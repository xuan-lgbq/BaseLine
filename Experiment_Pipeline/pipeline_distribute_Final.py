import os
import sys
import re
import time
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.nn as nn
import torch.optim as optim
import argparse

from config_mnist import training_config as config
from plotting import plot_training_loss, plot_top_k_eigenvalues, save_eigenvalue_csv
from utils_distribute import setup, cleanup, set_seed
from hessian_distribute import distributed_power_iteration, trace_distributed
from alignment import grad_dominant_alignment
from gradient_distribute import global_gradient_distribute
from Top_k_Dom_search import search_top_k_dominant_bulk_space
from data import load_dataset, DATASETS
from hessian_analysis_plugin import (
    compute_param_ranges, compute_eigenvalue_projections, save_projections_to_csv
)
from utilities import get_directory, get_optimizer, get_loss_and_acc, save_files, save_files_final
from utils_distribute import collect_eigenvalue_data, evaluate_and_log_test_accuracy
from archs import load_architecture
import swanlab

os.environ['MASTER_ADDR'] = '127.0.0.1'
os.environ['MASTER_PORT'] = '41900'
os.environ['GLOO_SOCKET_IFNAME'] = 'lo'
os.environ['NCCL_SOCKET_IFNAME'] = 'lo'
os.environ["CUDA_VISIBLE_DEVICES"] = "3,4,5,6,7"

def extract_step_from_ckpt(ckpt_path):
    match = re.search(r'model_step_(\d+)\.pth', os.path.basename(ckpt_path))
    if match:
        return int(match.group(1))
    else:
        return 0

def print_config(config):
    print("========== Training Config ==========")
    for k, v in config.items():
        print(f"{k}: {v}")
    print("=====================================")

def compute_global_loss(network: nn.Module, loss_functions: list, dataloader: DataLoader, world_size: int, rank: int):
    """
    分布式全局loss/acc计算，返回每个loss_fn的全局平均值。
    """
    L = len(loss_functions)
    total = torch.zeros(L, device=rank)
    total_samples = 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(rank), y.to(rank)
            preds = network(X)
            for l, loss_fn in enumerate(loss_functions):
                loss = loss_fn(preds, y)
                if hasattr(loss_fn, "reduction") and getattr(loss_fn, "reduction", None) == "mean":
                    loss = loss * X.size(0)  # 转为sum
                total[l] += loss.item() if loss.numel() == 1 else loss.sum().item()
            total_samples += X.size(0)
    dist.all_reduce(total, op=dist.ReduceOp.SUM)
    total_samples_tensor = torch.tensor(total_samples, device=rank)
    dist.all_reduce(total_samples_tensor, op=dist.ReduceOp.SUM)
    total_samples = total_samples_tensor.item()
    avg = total / total_samples
    return avg.cpu().tolist()

def train(rank, args):
    world_size = args.world_size
    set_seed(config["seed"])
    setup(rank, world_size)

    # Swanlab初始化
    if rank == 0:
        swanlab.init(
            project=config.get("swanlab_project_name", "default"),
            name=config.get("swanlab_experiment_name", "exp"),
            api_key=os.environ.get("SWANLAB_API_KEY", "")
        )
        swanlab.config.update(config)
        print_config(config)

    # ========== 数据准备 ==========
    save_dir = config.get("save_dir", "./save")
    os.makedirs(save_dir, exist_ok=True)
    train_dataset, test_dataset = load_dataset(config["dataset"], config["loss"])
    loss_fn, acc_fn = get_loss_and_acc(config["loss"])

    # ========== 模型与优化器 ==========
    model = load_architecture(config["arch_id"], config["dataset"]).cuda(rank)
    ddp_model = DDP(model, device_ids=[rank])
    
    optimizer = get_optimizer(ddp_model.parameters(), config["opt"], config["lr"], config["beta"], config.get("rho", 0))

    # ========== resume from checkpoint ==========
    resume_ckpt = getattr(args, "resume_ckpt", None)
    start_step = 0

    if resume_ckpt is not None and os.path.exists(resume_ckpt):
        map_location = {'cuda:%d' % 0: 'cuda:%d' % rank}
        ddp_model.module.load_state_dict(torch.load(resume_ckpt, map_location=map_location))
        start_step = extract_step_from_ckpt(resume_ckpt)
        if rank == 0:
            print(f"[Rank {rank}] Resumed from {resume_ckpt}, start from step {start_step}")
    else:
        if rank == 0 and resume_ckpt is not None:
            print(f"[Rank {rank}] Resume checkpoint not found: {resume_ckpt}")

    global_batch_size = config.get("batch_size", 50)
    physical_batch_size = config.get("physical_batch_size", 1000)
    max_steps = config["max_steps"]
    eig_freq = config["eig_freq"]
    neigs = config["neigs"]
    neigs_dom = config["neigs_dom"]
    save_freq = config["save_freq"]

    sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    train_loader = DataLoader(train_dataset, batch_size=global_batch_size // world_size, sampler=sampler, drop_last=True)
    computation_loader = DataLoader(train_dataset, batch_size=physical_batch_size // world_size, sampler=sampler, drop_last=True)

    # ========== 日志与历史 ==========
    train_loss = torch.zeros(max_steps)
    train_acc = torch.zeros(max_steps)
    eigs = torch.zeros(max_steps // eig_freq if eig_freq >= 0 else 0, neigs)
    evecs_grad_cos = torch.zeros(max_steps // eig_freq if eig_freq >= 0 else 0, neigs)
    evecs_update_cos = torch.zeros(max_steps // eig_freq if eig_freq >= 0 else 0, neigs)
    grad_norms = torch.zeros(max_steps)
    efflr_dom = torch.zeros(max_steps // eig_freq).to(torch.float64)
    efflr_orth = torch.zeros(max_steps // eig_freq).to(torch.float64)
    loss_history, step_history = [], []
    eigenvalue_history = {f"top_{i+1}": [] for i in range(neigs)}
    eigenvalue_step_history = []
    grad_dom_proj, grad_dom_proj_step_history = [], []
    accuracy_history, accuracy_step_history = [], []

    dist.barrier()  # 等待所有进程

    max_epochs = max_steps * global_batch_size // len(train_dataset)

    if resume_ckpt is not None and os.path.exists(resume_ckpt):
        step = start_step
    else:
        step = 0
    sys.stdout = open(os.path.join(save_dir, "train.log"), "a", 1) if rank == 0 else sys.stdout

    for epoch in range(max_epochs):
        if rank == 0:
            print(f"Epoch {epoch}")
        sampler.set_epoch(epoch)
        for X_batch, y_batch in train_loader:
            if step >= max_steps:
                break

            X_batch, y_batch = X_batch.to(rank), y_batch.to(rank)

            # 全局loss/acc（用computation_loader，增大batch）
            losses = compute_global_loss(ddp_model.module, [loss_fn, acc_fn], computation_loader, world_size, rank)
            train_loss[step], train_acc[step] = losses[0], losses[1]
            if rank == 0:
                print(f"{config['opt']} | {step}\t{train_loss[step]:.4f}\t{train_acc[step]:.4f}")
                swanlab.log({"Global Training Loss": train_loss[step].item()}, step=step)
                swanlab.log({"Global Training Accuracy": train_acc[step].item()}, step=step)

            # 全局梯度
            grad = global_gradient_distribute(
                model=ddp_model.module,
                criterion=loss_fn,
                dataloader=train_loader,
                world_size=world_size
            )
            grad_norms[step] = torch.norm(grad)
            print(f"gradient norm: {grad_norms[step].item()}")
            print(f"gradient shape: {grad.shape}")
            if rank == 0:
                swanlab.log({"Global Grad Norm": grad_norms[step].item()}, step=step)

            # Hessian分析
            if eig_freq != -1 and step % eig_freq == 0:
                start_time = time.time()
                eigenvalues, eigvectors = distributed_power_iteration(
                    model=ddp_model.module,
                    data=None,
                    dataloader=computation_loader,
                    criterion=loss_fn,
                    top_n=neigs,
                    max_iter=config.get("hessian_lanczos_steps", 250),
                    batch_size=None,
                    tol=1e-4,
                    hessian_type=config.get("hessian_type", "full")
                )
                duration = time.time() - start_time
                trace = trace_distributed(
                    model=ddp_model.module,
                    data=None,
                    dataloader=computation_loader,
                    criterion=loss_fn,
                    max_iter=config.get("hessian_lanczos_steps", 250),
                    tol=1e-4,
                    hessian_type=config.get("hessian_type", "full"),
                    batch_size=None
                )
                trace_mean = trace.mean().to(rank)
                dist.all_reduce(trace_mean, op=dist.ReduceOp.SUM)
                multi_trace_value = trace_mean.item() / world_size

                if rank == 0:
                    eigs[step // eig_freq, :] = eigenvalues
                    swanlab.log({f"Eigenvalue/Eigenvalue_{i+1}": eigenvalues[i].item() for i in range(len(eigenvalues))}, step=step)
                    print("eigenvalues: ", eigenvalues)
                    print(f"eigenvectors shape: {eigvectors .shape}")
                    evecs_grad_cos[step // eig_freq, :] = torch.tensor([
                        torch.nn.functional.cosine_similarity(eigvectors[:, i].cuda(), grad, dim=0, eps=1e-8).cpu().detach()
                        for i in range(neigs)
                    ])
                    print("cos(eigenvector, gradient): ", evecs_grad_cos[step // eig_freq, :])
                    params_vec = torch.nn.utils.parameters_to_vector(ddp_model.module.parameters()).cuda()
                    # 记录step
                    collect_eigenvalue_data(eigenvalue_history, eigenvalues)
                    eigenvalue_step_history.append(step)

            optimizer.zero_grad()
            loss = loss_fn(ddp_model(X_batch), y_batch) / (global_batch_size // world_size)
            loss.backward()
            optimizer.step()

            # batch loss（分布式聚合）
            with torch.no_grad():
                reduced_loss = loss.clone()
                dist.all_reduce(reduced_loss, op=dist.ReduceOp.SUM)
                reduced_loss /= world_size
            if rank == 0:
                loss_history.append(reduced_loss.item())
                step_history.append(step)
                swanlab.log({"Batch Training Loss": reduced_loss.item()}, step=step)

            # update cos & efflr
            if eig_freq != -1 and step % eig_freq == 0 and rank == 0:
                params_vec_next = torch.nn.utils.parameters_to_vector(ddp_model.module.parameters()).cuda()
                update = params_vec_next - params_vec
                evecs_update_cos[step // eig_freq, :] = torch.tensor([
                    torch.nn.functional.cosine_similarity(eigvectors[:, i].cuda(), update, dim=0, eps=1e-8).cpu().detach()
                    for i in range(neigs)
                ])
                update = update.to(torch.float64)
                grad = grad.to(torch.float64)
                grad_dom = torch.zeros_like(grad).cuda().to(torch.float64)
                for i in range(neigs_dom):
                    grad_dom += torch.dot(grad, eigvectors[:, i].cuda().to(torch.float64)) * eigvectors[:, i].cuda().to(torch.float64)
                grad_orth = grad - grad_dom
                efflr_dom[step // eig_freq] = - torch.dot(update, grad_dom) / (torch.norm(grad_dom)**2 + 1e-12)
                efflr_orth[step // eig_freq] = - torch.dot(update, grad_orth) / (torch.norm(grad_orth)**2 + 1e-12)
                swanlab.log({"EffLR_dom": efflr_dom[step // eig_freq].item(), "EffLR_orth": efflr_orth[step // eig_freq].item()}, step=step)

            # 其它分析与保存
            if eig_freq != -1 and step % eig_freq == 0 and rank == 0:
                space_result = search_top_k_dominant_bulk_space(eigenvalues.tolist(), method=config.get("search_method", "gap"))
                dominant_dim = space_result['dominant_end'] - space_result['dominant_start']
                grad_dom_align = grad_dominant_alignment(eigvectors[:, :dominant_dim], grad.float(), eps=1e-12)
                grad_dom_align_class = grad_dominant_alignment(eigvectors[:, :10], grad.float(), eps=1e-12)
                grad_dom_proj.append(grad_dom_align)
                grad_dom_proj_step_history.append(step)
                swanlab.log({"Grad-Dominant-Alignment": grad_dom_align.item()}, step=step)
                swanlab.log({"Grad-Dominant-Alignment-class": grad_dom_align_class.item()}, step=step)
                dominant_sum = eigenvalues[:dominant_dim].sum().item()
                trace_ratio = dominant_sum / (multi_trace_value + 1e-12)
                swanlab.log({"Dominant Trace Ratio": trace_ratio}, step=step)
                swanlab.log({"Hessian Trace": multi_trace_value}, step=step)
                swanlab.log({"Hessian Eig Time": duration}, step=step)
                param_ranges = compute_param_ranges(ddp_model.module.parameters())
                projections = compute_eigenvalue_projections(
                    eigenvectors=eigvectors, param_ranges=param_ranges, dominant_dim=None
                )
                save_projections_to_csv(projections=projections, save_dir=save_dir, step=step)
                save_eigenvalue_csv(eigenvalue_history, eigenvalue_step_history, config, save_dir)
                model_path = os.path.join(save_dir, "checkpoints", f"model_step_{step + 1}.pth")
                os.makedirs(os.path.dirname(model_path), exist_ok=True)
                torch.save(ddp_model.module.state_dict(), model_path)

            if save_freq != -1 and step % save_freq == 0 and rank == 0:
                save_files(save_dir, [("eigs", eigs[:step // eig_freq]), ("evecs_grad_cos", evecs_grad_cos[:step // eig_freq]),
                                    ("evecs_update_cos", evecs_update_cos[:step // eig_freq]),
                                    ("grad_norms", grad_norms[:step]),
                                    ("efflr_dom", efflr_dom[:step // eig_freq]), ("efflr_orth", efflr_orth[:step // eig_freq]),
                                    ("train_loss", train_loss[:step]), ("train_acc", train_acc[:step])])
            step += 1

    if rank == 0:
        save_files_final(save_dir,
            [("eigs", eigs[:(step) // eig_freq]), ("evecs_grad_cos", evecs_grad_cos[:(step) // eig_freq]),
             ("evecs_update_cos", evecs_update_cos[:(step) // eig_freq]),
             ("grad_norms", grad_norms[:step]),
             ("efflr_dom", efflr_dom[:step // eig_freq]), ("efflr_orth", efflr_orth[:step // eig_freq]),
             ("train_loss", train_loss[:step]), ("train_acc", train_acc[:step])])
        if len(loss_history) > 0:
            plot_training_loss(loss_history, step_history, config, save_dir, None)
        if len(eigenvalue_history) > 0:
            plot_top_k_eigenvalues(eigenvalue_history, eigenvalue_step_history, config, save_dir, None)
        if len(eigenvalue_history) > 0:
            save_eigenvalue_csv(eigenvalue_history, eigenvalue_step_history, config, save_dir)
        sys.stdout = sys.__stdout__

    cleanup()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Distributed Training Example')
    parser.add_argument('--world_size', type=int, default=5, help='Total number of processes (usually = num_gpus)')
    parser.add_argument('--backend', type=str, default='nccl', choices=['nccl', 'gloo'], help='Distributed backend (default: nccl)')
    parser.add_argument('--master_addr', type=str, default='127.0.0.1', help='Address of the master node (default: 127.0.0.1)')
    parser.add_argument('--master_port', type=str, default='46772', help='Port of the master node (default: 29500)')
    parser.add_argument('--resume_ckpt', type=str, default=None, help='Path to resume checkpoint (default: None)')
    args = parser.parse_args()

    mp.spawn(
        train,
        args=(args,),
        nprocs=args.world_size,
        join=True
    )

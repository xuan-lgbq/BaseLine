import sys
from os import makedirs

import torch
from torch.nn.utils import parameters_to_vector, vector_to_parameters

import argparse

from archs import load_architecture
from utilities import get_proj_directory, get_optimizer, get_loss_and_acc, compute_losses, save_files, save_files_final, get_hessian_eigenvalues, compute_gradient
from data import load_dataset, DATASETS


def main(proj: str, dataset: str, arch_id: str, loss: str, opt: str, lr: float, batch_size: int = 100, beta: float = 0, rho: float = 0, max_steps: int = 10000, start_step: int = 0, neigs: int = 0, neigs_dom: int = 0, physical_batch_size: int = 1000, save_freq: int = -1, save_model: bool = False, loss_goal: float = None, acc_goal: float = None, seed: int = 0, gpu_id: int = 0):
    torch.cuda.set_device(gpu_id)

    directory = get_proj_directory(proj, dataset, arch_id, loss, seed, opt, lr, beta, rho, batch_size, start_step)
    makedirs(directory, exist_ok=True)
    f = open(directory + '/train.log', "a", 1)
    print(f"output directory: {directory}")

    train_dataset, test_dataset = load_dataset(dataset, loss)

    loss_fn, acc_fn = get_loss_and_acc(loss)

    torch.manual_seed(seed)
    network = load_architecture(arch_id, dataset).cuda()
    if save_model:
        torch.save(network.state_dict(), f"{directory}/snapshot_init")

    optimizer = get_optimizer(network.parameters(), opt, lr, beta, rho)

    train_loss, train_acc = torch.zeros(max_steps), torch.zeros(max_steps)
    eigs = torch.zeros(max_steps, neigs)
    # evecs = torch.zeros(max_steps, len(parameters_to_vector(network.parameters())), neigs)
    evecs_grad_cos = torch.zeros(max_steps, neigs)
    grad_norms = torch.zeros(max_steps)

    max_epochs = max_steps * batch_size // len(train_dataset)
    step = 0
    data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    sys.stdout = f
    for epoch in range(max_epochs):
        print(f"Epoch {epoch}")
        for X_batch, y_batch in data_loader:
            train_loss[step], train_acc[step] = compute_losses(network, [loss_fn, acc_fn], train_dataset, physical_batch_size)

            grad = compute_gradient(network, loss_fn, train_dataset, physical_batch_size)
            grad_norms[step] = torch.norm(grad)
            if step < start_step:
                print(f"{opt} | {step}\t{train_loss[step]:.4f}\t{train_acc[step]:.4f}")
            else:
                print(f"{proj}-{opt} | {step}\t{train_loss[step]:.4f}\t{train_acc[step]:.4f}")
                if step == start_step and save_model:
                    torch.save(network.state_dict(), f"{directory}/snapshot_{step}")
                eigs[step, :], evec = get_hessian_eigenvalues(network, loss_fn, train_dataset, neigs=neigs, physical_batch_size=physical_batch_size)
                print("eigenvalues: ", eigs[step, :])
                
                with torch.no_grad():
                    for i in range(neigs):
                        evecs_grad_cos[step, i] = torch.nn.functional.cosine_similarity(evec[:,i].cuda(), grad, dim=0, eps=1e-8).cpu().detach()
                    # print("cos(eigenvector, gradient): ", evecs_grad_cos[step, :])

                    # save parameter vector
                    params_vec = parameters_to_vector(network.parameters()).cuda()

            # optimizer step
            optimizer.zero_grad()
            loss = loss_fn(network(X_batch.cuda()), y_batch.cuda()) / batch_size
            loss.backward()
            if opt == "sam":
                # first forward-backward pass
                optimizer.first_step(zero_grad=True)
                # second forward-backward pass
                (loss_fn(network(X_batch.cuda()), y_batch.cuda()) / batch_size).backward()
                optimizer.second_step(zero_grad=True)
            else:
                optimizer.step()

            if step>=start_step:
                with torch.no_grad():
                    params_vec_next = parameters_to_vector(network.parameters()).cuda()
                    update = params_vec_next - params_vec
                    update_dom = torch.zeros_like(update)
                    for i in range(neigs_dom):
                        update_dom += torch.dot(update, evec[:, i].cuda()) * evec[:, i].cuda()
                    if proj == "dom":
                        params_vec_next_proj = params_vec + update_dom
                    elif proj == "bulk":
                        params_vec_next_proj = params_vec_next - update_dom
                    vector_to_parameters(params_vec_next_proj, network.parameters())


            if save_freq != -1 and step % save_freq == 0:
                save_files(directory, [("eigs", eigs[:step]), ("evecs_grad_cos", evecs_grad_cos[:step]),
                                       ("grad_norms", grad_norms[:step]),
                                       ("train_loss", train_loss[:step]), ("train_acc", train_acc[:step])])      
            step = step + 1

        if (loss_goal != None and train_loss[step-1] < loss_goal) or (acc_goal != None and train_acc[step-1] > acc_goal):
            break

    save_files_final(directory, [("eigs", eigs[:step]), ("evecs_grad_cos", evecs_grad_cos[:step]),
                                 ("grad_norms", grad_norms[:step]),
                                 ("train_loss", train_loss[:step]), ("train_acc", train_acc[:step])]) 
    if save_model:
        torch.save(network.state_dict(), f"{directory}/snapshot_final")

    sys.stdout = sys.__stdout__
    f.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a neural network using projected updates.")
    parser.add_argument("proj", type=str, choices=["dom", "bulk"], help="which subspace to project")
    parser.add_argument("dataset", type=str, choices=DATASETS, help="which dataset to train")
    parser.add_argument("arch_id", type=str, help="which network architectures to train")
    parser.add_argument("loss", type=str, choices=["ce", "mse", "logtanh"], help="which loss function to use")
    parser.add_argument("opt", type=str, choices=["sgd", "sam", "adam"], help="which optimization method to use")
    parser.add_argument("lr", type=float, help="the learning rate")
    parser.add_argument("--max_steps", type=int, help="the maximum number of gradient steps to train for", default=1000)
    parser.add_argument("--start_step", type=int, help="the step to start projected method", default=0)
    parser.add_argument("--seed", type=int, help="the random seed used when initializing the network weights",
                        default=0)
    parser.add_argument("--beta", type=float, help="momentum parameter", default=0)
    parser.add_argument("--rho", type=float, help="perturbation radius for SAM", default=0)
    parser.add_argument("--physical_batch_size", type=int,
                        help="the maximum number of examples that we try to fit on the GPU at once", default=1000)
    parser.add_argument("--batch_size", type=int,
                        help="batch size of SGD", default=50)
    parser.add_argument("--acc_goal", type=float,
                        help="terminate training if the train accuracy ever crosses this value", default=1)
    parser.add_argument("--loss_goal", type=float, help="terminate training if the train loss ever crosses this value", default=0)
    parser.add_argument("--neigs", type=int, help="the number of top eigenvalues to compute")
    parser.add_argument("--neigs_dom", type=int, help="the number of dominant top eigenvalues")
    parser.add_argument("--save_freq", type=int, default=-1,
                        help="the frequency at which we save results")
    parser.add_argument("--save_model", type=bool, default=False,
                        help="if 'true', save model weights at end of training")
    parser.add_argument("--gpu_id", type=int, help="gpu (cuda device) id", default=0)
    args = parser.parse_args()

    main(proj=args.proj, dataset=args.dataset, arch_id=args.arch_id, loss=args.loss, opt=args.opt, lr=args.lr, batch_size=args.batch_size, beta=args.beta, rho=args.rho, max_steps=args.max_steps, start_step=args.start_step, neigs=args.neigs, neigs_dom=args.neigs_dom, physical_batch_size=args.physical_batch_size, save_freq=args.save_freq, save_model=args.save_model, loss_goal=args.loss_goal, acc_goal=args.acc_goal, seed=args.seed, gpu_id=args.gpu_id)
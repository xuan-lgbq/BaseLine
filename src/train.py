import sys
from os import makedirs
import os
import torch
from torch.nn.utils import parameters_to_vector, vector_to_parameters

import argparse


from config import training_config


from archs import load_architecture
from utilities import get_directory, get_optimizer, get_loss_and_acc, compute_losses, save_files, save_files_final, get_hessian_eigenvalues, compute_gradient
from data import load_dataset, DATASETS


from eigenvalues_analysis import compute_and_analyze_hessian

from eigenvalues_analysis import compute_and_analyze_hessian, analyze_dominant_space_layerwise, save_eigenvector_data 


def main(dataset: str, arch_id: str, loss: str, opt: str, lr: float, batch_size: int = 100, beta: float = 0, rho: float = 0, max_steps: int = 10000, neigs: int = 0, neigs_dom: int = 0, eig_freq: int = -1, physical_batch_size: int = 1000, save_freq: int = -1, save_model: bool = False, loss_goal: float = None, acc_goal: float = None, seed: int = 0, gpu_id: int = 0):
    
    torch.cuda.set_device(gpu_id)

    directory = get_directory(dataset, arch_id, loss, seed, opt, lr, beta, rho, batch_size)
    makedirs(directory, exist_ok=True)
    f = open(directory + '/train.log', "a", 1)
    print(f"Output directory: {directory}")

    train_dataset, test_dataset = load_dataset(dataset, loss)
    loss_fn, acc_fn = get_loss_and_acc(loss)

    torch.manual_seed(seed)
    network = load_architecture(arch_id, dataset).cuda()
    if save_model:
        torch.save(network.state_dict(), f"{directory}/snapshot_init")

    optimizer = get_optimizer(network.parameters(), opt, lr, beta, rho)

    train_loss, train_acc = torch.zeros(max_steps), torch.zeros(max_steps)
    
    # Calculate num_eig_computations more robustly for initialization
    num_eig_computations = max_steps // eig_freq if eig_freq > 0 else 0
    if eig_freq > 0 and max_steps % eig_freq != 0: # If it's not a perfect multiple, add one more slot
        num_eig_computations += 1

    eigs = torch.zeros(num_eig_computations, neigs)
    evecs_grad_cos = torch.zeros(num_eig_computations, neigs)
    evecs_update_cos = torch.zeros(num_eig_computations, neigs)
    grad_norms = torch.zeros(max_steps)
    efflr_dom = torch.zeros(num_eig_computations).to(torch.float64)
    efflr_orth = torch.zeros(num_eig_computations).to(torch.float64)

    max_epochs = max_steps * batch_size // len(train_dataset)
    step = 0
    data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    sys.stdout = f

    # Define a sub-directory for saving layer-wise analysis plots and data
    analysis_save_dir = os.path.join(directory, "hessian_layer_analysis")
    makedirs(analysis_save_dir, exist_ok=True)

    for epoch in range(max_epochs):
        print(f"Epoch {epoch}")
        for X_batch, y_batch in data_loader:
            train_loss[step], train_acc[step] = compute_losses(network, [loss_fn, acc_fn], train_dataset, physical_batch_size)
            print(f"{opt} | {step}\t{train_loss[step]:.4f}\t{train_acc[step]:.4f}")

            grad = compute_gradient(network, loss_fn, train_dataset, physical_batch_size)
            grad_norms[step] = torch.norm(grad)

            calculated_eigs = None
            calculated_evecs_dominant = None
            calculated_dominant_dim = None
            success = False # Initialize success flag

           
            if eig_freq > 0 and step % eig_freq == 0: 
                calculated_eigs, calculated_evecs_dominant, calculated_dominant_dim, calculated_gaps, success = \
                    compute_and_analyze_hessian(
                        model=network,
                        loss_function=loss_fn,
                        dataset_for_hessian=train_dataset, # Pass train_dataset directly
                        neigs=neigs, # Pass neigs
                        device=torch.device(f'cuda:{gpu_id}'),
                        config=training_config, # Pass training_config for method and physical_batch_size
                        step=step
                    )
                
                if success:
                    # Copy computed eigenvalues to the main eigs tensor
                    num_to_copy = min(neigs, len(calculated_eigs))
                    if num_to_copy > 0:
                        eigs[step // eig_freq, :num_to_copy] = calculated_eigs[:num_to_copy]
                    else:
                        print(f"Warning: Step {step} did not compute enough eigenvalues.")

                    print(f"Eigenvalues (copied to eigs tensor): {eigs[step // eig_freq, :num_to_copy].tolist()}")
                    
                    # --- NEW: Call the layer-wise analysis function ---
                    print(f"Starting layer-wise analysis for step {step}...")
                    analysis_data = analyze_dominant_space_layerwise(
                        model=network, 
                        eigenvectors=calculated_evecs_dominant, 
                        eigenvalues=calculated_eigs, 
                        dominant_dim=calculated_dominant_dim, # Use the actual dominant_dim from analysis
                        save_dir=analysis_save_dir, 
                        step=step
                    )
                    
                   
                    if analysis_data:
                        save_eigenvector_data(
                            analysis_data=analysis_data,
                            save_dir=analysis_save_dir,
                            step=step,
                            method=training_config.get("method", "max_gap"),
                            lr=lr,
                            seed=seed,
                            rank=calculated_dominant_dim, # Rank is the dominant dimension
                            num_layer=len(analysis_data['layer_info']) # Number of layers analyzed
                        )

                    # Now compute evecs_grad_cos using the dominant eigenvectors
                    with torch.no_grad():
                        if calculated_evecs_dominant is not None and calculated_evecs_dominant.shape[1] > 0:
                            # Iterate only up to the dominant dimension for cosine similarity
                            # or up to neigs if neigs_dom is larger, but evecs_grad_cos is sized by neigs.
                            for i in range(min(neigs, calculated_evecs_dominant.shape[1])):
                                evecs_grad_cos[step // eig_freq, i] = torch.nn.functional.cosine_similarity(
                                    calculated_evecs_dominant[:, i].cuda(), grad, dim=0, eps=1e-8
                                ).cpu().detach()
                        else:
                            print(f"Warning: No dominant eigenvectors for evecs_grad_cos at step {step}")

            params_vec = parameters_to_vector(network.parameters()).cuda() # Save parameters before optimizer step

            optimizer.zero_grad()
            loss = loss_fn(network(X_batch.cuda()), y_batch.cuda()) / batch_size
            loss.backward()
            if opt == "sam":
                optimizer.first_step(zero_grad=True)
                (loss_fn(network(X_batch.cuda()), y_batch.cuda()) / batch_size).backward()
                optimizer.second_step(zero_grad=True)
            else:
                optimizer.step()

            # Perform post-optimizer step calculations only if Hessian was computed successfully
            if eig_freq > 0 and step % eig_freq == 0 and success and calculated_evecs_dominant is not None:
                with torch.no_grad():
                    params_vec_next = parameters_to_vector(network.parameters()).cuda()
                    update = params_vec_next - params_vec
                    
                    # Compute evecs_update_cos
                    if calculated_evecs_dominant.shape[1] > 0:
                        # Iterate only up to the dominant dimension for cosine similarity
                        # or up to neigs if neigs_dom is larger, but evecs_update_cos is sized by neigs.
                        for i in range(min(neigs, calculated_evecs_dominant.shape[1])):
                            evecs_update_cos[step // eig_freq, i] = torch.nn.functional.cosine_similarity(
                                calculated_evecs_dominant[:, i].cuda(), update, dim=0, eps=1e-8
                            ).cpu().detach()
                    else:
                        print(f"Warning: No dominant eigenvectors for evecs_update_cos at step {step}")


                    update = update.to(torch.float64)
                    grad = grad.to(torch.float64)
                    
                    grad_dom = torch.zeros_like(grad).cuda().to(torch.float64)
                    # Use calculated_dominant_dim here for accuracy if available, otherwise neigs_dom
                    actual_neigs_dom_for_calc = calculated_dominant_dim if calculated_dominant_dim is not None else neigs_dom
                    actual_neigs_dom_for_calc = min(actual_neigs_dom_for_calc, calculated_evecs_dominant.shape[1])

                    for i in range(actual_neigs_dom_for_calc):
                        # Move eigenvector part to CUDA for dot product
                        evec_i_cuda = calculated_evecs_dominant[:, i].cuda().to(torch.float64)
                        grad_dom += torch.dot(grad, evec_i_cuda) * evec_i_cuda
                    
                    grad_orth = grad.to(torch.float64) - grad_dom

                    # Avoid division by zero
                    if torch.norm(grad_dom)**2 > 1e-12: 
                        efflr_dom[step // eig_freq] = - torch.dot(update, grad_dom) / torch.norm(grad_dom)**2
                    else:
                        efflr_dom[step // eig_freq] = float('nan') 

                    if torch.norm(grad_orth)**2 > 1e-12:
                        efflr_orth[step // eig_freq] = - torch.dot(update, grad_orth) / torch.norm(grad_orth)**2
                    else:
                        efflr_orth[step // eig_freq] = float('nan') 

            if save_freq != -1 and step % save_freq == 0:
                # Calculate the current index for saving based on eig_freq
                current_eig_idx = (step // eig_freq)
                
                # Adjust slice length to ensure it doesn't go out of bounds
                # and only includes data up to the last successful computation.
                slice_end_idx = current_eig_idx + (1 if (eig_freq > 0 and step % eig_freq == 0 and success) else 0)
                if slice_end_idx > num_eig_computations:
                    slice_end_idx = num_eig_computations

                save_files(directory, [("eigs", eigs[:slice_end_idx]), 
                                        ("evecs_grad_cos", evecs_grad_cos[:slice_end_idx]),
                                        ("evecs_update_cos", evecs_update_cos[:slice_end_idx]),
                                        ("grad_norms", grad_norms[:step]),
                                        ("efflr_dom", efflr_dom[:slice_end_idx]), 
                                        ("efflr_orth", efflr_orth[:slice_end_idx]),
                                        ("train_loss", train_loss[:step]), 
                                        ("train_acc", train_acc[:step])])          
            step = step + 1

        if (loss_goal is not None and train_loss[step-1] < loss_goal) or \
           (acc_goal is not None and train_acc[step-1] > acc_goal):
            break

    # Final save: Ensure all relevant data up to the final step is saved.
    # Calculate the final slice length for Hessian-related Tensors.
    final_eig_slice_len = (step // eig_freq)
    if eig_freq > 0 and step % eig_freq == 0 and success: # If the last step itself was an eig computation
        final_eig_slice_len += 1
    if final_eig_slice_len > num_eig_computations:
        final_eig_slice_len = num_eig_computations # Prevent out-of-bounds access

    save_files_final(directory,
                    [("eigs", eigs[:final_eig_slice_len]), 
                     ("evecs_grad_cos", evecs_grad_cos[:final_eig_slice_len]),
                     ("evecs_update_cos", evecs_update_cos[:final_eig_slice_len]),
                     ("grad_norms", grad_norms[:step]),
                     ("efflr_dom", efflr_dom[:final_eig_slice_len]), 
                     ("efflr_orth", efflr_orth[:final_eig_slice_len]),
                     ("train_loss", train_loss[:step]), 
                     ("train_acc", train_acc[:step])])
    if save_model:
        torch.save(network.state_dict(), f"{directory}/snapshot_final")

    sys.stdout = sys.__stdout__
    f.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a neural network.")
    parser.add_argument("dataset", type=str, choices=DATASETS, help="which dataset to train")
    parser.add_argument("arch_id", type=str, help="which network architectures to train")
    parser.add_argument("loss", type=str, choices=["ce", "mse", "logtanh"], help="which loss function to use")
    parser.add_argument("opt", type=str, choices=["sgd", "sam", "adam"], help="which optimization method to use")
    parser.add_argument("lr", type=float, help="the learning rate")
    parser.add_argument("--max_steps", type=int, help="the maximum number of gradient steps to train for", default=10000)
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
    parser.add_argument("--eig_freq", type=int, default=-1,
                        help="the frequency at which we compute the top Hessian eigenvalues (-1 means never)")
    parser.add_argument("--save_freq", type=int, default=-1,
                        help="the frequency at which we save results")
    parser.add_argument("--save_model", type=bool, default=False,
                        help="if 'true', save model weights at end of training")
    parser.add_argument("--gpu_id", type=int, help="gpu (cuda device) id", default=0)
    args = parser.parse_args()

   
    for key, value in training_config.items():
        
        if hasattr(args, key):
           
            if isinstance(value, bool):
                setattr(args, key, value)
          
            elif key in ["loss_goal", "acc_goal"] and value is None:
                setattr(args, key, None)
            # IMPORTANT: Exclude 'method' from direct args override if it's not a direct main() parameter.
            # It's typically used within `training_config` for the analysis function.
            elif key == "method": 
                continue 
            else:
                setattr(args, key, value)
   
    main(dataset=args.dataset, arch_id=args.arch_id, loss=args.loss, opt=args.opt, lr=args.lr, batch_size=args.batch_size, beta=args.beta, rho=args.rho, max_steps=args.max_steps, neigs=args.neigs, neigs_dom=args.neigs_dom, eig_freq=args.eig_freq, physical_batch_size=args.physical_batch_size, save_freq=args.save_freq, save_model=args.save_model, loss_goal=args.loss_goal, acc_goal=args.acc_goal, seed=args.seed, gpu_id=args.gpu_id)
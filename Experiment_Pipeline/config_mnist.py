from data import num_classes
from datetime import datetime
import os

training_config = {
    "seed": 0,
    "max_steps": 20000,
    "batch_size": 50,
    "physical_batch_size": 1000,
    "lr": 0.01,
    "dataset": "mnist-5k",
    "arch_id": "fc-tanh",
    "loss": "mse",
    "opt": "sgd",
    "beta": 0,
    "rho": 0,
    "neigs": 150,
    "neigs_dom": 10,
    "eig_freq": 100,
    "save_freq": -1,
    "save_model": False,
    "acc_goal": 1,
    "loss_goal": 0,
}
training_config["num_classes"] = num_classes(training_config["dataset"])
training_config["exp_name"] = f"{training_config['arch_id']}_{training_config['dataset']}_lr{training_config['lr']}_top{training_config['neigs']}_seed{training_config['seed']}"  
training_config["save_dir"] = f"/jumbo/yaoqingyang/ouyangzhuoli/Low_rank_identity/MNIST/experiments/{training_config['exp_name']}"

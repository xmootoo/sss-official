import os
import json
import socket
import argparse
import sys
from datetime import datetime
from sss.exp.exp import Experiment
from sss.config.config import Global, load_config
import torch
import torch.multiprocessing as mp
from dotenv import load_dotenv
from rich.console import Console
from rich.pretty import pprint
import warnings
from pydantic import BaseModel
from typing import Dict, Any
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent.resolve()

warnings.filterwarnings("ignore", message="h5py not installed, hdf5 features will not be supported.")

def check_patient_cluster(args):
    if args.open_neuro.patient_cluster != "all" and args.open_neuro.all_clusters is True:
        return True
    elif args.open_neuro.patient_cluster == "all" and args.open_neuro.all_clusters is False:
        return True
    else:
        return False

def check_moderntcn_list_lengths(args):
    list_lengths = []

    # Iterate through all fields of args.moderntcn
    for field_name, field in args.moderntcn.__fields__.items():
        # Get the attribute value
        attr_value = getattr(args.moderntcn, field_name)

        # Check if the attribute is a list
        if isinstance(attr_value, list):
            list_lengths.append(len(attr_value))

    # If no lists were found, return True
    if not list_lengths:
        raise ValueError("No list attributes found in ModernTCN")

    # Check if all list lengths are equal
    return all(length == list_lengths[0] for length in list_lengths)

def check_open_neuro_coeff(args):
    if args.open_neuro.alpha == args.open_neuro.beta and args.open_neuro.alpha!=1.0:
        return True
    else:
        return False

def check_open_neuro_loocv(args):
    if args.open_neuro.loocv:
        return bool(set(args.open_neuro.train_clusters) & set(args.open_neuro.test_clusters))
    else:
        return False

def update_global_config(ablation_config: Dict[str, Any], global_config: BaseModel, ablation_id: int) -> BaseModel:
    """
    Updates the global config for ablation studies and hyperparameter tuning. For example,
    if the ablation_config is {'global_config.sl.lr': 0.01}, then the global_config.sl.lr will be
    updated to 0.01.
    """

    for key, value in ablation_config.items():
        parts = key.split('.')
        if len(parts) == 2:
            sub_model, param = parts
            if hasattr(global_config, sub_model):
                sub_config = getattr(global_config, sub_model)
                if hasattr(sub_config, param):
                    setattr(sub_config, param, value)
                else:
                    print(f"Warning: {sub_model} does not have attribute {param}")
            else:
                print(f"Warning: global_config does not have attribute {sub_model}")
        elif len(parts) == 3:
            model, sub_model, param = parts
            if model == "global" and hasattr(global_config, sub_model):
                sub_config = getattr(global_config, sub_model)
                if hasattr(sub_config, param):
                    setattr(sub_config, param, value)
                else:
                    print(f"Warning: {sub_model} does not have attribute {param}")
            else:
                print(f"Warning: Invalid key format or 'global' not specified: {key}")
        else:
            print(f"Warning: Invalid key format: {key}")

    global_config.exp.ablation_id = ablation_id

    return global_config

def main(job_name="test", ablation=None, ablation_id=1):

    # Rich console
    load_dotenv()
    console = Console()

    # Load experimental configuration
    # base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    # args_path = os.path.join(base_dir, "sss", "sss", "jobs", "exp", job_name, "args.yaml")
    args_path = SCRIPT_DIR / "sss" / "jobs" / job_name / "args.yaml"
    
    args = load_config(args_path)
    if ablation is not None:
        args = update_global_config(ablation, args, ablation_id)
    console.print("Pydantic Configuration Loaded Successfully:", style="bold green")

    # Run experiment on multiple seeds (optional)
    seed_list = args.exp.seed_list
    tuning_score = 0 # Hyperparameter tuning score

    if not check_moderntcn_list_lengths(args):
        console.print("List attributes of ModernTCN are not of equal length. Skipping configuration.", style="bold red")
        return tuning_score # Return 0 tuning score

    if check_open_neuro_coeff(args):
        console.print("Alpha and Beta coefficients of OpenNeuro are equal. Skipping configuration.", style="bold red")
        return tuning_score

    if check_patient_cluster(args):
        console.print("Patient cluster incompatible with all cluster. Skipping configuration.", style="bold red")
        return tuning_score

    if check_open_neuro_loocv(args):
        console.print("Train and test clusters overlap. Skipping configuration.", style="bold red")
        return tuning_score

    if args.open_neuro.loocv:
        args.open_neuro.patient_cluster = "loocv_" + args.open_neuro.test_clusters[0]

    for i in range(len(seed_list)):
        args.exp.seed = seed_list[i]
        args.exp.time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        # Print args
        pprint(args, expand_all=True)

        # Initialize experiment
        exp = Experiment(args)

        # Run experiment
        if args.ddp.ddp:
            console.log("Using DDP")
            os.environ["TORCH_NCCL_BLOCKING_WAIT"] = "1" # Enable timeout
            os.environ["NCCL_BLOCKING_WAIT_TIMEOUT"] = "1200" # 1200 seconds timeout (20min)
            if args.ddp.slurm:
                os.environ["MASTER_ADDR"] = socket.gethostname()
            else:
                os.environ['MASTER_ADDR'] = args.ddp.master_addr
                os.environ['MASTER_PORT'] = args.ddp.master_port
            world_size = torch.cuda.device_count()
            mp.spawn(exp.run, args=(world_size,), nprocs=world_size, join=True)
        else:
            console.log("Using single device")
            exp.run()

        tuning_score += exp.tuning_score

    return tuning_score / len(seed_list) # Return the average tuning score (Raytune)

if __name__ == "__main__":
    # Non-hyperparameter tuning
    warnings.filterwarnings("ignore", message="h5py not installed, hdf5 features will not be supported.")
    parser = argparse.ArgumentParser(description="Run experiment")
    parser.add_argument("job_name", type=str, default="test", help="Name of the job")
    parser.add_argument("--venv_test", action="store_true", help="Flag for venv test")
    parser.add_argument("--ablation", type=str, default=None, help="Ablation study configuration")
    parser.add_argument("--ablation_id", type=int, default=1, help="Ablation study configuration")

    args = parser.parse_args()

    if args.venv_test:
        print("Running in venv test mode")
        # Add any venv test specific code here
    else:
        if args.ablation is not None:
            try:
                ablation = json.loads(args.ablation)
                main(job_name=args.job_name, ablation=ablation, ablation_id=args.ablation_id)
            except json.JSONDecodeError:
                print("Error: Invalid JSON string for ablation configuration")
                sys.exit(1)
        else:
            main(args.job_name)

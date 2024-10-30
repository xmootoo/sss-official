# Add the project root directory to the Python path
import sys
import os
import json
import shlex
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, project_root)

import itertools
import yaml
from typing import Dict, List, Any, Tuple
from main import main
from sss.config.config import Global, load_config
from sss.jobs.hpc.submit import submit_job, load_configs
from rich.console import Console
import argparse

def load_ablation_config(file_path: str) -> Tuple[Dict[str, List[Any]], str]:
    with open(file_path, 'r') as file:
        config = yaml.safe_load(file)
    ablation_config = config.get('ablations', {})
    cc_base = config.get('cc', None)
    return ablation_config, cc_base

def generate_ablation_combinations(ablation_config: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
    keys, values = zip(*ablation_config.items())
    return [dict(zip(keys, v)) for v in itertools.product(*values)]

def run_grid_search(job_name: str, cc: bool = False, acc: str = "None"):
    # Adjust the base_path to use the absolute path
    base_path = os.path.join(project_root, "sss", "jobs", "exp")
    job_path = os.path.join(base_path, job_name, "args.yaml")
    ablation_path = os.path.join(base_path, job_name, "ablation.yaml")

    # Rich
    console = Console()

    # Load the ablation configuration
    ablation_config, cc_base = load_ablation_config(ablation_path)

    # Generate all combinations of ablations
    ablation_combinations = generate_ablation_combinations(ablation_config)

    for i, ablation in enumerate(ablation_combinations):

        if cc:
            console.log(f"Submitting ablation (Compute Canada): {ablation}")

            # Load base configs for Compute Canada and SLURM
            cc_config, slurm_config = load_configs(cc_base)

            # Convert ablation to a JSON string
            ablation_json = json.dumps(ablation)
            escaped_ablation = shlex.quote(ablation_json)
            cli_input = f"{job_name} --ablation {escaped_ablation} --ablation_id {i}"

            # Update slurm account
            if acc != "None":
                console.log(f"Updating account to {acc}")
                slurm_config["slurm_account"] = acc

            # Submit job
            submit_job(
                cc_config["logdir"],
                cc_config["virtual_env"],
                cc_config["modules"],
                cli_input,
                cc_config["script_path"],
                cc_config["cluster"],
                **slurm_config
            )
        else:
            console.log(f"Running ablation {i} (locally): {ablation}")
            main(job_name=job_name, ablation=ablation, ablation_id=i)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("ablation", type=str, default="sss", help="Name of the ablation")
    parser.add_argument("--cc", action="store_true", help="Uses Compute Canada and instead submits an ablation job.")
    parser.add_argument("--acc", default="None", help="Switch Compute Canada accounts")
    args = parser.parse_args()
    run_grid_search(args.ablation, args.cc, args.acc)

import os
import socket
import yaml
import argparse
import subprocess
import sys
from rich.console import Console
from rich.pretty import pprint  # For pretty printing objects
import submitit

def run_script(virtual_env, modules, cli_input, script_path, cluster="narval"):
    """
    Run the script in a SLURM job (e.g., training) on a cluster. This function
    will run on the cluster and execute the script.

    Args:
        cli_input: A string of input arguments to pass to the training script.
        virtual_env (str): The name of the virtual environment to activate.
        modules (str): The list of modules to load.
        script_path (str): Path to the script to run.
    Returns:
        None
    """

    # Rich console (terminal output formatting)
    from rich.console import Console
    console = Console()

    # Load modules
    if cluster == "narval":
        load_modules = "module load StdEnv/2023; " \
                    "module load python/3.10; " \
                    "module load gdrcopy/2.3.1; " \
                    "module load flexiblas; " \
                    "module load blis; " \
                    "module load cudacore/.12.2.2; " \
                    "module load nccl/2.18.3; " \
                    "module load ucx-cuda/1.14.1; " \
                    "module load ucc-cuda/1.2.0; " \
                    "module load cudnn/8.9.5.29"
    else:
        load_modules = f"module load {modules}"

    # Activate environment
    py_env_activate = os.path.join(os.path.expanduser("~"),virtual_env, "bin", "activate")
    if not os.path.isfile(py_env_activate):
        console.log(f"FileNotFound: the file {py_env_activate} does not exist")
        return
    activate_env = f"source {py_env_activate}"

    # Full command
    if script_path.endswith(".py"):
        command = f"{load_modules} && {activate_env} && python {script_path} {cli_input}"
    elif script_path.endswith(".sh"):
        command = f"{load_modules} && {activate_env} && bash {script_path} {cli_input}"

    console.log(f"Run command {command}")

	# Run the command on the Operating System using subprocess.run
    result = subprocess.run(command, shell=True, capture_output=True, text=True)

    # Check if the command failed
    if result.returncode != 0:
        console.log(f"Command failed with return code {result.returncode}")
        console.log(f"Error output: {result.stderr}")
    else:
        console.log(f"Command succeeded with return code {result.returncode}")
        console.log(f"Command output: {result.stdout}")

def submit_job(logdir, virtual_env, modules, cli_input, script_path, cluster, /, **kwds):
    """
    Submits a job to the cluster using SLURM, from your local computer.

    Args:
        logdir (str): The directory to store the logs and output of the job.
        cli_input (str): The input argument to pass to the training script.
        virtual_env (str): The name of the virtual environment to activate.
        modules (str): The list of modules to load.
        script_path (str): Path to the script to run.
    Returns:
        None
    """

    # Rich console (terminal output formatting)
    from rich.console import Console
    console = Console()

    # Create an executor for SLURM
    executor = submitit.AutoExecutor(folder=logdir)

    # Set SLURM job parameters
    executor.update_parameters(
        gpus_per_node=kwds.get("gpus_per_node", 1),
        tasks_per_node=kwds.get("tasks_per_node", 1),
        cpus_per_task=kwds.get("cpus_per_task", 1),
        slurm_mem=kwds.get("slurm_mem", 12)*1024, # Memory per CPU (arg is GB)
        slurm_time=kwds.get("slurm_time","00:05:00"),
        slurm_array_parallelism=kwds.get("slurm_array_parallelism", 1),
        slurm_account=kwds.get("slurm_account", "def-hinat"), # Replace with your account
        slurm_mail_user=kwds.get("slurm_mail_user", "xmootoo@gmail.com"),  # Email for notifications
        slurm_mail_type=kwds.get("slurm_mail_type", "ALL"),
        )

    # Submit the job
    job = executor.submit(run_script, virtual_env, modules, cli_input, script_path, cluster)
    console.log(f"Job submitted: {job.job_id}")

def load_configs(exp_name):
    """
    Load the configuration files for the experiment.

    Args:
        exp_name (str): The folder name for the experiment  within 'exp' directory. This can include single folders, such as
                         'patchtst_electricity_96' or subfolders for nested experiments, such as 'patchtst/electricity_96'
    Returns:
        cc_config (dict): The Compute Canada configuration.
        slurm_config (dict): The SLURM configuration.
    """

    # Compute Canada Configuration

    with open(os.path.join("./time_series_jepa/jobs/hpc", exp_name, "compute_canada.yaml")) as f:
        cc_config = yaml.safe_load(f)

    # SLURM Configuration
    with open(os.path.join("./time_series_jepa/jobs/hpc", exp_name, "slurm.yaml")) as f:
        slurm_config = yaml.safe_load(f)

    return cc_config, slurm_config


if __name__ == "__main__":

    # Job submission arguments
    parser = argparse.ArgumentParser(description="HPC jobs submission")
    parser.add_argument("exp", type=str, default="test", help="Experiment name dir from time_series_jepa/jobs/exp")
    args = parser.parse_args()

    # Load configs
    cc_config, slurm_config = load_configs(args.exp)
    cli_input = cc_config["job_name"]

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

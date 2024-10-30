import argparse
import os
from pathlib import Path
import torch
import torch.nn as nn
import neptune
from rich.console import Console
from sss.config.config import load_config
from sss.utils.models import get_model, forward_pass
import yaml
from pydantic import BaseModel, Field
from sss.utils.dataloading import get_loaders
from sss.utils.calibration import CalibrationModel
from sss.utils.classification import get_metrics, update_stats, get_logger_mapping
from sss.utils.logger import epoch_logger, log_pydantic

def load(device, run_id, calibration_model, console, calibrator_only=False):
    # Console and Neptune
    console = Console()
    api_token = os.getenv('NEPTUNE_API_TOKEN')

    # Directories
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent.parent.parent
    model_folder = project_root / "checkpoint" / "analysis"
    model_folder.mkdir(parents=True, exist_ok=True) # Create the model folder if it does not exist

    # Load model weights
    model_path = os.path.join(model_folder, "local_weights.pth")
    run = neptune.init_run(
        project="xmootoo/soz-localization",
        api_token=api_token,
        with_id=run_id,
        mode="read-only"
    )

    run["model_checkpoints/PatchTSTBlind_sl"].download(destination=model_path)


    # Load args
    exp_args_path = project_root / "time-series-jepa" / "sss" / "jobs" / "exp" / "open_neuro" / "binary" / "sss" / "best" / "args.yaml"
    exp_args = load_config(exp_args_path)
    exp_args.exp.calibration_model = calibration_model
    exp_args.exp.seed = run["parameters/exp/seed"].fetch()
    exp_args.open_neuro.patient_cluster = run["parameters/open_neuro/patient_cluster"].fetch()
    run.stop()


    train_clusters = ["jh", "umf", "pt", "ummc"]

    if exp_args.open_neuro.patient_cluster=="all":
        exp_args.open_neuro.all_clusters = True
        console.log(f"Using all clusters")
    elif exp_args.open_neuro.patient_cluster.startswith("loocv"):
        exp_args.open_neuro.all_clusters = False
        exp_args.open_neuro.loocv = True
        exp_args.open_neuro.test_clusters = [exp_args.open_neuro.patient_cluster[6:]]
        train_clusters.remove(exp_args.open_neuro.patient_cluster[6:])
        exp_args.open_neuro.train_clusters = train_clusters
        console.log(f"Using loocv with test cluster {exp_args.open_neuro.test_clusters}")
    else:
        exp_args.open_neuro.all_clusters = False
        console.log(f"Using single cluster {exp_args.open_neuro.patient_cluster}")


    # Load weights and model
    state_dict = torch.load(model_path, map_location=torch.device('cpu'))
    model = get_model(exp_args)

    # Adjust the state_dict keys if the model was trained with DDP
    if exp_args.ddp.ddp:
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        exp_args.ddp.ddp = False

    model.load_state_dict(state_dict)
    model = model.to(device)

    loaders = get_loaders(exp_args, dataset_class="classification", loader_type="all")

    # Calibration
    calibrator = CalibrationModel(0, exp_args)

    return model, loaders, calibrator, exp_args

def parse_output(args, output, batch, device):
    ch_ids = torch.tensor(0, device=device).unsqueeze(-1)
    u = torch.tensor(0, device=device).unsqueeze(-1)

    if args.open_neuro.ch_loss:
        y_hat, y, ch_ids = (output, batch[1], batch[2])
    else:
        y_hat, y = (output, batch[1].to(device))

    if y_hat.dim() > 1 and y_hat.size(-1) == 1: y_hat = y_hat.unsqueeze(-1)

    return (y_hat, y.to(device), ch_ids.to(device), u.to(device))

def evaluate(args, model, model_id, loader, device, return_outputs=False, calibrator=None):
    """
        Evaluate the model return evaluation loss and/or evaluation accuracy.
    """
    stats = dict()
    num_examples = len(loader.dataset) # Total number of examples across all ranks

    model.eval()
    with torch.no_grad():

        # Initialize Metrics
        total_loss = torch.tensor(0.0, device=device)
        total_mae = torch.tensor(0.0, device=device)
        all_logits = []
        all_labels = []
        all_ch_ids = []
        all_u = []

        for i, batch in enumerate(loader):
            output = forward_pass(args, model, batch, model_id, device)
            y_hat, y, ch_ids, u = parse_output(args, output, batch, device)
            all_logits.append(y_hat); all_labels.append(y); all_ch_ids.append(ch_ids); all_u.append(u)

    if return_outputs:
        return torch.cat(all_logits), torch.cat(all_labels), torch.cat(all_ch_ids)

    # # Window Metrics
    # window_metrics = get_metrics(args, all_logits, all_labels, mode="window", rank=0, calibrator=calibrator)
    # update_stats(stats, window_metrics, args.open_neuro.task, args.exp.other_metrics, False, 0)

    ch_metrics = get_metrics(args, all_logits, all_labels, all_ch_ids, all_u, mode="channel", rank=0, calibrator=calibrator)
    update_stats(stats, ch_metrics, args.open_neuro.task, args.exp.other_metrics, True, 0)

    return stats

def train_calibrator(args, loaders, calibrator, model, device):

    # Combine train and val loaders into one
    train_logits, train_targets, train_ch_ids = evaluate(
        args=args,
        model=model,
        model_id=args.exp.model_id,
        loader=loaders[0],
        device=device,
        return_outputs=True,
        calibrator=None
    )

    val_logits, val_targets, val_train_ch_ids = evaluate(
        args=args,
        model=model,
        model_id=args.exp.model_id,
        loader=loaders[1],
        device=device,
        return_outputs=True,
        calibrator=None
    )

    all_logits = torch.cat([train_logits, val_logits], dim=0).squeeze()
    all_targets = torch.cat([train_targets, val_targets], dim=0).squeeze()
    all_ch_ids = torch.cat([train_ch_ids, val_train_ch_ids], dim=0).squeeze()

    probs, targets, ch_ids = calibrator.compile_predictions(all_logits, all_targets, all_ch_ids)
    calibrator.train(probs, targets, ch_ids)


def log_stats(args, logger, stats, flag, acc, ch_acc, mode, task="binary"):
    modes = {"val": "Validation", "test": "Test"}
    Mode = modes[mode]
    mapping = get_logger_mapping(args.open_neuro.task)

    if acc:
        acc_value = stats["acc"]*100
        epoch_logger(args, logger, f"{flag}_{mode}/accuracy", acc_value)

        if args.exp.other_metrics:
            for key, value in mapping.items():
                logger[f"{flag}_{mode}/{value}"] = stats[key]

    if ch_acc:
        ch_acc_value = stats["ch_acc"]*100
        epoch_logger(args, logger, f"{flag}_{mode}/channel_accuracy", ch_acc_value)

        if args.exp.other_metrics:
            for key, value in mapping.items():
                logger[f"{flag}_{mode}/channel_{value}"] = stats[f"ch_{key}"]


if __name__ == "__main__":
    from datetime import datetime

    parser = argparse.ArgumentParser()
    parser.add_argument("patient_cluster", type=str, default="loocv_pt")
    parser.add_argument("calibration_model", type=str, default="isotonic_regression")
    input_args = parser.parse_args()


    patient_clusters = {
        "loocv_pt":["SOZ-9577", "SOZ-9575", "SOZ-9574", "SOZ-9573", "SOZ-9576"],
        "loocv_jh":["SOZ-9580", "SOZ-9582", "SOZ-9578", "SOZ-9579", "SOZ-9581"],
        "loocv_ummc": ["SOZ-9585", "SOZ-9586", "SOZ-9584", "SOZ-9583", "SOZ-9587"],
        }


    # Initiate console
    console = Console()

    # Load the model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    # Set calibration model and run ids for the appropriate patient cluster
    calibration_model = input_args.calibration_model
    run_ids = patient_clusters[input_args.patient_cluster]

    for run_id in run_ids:

        # Load model, args, data and train calibrator
        model, loaders, calibrator, args = load(device, run_id, calibration_model, console)
        train_calibrator(args, loaders, calibrator, model, device)
        console.log(f"Calibration model {calibration_model} trained for run {run_id}")

        date_time = datetime.now()
        run = neptune.init_run(
            project=args.exp.project_name,
            api_token=args.exp.api_token,
            custom_run_id=datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        )
        log_pydantic(run, args)

        # Evaluate and upload test results
        stats = evaluate(args, model, args.exp.model_id, loaders[2], device, calibrator=calibrator)
        log_stats(args, run, stats, "sl", acc=False, ch_acc=True, mode="test", task="binary")
        run.stop()
        console.log(f"Test results uploaded for run {run_id} using {calibration_model} calibration")

from typing import Callable, List, Optional, Protocol, Tuple, Union

# Torch stack
import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

# Torchmetrics
import torchmetrics

# Pydantic
from pydantic import BaseModel
from sklearn.calibration import CalibratedClassifierCV

# Scikit-learn
from sklearn.isotonic import IsotonicRegression
from sklearn.svm import SVC
from torchmetrics.classification import (
    AUROC,
    BinaryF1Score,
    MulticlassConfusionMatrix,
    MulticlassF1Score,
)


def binary_classification_metrics(
    preds: torch.Tensor,
    targets: torch.Tensor,
    probs: torch.Tensor,
    device: torch.device,
) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """
    Computes evaluation metrics relevant for binary classification, including AUC-ROC.

    Args:
        preds (torch.Tensor): Binary predictions from the model of shape (N,) where N is the number of examples.
                              These are assumed to be thresholded at 0.5.
        targets (torch.Tensor): Ground truth labels of shape (N,) where N is the number of examples.
        probs (torch.Tensor): Probability predictions for the positive class of shape (N,).
        device (torch.device): The device to perform computations on.

    Returns:
        metrics (torch.Tensor): A tensor comprising of the following metrics:
                                - Accuracy
                                - True Positive Rate (Sensitivity/Recall)
                                - True Negative Rate (Specificity)
                                - False Positive Rate
                                - False Negative Rate
                                - F1 Score
                                - AUC-ROC
        roc_data (tuple): A tuple containing (fpr, tpr, thresholds) for ROC curve plotting.
    """
    # Move tensors to the specified device
    preds = preds.to(device)
    targets = targets.to(device)
    probs = probs.to(device)

    # Accuracy
    acc = (preds == targets).float().mean()

    # Confusion matrix
    conf_matrix = MulticlassConfusionMatrix(num_classes=2).to(device)
    cm = conf_matrix(preds, targets).to(device)
    tp, fp, fn, tn = cm[1, 1], cm[0, 1], cm[1, 0], cm[0, 0]

    # Compute metrics (avoid division by zero)
    tpr = torch.where(
        tp + fn != 0, tp / (tp + fn), torch.tensor(0.0, device=device)
    )  # True Positive Rate / Sensitivity / Recall
    tnr = torch.where(
        tn + fp != 0, tn / (tn + fp), torch.tensor(0.0, device=device)
    )  # True Negative Rate / Specificity
    fpr = torch.where(
        fp + tn != 0, fp / (fp + tn), torch.tensor(0.0, device=device)
    )  # False Positive Rate
    fnr = torch.where(
        fn + tp != 0, fn / (fn + tp), torch.tensor(0.0, device=device)
    )  # False Negative Rate

    # F1 Score
    f1 = BinaryF1Score().to(device)
    f1_score = f1(preds, targets).to(device)

    # AUC-ROC
    auroc = AUROC(task="binary").to(device)
    auroc_score = auroc(probs, targets).to(device)

    # Stack all metrics including AUC-ROC
    metrics = torch.stack([acc, tpr, tnr, fpr, fnr, f1_score, auroc_score])

    return metrics


def multi_classification_metrics(preds, targets, probs, device):
    """
    Computes multi-class classification metrics for OpenNeuro dataset. One set of metrics for SOZ vs non-SOZ
    and another set for Positive Outcome vs Negative Outcome (i.e. two confusion matrices). Key:
        #   0 - no SOZ, positive outcome
        #   1 - SOZ, positive outcome
        #   2 - no SOZ, negative outcome
        #   3 - SOZ, negative outcome
    """
    targets = targets.long()

    # Accuracy
    multi_acc = (preds == targets).float().mean().unsqueeze(0)

    # Compute metrics for SOZ vs non-SOZ
    preds_soz = torch.where(
        (preds == 1) | (preds == 3), torch.tensor(1), torch.tensor(0)
    )
    targets_soz = torch.where(
        (targets == 1) | (targets == 3), torch.tensor(1), torch.tensor(0)
    )
    probs_soz = probs[:, 1] + probs[:, 3]
    soz_metrics = binary_classification_metrics(
        preds_soz, targets_soz, probs_soz, device
    )  # TODO:

    # Compute metrics for Positive Outcome vs Negative Outcome
    preds_outcome = torch.where(
        (preds == 0) | (preds == 1), torch.tensor(1), torch.tensor(0)
    )
    targets_outcome = torch.where(
        (targets == 0) | (targets == 1), torch.tensor(1), torch.tensor(0)
    )
    probs_outcome = probs[:, 0] + probs[:, 1]
    outcome_metrics = binary_classification_metrics(
        preds_outcome, targets_outcome, probs_outcome, device
    )

    # Multiclass F1 (Macro-averaged F1)
    f1 = MulticlassF1Score(num_classes=4).to(device)
    f1_score = f1(preds, targets).unsqueeze(0).to(device)

    # Multiclass AUROC (Macro-averaged AUROC)
    auroc = AUROC(task="multiclass", num_classes=4).to(device)
    auroc_score = auroc(probs, targets).unsqueeze(0).to(device)

    # Stack and return metrics
    metrics = torch.cat(
        [multi_acc, soz_metrics, outcome_metrics, f1_score, auroc_score]
    ).to(device)

    return metrics


def compute_accuracy(logits, target, args, calibrator=None):
    """
    Computes the accuracy of the model's predictions.

    Args:
        logits (torch.Tensor): The model's logits of shape (batch_size, 1) for binary classification or
                               (batch_size, num_classes) for multiclass classification.
        target (torch.Tensor): The target labels of shape (batch_size,).
    Returns:
        acc (float): The accuracy of the model's predictions for the batch.

    """
    # Remove the singleton dimensions

    batch_size = logits.size(0)
    if batch_size != 1:
        logits = torch.squeeze(logits)
        targets = torch.squeeze(target).to(logits.device).view(-1)
    else:
        targets = target.to(logits.device)

    if args.open_neuro.task == "multi":
        if logits.dim() == 3:
            logits = logits.squeeze(0)
        elif logits.dim() == 1:
            logits = logits.unsqueeze(0)

    # Compute number of correct predictions
    if args.open_neuro.task == "binary":
        probs = F.sigmoid(logits)
        # preds = (probs > args.exp.thresh).float()
        num_classes = 2
    elif args.open_neuro.task == "multi":
        probs = F.softmax(logits, dim=-1)
        # preds = probs.argmax(dim=-1)
        num_classes = logits.size(-1)
    else:
        raise ValueError(
            f"Invalid loss_type: {args.open_neuro.task}. Use 'binary' or 'multi'."
        )

    # Calibrate the probabilities
    if calibrator and args.exp.calibration_type == "window":
        probs = calibrator.calibrate(probs)

    if args.open_neuro.task == "binary":
        preds = (probs > args.exp.thresh).float()
    elif args.open_neuro.task == "multi":
        preds = probs.argmax(dim=-1)

    if args.exp.other_metrics:
        if args.open_neuro.task == "binary":
            metrics = binary_classification_metrics(
                preds, targets, probs, logits.device
            )
        elif args.open_neuro.task == "multi":
            metrics = multi_classification_metrics(preds, targets, probs, logits.device)
        return metrics
    else:
        return (preds == targets).float().mean()


def compute_channel_accuracy(logits, target, ch_ids, u, args, calibrator=None):
    """
    Args:
        logits: Logits tensor of shape (batch_size, 1) if num_classes = 2 and (batch_size, num_classes) if num_classes > 2.
        target: Tensor of shape (batch_size, 1) with integer values corresponding to each class.
        ch_ids: Tensor of shape (batch_size, 1) with integer values corresponding to each channel ID.
        loss_type: Type of loss function used to compute the predictions. Use 'BCE' for binary classification and 'CE' for multiclass classification.
        thresh: Threshold value for binary classification predictions.
    Returns:
        acc: Accuracy of the model for channel predictions.
        preds: Predictions for each channel.
        targets: Ground truth labels for each channel.
        channel_df: Pandas DataFrame with channel-level predictions and probabilities.
    """

    target = target.to(logits.device)
    ch_ids = ch_ids.to(logits.device)

    if args.open_neuro.task == "multi":
        if logits.dim() == 3:
            logits = logits.squeeze(0)
        elif logits.dim() == 1:
            logits = logits.unsqueeze(0)

    # Compute predictions for each channel
    if args.open_neuro.ch_loss_type == "BCE":
        window_probs = F.sigmoid(logits).squeeze()
        num_classes = 2
    elif args.open_neuro.ch_loss_type == "CE":
        window_probs = F.softmax(logits, dim=-1)
        num_classes = logits.size(-1)
    else:
        raise ValueError(
            f"Invalid loss_type: {args.open_neuro.ch_loss_type}. Use 'BCE' or 'CE'."
        )

    window_probs_raw = window_probs
    window_probs_calibrated = window_probs

    if calibrator and args.exp.calibration_type == "window":
        window_probs_calibrated = calibrator.calibrate(window_probs_raw)

    # Compute predictions
    num_unique_ch = len(torch.unique(ch_ids))
    raw_preds = []
    raw_targets = []
    raw_probs = []
    calibrated_preds_list = []
    calibrated_probs_list = []
    channel_ids = []
    channel_lengths = []
    for ch_id in torch.unique(ch_ids):
        ch_id = ch_id.item()
        mask = (ch_ids == ch_id).squeeze()  # Create mask for the channel
        ch_probs_raw = window_probs_raw[
            mask
        ]  # Probability for each window in the channel (sigmoid or softmax)
        ch_probs_calibrated = window_probs_calibrated[mask]

        if args.open_neuro.ch_loss_type == "BCE":
            mean_prob_raw = ch_probs_raw.mean().unsqueeze(
                -1
            )  # Mean probability for the channel (1,)
            mean_prob_calibrated = ch_probs_calibrated.mean().unsqueeze(-1)
            ch_pred_raw = (
                mean_prob_raw > args.exp.thresh
            ).float()  # Predictions for the channel: 1 if mean probability > thresh, 0 otherwise
            ch_pred_calibrated = (mean_prob_calibrated > args.exp.thresh).float()
        elif args.open_neuro.ch_loss_type == "CE":
            mean_prob_raw = ch_probs_raw.mean(
                0
            )  # Mean probabilities over the channel (num_classes,)
            mean_prob_calibrated = ch_probs_calibrated.mean(0)
            ch_pred_raw = mean_prob_raw.argmax().unsqueeze(-1)
            ch_pred_calibrated = mean_prob_calibrated.argmax().unsqueeze(-1)

        ch_target = target[mask][0]  # Grab the target for the channel

        raw_probs.append(mean_prob_raw)
        raw_preds.append(ch_pred_raw)
        calibrated_probs_list.append(mean_prob_calibrated)
        calibrated_preds_list.append(ch_pred_calibrated)
        raw_targets.append(ch_target)
        channel_ids.append(ch_id)
        channel_lengths.append(int(mask.sum().item()))

    # Results
    raw_preds = torch.stack(raw_preds).view(-1)
    raw_targets = torch.stack(raw_targets).view(-1)
    raw_probs = torch.stack(raw_probs)
    targets = raw_targets

    calibrated_probs = torch.stack(calibrated_probs_list)
    calibrated_preds = torch.stack(calibrated_preds_list).view(-1)

    if calibrator and args.exp.calibration_type == "channel":
        calibrated_probs = calibrator.calibrate(calibrated_probs)
        if not torch.is_tensor(calibrated_probs):
            calibrated_probs = torch.as_tensor(
                calibrated_probs, device=raw_probs.device
            )
        if args.open_neuro.ch_loss_type == "BCE":
            calibrated_preds = (calibrated_probs > args.exp.thresh).float()
        elif args.open_neuro.ch_loss_type == "CE":
            calibrated_preds = calibrated_probs.argmax(dim=-1)

    probs = calibrated_probs
    preds = calibrated_preds
    raw_probs_cpu = raw_probs.detach().cpu()
    raw_preds_cpu = raw_preds.detach().cpu()
    raw_targets_cpu = raw_targets.detach().cpu()

    calibrated_probs_cpu = (
        calibrated_probs.detach().cpu()
        if torch.is_tensor(calibrated_probs)
        else torch.as_tensor(calibrated_probs).detach().cpu()
    )
    calibrated_preds_cpu = (
        calibrated_preds.detach().cpu()
        if torch.is_tensor(calibrated_preds)
        else torch.as_tensor(calibrated_preds).detach().cpu()
    )

    def _to_python(val):
        if torch.is_tensor(val):
            if val.numel() == 1:
                return val.item()
            return val.tolist()
        return val

    channel_df = pd.DataFrame(
        {
            "ch_id": channel_ids,
            "length": channel_lengths,
            "ch_prob": [_to_python(p) for p in raw_probs_cpu],
            "ch_prob_calibrated": [_to_python(p) for p in calibrated_probs_cpu],
            "ch_pred": [_to_python(p) for p in raw_preds_cpu],
            "ch_pred_calibrated": [_to_python(p) for p in calibrated_preds_cpu],
            "target": [_to_python(t) for t in raw_targets_cpu],
        }
    )
    channel_df["length_in_seconds"] = (channel_df["length"] * 12 + 12) / 1000

    if args.exp.other_metrics:
        if args.open_neuro.task == "binary":
            metrics = binary_classification_metrics(
                preds, targets, probs, logits.device
            )
        elif args.open_neuro.task == "multi":
            metrics = multi_classification_metrics(preds, targets, probs, logits.device)
        return metrics, channel_df
    else:
        return (preds == targets).float().mean(), channel_df


def sync(args):
    """
    Synchronizes all processes for Distributed Data Parallel (DDP).
    """
    if args.ddp.ddp:
        dist.barrier()
        # print("Synchronizing (classification.py)")


def gather_tensor(tensor):
    gathered = [torch.zeros_like(tensor) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered, tensor)
    return torch.cat(gathered)


def reduce_tensor(tensor):
    return dist.all_reduce(tensor, op=dist.ReduceOp.SUM)


def get_metrics(
    args: BaseModel,
    logits: Union[torch.Tensor, List[torch.Tensor]],
    target: Union[torch.Tensor, List[torch.Tensor]],
    ch_ids: Union[torch.Tensor, List[torch.Tensor]] = None,
    u: Union[torch.Tensor, List[torch.Tensor]] = None,
    mode: str = "channel",
    rank: int = 0,
    calibrator=None,
) -> torch.Tensor:
    # Window metrics
    if mode == "window":
        if args.exp.batchwise_metrics:
            assert all(isinstance(x, list) for x in [logits, target]), (
                "Logits and targets must be lists."
            )
            num_batches = len(logits)
            num_examples = torch.tensor(0.0, device=logits[0].device)
            total_metrics = None

            for i in range(num_batches):
                metrics = compute_accuracy(logits[i], target[i], args, calibrator)
                batch_size = logits[i].size(0)
                total_metrics = (
                    (total_metrics + metrics * batch_size)
                    if total_metrics is not None
                    else metrics * batch_size
                )
                num_examples += batch_size

            if args.ddp.ddp:
                sync(args)
                reduce_tensor(total_metrics)
                sync(args)
                reduce_tensor(num_examples)
            sync(args)
            return total_metrics / num_examples.item()
        else:
            logits = torch.cat(logits)
            target = torch.cat(target)
            if args.ddp.ddp:
                sync(args)
                logits = gather_tensor(logits)
                target = gather_tensor(target)
            return (
                compute_accuracy(logits, target, args, calibrator)
                if rank == 0
                else None
            )

    # Channel metrics
    elif mode == "channel":
        if args.exp.batchwise_metrics:
            assert all(isinstance(x, list) for x in [logits, target, ch_ids, u]), (
                "Logits, target, ch_ids, and u must be lists."
            )
            num_batches = len(logits)
            total_metrics = None
            num_examples = torch.tensor(0.0, device=logits[0].device)
            channel_dfs = []

            for i in range(num_batches):
                metrics, channel_df = compute_channel_accuracy(
                    logits[i], target[i], ch_ids[i], u[i], args, calibrator
                )
                channel_dfs.append(channel_df)
                batch_size = logits[i].size(0)
                total_metrics = (
                    (total_metrics + metrics * batch_size)
                    if total_metrics is not None
                    else metrics * batch_size
                )
                num_examples += batch_size

            if args.ddp.ddp:
                sync(args)
                reduce_tensor(total_metrics)
                sync(args)
                reduce_tensor(num_examples)
            sync(args)
            final_metrics = total_metrics / num_examples.item()
            final_channel_df = (
                pd.concat(channel_dfs, ignore_index=True)
                if len(channel_dfs) > 0
                else pd.DataFrame()
            )
            return final_metrics, final_channel_df
        else:
            # Get all outputs/targets/ch_ids/etc from all batches
            logits = torch.cat(logits)
            target = torch.cat(target)
            ch_ids = torch.cat(ch_ids)
            u = torch.cat(u)
            if args.ddp.ddp:
                sync(args)
                logits = gather_tensor(logits)
                target = gather_tensor(target)
                ch_ids = gather_tensor(ch_ids)
                u = gather_tensor(u)
            return (
                compute_channel_accuracy(logits, target, ch_ids, u, args, calibrator)
                if rank == 0
                else None
            )
    else:
        raise ValueError(f"Invalid mode: {mode}. Use 'window' or 'channel'.")


def update_stats(
    stats, metrics, task="binary", other_metrics=False, channel=False, rank=0
):
    if rank != 0:
        return
    if isinstance(metrics, tuple):
        metrics, _ = metrics
    binary = ["tpr", "tnr", "fpr", "fnr", "f1", "auroc"]
    multi = [
        "acc_soz",
        "tpr_soz",
        "tnr_soz",
        "fpr_soz",
        "fnr_soz",
        "f1_soz",
        "auroc_soz",
        "acc_outcome",
        "tpr_outcome",
        "tnr_outcome",
        "fpr_outcome",
        "fnr_outcome",
        "f1_outcome",
        "auroc_outcome",
        "f1_multi",
        "auroc_multi",
    ]

    keys = binary if task == "binary" else multi

    if channel:
        keys = ["ch_" + s for s in keys]

    acc_key = "ch_acc" if channel else "acc"
    stats[acc_key] = metrics[0].item()

    if other_metrics:
        for i, key in enumerate(keys):
            stats[key] = metrics[i + 1].item()


def get_logger_mapping(task="binary"):
    if task == "binary":
        mapping = {
            "tpr": "true_positive_rate",
            "tnr": "true_negative_rate",
            "fpr": "false_positive_rate",
            "fnr": "false_negative_rate",
            "f1": "f1_score",
            "auroc": "auroc",
        }
    elif task == "multi":
        mapping = {
            "acc_soz": "accuracy_soz",
            "tpr_soz": "true_positive_rate_soz",
            "tnr_soz": "true_negative_rate_soz",
            "fpr_soz": "false_positive_rate_soz",
            "fnr_soz": "false_negative_rate_soz",
            "f1_soz": "f1_score_soz",
            "auroc_soz": "auroc_soz",
            "acc_outcome": "accuracy_outcome",
            "tpr_outcome": "true_positive_rate_outcome",
            "tnr_outcome": "true_negative_rate_outcome",
            "fpr_outcome": "false_positive_rate_outcome",
            "fnr_outcome": "false_negative_rate_outcome",
            "f1_outcome": "f1_score_outcome",
            "auroc_outcome": "auroc_outcome",
            "f1_multi": "f1_score_multi",
            "auroc_multi": "auroc_multi",
        }
    else:
        raise ValueError("Invalid task. Please select 'binary' or 'multi'")

    return mapping


if __name__ == "__main__":
    # Inputs
    n = 10

    # preds = torch.randint(2, (n,))
    # targets = torch.randint(2, (n,))
    # preds = torch.tensor([0, 2, 3, 1, 0, 0, 1, 2, 3], dtype=torch.int64)
    # targets = torch.tensor([0, 1, 3, 2, 0, 1, 1, 2, 1], dtype=torch.int64)
    # print(f"x: {preds}")
    # print(f"y: {targets}")
    # num_classes = 4
    # device = torch.device("cpu")

    # metrics = multi_classification_metrics(preds, targets, device)

    # # Print the results
    # metric_names = ["ACC", "TPR", "TNR", "FPR", "FNR", "F1"] + ["ACC", "TPR", "TNR", "FPR", "FNR", "F1"]
    # print(metrics)
    # for name, value in zip(metric_names, metrics):
    #     print(f"{name}: {value.item():.4f}")

    from sss.config.config import Global

    args = Global()
    args.exp.batchwise_metrics = True
    args.exp.u_weight = False
    args.open_neuro.task = "binary"
    args.open_neuro.ch_loss_type = "BCE"
    args.exp.other_metrics = True

    # Function to create a test case
    num_batches = 10
    good_logits = [
        torch.tensor([-20, 300, -50]),
        torch.tensor([-500, -500]),
        torch.tensor([1000, -300]),
    ]
    bad_logits = [-x for x in good_logits]
    target = [torch.tensor([0, 1, 0]), torch.tensor([0, 0]), torch.tensor([1, 0])]
    ch_ids = [torch.tensor([0, 1, 0]), torch.tensor([0, 0]), torch.tensor([1, 0])]
    u = [None for _ in range(num_batches)]

    # Call your function with this test case
    result = get_metrics(args, bad_logits, target, ch_ids, u, mode="channel")
    print(result)
    print("[acc, tpr, tnr, fpr, fnr, f1_score, au_roc]")

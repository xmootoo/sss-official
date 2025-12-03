import datetime
import gc
import json
import os
import random
import shutil
import sys

# Timing
import time
import warnings
from typing import Any

# Logger
import neptune

# Torch
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn

# Pydantic
from pydantic import BaseModel

# Rich console
from rich.console import Console
from rich.pretty import pprint
from torch.distributed import destroy_process_group, init_process_group

# DDP
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

# Custom Modules
from sss.layers.patchtst_original.utils import adjust_learning_rate
from sss.utils.calibration import CalibrationModel
from sss.utils.classification import (
    binary_classification_metrics,
    get_logger_mapping,
    get_metrics,
    multi_classification_metrics,
    update_stats,
)
from sss.utils.dataloading import get_loaders
from sss.utils.logger import epoch_logger, format_time_dynamic, log_pydantic
from sss.utils.models import (
    compute_loss,
    ema_momentum_scheduler,
    forward_pass,
    get_criterion,
    get_downstream_model,
    get_downstream_optim,
    get_model,
    get_optim,
    get_scheduler,
    model_update,
)
from sss.utils.train import EarlyStopping

warnings.filterwarnings(
    "ignore", message="h5py not installed, hdf5 features will not be supported."
)


class Experiment:
    def __init__(self, args):
        self.args = args
        self.tuning_score = 0
        self.start_time = time.time()

    def run(self, rank=0, world_size=1):
        # Rank and World Size
        self.rank = rank
        self.world_size = world_size

        # Probability calibrator
        self.calibrator = None

        # Rich Console
        self.console = Console()

        # Reproducibility
        self.generator = torch.Generator().manual_seed(self.args.exp.seed)
        torch.manual_seed(self.args.exp.seed)  # CPU
        np.random.seed(self.args.exp.seed)  # Numpy
        random.seed(self.args.exp.seed)  # Python
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.args.exp.seed)  # GPU
            torch.cuda.manual_seed_all(self.args.exp.seed)  # multi-GPU

        # Logging
        self.init_logger()

        # DDP
        if self.args.ddp.ddp:
            self.init_ddp(rank, world_size)

        # Initialize Device
        self.init_device()

        # Learning Type
        if self.args.exp.learning_type == "ssl":
            if not self.args.downstream.ignore_pretrain:
                self.pretrain()
            self.downstream()
        elif self.args.exp.learning_type == "sl":
            self.supervised_train()

        # Stop Logger
        if self.rank == 0:
            if self.args.exp.neptune:
                self.logger.stop()
            else:
                # Save offline logging to JSON file
                with open(self.log_file, "w") as f:
                    json.dump(self.logger, f, indent=2)

        # Cleanup DDP Processes
        if self.args.ddp.ddp:
            destroy_process_group()
            torch.cuda.empty_cache()

    def init_ddp(self, rank, world_size):
        """
        Initialize distributive data parallel (DDP) for the given rank.
        """
        init_process_group(backend="nccl", rank=rank, world_size=world_size)
        torch.cuda.set_device(rank)  # Explicitly set the current CUDA device
        self.console.log(f"Initialized rank {rank} (DDP).")
        self.sync()
        self.print_master(
            f"Initialized Distributed Data Parallel aross {self.world_size} devices."
        )

    def init_device(self):
        """
        Initialize CUDA (or MPS) devices.
        """
        if self.args.ddp.ddp:
            self.device = torch.device(f"cuda:{self.rank}")
        elif self.args.exp.mps:
            self.device = torch.device(
                "mps" if torch.backends.mps.is_available() else "cpu"
            )
            self.console.log(f"MPS hardware acceleration activated.")
        elif not torch.cuda.is_available():
            self.device = torch.device("cpu")
            self.console.log("CUDA not available. Running on CPU.")
        else:
            self.device = torch.device(f"cuda:{self.args.exp.gpu_id}")
        self.sync()
        self.console.log(f"Rank {self.rank} device initialized to: {self.device}")

    def init_dataloaders(self, learning_type="sl", loader_type="train"):
        """
        Initialize the dataloaders depending on the learning type and loader type.

        Args:
            loader_type (str): Options: "train", "test", "all". "train" returns train and val loaders. "test" returns test loader. "all" returns all loaders.
            learning_type (str): Options: "sl", "ssl", "downstream". "sl" is supervised learning. "ssl" is self-supervised learning. "downstream" is downstream learning.
        """

        # Scikit-learn pipeline
        if self.args.exp.sklearn:
            self.seq_load(loader_type="all", learning_type="sl")
            self.print_master(
                f"{len(self.train_loader)} train samples. {len(self.val_loader)} validation samples. {len(self.test_loader)} test samples."
            )
            return

        # Deep learning (PyTorch) models
        if self.args.data.seq_load:
            self.seq_load(loader_type, learning_type)
        elif self.args.data.rank_seq_load:
            self.rank_seq_load(loader_type, learning_type)
        else:
            raise ValueError(
                f"Invalid dataloading option. Please set either data.seq_load or data.rank_seq_load to {True}."
            )

    def rank_seq_load(self, loader_type="train", learning_type="sl"):
        self.print_master(f"Running rank sequential dataloading ({loader_type}).")
        for i in range(self.world_size):
            self.sync()
            if self.rank == i:
                self.seq_load(loader_type, learning_type)
            self.sync()

    def seq_load(self, loader_type="train", learning_type="sl"):
        self.console.log(
            f"Running sequential dataloading on rank {self.rank} ({loader_type})."
        )
        self.free_memory()
        loaders = get_loaders(
            self.args,
            learning_type,
            self.generator,
            self.rank,
            self.world_size,
            self.args.sl.dataset_class,
            loader_type,
        )

        if loader_type == "train":
            self.train_loader, self.val_loader = loaders[:2]
            self.print_master(
                f"{len(self.train_loader.dataset)} train samples. {len(self.val_loader.dataset)} validation samples."
            )
        elif loader_type == "test":
            self.test_loader = loaders[0]
            self.print_master(f"{len(self.test_loader.dataset)} test samples.")
        elif loader_type == "all":
            self.train_loader, self.val_loader, self.test_loader = loaders[:3]
            if not self.args.exp.sklearn:
                self.print_master(
                    f"{len(self.train_loader.dataset)} train samples. {len(self.val_loader.dataset)} validation samples. {len(self.test_loader.dataset)} test samples."
                )
        else:
            raise ValueError("Invalid loader type.")

        if self.args.data.median_seq_len:
            self.args.data.seq_len = loaders[-1]
            self.logger["parameters/data/seq_len"] = loaders[-1]
            self.print_master(f"Sequence set to the median: {self.args.data.seq_len}.")

    def free_memory(self):
        for k in ["train", "val", "test"]:
            if hasattr(self, f"{k}_loader"):
                loader = getattr(self, f"{k}_loader")
                if hasattr(loader.dataset, "close"):
                    loader.dataset.close()
                del loader
                delattr(self, f"{k}_loader")

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def init_model(self):
        """
        Initialize the model.
        """
        self.model = get_model(self.args, self.generator)
        if self.args.exp.sklearn:
            self.print_master(
                f"Scikit-learn model {self.args.exp.model_id} initialized."
            )
            return
        elif self.args.ddp.ddp:
            torch.cuda.set_device(self.rank)
            self.model.to(self.rank)
            self.model = DDP(
                self.model,
                device_ids=[self.rank],
                find_unused_parameters=self.args.ddp.find_unused_parameters,
            )
        else:
            self.model.to(self.device)
        num_params = self.count_parameters()
        self.logger["parameters/sl/num_params"] = num_params
        self.print_master(
            f"{self.args.exp.model_id} model initialized with {num_params:,} parameters."
        )

    def init_optimizer(self):
        """
        Initialize the optimizer
        """
        self.optimizer = get_optim(self.args, self.model, self.args.sl.optimizer)
        self.print_master(f"{self.args.sl.optimizer} optimizer initialized.")

    def init_logger(self):
        """
        Initialize the logger
        """

        self.log_dir = os.path.join(
            "logs",
            f"{self.args.exp.ablation_id}_{self.args.exp.model_id}_{self.args.exp.id}",
            str(self.args.exp.seed),
        )
        if self.args.exp.neptune:
            # Initialize Neptune run with the time-based ID
            self.logger = neptune.init_run(
                project=self.args.exp.project_name,
                api_token=self.args.exp.api_token,
                custom_run_id=self.args.exp.time,
            )
            self.print_master("Neptune logger initialized.")
        else:
            self.logger = dict()
            self.log_file = os.path.join(self.log_dir, "log.json")

            if os.path.exists(self.log_dir):
                self.print_master(f"Using existing log directory: {self.log_dir}")
            else:
                os.makedirs(self.log_dir, exist_ok=True)
                self.print_master(f"Created new log directory: {self.log_dir}")

            self.print_master("Offline logger initialized.")

        log_pydantic(self.logger, self.args, "parameters")

    def init_earlystopping(self, path: str):
        self.early_stopping = EarlyStopping(
            patience=self.args.early_stopping.patience,
            verbose=self.args.early_stopping.verbose,
            delta=self.args.early_stopping.delta,
            path=path,
        )

    def sklearn_train(self, model, train_data, train_targets):
        self.print_master(f"Training scikit-learn model {self.args.exp.model_id}.")
        model.fit(train_data, train_targets)
        self.print_master("Sklearn training completed.")

    def sklearn_eval(
        self,
        model,
        criterion,
        inputs,
        targets,
        acc=False,
        mae=False,
        ch_acc=False,
        flag="sl",
    ):
        stats = dict()

        # Compute predictions
        num_examples = inputs.shape[0]
        preds = torch.from_numpy(model.predict(inputs))
        probs = torch.from_numpy(model.predict_proba(inputs)).to(self.device)

        if self.args.open_neuro.task == "binary":
            probs = probs[
                :, 1
            ]  # Get positive class value. Otherwise use multiclass values

        targets = torch.from_numpy(targets).to(self.device)

        # Loss
        if isinstance(criterion, nn.BCELoss):
            probs = probs.double()
            targets = targets.double()
        elif self.args.open_neuro.task == "multi":
            targets = targets.long()

        print(f"Probs: {probs.shape}, Targets: {targets.shape}")

        stats["loss"] = criterion(probs, targets).item()

        if mae:
            mae_loss = nn.L1Loss()
            stats["mae"] = mae_loss(preds, targets).item()

        if self.args.exp.other_metrics:
            if self.args.open_neuro.task == "binary":
                single_metrics = binary_classification_metrics(
                    preds, targets, probs, self.device
                )  # Shape: (7,)
            elif self.args.open_neuro.task == "multi":
                single_metrics = multi_classification_metrics(
                    preds, targets, probs, self.device
                )  # Shape: (15,)
            else:
                raise ValueError("Invalid task.")
        elif acc:
            single_metrics = ((preds == targets).float().mean()).unsqueeze(
                0
            )  # Shape: (1,)
        else:
            single_metrics = None

        # Window Metrics
        if acc or ch_acc:
            ch_acc = True if self.args.data.full_channels else ch_acc
            acc = False if self.args.data.full_channels else acc
            update_stats(
                stats,
                single_metrics,
                self.args.open_neuro.task,
                self.args.exp.other_metrics,
                ch_acc,
            )
            self.log_stats(stats, flag, mae, acc, ch_acc, mode="test")

        # Clear all tensors before returning stats
        del single_metrics
        torch.cuda.empty_cache()

    def train(
        self,
        model,
        model_id,
        optimizer,
        train_loader,
        best_model_path,
        criterion,
        val_loader=None,
        scheduler=None,
        flag="sl",
        ema=None,
        mae=False,
        acc=False,
        early_stopping=False,
        ch_acc=False,
    ):
        """
            Trains a model.

        Args:
            model (nn.Module): The model to train.
            model_id (str): The model ID.
            optimizer (torch.optim): The optimizer to use.
            train_loader (torch.utils.data.DataLoader): The training data.
            best_model_path (str): The path to save the best model.
            criterion (torch.nn): The loss function.
            val_loader (torch.utils.data.DataLoader): The validation data.
            scheduler (torch.optim.lr_scheduler): The learning rate scheduler.
            flag (str): The type of learning. Options: "sl", "ssl", "downstream".
        """

        # Scikit-learn pipeline
        if self.args.exp.sklearn:
            self.sklearn_train(model, train_loader.x, train_loader.y)
            return

        # Deep learning (PyTorch) pipeline
        num_examples = len(train_loader.dataset)
        self.print_master(f"Training on {num_examples} examples...")

        if early_stopping:
            self.init_earlystopping(best_model_path)
            self.print_master("Early stopping initialized.")

        # Synchronize before training starts
        self.sync()
        self.best_val_metric = float("inf")

        # <--------------- Training --------------->
        for epoch in range(eval(f"self.args.{flag}.epochs")):
            model.train()
            total_loss = torch.tensor(0.0, device=self.device)
            running_loss = torch.tensor(0.0, device=self.device)
            running_num_examples = torch.tensor(0.0, device=self.device)
            start_time = time.time()

            for i, batch in enumerate(train_loader):
                optimizer.zero_grad()
                output = forward_pass(self.args, model, batch, model_id, self.device)
                loss = compute_loss(
                    output, batch, criterion, model_id, self.args, self.device
                )

                # Metrics
                num_batch_examples = torch.tensor(batch[0].shape[0], device=self.device)
                total_loss += loss * num_batch_examples
                running_loss += loss * num_batch_examples
                running_num_examples += num_batch_examples

                # Update model parameters
                alpha = next(ema) if ema else 0.966
                model_update(model, loss, optimizer, model_id, alpha)

                # Periodic Logging
                if (i + 1) % 100 == 0:
                    # Aggregate loss metrics across all GPUs
                    self.print_rank(
                        f"Train Loss before all_reduce: {running_loss}. Rank: {self.rank}"
                    )
                    self.print_rank(
                        f"Num examples before all_reduce: {running_num_examples}. Rank: {self.rank}"
                    )
                    loss_tensor = running_loss.to(self.device)
                    num_examples_tensor = running_num_examples.to(self.device)

                    if self.args.ddp.ddp:
                        self.sync()
                        dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
                        dist.all_reduce(num_examples_tensor, op=dist.ReduceOp.SUM)
                        self.print_rank(
                            f"Train Loss after all_reduce: {loss_tensor.item()}. Rank: {self.rank}"
                        )
                        self.print_rank(
                            f"Num examples after all_reduce: {num_examples_tensor.item()}. Rank {self.rank}"
                        )

                    end_time = time.time()

                    # Only rank 0 prints and logs details
                    if self.rank == 0:
                        average_loss = loss_tensor.item() / num_examples_tensor.item()
                        self.print_master(
                            f"[Epoch {epoch}, Batch ({i + 1}/{len(train_loader)})]: {end_time - start_time:.3f}s. Loss: {average_loss:.6f}"
                        )
                        self.print_master(
                            f"EMA decay rate: {alpha}"
                        ) if flag == "ssl" else None

                    # Reset trackers
                    self.sync()
                    running_loss = torch.tensor(0.0, device=self.device)
                    running_num_examples = torch.tensor(0.0, device=self.device)
                    start_time = time.time()

                if scheduler:
                    if self.args.exp.model_id == "PatchTSTOG":
                        self.sync()
                        new_lr = adjust_learning_rate(
                            optimizer, scheduler, epoch, self.args
                        )
                        self.print_master(f"Learning rate adjusted to: {new_lr}") if (
                            (i + 1) % 100 == 0 or i == len(train_loader) - 1
                        ) else None
                    self.sync()
                    scheduler.step()

            # Average Loss + Logging
            if self.args.ddp.ddp:
                self.sync()
                dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
            if self.rank == 0:
                epoch_loss = total_loss.item() / num_examples
                self.print_master(f"Epoch {epoch}. Training loss: {epoch_loss:.6f}.")
                epoch_logger(self.args, self.logger, f"{flag}_train/loss", epoch_loss)

            # <--------------- Validation --------------->
            if val_loader:
                self.validate(
                    model,
                    val_loader,
                    model_id,
                    criterion,
                    flag,
                    mae,
                    acc,
                    epoch,
                    best_model_path,
                    early_stopping,
                    ch_acc,
                )

            self.sync()

            # Early stopping
            if early_stopping:
                if self.rank == 0 and self.early_stopping.early_stop:
                    self.print_master(f"EarlyStopping activated, ending training.")
                    # If DDP is enabled, broadcast the early stopping signal to all GPUs
                    if self.args.ddp.ddp:
                        should_stop_tensor = torch.tensor(
                            1, dtype=torch.int, device=self.device
                        )
                        dist.broadcast(should_stop_tensor, src=0)
                    break
                elif self.args.ddp.ddp:
                    # On non-master GPUs, receive the broadcast to stop
                    should_stop_tensor = torch.tensor(
                        0, dtype=torch.int, device=self.device
                    )
                    dist.broadcast(should_stop_tensor, src=0)
                    if should_stop_tensor.item() == 1:
                        break

                # Synchronize after early stopping
                self.sync()

            # Checkpoint (online)
            if self.rank == 0:
                if self.args.exp.neptune:
                    pass
                else:
                    run_time = time.time() - self.start_time
                    self.logger["parameters/running_time"] = format_time_dynamic(
                        run_time
                    )

            self.sync()

    def validate(
        self,
        model,
        val_loader,
        model_id,
        criterion,
        flag,
        mae,
        acc,
        epoch,
        best_model_path,
        early_stopping,
        ch_acc,
    ):
        """
        Validate the model.
        """
        stats = self.evaluate(
            model=model,
            model_id=model_id,
            loader=val_loader,
            criterion=criterion,
            flag=flag,
            mae=mae,
            acc=acc,
            ch_acc=ch_acc,
        )

        # Synchronize before validation
        self.sync()

        if self.rank == 0:
            ch_acc = True if self.args.data.full_channels else ch_acc
            acc = False if self.args.data.full_channels else acc
            val_loss = stats["loss"]
            self.log_stats(stats, flag, mae, acc, ch_acc, mode="val")

            if self.args.exp.best_model_metric == "loss":
                val_metric = val_loss
            elif self.args.exp.best_model_metric in {"acc", "ch_acc", "ch_f1"}:
                val_metric = -stats[self.args.exp.best_model_metric]
            else:
                raise ValueError(
                    f"Invalid best model metric: {self.args.exp.best_model_metric}"
                )

            # Save best model and apply early stopping
            if early_stopping:
                self.early_stopping(val_metric, model)

            else:
                if val_metric < self.best_val_metric:
                    if self.args.exp.best_model_metric == "loss":
                        self.print_master(
                            f"Validation loss decreased ({self.best_val_metric:.6f} --> {val_metric:.6f})."
                        )
                    elif self.args.exp.best_model_metric in {"acc", "ch_acc", "ch_f1"}:
                        self.print_master(
                            f"Validation {self.args.exp.best_model_metric} increased ({-self.best_val_metric * 100:.3f}% --> {-val_metric * 100:.3f}%)."
                        )
                    path_dir = os.path.abspath(os.path.dirname(best_model_path))
                    if not os.path.isdir(path_dir):
                        os.makedirs(path_dir)
                    torch.save(model.state_dict(), best_model_path)
                    self.print_master(f"Saving Model Weights at: {best_model_path}...")
                    self.best_val_metric = val_metric

        self.sync()

        self.print_master("Validation complete")

    def supervised_train(self):
        """
        Train the model in supervised mode.
        """

        # Load train loaders
        # self.init_dataloaders(loader_type="train", learning_type="sl")
        self.init_dataloaders(loader_type="all", learning_type="sl")
        self.sync()

        # Initialize Model and Optimizer
        self.init_model()
        self.init_optimizer()

        # Get supervised criterion
        self.sync()
        self.criterion = get_criterion(self.args, self.args.sl.criterion)
        self.print_master(f"{self.args.sl.criterion} initialized.")

        # Get supervised scheduler
        if self.args.sl.scheduler != "None":
            self.sl_scheduler = get_scheduler(
                self.args,
                self.args.sl.scheduler,
                "supervised",
                self.optimizer,
                len(self.train_loader),
            )
        else:
            self.sl_scheduler = None
        self.print_master("Starting Supervised Training...")

        # Supervised Training
        best_model_path = os.path.join(self.log_dir, f"supervised.pth")
        self.train(
            model=self.model,
            model_id=self.args.exp.model_id,
            optimizer=self.optimizer,
            train_loader=self.train_loader,
            best_model_path=best_model_path,
            criterion=self.criterion,
            val_loader=self.val_loader,
            scheduler=self.sl_scheduler,
            flag="sl",
            mae=self.args.exp.mae,
            acc=self.args.exp.acc,
            early_stopping=self.args.sl.early_stopping,
            ch_acc=self.args.exp.ch_acc,
        )

        # Upload best model to Neptune
        if self.args.exp.neptune and not self.args.exp.sklearn:
            self.print_master(f"Uploading best model to Neptune.")
            self.logger[f"model_checkpoints/{self.args.exp.model_id}_sl"].upload(
                best_model_path
            )

        # Test model
        self.print_master("Starting Supervised Testing...")
        self.sync()
        self.test(
            model=self.model,
            model_id=self.args.exp.model_id,
            best_model_path=best_model_path,
            criterion=self.criterion,
            flag="sl",
            mae=self.args.exp.mae,
            acc=self.args.exp.acc,
            ch_acc=self.args.exp.ch_acc,
        )

    def pretrain(self):
        """
        Pretrain the model using self-supervised learning (SSL).
        """

        # Initialize dataloaders
        self.init_dataloaders(loader_type="train", learning_type="ssl")
        self.sync()

        # Initialize Model and Optimizer
        self.init_model()
        self.init_optimizer()

        # SSL Learning rate scheduler
        self.ssl_scheduler = get_scheduler(
            self.args,
            self.args.ssl.scheduler,
            "pretrain",
            self.optimizer,
            len(self.train_loader),
        )

        # SSL criterion
        self.ssl_criterion = get_criterion(self.args, self.args.ssl.criterion)

        if self.args.ema.ema_momentum_scheduler:
            self.ema = ema_momentum_scheduler(
                len(self.train_loader),
                self.args.ssl.epochs,
                1,
                self.args.ema.beta_0,
                self.args.ema.beta_1,
            )

        # Self-Supervised Pretraining
        self.print_master("Starting SSL Pre-Training...")
        self.pretrained_model_path = os.path.join(self.log_dir, "pretrained.pth")
        self.train(
            model=self.model,
            model_id=self.args.exp.model_id,
            optimizer=self.optimizer,
            train_loader=self.train_loader,
            best_model_path=self.pretrained_model_path,
            criterion=self.ssl_criterion,
            val_loader=self.val_loader,
            scheduler=self.ssl_scheduler,
            flag="ssl",
            ema=self.ema,
            early_stopping=self.args.ssl.early_stopping,
        )

        self.print_master("End of SSL Pre-Training.")

    def downstream(self):
        """
        Train downstream model after pretraining.
        """

        # Initialize dataloaders
        self.init_dataloaders(loader_type="train", learning_type="downstream")

        # Load best model
        self.model = get_model(self.args, self.generator)
        self.model.load_state_dict(torch.load(self.pretrained_model_path))

        # Initialize downstream model
        self.downstream_model = get_downstream_model(self.args, self.model)
        self.downstream_model.to(self.device)

        # Downstream criterion
        self.downstream_criterion = get_criterion(
            self.args, self.args.downstream.criterion
        )

        # Downstream Optimizer
        self.downstream_optimizer = get_downstream_optim(
            self.args,
            self.downstream_model,
            self.args.downstream.eval,
            self.args.downstream.optimizer,
        )

        # Downstream Scheduler
        self.downstream_scheduler = get_scheduler(
            self.args,
            self.args.downstream.scheduler,
            "downstream",
            self.downstream_optimizer,
            len(self.downstream_train_loader),
        )

        # Downstream Training
        self.print_master("Starting Downstream Training...")
        best_model_path = os.path.join(self.log_dir, "downstream.pth")
        self.train(
            model=self.downstream_model,
            model_id=self.args.downstream.model_id,
            optimizer=self.downstream_optimizer,
            train_loader=self.train_loader,
            best_model_path=best_model_path,
            criterion=self.downstream_criterion,
            val_loader=self.val_loader,
            scheduler=self.downstream_scheduler,
            flag="downstream",
            mae=self.args.exp.mae,
            acc=self.args.exp.acc,
            early_stopping=self.args.downstream.early_stopping,
            ch_acc=self.args.exp.ch_acc,
        )

        # Initialize testloader
        self.init_dataloaders(loader_type="test", learning_type="downstream")

        # Test downstream
        self.print_master("Starting Downstream Testing...")
        self.test(
            model=self.downstream_model,
            model_id=self.args.downstream.model_id,
            best_model_path=best_model_path,
            criterion=self.downstream_criterion,
            flag="downstream",
            mae=self.args.exp.mae,
            acc=self.args.exp.acc,
        )
        self.print_master("End of Downstream.")

    def test(
        self,
        model,
        model_id,
        best_model_path,
        criterion,
        flag="sl",
        mae=False,
        acc=False,
        ch_acc=False,
    ):
        # Load data
        if self.args.exp.sklearn:
            pass
        elif self.args.exp.calibrate:
            self.init_dataloaders(loader_type="all", learning_type="sl")
        else:
            self.init_dataloaders(loader_type="test", learning_type="sl")

        # <---Scikit-learn pipeline--->
        if self.args.exp.sklearn:
            self.sklearn_eval(
                model,
                criterion,
                self.test_loader.x,
                self.test_loader.y,
                acc,
                mae,
                ch_acc,
                flag,
            )
            return

        # <---Deep learning (PyTorch) pipeline--->
        # Load best model
        self.sync()
        model_weights = torch.load(best_model_path)
        model.load_state_dict(model_weights)
        self.train_calibrator(
            model, model_id, criterion, flag
        )  # Optional: Train probability calibrator before evaluation

        # Test set evaluation
        self.sync()
        stats = self.evaluate(
            model=model,
            model_id=model_id,
            loader=self.test_loader,
            criterion=criterion,
            flag=flag,
            mae=mae,
            acc=acc,
            ch_acc=ch_acc,
        )

        if self.rank == 0:
            ch_acc = True if self.args.data.full_channels else ch_acc
            acc = False if self.args.data.full_channels else acc
            self.log_stats(stats, flag, mae, acc, ch_acc, mode="test")
            self.tuning_score = stats[
                f"{self.args.exp.tuning_metric}"
            ]  # For hyperparameter tuning

    def evaluate(
        self,
        model: nn.Module,
        model_id: str,
        loader: DataLoader,
        criterion: nn.Module,
        flag: str,
        mae=False,
        acc=False,
        ch_acc=False,
        calibrate=False,
    ):
        """
        Evaluate the model return evaluation loss and/or evaluation accuracy.
        """
        self.print_master("Evaluating...")
        stats = dict()
        mae_loss = nn.L1Loss()
        num_examples = len(loader.dataset)  # Total number of examples across all ranks

        self.sync()
        model.eval()
        with torch.no_grad():
            # Initialize Metrics
            total_loss = torch.tensor(0.0, device=self.device)
            total_mae = torch.tensor(0.0, device=self.device)
            all_logits = []
            all_labels = []
            all_ch_ids = []
            all_u = []

            for i, batch in enumerate(loader):
                output = forward_pass(self.args, model, batch, model_id, self.device)
                y_hat, y, ch_ids, u = self.parse_output(output, batch)
                all_logits.append(y_hat)
                all_labels.append(y)
                all_ch_ids.append(ch_ids)
                all_u.append(u)

                # Loss
                loss = compute_loss(
                    output, batch, criterion, model_id, self.args, self.device
                )
                num_batch_examples = torch.tensor(batch[0].shape[0], device=self.device)
                total_loss += loss * num_batch_examples

                # MAE
                if mae:
                    total_mae += (
                        mae_loss(output, batch[1].to(self.device)) * num_batch_examples
                    )

        if calibrate:
            return torch.cat(all_logits), torch.cat(all_labels), torch.cat(all_ch_ids)

        # Loss
        if self.args.ddp.ddp:
            self.sync()
            self.print_rank(
                f"Evaluation loss before all_reduce: {total_loss.item()}. Rank: {self.rank}"
            )
            dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
        if self.rank == 0:
            stats["loss"] = total_loss.item() / num_examples
        self.sync()
        self.print_rank(
            f"Evaluation loss after all_reduce: {total_loss.item()}. Rank: {self.rank}."
        )

        # MAE
        if mae:
            if self.args.ddp.ddp:
                self.sync()
                dist.all_reduce(total_mae, op=dist.ReduceOp.SUM)
            self.sync()
            if self.rank == 0:
                stats["mae"] = total_mae.item() / num_examples

        # Window Metrics
        if acc:
            window_metrics = get_metrics(
                self.args,
                all_logits,
                all_labels,
                mode="window",
                rank=self.rank,
                calibrator=self.calibrator,
            )
            channel = True if self.args.data.full_channels else False
            update_stats(
                stats,
                window_metrics,
                self.args.open_neuro.task,
                self.args.exp.other_metrics,
                channel,
                self.rank,
            )

        # Channel Metrics
        if ch_acc:
            ch_metrics, ch_df = get_metrics(
                self.args,
                all_logits,
                all_labels,
                all_ch_ids,
                all_u,
                mode="channel",
                rank=self.rank,
                calibrator=self.calibrator,
            )
            # ch_df.to_csv(os.path.join(self.log_dir, "channel_metrics.csv"), index=False)
            update_stats(
                stats,
                ch_metrics,
                self.args.open_neuro.task,
                self.args.exp.other_metrics,
                True,
                self.rank,
            )

        self.sync()
        return stats

    def train_calibrator(self, model, model_id, criterion, flag="sl"):
        # Calibration (window or channel probabilities)
        if self.args.exp.calibrate:
            self.calibrator = CalibrationModel(self.rank, self.args)

            # Combine train and val loaders into one
            train_probs, train_targets, train_ch_ids = self.evaluate(
                model=model,
                model_id=model_id,
                loader=self.train_loader,
                criterion=criterion,
                flag=flag,
                calibrate=True,
            )

            val_probs, val_targets, val_train_ch_ids = self.evaluate(
                model=model,
                model_id=model_id,
                loader=self.val_loader,
                criterion=criterion,
                flag=flag,
                calibrate=True,
            )

            all_probs = torch.cat([train_probs, val_probs], dim=0).squeeze()
            all_targets = torch.cat([train_targets, val_targets], dim=0).squeeze()
            all_ch_ids = torch.cat([train_ch_ids, val_train_ch_ids], dim=0).squeeze()

            probs, targets, ch_ids = self.calibrator.compile_predictions(
                all_probs, all_targets, all_ch_ids
            )
            self.calibrator.train(probs, targets, ch_ids)

    def parse_output(self, output, batch):
        ch_ids = torch.tensor(0, device=self.device).unsqueeze(-1)
        u = torch.tensor(0, device=self.device).unsqueeze(-1)
        if self.args.open_neuro.ch_aggr:
            y_hat, y = output[:2]
        elif self.args.qwa.qwa:
            y_hat = output[1]
            y, ch_ids = batch[1].to(self.device), batch[2].to(self.device)
        elif self.args.exp.u_weight:
            y_hat, u = output
            y, ch_ids = batch[1], batch[2]
        elif self.args.open_neuro.ch_loss:
            y_hat, y, ch_ids = (output, batch[1], batch[2])
        else:
            y_hat, y = (output, batch[1].to(self.device))

        if y_hat.dim() > 1 and y_hat.size(-1) == 1:
            y_hat = y_hat.unsqueeze(-1)

        return (y_hat, y.to(self.device), ch_ids.to(self.device), u.to(self.device))

    def gather_tensor(self, tensor):
        gathered = [torch.zeros_like(tensor) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered, tensor)
        return torch.cat(gathered)

    def log_stats(self, stats, flag, mae, acc, ch_acc, mode, task="binary"):
        modes = {"val": "Validation", "test": "Test"}
        Mode = modes[mode]
        mapping = get_logger_mapping(self.args.open_neuro.task)

        loss = stats["loss"]
        self.print_master(f"Model {Mode} Loss: {loss:.6f}")
        epoch_logger(self.args, self.logger, f"{flag}_{mode}/loss", loss)

        if mae:
            mae_value = stats["mae"]
            self.print_master(f"Model {Mode} MAE: {mae_value:.6f}")
            epoch_logger(self.args, self.logger, f"{flag}_{mode}/mae", mae_value)
        if acc and not self.args.open_neuro.ch_aggr:
            acc_value = stats["acc"] * 100
            self.print_master(f"Model {Mode} Accuracy: {acc_value:.2f}%")
            epoch_logger(self.args, self.logger, f"{flag}_{mode}/accuracy", acc_value)

            if self.args.exp.other_metrics:
                for key, value in mapping.items():
                    self.logger[f"{flag}_{mode}/{value}"] = stats[key]
        if ch_acc:
            ch_acc_value = stats["ch_acc"] * 100
            self.print_master(f"Model {Mode} Channel Accuracy: {ch_acc_value:.2f}%")
            epoch_logger(
                self.args, self.logger, f"{flag}_{mode}/channel_accuracy", ch_acc_value
            )

            if self.args.exp.other_metrics:
                for key, value in mapping.items():
                    self.logger[f"{flag}_{mode}/channel_{value}"] = stats[f"ch_{key}"]

    def count_parameters(self):
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    def print_master(self, message):
        """
        Prints statements to the rank 0 node.
        """
        if self.rank == 0:
            self.console.log(message)

    def print_rank(self, message):
        """
        Prints statement on every rank.
        """
        if self.args.exp.rank_verbose:
            self.console.log(message)

    def sync(self):
        """
        Synchronizes all processes for Distributed Data Parallel (DDP).
        """
        if self.args.ddp.ddp:
            dist.barrier()
            self.print_master("Synchronizing") if self.args.exp.rank_verbose else None

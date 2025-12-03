import warnings
warnings.filterwarnings("ignore", message="h5py not installed")
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Supervised Models
from sss.models.patchtst_blind import PatchTST as PatchTSTBlind
from sss.models.patchtst_original import PatchTST as PatchTSTOG
from sss.models.mlp_mixer import MLPMixerCI
from sss.models.recurrent import RecurrentModel
from sss.models.linear import Linear
from sss.models.dlinear import DLinear
from sss.models.modern_tcn import ModernTCN
from sss.models.timesnet import TimesNet
from sss.models.sss import SSS
from sss.models.tsmixer import TSMixer


# Layers
from sss.layers.patchtst_blind.backbone import SupervisedHead
from sss.layers.patchtst_blind.revin import RevIN

# Optimizers and Schedulers
from torch import optim
from sss.utils.schedulers import WarmupCosineSchedule, PatchTSTSchedule
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR

# Loss Functions
from sss.layers.channel_modules.ch_loss import ChannelLoss
from sss.layers.dynamic_weights.qwa_loss import QWALoss

def get_model(args, generator=torch.Generator()):
    if args.exp.model_id=="SSS":
        model = SSS(
            args=args,
            seq_len=args.data.seq_len,
            pred_len=args.data.pred_len,
            num_channels=args.data.num_channels,
            backbone_id=args.exp.backbone_id,
            mcd = args.mcd.mcd,
            mcd_samples = args.mcd.num_samples,
            mcd_stats = args.mcd.stats,
            mcd_prob = args.mcd.mcd_prob,
            revin=args.sl.revin,
            revin_affine=args.sl.revin_affine,
            revout=args.sl.revout,
            eps_revin=args.sl.eps_revin,
            patch_dim=args.data.patch_dim,
            patch_stride=args.data.patch_stride,
            norm_mode=args.sl.norm_mode,
            return_head=args.sl.return_head,
            pred_dropout=args.sl.pred_dropout,
            sparse_context=args.sparse_context,
            )
    elif args.exp.model_id == "PatchTSTOG":
        model = PatchTSTOG(num_channels=args.data.num_channels,
                           seq_len=args.data.seq_len,
                           pred_len=args.data.pred_len,
                           patch_dim=args.data.patch_dim,
                           stride=args.data.patch_stride,
                           num_enc_layers=args.sl.num_enc_layers,
                           d_model=args.sl.d_model,
                           num_heads=args.sl.num_heads,
                           d_ff=args.sl.d_ff,
                           norm_mode=args.sl.norm_mode,
                           dropout=args.sl.dropout,
                           attn_dropout=args.sl.attn_dropout,
                           ff_dropout=args.sl.ff_dropout,
                           pred_dropout=args.sl.pred_dropout,
                           revin=args.sl.revin,
                           revin_affine=args.sl.revin_affine,)
    elif args.exp.model_id == "PatchTSTBlind":
        model = PatchTSTBlind(num_enc_layers=args.sl.num_enc_layers,
                              d_model=args.sl.d_model,
                              d_ff=args.sl.d_ff,
                              num_heads=args.sl.num_heads,
                              num_channels=args.data.num_channels,
                              seq_len=args.data.seq_len,
                              pred_len=args.data.pred_len,
                              attn_dropout=args.sl.attn_dropout,
                              ff_dropout=args.sl.ff_dropout,
                              pred_dropout=args.sl.pred_dropout,
                              batch_first=args.sl.batch_first,
                              norm_mode=args.sl.norm_mode,
                              revin=args.sl.revin,
                              revout=args.sl.revout,
                              revin_affine=args.sl.revin_affine,
                              eps_revin=args.sl.eps_revin,
                              patch_dim=args.data.patch_dim,
                              stride=args.data.patch_stride,
                              head_type=args.sl.head_type,
                              ch_aggr=args.open_neuro.ch_aggr,
                              ch_reduction=args.open_neuro.ch_reduction,
                              cla_mix=args.clm.clm,
                              cla_mix_layers=args.clm.num_enc_layers,
                              cla_combination=args.clm.combo,
                              qwa=args.qwa.qwa,
                              qwa_num_networks=args.qwa.num_networks,
                              qwa_network_type=args.qwa.network_type,
                              qwa_hidden_dim=args.qwa.hidden_dim,
                              qwa_mlp_dropout=args.qwa.mlp_dropout,
                              qwa_attn_dropout=args.qwa.attn_dropout,
                              qwa_ff_dropout=args.qwa.ff_dropout,
                              qwa_norm_mode=args.qwa.norm_mode,
                              qwa_num_heads=args.qwa.num_heads,
                              qwa_num_enc_layers=args.qwa.num_enc_layers,
                              qwa_upper_quantile=args.qwa.upper_quantile,
                              qwa_lower_quantile=args.qwa.lower_quantile,)
    elif args.exp.model_id == "RecurrentModel":
        model = RecurrentModel(d_model=args.sl.d_model,
                               backbone_id=args.exp.backbone_id,
                               num_enc_layers=args.sl.num_enc_layers,
                               pred_len=args.data.pred_len,
                               bidirectional=args.sl.bidirectional,
                               dropout=args.sl.dropout,
                               seq_len=args.data.seq_len,
                               patching=args.data.patching,
                               patch_dim=args.data.patch_dim,
                               patch_stride=args.data.patch_stride,
                               num_channels=args.data.num_channels,
                               head_type=args.sl.head_type,
                               norm_mode=args.sl.norm_mode,
                               revin=args.sl.revin,
                               revout=args.sl.revout,
                               revin_affine=args.sl.revin_affine,
                               eps_revin=args.sl.eps_revin,
                               last_state=args.sl.last_state,
                               avg_state=args.sl.avg_state)
    elif args.exp.model_id == "Linear":
        model = Linear(in_features=args.data.seq_len,
                       out_features=args.data.pred_len,
                       norm_mode=args.sl.norm_mode)
    elif args.exp.model_id == "DLinear":
        model = DLinear(task=args.exp.task,
                        seq_len=args.data.seq_len,
                        pred_len=args.data.pred_len,
                        num_channels=args.data.num_channels,
                        num_classes=args.data.pred_len,
                        moving_avg=args.dlinear.moving_avg,
                        individual=args.dlinear.individual,
                        return_head=args.sl.return_head,
        )
    elif args.exp.model_id == "ModernTCN":
        model = ModernTCN(
            seq_len=args.data.seq_len,
            pred_len=args.data.pred_len,
            patch_dim=args.data.patch_dim,
            patch_stride=args.data.patch_stride,
            num_classes=args.data.pred_len,
            num_channels=args.data.num_channels,
            task=args.exp.task,
            return_head=args.sl.return_head,
            dropout=args.sl.dropout,
            class_dropout=args.moderntcn.class_dropout,
            ffn_ratio=args.moderntcn.ffn_ratio,
            num_enc_layers=args.moderntcn.num_enc_layers,
            large_size=args.moderntcn.large_size,
            d_model=args.moderntcn.d_model,
            revin=args.sl.revin,
            affine=args.sl.revin_affine,
            small_size=args.moderntcn.small_size,
            dw_dims=args.moderntcn.dw_dims,
        )

    elif args.exp.model_id == "TimesNet":
        model = TimesNet(
            seq_len=args.data.seq_len,
            pred_len=args.data.pred_len,
            num_channels=args.data.num_channels,
            d_model=args.timesnet.d_model,
            d_ff=args.timesnet.d_ff,
            num_enc_layers=args.sl.num_enc_layers,
            num_kernels=args.timesnet.num_kernels,
            c_out=args.timesnet.c_out,
            top_k=args.timesnet.top_k,
            dropout=args.sl.dropout,
            task=args.exp.task,
            revin=args.sl.revin,
            revin_affine=args.sl.revin_affine,
            revout=args.sl.revout,
            eps_revin=args.sl.eps_revin,
            return_head=args.sl.return_head,
        )
    elif args.exp.model_id == "TSMixer":
        model = TSMixer(
            seq_len=args.data.seq_len,
            pred_len=args.data.pred_len,
            num_enc_layers=args.sl.num_enc_layers,
            d_model=args.sl.d_model,
            num_channels=args.data.num_channels,
            dropout=args.sl.dropout,
            revin=args.sl.revin,
            revin_affine=args.sl.revin_affine,
            revout=args.sl.revout,
            eps_revin=args.sl.eps_revin,
        )
    elif args.exp.model_id == "MLPMixerCI":
        model = MLPMixerCI(num_enc_layers=args.sl.num_enc_layers,
                           d_model=args.sl.d_model,
                           tok_mixer_dim=args.mlp_mixer.tok_mixer_dim,
                           cha_mixer_dim=args.mlp_mixer.cha_mixer_dim,
                           num_channels=args.data.num_channels,
                           seq_len=args.data.seq_len,
                           pred_len=args.data.pred_len,
                           pos_enc_type=args.mlp_mixer.pos_enc_type,
                           pred_dropout=args.sl.pred_dropout,
                           dropout=args.mlp_mixer.dropout,
                           revin=args.sl.revin,
                           revout=args.sl.revout,
                           revin_affine=args.sl.revin_affine,
                           eps_revin=args.sl.eps_revin,
                           patch_dim=args.data.patch_dim,
                           patch_stride=args.data.patch_stride)
    else:
        raise ValueError("Please select a valid model_id.")
    return model

def get_downstream_model(args, model):
    if args.exp.model_id in {"JEPA", "DualJEPA"}:
        downstream_model = copy.deepcopy(model.context_encoder) # Copy context encoder (PatchTSTBlindMasked)
        downstream_model.backbone.return_head = True # Turn on linear head

        if args.downstream.revin and not args.ssl.revin: # Initialize RevIN/RevOUT if not used during SSL
            downstream_model._init_revin(args.downstream.revout, args.downstream.revin_affine)
        elif args.ssl.revin: # If RevIN initialized during SSL, initialize RevOUT
            downstream_model.revout = args.downstream.revout

        # Finetuning or Linear Probing
        if args.downstream.eval == "linear_probe":
            downstream_model.requires_grad_(False) # Freeze all weights
            downstream_model.backbone.head.requires_grad_(True) # Unfreeze linear head
            print("Initialized downstream model for linear probing.")
        elif args.downstream.eval == "finetune":
            downstream_model.requires_grad_(True)
            print("Initialized downstream model for finetuning.")
        else:
            raise ValueError("Please select a valid downstream_eval.")

        return downstream_model

# TODO: Implement exclude_weight_decay() in this function
def get_downstream_optim(args, model, downstream_eval="finetune", optimizer_type="adamw"):
    optimizer_classes = {"adam": optim.Adam, "adamw": optim.AdamW}
    if optimizer_type not in optimizer_classes:
        raise ValueError("Please select a valid optimizer.")
    optimizer_class = optimizer_classes[optimizer_type]

    if downstream_eval == "finetune":
        params = [
            {'params': [param for name, param in model.named_parameters() if 'backbone.head' not in name], "lr" : args.downstream.lr_encoder},
            {'params': model.backbone.head.parameters(), "lr" : args.downstream.lr_head}
        ]
        optimizer = optimizer_class(params, weight_decay=args.downstream.weight_decay)
    elif downstream_eval == "linear_probe":
        optimizer = optimizer_class(model.backbone.head.parameters(), lr=args.downstream.lr_head, weight_decay=args.downstream.weight_decay)
    else:
        raise ValueError("Please set downstream_eval to either 'linear_probe' or 'finetune'.")

    return optimizer

def get_optim(args, model, optimizer_type="adamw", flag="sl"):
    if args.exp.sklearn:
        return None

    optimizer_classes = {"adam": optim.Adam, "adamw": optim.AdamW}
    if optimizer_type not in optimizer_classes:
        raise ValueError("Please select a valid optimizer.")
    optimizer_class = optimizer_classes[optimizer_type]

    param_groups = exclude_weight_decay(model, args, flag) # Exclude bias and normalization parameters from weight decay

    optimizer = optimizer_class(param_groups) # Set optimizer

    return optimizer

def exclude_weight_decay(model, args, flag="sl"):
    # Separate parameters into those that will use weight decay and those that won't
    decay_params = []
    no_decay_params = []

    for name, param in model.named_parameters():
        if param.requires_grad:
            if 'bias' in name or isinstance(param, (nn.BatchNorm1d, nn.BatchNorm2d, nn.LayerNorm, RevIN)):
                no_decay_params.append(param)
            else:
                decay_params.append(param)

    # Create parameter groups
    param_groups = [
        {'params': decay_params, 'weight_decay': eval(f"args.{flag}.weight_decay")},
        {'params': no_decay_params, 'weight_decay': 0.0}
    ]

    return param_groups

def get_scheduler(args, scheduler_type, training_mode, optimizer, num_batches=0):
    if args.exp.sklearn:
        return None

    if scheduler_type == "cosine_warmup" and training_mode=="pretrain":
        scheduler = WarmupCosineSchedule(optimizer=optimizer,
                                         warmup_steps=args.scheduler.warmup_steps,
                                         start_lr=args.scheduler.start_lr,
                                         ref_lr=args.scheduler.ref_lr,
                                         T_max=args.scheduler.T_max,
                                         final_lr=args.scheduler.final_lr)
    elif scheduler_type == "cosine":
        if training_mode=="downstream":
            scheduler = CosineAnnealingLR(optimizer,
                                        T_max=args.downstream.epochs,
                                        eta_min=args.downstream.lr*1e-2,
                                        last_epoch=args.downstream.last_epoch)
        elif training_mode=="supervised":
            scheduler = CosineAnnealingLR(optimizer,
                                        T_max=args.sl.epochs,
                                        eta_min=args.sl.lr*1e-2,
                                        last_epoch=args.scheduler.last_epoch)
    elif scheduler_type == "patchtst" and training_mode=="supervised":
            scheduler = PatchTSTSchedule(optimizer, args, num_batches)
    elif scheduler_type == "onecyle" and training_mode=="supervised":
        scheduler = OneCycleLR(optimizer=optimizer,
                               steps_per_epoch=num_batches,
                               pct_start=args.scheduler.pct_start,
                               epochs = args.sl.epochs,
                               max_lr = args.sl.lr)
    elif scheduler_type is None:
        return None
    else:
        raise ValueError("Please select a valid scheduler_type.")
    return scheduler

def get_criterion(args, criterion_type):
    if criterion_type == "MSE":
        return nn.MSELoss()
    elif criterion_type == "SmoothL1":
        return nn.SmoothL1Loss()
    elif criterion_type == "BCE":
        return nn.BCEWithLogitsLoss()
    elif criterion_type == "BCE_normal":
        return nn.BCELoss()
    elif criterion_type == "CE":
        return nn.CrossEntropyLoss()
    elif criterion_type == "ChannelLossBCE":
        return (nn.BCEWithLogitsLoss(), ChannelLoss(loss_type="BCE", num_classes=args.data.pred_len, u_weight=args.exp.u_weight))
    elif criterion_type == "ChannelLossCE":
        return (nn.CrossEntropyLoss(), ChannelLoss(loss_type="CE", num_classes=args.data.pred_len, u_weight=args.exp.u_weight))
    elif criterion_type == "QWALoss":
        return QWALoss(num_classes=args.data.pred_len,
                       ch_loss_refined=args.qwa.ch_loss_refined,
                       ch_loss_coarse=args.qwa.ch_loss_coarse,
                       window_loss_refined=args.qwa.window_loss_refined,
                       window_loss_coarse=args.qwa.window_loss_coarse,
                       skew_loss=args.qwa.skew_loss,
                       delta=args.qwa.delta,
                       coeffs=args.qwa.coeffs,
                       loss_type=args.qwa.loss_type)
    else:
        raise ValueError("Please select a valid criterion_type.")

def forward_pass(args, model, batch, model_id, device):
    if model_id in {"SSS", "PatchTSTBlind", "PatchTSTOG", "TSMixer", "MLPMixerCI", "RecurrentModel", "Linear", "DLinear", "ModernTCN", "TimesNet", "RF_EMF_CNN", "RF_EMF_MLP", "RF_EMF_LSTM", "RF_EMF_Transformer"}:
        if args.open_neuro.ch_aggr or args.clm.clm or args.qwa.qwa:
            x, y, ch_ids = batch[0].to(device), batch[1].to(device), batch[2].to(device)
            output = model(x, y, ch_ids)
        if args.data.time_indices and model_id=="SSS":
            x, y, ch_ids, t = batch[0].to(device), batch[1].to(device), batch[2].to(device), batch[3].to(device)
            output = model(x, y, ch_ids, t)
        else:
            x = batch[0]
            x = x.to(device)
            output = model(x)
            # output = output.view(-1)[0].view(1) if args.sl.batch_size==1 and args.open_neuro.task else output
    elif model_id == "JEPA":
        x, context_mask, target_masks = batch
        x, context_mask, target_masks = x.to(device), context_mask.to(device), target_masks.to(device)
        output = model(x, context_mask, target_masks)
    elif model_id == "DualJEPA":
        if args.dual_jepa.stochastic:
            x1, x2, distances, context_mask1, context_mask2, target_masks1, target_masks2 = [batch[i].to(device) for i in range(len(batch))]
        else:
            x1, x2, context_mask1, context_mask2, target_masks1, target_masks2 = [batch[i].to(device) for i in range(len(batch))]
            distances = None
        output = model(x1, x2, distances, context_mask1, context_mask2, target_masks1, target_masks2)
    else:
        raise ValueError("Please select a valid model_id.")

    return output

def prepare_output_and_target(output, target, args, device):
    out = output.to(device).squeeze()  # (batch_size, 1, 1) -> (batch_size,) for binary task
    target = target.to(device)

    if args.open_neuro.task == "binary":
        if out.dim() == 0:  # scalar output for batch_size == 1
            out = out.unsqueeze(0)  # (,) -> (1,)
    elif args.open_neuro.task == "multi":
        if out.dim() == 1:  # (num_classes,) for batch_size == 1
            out = out.unsqueeze(0)  # (num_classes,) -> (1, num_classes)

    # Ensure out and target have the same size
    if args.open_neuro.task == "binary":
        assert out.size() == target.size(), f"Size mismatch: out {out.size()}, target {target.size()}"
    elif args.open_neuro.task == "multi":
        assert out.size(0) == target.size(0), f"Size mismatch: out {out.size()}, target {target.size()}"

    target = target.long() if args.open_neuro.task == "multi" else target.float()

    return out, target

def compute_loss(output, batch, criterion, model_id, args, device):
    if model_id in {"SSS", "PatchTSTBlind", "PatchTSTOG", "TSMixer", "MLPMixerCI", "RecurrentModel", "Linear", "DLinear", "ModernTCN", "TimesNet", "RF_EMF_CNN", "RF_EMF_MLP", "RF_EMF_LSTM", "RF_EMF_Transformer"}:
        if args.open_neuro.ch_loss:
            if args.clm.clm:
                out = output.squeeze().to(device)
                target = batch[1].to(device); ch_ids = batch[2].to(device)
                target = target.float() if args.open_neuro.task=="binary" else target.long()
            elif args.exp.u_weight:
                out = output[0].squeeze().to(device)
                u = output[1].squeeze().to(device)
                target = batch[1].to(device)
                ch_ids = batch[2].to(device)
            else:
                out, target = prepare_output_and_target(output, batch[1], args, device)
                ch_ids = batch[2].to(device)

            # Window loss
            normal_criterion = criterion[0]
            window_loss = normal_criterion(out, target)

            # Channel loss
            ch_criterion = criterion[1]
            if args.exp.u_weight:
                channel_loss, u_entropy = ch_criterion(out, target, ch_ids, u)
            else:
                channel_loss = ch_criterion(out, target, ch_ids)

            # Total Loss (window loss + channel loss)
            loss = args.open_neuro.alpha * window_loss + args.open_neuro.beta * channel_loss

            # Entropy diversity term for u_weight (optional)
            if args.exp.u_weight:
                loss += args.open_neuro.chi * u_entropy

        elif args.open_neuro.ch_aggr:
            y_hat_ch, y_ch = output[0].squeeze().to(device), output[1].to(device)

            if args.open_neuro.task=="binary":
                y_ch = y_ch.float()
            elif args.open_neuro.task=="multi":
                y_ch = y_ch.long()

            loss = criterion(y_hat_ch, y_ch)
        elif args.qwa.qwa:
            coarse_probs, refined_probs, qwa_coeffs = output[0].squeeze().to(device), output[1].squeeze().to(device), output[2]
            target = batch[1].to(device)
            ch_ids = batch[2].to(device)
            loss = criterion(coarse_probs, refined_probs, target, ch_ids, qwa_coeffs, device)
        else:
            out, target = prepare_output_and_target(output, batch[1], args, device)
            # output = output.squeeze()
            # target = batch[1].to(device)
            # target = target.squeeze() if args.sl.batch_size==1 else target
            loss = criterion(out, target)
    elif model_id == "JEPA":
        y_hat, y = output
        loss = criterion(y_hat, y)
    elif model_id == "DualJEPA":
        y_hat1, y_hat2, y1, y2 = output
        loss = (criterion(y_hat1[0], y1) + criterion(y_hat1[1], y1) + criterion(y_hat2[0], y2) + criterion(y_hat2[1], y2)) / 4
    else:
        raise ValueError("Please select a valid model_id.")

    return loss

def model_update(model, loss, optimizer, model_id, alpha=0.6):
    if model_id in {"SSS", "PatchTSTBlind", "PatchTSTOG", "TSMixer", "MLPMixerCI", "RecurrentModel", "Linear", "DLinear", "ModernTCN", "TimesNet", "RF_EMF_CNN", "RF_EMF_MLP", "RF_EMF_LSTM", "RF_EMF_Transformer"}:
        loss.backward()
        # check_gradients(model)
        optimizer.step()
    elif model_id in {"JEPA", "DualJEPA"}:
        loss.backward()
        optimizer.step()
        ema_update(model.context_encoder, model.target_encoder, alpha)
    else:
        raise ValueError("Please select a valid model_id.")

def ema_update(source_model: nn.Module, target_model: nn.Module, alpha: float):
    """
    NOTE: Fix better GPU Allocation so this does not occur
    Updates the target_model parameters by:

                theta_bar <- alpha*theta_bar + (1-alpha)*theta

    where theta_bar = target_model parameters and theta = source_model parameters.
    Adapted from: https://github.com/facebookresearch/ijepa/blob/main/src/train.py

    Args:
        source_model (nn.Module): Source model from which the parameters are used to update the target model.
        target_model (nn.Module): Target model whose parameters are updated.
        alpha (float): Exponential moving average decay factor.
    """
    with torch.no_grad():
        for theta, theta_bar in zip(source_model.parameters(), target_model.parameters()):
            # Store the original device of theta
            original_device = theta.device

            # Check if the devices are different and move theta to the same device as theta_bar if necessary
            if original_device != theta_bar.device:
                theta = theta.to(theta_bar.device)

            # Perform the update
            theta_bar.data.mul_(alpha).add_((1. - alpha) * theta.detach().data)

            # Move theta back to its original device
            if original_device != theta_bar.device:
                theta = theta.to(original_device)

def ema_momentum_scheduler(ipe, num_epochs, ipe_scale=1.0, beta_0=0.996, beta_1=1.0):
    """
    Args:
        beta_0 (float): Initial EMA decay factor.
        beta_1 (float): Final EMA decay factor.
        num_epochs (int): Number of epochs.
        ipe (int): Iterations per epoch, set to len(pretrain_loader) for self-supervised pretraining.
        ipe_scale (float): Iterations per epoch scale factor. A higher value leads to more a gradual transition
                           from beta_0 to beta_1, whereas a lower value leads to a more abrupt transition.
    Returns:
        momentum_coefficients (generator): Momentum coefficients for each iteration.
    """
    total_iterations = int(ipe*num_epochs*ipe_scale) + 1
    momentum_coefficients = (beta_0 + i*(beta_1 - beta_0)/(ipe*num_epochs*ipe_scale)
                             for i in range(total_iterations))

    return momentum_coefficients

def check_gradients(model, threshold_low=1e-5, threshold_high=1e2):
    vanishing = []
    exploding = []
    normal = []

    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            if grad_norm < threshold_low:
                vanishing.append((name, grad_norm))
            elif grad_norm > threshold_high:
                exploding.append((name, grad_norm))
            else:
                normal.append((name, grad_norm))

    print(f"Gradient statistics:")
    print(f"  Total parameters with gradients: {len(vanishing) + len(exploding) + len(normal)}")
    print(f"  Vanishing gradients: {len(vanishing)} ({len(vanishing) / (len(vanishing) + len(exploding) + len(normal)) * 100:.2f}%)")
    print(f"  Exploding gradients: {len(exploding)} ({len(exploding) / (len(vanishing) + len(exploding) + len(normal)) * 100:.2f}%)")
    print(f"  Normal gradients: {len(normal)} ({len(normal) / (len(vanishing) + len(exploding) + len(normal)) * 100:.2f}%)")

    if vanishing:
        print("\nVanishing gradients:")
        for name, grad_norm in vanishing[:10]:  # Print first 10
            print(f"  {name}: {grad_norm}")
        if len(vanishing) > 10:
            print(f"  ... and {len(vanishing) - 10} more")

    if exploding:
        print("\nExploding gradients:")
        for name, grad_norm in exploding[:10]:  # Print first 10
            print(f"  {name}: {grad_norm}")
        if len(exploding) > 10:
            print(f"  ... and {len(exploding) - 10} more")

    # Compute and print gradient statistics
    all_grads = [param.grad.norm().item() for name, param in model.named_parameters() if param.grad is not None]
    if all_grads:
        print("\nGradient norm statistics:")
        print(f"  Mean: {np.mean(all_grads):.6f}")
        print(f"  Median: {np.median(all_grads):.6f}")
        print(f"  Std: {np.std(all_grads):.6f}")
        print(f"  Min: {np.min(all_grads):.6f}")
        print(f"  Max: {np.max(all_grads):.6f}")

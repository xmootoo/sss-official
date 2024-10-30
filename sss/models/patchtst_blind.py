import torch
from torch import Tensor
import torch.nn as nn
from sss.layers.patchtst_blind.revin import RevIN
from sss.layers.patchtst_blind.pos_enc import PositionalEncoding
from sss.layers.patcher import Patcher
from sss.layers.patchtst_blind.backbone import PatchTSTBackbone
from sss.layers.patchtst_original.backbone import TSTiEncoder, Flatten_Head
from sss.utils.weight_init import xavier_init
from typing import Optional


class PatchTST(nn.Module):
    def __init__(self, num_enc_layers, d_model, d_ff, num_heads, num_channels, seq_len, pred_len, attn_dropout=0.0,
        ff_dropout=0.0, pred_dropout=0.0, batch_first=True, norm_mode="batch1d", revin=True, revout=True, revin_affine=True,
        eps_revin=1e-5, patch_dim=16, stride=1, return_head=True, head_type="linear", ch_aggr=False, ch_reduction="mean",
        cla_mix=False, cla_mix_layers=1, cla_combination="concat", qwa=False, qwa_num_networks=3,
        qwa_network_type="mlp", qwa_hidden_dim=2, qwa_mlp_dropout=0.0, qwa_attn_dropout=0.0, qwa_ff_dropout=0.0,
        qwa_norm_mode="batch1d", qwa_num_heads=3, qwa_num_enc_layers=1, qwa_upper_quantile=0.9, qwa_lower_quantile=0.1,):
        super(PatchTST, self).__init__()

        # Parameters
        self.num_patches = int((seq_len - patch_dim) / stride) + 2
        self.num_channels = num_channels
        self.eps_revin = eps_revin
        self.revin_affine = revin_affine

        # Initialize layers
        if revin:
            self._init_revin(revout, revin_affine)
        else:
            self._revin = None
            self.revout = None

        self.patcher = Patcher(patch_dim, stride)
        self.pos_enc = PositionalEncoding(patch_dim, d_model, self.num_patches)
        self.backbone = PatchTSTBackbone(num_enc_layers, d_model, d_ff, num_heads, num_channels, self.num_patches, pred_len,
                                         attn_dropout,ff_dropout, pred_dropout, batch_first, norm_mode, return_head, head_type,
                                         ch_aggr, ch_reduction, cla_mix, cla_mix_layers, cla_combination, qwa, qwa_num_networks,
                                         qwa_network_type, qwa_hidden_dim, qwa_mlp_dropout, qwa_attn_dropout, qwa_ff_dropout,
                                         qwa_norm_mode, qwa_num_heads, qwa_num_enc_layers, qwa_upper_quantile, qwa_lower_quantile)

        # Weight initialization
        self.apply(xavier_init)

    def _init_revin(self, revout:bool, revin_affine:bool):
        self._revin = True
        self.revout = revout
        self.revin_affine = revin_affine
        self.revin = RevIN(self.num_channels, self.eps_revin, self.revin_affine)

    def forward(self, x, y=None, ch_ids=None):

        # RevIN
        if self._revin:
            x = self.revin(x, mode="norm")

        # Patcher
        x = self.patcher(x)

        # Project + Positional Encoding
        x = self.pos_enc(x)

        # Transformer + Linear Head
        x = self.backbone(x, y, ch_ids)

        # RevOUT
        if self.revout:
            x = self.revin(x, mode="denorm")

        return x


class PatchTSTOG(nn.Module):
    def __init__(self, num_channels:int, seq_len:int, pred_len:int, patch_len:int, stride:int,
                 num_enc_layers:int=3, d_model=128, n_heads=16, d_k:Optional[int]=None, d_v:Optional[int]=None,
                 d_ff:int=256, norm:str='BatchNorm', attn_dropout:float=0., dropout:float=0., act:str="gelu", key_padding_mask:bool='auto',
                 padding_var:Optional[int]=None, attn_mask:Optional[Tensor]=None, res_attention:bool=True, pre_norm:bool=False, store_attn:bool=False,
                 pe:str='zeros', learn_pe:bool=True, fc_dropout:float=0., head_dropout = 0, padding_patch = None,
                 pretrain_head:bool=False, head_type = 'flatten', individual = False, revin = True, revin_affine = True, subtract_last = False,
                 verbose:bool=False, revout=True, eps_revin=1e-5, return_head=True, **kwargs):
        super(PatchTSTOG, self).__init__()

        # Parameters
        self.num_channels = num_channels
        self.eps_revin = eps_revin
        self.revin_affine = revin_affine

        # RevIn
        if revin:
            self._init_revin(revout, revin_affine)
        else:
            self.revin = None
            self.revout = None

        # Patching
        self.patcher = Patcher(patch_len, stride)
        num_patches = int((seq_len - patch_len) / stride) + 2

        # Backbone
        self.backbone = TSTiEncoder(num_channels=num_channels, num_patches=num_patches, patch_len=patch_len, num_enc_layers=num_enc_layers,
                                    d_model=d_model, n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, attn_dropout=attn_dropout, dropout=dropout,
                                    act=act, key_padding_mask=key_padding_mask, padding_var=padding_var, attn_mask=attn_mask, res_attention=res_attention,
                                    pre_norm=pre_norm, store_attn=store_attn, pe=pe, learn_pe=learn_pe, verbose=verbose, **kwargs)

        # Head
        self.head_nf = d_model * num_patches
        self.n_vars = num_channels
        self.pretrain_head = pretrain_head
        self.head_type = head_type
        self.individual = individual
        self.return_head = return_head

        self.head = Flatten_Head(self.individual, self.n_vars, self.head_nf, pred_len, head_dropout=head_dropout)

        # Weight initialization
        self.apply(xavier_init)

    def _init_revin(self, revout:bool, revin_affine:bool):
        self._revin = True
        self.revout = revout
        self.revin_affine = revin_affine
        self.revin = RevIN(self.num_channels, self.eps_revin, self.revin_affine)


    def forward(self, z):
        """"Applies the PatchTST model to the input data.
        Args:
            z (torch.Tensor): Input data. Shape (batch_size, num_channels, seq_len)
        Returns:
            torch.Tensor: Output data. Shape (batch_size, num_channels, pred_len)
        """

        # z: [bs x nvars x seq_len]

        # Instance normalization
        if self._revin:
            z = self.revin(z, "norm")

        # Patching
        z = self.patcher(z)                        # z: [bs x nvars x num_patches x patch_len]
        z = z.permute(0,1,3,2)                     # z: [bs x nvars x patch_len x num_patches]

        # Model
        z = self.backbone(z)                   # z: [bs x nvars x d_model x num_patches]

        if self.return_head:
            z = self.head(z)                       # z: [bs x nvars x pred_len]

            # Instance denormalization
            if self.revout:
                z = self.revin(z, "denorm")
        else:
            z = z.permute(0,1,3,2)

        return z


class PatchTSTBlindMasked(nn.Module):
    def __init__(self, num_enc_layers, d_model, d_ff, num_heads, num_channels, seq_len, pred_len, attn_dropout=0.0,
        ff_dropout=0.0, pred_dropout=0.0, batch_first=True, norm_mode="batch1d", revin=True, revout=True, revin_affine=True,
        eps_revin=1e-5, patch_dim=16, patch_stride=1, return_head=True):
        super(PatchTSTBlindMasked, self).__init__()

        # Parameters
        self.patch_dim = patch_dim
        self.num_patches = int((seq_len - patch_dim) / patch_stride) + 2
        self.num_channels = num_channels
        self.eps_revin = eps_revin
        self.revin_affine = revin_affine

        # Initialize layers
        if revin:
            self._init_revin(revout, revin_affine)
        else:
            self._revin = None
            self.revout = None

        self.patcher = Patcher(patch_dim, patch_stride)
        self.pos_enc = PositionalEncoding(patch_dim, d_model, self.num_patches)
        self.backbone = PatchTSTBackbone(num_enc_layers, d_model, d_ff, num_heads, num_channels, self.num_patches, pred_len,
                                 attn_dropout,ff_dropout, pred_dropout, batch_first, norm_mode, return_head)

        # Weight initialization
        self.apply(xavier_init)

    def _init_revin(self, revout:bool, revin_affine:bool):
        self._revin = True
        self.revout = revout
        self.revin_affine = revin_affine
        self.revin = RevIN(self.num_channels, self.eps_revin, self.revin_affine)


    def forward(self, x:torch.Tensor, mask:torch.Tensor=None):
        """
        PatchTSTBlind model with a masked mechanism.

        Args:
            x (torch.Tensor): The input time series tensor of shape (batch_size, num_channels, seq_len).
            mask (torch.Tensor): The mask tensor of shape (batch_size, num_channels, num_patches, 1).
        """

        # RevIN
        if self._revin:
            x = self.revin(x, mode="norm")

        # Patcher
        x = self.patcher(x)

        # Apply mask
        if isinstance(mask, torch.Tensor):
            x = x * mask.unsqueeze(-1)

        # Project + Positional Encoding
        x = self.pos_enc(x)

        # Transformer
        x = self.backbone(x)

        # RevOUT
        if self.revout:
            x = self.revin(x, mode="denorm")

        return x


if __name__ == "__main__":
    batch_size = 64
    num_channels = 1
    seq_len = 96
    patch_dim = 16
    patch_stride = 8
    pred_len = 1

    from sss.layers.dynamic_weights.qwa_loss import QWALoss

    criterion = QWALoss(num_classes=2,
                       ch_loss_refined=True,
                       ch_loss_coarse=True,
                       window_loss_refined=False,
                       window_loss_coarse=False,
                       skew_loss=False,
                       delta=0.1,
                       coeffs=[1., 1., 1., 1., 1.],
                       loss_type="BCE",)

    model = PatchTST(num_enc_layers=3,
                            d_model=128,
                            d_ff=512,
                            num_heads=4,
                            num_channels=num_channels,
                            seq_len=seq_len,
                            pred_len=pred_len,
                            attn_dropout=0.0,
                            ff_dropout=0.0,
                            pred_dropout=0.0,
                            batch_first=True,
                            norm_mode="batch1d",
                            revin=True,
                            revout=False,
                            revin_affine=True,
                            eps_revin=1e-5,
                            patch_dim=patch_dim,
                            stride=patch_stride,
                            return_head=False,
                            head_type="linear",

                            # QWA
                            qwa=True,
                            qwa_num_networks=3,
                            qwa_network_type="inception",
                            qwa_hidden_dim=32,
                            qwa_mlp_dropout=0.0,
                            qwa_attn_dropout=0.3,
                            qwa_ff_dropout=0.3,
                            qwa_norm_mode="None",
                            qwa_num_heads=4,
                            qwa_num_enc_layers=2,
                            qwa_upper_quantile=0.9,
                            qwa_lower_quantile=0.1,)

    # Count number of parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of parameters: {num_params}")

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    optimizer.zero_grad()
    x = torch.randn(batch_size, num_channels, seq_len) # (B, M, L)
    ch_ids = torch.randint(0, 4, (batch_size,))
    targets = torch.randint(0, 2, (batch_size,))

    print(f"Input shape: {x.shape}. Channel IDs shape: {ch_ids.shape}")
    coarse_probs, refined_probs, qwa_coeffs = model(x=x, y=None, ch_ids=ch_ids)
    print(f"Coarse probs: {coarse_probs.shape}. Refined probs: {refined_probs.shape}. QWA coeffs: {len(qwa_coeffs)}")
    device = x.device
    loss = criterion(coarse_probs, refined_probs, targets, ch_ids, qwa_coeffs, device)

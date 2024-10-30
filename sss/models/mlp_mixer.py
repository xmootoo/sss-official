import torch
import torch.nn as nn
import torch.nn.init as init

from sss.layers.patchtst_blind.backbone import SupervisedHead
from sss.layers.patchtst_blind.revin import RevIN
from sss.layers.patchtst_blind.pos_enc import PositionalEncoding
from sss.layers.patcher import Patcher

from sss.layers.mlp_mixer.backbone import MLPMixer
from sss.utils.weight_init import xavier_init
from sss.utils.sincos_pos_emb import *
from typing import Optional


class MLPMixerCI(nn.Module):
    def __init__(self,
                 num_enc_layers,
                 d_model,
                 tok_mixer_dim,
                 cha_mixer_dim,
                 num_channels,
                 seq_len,
                 pred_len,
                 pos_enc_type='sinusoidal',
                 pred_dropout=0.0,
                 dropout=0.0,
                 revin=True,
                 revout=True,
                 revin_affine=True,
                 eps_revin=1e-5,
                 patch_dim=16,
                 patch_stride=1):
        super(MLPMixerCI, self).__init__()

        """
        A Channel-Independent version of the MLP-Mixer model for time series.

        Args:
            pos_enc_type (str): The type of positional encoding to use. Either "learnable", "1d_sincos", or None.
        """

        # Parameters
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
        self.backbone = MLPMixer(self.num_patches,
                                 d_model,
                                 tok_mixer_dim,
                                 cha_mixer_dim,
                                 num_enc_layers,
                                 dropout)
        self.head = SupervisedHead(self.num_patches * d_model, pred_len, pred_dropout)
        self.proj = nn.Linear(patch_dim, d_model)

        # Positional Encoding
        if pos_enc_type == "learnable":
            self.pos_enc = nn.Parameter(torch.zeros(1, self.num_patches, d_model),
                                                    requires_grad=True)
            init.uniform_(self.pos_enc, -0.02, 0.02)

        # 1D sinusoidal encoding (Vanilla Transformer positional encoding)
        elif pos_enc_type == "1d_sincos":
            self.pos_enc = nn.Parameter(torch.zeros(1, self.num_patches, d_model),
                                                    requires_grad=False)
            sincos_enc = get_1d_sincos_pos_embed(embed_dim=d_model,
                                                 grid_size=self.num_patches,
                                                 cls_token=False)
            self.pos_enc.data.copy_(torch.from_numpy(sincos_enc).float().unsqueeze(0))
        else:
            self.pos_enc = None

        # Weight initialization
        self.apply(xavier_init)


    def _init_revin(self, revout :bool, revin_affine :bool):
        self._revin = True
        self.revout = revout
        self.revin_affine = revin_affine
        self.revin = RevIN(self.num_channels, self.eps_revin, self.revin_affine)

    def forward(self, x : torch.Tensor):

        # RevIN
        if self._revin:
            x = self.revin(x, mode="norm")

        # Patcher
        x = self.patcher(x)

        # Projection
        x = self.proj(x)

        # Positional Encoding
        if self.pos_enc is not None:
            x = x + self.pos_enc

        B, M, N, D = x.shape

        # MLPMixer
        x = x.view(B * M, N, D)
        x = self.backbone(x)

        # Linear Head
        x = x.view(B, M, N, D)
        x = self.head(x)

        # RevOUT
        if self.revout:
            x = self.revin(x, mode="denorm")

        return x


# Test
if __name__ == "__main__":
    batch_size = 32; num_channels = 7; seq_len = 512; pred_len = 96;
    x = torch.randn(batch_size, num_channels, seq_len)
    model = MLPMixerCI(num_enc_layers=6,
                       d_model=128,
                       tok_mixer_dim=128,
                       cha_mixer_dim=512,
                       num_channels=num_channels,
                       seq_len=seq_len,
                       pred_len=pred_len,
                       pos_enc_type="learnable")
    out = model(x)
    print(out.shape)

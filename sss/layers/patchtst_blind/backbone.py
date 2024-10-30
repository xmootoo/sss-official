import torch
import torch.nn as nn
import torch.nn.functional as F
from sss.utils.utils import *
from sss.layers.patchtst_blind import *
from sss.layers.channel_modules.channel_aggr import ChannelAggregator, ChannelLatentMixer
from sss.layers.dynamic_weights.qwa import QWA
from sss.layers.patchtst_blind.enc_block import SupervisedHead, EncoderBlock

class PatchTSTBackbone(nn.Module):
    def __init__(self, num_enc_layers, d_model, d_ff, num_heads, num_channels, num_patches, pred_len, attn_dropout=0.0,
        ff_dropout=0.0, pred_dropout=0.0, batch_first=True, norm_mode="batch1d", return_head=True, head_type="linear", ch_aggr=False,
        ch_reduction="mean", cla_mix=False, cla_mix_layers=1, cla_combination="concat_patch_dim", qwa=False, qwa_num_networks=3,
        qwa_network_type="mlp", qwa_hidden_dim=2, qwa_mlp_dropout=0.0, qwa_attn_dropout=0.0, qwa_ff_dropout=0.0,
        qwa_norm_mode="batch1d", qwa_num_heads=3, qwa_num_enc_layers=1, qwa_upper_quantile=0.9, qwa_lower_quantile=0.1,):
        super(PatchTSTBackbone, self).__init__()


        # Parameters
        self.num_channels = num_channels
        self.num_patches = num_patches
        self.d_model = d_model
        self.return_head = return_head
        self.cla = ch_aggr
        self.clm = cla_mix
        self.qwa = qwa
        self.pred_len = pred_len

        # Encoder
        self.enc = nn.Sequential(*(EncoderBlock(d_model, d_ff, num_heads, num_channels, num_patches, attn_dropout, ff_dropout,
                                                batch_first, norm_mode) for i in range(num_enc_layers)))

        # Channel Latent Mixing (optional)
        if self.clm:
            cla_mix_dim = 2*d_model if cla_combination=="concat_embed_dim" else d_model
            cla_ff_dim = 2*d_ff if cla_combination=="concat_embed_dim" else d_ff
            cla_num_patches = 2*num_patches if cla_combination=="concat_patch_dim" else num_patches
            self.cla_mix = ChannelLatentMixer(ch_reduction, cla_combination)
            self.cla_mix_enc = nn.Sequential(*(EncoderBlock(cla_mix_dim, cla_ff_dim, num_heads, num_channels, cla_num_patches, attn_dropout, ff_dropout,
                                                    batch_first, norm_mode) for i in range(cla_mix_layers)))

        # Channel Latent Aggregation (optional)
        if self.cla:
            self.ch_aggr = ChannelAggregator(num_patches*d_model, ch_reduction)

        # Quartile-based Dynamic Weighted Aggregation (optional)
        if self.qwa:
            self.qwa_module = QWA(num_networks=qwa_num_networks,
                                  network_type=qwa_network_type,
                                  hidden_dim=qwa_hidden_dim,
                                  d_model=d_model,
                                  d_ff=d_ff,
                                  num_patches=self.num_patches,
                                  mlp_dropout=qwa_mlp_dropout,
                                  attn_dropout=qwa_attn_dropout,
                                  ff_dropout=qwa_ff_dropout,
                                  batch_first=batch_first,
                                  norm_mode=qwa_norm_mode,
                                  num_heads=qwa_num_heads,
                                  num_channels=num_channels,
                                  num_enc_layers=qwa_num_enc_layers,
                                  upper_quantile=qwa_upper_quantile,
                                  lower_quantile=qwa_lower_quantile)

        # Prediction head
        head_in_dim = num_patches*d_model*2 if self.clm else num_patches*d_model
        head_hidden_dim = num_patches*d_model // 2 if self.clm else num_patches*d_model
        self.flatten = nn.Flatten(start_dim=-2)
        if head_type=="linear":
            self.head = SupervisedHead(head_in_dim, pred_len, pred_dropout)
        elif head_type=="mlp":
            self.head = nn.Sequential(nn.Linear(head_in_dim, head_hidden_dim),
                                        nn.GELU(),
                                        nn.Dropout(pred_dropout),
                                        nn.Linear(head_hidden_dim, pred_len))
        else:
            raise ValueError(f"Invalid head type: {head_type}")

    def forward(self, x:torch.Tensor, y:torch.Tensor=None, ch_ids:torch.Tensor=None) -> torch.Tensor:

        # Encoding
        batch_size = x.shape[0]
        x = x.view(batch_size * self.num_channels, self.num_patches, -1) # (batch_size * num_channels, num_patches, d_model)
        x = self.enc(x) # (batch_size * num_channels, num_patches, d_model)

        # Other modules
        if self.cla and self.return_head:
            x = self.flatten(x) # (batch_size, num_channels, num_patches*d_model)
            z = x.squeeze() # Remove any singleton dimensions: (batch_size, num_patches*d_model)
            z_ch, y_ch = self.ch_aggr(z, y, ch_ids) # Aggregate channels # (num_channels, num_patches*d_model)
            out = (self.head(z_ch), y_ch)
        elif self.clm:
            z = self.cla_mix(x, ch_ids) # (batch_size * num_channels, num_patches, 2*d_model) OR (batch_size * num_channels, 2*num_patches, d_model)
            z = self.cla_mix_enc(z) # (batch_size * num_channels, num_patches, 2*d_model) OR (batch_size * num_channels, 2*num_patches, d_model)
            z = self.flatten(z) # (batch_size, num_channels, 2*num_patches*d_model)
            out = self.head(z)
        elif self.qwa:
            z = x.clone()
            x = self.flatten(x)
            x = self.head(x)
            q = torch.sigmoid(x).squeeze() if self.pred_len==1 else torch.log_softmax(x, dim=-1).squeeze()
            refined_probs, qwa_coeffs = self.qwa_module(z, q, ch_ids)
            out = (q, refined_probs, qwa_coeffs)
        elif self.return_head:
            x = self.flatten(x) # (batch_size, num_channels, num_patches*d_model)
            out = self.head(x)
        else:
            out = x.view(batch_size, self.num_channels, self.num_patches, -1)

        return out

import torch
from sss.utils.weight_init import xavier_init
import torch.nn as nn
import numpy as np
import math

# Transformer Encoder Blocks
from sss.layers.patchtst_blind.enc_block import EncoderBlock
from sss.layers.patchtst_original.backbone import TSTEncoderLayer as OGEncoderBlock

# InceptionBlock
from sss.layers.timesnet.inception_block import InceptionNet1D

# Weight initialization
from sss.utils.weight_init import xavier_init


class QWA(nn.Module):
    def __init__(self, num_networks=3,
                       network_type="mlp",
                       hidden_dim=2,
                       d_model=128,
                       d_ff=256,
                       num_patches=64,
                       mlp_dropout=0.,
                       attn_dropout=0.,
                       ff_dropout=0.,
                       batch_first=True,
                       norm_mode="batch1d",
                       num_heads=3,
                       num_channels=1,
                       num_enc_layers=1,
                       upper_quantile=0.9,
                       lower_quantile=0.1,
                       append_q=False,
                       final_norm="None",
                       ddp=False,):
        super(QWA, self).__init__()

        """
        Quartile-based Dynamic Weighted Average (QWA) or (QDWA).
        """

        self.upper_quantile = upper_quantile
        self.lower_quantile = lower_quantile
        self.append_q = append_q
        self.d_model = (d_model + 1) if append_q else d_model
        self.latent_dim = self.d_model*num_patches
        self.hidden_dim = hidden_dim
        self.num_networks = num_networks
        self.append_q = append_q

        if network_type=="mlp":
            self.backbone = nn.ModuleList([
                        nn.Sequential(
                            nn.Flatten(start_dim=-2),
                            nn.Linear(self.latent_dim, self.hidden_dim),
                            nn.GELU(),
                            nn.Dropout(mlp_dropout),
                            nn.Linear(self.hidden_dim, self.latent_dim),
                        ) for _ in range(num_networks)
                    ])
        elif network_type=="attn":
            self.backbone = nn.ModuleList(
                        [nn.Sequential(
                            *(EncoderBlock(self.d_model, d_ff, num_heads, num_channels, num_patches, attn_dropout, ff_dropout,
                                           batch_first, norm_mode) for i in range(num_enc_layers)),
                            nn.Flatten(start_dim=-2)
                        ) for _ in range(num_networks)]
                    )
            # self.backbone = nn.ModuleList(
            #     [nn.Sequential(*(OGEncoderBlock(q_len=num_patches, d_model=d_model, num_heads=num_heads, d_k=None, d_v=None, d_ff=d_ff, norm_mode=norm_mode,
            #                                     attn_dropout=attn_dropout, dropout=ff_dropout, activation="gelu", res_attention=False,
            #                                     pre_norm=False, store_attn=False) for i in range(num_enc_layers))) for _ in range(num_networks)])
        elif network_type=="inception":
            self.backbone = nn.ModuleList([InceptionNet1D(in_channels=self.d_model, num_blocks=num_enc_layers, out_channels=hidden_dim, pool_size=2) for _ in range(num_networks)])
        elif network_type=="linear":
            self.backbone = nn.ModuleList(nn.Flatten(start_dim=-2) for _ in range(num_networks))
        else:
            raise ValueError(f"Invalid network type: {network_type}")

        self.head = nn.ModuleList([nn.Linear(self.latent_dim, 1) for _ in range(num_networks)]) if network_type!="inception" else nn.ModuleList([nn.Identity() for _ in range(num_networks)])

        self.apply(xavier_init)

    def quantile_separator(self, q, ch_ids):
        """
        For each channel, this function separates its coarse probabilites into quantiles (upper, normal, lower).

        Args:
            q (torch.Tensor): Coarse probabilities of shape (B,).
            ch_ids (torch.Tensor): Channel IDs of shape (B,).
        Returns:
            q_idx (list): A list where each entry is [ch_id, q_lower_idx, q_normal_idx, q_upper_idx], where ch_id is the
                            channel ID, q_lower_idx is the indices of its lower quantile, q_normal_idx is the indices its
                            normal quantile, and q_upper_idx is the indices of its upper quantile.

        """

        device = q.device
        q_idx = []
        q_upper_total = []
        q_lower_total = []
        q_normal_total = []

        for ch_id in torch.unique(ch_ids):
            mask = (ch_ids == ch_id).nonzero(as_tuple=True)[0]
            q_ch = q[mask]

            # Torch Version (preferred)
            n = len(q_ch)
            sample_adjusted_upper = torch.ceil(torch.tensor((n + 1) * self.upper_quantile, dtype=torch.float64, device=device)) / n
            sample_adjusted_lower = torch.floor(torch.tensor((n + 1) * self.lower_quantile, dtype=torch.float64, device=device)) / n

            # Use item() to get Python float for comparison
            if sample_adjusted_upper.item() > 1:
                sample_adjusted_upper = self.upper_quantile
            if sample_adjusted_lower.item() < 0:
                sample_adjusted_lower = self.lower_quantile

            # Compute quantiles
            upper_quantile = torch.quantile(q_ch.to(torch.float64), sample_adjusted_upper, interpolation='lower')
            lower_quantile = torch.quantile(q_ch.to(torch.float64), sample_adjusted_lower, interpolation='lower')


            # # NumPy Version (deprecated)
            # n = len(q_ch)
            # sample_adjusted_upper = math.ceil((n+1)*(self.upper_quantile))/n
            # sample_adjusted_lower = math.floor((n+1)*(self.lower_quantile))/n

            # if sample_adjusted_upper > 1:
            #     sample_adjusted_upper = self.upper_quantile
            # if sample_adjusted_lower < 0:
            #     sample_adjusted_lower = self.lower_quantile

            # # Compute quantiles
            # upper_quantile = np.quantile(q_ch.detach().cpu().numpy(), sample_adjusted_upper, method='lower')
            # lower_quantile = np.quantile(q_ch.detach().cpu().numpy(), sample_adjusted_lower, method='lower')

            # Compute indices of quantiles
            q_upper_idx = (q_ch >= upper_quantile).nonzero(as_tuple=True)[0]
            q_lower_idx = (q_ch <= lower_quantile).nonzero(as_tuple=True)[0]
            q_normal_idx = ((q_ch > lower_quantile) & (q_ch < upper_quantile)).nonzero(as_tuple=True)[0]

            q_idx.append([ch_id, q_lower_idx, q_normal_idx, q_upper_idx])
            q_upper_total.append(q_upper_idx)
            q_lower_total.append(q_lower_idx)
            q_normal_total.append(q_normal_idx)

        q_upper_tensor = torch.cat(q_upper_total)
        q_lower_tensor = torch.cat(q_lower_total)
        q_normal_tensor = torch.cat(q_normal_total)

        return q_idx, q_upper_tensor, q_lower_tensor, q_normal_tensor

    def forward(self, z, q, ch_ids):
        """
        Args:
            z (torch.Tensor): Latent representation of the input data of shape (B, N, D) where B is the batch size,
                              N is the number of patches, and D=d_model is the embedding dimension.
            q (torch.Tensor): Coarse probabilities of shape (B,).
            ch_ids (torch.Tensor): Channel IDs of shape (B,).
        Returns:
            refined_probs (torch.Tensor): Refined probabilities of shape (B,) in the same indices as q.
            qwa_coeffs (list): A qwa coefficients for each channel, where each entry is of the form
                              [ch_id, q_lower_coeff, q_normal_coeff, q_upper_coeff].
        """
        qwa_coeff = torch.zeros_like(q)
        logits = torch.zeros_like(q)
        refined_probs = torch.zeros_like(q)

        # Append coarse probabilities to each embedding (Optional)
        if self.append_q:
            # Append q to the final dimension of z
            B, N, _ = z.size()
            q_repeated = q.unsqueeze(1).expand(-1, N).unsqueeze(-1)  # (B, N, 1)
            z = torch.cat([z, q_repeated], dim=-1)  # (B, N, d_model) -> (B, N, d_model+1)

        # Get anomalies and normals for each channel. Separate z accordingly
        q_idx, q_upper, q_lower, q_normal = self.quantile_separator(q, ch_ids)
        z_upper, z_normal, z_lower = z[q_upper], z[q_normal], z[q_lower]

        # Refine z_upper, z_normal, z_lower and compute logits
        if self.num_networks==1:
            z = self.backbone[0](z)
            logits = self.head[0](z).squeeze()
        elif self.num_networks==2:
            z_anom = torch.cat([z_upper, z_lower], dim=0)
            q_anom = torch.cat([q_upper, q_lower], dim=0)
            z_anom = self.backbone[0](z_anom)
            z_normal = self.backbone[1](z_normal)

            # Insert anom_logits and normal_logits at correct indices according to q
            logits[q_anom] = self.head[0](z_anom).squeeze()
            logits[q_normal] = self.head[1](z_normal).squeeze()

        elif self.num_networks==3:
            z_upper = self.backbone[0](z_upper)
            z_normal = self.backbone[1](z_normal)
            z_lower = self.backbone[2](z_lower)
            logits[q_upper] = self.head[0](z_upper).squeeze()
            logits[q_normal] = self.head[1](z_normal).squeeze()
            logits[q_lower] = self.head[2](z_lower).squeeze()
        else:
            raise ValueError(f"Invalid number of networks: {self.num_networks}")

        qwa_coeff_list = []
        for c in q_idx:
            idx = torch.cat([c[1], c[2], c[3]], dim=0)
            qwa_coeff[idx] = torch.softmax(logits[idx], dim=0).squeeze(-1)

            # For logging purpose and QWA SkewLoss
            qwa_coeff_list.append([c[0], qwa_coeff[c[1]], qwa_coeff[c[2]], qwa_coeff[c[3]]])

        # Refine coarse probabilities through a convex combination over each channel's probabilities'
        refined_probs = q*qwa_coeff

        return refined_probs, qwa_coeff_list


# Test
if __name__ == "__main__":
    B = 8192
    d_model = 16
    d_ff = 64
    num_patches = 64
    mlp_dropout = 0.1
    attn_dropout = 0.1
    ff_dropout = 0.1
    batch_first = True
    norm_mode = "batch1d"
    num_heads = 2
    num_channels = 1
    num_enc_layers = 1
    upper_quantile = 0.9
    lower_quantile = 0.1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    z = torch.randn(B, num_patches, d_model).to(device)
    q = torch.rand(B,).to(device)
    ch_ids = torch.randint(0, 5, (B,)).to(device)

    print(f"Coarse probabilities: {q.shape}")
    print(f"Latent representations: {z.shape}")

    model = QWA(num_networks=2,
                network_type="mlp",
                hidden_dim=32,
                d_model=d_model,
                d_ff=d_ff,
                num_patches=num_patches,
                mlp_dropout=mlp_dropout,
                attn_dropout=attn_dropout,
                ff_dropout=ff_dropout,
                batch_first=batch_first,
                norm_mode=norm_mode,
                num_heads=num_heads,
                num_channels=num_channels,
                num_enc_layers=num_enc_layers,
                upper_quantile=upper_quantile,
                lower_quantile=lower_quantile,
                append_q=False).to(device)

    refined_probs, qwa_coeffs = model(z, q, ch_ids)
    print(f"Refined probabilities: {refined_probs.shape}")
    for q in qwa_coeffs:
        print(f"QWA coefficients for channel {q[0]}: {q[1].shape}, {q[2].shape}, {q[3].shape}")
        break

import torch
import torch.nn as nn
from pydantic import BaseModel
from sss.layers.patchtst_blind.revin import RevIN
from sss.layers.patchtst_blind.pos_enc import PositionalEncoding
from sss.layers.patcher import Patcher
from sss.layers.patchtst_blind.backbone import SupervisedHead
from sss.layers.channel_modules.channel_aggr import ChannelLatentMixer

# Supervised backbones
from sss.models.patchtst_blind import PatchTST
from sss.models.timesnet import TimesNet
from sss.models.dlinear import DLinear
from sss.models.modern_tcn import ModernTCN
from sss.models.recurrent import RecurrentModel

# Utils
from sss.utils.utils import Reshape

# Monte Carlo Dropout
from sss.layers.monte_carlo.mcd import MonteCarloDropout



class SSS(nn.Module):
    def __init__(
        self,
        args,
        seq_len,
        pred_len,
        num_channels,
        backbone_id,
        mcd=False,
        mcd_samples=30,
        mcd_stats=["mean", "var", "entropy"],
        mcd_prob=0.2,
        clm_config=None,
        clm=False,
        revin=False,
        revin_affine=False,
        revout=False,
        eps_revin=1e-5,
        patch_dim=16,
        patch_stride=8,
        norm_mode="batch1d",
        return_head=True,
        pred_dropout=0.0,
        wavelet_transform=None,
        sparse_context:BaseModel=None,
    ) -> None:
        super(SSS, self).__init__()
        """
        Stochastic Sparse Sampling (SSS).
        """

        # Parameters
        self.num_channels = num_channels
        self.return_head = return_head
        self.mcd = mcd
        self.clm = clm
        self.pred_len = pred_len
        self.seq_len = seq_len
        self.revin = revin
        self.eps_revin = eps_revin
        self.revin_affine = revin_affine
        self.revout = revout
        self.backbone_id = backbone_id
        self.patch_dim = patch_dim
        self.patch_stride = patch_stride
        self.backbone_config = getattr(args, backbone_id.lower())
        self.sparse_context = sparse_context

        # Layers
        if backbone_id=="PatchTST":
            self.num_patches = int((seq_len - patch_dim) / patch_stride) + 2
            self.patcher = Patcher(patch_dim, patch_stride)
            self.pos_enc = PositionalEncoding(patch_dim, self.backbone_config.d_model, self.num_patches)


        # TODO: Add utils/wavelets.py transform function here
        if wavelet_transform:
            self.wavelet_transform = ...

        # Backbone
        self.backbone = self.get_backbone(backbone_id)

        # Monte Carlo Dropout (optional)
        if self.mcd:
            self.mcd_layer = MonteCarloDropout(
                num_samples=mcd_samples,
                num_classes=pred_len,
                stats=mcd_stats,
                mcd_prob=mcd_prob)
            self.stats_enc = nn.Linear(len(mcd_stats), 1)

        # TODO: Implement Channel Latent Mixing (optional)
        if self.clm:
            # Clone backbone config and adjust for CLM
            clm_backbone_config = self.backbone_config.clone()

            if backbone_id=="PatchTST":
                clm_backbone_config.d_model = 2*self.backbone_config.d_model if clm_config.combo=="concat_embed_dim" else self.backbone_config.d_model
                cla_ff_dim = 2*clm_config.d_ff if clm_config.combo=="concat_embed_dim" else self.backbone_config.d_ff
                cla_num_patches = 2*self.num_patches if clm_config.combo=="concat_patch_dim" else self.num_patches
            self.clmixer = ChannelLatentMixer(clm_config.reduction, clm_config.combo)
            self.clm_backbone = self.get_backbone(backbone_id, clm_config)

        # Prediction head
        if backbone_id=="PatchTST":
            head_in_dim = self.num_channels*self.num_patches*self.backbone_config.d_model
            self.flatten = nn.Flatten(start_dim=-3)
        elif backbone_id=="TimesNet":
            head_in_dim = self.seq_len*self.backbone_config.d_model
            self.flatten = nn.Flatten(start_dim=-2)
        elif backbone_id=="DLinear":
            head_in_dim = self.num_channels*self.seq_len
            self.flatten = nn.Flatten(start_dim=-2)
        elif backbone_id=="ModernTCN":
            head_in_dim = self.backbone.head_in_dim
            self.flatten = nn.Flatten(start_dim=-3)
        else:
            raise ValueError(f"Backbone {backbone_id} not recognized. Please use one of ['PatchTST', 'TimesNet', 'DLinear', 'ModernTCN']")

        self.head = nn.Sequential(
                self.flatten,
                nn.Dropout(pred_dropout),
                nn.Linear(head_in_dim, self.pred_len)
            )

        self.full_backbone = nn.Sequential(
                self.backbone,
                self.head
            )

        if sparse_context.sparse_context:
            context_in_dim = sparse_context.d_model
            return_context_head = True if sparse_context.bidirectional else False

            self.context_embed = nn.Sequential(
                    self.flatten,
                    nn.Dropout(pred_dropout),
                    nn.Linear(head_in_dim, context_in_dim)
                )
            self.context_norm = nn.LayerNorm(context_in_dim)
            self.context_enc = RecurrentModel(
                d_model=context_in_dim,
                num_enc_layers=sparse_context.num_enc_layers,
                pred_len=context_in_dim,
                backbone_id=sparse_context.backbone_id,
                bidirectional=sparse_context.bidirectional,
                dropout=sparse_context.dropout,
                num_channels=context_in_dim,
                norm_mode=sparse_context.norm_mode,
                last_state=sparse_context.last_state,
                avg_state=sparse_context.avg_state,
                return_head=return_context_head
            )

            if sparse_context.final_norm=="batch1d":
                self.norm = nn.BatchNorm1d(head_in_dim)
            elif sparse_context.final_norm=="layer":
                self.norm = nn.LayerNorm(head_in_dim)
            else:
                self.norm = nn.Identity()

            if sparse_context.combine=="concat":
                self.context_head = nn.Sequential(self.norm, nn.Linear(context_in_dim*2, pred_len))
            elif sparse_context.combine=="add":
                self.context_head = nn.Sequential(self.norm, nn.Linear(context_in_dim, pred_len))
            else:
                raise ValueError(f"Combine method {sparse_context.combine} not recognized. Please use one of ['concat', 'add']")

    def sparse_context_enc(self, x:torch.Tensor, ch_ids:torch.Tensor, t:torch.Tensor) -> torch.Tensor:
        """

        Args:
            x (torch.tensor): Shape (batch_size, *) where * is the model output dimension, the model encodings for each window.
            ch_ids (torch.tensor): Shape (batch_size,) consisting of the channel IDs for each window.
            t (torch.tensor): Shape (batch_size,) where for each channel (upon masking), provides the unique time index for each window.
        """

        out_dim = self.sparse_context.d_model*2 if self.sparse_context.combine=="concat" else self.sparse_context.d_model
        logits = torch.zeros(x.size(0), self.sparse_context.d_model)

        for ch_id in torch.unique(ch_ids):
            mask = ch_ids==ch_id
            x_masked = x[mask]
            t_masked = t[mask]

            # Embedding -> Context Dimension
            x_embed = self.context_embed(x_masked) # (B, *) -> (B, context_dim)

            # Sort by time indices of the windows
            _, indices = torch.sort(t_masked)

            # Order x_masked by time indices
            x_embed_sort = x_embed[indices]

            # Apply RecurrentModel -> Context Encoding
            c = self.context_enc(x_embed_sort).squeeze()

            if self.sparse_context.combine == "add":
                logits[mask] = x_embed + c.unsqueeze(0)
            elif self.sparse_context.combine == "concat":
                # Append c to every example of x_embed_og
                c_exp = c.unsqueeze(0).expand(x_embed.size(0), -1)  # Shape: (num_windows, context_dim)
                logits[mask] = torch.cat([x_embed, c_exp], dim=-1) # Shape: (num_windows, 2*context_dim)

        return self.context_head(self.norm(logits))


    def get_backbone(self, backbone_id):
        config = self.backbone_config

        if backbone_id=="PatchTST":
            return PatchTST(num_enc_layers=config.num_enc_layers,
                            d_model=config.d_model,
                            d_ff=config.d_ff,
                            num_heads=config.num_heads,
                            num_channels=self.num_channels,
                            seq_len=self.seq_len,
                            pred_len=self.pred_len,
                            attn_dropout=config.attn_dropout,
                            ff_dropout=config.ff_dropout,
                            pred_dropout=0.0,
                            batch_first=True,
                            norm_mode=config.norm_mode,
                            return_head=False,
                            head_type="linear",
                            revin=self.revin,
                            revout=self.revout,
                            revin_affine=self.revin_affine,
                            eps_revin=self.eps_revin,
                            patch_dim=self.patch_dim,
                            stride=self.patch_stride,)
        elif backbone_id=="TimesNet":
            return TimesNet(self.seq_len,
                            self.pred_len,
                            self.num_channels,
                            config.d_model,
                            config.d_ff,
                            config.num_enc_layers,
                            config.num_kernels,
                            config.c_out,
                            config.top_k,
                            config.dropout,
                            task="classification",
                            revin=self.revin,
                            revin_affine=self.revin_affine,
                            revout=self.revout,
                            eps_revin=self.revin_affine,
                            return_head=False)
        elif backbone_id=="DLinear":
            return DLinear(task="classification",
                           seq_len=self.seq_len,
                           pred_len=self.pred_len,
                           num_channels=self.num_channels,
                           num_classes=self.pred_len,
                           moving_avg=config.moving_avg,
                           individual=config.individual,
                           return_head=False)
        elif backbone_id=="ModernTCN":
            return ModernTCN(seq_len=self.seq_len,
                             pred_len=self.pred_len,
                             patch_dim=self.patch_dim,
                             patch_stride=self.patch_stride,
                             num_classes=self.pred_len,
                             num_channels=self.num_channels,
                             task="classification",
                             return_head=False,
                             dropout=config.dropout,
                             class_dropout=config.class_dropout,
                             ffn_ratio=config.ffn_ratio,
                             num_enc_layers=config.num_enc_layers,
                             large_size=config.large_size,
                             small_size=config.small_size,
                             d_model=config.d_model,
                             dw_dims=config.dw_dims,
                             revin=self.revin,
                             affine=self.revin_affine)
        else:
            raise ValueError(f"Backbone {backbone_id} not recognized. Please use one of ['PatchTST', 'TimesNet', 'DLinear', 'ModernTCN']")

    def forward(self, x:torch.Tensor, y:torch.Tensor=None, ch_ids:torch.Tensor=None, t:torch.Tensor=None) -> torch.Tensor:
        """"

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_channels, seq_len).
            y (torch.Tensor): Target tensor of shape (batch_size, pred_len).
            ch_ids (torch.Tensor): Channel indices tensor of shape (batch_size,).
            t (torch.Tensor): Time indices tensor of shape tracking the relative time index for each window in each channel (batch_size,).
        Returns:
            out (torch.Tensor): Output tensor of shape (batch_size, pred_len).
            u (torch.Tensor): Uncertainty tensor of shape (batch_size, pred_len) (Optional).

        """

        # Other modules
        if self.clm:
            x = self.backbone(x)
            z = self.cla_mix(x, ch_ids)
            z = self.cla_mix_enc(z)
            out = self.head(z)
        elif self.mcd:
            out = self.full_backbone(x) # Logits
            _, s = self.mcd_layer(x, self.full_backbone) # MCD statistics
            u = self.stats_enc(s)
            return (out, u) # (logits, uncertainty/confidence embeddings)
        elif self.sparse_context.sparse_context:
            x = self.backbone(x)
            x = self.sparse_context_enc(x, ch_ids, t)
        else:
            out = self.full_backbone(x)
            return out

# Test
if __name__=="__main__":

    from sss.config.config import PatchTST as PatchTSTConfig, \
                                               TimesNet as TimesNetConfig, \
                                               DLinear as DLinearConfig, \
                                               ModernTCN as ModernTCNConfig


    # Test out each backbone (w/o CLM)
    batch_size = 32
    seq_len = 512
    pred_len = 1
    num_channels = 1
    clm_config=None
    clm=False
    revin=True
    revin_affine=True
    revout=False
    eps_revin=1e-5
    patch_dim=16
    patch_stride=8
    norm_mode="batch1d"
    return_head=True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.randn(batch_size, num_channels, seq_len).to(device)
    backbone_ids = ["PatchTST", "TimesNet", "DLinear", "ModernTCN"]

    from sss.config.config import Global
    args = Global()

    for i in range(4):
        print(f"Testing {backbone_ids[i]} backbone")
        # PatchTST
        backbone_id = backbone_ids[i]
        model = SSS(
            args,
            seq_len,
            pred_len,
            num_channels,
            backbone_id,
            mcd=True,
            mcd_samples=30,
            mcd_stats=["mean", "var", "entropy"],
            clm_config=None,
            clm=False,
            revin=revin,
            revin_affine=revin_affine,
            revout=revout,
            eps_revin=eps_revin,
            patch_dim=patch_dim,
            patch_stride=patch_stride,
            norm_mode=norm_mode,
            return_head=True,
            pred_dropout=0.1,).to(device)
        q, u = model(x)
        print(f"Input shape: {x.shape} for {backbone_id} backbone")
        print(f"Output shape: {q.shape} for {backbone_id} backbone")
        print(f"Uncertainty shape: {u.shape} for {backbone_id} backbone")

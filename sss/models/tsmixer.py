import torch
import torch.nn as nn
from sss.utils.weight_init import xavier_init
from sss.layers.patchtst_blind.revin import RevIN
# Taken from: https://github.com/thuml/Time-Series-Library/blob/main/models/TSMixer.py

class ResBlock(nn.Module):
    def __init__(
        self,
        seq_len,
        d_model,
        dropout,
        num_channels,):
        super(ResBlock, self).__init__()

        self.temporal = nn.Sequential(
            nn.Linear(seq_len, d_model),
            nn.GELU(),
            nn.Linear(d_model, seq_len),
            nn.Dropout(dropout)
        )

        self.channel = nn.Sequential(
            nn.Linear(num_channels, d_model),
            nn.GELU(),
            nn.Linear(d_model, num_channels),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        # x: [B, L, D]
        x = x + self.temporal(x.transpose(1, 2)).transpose(1, 2)
        x = x + self.channel(x)

        return x


class TSMixer(nn.Module):
    def __init__(
        self,
        seq_len,
        pred_len,
        num_enc_layers,
        d_model,
        num_channels,
        dropout=0.,
        revin=True,
        revin_affine=True,
        revout=False,
        eps_revin=1e-5):
        super(TSMixer, self).__init__()

        # Parameters
        self.num_enc_layers = num_enc_layers
        self.eps_revin = eps_revin
        self.revin_affine = revin_affine
        self.num_channels = num_channels
        self.pred_len = pred_len
        self.seq_len = seq_len

        # Layers
        self.backbone = nn.ModuleList([ResBlock(seq_len, d_model, dropout, num_channels)
                                    for _ in range(num_enc_layers)])

        self.head = nn.Linear(seq_len, pred_len)


        # Initialize layers
        if revin:
            self._init_revin(revout, revin_affine)
        else:
            self._revin = None
            self.revout = None

        self.apply(xavier_init)

    def _init_revin(self, revout:bool, revin_affine:bool):
        self._revin = True
        self.revout = revout
        self.revin_affine = revin_affine
        self.revin = RevIN(self.num_channels, self.eps_revin, self.revin_affine)

    def forward(self, x):

        # RevIN
        if self._revin:
            x = self.revin(x, mode="norm")

        x = x.permute(0, 2, 1) # (batch_size, num_channels, seq_len) => (batch_size, seq_len, num_channels)
        for i in range(self.num_enc_layers):
            x = self.backbone[i](x)

        out = self.head(x.transpose(1, 2))

        # RevOUT
        if self.revout:
            out = self.revin(out, mode="denorm")

        return out


# Test
if __name__ == "__main__":
    batch_size = 32
    seq_len = 512
    pred_len = 96
    num_channels = 7
    num_enc_layers = 3
    d_model = 16
    dropout = 0.1
    revin=True
    revin_affine=True
    revout=True
    eps_revin=1e-5

    x = torch.rand(batch_size, num_channels, seq_len)

    model = TSMixer(
        seq_len=seq_len,
        pred_len=pred_len,
        num_enc_layers=num_enc_layers,
        d_model=d_model,
        dropout=dropout,
        num_channels=num_channels,
        revin=revin,
        revin_affine=revin_affine,
        revout=revout,
        eps_revin=eps_revin
    )

    y = model(x)
    print(f"x: {x.shape} => y: {y.shape}")

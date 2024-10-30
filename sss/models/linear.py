import torch
import torch.nn as nn

class Linear(nn.Module):
    def __init__(self, in_features,
                       out_features,
                       norm_mode="layer"):
        super(Linear, self).__init__()

        # Normalization
        if norm_mode=="layer":
            self.norm = nn.LayerNorm(self.d_model)
        elif norm_mode=="batch1d":
            self.norm = nn.BatchNorm1d(self.d_model)
        else:
            self.norm = nn.Identity()

        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x):
        x = self.norm(x.squeeze())
        out = self.linear(x)
        return out

import torch
import torch.nn as nn
from sss.layers.patchtst_original.backbone import PatchTSTBackbone


class PatchTST(nn.Module):
    def __init__(self, num_channels=7, seq_len=512, pred_len=96, patch_dim=16, stride=8, num_enc_layers=3, d_model=128, num_heads=16, d_ff=256,
                       norm_mode="batch1d", attn_dropout=0., dropout=0., ff_dropout=0., pred_dropout=0., individual=False, revin=True, revin_affine=True):
        super(PatchTST, self).__init__()
        self.backbone = PatchTSTBackbone(num_channels=num_channels, seq_len=seq_len, pred_len=pred_len, patch_dim=patch_dim, stride=stride,
                                         num_enc_layers=num_enc_layers, d_model=d_model, num_heads=num_heads, d_ff=d_ff, norm_mode=norm_mode,
                                         attn_dropout=attn_dropout, dropout=dropout, ff_dropout=ff_dropout, pred_dropout=pred_dropout, individual=individual,
                                         revin=revin, revin_affine=revin_affine)

    def forward(self, x):
        """
        Args:
            x: torch.Tensor of shape (batch_size, num_channels, seq_len)
        Returns:
            out: torch.Tensor of shape (batch_size, num_channels, pred_len)
        """
        out = self.backbone(x)
        return out



if __name__ == "__main__":
    print("yo")
    model = PatchTST()
    x = torch.randn(32, 7, 512)
    out = model(x)
    print(out.shape)    # Expected output: "torch.Size([32, 512, 7])

import torch
import torch.nn as nn

from sss.layers.channel_modules.ch_loss import ChannelLoss

class QWALoss(nn.Module):
    def __init__(self, num_classes=2,
                       ch_loss_refined=True,
                       ch_loss_coarse=True,
                       window_loss_refined=False,
                       window_loss_coarse=False,
                       skew_loss=False,
                       delta=0.1,
                       coeffs=[1., 1., 1., 1., 1.],
                       loss_type="BCE",):
        super(QWALoss, self).__init__()

        # Parameters
        self.ch_loss_refined = ch_loss_refined
        self.ch_loss_coarse = ch_loss_coarse
        self.window_loss_refined = window_loss_refined
        self.window_loss_coarse = window_loss_coarse
        self.skew_loss = skew_loss
        self.delta = delta
        self.coeffs = coeffs

        # Loss Functions
        if loss_type=="BCE":
            self.window_loss = nn.BCELoss()
        elif loss_type=="CE":
            self.window_loss = nn.NLLLoss()

        if ch_loss_refined or ch_loss_coarse:
            self.ch_loss = ChannelLoss(loss_type=loss_type, num_classes=num_classes)

    def compute_skew_loss(self, qwa_coeffs, device):
        """
        Args:
            qwa_coeffs (list): A qwa coefficients for each channel, where each entry is of the form
                                [ch_id, q_lower_coeff, q_normal_coeff, q_upper_coeff].
        Returns:
            skew_loss (torch.Tensor): Skew loss value of shape (num_channels,), which computes the skewness of the
                                      QWA coefficients.
        """
        num_channels = len(qwa_coeffs)
        skew_losses = torch.zeros(num_channels, device=device)

        for i, qwa_coeff in enumerate(qwa_coeffs):
            lower_mass = torch.sum(qwa_coeff[1])
            normal_mass = torch.sum(qwa_coeff[2])
            upper_mass = torch.sum(qwa_coeff[3])
            skew_losses[i] = torch.relu(self.delta**2 - torch.norm(lower_mass - upper_mass, p=2))

        avg_skew_loss = torch.mean(skew_losses)

        return avg_skew_loss

    def forward(self, coarse_probs, refined_probs, targets, ch_ids, qwa_coeffs, device):
        """
        Args:
            targets (torch.Tensor): Target values of shape (B,).
            coarse_probs (torch.Tensor): Coarse probabilities of shape (B, 1) for binary classification and (B, num_classes) otherwise.
            refined_probs (torch.Tensor): Refined probabilities of shape (B, 1) for binary classification and (B, num_classes) otherwise.
            ch_ids (torch.Tensor): Channel IDs of shape (B,).
            qwa_coeffs (list): A qwa coefficients for each channel, where each entry is of the form
                                [ch_id, q_lower_coeff, q_normal_coeff, q_upper_coeff].
        Returns:
            loss (torch.Tensor): Loss value.
        """

        # Move to device
        device = coarse_probs.device
        targets = targets.to(device)
        ch_ids = ch_ids.to(device)

        # Channel Losses
        refined_ch_loss = self.coeffs[0] * self.ch_loss(refined_probs, targets, ch_ids, device) if self.ch_loss_refined else 0
        coarse_ch_loss = self.coeffs[1] * self.ch_loss(coarse_probs, targets, ch_ids, device) if self.ch_loss_coarse else 0

        # Window Losses
        refined_window_loss = self.coeffs[2] * self.window_loss(refined_probs, targets) if self.window_loss_refined else 0
        coarse_window_loss = self.coeffs[3] * self.window_loss(coarse_probs, targets) if self.window_loss_coarse else 0

        # Skew Loss
        skew_loss = self.coeffs[4] * self.compute_skew_loss(qwa_coeffs, device) if self.skew_loss else 0

        # Total loss
        total_loss = refined_ch_loss + coarse_ch_loss + refined_window_loss + coarse_window_loss + skew_loss

        return total_loss


if __name__ == "__main__":
    from sss.layers.dynamic_weights.qwa import QWA

    B = 64
    d_model = 32
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
    coarse_probs = torch.rand(B,).to(device).to(device)
    ch_ids = torch.randint(0, 5, (B,))
    targets = torch.randint(0, 2, (B,))

    model = QWA(num_networks=2,
                network_type="attn",
                hidden_exp_factor=2,
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
                lower_quantile=lower_quantile).to(device)

    refined_probs, qwa_coeffs = model(z, coarse_probs, ch_ids)
    refined_probs = refined_probs.to(device)

    qwa_loss = QWALoss(num_classes=2,
                       ch_loss_refined=True,
                       ch_loss_coarse=True,
                       skew_loss=True,
                       delta=0.1,
                       coeffs=[5., 5., 0., 0., 1],
                       loss_type="BCE").to(device)

    loss = qwa_loss(refined_probs, coarse_probs, targets, ch_ids, qwa_coeffs, device)
    print(f"Loss: {loss}")

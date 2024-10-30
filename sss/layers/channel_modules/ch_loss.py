import torch
import torch.nn as nn
import torch.nn.functional as F

class ChannelLoss(nn.Module):
    """
    A Channel-Wise loss for a single-channel classification model. This loss computes either the Binary Cross-Entropy
    or Cross-Entropy loss for each channel, where we use the mean probability of all windows in the channel as the
    channel output. This loss is useful when using the aggregation of the window probabilities as a proxy for the
    probability of the channel, with respect to some class, e.g., a channel being classified as anomolous. This class can
    either be binary (loss_type="BCE") or multiclass (loss_type="CE").
    """
    def __init__(self, loss_type="BCE", num_classes=3, u_weight=False):
        super(ChannelLoss, self).__init__()
        self.loss_type = loss_type
        self.u_weight = u_weight

        if loss_type == "BCE":
            self.criterion = nn.BCELoss()
        elif loss_type == "CE":
            self.criterion = nn.NLLLoss()
            self.num_classes = num_classes # For multiclass classification
        else:
            raise ValueError("Invalid channel loss type")

    def forward(self, output, target, ch_ids, u=None):
        """
        Args:
            output: Model predictions. Tensor of shape (batch_size,) for loss_type="BCE" and (batch_size, num_classes) for loss_type="CE".
            target: Tensor of shape (batch_size,) containing the target values with each entry being a binary value or positive integer.
            ch_ids: Tensor of shape (batch_size,) containing the channel ids with each entry being a positive integer.
            u: Uncertainty or confidence scores. Tensor of shape (batch_size,), representing some uncertainty or confidence for each window.
        Returns:
            loss: Scalar value of the channel loss.
        """

        if self.u_weight: u_entropy = 0.0 # Initialize uncertainty entropy

        # Compute sigmoid output
        if self.loss_type == "BCE":
            ch_probs = F.sigmoid(output) # (batch_size,)
            assert output.size() == target.size() == ch_ids.size(), "Output, target, and channel ids must have the same dimension"
        elif self.loss_type == "CE":
            ch_probs = F.log_softmax(output, dim=-1) # (batch_size, num_classes)
            assert len(output.size()) == 2, "Output must be a 2D tensor for multiclass classification"
            assert target.size() == ch_ids.size(), "Target and channel ids must have the same dimension"

        # Initialize channel targets and outputs
        ch_targets = torch.Tensor().to(output.device)
        ch_outs = torch.Tensor().to(output.device)

        # Enumerate through all unique channels
        for ch_id in torch.unique(ch_ids):
            ch_id = ch_id.item()
            mask = ch_ids == ch_id # Create mask for the channel
            ch_out = ch_probs[mask] # Probabilities for each window in the channel (sigmoid or softmax)

            # Mean Probability
            if self.u_weight:
                u_ch = u[mask] # (num_windows,) for a channel
                u_norm = F.softmax(u_ch, dim=-1) # (num_windows, 1)
                if self.loss_type == "BCE":
                    mean_out = torch.sum(ch_out * u_norm).unsqueeze(-1) # Weighted mean probability for the channel, shape: (1,
                elif self.loss_type == "CE":
                    mean_out = torch.sum(ch_out * u_norm.unsqueeze(-1), 0).unsqueeze(0)
            else:
                if self.loss_type == "BCE":
                    mean_out = ch_out.mean().unsqueeze(-1) # Mean probability for the channel, shape: (1,)
                elif self.loss_type == "CE":
                    mean_out = ch_out.mean(0).unsqueeze(0) # Mean probabilities over the channel, shape: (num_classes, 1)

            ch_outs = torch.cat((ch_outs, mean_out), 0) # Concatenate channel result

            # Targets
            ch_target = target[mask][0].unsqueeze(-1) # Current channel target
            ch_targets = torch.cat((ch_targets, ch_target), 0)

        if self.loss_type == "CE":
            ch_targets = ch_targets.long()
        elif self.loss_type == "BCE":
            ch_targets = ch_targets.float()

        loss = self.criterion(ch_outs, ch_targets)

        if self.u_weight:
            u_entropy /= len(torch.unique(ch_ids))
            return (loss, u_entropy)
        else:
            return loss


if __name__ == "__main__":

    # Prepare inputs
    loss_type = "CE"
    num_classes = 3 # Set to 1 if binary classification and > 2 for multiclass classification
    total_num_channels = 5
    batch_size = 8
    output = torch.tensor([[4], [1000], [5], [-8], [9], [-20], [4], [1000], [5], [-8], [9], [-20], [40], [1], [12], [-1], [-4], [16], [40], [1], [12], [-1], [-4], [16]]).view(8,3).float()
    print(f"output shape: {output.shape}")
    target = torch.randint(num_classes, (batch_size,))
    print(f"target: {target}")
    ch_ids = torch.randint(total_num_channels, (batch_size,))
    u = torch.randn(batch_size)

    target = target.to(output.device)
    ch_ids = ch_ids.to(output.device)

    criterion = ChannelLoss(loss_type=loss_type, num_classes=num_classes, u_weight=True)
    loss, u_entropy = criterion(output, target, ch_ids, u)
    print(f"loss: {loss}")
    print(f"u_entropy: {u_entropy}")

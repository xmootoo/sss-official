import torch
import torch.nn as nn

class ChannelAggregator(nn.Module):
    def __init__(self, latent_dim, reduction="mean"):
        super(ChannelAggregator, self).__init__()
        """
        Channel-wise head for binary/multiclass classification, aggregates the representations of the windows in a channel
        to a single representation. A linear layer or MLP can be optionally applied afterwards.

        Args:
            latent_dim (int): Dimension of the input representation.
            reduction (str): Type of reduction to use. Options: "mean" or "sum".
        """

        # Parameters
        self.reduction = reduction

    def forward(self, z, y, ch_ids):
        """
        Args:
            z (torch.tensor): Latent vector representation of the input data. Shape: (batch_size, latent_dim).
            ch_ids (torch.tensor): Tensor of shape (batch_size,) containing the channel ids with each entry being a positive integer.
            y (torch.tensor): Tensor of shape (batch_size,) containing the labels for each input.
        Return:
            z_ch (torch.tensor): Tensor of shape (num_channels, latent_dim) containing the aggregated latent vectors for each channel.
            y_ch (torch.tensor): Tensor of shape (num_channels,) containing the labels for each channel.
        """

        # Get unique channel ids
        unique_ch_ids = torch.unique(ch_ids)

        # Initialize lists to hold the aggregated representations and labels
        aggregated_representations = []
        y_ch = []

        for ch_id in unique_ch_ids:
            # Mask to select the latent vectors corresponding to the current channel
            mask = (ch_ids == ch_id).nonzero(as_tuple=True)[0]

            # Select the latent vectors for the current channel
            channel_representations = z[mask]

            # Aggregate the representations according to the reduction method
            if self.reduction == "mean":
                aggregated_representation = channel_representations.mean(dim=0)
            elif self.reduction == "sum":
                aggregated_representation = channel_representations.sum(dim=0)
            else:
                raise ValueError("Invalid reduction method. Choose between 'mean' or 'sum'.")

            aggregated_representations.append(aggregated_representation)

            # Select the label for the current channel (all entries for the same channel will have the same label)
            channel_label = y[mask][0]
            y_ch.append(channel_label)

        # Stack the aggregated representations and labels into single tensors
        z_ch = torch.stack(aggregated_representations)
        y_ch = torch.tensor(y_ch)

        return z_ch, y_ch

class ChannelLatentMixer(nn.Module):
    def __init__(self, reduction="mean", combination="concat"):
        super(ChannelLatentMixer, self).__init__()
        """
        Channel Latent Mixer takes the latent representation z and concatenates its corresponding channel mean representation
        along the embedding dimension.

        Args:
            latent_dim (int): Dimension of the input representation.
            reduction (str): Type of reduction to use. Options: "mean" or "sum".
        """

        # Parameters
        self.reduction = reduction
        self.combination = combination

    def forward(self, z, ch_ids):
        """
        Args:
            z (torch.tensor): Latent vector representation of the input data. Shape: (batch_size, 1, num_patches, d_model).
            ch_ids (torch.tensor): Tensor of shape (batch_size,) containing the channel ids with each entry being a positive integer.
        Return:
            new_z (torch.tensor): Tensor of shape (batch_size, num_patches, 2*d_model) or (batch_size, 2*num_patches, d_model)
                                  containing the aggregated latent vectors for univariate time series.
        """

        # Reshape z
        z = z.squeeze() # (batch_size, num_patches, d_model)

        # Get unique channel ids
        unique_ch_ids = torch.unique(ch_ids)

        # Initialize tensor to hold the aggregated representations and labels
        B, N, D = z.shape
        new_z = torch.zeros((B, N, 2*D)) if self.combination=="concat_embed_dim" else torch.zeros((B, 2*N, D))
        new_z = new_z.to(z.device)
        
        for ch_id in unique_ch_ids:
            # Mask to select the latent vectors corresponding to the current channel
            mask = (ch_ids == ch_id).nonzero(as_tuple=True)[0]

            # Select the latent vectors for the current channel
            ch_rep = z[mask] # Shape: (*, num_patches, d_model)

            # Aggregate the representations according to the reduction method
            if self.reduction == "mean":
                aggr_rep = ch_rep.mean(dim=0) # (num_patches, d_model)
            elif self.reduction == "sum":
                aggr_rep = ch_rep.sum(dim=0) # (num_patches, d_model)
            else:
                raise ValueError("Invalid reduction method. Choose between 'mean' or 'sum'.")

            # Repeat the aggregated representation for each example
            aggr_rep = aggr_rep.unsqueeze(0).repeat(ch_rep.shape[0], 1, 1) # (num_examples, num_patches, d_model)

            # Combine the channel representation with the aggregated representation
            if self.combination=="concat_embed_dim":
                new_rep = torch.cat([ch_rep, aggr_rep], dim=-1) # Shape: (num_examples, num_patches, 2*d_model)
            elif self.combination=="concat_patch_dim":
                new_rep = torch.cat([ch_rep, aggr_rep], dim=-2)# Shape: (num_examples, 2*num_patches, d_model)
            else:
                raise ValueError("Invalid combination method. Choose between 'concat_embed_dim' or 'concat_patch_dim'.")

            new_z[mask] = new_rep

        return new_z


# Example usage
if __name__ == "__main__":
    # batch_size = 5
    # latent_dim = 10

    # z = torch.randn(batch_size, latent_dim)
    # print(f"z: {z}")
    # ch_ids = torch.randint(0, 3, (batch_size,))
    # print(f"ch_ids: {ch_ids}")
    # y = torch.randint(0, 2, (batch_size,))
    # print(f"y: {y}")

    # model = ChannelAggregator(latent_dim=latent_dim, reduction="mean")
    # z_ch, y_ch = model(z, ch_ids, y)
    # print(z_ch)
    # print(y_ch)


    # ChannelLatentMixer
    batch_size = 8
    num_patches = 4
    d_model = 2

    z = torch.randn(batch_size, 1, num_patches, d_model)
    print(f"z: {z.shape}")
    ch_ids = torch.tensor([0, 0, 2, 4, 4, 1, 3, 0])
    print(f"ch_ids: {ch_ids}")
    y = torch.tensor([0, 0, 1, 0, 0, 1, 1, 0])
    print(f"y: {y}")

    clm = ChannelLatentMixer(combination="concat_embed_dim")
    new_z  = clm(z, ch_ids)
    print(f"z_ch: {new_z.shape}")

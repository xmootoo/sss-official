import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
import scipy.stats
from typing import List

def has_intersection(list1, list2):
    return bool(set(list1) & set(list2))

class MonteCarloDropout(nn.Module):
    def __init__(self, num_samples: int, num_classes: int, stats: List[str], mcd_prob: float) -> None:
        super(MonteCarloDropout, self).__init__()
        self.num_samples = num_samples
        self.num_classes = num_classes
        self.stats = stats
        self.mcd_prob = mcd_prob

        if num_classes==1:
            self.head = nn.Sigmoid()
        elif num_classes > 1:
            self.head = nn.Softmax(dim=-1)
        else:
            raise ValueError("Invalid number of classes")

    def enable_dropout(self, model: nn.Module) -> None:
        """
        Function to enable the dropout layers during test-time and set custom dropout probability,
        while keeping BatchNorm and LayerNorm layers in eval mode.

        Args:
        model (nn.Module): The PyTorch model
        mcd_prob (float): The desired dropout probability for Monte Carlo Dropout
        """
        model.eval()  # Set model to evaluation mode
        for module in model.modules():
            if isinstance(module, nn.Dropout):
                module.train()  # Set dropout layers to train mode
                module.p = self.mcd_prob  # Set the dropout probability

    def forward(self, x: torch.Tensor, model: nn.Module) -> torch.Tensor:
        """

        Forward pass through the model with Monte Carlo Dropout

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, *)

        """
        eval = not model.training
        self.enable_dropout(model) # Sets model.eval() and enables dropout layers

        with torch.no_grad():
            samples = torch.stack([self.head(model(x)) for _ in range(self.num_samples)], dim=0).permute(1, 0, 2).squeeze(-1) # (batch_size, num_samples, num_classes)

            if self.num_classes==1:
                stats = self.get_stats(samples)
            elif self.num_classes > 1:
                stats = self.get_multiclass_stats(samples)

        # Re-enable eval if model was in validation or testing loop, otherwise re-enable for training loop
        if eval:
            model.eval()
        else:
            model.train()

        return stats

    def get_stats(self, samples):
        """
        Calculate various statistical estimators for model uncertainty in binary classification.

        Args:
        samples (torch.Tensor): Tensor of shape (batch_size, num_samples) for binary classification containing probabilities from
                                Monte Carlo dropout trials.

        Returns:
            estimators (dict): A dictionary containing all calculated estimators.
            estimators_tensor (torch.Tensor): A tensor of shape (batch_size, num_estimators) containing all calculated estimators.
        """

        estimators = {}

        # Mean
        if has_intersection(["mean", "covar", "entropy", "entropy_confidence", "mi_confidence"], self.stats):
            mean = samples.mean(dim=1)
            estimators['mean'] = mean

        # Variance
        if "var" in self.stats:
            variance = samples.var(dim=1)
            estimators['var'] = variance

        # Standard deviation
        if "std" in self.stats:
            std = samples.std(dim=1)
            estimators['std'] = std

        # Coefficient of variation
        if "cv" in self.stats:
            cv = std / mean
            estimators['cv'] = cv

        # Quantiles (25th and 75th percentiles)
        if "q25" in self.stats and "q75" in self.stats:
            q25, q75 = torch.quantile(samples, torch.tensor([0.25, 0.75]), dim=1)
            estimators['q25'] = q25
            estimators['q75'] = q75

        # Interquartile range (IQR)
        if "iqr" in self.stats:
            iqr = q75 - q25
            estimators['iqr'] = iqr

        # 1. Entropy. Computes the overall uncertainty in the average prediction.
        if has_intersection(["entropy", "entropy_confidence", "mutual_info", "mi_confidence"], self.stats):
            entropy = -(mean * torch.log(mean + 1e-8) + (1 - mean) * torch.log(1 - mean + 1e-8))
            estimators['entropy'] = entropy

            # Confidence score (entropy). Higher values indicate more certainty in the average prediction. A value close to 1 means the
            # average prediction is very certain (close to 0 or 1), while a value close to 0 indicates maximum uncertainty (random guessing).
            if "entropy_confidence" in self.stats:
                entropy_confidence = 1 - (entropy / torch.log(torch.tensor(2.0)))
                estimators['entropy_confidence'] = entropy_confidence

        # 2. Mutual information. How much the dropout mask affects the prediction I(y_hat; d) where d is the random variable representing the
        # dropout mask and y_hat is the random variable representing the prediction fo the model with dropout mask d applied.
        if has_intersection(["mutual_info", "mi_confidence"], self.stats):
            entropy_samples = -(samples * torch.log(samples + 1e-8) + (1 - samples) * torch.log(1 - samples + 1e-8))
            entropy_mean = entropy_samples.mean(dim=1)
            mutual_info = entropy - entropy_mean # Confidence score in [0, 1]
            estimators['mutual_info'] = mutual_info

            # Confidence score (mutual information). Higher values indicate that the dropout mask has less impact on the prediction. A value close to 1
            # suggests that different dropout masks produce consistent predictions, indicating lower model uncertainty.
            if "mi_confidence" in self.stats:
                mi_confidence = 1 - (mutual_info / torch.log(torch.tensor(2.0)))
                estimators['mi_confidence'] = mi_confidence

        # 3. Variation ratio. Proportion of predictions that disagree with the most likely prediction.
        if has_intersection(["variation_ratio", "variation_confidence"], self.stats):
            mode_prediction = (samples.mean(dim=1) > 0.5).float()
            variation_ratio = (samples != mode_prediction.unsqueeze(1)).float().mean(dim=1)
            estimators['variation_ratio'] = variation_ratio

            # Confidence score (variation ratio). Higher values indicate more agreement among the predictions from different dropout masks. A value close to 1
            # means most predictions agree with the mode prediction, suggesting higher confidence.
            if "variation_confidence" in self.stats:
                confidence = 1 - variation_ratio # Confidence score in [0, 1]
                estimators['variation_confidence'] = confidence

        # Kullback-Leibler divergence
        if "kl_div" in self.stats:
            kl_div = F.kl_div(samples.log(), mean.unsqueeze(1).expand_as(samples), reduction='none').sum(dim=1)
            estimators['kl_div'] = kl_div

        # Jensen-Shannon divergence
        if "js_div" in self.stats:
            m = 0.5 * (samples + mean.unsqueeze(1))
            js_div = 0.5 * (F.kl_div(samples.log(), m, reduction='none').sum(dim=1) +
                            F.kl_div(mean.unsqueeze(1).log(), m, reduction='none').sum(dim=1))
            estimators['js_div'] = js_div

        # Confidence intervals (95%)
        if "ci_lower" in self.stats and "ci_upper" in self.stats:
            ci_lower, ci_upper = torch.quantile(samples, torch.tensor([0.025, 0.975]), dim=1)
            estimators['ci_lower'] = ci_lower
            estimators['ci_upper'] = ci_upper

        # Concatenate all estimators into a single tensor
        estimators_tensor = torch.stack([estimators[stat] for stat in self.stats], dim=1)

        return estimators, estimators_tensor

    def get_multiclass_stats(self, samples):
        """
        Calculate various statistical estimators for model uncertainty in multiclass classification.
        Args:
        samples (torch.Tensor): Tensor of shape (batch_size, num_samples, num_classes) containing
                                softmax probabilities from Monte Carlo dropout trials.
        Returns:
            estimators (dict): A dictionary containing all calculated estimators.
            estimators_tensor (torch.Tensor): A tensor containing all calculated estimators.
        """
        estimators = {}
        batch_size, num_samples, num_classes = samples.shape

         # Mean
        if has_intersection(["mean", "covar", "entropy", "entropy_confidence", "mi_confidence"], self.stats):
            mean = samples.mean(dim=1)
            estimators['mean'] = mean

        # Variance
        if "var" in self.stats:
            variance = samples.var(dim=1)
            estimators['var'] = variance

        # Standard deviation
        if "std" in self.stats:
            std = samples.std(dim=1)
            estimators['std'] = std

        # Covariance
        if "covar" in self.stats:
            centered = samples - mean.unsqueeze(1)
            cov = torch.bmm(centered.transpose(1, 2), centered) / (num_samples - 1)
            estimators['covar'] = cov

        # Entropy. Computes the overall uncertainty in the average prediction.
        if "entropy" in self.stats:
            entropy = -(mean * torch.log(mean + 1e-8) + (1 - mean) * torch.log(1 - mean + 1e-8))
            estimators['entropy'] = entropy

            #
            if "entropy_confidence" in self.stats:
                entropy_confidence = 1 - (entropy / torch.log(torch.tensor(2.0)))
                estimators['entropy_confidence'] = entropy_confidence

        # AUROC (Area Under the Receiver Operating Characteristic curve)
        if "auroc" in self.stats:
            sorted_probs, _ = torch.sort(mean, dim=1, descending=True)
            auroc = torch.trapz(sorted_probs.cumsum(dim=1), sorted_probs)
            estimators['auroc'] = auroc

        # Mutual information. How much the dropout mask affects the prediction I(y_hat; d) where d is the random variable representing the
        # dropout mask and y_hat is the random variable representing the prediction fo the model with dropout mask d applied.
        if "mutual_info" in self.stats and "entropy" in self.stats:
            entropy_samples = -(samples * torch.log(samples + 1e-8) + (1 - samples) * torch.log(1 - samples + 1e-8))
            entropy_mean = entropy_samples.mean(dim=1)
            mutual_info = entropy - entropy_mean
            estimators['mutual_info'] = mutual_info

            if "mi_confidence" in self.stats:
                mi_confidence = 1 - (mutual_info / torch.log(torch.tensor(2.0)))
                estimators['mi_confidence'] = mi_confidence

        # Variation ratio. Proportion of predictions that disagree with the most likely prediction.
        if "variation_ratio" in self.stats:
            mode_prediction = (samples.mean(dim=1) > 0.5).float()
            variation_ratio = (samples != mode_prediction.unsqueeze(1)).float().mean(dim=1)
            estimators['variation_ratio'] = variation_ratio

            if "variation_confidence" in self.stats:
                confidence = 1 - variation_ratio
                estimators['variation_confidence'] = confidence

        # Differential entropy (assuming multivariate normal distribution)
        if "diff_entropy" in self.stats:
            diff_entropy = 0.5 * torch.logdet(cov + 1e-8 * torch.eye(num_classes).unsqueeze(0))
            estimators['diff_entropy'] = diff_entropy

        # Total variation distance (average pairwise)
        if "tv_distance" in self.stats:
            tv_distance = 0.5 * torch.abs(samples.unsqueeze(2) - samples.unsqueeze(3)).sum(dim=-1).mean(dim=(1, 2))
            estimators['tv_distance'] = tv_distance

        # Confidence (probability of most likely class)
        if "confidence" in self.stats:
            confidence, _ = mean.max(dim=-1)
            estimators['confidence'] = confidence

        # Predictive uncertainty (1 - confidence)
        if "uncertainty" in self.stats:
            pred_uncertainty = 1 - confidence
            estimators['uncertainty'] = pred_uncertainty

        # Gini impurity
        if "gini" in self.stats:
            gini = 1 - (mean ** 2).sum(dim=-1)
            estimators['gini'] = gini

        # TODO: Create a smart concatenation of all estimators in a tensor based off of self.stats keys
        # estimators_tensor = torch.cat([
        #     mean.reshape(batch_size, -1),
        #     variance.reshape(batch_size, -1),
        #     cov.reshape(batch_size, -1),
        #     std.reshape(batch_size, -1),
        #     entropy.unsqueeze(1),
        #     auroc.unsqueeze(1),
        #     mutual_info.unsqueeze(1),
        #     diff_entropy.unsqueeze(1),
        #     tv_distance.unsqueeze(1),
        #     confidence.unsqueeze(1),
        #     pred_uncertainty.unsqueeze(1),
        #     gini.unsqueeze(1)
        # ], dim=1)

        return estimators, estimators_tensor


if __name__ == "__main__":
    x = torch.randn(64, 32)* (-500)
    print(f"Input shape: {x}")
    model = nn.Sequential(nn.Linear(32, 16),
                            nn.Dropout(p=0.3),
                            nn.Linear(16, 1))

    model.train()
    mcd = MonteCarloDropout(num_samples=100, num_classes=1, stats=["entropy_confidence", "mi_confidence", "variation_confidence"], mcd_prob=0.2)

    results, results_tensor = mcd(x, model)

    print(f"Results shape: {results_tensor.shape}")

    # Print results
    print(f"entropy_confidence: {results['entropy_confidence']}")
    print(f"mi_confidence: {results['mi_confidence']}")
    print(f"variation_confidence: {results['variation_confidence']}")

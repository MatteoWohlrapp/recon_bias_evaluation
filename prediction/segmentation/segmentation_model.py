"""
Segmentation wrappers for both training and evaluation.
"""

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from skimage.metrics import peak_signal_noise_ratio as psnr

from .model_wrapper import ModelWrapper
import torch
import torch.nn as nn


class DiceLoss(nn.Module):
    def __init__(self, epsilon=1e-6):
        """
        Dice Loss for binary segmentation.
        Args:
            epsilon (float): Small constant to prevent division by zero.
        """
        super(DiceLoss, self).__init__()
        self.epsilon = epsilon

    def forward(self, predictions, targets):
        """
        Compute Dice Loss.
        Args:
            predictions (torch.Tensor): Predicted tensor of shape (N, C, H, W) for logits or probabilities.
            targets (torch.Tensor): Ground truth tensor of shape (N, C, H, W).
        Returns:
            torch.Tensor: Dice loss value.
        """
        # Apply sigmoid to predictions if they're logits
        predictions = torch.sigmoid(predictions)

        # Flatten tensors to compute per-channel dice loss
        predictions = predictions.view(predictions.size(0), -1)
        targets = targets.view(targets.size(0), -1)

        # Compute Dice Score
        intersection = (predictions * targets).sum(dim=1)
        union = predictions.sum(dim=1) + targets.sum(dim=1)
        dice_score = (2.0 * intersection + self.epsilon) / (union + self.epsilon)

        # Dice Loss
        dice_loss = 1 - dice_score.mean()
        return dice_loss


class SegmentationModel(ModelWrapper):
    """
    Reconstruction base class.
    """

    def __init__(self):
        super().__init__()
        self.dice_loss = DiceLoss()
        self.l1_loss = nn.L1Loss()

    @property
    def name(self):
        return self.network.__class__.__name__

    def target_transformation(self, y):
        return y

    def criterion(self, x, y):
        return 0.5 * self.dice_loss(x, y) + 0.5 * self.l1_loss(x, y)

    def evaluation_performance_metric(self, x, y):
        return torch.tensor(0.0)

    def epoch_performance_metric(self, x, y):
        return torch.tensor(0.0), 1

    @property
    def performance_metric_name(self):
        return "n/a"

    @property
    def performance_metric_input_value(self):
        return "prediction"

    def save_snapshot(self, x, y, y_pred, path, device, epoch):
        # save image next to each other
        plt.clf()
        x = x.cpu().numpy()
        y = y.cpu().numpy()
        y_pred = y_pred.cpu().numpy()
        x = x.squeeze()
        y = y.squeeze()
        y_pred = y_pred.squeeze()

        fig, ax = plt.subplots(1, 5, figsize=(10, 5))
        ax[0].imshow(x.squeeze(), cmap="gray")
        ax[0].set_title("Image")
        ax[0].axis("off")
        ax[1].imshow(y.squeeze(), cmap="gray")
        ax[1].set_title("Original segmentation")
        ax[1].axis("off")
        ax[2].imshow(y_pred.squeeze(), cmap="gray")
        ax[2].set_title("Output logits")
        ax[2].axis("off")
        ax[3].imshow(y_pred.squeeze() > 0.5, cmap="gray")
        ax[3].set_title("Segmentation with threshold")
        ax[3].axis("off")
        ax[4].imshow(
            np.abs(y - y_pred),
            cmap="viridis",
        )
        ax[4].set_title("Difference")
        ax[4].axis("off")
        plt.savefig(path)
        plt.close()

    @property
    def evaluation_groups(self):
        return [
            (
                ["sex"],
                {
                    "x": "sex",
                    "x_label": "Sex",
                    "facet_col": None,
                    "facet_col_label": None,
                },
                "sex",
            ),
            (
                ["age_bin"],
                {
                    "x": "age_bin",
                    "x_label": "Age Group",
                    "facet_col": None,
                    "facet_col_label": None,
                },
                "age",
            ),
            (
                ["sex", "age_bin"],
                {
                    "x": "sex",
                    "x_label": "Sex",
                    "facet_col": "age_bin",
                    "facet_col_label": "Age Group",
                },
                "sex_age",
            ),
        ]

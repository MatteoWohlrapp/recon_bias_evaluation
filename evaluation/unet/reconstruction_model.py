"""
Classifier wrappers for both training and evaluation.
"""

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from skimage.metrics import peak_signal_noise_ratio as psnr

from .model_wrapper import ModelWrapper


class ReconstructionModel(ModelWrapper):
    """
    Reconstruction base class.
    """

    def __init__(self):
        super().__init__()
        self.loss = torch.nn.MSELoss()

    @property
    def name(self):
        return self.network.__class__.__name__

    def target_transformation(self, y):
        return y

    def criterion(self, x, y):
        return self.loss(x, y)

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

        fig, ax = plt.subplots(1, 4, figsize=(10, 5))
        ax[0].imshow(x.squeeze(), cmap="gray")
        ax[0].set_title("Low-Dose CT")
        ax[0].axis("off")
        ax[1].imshow(y.squeeze(), cmap="gray")
        ax[1].set_title("Original")
        ax[1].axis("off")
        ax[2].imshow(y_pred.squeeze(), cmap="gray")
        ax[2].set_title("Reconstruction")
        ax[2].axis("off")
        ax[3].imshow(
            np.abs(y - y_pred),
            cmap="viridis",
        )
        ax[3].set_title("Difference")
        ax[3].axis("off")
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

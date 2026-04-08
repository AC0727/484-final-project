# cat_classifier.py

import torch
import torch.nn as nn
from torchvision.models import (
    resnet18,
    ResNet18_Weights,
)

# we use resnet18 over 34 since we prioritize speed and efficiency here, as our classifier
# task is quite simple: output 1 if the image contains a cat, 0 otherwise


class CatClassifier(nn.Module):
    """
    Binary classifier for:
        1 -> image contains at least one cat
        0 -> image does not contain any cats

    Output:
        One logit per image, shape [B, 1]
    """

    def __init__(
        self,
        backbone: str = "resnet18",
        pretrained: bool = True,
        freeze_backbone: bool = False,
    ):
        super().__init__()

        weights = ResNet18_Weights.DEFAULT if pretrained else None
        self.model = resnet18(weights=weights)

        if freeze_backbone:
            for param in self.model.parameters():
                param.requires_grad = False

        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: image tensor of shape [B, 3, H, W]

        Returns:
            logits: tensor of shape [B, 1]
        """
        return self.model(x)

    def predict_probs(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns probabilities in [0, 1], shape [B, 1]
        """
        logits = self.forward(x)
        return torch.sigmoid(logits)

    def predict(self, x: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        """
        Returns binary predictions:
            1 -> contains cat
            0 -> does not contain cat
        """
        probs = self.predict_probs(x)
        return (probs >= threshold).float()

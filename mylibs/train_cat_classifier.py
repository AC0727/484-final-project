import torch
import torch.nn as nn
import torch.nn.functional as F
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
        One logit per image, shape [B, 1] where B is for batch size
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

        # storing the gradients for GradCAM
        self.gradients = None
        self.activations = None

        # register the hooks needed to track the gradients for GradCAM
        target_layer = self.model.layer4[-1]

        target_layer.register_forward_hook(self._forward_hook)
        target_layer.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, module, input, output):
        # Save feature maps
        self.activations = output

    def _backward_hook(self, module, grad_input, grad_output):
        # Save gradients (grad_output is a tuple)
        self.gradients = grad_output[0]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: image tensor of shape [B, 3, H, W]

        Returns:
            logits: tensor of shape [B, 1], one logit per image
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

    def generate_gradcam(self, x: torch.Tensor, upsample: bool = True):
        """
        Args:
            x: shape [B, 3, H, W] where 3 is the number of channels (RGB)
        Returns:
            cam: shape [B, 1, H, W]
        """
        self.eval()

        # Forward pass
        logits = self.forward(x) # [B, 1]

        self.zero_grad()
        logits.backward(torch.ones_like(logits)) # gradients of the 'cat' score

        # Get stored data from the hooks
        gradients = self.gradients # [B, C, H, W], C is number of feature maps
        activations = self.activations # [B, C, H, W]

        # Global average pooling on gradients
        weights = gradients.mean(dim=(2, 3), keepdim=True)  # [B, C, 1, 1]

        # Weighted sum, collapse all feature maps into one heatmap
        cam = (weights * activations).sum(dim=1, keepdim=True)  # [B, 1, H, W]

        # ReLU, keep only positive contributions to the heatmap
        cam = torch.relu(cam)

        # Normalize per image
        B, _, H, W = cam.shape
        cam = cam.view(B, -1) # flatten to [B, H*W]
        cam = cam - cam.min(dim=1, keepdim=True)[0]
        cam = cam / (cam.max(dim=1, keepdim=True)[0] + 1e-8)
        cam = cam.view(B, 1, H, W) # reshape

        if upsample:
            cam = F.interpolate(cam, size=x.shape[2:], mode="bilinear", align_corners=False)

        return cam

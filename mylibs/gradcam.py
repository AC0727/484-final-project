import torch
import torch.nn as nn
import torch.nn.functional as F

"""
Example Usage:

model = CatClassifier().to(device)

gradcam = GradCAM(
    model=model,
    target_layer=model.model.layer4[-1]
)

cam = gradcam.generate(image)
"""


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer

        # storing the gradients and feature maps for GradCAM
        self.gradients = None
        self.activations = None

        self._register_hooks()
    
    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)
    
    def generate(self, x: torch.Tensor, upsample=True):
        """
        Args:
            x: shape [B, 3, H, W] where 3 is the number of channels (RGB)
        Returns:
            cam: shape [B, 1, H, W]
        """
        self.model.eval()

        # Forward pass
        logits = self.model(x) # [B, 1]

        self.model.zero_grad()
        logits.backward(torch.ones_like(logits)) # gradients of the 'cat' score

        # Get stored data from the hooks
        gradients = self.gradients # [B, C, H, W], C is number of feature maps
        activations = self.activations # [B, C, H, W]

        # Global average pooling on gradients
        weights = gradients.mean(dim=(2, 3), keepdim=True) # [B, C, 1, 1]

         # Weighted sum, collapse all feature maps into one heatmap
        cam = (weights * activations).sum(dim=1, keepdim=True) # [B, 1, H, W]

        # ReLU, keep only positive contributions to the heatmap
        cam = torch.relu(cam)

        # normalize
        B, _, H, W = cam.shape
        cam = cam.view(B, -1) # flatten to [B, H*W]
        cam = cam - cam.min(dim=1, keepdim=True)[0]
        cam = cam / (cam.max(dim=1, keepdim=True)[0] + 1e-8)
        cam = cam.view(B, 1, H, W) # reshape

        if upsample:
            import torch.nn.functional as F
            cam = F.interpolate(cam, size=x.shape[2:], mode="bilinear", align_corners=False)

        return cam




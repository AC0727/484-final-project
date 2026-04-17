import torch
import numpy as np
from scipy.ndimage import label
import matplotlib.pyplot as plt

"""
Example Usage:

cam = gradcam.generate(image)[0]  # [1, H, W] → [H, W]

center, bbox, mask = cam_to_bbox_and_center(cam, threshold=0.4)

visualize_cam_bbox(image[0], cam, bbox, center)
"""

# NOTE: For this to work properly, cam needs to be upsampled to the original image size

# NOTE: The value of the threshold matters a lot, somehting we can experiment with
def cam_to_binary_mask(cam: torch.Tensor, threshold: float = 0.5):
    """
    cam: [H, W] or [1, H, W]
    returns: binary mask [H, W]
    """
    if cam.dim() == 3:
        cam = cam.squeeze(0)  # If the dimension is [1, H, W], modify it to [H, W]

    return (cam > threshold).float()


# NOTE: This is another way to threshold for the binary mask that is supposedly better, maybe we can compare the different
# ways to threshold?
def percentile_threshold_mask(cam: torch.Tensor, percentile=80):
    """
    Keeps the top (100 - percentile)% of pixels in the binary mask
    cam: [H, W] or [1, H, W]
    percentile: keep top (100 - percentile)% pixels

    returns: binary mask
    """
    if cam.dim() == 3:
        cam = cam.squeeze(0)

    flat = cam.view(-1)
    thresh = torch.quantile(flat, percentile / 100.0)

    return (cam >= thresh).float()


def largest_connected_component(binary_mask: torch.Tensor):
    """
    binary_mask: mask [H, W]
    returns: mask with only largest component
    """
    mask_np = binary_mask.detach().cpu().numpy() 

    labeled, num_features = label(mask_np)

    if num_features == 0:  # nothing found
        return binary_mask

    largest_label = 1 + np.argmax([
        (labeled == i).sum() for i in range(1, num_features + 1)
    ])  # the feature with the most labelled pixels

    largest_mask = (labeled == largest_label).astype(np.float32)

    return torch.from_numpy(largest_mask).to(binary_mask.device)


def compute_centroid(binary_mask: torch.Tensor):
    """
    Compute the centre of the binary map
    binary_mask: [H, W]
    returns: (x, y)
    """
    ys, xs = torch.where(binary_mask > 0)

    if len(xs) == 0:
        return None

    x_center = xs.float().mean().item()
    y_center = ys.float().mean().item()

    return (x_center, y_center)


def mask_to_bbox(binary_mask: torch.Tensor):
    """
    binary_mask: [H, W]
    returns: (x_min, y_min, x_max, y_max)
    """
    ys, xs = torch.where(binary_mask > 0)

    if len(xs) == 0:
        return None

    x_min = xs.min().item()
    x_max = xs.max().item()
    y_min = ys.min().item()
    y_max = ys.max().item()

    return (x_min, y_min, x_max, y_max)


def cam_to_bbox_and_center(cam: torch.Tensor, mask_fn, **mask_kwargs):
    """
    cam: [1, H, W] or [H, W]
    mask_fn: function that converts CAM → binary mask
    mask_kwargs: extra args for the mask function

    returns:
        center: (x, y)
        bbox: (x_min, y_min, x_max, y_max)
        mask: refined mask featuring only the largest connected component
    """
    mask = mask_fn(cam, **mask_kwargs)
    mask = largest_connected_component(mask)

    center = compute_centroid(mask)
    bbox = mask_to_bbox(mask)

    return center, bbox, mask


def visualize_cam_bbox(image, cam, bbox=None, center=None):
    """
    Visualize the bounding box and centroid on the actual image
    image: [3, H, W]
    cam: [H, W] or [1, H, W]
    """
    img = image.permute(1, 2, 0).detach().cpu().numpy()

    if cam.dim() == 3:
        cam = cam.squeeze(0)

    cam = cam.detach().cpu().numpy()

    plt.imshow(img)
    plt.imshow(cam, cmap='jet', alpha=0.5)

    if bbox is not None:
        x_min, y_min, x_max, y_max = bbox
        rect = plt.Rectangle(
            (x_min, y_min),
            x_max - x_min,
            y_max - y_min,
            edgecolor='red',
            facecolor='none',
            linewidth=2
        )
        plt.gca().add_patch(rect)

    if center is not None:
        plt.scatter(center[0], center[1], c='white', s=50)

    plt.axis('off')
    plt.show()


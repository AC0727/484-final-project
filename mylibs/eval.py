from __future__ import annotations

import math
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.datasets import VOCDetection

from train_cat_classifier import CatClassifier
from gradcam import GradCAM
from gradcam_utils import cam_to_bbox_and_center, cam_to_binary_mask

"""
Example Usage:

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset = VOCCatEvalDataset(
    root="./data",
    year="2007",
    image_set="val",
    download=False,
)

dataloader = DataLoader(
    dataset,
    batch_size=16,
    shuffle=False,
    num_workers=4,
    collate_fn=eval_collate_fn,
)

model = build_model(
    checkpoint_path="checkpoints/best_model.pth",
    device=device,
    backbone="resnet18",
)

criterion = nn.BCEWithLogitsLoss()

gradcam = GradCAM(
    model=model,
    target_layer=model.model.layer4[-1],
)

metrics = evaluate(model, gradcam, dataloader, criterion, device)

print(metrics)
"""

IMAGE_SIZE = 224

# NOTE: we can tune these!
THRESHOLD = 0.4
BATCH_SIZE = 16
NUM_WORKERS = 4

class VOCCatEvalDataset(Dataset):
    """
    Returns:
        image_tensor: [3, H, W]
        label: float tensor scalar (1 if contains cat, else 0)
        gt_boxes: tensor of shape [N, 4] in resized image coordinates
                  each box is (x_min, y_min, x_max, y_max)
    """
    def __init__(self, root: str, year: str = "2007", image_set: str = "val", download: bool = False):
        self.dataset = VOCDetection(
            root=root,
            year=year,
            image_set=image_set,
            download=download,
        )

        self.transform = transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx: int):
        image, target = self.dataset[idx]

        orig_w, orig_h = image.size
        image_tensor = self.transform(image)

        objects = target["annotation"].get("object", [])
        if isinstance(objects, dict):
            objects = [objects]

        gt_boxes = []
        contains_cat = False

        scale_x = IMAGE_SIZE / orig_w
        scale_y = IMAGE_SIZE / orig_h

        for obj in objects:
            if obj["name"] != "cat":
                continue

            contains_cat = True
            bbox = obj["bndbox"]

            # VOC stores coordinates as strings
            x_min = (float(bbox["xmin"]) - 1.0) * scale_x
            y_min = (float(bbox["ymin"]) - 1.0) * scale_y
            x_max = (float(bbox["xmax"]) - 1.0) * scale_x
            y_max = (float(bbox["ymax"]) - 1.0) * scale_y

            gt_boxes.append([x_min, y_min, x_max, y_max])

        if len(gt_boxes) == 0:
            gt_boxes = torch.zeros((0, 4), dtype=torch.float32)
        else:
            gt_boxes = torch.tensor(gt_boxes, dtype=torch.float32)

        label = torch.tensor(float(contains_cat), dtype=torch.float32)
        return image_tensor, label, gt_boxes


def eval_collate_fn(batch):
    """
    gt_boxes has variable length, so we keep it as a list.
    """
    images = torch.stack([item[0] for item in batch], dim=0)
    labels = torch.stack([item[1] for item in batch], dim=0)
    gt_boxes = [item[2] for item in batch]
    return images, labels, gt_boxes


def build_model(checkpoint_path: str, device: torch.device, backbone: str = "resnet18"):
    model = CatClassifier(backbone=backbone, pretrained=False)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model


def point_in_box(center, box):
    cx, cy = center
    x1, y1, x2, y2 = box
    return x1 <= cx <= x2 and y1 <= cy <= y2


def box_center(box):
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def euclidean_distance(p1, p2):
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def compute_iou(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)

    union = area_a + area_b - inter_area
    if union <= 0:
        return 0.0
    return inter_area / union


def evaluate(model, gradcam, dataloader, criterion, device):
    total_loss = 0.0
    total_examples = 0
    correct = 0

    tp = fp = fn = tn = 0

    positive_images = 0
    center_inside_count = 0
    iou_sum = 0.0
    valid_distance_count = 0
    distance_sum = 0.0

    with torch.no_grad():
        for images, labels, gt_boxes_list in dataloader:
            images = images.to(device)
            labels = labels.to(device).unsqueeze(1)  # [B, 1]

            logits = model(images)
            loss = criterion(logits, labels)

            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()

            batch_size = images.size(0)
            total_loss += loss.item() * batch_size
            total_examples += batch_size

            correct += (preds == labels).sum().item()

            tp += ((preds == 1) & (labels == 1)).sum().item()
            fp += ((preds == 1) & (labels == 0)).sum().item()
            fn += ((preds == 0) & (labels == 1)).sum().item()
            tn += ((preds == 0) & (labels == 0)).sum().item()

    # Need gradients for Grad-CAM, so do localization in a second pass
    for images, labels, gt_boxes_list in dataloader:
        images = images.to(device)

        cams = gradcam.generate(images)  # [B, 1, H, W]

        probs = torch.sigmoid(model(images))
        preds = (probs > 0.5).float().squeeze(1)

        for i in range(images.size(0)):
            gt_boxes = gt_boxes_list[i]

            # Evaluate localization only on GT-positive images
            if gt_boxes.shape[0] == 0:
                continue

            positive_images += 1

            # Only try localization if classifier predicts cat
            if preds[i].item() == 0:
                continue

            cam = cams[i] # [1, H, W]

            pred_center, pred_bbox, pred_mask = cam_to_bbox_and_center(
                cam,
                mask_fn=cam_to_binary_mask,
                threshold=THRESHOLD,
            )

            if pred_center is not None:
                inside_any = any(point_in_box(pred_center, gt_box.tolist()) for gt_box in gt_boxes)
                if inside_any:
                    center_inside_count += 1

                gt_centers = [box_center(gt_box.tolist()) for gt_box in gt_boxes]
                nearest_dist = min(euclidean_distance(pred_center, c) for c in gt_centers)
                distance_sum += nearest_dist
                valid_distance_count += 1

            if pred_bbox is not None:
                best_iou = max(compute_iou(pred_bbox, gt_box.tolist()) for gt_box in gt_boxes)
                iou_sum += best_iou
            else:
                iou_sum += 0.0

    avg_loss = total_loss / total_examples if total_examples > 0 else 0.0
    accuracy = correct / total_examples if total_examples > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    center_inside_rate = center_inside_count / positive_images if positive_images > 0 else 0.0
    mean_iou = iou_sum / positive_images if positive_images > 0 else 0.0
    mean_center_distance = distance_sum / valid_distance_count if valid_distance_count > 0 else 0.0

    return {
        "loss": avg_loss,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "positive_images": positive_images,
        "center_inside_rate": center_inside_rate,
        "mean_iou": mean_iou,
        "mean_center_distance": mean_center_distance,
    }


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = "checkpoints/best_model.pth"

    dataset = VOCCatEvalDataset(
        root="./data",
        year="2007",
        image_set="val",
        download=False,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        collate_fn=eval_collate_fn,
    )

    model = build_model(checkpoint_path, device, backbone="resnet18")
    criterion = nn.BCEWithLogitsLoss()

    gradcam = GradCAM(
        model=model,
        target_layer=model.model.layer4[-1],
    )

    metrics = evaluate(model, gradcam, dataloader, criterion, device)

    print("\nClassification Results")
    print(f"Loss:      {metrics['loss']:.4f}")
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1:        {metrics['f1']:.4f}")
    print(f"TP: {metrics['tp']} | FP: {metrics['fp']} | FN: {metrics['fn']} | TN: {metrics['tn']}")

    print("\nLocalization Results")
    print(f"Positive images:       {metrics['positive_images']}")
    print(f"Center inside GT box:  {metrics['center_inside_rate']:.4f}")
    print(f"Mean IoU:              {metrics['mean_iou']:.4f}")
    print(f"Mean center distance:  {metrics['mean_center_distance']:.4f}")


if __name__ == "__main__":
    main()
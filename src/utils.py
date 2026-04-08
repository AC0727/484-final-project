import torch

def classifier_collate_fn(batch):
    images, labels, targets = zip(*batch)
    images = torch.stack(images, dim=0)   # [B, 3, 224, 224]
    labels = torch.stack(labels, dim=0)   # [B]
    return images, labels, list(targets)
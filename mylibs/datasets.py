import torch
from torch.utils.data import Dataset
from torchvision.datasets import VOCDetection

IMG_SIZE = 224

class VOCCatBinaryDataset(Dataset):
    """
    Returns:
        image: FloatTensor [3, 224, 224]
        label: FloatTensor [] where 1 = cat present, 0 = no cat
        target: dict with:
            boxes: FloatTensor [N, 4] scaled to resized image coordinates
            orig_size: tuple (orig_h, orig_w)
            new_size: tuple (224, 224)
            has_cat: int
            image_id: str
    """
    def __init__(self, root, year="2012", image_set="train", transform=None, download=False):
        self.dataset = VOCDetection(
            root=root,
            year=year,
            image_set=image_set,
            download=download
        )
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def _scale_box(self, box, orig_w, orig_h, new_w=IMG_SIZE, new_h=IMG_SIZE):
        xmin, ymin, xmax, ymax = box

        x_scale = new_w / orig_w
        y_scale = new_h / orig_h

        xmin = xmin * x_scale
        xmax = xmax * x_scale
        ymin = ymin * y_scale
        ymax = ymax * y_scale

        return [xmin, ymin, xmax, ymax]

    def __getitem__(self, idx):
        image, annotation = self.dataset[idx]

        orig_w, orig_h = image.size  # PIL image.size = (width, height)

        ann = annotation["annotation"]
        image_id = ann["filename"]

        objects = ann.get("object", [])
        if not isinstance(objects, list):
            objects = [objects]

        cat_boxes = []

        for obj in objects:
            if obj["name"] == "cat":
                bndbox = obj["bndbox"]
                xmin = float(bndbox["xmin"])
                ymin = float(bndbox["ymin"])
                xmax = float(bndbox["xmax"])
                ymax = float(bndbox["ymax"])

                scaled_box = self._scale_box(
                    [xmin, ymin, xmax, ymax],
                    orig_w=orig_w,
                    orig_h=orig_h,
                    new_w=IMG_SIZE,
                    new_h=IMG_SIZE
                )
                cat_boxes.append(scaled_box)

        has_cat = 1 if len(cat_boxes) > 0 else 0

        if len(cat_boxes) > 0:
            boxes = torch.tensor(cat_boxes, dtype=torch.float32)   # [N, 4]
        else:
            boxes = torch.zeros((0, 4), dtype=torch.float32)       # [0, 4]

        if self.transform is not None:
            image = self.transform(image)

        label = torch.tensor(has_cat, dtype=torch.float32)

        target = {
            "boxes": boxes,
            "orig_size": (orig_h, orig_w),
            "new_size": (IMG_SIZE, IMG_SIZE),
            "has_cat": has_cat,
            "image_id": image_id
        }

        return image, label, target
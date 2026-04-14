import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch

from mylibs.datasets import VOCCatBinaryDataset
from mylibs.transforms import get_image_transform

def unnormalize(img_tensor):
    """
    img_tensor: Tensor [3, 224, 224]
    returns: Tensor [3, 224, 224] in range [0, 1] for display
    """
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    img_tensor = img_tensor * std + mean
    return torch.clamp(img_tensor, 0, 1)

def main():
    dataset = VOCCatBinaryDataset(
        root="data",
        year="2012",
        image_set="train",
        transform=get_image_transform(),
        download=False
    )

    shown = 0
    i = 0

    while shown < 3 and i < len(dataset):
        image, label, target = dataset[i]

        if label.item() == 1.0:
            image = unnormalize(image)
            image_np = image.permute(1, 2, 0).numpy()   # [224, 224, 3]

            fig, ax = plt.subplots(1, figsize=(6, 6))
            ax.imshow(image_np)

            for box in target["boxes"]:
                xmin, ymin, xmax, ymax = box.tolist()
                rect = patches.Rectangle(
                    (xmin, ymin),
                    xmax - xmin,
                    ymax - ymin,
                    linewidth=2,
                    edgecolor="r",
                    facecolor="none"
                )
                ax.add_patch(rect)

            ax.set_title(f'label={int(label.item())}, image_id={target["image_id"]}')
            plt.show()
            shown += 1

        i += 1

if __name__ == "__main__":
    main()
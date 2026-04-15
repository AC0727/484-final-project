import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch

def classifier_collate_fn(batch):
    images, labels, targets = zip(*batch)
    images = torch.stack(images, dim=0)   # [B, 3, 224, 224]
    labels = torch.stack(labels, dim=0)   # [B]
    return images, labels, list(targets)


# Get a balanced set of indices of cat vs no cat images
def get_balanced_indices(dataset, max_per_class=None):
    cat_indices = []
    no_cat_indices = []

    for i in range(len(dataset.dataset)):  # access VOC dataset directly to make faster, no need to load images
        _, annotation = dataset.dataset[i]

        objects = annotation["annotation"].get("object", [])
        if not isinstance(objects, list):
            objects = [objects]

        has_cat = any(obj["name"] == "cat" for obj in objects)

        if has_cat:
            cat_indices.append(i)
        else:
            no_cat_indices.append(i)

    n = min(len(cat_indices), len(no_cat_indices))

    if max_per_class:
        n = min(n, max_per_class)

    return cat_indices[:n] + no_cat_indices[:n]


def unnormalize(img_tensor):
    """
    img_tensor: Tensor [3, 224, 224]
    returns: Tensor [3, 224, 224] in range [0, 1] for display
    """
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    img_tensor = img_tensor * std + mean
    return torch.clamp(img_tensor, 0, 1)


def visualize_cat_images(dataset, num_cats = 3):
    fig, axes = plt.subplots(1, num_cats, figsize=(5 * num_cats, 5))
    shown = 0
    i = 0

    if num_cats == 1:
        axes = [axes]

    while shown < num_cats and i < len(dataset):
        image, label, target = dataset[i]

        if label.item() == 1.0:
            image = unnormalize(image)
            image_np = image.permute(1, 2, 0).numpy()   # [224, 224, 3]

            ax = axes[shown]
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
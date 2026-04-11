from mylibs.datasets import VOCCatBinaryDataset
from mylibs.transforms import get_image_transform

def main():
    dataset = VOCCatBinaryDataset(
        root="data",
        year="2012",
        image_set="train",
        transform=get_image_transform(),
        download=False
    )

    print("Dataset size:", len(dataset))
    print()

    found_cat = 0

    for i in range(20):
        image, label, target = dataset[i]

        print(f"Index: {i}")
        print("Image shape:", image.shape)         # [3, 224, 224]
        print("Label:", label.item())              # 0.0 or 1.0
        print("Boxes shape:", target["boxes"].shape)
        print("Boxes:", target["boxes"])
        print("Original size:", target["orig_size"])
        print("New size:", target["new_size"])
        print("Image ID:", target["image_id"])
        print("-" * 50)

        if label.item() == 1.0:
            found_cat += 1

        if found_cat >= 3:
            break

if __name__ == "__main__":
    main()
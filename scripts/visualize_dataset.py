from mylibs.datasets import VOCCatBinaryDataset
from mylibs.transforms import get_image_transform
from mylibs.utils import unnormalize, visualize_cat_images

def main():
    dataset = VOCCatBinaryDataset(
        root="data",
        year="2012",
        image_set="train",
        transform=get_image_transform(),
        download=False
    )

    visualize_cat_images(dataset, 3)

if __name__ == "__main__":
    main()
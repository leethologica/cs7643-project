import os
import json
import torch
import numpy as np

from PIL import Image

class CocoFake(torch.utils.data.Dataset):
    def __init__(
        self,
        coco_path,
        cocofake_path,
        real_transform=None,
        fake_transform=None,
        split="train",
        limit=-1,
        test_split=None,
    ):
        if split == "test" or split == "val":
            self.cocofake_path = os.path.join(cocofake_path, "val2014")
            self.coco_path = os.path.join(coco_path, "validation", "data")
        else:
            self.cocofake_path = os.path.join(cocofake_path, f"train2014")
            self.coco_path = os.path.join(coco_path, "train", "data")

        self.real_images = sorted(os.listdir(self.coco_path))[:limit]
        if test_split is not None:
            n = int(len(self.real_images) * test_split)
            if split == "test":
                self.real_images = self.real_images[n:]
            else:
                self.real_images = self.real_images[:n]
        self.real_transform = real_transform
        self.fake_transform = fake_transform

    def __len__(self):
        return len(self.real_images)

    def __getitem__(self, index):
        path = os.path.join(self.coco_path, self.real_images[index])
        img_id = os.path.basename(path).split(".")[0]
        real_image = np.array(Image.open(path))
        if len(real_image.shape) != 3:
            real_image = np.array(Image.open(path).convert("RGB"))
        fake_image_paths = sorted(os.listdir(os.path.join(self.cocofake_path, img_id)))
        fake_images = [
            np.array(Image.open(os.path.join(self.cocofake_path, img_id, x)))
            for x in fake_image_paths
        ]

        if self.real_transform is not None:
            real_image = self.real_transform(real_image)
        if self.fake_transform is not None:
            fake_images = [self.fake_transform(x) for x in fake_images]

        real_image = np.transpose(real_image, (2, 0, 1)).astype(np.float32)
        fake_images = [np.transpose(x, (2, 0, 1)).astype(np.float32) for x in fake_images]

        return {"real": real_image, "fake": fake_images}


def get_cocofake(
    coco_path,
    cocofake_path,
    train_transform=None,
    val_transform=None,
    batch_size=1,
    train_limit=-1,
    val_limit=-1,
    test_split=0.5,
):
    """
    Retrieve train, validation, and test dataloaders.

    Arguments
    ---------
    coco_path : str
        Path to the coco-2014 root directory
    cocofake_path : str
        Path to the cocofake root directory
    transforms : callable
        Transformation functions to preprocess the images
    batch_size : int
        Batch size
    train_limit : int
        Maximum number of images to include in the training set
    val_limit : int
        Maximum number of images to include in the validation set
    test_split : float
        Proportion of the validation set to use as the test set.
        Note: the test set is split after val_limit is applied.
        For example, if val_limit=10000 and test_split=0.5, then the validation set will have
        5000 images and the test set will have 5000 images.

    Returns
    -------
    train : torch.util.data.DataLoader
    val : torch.util.data.DataLoader
    test : torch.util.data.DataLoader
    """
    if not os.path.exists(coco_path):
        import fiftyone
        coco = fiftyone.zoo.load_zoo_dataset("coco-2014")

    assert os.path.exists(cocofake_path)

    train = CocoFake(coco_path, cocofake_path, real_transform=train_transform, fake_transform=train_transform, split="train", limit=train_limit)
    val = CocoFake(coco_path, cocofake_path, real_transform=val_transform, fake_transform=val_transform, split="val", limit=val_limit, test_split=test_split)
    test = CocoFake(coco_path, cocofake_path, real_transform=val_transform, fake_transform=val_transform, split="test", limit=val_limit, test_split=test_split)

    train_dataloader = torch.utils.data.DataLoader(train, batch_size=batch_size)
    val_dataloader = torch.utils.data.DataLoader(val, batch_size=batch_size)
    test_dataloader = torch.utils.data.DataLoader(test, batch_size=batch_size)

    return train_dataloader, val_dataloader, test_dataloader


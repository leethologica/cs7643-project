import os
import json
import torch
import numpy as np

from PIL import Image

class CocoFake(torch.utils.data.Dataset):
    def __init__(self, coco_path, cocofake_path, transforms=None, split="train", limit=-1):
        self.cocofake_path = os.path.join(cocofake_path, f"{split}2014")
        if split == "val":  # coco-2014 val folder is called validation while cocofake is called val2014
            split = "validation"
        self.coco_path = os.path.join(coco_path, split, "data")

        self.real_images = sorted(os.listdir(self.coco_path))[:limit]
        self.transforms = transforms

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

        if self.transforms is not None:
            real_image = self.transforms(real_image)
            fake_images = [self.transform(x) for x in fake_images]

        real_image = np.transpose(real_image, (2, 0, 1)).astype(np.float32)
        fake_images = [np.transpose(x, (2, 0, 1)).astype(np.float32) for x in fake_images]

        return {"real": real_image, "fake": fake_images}


def get_cocofake(coco_path, cocofake_path, transforms=None, batch_size=1, n=-1):
    if not os.path.exists(coco_path):
        import fiftyone
        coco = fiftyone.zoo.load_zoo_dataset("coco-2014")

    assert os.path.exists(cocofake_path)

    train = CocoFake(coco_path, cocofake_path, transforms=transforms, split="train", limit=n)
    val = CocoFake(coco_path, cocofake_path, transforms=transforms, split="val", limit=n)

    train_dataloader = torch.utils.data.DataLoader(train, batch_size=batch_size)
    val_dataloader = torch.utils.data.DataLoader(val, batch_size=batch_size)

    return train_dataloader, val_dataloader


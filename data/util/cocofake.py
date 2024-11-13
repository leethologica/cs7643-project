import os
import json
import torch
import numpy as np
import random

from PIL import Image, ImageOps

# Marcus added a pad_to_square index to take the first fake image.
def pad_to_square(image, target_size=(640, 640)):
    """
    Conduct padding on the image to make it 640 by 640
    
    Args:
    - image: PIL Image object
    - target_size: tuple (width, height), the desired size of the output image
    
    Returns:
    - Padded PIL Image
    """
    # Get the original size of the image
    width, height = image.size

    # Calculate padding for left, right, top, and bottom
    left = (target_size[0] - width) // 2
    right = target_size[0] - width - left
    top = (target_size[1] - height) // 2
    bottom = target_size[1] - height - top


    # Add padding to the image
    padding_color = (0, 0, 0) if image.mode == "RGB" else 0 #There's some grayscale images, hence need to change to grayscale instead of a tuple.
    padded_image = ImageOps.expand(image, (left, top, right, bottom), padding_color)

    return padded_image

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
        fake_prob = 0.2
    ):
        if split == "test" or split == "val":
            self.cocofake_path = os.path.join(cocofake_path, "val2014") 
            self.coco_path = os.path.join(coco_path, "val2014") #Updated with file path
        else:
            self.cocofake_path = os.path.join(cocofake_path, f"train2014")
            self.coco_path = os.path.join(coco_path, "train2014") #Updated with file path

        self.real_images = sorted(os.listdir(self.coco_path))[:limit]
        if test_split is not None:
            n = int(len(self.real_images) * test_split)
            if split == "test":
                self.real_images = self.real_images[n:]
            else:
                self.real_images = self.real_images[:n]

        # self.transforms = transforms #commeting this out because this is not used - random variable initialized.
        self.real_transform = real_transform
        self.fake_transform = fake_transform

        # Adding prob of fake images
        self.fake_prob = fake_prob

    def __len__(self):
        return len(self.real_images)

    def __getitem__(self, index):
        path = os.path.join(self.coco_path, self.real_images[index])
        img_id = os.path.basename(path).split(".")[0]

        real_image = Image.open(path).convert('RGB') #Marcus: Breakout real image into open image & array so that we can insert pad_to_square
        real_image = pad_to_square(real_image, target_size=(640, 640)) # Marcus added pad_to_square    
        real_image = np.array(real_image) # Converting to array

        fake_image_paths = sorted(os.listdir(os.path.join(self.cocofake_path, img_id)))
        fake_images = [
            Image.open(os.path.join(self.cocofake_path, img_id, x)).convert('RGB')
            for x in fake_image_paths
        ] #Breaking out fake images into open and array
        fake_images = [pad_to_square(img, target_size=(640, 640)) for img in fake_images] # Marcus added padding to the square image
        fake_images = [np.array(img) for img in fake_images] # Converting into a list of arrays

        if self.real_transform is not None:
            real_image = self.real_transform(real_image)
        if self.fake_transform is not None:
            fake_images = [self.fake_transform(x) for x in fake_images] 

        real_image = np.transpose(real_image, (2, 0, 1)).astype(np.float32)
        fake_images = [np.transpose(x, (2, 0, 1)).astype(np.float32) for x in fake_images] 

        if random.random() > self.fake_prob:
            fake_images[0] = np.zeros_like(real_image)

        return {"real": real_image, "fake": fake_images[0]} # Marcus added a 0 index to take the first fake image. WE WILL CHANGE THIS IN THE FUTURE WHEN WE TRAIN ON UNBALANCED DATASET



def get_cocofake(
    coco_path,
    cocofake_path,
    transforms=None,
    batch_size=1,
    train_limit=-1,
    val_limit=-1,
    test_split=0.5,
    fake_prob = 0.2
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

    train = CocoFake(coco_path, cocofake_path, split="train", limit=train_limit, fake_prob = fake_prob) #removed: , transforms=transforms. added: fake_prob
    val = CocoFake(coco_path, cocofake_path, split="val", limit=val_limit, test_split=test_split, fake_prob = 1) #removed: , transforms=transforms. added: fake_prob; val_loader should always be 1 to return all images
    test = CocoFake(coco_path, cocofake_path, split="test", limit=val_limit, test_split=test_split, fake_prob = 1) #removed: , transforms=transforms added: fake_prob; val_loader should always be 1 to return all images

    train_dataloader = torch.utils.data.DataLoader(train, batch_size=batch_size)
    val_dataloader = torch.utils.data.DataLoader(val, batch_size=batch_size)
    test_dataloader = torch.utils.data.DataLoader(test, batch_size=batch_size)

    return train_dataloader, val_dataloader, test_dataloader
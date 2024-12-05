import torch
import torch.nn as nn
from torchvision.transforms import v2

from captum.attr import GuidedGradCam, Saliency, IntegratedGradients

import matplotlib.pyplot as plt

import numpy as np
from PIL import Image

from matplotlib.colors import LinearSegmentedColormap

from captum.attr import visualization as viz
from captum.attr import NoiseTunnel

import os

dir_path = os.path.dirname(os.path.realpath(__file__))
import sys ; sys.path.append(f"{dir_path}/../..")
from data.util import get_cocofake
from models.resnet import ResnetFakeDetector

def visualize_captum(attributions, images, X, y, pred_labels, class_names, sign, path):

    N = attributions.shape[0]
    print(attributions.shape)
    print(N)
    fig, ax = plt.subplot_mosaic([
        [f"i{i}" for i in range(len(images))],
        [f"a{i}" for i in range(N)]
        ], figsize=(N*3.5, 7))

    default_cmap = LinearSegmentedColormap.from_list('custom blue', 
                                                 [(0, '#ffffff'),
                                                  (0.25, '#000000'),
                                                  (1, '#000000')], N=256)

    for i in range(len(images)):
        ax[f"i{i}"].imshow(images[i])
        ax[f"i{i}"].axis('off')
        ax[f"i{i}"].set_title(f"True: {class_names[y[i]]} Pred: {class_names[pred_labels[i]]}")

    for i in range(N):
        _ = viz.visualize_image_attr(np.transpose(attributions[i].cpu().detach().numpy(), (1,2,0)),
                                    method='heat_map',
                                    cmap=default_cmap,
                                    sign=sign,
                                    plt_fig_axis=(fig, ax[f"a{i}"]),
                                    outlier_perc=1)
    plt.savefig(path)

def visualize(path, X, y, pred_labels, class_names, attributions, titles, attr_preprocess=lambda attr: attr.permute(1, 2, 0).detach().numpy(),
                        cmap='viridis', alpha=0.7, clip=False):
    N = attributions[0].shape[0]
    fig, ax = plt.subplot_mosaic([
        [f"i{i}" for i in range(N)],
        [f"a{i}" for i in range(N)]
        ], figsize=(N*3.5, 7))
    
    for i in range(len(X)):
        ax[f"i{i}"].imshow(X[i])
        ax[f"i{i}"].axis('off')
        ax[f"i{i}"].set_title(f"True: {class_names[y[i]]} Pred: {class_names[pred_labels[i]]}")

    for j in range(len(attributions)):
        for i in range(N):
            attr = np.array(attr_preprocess(attributions[j][i].cpu()))
            if clip:
                attr = (attr - np.mean(attr)) / np.std(attr).clip(1e-20)
                attr = attr * 0.2 + 0.5
                attr = attr.clip(0, 1.0)
            ax[f"a{i}"].imshow(attr, cmap=cmap, alpha=alpha)
            ax[f"a{i}"].axis("off")
    plt.savefig(path)

def get_data(half_batch, transform, device): 
    coco_path = "../../data/coco-2014/validation/data"
    cocofake_path = "../../data/cocofake/val2014"
    real_images = sorted(os.listdir(coco_path))
    
    real_image_tensors = []
    fake_image_tensors = []
    real_image_arrays = []
    fake_image_arrays = []

    idx = []
    for i in range(half_batch):
        index = np.random.randint(0, len(real_images))
        idx.append(index)
        path = os.path.join(coco_path, real_images[index])
        img_id = os.path.basename(path).split(".")[0]
        real_image = Image.open(path).convert("RGB")
        fake_image_paths = sorted(os.listdir(os.path.join(cocofake_path, img_id)))
        fake_image = Image.open(os.path.join(cocofake_path, img_id, fake_image_paths[0]))

        real_image_arrays.append(np.asarray(real_image))
        fake_image_arrays.append(np.asarray(fake_image))

        real_image_tensors.append(transform(real_image))
        fake_image_tensors.append(transform(fake_image))
    
    real_image_tensors = torch.stack(real_image_tensors, dim=0)
    fake_image_tensors = torch.stack(fake_image_tensors, dim=0)

    images = real_image_arrays + fake_image_arrays

    batch = torch.cat([real_image_tensors, fake_image_tensors], dim=0).to(device)

    labels = torch.cat([torch.zeros(half_batch, 1),
                        torch.ones(half_batch, 1)], dim=0).long().squeeze(1).to(device)
    
    return batch, labels, images, idx

normalize = v2.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

if __name__ == "__main__":
    train_preprocess = v2.Compose([
        v2.RandomResizedCrop(224),
        v2.RandomHorizontalFlip(),
        v2.ToTensor(),
        normalize
    ])

    val_preprocess = v2.Compose([
        v2.Resize(256),
        v2.CenterCrop(224),
        v2.ToTensor(),
        normalize
    ])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = 2
    backbone = "resnet18"
    epoch = 25

    num_examples = 10
    np.random.seed(0)
    #np.random.seed(69)
    print(device)
    
    save_dir = backbone

    model = ResnetFakeDetector(backbone=backbone, hidden_dim=1024, device=device)

    model.load_state_dict(torch.load(f"checkpoints/{backbone}_epoch{epoch}.pt"))
    #model.load_state_dict(torch.load(f"checkpoints/archive/{backbone}_epoch{epoch}.pt"))
    model.zero_grad()
    

    guided_gc = GuidedGradCam(model=model, layer=model.resnet.layer4)

    int_grad = IntegratedGradients(model)
    sal = Saliency(model)


    for _ in range(num_examples):
        batch, labels, images, idx = get_data(batch_size, val_preprocess, device)
        with torch.no_grad():
            pred = model(batch)
            pred = nn.functional.sigmoid(pred)
            pred_labels = (pred > 0.5).cpu().squeeze(1)

        attribution = guided_gc.attribute(batch)
        class_names = ["real", "fake"]
        visualize(f"attribution/{save_dir}/guided_gradcam_captum_{backbone}_epoch{epoch}_{sum(idx)}.png", images, labels, pred_labels, class_names, [attribution], ['Guided GradCam'], clip=True)

        noise_tunnel = NoiseTunnel(int_grad)

        attributions_ig_nt = noise_tunnel.attribute(batch, nt_samples=10, nt_type='smoothgrad_sq', nt_samples_batch_size=1)
        visualize_captum(attributions_ig_nt, images, batch, labels, pred_labels, class_names, "positive",
                        f"attribution/{save_dir}/integrated_gradients_{backbone}_epoch{epoch}_{sum(idx)}.png")


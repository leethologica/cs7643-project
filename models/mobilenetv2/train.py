import os
import sys ; sys.path.append("../../")
import torch
import argparse
import numpy as np

from tqdm.auto import tqdm
from torchvision import models, transforms

from data.util.cocofake import get_cocofake

def train(
    model,
    train,
    val,
    lr: float,
    wd: float,
    criterion,
    n_epochs: int,
    savepath: str,
    device: torch.device,
    patience: int = 20,
):
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr,
        weight_decay=wd,
    )

    best_model = model.state_dict()
    best_val_loss = np.inf
    stale = 0

    for epoch in range(n_epochs):
        if stale > patience:
            print(f"Model validation loss has not improved in {patience} epochs, exiting early.")
            break

        for phase in ["train", "val"]:
            if phase == "train":
                model.train(True)
                data = train
            else:
                model.train(False)
                data = val

            running_loss = 0
            running_correct = 0
            total = 0

            with torch.set_grad_enabled(phase == "train"):
                for i, batch in tqdm(enumerate(data), total=len(data)):
                    outputs = None
                    optimizer.zero_grad()
                    gt = []
                    real_images = batch["real"]
                    real_labels = torch.zeros(real_images.shape[0], dtype=torch.long)
                    fake_images = torch.vstack(batch["fake"])
                    fake_labels = torch.ones(fake_images.shape[0], dtype=torch.long)
                    inputs = torch.cat((real_images, fake_images), dim=0).to(device)
                    labels = torch.cat((real_labels, fake_labels), dim=0).to(device)

                    # shuffle inputs
                    idx = torch.randperm(inputs.shape[0])
                    inputs = inputs[idx]
                    labels = labels[idx]

                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    _, preds = torch.max(outputs, 1)
                    running_correct += (preds == labels).sum().item()
                    total += labels.shape[0]

                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                    running_loss += loss

                accuracy = running_correct / total
                print(f"{phase} avg batch loss: {running_loss / len(data)} | {phase} accuracy: {accuracy}")
                if phase == "val":
                    if running_loss < best_val_loss:
                        stale = 0
                        best_val_loss = running_loss
                        best_model = model.state_dict()
                        torch.save(model, savepath)
                    else:
                        stale += 1


def eval(
    model,
    data,
    n_runs: int,
):
    pass

def parse_args():
    argparser = argparse.ArgumentParser(
        "Fine-tune MobileNetV2", formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    argparser.add_argument("--gpu", type=int, default=0, help="Which gpu to use. -1 uses CPU")
    argparser.add_argument("--n-runs", type=int, default=1, help="Number of runs")
    argparser.add_argument("--n-epochs", type=int, default=50, help="Number of epochs")
    argparser.add_argument("--batch", type=int, default=16, help="Batch size")
    argparser.add_argument("--lr", type=float, default=1e-6, help="Learning rate")
    argparser.add_argument("--wd", type=float, default=1e-4, help="Weight decay")
    argparser.add_argument("--eval-only", action="store_true", help="No training")
    argparser.add_argument("--cocopath", type=str, required=True, help="Path to coco-2014")
    argparser.add_argument("--cocofakepath", type=str, required=True, help="Path to cocofake")
    argparser.add_argument("--savepath", type=str, default="./best_mobilenetv2.pt", help="Path to save model to")
    argparser.add_argument("--train-lim", type=int, default=-1, help="Maximum number of training samples to use")
    argparser.add_argument("--val-lim", type=int, default=-1, help="Maximum number of validation samples to use")
    argparser.add_argument("--num-fake", type=int, default=5, help="Number of fake images to include per real image, between 1 and 5")
    argparser.add_argument("--patience", type=int, default=20, help="Number of no-change validation epochs before early stopping")
    args = argparser.parse_args()

    return args


# python train.py --cocopath /u00/data/coco-2014 --cocofakepath /u00/data/cocofake --train-lim 10000 --val-lim 2500
if __name__ == "__main__":
    args = parse_args()

    if os.path.exists(args.savepath):
        response = input(f"File already exists at {args.savepath}. Overwrite? (y/n) ")
        if response.lower()[0] == "n":
            model = torch.load(args.savepath)
            args.eval_only = True

    device = torch.device(f"cuda:{args.gpu}" if args.gpu >= 0 else "cpu")

    #model = torch.hub.load("pytorch/vision", "mobilenet_v2", pretrained=True).to(device)
    model = models.mobilenet_v2(pretrained=True)
    model.classifier[1] = torch.nn.Linear(in_features=model.classifier[1].in_features, out_features=2)
    model = model.to(device)

    real_transform = lambda x: transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )(transforms.CenterCrop(512)(x))

    train_data, val_data, test_data, = get_cocofake(
        args.cocopath, args.cocofakepath, train_limit=args.train_lim, val_limit=args.val_lim,
        real_transform=real_transform, batch_size=args.batch, train_n_workers=8, val_n_workers=2, num_fake=args.num_fake
    )

    #criterion = torch.nn.BCELoss()
    criterion = torch.nn.CrossEntropyLoss()
    if not args.eval_only:
        train(
            model=model,
            train=train_data,
            val=val_data,
            n_epochs=args.n_epochs,
            criterion=criterion,
            lr=args.lr,
            wd=args.wd,
            savepath=args.savepath,
            patience=args.patience,
            device=device,
        )



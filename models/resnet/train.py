import torch
import torch.nn as nn
from torchvision.models import resnet50
from torchvision.models.feature_extraction import get_graph_node_names
from torchvision.models.feature_extraction import create_feature_extractor

from torchvision.transforms import v2

from tqdm.auto import tqdm
import os

dir_path = os.path.dirname(os.path.realpath(__file__))
import sys ; sys.path.append(f"{dir_path}/../..")
from data.util import get_cocofake

import matplotlib.pyplot as plt

train_nodes, eval_nodes = get_graph_node_names(resnet50())
#print(train_nodes)
#print(eval_nodes)

class Resnet50FakeDetector(torch.nn.Module):
    def __init__(self, return_nodes, hidden_dim, device, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.device = device

        self.hidden_dim = hidden_dim

        m = resnet50()
        self.fe = create_feature_extractor(m, return_nodes=return_nodes).to(self.device)

        # Dry run to get number of channels for FPN
        inp = torch.randn(1, 3, 224, 224).to(self.device)
        with torch.no_grad():
            out = self.fe(inp)
        self.feat_shape = out["feat"].shape

        self.classification_head = nn.Sequential(
            nn.Linear(self.feat_shape[1], self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, 1),
            nn.Sigmoid()
        ).to(device)

    
    def forward(self, x):
        x = self.fe(x)
        x = self.classification_head(x["feat"])
        return x

def accuracy(pred, target):
    pred_labels = (pred > 0.5)
    n = pred.shape[0]
    acc = (pred_labels == target).sum().item() / n
    return acc

normalize = v2.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])


if __name__ == "__main__":
    # input needs to be [N, 3, 244, 244]
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
    return_nodes = {
        "flatten": "feat"
    }
    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = 64
    lr = 0.01
    epochs = 15

    print(device)
    model = Resnet50FakeDetector(return_nodes=return_nodes, hidden_dim=1024, device=device)
    train, val, test, = get_cocofake("../../data/coco-2014", "../../data/cocofake",
                                     train_limit=-1, val_limit=10000, batch_size=batch_size,
                                     train_transform=train_preprocess,
                                     val_transform=val_preprocess,
                                     train_n_workers=8,
                                     val_n_workers=2
                                    )
    loss_fn = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_losses = []
    val_losses = []

    train_acc = []
    val_acc = []
    for e in range(epochs):
        running_loss = 0
        train_b_count = 0

        running_train_acc = 0
        n_points = 0
        model.train(True)
        for batch in tqdm(train):
            optimizer.zero_grad()

            x = torch.cat([batch["real"], batch["fake"]], dim=0).to(device)
            # pred on real
            out = model(x)
            labels = torch.cat([torch.zeros(batch["real"].shape[0], 1), torch.ones(batch["fake"].shape[0], 1)], dim=0).to(device)
            # gonna make real class 0
            # fake is class 1
            loss = loss_fn(out, labels)

            loss.backward()
            optimizer.step()

            with torch.no_grad():
                running_loss += loss.cpu()
                running_train_acc += accuracy(out, labels)
                n_points += out.shape[0]

            train_b_count += 1
        avg_train_loss = running_loss/train_b_count
        avg_train_acc = running_train_acc/n_points
        train_acc.append(avg_train_acc)
        train_losses.append(avg_train_loss)
        print(f"Train Loss: {avg_train_loss}")
        print(f"Train Accuracy: {avg_train_acc}")
        torch.save(model.state_dict(), f"checkpoints/resnet50_epoch{e}.pt")
        

        model.eval()
        avg_val_loss = 0.0
        val_b_count = 0

        running_val_acc = 0
        val_n_points = 0
        with torch.no_grad():
            for batch in tqdm(val):

                x = torch.cat([batch["real"], batch["fake"]], dim=0).to(device)
                # pred on real
                out = model(x)
                labels = torch.cat([torch.zeros(batch["real"].shape[0], 1), torch.ones(batch["fake"].shape[0], 1)], dim=0).to(device)
                # gonna make real class 0
                # fake is class 1
                val_loss = loss_fn(out, labels)

                avg_val_loss += loss.cpu()
                val_b_count += 1

                # compute accuracy stuff
                running_val_acc += accuracy(out, labels)
                val_n_points += out.shape[0]
        avg_val_loss /= val_b_count
        avg_val_acc = running_val_acc/val_n_points
        val_acc.append(avg_val_acc)
        val_losses.append(avg_val_loss)
        print(f"Validation Loss: {avg_val_loss}")
        print(f"Validation Accuracy: {avg_val_acc}")
    
    # plot ce loss
    fig = plt.figure()
    plt.plot(train_losses, label="Train")
    plt.plot(val_losses, label="Validation")
    plt.legend()
    plt.savefig("figures/resnet18_CE_loss_lc.png")

    # plot acc learning curve
    fig = plt.figure()
    plt.plot(train_acc, label="Train")
    plt.plot(val_acc, label="Validation")
    plt.legend()
    plt.savefig("figures/resnet18_acc_lc.png")
    
    

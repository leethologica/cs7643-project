import torch
import torch.nn as nn
from torchvision.models import resnet18
from torchvision.models.feature_extraction import get_graph_node_names
from torchvision.models.feature_extraction import create_feature_extractor

from torchvision.transforms import v2

from tqdm.auto import tqdm
import os

dir_path = os.path.dirname(os.path.realpath(__file__))
import sys ; sys.path.append(f"{dir_path}/../..")
from data.util import get_cocofake

train_nodes, eval_nodes = get_graph_node_names(resnet18())
#print(train_nodes)
#print(eval_nodes)

class Resnet18FakeDetector(torch.nn.Module):
    def __init__(self, return_nodes, hidden_dim, device, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.device = device

        self.hidden_dim = hidden_dim

        m = resnet18()
        self.fe = create_feature_extractor(m, return_nodes=return_nodes).to(self.device)

        # Dry run to get number of channels for FPN
        inp = torch.randn(1, 3, 224, 224).to(self.device)
        with torch.no_grad():
            out = self.fe(inp)
        self.feat_shape = out["feat"].shape

        self.classification_head = nn.Sequential(
            nn.Linear(self.feat_shape[1], self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, 2)
        ).to(device)

    
    def forward(self, x):
        x = self.fe(x)
        x = self.classification_head(x["feat"])
        return x

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
    lr = 0.001
    epochs = 5

    print(device)
    model = Resnet18FakeDetector(return_nodes=return_nodes, hidden_dim=128, device=device)
    train, val, test, = get_cocofake("../../data/coco-2014", "../../data/cocofake",
                                     train_limit=-1, val_limit=5000, batch_size=batch_size,
                                     train_transform=train_preprocess,
                                     val_transform=val_preprocess,
                                     train_n_workers=8,
                                     val_n_workers=2
                                    )
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for e in range(epochs):
        running_loss = 0
        train_b_count = 0
        for batch in tqdm(train):
            optimizer.zero_grad()
            # pred on real
            real_out = model(batch["real"].to(device))
            # gonna make real class 0
            # fake is class 1
            real_label = torch.Tensor([1, 0])
            batch_real_labels = real_label.repeat(real_out.shape[0], 1).to(device)

            loss = loss_fn(real_out, batch_real_labels)
            running_loss += loss

            loss.backward()
            # pred on fake
            fake_batch = torch.cat(batch["fake"], dim=0).to(device)
            fake_out = model(fake_batch)

            # fake is class 1
            fake_label = torch.Tensor([0,1])
            batch_fake_labels = fake_label.repeat(fake_out.shape[0], 1).to(device)
            
            loss = loss_fn(fake_out, batch_fake_labels)

            running_loss += loss

            loss.backward()
            optimizer.step()
            train_b_count += 1
        print(f"Train Loss: {running_loss/train_b_count}")
        torch.save(model.state_dict(), f"checkpoints/resnet18_epoch{e}.pt")
        

        model.eval()
        avg_val_loss = 0.0
        val_b_count = 0
        with torch.no_grad():
            for batch in tqdm(val):

                real_out = model(batch["real"].to(device))
                # gonna make real class 0
                # fake is class 1
                real_label = torch.Tensor([1, 0])
                batch_real_labels = real_label.repeat(real_out.shape[0], 1).to(device)

                val_loss = loss_fn(real_out, batch_real_labels)
                avg_val_loss += loss

                # pred on fake
                fake_batch = torch.cat(batch["fake"], dim=0).to(device)
                fake_out = model(fake_batch)

                # fake is class 1
                fake_label = torch.Tensor([0,1])
                batch_fake_labels = fake_label.repeat(fake_out.shape[0], 1).to(device)
                
                val_loss = loss_fn(fake_out, batch_fake_labels)
                avg_val_loss += loss
                val_b_count += 1
        avg_val_loss /= val_b_count
        print(f"Validation Loss: {avg_val_loss}")

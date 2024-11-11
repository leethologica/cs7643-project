import torch
import torch.nn as nn
from torchvision.models import resnet18
from torchvision.models.feature_extraction import get_graph_node_names
from torchvision.models.feature_extraction import create_feature_extractor

import torchvision.transforms as transforms

from tqdm.auto import tqdm
import os

dir_path = os.path.dirname(os.path.realpath(__file__))
os.environ['KMP_DUPLICATE_LIB_OK']='True'
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
        print(self.feat_shape)

        self.classification_head = nn.Sequential(
            nn.Linear(self.feat_shape[1], self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, 2)
        )

    
    def forward(self, x):
        x = self.fe(x)
        x = self.classification_head(x["feat"])
        return x

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

# input needs to be [N, 3, 244, 244]
train_preprocess = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize
])

val_preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    normalize
])

if __name__ == "__main__":
    return_nodes = {
        "flatten": "feat"
    }
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)
    model = Resnet18FakeDetector(return_nodes=return_nodes, hidden_dim=128, device=device)

    train, val, test, = get_cocofake("../../data/coco-2014", "../../data/cocofake", train_limit=1000, val_limit=500)
    breakpoint()
    for batch in tqdm(train):
        print(batch)
        break

    for batch in tqdm(val):
        pass

    for batch in tqdm(test):
        pass



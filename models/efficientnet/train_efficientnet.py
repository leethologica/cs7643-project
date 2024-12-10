import os
import torch
import numpy as np
import random
from PIL import Image
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision import transforms
import torchvision.models as models
from torchvision.models import SqueezeNet1_1_Weights
import torch.nn as nn
import torch.optim as optim
from cocofake import CocoFake, get_cocofake

import matplotlib.pyplot as plt

# Largest image 640 x 640
# from Marcus's training script

def get_device():
    if torch.cuda.is_available():
        print("cuda available")
        return "cuda"
    elif torch.backends.mps.is_available():
        print("mps available")
        return "mps"
    else:
        print("cpu device")
        return "cpu"

# SEEDING FUNCTION:
def set_seed(seed):
    """
    Set the seed for reproducibility across:
    - Python's built-in random module
    - Numpy
    - PyTorch
    - CUDA (if using a GPU)
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU setups

set_seed(42)

######################################################################### TRAINING FOR BENCHMARK #########################################################################

###### PART 1: DEFINING VARIABLES ######
coco_path = '/home/hice1/tsun307/scratch/dl_project/data/coco-2014'
cocofake_path = '/home/hice1/tsun307/scratch/dl_project/data/cocofake' # Path definitions

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # Device definition

batch_size = 16#128 # Defining the batch size for training
train_limit = 10000 # Training only on a subset of dataset
val_limit = 5000 # Validating only on a subset of validation dataset
test_split = 0.5 # splitting up the val dataset into val and test
fake_prob = 1 #the percentage of fake images to keep

criterion = nn.CrossEntropyLoss() #Loss function
learning_rate = 0.001

num_epochs = 30


print('Is Cuda available:', torch.cuda.is_available())
print('Variables Defined!')

###### PART 2: TRANSFORM & GENERATING DATA - Simple transformation to fit your model size. ######
transforms = transforms.Compose([
    transforms.Resize(224),  # Resize the image to 224x224 as required by Squeezenet
    transforms.ToTensor(),
    # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  #Normalization used based on ImageNet's means/SD
])

# Getting dataset loaders
train_loader, val_loader, test_loader = get_cocofake(
    coco_path=coco_path,
    cocofake_path=cocofake_path,
    transforms=transforms, 
    batch_size=batch_size,  
    train_limit=train_limit,
    val_limit=val_limit, 
    test_split=test_split,
    fake_prob = fake_prob  # Split validation data into test and validation sets
)

print('Dataloading/Transformation Completed!')

###### PART 3: MODEL BUILDING & EVALUATION ######

# Importing 
model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT) 
model.num_classes = 2

model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)  # Binary classification (2 classes)

# Pushing model to GPU
model = model.to(device)

# Model optimizer
optimizer = optim.Adam(model.parameters(), lr=0.001)

print('Model Defined!')

# TRAINING FUNCTION!
def train(model, train_loader, criterion, optimizer, device):
    model.train()  # Set model to training mode
    running_loss = 0.0
    correct = 0
    total=0
    batch_count = 0

    for batch in train_loader:
        #print("starting new batch", batch_count)
        #batch_count += 1

        # Real image: creating labels
        real_images = batch["real"].to(device)
        real_labels = torch.zeros(real_images.size(0), dtype=torch.long).to(device)

        # # Fake image & concatenating real + fake images if possible
        # print(batch.keys())

        # if batch.get('fake') is None:
        #     inputs = real_images
        #     labels = real_labels
            
        # else:
        fake_images = batch["fake"].to(device)   
        fake_labels = torch.ones(fake_images.size(0), dtype=torch.long).to(device)

        # Concatenate images and labels to make a single batch
        inputs = torch.cat((real_images, fake_images), dim=0)
        labels = torch.cat((real_labels, fake_labels), dim=0)
   
        # print('labels here', labels)
        # print('inptus here', inputs)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        epoch_loss = running_loss / len(train_loader.dataset)
        accuracy = correct / total

    epoch_loss = running_loss / len(train_loader.dataset)
    accuracy = correct / total
    #print(f"epoch_loss: {epoch_loss}")
    #print(f"accuracy: {accuracy} " )

    return epoch_loss, accuracy

# VALIDATION FUNCTION!
def validate(model, val_loader, criterion, device):
    model.eval()  # Set model to evaluation mode
    running_loss = 0.0
    correct = 0
    total=0

    with torch.no_grad():
        for batch in val_loader:
            real_images = batch["real"].to(device)
            fake_images = batch["fake"].to(device)


            real_labels = torch.zeros(real_images.size(0), dtype=torch.long).to(device)
            fake_labels = torch.ones(fake_images.size(0), dtype=torch.long).to(device)

            inputs = torch.cat((real_images, fake_images), dim=0)
            labels = torch.cat((real_labels, fake_labels), dim=0)

            outputs = model(inputs)
            print(outputs)
            # print(inputs.shape)
            # print(inputs)

            # print('outputs here', outputs)

            # if torch.equal(outputs[1], torch.tensor([0.0, 0.0], device=device)):
                
            #     print('equal!')
            #     print(inputs)
            #     print(outputs)


            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)

            _, preds = torch.max(outputs, 1)

            # print('predshere',preds)

            correct += (preds == labels).sum().item()
            total += labels.size(0)
            
            epoch_loss = running_loss / len(train_loader.dataset)
            accuracy = correct / total
            #print("epoch_loss: ", epoch_loss)
            #print("accuracy: ", accuracy)
    
    epoch_loss = running_loss / total
    accuracy = correct / total
    return epoch_loss, accuracy


###### PART 4: CHECKPOINTING/SEED SETTING ######

# SAVING CHECKPOINT!
def save_checkpoint(model, optimizer, epoch, train_loss, val_loss, dataloader_state = 'seed', filename="checkpoint.pth"):
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(), #Saving optimizer
        "train_loss": train_loss,
        "val_loss": val_loss,
        "random_state": 42,  # Fixed at 42
        "dataloader_state": dataloader_state #Fixing at seed 
    }
    torch.save(checkpoint, filename)
    print(f"Checkpoint saved at epoch {epoch}") # We are saving by epochs not by bach size

# LOADING CHECKPOINT!
def load_checkpoint(model, optimizer, filename="checkpoint.pth", device="cuda"):
    checkpoint = torch.load(filename, map_location=device)
    
    
    # NEED TO REMOVE THIS IF YOU'RE USING MULTIPLE GPU. PROBLEM OF 'module.' BEING A PREFIX IN THE KEYS WHEN YOU USE DATAPARALLEL
    state_dict = checkpoint["model_state_dict"]
    new_state_dict = {}
    
    for key, value in state_dict.items():
        new_key = key.replace("module.", "")  # Remove 'module.' prefix
        new_state_dict[new_key] = value
    
    # Load the updated state dicitonaries
    model.load_state_dict(new_state_dict)

    # Loading the rest of the checkpoints
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    epoch = checkpoint["epoch"]
    train_loss = checkpoint["train_loss"]
    val_loss = checkpoint["val_loss"]
    random_state = checkpoint["random_state"]
    dataloader_state = checkpoint["dataloader_state"]
    
    print(f"Checkpoint loaded from epoch {epoch}")

    set_seed(42)

    return model, optimizer, epoch, train_loss, val_loss, random_state, dataloader_state

def log_metrics(epoch, train_loss, train_acc, val_loss=None, val_acc=None, log_file="training_log_first_fake.txt"):
    with open(log_file, 'a') as f:
        log_entry = f"Epoch {epoch + 1}: Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.4f}"
        if val_loss is not None and val_acc is not None:
            log_entry += f", Val Loss: {val_loss:.4f}, Val Accuracy: {val_acc:.4f}"
        log_entry += "\n"
        f.write(log_entry)

###### PART 5: MODEL BUILDING & EVALUATION ######
ep = None
# Loading checkpoint in case of GPU FAILURE. TO BE UNCOMMENTED OUT WHEN YOU NEED TO LOAD CHECKPOINT!
#model, optimizer, ep, train_loss, val_loss, random_state, dataloader_state = load_checkpoint(model, optimizer, "checkpoint.pth")
#try: # Setting this so as to use this for the range function in the training methods below.
 #   print(ep)
#except:
 #   ep = 0

if not ep:
    ep = 0
    
print(f"Training from ep {ep}")

# Simple Parallelization
if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs!")
    model = torch.nn.DataParallel(model)


print('Training Starts!')

log_file = "training_log.txt"

for epoch in range(ep, num_epochs):

    train_loss, train_accuracy = train(model, train_loader, criterion, optimizer, device)
    print(f"train loss of epoch {epoch+1}: {train_loss}")
    print(f"train accuracy of epoch {epoch+1}: {train_accuracy}")
    val_loss, val_accuracy = validate(model, val_loader, criterion, device)
    print(f"val loss of epoch {epoch+1}: {val_loss}")
    print(f"val accuracy of epoch {epoch+1}: {val_accuracy}")
    
    print(f"Completed Epoch {epoch + 1}/{num_epochs}")
    print(f"Train Loss: {train_loss:.4f} | Train accuracy: {train_accuracy * 100:.2f}%")
    print(f"Val Loss: {val_loss:.4f} | Val Accuracy: {val_accuracy * 100:.2f}%")
    log_metrics(epoch, train_loss, train_accuracy, val_loss, val_accuracy, log_file)

    save_checkpoint(model, optimizer, epoch, train_loss, val_loss)





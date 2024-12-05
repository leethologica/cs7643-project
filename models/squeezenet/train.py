import torch
import numpy as np
import random
import os
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
coco_path = '/home/hice1/mtan75/scratch/dlproject/dataset_real'
cocofake_path = '/home/hice1/mtan75/scratch/dlproject/dataset_fake' # Path definitions

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # Device definition

batch_size = 128 # Defining the batch size for training
fake_size = batch_size/2
train_limit = -1 # Training only on a subset of dataset
val_limit = -1 # Validating only on a subset of validation dataset
test_split = 0.7 # splitting up the val dataset into val and test
fake_prob = 1 #the percentage of fake images to keep

criterion = nn.CrossEntropyLoss() #Loss function; add [0.7, 0.3] to weigh the losses!
learning_rate = 0.001

num_epochs = 10

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
    fake_prob = fake_prob,
    train_n_workers = 8  # Split validation data into test and validation sets
)

print('Dataloading/Transformation Completed!')

###### PART 3: MODEL BUILDING & EVALUATION ######

# Importing Squeezenets
model = models.squeezenet1_1(weights=SqueezeNet1_1_Weights.DEFAULT) # Depends on whether you want pretrained weights or not
# model = models.squeezenet1_1(pretrained=False)
model.num_classes = 2

# Updating the last layer of Squeenet
model.classifier[1] = nn.Conv2d(512, 2, kernel_size=(1, 1), stride=(1, 1))  # 2 classes

# Pushing model to GPU
model = model.to(device)

# Model optimizer
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

print('Model Defined!')

# TRAINING FUNCTION!
def train(model, train_loader, criterion, optimizer, device):
    model.train()  # Set model to training mode
    running_loss = 0.0
    correct = 0
    total=0

    for batch in train_loader:

        # Real image: creating labels
        real_images = batch["real"].to(device)
        real_labels = torch.zeros(real_images.size(0), dtype=torch.long).to(device)

        fake_images = batch["fake"][fake_size:-1].to(device) # This generates a smaller amount of fake images
        # fake_images = batch["fake"][-1].unsqueeze(0).to(device) # This generates a smaller amount of fake images. In case you only wants one image
        fake_images = batch["fake"].to(device)
        fake_labels = torch.ones(fake_images.size(0), dtype=torch.long).to(device)

        # Concatenate images and labels to make a single batch
        inputs = torch.cat((real_images, fake_images), dim=0)
        labels = torch.cat((real_labels, fake_labels), dim=0)

        # Shuffling the inputs and labels - ensure no memorization of the labels of the data through the sequences.
        permutation = torch.randperm(inputs.size(0))
        inputs = inputs[permutation]
        labels = labels[permutation]

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

        # print(batch.keys())

        running_loss += loss.item() * inputs.size(0)

    epoch_loss = running_loss / len(train_loader.dataset)
    accuracy = correct / total

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

            permutation = torch.randperm(inputs.size(0))
            inputs = inputs[permutation]
            labels = labels[permutation]


            outputs = model(inputs)
            # print(outputs)
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

            # print('this is the predictor', preds)
            # print('this is the label', labels)

            # print('predshere',preds)

            correct += (preds == labels).sum().item()
            total += labels.size(0)
    
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


###### PART 5: MODEL BUILDING & EVALUATION ######

# Loading checkpoint in case of GPU FAILURE. TO BE UNCOMMENTED OUT WHEN YOU NEED TO LOAD CHECKPOINT!
# model, optimizer, ep, train_loss, val_loss, random_state, dataloader_state = load_checkpoint(model, optimizer, "checkpoint.pth")
try: # Setting this so as to use this for the range function in the training methods below.
    print(ep)
except:
    ep = 0

# Simple Parallelization
if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs!")
    model = torch.nn.DataParallel(model)

print('Training Starts!')

epoch_list = []
train_loss_list = []
train_accuracy_list = []
val_loss_list = []
val_accuracy_list = []

for epoch in range(ep, num_epochs):

    train_loss, train_accuracy = train(model, train_loader, criterion, optimizer, device)
    val_loss, val_accuracy = validate(model, test_loader, criterion, device)
    
    print(f"Completed Epoch {epoch + 1}/{num_epochs}")
    print(f"Train Loss: {train_loss:.4f} | Train accuracy: {train_accuracy * 100:.2f}%")
    print(f"Val Loss: {val_loss:.4f} | Val Accuracy: {val_accuracy * 100:.2f}%")

    save_checkpoint(model, optimizer, epoch, train_loss, val_loss)

    epoch_list.append(epoch)
    train_loss_list.append(train_loss)
    train_accuracy_list.append(train_accuracy)
    val_loss_list.append(val_loss)
    val_accuracy_list.append(val_accuracy)

# Plots #1 - Loss

# Plot the first line
plt.plot(epoch_list, train_loss_list, label='Training Loss', color='blue')

# Plot the second line
plt.plot(epoch_list, val_loss_list, label='Val Loss', color='red')

# Add labels, title, and legend
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss Curve')
plt.legend()

plt.savefig('Loss_Balanced_unPretrained.png')  # Save as PNG

plt.clf()


# Plots #2 - Accuracy

# Plot the first line
plt.plot(epoch_list, train_accuracy_list, label='Training Accuracy', color='blue')

# Plot the second line
plt.plot(epoch_list, val_accuracy_list, label='Val Accuracy', color='red')

# Add labels, title, and legend
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Accuracy Curve')
plt.legend()

plt.savefig('Accuracy_Balanced_unPretrained.png')  # Save as PNG

plt.clf()


# print(ep)

# for name, param in model.state_dict().items():
#     print(f"Layer: {name}, Weights: {param}")
# WHAT HAPPENS IF YOU ONLY USE 1 GPU?-> Tried with 1 gpu - still getting 99% accuracy 
# WHAT HAPPENS if we reduce the fake dataset? -> reducing the number of fake dataset reduces the accuracy scores.



######################################################################### MISC SECTION - TESTING ON DATASIZE ETC. #########################################################################



# Define the path to your folder with images
# folder_path = "/home/hice1/mtan75/scratch/dlproject/dataset_real/train2014"  # Replace with your folder path
# folder_path = '/home/hice1/mtan75/scratch/dlproject/dataset_fake/train2014/COCO_train2014_000000057300'


# ctr = 1
# # List all files in the folder
# for filename in os.listdir(folder_path):
#     # Construct the full file path
#     file_path = os.path.join(folder_path, filename)

#     # Check if it's an image file by its extension (optional but recommended)
#     if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')):
#         try:
#             # Open the image and get its dimensions
#             with Image.open(file_path) as img:
#                 width, height = img.size
#                 print(f"Image: {filename}, Width: {width}, Height: {height}")
#         except Exception as e:
#             print(f"Could not open {filename}: {e}")
#     ctr+=1


# Initialize variables to track max dimensions
# max_width = 0
# max_height = 0
# max_image = ''

# ctr = 0
# directory_path = '/home/hice1/mtan75/scratch/dlproject/dataset_real/train2014'

# # Loop through all files in the directory
# for filename in os.listdir(directory_path):
#     if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
#         image_path = os.path.join(directory_path, filename)
        
#         try:
#             # Open the image using PIL
#             with Image.open(image_path) as img:
#                 # Get image dimensions (width, height)
#                 width, height = img.size
#                 print(f"Image: {filename}, Width: {width}, Height: {height}")
                
#                 # Update max dimensions if the current image is larger
#                 if width > max_width or height > max_height:
#                     max_width = width
#                     max_height = height
#                     max_image = filename
#         except Exception as e:
#             print(f"Error opening {filename}: {e}")



#     ctr +=1
#     if ctr >= 40000:
#         break

# # Output the results
# print(f"The largest image is {max_image} with dimensions: {max_width}x{max_height}")


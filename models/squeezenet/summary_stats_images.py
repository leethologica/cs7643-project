

import torch
import numpy as np
import random
import os
from PIL import Image
# from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision import transforms
import torchvision.models as models
from torchvision.models import SqueezeNet1_1_Weights
import torch.nn as nn
import torch.optim as optim
from cocofake import CocoFake, get_cocofake


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
train_limit = -1 # Training only on a subset of dataset
val_limit = -1 # Validating only on a subset of validation dataset
test_split = 0.5 # splitting up the val dataset into val and test
fake_prob = 1 #the percentage of fake images to keep

criterion = nn.CrossEntropyLoss() #Loss function
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

print('DATASET LOADED')

###### PART 3: Train Loader Loaded In! ######

import cv2
from skimage.feature import local_binary_pattern

# Function to process the dataset and aggregate features
def process_dataset(loader):
    # Initialize lists to store features for real and fake images
    all_real_features = {
        'mean_intensity': [],
        'contrast': [],
        'sharpness': [],
        'edge_density': [],
        'lbp_uniformity': [],
        'red_hist_mean': [],
        'green_hist_mean': [],
        'blue_hist_mean': [],
        'red_hist_var': [],
        'green_hist_var': [],
        'blue_hist_var': []
    }
    
    all_fake_features = {
        'mean_intensity': [],
        'contrast': [],
        'sharpness': [],
        'edge_density': [],
        'lbp_uniformity': [],
        'red_hist_mean': [],
        'green_hist_mean': [],
        'blue_hist_mean': [],
        'red_hist_var': [],
        'green_hist_var': [],
        'blue_hist_var': []
    }

    # Iterate over batches in the loader
    for batch in loader:
        # Process 'real' images
        real_features = process_batch(batch['real'])
        # Process 'fake' images
        fake_features = process_batch(batch['fake'])

        # Accumulate real and fake features
        for feature_name in all_real_features:
            all_real_features[feature_name].extend(real_features[feature_name])
            all_fake_features[feature_name].extend(fake_features[feature_name])

    # Calculate consolidated statistics (mean, std, etc.) for both real and fake images
    consolidated_real_features = {}
    consolidated_fake_features = {}

    for feature_name, values in all_real_features.items():
        consolidated_real_features[feature_name + '_mean'] = np.mean(values)
        consolidated_real_features[feature_name + '_std'] = np.std(values)

    for feature_name, values in all_fake_features.items():
        consolidated_fake_features[feature_name + '_mean'] = np.mean(values)
        consolidated_fake_features[feature_name + '_std'] = np.std(values)

    return consolidated_real_features, consolidated_fake_features


def process_dataset(loader):
    # Initialize lists to store features for real and fake images
    all_real_features = {
        'mean_intensity': [],
        'contrast': [],
        'sharpness': [],
        'edge_density': [],
        'lbp_uniformity': [],
        'red_hist_mean': [],
        'green_hist_mean': [],
        'blue_hist_mean': [],
        'red_hist_var': [],
        'green_hist_var': [],
        'blue_hist_var': []
    }
    
    all_fake_features = {
        'mean_intensity': [],
        'contrast': [],
        'sharpness': [],
        'edge_density': [],
        'lbp_uniformity': [],
        'red_hist_mean': [],
        'green_hist_mean': [],
        'blue_hist_mean': [],
        'red_hist_var': [],
        'green_hist_var': [],
        'blue_hist_var': []
    }

    # Iterate over batches in the loader
    for batch in loader:
        # Process 'real' images
        real_features = process_batch(batch['real'])
        # Process 'fake' images
        fake_features = process_batch(batch['fake'])

        # Accumulate real and fake features for all images
        for feature_name in all_real_features:
            all_real_features[feature_name].extend(real_features[feature_name])
            all_fake_features[feature_name].extend(fake_features[feature_name])

    return all_real_features, all_fake_features

# Function to process each batch (each batch contains 'real' and 'fake' images)
def process_batch(batch):
    batch_features = {
        'mean_intensity': [],
        'contrast': [],
        'sharpness': [],
        'edge_density': [],
        'lbp_uniformity': [],
        'red_hist_mean': [],
        'green_hist_mean': [],
        'blue_hist_mean': [],
        'red_hist_var': [],
        'green_hist_var': [],
        'blue_hist_var': []
    }

    # Process each image in the batch
    for i in range(batch.size(0)):  # Loop through batch dimension (N)
        image = batch[i]  # Shape: (3, 224, 224)
        features = extract_features_tensor(image)  # Extract features for this image

        # Add the features of this image to the batch_features dictionary
        for feature_name, value in features.items():
            batch_features[feature_name].append(value)
    
    return batch_features

# Function to compute advanced metrics for one tensor (image)
def extract_features_tensor(tensor):
    features = {}
    
    # Convert tensor to NumPy array (H, W, C format)
    if tensor.dim() == 3:  # (C, H, W)
        tensor = tensor.permute(1, 2, 0)  # (H, W, C)
    image = tensor.numpy()  # Convert to NumPy array
    
    # Convert to grayscale
    if image.max() > 1:  # if pixel values are in [0, 255]
        image = image / 255.0  # normalize to [0, 1] range  
    gray = 0.2989 * image[:, :, 0] + 0.5870 * image[:, :, 1] + 0.1140 * image[:, :, 2]

    # Mean and variance of pixel intensities (contrast)
    features['mean_intensity'] = gray.mean()
    features['contrast'] = gray.var()
    
    # Sharpness using Laplacian variance
    gray_rescaled = gray * 255.0
    gray_rescaled = gray_rescaled.astype(np.float32)  # Convert to float32 for Laplacian

    grad_x = cv2.Sobel(gray_rescaled, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray_rescaled, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)

    sharpness = gradient_magnitude.var()
    features['sharpness'] = sharpness

    # Edge density using Canny edge detector
    edges = cv2.Canny((gray * 255).astype(np.uint8), 100, 200)  # Canny expects uint8
    features['edge_density'] = np.sum(edges > 0) / gray.size
    
    # Color histograms for each channel (normalized)
    for i, color in enumerate(['red', 'green', 'blue']):  # Assuming RGB channels
        channel = image[:, :, i]
        hist, _ = np.histogram(channel, bins=256, range=(0, 1))
        features[f'{color}_hist_mean'] = hist.mean() / channel.size
        features[f'{color}_hist_var'] = hist.var() / channel.size
    
    # Texture using Local Binary Pattern (LBP)
    lbp = local_binary_pattern(gray_rescaled, P=8, R=1, method='uniform')  # 8 neighbors, radius=1
    features['lbp_uniformity'] = np.sum(lbp == 0) / lbp.size  # Proportion of uniform patterns
    
    return features

print('FUNCTIONS TO PROCESS DATA COMPLETED')

###### PART 4: PROCESSING DATA AND CONDUCTING COMPARISONS! ######


# Assuming `train_loader` is your DataLoader for the training dataset
consolidated_real_features, consolidated_fake_features = process_dataset(train_loader)

print('FEATURES CONSOLIDATED HERE')


###### PART 5: ANALYZING DIFFERENCES! ######

# Function to perform t-test between real and fake features
from scipy import stats

def compare_features(real_features, fake_features):
    results = {}
    
    for feature_name in real_features.keys():
        real_values = real_features[feature_name]
        fake_values = fake_features[feature_name]

        real_mean = np.mean(real_values)
        fake_mean = np.mean(fake_values)

        # Perform a t-test to compare the means of real and fake features
        t_stat, p_value = stats.ttest_ind(real_values, fake_values, equal_var=False)
        
        # Store the results
        results[feature_name] = {
            'real_mean': real_mean,
            'fake_mean': fake_mean,
            't_stat': t_stat,
            'p_value': p_value
        }
    

        # break
    return results

# Function to print the comparison results
def print_comparison_results(comparison_results):
    for feature_name, result in comparison_results.items():
        real_mean = result['real_mean']
        fake_mean = result['fake_mean']
        t_stat = result['t_stat']
        p_value = result['p_value']
        
        # Print the means, t-statistic, and p-value
        print(f"Feature: {feature_name}")
        print(f"  Real Mean: {real_mean:.4f}")
        print(f"  Fake Mean: {fake_mean:.4f}")
        print(f"  t-statistic: {t_stat:.4f}")
        print(f"  p-value: {p_value:.4e}")
        
        # Interpret the result based on the p-value
        if p_value < 0.05:
            print(f"  Significant difference detected between real and fake for this feature.\n")
        else:
            print(f"  No significant difference detected between real and fake for this feature.\n")

# Perform the comparison between real and fake features
comparison_results = compare_features(consolidated_real_features, consolidated_fake_features)

# Print out the comparison results
print_comparison_results(comparison_results)
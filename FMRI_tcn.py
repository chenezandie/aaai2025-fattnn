#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import nibabel as nib
from nilearn import datasets

# Fetch sample dataset
haxby_dataset = datasets.fetch_haxby()
# fMRI data and anatomical image
fmri_filename = haxby_dataset.func[0]
# Load the fMRI data using nibabel
fmri_img_nii = nib.load(fmri_filename)
fmri_data = fmri_img_nii.get_fdata()

# Normalize the data
fmri_data_normalized = (fmri_data - np.mean(fmri_data)) / np.std(fmri_data)

class FMRISequenceDataset(Dataset):
    def __init__(self, fmri_volumes, num_input_slices, num_predict_slices):
        """
        Args:
            fmri_volumes (numpy.ndarray): 4D array of fMRI data (depth, height, width, time).
            num_input_slices (int): Number of slices in each input sequence.
            num_predict_slices (int): Number of slices to predict.
        """
        # Assuming that you only want to consider a specific depth slice across all times,
        # here using the middle slice for simplicity. Adjust index as needed.
        middle_slice_index = fmri_volumes.shape[0] // 2
        self.fmri_volumes = fmri_volumes[middle_slice_index]  # Shape now should be (64, 64, 1452)
        self.num_input_slices = num_input_slices
        self.num_predict_slices = num_predict_slices
        self.total_volumes = self.fmri_volumes.shape[2]  # The time dimension size

    def __len__(self):
        return self.total_volumes - self.num_input_slices - self.num_predict_slices + 1

    def __getitem__(self, idx):
        X = self.fmri_volumes[:, :, idx:idx + self.num_input_slices]  # (64, 64, num_input_slices)
        Y = self.fmri_volumes[:, :, idx + self.num_input_slices:idx + self.num_input_slices + self.num_predict_slices]  # (64, 64, num_predict_slices)
        return torch.tensor(X, dtype=torch.float32).permute(2, 0, 1), torch.tensor(Y, dtype=torch.float32).permute(2, 0, 1)

num_input_slices = 25
num_predict_slices = 1
dataset = FMRISequenceDataset(fmri_data_normalized, num_input_slices=num_input_slices, num_predict_slices=num_predict_slices)
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)


# In[4]:


import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import nibabel as nib
from nilearn import datasets

# Fetch sample dataset
haxby_dataset = datasets.fetch_haxby()
# fMRI data and anatomical image
fmri_filename = haxby_dataset.func[0]
# Load the fMRI data using nibabel
fmri_img_nii = nib.load(fmri_filename)
fmri_data = fmri_img_nii.get_fdata()


# In[5]:


fmri_data.shape


# In[6]:


from sklearn.model_selection import train_test_split

# Split indices to create training and validation sets
indices = list(range(len(dataset)))  # get indices of the whole dataset
train_indices, val_indices = train_test_split(indices, test_size=0.2, random_state=42)

# Create PyTorch data samplers and loaders:
from torch.utils.data import SubsetRandomSampler

train_sampler = SubsetRandomSampler(train_indices)
val_sampler = SubsetRandomSampler(val_indices)

train_loader = DataLoader(dataset, batch_size=1, sampler=train_sampler)
val_loader = DataLoader(dataset, batch_size=1, sampler=val_sampler)


# In[7]:


def loader_to_tensor(data_loader):
    x_batches = []
    y_batches = []
    
    # Iterate over the DataLoader
    for x_batch, y_batch in data_loader:
        x_batches.append(x_batch)
        y_batches.append(y_batch)
    
    # Concatenate all batches to form a single tensor
    X_tensor = torch.cat(x_batches, dim=0)  # Assuming dim=0 is your batch dimension
    Y_tensor = torch.cat(y_batches, dim=0)

    return X_tensor, Y_tensor

# Get tensors from loaders
X_train_tensor, Y_train_tensor = loader_to_tensor(train_loader)
X_test_tensor, Y_test_tensor = loader_to_tensor(val_loader)


# In[8]:


from torchvision import transforms

single_transform = transforms.Compose([
    transforms.ToPILImage(),
    #transforms.Resize((64, 64)),
    transforms.Grayscale(),  # Apply if needed
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

def transform_input(images):
    transformed_images = [single_transform(image[:, :, np.newaxis]).numpy() for image in images]  # Ensure image has 3 dimensions
    return np.stack(transformed_images, axis=0)  # Stack along new axis to create a multi-channel tensor

X_train_transformed = np.array([transform_input(images) for images in X_train_tensor.numpy()])
#Y_train_transformed = np.array([transform_input(images) for images in Y_train_tensor.numpy()])
X_test_transformed = np.array([transform_input(images) for images in X_test_tensor.numpy()])
#Y_test_transformed = np.array([transform_input(images) for images in Y_test_tensor.numpy()])
X_train_transformed = X_train_transformed.transpose(0,2,1,3,4)
X_test_transformed = X_test_transformed.transpose(0,2,1,3,4)


# In[9]:


X_train_tensor = torch.tensor(X_train_transformed, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test_transformed, dtype=torch.float32)



import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
import tensorly as tl
from tensorly.decomposition import tucker

class TemporalConvNet2D(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet2D, self).__init__()
        self.num_channels = num_channels
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            padding = (kernel_size - 1) * dilation_size
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [nn.Conv2d(in_channels, out_channels, kernel_size,
                                 stride=1, padding=padding, dilation=dilation_size),
                       nn.ReLU(),
                       nn.Dropout(dropout)]
        self.network = nn.Sequential(*layers)
        # Adding a final convolutional layer to adjust channel dimensions
        self.final_conv = nn.Conv2d(num_channels[-1], num_predict_slices, 1)  # Reduce to 1 channel
        self.adaptive_pool = nn.AdaptiveAvgPool2d((64, 64))  # Ensure spatial dimensions are 72x72

    def forward(self, x):
        batch_size, channels, seq_length, height, width = x.size()
        x = x.view(batch_size, channels * seq_length, height, width)  # Reshape input
        x = self.network(x)
        x = self.final_conv(x)  # Reduce channel dimensions to 1
        x = self.adaptive_pool(x)  # Ensure output dimensions are 72x72
        return x

def train_tcn_model(model, X_train, Y_train, epochs=100, lr=0.001):
    optimizer = Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()  # Use MSE for image prediction
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        output = model(X_train)
        print("Output shape:", output.shape)
        print("Target shape:", Y_train.shape)
        loss = criterion(output, Y_train)
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch+1}, Loss: {loss.item()}")

# Run training
torch.manual_seed(1)
np.random.seed(1)
        
num_channels = [50,50,50]
kernel_size = 3
dropout = 0.2
tcn_model = TemporalConvNet2D(num_inputs=ranks[0], num_channels=num_channels, kernel_size=kernel_size, dropout=dropout)
start_time = time.time()
train_tcn_model(tcn_model, X_train_tensor,Y_train_tensor, epochs=200, lr=0.001)
end_time = time.time()
# Calculate and print the duration
duration = end_time - start_time
print(f"Training completed in {duration:.2f} seconds.")


tcn_model.eval()
with torch.no_grad():
    predicted = tcn_model(X_test_tensor)
    mse_loss = nn.MSELoss()
    mse = mse_loss(predicted, Y_test_tensor)
    print(f"Test MSE: {mse.item()}")







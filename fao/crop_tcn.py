#!/usr/bin/env python
# coding: utf-8

# In[8]:


import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import Adam
df = pd.read_csv('crop1.csv')
df2 = pd.read_csv('crop2.csv')


# In[18]:


# Creating a pivot table
pivot_df = df.pivot_table(
    values='Value', 
    index=['Year', 'Area Code (M49)', 'Element Code', 'Item Code (CPC)'], 
    aggfunc=np.sum
).unstack(fill_value=0)  # Unstack to move the last level of the index to columns

pivot_df2 = df2.pivot_table(
    values='Value', 
    index=['Year', 'Area Code (M49)', 'Element Code', 'Item Code (CPC)'], 
    aggfunc=np.sum
).unstack(fill_value=0)  # Unstack to move the last level of the index to columns

tensor = pivot_df.values.reshape((62, len(df['Area Code (M49)'].unique()), len(df['Element Code'].unique()), len(df['Item Code (CPC)'].unique())))
tensor2 = pivot_df2.values.reshape((62, len(df2['Area Code (M49)'].unique()), len(df2['Element Code'].unique()), len(df2['Item Code (CPC)'].unique())))


# In[10]:


tensor_t = tensor.transpose(2,0,1,3)
tensor_t = tensor_t[:2]
tensor = tensor_t.transpose(1,2,0,3)


# In[11]:


tensor_t = tensor2.transpose(2,0,1,3)
tensor_t = tensor_t[:2]
tensor2 = tensor_t.transpose(1,2,0,3)


# In[12]:


X_train = np.array(tensor)
Y_train = np.array(tensor2)


# In[13]:


X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
Y_train_tensor = torch.tensor(Y_train, dtype=torch.float32)
Y_train_tensor = torch.where(Y_train_tensor < 0, torch.zeros_like(Y_train_tensor), Y_train_tensor)
X_train_tensor = (X_train_tensor+0.0001).log()
Y = (Y_train_tensor+0.0001).log()
X = X_train_tensor.unsqueeze(1)
#X_train_tensor = (X_train_tensor-X_train_tensor.min())/(X_train_tensor.max()-X_train_tensor.min())
#Y_train_tensor = (Y_train_tensor-Y_train_tensor.min())/(Y_train_tensor.max()-Y_train_tensor.min())
num_train = int(X.shape[0]*0.8)
X_test_tensor = X[num_train:]
X_train_tensor = X[:num_train]
Y_test_tensor = Y[num_train:]
Y_train_tensor = Y[:num_train]


# In[16]:


Y_mean_obs = torch.sum(Y, dim=(0))/Y.shape[0]
cores = []
for i in range(Y_test_tensor.shape[0]):
    abcd = Y_test_tensor[i]-Y_mean_obs
    cores.append(abcd)
denominator = []
for i in range(Y_test_tensor.shape[0]):
    abcde = cores[i]
    f_norm = torch.norm(abcde, p='fro')**2 
    denominator.append(f_norm)  
denominator = sum(denominator).numpy()


# In[24]:


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2, output_shape=None):
        super(TemporalConvNet, self).__init__()
        self.output_shape = output_shape
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            padding = (kernel_size - 1) * dilation_size
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [nn.Conv3d(in_channels, out_channels, kernel_size,
                                 stride=1, dilation=dilation_size, padding=padding),
                       nn.ReLU(),
                       nn.Dropout(dropout)]
        self.network = nn.Sequential(*layers)
        self.final_conv = nn.Conv3d(num_channels[-1], 1, kernel_size=1)  # Ensuring the final output has 1 channel
        self.adaptive_layer = nn.AdaptiveAvgPool3d((output_shape[0], output_shape[1], output_shape[2]))

    def forward(self, x):
        x = self.network(x)
        x = self.final_conv(x)
        x = self.adaptive_layer(x)
        x = x.squeeze(1)  # Remove the singleton channel dimension to match the target shape
        return x

def train_tcn_model(model, X_train, Y_train, epochs=100, lr=0.001, weight_decay=1e-4):
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.MSELoss()
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

import torch
import numpy as np
import time  # Import the time module

# Set random seeds for reproducibility
torch.manual_seed(123)
np.random.seed(123)
output_shape = (13, 2, 11)  # Expected output shape
tcn_model = TemporalConvNet(num_inputs=1, num_channels=[50, 50, 50], kernel_size=3, dropout=0.2, output_shape=output_shape)
start_time = time.time()
train_tcn_model(tcn_model, X_train_tensor, Y_train_tensor, epochs=300, lr=0.003)
end_time = time.time()

# Calculate and print the duration
duration = end_time - start_time
print(f"Training completed in {duration:.2f} seconds.")


# In[25]:


import torch
import torch.nn as nn

# Set model to evaluation mode
tcn_model.eval()

# No gradient computation to speed up calculations and reduce memory usage
with torch.no_grad():
    predicted = tcn_model(X_test_tensor)
    modified_predicted = torch.where(predicted < 0, torch.zeros_like(predicted), predicted)
    modified_Y_test_tensor = torch.where(Y_test_tensor < 0, torch.zeros_like(Y_test_tensor), Y_test_tensor)
    mse_loss = nn.MSELoss()
    mse = mse_loss(modified_predicted, modified_Y_test_tensor)
    print(f"Test MSE: {mse.item()}")


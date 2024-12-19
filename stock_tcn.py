#!/usr/bin/env python
# coding: utf-8

# In[31]:


import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import Adam
import tensorly as tl
df = pd.read_csv('crop.csv')
df2 = pd.read_csv('stock.csv')


# In[32]:


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
countries = [
    "Austria", "Bulgaria", "Canada", "China, mainland", "Croatia", "Cyprus", "Czechia",
    "Democratic Peoples Republic of Korea", "Denmark", "Estonia", "Finland", "France",
    "Germany", "Greece", "Hungary", "Ireland", "Italy", "Japan", "Latvia", "Lithuania",
    "Luxembourg", "Malta", "Mongolia", "Netherlands (Kingdom of the)", "Poland", "Portugal",
    "Republic of Korea", "Romania", "Slovakia", "Slovenia", "Spain", "Sweden", "United States of America"
]

european_countries = [
    "Austria", "Bulgaria", "Croatia", "Cyprus", "Czechia", "Denmark", "Estonia",
    "Finland", "France", "Germany", "Greece", "Hungary", "Ireland", "Italy",
    "Latvia", "Lithuania", "Luxembourg", "Malta", "Netherlands (Kingdom of the)",
    "Poland", "Portugal", "Romania", "Slovakia", "Slovenia", "Spain", "Sweden"
]

indices = [countries.index(country) for country in european_countries if country in countries]
tensor = tensor[:, indices, :, :]
tensor2 = tensor2[:, indices, :, :]


# In[33]:


X_train = np.array(tensor2)
Y_train = np.array(tensor)
X_train = np.log(X_train+0.0001)
Y_train = np.log(Y_train+0.0001)


# In[34]:


import torch
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
Y = torch.tensor(Y_train, dtype=torch.float32)
X = X_train_tensor.unsqueeze(1)
num_train = int(X.shape[0]*0.8)
X_test_tensor = X[num_train:]
X_train_tensor = X[:num_train]
Y_test_tensor = Y[num_train:]
Y_train_tensor = Y[:num_train]


# In[35]:


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

def train_tcn_model(model, X_train, Y_train, epochs=100, lr=0.001):
    optimizer = Adam(model.parameters(), lr=lr)
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
torch.manual_seed(1)
np.random.seed(1)

# Specify the output shape
output_shape = (26, 3, 11)  # Adjusted for the European countries tensor

# Initialize your model
tcn_model = TemporalConvNet(num_inputs=X_train_tensor.shape[1], num_channels=[25, 25, 25, 25, 25], kernel_size=2, dropout=0.2, output_shape=output_shape)

# Record the start time
start_time = time.time()

# Train the model
train_tcn_model(tcn_model, X_train_tensor, Y_train_tensor, epochs=200, lr=0.002)

# Record the end time
end_time = time.time()

# Calculate and print the duration
duration = end_time - start_time
print(f"Training completed in {duration:.2f} seconds.")


# In[39]:


import torch
import torch.nn as nn

# Assuming tcn_model, X_test_tensor, and Y_test_tensor are defined and properly set up

# Set model to evaluation mode
tcn_model.eval()

# No gradient computation to speed up calculations and reduce memory usage
with torch.no_grad():
    predicted = tcn_model(X_test_tensor)
    
    # Change all entries in predicted that are smaller than -1 to 0
    modified_predicted = torch.where(predicted < 0, torch.zeros_like(predicted), predicted)
    modified_Y_test_tensor = torch.where(Y_test_tensor < 0, torch.zeros_like(Y_test_tensor), Y_test_tensor)

    # Calculate MSE using the modified predictions
    mse_loss = nn.MSELoss()
    mse = mse_loss(modified_predicted, modified_Y_test_tensor)

    print(f"Test MSE: {mse.item()}")



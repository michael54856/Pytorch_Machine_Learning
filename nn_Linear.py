import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset

# Input (temp, rainfall, humidity)
inputs = np.array([[73, 67, 43], 
                   [91, 88, 64], 
                   [87, 134, 58], 
                   [102, 43, 37], 
                   [69, 96, 70], 
                   [74, 66, 43], 
                   [91, 87, 65], 
                   [88, 134, 59], 
                   [101, 44, 37], 
                   [68, 96, 71], 
                   [73, 66, 44], 
                   [92, 87, 64], 
                   [87, 135, 57], 
                   [103, 43, 36], 
                   [68, 97, 70]], 
                  dtype='float32')

# Targets (apples, oranges)
targets = np.array([[56, 70], 
                    [81, 101], 
                    [119, 133], 
                    [22, 37], 
                    [103, 119],
                    [57, 69], 
                    [80, 102], 
                    [118, 132], 
                    [21, 38], 
                    [104, 118], 
                    [57, 69], 
                    [82, 100], 
                    [118, 134], 
                    [20, 38], 
                    [102, 120]], 
                   dtype='float32')

inputs = torch.from_numpy(inputs)
targets = torch.from_numpy(targets)

train_ds = TensorDataset(inputs, targets)

batch_size = 5
train_dl = DataLoader(train_ds,batch_size,shuffle=True)

model = nn.Linear(3, 2) #linear的模型,3個inputs,2個outputs

optimizer = torch.optim.SGD(model.parameters(), lr=0.00001)

lossFunction = nn.MSELoss()

for epoch in range(500): #200 times training loop
    
    # Train with batches of data
    for xb,yb in train_dl: #分割成幾個比較小的資料去執行
        
        # 1. Generate predictions
        pred = model(xb)
        
        # 2. Calculate loss
        loss = lossFunction(pred, yb)
        
        # 3. Compute gradients
        loss.backward()
        
        # 4. Update parameters using gradients
        optimizer.step()
        
        # 5. Reset the gradients to zero
        optimizer.zero_grad()
    # Print the progress
    if (epoch+1) % 10 == 0:
        print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, 200, loss.item()))

pred = model(inputs)
print(pred)
print(targets)

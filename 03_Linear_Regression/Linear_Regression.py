import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets

#(0) prepare data

X_numpy , Y_numpy = datasets.make_regression(n_samples=100, n_features=1, noise=20, random_state=1) #建立樣本數100,只有一種特徵

X = torch.from_numpy(X_numpy.astype(np.float32)) #[100][1]
Y = torch.from_numpy(Y_numpy.astype(np.float32)) #[100]

Y = Y.view(Y.shape[0], 1) #reshape Y to [100][1]

n_samples , n_features = X.shape #n_samples為共有幾組要比對,n_features為每一組資料裡有多少資料

#(1)model

input_size = n_features
output_size = 1

model = nn.Linear(input_size,output_size)

#(2)loss and optimizer
lossFunction = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

#(3)training loop
trainingTimes = 100

for epoch in range(trainingTimes) :

    #forward_Pass 
    pred_y = model(X)
    loss = lossFunction(pred_y,Y)

    #backward_Pass
    loss.backward()

    #update weights
    optimizer.step()

    #zero gradient
    optimizer.zero_grad()

    print(f"epoch: {epoch+1} , loss = {loss.item():.4f}")

#(4)plot

#detach截斷反向傳播
predicted = model(X).detach().numpy()
plt.plot(X_numpy,Y_numpy, 'ro') #點
plt.plot(X_numpy,predicted, 'b') #線段
plt.show()

#因為我們是使用MSELoss()來當作loss function,所以可以預期loss會越來越小,也就是線性回歸



    








    







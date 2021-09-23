import torch
import torchvision
from torchvision.datasets import MNIST
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from torch.utils.data import random_split
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F

input_size = 28*28
num_classes = 10 #輸出有10種可能

class MnistModel(nn.Module): #繼承pytorch的模型
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(input_size, num_classes)
        
    def forward(self, xb): #定義forwardPass
        xb = xb.reshape(-1, 784) #reshape要傳入的參數,-1會讓python自行調整
        out = self.linear(xb)
        return out

# training dataset
dataset = MNIST(root='data/',  train=True, transform=transforms.ToTensor())

#dataset[i][0] = imageTensor , dataset[i][1] = label

#圖片轉為tensor後為1x28x28,代表1個channel且像素為28x28

train_ds, val_ds = random_split(dataset, [50000, 10000]) #分成50000訓練,10000驗證

train_loader = DataLoader(train_ds, 128, shuffle=True) #128為1組
val_loader = DataLoader(val_ds, 128) #128為1組


model = MnistModel() #現在model沒有weight跟bias,取而代之可以用parameter回傳這個list
model.load_state_dict(torch.load("mnist-logistic.pth"))
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
loss_fn = F.cross_entropy

#model.weight.shape = [10,784] , 代表有10個output,每個output有784個weight
#model.bias = 10 , 有10個output,所以有10個bias

totalEpochs = 0

for epoch in range(totalEpochs):
    averageLoss = 0
    totalIterations = 0
    for images, labels in train_loader:
        outputs = model(images) #outputs的大小為128x10,代表有128個測資,每個測資有10個輸出 , 

        #probs = F.softmax(outputs, dim=1) 每個outputRow都應用softmax,softmax會將數值轉成機率,並讓每個row sum為1
        #max_probs, preds = torch.max(probs, dim=1) 取得每列的最大元素,並回傳他的index

        #使用 cross_entropy的時候 會計算 [ln機率] 當機率很高的時候數值會較低,當機率很低的時候數值會較高,這可以當作我們的lossFunction,我們只求最關鍵的那個值而已
        
        #cross_entropy的步驟是先對data使用softMax,再針對我真正的值去取出loss
        loss = loss_fn(outputs, labels) #不用softMax , cross_entropy會自動幫我們用
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        averageLoss += loss.item()
        totalIterations += 1

    print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, totalEpochs, averageLoss/totalIterations))

counter = 0
for images, labels in val_loader:

    outputs = model(images)
    probs = F.softmax(outputs, dim=1) #每個outputRow都應用softmax,softmax會將數值轉成機率,並讓每個row sum為1
    max_probs, preds = torch.max(probs, dim=1) #取得每列的最大元素,並回傳他的index

    counter += torch.sum(preds == labels).item()

# torch.save(model.state_dict(), "mnist-logistic.pth")
print(f"total accepted:{counter} , accuracy:{counter/10000}")


def getNumber(img):
    outputs = model(img)
    probs = F.softmax(outputs, dim=1) #每個outputRow都應用softmax,softmax會將數值轉成機率,並讓每個row sum為1
    max_probs, preds = torch.max(probs, dim=1) #取得每列的最大元素,並回傳他的index
    return preds

        
#最後測試
myNumber = 2734 #1~10000
test_dataset = MNIST(root='data/', train=False,transform=transforms.ToTensor())#用額外的一些資料來測試
img, label = test_dataset[myNumber]
plt.imshow(img[0], cmap='gray')
plt.show()
print('Label:', label, ', Predicted:', getNumber(img).item())








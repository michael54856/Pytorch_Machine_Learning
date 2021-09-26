import torch
import torchvision
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
from torchvision.datasets import MNIST
from torchvision.transforms import transforms
from torchvision.utils import make_grid
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import random_split

def getNumber(img):
        outputs = model(img)
        probs = F.softmax(outputs, dim=1) #每個outputRow都應用softmax,softmax會將數值轉成機率,並讓每個row sum為1
        max_probs, preds = torch.max(probs, dim=1) #取得每列的最大元素,並回傳他的index
        return preds


def get_default_device():
    """Pick GPU if available, else CPU"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')


class MnistModel(nn.Module):
    # Feedfoward neural network with 1 hidden layer
    def __init__(self, in_size, hidden_size, out_size):
        super().__init__()
        # hidden layer
        self.linear1 = nn.Linear(in_size, hidden_size) #第一層
        # output layer
        self.linear2 = nn.Linear(hidden_size, out_size) #第二層
        
    def forward(self, xb):
        # 把資料弄成(x,784),用xb.size(0)可以取得第一層的維度,再用-1讓python自己去偵測
        xb = xb.view(xb.size(0), -1)
        # 使用第一層的linear,input:[x,784],output:[x,32] , x代表一次幾組
        out = self.linear1(xb)
        # 套用RELU , max(0,input_)
        out = F.relu(out)
        # Get predictions using output layer
        out = self.linear2(out)
        return out #最終回傳0~9的各數值

if __name__=='__main__':
    # training dataset
    dataset = MNIST(root='data/',  train=True, transform=transforms.ToTensor())

    device = get_default_device()

    train_size = 50000
    val_size = 10000

    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    batch_size = 128

    input_size = 784
    num_classes = 10

    train_loader = DataLoader(train_ds, batch_size, shuffle=True, num_workers=4, pin_memory=True) #使用4個進程,而pin_memory可以在gpu中加快 , especially true for 3D data or very large batch sizes
    val_loader = DataLoader(val_ds, batch_size*2, num_workers=4, pin_memory=True) #使用4個進程,而pin_memory可以在gpu中加快 , especially true for 3D data or very large batch sizes

    model = MnistModel(input_size, 32, num_classes) #hidden size 為 32
    model.load_state_dict(torch.load("mnist-logistic-gpu.pth"))

    totalEpochs = 0
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    if totalEpochs > 0 :
        myTraining_loader = [[images.to(device, non_blocking=True),labels.to(device, non_blocking=True)] for images, labels in train_loader] #把train_loader轉換成list,並修改device,存成一個list


    for epoch in range(totalEpochs):
        averageLoss = 0
        totalIterations = 0
        for images, labels in myTraining_loader:

            outputs = model(images)

            loss = F.cross_entropy(outputs, labels) #不用softMax , cross_entropy會自動幫我們用
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            averageLoss += loss.item()
            totalIterations += 1
            
        print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, totalEpochs, averageLoss/totalIterations))

    model = model.to("cpu")
    counter = 0
    for images, labels in val_loader:
        outputs = model(images)
        probs = F.softmax(outputs, dim=1) #每個outputRow都應用softmax,softmax會將數值轉成機率,並讓每個row sum為1
        max_probs, preds = torch.max(probs, dim=1) #取得每列的最大元素,並回傳他的index

        counter += torch.sum(preds == labels).item()

    #torch.save(model.state_dict(), "mnist-logistic-gpu.pth")
    print(f"total accepted:{counter} , accuracy:{counter/10000}")

            
    #最後測試
    myNumber = 2734 #1~10000
    test_dataset = MNIST(root='data/', train=False,transform=transforms.ToTensor())#用額外的一些資料來測試
    img, label = test_dataset[myNumber]
    plt.imshow(img[0], cmap='gray')
    plt.show()
    print('Label:', label, ', Predicted:', getNumber(img).item())
from numpy import float32, true_divide
from numpy.core.fromnumeric import mean
import torch

X = torch.tensor([1,2,3,4], dtype=torch.float32)
Y = torch.tensor([3,6,9,12], dtype=torch.float32)

w = torch.tensor(0.0, dtype=torch.float32, requires_grad=True) # w是我們要訓練的東西,所以需要requires_grad

def forward(x) :
    return w*x

def loss(y, trainingY) :
    return ((y-trainingY)**2).mean()

print(f"訓練之前f(5) = {forward(5):.3f}") #一開始輸出0,因為預設w為0,所以5*w = 0

learningRate = 0.01
iterateTimes = 100

for epoch in range(iterateTimes) :

    #forward pass
    traingY = forward(X)

    #loss
    l = loss(Y,traingY)

    l.backward() #呼叫最終loss的backward function

    #update weight
    with torch.no_grad() :
        w -= learningRate * w.grad

    #zero gradient
    w.grad.zero_()

    print(f"epoch : {epoch+1}: w = {w:.3f}, loss = {l:.8f}")

print(f"訓練之後f(5) = {forward(5):.3f}") #輸出最後w*5的結果
    


    







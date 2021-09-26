import torch
import torch.nn as nn

X = torch.tensor([[1],[2],[3],[4]], dtype=torch.float32)
Y = torch.tensor([[3],[6],[9],[12]], dtype=torch.float32)

n_samples,n_features = X.shape 

# n_samples代表有幾組資料要去比對 row
# n_features輸入的節點量 column

input_size = n_features
output_size = n_features

model = nn.Linear(input_size,output_size) #一個輸入,一個輸出

X_test = torch.tensor([5], dtype=torch.float32)

# model(X_test).item() 代表呼叫f(X_test) ,需要item是因為不需要整個tensor
print(f"訓練之前f(5) = {model(X_test).item():.3f}") #一開始輸出0,因為預設w為0,所以5*w = 0

learningRate = 0.01
iterateTimes = 1000

loss = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learningRate) #把w帶換成model.parameters()

for epoch in range(iterateTimes) :

    #由輸入X,與預期Y,裡面的weight由pytorch處理

    #forward pass
    traingY = model(X)

    #loss
    l = loss(Y,traingY)

    l.backward() #呼叫最終loss的backward function,可求出requires_grad=True的weight

    #update weight
    optimizer.step() #自動更新w的數值 , 會自動更新optimizer裡頭的參數 (w -= learningRate * w.grad)
       
    #zero gradient
    optimizer.zero_grad() # optimizer裡的參數,將其grad設為0

    [w,b] = model.parameters()
    print(f"epoch : {epoch+1}: w = {w[0][0].item():.3f}, loss = {l:.8f}")

print(f"訓練之後f(5) = {model(X_test).item():.3f}") #輸出最後w*5的結果
    


    







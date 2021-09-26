from scipy.sparse import data
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from torch.nn.modules import linear

#(0) prepare data

bc = datasets.load_breast_cancer() #癌症的資料集,分成malignant(惡性)，benign(良性)兩類
X,Y = bc.data, bc.target

# X :  [569][30]
# Y :  [569]

n_samples, n_features = X.shape #n_samples = 569筆資料, n_features = 每筆資料30維度

X_train, X_test ,Y_train, Y_test = train_test_split(X,Y, test_size=0.2, random_state=1234)#對於X,Y把20%用來比對,剩下80%用來訓練,random_state只是seed

#scale
sc = StandardScaler() #標準化,平均為0,方差為1,標準化並不會破壞資料之間的關係

#fit_transform會用在 trainging data
#transform會用在 test data
#對剩餘的數據（testData）使用均值、變量、最大使用等指標進行轉換（testData），從而保證訓練、測試處理方式相同

X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#train是我們訓練時用的資料,test是訓練完後用來對照訓練結果,會把X_test丟進去,看跟Y_test差多少

X_train = torch.from_numpy(X_train.astype(np.float32))
X_test = torch.from_numpy(X_test.astype(np.float32))
Y_train = torch.from_numpy(Y_train.astype(np.float32))
Y_test = torch.from_numpy(Y_test.astype(np.float32))

Y_train = Y_train.view(Y_train.shape[0], 1) #change row vector to column vector
Y_test = Y_test.view(Y_test.shape[0], 1) #change row vector to column vector

#(1) model
# f = wx + b , sigmod at the end
class LogisticRegression(nn.Module): #固定以這個形式來定義model

    def __init__(self, n_input_features):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(n_input_features, 1) #設定n_input_features個input, 1個output , linear是class的變數
    
    def forward(self, x):
        y_pred = torch.sigmoid(self.linear(x))#return 0~1之間的數
        return y_pred

model = LogisticRegression(n_features)

#(2) loss and optimizer

criterion = nn.BCELoss() #應用在二分類問題
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)


#(3) training loop
trainTimes = 100

for epoch in range(trainTimes) :

    #forward pass
    y_predicted = model(X_train)
    loss = criterion(y_predicted, Y_train)

    #backward pass
    loss.backward()

    #update weight
    optimizer.step()

    #empty gradient
    optimizer.zero_grad()

    print(f"epoch= {epoch+1} , loss={loss.item():.4f}")

with torch.no_grad() :
    y_predicted = model(X_test)
    y_predicted_cls = y_predicted.round() #因為最後的結果只有0跟1,所以四捨五入
    accuracy = y_predicted_cls.eq(Y_test).sum() / float(Y_test.shape[0])
    print(f"accuracy = {accuracy}")








    







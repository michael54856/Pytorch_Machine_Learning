from __future__ import unicode_literals, print_function, division
from io import open
import glob
import os
import unicodedata
import string
import torch
import torch.nn as nn
import random
import matplotlib.pyplot as plt

#=======================================資料預處理=====================================================

def findFiles(path): return glob.glob(path) #回傳路徑

def unicodeToAscii(s): #把Unicode轉成ASCII
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in all_letters
    )

def readLines(filename): #開啟傳過來的檔名,並把名子做分割,弄成list
    lines = open(filename, encoding='utf-8').read().strip().split('\n')
    return [unicodeToAscii(line) for line in lines]

category_lines = {} #每個國家會有對應的name list
all_categories = [] #紀錄所有國家

all_letters = string.ascii_letters + " .,;'"
n_letters = len(all_letters)


for filename in findFiles('./names/*.txt') : #開啟每個檔案,並把名子弄成list丟進dicrionary中
    category = os.path.splitext(os.path.basename(filename))[0]
    all_categories.append(category)
    lines = readLines(filename)
    category_lines[category] = lines

n_categories = len(all_categories)
    
#=================================================================================================

#我們需要把每個字元轉成tensor,這裡使用的是<one-hot vector>

# *output的數量=hidden_size* (重要!!!!)
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers,output_size):
        super(RNN, self).__init__()

        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(input_size,hidden_size, num_layers,batch_first=True) #feature為input_size(也就是所有的字元數量),hidden_size=128
        #nn.RNN會回傳2個東西,一個是output,一個是hiddenState(長度都是根據hiddenSize)
        self.fc = nn.Linear(hidden_size,output_size) #讓最後一個時間序列 對到calssify的tensor
        self.myLogSoftMax = nn.LogSoftmax(dim=1)

    def forward(self, input):
        init_hidden = torch.zeros(self.num_layers, input.size(0),self.hidden_size)
        output,next_hidden = self.rnn(input,init_hidden)
        output = self.fc(output)
        output = output[:,-1] #只需要取得最後一列
        output = self.myLogSoftMax(output)
        return output #回傳的output與hidden是tensor


n_hidden = 128
criterion = nn.NLLLoss()
layers = 1
model = RNN(n_letters, n_hidden, layers,n_categories) #inputSize為幾個letter
optimizer = torch.optim.SGD(model.parameters(), lr=0.005)

def letterToTensor(letter): #把字元轉成tensor
    tensor = torch.zeros(1, n_letters) #先建立一個 1xn 的tensor,全部設為0
    tensor[0][all_letters.find(letter)] = 1 #使用one-hot,設為!
    return tensor

def randomChoice(l): #返回list中的隨機一個元素
    return l[random.randint(0, len(l) - 1)]

def lineToTensor(line): #把名子轉成tensor
    myTensor = torch.zeros(len(line), 1, n_letters) #看名子的長度是多少,tensor就要幾列
    for index, letter in enumerate(line):
        myTensor[index][0][all_letters.find(letter)] = 1
    return myTensor

def categoryFromOutput(output): #給予output tensor , 返回國家
    top_n, top_i = output.topk(1) #top_n是取到的值,top_i是他的index,回傳為tensor
    return all_categories[top_i[0].item()] #回傳國家

def randomTrainingExample(): #給予隨機一筆資料
    category = randomChoice(all_categories) #取得隨機一個國家
    name = randomChoice(category_lines[category]) #取得該國家人名的隨機一個
    category_tensor = torch.tensor([all_categories.index(category)], dtype=torch.long) #取得該國家的index(型態為tensor)
    name_tensor = lineToTensor(name) #人名轉換成tensor
    return category, name, category_tensor, name_tensor

def train(category_tensor, line_tensor): #傳入國家的tensor形式 以及 人名的tensor形式

    #訓練的時候會把全部字元forward後,並回傳最後的output
    line_tensor = torch.reshape(line_tensor, (1,line_tensor.size(0),-1))
    model.zero_grad()

    output = model(line_tensor)

    loss = criterion(output, category_tensor) #比較output 跟 category_tensor之間的關係,會返回一個數字(1x1 tensor)
    loss.backward()
    optimizer.step()
    return output, loss.item()


currentLoss = 0
all_losses = []
n_iters = 100000
plot_steps, print_steps = 1000,5000

for i in range(n_iters):
    category, line, category_tensor, line_tensor = randomTrainingExample()

    output, loss = train(category_tensor,line_tensor)

    currentLoss += loss

    if (i+1) % plot_steps == 0 : #繪製趨勢圖
        all_losses.append(currentLoss/plot_steps)
        currentLoss = 0

    if (i+1) % print_steps == 0:
        guess = categoryFromOutput(output)

        correctOrNot = "X"
        if guess == category:
            correctOrNot = "O"
        
        print(f"{i+1} / {n_iters} | {category} / {guess} : {correctOrNot}  loss:{loss:.4f}")


plt.figure()
plt.plot(all_losses)
plt.show()

def predict(line):
    with torch.no_grad():
        line_tensor = lineToTensor(line)
        line_tensor = torch.reshape(line_tensor, (1,line_tensor.size(0),-1))
        output = model(line_tensor)
        print(f"country : {categoryFromOutput(output)}")

while True:
    inputName = input("Input:")
    if inputName == "quit":
        break
    predict(inputName)




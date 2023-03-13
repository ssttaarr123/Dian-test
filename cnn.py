import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
import pandas as pd
from torchvision import datasets,transforms
from torch.utils.data import DataLoader

#超参数
batch_size = 64
kernel_size = 3   #卷积核
learning_rate = 1e-3
momentum = 0.9  #动量，缓解局部最优


#导入训练集和测试集
train_dataset = datasets.MNIST(root='./mnist_data/',
                               train=True,
                               transform=transforms.Compose([
                                   transforms.ToTensor(),
                                   transforms.Normalize(
                                       (0.1307,),(0.3081,))
                               ]),
                               download=True)

test_dataset = datasets.MNIST(root='./mnist_data/',
                              train=False,
                              download=True,
                              transform=transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize(
                                    (0.1307,),(0.3081,)) 
                              ])
                              )

train_loader = DataLoader(train_dataset,batch_size = batch_size,shuffle=True,drop_last = True)
test_loader = DataLoader(test_dataset,batch_size,True,drop_last = True)

# eee = iter(train_loader)
# a,b = next(eee)
# print(a.shape)
# [64, 1, 28, 28]

class MyCNN_net(nn.Module):
    def __init__(self) :
        super(MyCNN_net,self).__init__()
        self.mymetrix = nn.Sequential(
            nn.Conv2d(1,6,kernel_size=kernel_size,stride=1,padding=0),
            nn.MaxPool2d(kernel_size = 2,stride=2,padding=0),
            nn.Conv2d(6,10,kernel_size=kernel_size,stride=1,padding=0),
            nn.MaxPool2d(kernel_size=2,stride=2,padding=0),
        )
        self.mylinear = nn.Sequential(
            nn.Linear(250,120),
            nn.ReLU(),
            nn.Linear(120,60),
            nn.ReLU(),
            nn.Linear(60,10)
        )

        # tryy = torch.rand(64,1,28,28)
        # anss = self.mymetrix(tryy)
        # print(anss.shape)
        # [64, 10, 5, 5]

    def forward(self,inputt):
        inputt = self.mymetrix(inputt)
        inputt = inputt.view(batch_size,-1)
        out = self.mylinear(inputt)
        return out

mynet = MyCNN_net()

criterion = torch.nn.CrossEntropyLoss()  
optimizer = torch.optim.SGD(mynet.parameters(), lr=learning_rate, momentum=momentum)  


def train(epoch):
    sum_loss = 0.0
    num_total = 0
    for index,(data , Target) in enumerate(train_loader):
        num_total = index + 1
        optimizer.zero_grad()
        digist = mynet(data)
        loss = criterion(digist,Target)
        loss.backward()
        optimizer.step()
        sum_loss += loss.item()
    print('Times :%d' %(epoch+1))
    print('     average training loss : %.3f' %(sum_loss/num_total))

def test():
    num_ac = 0
    num_total = 0
    with torch.no_grad():
        for data, ans in test_loader:
            out = mynet(data)
            _, pred = torch.max(out.data,dim=1)
            num_total += ans.size(0)
            num_ac += (pred ==  ans).sum().item()
    ac_rate = num_ac / num_total
    print('     Accuracy on the test set : %.3f' %(ac_rate))
    return  ac_rate


if __name__ == '__main__':
    for epoch in range(15):
        train(epoch)
        test()

'''Times :1
     average training loss : 1.352
     Accuracy on the test set : 0.907
Times :2
     average training loss : 0.245
     Accuracy on the test set : 0.947
Times :3
     average training loss : 0.164
     Accuracy on the test set : 0.959
Times :4
     average training loss : 0.127
     Accuracy on the test set : 0.966
Times :5
     average training loss : 0.106
     Accuracy on the test set : 0.974
Times :6
     average training loss : 0.091
     Accuracy on the test set : 0.968
Times :7
     average training loss : 0.081
     Accuracy on the test set : 0.972
Times :8
     average training loss : 0.072
     Accuracy on the test set : 0.977
Times :9
     average training loss : 0.066
     Accuracy on the test set : 0.979
Times :10
     average training loss : 0.059
     Accuracy on the test set : 0.978
Times :11
     average training loss : 0.054
     Accuracy on the test set : 0.981
Times :12
     average training loss : 0.051
     Accuracy on the test set : 0.982
Times :13
     average training loss : 0.048
     Accuracy on the test set : 0.982
Times :14
     average training loss : 0.044
     Accuracy on the test set : 0.981
Times :15
     average training loss : 0.041
     Accuracy on the test set : 0.981'''


         







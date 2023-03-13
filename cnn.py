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

train_loader = DataLoader(train_dataset,batch_size = batch_size,shuffle=True)
test_loader = DataLoader(test_dataset,batch_size,True)

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
        inputt = inputt.view(batch_size,250)
        out = self.mylinear(inputt)
        return out

mynet = MyCNN_net()

criterion = torch.nn.CrossEntropyLoss()  
optimizer = torch.optim.SGD(mynet.parameters(), lr=learning_rate, momentum=momentum)  






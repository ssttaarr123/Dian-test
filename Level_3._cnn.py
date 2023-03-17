from nn.function import *
import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
import pandas as pd
from torchvision import datasets,transforms
from torch.utils.data import DataLoader
torch.manual_seed(123)
#超参数
batch_size = 64
kernel_size = 3   #卷积核
learning_rate = 1e-3
momentum = 0.9  #动量，缓解局部最优
torch.set_grad_enabled(False)

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
#对relu的求导数
def relu_back(input,out_grad):
     batch,len = input.shape
     in_grad = torch.zeros(batch,len)
     for i in range(batch):
          for j in range(len):
               if input[i,j] > 0:
                    in_grad[i,j] = out_grad[i,j]
               else:
                    in_grad[i,j] = 0
     return in_grad

class MyCNN_net(nn.Module):
    def __init__(self) :
        super(MyCNN_net,self).__init__()
        #卷积部分
        self.mymetrix1 = Conv2d(1,2,kernel_size=kernel_size,stride=1,padding=0)
        self.mymetrix2 = Conv2d(2,1,kernel_size=kernel_size,stride=1,padding=0)
        #线性部分
        self.mylinear1 = Linear(576,20)
        self.relu1 = nn.ReLU()
        self.mylinear3 = Linear(20,10)
        

        # tryy = torch.rand(64,1,28,28)
        # anss = self.mymetrix(tryy)
        # print(anss.shape)
        # [64, 10, 5, 5]

    def forward(self,inputt):
        self.inputt = inputt
        self.x1 = self.mymetrix1(inputt)
        self.x2 = self.mymetrix2(self.x1)
        
        self.x3 = self.x2.view(batch_size,-1)
        
        self.x6 = self.mylinear1.forward(self.x3)
        self.x7 = self.relu1(self.x6)
 
        out = self.mylinear3.forward(self.x7)
        return out
    def backward(self,out_grad):
 
        out_grad = self.mylinear3.backward(out_grad)
        #temp = torch.zeros_like(self.x8)

        out_grad = relu_back(self.x6,out_grad)
        out_grad = self.mylinear1.backward(out_grad)
        
        out_grad = out_grad.view(batch_size,1,24,24)

        out_grad = self.mymetrix2.backward(out_grad)
        out_grad = self.mymetrix1.backward(out_grad)

 

        
        
        
mynet = MyCNN_net()
criterion = CrossEntropyLoss()  
optimizer = torch.optim.SGD(mynet.parameters(), lr=learning_rate, momentum=momentum)  


def train(epoch):
    sum_loss = 0.0
    num_total = 0
    for index,(data , Target) in enumerate(train_loader):
        num_total = index + 1
        optimizer.zero_grad()
        
        digist = mynet.forward(data)
        loss = criterion(digist,Target)
        mynet.backward(criterion.backward())
        optimizer.step()
        sum_loss += loss.item()
        print(index)
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


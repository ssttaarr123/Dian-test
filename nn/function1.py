
from typing import Union
from torch import Tensor, zeros
from torch.nn.modules import Module
from torch.nn.parameter import Parameter
import torch
import numpy as np
from torch.nn.modules.conv import _ConvNd
# from cnnbase import ConvBase
from torch.nn.modules.utils import _pair
from torch.nn.common_types import _size_2_t
def im2coll(input,kernal_size):
    batch_size,cha_in,h_in,w_in = input.shape
    h_out = h_in - kernal_size+1
    w_out = w_in - kernal_size+1
    col = torch.zeros(batch_size*h_out*w_out, cha_in*kernal_size*kernal_size)
    for i in range(h_out):
        h_start = i
        for j in range(w_out):
            w_start = j
            col[i*h_out+j::h_out*w_out,:] = input[:,:, h_start:h_start+kernal_size , w_start:w_start+kernal_size ].reshape(batch_size,-1)
    return col

def coll2im(col,kernal_size,batch_size,h_out,w_out):
    cha_in = col.shape[1] // (kernal_size*kernal_size)


    input = torch.zeros(batch_size,cha_in,h_out+kernal_size-1,w_out+kernal_size-1)
    temp = col.reshape(batch_size,w_out*h_out,-1)
    for i in range(h_out):
        for j in range(w_out):
            input[:,:,i:i+kernal_size,j:j+kernal_size] += temp[:,i*h_out+j,:].reshape(batch_size,cha_in,kernal_size,kernal_size)
    return input



class Conv2d(_ConvNd):
   
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_2_t,
        stride: _size_2_t = 1,
        padding: Union[str, _size_2_t] = 0,
        dilation: _size_2_t = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = 'zeros',  # TODO: refine this type
        device=None,
        dtype=None
    ):
        factory_kwargs = {'device': device, 'dtype': dtype}
        kernel_size_ = _pair(kernel_size)
        stride_ = _pair(stride)
        padding_ = padding if isinstance(padding, str) else _pair(padding)
        dilation_ = _pair(dilation)
        super(Conv2d, self).__init__(
            in_channels, out_channels, kernel_size_, stride_, padding_, dilation_,
            False, _pair(0), groups, bias, padding_mode, **factory_kwargs)
    
   

    def conv2d(self, input, kernel, bias = 0, stride=1, padding=0):
        self.stride = stride
        self.input = input
        batch_size, cha_in,h_in,w_in = input.shape
        cha_out,cha_in,kernal_size,_ = kernel.shape
        h_out = (h_in - kernal_size + 2*padding ) // stride + 1
        w_out = (w_in - kernal_size + 2*padding) // stride + 1
        col = im2coll(input,kernal_size)
        self.col_input = col
        out = torch.mm(col,kernel.reshape(cha_out,-1).t())
        out = out+bias
        out = out.reshape(batch_size,-1,cha_out)
        out = out.transpose(1,2).reshape(batch_size,cha_out,h_out,w_out)
        self.output = out
        return self.output
    
    def forward(self, input: Tensor):
        weight = self.weight
        bias = self.bias
        return self.conv2d(input, weight, bias)
    
    def backward(self, ones: Tensor):
        batch_size,cha_out,h_out,w_out = ones.shape
        batch_size, cha_in,h_in,w_in = self.input.shape
        cha_out,cha_in,kernal_size,_ = self.weight.shape
        #print(ones.shape)
        self.bias.grad = torch.zeros(cha_out)
        grad_out = ones.reshape(batch_size,cha_out,-1).transpose(1,2).reshape(batch_size*w_out*h_out,-1)
        #print(grad_out)
        kernal_grad_col = torch.mm(grad_out.t(),self.col_input).t()
  
        self.weight.grad = kernal_grad_col.t().reshape(cha_out,cha_in,kernal_size,kernal_size)

        input_grad_col = torch.mm(grad_out,self.weight.reshape(cha_out,-1))
        self.input.grad = coll2im(input_grad_col,kernal_size,batch_size,h_out,w_out)

        self.bias.grad = torch.sum(grad_out,dim=0)
        return self.input.grad

    
class Linear(Module):
    __constants__ = ['in_features', 'out_features']
    in_features: int
    out_features: int
    weight: Tensor

    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 device=None, dtype=None):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(Linear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.wheb = bias
        self.weight = Parameter(torch.randn((out_features, in_features), **factory_kwargs)*0.1)#随机weight
        if bias:
            self.bias = Parameter(torch.randn(out_features, **factory_kwargs)*0.1)
            
            
    def forward(self, input):
        self.input = input
        mid = torch.mm(input,self.weight.t())
        if self.wheb:
            self.output = mid + self.bias
        else:
            self.output = mid
        return self.output
    def backward(self, ones: Tensor):
        self.input.grad = torch.mm(ones,self.weight)
        self.weight.grad = torch.mm(ones.t(), self.input)
        self.bias.grad = torch.sum(ones,dim = 0)
        return self.input.grad

class CrossEntropyLoss():
    def __init__(self):
        pass
    def __call__(self, input, target):

        self.input = input
        self.target = target
        value_max,_ = torch.max(input, axis=1, keepdims=True)
        temp1_exp = torch.exp(input - value_max)
        softmax = temp1_exp / torch.sum(a=temp1_exp, axis=1, keepdims=True)
        self.softmax = softmax
        batch_size = input.shape[0]
        loss = 0.0
        for i in range(batch_size):
            loss += -torch.log(softmax[i][target[i]])
        loss = loss / batch_size
        self.output = loss
        return self.output
    def backward(self):
        self.input.grad = torch.zeros(self.input.shape[0],self.input.shape[1])
        batch_size = self.input.shape[0]
        tar = torch.zeros(batch_size,self.input.shape[1])
        for i in range(batch_size):
            tar[i,self.target[i]] = 1
        self.input.grad = (self.softmax - tar) / batch_size
        
        return self.input.grad
        


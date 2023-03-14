import imp
from typing import Union
from torch import Tensor
from torch.nn.modules import Module
from torch.nn.parameter import Parameter
import torch
import numpy as np
from torch.nn.modules.conv import _ConvNd
# from cnnbase import ConvBase
from torch.nn.modules.utils import _pair
from torch.nn.common_types import _size_2_t

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
        batch_size, cha_in,w_in,h_in = input.shape
        cha_out,cha_in,w_kernal,h_kernal = kernel.shape
        w_out = (w_in - w_kernal + 2*padding ) // stride + 1
        h_out = (h_in - h_kernal + 2*padding) // stride + 1
        self.output = torch.zeros(batch_size,cha_out,w_out,h_out)
        for b in range(batch_size):
            for o in range(cha_out):

                for i in range(w_out):
                    for j in range(h_out):
                        temp = 0
                        for k in range(cha_in):
                            temp += (input[b,k,i:i+w_kernal,j:j+h_kernal] * kernel[o,k,:,:] ).sum()
                        self.output[b,o,i,j] = temp + bias[o]
        return self.output
    
    def forward(self, input: Tensor):
        weight = self.weight
        bias = self.bias
        return self.conv2d(input, weight, bias)
    
    def backward(self, ones: Tensor):
        '''TODO backward的计算方法''' 
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
        self.weight = Parameter(torch.empty((out_features, in_features), **factory_kwargs))#随机weight
        if bias:
            self.bias = Parameter(torch.empty(out_features, **factory_kwargs))
            
            
    def forward(self, input):
        mid = torch.mm(input,self.weight.t())
        if self.wheb:
            self.output = mid + self.bias
        else:
            self.output = mid
        return self.output
    def backward(self, ones: Tensor):
        '''TODO'''
        return self.input.grad

class CrossEntropyLoss():
    def __init__(self):
        pass
    def __call__(self, input, target):
        value_max,_ = torch.max(input, axis=1, keepdims=True)
        temp1_exp = torch.exp(input - value_max)
        softmax = temp1_exp / torch.sum(a=temp1_exp, axis=1, keepdims=True)
        batch_size = input.shape[0]
        loss = 0.0
        for i in range(batch_size):
            loss += -torch.log(softmax[i][target[i]])
        loss = loss / batch_size
        self.output = loss
        return self.output
    def backward(self):
        '''TODO'''
        return self.input.grad
        

from torch import nn
import torch.nn.functional as F
import torch
import importlib
import custom_layers as cl
importlib.reload(cl)
import math

from utils import *

class CustomLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super(CustomLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        # weights - standard pytorch convention - shape is out x input
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features))

        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, input):
        return cl.CustomLinearLayer.apply(input, self.weight, self.bias)

class CustomLinearLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, bias):
        out = input.matmul(weight.t())
        if bias is not None:
            out = out + bias
        ctx.save_for_backward(input, weight, bias)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        input, weight, bias = ctx.saved_tensors
        grad_input = grad_weight = grad_bias = None
        if ctx.needs_input_grad[0]:
            grad_input  = grad_output.matmul(weight)
        if ctx.needs_input_grad[1]:
            grad_weight = grad_output.t().matmul(input)
        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias   = grad_output.sum(dim=0)
        return grad_input, grad_weight, grad_bias
        
class CustomSoftmax(nn.Module):
    def __init__(self, dim):
        super(CustomSoftmax, self).__init__()
        self.dim = dim

    def forward(self, input):
        return cl.CustomSoftmaxLayer.apply(input, self.dim)

class CustomReLU(nn.Module):
    def __init__(self):
        super(CustomReLU, self).__init__()

    def forward(self, input):
        return cl.CustomReLULayer.apply(input)

## This is a reference of the neural network structure
class RefMLP(nn.Module):
    def __init__(s):
        super().__init__()

        s.model = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),

            nn.Linear(512, 128),
            nn.ReLU(),

            nn.Linear(128, 10),
            nn.Softmax(1),
        )

    def forward(s, x):
        x = torch.flatten(x, 1)
        return s.model(x)
## This is our custom neural network using our custom classes
class CustomMLP(nn.Module):
    def __init__(s):
        super().__init__()

        s.model = nn.Sequential(

            CustomLinear(1024, 512),
            CustomReLU(),
            
            CustomLinear(512, 128),
            CustomReLU(),
            
            CustomLinear(128, 10),
            CustomSoftmax(1),
        )

    def forward(s, x):
        x = torch.flatten(x, 1)
        return s.model(x)
    
class CustomConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(CustomConv2d, self).__init__()
        # Initialize weight and bias
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.randn(out_channels))
        self.kernel_size = kernel_size
        self.stride = stride

    def forward(self, x):
        # Call the custom autograd function
        return cl.CustomConvLayer.apply(x, self.weight, self.bias, self.stride, self.kernel_size)


        
class CustomCNN(nn.Module):
    def __init__(s):
        super().__init__()

        s.conv = nn.Sequential(
            CustomConv2d(1, 16, kernel_size=3, stride=2),
            CustomReLU(),

            CustomConv2d(16, 64, kernel_size=3, stride=2),
            CustomReLU(),
        )

        s.model = nn.Sequential(

            CustomLinear(3136, 512),
            CustomReLU(),

            CustomLinear(512, 128),
            CustomReLU(),

            CustomLinear(128, 10),
            CustomSoftmax(1),
        )


    def forward(s, x):
        x = s.conv(x)
        x = torch.flatten(x, 1)
        x = s.model(x)
        return x
    
class RefCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv_layer_1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=2)  
        self.relu_1 = nn.ReLU(inplace=True)
        
        self.conv_layer_2 = nn.Conv2d(in_channels=16, out_channels=64, kernel_size=3, stride=2) 
        self.relu_2 = nn.ReLU(inplace=True)

        self.fc1 = nn.Linear(3136, 512)
        self.relu_3 = nn.ReLU(inplace=True)

        self.fc2 = nn.Linear(512, 128)
        self.relu_4 = nn.ReLU(inplace=True)

        self.fc3 = nn.Linear(128, 10)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.conv_layer_1(x)
        x = self.relu_1(x)

        x = self.conv_layer_2(x)
        x = self.relu_2(x)

        x = torch.flatten(x, 1) 

        x = self.fc1(x)
        x = self.relu_3(x)

        x = self.fc2(x)
        x = self.relu_4(x)

        logits = self.fc3(x)
        probs = self.softmax(logits)
        return probs
    
import torch
import logging
logging.basicConfig(level=logging.INFO)
import torch.nn.functional as F
logger = logging.getLogger(__name__)

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
        
class CustomReLULayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        # y = max(0, x)
        y = torch.clamp(input, min=0)
        # save input to build the indicator 1_{x>0} in backward
        ctx.save_for_backward(input)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        # dL/dx = dL/dy * 1_{x>0}
        mask = (x > 0).to(grad_output.dtype)
        grad_input = grad_output * mask
        return grad_input


class CustomSoftmaxLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, dim):
        m = input.max(dim=dim, keepdim=True).values              
        x_hat = input - m                                        
        exp_x = torch.exp(x_hat)                                 
        Z = exp_x.sum(dim=dim, keepdim=True)                     
        y = exp_x / Z                                            

        ctx.save_for_backward(y)
        ctx.dim = dim
        return y
        
    @staticmethod
    def backward(ctx, grad_output):
        (y,) = ctx.saved_tensors
        dim = ctx.dim
        dot = (grad_output * y).sum(dim=dim, keepdim=True)       
        grad_input = y * (grad_output - dot)                     
        return grad_input, None

class CustomConvLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, bias, stride, kernel_size):
        out = F.conv2d(input, weight, bias, stride=stride, padding=0)
        ctx.save_for_backward(input, weight, bias)
        ctx.stride = stride
        ctx.kernel_size = kernel_size
        return out

    @staticmethod
    def backward(ctx, grad_output):
        input, weight, bias = ctx.saved_tensors
        stride = ctx.stride
        K = ctx.kernel_size

        grad_input = grad_weight = grad_bias = None

        if ctx.needs_input_grad[0]:
            grad_input = F.conv_transpose2d(
                grad_output, weight,
                stride=stride,
                padding=0
            )

        if ctx.needs_input_grad[1]:
            N, C_in, H, W = input.shape
            C_out, _, _, _ = weight.shape

            cols = F.unfold(input, kernel_size=K, padding=0, stride=stride)
            N2, C_out2, H_out, W_out = grad_output.shape
            assert N2 == N and C_out2 == C_out
            L = H_out * W_out
            grad_out_flat = grad_output.view(N, C_out, L)

            grad_weight = torch.zeros_like(weight)      
            grad_weight_flat = torch.zeros(C_out, C_in*K*K, device=input.device, dtype=input.dtype)

            grad_w_batch = torch.bmm(
                grad_out_flat,                
                cols.transpose(1, 2)          
            )                                 

            grad_weight_flat = grad_w_batch.sum(dim=0)     
            grad_weight = grad_weight_flat.view_as(weight)

        
        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(dim=(0, 2, 3))

        return grad_input, grad_weight, grad_bias, None, None
import numpy as np 
import math
import torch.nn as nn
import torch

seed = 42
np.random.seed(seed)
torch.manual_seed(seed)

def xavier_init_conv(out_channels, in_channels, kernel_size):
    xavier_stddev = np.sqrt(2.0 / (in_channels * kernel_size * kernel_size + out_channels))
    weights = np.random.randn(out_channels, in_channels, kernel_size, kernel_size) * xavier_stddev
    return weights

class Conv2D():
    
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, bias=True):
        self.in_channels  = in_channels
        self.out_channels = out_channels
        self.kernel_size  = kernel_size
        self.stride       = stride
        self.use_bias     = bias
        self.ctx          = None
         
        self.weight = xavier_init_conv(out_channels, in_channels, kernel_size)
        self.w_grad = np.zeros((out_channels, in_channels, kernel_size, kernel_size))
        
        if self.use_bias:
            self.bias   = np.zeros(out_channels)
            self.b_grad = np.zeros((out_channels)) 
        else:
            self.bias   = None
            self.b_grad = None
    
    def conv(self, input):
        B_size, C_in, H_in, W_in = input.shape
        assert (H_in == W_in), "invalid input shape"
        assert ((H_in - self.kernel_size) % self.stride == 0), "invalid stride"
        assert ((W_in - self.kernel_size) % self.stride == 0), "invalide stride"
         
        H_out = math.floor((H_in - self.kernel_size)/self.stride + 1)
        W_out = math.floor((W_in - self.kernel_size)/self.stride + 1)
        C_out = self.out_channels
        
        result = np.zeros((B_size, C_out, H_out, W_out))
        
        for batch in range(B_size):
            for c_out in range(C_out):
                for h_out in range(H_out):
                    for w_out in range(W_out):
                        cur_kernel = self.weight[c_out].copy()
                        
                        h_start = h_out*self.stride
                        h_end   = h_start + self.kernel_size
                        w_start = w_out*self.stride
                        w_end   = w_start + self.kernel_size
                        
                        cur_input  = input[batch, :, h_start:h_end, w_start:w_end].copy()
                        cur_bias   = 0
                        if self.bias is not None:
                            cur_bias = self.bias[c_out].copy()
                            
                        result[batch, c_out, h_out, w_out] = np.sum(cur_kernel*cur_input) + cur_bias
                        
        return result
    
    @staticmethod
    def norm(a, b):
        return torch.norm((a - b), p='fro')
    
    @staticmethod
    def output_check(conv, input):
        custom_conv = Conv2D(conv.in_channels, conv.out_channels, conv.kernel_size, conv.stride, conv.use_bias)
        conv = nn.Conv2d(conv.in_channels, conv.out_channels, conv.kernel_size, conv.stride, bias=conv.use_bias)
        
        conv.weight = nn.Parameter(torch.tensor(custom_conv.weight, dtype=torch.float64))
        if conv.bias is not None:
            conv.bias   = nn.Parameter(torch.tensor(custom_conv.bias, dtype=torch.float64))
        torch_input = torch.tensor(input, dtype=torch.float64)
        
        Frobenuis_norm = torch.norm(torch.tensor(custom_conv(input), dtype=torch.float64) - conv(torch_input), p='fro')
        print("Diffence of output = {}".format(Frobenuis_norm.item())) 
    
    @staticmethod
    def grad_check(conv1, input1):
        conv2 = nn.Conv2d(conv1.in_channels, conv1.out_channels, conv1.kernel_size, bias=conv1.use_bias)
        conv2.weight = nn.Parameter(torch.from_numpy(conv1.weight.copy()))
        conv2.bias   = nn.Parameter(torch.from_numpy(conv1.bias.copymodel))
        input2 = tensor(input1).requires_grad_(True)

        out1 = conv1(input1)
        B, C, H, W = out1.shape
        conv1.param_grad(np.ones((B, C, H, W)))

        out2 = torch.sum(conv2(input2))
        out2.backward()
        
        print("Difference of weight grad = {}".format(Conv2D.norm(torch.from_numpy(conv1.w_grad.copy()), conv2.weight.grad)))
        
        if conv1.use_bias:
            print("Diffence of bias grad = {}".format(Conv2D.norm(torch.from_numpy(conv1.b_grad.copy()), conv2.bias.grad)))
            
        print("Diffence of input grad = {}".format(Conv2D.norm(torch.from_numpy(conv1.input_grad(np.ones((B, C, H, W))).copy()), input2.grad)))
            
    def forward(self, input):
        self.ctx = input
        return self.conv(input)
    
    def param_grad(self, output_grad):
        
        B_out, C_out, H_out, W_out = output_grad.shape
        B_in, C_in, H_in, W_in     = self.ctx.shape
        
        for batch in range(B_out):
            for c_out in range(C_out):
                for h_out in range(H_out):
                    for w_out in range(W_out):
                        h_start = h_out * self.stride
                        h_end   = h_start + self.kernel_size
                        w_start = w_out * self.stride
                        w_end   = w_start + self.kernel_size
                        d_cur   = self.ctx[batch, :, h_start:h_end, w_start:w_end].copy()
                        d_out   = output_grad[batch, c_out, h_out, w_out].copy()
                        
                        self.w_grad[c_out] += d_cur*d_out
                        
                        if self.use_bias:
                            self.b_grad[c_out] += d_out
    
    def input_grad(self, output_grad):
        B_out, C_out, H_out, W_out = output_grad.shape
        B_in, C_in, H_in, W_in     = self.ctx.shape
        
        result = np.zeros((B_in, C_in, H_in, W_in))

        for batch in range(B_out):
            for c_out in range(C_out):
                for h_out in range(H_out):
                    for w_out in range(W_out):
                        h_start = h_out * self.stride
                        h_end   = h_start + self.kernel_size
                        w_start = w_out * self.stride
                        w_end   = w_start + self.kernel_size
                        
                        d_cur   = self.weight[c_out].copy()
                        d_in    = output_grad[batch, c_out, h_out, w_out].copy()

                        result[batch, :, h_start:h_end, w_start:w_end] += d_cur*d_in
        
        return result
    
    def backward(self, output_grad):
        self.param_grad(output_grad)
        return self.input_grad(output_grad)
    
    def __call__(self, input):
        return self.forward(input)
    
    def step(self, eta):
        self.weight -= eta*self.w_grad
        self.w_grad = np.zeros_like(self.w_grad)

        if self.use_bias:
            self.bias -= eta*self.b_grad
            self.b_grad = np.zeros_like(self.b_grad)

class Padding():
    
    def __init__(self, size):
        self.ctx  = None
        self.size = size 
    
    def __call__(self, input):
        self.ctx = input
        B, C, H, W = input.shape
        H_new = H + 2*self.size
        W_new = W + 2*self.size

        output = np.zeros((B, C, H_new, W_new))
        for b in range(B):
            for c in range(C):
                for h in range(H):
                    for w in range(W):
                        output[b, c, h + self.size, w + self.size] = input[b, c, h, w]
        return output
    
    def backward(self, output_grad):
        B, C, H, W = self.ctx.shape
        result     = np.ones((B, C, H, W))
        for b in range(B):
            for c in range(C):
                for h in range(H):
                    for w in range(W):
                        result[b, c, h, w] *= output_grad[b, c, h + self.size, w + self.size]
        return result

class MaxPool():

    def __init__(self, size):
        self.ctx  = None
        self.size = size

    def __call__(self, input):
        self.ctx = input

        B, C, H, W = input.shape
        assert H % self.size == 0, "invalid input shape in maxpooling"
        assert W % self.size == 0, "invalid input shape in maxpooling"

        H_new = int(H / self.size)
        W_new = int(W / self.size)

        self.h = np.zeros((B, C, H_new, W_new))
        self.w = np.zeros((B, C, H_new, W_new))
        output    = np.zeros((B, C, H_new, W_new))

        for b in range(B):
            for c in range(C):
                for h in range(H_new):
                    for w in range(W_new):
                        h_start = h * self.size
                        h_end   = h_start + self.size
                        w_start = w * self.size
                        w_end   = w_start + self.size

                        h_max = h_start
                        w_max = w_start
                        
                        for i in range(h_start, h_end):
                            for j in range(w_start, w_end):
                                if (input[b, c, i, j] > input[b, c, h_max, w_max]):
                                    h_max = i
                                    w_max = j
                        output[b, c, h, w] = input[b, c, h_max, w_max]
                        self.h[b, c, h, w] = h_max
                        self.w[b, c, h, w] = w_max

        return output

    def backward(self, output_grad):
        B, C, H, W = self.ctx.shape
        result     = np.zeros((B, C, H, W))

        H_new = int(H / self.size)
        W_new = int(W / self.size)

        for b in range(B):
            for c in range(C):
                for h in range(H_new):
                    for w in range(W_new):
                        result[b, c, int(self.h[b, c, h, w]), int(self.w[b, c, h, w])] = output_grad[b, c, h, w]
        return result
    
class Flatten():

    def __call__(self, input):
        self.ctx = input
        B, C, H, W = input.shape

        output = np.zeros((B, C*H*W, 1, 1))

        for b in range(B):
            for c in range(C):
                for h in range(H):
                    for w in range(W):
                        output[b, c*H*W + h*W + w] = input[b, c, h, w]
        
        return output
    
    def backward(self, output_grad):
        B, C, H, W = self.ctx.shape
        result     = np.ones((B, C, H, W))

        for b in range(B):
            for c in range(C):
                for h in range(H):
                    for w in range(W):
                        result[b, c, h, w] *= output_grad[b, c*H*W + h*W + w]
        return result
        
        
    
class ReLU():

    def __init__(self):
        self.ctx = None
    
    def __call__(self, input):
        self.ctx = input
        return input * (input >= 0)
    
    def backward(self, output_grad):
        return np.array([1.0])*(self.ctx >= 0)*output_grad
    
    
class Loss():
    
    def __init__(self):
        self.truth = None
        self.ctx  = None
        
    def __call__(self, input, truth):
        self.ctx = input
        self.truth = truth
        B, C, H, W = input.shape
        assert (H == W == 1), "invalid input shape"
        result = np.zeros(B) 
        for batch in range(B):
            denominator = 0 
            mx          = 0
            for i in range(C):
                mx = max(mx, input[batch, i, 0, 0].copy())
                
            for i in range(C):
                #print(input[batch, i, 0, 0] - mx)
                denominator += np.exp(input[batch, i, 0, 0].copy() - mx)

            result[batch] = -np.log(np.exp(input[batch, int(truth[batch]), 0, 0].copy() - mx) / denominator)
        return result
            
    def backward(self):
        B, C, H, W = self.ctx.shape
        result = np.zeros((B, C, H, W))
        input = self.ctx
        
        for batch in range(B):
            denominator = 0
            mx          = 0
            for i in range(C):
                mx = max(mx, input[batch, i, 0, 0].copy())
                
            for i in range(C):
                denominator += np.exp(input[batch, i].copy() - mx)
                
            for i in range(C):
                if i == int(self.truth[batch]):
                    result[batch, i] += np.exp(input[batch, i].copy() - mx) / denominator - 1.0
                else:
                    result[batch, i] += np.exp(input[batch, i].copy() - mx) / denominator
        
        return result

def tensor(a):
    return torch.tensor(a, dtype=torch.float64)

t = Conv2D(3, 5, 3)
input = np.random.randn(8, 3, 28, 28)

#Conv2D.output_check(t, input)
#Conv2D.grad_check(t, input)

class Model():
    
    def __init__(self, input_dim, eta=1e-4):
        self.input_dim = input_dim
        self.eta = eta

        self.conv1    = Conv2D(1, 16, 3)
        self.padding1 = Padding(1)
        self.relu1    = ReLU()

        self.conv2    = Conv2D(16, 32, 3)
        self.padding2 = Padding(1)
        self.relu2    = ReLU()
        self.maxpool  = MaxPool(2)

        self.conv3    = Conv2D(32, 64, 3)
        self.relu3    = ReLU()

        self.conv4    = Conv2D(64, 64, 3)
        self.relu4    = ReLU()

        self.flatten  = Flatten()

        self.conv5    = Conv2D(64*10*10, 10, 1)

        self.loss  = Loss()
        
    def __call__(self, input):
        out1 = self.relu1(self.conv1(self.padding1(input)))
        out2 = self.maxpool(self.relu2(self.conv2(self.padding2(out1))))
        out3 = self.relu3(self.conv3(out2))
        out4 = self.relu4(self.conv4(out3))
        out5 = self.flatten(out4)
        out6 = self.conv5(out5)
        return out6
    
    def torch(self, input):
        pass 
    
    def calc_loss(self, input, truth):
        return self.loss(input, truth)

    def backward(self):
        out_grad = self.loss.backward()
        
        
        out_grad = self.conv5.backward(out_grad)
        self.conv5.step(self.eta)

        out_grad = self.flatten.backward(out_grad)

        out_grad = self.conv4.backward(self.relu4.backward(out_grad))
        self.conv4.step(self.eta)
        
        out_grad = self.conv3.backward(self.relu3.backward(out_grad))
        self.conv3.step(self.eta)
        
        out_grad = self.padding2.backward(self.conv2.backward(self.relu2.backward(self.maxpool.backward(out_grad))))
        self.conv2.step(self.eta)
        
        out_grad = self.padding1.backward(self.conv1.backward(self.relu1.backward(out_grad)))
        #print(self.conv1.w_grad)
        a = torch.from_numpy(self.conv1.weight.copy())
        #print(a)
        #print("#############")
        self.conv1.step(self.eta)
        b = torch.from_numpy(self.conv1.weight.copy())
        #print(a)
        #print("############")
        #print(b)
        print("###### : {}".format(Conv2D.norm(a, b)))
        #print(a - b)
        
    def save_weight(self):
        np.savez('checkpoint.npz', conv1_w = self.conv1.weight, conv1_b = self.conv1.bias,
                                   conv2_w = self.conv2.weight, conv2_b = self.conv2.bias,
                                   conv3_w = self.conv3.weight, conv3_b = self.conv3.bias,
                                   conv4_w = self.conv4.weight, conv4_b = self.conv4.bias,
                                   conv5_w = self.conv5.weight, conv5_b = self.conv5.bias)
        
    
    def load_weight(self):
        weight = np.load('checkpoint.npz')
        
        self.conv1.weight = weight['conv1_w']
        self.conv1.bias   = weight['conv1_b']
        
        self.conv2.weight = weight['conv2_w']
        self.conv2.bias   = weight['conv2_b']
        
        self.conv3.weight = weight['conv3_w']
        self.conv3.bias   = weight['conv3_b']

        self.conv4.weight = weight['conv4_w']
        self.conv4.bias   = weight['conv4_b']

        self.conv5.weight = weight['conv5_w']
        self.conv5.bias   = weight['conv5_b']
import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Function, Variable
from torch.nn.parameter import Parameter
from torch.distributions import Bernoulli


class RoundFn(Function):
    @staticmethod
    def forward(ctx, input, pwr_coef):
        # the standard quantization function quantized to k bit, where 2^k=pwr_coef, the input must scale to [0,1]
        return (input * (float(pwr_coef)-1.0)).round() / (float(pwr_coef)-1.0)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None

class ActQuant_init(nn.Module):
    def __init__(self, act_bit=4, scale_coef=10.0, extern_init=False, init_model=nn.Sequential()):
        super(ActQuant_init, self).__init__()
        self.pwr_coef = 2**act_bit
        self.act_bit=act_bit
        self.scale_coef = Parameter(torch.ones(1)*scale_coef)
        if extern_init:
            param=list(init_model.parameters())
            if param[0]>0.1 and param[0]<10.0:
                self.scale_coef=Parameter(param[0])
            else:
                self.scale_coef=Parameter(torch.ones(1)*1.0)

    def forward(self, x):
        if self.act_bit==32:
            out=0.5*(x.abs() - (x-self.scale_coef.abs()).abs()+self.scale_coef.abs())/self.scale_coef.abs()
        else:
            out = 0.5*(x.abs() - (x-self.scale_coef.abs()).abs()+self.scale_coef.abs())
            out = RoundFn.apply(out / self.scale_coef.abs(), self.pwr_coef)
        return out*2.0

class ActQuant_init_fixed(nn.Module):
    def __init__(self, act_bit=4, scale_coef=10.0, fixed_rescale=6.0, extern_init=False, init_model=nn.Sequential()):
        super(ActQuant_init_fixed, self).__init__()
        self.pwr_coef = 2**act_bit
        self.act_bit=act_bit
        self.fixed_rescale=fixed_rescale
        self.scale_coef = Parameter(torch.ones(1)*scale_coef)
        if extern_init:
            param=list(init_model.parameters())
            if param[0]>0.1 and param[0]<10.0:
                self.scale_coef=Parameter(param[0])
            else:
                self.scale_coef=Parameter(torch.ones(1)*1.0)

    def forward(self, x):
        if self.act_bit==32:
            out=0.5*(x.abs() - (x-self.scale_coef.abs()).abs()+self.scale_coef.abs())/self.scale_coef.abs()
        else:
            out = 0.5*(x.abs() - (x-self.scale_coef.abs()).abs()+self.scale_coef.abs())
            out = RoundFn.apply(out / self.scale_coef.abs(), self.pwr_coef)
        return out*self.fixed_rescale


class QuantConv_init(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, bit=32, extern_init=False, init_model=nn.Sequential()):
        super(QuantConv_init, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            groups, bias)
        self.bit = bit
        self.pwr_coef = 2**bit

        nn.init.kaiming_normal_(self.weight,mode='fan_out',nonlinearity='relu')
        if extern_init:
            param=list(init_model.parameters())
            self.weight=Parameter(param[0])
            if bias:
                self.bias=Parameter(param[1])
            

    def forward(self, x):
        if self.bit == 32:
            return F.conv2d(
                x, self.weight, self.bias, self.stride, self.padding,
                self.dilation, self.groups)
        else:
            w = torch.tanh(self.weight)
            wq = RoundFn.apply(
                w / (2 * torch.max(w.abs())) + 0.5, self.pwr_coef)
            wq = wq * (self.pwr_coef-1.0)
            wq = wq - (self.pwr_coef/2.0)
            wq = wq / (self.pwr_coef/2.0)
            #wq = wq - (self.pwr_coef/2.0) / (self.pwr_coef-1.0)
            #print(wq)
            return F.conv2d(
                x, wq, self.bias, self.stride, self.padding, self.dilation,
                self.groups)


class QuantLinear_init(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, bit=32, extern_init=False, init_model=nn.Sequential()):
        super(QuantLinear_init, self).__init__(in_features, out_features, bias)
        self.bit = bit
        self.pwr_coef = 2**bit

        nn.init.kaiming_normal_(self.weight,mode='fan_out',nonlinearity='relu')
        if extern_init:
            param=list(init_model.parameters())
            self.weight=Parameter(param[0])
            if bias:
                self.bias=Parameter(param[1])       

    def forward(self, x):
        if self.bit == 32:
            return F.linear(x, self.weight, self.bias)
        else:
            w = torch.tanh(self.weight)
            wq = RoundFn.apply(
                w / (2 * torch.max(w.abs())) + 0.5, self.pwr_coef) * (float(self.pwr_coef)-1.0)
            wq = wq - (self.pwr_coef//2+1)
            wq = wq / (float(self.pwr_coef)-1.0)

            if self.bias!=None:
                b=torch.tanh(self.bias)
                bq = RoundFn.apply(
                    b / (2 * torch.max(b.abs())) + 0.5, self.pwr_coef) * (float(self.pwr_coef)-1.0)
                bq = bq - (self.pwr_coef//2+1)
                bq = bq / (float(self.pwr_coef)-1.0)
            return F.linear(x, wq, bq)


class Shift3x3(nn.Module):
    def __init__(self, planes):
        super(Shift3x3, self).__init__()

        self.planes = planes
        kernel = np.zeros((planes, 1, 3, 3), dtype=np.float32)

        for i in range(planes):
            if i%5==0:
                kernel[i,0,0,1]=1.0
            elif i%5==1:
                kernel[i,0,1,0]=1.0
            elif i%5==2:
                kernel[i,0,1,1]=1.0
            elif i%5==3:
                kernel[i,0,1,2]=1.0
            else:
                kernel[i,0,2,1]=1.0

        self.register_parameter('bias', None)
        self.kernel = nn.Parameter(torch.from_numpy(kernel), requires_grad=False)

    def forward(self, input):
        return F.conv2d(input,
                        self.kernel,
                        self.bias,
                        (1, 1), # stride
                        (1, 1), # padding
                        1, # dilation
                        self.planes, #groups
                       )


class DiracDeltaBlock(nn.Module):

    def __init__(self, in_planes, out_planes, stride=1, expansion=1, weight_bit=3, act_bit=4, first_block=False, extern_init=False, init_model=nn.Sequential()):
        super(DiracDeltaBlock, self).__init__()
        self.is_downsample=False
        if stride != 1 or in_planes != out_planes:
            self.is_downsample=True
        self.first_block=first_block
        self.expansion = expansion
        if (not self.is_downsample) or self.first_block:
            self.in_planes = int(in_planes/2)
            self.out_planes = int(out_planes/2)
            self.mid_planes = mid_planes = int(out_planes/2 * self.expansion)
        else:
            self.in_planes = in_planes
            self.out_planes = int(out_planes/2)
            self.mid_planes = mid_planes = int(out_planes/2 * self.expansion)
        
        self.weight_bit=weight_bit
        self.act_bit=act_bit

        if not extern_init:
            self.conv1 = QuantConv_init(
                self.in_planes, self.mid_planes, kernel_size=1, bias=False, stride=1, bit=self.weight_bit)
            self.bn1 = nn.BatchNorm2d(self.mid_planes)

            self.shift2 = Shift3x3(self.mid_planes)

            self.conv3 = QuantConv_init(
                self.mid_planes, self.out_planes, kernel_size=1, bias=False, stride=1, bit=self.weight_bit)
            self.bn3 = nn.BatchNorm2d(self.out_planes)
            self.quant_act1=ActQuant_init(self.act_bit,2.0)
            self.quant_act2=ActQuant_init(self.act_bit,2.0)

            nn.init.constant_(self.bn1.weight, 1)
            nn.init.constant_(self.bn1.bias, 0)
            nn.init.constant_(self.bn3.weight, 1)
            nn.init.constant_(self.bn3.bias, 0)
        else:
            self.conv1 = QuantConv_init(
                self.in_planes, self.mid_planes, kernel_size=1, bias=False, stride=1, bit=self.weight_bit, extern_init=True, init_model=init_model.conv1)
            self.bn1 = init_model.bn1

            self.shift2 = Shift3x3(self.mid_planes)

            self.conv3 = QuantConv_init(
                self.mid_planes, self.out_planes, kernel_size=1, bias=False, stride=1, bit=self.weight_bit, extern_init=True, init_model=init_model.conv3)
            self.bn3 = init_model.bn3
            self.quant_act1=ActQuant_init(self.act_bit,2.0, extern_init=True, init_model=init_model.quant_act1)
            self.quant_act2=ActQuant_init(self.act_bit,2.0, extern_init=True, init_model=init_model.quant_act2)


        self.shortcut = nn.Sequential()
        if self.is_downsample or self.first_block:
            if not extern_init:
                self.shortcut = nn.Sequential(
                    QuantConv_init(self.in_planes, self.out_planes, kernel_size=1, stride=1, bias=False, bit=self.weight_bit),
                    nn.BatchNorm2d(self.out_planes)
                )
                nn.init.constant_(self.shortcut[1].weight, 1)
                nn.init.constant_(self.shortcut[1].bias, 0)
            else:
                init_shortcut=init_model.shortcut
                init_layer=list(init_shortcut.children())
                shortcut_conv2=QuantConv_init(self.in_planes, self.out_planes, kernel_size=1, stride=1, bias=False, bit=self.weight_bit, extern_init=True,init_model=init_layer[0])
                shortcut_bn2=init_layer[1]
                self.shortcut = nn.Sequential(
                    shortcut_conv2,
                    shortcut_bn2
                )


    def forward(self, x):
        #channel split
        size=x.size()
        if (not self.is_downsample) or self.first_block:
            x1=x[:,:int(size[1]/2),:,:]
            x2=x[:,int(size[1]/2):,:,:]
        else:
            x1=x
            x2=x   

        if (self.is_downsample) or self.first_block:
            x1=F.max_pool2d(x1, kernel_size=2, stride=2)
        out1=self.shortcut(x1)
        out1=F.relu(out1)

        out2 = F.relu(self.bn1(self.conv1(x2)))
        out2 = self.quant_act1(out2)
        if (self.is_downsample) or self.first_block:
            out2=F.max_pool2d(out2, kernel_size=2, stride=2)
        out2 = self.shift2(out2)
        out2 = F.relu(self.bn3(self.conv3(out2)))

        if self.is_downsample:
            out1=self.quant_act2(out1)
        out2=self.quant_act2(out2)

        #channel shuffle and concat
        result=torch.cat([out1,out2],1)
        size_r=result.size()
        tmp1=result[:,int(3*size_r[1]/4):,:,:]
        tmp2=result[:,:int(3*size_r[1]/4),:,:]
        result=torch.cat([tmp1,tmp2],1)

        return result


class DiracDeltaNet(nn.Module):
    def __init__(self, block, num_blocks, base_channelsize=48, num_classes=1000, weight_bit=4, act_bit=4, first_weight_bit=32, first_act_bit=32, last_weight_bit=32, last_act_bit=32, fc_bit=32, extern_init=False, init_model=nn.Sequential()):
        super(DiracDeltaNet, self).__init__()
        self.num_classes = num_classes
        self.in_planes = 64
        self.base_channelsize = base_channelsize
        self.weight_bit=weight_bit
        self.act_bit=act_bit
        self.first_weight_bit=first_weight_bit
        self.first_act_bit=first_act_bit
        self.last_weight_bit=last_weight_bit
        self.last_act_bit=last_act_bit

        if not extern_init:
            self.conv1 = QuantConv_init(3, 32, kernel_size=1, stride=1, bias=False, bit=self.first_weight_bit)
            self.bn1 = nn.BatchNorm2d(32)
            self.quant_act1 = ActQuant_init_fixed(self.first_act_bit,6.0,fixed_rescale=6.0)
            self.shift1 = Shift3x3(32)
            self.conv2 = QuantConv_init(32, 64, kernel_size=1, stride=1, bias=False, bit=self.first_weight_bit)
            self.bn2 = nn.BatchNorm2d(64)
            self.quant_act2=ActQuant_init(self.act_bit,2.0)
            self.shift2 = Shift3x3(64)

            self.layer1 = self._make_layer(block, self.base_channelsize, num_blocks[0], stride=2, weight_bit=self.weight_bit, act_bit=self.act_bit, first_block=True)
            self.layer2 = self._make_layer(block, self.base_channelsize*2, num_blocks[1], stride=2, weight_bit=self.weight_bit, act_bit=self.act_bit, first_block=False)
            self.layer3 = self._make_layer(block, self.base_channelsize*4, num_blocks[2], stride=2, weight_bit=self.weight_bit, act_bit=self.act_bit, first_block=False)
            self.conv3 = QuantConv_init(self.base_channelsize*4, 1024, kernel_size=1, stride=1, bias=False, bit=self.last_weight_bit)
            self.bn3 = nn.BatchNorm2d(1024)
            self.quant_act3=ActQuant_init_fixed(self.last_act_bit,10.0,fixed_rescale=10.0)
            self.linear=QuantLinear_init(1024, num_classes, True, fc_bit)
            nn.init.constant_(self.bn1.weight, 1)
            nn.init.constant_(self.bn1.bias, 0)
            nn.init.constant_(self.bn2.weight, 1)
            nn.init.constant_(self.bn2.bias, 0)
            nn.init.constant_(self.bn3.weight, 1)
            nn.init.constant_(self.bn3.bias, 0)
        else:
            self.conv1 = QuantConv_init(3, 32, kernel_size=1, stride=1, bias=False, bit=self.first_weight_bit, extern_init=True, init_model=init_model.conv1)
            self.bn1 = init_model.bn1
            self.quant_act1=ActQuant_init_fixed(self.first_act_bit,6.0,fixed_rescale=6.0, extern_init=True, init_model=init_model.quant_act1)
            self.shift1 = Shift3x3(32)
            self.conv2 = QuantConv_init(32, 64, kernel_size=1, stride=1, bias=False, bit=self.weight_bit, extern_init=True, init_model=init_model.conv2)
            self.bn2 = init_model.bn2
            self.quant_act2=ActQuant_init(self.act_bit,2.0, extern_init=True, init_model=init_model.quant_act2)
            self.shift2 = Shift3x3(64)

            self.layer1 = self._make_layer(block, self.base_channelsize, num_blocks[0], stride=2, weight_bit=self.weight_bit, act_bit=self.act_bit, first_block=True, extern_init=True, init_model=init_model.layer1)
            self.layer2 = self._make_layer(block, self.base_channelsize*2, num_blocks[1], stride=2, weight_bit=self.weight_bit, act_bit=self.act_bit, first_block=False, extern_init=True, init_model=init_model.layer2)
            self.layer3 = self._make_layer(block, self.base_channelsize*4, num_blocks[2], stride=2, weight_bit=self.weight_bit, act_bit=self.act_bit, first_block=False, extern_init=True, init_model=init_model.layer3)
            self.conv3 = QuantConv_init(self.base_channelsize*4, 1024, kernel_size=1, stride=1, bias=False, bit=self.last_weight_bit, extern_init=True, init_model=init_model.conv3)
            self.bn3 = init_model.bn3
            self.quant_act3=ActQuant_init_fixed(self.last_act_bit,10.0,fixed_rescale=10.0, extern_init=True, init_model=init_model.quant_act3)
            self.linear=QuantLinear_init(1024, num_classes, True, fc_bit, extern_init=True, init_model=init_model.linear)


    def _make_layer(self, block, planes, num_blocks, stride, weight_bit, act_bit, first_block=False, extern_init=False, init_model=nn.Sequential()):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        planes = int(planes)
        i=0
        init_layer=list(init_model.children())
        my_first_block=first_block
        for stride in strides:
            if not extern_init:
                new_block=block(self.in_planes, planes, stride, weight_bit=weight_bit, act_bit=act_bit, first_block=my_first_block)
            else:
                new_block=block(self.in_planes, planes, stride, weight_bit=weight_bit, act_bit=act_bit, first_block=my_first_block, extern_init=True, init_model=init_layer[i])
            layers.append(new_block)
            self.in_planes = planes
            i=i+1
            my_first_block=False
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.quant_act1(out)
        out = F.max_pool2d(out,kernel_size=2,stride=2,padding=1)
        out = self.shift1(out)
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.quant_act2(out)
        out = F.max_pool2d(out, kernel_size=2, stride=2, padding=1)
        out = self.shift2(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.relu(self.bn3(self.conv3(out)))
        out = self.quant_act3(out)
        out = F.avg_pool2d(out, 7)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def DiracDeltaNet_wrapper(expansion=1, base_channelsize=48, num_classes=1000, weight_bit=32, act_bit=32, first_weight_bit=32, first_act_bit=32, last_weight_bit=32, last_act_bit=32, fc_bit=32, extern_init=False, init_model=nn.Sequential()):
    block = lambda in_planes, out_planes, stride, weight_bit, act_bit, first_block=False, extern_init=False, init_model=nn.Sequential(): \
        DiracDeltaBlock(in_planes, out_planes, stride, expansion=expansion, weight_bit=weight_bit, act_bit=act_bit, first_block=first_block, extern_init=extern_init, init_model=init_model)
    return DiracDeltaNet(block, [4,8,4], base_channelsize=base_channelsize, num_classes=num_classes, weight_bit=weight_bit, act_bit=act_bit, first_weight_bit=first_weight_bit, first_act_bit=first_act_bit, last_weight_bit=last_weight_bit, last_act_bit=last_act_bit, fc_bit=fc_bit, extern_init=extern_init, init_model=init_model)


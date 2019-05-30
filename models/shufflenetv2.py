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
            

    def foward(self, x):
        if self.bit == 32:
            return F.conv2d(
                x, self.weight, self.bias, self.stride, self.padding,
                self.dilation, self.groups)
        else:
            w = torch.tanh(self.weight)
            wq = RoundFn.apply(
                w / (2 * torch.max(w.abs())) + 0.5, self.pwr_coef)
            wq = 2 * wq - 1
            return F.conv2d(
                x, wq, self.bias, self.stride, self.padding, self.dilation,
                self.groups)



class ShuffleBlock(nn.Module):

    def __init__(self, in_planes, out_planes, stride=1, expansion=1, weight_bit=3, act_bit=4, extern_init=False, init_model=nn.Sequential()):
        super(ShuffleBlock, self).__init__()
        self.is_downsample=False
        if stride != 1 or in_planes != out_planes:
            self.is_downsample=True
        self.expansion = expansion
        if not self.is_downsample:
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
                self.in_planes, self.mid_planes, kernel_size=1, bias=False, bit=self.weight_bit)
            self.bn1 = nn.BatchNorm2d(self.mid_planes)

            self.conv2 = QuantConv_init(self.mid_planes, self.mid_planes, kernel_size=3, padding=1,
                               stride=stride, bias=False, groups=self.mid_planes, bit=self.weight_bit)
            self.bn2 = nn.BatchNorm2d(self.mid_planes)

            self.conv3 = QuantConv_init(
                self.mid_planes, self.out_planes, kernel_size=1, bias=False, stride=1, bit=self.weight_bit)
            self.bn3 = nn.BatchNorm2d(self.out_planes)

            nn.init.constant_(self.bn1.weight, 1)
            nn.init.constant_(self.bn1.bias, 0)
            nn.init.constant_(self.bn2.weight, 1)
            nn.init.constant_(self.bn2.bias, 0)
            nn.init.constant_(self.bn3.weight, 1)
            nn.init.constant_(self.bn3.bias, 0)
        else:
            self.conv1 = QuantConv_init(
                self.in_planes, self.mid_planes, kernel_size=1, bias=False, bit=self.weight_bit, extern_init=True, init_model=init_model.conv1)
            self.bn1 = init_model.bn1

            self.conv2 = QuantConv_init(self.mid_planes, self.mid_planes, kernel_size=3, padding=1,
                               stride=stride, bias=False, groups=self.mid_planes, bit=self.weight_bit, extern_init=True, init_model=init_model.conv2)
            self.bn2 = init_model.bn2

            self.conv3 = QuantConv_init(
                self.mid_planes, self.out_planes, kernel_size=1, bias=False, stride=1, bit=self.weight_bit, extern_init=True, init_model=init_model.conv3)
            self.bn3 = init_model.bn3


        self.shortcut = nn.Sequential()
        if self.is_downsample:
            if not extern_init:
                self.shortcut = nn.Sequential(
                    QuantConv_init(self.in_planes, self.in_planes, kernel_size=3, padding=1, stride=stride, bias=False, groups=self.in_planes, bit=self.weight_bit),
                    nn.BatchNorm2d(self.in_planes),
                    QuantConv_init(self.in_planes, self.out_planes, kernel_size=1, stride=1, bias=False, bit=self.weight_bit),
                    nn.BatchNorm2d(self.out_planes)
                )
                nn.init.constant_(self.shortcut[1].weight, 1)
                nn.init.constant_(self.shortcut[1].bias, 0)
                nn.init.constant_(self.shortcut[3].weight, 1)
                nn.init.constant_(self.shortcut[3].bias, 0)
            else:
                init_shortcut=init_model.shortcut
                init_layer=list(init_shortcut.children())
                shortcut_conv1=QuantConv_init(self.in_planes, self.in_planes, kernel_size=3, padding=1, stride=stride, bias=False, groups=self.in_planes, bit=self.weight_bit, extern_init=True,init_model=init_layer[0])
                shortcut_bn1=init_layer[1]
                shortcut_conv2=QuantConv_init(self.in_planes, self.out_planes, kernel_size=1, stride=1, bias=False, bit=self.weight_bit, extern_init=True,init_model=init_layer[2])
                shortcut_bn2=init_layer[3]
                self.shortcut = nn.Sequential(
                    shortcut_conv1,
                    shortcut_bn1,
                    shortcut_conv2,
                    shortcut_bn2
                )


    def forward(self, x):
        #channel split
        size=x.size()
        if not self.is_downsample:
            x1=x[:,:int(size[1]/2),:,:]
            x2=x[:,int(size[1]/2):,:,:]
        else:
            x1=x
            x2=x   

        out1=self.shortcut(x1)
        out1=F.relu(out1)

        out2 = F.relu(self.bn1(self.conv1(x2)))
        out2 = self.bn2(self.conv2(out2))
        out2 = F.relu(self.bn3(self.conv3(out2)))

        #channel shuffle and concat
        size1=out1.size()
        size2=out2.size()
        out1=out1.view(size1[0],size1[1],1,size1[2],size1[3])
        out2=out2.view(size2[0],size2[1],1,size2[2],size2[3])

        result=torch.cat([out1,out2],2)
        result=result.view(size[0],size1[1]+size2[1],size1[2],size1[3])

        return result


class ShuffleNetv2(nn.Module):
    def __init__(self, block, num_blocks, base_channelsize=48, num_classes=1000, weight_bit=4, act_bit=4, extern_init=False, init_model=nn.Sequential()):
        super(ShuffleNetv2, self).__init__()
        self.num_classes = num_classes
        self.in_planes = 24
        self.base_channelsize = base_channelsize
        self.weight_bit=weight_bit
        self.act_bit=act_bit

        if not extern_init:
            self.conv1 = QuantConv_init(3, 24, kernel_size=3, stride=2, padding=1, bias=False, bit=self.weight_bit)
            self.bn1 = nn.BatchNorm2d(24)
            self.layer1 = self._make_layer(block, self.base_channelsize, num_blocks[0], stride=2, weight_bit=self.weight_bit, act_bit=self.act_bit)
            self.layer2 = self._make_layer(block, self.base_channelsize*2, num_blocks[1], stride=2, weight_bit=self.weight_bit, act_bit=self.act_bit)
            self.layer3 = self._make_layer(block, self.base_channelsize*4, num_blocks[2], stride=2, weight_bit=self.weight_bit, act_bit=self.act_bit)
            self.conv2 = QuantConv_init(self.base_channelsize*4, 1024, kernel_size=1, stride=1, bias=False, bit=self.weight_bit)
            self.bn2 = nn.BatchNorm2d(1024)
            self.linear = nn.Linear(1024, num_classes)
            nn.init.constant_(self.bn1.weight, 1)
            nn.init.constant_(self.bn1.bias, 0)
            nn.init.constant_(self.bn2.weight, 1)
            nn.init.constant_(self.bn2.bias, 0)
        else:
            self.conv1 = QuantConv_init(3, 24, kernel_size=3, stride=2, padding=1, bias=False, bit=self.weight_bit, extern_init=True, init_model=init_model.conv1)
            self.bn1 = init_model.bn1
            self.layer1 = self._make_layer(block, self.base_channelsize, num_blocks[0], stride=2, weight_bit=self.weight_bit, act_bit=self.act_bit, extern_init=True, init_model=init_model.layer1)
            self.layer2 = self._make_layer(block, self.base_channelsize*2, num_blocks[1], stride=2, weight_bit=self.weight_bit, act_bit=self.act_bit, extern_init=True, init_model=init_model.layer2)
            self.layer3 = self._make_layer(block, self.base_channelsize*4, num_blocks[2], stride=2, weight_bit=self.weight_bit, act_bit=self.act_bit, extern_init=True, init_model=init_model.layer3)
            self.conv2 = QuantConv_init(self.base_channelsize*4, 1024, kernel_size=1, stride=1, bias=False, bit=self.weight_bit, extern_init=True, init_model=init_model.conv2)
            self.bn2 = init_model.bn2
            self.linear = nn.Linear(1024, num_classes)
            self.linear.weight=Parameter(init_model.linear.weight)
            self.linear.bias=Parameter(init_model.linear.bias)


    def _make_layer(self, block, planes, num_blocks, stride, weight_bit, act_bit, extern_init=False, init_model=nn.Sequential()):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        planes = int(planes)
        i=0
        init_layer=list(init_model.children())
        for stride in strides:
            if not extern_init:
                new_block=block(self.in_planes, planes, stride, weight_bit=weight_bit, act_bit=act_bit)
            else:
                new_block=block(self.in_planes, planes, stride, weight_bit=weight_bit, act_bit=act_bit, extern_init=True, init_model=init_layer[i])
            layers.append(new_block)
            self.in_planes = planes
            i=i+1
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.max_pool2d(out,kernel_size=3,stride=2,padding=1)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.relu(self.bn2(self.conv2(out)))
        out = F.avg_pool2d(out, 7)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def ShuffleNetv2_wrapper(expansion=1, base_channelsize=48, num_classes=1000, weight_bit=32, act_bit=32, extern_init=False, init_model=nn.Sequential()):
    block = lambda in_planes, out_planes, stride, weight_bit, act_bit, extern_init=False, init_model=nn.Sequential(): \
        ShuffleBlock(in_planes, out_planes, stride, expansion=expansion, weight_bit=weight_bit, act_bit=act_bit, extern_init=extern_init, init_model=init_model)
    return ShuffleNetv2(block, [4,8,4], base_channelsize=base_channelsize, num_classes=num_classes, weight_bit=weight_bit, act_bit=act_bit, extern_init=extern_init, init_model=init_model)


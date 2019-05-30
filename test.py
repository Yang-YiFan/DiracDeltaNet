import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import os
import argparse

from torch.autograd import Variable
from extensions.utils import progress_bar
from extensions.model_refinery_wrapper import ModelRefineryWrapper
from extensions.refinery_loss import RefineryLoss

from models import ShuffleNetv2_wrapper
from models import DiracDeltaNet_wrapper


parser = argparse.ArgumentParser(description='PyTorch imagenet inference')
parser.add_argument('--datadir', help='path to dataset')
parser.add_argument('--inputdir', help='path to input model')
args = parser.parse_args()


# Data
print('==> Preparing data..')
# Data loading code
valdir = os.path.join(args.datadir, 'val')
    
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])


transform_test = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])

#imagenet
testset = datasets.ImageFolder(valdir, transform_test)
num_classes=1000

testloader = torch.utils.data.DataLoader(testset, batch_size=1000, shuffle=False, pin_memory=True, num_workers=30)


use_cuda = torch.cuda.is_available()

print('Using input path: %s' % args.inputdir)
checkpoint = torch.load(args.inputdir)
init_net = checkpoint['net']
net=init_net.to('cpu')

label_refinery=torch.load('./resnet50.t7')
net = ModelRefineryWrapper(net, label_refinery)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    net = nn.DataParallel(net)

net=net.to(device)

criterion = RefineryLoss()



def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k)
        return res


def test():
    net.eval()
    criterion.eval()
    test_loss = 0
    correct_1 = 0
    correct_5 = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(testloader):
        if use_cuda:
            inputs, targets = inputs.cuda(device), targets.cuda(device)
        with torch.no_grad():
            outputs = net(inputs)
        loss = criterion(outputs, targets)

        if isinstance(loss, tuple):
            loss_value, outputs = loss
        else:
            loss_value = loss

        test_loss += loss_value.item()
        prec1, prec5 = accuracy(outputs, targets, topk=(1, 5))
        total += targets.size(0)
        correct_1 += prec1
        correct_5 += prec5

        progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (test_loss/(batch_idx+1), 100.*float(correct_1)/float(total), correct_1, total))

    return 100.*float(correct_1)/float(total),100.*float(correct_5)/float(total),test_loss

acc1,acc5,loss=test()
print('top-1 accuracy: {0}, top-5 accuracy: {1}'.format(acc1,acc5))

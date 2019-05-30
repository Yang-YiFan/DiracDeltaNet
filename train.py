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

from models import ShuffleNetv2_wrapper
from models import DiracDeltaNet_wrapper

from torch.autograd import Variable
from extensions.utils import progress_bar
from extensions.model_refinery_wrapper import ModelRefineryWrapper
from extensions.refinery_loss import RefineryLoss


parser = argparse.ArgumentParser(description='PyTorch imagenet Training in quant')

parser.add_argument('--datadir', help='path to dataset')
parser.add_argument('--inputdir', help='path to input model')
parser.add_argument('--outputdir', help='path to output model')
parser.add_argument('--logdir', default='./log/log.txt', help='path to log')

parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')

parser.add_argument('--lr', default=0.5, type=float, help='learning rate')
parser.add_argument('--lr_policy', choices=('step', 'linear'), help='learning rate decay policy', default='linear')
parser.add_argument('--totalepoch', default=90, type=int, help='how many epoch')
parser.add_argument('--batch_size', '-b', default=1024, type=int, help='batch size')
parser.add_argument('--weight_decay', '--wd', default=4e-5, type=float, help='weight decay (default: 4e-5)')
parser.add_argument('--crop_scale', default=0.2, type=float, help='random resized crop scale')

parser.add_argument('--expansion', '-e', default=2.0, type=float, help='expansion rate for the middle plate')
parser.add_argument('--base_channel_size', default=116, type=int, help='base channel size of the shuffle block')
parser.add_argument('--weight_bit', default=32, type=int, help='conv weight bitwidth')
parser.add_argument('--act_bit', default=32, type=int, help='activation bitwidth')
parser.add_argument('--first_weight_bit', default=32, type=int, help='first conv weight bitwidth')
parser.add_argument('--first_act_bit', default=32, type=int, help='first conv activation bitwidth')
parser.add_argument('--last_weight_bit', default=32, type=int, help='last conv weight bitwidth')
parser.add_argument('--last_act_bit', default=32, type=int, help='last conv activation bitwidth')
parser.add_argument('--fc_bit', default=32, type=int, help='fc weight bitwidth')

args = parser.parse_args()


# Data
print('==> Preparing data..')
# Data loading code
traindir = os.path.join(args.datadir, 'train')
valdir = os.path.join(args.datadir, 'val')
    
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

transform_train=transforms.Compose([
        transforms.RandomResizedCrop(224,scale=(args.crop_scale,1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])


transform_test = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])

#imagenet
trainset = datasets.ImageFolder(traindir, transform_train)
testset = datasets.ImageFolder(valdir, transform_test)
num_classes=1000

trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=30)
testloader = torch.utils.data.DataLoader(testset, batch_size=1000, shuffle=False, pin_memory=True, num_workers=30)


use_cuda = torch.cuda.is_available()
best_acc = 0.0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

if args.inputdir != None:
    input_path = args.inputdir
    print('Using input path: %s to fine tune' % input_path)
    checkpoint = torch.load(input_path)
    init_net = checkpoint['net']
    init_net=init_net.to('cpu')
else:
    if args.resume:
        print('Resuming from checkpoint')
    else:
        print('Training from scratch')

output_path =args.outputdir
print('Using output path: %s' % output_path)


if not args.resume:
    if args.inputdir != None:
        net=DiracDeltaNet_wrapper(expansion=args.expansion, base_channelsize=args.base_channel_size, num_classes=num_classes, 
            weight_bit=args.weight_bit, act_bit=args.act_bit, first_weight_bit=args.first_weight_bit, first_act_bit=args.first_act_bit, 
            last_weight_bit=args.last_weight_bit, last_act_bit=args.last_act_bit, fc_bit=args.fc_bit, extern_init=True, init_model=init_net)
    else:
        net=DiracDeltaNet_wrapper(expansion=args.expansion, base_channelsize=args.base_channel_size, num_classes=num_classes, 
            weight_bit=args.weight_bit, act_bit=args.act_bit, first_weight_bit=args.first_weight_bit, first_act_bit=args.first_act_bit, 
            last_weight_bit=args.last_weight_bit, last_act_bit=args.last_act_bit, fc_bit=args.fc_bit)
else:
    checkpoint = torch.load(output_path)
    net = checkpoint['net']
    best_acc = checkpoint['acc_1']
    start_epoch = checkpoint['epoch']+1

label_refinery=torch.load('./resnet50.t7')
net = ModelRefineryWrapper(net, label_refinery)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    net = nn.DataParallel(net)
net=net.to(device)

criterion = RefineryLoss()

model_trainable_parameters = filter(lambda x: x.requires_grad, net.parameters())
optimizer = optim.SGD(model_trainable_parameters, lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)

if not args.resume:
    iteration=0
else:
    iteration=start_epoch*(int(1281167/args.batch_size)+1)

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


def adjust_learning_rate(optimizer, iteration, lr):
    if args.lr_policy == 'linear' :
        #linear lr decay
        total_iteration=(int(1281167/args.batch_size)+1)*args.totalepoch
        new_lr=lr-lr*float(iteration)/(float(total_iteration-1.0))

    else:
        #step lr decay for fine tuning
        if epoch<20:
            new_lr=lr
        elif epoch<30:
            new_lr=lr/5.0
        elif epoch<40:
            new_lr=lr/25.0
        else:
            new_lr=lr/125.0

    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr
    return new_lr


# Training
def train(epoch):
    global iteration
    print('\nEpoch: %d' % epoch)
    net.train()
    criterion.train()
    net.to(device)
    train_loss = 0
    correct_1 = 0 # moniter top 1
    correct_5 = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        adjust_learning_rate(optimizer, iteration, args.lr)
        if use_cuda:
            inputs, targets = inputs.cuda(device,non_blocking=True), targets.cuda(device,non_blocking=True)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        
        if isinstance(loss, tuple):
            loss_value, outputs = loss
        else:
            loss_value = loss
        loss_value.backward()
        optimizer.step()

        train_loss += loss_value.item()
        prec1, prec5 = accuracy(outputs, targets, topk=(1, 5))
        total += targets.size(0)
        correct_1 += prec1
        correct_5 += prec5

        iteration=iteration+1

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (train_loss/(batch_idx+1), 100.*float(correct_1)/float(total), correct_1, total))
    return 100.*float(correct_1)/float(total),100.*float(correct_5)/float(total),train_loss

def test(epoch):
    global best_acc
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

    # Save checkpoint.
    acc_1 = 100.*float(correct_1)/float(total)
    if acc_1 > best_acc:
        print('Saving..')
        state = {
            'net': net.module.model if use_cuda and torch.cuda.device_count() > 1 else net.model,
            'acc_1': acc_1,
            'acc_5': 100.*float(correct_5)/float(total),
            'lr': args.lr,
            'epoch': epoch,
            'weight_decay': args.weight_decay,
            'batch_size': args.batch_size,
        }
        torch.save(state, output_path)
        print('* Saved checkpoint to %s' % output_path)
        best_acc = acc_1

    return 100.*float(correct_1)/float(total),100.*float(correct_5)/float(total),test_loss

if not args.resume:
    f=open(args.logdir,'w')
else:
    f=open(args.logdir,'a')
for epoch in range(start_epoch, int(args.totalepoch)):
    acc1,acc5,loss=train(epoch)
    f.write(str(epoch)+' '+str(acc1)+' '+str(acc5)+' '+str(loss)+' ')
    acc1,acc5,loss=test(epoch)
    f.write(str(acc1)+' '+str(acc5)+' '+str(loss)+'\n')

f.close()

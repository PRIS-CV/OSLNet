'''Train CIFAR10 with PyTorch.'''
from __future__ import print_function
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import time
import math
import torch
import logging
import argparse
import torchvision
# from models import *
import torch.nn as nn
import numpy as np
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
from torch.nn.modules.module import Module
from torch.nn.parameter import Parameter
import torchvision.transforms as transforms
from itertools import combinations, permutations
#from utils import progress_bar
logging.basicConfig(level=logging.INFO)
parser = argparse.ArgumentParser(description='PyTorch CIFAR100 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
args = parser.parse_args()
logging.info(args)

store_name = "OS-CNN"
nb_epoch = 400
# setup output


use_cuda = torch.cuda.is_available()


# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])



transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=256, shuffle=True, num_workers=8)



testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=256, shuffle=False, num_workers=8)


def design_tensor_C(previous_hidden_num =256,next_hidden_num=100,classes=100):
    tensor_C = np.zeros(previous_hidden_num*next_hidden_num).reshape(previous_hidden_num,next_hidden_num)

    top_left_nums = int(math.floor(previous_hidden_num / classes))
    column =top_left_nums
    row =  int(math.floor(next_hidden_num /classes))
    top_left = [[i * column, i * row] for i in range(classes)]

    remainder_1 = previous_hidden_num % classes
    remainder_2 = next_hidden_num % classes

    base_matrix = []
    for i in range(column):
        for j in range(row):
            base_matrix.append([i,j])
    base_matrix_1 = np.array(base_matrix)

    base_matrix = []
    for i in range(column+remainder_1):
        for j in range(row+remainder_2):
            base_matrix.append([i,j])
    base_matrix_2 = np.array(base_matrix)


    matrix_one_1 = [(base_matrix_1 + i).tolist() for i in top_left[:-1]]

    matrix_one_1_1 = []    
    for item in matrix_one_1:
        matrix_one_1_1 = matrix_one_1_1 + item

    matrix_one_2 = (base_matrix_2 + top_left[-1]).tolist()
    matrix_one = matrix_one_1_1 + matrix_one_2

    for item in range(len(matrix_one)):
        tensor_C[matrix_one[item][0],matrix_one[item][1]] = 1
    
    tensor_C = Variable(torch.from_numpy(tensor_C.astype("float32")).cuda())
    return tensor_C

tensor_C = design_tensor_C()

class OS_Linear(Module):

    def __init__(self, in_features, out_features, bias=True):
        super(OS_Linear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input):
        if input.dim() == 2 and self.bias is not None:

            output = input.matmul(self.weight.t()* tensor_C)

            if self.bias is not None:
                output += self.bias

            return output


    def __repr__(self):
        return self.__class__.__name__ + ' (' \
            + str(self.in_features) + ' -> ' \
            + str(self.out_features) + ')'



cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class OS_VGG(nn.Module):
    def __init__(self, vgg_name):
        super(OS_VGG, self).__init__()
        self.features = self._make_layers(cfg[vgg_name])
        self.classifier = nn.Sequential(
            nn.Linear(512,256),
            OS_Linear(256, 100)
        )
    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)

# Model


print('==> Building model..')

net = OS_VGG('VGG16')

if use_cuda:
    net.cuda()

    cudnn.benchmark = True

criterion = nn.CrossEntropyLoss()

def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    idx = 0
    

    for batch_idx, (inputs, targets) in enumerate(trainloader):
        idx = batch_idx
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        optimizer.zero_grad()
        inputs, targets = Variable(inputs), Variable(targets)
        outputs = net(inputs)


        loss = criterion(outputs, targets)


        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum().item()


    train_acc = 100.*correct/total
    train_loss = train_loss/(idx+1)
    logging.info('Iteration %d, train_acc = %.5f,train_loss = %.6f' % (epoch, train_acc,train_loss))


def test(epoch):
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    idx = 0
    for batch_idx, (inputs, targets) in enumerate(testloader):
        with torch.no_grad():
            idx = batch_idx
            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
            inputs, targets = Variable(inputs), Variable(targets)
            outputs = net(inputs)


            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += predicted.eq(targets.data).cpu().sum().item()


    test_acc = 100.*correct/total
    test_loss = test_loss/(idx+1)
    logging.info('Iteration %d, test_acc = %.4f,test_loss = %.4f' % (epoch, test_acc,test_loss))
    return test_acc


def cosine_anneal_schedule(t):
    cos_inner = np.pi * (t % (nb_epoch  ))  # t - 1 is used when t has 1-based indexing.
    cos_inner /= (nb_epoch )
    cos_out = np.cos(cos_inner) + 1
    return float(args.lr / 2 * cos_out)



optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)


max_val_acc = 0
for epoch in range(nb_epoch):
    lr = cosine_anneal_schedule(epoch)
    for param_group in optimizer.param_groups:
        print(param_group['lr'])
        param_group['lr'] = lr
    train(epoch)
    test_acc = test(epoch)
    
    if test_acc >max_val_acc:
        max_val_acc = test_acc
    print("max_val_acc", max_val_acc)








#!/usr/bin/env python
# -*- coding: utf-8 -*-

import time
import os
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torchvision
from copy import deepcopy
import torchvision.transforms as transforms
from torch.utils.data.dataset import Subset
from PreResNet_cifar import ResNet18
import argparse


# Mix-up data augmentation
def mixup_data(x, y, alpha=1.0, use_cuda=True):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


# Loss function of mix-up augmented data
def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def get_dataset(dataset):
    global trainset
    if dataset == "CIFAR10":
        # CIFAR meta
        mean = [0.4914, 0.4822, 0.4465]
        std = [0.2023, 0.1994, 0.2010]

        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

        datapath = os.path.abspath('..') + '/scratch/jingyi/cifar/'
        print(datapath)
        trainset = torchvision.datasets.CIFAR10(root=datapath, train=True, transform=train_transform, download=True)
        testset = torchvision.datasets.CIFAR10(root=datapath, train=False, transform=test_transform)
    return trainset, testset


def add_noise_CIFAR10(dataset, noise_type, noise_level):
    # Add noise
    torch.manual_seed(index)
    np.random.seed(index)
    y_train = np.array(dataset.targets)
    y_train_noisy = np.array(dataset.targets)
    if noise_type == "syn":
        probs_to_change = np.random.randint(0, 100, (len(y_train_noisy),))
        idx_to_change = np.where(probs_to_change >= (100.0 - noise_level * 100))[0]
        y_train_noisy[idx_to_change] = np.random.randint(0, 10, (len(idx_to_change),))

    elif noise_type == "asyn":
        for i in range(10):
            indices = np.where(y_train == i)[0]
            num_noise = int(noise_level * len(indices))
            idx_to_change = np.random.choice(indices, num_noise, replace=False)
            # truck -> automobile
            if i == 9:
                y_train_noisy[idx_to_change] = [1] * num_noise
            # bird -> airplane
            if i == 2:
                y_train_noisy[idx_to_change] = [0] * num_noise
            # cat -> dog
            if i == 3:
                y_train_noisy[idx_to_change] = [5] * num_noise
            # dog -> cat
            if i == 5:
                y_train_noisy[idx_to_change] = [3] * num_noise
            # deer -> horse
            if i == 4:
                y_train_noisy[idx_to_change] = [7] * num_noise
    return np.array(y_train_noisy)


def main(lr, noise_level, noise_type, mixup, recordpath, index, dataset="CIFAR10"):
    seed = index
    torch.cuda.manual_seed_all(seed)  # GPU seed
    np.random.seed(seed)
    torch.manual_seed(seed)

    if not os.path.exists(recordpath):
        os.makedirs(recordpath)

    trainset, testset = get_dataset(dataset)

    y_train = np.array(trainset.targets)
    y_test = np.array(testset.targets)

    n_train = len(y_train)
    n_test = len(y_test)

    # Add noise
    y_train_noisy = add_noise_CIFAR10(trainset, noise_type, noise_level)
    real_pi = np.mean(y_train == y_train_noisy)
    p_index = np.where(y_train == y_train_noisy)[0]
    n_index = np.where(y_train != y_train_noisy)[0]
    print(real_pi)
    trainset.targets = y_train_noisy
    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=1)
    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=1)

    milestones = [30, 50, 80]
    net = ResNet18(num_classes=10).to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.1)
    criterion = nn.CrossEntropyLoss()
    f = open(recordpath + "/%d.txt" % index, "a+")
    f.write("real pi: %.3f \n" % real_pi)
    for epoch in range(epoch_nums):
        net.train()
        loss_sum = 0
        start_time = time.time()
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            if mixup:
                inputs, labels_a, labels_b, lam = mixup_data(inputs, labels, 1)
                inputs, targets_a, targets_b = map(Variable, (inputs, labels_a, labels_b))
                optimizer.zero_grad()
                outputs = net(inputs)
                loss = mixup_criterion(criterion, outputs, labels_a, labels_b, lam)
            else:
                optimizer.zero_grad()
                outputs = net(inputs)
                loss = criterion(outputs, labels)

            loss.backward()
            loss_sum += loss.item()
            optimizer.step()
        scheduler.step()

        net.eval()
        with torch.no_grad():
            test_acc = 0.0

            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                outputs = net(x)
                _, predicted = torch.max(outputs.data, 1)
                test_acc += (predicted == y).sum()

        f.write('epoch: %d, train loss: %.03f, test acc: %.03f, time cost: %.1f sec \n' % (
            epoch, loss_sum / len(train_loader), test_acc.item() / n_test, time.time() - start_time))
        f.flush()

    f.close()


parser = argparse.ArgumentParser(description='Train Baseline for CIFAR-10')
parser.add_argument('--batch_size', type=int, default=128,
                    help='input batch size for training (default: 128)')
parser.add_argument('--epoch_nums', type=int, default=100,
                    help='number of epochs (default: 100)')
parser.add_argument('--index', type=int, default=0,
                    help='index of experiment (also used as seed)')
parser.add_argument('--learning_rate', type=float, default=0.05,
                    help='learning rate for training (default: 0.05)')
parser.add_argument('--noise_type', type=str, default='syn',
                    help='type of noise, ("syn": symmetric noise; "asyn": asymmetric noise (default: "syn")')
parser.add_argument('--noise_level', type=float, default=0.3,
                    help='noise level (default: 0.3)')
parser.add_argument('--mixup', dest='mixup', action='store_true', default=False)

args = parser.parse_args()

batch_size = args.batch_size
epoch_nums = args.epoch_nums
learning_rate = args.learning_rate
noise_type = args.noise_type
noise_level = args.noise_level
mixup = args.mixup
index = args.index

device = torch.device("cuda")


if __name__ == '__main__':
    recordpath = "record/CIFAR10/baseline/{noise_type}_{noise_level}_lr_{lr}".format(noise_type=noise_type, noise_level=noise_level, lr=learning_rate)
    if mixup:
        recordpath += "_mixup"

    main(learning_rate, noise_level, noise_type, mixup, recordpath, index)

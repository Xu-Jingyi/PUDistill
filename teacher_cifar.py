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


def get_augmented_clean_set(clean_data_ratio, threshold, add_criterion, lr=0.01):
    dirs = "record/CIFAR10/generate_clean_set/clean_ratio_{clean_data_ratio}_threshold_{threshold}_criterion_{add_criterion}_lr_{learning_rate}".format(
        clean_data_ratio=clean_data_ratio, threshold=threshold, add_criterion=add_criterion, learning_rate=lr)
    add_index = []
    add_num = []
    for label in range(10):
        training_file = open(dirs + '/training_label%d.txt' % label)
        lines = training_file.readlines()
        training_precision = '['
        for line in lines:
            training_precision += line.replace('\n', '').replace(' ', ',')
        training_precision += ']'
        training_precision = eval(
            training_precision.replace('][', '], [').replace(',,,,', ',').replace(',,,', ',').replace(',,', ',').replace('[,', '['))
        add_index.append(training_precision[index])
        add_num.append(len(add_index[-1]))
    return add_index, min(add_num)


def main(clean_data_ratio, add_criterion, threshold, learning_rate, batch_size, epoch_nums, index, recordpath,
         modelpath, mixup=False, entropy_reg=False, beta=0, bagging=False):
    seed = 99
    torch.cuda.manual_seed_all(seed)  # GPU seed
    np.random.seed(seed)
    torch.manual_seed(seed)

    if not os.path.exists(recordpath):
        os.makedirs(recordpath)
    if not os.path.exists(modelpath):
        os.makedirs(modelpath)

    trainset, testset = get_dataset(dataset)

    y_train = np.array(trainset.targets)
    y_train_unlabel = np.array(trainset.targets)
    y_test = np.array(testset.targets)

    n_train = len(y_train)
    n_test = len(y_test)

    # Generate clean dataset
    clean_index = []
    clean_data_size = int(n_train * clean_data_ratio / 10)
    for i in range(10):
        positive_index = list(np.where(y_train == i)[0])
        clean_index = np.append(clean_index, np.random.choice(positive_index, clean_data_size, replace=False)).astype(
            int)
    noisy_index = list(set(range(n_train)) - set(clean_index))
    y_train_unlabel[noisy_index] = [-1] * len(noisy_index)
    trainset.targets = y_train_unlabel

    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=1)
    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=1)

    # Get additional clean data indices
    add_index, bootstrap_size = get_augmented_clean_set(clean_data_ratio, threshold, add_criterion)

    # Generate indices of bootstrap subset to train the teacher model
    train_index = clean_index
    y_slice = sum([[i] * clean_data_size for i in range(10)], [])
    for label in range(10):
        train_index = np.concatenate([train_index, add_index[label]])
        y_slice += [label] * len(add_index[label])

    bootstrap_train_set = Subset(trainset, train_index)
    train_loader_bootstrap = torch.utils.data.DataLoader(
        dataset=bootstrap_train_set,
        batch_size=batch_size,
        shuffle=True,
    )

    y_train_new = deepcopy(y_train_unlabel)
    y_train_new[train_index] = y_slice
    trainset.targets = y_train_new

    # Create model
    milestones = [30, 50, 80]
    net = ResNet18(num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.1)

    f = open(recordpath + "/%d.txt" % index, "a+")
    model_name = modelpath + '/%d.pkl' % index

    best_test_acc = 0
    for epoch in range(epoch_nums):
        if bagging:
            # Generate indices of bootstrap subset to train the teacher model
            train_index = clean_index
            y_slice = sum([[i] * clean_data_size for i in range(10)], [])
            for label in range(10):
                index_bootstrap = np.random.choice(add_index[label], bootstrap_size, replace=False)
                train_index = np.concatenate([train_index, index_bootstrap])
                y_slice += [label] * bootstrap_size

            bootstrap_train_set = Subset(trainset, train_index)
            train_loader_bootstrap = torch.utils.data.DataLoader(
                dataset=bootstrap_train_set,
                batch_size=batch_size,
                shuffle=True,
            )

            y_train_new = deepcopy(y_train_unlabel)
            y_train_new[train_index] = y_slice
            trainset.targets = y_train_new
            print('percentage of correct labels:', np.mean(y_train_new[train_index] == y_train[train_index]))

        net.train()
        loss_sum = 0
        start_time = time.time()
        for inputs, labels in train_loader_bootstrap:
            inputs, labels = inputs.to(device), labels.to(device)
            labels = labels.long()
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
            if entropy_reg:
                s = F.softmax(outputs, dim=1)
                L_entropy = -torch.mean(torch.sum(F.log_softmax(outputs) * s, axis=1))
                loss = loss + beta * L_entropy
            loss.backward()
            loss_sum += loss.item()

            optimizer.step()
        scheduler.step()

        net.eval()
        with torch.no_grad():
            test_acc = 0.0
            total = 0

            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                outputs = net(x)
                _, predicted = torch.max(outputs.data, 1)
                total += y.shape[0]
                acc = (predicted == y).sum()
                test_acc += acc

            if (test_acc.item() / total) > best_test_acc:
                best_test_acc = test_acc.item() / total
                torch.save(net.state_dict(), model_name)

        f.write('epoch: %d, precision: %.03f, train loss: %.03f, test acc: %.03f, time cost: %.1f sec \n' % (
            epoch, np.mean(y_train_new[train_index] == y_train[train_index]), loss_sum,
            test_acc.item() / total, time.time() - start_time))
        f.flush()
    f.close()


parser = argparse.ArgumentParser(description='Train Teacher Models for CIFAR-10')
parser.add_argument('--batch_size', type=int, default=128,
                    help='input batch size for training (default: 128)')
parser.add_argument('--epoch_nums', type=int, default=100,
                    help='number of epochs (default: 100)')
parser.add_argument('--clean_data_ratio', type=float, default=0.03,
                    help='size of the original clean set (default: 200)')
parser.add_argument('--criterion', type=int, default=19,
                    help='criterion to augment clean set (default: 19)')
parser.add_argument('--threshold', type=float, default=0.9,
                    help='threshold to classify a sample as positive (default: 0.9)')
parser.add_argument('--n', type=int, default=5,
                    help='number of teacher model (default: 5)')
parser.add_argument('--learning_rate', type=float, default=0.03,
                    help='learning rate for training (default: 0.03)')
parser.add_argument('--mixup', dest='mixup', action='store_true', default=False)
parser.add_argument('--entropy_reg', dest='entropy_reg', action='store_true', default=False)
parser.add_argument('--beta', type=float, default=0.8,
                    help='weight of entropy regularization (default: 0.8)')
parser.add_argument('--bagging', dest='bagging', action='store_true', default=False)

args = parser.parse_args()

dataset = "CIFAR10"

batch_size = args.batch_size
epoch_nums = args.epoch_nums
n = args.n
clean_data_ratio = args.clean_data_ratio
add_criterion = args.criterion
threshold = args.threshold
learning_rate = args.learning_rate
mixup = args.mixup
entropy_reg = args.entropy_reg
beta = args.beta
bagging = args.bagging

device = torch.device("cuda")

if __name__ == '__main__':
    for index in range(n):
        recordpath = "record/CIFAR10/teacher/clean_ratio_{clean_data_ratio}_threshold_{threshold}_criterion_{add_criterion}_lr_{learning_rate}".format(
            clean_data_ratio=clean_data_ratio, threshold=threshold, add_criterion=add_criterion, learning_rate=learning_rate)
        modelpath = "model/CIFAR10/teacher/clean_ratio_{clean_data_ratio}_threshold_{threshold}_criterion_{add_criterion}_lr_{learning_rate}".format(
            clean_data_ratio=clean_data_ratio, threshold=threshold, add_criterion=add_criterion, learning_rate=learning_rate)
        if mixup:
            recordpath += "_mixup"
            modelpath += "_mixup"
        if entropy_reg:
            recordpath += "_entropy_" + str(beta)
            modelpath += "_entropy_" + str(beta)
        if bagging:
            recordpath += "_bagging"
            modelpath += "_bagging"
        main(clean_data_ratio, add_criterion, threshold, learning_rate, batch_size, epoch_nums, index, recordpath,
             modelpath, mixup, entropy_reg, beta, bagging)

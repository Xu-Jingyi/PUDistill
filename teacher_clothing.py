#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on 2019/10/12 14:28

@author: Jingyi
"""

import time
import os
import numpy as np
import torch
import torch.nn as nn
from ResNet_clothing import ResNet50
from torch.autograd import Variable
import dataloader_clothing
import torch.nn.functional as F

np.set_printoptions(threshold=np.inf)

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


def main(lr, recordpath, modelpath, epoch_nums, bagging, mixup, index):
    seed = 99
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if not os.path.exists(recordpath):
        os.makedirs(recordpath)
    if not os.path.exists(modelpath):
        os.makedirs(modelpath)
    if mixup:
        modelname = modelpath + 'lr_%.4f_%d_mixup.pkl'%(lr, index)
        recordname = recordpath + "lr_%.4f_%d_mixup.txt" % (lr, index)
    else:
        modelname = modelpath + 'lr_%.4f_%d.pkl'%(lr, index)
        recordname = recordpath + "lr_%.4f_%d.txt" % (lr, index)

    loader = dataloader_clothing.clothing_dataloader(batch_size=batch_size, num_workers=1, shuffle=True)
    train_loader, train_loader_trace, valid_loader, test_loader = loader.run()
    clean_train_list = loader.clean_train_list()
    clean_train_list = sum(clean_train_list, [])
    y_train = np.array(train_loader.dataset.train_labels)
    y_valid = np.array(valid_loader.dataset.val_labels)
    y_test = np.array(test_loader.dataset.test_labels)
    num_train = len(y_train)
    num_valid = len(y_valid)
    num_test = len(y_test)

    '''
    Get augmented clean data index.
    The file is generated based on the clean set augmentation step, 
    the detailed process is shown in code for CIFAR10 (teacher_cifar.py, line 70 - line 86).
    '''
    f = open("record/additional_clean_set_idx_clothing.txt")
    lines = f.readlines()
    add_index = ""
    for line in lines:
        add_index += line.strip()
    add_index = eval(add_index2)

    add_num = []
    for label in range(14):
        add_num.append(len(add_index[label]))
    bootstrap_size = min(add_num)

    # Generate indices of bootstrap subset to train the teacher model
    train_index = clean_train_list
    y_slice = list(y_train[clean_train_list])
    for label in range(14):
        train_index = np.concatenate([train_index, add_index[label]])
        y_slice += [label] * len(add_index[label])

    train_loader_bootstrap = loader.subset_train_loader(train_index, y_slice)

    net = ResNet50(num_classes=1000, pretrained=True)
    num_ftrs = net.fc.in_features
    net.fc = nn.Linear(num_ftrs, 14)
    net = net.to(device)
    torch.save(net.state_dict(), modelname)

    milestones = [3]
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.1)

    f = open(recordname, 'a+')
    best_val_acc = 0
    for epoch in range(epoch_nums):
        torch.cuda.empty_cache()
        net.train()
        batch = 0

        if bagging:
            # Generate indices of bootstrap subset to train the teacher model
            train_index = clean_train_list
            y_slice = list(y_train[clean_train_list])
            for label in range(14):
                train_index = np.concatenate(
                    [train_index, np.random.choice(add_index[label], bootstrap_size, replace=False)])
                y_slice += [label] * bootstrap_size
            train_loader_bootstrap = loader.subset_train_loader(train_index, y_slice)

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
            if entropy:
                s = F.softmax(outputs, dim=1)
                L_entropy = -torch.mean(torch.sum(F.log_softmax(outputs) * s, axis=1))
                loss = loss + 0.8 * L_entropy
            loss.backward()
            optimizer.step()
            batch += 1

            if batch % 100 == 0:
                net.eval()
                with torch.no_grad():
                    test_acc = 0.0
                    valid_acc = 0.0

                    for x, y in test_loader:
                        x, y = x.to(device), y.to(device)
                        outputs = net(x)
                        _, predicted = torch.max(outputs.data, 1)
                        acc = (predicted == y).sum()
                        test_acc += acc

                    for x, y in valid_loader:
                        x, y = x.to(device), y.to(device)
                        outputs = net(x)
                        _, predicted = torch.max(outputs.data, 1)
                        acc = (predicted == y).sum()
                        valid_acc += acc

                if (valid_acc.item() / num_valid) > best_val_acc:
                    best_val_acc = valid_acc.item() / num_valid
                    torch.save(net.state_dict(), modelname)
                f.write("epoch: %d, batch: %d, test accuracy: %.5f, valid accuracy: %.5f, time: %.1f \n" % (
                epoch, batch, test_acc / num_test, valid_acc / num_valid, time.time() - start_time))
                f.flush()

        scheduler.step()


parser = argparse.ArgumentParser(description='Train Teacher Models for Clothing1M')
parser.add_argument('--batch_size', type=int, default=32,
                    help='input batch size for training (default: 32)')
parser.add_argument('--epoch_nums', type=int, default=5,
                    help='number of epochs (default: 5)')
parser.add_argument('--threshold', type=float, default=0.95,
                    help='decision threshold of binary classifiers (default: 0.95)')
parser.add_argument('--add_criterion', type=int, default=10,
                    help='criterion to move an unsure sample to clean set (default: 10)')
parser.add_argument('--n', type=int, default=5,
                    help='number of teacher model (default: 5)')
parser.add_argument('--learning_rate', type=float, default=0.003,
                    help='learning rate for training (default: 0.003)')
parser.add_argument('--mixup', dest='mixup', action='store_true', default=False)
parser.add_argument('--entropy_reg', dest='entropy_reg', action='store_true', default=False)
parser.add_argument('--beta', type=float, default=0.8,
                    help='weight of entropy regularization (default: 0.8)')
parser.add_argument('--bagging', dest='bagging', action='store_true', default=False)

args = parser.parse_args()

device = torch.device("cuda")
batch_size = args.batch_size
epoch_nums = args.epoch_nums
n = args.n
add_criterion = args.criterion
threshold = args.threshold
lr = args.learning_rate
mixup = args.mixup
entropy = args.entropy_reg
beta = args.beta
bagging = args.bagging

if __name__ == '__main__':
    recordpath = "record/clothing/teacher/threshold_{threshold}_criterion_{add_criterion}/.format(threshold=threshold, add_criterion=add_criterion)"
    modelpath = "model/clothing/teacher/threshold_{threshold}_criterion_{add_criterion}/.format(threshold=threshold, add_criterion=add_criterion)"

    if not bagging:
        recordpath += "whole_"
        modelpath += "whole_"
    if entropy:
        recordpath += "entropy_"
        modelpath += "entropy_"
    for index in range(n):
        main(lr, recordpath, modelpath, epoch_nums, bagging, mixup, index)


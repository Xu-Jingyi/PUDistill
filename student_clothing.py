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
import argparse
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


# Cross-entropy loss function
def CELoss(predicted, target):
    return -(target * torch.log_softmax(predicted, dim=1)).sum(dim=1).mean()


def main(learning_rate, label_type, eta, student_lambda, mixup_stu, entropy_reg_stu, beta, recordpath, n, teacherpath, index):
    seed = 99
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if not os.path.exists(recordpath):
        os.makedirs(recordpath)
    if not os.path.exists(modelpath):
        os.makedirs(modelpath)
    if mixup_stu:
        modelname = modelpath + '/%d_mixup.pkl' % (index)
        recordname = recordpath + "/%d_mixup.txt" % (index)
    else:
        modelname = modelpath + '/%d.pkl' % (index)
        recordname = recordpath + "/%d.txt" % (index)

    loader = dataloader_clothing.clothing_dataloader(batch_size=batch_size, num_workers=1, shuffle=True)
    train_loader, train_loader_trace, valid_loader, test_loader = loader.run()
    clean_train_list = loader.clean_train_list()
    clean_train_list = sum(clean_train_list, [])
    y_train = np.array(train_loader.dataset.train_labels)
    y_pseudo = np.array(train_loader.dataset.train_labels)
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

    for label in range(14):
        y_pseudo[add_index[label]] = [label] * len(add_index[label])

    # Get the output of teacher model
    pred_teacher = np.zeros((n, num_train, 14))
    for i in range(n):
        if mixup:
            model_name = teacherpath + 'lr_0.0010_%d_mixup.pkl'%(i)
        else:
            model_name = teacherpath + 'lr_0.0010_%d.pkl'%(i)
        net = ResNet50(num_classes=14).to(device)
        net.load_state_dict(torch.load(model_name))
        start_index = 0
        with torch.no_grad():
            for x, y in train_loader_trace:
                x, y = x.to(device), y.to(device)
                outputs = net(x)
                outputs_ = F.softmax(outputs, dim=1)
                pred_teacher[i, start_index:start_index + len(y), :] = outputs_.cpu().numpy()
                start_index += len(y)
    pred_teacher_aver = np.mean(pred_teacher, axis=0)
    pred_teacher_max = np.max(pred_teacher_aver, axis=1)

    # Generate pseudo label
    index_update_teacher = np.where(pred_teacher_max >= eta)[0]
    if label_type == 'hard_bootstrap':
        y_pseudo = np.eye(14)[y_pseudo]
        y_pseudo[index_update_teacher] = student_lambda * np.eye(10)[
            np.argmax(pred_teacher_aver[index_update_teacher], axis=1)] + (1 - student_lambda) * y_pseudo[index_update_teacher]
    elif label_type == 'soft_bootstrap':
        y_pseudo = np.eye(14)[y_pseudo]
        y_pseudo[index_update_teacher] = student_lambda * pred_teacher_aver[index_update_teacher] \
                                         + (1 - student_lambda) * y_pseudo[index_update_teacher]

    train_loader.dataset.train_labels = y_pseudo

    net = ResNet50(num_classes=1000, pretrained=True)
    num_ftrs = net.fc.in_features
    net.fc = nn.Linear(num_ftrs, 14)
    net = net.to(device)
    torch.save(net.state_dict(), modelname)

    milestones = [3, 6]
    criterion = CELoss
    optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.1)

    f = open(recordname, 'a+')
    best_val_acc = 0
    for epoch in range(epoch_nums):
        torch.cuda.empty_cache()
        net.train()
        batch = 0

        start_time = time.time()
        for inputs, labels in train_loader:
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

            if entropy_reg_stu:
                s = F.softmax(outputs, dim=1)
                L_entropy = -torch.mean(torch.sum(F.log_softmax(outputs) * s, axis=1))
                loss = loss + beta * L_entropy

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


parser = argparse.ArgumentParser(description='Train Student Models for Clothing1M')
parser.add_argument('--batch_size', type=int, default=32,
                    help='input batch size for training (default: 32)')
parser.add_argument('--epoch_nums', type=int, default=10,
                    help='number of epochs (default: 10)')
parser.add_argument('--n', type=int, default=5,
                    help='number of teacher model (default: 5)')
parser.add_argument('--learning_rate', type=float, default=0.001,
                    help='learning rate for training (default: 0.001)')
parser.add_argument('--label_type', type=str, default='soft_bootstrap',
                    help='type of pseudo label, "soft_bootstrap" or "hard_bootstrap" (default: "soft_bootstrap")')
parser.add_argument('--student_lambda', type=float, default=0.8,
                    help='weight of the teacher output (default: 0.8)')
parser.add_argument('--criterion', type=int, default=10,
                    help='criterion to augment clean set (default: 10)')
parser.add_argument('--threshold', type=float, default=0.95,
                    help='threshold to choose the training data with high confidence (default: 0.95)')
parser.add_argument('--mixup', dest='mixup', action='store_true', default=False)
parser.add_argument('--mixup_stu', dest='mixup_stu', action='store_true', default=False)
parser.add_argument('--entropy_reg_stu', dest='entropy_reg_stu', action='store_true', default=False)
parser.add_argument('--beta', type=float, default=0.8,
                    help='weight of entropy regularization (default: 0.8)')
parser.add_argument('--eta', type=float, default=0.8,
                    help='threshold to choose the subset to assign pseudo label (default: 0.8)')

args = parser.parse_args()

batch_size = args.batch_size
epoch_nums = args.epoch_nums
n = args.n
learning_rate = args.learning_rate
label_type = args.label_type
student_lambda = args.student_lambda
add_criterion = args.criterion
threshold = args.threshold
mixup = args.mixup
mixup_stu = args.mixup_stu
entropy_reg_stu = args.entropy_reg_stu
beta = args.beta
eta = args.eta

device = torch.device("cuda")

if __name__ == '__main__':
    recordpath = "record/clothing/student/threshold_{threshold}_criterion_{add_criterion}_eta_{eta}_lambda_{student_lambda}_lr_{lr}".format(
        threshold=threshold, add_criterion=add_criterion, eta=eta, student_lambda=student_lambda, lr=learning_rate)
    modelpath = "model/clothing/student/threshold_{threshold}_criterion_{add_criterion}_eta_{eta}_lambda_{student_lambda}_lr_{lr}".format(
        threshold=threshold, add_criterion=add_criterion, eta=eta, student_lambda=student_lambda, lr=learning_rate)
    if mixup:
        recordpath += "_mixup"
        modelpath += "_mixup"
    if mixup_stu:
        recordpath += "_mixup_stu"
        modelpath += "_mixup_stu"
    if entropy_reg_stu:
        recordpath += "_entropy_" + str(beta)
        modelpath += "_entropy_" + str(beta)

    teacherpath = "model/clothing//teacher/threshold_{threshold}_criterion_{add_criterion}/".format(
        threshold=threshold, add_criterion=add_criterion)

    index = 0
    main(learning_rate, label_type, eta, student_lambda, mixup_stu, entropy_reg_stu, beta, recordpath, n, teacherpath, index)

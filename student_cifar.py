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


def get_augmented_clean_set(clean_data_dir):
    add_index = []
    add_num = []
    for label in range(10):
        training_file = open(clean_data_dir + '/training_label%d.txt' % label)
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


def add_noise_CIFAR10(dataset, noise_type, noise_level, clean_index, clean_data_ratio):
    # Add noise
    torch.manual_seed(13)
    np.random.seed(13)
    y_train = np.array(dataset.targets)
    y_train_noisy = np.array(dataset.targets)
    noise_level = noise_level / (1 - clean_data_ratio)
    if noise_type == "syn":
        probs_to_change = np.random.randint(0, 100, (len(y_train_noisy),))
        idx_to_change = np.where(probs_to_change >= (100.0 - noise_level * 100))[0]
        idx_to_change = list(set(idx_to_change) - set(clean_index))
        y_train_noisy[idx_to_change] = np.random.randint(0, 10, (len(idx_to_change),))

    elif noise_type == "asyn":
        for i in range(10):
            indices = np.where(y_train == i)[0]
            num_noise = int(noise_level * len(indices))
            idx_to_change = np.random.choice(indices, num_noise, replace=False)
            idx_to_change = list(set(idx_to_change) - set(clean_index))
            num_noise = len(idx_to_change)
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


# Cross-entropy loss function
def CELoss(predicted, target):
    return -(target * torch.log_softmax(predicted, dim=1)).sum(dim=1).mean()


def main(lr, clean_data_ratio, noise_level, noise_type, label_type, eta, student_lambda, mixup,
         entropy_reg, recordpath, n, teacherpath, clean_data_dir, index, dataset="CIFAR10"):
    seed = 99
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

    # Generate clean dataset
    clean_index = []
    clean_data_size = int(n_train * clean_data_ratio / 10)
    for i in range(10):
        positive_index = list(np.where(y_train == i)[0])
        clean_index = np.append(clean_index, np.random.choice(positive_index, clean_data_size, replace=False)).astype(
            int)
    noisy_index = list(set(range(n_train)) - set(clean_index))

    # Add noise
    y_train_noisy = add_noise_CIFAR10(trainset, noise_type, noise_level, clean_index, clean_data_ratio)
    real_pi = np.mean(y_train == y_train_noisy)
    p_index = np.where(y_train == y_train_noisy)[0]
    n_index = np.where(y_train != y_train_noisy)[0]
    print(real_pi)

    # Get additional clean data indices and relabel the noisy training set
    add_index, _ = get_augmented_clean_set(clean_data_dir)
    for label in range(10):
        y_train_noisy[add_index[label]] = [label] * len(add_index[label])
    pi_relabeled_by_clean = np.mean(y_train == y_train_noisy)
    trainset.targets = y_train_noisy
    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=1)
    train_loader_trace = DataLoader(trainset, batch_size=batch_size, shuffle=False, num_workers=1)
    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=1)

    # Get the output of teacher model
    pred_teacher = np.zeros((n, n_train, 10))
    for i in range(n):
        model_name = teacherpath + '/%d.pkl' % i
        net = ResNet18(num_classes=10).to(device)
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

    milestones = [30, 50, 80]
    net = ResNet18(num_classes=10).to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.1)

    # Generate pseudo label
    index_update_teacher = np.where(pred_teacher_max >= eta)[0]
    y_pseudo = np.array(trainset.targets)
    if label_type == 'hard':
        y_pseudo[index_update_teacher] = np.argmax(pred_teacher_aver[index_update_teacher], axis=1)
        criterion = nn.CrossEntropyLoss()
        pi_relabeled_by_teacher = np.mean(y_train == y_pseudo)
    elif label_type == 'hard_bootstrap':
        y_pseudo = np.eye(10)[y_pseudo]
        y_pseudo[index_update_teacher] = student_lambda * np.eye(10)[
            np.argmax(pred_teacher_aver[index_update_teacher], axis=1)] + (1 - student_lambda) * y_pseudo[index_update_teacher]
        criterion = CELoss
        pi_relabeled_by_teacher = np.mean(y_train == np.argmax(y_pseudo, axis=1))
    elif label_type == 'soft_bootstrap':
        y_pseudo = np.eye(10)[y_pseudo]
        y_pseudo[index_update_teacher] = student_lambda * pred_teacher_aver[index_update_teacher] + (1 - student_lambda) * y_pseudo[index_update_teacher]
        criterion = CELoss
        pi_relabeled_by_teacher = np.mean(y_train == np.argmax(y_pseudo, axis=1))
    trainset.targets = y_pseudo

    f = open(recordpath + "/%d.txt" % index, "a+")
    f.write("real pi: %.3f, pi after relabeling by clean set: %.3f, pi after relabeling by teacher model: %.3f, number of pseudo label: %d \n" % (
            real_pi, pi_relabeled_by_clean, pi_relabeled_by_teacher, len(index_update_teacher)))

    # Training
    for epoch in range(epoch_nums):
        net.train()
        loss_sum = 0
        start_time = time.time()
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            if label_type == 'hard':
                labels = labels.long()
            elif label_type == 'hard_bootstrap' or label_type == 'soft_bootstrap':
                labels = labels.float()

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
                loss = loss + 0.8 * L_entropy

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

        f.write('epoch: %d, train loss: %.03f, test acc: %.05f, time cost: %.1f sec \n' % (
            epoch, loss_sum / len(train_loader), test_acc.item() / n_test, time.time() - start_time))
        f.flush()

    f.close()


parser = argparse.ArgumentParser(description='Train Student Models for CIFAR-10')
parser.add_argument('--batch_size', type=int, default=128,
                    help='input batch size for training (default: 128)')
parser.add_argument('--epoch_nums', type=int, default=100,
                    help='number of epochs (default: 100)')
parser.add_argument('--clean_data_ratio', type=float, default=0.05,
                    help='proportion of the original clean set (default: 0.05)')
parser.add_argument('--n', type=int, default=5,
                    help='number of teacher model (default: 5)')
parser.add_argument('--learning_rate', type=float, default=0.05,
                    help='learning rate for training (default: 0.05)')
parser.add_argument('--noise_type', type=str, default='syn',
                    help='type of noise, ("syn": symmetric noise; "asyn": asymmetric noise (default: "syn")')
parser.add_argument('--noise_level', type=float, default=0.3,
                    help='noise level (default: 0.3)')
parser.add_argument('--label_type', type=str, default='soft',
                    help='type of pseudo label, "soft_bootstrap" or "hard_bootstrap" or "hard" (default: "soft")')
parser.add_argument('--student_lambda', type=float, default=0.8,
                    help='weight of the teacher output (default: 0.8)')
parser.add_argument('--criterion', type=int, default=19,
                    help='criterion to augment clean set (default: 19)')
parser.add_argument('--threshold', type=float, default=0.9,
                    help='threshold to choose the training data with high confidence (default: 0.9)')
parser.add_argument('--mixup', dest='mixup', action='store_true', default=False)
parser.add_argument('--mixup_stu', dest='mixup_stu', action='store_true', default=False)
parser.add_argument('--entropy_reg', dest='entropy_reg', action='store_true', default=False)
parser.add_argument('--entropy_reg_stu', dest='entropy_reg_stu', action='store_true', default=False)
parser.add_argument('--bagging', dest='bagging', action='store_true', default=True)
parser.add_argument('--eta', type=float, default=0.8,
                    help='threshold to choose the subset to assign pseudo label (default: 0.8)')

args = parser.parse_args()

batch_size = args.batch_size
epoch_nums = args.epoch_nums
n = args.n
clean_data_ratio = args.clean_data_ratio
learning_rate = args.learning_rate
noise_type = args.noise_type
noise_level = args.noise_level
label_type = args.label_type
student_lambda = args.student_lambda
add_criterion = args.criterion
threshold = args.threshold
mixup = args.mixup
mixup_stu = args.mixup_stu
entropy_reg = args.entropy_reg
entropy_reg_stu = args.entropy_reg_stu
eta = args.eta
bagging = args.bagging

device = torch.device("cuda")

if __name__ == '__main__':
    recordpath = "record/CIFAR10/student/clean_ratio_{clean_data_ratio}_{noise_type}_{noise_level}_threshold_{threshold}_criterion_{add_criterion}_eta_{eta}".format(
        clean_data_ratio=clean_data_ratio, noise_type=noise_type, noise_level=noise_level,
        threshold=threshold, add_criterion=add_criterion, eta=eta)
    if mixup:
        recordpath += "_mixup"
    if entropy_reg:
        recordpath += "_entropy_0.8"
    recordpath += "/{label_type}_{student_lambda}_lr_{lr}".format(label_type=label_type, student_lambda=student_lambda,
                                                                  lr=learning_rate)
    if mixup_stu:
        recordpath += "_mixup"
    if entropy_reg_stu:
        recordpath += "_entropy_0.8"
    teacherpath = "model/CIFAR10/teacher/clean_ratio_{clean_data_ratio}_threshold_{threshold}_criterion_{add_criterion}_lr_0.03".format(
        clean_data_ratio=clean_data_ratio, threshold=threshold, add_criterion=add_criterion)
    if mixup:
        teacherpath += "_mixup"
    if entropy_reg:
        teacherpath += "_entropy_0.8"
    if bagging:
        teacherpath += "_bagging"

    clean_data_dir = "record/CIFAR10/generate_clean_set/clean_ratio_{clean_data_ratio}_threshold_{threshold}_criterion_{add_criterion}_lr_0.01".format(
        clean_data_ratio=clean_data_ratio, threshold=threshold, add_criterion=add_criterion)

    for index in range(1):
        main(learning_rate, clean_data_ratio, noise_level, noise_type, label_type, eta, student_lambda, mixup_stu,
             entropy_reg_stu, recordpath, n, teacherpath, clean_data_dir, index, dataset="CIFAR10")

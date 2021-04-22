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
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from torch.utils.data.dataset import Subset
from PreResNet_cifar import ResNet18
import argparse

np.set_printoptions(threshold=np.inf)


def get_dataset(dataset):
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
    return (trainset, testset)


def main(threshold, add_criterion, lr, label, recordpath, modelpath):
    seed = 99
    if device == "cuda":
        torch.cuda.manual_seed_all(seed)  # GPU seed
    np.random.seed(seed)
    torch.manual_seed(seed)

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

    train_loader_track = DataLoader(trainset, batch_size=batch_size, shuffle=False, num_workers=1)
    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=1)

    if not os.path.exists(recordpath):
        os.makedirs(recordpath)
    if not os.path.exists(modelpath):
        os.makedirs(modelpath)

    # Initialize N binary classifiers
    for n in range(N_bagging):
        net = ResNet18(num_classes=1).to(device)
        model_name = modelpath + "/model" + str(n) + ".pkl"
        torch.save(net.state_dict(), model_name)
    torch.cuda.empty_cache()
    additional_data_index = []

    for k in range(K_iteration):
        y_pred_train = np.zeros((n_train, N_bagging))
        y_pred_test = np.zeros((n_test, N_bagging))

        # train N binary classifiers for each class
        for n in range(N_bagging):
            torch.cuda.empty_cache()
            print('iteration: %d, label: %d, model: %d' % (k, label, n))
            net = ResNet18(num_classes=1).to(device)
            model_name = modelpath + '/model' + str(n) + '.pkl'
            net.load_state_dict(torch.load(model_name))
            milestones = [20]
            criterion = nn.BCEWithLogitsLoss()
            optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.1)

            ''' 
            generate indices of bootstrap training set
            Positives: original clean set C_i for this class i
                       & additional clean set (if the size is smaller than C_i), 
                         or a bootstrap subset of additional clean set (otherwise)
                      
            Negatives: a bootstrap subset of original clean set C_j (j!=i) for other 9 classes
                       & a bootstrap subset of the remaining noisy set.
            '''
            clean_pos_index = clean_index[label * clean_data_size:(label + 1) * clean_data_size]
            if len(additional_data_index) < clean_data_size:
                noisy_pos_index = additional_data_index
            else:
                noisy_pos_index = np.random.choice(additional_data_index, clean_data_size, replace=False)

            n_pos = len(clean_pos_index) + len(noisy_pos_index)
            n_neg_clean = n_pos // 2
            n_neg_noisy = n_pos - n_neg_clean

            clean_neg_index = np.random.choice(list(set(clean_index) - set(clean_pos_index)), n_neg_clean,
                                               replace=False)
            noisy_neg_index = np.random.choice(list(set(noisy_index) - set(additional_data_index)), n_neg_noisy,
                                               replace=False)

            train_index = np.concatenate((clean_pos_index, noisy_pos_index, clean_neg_index, noisy_neg_index)).astype(
                int)
            # Binary labels
            y_slice = [1] * n_pos + [0] * n_pos

            bootstrap_train_set = Subset(trainset, train_index)
            train_loader_bootstrap = DataLoader(dataset=bootstrap_train_set, batch_size=batch_size, shuffle=True)

            y_train_binary = np.array(trainset.targets)
            y_train_binary[train_index] = y_slice
            trainset.targets = y_train_binary

            for epoch in range(epoch_nums):
                scheduler.step()
                net.train()
                loss_sum = 0
                start_time = time.time()
                for inputs, labels in train_loader_bootstrap:
                    inputs, labels = inputs.to(device), labels.to(device)
                    labels = labels.float()
                    optimizer.zero_grad()
                    outputs = net(inputs).reshape(-1)
                    loss = criterion(outputs, labels)
                    loss_sum += loss.item()
                    loss.backward()
                    optimizer.step()

                net.eval()
                with torch.no_grad():
                    test_acc = 0.0
                    test_acc_pos = 0.0
                    test_acc_neg = 0.0
                    total = 0
                    total_pos = 0.0
                    total_neg = 0.0

                    for x, y in test_loader:
                        x, y = x.to(device), y.to(device)
                        outputs = net(x)
                        outputs = torch.sigmoid(outputs).reshape(-1)
                        test_acc += ((outputs > 0.5) == (y == label)).sum()
                        test_acc_pos += ((outputs[y == label] > 0.5) == (y[y == label] == label)).sum()
                        test_acc_neg += ((outputs[y != label] > 0.5) == (y[y != label] == label)).sum()
                        total += y.shape[0]
                        total_pos += y[y == label].shape[0]
                        total_neg += y[y != label].shape[0]

                print(
                    'epoch: %d, train loss: %.03f, test acc: %.03f, (positive: %.3f, negative: %.3f), time cost: %.1f sec' % (
                    epoch, loss_sum, test_acc.item() / total, test_acc_pos.item() / total_pos,
                    test_acc_neg.item() / total_neg, time.time() - start_time))
            torch.save(net.state_dict(), model_name)

            # Record the prediction
            net.eval()
            pred_train = []
            pred_test = []
            for x, y in train_loader_track:
                x, y = x.to(device), y.to(device)
                outputs = net(x)
                outputs = torch.sigmoid(outputs).reshape(-1)
                pred_train += outputs.tolist()
            y_pred_train[:, n] = pred_train
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                outputs = net(x)
                outputs = torch.sigmoid(outputs).reshape(-1)
                pred_test += outputs.tolist()
            y_pred_test[:, n] = pred_test

        # Obtain additional clean set from training data set
        pos_pred_train = np.sum(y_pred_train > threshold, axis=1)
        add_index = np.where(pos_pred_train >= add_criterion)[0]
        additional_data_index = add_index
        record_file = open(os.path.join(recordpath, 'training_label%d.txt' % label), 'a+')
        record_file.write(str(additional_data_index) + '\n')
        record_file.close()

        # Test the precision score of extracted clean set on test set
        pos_pred_test = np.sum(y_pred_test > threshold, axis=1)
        add_index_test = np.where(pos_pred_test >= add_criterion)[0]

        record_file = open(os.path.join(recordpath, 'test_label%d.txt' % label), 'a+')
        record_file.write(str(add_index_test) + '\n')
        record_file.close()

        evaluate_file = open(os.path.join(recordpath, 'evaluation_label%d.txt' % label), 'a+')
        train_precision = len(list(set(np.where(y_train == label)[0]) & set(additional_data_index))) / max(1, len(
            additional_data_index))
        evaluate_file.write('iteration: %d, training set precision: %.3f, number: %d' % (
        k, train_precision, len(additional_data_index)) + '\n')
        test_precision = len(list(set(np.where(y_test == label)[0]) & set(add_index_test))) / (
            max(1, len(add_index_test)))
        evaluate_file.write(
            'iteration: %d, test set precision: %.3f, number: %d' % (k, test_precision, len(add_index_test)) + '\n')
        evaluate_file.close()
        torch.cuda.empty_cache()


parser = argparse.ArgumentParser(description='Generate Clean Set for CIFAR-10')
parser.add_argument('--clean_data_ratio', type=float, default=0.05,
                    help='ratio of given clean samples (default: 0.05)')
parser.add_argument('--batch_size', type=int, default=64,
                    help='input batch size for training (default: 64)')
parser.add_argument('--epoch_nums', type=int, default=30,
                    help='number of epochs for each binary classifier (default: 30)')
parser.add_argument('--learning_rate', type=float, default=0.01,
                    help='learning rate for training (default: 0.01)')
parser.add_argument('--N_bagging', type=int, default=20,
                    help='number of binary classifiers for each class (default: 20)')
parser.add_argument('--K_iteration', type=int, default=10,
                    help='number of iterations to generate clean set (default: 10)')
parser.add_argument('--threshold', type=float, default=0.9,
                    help='decision threshold of binary classifiers (default: 0.9)')
parser.add_argument('--add_criterion', type=int, default=19,
                    help='criterion to move an unsure sample to clean set (default: 19)')



args = parser.parse_args()

# gpu or cpu
device = torch.device("cuda")

dataset = "CIFAR10"
clean_data_ratio = args.clean_data_ratio
batch_size = args.batch_size
epoch_nums = args.epoch_nums
lr = args.learning_rate
N_bagging = args.N_bagging
K_iteration = args.K_iteration
threshold = args.threshold
add_criterion = args.add_criterion

if __name__ == '__main__':
    recordpath = "record/CIFAR10/generate_clean_set/clean_ratio_{clean_data_ratio}_threshold_{threshold}_criterion_{add_criterion}_lr_{learning_rate}".format(
        clean_data_ratio=clean_data_ratio, threshold=threshold, add_criterion=add_criterion, learning_rate=lr)
    modelpath = "model/CIFAR10/generate_clean_set/clean_ratio_{clean_data_ratio}_threshold_{threshold}_criterion_{add_criterion}_lr_{learning_rate}".format(
        clean_data_ratio=clean_data_ratio, threshold=threshold, add_criterion=add_criterion, learning_rate=lr)
    for label in range(10):
        main(threshold, add_criterion, lr, label, recordpath, modelpath)

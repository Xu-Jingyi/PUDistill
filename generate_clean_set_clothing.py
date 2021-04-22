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
from torch.autograd import Variable
import torch.nn.functional as F
import torchvision
from copy import deepcopy
import torchvision.transforms as transforms
from torch.utils.data.dataset import Subset
from ResNet_clothing import ResNet50
import dataloader_clothing
import argparse

np.set_printoptions(threshold=np.inf)


def train_binary_classifiers(k, label, n, modelpath, additional_data_index, epoch_nums):
    loader = dataloader_clothing.clothing_dataloader(batch_size=batch_size, num_workers=1, shuffle=True)
    train_loader, train_loader_trace, valid_loader, test_loader = loader.run()
    clean_train_list = loader.clean_train_list()
    y_train = train_loader.dataset.train_labels
    y_valid = valid_loader.dataset.val_labels
    y_test = test_loader.dataset.test_labels
    num_train = len(y_train)
    num_valid = len(y_valid)

    if len(additional_data_index) > 0:
        true_positive_index = list(np.where(np.array(y_train) == label)[0])
        TP = len(list(set(additional_data_index) & set(true_positive_index)))
        print(len(additional_data_index), TP / len(additional_data_index))

    clean_index = sum(clean_train_list, [])
    noisy_index = list(set(range(num_train)) - set(clean_index))
    clean_pos_index = clean_train_list[label]
    if len(additional_data_index) < len(clean_pos_index):
        noisy_pos_index = additional_data_index
    else:
        noisy_pos_index = np.random.choice(additional_data_index, len(clean_pos_index), replace=False)

    ''' 
    generate indices of bootstrap training set
    Positives: original clean set C_i for this class i
               & additional clean set (if the size is smaller than C_i), 
                 or a bootstrap subset of additional clean set (otherwise)
              
    Negatives: a bootstrap subset of original clean set C_j (j!=i) for other 9 classes
               & a bootstrap subset of the remaining noisy set.
    '''
    n_pos = len(clean_pos_index) + len(noisy_pos_index)
    n_neg_clean = n_pos // 2
    n_neg_noisy = n_pos - n_neg_clean

    clean_neg_index = np.random.choice(list(set(clean_index) - set(clean_pos_index)), n_neg_clean, replace=False)
    noisy_neg_index = np.random.choice(list(set(noisy_index) - set(additional_data_index)), n_neg_noisy, replace=False)

    train_index = np.concatenate((clean_pos_index, noisy_pos_index, clean_neg_index, noisy_neg_index)).astype(int)
    y_slice = [1] * n_pos + [0] * n_pos

    train_loader_bootstrap = loader.subset_train_loader(train_index, y_slice)

    net = ResNet50(num_classes=1, pretrained=False).to(device)
    model_name = os.path.join(modelpath, 'model%d.pkl' % n)
    net.load_state_dict(torch.load(model_name))
    milestones = [10, 15]
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.1)

    print('iteration: %d, label: %d, model: %d' % (k, label, n))
    for epoch in range(epoch_nums):
        torch.cuda.empty_cache()
        net.train()
        start_time = time.time()
        for inputs, labels in train_loader_bootstrap:
            inputs, labels = inputs.to(device), labels.to(device)
            labels = labels.float()
            optimizer.zero_grad()
            outputs = net(inputs).reshape(-1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        scheduler.step()
        net.eval()
        with torch.no_grad():
            valid_acc = 0.0
            valid_acc_pos = 0.0
            valid_acc_neg = 0.0
            total = 0
            total_pos = 0.0
            total_neg = 0.0

            for x, y in valid_loader:
                x, y = x.to(device), y.to(device)
                outputs = net(x)
                outputs = torch.sigmoid(outputs).reshape(-1)
                valid_acc += ((outputs > 0.5) == (y == label)).sum()
                valid_acc_pos += ((outputs[y == label] > 0.5) == (y[y == label] == label)).sum()
                valid_acc_neg += ((outputs[y != label] > 0.5) == (y[y != label] == label)).sum()
                total += y.shape[0]
                total_pos += y[y == label].shape[0]
                total_neg += y[y != label].shape[0]

        print('epoch: %d, valid acc: %.03f, (positive: %.3f, negative: %.3f), time cost: %.1f sec' % (
            epoch, valid_acc.item() / total, valid_acc_pos.item() / total_pos,
            valid_acc_neg.item() / total_neg, time.time() - start_time))
    torch.save(net.state_dict(), model_name)


def augment_clean_set(N_bagging, path, model_path, threshold, add_criterion, label, k):
    loader = dataloader_clothing.clothing_dataloader(batch_size=batch_size, num_workers=1, shuffle=True)
    train_loader, train_loader_trace, valid_loader, test_loader = loader.run()
    y_train = train_loader.dataset.train_labels
    y_valid = valid_loader.dataset.val_labels
    num_batch = int(len(y_train) / batch_size)

    additional_data_index = []
    additional_data_index_valid = []
    net_list = []

    # Record the prediction
    for n in range(N_bagging):
        torch.cuda.empty_cache()
        net = ResNet50(num_classes=1, pretrained=False).to(device)
        model_name = os.path.join(model_path, 'model%d.pkl' % n)
        net.load_state_dict(torch.load(model_name))

        net_list.append(net)

    start_index = 0
    batch = 0
    for x, y in train_loader_trace:
        batch += 1
        if batch % 100 == 1:
            print("batch %d out of %d" % (batch, num_batch))
        x, y = x.to(device), y.to(device)
        pred_train = np.zeros((len(y), N_bagging))
        for n in range(N_bagging):
            net = net_list[n]
            net.eval()
            outputs = net(x)
            outputs = torch.sigmoid(outputs).reshape(-1)
            pred_train[:, n] = outputs.tolist()

        # Obtain additional clean set from training data set
        pos_pred_train = np.sum(pred_train >= threshold, axis=1)
        add_index = np.where(pos_pred_train >= add_criterion)[0]
        additional_data_index += list(add_index + start_index)

        start_index += len(y)
        torch.cuda.empty_cache()

    record_file = open(os.path.join(path, 'training_label%d_iteration%d.txt' % (label, k)), 'a+')
    record_file.write(str(additional_data_index) + '\n')
    record_file.close()

    start_index = 0
    for x, y in valid_loader:
        x, y = x.to(device), y.to(device)
        pred_valid = np.zeros((len(y), N_bagging))
        for n in range(N_bagging):
            net = net_list[n]
            net.eval()
            outputs = net(x)
            outputs = torch.sigmoid(outputs).reshape(-1)
            pred_valid[:, n] = outputs.tolist()

        # Generate additional data from training data set
        pos_pred_valid = np.sum(pred_valid >= threshold, axis=1)
        add_index = np.where(pos_pred_valid >= add_criterion)[0]
        additional_data_index_valid += list(add_index + start_index)

        start_index += len(y)

    record_file = open(os.path.join(path, 'validation_label%d_iteration%d.txt' % (label, k)), 'a+')
    record_file.write(str(additional_data_index_valid) + '\n')
    record_file.close()

    # Test the precision score of extracted clean set on training set and validation set
    evaluate_file = open(os.path.join(path, 'evaluation_label%d.txt' % label), 'a+')
    train_precision = len(list(set(np.where(np.array(y_train) == label)[0]) & set(additional_data_index))) / (
        max(0.1, len(additional_data_index)))
    evaluate_file.write('iteration: %d, training set precision: %.3f, number: %d' % (
        k, train_precision, len(additional_data_index)) + '\n')
    valid_precision = len(list(set(np.where(np.array(y_valid) == label)[0]) & set(additional_data_index_valid))) / (
        max(0.1, len(additional_data_index_valid)))
    evaluate_file.write('iteration: %d, validation set precision: %.3f, number: %d' % (
        k, valid_precision, len(additional_data_index_valid)) + '\n')
    evaluate_file.close()
    return additional_data_index


def main(threshold, add_criterion, lr, label, pretrained, recordpath, modelpath):
    seed = 99
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if not os.path.exists(recordpath):
        os.makedirs(recordpath)
    if not os.path.exists(modelpath):
        os.makedirs(modelpath)

    # Initialize N binary classifiers
    for n in range(N_bagging):
        if pretrained:
            net = ResNet50(num_classes=1000, pretrained=True).to(device)
            num_ftrs = net.fc.in_features
            net.fc = nn.Linear(num_ftrs, 1)
        else:
            net = ResNet50(num_classes=1, pretrained=False).to(device)
        model_name = os.path.join(modelpath, 'model%d.pkl' % n)
        torch.save(net.state_dict(), model_name)

    additional_data_index = []

    for k in range(K_iteration):
        for n in range(N_bagging):
            train_binary_classifiers(k, label, n, modelpath, additional_data_index, epoch_nums)
            torch.cuda.empty_cache()
        additional_data_index = augment_clean_set(N_bagging, recordpath, modelpath, threshold, add_criterion, label, k)


parser = argparse.ArgumentParser(description='Generate Clean Set for Clothing1M')
parser.add_argument('--batch_size', type=int, default=32,
                    help='input batch size for training (default: 32)')
parser.add_argument('--epoch_nums', type=int, default=20,
                    help='number of epochs for each binary classifier (default: 20)')
parser.add_argument('--learning_rate', type=float, default=0.03,
                    help='learning rate for training (default: 0.03)')
parser.add_argument('--N_bagging', type=int, default=10,
                    help='number of binary classifiers for each class (default: 5)')
parser.add_argument('--K_iteration', type=int, default=5,
                    help='number of iterations to generate clean set (default: 10)')
parser.add_argument('--threshold', type=float, default=0.95,
                    help='decision threshold of binary classifiers (default: 0.95)')
parser.add_argument('--add_criterion', type=int, default=10,
                    help='criterion to move an unsure sample to clean set (default: 10)')

args = parser.parse_args()

device = torch.device("cuda")

batch_size = args.batch_size
epoch_nums = args.epoch_nums
lr = args.learning_rate
N_bagging = args.N_bagging
K_iteration = args.K_iteration
threshold = args.threshold
add_criterion = args.add_criterion

pretrained = True


if __name__ == '__main__':
    for label in range(14):
        recordpath = "record/clothing/generate_clean_set/threshold_{threshold}_criterion_{add_criterion}_lr_{learning_rate}".format(
            threshold=threshold, add_criterion=add_criterion, learning_rate=lr)
        modelpath = "model/clothing/generate_clean_set6/threshold_{threshold}_criterion_{add_criterion}_lr_{learning_rate}/{label}".format(
            threshold=threshold, add_criterion=add_criterion, learning_rate=lr, label=label)
    
        main(threshold, add_criterion, lr, label, pretrained, recordpath, modelpath)

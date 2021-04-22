from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torch.utils.data.dataset import Subset
import torch
import random
import numpy as np
from PIL import Image
import os


class clothing_dataset(Dataset):
    def __init__(self, transform, mode):
        self.train_imgs = []
        self.test_imgs = []
        self.val_imgs = []
        self.noisy_labels = {}
        self.clean_labels = {}
        self.train_labels = []
        self.test_labels = []
        self.val_labels = []
        self.transform = transform
        self.mode = mode
        self.clean_train_list = [[] for i in range(14)]

        datapath = os.path.abspath('..') + '/scratch/jingyi/clothing1M/'
        # datapath = "/dgxdata/jingyi/clothing1M/"

        with open(datapath + 'noisy_label_kv.txt', 'r') as f:
            lines = f.read().splitlines()
        for l in lines:
            entry = l.split()
            img_path = datapath + entry[0]
            self.noisy_labels[img_path] = int(entry[1])

        with open(datapath + 'clean_label_kv.txt', 'r') as f:
            lines = f.read().splitlines()
        for l in lines:
            entry = l.split()
            img_path = datapath + entry[0]
            self.clean_labels[img_path] = int(entry[1])

        with open(datapath + 'clean_train_key_list.txt', 'r') as f:
            lines = f.read().splitlines()
        for i in range(len(lines)):
            l = lines[i]
            img_path = datapath + l
            self.train_imgs.append(img_path)
            target = self.clean_labels[img_path]
            self.clean_train_list[target].append(i)
            self.train_labels.append(target)

        with open(datapath + 'noisy_train_key_list.txt', 'r') as f:
            lines = f.read().splitlines()
        for l in lines:
            img_path = datapath + l
            self.train_imgs.append(img_path)
            target = self.noisy_labels[img_path]
            self.train_labels.append(target)

        with open(datapath + 'clean_test_key_list.txt', 'r') as f:
            lines = f.read().splitlines()
        for l in lines:
            img_path = datapath + l
            self.test_imgs.append(img_path)
            target = self.clean_labels[img_path]
            self.test_labels.append(target)

        with open(datapath + 'clean_val_key_list.txt', 'r') as f:
            lines = f.read().splitlines()
        for l in lines:
            img_path = datapath + l
            self.val_imgs.append(img_path)
            target = self.clean_labels[img_path]
            self.val_labels.append(target)

    def __getitem__(self, index):
        if self.mode == 'train':
            img_path = self.train_imgs[index]
            target = self.train_labels[index]
        elif self.mode == 'test':
            img_path = self.test_imgs[index]
            target = self.test_labels[index]
        elif self.mode == 'val':
            img_path = self.val_imgs[index]
            target = self.val_labels[index]
        image = Image.open(img_path).convert('RGB')
        img = self.transform(image)
        return img, target

    def __len__(self):
        if self.mode == 'train':
            return len(self.train_imgs)
        elif self.mode == 'test':
            return len(self.test_imgs)
        elif self.mode == 'val':
            return len(self.val_imgs)


class clothing_dataloader():
    def __init__(self, batch_size, num_workers, shuffle):
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle

    def run(self):
        transform_train = transforms.Compose([
            transforms.Resize(256),
            # transforms.RandomSizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])  # meanstd transformation

        transform_test = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

        train_dataset = clothing_dataset(transform=transform_train, mode='train')
        test_dataset = clothing_dataset(transform=transform_test, mode='test')
        val_dataset = clothing_dataset(transform=transform_test, mode='val')

        train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle)
        train_loader_trace = DataLoader(
            dataset=train_dataset,
            batch_size=self.batch_size,
            shuffle=False)
        test_loader = DataLoader(
            dataset=test_dataset,
            batch_size=self.batch_size,
            shuffle=False)
        val_loader = DataLoader(
            dataset=val_dataset,
            batch_size=self.batch_size,
            shuffle=False)
        return train_loader, train_loader_trace, val_loader, test_loader

    def clean_train_list(self):
        transform_train = transforms.Compose([
            transforms.Resize(256),
            # transforms.RandomSizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])  # meanstd transformation
        train_dataset = clothing_dataset(transform=transform_train, mode='train')
        return train_dataset.clean_train_list

    def subset_train_loader(self, train_index, y_slice):
        transform_train = transforms.Compose([
            transforms.Resize(256),
            # transforms.RandomSizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])  # meanstd transformation
        train_dataset = clothing_dataset(transform=transform_train, mode='train')
        y_train_new = np.array(train_dataset.train_labels)
        y_train_new[train_index] = y_slice
        train_dataset.train_labels = y_train_new
        subset = Subset(train_dataset, train_index)
        train_loader_subset = torch.utils.data.DataLoader(
            dataset=subset,
            batch_size=self.batch_size,
            shuffle=True)
        return train_loader_subset

    def train_clean_loader(self):
        transform_train = transforms.Compose([
            transforms.Resize(256),
            # transforms.RandomSizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])  # meanstd transformation
        train_dataset = clothing_dataset(transform=transform_train, mode='train')
        train_index = self.clean_train_list()
        subset = Subset(train_dataset, train_index)
        train_loader_subset = torch.utils.data.DataLoader(
            dataset=subset,
            batch_size=32,
            shuffle=True)
        return train_loader_subset

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 15 17:44:59 2022

@author: nvidia
"""
# %% dataset class
import random
import numpy as np
import glob
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image

# %% datasets path
SEM_CASE_1 = 'datasets/simulation_data/SEM/Case_1/*/*.png'
DEPTH_CASE_1A = 'datasets/simulation_data/Depth/Case_1/*/*.png'
DEPTH_CASE_1B = 'datasets/simulation_data/Depth/*/*/*.png'

SEM_CASE_2 = 'datasets/simulation_data/SEM/Case_1/*/*.png'
DEPTH_CASE_2A = 'datasets/simulation_data/Depth/Case_1/*/*.png'
DEPTH_CASE_2B = 'datasets/simulation_data/Depth/*/*/*.png'

SEM_CASE_3 = 'datasets/simulation_data/SEM/Case_1/*/*.png'
DEPTH_CASE_3A = 'datasets/simulation_data/Depth/Case_1/*/*.png'
DEPTH_CASE_3B = 'datasets/simulation_data/Depth/*/*/*.png'

SEM_CASE_4 = 'datasets/simulation_data/SEM/Case_1/*/*.png'
DEPTH_CASE_4A = 'datasets/simulation_data/Depth/Case_1/*/*.png'
DEPTH_CASE_4B = 'datasets/simulation_data/Depth/*/*/*.png'

TRAIN_SEM_PATH = 'datasets/train/SEM/*/*/*.png'
TEST_SEM_PATH = 'datasets/test/SEM/*.png'

CASE_LIST_PATH = 'datasets/train/SEM/*/*/*.png'


# %%
class CustomDataset(Dataset):
    def __init__(self, sem_path_list, depth_path_list):
        self.sem_path_list = sem_path_list
        self.depth_path_list = depth_path_list

    def __getitem__(self, index):
        sem_path = self.sem_path_list[index]
        sem_img = cv2.imread(sem_path)
        if np.shape(np.array(sem_img)) == (72, 48, 3):
            sem_img = sem_img[:, :, 0]
        temp = cv2.GaussianBlur(sem_img, (0, 0), 1)
        for i in range(72):
            for j in range(48):
                if temp[i, j] > 50:
                    temp[i, j] = temp[i, j] * 1.15
                if temp[i, j] < 50:
                    temp[i, j] = temp[i, j] + 4
        sem_img = np.array(temp)
        sem_img = sem_img / 255.
        if self.depth_path_list is not None:
            depth_path = self.depth_path_list[index]
            depth_img = Image.open(depth_path)
            depth_img = np.array(depth_img)
            depth_img = depth_img / 255.
            return torch.Tensor(sem_img).unsqueeze(0), torch.Tensor(depth_img)  # B,C,H,W
        else:
            img_name = sem_path.split('/')[-1]

            return torch.Tensor(sem_img).unsqueeze(0), img_name  # C,B,H,W.unsqueeze(0)

    def __len__(self):
        return len(self.sem_path_list)


# %%
simulation_sem_paths1 = sorted(glob.glob(SEM_CASE_1))
simulation_depth_paths1 = sorted(glob.glob(DEPTH_CASE_1A) + glob.glob(DEPTH_CASE_1B))

simulation_sem_paths2 = sorted(glob.glob(SEM_CASE_2))
simulation_depth_paths2 = sorted(glob.glob(DEPTH_CASE_2A) + glob.glob(DEPTH_CASE_2B))

simulation_sem_paths3 = sorted(glob.glob(SEM_CASE_3))
simulation_depth_paths3 = sorted(glob.glob(DEPTH_CASE_3A) + glob.glob(DEPTH_CASE_3B))

simulation_sem_paths4 = sorted(glob.glob(SEM_CASE_4))
simulation_depth_paths4 = sorted(glob.glob(DEPTH_CASE_4A) + glob.glob(DEPTH_CASE_4B))

test_sem_path_list = sorted(glob.glob(TEST_SEM_PATH))

# %%checking datalen
data_len1 = len(simulation_sem_paths1)
simul_len1 = len(simulation_depth_paths1)
data_len2 = len(simulation_sem_paths2)
simul_len2 = len(simulation_depth_paths2)
data_len3 = len(simulation_sem_paths3)
simul_len3 = len(simulation_depth_paths3)
data_len4 = len(simulation_sem_paths4)
simul_len4 = len(simulation_depth_paths4)

# %%split
random.Random(19991006).shuffle(simulation_sem_paths1)
random.Random(19991006).shuffle(simulation_depth_paths1)
random.Random(19991006).shuffle(simulation_sem_paths2)
random.Random(19991006).shuffle(simulation_depth_paths2)
random.Random(19991006).shuffle(simulation_sem_paths3)
random.Random(19991006).shuffle(simulation_depth_paths3)
random.Random(19991006).shuffle(simulation_sem_paths4)
random.Random(19991006).shuffle(simulation_depth_paths4)

train_sem_paths1 = simulation_sem_paths1[:int(data_len1 * 0.9)]
train_depth_paths1 = simulation_depth_paths1[:int(data_len1 * 0.9)]

val_sem_paths1 = simulation_sem_paths1[int(data_len1 * 0.9):]
val_depth_paths1 = simulation_depth_paths1[int(data_len1 * 0.9):]

train_sem_paths2 = simulation_sem_paths1[:int(data_len2 * 0.9)]
train_depth_paths2 = simulation_depth_paths1[:int(data_len2 * 0.9)]

val_sem_paths2 = simulation_sem_paths1[int(data_len2 * 0.9):]
val_depth_paths2 = simulation_depth_paths1[int(data_len2 * 0.9):]

train_sem_paths3 = simulation_sem_paths1[:int(data_len3 * 0.9)]
train_depth_paths3 = simulation_depth_paths1[:int(data_len3 * 0.9)]

val_sem_paths3 = simulation_sem_paths1[int(data_len3 * 0.9):]
val_depth_paths3 = simulation_depth_paths1[int(data_len3 * 0.9):]

train_sem_paths4 = simulation_sem_paths1[:int(data_len4 * 0.9)]
train_depth_paths4 = simulation_depth_paths1[:int(data_len4 * 0.9)]

val_sem_paths4 = simulation_sem_paths1[int(data_len4 * 0.9):]
val_depth_paths4 = simulation_depth_paths1[int(data_len4 * 0.9):]

# %%make dataset
train_dataset1 = CustomDataset(train_sem_paths1, train_depth_paths1)
train_loader1 = DataLoader(train_dataset1, batch_size=64, shuffle=True, num_workers=4)

val_dataset1 = CustomDataset(val_sem_paths1, val_depth_paths1)
val_loader1 = DataLoader(val_dataset1, batch_size=64, shuffle=False, num_workers=4)

train_dataset2 = CustomDataset(train_sem_paths1, train_depth_paths1)
train_loader2 = DataLoader(train_dataset1, batch_size=64, shuffle=True, num_workers=4)

val_dataset2 = CustomDataset(val_sem_paths1, val_depth_paths1)
val_loader2 = DataLoader(val_dataset1, batch_size=64, shuffle=False, num_workers=4)

train_dataset3 = CustomDataset(train_sem_paths1, train_depth_paths1)
train_loader3 = DataLoader(train_dataset1, batch_size=64, shuffle=True, num_workers=4)

val_dataset3 = CustomDataset(val_sem_paths1, val_depth_paths1)
val_loader3 = DataLoader(val_dataset1, batch_size=64, shuffle=False, num_workers=4)

train_dataset4 = CustomDataset(train_sem_paths1, train_depth_paths1)
train_loader4 = DataLoader(train_dataset1, batch_size=64, shuffle=True, num_workers=4)

val_dataset4 = CustomDataset(val_sem_paths1, val_depth_paths1)
val_loader4 = DataLoader(val_dataset1, batch_size=64, shuffle=False, num_workers=4)

test_dataset = CustomDataset(test_sem_path_list, None)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4)


# %%
class CaseDataset(Dataset):
    def __init__(self, sem_path_list):
        self.sem_path_list = sem_path_list

    def __getitem__(self, index):
        sem_path = self.sem_path_list[index]
        sem_img = cv2.imread(sem_path)
        if np.shape(np.array(sem_img)) == (72, 48, 3):
            sem_img = sem_img[:, :, 0]
        temp = cv2.GaussianBlur(sem_img, (0, 0), 1)
        for i in range(72):
            for j in range(48):
                if temp[i, j] > 50:
                    temp[i, j] = temp[i, j] * 1.15
                if temp[i, j] < 50:
                    temp[i, j] = temp[i, j] + 4
        sem_img = np.array(temp)
        sem_img = sem_img / 255.
        if sem_path.split('/')[-3] == "Case_1" or sem_path.split('/')[-3] == "Depth_110":
            case_name = 0

        if sem_path.split('/')[-3] == "Case_2" or sem_path.split('/')[-3] == "Depth_120":
            case_name = 1

        if sem_path.split('/')[-3] == "Case_3" or sem_path.split('/')[-3] == "Depth_130":
            case_name = 2

        if sem_path.split('/')[-3] == "Case_4" or sem_path.split('/')[-3] == "Depth_140":
            case_name = 3

        return torch.Tensor(sem_img).unsqueeze(0), case_name  # C,B,H,W.unsqueeze(0)

    def __len__(self):
        return len(self.sem_path_list)


# %%
caselist = sorted(glob.glob(CASE_LIST_PATH))
case_data_len = len(caselist)
random.Random(19991006).shuffle(caselist)
train_case_paths = caselist[:int(case_data_len * 0.9)]
validation_case_paths = caselist[int(case_data_len * 0.9):]

case_train_dataset = CaseDataset(train_case_paths)
case_validation_dataset = CaseDataset(validation_case_paths)

case_train_loader = DataLoader(case_train_dataset, batch_size=64, shuffle=True, num_workers=4)
case_validation_loader = DataLoader(case_validation_dataset, batch_size=64, shuffle=True, num_workers=4)
# %%

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 15 17:46:57 2022

@author: nvidia
"""
# %%model class
import torch.nn as nn


# %%
class depthmodel(nn.Module):
    def __init__(self):
        super(depthmodel, self).__init__()
        self.fc1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3), bias=True),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            nn.ReLU()
        )
        self.yp1 = nn.Sequential(
            nn.Conv2d(64, 256, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), bias=True),
            nn.BatchNorm2d(256)
        )
        self.xp1 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), bias=True),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=True),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=True),
            nn.BatchNorm2d(256)
        )
        self.green1 = nn.Sequential(
            nn.Conv2d(256, 64, kernel_size=(1, 1), stride=(1, 1), bias=True),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=True),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), padding=0, bias=True),
            nn.BatchNorm2d(256)
        )
        self.yp2 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), bias=True),
            nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            nn.BatchNorm2d(512)
        )
        self.xp2 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), bias=True),
            nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=True),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=True),
            nn.BatchNorm2d(512)
        )
        self.green2 = nn.Sequential(
            nn.Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1), bias=True),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=True),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=True),
            nn.BatchNorm2d(512)
        )
        self.yp3 = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), bias=True),
            nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            nn.BatchNorm2d(1024)
        )
        self.xp3 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), bias=True),
            nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=True),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=True),
            nn.BatchNorm2d(1024)
        )
        self.green3 = nn.Sequential(
            nn.Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=True),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=True),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=True),
            nn.BatchNorm2d(1024)
        )
        self.fc2 = nn.Sequential(
            nn.Conv2d(1024, 516, kernel_size=(1, 1), stride=(1, 1), bias=True),
            nn.BatchNorm2d(516)
        )
        self.upconv1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(516, 256, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), bias=True),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        self.upconv2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(256, 128, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), bias=True),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        self.upconv3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(128, 64, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), bias=True),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.fc3 = nn.Sequential(
            nn.Conv2d(64, 1, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=True),
            nn.ReLU()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.fc1(x)
        # purple
        y = self.yp1(x)
        x = self.xp1(x)
        x = x + y
        x = self.relu(x)
        # green
        for i in range(2):
            y = x
            x = self.green1(x)
            x = x + y
            x = self.relu(x)
        # purple
        y = self.yp2(x)
        x = self.xp2(x)
        x = x + y
        x = self.relu(x)
        # green
        for i in range(3):
            y = x
            x = self.green2(x)
            x = x + y
            x = self.relu(x)
        # purple
        y = self.yp3(x)
        x = self.xp3(x)
        x = x + y
        x = self.relu(x)
        # green
        for i in range(8):
            y = x
            x = self.green3(x)
            x = x + y
            x = self.relu(x)
        x = self.fc2(x)
        x = self.upconv1(x)
        x = self.upconv2(x)
        x = self.upconv3(x)
        x = self.fc3(x)
        return x

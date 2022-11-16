#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import glob
import cv2

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import random
import copy


# %% datasets path
DF_PATH = 'datasets/train/average_depth.csv'

CASE11_PATH = 'datasets/simulation_data/SEM/Case_1/*/*.png'
CASE22_PATH = 'datasets/simulation_data/SEM/Case_2/*/*.png'
CASE33_PATH = 'datasets/simulation_data/SEM/Case_3/*/*.png'
CASE44_PATH = 'datasets/simulation_data/SEM/Case_4/*/*.png'

DEPTH11_PATH = 'datasets/train/SEM/Depth_110/*/*.png'
DEPTH22_PATH = 'datasets/train/SEM/Depth_120/*/*.png'
DEPTH33_PATH = 'datasets/train/SEM/Depth_130/*/*.png'
DEPTH44_PATH = 'datasets/train/SEM/Depth_140/*/*.png'

LABEL11_PATH = 'datasets/simulation_data/Depth/Case_1/*/*.png'
LABEL22_PATH = 'datasets/simulation_data/Depth/Case_2/*/*.png'
LABEL33_PATH = 'datasets/simulation_data/Depth/Case_3/*/*.png'
LABEL44_PATH = 'datasets/simulation_data/Depth/Case_4/*/*.png'

# %%
df1 = pd.read_csv(DF_PATH)
df1.head()
df2 = df1.sort_values(by=['0'])
print(df2['1'])


# %%
def plots(figs):
    k = len(figs)
    if k > 1:
        fig, axes = plt.subplots(1, k, sharex=True, sharey=True, figsize=(10, 10))
        for i in range(k):
            axes[i].imshow(figs[i], cmap='gray', vmin=0, vmax=255)
    elif k == 1:
        plt.imshow(figs[0], cmap='gray', vmin=0, vmax=255)


# %%
case11 = sorted(glob.glob(CASE11_PATH))
case22 = sorted(glob.glob(CASE22_PATH))
case33 = sorted(glob.glob(CASE33_PATH))
case44 = sorted(glob.glob(CASE44_PATH))
# len=43326
depth11 = sorted(glob.glob(DEPTH11_PATH))
depth22 = sorted(glob.glob(DEPTH22_PATH))
depth33 = sorted(glob.glob(DEPTH33_PATH))
depth44 = sorted(glob.glob(DEPTH44_PATH))
# len=15166
label11 = sorted(glob.glob(LABEL11_PATH))
label22 = sorted(glob.glob(LABEL22_PATH))
label33 = sorted(glob.glob(LABEL33_PATH))
label44 = sorted(glob.glob(LABEL44_PATH))
# %%
case1 = case11[random.randint(0, 43326)]
case2 = case22[random.randint(0, 43326)]
case3 = case33[random.randint(0, 43326)]
case4 = case44[random.randint(0, 43326)]

depth1 = depth11[random.randint(0, 15166)]
depth2 = depth22[random.randint(0, 15166)]
depth3 = depth33[random.randint(0, 15166)]
depth4 = depth44[random.randint(0, 15166)]

label1 = label11[1]
label2 = label22[1]
label3 = label33[1]
label4 = label44[1]

case1 = cv2.imread(case1)
case2 = cv2.imread(case2)
case3 = cv2.imread(case3)
case4 = cv2.imread(case4)
depth1 = cv2.imread(depth1)
depth2 = cv2.imread(depth2)
depth3 = cv2.imread(depth3)
depth4 = cv2.imread(depth4)
label1 = cv2.imread(label1)
label2 = cv2.imread(label2)
label3 = cv2.imread(label3)
label4 = cv2.imread(label4)
for i in range(1, 5):
    if np.shape(globals()["case{}".format(i)]) == (72, 48, 3):
        globals()["case{}".format(i)] = globals()["case{}".format(i)][:, :, 0]
for i in range(1, 5):
    if np.shape(globals()["label{}".format(i)]) == (72, 48, 3):
        globals()["label{}".format(i)] = globals()["label{}".format(i)][:, :, 0]
for i in range(1, 5):
    if np.shape(globals()["case{}".format(i)]) == (72, 48, 3):
        globals()["depth{}".format(i)] = globals()["depth{}".format(i)][:, :, 0]

plots([case1, depth1, label1])
# %%
plots([case2, depth2, label2])
# %%
plots([case3, depth3, label3])
# %%
plots([case4, depth4, label4])
# %%
plots([case1, case2, case3, case4])
# %%
plots([depth1, depth2, depth3, depth4])
# %%
hist1 = cv2.calcHist([case1], [0], None, [256], [0, 256])
hist2 = cv2.calcHist([case2], [0], None, [256], [0, 256])
hist3 = cv2.calcHist([case3], [0], None, [256], [0, 256])
hist4 = cv2.calcHist([case4], [0], None, [256], [0, 256])
hist11 = cv2.calcHist([depth1], [0], None, [256], [0, 256])
hist22 = cv2.calcHist([depth2], [0], None, [256], [0, 256])
hist33 = cv2.calcHist([depth3], [0], None, [256], [0, 256])
hist44 = cv2.calcHist([depth4], [0], None, [256], [0, 256])

print(label1[1, 30], label2[0, 0], label3[0, 0], label4[0, 0])
plt.plot(hist1)
plt.plot(hist11)
plt.title("case1")
plt.show()
plt.plot(hist2)
plt.plot(hist22)
plt.title("case2")
plt.show()
plt.plot(hist3)
plt.plot(hist33)
plt.title("case3")
plt.show()
plt.plot(hist4)
plt.plot(hist44)
plt.title("case4")
plt.show()

# %%
dst0 = cv2.GaussianBlur(depth1, (0, 0), 0.5)
dst1 = cv2.GaussianBlur(case1, (0, 0), 1)
plots([dst0, dst1])

# %%
hist0 = cv2.calcHist([dst0], [0], None, [256], [0, 256])
hist1 = cv2.calcHist([dst1], [0], None, [256], [0, 256])
plt.plot(hist0)
plt.plot(hist1)
plt.show()

# %%
dst3 = copy.deepcopy(dst1)
height, width = dst3.shape

# case1,2,3,4
for i in range(72):
    for j in range(48):
        if dst3[i, j] > 50:
            dst3[i, j] = dst3[i, j] * 1.15
        if dst3[i, j] < 50:
            dst3[i, j] = dst3[i, j] + 4
plots([dst0, dst1, dst3])
# %%
hist0 = cv2.calcHist([dst0], [0], None, [256], [0, 256])
hist1 = cv2.calcHist([dst1], [0], None, [256], [0, 256])
hist2 = cv2.calcHist([dst3], [0], None, [256], [0, 256])
plt.plot(hist1)
plt.show()
plt.plot(hist0)
plt.plot(hist2)
plt.show()

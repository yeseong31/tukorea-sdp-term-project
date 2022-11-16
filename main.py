#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# %%import
import warnings
import sys

import matplotlib.pyplot as plt

from seed import seed_everything
from customdataset import *
from depthmodel import depthmodel
from train import train, casetrain
from inference import inference
from casemodel import Casemodel

# %% datasets path
CASE_BEST_MODEL_PATH = 'datasets/case_best_model.pth'
BEST_MODEL1_PATH = 'datasets/best_model1.pth'
BEST_MODEL2_PATH = 'datasets/best_model2.pth'
BEST_MODEL3_PATH = 'datasets/best_model3.pth'
BEST_MODEL4_PATH = 'datasets/best_model4.pth'

# %%
# sys.path.append("/home/nvidia/Workspace/Samsung")

# %%
warnings.filterwarnings(action='ignore')

# %%device
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(device)

# %%CFG
CFG = {
    'WIDTH': 48,
    'HEIGHT': 72,
    'EPOCHS': 40,
    'LEARNING_RATE': 1e-7,
    'CASE_LEARNING_RATE': 1e-4,
    'CASE_EPOCHS': 60,
    'BATCH_SIZE': 64,
    'SEED': 44
}

# %%Random seed
seed_everything(CFG['SEED'])  # Seed 고정

# %%model setting
case_model = Casemodel()
case_model = case_model.to(device)
case_model.eval()

model1 = depthmodel()
model1 = model1.to(device)
model1.eval()

model2 = depthmodel()
model2 = model2.to(device)
model2.eval()

model3 = depthmodel()
model3 = model3.to(device)
model3.eval()

model4 = depthmodel()
model4 = model4.to(device)
model4.eval()

# %%parameters setting
optimizer1 = torch.optim.Adam(params=model1.parameters(), lr=CFG["LEARNING_RATE"], weight_decay=0.00001)
optimizer2 = torch.optim.Adam(params=model2.parameters(), lr=CFG["LEARNING_RATE"], weight_decay=0.00001)
optimizer3 = torch.optim.Adam(params=model3.parameters(), lr=CFG["LEARNING_RATE"], weight_decay=0.00001)
optimizer4 = torch.optim.Adam(params=model4.parameters(), lr=CFG["LEARNING_RATE"], weight_decay=0.00001)

case_optimizer = torch.optim.Adam(params=case_model.parameters(), lr=CFG["CASE_LEARNING_RATE"], weight_decay=0.00001)
case_criterion = torch.nn.CrossEntropyLoss()
scheduler = None

# %%
case_checkpoint = torch.load(CASE_BEST_MODEL_PATH)
case_model = Casemodel()
case_model = case_model.to(device)
case_model.load_state_dict(case_checkpoint)

# %%
best_acc, best_epoch, vali_acc, val_loss, tr_loss = casetrain(case_model, case_optimizer, case_criterion,
                                                              case_train_loader, case_validation_loader,
                                                              CFG['CASE_EPOCHS'], scheduler, device)
# best_acc,best_epoch,vali_acc,val_loss,tr_loss=casetrain(case_model,case_optimizer,case_criterion,case_train_loader,case_validation_loader,CFG['EPOCHS'],scheduler,device,best_acc)

# %%visible result
plt.title('Loss')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.grid()
plt.plot(tr_loss[:], label='train loss')
plt.plot(val_loss[:], label='validation loss')
plt.legend(loc='best')
plt.show()

# %%
checkpoint = torch.load(BEST_MODEL1_PATH)
model1 = depthmodel()
model1 = model1.to(device)
model1.load_state_dict(checkpoint)

checkpoint = torch.load(BEST_MODEL2_PATH)
model2 = depthmodel()
model2 = model2.to(device)
model2.load_state_dict(checkpoint)

checkpoint = torch.load(BEST_MODEL3_PATH)
model3 = depthmodel()
model3 = model3.to(device)
model3.load_state_dict(checkpoint)

checkpoint = torch.load(BEST_MODEL4_PATH)
model4 = depthmodel()
model4 = model4.to(device)
model4.load_state_dict(checkpoint)

case_checkpoint = torch.load(CASE_BEST_MODEL_PATH)
case_model = Casemodel()
case_model = case_model.to(device)
case_model.load_state_dict(case_checkpoint)

# %%each train
# train first time
train_loss, validation_loss, best_loss, epochs = train(model1, 1, optimizer1, train_loader1, val_loader1, scheduler,
                                                       device, CFG['EPOCHS'])
train_loss, validation_loss, best_loss, epochs = train(model2, 2, optimizer2, train_loader2, val_loader2, scheduler,
                                                       device, CFG['EPOCHS'])
train_loss, validation_loss, best_loss, epochs = train(model3, 3, optimizer3, train_loader3, val_loader3, scheduler,
                                                       device, CFG['EPOCHS'])
train_loss, validation_loss, best_loss, epochs = train(model4, 4, optimizer4, train_loader4, val_loader4, scheduler,
                                                       device, CFG['EPOCHS'])

# train after second time
"""
train_loss,validation_loss,best_loss,epochs = train(model1,1, optimizer, train_loader1, val_loader1, scheduler, device, CFG['EPOCHS'], train_loss, validation_loss, best_loss, epochs)
train_loss,validation_loss,best_loss,epochs = train(model2,2, optimizer, train_loader2, val_loader2, scheduler, device, CFG['EPOCHS'], train_loss, validation_loss, best_loss, epochs)
train_loss,validation_loss,best_loss,epochs = train(model3,3, optimizer, train_loader3, val_loader3, scheduler, device, CFG['EPOCHS'], train_loss, validation_loss, best_loss, epochs)
train_loss,validation_loss,best_loss,epochs = train(model4,4, optimizer, train_loader4, val_loader4, scheduler, device, CFG['EPOCHS'], train_loss, validation_loss, best_loss, epochs)
"""

# %%visible result
plt.title('Loss')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.grid()
plt.plot(train_loss[:], label='train loss')
plt.plot(validation_loss[:], label='validation loss')
plt.legend(loc='best')
plt.show()

# %%get best models
checkpoint = torch.load(BEST_MODEL1_PATH)
model1 = depthmodel()
model1 = model1.to(device)
model1.load_state_dict(checkpoint)

checkpoint = torch.load(BEST_MODEL2_PATH)
model2 = depthmodel()
model2 = model2.to(device)
model2.load_state_dict(checkpoint)

checkpoint = torch.load(BEST_MODEL3_PATH)
model3 = depthmodel()
model3 = model3.to(device)
model3.load_state_dict(checkpoint)

checkpoint = torch.load(BEST_MODEL4_PATH)
model4 = depthmodel()
model4 = model4.to(device)
model4.load_state_dict(checkpoint)

case_checkpoint = torch.load(CASE_BEST_MODEL_PATH)
case_model = Casemodel()
case_model = case_model.to(device)
case_model.load_state_dict(case_checkpoint)

# %%submit!!
inference(model1, model2, model3, model4, case_model, test_loader, device)

# %%summary
"""
from torchsummary import summary
summary(model, input_size=(1, CFG['WIDTH'], CFG['HEIGHT']), device=device.type)

print(model)
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm


# %% datasets path
CASE_BEST_MODEL_PATH = 'datasets/case_best_model.pth'
BEST_MODEL1_PATH = 'datasets/best_model1.pth'
BEST_MODEL2_PATH = 'datasets/best_model2.pth'
BEST_MODEL3_PATH = 'datasets/best_model3.pth'
BEST_MODEL4_PATH = 'datasets/best_model4.pth'


# %%train class
def train(model, modeltype, optimizer, train_loader, val_loader, scheduler, device, epoch=0,
          tr_loss=None, vali_loss=None, best_rmse=999999, last_epoch=0):
    if vali_loss is None:
        vali_loss = []
    if tr_loss is None:
        tr_loss = []
    model.to(device)
    criterion = nn.MSELoss().to(device)
    a = last_epoch

    for epoch in range(1, epoch):
        model.train()
        train_loss = []
        for sem, depth in tqdm(iter(train_loader)):
            sem = sem.float().to(device)
            depth = depth.float().to(device)

            optimizer.zero_grad()
            model_pred = model(sem)

            loss = criterion(model_pred, depth)

            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())

        val_loss, val_rmse = validation(model, nn.MSELoss().to(device), val_loader, device)
        print(f'Epoch : [{epoch + a}] Train Loss : [{np.mean(train_loss):.5f}]' +
              f'Val Loss : [{val_loss:.5f}] Val RMSE : [{val_rmse:.5f}]')
        print(f'model tpye : [{modeltype}]')
        tr_loss.append(np.mean(train_loss))
        vali_loss.append(val_rmse)
        if best_rmse > val_rmse:
            best_rmse = val_rmse
            if modeltype == 1:
                torch.save(model.state_dict(), BEST_MODEL1_PATH)
            if modeltype == 2:
                torch.save(model.state_dict(), BEST_MODEL2_PATH)
            if modeltype == 3:
                torch.save(model.state_dict(), BEST_MODEL3_PATH)
            if modeltype == 4:
                torch.save(model.state_dict(), BEST_MODEL4_PATH)

            print('Model Saved\n')

        if scheduler is not None:
            scheduler.step()
    return tr_loss, vali_loss, best_rmse, epoch


# %%validation class
def validation(model, criterion, val_loader, device):
    model.eval()
    val_loss = []
    val_rmse = []
    with torch.no_grad():
        for sem, depth in tqdm(iter(val_loader)):
            sem = sem.float().to(device)
            depth = depth.float().to(device)
            model_pred = model(sem)

            loss = criterion(model_pred, depth)

            pred = (model_pred * 255.).type(torch.int8).float()
            true = (depth * 255.).type(torch.int8).float()

            b_rmse = torch.sqrt(criterion(pred, true))

            val_loss.append(loss.item())
            val_rmse.append(b_rmse.item())

    return np.mean(val_loss), np.mean(val_rmse)


# %%
def casetrain(model, optimizer, criterion, train_loader, val_loader, epochs, scheduler, device, bestacc=0.0):
    model.to(device)
    n = len(train_loader)
    best_acc = bestacc
    best_apoch = 0
    tr_loss = []
    val_loss = []
    # Loss Function 정의
    # criterion = nn.CrossEntropyLoss().to(device)

    for epoch in range(1, epochs + 1):  # 에포크 설정
        model.train()  # 모델 학습
        running_loss = 0.0

        for img, label in tqdm(iter(train_loader)):
            img, label = img.to(device), label.to(device)  # 배치 데이터
            optimizer.zero_grad()  # 배치마다 optimizer 초기화

            # Data -> Model -> Output
            logit = model(img)  # 예측값 산출
            loss = criterion(logit, label)  # 손실함수 계산

            # 역전파
            loss.backward()  # 손실함수 기준 역전파
            optimizer.step()  # 가중치 최적화
            running_loss += loss.item()
            train_loss = running_loss / len(train_loader)
        print('[%d] Train loss: %.10f' % (epoch, running_loss / len(train_loader)))
        tr_loss.append(running_loss / len(train_loader))
        if scheduler is not None:
            scheduler.step()

        # Validation set 평가
        model.eval()  # evaluation 과정에서 사용하지 않아야 하는 layer들을 알아서 off 시키도록 하는 함수
        vali_loss = 0.0
        correct = 0

        with torch.no_grad():  # 파라미터 업데이트 안하기 때문에 no_grad 사용
            for img, label in tqdm(iter(val_loader)):
                img, label = img.to(device), label.to(device)

                logit = model(img)
                vali_loss += criterion(logit, label)
                pred = logit.argmax(dim=1, keepdim=True)
                correct += pred.eq(label.view_as(pred)).sum().item()  # 예측값과 실제값이 맞으면 1 아니면 0으로 합산
        vali_acc = 100.0 * correct / len(val_loader.dataset) * 1.0
        print('Vail set: Loss: {:.4f}, Accuracy: {}/{} ( {:.0f}%)\n'.format(vali_loss / len(val_loader), correct,
                                                                            len(val_loader.dataset),
                                                                            100.0 * correct / len(
                                                                                val_loader.dataset) * 1.0))
        val_loss.append(vali_loss / len(val_loader))
        # 베스트 모델 저장
        if best_acc < vali_acc:
            best_acc = vali_acc
            torch.save(model.state_dict(), CASE_BEST_MODEL_PATH)  # 이 디렉토리에 best_model.pth을 저장
            print('Model Saved.')
    return best_acc, best_apoch, vali_acc, val_loss, tr_loss

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
# %%inference classF
import zipfile

import cv2
import torch
from tqdm import tqdm


# %% datasets path
MAKE_SUBMISSION_PATH = 'datasets/submission/'
SUBMISSION_ZIP_FILE_PATH = MAKE_SUBMISSION_PATH + 'beforeattach.zip'


# %%
def inference(model1, model2, model3, model4, case_model, test_loader, device):
    model1.to(device)
    model1.eval()
    model2.to(device)
    model2.eval()
    model3.to(device)
    model3.eval()
    model4.to(device)
    model4.eval()
    case_model.to(device)
    case_model.eval()
    a1 = torch.tensor([[0]]).to(device)
    a2 = torch.tensor([[1]]).to(device)
    a3 = torch.tensor([[2]]).to(device)
    a4 = torch.tensor([[3]]).to(device)
    result_name_list = []
    result_list = []
    with torch.no_grad():
        for sem, name in tqdm(iter(test_loader)):
            sem = sem.float().to(device)
            logit = case_model(sem)
            case = logit.argmax(dim=1, keepdim=True)
            global model_pred
            if case == a1:
                model_pred = model1(sem)
            elif case == a2:
                model_pred = model2(sem)
            elif case == a3:
                model_pred = model3(sem)
            elif case == a4:
                model_pred = model4(sem)
            else:
                print('something wrong')
            for pred, img_name in zip(model_pred, name):
                # pred=pred.cpu().numpy()*255.
                pred = pred.cpu().numpy().transpose(1, 2, 0) * 255.
                save_img_path = f'{img_name}'
                # im=torchvision.transforms.functional.to_pil_image(pred)
                # im.save(save_img_path)
                cv2.imwrite(save_img_path, pred)
                result_name_list.append(save_img_path)
                result_list.append(pred)
    os.makedirs(MAKE_SUBMISSION_PATH, exist_ok=True)
    os.chdir(MAKE_SUBMISSION_PATH)
    sub_imgs = []
    for path, pred_img in zip(result_name_list, result_list):
        cv2.imwrite(path, pred_img)
        sub_imgs.append(path)
    submission = zipfile.ZipFile(SUBMISSION_ZIP_FILE_PATH, 'w')
    for path in sub_imgs:
        submission.write(path)
    submission.close()

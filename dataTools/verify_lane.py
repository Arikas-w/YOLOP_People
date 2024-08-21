# -*- coding: UTf-8 -*-

import cv2
import tqdm
import os
import numpy as np
import shutil


rootPath = "/data/litongxin/data/annotation/"
rootImgPath = rootPath + "img"
rootlLanePath = rootPath + "line"

outPath = "./dataTools/res/"
if os.path.exists(outPath):
    shutil.rmtree(outPath)

os.makedirs(outPath, exist_ok=True)

for i in range(6):
    imgNames = os.listdir(os.path.join(rootImgPath, str(i)))
    for imgName in tqdm.tqdm(imgNames):
        # if imgName != "1682668277742735872.jpg":
        #     continue
        imgPath = os.path.join(rootImgPath, str(i), imgName)
        labelPath = os.path.join(rootlLanePath, str(i), imgName)

        if not os.path.exists(imgPath) or not os.path.exists(labelPath):
            continue

        img = cv2.imread(imgPath)
        labelMask = cv2.imread(labelPath)
        if img is None or labelMask is None:
            print(f"img or label is None. {imgName}")
            continue

        mask = np.mean(labelMask, 2)
        color_area = labelMask.copy()
        color_area[mask != 0] = [0, 0, 255]
        img[mask != 0] = img[mask != 0] * 0.5 + color_area[mask != 0] * 0.5
        img = img.astype(np.uint8)

        cv2.imwrite(os.path.join(outPath, imgName), img)
        





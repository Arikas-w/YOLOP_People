# -*- coding:UTF-8 -*-

import os
import tqdm

indicator = "train"
imgRootPath = "/data/linkejun/dataset/self_data/lane/img/" + indicator
labelRootPath = "/data/linkejun/dataset/self_data/lane/src_json_label/" + indicator
daMaskRootPath = "/data/linkejun/dataset/self_data/lane/lane_gt/" + indicator
llMaskRootPath = "/data/linkejun/dataset/self_data/lane/lane_gt/" + indicator

outFile = "/data/linkejun/dataset/self_data/lane/" + f"{indicator}.txt"
outData = open(outFile, 'w') 

imgNames = os.listdir(imgRootPath)
for imgName in tqdm.tqdm(imgNames):
    imgPath = os.path.join(imgRootPath, imgName)
    baseName = imgName[:imgName.rfind(".")]
    labelPath = os.path.join(labelRootPath, f"{baseName}.json")
    daMaskPath = os.path.join(daMaskRootPath, f"{baseName}.png")
    llMaskPath = os.path.join(llMaskRootPath, f"{baseName}.png")

    if not os.path.exists(imgPath) or not os.path.exists(labelPath) or not os.path.exists(daMaskPath) \
        or not os.path.exists(llMaskPath):
        print("file not exist.")
        continue

    outData.writelines(f"{imgPath},{labelPath},{daMaskPath},{llMaskPath}\n")
outData.close()





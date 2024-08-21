# -*- coding: UTF-8 -*-

import json
import os
import tqdm
import cv2
import numpy as np

id_dict_fd = {'person': 0, 'rider': 0, 'car': 1, 'bus': 1, 'truck': 1, 'bike': 2, 'motor': 3}


def convert_data():
    rootLabelPath = "/data/linkejun/dataset/bdd100k/labels/author_offer/100k/val"
    outRootPath = "/data/linkejun/dataset/bdd100k/labels/author_offer/det_labels/val"

    labelNames = os.listdir(rootLabelPath)
    for labelName in tqdm.tqdm(labelNames[:]):
        labelPath = os.path.join(rootLabelPath, labelName)
        with open(labelPath, 'rb') as f:
            infos = json.load(f)

        shape = infos['shape']  # h, w
        name = infos['name']
        allLabels = []
        for info in infos['frames']:
            if len(info['objects']) < 1:
                continue
            

            for obj in info['objects']:
                category = obj['category']
                obj_id = obj['id']
                attr = obj['attributes']
                if category not in id_dict_fd.keys():
                    continue
                
                label = []
                if 'box2d' in obj:
                    x1 = obj['box2d']['x1']
                    y1 = obj['box2d']['y1']
                    x2 = obj['box2d']['x2']
                    y2 = obj['box2d']['y2']

                    w = x2 - x1
                    h = y2 - y1
                    x_center = (x1 - w / 2.0) / shape[1]
                    y_center = (y1 - h / 2.0) / shape[0]
                    w /= shape[1]
                    h /= shape[0]

                    label.append(id_dict_fd[category])
                    label.append(max(0, x_center))
                    label.append(max(0, y_center))
                    label.append(w)
                    label.append(h)
                    allLabels.append(label)
        
        if len(allLabels) > 0:
            with open(os.path.join(outRootPath, f"{name}.txt"), 'w') as f:
                for l in allLabels:
                    f.writelines(f"{l[0]} {l[1]:.6f} {l[2]:.6f} {l[3]:.6f} {l[4]:.6f}\n")


def gen_train_val():
    rootImgPath = "/data/linkejun/dataset/bdd100k/images/100k/val"
    rootLabelPath = "/data/linkejun/dataset/bdd100k/labels/author_offer/det_labels/val"

    outTrainTxt = "/data/linkejun/dataset/bdd100k/labels/author_offer/det_labels/val.txt"
    outTrainData = open(outTrainTxt, 'w')

    imgNames = os.listdir(rootImgPath)
    for imgName in tqdm.tqdm(imgNames):
        name = imgName[:imgName.rfind('.')]
        imgPath = os.path.join(rootImgPath, imgName)
        labelPath = os.path.join(rootLabelPath, f"{name}.txt")
        if (not os.path.exists(imgPath)) or (not os.path.exists(labelPath)):
            print("imgPath or labelPath not exist.")
            continue
        outTrainData.writelines(f"{imgPath},{labelPath}\n")

    outTrainData.close()



if __name__ == "__main__":
    # convert_data()
    gen_train_val()

                

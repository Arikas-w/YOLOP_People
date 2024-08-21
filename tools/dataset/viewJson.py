# -*- coding: UTF-8 -*-

import json
import os
import tqdm
import cv2
import numpy as np

rootLabelPath = "/data/linkejun/dataset/bdd100k/labels/author_offer/100k_resize/train"
rootImgPath = "/data/linkejun/dataset/bdd100k/images/100k_resize/train"
outRootPath = "/data/linkejun/dataset/sourceImg/bdd100k_show"
cate = {}
labelNames = os.listdir(rootLabelPath)
for labelName in tqdm.tqdm(labelNames[:]):
    labelPath = os.path.join(rootLabelPath, labelName)
    with open(labelPath, 'rb') as f:
        infos = json.load(f)
    imgPath = os.path.join(rootImgPath, f"{infos['name']}.jpg")
    if not os.path.exists(imgPath):
        print(f"file not exist: {imgPath}")
        continue
    img = cv2.imread(imgPath)
    img = cv2.resize(img, (640, 360), interpolation=cv2.INTER_LINEAR)
    imgFormat = os.path.basename(imgPath)[os.path.basename(imgPath).rfind('.'):]
    infos['shape'] = [360, 640] # list(img.shape[:2])
    infos['imgFormat'] = ".jpg"
    # with open(os.path.join(rootLabelPath, labelName), 'w') as f:
    #     json.dump(infos, f)
    # subCate = []
    for info in infos['frames']:
        if len(info['objects']) < 1:
            continue

        for obj in info['objects']:
            category = obj['category']
            obj_id = obj['id']
            attr = obj['attributes']
            if 'box2d' in obj:
                x1 = obj['box2d']['x1']
                y1 = obj['box2d']['y1']
                x2 = obj['box2d']['x2']
                y2 = obj['box2d']['y2']

                # obj['box2d']['x1'] = 0.5 * obj['box2d']['x1']
                # obj['box2d']['y1'] = 0.5 * obj['box2d']['y1']
                # obj['box2d']['x2'] = 0.5 * obj['box2d']['x2']
                # obj['box2d']['y2'] = 0.5 * obj['box2d']['y2']
                # if category not in cate:
                #     cate[category] = 0
                # else:
                #     cate[category] += 1
                # if "traffic" in category:
                #     continue

                cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 1)
                cv2.putText(img, category, (int((x1+x2)/2), int((y1+y2)/2)), cv2.FONT_HERSHEY_COMPLEX, 1.0, (0,255,255), 1)
                # subCate.append(category)
    # if ("rider" not in subCate) and ("bike" not in subCate):
    # if ("rider" not in subCate):
    #     continue
    cv2.imwrite(os.path.join(outRootPath, f"{infos['name']}.jpg"), img)

    # with open(os.path.join(rootLabelPath, labelName), 'w') as f:
    #     json.dump(infos, f)
# print(cate)

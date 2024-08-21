# -*- coding: UTF-8 -*-

import numpy as np
import tqdm

trainFile_1 = "/data/linkejun/dataset/bdd100k_selfData_txt/train.txt"
valFile_1 = "/data/linkejun/dataset/bdd100k_selfData_txt/val.txt"

trainFile_2 = "/data/linkejun/dataset/self_data/lane/train.txt"
valFile_2 = "/data/linkejun/dataset/self_data/lane/val.txt"

with open(trainFile_1, 'r') as ft1:
    trainLines_1 = ft1.readlines()
tl_1 = trainLines_1.__len__()

with open(trainFile_2, 'r') as ft2:
    trainLines_2 = ft2.readlines()
tl_2 = trainLines_2.__len__()

with open(valFile_1, 'r') as fv1:
    valLines_1 = fv1.readlines()
vl_1 = valLines_1.__len__()

with open(valFile_2, 'r') as fv2:
    valLines_2 = fv2.readlines()
vl_2 = valLines_2.__len__()


# add train info
for l in tqdm.tqdm(trainLines_2):
    trainLines_1.append(l)

# add val info
for l in tqdm.tqdm(valLines_2):
    valLines_1.append(l)

# shuffle
for _ in range(10): 
    np.random.shuffle(trainLines_1)
    np.random.shuffle(valLines_1)

print(f"train, lines1: {tl_1}, lines2:{tl_2}, total:{trainLines_1.__len__()}")
print(f"val, lines1: {vl_1}, lines2:{vl_2}, total:{valLines_1.__len__()}")

with open(trainFile_1, 'w') as f:
    for l in tqdm.tqdm(trainLines_1):
        f.writelines(l)

with open(valFile_1, 'w') as f:
    for l in tqdm.tqdm(valLines_1):
        f.writelines(l)
        
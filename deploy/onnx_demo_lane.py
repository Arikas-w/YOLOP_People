# -*- coding: UTF-8 -*-

import argparse
import onnxruntime as ort
import onnx
import collections
import os
import cv2
import numpy as np
import shutil


parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='/data/linkejun/models/freespace/yolopv2-tiny/ttt-lane-selfData/best.onnx', help='.onnx model path')
parser.add_argument('--input_w', type=int, default=640, help='model input width')
parser.add_argument('--input_h', type=int, default=384, help='model input height')
parser.add_argument('--imgFile', type=str, default='./inference/images/1920x1080.jpg', help='img path') 
parser.add_argument('--out', type=str, default="./inference/output", help='result path to save')


if __name__ == "__main__":
    args = parser.parse_args()

    if os.path.exists(args.out):
        shutil.rmtree(args.out)
    os.makedirs(args.out, exist_ok=True) 
    model = onnx.load(args.model)

    img_ori = cv2.imread(args.imgFile)
    img_draw = img_ori.copy()

    # pre-process
    ratio = max(args.input_w, args.input_h) / max(img_ori.shape[:2])
    img_re = cv2.resize(img_ori, (int(img_ori.shape[1]*ratio), int(img_ori.shape[0]*ratio)), interpolation=cv2.INTER_LINEAR)
    
    # pad_w = int((32 - img_re.shape[1] % 32 if img_re.shape[1] % 32 != 0 else 0) / 2)
    # pad_h = int((32 - img_re.shape[0] % 32 if img_re.shape[0] % 32 != 0 else 0) / 2)
    pad_w = int((args.input_w - img_re.shape[1]) / 2)
    pad_h = int((args.input_h - img_re.shape[0]) / 2)
    # img_RGB = cv2.cvtColor(img_re, code=cv2.COLOR_BGR2RGB)  # BGR to RGB
    img_pad = cv2.copyMakeBorder(img_re, pad_h, pad_h, pad_w, pad_w, cv2.BORDER_CONSTANT, value=(114, 114, 114))
    img_pad = np.float32(img_pad)
    # img_pad /= 255
    img_input = img_pad.transpose(2, 0, 1)[np.newaxis, :, :, :]

    # providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    sess = ort.InferenceSession(args.model, providers=ort.get_available_providers())
    input_name = sess.get_inputs()[0].name

    # forward
    output = sess.run(None, input_feed={input_name: img_input})
    outputNames = [x.name for x in  sess.get_outputs()]
    outputShapes = [x.shape for x in  sess.get_outputs()]
    print(f"**********output name and shape**********\n{collections.OrderedDict(zip(outputNames, outputShapes))}")

    # post process
    ll = output[0]  # (384, 640) ll
    
    ll_mask = ll[pad_h:ll.shape[0]-pad_h, pad_w:ll.shape[1]-pad_w]
    color_area = np.zeros((img_re.shape[0], img_re.shape[1], 3), dtype=np.uint8)
    color_area[ll_mask == 1] = [0, 0, 255]
    color_area = cv2.resize(color_area, (int(color_area.shape[1] / ratio), int(color_area.shape[0] / ratio)), interpolation=cv2.INTER_CUBIC)
    color_mask = np.mean(color_area, 2)
    # color_mask = color_area
    img_draw[color_mask != 0] = img_draw[color_mask != 0] * 0.5 + color_area[color_mask != 0] * 0.5
    # img_draw = img_draw * 0.5 + color_area * 0.5
    img_draw = img_draw.astype(np.uint8)
    
    ll_mask_show = np.zeros((ll_mask.shape[0], ll_mask.shape[1], 3), dtype=np.uint8)
    ll_mask_show[ll_mask == 1] = [255, 255, 255]
    ll_mask_show = cv2.resize(ll_mask_show, (int(ll_mask_show.shape[1] / ratio), int(ll_mask_show.shape[0] / ratio)), interpolation=cv2.INTER_CUBIC)
    
    # img_draw = cv2.addWeighted(img_draw, 0.5, da_mask_show, 0.5, 0)
    # img_draw = cv2.addWeighted(img_draw, 0.5, ll_mask_show, 0.5, 0)
    if not os.path.exists(args.out):
        os.makedirs(args.out)

    cv2.imwrite(os.path.join(args.out, "onnx_ll_mask.png"), ll_mask_show)
    cv2.imwrite(os.path.join(args.out, "onnx_color_area.png"), color_area)
    cv2.imwrite(os.path.join(args.out, "onnx_pred.png"), img_draw)

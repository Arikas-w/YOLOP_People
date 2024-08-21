# -*- coding: UTF-8 -*-

import argparse
import onnxruntime as ort
import onnx
import collections
import os
import cv2
import numpy as np
import shutil
from utils import non_max_suppression, scale_coords
import torch


parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='/home/lkj/models/yolop/aiDog-det-da-0.5/epoch-best.onnx', help='.onnx model path')
parser.add_argument('--input_w', type=int, default=640, help='model input width')
parser.add_argument('--input_h', type=int, default=480, help='model input height')
parser.add_argument('--imgFile', type=str, default='inference/images/90.jpg', help='img path') 
parser.add_argument('--conf', type=float, default=0.25, help='obj confidence')
parser.add_argument('--iou', type=float, default=0.45, help='IoU for NMS')
parser.add_argument('--out', type=str, default="./inference/output", help='result path to save')


if __name__ == "__main__":
    args = parser.parse_args()
    if not os.path.exists(args.out):
        os.makedirs(args.out, exist_ok=True)

    model = onnx.load(args.model)

    color_dict = {}

    img_ori = cv2.imread(args.imgFile)
    img_draw = img_ori.copy()

    # pre-process
    ratio = min(args.input_w / img_ori.shape[1], args.input_h / img_ori.shape[0])
    img_re = cv2.resize(img_ori, (int(img_ori.shape[1]*ratio), int(img_ori.shape[0]*ratio)), interpolation=cv2.INTER_LINEAR)
    
    pad_w = int((args.input_w - img_re.shape[1]) / 2)
    pad_h = int((args.input_h - img_re.shape[0]) / 2)
    # img_RGB = cv2.cvtColor(img_re, code=cv2.COLOR_BGR2RGB)  # BGR to RGB
    img_pad = cv2.copyMakeBorder(img_re, pad_h, pad_h, pad_w, pad_w, cv2.BORDER_CONSTANT, value=(114, 114, 114))
    img_pad = np.float32(img_pad)
    # img_pad /= 255
    img_input = img_pad.transpose(2, 0, 1)[np.newaxis, :, :, :]

    # providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    # providers=ort.get_available_providers()
    sess = ort.InferenceSession(args.model, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    input_name = sess.get_inputs()[0].name

    # forward
    output = sess.run(None, input_feed={input_name: img_input})
    outputNames = [x.name for x in  sess.get_outputs()]
    outputShapes = [x.shape for x in  sess.get_outputs()]
    print(f"**********output name and shape**********\n{collections.OrderedDict(zip(outputNames, outputShapes))}")

    # post process
    det = output[0]  # (18900, 18(13 + 5)) det
    da = output[1]   # (480, 640) da
    
    # det
    det_preds = non_max_suppression(torch.from_numpy(det),
                                   conf_thres=args.conf,
                                   iou_thres=args.iou,
                                   agnostic=False
                                   )
    
    # scale 
    det_preds[:, :4] = scale_coords(img_input.shape[2:], det_preds[:, :4], img_ori.shape)
    for i in range(det_preds.shape[0]):
        # draw
        conf = det_preds[i][4]
        cls = int(det_preds[i][5])
        if cls not in color_dict:
            color_dict[cls] = (np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255))

        color = color_dict[cls]
        cv2.rectangle(img_draw, (int(det_preds[i][0]), int(det_preds[i][1])), (int(det_preds[i][2]), int(det_preds[i][3])), color, 2)
        cv2.putText(img_draw, f"{cls}:{conf:.2f}", (int(det_preds[i][0]), int(det_preds[i][1])), cv2.FONT_HERSHEY_COMPLEX, 0.6, color, 1)

    # drivable
    da_mask = da[pad_h:da.shape[0] - pad_h, pad_w:da.shape[1] - pad_w]
    color_area = np.zeros((img_re.shape[0], img_re.shape[1], 3), dtype=np.uint8)
    color_area[da_mask == 1] = [0, 255, 0]
    color_area = cv2.resize(color_area, (int(color_area.shape[1] / ratio), int(color_area.shape[0] / ratio)), interpolation=cv2.INTER_CUBIC)
    color_mask = np.mean(color_area, 2)
    # color_mask = color_area
    img_draw[color_mask != 0] = img_draw[color_mask != 0] * 0.5 + color_area[color_mask != 0] * 0.5
    img_draw = img_draw.astype(np.uint8)
    
    da_mask_show = np.zeros((da_mask.shape[0], da_mask.shape[1], 3), dtype=np.uint8)
    da_mask_show[da_mask == 1] = [255, 255, 255]
    da_mask_show = cv2.resize(da_mask_show, (int(da_mask_show.shape[1] / ratio), int(da_mask_show.shape[0] / ratio)), interpolation=cv2.INTER_CUBIC)
    
    # img_draw = cv2.addWeighted(img_draw, 0.5, da_mask_show, 0.5, 0)
    if not os.path.exists(args.out):
        os.makedirs(args.out)

    cv2.imwrite(os.path.join(args.out, "onnx_da_mask.png"), da_mask_show)
    cv2.imwrite(os.path.join(args.out, "onnx_color_area.png"), color_area)
    cv2.imwrite(os.path.join(args.out, "onnx_pred.png"), img_draw)

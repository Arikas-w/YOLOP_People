# -*- coding:UTF-8 -*-

import argparse
import cv2
import os
import tensorrt as trt
import numpy as np
import torch
from utils import non_max_suppression, scale_coords
import math
import sys
import shutil


class TRTModule(torch.nn.Module):
    def __init__(self, engine=None, input_names=None, output_names=None):
        super(TRTModule, self).__init__()
        self.engine = engine
        self.input_names = input_names
        self.output_names = output_names

        if self.engine is not None:
            # engine创建执行context
            self.context = self.engine.create_execution_context()

    def forward(self, x):
        bindings = [None] * (len(self.input_names) + len(self.output_names))
        # 创建输入tensor，并分配内存
        for i, input_name in enumerate(self.input_names):
            idx = self.engine.get_binding_index(input_name)
            bindings[idx] = x.contiguous().data_ptr()
        
        # 创建输出tensor，并分配内存
        outputs = [None] * len(self.output_names)
        for i, output_name in enumerate(self.output_names):
            idx = self.engine.get_binding_index(output_name)
            dtype = self.torch_dtype_from_trt(self.engine.get_binding_dtype(idx))
            shape = self.engine.get_binding_shape(idx)
            device = self.torch_device_from_trt(self.engine.get_location(idx))
            output = torch.empty(size=tuple(shape), dtype=dtype, device=device)
            outputs[i] = output
            bindings[idx] = output.data_ptr() 
        
        self.context.execute_async(1, bindings, torch.cuda.current_stream().cuda_stream)
        return outputs

    def torch_device_from_trt(self, device):
        if device == trt.TensorLocation.DEVICE:
            return torch.device("cuda")
        elif device == trt.TensorLocation.HOST:
            return torch.device("cpu")
        else:
            return TypeError("%s is not supported by torch." % device)


    def torch_dtype_from_trt(self, dtype):
        if dtype == trt.int8:
            return torch.int
        elif trt.__version__ >= '7.0' and dtype == trt.bool:
            return torch.bool
        elif dtype == trt.int32:
            return torch.int32
        elif dtype == trt.float16:
            return torch.float16
        elif dtype == trt.float32:
            return torch.float32
        else:
            raise TypeError("%s is not supported by torch." % dtype)


parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='/home/wyh/models/yolop/aiDog-det-da-0.5/epoch-best.engine', help='.onnx model path')
parser.add_argument('--input_w', type=int, default= 640, help='model input width')
parser.add_argument('--input_h', type=int, default=480, help='model input height')
parser.add_argument('--imgFile', type=str, default='inference/images/90.jpg', help='img path') 
parser.add_argument('--conf', type=float, default=0.25, help='obj confidence')
parser.add_argument('--iou', type=float, default=0.45, help='IoU for NMS')
parser.add_argument('--out', type=str, default="./inference/output", help='result path to save')
parser.add_argument('--show-net-input-output', type=bool, default=True, help='whether show net input/output')


if __name__ == "__main__":

    args = parser.parse_args()

    if not os.path.exists(args.out):
        os.makedirs(args.out, exist_ok=True) 

    logger = trt.Logger(trt.Logger.INFO)

    color_dict = {}
    # load model
    print(f"load model, {args.model}")
    with open(args.model, 'rb') as f, trt.Runtime(logger) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())
    
    # show input and output
    if args.show_net_input_output:
        for idx in range(engine.num_bindings):
            is_input = engine.binding_is_input(idx)
            name = engine.get_binding_name(idx)
            op_type = engine.get_binding_dtype(idx)
            shape = engine.get_binding_shape(idx)

            print("input_id: ", idx, " , is input: ", is_input, " , binding name: ", name, \
                " , shape: ",  shape, " , type: ", op_type)
    # model init
    trt_model = TRTModule(engine, ['image'], ['det', 'da'])

    img_ori = cv2.imread(args.imgFile)
    if img_ori is None:
        print(f"could not read image: {args.imgFile}")
        sys.exit()

    img_draw = img_ori.copy()

    # pre-process
    ratio = max(args.input_w, args.input_h) / max(img_ori.shape[:2])
    img_re = cv2.resize(img_ori, (int(img_ori.shape[1]*ratio), int(img_ori.shape[0]*ratio)), interpolation=cv2.INTER_LINEAR)
    

    pad_w = int((args.input_w - img_re.shape[1]) / 2)
    pad_h = int((args.input_h - img_re.shape[0]) / 2)
    # img_RGB = cv2.cvtColor(img_re, code=cv2.COLOR_BGR2RGB)  # BGR to RGB
    img_pad = cv2.copyMakeBorder(img_re, pad_h, pad_h, pad_w, pad_w, cv2.BORDER_CONSTANT, value=(114, 114, 114))
    img_pad = np.float32(img_pad)
    # img_pad /= 255
    img_input = img_pad.transpose(2, 0, 1)[np.newaxis, :, :, :]
    img_input = torch.from_numpy(img_input).to("cuda:0")

    # forward
    output = trt_model(img_input)

    # get output
    # det = output[0].data.cpu().numpy()
    da = output[1].data.cpu().numpy()

    # post-process
    # det
    det_preds = non_max_suppression(output[0],
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

    cv2.imwrite(os.path.join(args.out, "trt_da_mask.png"), da_mask_show)
    cv2.imwrite(os.path.join(args.out, "trt_color_area.png"), color_area)
    cv2.imwrite(os.path.join(args.out, "trt_pred.png"), img_draw)
    
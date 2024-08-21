# -*- coding:UTF-8 -*-

import argparse
import cv2
import os
import tensorrt as trt
import numpy as np
import torch
from utils import fitCubic, FindPts, fitBezier3, cal_line_coef, fitBezier2, fitPolyDP, fitSplines, fitBezier3_track
import math
from pycubicspline import calc_2d_spline_interpolation
import tqdm
import sys
import shutil
from utils import getFilePaths, curve


os.environ["CUDA_VISIBLE_DEVICES"] = "1"

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
parser.add_argument('--model', type=str, 
    default='/data/linkejun/models/freespace/yolopv2-tiny/seg_lane/best.engine', help='trt model path')
# parser.add_argument('--model', type=str, default='./weights/trt_8.2.0.6/yolopv2_384x640_bn_fp16.engine', help='.onnx model path')
parser.add_argument('--input_w', type=int, default=640, help='model input width')
parser.add_argument('--input_h', type=int, default=384, help='model input height')
parser.add_argument('--imgDir', type=str, default='./inference/images', help='img path') 
# parser.add_argument('--imgDir', type=str, default='/data/linkejun/dataset/sourceImg/front', help='img path') 
parser.add_argument('--out', type=str, default="./inference/output", help='result path to save')
parser.add_argument('--show-net-input-output', type=bool, default=False, help='whether show net input/output')


if __name__ == "__main__":
    args = parser.parse_args()
    if os.path.exists(args.out):
        shutil.rmtree(args.out)
    os.makedirs(args.out, exist_ok=True) 

    logger = trt.Logger(trt.Logger.INFO)

    # load model
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
    # trt_model = TRTModule(engine, ['image'], ['da', 'll'])
    trt_model = TRTModule(engine, ['image'], ['ll'])

    if not os.path.exists(args.out):
        os.makedirs(args.out)

    imgPaths = getFilePaths(args.imgDir)
    imgPaths.sort()

    left_curve = curve()
    right_curve = curve()

    # imgNames = os.listdir(args.imgDir)
    for imgPath in tqdm.tqdm(imgPaths):
        imgName = os.path.basename(imgPath)
        # if imgName != "1694409993684889.jpg":
        if imgName != "1920x1080.jpg":
            continue

        img_ori = cv2.imread(imgPath)
        if img_ori is None:
            print(f"could not read image: {imgPath}")
            continue

        img_draw = img_ori.copy()
        # for _ in range(200):
        # pre_proc = cv2.getTickCount()
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
        img_input = torch.from_numpy(img_input).to("cuda:0")

        # forwad_start = cv2.getTickCount()
        # forward
        output = trt_model(img_input)
        # forward_end = cv2.getTickCount()

        # get output
        # da = output[0].data.cpu().numpy()
        ll = output[0].data.cpu().numpy()
        # gpu2cpu_end = cv2.getTickCount()

        # post-process
        # da_pred = da[pad_h:da.shape[0]-pad_h, pad_w:da.shape[1]-pad_w]
        ll_pred = ll[pad_h:ll.shape[0]-pad_h, pad_w:ll.shape[1]-pad_w]

        # da_output = np.zeros((da_pred.shape[0], da_pred.shape[1], 1), dtype=np.uint8)
        ll_output = np.zeros((ll_pred.shape[0], ll_pred.shape[1], 1), dtype=np.uint8)

        # da_output[da_pred == 1] = 255
        ll_output[ll_pred == 1] = 255

        # cv2.imwrite("./inference/output/da_output_net.png", da_output)
        # cv2.imwrite("./inference/output/ll_output_net" + imgName[:imgName.rfind('.')] + ".png", ll_output)
        cv2.imwrite("./inference/output/ll_output_net.png", ll_output)
        
        ll_pred_src = cv2.resize(ll_pred.astype(np.uint8), (img_ori.shape[1], img_ori.shape[0]), interpolation=cv2.INTER_LINEAR)
        # img_re[ll_pred != 0] = [0, 0, 255] 
        color_seg = np.zeros((img_draw.shape[0], img_draw.shape[1], 3), dtype=np.uint8)
        color_seg[ll_pred_src != 0] = [0, 0, 255]
        img_draw[ll_pred_src != 0] = img_draw[ll_pred_src != 0] * 0.5 + color_seg[ll_pred_src != 0] * 0.5
        cv2.imwrite(os.path.join(args.out, imgName[:imgName.rfind('.')] + ".jpg"), img_draw)

        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 255, 255)]
        ll_pts = FindPts(ll_output)
        # print(f"num lines: {len(ll_pts)}")
        
        # 两个 二阶贝塞尔曲线拟合
        # img_draw = fitBezier2(img_draw, ll_pts, ratio)

        # 三阶贝塞尔曲线拟合
        # img_draw = fitBezier3(img_draw, ll_pts, ratio)
        img_draw = fitBezier3_track(img_draw, ll_pts, left_curve, right_curve, ratio, middle_x=320)

        # 多边形拟合
        # img_draw = fitPolyDP(img_draw, ll_pts, ratio)

        # 样条插值
        # img_draw = fitSplines(img_draw, ll_pts, ratio)

        # 一元三次方程
        # img_draw = fitCubic(img_draw, ll_pts, ll_output.shape[1], ll_output.shape[0])

        cv2.imwrite(os.path.join(args.out, imgName[:imgName.rfind('.')] + ".jpg"), img_draw)


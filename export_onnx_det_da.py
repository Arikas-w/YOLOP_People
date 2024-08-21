import torch
import torch.nn as nn
# from lib.models.common import Conv, SPP, Bottleneck, BottleneckCSP, Focus, Concat, Detect, SharpenConv
from lib.models.common2 import Conv, Bottleneck, ConvTranspose, ResnetBolck, ELAN, ELAN_X, CBAM, M2C, SPPF, Concat

from lib.models.common import Detect, C2f
from torch.nn import Upsample
from lib.utils import check_anchor_order
from lib.utils import initialize_weights
from lib.utils.utils import fuse_conv_and_bn
import argparse
import onnx
import onnxruntime as ort
import onnxsim

import math
import cv2
from lib.models.YOLOP import YOLOP, YOLOPv2, YOLOv8
from collections import OrderedDict
from thop import profile
from thop import clever_format


class MCnet(nn.Module):
    def __init__(self, block_cfg, device="cuda:0"):
        super(MCnet, self).__init__()
        layers, save = [], []
        self.detector_index = -1
        self.det_out_idx = block_cfg[0][0]
        self.da_out_idx = block_cfg[0][1]
        # Build model
        for i, (from_, block, args) in enumerate(block_cfg[1:]):
            block = eval(block) if isinstance(block, str) else block  # eval strings
            if block is Detect:
                self.detector_index = i
            block_ = block(*args)
            block_.index, block_.from_ = i, from_
            layers.append(block_)
            save.extend(x % i for x in ([from_] if isinstance(from_, int) else from_) if x != -1)  # append to savelist

        assert self.detector_index == block_cfg[0][0]

        self.model, self.save = nn.Sequential(*layers), sorted(save)

        # set stride、anchor for detector
        # Detector = self.model[self.detector_index]  # detector
        # if isinstance(Detector, Detect):
        #     s = 128  # 2x min stride
        #     # for x in self.forward(torch.zeros(1, 3, s, s)):
        #     #     print (x.shape)
        #     with torch.no_grad():
        #         model_out = self.forward(torch.zeros(1, 3, s, s))
        #         detects, _ = model_out
        #         Detector.stride = torch.tensor([s / x.shape[-2] for x in detects])  # forward
        #     # print("stride"+str(Detector.stride ))
        #     Detector.anchors /= Detector.stride.view(-1, 1, 1)  # Set the anchors for the corresponding scale
        #     check_anchor_order(Detector)
        #     self.stride = Detector.stride
        #     # self._initialize_biases()
        # initialize_weights(self)

    def forward(self, x):
        cache = []
        out = []
        det_out = None
        for i, block in enumerate(self.model):
            if block.from_ != -1:
                x = cache[block.from_] if isinstance(block.from_, int) \
                    else [x if j == -1 else cache[j] for j in
                          block.from_]  # calculate concat detect
            x = block(x)

            # save da
            if i == self.da_out_idx:  # save driving area segment result
                # m = nn.Sigmoid()
                # out.append(m(x))
                out.append(torch.sigmoid(x))

            # save det
            if i == self.detector_index:
                det_out = x[0]  # (torch.cat(z, 1), input_feat) if test

            cache.append(x if block.index in self.save else None)

        return torch.squeeze(det_out, dim=0), torch.squeeze(torch.max(out[0], 1)[1], dim=0)  # det, da

        # return torch.squeeze(torch.max(out[0], 1)[1], dim=0), torch.squeeze(torch.max(out[1], 1)[1], dim=0)  # da ll

        # (1,na*ny*nx*nl,no=2+2+1+nc=xy+wh+obj_conf+cls_prob), (1,2,h,w) (1,2,h,w)

    def _initialize_biases(self, cf=None):  # initialize biases into Detect(), cf is class frequency
        # https://arxiv.org/abs/1708.02002 section 3.3
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1.
        # m = self.model[-1]  # Detect() module
        m = self.model[self.detector_index]  # Detect() module
        for mi, s in zip(m.m, m.stride):  # from
            b = mi.bias.view(m.na, -1)  # conv.bias(255) to (3,85)
            b[:, 4] += math.log(8 / (640 / s) ** 2)  # obj (8 objects per 640 image)
            b[:, 5:] += math.log(0.6 / (m.nc - 0.99)) if cf is None else torch.log(cf / cf.sum())  # cls
            mi.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)
    
    def fuse(self):
        print("Fusing layers... ")
        for m in self.model.modules():
            if isinstance(m, Conv) and hasattr(m, 'bn'):
                m.conv = fuse_conv_and_bn(m.conv, m.bn)
                delattr(m, 'bn')
                m.forward = m.fuseforward
            elif isinstance(m, C2f):
                # EdgeTPU does not support FlexSplitV while split provides cleaner ONNX graph
                m.forward = m.forward_split
        return self


class YOLOPONNX(torch.nn.Module):
    def __init__(self, model, use_bn=True, device="cuda:0"):
        super(YOLOPONNX, self).__init__()

        self.model = model
        self.use_bn = use_bn
        self.mean = [0.0, 0.0, 0.0]  # mean
        self.std = [255.0**2, 255.0**2, 255.0**2]    # std
        self.bn = torch.nn.BatchNorm2d(3, device=device)
        self.bn.eval()
        self.bn.running_mean = torch.tensor(self.mean).to(device)
        self.bn.running_var = torch.tensor(self.std).to(device)
    
    def forward(self, x):
        return self.model(self.bn(x)) if self.use_bn else self.model(x)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--height', type=int, default=480, help="model input height")  # height
    parser.add_argument('--width', type=int, default=640, help="model input width")  # width
    parser.add_argument('--use_bn', type=bool, default=True, help="merge divide 255 into bn layer") 
    parser.add_argument('--do_simplify', type=bool, default=True, help="whether do simplify when convert onnx model") 
    parser.add_argument('--weight', type=str, 
                        default="/home/lkj/models/yolop/aiDog-det-da-yolov8-0.5/epoch-best.pth",
                        help="pytorch model path") 
    parser.add_argument('--onnx_path', type=str, default='/home/lkj/models/yolop/aiDog-det-da-yolov8-0.5/epoch-best.onnx',
                         help="onnx model save path") 
    
    args = parser.parse_args()


    # device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    device = 'cpu'
    model = MCnet(YOLOv8)

    ckpt = torch.load(args.weight, map_location=device)
    model.load_state_dict(ckpt['state_dict'])
    model.fuse()
    model.to(device)
    model.eval()

    yoloponnx = YOLOPONNX(model, use_bn=args.use_bn, device=device)
    yoloponnx.to(device)
    yoloponnx.eval()
    
    height = args.height
    width = args.width
    # print("Load ./weights/End-to-end.pth done!")
    inputs = torch.randn(1, 3, height, width, device=device)

    # 统计计算量以及参数量
    print("cal model flops and params...")
    flops, params = profile(yoloponnx, inputs=(inputs, ), verbose=True)
    flops, params = clever_format([flops, params], "%.3f")
    print(f"model flops:{flops}, params:{params}")

    # 导出模型
    print(f"Converting to {args.onnx_path}")
    in_name = ['image']
    out_names = ['det', 'da']
    torch.onnx.export(yoloponnx, inputs, args.onnx_path,
                      verbose=False, opset_version=12, input_names=in_name, output_names=out_names)
    print('convert', args.onnx_path, 'to onnx finish!!!')
   
    # Checks
    model_onnx = onnx.load(args.onnx_path)  # load onnx model
    onnx.checker.check_model(model_onnx)  # check onnx model
    # print(onnx.helper.printable_graph(model_onnx.graph))  # print

    if args.do_simplify:
        print(f'simplifying with onnx-simplifier {onnxsim.__version__}...')
        model_onnx, check = onnxsim.simplify(model_onnx, check_n=1)
        assert check, 'assert check failed'
        onnx.save(model_onnx, args.onnx_path)

    x = inputs.cpu().numpy()
    try:
        sess = ort.InferenceSession(args.onnx_path, providers=['CPUExecutionProvider'])

        for ii in sess.get_inputs():
            print("Input: ", ii)
        for oo in sess.get_outputs():
            print("Output: ", oo)

        output = sess.run(None, input_feed={in_name[0]: x})
        print(f"det out shape: {output[0].shape}, da out shape: {output[1].shape}")
        print('read onnx using onnxruntime sucess')
    except Exception as e:
        print('read failed')
        raise e

    """
    PYTHONPATH=. python3 ./export_onnx.py --height 640 --width 640
    PYTHONPATH=. python3 ./export_onnx.py --height 1280 --width 1280
    PYTHONPATH=. python3 ./export_onnx.py --height 320 --width 320
    """

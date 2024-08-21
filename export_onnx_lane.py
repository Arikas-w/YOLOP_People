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
from lib.models.YOLOP import YOLOP, YOLOPv2, YOLOv8, YOLOPv2_tiny
from collections import OrderedDict
from thop import profile
from thop import clever_format


class MCnet(nn.Module):
    def __init__(self, block_cfg, del_idx=None, ncnn=False, rknn=False, device="cuda:0"):
        super(MCnet, self).__init__()
        self.del_idx = del_idx
        self.ncnn = ncnn
        self.rknn = rknn
        layers, save = [], []
        self.ll_out_idx = [block_cfg[0][-1] - len(self.del_idx)]
       
        # Build model
        for i, (from_, block, args) in enumerate(block_cfg[1:]):
            block = eval(block) if isinstance(block, str) else block  # eval strings
            if i in self.del_idx:
                continue
            block_ = block(*args)
            block_.index, block_.from_ = i, from_
            layers.append(block_)
            save.extend(x % i for x in ([from_] if isinstance(from_, int) else from_) if x != -1)  # append to savelist

        self.model, self.save = nn.Sequential(*layers), sorted(save)
        initialize_weights(self)

    def forward(self, x):
        cache = []
        ll_out = []
        for i, block in enumerate(self.model):
            if block.from_ != -1:
                x = cache[block.from_] if isinstance(block.from_, int) \
                    else [x if j == -1 else cache[j] for j in
                          block.from_]  # calculate concat detect
            x = block(x)
            if i in self.ll_out_idx:  # save driving area segment result
                # m = nn.Sigmoid()
                # out.append(m(x))
                ll_out.append(torch.sigmoid(x))
            cache.append(x if block.index in self.save else None)

        if self.ncnn:
            return ll_out[0]
        elif self.rknn:
            return torch.squeeze(ll_out[0], dim=0)
        else:
            return torch.squeeze(torch.max(ll_out[0], 1)[1], dim=0)


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
    parser.add_argument('--height', type=int, default=384, help="model input height")  # height
    parser.add_argument('--width', type=int, default=640, help="model input width")  # width
    parser.add_argument('--ncnn', type=bool, default=False, help="whether export for ncnn") 
    parser.add_argument('--rknn', type=bool, default=False, help="whether export for rknn") 
    parser.add_argument('--use_bn', type=bool, default=True, help="merge divide 255 into bn layer") 
    parser.add_argument('--do_simplify', type=bool, default=True, help="whether do simplify when convert onnx model") 
    parser.add_argument('--onnx_path', type=str, 
                        default='weights/epoch-best-0.5-lane.onnx',
                         help="onnx model save path") 
    parser.add_argument('--weight', type=str, 
                        default="weights/epoch-best-0.5.pth",
                        help="pytorch model path") 

    args = parser.parse_args()

    if args.ncnn:
        print("convert model to ncnn...")
    elif args.rknn:
        print("convert model to rknn...")
    else:
        print("convert model to mnn...")

    # yolopv2
    del_idx = [i for i in range(27, 51)]  # det da

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print("build model...")
    # model = MCnet(YOLOPv2, del_idx=del_idx, ncnn=args.ncnn)
    model = MCnet(YOLOPv2_tiny, del_idx=del_idx, ncnn=args.ncnn, rknn=args.rknn, device=device)

    print(f"load model {args.weight}...")
    checkpoint = torch.load(args.weight, map_location=device)
    new_dict = OrderedDict()
    for k, v in checkpoint['state_dict'].items():
        num = int(k.split(".")[1])
        if num in del_idx:
            continue
        elif num > del_idx[-1]:
            sp = k.split(".")
            sp[1] = str(num-len(del_idx))
            k = '.'.join(sp)
        new_dict[k] = v

    model.load_state_dict(new_dict)
    model.fuse()
    model.to(device)
    model.eval()

    yoloponnx = YOLOPONNX(model, use_bn=args.use_bn, device=device)
    inputs = torch.randn(1, 3, args.height, args.width, device=device)

    # 统计计算量以及参数量
    print("cal model flops and params...")
    flops, params = profile(yoloponnx, inputs=(inputs, ), verbose=True)
    flops, params = clever_format([flops, params], "%.3f")
    print(f"model flops:{flops}, params:{params}")

    # 导出模型
    print(f"Converting to {args.onnx_path}...")
    out_names = ['ll']
    torch.onnx.export(yoloponnx, inputs, args.onnx_path,
                      verbose=False, opset_version=12, input_names=['image'], output_names=out_names)
    print('convert ', args.onnx_path, ' to onnx finish!!!')
   
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
        sess = ort.InferenceSession(args.onnx_path, providers=ort.get_available_providers())

        for ii in sess.get_inputs():
            print("Input: ", ii)
        for oo in sess.get_outputs():
            print("Output: ", oo)

        print('read onnx using onnxruntime sucess')
    except Exception as e:
        print('read failed')
        raise e

    """
    PYTHONPATH=. python3 ./export_onnx.py --height 640 --width 640
    PYTHONPATH=. python3 ./export_onnx.py --height 1280 --width 1280
    PYTHONPATH=. python3 ./export_onnx.py --height 320 --width 320
    """

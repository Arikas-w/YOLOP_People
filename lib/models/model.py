import torch
from torch import tensor
import torch.nn as nn
import sys,os
import math
import sys
sys.path.append(os.getcwd())
#sys.path.append("lib/models")
#sys.path.append("lib/utils")
#sys.path.append("/workspace/wh/projects/DaChuang")
from lib.utils import initialize_weights
# from lib.models.common2 import DepthSeperabelConv2d as Conv
# from lib.models.common2 import SPP, Bottleneck, BottleneckCSP, Focus, Concat, Detect
# from lib.models.common import Conv, SPP, Bottleneck, BottleneckCSP, Focus, Concat, SharpenConv, DepthSeperabelConv2d, SPPF
from lib.models.common import C2f, Detect
from torch.nn import Upsample
from lib.utils import check_anchor_order
from lib.core.evaluate import SegmentationMetric
from lib.utils.utils import time_synchronized
# from lib.dataset.convert import id_to_name
from lib.models.YOLOP import YOLOP, YOLOv8, YOLOv8n, YOLOPv2, YOLOPv2_tiny
from lib.utils.utils import fuse_conv_and_bn
from lib.config import cfg
from lib.models.common2 import ELAN, ELAN_X, ResnetBolck, M2C, CBAM, ConvTranspose, Conv, SPPF, Concat

class MCnet(nn.Module):
    def __init__(self, block_cfg, cfg):
        super(MCnet, self).__init__()
        layers, save= [], []
        self.nc = cfg.NUM_CLASSES
        self.detector_index = -1
        self.det_out_idx = block_cfg[0][0]
        self.seg_out_idx = block_cfg[0][1]

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

        # id_to_name = {}
        # for k, v in id_dict.items():
        #     id_to_name[str(v)] = k

        self.names = cfg.OBJ_NAMES
        # set stride、anchor for detector
        Detector = self.model[self.detector_index]  # detector
        if isinstance(Detector, Detect):
            s = 128  # 2x min stride
            # for x in self.forward(torch.zeros(1, 3, s, s)):
            #     print (x.shape)
            with torch.no_grad():
                model_out = self.forward(torch.zeros(1, 3, s, s))
                detects, _= model_out
                Detector.stride = torch.tensor([s / x.shape[-2] for x in detects])  # forward
            # print("stride"+str(Detector.stride ))
            Detector.anchors /= Detector.stride.view(-1, 1, 1)  # Set the anchors for the corresponding scale
            check_anchor_order(Detector)
            self.stride = Detector.stride
            self._initialize_biases()
        
        initialize_weights(self)

    def forward(self, x):
        cache = []
        out = []
        det_out = None
        Da_fmap = []
        for i, block in enumerate(self.model):
            # print(f"layer {i}")
            if block.from_ != -1:
                x = cache[block.from_] if isinstance(block.from_, int) else [x if j == -1 else cache[j] for j in block.from_]       #calculate concat detect
            x = block(x)
            if i == self.seg_out_idx:     #save driving area segment result
                m=nn.Sigmoid()
                out.append(m(x))
            if i == self.detector_index:
                det_out = x
            cache.append(x if block.index in self.save else None)
        out.insert(0,det_out)
        return out
            
    
    def _initialize_biases(self, cf=None):  # initialize biases into Detect(), cf is class frequency
        # https://arxiv.org/abs/1708.02002 section 3.3
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1.
        # m = self.model[-1]  # Detect() module
        m = self.model[self.detector_index]  # Detect() module
        for mi, s in zip(m.m, m.stride):  # from
            b = mi.bias.view(m.na, -1)  # conv.bias(255) to (3,85)
            b.data[:, 4] += math.log(8 / (640 / s) ** 2)  # obj (8 objects per 640 image)
            b.data[:, 5:] += math.log(0.6 / (m.nc - 0.99)) if cf is None else torch.log(cf / cf.sum())  # cls
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

def get_net(cfg, **kwargs): 
    m_block_cfg = eval(cfg.MODEL.NAME)
    model = MCnet(m_block_cfg, cfg=cfg)
    return model


if __name__ == "__main__":
    from torch.utils.tensorboard import SummaryWriter
    import time, tqdm
    model = get_net(cfg).fuse()
    model.to("cuda:0")
    model.eval()
    input_ = torch.randn(1, 3, 384, 640, device="cuda:0")
    # gt_ = torch.rand((1, 2, 384, 640))
    # metric = SegmentationMetric(2)
    # count = 100
    # start = time.time()
    # for _ in tqdm.tqdm(range(count)):
    #     model_out = model(input_)
    # end = time.time()
    # print(f"run {count} turns, cost {(end-start):.3f}s")
    # detects, driving_area_seg, lane_line_seg = model_out
    # for i, det in enumerate(detects):
    #     if i == 0:
    #         print("det cat: ", det.shape)
    #     else:
    #         print("det layer1: ", det[0].shape)
    #         print("det layer2: ", det[1].shape)
    #         print("det layer3: ", det[2].shape)
    # print("driving area: ", driving_area_seg.shape)
    # print("lane line: ", lane_line_seg.shape)
    torch.onnx.export(model, input_, './yolopv2_self.onnx', verbose=True, opset_version=12, input_names=['image'])
 

import os
from yacs.config import CfgNode as CN

batch_size_per_gpu = 6 # 64
start_epoch = 0
num_epoch = 300
_C = CN()

_C.OBJ_NAMES = ["car", "truck", "bus", "motorcycle", "bicycle", "person",  "cone-tank", "tl-green",
               "tl-red", "building-turnstile", "building-turnstile-open", "lift-half", "lift-closed"]
# _C.OBJ_NAMES = ['person', 'rider', 'car', 'bus', 'truck', 'bike', 'motor', 'tl_green', 'tl_red', 'tl_yellow', 'tl_none', 'traffic sign', 'train']

_C.SELF_DATA = False
_C.CACHE = '' # ram  None  cache images into memory
_C.NUM_CLASSES = len(_C.OBJ_NAMES)    # num classes for detector
_C.LOG_DIR = 'runs/'
_C.GPUS = (0,)     
_C.WORKERS = 8
# _C.PRINT_FREQ = 10
_C.AUTO_RESUME = True       # Resume from the last training interrupt
_C.NEED_AUTOANCHOR = False      # Re-select the prior anchor(k-means)    When training from scratch (epoch=0), set it to be ture!
_C.DEBUG = False
_C.num_seg_class = 2

# Cudnn related params
_C.CUDNN = CN()
_C.CUDNN.BENCHMARK = True
_C.CUDNN.DETERMINISTIC = False
_C.CUDNN.ENABLED = True


# common params for NETWORK
_C.MODEL = CN(new_allowed=True)
_C.MODEL.NAME = 'YOLOv8'  # YOLOP YOLOv8 YOLOv8n YOLOPv2 YOLOPv2_tiny
_C.MODEL.RATIO = 0.5
_C.MODEL.STRU_WITHSHARE = False     #add share_block to segbranch
_C.MODEL.HEADS_NAME = ['']
_C.MODEL.PRETRAINED = ""
_C.MODEL.PRETRAINED_DET = ""
_C.MODEL.AUTHOR_PRETRAINED = "" # "./weights/End-to-end.pth"
_C.MODEL.IMAGE_SIZE = [640, 640]  # width * height, ex: 192 * 256
_C.MODEL.EXTRA = CN(new_allowed=True)


# loss params
_C.LOSS = CN(new_allowed=True)
_C.LOSS.LOSS_NAME = ''
_C.LOSS.MULTI_HEAD_LAMBDA = None
_C.LOSS.FL_GAMMA = 0.0  # focal loss gamma
_C.LOSS.CLS_POS_WEIGHT = 1.0  # classification loss positive weights
_C.LOSS.OBJ_POS_WEIGHT = 1.0  # object loss positive weights
_C.LOSS.SEG_POS_WEIGHT = 1.0  # segmentation loss positive weights
_C.LOSS.BOX_GAIN = 0.05  # box loss gain
_C.LOSS.CLS_GAIN = 0.5  # classification loss gain
_C.LOSS.OBJ_GAIN = 1.0  # object loss gain
_C.LOSS.DA_SEG_GAIN = 0.2  # driving area segmentation loss gain


# DATASET related params
_C.DATASET = CN(new_allowed=True)
# _C.DATASET.DATAROOT = '/data/linkejun/dataset/bdd100k/images/100k'                       # the path of images folder
# _C.DATASET.LABELROOT = '/data/linkejun/dataset/bdd100k/labels/author_offer/100k'         # the path of det_annotations folder
# _C.DATASET.MASKROOT = '/data/linkejun/dataset/bdd100k/labels/author_offer/bdd_seg_gt'    # the path of da_seg_annotations folder
# _C.DATASET.LANEROOT = '/data/linkejun/dataset/bdd100k/labels/author_offer/bdd_lane_gt'   # the path of ll_seg_annotations folder

# bdd100k datasets
_C.DATASET.TRAIN_FILE = '/home/lkj/datasets/companyData/det_segment/yolop-train-da.txt'
_C.DATASET.TEST_FILE = '/home/lkj/datasets/companyData/det_segment/yolop-val-da.txt'

# self-data
# _C.DATASET.TRAIN_FILE = '/data/linkejun/dataset/self_data/lane/train.txt'
# _C.DATASET.TEST_FILE = '/data/linkejun/dataset/self_data/lane/val.txt'

# merge data
# _C.DATASET.TRAIN_FILE = '/data/linkejun/dataset/bdd100k_selfData_txt/train.txt'
# _C.DATASET.TEST_FILE = '/data/linkejun/dataset/bdd100k_selfData_txt/val.txt'

# _C.DATASET.DATA_FORMAT = 'jpg'
# _C.DATASET.SELECT_DATA = False
# _C.DATASET.ORG_IMG_SIZE = [720, 1280]

# training data augmentation
_C.DATASET.FLIP = True
_C.DATASET.SCALE_FACTOR = 0.25
_C.DATASET.ROT_FACTOR = 10
_C.DATASET.TRANSLATE = 0.1
_C.DATASET.SHEAR = 0.0
_C.DATASET.COLOR_RGB = False
_C.DATASET.HSV_H = 0.015  # image HSV-Hue augmentation (fraction)
_C.DATASET.HSV_S = 0.7  # image HSV-Saturation augmentation (fraction)
_C.DATASET.HSV_V = 0.4  # image HSV-Value augmentation (fraction)
# TODO: more augmet params to add


# train
_C.TRAIN = CN(new_allowed=True)
_C.TRAIN.LR0 = 0.001  # initial learning rate (SGD=1E-2, Adam=1E-3)
_C.TRAIN.LRF = 0.2  # final OneCycleLR learning rate (lr0 * lrf)
_C.TRAIN.WARMUP_EPOCHS = 3.0
_C.TRAIN.WARMUP_BIASE_LR = 0.1
_C.TRAIN.WARMUP_MOMENTUM = 0.8

_C.TRAIN.OPTIMIZER = 'adam'
_C.TRAIN.MOMENTUM = 0.937
_C.TRAIN.WD = 0.0005
_C.TRAIN.NESTEROV = True
_C.TRAIN.GAMMA1 = 0.99
_C.TRAIN.GAMMA2 = 0.0

_C.TRAIN.BEGIN_EPOCH = start_epoch
_C.TRAIN.END_EPOCH = num_epoch

_C.TRAIN.VAL_FREQ = 1
_C.TRAIN.BATCH_SIZE_PER_GPU = batch_size_per_gpu
_C.TRAIN.SHUFFLE = True

_C.TRAIN.IOU_THRESHOLD = 0.2
_C.TRAIN.ANCHOR_THRESHOLD = 4.0

# if training 3 tasks end-to-end, set all parameters as True
# Alternating optimization
_C.TRAIN.SEG_ONLY = False           # Only train two segmentation branchs
_C.TRAIN.DET_ONLY = False           # Only train detection branch
_C.TRAIN.ENC_SEG_ONLY = False       # Only train encoder and two segmentation branchs
_C.TRAIN.ENC_DET_ONLY = False       # Only train encoder and detection branch
_C.TRAIN.ENC_DET_DRIVABLE_ONLY = False  # Only train encoder / detection and da_segmentation branch

# Single task 
_C.TRAIN.DRIVABLE_ONLY = False      # Only train da_segmentation task


_C.TRAIN.PLOT = False                # 

# testing
_C.TEST = CN(new_allowed=True)
_C.TEST.BATCH_SIZE_PER_GPU = batch_size_per_gpu
_C.TEST.MODEL_FILE = ''
_C.TEST.SAVE_JSON = False
_C.TEST.SAVE_TXT = False
_C.TEST.PLOTS = False
_C.TEST.NMS_CONF_THRESHOLD  = 0.001
_C.TEST.NMS_IOU_THRESHOLD  = 0.6


def update_config(cfg, args):
    cfg.defrost()
    # cfg.merge_from_file(args.cfg)

    # if args.modelDir:
    #     cfg.OUTPUT_DIR = args.modelDir

    if args.logDir:
        cfg.LOG_DIR = args.logDir
    
    cfg.GPUS = args.gpu_nums
    
    # if args.conf_thres:
    #     cfg.TEST.NMS_CONF_THRESHOLD = args.conf_thres

    # if args.iou_thres:
    #     cfg.TEST.NMS_IOU_THRESHOLD = args.iou_thres
    


    # cfg.MODEL.PRETRAINED = os.path.join(
    #     cfg.DATA_DIR, cfg.MODEL.PRETRAINED
    # )
    #
    # if cfg.TEST.MODEL_FILE:
    #     cfg.TEST.MODEL_FILE = os.path.join(
    #         cfg.DATA_DIR, cfg.TEST.MODEL_FILE
    #     )

    cfg.freeze()

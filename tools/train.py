import argparse
import os, sys
import math
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

# os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"

import pprint
import time
import torch
import torch.nn.parallel
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda import amp
import torch.distributed as dist
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import numpy as np
from lib.utils import DataLoaderX, torch_distributed_zero_first
from lib.utils.utils import InfiniteDataLoader
from tensorboardX import SummaryWriter

import lib.dataset as dataset
from lib.config import cfg
from lib.config import update_config
from lib.core.loss import get_loss
from lib.core.function import train
from lib.core.function import validate
from lib.core.general import fitness
from lib.models import get_net
from lib.utils import is_parallel
from lib.utils.utils import get_optimizer
from lib.utils.utils import save_checkpoint
from lib.utils.utils import create_logger, select_device
from lib.utils import run_anchor


def parse_args():
    parser = argparse.ArgumentParser(description='Train Multitask network')
    parser.add_argument('--logDir', help='log directory', type=str, default='runs/')
    parser.add_argument('--sync-bn', action='store_true', help='use SyncBatchNorm, only available in DDP mode')
    parser.add_argument('--local_rank', type=int, default=-1, help='DDP parameter, do not modify')
    parser.add_argument('--conf-thres', type=float, default=0.001, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.6, help='IOU threshold for NMS')
    args = parser.parse_args()

    return args


def main():
    # set all the configurations
    args = parse_args()
    args.gpu_nums = tuple(range(torch.cuda.device_count()))
    update_config(cfg, args)

    # Set DDP variables
    world_size = int(os.environ['WORLD_SIZE']) if 'WORLD_SIZE' in os.environ else 1
    global_rank = int(os.environ['RANK']) if 'RANK' in os.environ else -1

    rank = global_rank
    #print(rank)
    # TODO: handle distributed training logger
    # set the logger, tb_log_dir means tensorboard logdir

    logger, final_output_dir, tb_log_dir = create_logger(cfg, 'train', rank=rank)

    if rank in [-1, 0]:
        # logger.info(pprint.pformat(args))
        # logger.info(cfg)

        writer_dict = {
            'writer': SummaryWriter(log_dir=tb_log_dir),
            'train_global_steps': 0,
            'valid_global_steps': 0,
        }
    else:
        writer_dict = None

    # cudnn related setting
    cudnn.benchmark = cfg.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED

    # bulid up model
    # start_time = time.time()
    print(f"begin to bulid up model, {cfg.MODEL.NAME}...")
    # DP mode
    device = select_device(logger, batch_size=cfg.TRAIN.BATCH_SIZE_PER_GPU* len(cfg.GPUS)) if not cfg.DEBUG \
        else select_device(logger, 'cpu')

    if args.local_rank != -1:
        assert torch.cuda.device_count() > args.local_rank
        torch.cuda.set_device(args.local_rank)
        device = torch.device('cuda', args.local_rank)
        dist.init_process_group(backend='nccl', init_method='env://')  # distributed backend

    nc = cfg.NUM_CLASSES
    model = get_net(cfg, nc=nc).to(device)
    # print("load finished")
    #model = model.to(device)
    # print("finish build model")
    
    model.nc = nc
    # define loss function (criterion) and optimizer
    criterion = get_loss(cfg, device=device)
    optimizer = get_optimizer(cfg, model)


    # load checkpoint model
    best_perf = 0.0

    if cfg.MODEL.NAME == "YOLOP":
        Encoder_para_idx = [str(i) for i in range(0, 17)]
        Det_Head_para_idx = [str(i) for i in range(17, 25)]
        Da_Seg_Head_para_idx = [str(i) for i in range(25, 34)]
        Ll_Seg_Head_para_idx = [str(i) for i in range(34,43)]
    elif cfg.MODEL.NAME == "YOLOv8":
        Encoder_para_idx = [str(i) for i in range(0, 17)]
        Det_Head_para_idx = [str(i) for i in range(17, 24)]
        Da_Seg_Head_para_idx = [str(i) for i in range(24, 36)]
        Ll_Seg_Head_para_idx = [str(i) for i in range(36, 43)]
    elif cfg.MODEL.NAME == "YOLOPv2" or cfg.MODEL.NAME == "YOLOPv2_tiny":
        Encoder_para_idx = [str(i) for i in range(0, 26)]
        Det_Head_para_idx = [str(i) for i in range(27, 38)]
        Da_Seg_Head_para_idx = [str(i) for i in range(38, 51)]
        Ll_Seg_Head_para_idx = [str(i) for i in range(51, 63)]
    elif cfg.MODEL.NAME == "YOLOv8n":
        Encoder_para_idx = []
        Det_Head_para_idx = [str(i) for i in range(0, 23)]
        Da_Seg_Head_para_idx = [str(i) for i in range(23, 35)]
    else:
        raise NameError(f"model not implement: {cfg.MODEL.NAME}")

    lf = lambda x: ((1 + math.cos(x * math.pi / cfg.TRAIN.END_EPOCH)) / 2) * \
                   (1 - cfg.TRAIN.LRF) + cfg.TRAIN.LRF  # cosine
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    begin_epoch = cfg.TRAIN.BEGIN_EPOCH

    if rank in [-1, 0]:
        checkpoint_file = os.path.join(final_output_dir, 'epoch-last.pth')
        # print(checkpoint_file)
        if os.path.exists(cfg.MODEL.AUTHOR_PRETRAINED):
            logger.info("=> loading author provided single class model weight from '{}'".format(cfg.MODEL.AUTHOR_PRETRAINED))
            if model.nc > 1:
                idx_range = [str(i) for i in range(0,17)] + [str(i) for i in range(25,43)]
            else:
                idx_range = [str(i) for i in range(0,43)]
            model_dict = model.state_dict()
            checkpoint = torch.load(cfg.MODEL.AUTHOR_PRETRAINED)
            checkpoint_dict = {k: v for k, v in checkpoint['state_dict'].items() if k.split(".")[1] in idx_range}
            model_dict.update(checkpoint_dict)
            model.load_state_dict(model_dict)
            logger.info("=> loaded author provided single class model checkpoint '{}' ".format(cfg.MODEL.AUTHOR_PRETRAINED))

        elif os.path.exists(cfg.MODEL.PRETRAINED):
            logger.info("=> loading model '{}'".format(cfg.MODEL.PRETRAINED))
            checkpoint = torch.load(cfg.MODEL.PRETRAINED)
            begin_epoch = 0
            best_perf = 0.0  # checkpoint.get('best_perf', 0.0)
            # last_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            # optimizer.load_state_dict(checkpoint['optimizer'])
            logger.info("=> loaded checkpoint '{}' (epoch {})".format(
                cfg.MODEL.PRETRAINED, checkpoint['epoch']))
            #cfg.NEED_AUTOANCHOR = False     #disable autoanchor
        
        elif os.path.exists(cfg.MODEL.PRETRAINED_DET):
            logger.info("=> loading model weight in det branch from '{}'".format(cfg.MODEL.PRETRAINED))
            det_idx_range = [str(i) for i in range(0,25)]
            model_dict = model.state_dict()
            checkpoint_file = cfg.MODEL.PRETRAINED_DET
            checkpoint = torch.load(checkpoint_file)
            begin_epoch = checkpoint['epoch']
            last_epoch = checkpoint['epoch']
            checkpoint_dict = {k: v for k, v in checkpoint['state_dict'].items() if k.split(".")[1] in det_idx_range}
            model_dict.update(checkpoint_dict)
            model.load_state_dict(model_dict)
            logger.info("=> loaded det branch checkpoint '{}' ".format(checkpoint_file))
        
        elif cfg.AUTO_RESUME and os.path.exists(checkpoint_file):
            logger.info("Resume...")
            logger.info("=> loading checkpoint '{}'".format(checkpoint_file))
            checkpoint = torch.load(checkpoint_file)
            begin_epoch = checkpoint['epoch']
            best_perf = checkpoint.get('best_perf', 0.0)
            last_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            # optimizer = get_optimizer(cfg, model)
            optimizer.load_state_dict(checkpoint['optimizer'])
            logger.info("=> loaded checkpoint '{}' (epoch {})".format(
                checkpoint_file, checkpoint['epoch']))
            #cfg.NEED_AUTOANCHOR = False     #disable autoanchor
        # model = model.to(device)

        if cfg.TRAIN.SEG_ONLY:  #Only train two segmentation branchs
            logger.info('freeze encoder and Det head...')
            for k, v in model.named_parameters():
                v.requires_grad = True  # train all layers
                if k.split(".")[1] in Encoder_para_idx + Det_Head_para_idx:
                    # print('freezing %s' % k)
                    v.requires_grad = False

        elif cfg.TRAIN.DET_ONLY:  #Only train detection branch
            logger.info('freeze encoder and da Seg heads...')
            # print(model.named_parameters)
            for k, v in model.named_parameters():
                v.requires_grad = True  # train all layers
                if k.split(".")[1] in Encoder_para_idx + Da_Seg_Head_para_idx:
                    # print('freezing %s' % k)
                    v.requires_grad = False

        elif cfg.TRAIN.ENC_SEG_ONLY:  # Only train encoder and two segmentation branchs
            logger.info('freeze Det head...')
            for k, v in model.named_parameters():
                v.requires_grad = True  # train all layers 
                if k.split(".")[1] in Det_Head_para_idx:
                    # print('freezing %s' % k)
                    v.requires_grad = False

        elif cfg.TRAIN.ENC_DET_ONLY or cfg.TRAIN.DET_ONLY:    # Only train encoder and detection branchs
            logger.info('freeze two Seg heads...')
            for k, v in model.named_parameters():
                v.requires_grad = True  # train all layers
                if k.split(".")[1] in Da_Seg_Head_para_idx:
                    # print('freezing %s' % k)
                    v.requires_grad = False

        elif cfg.TRAIN.DRIVABLE_ONLY:
            logger.info('freeze encoder and Det head...')
            # print(model.named_parameters)
            for k, v in model.named_parameters():
                v.requires_grad = True  # train all layers
                if k.split(".")[1] in Encoder_para_idx + Det_Head_para_idx:
                    # print('freezing %s' % k)
                    v.requires_grad = False
        
    if rank == -1 and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model, device_ids=cfg.GPUS)
        # model = torch.nn.DataParallel(model, device_ids=cfg.GPUS).cuda()
    # # DDP mode
    if rank != -1:
        model = DDP(model, device_ids=[args.local_rank], output_device=args.local_rank,find_unused_parameters=True)


    # assign model params
    model.gr = 1.0
    model.nc = nc
    # print('bulid model finished')

    print("begin to load data")
    # Data loading
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )

    train_dataset = dataset.BddDataset(
        cfg=cfg,
        is_train=True,
        inputsize=cfg.MODEL.IMAGE_SIZE,
        transform=transforms.Compose([
            transforms.ToTensor(),
            # normalize,
        ]),
        self_data=cfg.SELF_DATA
    )
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset) if rank != -1 else None

    train_loader = DataLoaderX(
        train_dataset,
        batch_size=cfg.TRAIN.BATCH_SIZE_PER_GPU * len(cfg.GPUS),
        shuffle=(cfg.TRAIN.SHUFFLE & rank == -1),
        num_workers=cfg.WORKERS,
        sampler=train_sampler,
        pin_memory=True,
        prefetch_factor=8,
        collate_fn=dataset.AutoDriveDataset.collate_fn
    )
    num_batch = len(train_loader)

    if rank in [-1, 0]:
        valid_dataset = dataset.BddDataset(
            cfg=cfg,
            is_train=False,
            inputsize=cfg.MODEL.IMAGE_SIZE,
            transform=transforms.Compose([
                transforms.ToTensor(),
                # normalize,
            ]),
            self_data=cfg.SELF_DATA
        )

        valid_loader = DataLoaderX(
            valid_dataset,
            batch_size=cfg.TEST.BATCH_SIZE_PER_GPU * len(cfg.GPUS),
            shuffle=False,
            num_workers=cfg.WORKERS,
            pin_memory=True,
            prefetch_factor=8,
            collate_fn=dataset.AutoDriveDataset.collate_fn
        )
        print('load data finished')
    
    if rank in [-1, 0]:
        if cfg.NEED_AUTOANCHOR:
            logger.info("begin check anchors")
            run_anchor(logger,train_dataset, model=model, thr=cfg.TRAIN.ANCHOR_THRESHOLD, imgsz=min(cfg.MODEL.IMAGE_SIZE))
        else:
            logger.info("anchors loaded successfully")
            det = model.module.model[model.module.detector_index] if is_parallel(model) \
                else model.model[model.detector_index]
            # logger.info(str(det.anchors))

    # training
    num_warmup = max(round(cfg.TRAIN.WARMUP_EPOCHS * num_batch), 1000)
    scaler = amp.GradScaler(enabled=device.type != 'cpu')
    print('=> start training...')
    for epoch in range(begin_epoch+1, cfg.TRAIN.END_EPOCH+1):
        if rank != -1:
            train_loader.sampler.set_epoch(epoch)
        # train for one epoch
        train(cfg, train_loader, model, criterion, optimizer, scaler,
              epoch, num_batch, num_warmup, writer_dict, logger, device, rank)
        
        lr_scheduler.step()

        # evaluate on validation set
        val_epoch_freq = cfg.TRAIN.VAL_FREQ if epoch > 20 else 5
        val_flag = ((epoch % val_epoch_freq == 0 or epoch == cfg.TRAIN.END_EPOCH) and rank in [-1, 0])
        # if (epoch % cfg.TRAIN.VAL_FREQ == 0 or epoch == cfg.TRAIN.END_EPOCH) and rank in [-1, 0]:
        # if val_flag:
        val_flag = True
        if val_flag:
            # print('validate')
            da_segment_results, detect_results, total_loss, maps, times = validate(
                cfg, valid_loader, model, criterion, final_output_dir, device=device, half=True, rank=rank)
            
            fi_det = fitness(np.array(detect_results).reshape(1, -1), item="det")     # 目标检测评价指标
            fi_da = fitness(np.array(da_segment_results).reshape(1, -1), item="seg")  # 可行驶区域检测评价指标
            if cfg.TRAIN.DET_ONLY or cfg.TRAIN.ENC_DET_ONLY:
                fi = fi_det
            elif cfg.TRAIN.DRIVABLE_ONLY:
                fi = fi_da
            else:
                fi = fi_det + fi_da
                # fi = fi_ll
            
            if fi > best_perf:
                best_perf = fi

            msg = 'Epoch: [{0}]    Loss({loss:.3f})\n' \
                      'Driving area Segment: Acc({da_seg_acc:.3f})    IOU ({da_seg_iou:.3f})    mIOU({da_seg_miou:.3f})\n' \
                      'Detect: P({p:.3f})  R({r:.3f})  mAP@0.5({map50:.3f})  mAP@0.5:0.95({map:.3f})\n'\
                      'Time: inference({t_inf:.4f}s/frame)  nms({t_nms:.4f}s/frame)'.format(
                          epoch,  loss=total_loss, da_seg_acc=da_segment_results[0],da_seg_iou=da_segment_results[1],da_seg_miou=da_segment_results[2],
                          p=detect_results[0],r=detect_results[1],map50=detect_results[2],map=detect_results[3],
                          t_inf=times[0], t_nms=times[1])
            logger.info(msg)

        # save checkpoint model and best model
        if rank in [-1, 0]:
            logger.info('=> saving last checkpoint to {}'.format(final_output_dir))
            
            # last 
            save_checkpoint(
                epoch=epoch,
                name=cfg.MODEL.NAME,
                model=model,
                optimizer=optimizer,
                best_perf=best_perf,
                output_dir=final_output_dir,
                filename=f'epoch-last.pth'
            )

            # best
            if val_flag:
                if best_perf == fi:
                    logger.info('=> saving best checkpoint to {}'.format(final_output_dir))
                    save_checkpoint(
                        epoch=epoch,
                        name=cfg.MODEL.NAME,
                        model=model,
                        optimizer=optimizer,
                        best_perf=best_perf,
                        output_dir=final_output_dir,
                        filename=f'epoch-best.pth'
                    )

    # save final model
    if rank in [-1, 0]:
        if False:
            final_model_state_file = os.path.join(
                final_output_dir, 'final_state.pth'
            )
            logger.info('=> saving final model state to {}'.format(
                final_model_state_file)
            )
            model_state = model.module.state_dict() if is_parallel(model) else model.state_dict()
            torch.save(model_state, final_model_state_file)
        writer_dict['writer'].close()
    else:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
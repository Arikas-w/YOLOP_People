import numpy as np
import json

from .AutoDriveDataset import AutoDriveDataset
from .convert import convert, id_dict, id_dict_single
from tqdm import tqdm
import os
from pathlib import Path
from itertools import repeat
from multiprocessing.pool import ThreadPool, Pool
from threading import Thread
import cv2

single_cls = False        # just detect vehicle
NUM_THREADS = min(8, os.cpu_count())  # number of multiprocessing threads


class BddDataset(AutoDriveDataset):
    def __init__(self, cfg, is_train, inputsize, transform=None, self_data=False):
        super().__init__(cfg, is_train, inputsize, transform)
        self.prefix = "train" if is_train else "val"
        self.self_data = self_data
        self.db, self.shapes = self._get_db()
        self.n = len(self.shapes)

        self.imgs, self.imgs_npy = [None] * self.n, [None] * self.n
        self.segs, self.segs_npy = [None] * self.n, [None] * self.n
        # self.cfg = cfg
        # Cache images into memory for faster training (WARNING: large datasets may exceed system RAM)
        if self.cfg.CACHE:
            self.cache_file()

    def _get_db(self):
        """
        get database from the annotation file

        Inputs:

        Returns:
        gt_db: (list)database   [a,b,c,...]
                a: (dictionary){'image':, 'information':, ......}
        image: image path
        mask: path of the segmetation label
        label: [cls_id, center_x//256, center_y//256, w//256, h//256] 256=IMAGE_SIZE
        """
        print(f'building {self.prefix} database...')
        gt_db = []
        shapes = []
        # img_paths = []
        # seg_paths = []
        # lane_paths = []
        # height, width = self.shapes
        with open(self.train_file, 'r') as f:
            lines = f.readlines()
        # for mask in tqdm(list(self.mask_list)[:]):
        num = len(lines) if self.is_train else 50000
        for line in tqdm(lines[:num]):
            image_path, label_path, mask_path, lane_path = line.strip().split(',')
            
            image = cv2.imread(image_path)
            height, width, _ = image.shape
            if not self.self_data:
                with open(label_path, 'r') as f:
                    labelInfos = f.readlines()
                
                gt = np.zeros((len(labelInfos), 5))
                for idx, labelInfo in enumerate(labelInfos):
                    labelInfo = [float(l) for l in labelInfo.strip().split()]
                    cls_id = int(labelInfo[0])
                    if single_cls:
                        cls_id = 0
                    gt[idx][0] = cls_id
                    box = labelInfo[1:]
                    gt[idx][1:] = box

            rec = [{
                'image': image_path,
                'label': gt if not self.self_data else np.array([]),
                'mask': mask_path,
            }]

            gt_db += rec
            shapes.append([height, width])
        print(f'database {self.prefix} build finish')
        return gt_db, np.array(shapes, dtype=np.float64)

    def filter_data(self, data):
        remain = []
        for obj in data:
            if 'box2d' in obj.keys():  # obj.has_key('box2d'):
                remain.append(obj)
        return remain

    def evaluate(self, cfg, preds, output_dir, *args, **kwargs):
        """  
        """
        pass
    
    def cache_file(self):
        
        if self.cfg.CACHE == "disk":
            # image
            self.im_cache_dir = Path(Path(self.db[0]['image']).parent.as_posix() + '_npy')
            self.imgs_npy = [self.im_cache_dir / Path(d['image']).with_suffix('.npy').name for d in self.db]
            self.im_cache_dir.mkdir(parents=True, exist_ok=True)

            # seg
            self.seg_cache_dir = Path(Path(self.db[0]['mask']).parent.as_posix() + '_npy')
            self.segs_npy = [self.seg_cache_dir / Path(d['mask']).with_suffix('.npy').name for d in self.db]
            self.seg_cache_dir.mkdir(parents=True, exist_ok=True)
        
        # cache image 
        gb_img = 0  # Gigabytes of cached images
        results = ThreadPool(NUM_THREADS).imap(lambda x: self.load_image(x), range(self.n))
        pbar = tqdm(enumerate(results), total=self.n)
        for i, x in pbar:
            if self.cfg.CACHE == 'disk':
                if not self.imgs_npy[i].exists():
                    np.save(self.imgs_npy[i].as_posix(), x)
                gb_img += self.imgs_npy[i].stat().st_size
            else:
                self.imgs[i] = x 
                gb_img += self.imgs[i].nbytes
            pbar.desc = f"{'image train ' if self.is_train else 'image val '}Caching images ({gb_img / 1E9:.1f}GB {self.cfg.CACHE})"
        pbar.close()

        # cache seg
        gb_img = 0  # Gigabytes of cached images
        results = ThreadPool(NUM_THREADS).imap(lambda x: self.load_seg(x), range(self.n))
        pbar = tqdm(enumerate(results), total=self.n)
        for i, x in pbar:
            if self.cfg.CACHE == 'disk':
                if not self.segs_npy[i].exists():
                    np.save(self.segs_npy[i].as_posix(), x)
                gb_img += self.segs_npy[i].stat().st_size
            else:
                self.segs[i] = x 
                gb_img += self.segs[i].nbytes
            pbar.desc = f"{'seg train ' if self.is_train else 'seg val '}Caching images ({gb_img / 1E9:.1f}GB {self.cfg.CACHE})"
        pbar.close()

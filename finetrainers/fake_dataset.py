import io
import os
import csv
import json
import random
import tarfile
import torch
import logging
import numpy as np
import tqdm
import copy
import threading
import torchvision

from PIL import Image
from collections import defaultdict
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torch.utils.data import default_collate, get_worker_info

class FakeDataset(Dataset):
    def __init__(self, dataset_name=None, interval=1, image_size=256, view_num=9, cache_dir='/mnt/dataset_cache', file_name='infos_train_0.json', fix_interval=False, consecutive_idx=True, norm_t=False, shuffle=True, color_aug=False, random_mask_xyz=False):
        self.interval = interval
        self.view_num = view_num
        self.random_mask_xyz = random_mask_xyz

        self.path = '/mnt/petrelfs/liangzhengyang.d/qh_projects/T2I/re10k/'
        self.num_samples = len(os.listdir(self.path))

        self.resolution_buckets = [(9, 256, 256)]

    def __len__(self):
        return 10000

    def __getitem__(self, idx):
        if isinstance(idx, list):
            # Here, index is actually a list of data objects that we need to return.
            # The BucketSampler should ideally return indices. But, in the sampler, we'd like
            # to have information about num_frames, height and width. Since this is not stored
            # as metadata, we need to read the video to get this information. You could read this
            # information without loading the full video in memory, but we do it anyway. In order
            # to not load the video twice (once to get the metadata, and once to return the loaded video
            # based on sampled indices), we cache it in the BucketSampler. When the sampler is
            # to yield, we yield the cache data instead of indices. So, this special check ensures
            # that data is not loaded a second time. PRs are welcome for improvements.
            return idx

        idx = idx % self.num_samples
        op = torch.nn.Sequential(
         torchvision.transforms.Resize(256),
         torchvision.transforms.CenterCrop(256),
        )
        path = os.path.join(self.path, str(idx))
        files = os.listdir(path)
        npy = [each for each in files if 'npy' in each]
        npy = sorted(npy, key=lambda x: int(x.split('.')[0]))
        jpg = [each[:-4] for each in npy]

        xyz = [np.load(os.path.join(path, each)) for each in npy]
        rgb = [np.array(Image.open(os.path.join(path, each))) for each in jpg]

        xyz = op(torch.from_numpy(np.stack(xyz, 0)).permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        rgb = op(torch.from_numpy(np.stack(rgb, 0)).permute(0, 3, 1, 2)) / 255 * 2 - 1

        cameras = json.load(open(os.path.join(path, 'rescaled_camera.json')))
        cameras = [cameras[each[:-4]] for each in npy]

        n_view = len(cameras)
        # max_interval = min(self.interval, (n_view - 1) // (self.view_num-1))
        # interval = random.randint(0, max_interval)
        interval = self.interval
        start = random.randint(0, n_view - (self.view_num - 1) * interval - 1)
        end = start + (self.view_num - 1) * interval
        idx = [start + _ * interval for _ in range(self.view_num)]

        xyz_img = torch.stack([xyz[i] for i in idx], 0)
        img = torch.stack([rgb[i] for i in idx], 0)
        xyz_img = xyz_img.numpy()
        raw_xyz = xyz_img.copy()
        xyz_img = xyz_img - xyz_img.mean((0,1,2,), keepdims=True)
        l, r = np.quantile(xyz_img, 0.02), np.quantile(xyz_img, 0.98)
        xyz_img = np.clip(xyz_img, l, r)
        xyz_img = (xyz_img - l) / (r - l + 1e-5)
        xyz_img = torch.from_numpy(xyz_img * 2 - 1).permute(0, 3, 1, 2)

        ray_map = np.zeros([self.view_num,32,32, 3])
        caption = np.zeros([234])
        tokens = np.zeros([47])
        tokens[0] = 1
        caption[0] = 1
        # idx = np.arange(self.view_num)
        intrinsic = torch.tensor([cameras[i]['intrinsic'] for i in idx])
        w2c = torch.linalg.inv(torch.tensor([cameras[i]['pose'] for i in idx]))
        H = W = 256
        view_num = self.view_num if self.view_num !=-1 else 8
        if self.random_mask_xyz:
            mask_idx = np.random.choice(view_num * 2, size=1, replace=False)
        else:
            mask_idx = np.random.choice(view_num, size=1, replace=False)
        mask = np.zeros([view_num * 2])
        mask[mask_idx] = 1
        prompt = ""

        return dict(video=img, prompt="",
                    video_metadata=dict(num_frames=img.shape[0], height=img.shape[2], width=img.shape[3]),
                    ray_map=ray_map, caption=caption, tokens=tokens, timestep=torch.tensor(idx), time_idxs=torch.tensor(idx), intrinsic=intrinsic, w2c=w2c, H=H, W=W,
                    mask=mask, mask_idx=mask_idx, xyz_img=xyz_img, raw_xyz = raw_xyz)

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

        self.num_samples = 10

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
        img = torch.zeros([self.view_num, 3, 256, 256])
        xyz_img = torch.zeros([self.view_num, 3, 256, 256])

        prompt = ""

        return dict(video=img, xyz_img=xyz_img, prompt="",
                    video_metadata=dict(num_frames=img.shape[0], height=img.shape[2], width=img.shape[3]),)

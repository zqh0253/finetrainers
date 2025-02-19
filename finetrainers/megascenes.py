import io
import os
import csv
import json
import random
import tarfile
import torch
import logging
logging.getLogger('PIL').setLevel(logging.INFO)
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

class MegascenesDataset(Dataset):
    def __init__(self, json_path='', image_size=256, view_num=9, color_aug=False, random_mask_xyz=False):

        with open(json_path, 'r') as f:
            self.data = json.load(f)
        self.num_samples = len(self.data)

        self.view_num = view_num
        self.random_mask_xyz = random_mask_xyz
        self.resolution_buckets = [(9, 256, 256)]

    def __len__(self):
        return self.num_samples
    
    def read_image(self, path):
        image = Image.open(path).convert('RGB')
        image_np = np.array(image)
        image_np = image_np.transpose(2, 0, 1)
        return torch.from_numpy(image_np)

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

        total_view_num = len(self.data[idx])

        op = torch.nn.Sequential(
         torchvision.transforms.Resize(256),
         torchvision.transforms.CenterCrop(256),
        )

        selected_indices = random.sample(range(total_view_num), min(self.view_num, total_view_num))
        rgbs, masks, xyzs, Hs, Ws = [], [], [], [], []
        for index in selected_indices:
            path, mask_path, xyz_path, orientation, description, c2w, K = self.data[idx][index]
            rgb = self.read_image(path)
            _, H, W = rgb.shape
            rgb = op(rgb)
            mask = self.read_image(mask_path)
            xyz = torch.from_numpy(np.load(xyz_path)).permute(2, 0, 1)
            xyz = op(xyz)
            rgbs.append(rgb)
            masks.append(mask)
            xyzs.append(xyz)
            Hs.append(H)
            Ws.append(W)

        img = torch.stack(rgbs) / 255.0 * 2 - 1
        xyz_img = torch.stack(xyzs)
        xyz_min = xyz_img.amin(dim=(0,2,3), keepdim=True) 
        xyz_max = xyz_img.amax(dim=(0,2,3), keepdim=True)
        xyz_img = 2 * (xyz_img - xyz_min) / (xyz_max - xyz_min) - 1


        # tmp 
        img = img[0:1].repeat(self.view_num, 1, 1, 1)
        xyz_img = xyz_img[0:1].repeat(self.view_num, 1, 1, 1)

        prompt = description

        return dict(video=img, xyz_img=xyz_img, prompt=prompt, H=Hs, W=Ws,
                    video_metadata=dict(num_frames=img.shape[0], height=img.shape[2], width=img.shape[3]),)


if __name__ == "__main__":
    dataset = MegascenesDataset(json_path='/home/qihang/data/megascenes_label/infos.json')
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    for data in dataloader:
        print(data['prompt'], data['video'].shape)
        import pdb; pdb.set_trace()
        # print([(a, b, a/b) for a, b in zip(data['H'], data['W'])])

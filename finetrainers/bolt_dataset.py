import io
import os
import json
import re
import random
try:
    import boto3
    from botocore.config import Config
    from boto3.s3.transfer import TransferConfig
    NO_BOTO = False
except:
    print('boto3 is not installed. Return all-zero samples!')
    NO_BOTO = True
import tarfile
import torch
import numpy as np
import time
import math

from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as F
from torchvision import transforms
from .distributed import get_rank
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
import cv2
from io import BytesIO
from scipy.spatial.transform import Rotation 


# ================= helper functions ========================= #

def get_np_array_from_tar_object(tar_extractfl):
     '''converts a buffer from a tar file in np.array'''
     return np.asarray(
        bytearray(tar_extractfl.read())
        , dtype=np.uint8)

def find_factors(n):
    factors = set()
    for i in range(1, int(math.sqrt(n)) + 1):
        if n % i == 0:
            factors.add(i)
            factors.add(n // i)
    return sorted(factors, reverse=True)

def find_max_scale_factor(A, B):
    gcd = math.gcd(A, B)
    
    factors = find_factors(gcd)
    
    for factor in factors:
        if A // factor >= 32 and B // factor >= 32 and abs(A-B)//factor % 2 ==0:
            return factor
    
    return 1 

# ==========================================


class InfiniteDataLoader(torch.utils.data.DataLoader):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Initialize an iterator over the dataset.
        self.dataset_iterator = super().__iter__()

    def __iter__(self):
        return self

    def __next__(self):
        try:
            batch = next(self.dataset_iterator)
        except StopIteration:
            # Dataset exhausted, use a new fresh iterator.
            print('Another Loop over the dataset', flush=True)
            self.dataset_iterator = super().__iter__()
            batch = next(self.dataset_iterator)
        return batch


class DataLoaderWrapper(InfiniteDataLoader):
    def __iter__(self):
        return IterWrapper(super().__iter__())


class IterWrapper:
    def __init__(self, obj):
        self.obj = obj

    def __iter__(self):
        return self
    
    def __next__(self):
        return self.next()

    def next(self):
        return next(self.obj)

    
def _get_plucker_embedding(intrinsic_parameters, w2c_matrices, height, width, norm_t=False, mask_idx=[0], project=False):
    return np.concatenate([
        get_plucker_embedding(intrinsic_parameters, w2c_matrices, height, width, norm_t, idx, project) 
        for idx in mask_idx], -1)
    

def get_plucker_embedding(intrinsic_parameters, w2c_matrices, height, width, norm_t=False, mask_idx=0, project=True):
    """
        intrinsic_parameters.shape = [b f 4]
        c2w_matrices.shape = [b f 4 4]
    """

    num_frames = intrinsic_parameters.shape[0]
    c2w_matrices = np.linalg.inv(w2c_matrices)

    if project:
        w2c_cond_matrices = w2c_matrices[mask_idx: mask_idx+1]
        c2w_matrices = w2c_cond_matrices @ c2w_matrices # relative pose to the first frame

    if norm_t:
        offset = c2w_matrices[:, :3, -1:]  # f, 3, 1
        offset = offset / (np.abs(offset).max(axis=(1, 2), keepdims=True) + 1e-7)
        c2w_matrices[:, :3, -1:] = offset

    ys, xs = np.meshgrid(
        np.linspace(0, height - 1, height, dtype=c2w_matrices.dtype),
        np.linspace(0, width - 1, width, dtype=c2w_matrices.dtype), indexing='ij')
    ys = np.tile(ys.reshape([1, height * width]), [num_frames, 1])  +0.5
    xs = np.tile(xs.reshape([1, height * width]), [num_frames, 1])  +0.5

    fx, fy, cx, cy = np.split(intrinsic_parameters, 4, -1)
    fx, fy, cx, cy = fx * width, fy * height, cx * width, cy * height

    zs_cam = np.ones_like(xs)
    xs_cam = (xs - cx) / fx * zs_cam
    ys_cam = (ys - cy) / fy * zs_cam
    directions = np.stack((xs_cam, ys_cam, zs_cam), -1)
    directions = directions / np.linalg.norm(directions, axis=-1, keepdims=True)
    
    ray_directions_w = (c2w_matrices[..., :3, :3] @ directions.transpose(0, 2, 1)).transpose(0, 2, 1)
    ray_origin_w = np.expand_dims(c2w_matrices[..., :3, 3], axis=-2)
    ray_origin_w = np.broadcast_to(ray_origin_w, ray_directions_w.shape)
    ray_dxo = np.cross(ray_origin_w, ray_directions_w)
    plucker_embedding = np.concatenate([ray_dxo, ray_directions_w], -1).reshape(num_frames, height, width, 6)

    return plucker_embedding


def label_to_camera(dataset_name, label):
    num_frames = label.shape[0]
    bottom = np.zeros([num_frames, 1, 4])
    bottom[:, :, -1] = 1
    if dataset_name == 'arkit':
        # TODO: needs check!
        w2c = label[:, :16].reshape(num_frames, 4, 4)
        intrinsic = label[:, 16:]
        fx, fy, cx, cy = intrinsic[:, 0:1], intrinsic[:, 4:5], intrinsic[:, 2:3], intrinsic[:, 5:6]
        intrinsic = np.concatenate([fx, fy, cx, cy],1)
    elif dataset_name == 'scannet':
        intrinsic = label[:, :4]
        c2w = label[:, 4:].reshape(num_frames, 3, 4)
        c2w = np.concatenate([c2w, bottom], 1)
        w2c = np.linalg.inv(c2w)
    elif dataset_name in ['RealEstate10K' ,'re10k_rgbxyz', 're10k_rgbxyz_samefocal_2']:
        intrinsic = label[:, 1:5]
        w2c = label[:, 7:].reshape(num_frames, 3, 4)
        w2c = np.concatenate([w2c, bottom], 1)
    elif dataset_name in ['re10k_rgbxyz_test']:
        intrinsic = label[:, 0:4]
        w2c = label[:, 6:].reshape(num_frames, 3, 4)
        w2c = np.concatenate([w2c, bottom], 1)
    elif dataset_name in ['co3d_rgbxyz']:
        fx, fy, cx, cy = label[:, 2:3], label[:, 4:5], label[:, 6:7], label[:, 7:8]
        fx, fy, cx, cy = fx / label[:, 1:2], fy / label[:, 0:1], cx / label[:, 1:2], fy / label[:, 0:1]
        intrinsic = np.concatenate([fx, fy, cx, cy],1)
        
        tmp = label[:, 11:27].reshape(num_frames, 4, 4)
        R = tmp[:, :3, :3]
        T = tmp[:, :3, 3]
        tmp[:, :3, :3] = tmp[:, :3, :3].transpose(0, 2, 1)
        w2c = tmp
    elif dataset_name in ['habitat_v2']:
        intrinsic = label[:, 0:4]
        w2c = label[:, 4:].reshape(num_frames, 4, 4)

        # w2c = np.empty((num_frames, 4, 4), dtype=np.float32)
        # w2c[:, :3, :3] = Rotation.from_quat(label[:, :4]).as_matrix()
        # w2c[:, :3, 3] = label[:, 4:]
        # w2c[:, 3, 3] = 1
        
        # hfov = 90.0 * np.pi / 180.
        # f = 1 / np.tan(hfov / 2.)
        # intrinsic = np.array([f, f, 0.5, 0.5], dtype=np.float32)
        # intrinsic = np.tile(intrinsic, (num_frames, 1))
    elif dataset_name == 'MVImgNet' or dataset_name == 'mvimgnet_rgbxyz':
        h, w, f = label[:, 4:5], label[:, 9:10], label[:, 14:15]
        fx, fy, cx, cy = f / w, f / h, np.ones_like(h) * .5, np.ones_like(h) * .5
        intrinsic = np.concatenate([fx, fy, cx, cy],1)
        c2w = np.concatenate([label[:, 0:4], label[:, 5:9], label[:, 10:14]], 1).reshape(num_frames, 3, 4)
        #recover this step: https://github.com/Fyusion/LLFF/blob/master/llff/poses/pose_utils.py#L51
        c2w = np.concatenate([c2w[..., 1:2], c2w[..., 0:1], -c2w[..., 2:3], c2w[..., 3:4]], 2) 
        c2w = np.concatenate([c2w, bottom], 1)
        w2c = np.linalg.inv(c2w)
    elif dataset_name == 'DL3DV_new':
        # [w, h, flx, fly] + camera_model[0] + camera_model[1] + camera_model[2] + camera_model[3]
        w, h, fx, fy = label[:, 0:1], label[:, 1:2], label[:, 2:3], label[:, 3:4]
        fx, fy = fx / w, fy / h
        c2w = label[:, 4:].reshape(num_frames, 4, 4)
        c2w[:, 2, :] *= -1
        c2w = c2w[:, np.array([1, 0, 2, 3]), :]
        c2w[:, 0:3, 1:3] *= -1
        w2c = np.linalg.inv(c2w)
        intrinsic = np.concatenate([fx, fy, np.ones_like(fx) * .5, np.ones_like(fx) * .5], 1)
    elif dataset_name == 'gobj_rawtar':
        num_frames = len(label)
        c2w = np.eye(4)[None].repeat(num_frames, 0)
        x = np.stack([each['x'] for each in label])
        y = np.stack([each['y'] for each in label])
        z = np.stack([each['z'] for each in label])
        o = np.stack([each['origin'] for each in label])
        fx = np.stack([each['x_fov'] for each in label])[:, None]
        fy = np.stack([each['y_fov'] for each in label])[:, None]
        fx = fy = np.ones_like(fx) * 1422.222 / 1024  # it is hard coded according to: https://github.com/modelscope/richdreamer/issues/10#issuecomment-1890870640 
        c2w[:, :3, 0] = np.array(x)
        c2w[:, :3, 1] = np.array(y)
        c2w[:, :3, 2] = np.array(z)
        c2w[:, :3, 3] = np.array(o)
        # print(fx, fy)
        w2c = np.linalg.inv(c2w)
        intrinsic = np.concatenate([fx, fy, np.ones_like(fx) * .5, np.ones_like(fx) * .5], 1)
    # elif dataset_name == 're10k_rgbxyz':
    #     f = label[:, 0:1]
    #     H, W = 288, 512
    #     fx, fy = f / W, f / H
    #     intrinsic = np.concatenate([fx, fy, np.ones_like(fx) * .5, np.ones_like(fx) * .5], 1)
    #     c2w = label[:, 1:].reshape(num_frames, 4, 4)
    #     w2c = np.linalg.inv(c2w)
    else:
        raise NotImplementedError
    
    return intrinsic, w2c


class MultiViewXYZDataset(Dataset):
    if not NO_BOTO:
        # create conductor session
        retry_config = Config(
        retries={
                'max_attempts': 10,
                'mode': 'adaptive'
            }
        )
        transfer_config = TransferConfig(
            # max_concurrency=10,  # Number of threads to use for the transfer
            use_threads=True,     # Set to False to disable threading
            multipart_chunksize=8 * 1024 * 1024,  
            multipart_threshold=8 * 1024 * 1024,
            # max_bandwidth = 50 * 1024 * 1024
        )
        session = boto3.Session(profile_name="conductor-notary")
        conductor_endpoint = "https://conductor.data.apple.com"
        conductor = session.client('s3', endpoint_url=conductor_endpoint, config=retry_config)
        
    def __init__(self, dataset_name, interval=1, image_size=256, view_num=4, cache_dir='/mnt/dataset_cache', file_name='infos_train_0.json', fix_interval=False, consecutive_idx=True, norm_t=False, shuffle=True, color_aug=False, random_mask_xyz=False, project_camera=True):
        self.dataset_name = dataset_name
        self.view_num = view_num
        self.cache_dir = cache_dir
        self.interval = interval
        self.consecutive_idx = consecutive_idx
        self.norm_t = norm_t
        self.fix_interval = fix_interval
        self.color_aug = color_aug
        self.random_mask_xyz = random_mask_xyz
        self.project_camera = project_camera

        self.resolution_buckets = [(view_num, image_size, image_size)]

        seed = 1234 + get_rank()
        self.rng = np.random.default_rng(seed)

        # load dataset json file
        self.info = json.load(open(file_name))

        # filter sequences with inadequate number of views
        length = [v['meta']['n_views'] if 'n_views' in v['meta'] else len(v['views']) for k, v in self.info.items()]
        self.seq_names_and_copy = [(k, v['meta']['copy']) for (k, v, l) in zip(self.info.keys(), self.info.values(), length) if l >= (self.view_num-1) * (self.interval if self.fix_interval else 2) + 1]
        self.seq_names = [seq[0] for seq in self.seq_names_and_copy for _ in range(seq[1])]
        if shuffle:
            random.shuffle(self.seq_names)  # random shuffle for mixed-dataset training
        # self.seq_names = list(self.info.keys()) 

        # define image transform
        self.transform = transforms.Compose([
            transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BILINEAR, antialias=True),
            transforms.CenterCrop(image_size),
        ])
        self.color_transform = transforms.ColorJitter(brightness=.5, hue=.1)
        self.depth_transform = transforms.Compose([
            transforms.Resize(image_size, interpolation=transforms.InterpolationMode.NEAREST_EXACT, antialias=False),
            transforms.CenterCrop(image_size),
        ])
        self.raymap_transform = transforms.Compose([
            transforms.Resize(32, antialias=True),
            transforms.CenterCrop(32),
        ])

        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)

        self.max_retry_n = 10
        
    def __len__(self):
        return len(self.seq_names)

    def __getitem__(self, index):
        seq_name = self.seq_names[index]
        info = self.info[seq_name]
        dataset_name = info['meta']['dataset_name']
    
        i = 0
        while 1:
            if i == self.max_retry_n:            
                print('FAIL:', f'{dataset_name}/{seq_name}.tar', flush=True)
                # raise ValueError
                return self.__getitem__((index + 1) % len(self))
            try:
                file_stream = io.BytesIO()
                self.conductor.download_fileobj('aws-qihang-1', f'{dataset_name}/{seq_name}.tar', file_stream, Config=self.transfer_config)
                file_stream.seek(0)
                tar = tarfile.open(fileobj=io.BytesIO(file_stream.read()), mode='r')
                break
            except:
                i += 1
                pass
        
        n_view = info['meta']['n_views'] if 'n_views' in info['meta'] else len(info['views'])
        if self.view_num == -1:
            start, end = 0, n_view
            idx = list(range(start, end, self.interval))
        else:
            if not self.consecutive_idx:
                idx = random.sample(range(n_view), self.view_num)
            else:
                ratio = 4 if dataset_name in ['habitat_v2'] else 1
                if self.fix_interval:
                    interval = self.interval
                    if dataset_name in ['habitat_v2']:
                        interval = interval // 4
                
                else:
                    max_interval = min(self.interval, (n_view - 1) // (self.view_num-1))
                    interval = self.rng.integers(1, max_interval // ratio + 1)
                fps = 30 / float(interval * ratio)
                start = self.rng.integers(0, n_view - (self.view_num - 1) * interval)
                end = start + (self.view_num - 1) * interval
                
                # idx = list(range(start, end+1, interval))
                idx = [start + _ * interval for _ in range(self.view_num)]
                      
        sort_fn = lambda x: int(re.search(r'(\d+)', x).group(1))
        keys = sorted(list(info['views'].keys()), key=sort_fn)
        names = [keys[_] for _ in idx]
        try:
            raw_img = [Image.open(tar.extractfile(name)) for name in names]
        except:
            print('info:', dataset_name, seq_name, names, flush=True)
            assert 1== 0

        img = [self.transform(_.convert('RGB')) for _ in raw_img]
        img = np.stack(img, 0)

        # color transformation
        if self.color_aug:
            img = torch.from_numpy(img).permute(0, 3, 1, 2) / 255
            img = self.color_transform(img)
            img = (img.permute(0, 2, 3, 1) * 255).numpy().astype(np.uint8)

        if dataset_name in [
            're10k_rgbxyz','re10k_rgbxyz_samefocal_2', 'arkit', 'scannet', 
            'mvimgnet_rgbxyz', 'co3d_rgbxyz', 'habitat_v2', 'dl3dv_rgbxyz']:
            depth_names = [each+'.npy' for each in names]
            xyz = [np.load(BytesIO(tar.extractfile(name).read())) for name in depth_names]
            if dataset_name == 'habitat_v2':
                xyz = [each[:3].reshape(3, 256, 256).transpose(1, 2, 0) for each in xyz]
            xyz = [self.depth_transform(torch.from_numpy(each).permute(2, 0, 1)) for each in xyz]
        elif dataset_name in ['re10k_rgbxyz_test']:
            # TODO: I haven't run dust3r reconstruction on re10k test set.
            xyz = [np.zeros_like(each).transpose(2, 0, 1) for each in img]
        else:
            pass

        if 'meta' in info.keys():
            meta = info['meta']
              
        if 'views' in info.keys() and info['views'] is not None:
            view_num = self.view_num if self.view_num !=-1 else 8
            if self.random_mask_xyz:
                mask_idx = np.random.choice(view_num * 2, size=1, replace=False)
            else:
                mask_idx = np.random.choice(view_num, size=1, replace=False)
            mask = np.zeros([view_num * 2])
            mask[mask_idx] = 1

            # XYZ maps
            height, width = img.shape[1:3]
            xyz_img = np.stack(xyz)[:, :3]  # HACK: ignore the fourth-channel for now
            xyz_center = xyz_img.mean((0,2,3,), keepdims=True)
            xyz_img = xyz_img - xyz_center
            l, r = np.quantile(xyz_img, 0.02), np.quantile(xyz_img, 0.98)
            xyz_img = np.clip(xyz_img, l, r)
            xyz_img = (xyz_img - l) / (r - l + 1e-5)
            xyz_img = xyz_img * 255
            xyz_img = xyz_img.astype(np.uint8).transpose([0, 2, 3, 1])[..., :3]
            
            # load cameras
            if dataset_name not in ['habitat_v2', 're10k_rgbxyz_test']:
                # load the rescaled camera
                try:
                    cameras = json.load(tar.extractfile('rescaled_camera.json'))
                except Exception as e:
                    print("missing camera file:", dataset_name, seq_name)
                    return self.__getitem__(index + 137)
                
                cameras = [cameras[name] for name in names]
                w2c = np.asarray([np.linalg.inv(np.asarray(c['pose'])) for c in cameras])
                intrinsic = np.asarray([c['intrinsic'] for c in cameras]).reshape(-1, 9)
                intrinsic = np.concatenate([intrinsic[:, 0:1], intrinsic[:, 4:5], intrinsic[:, 2:3], intrinsic[:, 5:6]], -1)
            else:
                # habitat camera is already rescaled
                labels = info['views']    
                label = [labels[name] for name in names]
                label = np.array(label)
                intrinsic, w2c = label_to_camera(dataset_name, label)
            
            def normalize_camera(w2c, center, scale):
                c2w = np.linalg.inv(w2c)
                c2w[:3, 3] = c2w[:3, 3] - center
                c2w[:3, :] = c2w[:3, :] / scale
                w2c = np.linalg.inv(c2w)
                return w2c
                 
            # shift camera to match xyz
            w2c = np.array([normalize_camera(w, xyz_center[0, :, 0, 0]+l, (r-l+1e-5)) for w in w2c])
            
            # construct raymaps    
            # factor = find_max_scale_factor(height, width) 
            # H, W = height // factor, width // factor
            H = W = 32
            ray_map = _get_plucker_embedding(intrinsic, w2c, H, W, 
                                             norm_t=self.norm_t, 
                                             mask_idx=mask_idx % view_num,
                                             project=self.project_camera)
            ray_map = torch.from_numpy(ray_map).permute(0, 3, 1, 2)
            ray_map = F.resize(transforms.CenterCrop(min(H, W))(ray_map), 32).permute(0, 2, 3, 1)

        else:
            intrinsic = w2c = H = W = ray_map = 0
        
        idx = [cnt for cnt, each in enumerate(idx)]
        return dict(
            videos=img, xyz_videos=xyz_img, ray_map=ray_map, prompts="",
            video_metadata=dict(num_frames=img.shape[0], height=img.shape[1], width=img.shape[2]), 
            timestep=torch.tensor(idx), time_idxs=torch.tensor(idx),
            fps=torch.tensor([fps]),
            intrinsic=intrinsic, w2c=w2c, H=H, W=W, 
            mask=mask, mask_idx=mask_idx)

    
if __name__ == '__main__':
    import time
    from nerfvis import scene
    from utils.vis import HtmlPageVisualizer
    from tqdm import trange
    import open3d as o3d

    # test RealEstate10K
    # dataset = MultiViewDataset('RealEstate10K')
    dataset = MultiViewXYZDataset('f', file_name='infos_train_0.json', interval=2, fix_interval=False,  view_num=21)
    import pdb;pdb.set_trace()
    N = 10
    page = HtmlPageVisualizer(num_rows=N*4, num_cols=12)

    scene.set_title(f'Scene')
    scene.set_opencv()
    mlist = []
    for i in trange(N):
        data = dataset[i]
        # del data
        # memory = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024  # MB
        # mlist.append(memory)
        # print(memory)
        for j in range(12):
            page.set_cell(4*i, j, image=data['image'][j])
            noisy_rgb = np.clip((data['image'][j] + np.random.randn(256, 256, 3) * 40), 0, 255).astype(np.uint8)
            page.set_cell(4*i+1, j, image=noisy_rgb)
            page.set_cell(4*i+2, j, image=data['xyz_img'][j])
            noisy_xyz = np.clip((data['xyz_img'][j] + np.random.randn(256, 256, 3) * 40), 0, 255).astype(np.uint8)
            page.set_cell(4*i+3, j, image=noisy_xyz)
        
        for j in range(data['xyz_img'].shape[0]):
            scene.add_points(f"points/{i}/{j}", data['xyz_img'][j].reshape(-1, 3), vert_color=data['image'][j].reshape(-1, 3))

            point_cloud = o3d.geometry.PointCloud()
            point_cloud.colors = o3d.utility.Vector3dVector(data['image'][j].reshape(-1, 3) / 255)
            point_cloud.points = o3d.utility.Vector3dVector(data['xyz_img'][j].reshape(-1, 3))
           
            o3d.io.write_point_cloud(f"nerfvis/{i}_{j}_colored_point_cloud.ply", point_cloud)

    scene.export(f'nerfvis')
    page.save('nerfvis.html')

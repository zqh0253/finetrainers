import argparse
import boto3.session
from concurrent.futures import ProcessPoolExecutor, as_completed, wait, FIRST_COMPLETED
import csv
from pathlib import Path
import logging
import os
from pathlib import Path
import s3_helpers
import shutil
import tempfile
import time
import yaml
import subprocess
import tarfile
from PIL import Image
from io import BytesIO
import json
from tqdm import tqdm

def main(args):
    session = boto3.Session(profile_name="conductor-notary")
    dataset_config_files = args.dataset_config_file.split(':')
    if args.copy_ratio is None:
        copy_ratios = [1 for _ in dataset_config_files]
    else:
        copy_ratios = args.copy_ratio.split(':')
        copy_ratios = [int(_) for _ in copy_ratios]
        assert len(copy_ratios) == len(dataset_config_files)
    total_json = dict()
    total_eval_json = dict()
    for it, dataset_config_file in enumerate(dataset_config_files):
        copy_ratio = copy_ratios[it]
        with open(dataset_config_file, 'r') as file:
            config = yaml.safe_load(file)
        assert args.subset in ['train', 'eval'] 
        endpoint_url = config[args.subset].get('endpoint_url', args.endpoint_url)
        conductor = session.client('s3', endpoint_url=endpoint_url)
        
        
        # download train sets
        files = []
        patterns = config['train']['files']
        for pattern in patterns:
            cur_file = s3_helpers.get_file_list(pattern, endpoint_url=endpoint_url)
            files.extend(cur_file)
        
        node_idx = args.worker_id
        total_num = args.num_downloaders
        # n = len(files) // total_num
        # start, end = node_idx * n, node_idx * n + n if node_idx < total_num - 1 else len(files)
        for file_name in tqdm(files[node_idx:][::total_num]):
            splitted_file_name = file_name.split('/')
            max_try, cur_cnt = 10, 0
            while 1:
                try: 
                    file_stream = BytesIO()
                    conductor.download_fileobj(splitted_file_name[2], '/'.join(splitted_file_name[3:]), file_stream)
                    file_stream.seek(0)
                    tmp_json = json.load(file_stream)
                    for k, v in tmp_json.items():
                        tmp_json[k]['meta']['dataset_name'] = '/'.join(splitted_file_name[3:-2])
                        tmp_json[k]['meta']['copy'] = copy_ratio
                    total_json.update(tmp_json)
                    break
                except:
                    cur_cnt += 1
                    if cur_cnt == max_try:
                        print('max try!', '/'.join(splitted_file_name[3:]), flush=True)
                        assert 1==0
                    else:
                        pass     
                    
        # download eval sets
        patterns = config['eval']['files']
        files = []
        for pattern in patterns:
            cur_file = s3_helpers.get_file_list(pattern, endpoint_url=endpoint_url)
            files.extend(cur_file)

        for file_name in tqdm(files):
            splitted_file_name = file_name.split('/')
            max_try, cur_cnt = 10, 0
            while 1:
                try: 
                    file_stream = BytesIO()
                    conductor.download_fileobj(splitted_file_name[2], '/'.join(splitted_file_name[3:]), file_stream)
                    file_stream.seek(0)
                    tmp_json = json.load(file_stream)
                    for k, v in tmp_json.items():
                        tmp_json[k]['meta']['dataset_name'] = '/'.join(splitted_file_name[3:-2])
                        tmp_json[k]['meta']['copy'] = copy_ratio
                    total_eval_json.update(tmp_json)
                    break
                except:
                    cur_cnt += 1
                    if cur_cnt == max_try:
                        print('max try!', '/'.join(splitted_file_name[3:]), flush=True)
                        assert 1==0
                    else:
                        pass
        
        print(dataset_config_file, len(total_json), len(total_eval_json))
    
    json_path = f'infos_{args.subset}_{node_idx}.json'
    with open(json_path, 'w') as f:
        json.dump(total_json, f)
    print(f'dump to {json_path}!', flush=True)
    json_path = f'infos_eval.json'
    with open(json_path, 'w') as f:
        json.dump(total_eval_json, f)
    print(f'dump to {json_path}!', flush=True)
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Download tar files referred to in index file from mlx')
    parser.add_argument('--dataset-config-file',
                        type=str,
                        default='',
                        help='yaml file with dataset names'
                        )
    parser.add_argument('--copy-ratio',
                        type=str,
                        default=None,
                        help='the mix ratio of datasets')
    parser.add_argument('--worker-id',
                        type=int,
                        default=0,
                        help='current worker in [0, num-downloaders -1]'
                        )
    parser.add_argument('--num-downloaders',
                        type=int,
                        default=1,
                        help='number of parallel downloaders'
                        )
    parser.add_argument('--no_bandwidth', action='store_true')
    parser.add_argument('--download_tar', action='store_true',
                        help='whether or not to download tar files also'
                        )
    parser.add_argument('--pretrained-text-embeddings',
                        type=str, default=None)
    parser.add_argument('--endpoint-url',
                        type=str,
                        default='https://blob.mr3.simcloud.apple.com',
                        help='end point for the s3 bucket'
                        )
    parser.add_argument('--subset',
                        type=str,
                        default='train',
                        choices=['train', 'eval'],
                        help='subset to download [train|eval]'
                        )
    args = parser.parse_args()
    logging.basicConfig(level='INFO',
                        format=('[%(asctime)s] {%(pathname)s:%(lineno)d}'
                                '%(levelname)s - %(message)s'),
                        datefmt='%H:%M:%S')
    print(args)
    main(args)
import turibolt as bolt
import argparse
import tqdm
import shutil, os, time

parser = argparse.ArgumentParser()
parser.add_argument('ids', nargs='+')
parser.add_argument('--sleep', type=int, default=3600)
args = parser.parse_args()
workspace = bolt.ARTIFACT_DIR


while 1:
    if os.path.exists(workspace):
        shutil.rmtree(workspace)
    os.makedirs(workspace, exist_ok=True)
    for id in tqdm.tqdm(args.ids):
        task = bolt.get_task(id)
        name = task.name
        for fd in task.artifacts.list():
            if fd[-1] == '/' and f'{fd}logs/' in task.artifacts.list(fd):
                task.artifacts.download_dir(f'{fd}logs/', f"{workspace}/{id}:{name}/{fd}")
    time.sleep(args.sleep)    
            
# 4sg4xwmsr9 3jg6684dh2 au2hha3bqx wjvrcmwmrz gs5jy4uvn9 i2mb933nz3 yjrh4z6u9c 5zukgn6inq
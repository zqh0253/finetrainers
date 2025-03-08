import yaml
import turibolt as bolt
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('--config_file', type=str, default='accelerate_config.yaml')
args = parser.parse_args()

my_task = bolt.get_task(bolt.get_current_task_id())
try:
    subtasks = list(my_task.distributed_subtasks())
    main_task = [t for t in subtasks if t.global_rank==0][0]
    gpus = [t.resources.num_gpus for t in subtasks]
except:
    subtasks = [my_task]
    main_task = my_task
    gpus = [main_task.resources.num_gpus]

config = {
    'compute_environment': 'LOCAL_MACHINE',
    'distributed_type': 'MULTI_GPU',
    'downcast_bf16': 'no',
    'gpu_ids': 'all',
    'machine_rank': my_task.global_rank,
    'main_process_ip': main_task.host_ip_address,
    'main_process_port': main_task.distributed_port,
    'main_training_function': 'main',
    'mixed_precision': 'fp16',
    'num_machines': len(subtasks),
    'num_processes': sum(gpus),
    'rdzv_backend': 'static',
    'same_network': False,
    'tpu_env': [],
    'tpu_use_cluster': False,
    'tpu_use_sudo': False,
    'use_cpu': False
}

with open(args.config_file, 'w') as f:
    yaml.dump(config, f)
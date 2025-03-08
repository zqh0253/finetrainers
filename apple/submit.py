import yaml
import turibolt as bolt
import click


@click.command()

# Required.
@click.option('--name', help='task name', default=None)
@click.option('--config', help='config file', required=True)
@click.option('--ngpu', default=8)
@click.option('--days', default=14)
@click.option('--tags', default=None)
@click.option('--interactive', default=False)
@click.option('--run', default=None)
@click.option('--wandb_key', default=None)
def main(name, config, ngpu, days, tags, run, interactive, wandb_key):
    with open(config) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    if name is not None:
        config['name'] = name
    if 'num_nodes' in config['resources']:
        num_nodes = config['resources']['num_nodes']
    else:
        num_nodes = 1
    config['resources']['num_gpus'] = ngpu
    config['resources']['timeout'] = f'{days}d'
    if tags is not None:
        for c in tags.split(','):
            config['tags'].append(c)
    if run is not None:
        config['command'] = f"bash apple/lunch.sh {run} {ngpu * num_nodes}" # assuming scripts accepts #GPUs and name

    config['attribution']['project'] = config['name']
    config['attribution']['fm_development_type'] = "training"


    LD_LIBRARY_PATH = ["/usr/local/cuda/compat", "/usr/local/cuda/compat/lib",
                        "/opt/aws-ofi-nccl/install/lib",
                        "/miniforge/envs/iris/lib/python3.10/site-packages/nvidia/cuda_nvrtc/lib"]
    LD_LIBRARY_PATH = ":".join(LD_LIBRARY_PATH)
    cluster = config['resources']['cluster']
    if 'gcp' in cluster:
        config['environment_variables'] = {
            'NCCL_NET_PLUGIN': 'gcp',
            "LD_LIBRARY_PATH": f"{LD_LIBRARY_PATH}:$LD_LIBRARY_PATH",
            }
    elif 'aws' in cluster:
        config['environment_variables'] = {
            "FI_EFA_USE_DEVICE_RDMA": '1',
            "FI_PROVIDER": 'efa',
            "RDMAV_FORK_SAFE": '1',
            "LD_LIBRARY_PATH": f"{LD_LIBRARY_PATH}:$LD_LIBRARY_PATH",
            }
        config['resources']['rdma'] = True
    else:
        raise NotImplementedError(f"Cluster {cluster} not supported")

    
    if wandb_key is not None:
        config['environment_variables']['WANDB_API_KEY'] = wandb_key

    config['environment_variables']['EXPERIMENT_NAME'] = config['name']
    # submit
    bolt.submit(config, tar='.', interactive=interactive)

if __name__ == "__main__":
    main() # pylint: disable=no-value-for-parameter



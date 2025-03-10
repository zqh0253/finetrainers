import datetime
import os
import logging
import torch
import torch.distributed as dist
try:
    import irisctl.api as irisctl
except:
    print('no irisctl')

from fnmatch import fnmatch
from functools import partial

from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    CheckpointImpl,
    apply_activation_checkpointing,
    checkpoint_wrapper,
)
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.api import (
    FullOptimStateDictConfig,
    FullStateDictConfig,
    StateDictType,
)
from torch.distributed.fsdp import ShardingStrategy
from torch.distributed.fsdp.wrap import ModuleWrapPolicy


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def init_distributed_multinode(timeout=0):

    master_host = ""
    world_size = 0
    for tasklet in irisctl.distributed_tasklets():
        if tasklet.role_rank == 0:
            master_host = f"{tasklet.host_ip_address}:{tasklet.distributed_port}"
        world_size += 1
    print(
        f"Init PyTorch DDP with master host {master_host}, "
        f"world size {world_size}, rank {irisctl.role_rank()}"
    )
    if timeout == 0:
        timeout = dist.default_pg_timeout
    else:
        timeout = datetime.timedelta(seconds=timeout)

    logging.info(f'Default timeout: {timeout}')
    if world_size >= 1:
        torch.distributed.init_process_group(
            backend="nccl",
            init_method="tcp://" + master_host,
            world_size=world_size,
            timeout=timeout,
            rank=irisctl.role_rank(),
        )

    logging.info("Starting {} workers with rank {}".format(world_size, irisctl.role_rank()))
    # Pick a GPU based on the local rank
    torch.cuda.set_device(irisctl.local_rank())

    dist.barrier()
    setup_for_distributed(irisctl.local_rank() == 0)
    return irisctl.local_rank(), irisctl.role_rank(), world_size


def init_distributed_singlenode(timeout=0):
    # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
    dist_url = "env://" # default

    # only works with torch.distributed.launch // torch.run
    rank = int(os.environ["RANK"])
    world_size = int(os.environ['WORLD_SIZE'])
    local_rank = int(os.environ['LOCAL_RANK'])

    if timeout == 0:
        timeout = dist.default_pg_timeout
    else:
        timeout = datetime.timedelta(seconds=timeout)

    logging.info(f'Default timeout: {timeout}')
    dist.init_process_group(
            backend="nccl",
            init_method=dist_url,
            world_size=world_size,
            timeout=timeout,
            rank=rank)

    # this will make all .cuda() calls work properly
    torch.cuda.set_device(local_rank)
    # synchronizes all the threads to reach this point before moving on
    dist.barrier()
    logging.info(f'setting up local_rank {local_rank} global_rank {rank} world size {world_size}')
    setup_for_distributed(rank == 0)
    return local_rank, rank, world_size


def get_rank():
    return torch.distributed.get_rank() if torch.distributed.is_initialized() else 0


def get_local_rank():
    try:
        return irisctl.local_rank()
    except:
        return int(os.environ.get('LOCAL_RANK', '0'))
    

def get_world_size():
    return torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1


def set_barriar():
    if torch.distributed.is_initialized():
        torch.distributed.barrier()


def print0(*args, **kwargs):
    if get_rank() == 0:
        print(*args, **kwargs)


def init_fsdp(model, module_classes, 
              ignored_classes=[], 
              grad_checkpoint_classes=[],
              grad_ckpt_every=1):
    # Create set of modules to FSDP wrap
    layers = set()
    for module in model.modules():
        name = module.__class__.__name__
        for layer in module_classes:
            if fnmatch(name, layer):
                layers.add(module.__class__)
    
    ignore_layers = set()
    for module in model.modules():
        name = module.__class__.__name__
        for layer in ignored_classes:
            if fnmatch(name, layer):
                ignore_layers.add(module)

    # FSDP wrap the model
    model = torch.distributed.fsdp.FullyShardedDataParallel(
        model,
        sharding_strategy=ShardingStrategy.HYBRID_SHARD,
        auto_wrap_policy=ModuleWrapPolicy(layers),
        ignored_modules=ignore_layers,
        device_id=torch.cuda.current_device(),
        mixed_precision=torch.distributed.fsdp.MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.bfloat16,
            buffer_dtype=torch.float32,
            cast_forward_inputs=False,
        ),
        sync_module_states=False,
        use_orig_params=True,
    )

    # Gradient Checkpointing wrapper
    if len(grad_checkpoint_classes) > 0:
        layers_grad_checkpoint = set()
        for module in model.modules():
            name = module.__class__.__name__
            for layer in grad_checkpoint_classes:
                if fnmatch(name, layer):
                    layers_grad_checkpoint.add(module.__class__)
        wrapper = partial(
            checkpoint_wrapper,
            checkpoint_impl=CheckpointImpl.NO_REENTRANT,
        )
        check_fn = lambda submodule: (  # noqa: E731
            any(isinstance(submodule, layer) for layer in layers_grad_checkpoint) and \
            getattr(submodule, 'layer_index', 0) % grad_ckpt_every == 0
        )
        apply_activation_checkpointing(
            model, checkpoint_wrapper_fn=wrapper, check_fn=check_fn
        )

    FSDP.set_state_dict_type(
        model,
        StateDictType.FULL_STATE_DICT,
        FullStateDictConfig(rank0_only=True, offload_to_cpu=True),
        FullOptimStateDictConfig(rank0_only=True, offload_to_cpu=True),
    )

    return model


def get_fsdp_optim_state(model, optim):
    original_osd = optim.state_dict()
    optim_state_dict = FSDP.optim_state_dict(
        model, optim, 
        optim_state_dict=original_osd)
    return optim_state_dict


def load_fsdp_optim_state(model, optim, optim_state_dict):
    optim_state_dict = FSDP.optim_state_dict_to_load(
        model, optim, optim_state_dict)
    optim.load_state_dict(optim_state_dict)
    

def gather(tensor):
    if torch.distributed.is_initialized():
        gathered_tensors = [torch.zeros_like(tensor) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered_tensors, tensor)
    else:
        gathered_tensors = [tensor]
    return gathered_tensors


def mean(tensor):
    if torch.distributed.is_initialized():
        tensor_clone = tensor.clone()
        dist.all_reduce(tensor_clone, op=dist.ReduceOp.SUM)
        tensor_mean = tensor_clone / dist.get_world_size()
    else:
        tensor_mean = tensor
    return tensor_mean

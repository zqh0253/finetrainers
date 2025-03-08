from modeling.mtok import MTok, MTokPrior
from modeling.tatitok import TATiTok
import turibolt as bolt
import demo_util
import os, torch

def load_bolt_tokenizer(task_id, device=None):
    workspace = os.environ.get('WORKSPACE', '')
    task = bolt.get_task(task_id)
    for fd in task.artifacts.list():
        if fd[-1] == '/':
            model_folder = fd[:-1]
            iters = sorted([int(x[:-1].split('-')[-1]) 
                            for x in task.artifacts.list(fd) if x.startswith(f'{model_folder}/checkpoint-')])[-1]  
            break
    print(task_id, model_folder, iters)
    
    os.makedirs(f"{workspace}/{task_id}", exist_ok=True)
    if not os.path.exists(f"{workspace}/{task_id}/pytorch_model_{iters}.bin"):
        task.artifacts.download_file(src=f"{model_folder}/config.yaml", dest=f"{workspace}/{task_id}/config.yaml", overwrite=True)
        task.artifacts.download_file(src=f"{model_folder}/checkpoint-{iters}/ema_model/pytorch_model.bin", dest=f"{workspace}/{task_id}/pytorch_model_{iters}.bin", overwrite=True)

    model_path = f"{workspace}/{task_id}"
    config = demo_util.get_config(f"{model_path}/config.yaml")
    project = config.experiment.project
    if project == 'tatitok':
        tokenizer = TATiTok(config)
    elif project == 'mtok':
        tokenizer = MTok(config)
    else:
        raise ValueError(f"Unknown project: {project}")
    tokenizer.load_state_dict(torch.load(f"{model_path}/pytorch_model_{iters}.bin", map_location="cpu"))
    tokenizer.eval()
    tokenizer.requires_grad_(False)
    tokenizer.to(device)
    return tokenizer


def load_bolt_generator(task_id, device=None):
    workspace = os.environ.get('WORKSPACE', '')
    task = bolt.get_task(task_id)
    for fd in task.artifacts.list():
        if fd[-1] == '/':
            model_folder = fd[:-1]
            iters = sorted([int(x[:-1].split('-')[-1]) 
                            for x in task.artifacts.list(fd) if x.startswith(f'{model_folder}/checkpoint-')])[-1]  
            break
    print(task_id, model_folder, iters)
    
    os.makedirs(f"{workspace}/{task_id}", exist_ok=True)
    if not os.path.exists(f"{workspace}/{task_id}/pytorch_model_{iters}.bin"):
        task.artifacts.download_file(src=f"{model_folder}/config.yaml", dest=f"{workspace}/{task_id}/config.yaml", overwrite=True)
        task.artifacts.download_file(src=f"{model_folder}/checkpoint-{iters}/unwrapped_model/pytorch_model.bin", dest=f"{workspace}/{task_id}/pytorch_model_{iters}.bin", overwrite=True)

    model_path = f"{workspace}/{task_id}"
    config = demo_util.get_config(f"{model_path}/config.yaml")
    tokenizer_id = config.model.vq_model.tokenizer_task_id
    tokenizer = load_bolt_tokenizer(tokenizer_id, device)
    config.model.vq_model.update(tokenizer.config.model.vq_model)
    
    generator = MTokPrior(config)
    generator.load_state_dict(torch.load(f"{model_path}/pytorch_model_{iters}.bin", map_location="cpu"))
    generator.eval()
    generator.requires_grad_(False)
    generator.to(device)
    return tokenizer, generator
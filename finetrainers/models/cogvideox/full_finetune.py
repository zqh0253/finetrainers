from diffusers import CogVideoXPipeline

from .lora import (
    calculate_noisy_latents,
    collate_fn_t2v,
    forward_pass,
    initialize_pipeline,
    load_condition_models,
    load_diffusion_models,
    load_latent_models,
    post_latent_preparation,
    prepare_conditions,
    prepare_latents,
    validation,
)


# TODO(aryan): refactor into model specs for better re-use
COGVIDEOX_T2V_FULL_FINETUNE_CONFIG = {
    "pipeline_cls": CogVideoXPipeline,
    "load_condition_models": load_condition_models,
    "load_latent_models": load_latent_models,
    "load_diffusion_models": load_diffusion_models,
    "initialize_pipeline": initialize_pipeline,
    "prepare_conditions": prepare_conditions,
    "prepare_latents": prepare_latents,
    "post_latent_preparation": post_latent_preparation,
    "collate_fn": collate_fn_t2v,
    "calculate_noisy_latents": calculate_noisy_latents,
    "forward_pass": forward_pass,
    "validation": validation,
}

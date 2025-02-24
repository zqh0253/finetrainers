from typing import Any, Dict, List, Optional, Union
import argparse
import json
import torch
import numpy as np
import os
import cv2
import tqdm
from diffusers import CogVideoXPipeline

from finetrainers.io import load_np, load_image
from finetrainers.visualizer import HtmlPageVisualizer

def _get_t5_prompt_embeds(
    tokenizer,
    text_encoder,
    prompt: Union[str, List[str]] = None,
    max_sequence_length: int = 226,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
):
    prompt = [prompt] if isinstance(prompt, str) else prompt

    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=max_sequence_length,
        truncation=True,
        add_special_tokens=True,
        return_tensors="pt",
    )
    text_input_ids = text_inputs.input_ids

    prompt_embeds = text_encoder(text_input_ids.to(device))[0]
    prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)

    return {"prompt_embeds": prompt_embeds}


def get_text_conditions(prompt, tokenizer, text_encoder, device, dtype, max_sequence_length=226):
    device = device or text_encoder.device
    dtype = dtype or text_encoder.dtype
    return _get_t5_prompt_embeds(
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        prompt=prompt,
        max_sequence_length=max_sequence_length,
        device=device,
        dtype=dtype,
    )

def get_vae_latents(model, img, device):
    img = img.to(model.vae.dtype)
    latents = model.vae.encode(img.to(model.vae.dtype)).latent_dist
    latents_params = latents.parameters
    
    return latents_params

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--json_path", type=str, required=True, help="Path to the json file")
    parser.add_argument("--model_path", type=str, default="THUDM/CogVideoX-2b", help="Path to the model")
    parser.add_argument("--output_dir", type=str, default="latents", help="Path to the output directory")
    parser.add_argument("--downsample_factor", type=int, default=1, help="Downsample factor")
    args = parser.parse_args()

    with open(args.json_path, "r") as f:
        data = json.load(f)
    
    model = CogVideoXPipeline.from_pretrained(args.model_path, torch_dtype=torch.bfloat16).to("cuda")

    new_infos = []
    with torch.no_grad():
        for scene in tqdm.tqdm(data):    
            for rgb_path, mask_path, xyz_path, orientation, description, c2w, K, scale in scene:
                rgb = np.array(load_image(rgb_path))
                mask = np.array(load_image(mask_path))
                xyz = np.clip(load_np(xyz_path), 0, 1)

                h, w, _ = rgb.shape
                new_h, new_w = h // args.downsample_factor, w // args.downsample_factor
                new_h, new_w = round(new_h / 8) * 8, round(new_w / 8) * 8
                K[0] = K[0] * new_w / w
                K[2] = K[2] * new_w / w
                K[4] = K[4] * new_h / h
                K[5] = K[5] * new_h / h

                rgb = cv2.resize(rgb, (new_w, new_h))
                mask = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST_EXACT)
                xyz = cv2.resize(xyz, (new_w, new_h), interpolation=cv2.INTER_NEAREST_EXACT)
                xyz = (xyz * 255).astype(np.uint8)

                xyz[mask<=0] = 0
                xyz = torch.from_numpy(xyz).permute(2, 0, 1) / 255 * 2 - 1
                xyz = xyz[None, :, None].cuda()
                xyz_latents = get_vae_latents(model, xyz, "cuda")
                
                rgb = torch.from_numpy(rgb).permute(2, 0, 1) / 255 * 2 - 1
                rgb = rgb[None, :, None].cuda()
                rgb_latents = get_vae_latents(model, rgb, "cuda")

                text_conditions = get_text_conditions(description, model.tokenizer, model.text_encoder, "cuda", torch.bfloat16)

                folder_name = rgb_path.replace('/', '-').replace(' ', '_')
                rgb_latents_path = os.path.join(args.output_dir, 'latents', folder_name, "rgb_latents.pt")
                xyz_latents_path = os.path.join(args.output_dir, 'latents', folder_name, "xyz_latents.pt")
                text_conditions_path = os.path.join(args.output_dir, 'latents', folder_name, "text_conditions.pt")

                os.makedirs(os.path.dirname(rgb_latents_path), exist_ok=True)
                os.makedirs(os.path.dirname(xyz_latents_path), exist_ok=True)
                os.makedirs(os.path.dirname(text_conditions_path), exist_ok=True)

                torch.save(text_conditions, text_conditions_path)
                torch.save(rgb_latents, rgb_latents_path)
                torch.save(xyz_latents, xyz_latents_path)

                infos = [
                    rgb_path, mask_path, xyz_path, orientation, description, c2w, K, scale,
                    rgb_latents_path, xyz_latents_path, text_conditions_path
                ]
                new_infos.append(infos)

    with open(os.path.join(args.output_dir, "precomputed_infos.json"), "w") as f:
        json.dump(new_infos, f)

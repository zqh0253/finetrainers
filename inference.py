import torch
import numpy as np
from einops import rearrange
from finetrainers.models.cogvideox.cogvideox_transformer_3d import CogVideoXTransformer3DModel
from finetrainers.models.cogvideox.cogvideox_pipeline_3d import CogVideoXPipeline3D
from finetrainers.models.cogvideox import COGVIDEOX_T2V_FULL_FINETUNE_CONFIG as model_config
from diffusers.utils import export_to_video
from PIL import Image

# prompt = "The image depicts a serene and picturesque scene of an old wooden church nestled in a lush, green environment. The weather appears to be clear and sunny, as indicated by the bright blue sky with minimal cloud cover."
# prompt = "a grand building in a bustling city, night with no clouds"

@torch.no_grad()
def inference(prompt, ckpt_path, model_path, json_path, data_root, sample_idx, output_path, unconditional=False, guidance_scale=6):
    transformer = CogVideoXTransformer3DModel.from_pretrained(
        ckpt_path, torch_dtype=torch.bfloat16
    )
    pipe = CogVideoXPipeline3D.from_pretrained(model_path, 
                                               transformer=transformer, torch_dtype=torch.bfloat16)
    pipe.to("cuda")

    src = torch.from_numpy(np.array(Image.open("dd.jpeg"))) / 127.5 - 1  # Convert to -1-1 range float
    src = rearrange(torch.tensor(src, device="cuda"), "h w c -> () () c h w")
    src_latent_conditions = model_config["prepare_latents"](
                            vae=pipe.vae,
                            image_or_video=src,
                            patch_size=transformer.config.patch_size,
                            patch_size_t=transformer.config.patch_size_t,
                            device="cuda")['latents'] 
    output = pipe(
        prompt, height=256, width=256,
        num_frames=21,
        src_latents=src_latent_conditions,
        num_inference_steps=50,
        guidance_scale=guidance_scale)
    video = output['frames'][0]
    export_to_video(video, "output.mp4", fps=4)
    # dataset = MegascenesDataset(data_root, json_path)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, default="")
    parser.add_argument("--ckpt_path", type=str, default="debug_ckpt/checkpoint-10000/transformer")
    parser.add_argument("--model_path", type=str, default="THUDM/CogVideoX-2b")
    parser.add_argument("--json_path", type=str, default="/users/qihang/data/megascenes_all_label/precomputed/merge_new.json")  
    parser.add_argument("--data_root", type=str, default="/users/qihang/data/")
    parser.add_argument("--sample_idx", type=int, default=0)
    parser.add_argument("--guidance_scale", type=float, default=6)
    parser.add_argument("--output_path", type=str, default="vis.png")
    parser.add_argument("--unconditional", action="store_true")
    args = parser.parse_args()
    inference(args.prompt, args.ckpt_path, args.model_path, args.json_path, args.data_root, args.sample_idx, args.output_path, args.unconditional, args.guidance_scale)
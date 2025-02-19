import torch
from finetrainers.models.cogvideox.cogvideox_transformer_3d import CogVideoXTransformer3DModel
from finetrainers.models.cogvideox.cogvideox_pipeline import CogVideoXPipeline_vdm
from diffusers.utils import export_to_video

transformer = CogVideoXTransformer3DModel.from_pretrained(
    "debug_ckpt_rgbrgb/checkpoint-15900/transformer", torch_dtype=torch.bfloat16
)
pipe = CogVideoXPipeline_vdm.from_pretrained("THUDM/CogVideoX-2b", transformer=transformer, torch_dtype=torch.bfloat16)
# pipe = CogVideoXPipeline_vdm.from_pretrained("THUDM/CogVideoX-2b", torch_dtype=torch.bfloat16)
pipe.to("cuda")

# pipe.load_lora_weights("my-awesome-name/my-awesome-lora", adapter_name="cogvideox-lora")
# pipe.set_adapters(["cogvideox-lora"], [0.75])

prompt = "The image depicts a serene and picturesque scene of an old wooden church nestled in a lush, green environment. The weather appears to be clear and sunny, as indicated by the bright blue sky with minimal cloud cover."
video = pipe(prompt, num_frames=9, height=256, width=256, num_inference_steps=50).frames[0]
export_to_video(video, "output.mp4")

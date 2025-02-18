import torch
from finetrainers.models.cogvideox.cogvideox_transformer_3d import CogVideoXTransformer3DModel
from finetrainers.models.cogvideox.cogvideox_pipeline import CogVideoXPipeline_vdm
from diffusers.utils import export_to_video

pipe = CogVideoXPipeline_vdm.from_pretrained(
            "THUDM/CogVideoX-2b", torch_dtype=torch.bfloat16
            ).to("cuda")
pipe.load_lora_weights("debug_ckpt/checkpoint-400", adapter_name="cogvideox-lora")
pipe.set_adapters(["cogvideox-lora"], [0.75])

prompt = "The image showcases a sunny day with blue skies and some clouds. The lighting is bright, casting shadows on the white walls of buildings surrounding an old stone tower with bells in its belfry. A palm tree stands tall beside it, while various cars are parked nearby including a van and several smaller vehicles."
video = pipe(prompt, num_frames=9).frames[0]
export_to_video(video, "output.mp4")

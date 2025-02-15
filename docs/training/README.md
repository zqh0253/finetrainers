# FineTrainers training documentation

This directory contains the training-related specifications for all the models we support in `finetrainers`. Each model page has:
- an example training command
- inference example
- numbers on memory consumption

By default, we don't include any validation-related arguments in the example training commands. To enable validation inference, one can pass:

```diff
+ --validation_prompts "$ID_TOKEN A black and white animated scene unfolds with an anthropomorphic goat surrounded by musical notes and symbols, suggesting a playful environment. Mickey Mouse appears, leaning forward in curiosity as the goat remains still. The goat then engages with Mickey, who bends down to converse or react. The dynamics shift as Mickey grabs the goat, potentially in surprise or playfulness, amidst a minimalistic background. The scene captures the evolving relationship between the two characters in a whimsical, animated setting, emphasizing their interactions and emotions.@@@49x512x768:::$ID_TOKEN A woman with long brown hair and light skin smiles at another woman with long blonde hair. The woman with brown hair wears a black jacket and has a small, barely noticeable mole on her right cheek. The camera angle is a close-up, focused on the woman with brown hair's face. The lighting is warm and natural, likely from the setting sun, casting a soft glow on the scene. The scene appears to be real-life footage@@@49x512x768" \
+ --num_validation_videos 1 \
+ --validation_steps 100
```

Supported models:
- [CogVideoX](./cogvideox.md)
- [LTX-Video](./ltx_video.md)
- [HunyuanVideo](./hunyuan_video.md)

Supported training types:
- LoRA (`--training_type lora`)
- Full finetuning (`--training_type full-finetune`)

Arguments for training are well-documented in the code. For more information, please run `python train.py --help`.

## How do we handle `mixed_precision`?

The accelerate config files (the ones seen [here](../../accelerate_configs/)) that are being supplied while launching training should contain a field called `mixed_precision` and `accelerate` makes use of that if specified. We don't let users explicitly pass that from the CLI args because it can be confusing to have `transformer_dtype` and `mixed_precision` in the codebase.

`transformer_dtype` is the ultimate source of truth for the precision to be used when training. It will also most likely always have to be `torch.bfloat16` because:

* All models currently supported (except Cog-2b) do not work well in FP16 for inference, so training would be broken as well. This can be revisited if it makes sense to train in FP16 for other models added.
* The `accelerate` config files default to using "bf16", but modifying that would be at the risk of user and assumes they understand the significance of their changes.

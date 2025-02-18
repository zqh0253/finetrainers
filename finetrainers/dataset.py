import json
import os
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torchvision.transforms as TT
import torchvision.transforms.functional as TTF
from accelerate.logging import get_logger
from torch.utils.data import Dataset, Sampler
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from torchvision.transforms.functional import resize


# Must import after torch because this can sometimes lead to a nasty segmentation fault, or stack smashing error
# Very few bug reports but it happens. Look in decord Github issues for more relevant information.
import decord  # isort:skip

decord.bridge.set_bridge("torch")

from .constants import (  # noqa
    COMMON_LLM_START_PHRASES,
    PRECOMPUTED_CONDITIONS_DIR_NAME,
    PRECOMPUTED_DIR_NAME,
    PRECOMPUTED_LATENTS_DIR_NAME,
    PRECOMPUTED_XYZ_LATENTS_DIR_NAME,
)


logger = get_logger(__name__)


# TODO(aryan): This needs a refactor with separation of concerns.
# Images should be handled separately. Videos should be handled separately.
# Loading should be handled separately.
# Preprocessing (aspect ratio, resizing) should be handled separately.
# URL loading should be handled.
# Parquet format should be handled.
# Loading from ZIP should be handled.
class ImageOrVideoDataset(Dataset):
    def __init__(
        self,
        data_root: str,
        caption_column: str,
        video_column: str,
        resolution_buckets: List[Tuple[int, int, int]],
        dataset_file: Optional[str] = None,
        id_token: Optional[str] = None,
        remove_llm_prefixes: bool = False,
    ) -> None:
        super().__init__()

        self.data_root = Path(data_root)
        self.dataset_file = dataset_file
        self.caption_column = caption_column
        self.video_column = video_column
        self.id_token = f"{id_token.strip()} " if id_token else ""
        self.resolution_buckets = resolution_buckets

        # Four methods of loading data are supported.
        #   - Using a CSV: caption_column and video_column must be some column in the CSV. One could
        #     make use of other columns too, such as a motion score or aesthetic score, by modifying the
        #     logic in CSV processing.
        #   - Using two files containing line-separate captions and relative paths to videos.
        #   - Using a JSON file containing a list of dictionaries, where each dictionary has a `caption_column` and `video_column` key.
        #   - Using a JSONL file containing a list of line-separated dictionaries, where each dictionary has a `caption_column` and `video_column` key.
        # For a more detailed explanation about preparing dataset format, checkout the README.
        if dataset_file is None:
            (
                self.prompts,
                self.video_paths,
            ) = self._load_dataset_from_local_path()
        elif dataset_file.endswith(".csv"):
            (
                self.prompts,
                self.video_paths,
            ) = self._load_dataset_from_csv()
        elif dataset_file.endswith(".json"):
            (
                self.prompts,
                self.video_paths,
            ) = self._load_dataset_from_json()
        elif dataset_file.endswith(".jsonl"):
            (
                self.prompts,
                self.video_paths,
            ) = self._load_dataset_from_jsonl()
        else:
            raise ValueError(
                "Expected `--dataset_file` to be a path to a CSV file or a directory containing line-separated text prompts and video paths."
            )

        if len(self.video_paths) != len(self.prompts):
            raise ValueError(
                f"Expected length of prompts and videos to be the same but found {len(self.prompts)=} and {len(self.video_paths)=}. Please ensure that the number of caption prompts and videos match in your dataset."
            )

        # Clean LLM start phrases
        if remove_llm_prefixes:
            for i in range(len(self.prompts)):
                self.prompts[i] = self.prompts[i].strip()
                for phrase in COMMON_LLM_START_PHRASES:
                    if self.prompts[i].startswith(phrase):
                        self.prompts[i] = self.prompts[i].removeprefix(phrase).strip()

        self.video_transforms = transforms.Compose(
            [
                transforms.Lambda(self.scale_transform),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
            ]
        )

    @staticmethod
    def scale_transform(x):
        return x / 255.0

    def __len__(self) -> int:
        return len(self.video_paths)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        if isinstance(index, list):
            # Here, index is actually a list of data objects that we need to return.
            # The BucketSampler should ideally return indices. But, in the sampler, we'd like
            # to have information about num_frames, height and width. Since this is not stored
            # as metadata, we need to read the video to get this information. You could read this
            # information without loading the full video in memory, but we do it anyway. In order
            # to not load the video twice (once to get the metadata, and once to return the loaded video
            # based on sampled indices), we cache it in the BucketSampler. When the sampler is
            # to yield, we yield the cache data instead of indices. So, this special check ensures
            # that data is not loaded a second time. PRs are welcome for improvements.
            return index

        prompt = self.id_token + self.prompts[index]

        video_path: Path = self.video_paths[index]
        if video_path.suffix.lower() in [".png", ".jpg", ".jpeg"]:
            video = self._preprocess_image(video_path)
        else:
            video = self._preprocess_video(video_path)

        return {
            "prompt": prompt,
            "video": video,
            "video_metadata": {
                "num_frames": video.shape[0],
                "height": video.shape[2],
                "width": video.shape[3],
            },
        }

    def _load_dataset_from_local_path(self) -> Tuple[List[str], List[str]]:
        if not self.data_root.exists():
            raise ValueError("Root folder for videos does not exist")

        prompt_path = self.data_root.joinpath(self.caption_column)
        video_path = self.data_root.joinpath(self.video_column)

        if not prompt_path.exists() or not prompt_path.is_file():
            raise ValueError(
                "Expected `--caption_column` to be path to a file in `--data_root` containing line-separated text prompts."
            )
        if not video_path.exists() or not video_path.is_file():
            raise ValueError(
                "Expected `--video_column` to be path to a file in `--data_root` containing line-separated paths to video data in the same directory."
            )

        with open(prompt_path, "r", encoding="utf-8") as file:
            prompts = [line.strip() for line in file.readlines() if len(line.strip()) > 0]
        with open(video_path, "r", encoding="utf-8") as file:
            video_paths = [self.data_root.joinpath(line.strip()) for line in file.readlines() if len(line.strip()) > 0]

        if any(not path.is_file() for path in video_paths):
            raise ValueError(
                f"Expected `{self.video_column=}` to be a path to a file in `{self.data_root=}` containing line-separated paths to video data but found atleast one path that is not a valid file."
            )

        return prompts, video_paths

    def _load_dataset_from_csv(self) -> Tuple[List[str], List[str]]:
        df = pd.read_csv(self.dataset_file)
        prompts = df[self.caption_column].tolist()
        video_paths = df[self.video_column].tolist()
        video_paths = [self.data_root.joinpath(line.strip()) for line in video_paths]

        if any(not path.is_file() for path in video_paths):
            raise ValueError(
                f"Expected `{self.video_column=}` to be a path to a file in `{self.data_root=}` containing line-separated paths to video data but found atleast one path that is not a valid file."
            )

        return prompts, video_paths

    def _load_dataset_from_json(self) -> Tuple[List[str], List[str]]:
        with open(self.dataset_file, "r", encoding="utf-8") as file:
            data = json.load(file)

        prompts = [entry[self.caption_column] for entry in data]
        video_paths = [self.data_root.joinpath(entry[self.video_column].strip()) for entry in data]

        if any(not path.is_file() for path in video_paths):
            raise ValueError(
                f"Expected `{self.video_column=}` to be a path to a file in `{self.data_root=}` containing line-separated paths to video data but found atleast one path that is not a valid file."
            )

        return prompts, video_paths

    def _load_dataset_from_jsonl(self) -> Tuple[List[str], List[str]]:
        with open(self.dataset_file, "r", encoding="utf-8") as file:
            data = [json.loads(line) for line in file]

        prompts = [entry[self.caption_column] for entry in data]
        video_paths = [self.data_root.joinpath(entry[self.video_column].strip()) for entry in data]

        if any(not path.is_file() for path in video_paths):
            raise ValueError(
                f"Expected `{self.video_column=}` to be a path to a file in `{self.data_root=}` containing line-separated paths to video data but found atleast one path that is not a valid file."
            )

        return prompts, video_paths

    def _preprocess_image(self, path: Path) -> torch.Tensor:
        # TODO(aryan): Support alpha channel in future by whitening background
        image = TTF.Image.open(path.as_posix()).convert("RGB")
        image = TTF.to_tensor(image)
        image = image * 2.0 - 1.0
        image = image.unsqueeze(0).contiguous()  # [C, H, W] -> [1, C, H, W] (1-frame video)
        return image

    def _preprocess_video(self, path: Path) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        r"""
        Loads a single video, or latent and prompt embedding, based on initialization parameters.

        Returns a [F, C, H, W] video tensor.
        """
        video_reader = decord.VideoReader(uri=path.as_posix())
        video_num_frames = len(video_reader)

        indices = list(range(0, video_num_frames, video_num_frames // self.max_num_frames))
        frames = video_reader.get_batch(indices)
        frames = frames[: self.max_num_frames].float()
        frames = frames.permute(0, 3, 1, 2).contiguous()
        frames = torch.stack([self.video_transforms(frame) for frame in frames], dim=0)
        return frames


class ImageOrVideoDatasetWithResizing(ImageOrVideoDataset):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.max_num_frames = max(self.resolution_buckets, key=lambda x: x[0])[0]

    def _preprocess_image(self, path: Path) -> torch.Tensor:
        # TODO(aryan): Support alpha channel in future by whitening background
        image = TTF.Image.open(path.as_posix()).convert("RGB")
        image = TTF.to_tensor(image)

        nearest_res = self._find_nearest_resolution(image.shape[1], image.shape[2])
        image = resize(image, nearest_res)

        image = image * 2.0 - 1.0
        image = image.unsqueeze(0).contiguous()
        return image

    def _preprocess_video(self, path: Path) -> torch.Tensor:
        video_reader = decord.VideoReader(uri=path.as_posix())
        video_num_frames = len(video_reader)
        nearest_frame_bucket = min(
            [bucket for bucket in self.resolution_buckets if bucket[0] <= video_num_frames],
            key=lambda x: abs(x[0] - min(video_num_frames, self.max_num_frames)),
            default=1,
        )[0]

        frame_indices = list(range(0, video_num_frames, video_num_frames // nearest_frame_bucket))

        frames = video_reader.get_batch(frame_indices)
        frames = frames[:nearest_frame_bucket].float()
        frames = frames.permute(0, 3, 1, 2).contiguous()

        nearest_res = self._find_nearest_resolution(frames.shape[2], frames.shape[3])
        frames_resized = torch.stack([resize(frame, nearest_res) for frame in frames], dim=0)
        frames = torch.stack([self.video_transforms(frame) for frame in frames_resized], dim=0)

        return frames

    def _find_nearest_resolution(self, height, width):
        nearest_res = min(self.resolution_buckets, key=lambda x: abs(x[1] - height) + abs(x[2] - width))
        return nearest_res[1], nearest_res[2]


class ImageOrVideoDatasetWithResizeAndRectangleCrop(ImageOrVideoDataset):
    def __init__(self, video_reshape_mode: str = "center", *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.video_reshape_mode = video_reshape_mode
        self.max_num_frames = max(self.resolution_buckets, key=lambda x: x[0])[0]

    def _resize_for_rectangle_crop(self, arr, image_size):
        reshape_mode = self.video_reshape_mode
        if arr.shape[3] / arr.shape[2] > image_size[1] / image_size[0]:
            arr = resize(
                arr,
                size=[image_size[0], int(arr.shape[3] * image_size[0] / arr.shape[2])],
                interpolation=InterpolationMode.BICUBIC,
            )
        else:
            arr = resize(
                arr,
                size=[int(arr.shape[2] * image_size[1] / arr.shape[3]), image_size[1]],
                interpolation=InterpolationMode.BICUBIC,
            )

        h, w = arr.shape[2], arr.shape[3]
        arr = arr.squeeze(0)

        delta_h = h - image_size[0]
        delta_w = w - image_size[1]

        if reshape_mode == "random" or reshape_mode == "none":
            top = np.random.randint(0, delta_h + 1)
            left = np.random.randint(0, delta_w + 1)
        elif reshape_mode == "center":
            top, left = delta_h // 2, delta_w // 2
        else:
            raise NotImplementedError
        arr = TT.functional.crop(arr, top=top, left=left, height=image_size[0], width=image_size[1])
        return arr

    def _preprocess_video(self, path: Path) -> torch.Tensor:
        video_reader = decord.VideoReader(uri=path.as_posix())
        video_num_frames = len(video_reader)
        nearest_frame_bucket = min(
            [bucket for bucket in self.resolution_buckets if bucket <= video_num_frames],
            key=lambda x: abs(x[0] - min(video_num_frames, self.max_num_frames)),
            default=1,
        )[0]

        frame_indices = list(range(0, video_num_frames, video_num_frames // nearest_frame_bucket))

        frames = video_reader.get_batch(frame_indices)
        frames = frames[:nearest_frame_bucket].float()
        frames = frames.permute(0, 3, 1, 2).contiguous()

        nearest_res = self._find_nearest_resolution(frames.shape[2], frames.shape[3])
        frames_resized = self._resize_for_rectangle_crop(frames, nearest_res)
        frames = torch.stack([self.video_transforms(frame) for frame in frames_resized], dim=0)
        return frames

    def _find_nearest_resolution(self, height, width):
        nearest_res = min(self.resolutions, key=lambda x: abs(x[1] - height) + abs(x[2] - width))
        return nearest_res[1], nearest_res[2]


class PrecomputedDataset(Dataset):
    def __init__(self, data_root: str, model_name: str = None, cleaned_model_id: str = None) -> None:
        super().__init__()

        self.data_root = Path('precompute')

        if model_name and cleaned_model_id:
            precomputation_dir = self.data_root / f"{model_name}_{cleaned_model_id}_{PRECOMPUTED_DIR_NAME}"
            self.latents_path = precomputation_dir / PRECOMPUTED_LATENTS_DIR_NAME
            self.conditions_path = precomputation_dir / PRECOMPUTED_CONDITIONS_DIR_NAME
            self.xyz_latents_path = precomputation_dir / PRECOMPUTED_XYZ_LATENTS_DIR_NAME
        else:
            self.latents_path = self.data_root / PRECOMPUTED_DIR_NAME / PRECOMPUTED_LATENTS_DIR_NAME
            self.conditions_path = self.data_root / PRECOMPUTED_DIR_NAME / PRECOMPUTED_CONDITIONS_DIR_NAME
            self.xyz_latents_path = self.data_root / PRECOMPUTED_DIR_NAME / PRECOMPUTED_XYZ_LATENTS_DIR_NAME
        self.latent_conditions = sorted(os.listdir(self.latents_path))
        self.text_conditions = sorted(os.listdir(self.conditions_path))
        self.xyz_latent_conditions = sorted(os.listdir(self.xyz_latents_path))
        assert len(self.latent_conditions) == len(self.text_conditions), "Number of captions and videos do not match"

    def __len__(self) -> int:
        return len(self.latent_conditions)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        conditions = {}
        latent_path = self.latents_path / self.latent_conditions[index]
        condition_path = self.conditions_path / self.text_conditions[index]
        xyz_latent_path = self.xyz_latents_path / self.xyz_latent_conditions[index]
        conditions["latent_conditions"] = torch.load(latent_path, map_location="cpu", weights_only=True)
        conditions["text_conditions"] = torch.load(condition_path, map_location="cpu", weights_only=True)
        conditions["xyz_latent_conditions"] = torch.load(xyz_latent_path, map_location="cpu", weights_only=True)
        return conditions


class BucketSampler(Sampler):
    r"""
    PyTorch Sampler that groups 3D data by height, width and frames.

    Args:
        data_source (`ImageOrVideoDataset`):
            A PyTorch dataset object that is an instance of `ImageOrVideoDataset`.
        batch_size (`int`, defaults to `8`):
            The batch size to use for training.
        shuffle (`bool`, defaults to `True`):
            Whether or not to shuffle the data in each batch before dispatching to dataloader.
        drop_last (`bool`, defaults to `False`):
            Whether or not to drop incomplete buckets of data after completely iterating over all data
            in the dataset. If set to True, only batches that have `batch_size` number of entries will
            be yielded. If set to False, it is guaranteed that all data in the dataset will be processed
            and batches that do not have `batch_size` number of entries will also be yielded.
    """

    def __init__(
        self, data_source: ImageOrVideoDataset, batch_size: int = 8, shuffle: bool = True, drop_last: bool = False
    ) -> None:
        self.data_source = data_source
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last

        self.buckets = {resolution: [] for resolution in data_source.resolution_buckets}

        self._raised_warning_for_drop_last = False

    def __len__(self):
        if self.drop_last and not self._raised_warning_for_drop_last:
            self._raised_warning_for_drop_last = True
            logger.warning(
                "Calculating the length for bucket sampler is not possible when `drop_last` is set to True. This may cause problems when setting the number of epochs used for training."
            )
        return (len(self.data_source) + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        for index, data in enumerate(self.data_source):
            video_metadata = data["video_metadata"]
            f, h, w = video_metadata["num_frames"], video_metadata["height"], video_metadata["width"]

            self.buckets[(f, h, w)].append(data)
            if len(self.buckets[(f, h, w)]) == self.batch_size:
                if self.shuffle:
                    random.shuffle(self.buckets[(f, h, w)])
                yield self.buckets[(f, h, w)]
                del self.buckets[(f, h, w)]
                self.buckets[(f, h, w)] = []

        if self.drop_last:
            return

        for fhw, bucket in list(self.buckets.items()):
            if len(bucket) == 0:
                continue
            if self.shuffle:
                random.shuffle(bucket)
                yield bucket
                del self.buckets[fhw]
                self.buckets[fhw] = []

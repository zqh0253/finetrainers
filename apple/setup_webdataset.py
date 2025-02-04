import webdataset as wds
import gc, os
from apple_fsspec import open as apple_open
from torch.utils.data import default_collate
from torchvision import transforms

os.environ["APPLE_FSSPEC__CACHE__PARALLELISM"] = "4"  # noqa
os.environ["APPLE_FSSPEC__S3LIKE_CACHE_TYPE"] = "fsspec"  # noqa

def url_opener(data, handler=wds.handlers.reraise_exception, **kw):
    """
    https://github.pie.apple.com/foundation-models/ajax/blob/35cbd3327f798ebfe0dc74072fa9c4b8215791c7/ajax/experiments/polymath/common/filters/wds_fixes.py#L27
    """
    for sample in data:
        url = sample["url"]
        try:
            with apple_open(url, mode="rb") as stream:
                sample["stream"] = stream
                yield sample
        except Exception as exn:
            exn.args = exn.args + (url,)
            if handler(exn):
                continue
            else:
                break
        gc.collect()


def tarfile_samples(
    src,
    handler,
    select_files,
    rename_files,
):
    streams = url_opener(src, handler=handler)
    files = wds.tariterators.tar_file_expander(
        streams, handler=handler, select_files=select_files, rename_files=rename_files
    )
    samples = wds.tariterators.group_by_keys(files, handler=handler)
    return samples


wds.tariterators.url_opener = url_opener
wds.tariterators.tarfile_to_samples = wds.filters.pipelinefilter(tarfile_samples)
print('setup webdataset with apple_fsspec')

if __name__ == "__main__":
    # Create a ResampledShards dataset
    train_processing_pipeline = [
        wds.decode(wds.autodecode.ImageHandler("pil", extensions=["webp", "png", "jpg", "jpeg"])),
        wds.rename(
            image="jpg;png;jpeg;webp",
            class_id="cls",
            handler=wds.warn_and_continue,
            ),
        wds.map_dict(
            image=transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(256),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]),
            class_id=lambda x: int(x),
            handler=wds.warn_and_continue,
        ),
    ]

    # Define the list of shard URLs
    shard_urls = "conductor://mlx/datasets/imagenet_wds/imagenet-train-{000000..000320}.tar"
    pipeline = [wds.ResampledShards(shard_urls),
                wds.tarfile_to_samples(handler=wds.warn_and_continue),
                wds.shuffle(bufsize=5000,
                            initial=1000),
                *train_processing_pipeline,
                wds.batched(8, partial=False, collation_fn=default_collate),]

    # Optionally, apply additional transformations
    dataset = wds.DataPipeline(*pipeline)

    # Iterate through the dataset
    for batch in dataset:
        print(batch['class_id'])
        from torchvision.utils import save_image
        save_image(batch['image'], 'debug.png', normalize=True)
        break
import boto3.session
from boto3.s3.transfer import TransferConfig
from concurrent.futures import ProcessPoolExecutor, as_completed, wait
from pathlib import Path
import logging
import os
import re


def download_object(
    bucket_name,
    file_name,
    download_path=None,
    endpoint_url='https://blob.mr3.simcloud.apple.com',
    max_bandwidth=None
):
    """Downloads an object from S3 to local."""
    session = boto3.session.Session()
    s3_client = session.client(service_name='s3', endpoint_url=endpoint_url)
    if download_path is None:
        download_path = os.path.basename(file_name)
    Path(download_path).parent.mkdir(parents=True, exist_ok=True)
    try:
        # command = f"aws --endpoint-url {endpoint_url} --cli-read-timeout 300 s3 cp "
        # command += f"s3://{bucket_name}/{file_name} {download_path}"
        # os.system(command)
        s3_client.download_file(
            bucket_name,
            file_name,
            download_path,
            Config=TransferConfig(num_download_attempts=10, max_bandwidth=max_bandwidth
        ))
    except Exception as e:
        logging.error(f'Error thrown while downloading file: {file_name}. {e}')
    return download_path


def download_object_from_full_path(
    path,
    download_path=None,
    endpoint_url='https://blob.mr3.simcloud.apple.com'
):
    bucket_name, parent_path, basename = _parse_path(path)
    file_name = os.path.join(parent_path, basename)
    return download_object(bucket_name, file_name,
                           download_path=download_path,
                           endpoint_url=endpoint_url)


def upload_object(
    bucket_name, file_name, upload_path,
    endpoint_url='https://blob.mr3.simcloud.apple.com'
):
    """Uload an object from S3 to local."""

    session = boto3.session.Session()
    s3_client = session.client(service_name='s3', endpoint_url=endpoint_url)
    s3_client.upload_file(file_name, bucket_name, upload_path)
    return "Success"


def _parse_path(tsv_pattern):
    # for now, expected path is s3://mlx/datasets/{datset_name}/rest
    parts = tsv_pattern.split('/')
    assert parts[0] == 's3:'
    assert parts[1] == ''
    # assert parts[3] == 'datasets' # for now, since everything is in mlx/datasets
    bucket = parts[2]
    pattern = parts[-1]
    return bucket, '/'.join(parts[3:-1]), pattern


def get_file_list(
    tsv_pattern,
    endpoint_url='https://blob.mr3.simcloud.apple.com'
):
    bucket_name, parent_path, pattern = _parse_path(tsv_pattern)
    resource = boto3.resource('s3', endpoint_url=endpoint_url)
    bucket = resource.Bucket(bucket_name)
    fnames = []
    pattern = re.compile(pattern)
    for obj in bucket.objects.filter(Prefix=parent_path + '/'):
        fname = obj.key
        if pattern.search(fname):
            fnames.append(f's3://{bucket_name}/{fname}')

    return fnames


def download_parallel(files, endpoint_url='https://blob.mr3.simcloud.apple.com'):
    logging.info('Doing parallel download')
    with ProcessPoolExecutor() as executor:
        logging.info(f'Submitting {files}')
        future_to_key = {
                executor.submit(download_object_from_full_path, key,
                    f'{index}.tsv', endpoint_url=endpoint_url): key
                for index, key in enumerate(files)
                }
        for future in as_completed(future_to_key):
            key = future_to_key[future]
            exception = future.exception()
            if not exception:
                yield key, future.result()
            else:
                yield key, exception


def s3_download_file(bucket, k, local):
    session = boto3.Session(profile_name="conductor-notary")
    conductor_endpoint = "https://conductor.data.apple.com"
    client = session.client('s3', endpoint_url=conductor_endpoint)
    if not os.path.exists(local):
        client.download_file(bucket, k, local)
    

def s3_download_dir(prefix, local, bucket):
    """
    params:
    - prefix: pattern to match in s3
    - local: local path to folder in which to place files
    - bucket: s3 bucket with target contents
    - client: initialized s3 client object
    """
    session = boto3.Session(profile_name="conductor-notary")
    conductor_endpoint = "https://conductor.data.apple.com"
    client = session.client('s3', endpoint_url=conductor_endpoint)
    
    keys = []
    dirs = []
    next_token = ''
    base_kwargs = {
        'Bucket':bucket,
        'Prefix':prefix,
    }
    while next_token is not None:
        kwargs = base_kwargs.copy()
        if next_token != '':
            kwargs.update({'ContinuationToken': next_token})
        results = client.list_objects_v2(**kwargs)
        contents = results.get('Contents')
        for i in contents:
            k = i.get('Key')
            if k[-1] != '/':
                keys.append(k)
            else:
                dirs.append(k)
        next_token = results.get('NextContinuationToken')

    for d in dirs:
        dest_pathname = os.path.join(local, d)
        if not os.path.exists(os.path.dirname(dest_pathname)):
            os.makedirs(os.path.dirname(dest_pathname))
            
    with ProcessPoolExecutor() as executor:
        futures = []
        for k in keys:
            kk = k[len(prefix)+1:]
            dest_pathname = os.path.join(local, kk)
            if not os.path.exists(os.path.dirname(dest_pathname)):
                os.makedirs(os.path.dirname(dest_pathname))
            print(bucket, k, dest_pathname)
            futures.append(
                executor.submit(
                    s3_download_file, bucket, k, dest_pathname
                )
            )
        for future in as_completed(futures):
            future.result()
    

def snapshot_download(remote, local_dir):
    try:
        s3_download_dir(f'hf_snapshot/{remote}', local_dir, 'jiatao-datasets')
    except Exception as e:
        print(e)
        from huggingface_hub import snapshot_download
        snapshot_download(remote, local_dir=local_dir)  # download from huggingface
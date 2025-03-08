DATA=${1:-imagenet.zip}

mkdir -p ../datasets
pushd ../datasets
aws --endpoint-url https://conductor.data.apple.com s3 cp s3://szhai/datasets/$DATA .
unzip -q $DATA
popd
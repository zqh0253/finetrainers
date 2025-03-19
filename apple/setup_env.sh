apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
dpkg -i cuda-keyring_1.1-1_all.deb
apt-get -y install cuda-toolkit-12-4
rm -rf cuda-keyring_1.1-1_all.deb
pip install --upgrade turibolt iris-ml-ctl apple_fsspec --index https://pypi.apple.com/simple
pip install -r requirements.txt
pip install deepspeed
bash apple/conductor.sh

apt-get update && apt-get install ffmpeg libsm6 libxext6  -y
pip install --upgrade turibolt iris-ml-ctl apple_fsspec --index https://pypi.apple.com/simple
pip install -r requirements.txt
bash apple/conductor.sh

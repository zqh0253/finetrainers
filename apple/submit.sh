if [ "$2" == "i" ]; then
    bolt task submit --tar . --config $1 --interactive --exclude cifar* *.pth *.tar.gz .ipynb_checkpoints __pycache__
else
    bolt task submit --tar . --config $1 --exclude cifar* *.pth *.tar.gz .ipynb_checkpoints __pycache__
fi
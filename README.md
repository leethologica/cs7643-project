CS7643 Final Project

Authors:
* Michael Lee
* Lucas Sheldon
* Te Sun
* Marcus Tan

Data sources:
* **COCO**: https://cocodataset.org/#download
* **COCOFake**: https://github.com/aimagelab/COCOFake


Notes for Lucas:
`pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu124`

salloc -N1 -t0:15:00 --cpus-per-task 8 --ntasks-per-node=1 --gres=gpu:V100:1 --mem-per-gpu=32G
make sure to set this so that I don't kill my home dir
hub.set_dir(<PATH_TO_HUB_DIR>)
# Real-time-Recon-Tool
Real-time 3D reconstruction tool based on DROID SLAM and CDS MVSNet.

## Install
```shell
conda env create -f recon.yaml
pip install evo --upgrade --no-binary evo
pip install gdown
```

* pretrained checkpoints
[DROID-SLAM-pth](https://drive.google.com/file/d/1RvKCXebZ3gQw67eoOgiDzeoEhPAKBu4c/view?usp=sharing),
[CDS-MVSNet-pth](https://drive.google.com/file/d/1gKwyW8NnGQV7Xu5-EkRBvnDJGq4y83yg/view?usp=sharing)
### Here some tips for successful compilation.

1. Soft link of Eigen
```shell
sudo ln -s /usr/include/eigen3/Eigen /usr/include/Eigen
```
2. CUDA setting
```shell
sudo gedit ~/.bashrc
export CUDA_HOME=/usr/local/cuda-11.3
source ~/.bashrc
```
3. pillow version degrade
```shell
pip install pillow==6.2.1
```
4. OpenCV version
```shell
conda remove opencv
conda install -c menpo opencv
pip install opencv-contrib-python
```
5.Requirement
* Make sure 11G memory in your GPU. Example is based on Nvidia RTX 3090.
```shell
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113
```

## Compile extensions
```shell
python setup.py install
```

## Demo
```shell
python demo.py --imagedir=data/xxx(imgs) --calib=calib/xxx.txt(intrinsic) --stride=2 --buffer 384 --mvsnet_ckpt cds_mvsnet.pth
```
[demo_0](https://www.youtube.com/watch?v=TY_y1AmOvuA),
[demo_1](https://www.youtube.com/watch?v=31lwz_Hn4Us)

# All-In-One Image Restoration for Unknown Corruption (AirNet) ![](https://visitor-badge.glitch.me/badge?page_id=XLearning-SCU.2022-CVPR-AirNet)

PyTorch implementation for All-In-One Image Restoration for Unknown Corruption (AirNet) (CVPR 2022). [[paper](http://pengxi.me/wp-content/uploads/2022/03/All-In-One-Image-Restoration-for-Unknown-Corruption.pdf)]

## Dependencies

* Python == 3.8.11
* Pytorch == 1.7.0 
* mmcv-full == 1.3.11 

We also export our conda virtual environment as airnet.yaml. You can use the following command to create the environment.

```bash
conda env create -f airnet.yaml
```

## Dataset

You could find the dataset we used in the paper at following:

Denoising: [BSD400](https://drive.google.com/file/d/1idKFDkAHJGAFDn1OyXZxsTbOSBx9GS8N/view?usp=sharing), [WED]([https://ece.uwaterloo.ca/~k29ma/exploration](http://ivc.uwaterloo.ca/database/WaterlooExploration/exploration_database_and_code.rar)), [Urban100](https://drive.google.com/drive/folders/1B3DJGQKB6eNdwuQIhdskA64qUuVKLZ9u)

Deraining: [Train100L&Rain100L](https://drive.google.com/drive/folders/1-_Tw-LHJF4vh8fpogKgZx1EQ9MhsJI_f?usp=sharing)

Dehazing: [RESIDE](https://sites.google.com/view/reside-dehaze-datasets/reside-v0) (OTS)

## Demo

You could download the pre-trained model from [Google Drive](https://drive.google.com/drive/folders/1DS_iJsP5Epzz78fZRz8lEINcnhBF6Uws?usp=sharing) and [Baidu Netdisk](https://pan.baidu.com/s/1usrpGA8FIyj1ogsZQE_Emg) (password: cr7d). Remember to put the pre-trained model into ckpt/

If you only need the visual results, you could put the test images into test/demo/ and use the following command to restore the test image:

```bash
python demo.py --mode 3
```

where mode == 3 means we use the checkpoint trained on all-in-one setting. (0 for denoising, 1 for deraining and 2 for dehazing)

## Training

If you want to re-train our model, you need to first put the training set into the data/, and use the following command:

```bash
python train.py
```

ps. To train with different combinations of corruptions, you could modify the "de_type" in option.py.

## Testing

If you want to test our model and get the psnr and ssim, you need to put the testing set into the test/, where several examples are given. Then, you could use the following command:

```bash
python test.py --mode 3
```

where mode == 3 means we use the checkpoint trained on all-in-one setting. (0 for denoising, 1 for deraining and 2 for dehazing)

## Citation

If you find AirNet useful in your research, please consider citing:

```
@inproceedings{AirNet,
author = {Li, Boyun and Liu, Xiao and Hu, Peng and Wu, Zhongqin and Lv, Jiancheng and Peng, Xi},
title = {{All-In-One Image Restoration for Unknown Corruption}},
booktitle = {IEEE Conference on Computer Vision and Pattern Recognition},
year = {2022},
address = {New Orleans, LA},
month = jun
}
```

## Acknowledgement

This repo is built upon the framework of [DASR](https://github.com/The-Learning-And-Vision-Atelier-LAVA/DASR), and we borrow some code from [mmcv](https://github.com/open-mmlab/mmcv), thanks for their excellent work!


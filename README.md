# Semantic segmentation using generative knowledge distillation for crack detection


This repo is the pytorch implementation of the following paper:

The detailed paper will be updated when the publication is completed.
The most highlight of this proposed method is to improve m-IoU and F1 without any additional crack image. The generated knowledge plays a role of a new crack image. This knowledge is transferred to crack detection newtork, then the accuracy is increased regrardless of newtork architecture.

![](figs/semi_overview.jpg)

This codes are inspired by Jun-Yan Zhu*, Taesung Park*, Phillip Isola, and Alexei A. Efros. "Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks", in IEEE International Conference on Computer Vision (ICCV), 2017. In particular, 'option' based implementation is heavily borrowed from their code ([Link](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/tree/master/options)).

The full dataset is provided by Bianchi, Eric; Hebdon, Matthew (2021). Concrete Crack Conglomerate Dataset. University Libraries, Virginia Tech. Dataset. This dataset can be download at ([Link](https://data.lib.vt.edu/articles/dataset/Concrete_Crack_Conglomerate_Dataset/16625056)). This code has been implemented to operate using sample data. 


Please cite our paper if you find it useful for your research.
```
@article{Shim_CACAIE_2023,
  title={Self-training approach for crack detection using synthesized crack images based on conditional generative adversarial network},
  author={Shim, Seungbo},
  journal={Computer-Aided Civil and Infrastructure Engineering},
  volume={39},
  number={7},
  pages={1019--1041},
  year={2024},
  publisher={Wiley Online Library}
}
```

## Prerequisite

* CUDA 12.1
* pytorch == 2.1.2 
* python-opencv == 4.9.0.80
* matplotlib == 3.8.2

## Installation
```bash
git clone https://github.com//phylun/GenKDCrack.git
```

## Download dataset
Download the sample dataset via the link ([Link](https://drive.google.com/file/d/1f6kgHManFRST8NMJGyrrlsTls304V6in/view?usp=sharing)). Unzip it into the folder `dataset_sample`
```
dataset_sample/labeled/trainConc/JPEGImages
                                /SegmentationClass
                      /valConc/JPEGImages
                              /SegmentationClass
                      /testConc/JPEGImages
                               /SegmentationClass
              /unlabeled/trainConc/JPEGImages
```

## Pre-trained model for knowledge distillation
This code needs pre-trained weight models for teacher networks. The teacher networks are FRRNA and FRRNB whose weights can be downloaded via ([Link](https://drive.google.com/file/d/11-nly73F10iI0xmNiCIoouDPgUt7PX6o/view?usp=sharing)) and ([Link](https://drive.google.com/file/d/1ARZ4W95gH0F212TKg260yiYqS3pdy-uo/view?usp=sharing)), respectively. Move them in the folder `pretrained`


## Training on sample dataset
```
python EnsemKD_SegTrain.py --name project_name \
                           --Snet RegSeg \
                           --n_epoch 3000 \
                           --consist_weight 0.001 \
                           --gpu_ids 0
```

Use your own project name (e.g. KDEnsem_FRRNAB_RegSeg). Snet means segmenation model. In this code, eight models are provided such as CGNet, FDDWNet, ERFNet, DDRNet, RegSeg, PIDNet, Deeplabv3p, and LEDNet. 

In case of supervised learning, use the below script
```
python Super_SegTrain.py --name project_name \
                         --Snet RegSeg \
                         --gpu_ids 0
```

## Evaluate the segmentation model on sample dataset
```
python SegTest.py --name project_name \
                  --Snet RegSeg \
                  --gpu_ids 0
```
This script evaluates the accuracy for all saved models in the folder `outputs/project_name`. When the option of `--epoch 100`, then 100th saved model is called and m-IoU is assessed. Also, adding `--phase test`, 


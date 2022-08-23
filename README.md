# Towards High Performance One-Stage Human Pose Estimation
This is an official pytorch implementation of [*Towards High Performance One-Stage Human Pose Estimation*]. 

## Introduction
In this paper, we aim to largely advance the human pose estimation results of Mask-RCNN and still keep the efficiency. Specifically, we make improvements on the whole process of pose estimation, which contains feature extraction, keypoint detection, and results post-processing. The part of feature extraction is ensured to get enough and valuable information of pose. Then, we introduce a Global Context Module into the keypoints detection branch to enlarge the receptive field, as it is crucial to successful human pose estimation. Lastly, the results post-processing steps are used to further improve the performance. 

The framework of our model is illustrated below:

<img src="/fig/structure.png" style="zoom: 50%;float:left;width: 500px" />

Among them, the network structure of keypoint branches is as follows:

<img src="/fig/network.png" style="zoom: 50%;float:left;width: 500px" />

## Main Results

### Detection results on COCO val2017 set

|       Methods       |  backbone  |  AP  | Ap  .5 | AP .75 | AP (M) | AP (L) |
|---------------------|------------|------|--------|--------|--------|--------|
|      Mask RCNN      | ResNet-50  | 65.5 |  87.2  |  71.1  |  61.3  |  73.4  |
|      Mask RCNN      | ResNet-101 | 66.1 |  87.4  |  72.0  |  61.5  |  74.4  |
|   SimpleBaseline*   | ResNet-50  | 68.9 |  88.2  |  76.5  |  65.5  |  75.2  |
|   SimpleBaseline*   | ResNet-101 | 70.3 |  89.0  |  78.0  |  67.0  |  76.4  |
|     Our method      | ResNet-50  | 68.1 |  88.0  |  74.5  |  63.7  |  76.2  |
|     Our method      | ResNet-101 | 68.3 |  88.0  |  74.8  |  63.8  |  76.6  |

#### Note:
- The experimental data of Mask RCNN comes from [*Detectron2*](https://github.com/facebookresearch/detectron2) project.
- The SimpleBaseline network is based on the [*code implementation*](https://github.com/leoxiaobin/deep-high-resolution-net.pytorch) of the HRNet project. However, flip test is turned off. And person detector has person AP of 55.4 (ResNet-50) and 56.1 (ResNet-101) on COCO val2017 set.

### Detection results on COCO test-dev2017 set

|       Methods       |  backbone  |  AP  | Ap  .5 | AP .75 | AP (M) | AP (L) |
|---------------------|------------|------|--------|--------|--------|--------|
|      Mask RCNN      | ResNet-50  | 63.1 |  87.3  |  68.7  |  57.8  |  71.4  |
|   SimpleBaseline*   | ResNet-50  | 67.7 |  89.0  |  75.2  |  64.4  |  73.6  |
|   SimpleBaseline*   | ResNet-101 | 69.0 |  89.9  |  77.1  |  65.9  |  74.7  |
|     Our method      | ResNet-50  | 66.4 |  88.4  |  73.1  |  62.2  |  73.9  |
|     Our method      | ResNet-101 | 67.1 |  89.0  |  74.0  |  63.0  |  74.7  |


#### Note:
- The experimental data of Mask RCNN comes from the paper [*Mask R-CNN*](https://arxiv.org/abs/1703.06870).
- The SimpleBaseline network is based on the [*code implementation*](https://github.com/leoxiaobin/deep-high-resolution-net.pytorch) of the HRNet project. However, flip test is turned off.

## Environment

The code is developed using python 3.6.6 on Ubuntu 18.04. NVIDIA GPUs are needed. The code is developed and tested using 1 TITAN V GPU.

## Quick start

### Installation

Our code is based on the [*Detectron2*](https://github.com/facebookresearch/detectron2). Therefore, the environment configuration refers to the [*installation instructions*](https://detectron2.readthedocs.io/en/latest/tutorials/install.html) in Detectron2.

Please refer to [*Installation Instructions*](https://detectron2.readthedocs.io/en/latest/tutorials/install.html) for the configuration of the Detectron2 environment and [*Use Builtin Datasets*](/datasets/README.md) for the preparation of the datasets. Then just overwrite our file in Detectron2.

## Training and Testing 

##### Train on COCO val2017 set

```python
# ResNet-50 backbone
./tools/train_net.py --config-file configs/COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml --num-gpus 1 SOLVER.IMS_PER_BATCH 2 SOLVER.BASE_LR 0.0025 SOLVER.MAX_ITER 2160000 SOLVER.STEPS 1680000,2000000

# ResNet-101 backbone
./tools/train_net.py --config-file configs/COCO-Keypoints/keypoint_rcnn_R_101_FPN_3x.yaml --num-gpus 1 SOLVER.IMS_PER_BATCH 2 SOLVER.BASE_LR 0.0025 SOLVER.MAX_ITER 2160000 SOLVER.STEPS 1680000,2000000
```

##### Testing on COCO test-dev2017 set. Our trained model is provided in [best_model_set](https://pan.baidu.com/s/1afpCHab3f6Qj_1ETPfDKEw?pwd=2hub).

```python
# ResNet-50 backbone
./tools/train_net.py --config-file configs/COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml --eval_only --num-gpus 1 MODEL.WEIGHTS /best_model_set/model_r50.pth

# ResNet-101 backbone
./tools/train_net.py --config-file configs/COCO-Keypoints/keypoint_rcnn_R_101_FPN_3x.yaml --eval_only --num-gpus 1 MODEL.WEIGHTS /best_model_set/model_r101.pth
```

### Citation

If you use our code or models in your research, please cite with:

```

```

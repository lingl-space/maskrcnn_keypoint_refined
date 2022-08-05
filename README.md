# Towards High Performance One-Stage Human Pose Estimation
This is an official pytorch implementation of [*Towards High Performance One-Stage Human Pose Estimation*]. 

## Introduction
In this paper, we aim to largely advance the human pose estimation results of Mask-RCNN and still keep the efficiency. Specifically, we make improvements on the whole process of pose estimation, which contains feature extraction, keypoint detection, and results post-processing. The part of feature extraction is ensured to get enough and valuable information of pose. Then, we introduce a Global Context Module into the keypoints detection branch to enlarge the receptive field, as it is crucial to successful human pose estimation. Lastly, the results post-processing steps are used to further improve the performance. 

The framework of our model is illustrated below:

<img src="/fig/structure.png" style="zoom: 80%;float:left" />

Among them, the network structure of keypoint branches is as follows:

<img src="/fig/network.png" style="zoom: 80%;float:left" />

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

### Detection results on COCO test-dev2017 set

|       Methods       |  backbone  |  AP  | Ap  .5 | AP .75 | AP (M) | AP (L) |
|---------------------|------------|------|--------|--------|--------|--------|
|      Mask RCNN      | ResNet-50  | 63.1 |  87.3  |  68.7  |  57.8  |  71.4  |
|   SimpleBaseline*   | ResNet-50  | 67.7 |  89.0  |  75.2  |  64.4  |  73.6  |
|   SimpleBaseline*   | ResNet-101 | 69.0 |  89.9  |  77.1  |  65.9  |  74.7  |
|     Our method      | ResNet-50  | 66.4 |  88.4  |  73.1  |  62.2  |  73.9  |
|     Our method      | ResNet-101 | 67.1 |  89.0  |  74.0  |  63.0  |  74.7  |


# Towards High Performance One-Stage Human Pose Estimation
This is an official pytorch implementation of [*Towards High Performance One-Stage Human Pose Estimation*]. 

## Introduction
In this paper, we aim to largely advance the human pose estimation results of Mask-RCNN and still keep the efficiency. Specifically, we make improvements on the whole process of pose estimation, which contains feature extraction, keypoint detection, and results post-processing. The part of feature extraction is ensured to get enough and valuable information of pose. Then, we introduce a Global Context Module into the keypoints detection branch to enlarge the receptive field, as it is crucial to successful human pose estimation. Lastly, the results post-processing steps are used to further improve the performance. 

The framework of our model is illustrated below:

# Crack-Segmentation
EE511 Computer Vision Course Project On Image Segmentation
**UNet with DAM (Dual Attention Module) for Crack Detection**
**Overview**
This repository contains the implementation of a crack detection system using a modified UNet architecture with a Dual Attention Module (DAM). The UNet architecture used is consistent with the standard UNet, as proposed in the earlier part. The output from the bottleneck of UNet is passed through the DAM, which consists of both the Convolutional Block Attention Module (CBAM) and the Squeeze and Excitation Module. This integration of attention mechanisms enhances the model's ability to distinguish between crack and non-crack regions effectively.

**Architecture**
The architecture follows these key steps:

**Data Pre-processing:**
Features Extracted from Transformer based DINO Architecture.
-"https://github.com/facebookresearch/dino"

The input image size is originally 448 x 448 x 3.
As the UNet architecture requires input of size 128 x 128 x 3, the input image is resized accordingly.
DAM (Dual Attention Module):

DAM is composed of CBAM and Squeeze and Excitation Module.
CBAM extracts both spatial and channel information, while the Squeeze and Excitation Module generates local descriptors providing information about different channels.
The combination of these modules enhances the information used for distinguishing between crack and non-crack areas.
Decoder Output:

The output of the decoder contains 2 channels, representing the probability of each pixel belonging to either the crack or non-crack class.


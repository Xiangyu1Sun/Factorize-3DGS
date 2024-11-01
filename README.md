# F-3DGS: Factorized Coordinates and Representations for 3D Gaussian Splatting (ACM MM 2024)
### Xiangyu Sun, Joo Chan Lee, Daniel Rho, Jong Hwan Ko, Usman Ali and Eunbyung Park

### [[Project Page](https://xiangyu1sun.github.io/Factorize-3DGS/)] [[Paper](https://dl.acm.org/doi/pdf/10.1145/3664647.3681116)]

Our code is based on [3D Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting).

## Method Overview
<div style="background-color:white; padding:10px; display:inline-block;">
 <img src="https://github.com/Xiangyu1Sun/Factorize-3DGS/blob/page/images/factorized_coordinates.png">
</div>
<p align="justify">Examples of factorized coordinates: (a) 25 normal coordinates, (b) 5 &times; 5 factorized coordinates. 
each x and y axis has 5 points, and both represent 25 (5 &times; 5) points. (c) two 5 &times; 5 factorized 
coordinates and a total of 50 points are represented (2 &times; 5 &times; 5), (d) multi-resolution factorized 
coordinates, where two factorized coordinates have different resolutions (3 &times; 3 and 5 &times; 5), represent total 34 points, 
(e) two 3 &times; 3 and one 5 &times; 5 factorized coordinates. A total of 43 points are represented. The best-viewed in color.</p>

## Setup

For installation, we use the same environment as 3D Gaussian Splatting.

We use the same environment as original 3DGS, please follow the link of 3D-GS to install all packages.

https://github.com/graphdeco-inria/gaussian-splatting

Thanks for the excellent work in 3D-GS！


## Training
 
Here, we will explain how to use our codes.
Our codes are divided into two parts, one for synthetic-nerf dataset and other for Tanks&Temples dataset.

## Pre-processing

### Step 1.  Get hist path of original 3DGS

To use our code, first we need to use original 3DGS code to train each scene and get the .ply file for each scene. The distribution of original Gaussians will be used in next step.

### Step 2. Using the code in train.sh file to train synthetic-nerf and Tanks&Temples dataset


## Running

### In nerf-synthetic dataset

```shell
python train.py -s /workspace/datasets/nerf_synthetic/chair -m exp/chair   --eval --hist_path /gaussian-ori/gaussian-splatting/exp/chair/point_cloud/iteration_30000/point_cloud.ply   
```

#### -s
the source of dataset
#### --hist_path
the path of .ply file trained by original 3DGS
#### -m
the output of the model


### Tanks&Temples dataset

```shell
python train.py -s /workspace/datasets/TanksAndTemple/Barn  -m TanksAndTemple/Barn  --eval -r 2  -w --hist_path /gaussian-ori/gaussian-splatting/TanksAndTemple/Barn/point_cloud/iteration_30000/point_cloud.ply
```

#### -s       
the source of dataset
#### --hist_path
the path of .ply file trained by original 3DGS
#### -m
the output of the model
#### -r
the resolution of images
#### -w
the background is white

#### Refer to other arguments of [3DGS](https://github.com/graphdeco-inria/gaussian-splatting).

## BibTeX
```
@inproceedings{10.1145/3664647.3681116,
author = {Sun, Xiangyu and Lee, Joo Chan and Rho, Daniel and Ko, Jong Hwan and Ali, Usman and Park, Eunbyung},
title = {F-3DGS: Factorized Coordinates and Representations for 3D Gaussian Splatting},
year = {2024},
isbn = {9798400706868},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3664647.3681116},
doi = {10.1145/3664647.3681116},
abstract = {The neural radiance field (NeRF) has made significant strides in representing 3D scenes and synthesizing novel views. Despite its advancements, the high computational costs of NeRF have posed challenges for its deployment in resource-constrained environments and real-time applications. As an alternative to NeRF-like neural rendering methods, 3D Gaussian Splatting (3DGS) offers rapid rendering speeds while maintaining excellent image quality. However, as it represents objects and scenes using a myriad of Gaussians, it requires substantial storage to achieve high-quality representation. To mitigate the storage overhead, we propose Factorized 3D Gaussian Splatting (F-3DGS), a novel approach that drastically reduces storage requirements while preserving image quality. Inspired by classical matrix and tensor factorization techniques, our method represents and approximates dense clusters of Gaussians with significantly fewer Gaussians through efficient factorization. We aim to efficiently represent dense 3D Gaussians by approximating them with a limited amount of information for each axis and their combinations. This method allows us to encode a substantially large number of Gaussians along with their essential attributes'such as color, scale, and rotation-necessary for rendering using a relatively small number of elements. Extensive experimental results demonstrate that F-3DGS achieves a significant reduction in storage costs while maintaining comparable quality in rendered images. Our project page is available at https://xiangyu1sun.github.io/Factorize-3DGS/.},
booktitle = {Proceedings of the 32nd ACM International Conference on Multimedia},
pages = {7957–7965},
numpages = {9},
keywords = {3d reconstruction, compression, real-time rendering, tensor factorization},
location = {Melbourne VIC, Australia},
series = {MM '24}
}
```

# F-3DGS: Factorized Coordinates and Representations for 3D Gaussian Splatting (ACM MM 2024)
### Xiangyu Sun, Joo Chan Lee, Daniel Rho, Jong Hwan Ko, Usman Ali and Eunbyung Park

### [[Project Page](https://xiangyu1sun.github.io/Factorize-3DGS/)] [[Paper(arxiv)](https://arxiv.org/abs/2405.17083)]

Our code is based on [3D Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting).

## Method Overview
<img src="https://github.com/Xiangyu1Sun/Factorize-3DGS/tree/page/images/factorized_coordinates.png">

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
@article{sun2024f,
  title={F-3DGS: Factorized Coordinates and Representations for 3D Gaussian Splatting},
  author={Sun, Xiangyu and Lee, Joo Chan and Rho, Daniel and Ko, Jong Hwan and Ali, Usman and Park, Eunbyung},
  journal={arXiv preprint arXiv:2405.17083},
  year={2024}
}
```

# SOLUTION

## Overview

This repository contains implementations of core 3D Computer Vision and Neural Rendering algorithms developed for educational and research purposes.

The project combines classical geometric computer vision methods with modern neural rendering approaches.

Implemented modules:

- Camera Calibration (PnP)
- Structure from Motion (SfM)
- Differentiable Rasterization
- Neural Radiance Fields (NeRF)

Main technologies:
- Python
- PyTorch
- OpenCV
- NumPy
- SciPy
- nvdiffrast

---

# 1. Camera Calibration

## Goal

Estimate the camera projection matrix from 2D-3D correspondences and recover intrinsic/extrinsic camera parameters.

## Implemented Methods

- Direct Linear Transform (DLT)
- Levenberg-Marquardt optimization
- Projection matrix factorization

## Pipeline

1. Build linear system:

A · m = 0

2. Solve using eigenvector decomposition

3. Refine projection matrix by minimizing reprojection error

4. Decompose:

M = K [R | t]

## Results

- Accurate intrinsic parameter recovery
- Stable camera pose estimation
- Low reprojection error (< 0.5 px)

---

# 2. Structure from Motion

## Goal

Recover sparse 3D scene structure and camera trajectories from image sequences.

## Implemented Methods

- ORB feature extraction
- Lowe ratio test matching
- Fundamental matrix estimation with RANSAC
- Track construction using Union-Find
- Triangulation
- PnP pose estimation

## Features

- Feature caching
- Geometric outlier filtering
- Reprojection error validation

## Results

- Stable trajectory reconstruction
- Sparse 3D point cloud recovery
- Robust pose estimation for unknown frames

---

# 3. Differentiable Rasterization

## Goal

Optimize mesh textures using differentiable rendering.

## Implemented Methods

- Rasterization with nvdiffrast
- UV interpolation
- Mipmap texture sampling
- Gradient-based texture optimization

## Optimization

The texture is optimized directly through image reconstruction loss:

loss = F.mse_loss(rendered_image, target_image)
## Results

- Successful texture reconstruction
- Stable optimization process
- Final texture MSE ≈ 0.0011

---

# 4. Neural Radiance Fields (NeRF)

## Goal

Learn continuous volumetric scene representations from multi-view images.

## Implemented Components

- Positional encoding
- Ray generation
- Volume rendering
- Neural radiance field MLP
- Neural optimization

## Architecture

- Positional encoding (L=10)
- MLP with 3 hidden layers
- RGB + density prediction

## Volume Rendering

:contentReference[oaicite:0]{index=0}

The model integrates colors and densities along camera rays to synthesize novel views.

## Results

- Novel view synthesis
- Continuous scene representation
- Test MSE ≈ 0.00013

---

# Conclusion

This project explores both classical and neural approaches to 3D scene reconstruction.

The implementation helped develop practical understanding of:

- geometric computer vision
- differentiable rendering
- neural scene representations
- optimization methods in 3D AI systems

Future improvements may include:
- hierarchical NeRF sampling
- bundle adjustment
- Gaussian Splatting
- dense reconstruction methods
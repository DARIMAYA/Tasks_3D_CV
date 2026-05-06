# Tasks_3D_CV

Implementation of key 3D Computer Vision algorithms from scratch.

## Table of Contents

| Task | Methods |
|------|---------|
| Camera Calibration (PnP) | DLT + Levenberg-Marquardt, factorization into K, R, t |
| Structure from Motion | ORB, ratio test, RANSAC, Union-Find tracking, PnP |
| Differentiable Rasterization | nvdiffrast, texture optimization |
| NeRF | Positional encoding, volumetric rendering |

## Для работы в Google Colab

**Все ноутбуки написаны и протестированы в Google Colab.** Рекомендуется запускать их там же.

[Open In Colab](https://colab.research.google.com/)

---

## Быстрый старт в Google Colab

1. Открой [Google Colab](https://colab.research.google.com/)
2. Загрузи ноутбук: `File` → `Upload notebook` → выбери `.ipynb` файл
3. Выбери среду с GPU: `Runtime` → `Change runtime type` → `T4 GPU` или `A100`
4. Выполняй ячейки последовательно (`Shift + Enter`)

### Важно для Colab:

При каждом запуске Colab нужно установить библиотеки. Добавь в **первую ячейку** ноутбука:

```python
!pip install numpy opencv-python matplotlib scipy torch torchvision trimesh xatlas ninja
!pip install git+https://github.com/yzhq97/nvdiffrast.git

## Quick Start

```bash
git clone https://github.com/DARIMAYA/Tasks_3D_CV.git
cd Tasks_3D_CV
pip install -r requirements.txt
```

## Camera Calibration (PnP)

**Goal:** Estimate projection matrix M from 2D-3D correspondences and decompose into K, R, t.

**Steps:**
1. Build linear system A·m = 0 (DLT)
2. Find eigenvector for smallest eigenvalue
3. Refine using Levenberg-Marquardt (reprojection error)
4. Factorize M = K[R|t] using cv2.decomposeProjectionMatrix

```python
M = dlt_initialization(x3d, x2d)
M_refined = levenberg_marquardt(M, x3d, x2d)
K, R, t = factorize_projection_matrix(M_refined)
```

## Structure from Motion

**Goal:** Recover camera trajectories from image sequence.

**Pipeline:**
- ORB feature detection with caching
- Feature matching with Lowe's ratio test (0.75)
- Fundamental matrix filtering with RANSAC (threshold=3.0px)
- Track construction using Union-Find
- Triangulation with reprojection error filtering (<10px)
- PnP RANSAC for unknown camera poses

```python
estimate_trajectory("data/", "output/")
# Result: output/all_poses.txt
```

## Differentiable Rasterization

**Goal:** Optimize mesh texture so renders match real images.

**Tech stack:** nvdiffrast, PyTorch

```python
texture = torch.full((512, 512, 3), 0.5, requires_grad=True, device="cuda")
optimizer = torch.optim.Adam([texture], lr=1e-3)

for iteration in range(1000):
    rendered, _ = render_textured(mesh, mvp, texture=texture)
    loss = F.mse_loss(rendered, target_image)
    loss.backward()
    optimizer.step()
    texture.clamp_(0.0, 1.0)
```

**Result:** MSE < 0.0012 after 1000 iterations.

## NeRF (Neural Radiance Fields)

**Goal:** Learn continuous volumetric scene representation from images.

**Architecture:**
- Positional encoding (L=10)
- MLP: 256 → 256 → 256 → 4 (RGB + sigma)
- Volumetric rendering with 64 bins

```python
radiance_field = VanillaNeRF().cuda()

for epoch in range(10):
    for batch in dataloader:
        pred = render_radiance_field(rays, radiance_field)
        loss = F.mse_loss(pred, target)
        loss.backward()
        optimizer.step()
```

**Results after 10 epochs:** Clean 3D scene representation.

## Results Summary

| Method | Metric | Value |
|--------|--------|-------|
| DLT + LM | Reprojection error | < 0.5 px |
| SfM | Recovered poses (out of 100) | ~95 |
| Diff Rasterization | MSE after 1000 iters | 0.0011 |
| Vanilla NeRF | Test MSE | 0.00013 |

## Project Structure

```
Tasks_3D_CV/
├── camera_calibration/     # PnP, DLT, factorization
├── sfm/                    # ORB, tracks, PnP, trajectory
├── diff_rasterization/     # nvdiffrast, texture optimization
├── nerf/                   # NeRF, volumetric rendering
└── requirements.txt
```

## Requirements

- Python 3.12+
- PyTorch 2.10+ with CUDA
- OpenCV 4.10+
- nvdiffrast
- trimesh, xatlas, ninja

```bash
pip install numpy opencv-python matplotlib scipy torch torchvision trimesh xatlas ninja
pip install git+https://github.com/NVlabs/nvdiffrast.git
```

## References

- Hartley, Zisserman - "Multiple View Geometry"
- Mildenhall et al. - "NeRF: Representing Scenes as Neural Radiance Fields"
- NVlabs - nvdiffrast

---

## requirements.txt

```txt
numpy>=1.20
opencv-python>=4.5
matplotlib>=3.5
scipy>=1.10
torch>=2.0
torchvision
trimesh>=4.0
xatlas>=0.0.11
ninja>=1.10
nvdiffrast@git+https://github.com/NVlabs/nvdiffrast.git
```


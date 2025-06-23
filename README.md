# 基于 TensoRF 的三维场景重建与新视角合成

本项目旨在使用 TensoRF 技术实现高效的三维场景重建与高质量的新视角合成。

## 1. 关于 TensoRF

近年来，神经辐射场（Neural Radiance Fields, NeRF）在三维场景重建和新视角合成领域取得了突破性进展，能够生成高度逼真的渲染效果。然而，NeRF 也存在一些固有的局限性，最主要的是其高昂的计算成本和缓慢的训练速度。为了解决这些问题，研究者们提出了一系列优化和加速技术，其中 **TensoRF (Tensorial Radiance Fields)** 是一种非常高效且出色的代表性工作。

### 1.1 NeRF: 神经辐射场简介

NeRF 的核心思想是使用一个全连接的深度神经网络（通常是多层感知机 MLP）来隐式地表示一个连续的五维（5D）场景函数。这个函数输入一个三维空间点坐标 $(x, y, z)$ 和一个二维观测方向 $(\theta, \phi)$，输出该空间点的体积密度（Volume Density, $\sigma$）和与方向相关的颜色（Color, $c$)。

$$
MLP: (x, y, z, \theta, \phi) \rightarrow (c, \sigma)
$$

通过沿相机射线对这些密度和颜色值进行积分的经典体渲染（Volume Rendering）技术，NeRF 可以合成任意新视角的图像。尽管效果惊艳，但 NeRF 的训练过程极其耗时，因为它需要对每个像素的每条光线上的大量采样点都执行一次庞大的 MLP 网络前向传播，导致训练一个场景通常需要数小时甚至数天。

### 1.2 TensoRF: 张量化辐射场

为了打破 NeRF 的性能瓶颈，TensoRF 提出了一种全新的思路：不再使用大型 MLP 来表示整个场景，而是将场景的辐射场显式地建模为一个四维张量（4D Tensor），可以理解为一个三维体素网格（3D Voxel Grid），每个体素都包含多通道的特征。

TensoRF 的关键创新在于，它没有直接优化这个巨大的 4D 张量，而是通过 **张量分解（Tensor Factorization）** 的方法，将其分解为多个紧凑的、低秩的张量分量（主要是向量和矩阵）。这种分解极大地降低了模型的参数量和计算复杂度。

TensoRF 主要使用了两种分解方式：

1.  **CP 分解 (CANDECOMP/PARAFAC Decomposition)**: 将 4D 张量分解为一系列秩为1的向量外积之和。这种方式可以得到极度紧凑的模型（小于 4MB），渲染质量优于 NeRF，并且训练速度快得多。

2.  **VM 分解 (Vector-Matrix Decomposition)**: 这是一种新的分解方法，将 4D 张量分解为多个向量和矩阵的外积。VM 分解在保持模型紧凑（小于 75MB）和快速训练（10分钟以内）的同时，能够实现比 CP 分解及其他SOTA方法更高的渲染质量。

通过这种方式，TensoRF 将场景表示从隐式的神经网络查询转变为显式的、可分解的张量计算。这使得它可以在不依赖定制化 CUDA 内核的情况下，仅使用标准的 PyTorch 框架就实现极高的效率，兼顾了高渲染质量、快速训练和低内存占用等多个优点。

## 2. 数据集与参数

### 2.1 数据集: Tanks and Temples ("Truck" 场景)

我们采用了业界知名的三维重建基准数据集 **Tanks and Temples**，并选用其中的 **"Truck"** 场景。

该数据集由英特尔实验室发布，专门用于评估和比较三维重建算法的性能。因其高质量、高分辨率和场景多样性，已成为衡量算法在新视角合成、三维重建完整性和精细度方面的"黄金标准"。"Truck" 场景提供了 250 张对一辆大型卡车拍摄的多视角、高分辨率图像，其中包含了丰富的纹理、复杂的几何形状以及自然光照下的阴影和反光，非常适合用于评估模型对于真实世界场景的细节重建能力。

-   **训练集**: 从中随机抽取 218 张图像，用于学习场景的几何与外观。
-   **测试集**: 另外随机抽取 32 张图像，用于评估模型对未见过视角的渲染能力。

### 2.2 实验配置

模型训练和评估所使用的主要参数如下：

| 参数分类 | 参数名称 | 值 |
| :--- | :--- | :--- |
| **模型配置** | `model_name` | `TensorVMSplit` |
| | `shadingMode` | `MLP_Fea` |
| | `n_lamb_sigma` (密度特征维度) | `[16, 16, 16]` |
| | `n_lamb_sh` (外貌特征维度) | `[48, 48, 48]` |
| | `fea2denseAct` (激活函数) | `softplus` |
| | `N_voxel_init` (初始体素数) | `128^3` |
| | `N_voxel_final` (最终体素数) | `300^3` |
| **训练参数** | `n_iters` (总迭代次数) | `30000` |
| | `batch_size` | `4096` |
| | `upsamp_list` (上采样迭代点) | `[2000, 3000, 4000, 5500, 7000]` |
| | `update_AlphaMask_list` (Alpha掩码更新点)| `[2000, 4000]` |
| | `optimizer` | Adam (默认) |
| | `learning_rate` | (默认: `lr_init=0.02`, `lr_basis=0.001`) |
| | `loss_function` | MSE Loss (L2 Loss) |
| **正则化** | `TV_weight_density` (TV正则项-密度) | `0.1` |
| | `TV_weight_app` (TV正则项-外貌) | `0.01` |
| **评价指标** | | PSNR, SSIM, LPIPS |

## 3. 实验结果

在测试集上取得的平均性能指标如下：

| Metric       |   Value |
|:-------------|--------:|
| PSNR         | 26.9014 |
| SSIM         |  0.9133 |
| LPIPS (VGG)  |  0.1268 |
| LPIPS (Alex) |  0.1471 |

## 4. 如何使用

1.  **克隆并安装代码库**
    ```bash
    git clone https://github.com/apchenstu/TensoRF.git
    conda create -n TensoRF python=3.8
    conda activate TensoRF
    pip install torch torchvision
    pip install tqdm scikit-image opencv-python configargparse lpips imageio-ffmpeg kornia lpips tensorboard
    ```

2.  **下载预训练模型**
    
    从[此链接](https://drive.google.com/drive/folders/1JGgeRJzs-e7Jrn7pz95RWnfoXpcRkBWk?usp=drive_link)下载模型权重文件。

3.  **执行渲染**
    
    ```bash
    python train.py --config configs/your_own_data.txt --ckpt path/to/your/checkpoint --render_only 1 --render_test 1 
    ```

## 5. 资源下载

*   [预训练模型权重与效果演示视频](https://drive.google.com/drive/folders/1JGgeRJzs-e7Jrn7pz95RWnfoXpcRkBWk?usp=drive_link) 

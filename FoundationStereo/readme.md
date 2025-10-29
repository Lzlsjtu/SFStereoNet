# FoundationStereo: Zero-Shot Stereo Matching

这是我们 CVPR 2025 Oral (**最佳论文提名**) 的官方实现。

- [官网](https://nvlabs.github.io/FoundationStereo/)  
- [论文](https://arxiv.org/abs/2501.09898)  
- [演示视频](https://www.youtube.com/watch?v=R7RgHxEXB3o)

作者：Bowen Wen, Matthew Trepte, Joseph Aribido, Jan Kautz, Orazio Gallo, Stan Birchfield

## 简介
FoundationStereo 是一个用于立体匹配的基础模型，专注于零样本泛化能力。该模型通过大规模合成数据集训练（约 100 万对立体图像），并结合网络结构优化（如侧调特征骨干和长程上下文推理），实现跨域的高鲁棒性和高精度立体匹配。

**功能**：输入左右立体图像，输出稠密视差图，可进一步生成深度图或 3D 点云。

## 环境安装

建议在 Linux GPU 环境下运行（3090, 4090, A100, V100, Jetson Orin 等）：

```bash
conda env create -f environment.yml
conda run -n foundation_stereo pip install flash-attn
conda activate foundation_stereo
````

> 注意：`flash-attn` 需要单独安装以避免环境创建错误。

## 模型权重下载

请下载权重文件夹（如 `23-51-11` 或 `11-33-40`）并放入项目路径下的 `FoundationStereo/pretrained_models/` 目录：

| 模型       | 下载地址                                                                                                 | 说明                      |
| -------- | ---------------------------------------------------------------------------------------------------- | ----------------------- |
| 23-51-11 | [Google Drive](https://drive.google.com/drive/folders/1VhPebc_mMxWKccrv7pdQLTvXYVcLYpsf?usp=sharing) | 性能最佳，基于 Vit-large       |
| 11-33-40 | [Google Drive](https://drive.google.com/drive/folders/1VhPebc_mMxWKccrv7pdQLTvXYVcLYpsf?usp=sharing) | 精度略低，但推理更快，基于 Vit-small |

> 下载完成后，文件夹结构示例：
>
> ```
> FoundationStereo/
> ├─ pretrained_models/
> │  └─ 23-51-11/
> │     ├─ model_best_bp2.pth
> │     └─ ... (其他权重文件)
> ├─ core/
> ├─ environment.yml
> ├─ ply_reconstruction.py
> ├─ Utils.py
> └─ readme.md
> ```

输出包括稠密视差图及 3D 点云。

## 使用注意

* 左右图像需经过**校正与去畸变**，确保极线对齐。
* 推荐使用 PNG 图像，RGB 彩色立体效果最佳。
* 高分辨率图像可使用 `--hiera 1` 或缩放参数 `--scale` 控制速度和精度。

## 更多选项

请使用 `python scripts/run_demo.py --help` 查看所有参数说明。



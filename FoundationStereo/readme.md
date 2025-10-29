
# FoundationStereo：零样本立体匹配

这是我们的论文（被CVPR 2025接收，**全强接收**）的官方实现

[[项目主页]](https://nvlabs.github.io/FoundationStereo/) [[论文]](https://arxiv.org/abs/2501.09898) [[视频]](https://www.youtube.com/watch?v=R7RgHxEXB3o)

作者：Bowen Wen, Matthew Trepte, Joseph Aribido, Jan Kautz, Orazio Gallo, Stan Birchfield

# 摘要
通过针对特定领域的微调，深度立体匹配在基准数据集上取得了巨大进展。然而，实现强大的零样本泛化能力——这是其他计算机视觉任务中基础模型的标志——仍然是立体匹配面临的挑战。我们提出了FoundationStereo，一个专为深度立体估计设计的基础模型，旨在实现强大的零样本泛化能力。为此，我们首先构建了一个大规模（100万立体对）的合成训练数据集，具有高度多样性和真实感，随后通过自动自净流程去除模糊样本。接着我们设计了多个增强可扩展性的网络架构组件，包括一个侧调谐特征主干网络（将视觉基础模型中的单目先验知识进行适配以缩小仿真与真实差距），以及用于有效代价体滤波的长程上下文推理模块。这些组件共同实现了跨域的强鲁棒性和高精度，为零样本立体深度估计树立了新标杆。

<p align="center">
  <img src="https://raw.githubusercontent.com/NVlabs/FoundationStereo/website/static/images/intro.jpg" width="800"/>
</p>

**精简版**：我们的方法输入一对立体图像，输出密集视差图，可转换为公制尺度深度图或3D点云。

<p align="center">
  <img src="./teaser/input_output.gif" width="600"/>
</p>

# 排行榜 🏆
我们在全球[Middlebury排行榜](https://vision.middlebury.edu/stereo/eval3/)和[ETH3D排行榜](https://www.eth3d.net/low_res_two_view)上获得第一名。

<p align="center">
  <img src="https://raw.githubusercontent.com/NVlabs/FoundationStereo/website/static/images/middlebury_leaderboard.jpg" width="700"/>
  <br>
  <img src="https://raw.githubusercontent.com/NVlabs/FoundationStereo/website/static/images/eth_leaderboard.png" width="700"/>
</p>

# 与单目深度估计的对比
我们的方法在不同场景的零样本立体匹配任务中超越了现有方法。

<p align="center">
  <img src="https://raw.githubusercontent.com/NVlabs/FoundationStereo/website/static/images/mono_comparison.png" width="700"/>
</p>

# 安装
```
conda env create -f environment.yml
conda activate foundation_stereo
```

# 模型权重
• 从[这里](https://drive.google.com/drive/folders/1VhPebc_mMxWKccrv7pdQLTvXYVcLYpsf?usp=sharing)下载基础模型用于零样本推理。将整个文件夹（如`23-51-11`）放入`./pretrained_models/`目录。

# 运行示例
```
python scripts/run_demo.py --left_file ./assets/left.png --right_file ./assets/right.png --ckpt_dir ./pretrained_models/model_best_bp2.pth --out_dir ./test_outputs/
```
可查看输出的点云结果。

<p align="center">
  <img src="./teaser/output.jpg" width="700"/>
</p>

使用提示：
• 输入左右图像需经过**校正和去畸变**（即无鱼眼畸变且极线水平对齐）。使用如Zed等立体相机拍摄的图像通常已处理过该问题
• 不要混淆左右图像，左图像应来自左侧相机（物体在图像中位置更靠右）
• 建议使用无损压缩的PNG格式
• 查看所有选项：`python scripts/run_demo.py --help`
• 生成自有数据的点云需指定相机内参。内参文件第一行为9维内参矩阵，第二行为基线距离（米）
• 高分辨率图像（>1000像素）可启用`--hiera 1`进行分层推理
• 加速推理可降低分辨率（如`--scale 0.5`）或减少迭代次数（如`--valid_iters 16`）

# ONNX/TensorRT推理（实验性）
生成ONNX模型：
• 修改代码以禁用flash-attention（见[此issue](https://github.com/NVlabs/FoundationStereo/issues/13#issuecomment-2708791825)）
• 生成ONNX：
```
export XFORMERS_DISABLED=1
python scripts/make_onnx.py --save_path ./output/foundation_stereo.onnx --ckpt_dir ./pretrained_models/23-51-11/model_best_bp2.pth --height 480 --width 640 --valid_iters 22
```
• 转换为TensorRT：
```
trtexec --onnx=./output/foundation_stereo.onnx --saveEngine=./output/foundation_stereo.engine --fp16 --verbose
```
在3090 GPU上可获得6倍加速。此功能尚处实验阶段，欢迎贡献代码！

# FSD数据集
<p align="center">
  <img src="https://raw.githubusercontent.com/NVlabs/FoundationStereo/website/static/images/sdg_montage.jpg" width="800"/>
</p>

完整数据集[下载](https://drive.google.com/drive/folders/1YdC2a0_KTZ9xix_HyqNMPCrClpm0-XFU?usp=sharing)（>1TB），提供[样例数据](https://drive.google.com/file/d/1dJwK5x8xsaCazz5xPGJ2OKFIWrd9rQT5/view?usp=drive_link)（3GB）。每个数据点包含：
• 左右图像
• 真值视差图

使用示例脚本可视化：
```
python scripts/vis_dataset.py --dataset_path ./DATA/sample/manipulation_v5_realistic_kitchen_2500_1/dataset/data/
```

# 常见问题
• Q: Conda安装失败？<br>
  A: 参考[此issue](https://github.com/NVlabs/FoundationStereo/issues/20)

• Q: GPU不支持Flash attention？<br>
  A: 见[此解决方案](https://github.com/NVlabs/FoundationStereo/issues/13#issuecomment-2708791825)

• Q: cuDNN错误提示？<br>
  A: 可能是显存不足，尝试降低分辨率

• Q: 如何搭配RealSense使用？<br>
  A: 参考[此讨论](https://github.com/NVlabs/FoundationStereo/issues/26)

# 引用
```
@article{wen2025stereo,
  title={FoundationStereo: Zero-Shot Stereo Matching},
  author={Bowen Wen and Matthew Trepte and Joseph Aribido and Jan Kautz and Orazio Gallo and Stan Birchfield},
  journal={CVPR},
  year={2025}
}
```

# 致谢
感谢NVIDIA Isaac团队成员的技术支持，感谢DINOv2、DepthAnything V2、Selective-IGEV和RAFT-Stereo等项目的代码开源，感谢CVPR审稿人的宝贵意见。

# 联系方式
技术咨询请联系[Bowen Wen](https://wenbowen123.github.io/) (bowenw@nvidia.com)。

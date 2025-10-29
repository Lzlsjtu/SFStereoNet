# SegFace: 高精度人脸分割工具

SegFace 是一个专注于人脸语义分割的深度学习模型，能够将人脸图像划分为关键面部区域（如眼睛、鼻子、嘴巴、耳朵、颈部、头发等），特别擅长处理长尾类别（如耳环、项链等不常见类别），广泛应用于面部编辑、换脸、化妆增强等任务。

## 模型优势
- **长尾类别分割性能突出**：采用轻量级 Transformer 解码器与可学习的类别特定 token，实现每个 token 独立建模各类别，提高长尾类别分割精度。
- **多尺度特征融合**：结合 MLP 融合策略，有效增强细粒度面部特征，同时缓解主类别（如脸部、头发）训练主导问题。
- **高效与可移植**：可替换为 MobileNetV3 主干网络，实现高达 95.96 FPS 的快速推理，适用于低算力设备。
- **改进方法**：采用“两次调用粗-细分割”的策略：
  1. **第一次粗分割**：快速定位人脸或目标区域；
  2. **第二次精细分割**：在裁剪的正方形区域上再次预测，提高边缘精度，显著减少锯齿效果。

## 模型权重下载与放置
预训练模型可通过 [HuggingFace SegFace](https://huggingface.co/kartiknarayan/SegFace) 下载，下载后应放置在项目 `ckpts` 文件夹下对应子目录中：

- **示例 Python 下载方式**：
```python
from huggingface_hub import hf_hub_download

# 下载 Swin Base backbone 在 CelebA 数据集上的权重
hf_hub_download(
    repo_id="kartiknarayan/SegFace",
    filename="swin_base_lapa_512/model_299.pt",
    local_dir="D:/selfprogram/3DFaceProjects/SFStereoNet-main/SegFace/ckpts/swin_base_lapa_512"
)
````

* **目录结构示例**：

```
SegFace/
├── ckpts/
│   ├── convnext_celeba_512/
│   │       model_299.pt
│   └── swin_base_lapa_512/
│           model_299.pt
│           test.log
```

> 根据需要可选择不同 backbone 和数据集权重，确保放置在对应子文件夹下。

## 文件夹结构

```
SegFace/
├── .env
├── dataloader.py
├── face_processing.py
├── SegFace_demo.py
├── README.md
├── ckpts/                  # 模型权重
├── datasets/               # 数据集脚本
│   ├── celebamask_hq.py
│   ├── helen.py
│   └── lapa.py
├── loss/                   # 损失函数实现
│   ├── dice_loss.py
│   ├── focal_loss.py
│   ├── weighted_ce.py
│   └── __init__.py
├── network/                # 模型定义
│   ├── __init__.py
│   └── models/
│       ├── segface_celeb.py
│       ├── segface_helen.py
│       ├── segface_lapa.py
│       ├── transformer.py
│       ├── utils_models.py
│       └── __init__.py
└── utils/
    └── utils.py
```

## 安装

```bash
conda env create --file environment.yml
conda activate segface

# 配置 .env 文件
echo 'ROOT_PATH=D:/selfprogram/3DFaceProjects/SFStereoNet-main/SegFace' > .env
echo 'DATA_PATH=/path/to/dataset' >> .env
echo 'LOG_PATH=/path/to/checkpoints' >> .env
```

## 使用方法

### 推理示例

```python
from face_processing import extract_and_Seg_Faces

args.image_folder = "/path/to/input_images"
args.output_dir = "/path/to/output"
args.model_path = "D:/selfprogram/3DFaceProjects/SFStereoNet-main/SegFace/ckpts/swin_base_lapa_512/model_299.pt"
args.input_resolution = 512
args.backbone = "swin_base"
args.model = "swin_base"
args.save_parts = True

# 分割类型: 'face', 'face_neck', 'face_ears', 'full', 'manual'
extract_and_Seg_Faces(args, Type="face", saveparams=True)
```
```bash
# 原项目的推理代码在SegFace_demo.py文件中
python SegFace_demo.py
```
### 训练示例

```bash
python train.py --dataset celebamask_hq --model swin_base --input_resolution 512
```

## 数据集下载

* [CelebAMask-HQ](https://mmlab.ie.cuhk.edu.hk/projects/CelebA/CelebAMask_HQ.html)
* [LaPa](https://github.com/jd-opensource/lapa-dataset)
* [Helen](https://github.com/JPlin/Relabeled-HELEN-Dataset)

## 引用

```bibtex
@inproceedings{narayan2025segface,
  title={Segface: Face segmentation of long-tail classes},
  author={Narayan, Kartik and Vs, Vibashan and Patel, Vishal M},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={39},
  number={6},
  pages={6182--6190},
  year={2025}
}
```

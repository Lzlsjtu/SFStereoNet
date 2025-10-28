# SFStereoNet
本项目构建了一套高精度双目人脸三维重建系统，通过软硬件协同优化，在Facescape数据集上实现了亚毫米级（0.4 mm以内）的重建精度。在算法方面，采用开源模型 SegFace 以提升面部细长尾部位（如耳朵、鼻翼）的语义分割精度，并结合 NVIDIA FoundationStereo 模型进行高精度双目视差估计；在硬件方面，系统由 MVL-KF0814M-12MPE 镜头与 MV-CH050-10UC 相机组成双目视觉采集平台，配合高精度多自由度夹持装置，保证数据采集的稳定性与一致性。计算平台配备 NVIDIA RTX 3060 GPU 和 AMD Ryzen 7 5800H CPU，软件环境基于 WSL2 开发与训练。该系统可广泛应用于高精度人脸建模、表情重建、医学影像及数字人生成等领域。

---

## 🧠 数据与模型说明

本工程综合利用了 **FaceScape**、**SegFace** 与 **FoundationStereo** 三大关键组件，构建出一个兼具数据多样性、语义精细度与几何通用性的高精度三维人脸重建体系。

---

### 📚 FaceScape 数据集（CVPR 2020 / TPAMI 2023）

[FaceScape](https://facescape.nju.edu.cn/) 是迄今最具规模与精度的高质量三维人脸数据集之一，包含：
- **847 位受试者 × 20 种表情**，共约 **16,940 个拓扑一致的 3D 面部模型**；
- 每个样本均具备 **高分辨率多视角图像**、**精确相机参数** 以及 **位移纹理贴图（4K 分辨率）**；
- 数据涵盖性别、年龄等多维度人群差异，适合模型的跨域泛化研究；
- 提供了官方 **Bilinear 参数模型**、**渲染工具链**、**对齐与配准工具**，便于从点云到网格的系统重建。

> 在本工程中，FaceScape 被用于 **提供真值三维形状** 与 **模拟多视角观测数据**，以实现基于真实参数的仿真验证。

---

### 🎯 SegFace（AAAI 2025）: 面向长尾类别的人脸语义分割

[SegFace](https://huggingface.co/kartiknarayan/SegFace) 是首个针对人脸语义分割中**长尾类别（如耳环、帽子、眼镜等**的高性能方法。其主要特点包括：

- 🧩 **类特定可学习 token**：每个类别拥有独立的 transformer token，提升了稀有类的表现；
- ⚙️ **轻量化结构**：可在移动端以 >95 FPS 运行；
- 🧠 **多尺度特征融合**：结合 MLP 融合与跨注意力机制，实现对头部和长尾类的平衡建模；
- 🏆 在 CelebAMask-HQ 与 LaPa 数据集上均刷新 SOTA（最高 93.03 mean F1）。

> 在本项目中，SegFace 用于 **人脸语义掩膜生成**，辅助 FoundationStereo 提取面部有效区域并去除背景干扰，从而提升立体匹配的鲁棒性与几何一致性。

---

### 🌍 FoundationStereo（CVPR 2025）: 通用零样本立体匹配框架

[FoundationStereo](https://github.com/NVlabs/FoundationStereo) 是 NVIDIA 提出的立体匹配基础模型，具备 **强大的零样本泛化能力**。它刚发布时在 Middlebury 与 ETH3D 立体匹配精度排行榜上取得了全球第一的成绩。核心特征如下：

- 🧠 **基础模型架构**：基于侧调谐（side-tuning）特征主干，将视觉基础模型的单目先验迁移至立体估计；
- 🌀 **百万级合成训练数据**：利用自动自净流程构建高质量的 100 万对立体图像数据；
- 🔭 **长程上下文滤波模块**：显著提升跨场景视差一致性；
- 🧩 **零样本泛化能力**：无需针对特定相机或场景进行微调即可稳定生成高精度视差图；
- 🚀 **ONNX / TensorRT 加速**：支持推理部署与高分辨率图像输入。

> 在本工程中，FoundationStereo 是**核心深度重建模块**，其通用性使得系统能在多种双目设备、光照条件及人群类型下保持稳定性能。这一特性确保了我们的系统具备“**跨数据集可迁移性**”与“**领域无关性**”，从而成为**面向通用人脸立体重建任务的基础方案**。

---

## 🧩 算法整体流程

以下为系统的整体管线示意（Algorithm Pipeline）：

<p align="center">
  <img src="./docs/pipeline.png" width="85%"/>
</p>

流程说明：
1. **多视角输入采集**：基于 FaceScape 参数渲染或真实双目采集；
2. **语义分割（SegFace）**：提取面部区域掩膜；
3. **视差估计（FoundationStereo）**：生成密集视差图；
4. **深度恢复与点云融合**：根据相机参数计算真实尺度深度；
6. **误差评估**：与 FaceScape 真值网格使用cloudcompare进行配准后精度验证。

---

## 📊 实验结果与精度评估

我们在 FaceScape 数据集的多个表情与人物下验证了系统性能，结果表明：

- 平均欧氏误差（Mean Error）：**0.39 mm**
- 均方根误差（RMS Error）：**0.46 mm**
- 95% 百分位误差（P95）：**0.63 mm**
- Hausdorff 距离（HD）：**1.02 mm**
- 重建区域一致性（IoU）：**>96%**

<p align="center">
  <img src="./docs/results_comparison.png" width="80%"/>
</p>

图示说明：左为 FoundationStereo 输出点云，右为真值 FaceScape 网格。可以看到，本系统在鼻翼、下颌等高曲率区域仍保持良好的几何精度。

---

## 🧭 通用性与可扩展性

得益于 FoundationStereo 的零样本立体匹配能力与 SegFace 的跨域语义掩膜优化，本项目具有极强的**通用性**：
- 无需针对特定人物或相机重新训练；
- 可直接部署于医疗扫描、文化遗产数字化、虚拟人建模等多领域；
- 仅需调整相机内外参，即可适配任意双目系统。

---

## 📚 参考文献

[1] H. Yang *et al.*, “FaceScape: A Large-Scale High Quality 3D Face Dataset and Detailed Riggable 3D Face Prediction,” *CVPR*, 2020.  
[2] H. Zhu *et al.*, “FaceScape: 3D Facial Dataset and Benchmark for Single-View 3D Face Reconstruction,” *IEEE TPAMI*, 2023.  
[3] K. Narayan *et al.*, “SegFace: Face Segmentation of Long-Tail Classes,” *AAAI*, 2025.  
[4] B. Wen *et al.*, “FoundationStereo: Zero-Shot Stereo Matching,” *CVPR*, 2025.

---

## 🧾 License
本项目仅供学术研究使用，禁止任何形式的商业用途。  
若需在其他项目中引用或扩展，请遵守各依赖项目的 License 条款。


# FaceScape 文件夹说明及 TU-Model 下载说明

[TOC]

该文件夹提供了 **FaceScape** 数据集的处理脚本和工具，包括 3D 面部渲染、点云评估以及表情和视角的批量处理。同时包含了 FaceScape 的 **TU-Model**（Topologically Uniformed Model）数据及下载方法。

---

## FaceScape TU-Model

### 数据概览
- 数据包含 847 个身份 × 20 个表情，总计 16940 个模型，约占完整数据的 90%。其余 10% 数据暂未公开，用于未来潜在评测或基准测试。
- 每个身份的模型包含：
  - 20 个基础网格模型：`/models_reg/$IDENTITY$_$EXPRESSION$.obj`
  - 20 个位移图：`/dpmap/$IDENTITY$_$EXPRESSION$.png`
  - 1 个基础材质文件：`/models_reg/$IDENTITY$_$EXPRESSION$.obj.mtl`
  - 1 张纹理图：`/models_reg/$IDENTITY$_$EXPRESSION$.jpg`
- 其中：
  - `$IDENTITY$`：身份索引（1 - 847）
  - `$EXPRESSION$`：表情索引（0 - 20）
  - 部分纹理（索引 360 - 847）对眼部进行马赛克处理以保护隐私。

### 特性
- **拓扑统一**：不同身份、不同表情的几何网格共享相同拓扑，便于特征对齐和 3D 可变形模型构建。
- **位移图 + 基础网格**：基础网格表示粗略几何形状，位移图表示细节几何，形成两层表示。基础网格可通过 MeshLab 可视化，位移图可在 MAYA、ZBrush、3D MAX 等软件中加载。
- **20 种特定表情**：中性、微笑、嘴巴拉伸、愤怒、下颌左/右/前移、嘴巴左/右、酒窝、下巴抬起、嘴唇嘟起、嘴唇漏斗、悲伤、嘴唇翻卷、咧嘴笑、鼓腮、闭眼、眉毛上抬/下压等。
- **高分辨率**：纹理图和位移图均为 4K 分辨率，保留最大细节。

### 附加信息
- **信息列表**：包含年龄和性别，格式为 `[索引, 性别(m/f), 年龄, 有效标签]`。`-` 表示信息缺失。有效标签为 [1+4] 二进制数：第一个表示模型完整性，剩余四位表示 obj/mtl/jpg/png 是否缺失。
- **可发表列表**：包含可用于论文或展示的模型索引。遵循许可条款第 4 条，列表可能更新。

### 下载方法
1. 前往官方 FaceScape 数据下载页面（通常需注册或填写申请表）。
2. 获取 TU-Model 数据压缩包（约数百 GB，包含 models_reg、dpmap 等文件夹）。
3. 解压到本地指定目录，例如：

---

## 文件夹结构
```

facescape/
│
├─ facescape_trainset_001_100/       # 原始 FaceScape 网格数据集
│   ├─ 001/
│   │   └─ models_reg/               # 已注册的 3D 面部网格 (.obj)，包含20种表情
│   ├─ 002/
│   │   └─ models_reg/
│   └─ ...
│
├─ output/                           # 默认渲染图片输出目录（可调整）
│   └─ <expression_name>/			 # 表情类型
│       └─ viewpointX/				 # 人物序号
│           ├─ 0.bmp                 # 左视图
│           ├─ 1.bmp                 # 右视图
│           └─ params.json           # 相机内参与外参
│
├─ renderer.py                       # 渲染相关函数
├─ evaluation.py          			 # 点云与网格误差评估函数
└─ process_TUModels.py               # 处理TUModels的主函数

```

---

## 脚本与模块功能

### 1. 网格渲染 (`renderer.py`)
提供多种渲染函数：
- **render_glcam()**：OpenGL 风格渲染，输出 RGB 图像和深度图。
- **render_cvcam()**：OpenCV 风格相机转换为 OpenGL 风格渲染，适用于 FaceScape。
- **render_orthcam()**：正射投影渲染，可指定 XY 放大倍数。

**特点**：
- 支持平面光照或标准光照
- 返回 BGR 格式图像，兼容 OpenCV
- 光源位置可自定义

---

### 2. FaceScape 网格多视角渲染 (`process_TUModels.py`)
- 遍历 `facescape_trainset_001_100` 下所有表情文件夹
- 加载 `models_reg` 下 `.obj` 网格
- 使用 `render_cvcam` 渲染左右视图
- 输出图片和相机参数至：
```

output_root/<expression_name>/viewpointX/

````

**相机参数示例**：
```python
K = np.array([[2000, 0, 256],
            [0, 2000, 256],
            [0, 0, 1]])
Rt_left = ...
Rt_right = ...
baseline = 40
image_size = (512, 512)
````

---

### 3. 点云与网格误差评估 (`evaluation.py`)

主要函数：

* `evaluate_cloud_to_mesh_distance(cloud_path, mesh_path, ...)`
  计算点到网格面的距离、RMS、均值、标准差、最大/最小误差、异常点比例、Hausdorff 距离、可选对称 Chamfer 距离。支持彩色点云、误差直方图、CSV 和 summary.txt 输出。
* `batch_evaluate_all_faces(mesh_root, cloud_root, output_root, ...)`
  批量评估所有表情和视角，可指定人物 ID 或视角范围。
* `summarize_evaluation_results(eval_root, output_excel)`
  汇总每个表情下所有人物的误差统计，输出 Excel 文件。

##**本文件夹文件使用流程**

1. 使用 `renderer.py` 渲染网格
2. 进行点云重建（外部过程）
3. 评估点云与原始网格误差
4. 汇总结果到 Excel

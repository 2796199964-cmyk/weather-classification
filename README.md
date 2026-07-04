# SkyWeather: 基于三通道 Sky Mask + ConvNeXt Cross-Attention 的天气图像分类

## 项目简介

### 比赛任务

给定一张室外场景图片，判断图片所展示的天气状况，输出四个类别之一：`sunny`（晴天）、`cloudy`（多云）、`rainy`（雨天）、`snowy`（雪天）。

### 方案概述

本项目为**天气图像四分类比赛**参赛方案，对输入图片预测天气类别：`sunny`（晴天）、`cloudy`（多云）、`rainy`（雨天）、`snowy`（雪天）。

核心思路：使用 SegFormer 语义分割提取精细的**三通道天空掩膜**，通过 ConvNeXt-Tiny 深层特征与 Mask 特征做空间 Cross-Attention 融合，最终分类。

**最终跑分稳定在 90% 左右。**

---

## 比赛经历

- **第一天**：查看训练图片集，发现部分图片底部会有媒体平台的标签信息，为防止模型把这个标签当作特征，把所有图片底部都截了。比赛系统特别卡，GPU 资源根本排不上，无法进行模型训练。
- **第二天**：系统更卡了，卡到连文件都上传不了、终端命令行也输不了，训练好的模型无法检验效果。下午赛事支持方升级了服务器，我们的工作才步入正轨，根据说明文档的接口编写推理程序，但服务器环境问题导致无法正常跑分。
- **第三天**：将 Python 环境安装打包后一起上传到服务器，在程序中引用本地环境，得以正常跑分，最终跑分稳定在 **90% 左右**。

---

## 目录结构与脚本说明

```
SkyWeather_upload/
├── 1截底.py                    # 训练 Step 1：图片底部裁剪预处理（去除水印/信息条）
├── 2sky云分割.py                # 训练 Step 2：SegFormer 三通道天空语义分割
├── 3ConvNeXtCrossMask2.py       # 训练 Step 3：ConvNeXt + Cross-Attention 模型训练
├── main.py                      # 推理算法模块：加载模型 + 在线分割 + 天气预测
├── predict.py                   # 模拟比赛系统：批量调用 main.predict 评估
├── main.ipynb                   # Jupyter Notebook 实验记录
├── 图标.jpg                     # 天气图标素材
├── requirements.txt             # Python 依赖列表
└── README.md                    # 本文件
```

| 脚本 | 类型 | 说明 |
|------|------|------|
| `1截底.py` | 训练程序 | 将所有图片底部 24 像素涂白，去除媒体平台水印（控制变量，统一处理） |
| `2sky云分割.py` | 训练程序 | SegFormer-B1 语义分割，生成三通道 Mask（clouds+fog / sky-other / 地面） |
| `3ConvNeXtCrossMask2.py` | 训练程序 | ConvNeXt-Tiny + 三通道 MaskEncoder + Spatial Cross-Attention 训练 |
| `main.py` | 推理模块 | 核心算法，提供 `predict(X)` 接口，比赛跑分系统直接调用 |
| `predict.py` | 模拟评测 | 本地模拟比赛系统，批量传图给 `main.py` 并统计准确率 |

---

## 技术路线

### 训练流水线

```
原始图片 ──► 1截底.py ──► 2sky云分割.py ──► 3ConvNeXtCrossMask2.py
                │                │                     │
         去除底部水印      三通道天空分割           模型训练
         输出: 1cut/       输出: sky/             输出: .pth 权重
```

### 推理流程

```
比赛系统传入图片
        │
   main.predict(X)
        │
   ┌────┴────┐
   │         │
 SegFormer   ConvNeXt-Tiny
 在线分割     图像特征
   │         │
 三通道Mask   RGB特征 (768,7,7)
   │         │
 MaskEncoder  │
 (3→768)     │
   │         │
   └────┬────┘
   Spatial Cross-Attention
        │
   Classifier → sunny / cloudy / rainy / snowy
```

### 三通道 Mask

| 通道 | 含义 | 像素值 |
|------|------|--------|
| 0 | Clouds + Fog（云层/雾气） | 200 |
| 1 | Sky-Other（晴空，需贴顶过滤） | 255 |
| 2 | Ground / Other（地面/其它） | 128 |

---

## 推理接口

`main.py` 提供标准预测接口，比赛系统直接调用：

```python
import cv2
from main import predict

# X: np.ndarray, 由 cv2.imread 读取的图片, shape (H, W, 3)
img = cv2.imread("test.jpg")

# 返回: str, 取值为 'sunny' / 'cloudy' / 'rainy' / 'snowy'
result = predict(img)
```

---

## 模型架构

**ConvNeXtCrossMask**：RGB 图像经 ConvNeXt-Tiny backbone 提取深层特征，三通道 Mask 经 MaskEncoder 编码到相同维度，两者通过 Spatial Cross-Attention 融合后分类。

**SegFormer-B1**：基于 COCO-Stuff 164K 预训练的 172 类语义分割模型，用于在线提取天空/云层/地面区域。

---

## 服务器部署

比赛服务器环境不完整，解决方案：将本地 Python 环境（含 PyTorch、torchvision、albumentations 等依赖）打包后上传，在 `main.py` 中注入本地包路径：

```python
_LOCAL_PACKAGES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs", "local_packages")
if os.path.isdir(_LOCAL_PACKAGES) and _LOCAL_PACKAGES not in sys.path:
    sys.path.insert(0, _LOCAL_PACKAGES)
```

设备自动检测：CUDA → DirectML → CPU，兼容不同服务器环境。

---

## 依赖环境

```
torch>=2.0.0
torchvision>=0.15.0
opencv-python>=4.7.0
albumentations>=1.3.0
numpy>=1.24.0
```

---

## 预训练模型

本项目需要以下预训练模型（需提前下载放到指定路径）：

### 1. SegFormer-B1 语义分割模型（训练 + 推理均需要）

| 项目 | 说明 |
|------|------|
| 模型 | SegFormer-B1, COCO-Stuff 164K, 172 类语义分割 |
| 来源 | [ModelScope: damo/cv_segformer-b1](https://www.modelscope.cn/models/damo/cv_segformer-b1_image_semantic-segmentation_coco-stuff164k/summary) |
| 文件 | `pytorch_model.pt`（约 157 MB） |
| 放置路径 | `runs/segformer-b1/pytorch_model.pt` |

用途：
- `2sky云分割.py`（训练时生成三通道 Mask）
- `main.py`（推理时在线生成三通道 Mask）

下载方式：
```bash
# 方式1: ModelScope SDK
pip install modelscope
from modelscope import snapshot_download
snapshot_download('damo/cv_segformer-b1_image_semantic-segmentation_coco-stuff164k',
                  cache_dir='runs/segformer-b1')

# 方式2: 手动从 ModelScope 页面下载 pytorch_model.pt
```

### 2. ConvNeXt-Tiny ImageNet 预训练权重（仅训练时需要）

| 项目 | 说明 |
|------|------|
| 模型 | ConvNeXt-Tiny, ImageNet-1K 分类 |
| 来源 | torchvision 内置 (`models.ConvNeXt_Tiny_Weights.DEFAULT`) |
| 文件 | 首次运行自动下载（约 110 MB），缓存到 `~/.cache/torch/hub/` |

用途：
- `3ConvNeXtCrossMask2.py` 训练时自动加载 ImageNet 预训练权重

### 3. 训练好的天气分类模型（推理时需要）

| 项目 | 说明 |
|------|------|
| 文件 | `ConvNeXtCrossMask2_best_*.pth`（约 216 MB） |
| 放置路径 | `runs/convnext_crossmask2/` |

该模型由 `3ConvNeXtCrossMask2.py` 训练生成，推理时 `main.py` 自动扫描该目录加载最新权重。若需从零训练，无需提前准备此文件。

---

## 引用

- [SegFormer](https://arxiv.org/abs/2105.15203) (NeurIPS 2021)
- [ConvNeXt](https://arxiv.org/abs/2201.03545) (CVPR 2022)
- [COCO-Stuff 164K](https://arxiv.org/abs/2104.10900) (CVPR 2022)

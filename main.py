import sys
import os
#三通道sky_mask
# ================= 0. 本地包路径注入（必须在最顶部）=================
_LOCAL_PACKAGES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs", "local_packages")
if os.path.isdir(_LOCAL_PACKAGES) and _LOCAL_PACKAGES not in sys.path:
    print(r"本地包1")
    sys.path.insert(0, _LOCAL_PACKAGES)

import pickle
import io
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import albumentations as A
from albumentations.pytorch import ToTensorV2

# ================= 2. 路径与基础配置 =================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = r"runs/convnext_crossmask"
SEGFORMER_WEIGHT_PATH = r"runs/segformer-b1/pytorch_model.pt"
IMG_SIZE = 224
BOTTOM_PIXELS = 24

# ================= 3. 类别与标签映射 =================
WEATHER_CLASSES_EN = ["sunny", "cloudy", "rainy", "snowy"]
WEATHER_CLASSES_CN = ["晴天", "多云", "雨天", "雪天"]
CLASS_TO_IDX = {name: i for i, name in enumerate(WEATHER_CLASSES_EN)}
IDX_TO_CLASS = {i: name for name, i in CLASS_TO_IDX.items()}
WEATHER_NUM_CLASSES = len(WEATHER_CLASSES_EN)
OUTPUT_MAPPING = {"sunny": "sunny", "cloudy": "cloudy", "rainy": "rainy", "snow": "snowy"}

# ================= 4. 图像预处理参数 =================
RGB_MEAN = [0.485, 0.456, 0.406]
RGB_STD = [0.229, 0.224, 0.225]
SEGFORMER_MEAN = np.array([123.675, 116.28, 103.53])
SEGFORMER_STD = np.array([58.395, 57.12, 57.375])

# ================= 5. SegFormer 模型超参数 =================
SEGFORMER_NUM_CLASSES = 172
BACKBONE_EMBED_DIMS = [64, 128, 320, 512]
BACKBONE_NUM_HEADS = [1, 2, 5, 8]
BACKBONE_MLP_RATIOS = [4, 4, 4, 4]
BACKBONE_DEPTHS = [2, 2, 2, 2]
BACKBONE_SR_RATIOS = [8, 4, 2, 1]
SEGFORMER_DROP_RATE = 0.0
SEGFORMER_DROP_PATH_RATE = 0.1
DECODE_HEAD_HIDDEN_DIM = 256
DECODE_HEAD_DROPOUT = 0.1
KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

COCO_CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
    'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
    'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
    'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
    'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
    'teddy bear', 'hair drier', 'toothbrush', 'banner', 'blanket', 'branch', 'bridge', 'building-other',
    'bush', 'cabinet', 'cage', 'cardboard', 'carpet', 'ceiling-other', 'ceiling-tile', 'cloth', 'clothes',
    'clouds', 'counter', 'cupboard', 'curtain', 'desk-stuff', 'dirt', 'door-stuff', 'fence', 'floor-marble',
    'floor-other', 'floor-stone', 'floor-tile', 'floor-wood', 'flower', 'fog', 'food-other', 'fruit',
    'furniture-other', 'grass', 'gravel', 'ground-other', 'hill', 'house', 'leaves', 'light', 'mat', 'metal',
    'mirror-stuff', 'moss', 'mountain', 'mud', 'napkin', 'net', 'paper', 'pavement', 'pillow', 'plant-other',
    'plastic', 'platform', 'playingfield', 'railing', 'railroad', 'river', 'road', 'rock', 'roof', 'rug',
    'salad', 'sand', 'sea', 'shelf', 'sky-other', 'skyscraper', 'snow', 'solid-other', 'stairs', 'stone',
    'straw', 'structural-other', 'table', 'tent', 'textile-other', 'towel', 'tree', 'vegetable', 'wall-brick',
    'wall-concrete', 'wall-other', 'wall-panel', 'wall-stone', 'wall-tile', 'wall-wood', 'water-other',
    'waterdrops', 'window-blind', 'window-other', 'wood'
]
COCO_IDX = {name: i for i, name in enumerate(COCO_CLASSES)}
CLOUDS_ID = COCO_IDX.get('clouds', 77)
FOG_ID = COCO_IDX.get('fog', 85)
SKY_OTHER_ID = COCO_IDX.get('sky-other', 135)

# ================= 6. ConvNeXt + CrossMask 模型超参数 =================
MASK_ENCODER_OUT_CHANNELS = 768
CROSS_ATTN_CHANNELS = 768
CROSS_ATTN_NUM_HEADS = 8
CROSS_ATTN_DROPOUT = 0.1
CLASSIFIER_DROPOUT = 0.25


# ================= 辅助函数 =================
def safe_imread_gray(path):
    if path is None or not os.path.exists(path): return None
    try:
        arr = np.fromfile(path, dtype=np.uint8)
        return cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
    except Exception:
        return None


def keep_boundary_connected(mask):
    """保留与图像任意边界连通的区域"""
    if not np.any(mask): return mask.astype(bool)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    valid_labels = set()
    for lbl in range(1, num_labels):
        lbl_mask = (labels == lbl)
        if (np.any(lbl_mask[0, :]) or np.any(lbl_mask[-1, :]) or np.any(lbl_mask[:, 0]) or np.any(lbl_mask[:, -1])):
            valid_labels.add(lbl)
    return np.isin(labels, list(valid_labels)) if valid_labels else np.zeros_like(mask, dtype=bool)


# ================= SegFormer 模型结构 =================
class OverlapPatchEmbed(nn.Module):
    def __init__(self, patch_size=7, stride=4, in_chans=3, embed_dim=64):
        super().__init__()
        self.projection = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride, padding=patch_size // 2)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.projection(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x, H, W


class EfficientAttention(nn.Module):
    def __init__(self, dim, num_heads=8, sr_ratio=1):
        super().__init__()
        self.dim, self.num_heads, self.head_dim = dim, num_heads, dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.sr_ratio = sr_ratio
        self.q = nn.Linear(dim, dim, bias=True)
        self.kv = nn.Linear(dim, dim * 2, bias=True)
        self.proj = nn.Linear(dim, dim, bias=True)
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        if self.sr_ratio > 1:
            x_kv = x.permute(0, 2, 1).reshape(B, C, H, W)
            x_kv = self.sr(x_kv).reshape(B, C, -1).permute(0, 2, 1)
            x_kv = self.norm(x_kv)
        else:
            x_kv = x
        kv = self.kv(x_kv).reshape(B, -1, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj(out)


class FFN(nn.Module):
    def __init__(self, in_features, hidden_features=None, drop=0.):
        super().__init__()
        hidden_features = hidden_features or in_features
        self.layers = nn.ModuleList([
            nn.Conv2d(in_features, hidden_features, 1),
            nn.Conv2d(hidden_features, hidden_features, 3, 1, 1, groups=hidden_features),
            nn.GELU(), nn.Dropout(drop), nn.Conv2d(hidden_features, in_features, 1), nn.Dropout(drop),
        ])

    def forward(self, x, H, W):
        x = x.transpose(1, 2).reshape(-1, x.shape[-1], H, W)
        for layer in self.layers: x = layer(x)
        return x.flatten(2).transpose(1, 2)


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., sr_ratio=1, drop=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = EfficientAttention(dim, num_heads=num_heads, sr_ratio=sr_ratio)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = FFN(dim, hidden_features=int(dim * mlp_ratio), drop=drop)

    def forward(self, x, H, W):
        x = x + self.attn(self.norm1(x), H, W)
        x = x + self.ffn(self.norm2(x), H, W)
        return x


class MixVisionTransformer(nn.Module):
    def __init__(self, in_chans=3, embed_dims=BACKBONE_EMBED_DIMS, num_heads=BACKBONE_NUM_HEADS,
                 mlp_ratios=BACKBONE_MLP_RATIOS, drop_rate=SEGFORMER_DROP_RATE, drop_path_rate=SEGFORMER_DROP_PATH_RATE,
                 depths=BACKBONE_DEPTHS, sr_ratios=BACKBONE_SR_RATIOS, num_stages=4):
        super().__init__()
        self.num_stages = num_stages
        self.layers = nn.ModuleList()
        for i in range(num_stages):
            patch_embed = OverlapPatchEmbed(
                patch_size=7 if i == 0 else 3, stride=4 if i == 0 else 2,
                in_chans=in_chans if i == 0 else embed_dims[i - 1], embed_dim=embed_dims[i]
            )
            blocks = nn.ModuleList([
                Block(dim=embed_dims[i], num_heads=num_heads[i], mlp_ratio=mlp_ratios[i],
                      sr_ratio=sr_ratios[i], drop=drop_rate) for _ in range(depths[i])
            ])
            norm = nn.LayerNorm(embed_dims[i])
            self.layers.append(nn.ModuleList([patch_embed, blocks, norm]))

    def forward(self, x):
        outs = []
        for i in range(self.num_stages):
            patch_embed, blocks, norm = self.layers[i]
            x, H, W = patch_embed(x)
            for block in blocks: x = block(x, H, W)
            x = norm(x)
            x = x.reshape(-1, H, W, x.shape[-1]).permute(0, 3, 1, 2).contiguous()
            outs.append(x)
        return outs


class SegformerHead(nn.Module):
    def __init__(self, in_channels=BACKBONE_EMBED_DIMS, num_classes=SEGFORMER_NUM_CLASSES,
                 hidden_dim=DECODE_HEAD_HIDDEN_DIM):
        super().__init__()
        self.convs = nn.ModuleList()
        for in_ch in in_channels:
            self.convs.append(nn.Sequential(
                nn.Conv2d(in_ch, hidden_dim, 1, bias=False), nn.BatchNorm2d(hidden_dim), nn.ReLU(inplace=True)
            ))
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(len(in_channels) * hidden_dim, hidden_dim, 1, bias=False), nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        self.dropout = nn.Dropout(DECODE_HEAD_DROPOUT)
        self.conv_seg = nn.Conv2d(hidden_dim, num_classes, 1)

    def forward(self, x):
        c1, c2, c3, c4 = x
        size = c1.size()[2:]
        outs = []
        for i, feat in enumerate([c1, c2, c3, c4]):
            out = self.convs[i](feat)
            if out.size()[2:] != size:
                out = F.interpolate(out, size=size, mode='bilinear', align_corners=False)
            outs.append(out)
        out = torch.cat(outs, dim=1)
        out = self.fusion_conv(out)
        out = self.dropout(out)
        return self.conv_seg(out)


class Segformer(nn.Module):
    def __init__(self, num_classes=SEGFORMER_NUM_CLASSES):
        super().__init__()
        self.backbone = MixVisionTransformer()
        self.decode_head = SegformerHead(num_classes=num_classes)

    def forward(self, x):
        return self.decode_head(self.backbone(x))


# ================= 天气分类模型结构 =================
class MaskEncoder(nn.Module):
    def __init__(self, out_channels=MASK_ENCODER_OUT_CHANNELS):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 64, 7, 4, 3, bias=False), nn.BatchNorm2d(64), nn.GELU(),
            nn.Conv2d(64, 128, 3, 2, 1, bias=False), nn.BatchNorm2d(128), nn.GELU(),
            nn.Conv2d(128, 256, 3, 2, 1, bias=False), nn.BatchNorm2d(256), nn.GELU(),
            nn.Conv2d(256, 512, 3, 2, 1, bias=False), nn.BatchNorm2d(512), nn.GELU(),
            nn.Conv2d(512, out_channels, 3, 2, 1, bias=False), nn.BatchNorm2d(out_channels), nn.GELU(),
        )

    def forward(self, mask): return self.net(mask)


class SpatialCrossAttention(nn.Module):
    def __init__(self, channels=CROSS_ATTN_CHANNELS, num_heads=CROSS_ATTN_NUM_HEADS, dropout=CROSS_ATTN_DROPOUT):
        super().__init__()
        self.norm_q = nn.LayerNorm(channels)
        self.norm_kv = nn.LayerNorm(channels)
        self.attn = nn.MultiheadAttention(channels, num_heads, dropout=dropout, batch_first=True)
        self.gamma = nn.Parameter(torch.tensor(0.1))
        self.ffn = nn.Sequential(nn.LayerNorm(channels), nn.Linear(channels, channels * 2), nn.GELU(),
                                 nn.Dropout(dropout), nn.Linear(channels * 2, channels))

    def forward(self, rgb_feat, mask_feat):
        b, c, h, w = rgb_feat.shape
        q = rgb_feat.flatten(2).transpose(1, 2)
        kv = mask_feat.flatten(2).transpose(1, 2)
        attn_out, _ = self.attn(self.norm_q(q), self.norm_kv(kv), self.norm_kv(kv), need_weights=False)
        x = q + self.gamma * attn_out
        x = x + self.gamma * self.ffn(x)
        return x.transpose(1, 2).reshape(b, c, h, w)


class ConvNeXtCrossMask(nn.Module):
    def __init__(self, num_classes=WEATHER_NUM_CLASSES):
        super().__init__()
        backbone = models.convnext_tiny(weights=None)
        self.rgb_features = backbone.features
        self.mask_encoder = MaskEncoder()
        self.cross_attn = SpatialCrossAttention()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.LayerNorm(CROSS_ATTN_CHANNELS),
            nn.Dropout(CLASSIFIER_DROPOUT),
            nn.Linear(CROSS_ATTN_CHANNELS, num_classes)
        )

    def forward(self, image, mask, ratio=None):
        rgb_feat = self.rgb_features(image)
        mask_feat = self.mask_encoder(mask)
        if mask_feat.shape[2:] != rgb_feat.shape[2:]:
            mask_feat = F.interpolate(mask_feat, size=rgb_feat.shape[2:], mode="bilinear", align_corners=False)
        fused = self.cross_attn(rgb_feat, mask_feat)
        feat = self.pool(fused).flatten(1)
        logits = self.classifier(feat)
        return logits, feat


# ================= 设备检测 =================
def get_infer_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    try:
        import torch_directml
        if torch_directml.is_available():
            return torch_directml.device()
    except ImportError:
        pass
    return torch.device("cpu")


# ================= 核心推理类 =================
class WeatherClassifier:
    def __init__(self, model_dir=MODEL_DIR, img_size=IMG_SIZE):
        self.img_size = img_size
        self.device = get_infer_device()
        print(f"[INFO] 推理设备: {self.device}")

        self.model = self._load_model(model_dir)
        self.segformer = self._load_segformer()

        self.transform = A.Compose([
            A.Normalize(mean=RGB_MEAN, std=RGB_STD),
            ToTensorV2(),
        ], additional_targets={"mask": "image"})

    def _load_model(self, model_dir):
        md = os.path.join(SCRIPT_DIR, model_dir)
        if not os.path.isdir(md):
            raise FileNotFoundError(f"❌ 模型目录不存在: {md}")
        print(f"[INFO] 正在扫描天气分类模型目录: {md}")
        cands = [os.path.join(md, f) for f in os.listdir(md) if f.lower().endswith(".pth")]
        if not cands:
            raise FileNotFoundError(f"未找到任何 .pth 模型文件: {md}")
        model_path = max(cands, key=os.path.getmtime)
        print(f"[INFO] ✅ 成功加载天气分类模型: {os.path.basename(model_path)}")

        # 👇 核心修复：Monkey-patch PyTorch 底层张量重建函数，强制映射到 CPU
        import torch._utils

        # 备份原始函数
        _orig_rebuild_tensor_v2 = getattr(torch._utils, '_rebuild_tensor_v2', None)
        _orig_rebuild_device_tensor = getattr(torch._utils, '_rebuild_device_tensor_from_numpy', None)

        def _patched_rebuild_tensor_v2(storage, storage_offset, size, stride, requires_grad, backward_hooks):
            # 强制将 storage 转换为 CPU storage
            if hasattr(storage, '_cdata') and storage.device.type != 'cpu':
                # 如果 storage 不是 CPU，尝试通过 numpy 转换（兜底）
                pass
                # 实际上 torch.load(map_location='cpu') 已经处理了 storage 的映射，
            # 这里主要是为了防止 rebuild 阶段调用不支持的后端算子。
            tensor = torch.empty(0).set_(storage, storage_offset, size, stride)
            if requires_grad:
                tensor.requires_grad_()
            return tensor

        def _patched_rebuild_device_tensor_from_numpy(data, dtype, device, requires_grad=False):
            # 忽略原始的 device (可能是 PrivateUse1)，强制使用 CPU
            tensor = torch.from_numpy(data).to(dtype=dtype, device='cpu')
            if requires_grad:
                tensor.requires_grad_()
            return tensor

        try:
            # 注入补丁
            if _orig_rebuild_tensor_v2:
                torch._utils._rebuild_tensor_v2 = _patched_rebuild_tensor_v2
            if _orig_rebuild_device_tensor:
                torch._utils._rebuild_device_tensor_from_numpy = _patched_rebuild_device_tensor_from_numpy

            # 使用安全的 weights_only=False 和 map_location='cpu'
            ckpt = torch.load(model_path, map_location='cpu', weights_only=False)

        except Exception as e:
            print(f"[WARN] ⚠️ 标准加载失败 ({type(e).__name__})，尝试终极兼容模式...")

            # 终极兜底：如果 monkey-patch 仍然失败，使用自定义 Unpickler 配合 persistent_load
            class CpuUnpickler(pickle.Unpickler):
                def find_class(self, module, name):
                    if module == 'torch._utils' and name == '_rebuild_device_tensor_from_numpy':
                        return _patched_rebuild_device_tensor_from_numpy
                    return super().find_class(module, name)

                def persistent_load(self, pid):
                    # 处理 PyTorch 的 persistent_id (通常是 tensor storage)
                    # 这里我们简单地返回一个空的 CPU tensor 或让 torch 内部处理
                    # 实际上 torch.load 内部有自己的 persistent_load，我们很难完全复刻
                    # 所以这个兜底方案主要用于旧版 PyTorch 格式
                    raise pickle.UnpicklingError(f"Unsupported persistent_id: {pid}")

            with open(model_path, 'rb') as f:
                ckpt = CpuUnpickler(f).load()
        finally:
            # 恢复原始函数
            if _orig_rebuild_tensor_v2:
                torch._utils._rebuild_tensor_v2 = _orig_rebuild_tensor_v2
            if _orig_rebuild_device_tensor:
                torch._utils._rebuild_device_tensor_from_numpy = _orig_rebuild_device_tensor

        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            state_dict = ckpt["model_state_dict"]
        elif hasattr(ckpt, 'state_dict'):
            state_dict = ckpt.state_dict()
        else:
            state_dict = ckpt

        model = ConvNeXtCrossMask().to(self.device)
        model.load_state_dict(state_dict, strict=True)
        model.eval()
        return model

    def _load_segformer(self):
        if not os.path.exists(SEGFORMER_WEIGHT_PATH):
            print(f"[WARN] ⚠️ 未找到 SegFormer 权重: {SEGFORMER_WEIGHT_PATH}，将禁用实时云分割！")
            return None
        print(f"[INFO] 正在加载 SegFormer 云分割模型...")
        model = Segformer()

        checkpoint = torch.load(SEGFORMER_WEIGHT_PATH, map_location='cpu', weights_only=False)

        state_dict = checkpoint.get('state_dict', checkpoint)
        mapped_state_dict, attn_splits = {}, {}
        for k, v in state_dict.items():
            new_k = k
            if 'decode_head.convs.' in k or 'decode_head.fusion_conv.' in k:
                if '.conv.' in k or k.endswith('.conv'):
                    new_k = k.replace('.conv.', '.0.')
                elif '.bn.' in k or k.endswith('.bn'):
                    new_k = k.replace('.bn.', '.1.')
            elif '.attn.attn.out_proj.' in k:
                new_k = k.replace('.attn.attn.out_proj.', '.attn.proj.')
            elif '.attn.attn.in_proj_' in k:
                attn_splits[k] = v
                continue
            mapped_state_dict[new_k] = v
        for k, v in attn_splits.items():
            prefix = k.replace('.attn.attn.in_proj_weight', '.attn').replace('.attn.attn.in_proj_bias', '.attn')
            dim = v.shape[0] // 3
            if 'weight' in k:
                mapped_state_dict[f"{prefix}.q.weight"] = v[0:dim]
                mapped_state_dict[f"{prefix}.kv.weight"] = torch.cat([v[dim:2 * dim], v[2 * dim:3 * dim]], dim=0)
            else:
                mapped_state_dict[f"{prefix}.q.bias"] = v[0:dim]
                mapped_state_dict[f"{prefix}.kv.bias"] = torch.cat([v[dim:2 * dim], v[2 * dim:3 * dim]], dim=0)
        model.load_state_dict(mapped_state_dict, strict=False)
        model = model.to(self.device)
        model.eval()
        print(f"[INFO] ✅ SegFormer 云分割模型加载成功！")
        return model

    def _generate_mask_online(self, img_bgr_cropped):
        """在线生成三通道 Mask，与 2sky云分割.py 的三通道标记一致"""
        if self.segformer is None:
            return np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)
        img_rgb = cv2.cvtColor(img_bgr_cropped, cv2.COLOR_BGR2RGB).astype(np.float32)
        img_norm = (img_rgb - SEGFORMER_MEAN) / SEGFORMER_STD
        img_tensor = torch.from_numpy(img_norm.transpose(2, 0, 1)).unsqueeze(0).float().to(self.device)
        with torch.no_grad():
            logits = self.segformer(img_tensor)
            if logits.shape[-2:] != (self.img_size, self.img_size):
                logits = F.interpolate(logits, size=(self.img_size, self.img_size), mode='bilinear',
                                       align_corners=False)
            pred_class = logits.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)

        h, w = pred_class.shape
        # 三通道 Mask
        mask_out = np.zeros((h, w, 3), dtype=np.uint8)

        # 通道0: clouds + fog (mask 像素值 200)
        clouds_fog = (pred_class == CLOUDS_ID) | (pred_class == FOG_ID)
        clouds_fog = cv2.morphologyEx(clouds_fog.astype(np.uint8), cv2.MORPH_CLOSE, KERNEL).astype(bool)

        # 通道1: sky-other (mask 像素值 255)
        sky_other = keep_boundary_connected((pred_class == SKY_OTHER_ID))
        sky_other = cv2.morphologyEx(sky_other.astype(np.uint8), cv2.MORPH_CLOSE, KERNEL).astype(bool)

        # 底部 BOTTOM_PIXELS 像素强制归为地面
        if h > BOTTOM_PIXELS:
            clouds_fog[-BOTTOM_PIXELS:, :] = False
            sky_other[-BOTTOM_PIXELS:, :] = False

        mask_out[clouds_fog, 0] = 255
        mask_out[sky_other, 1] = 255

        # 通道2: 地面 / 其它 (mask 像素值 128)
        sky_mask = clouds_fog | sky_other
        ground_mask = ~sky_mask
        mask_out[ground_mask, 2] = 255

        return mask_out

    def infer(self, X):
        """核心推理逻辑，接收 224x224 的 BGR 图像"""
        img_bgr = X.copy()
        img_h, img_w = img_bgr.shape[:2]
        if img_h > BOTTOM_PIXELS:
            img_bgr[img_h - BOTTOM_PIXELS: img_h, :] = [255, 255, 255]

        mask_3ch = self._generate_mask_online(img_bgr)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        aug = self.transform(image=img_rgb, mask=mask_3ch)
        img_tensor = aug["image"].unsqueeze(0).to(self.device)

        mask_tensor = aug["mask"]
        if not torch.is_tensor(mask_tensor):
            mask_tensor = torch.from_numpy(mask_tensor)
        # Albumentations ToTensorV2: (H,W,3) -> (3,H,W)
        if mask_tensor.ndim == 3 and mask_tensor.shape[-1] == 3:
            mask_tensor = mask_tensor.permute(2, 0, 1)
        elif mask_tensor.ndim == 2:
            mask_tensor = mask_tensor.unsqueeze(0).repeat(3, 1, 1)
        # 添加 batch 维度 -> (B, 3, H, W)
        mask_tensor = mask_tensor.unsqueeze(0).float().to(self.device)
        if mask_tensor.max() > 1.0:
            mask_tensor = mask_tensor / 255.0
        # 三通道各自的面积比例
        ratio = mask_tensor.mean(dim=(2, 3)).float()

        with torch.no_grad():
            logits, _ = self.model(img_tensor, mask_tensor, ratio)
            probs = torch.softmax(logits, dim=1)

        pred_idx = probs.argmax(dim=1).item()
        return WEATHER_CLASSES_EN[pred_idx]


# ================= 全局单例与标准接口 =================
_classifier_instance = None


def _get_classifier():
    """延迟加载单例模型，避免每次调用 predict 都重新加载权重"""
    global _classifier_instance
    if _classifier_instance is None:
        print("🚀 初始化推理引擎...")
        _classifier_instance = WeatherClassifier()
    return _classifier_instance


def predict(X):
    """
    模型预测
    param：
        X : np.ndarray，由 cv2.imread 读取的图片数据，shape(224,224,3)。
    return：
        y_predict : str, 数据 label，取值为 'sunny', 'cloudy', 'rainy', 'snowy' 之一。
    """
    classifier = _get_classifier()
    raw_pred = classifier.infer(X)
    y_predict = OUTPUT_MAPPING.get(raw_pred, raw_pred)
    return y_predict
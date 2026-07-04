"""
模型: sky/segformer-b1
Mask 标记:
    clouds + fog = 200
    sky-other    = 255
    地面/其它    = 128
说明:
    不再标记 snow；训练脚本可通过 mask > 200 或 mask == 255 提取不同天空区域。
"""
import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from concurrent.futures import ThreadPoolExecutor
import time
import glob

# ========== 配置 ==========
WEIGHT_PATH = r"runs/segformer-b1/pytorch_model.pt"  # 后续会基于 SCRIPT_DIR 解析为绝对路径
# SegFormer 模型配置
# B1: embed_dims=[64,128,320,512], B0: [32,64,160,256]
BACKBONE_EMBED_DIMS = [64, 128, 320, 512]
BACKBONE_NUM_HEADS = [1, 2, 5, 8]
# 脚本所在目录，用于解析相对路径
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# 输入根目录（递归扫描所有子文件夹）
INPUT_ROOTS = [os.path.join(SCRIPT_DIR, "1cut")]
IMG_SIZE = (224, 224)
# 输出根目录，每个类别会生成对应的子目录
OUTPUT_ROOT_BASE = os.path.join(SCRIPT_DIR, "sky")

BATCH_SIZE = 32      # batch 过大（128）会死机，核显显存有限
MAX_WORKERS = 8      # 后处理线程数

# 限制 PyTorch 自身 CPU 线程数
torch.set_num_threads(8)

# 形态学卷积核
KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

# ========== 设备配置 ==========
USE_CUDA = torch.cuda.is_available()
USE_DML = False
if not USE_CUDA:
    try:
        import torch_directml
        if torch_directml.is_available():
            USE_DML = True
    except ImportError:
        pass

if USE_CUDA:
    DEVICE = torch.device("cuda")
    print(f"[INFO] 使用 CUDA: {torch.cuda.get_device_name(0)}")
elif USE_DML:
    import torch_directml
    DEVICE = torch_directml.device()
    print(f"[INFO] 使用 DirectML: {torch_directml.device_name(0)}")
else:
    DEVICE = torch.device("cpu")
    print("[INFO] 使用 CPU")


# ============================== 类别定义 ==============================
CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train',
    'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag',
    'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite',
    'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
    'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon',
    'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
    'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant',
    'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
    'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
    'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
    'hair drier', 'toothbrush', 'banner', 'blanket', 'branch', 'bridge',
    'building-other', 'bush', 'cabinet', 'cage', 'cardboard', 'carpet',
    'ceiling-other', 'ceiling-tile', 'cloth', 'clothes', 'clouds', 'counter',
    'cupboard', 'curtain', 'desk-stuff', 'dirt', 'door-stuff', 'fence',
    'floor-marble', 'floor-other', 'floor-stone', 'floor-tile', 'floor-wood',
    'flower', 'fog', 'food-other', 'fruit', 'furniture-other', 'grass',
    'gravel', 'ground-other', 'hill', 'house', 'leaves', 'light', 'mat',
    'metal', 'mirror-stuff', 'moss', 'mountain', 'mud', 'napkin', 'net',
    'paper', 'pavement', 'pillow', 'plant-other', 'plastic', 'platform',
    'playingfield', 'railing', 'railroad', 'river', 'road', 'rock', 'roof',
    'rug', 'salad', 'sand', 'sea', 'shelf', 'sky-other', 'skyscraper', 'snow',
    'solid-other', 'stairs', 'stone', 'straw', 'structural-other', 'table',
    'tent', 'textile-other', 'towel', 'tree', 'vegetable', 'wall-brick',
    'wall-concrete', 'wall-other', 'wall-panel', 'wall-stone', 'wall-tile',
    'wall-wood', 'water-other', 'waterdrops', 'window-blind', 'window-other',
    'wood'
]

CLOUDS_NAME = 'clouds'
FOG_NAME = 'fog'
SKY_OTHER_NAME = 'sky-other'

CLASS_TO_IDX = {name: i for i, name in enumerate(CLASSES)}
CLOUDS_ID = CLASS_TO_IDX.get(CLOUDS_NAME, -1)
FOG_ID = CLASS_TO_IDX.get(FOG_NAME, -1)
SKY_OTHER_ID = CLASS_TO_IDX.get(SKY_OTHER_NAME, -1)


# ============================== SegFormer-B0 架构 (匹配 EasyCV 权重) ==============================
class OverlapPatchEmbed(nn.Module):
    """重叠 Patch 嵌入，权重 key: projection/norm"""
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
    """高效注意力 (Spatial Reduction Attention)，手动实现以避免 DML fallback 到 CPU"""
    def __init__(self, dim, num_heads=8, sr_ratio=1):
        super().__init__()
        assert dim % num_heads == 0
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.sr_ratio = sr_ratio

        # 手动实现 q/kv/proj
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
        out = self.proj(out)
        return out


class FFN(nn.Module):
    """前馈网络，权重 key: layers.0/1/4 (EasyCV 用 1x1 Conv2d 代替 Linear)"""
    def __init__(self, in_features, hidden_features=None, drop=0.):
        super().__init__()
        hidden_features = hidden_features or in_features
        self.layers = nn.ModuleList([
            nn.Conv2d(in_features, hidden_features, 1),  # 0: 1x1 conv 等价于 linear
            nn.Conv2d(hidden_features, hidden_features, 3, 1, 1, groups=hidden_features),  # 1: DWConv
            nn.GELU(),  # 2
            nn.Dropout(drop),  # 3
            nn.Conv2d(hidden_features, in_features, 1),  # 4: 1x1 conv 等价于 linear
            nn.Dropout(drop),  # 5
        ])

    def forward(self, x, H, W):
        # x: (B, N, C) -> (B, C, H, W)
        x = x.transpose(1, 2).reshape(-1, x.shape[-1], H, W)
        x = self.layers[0](x)
        x = self.layers[1](x)
        x = self.layers[2](x)
        x = self.layers[3](x)
        x = self.layers[4](x)
        x = self.layers[5](x)
        x = x.flatten(2).transpose(1, 2)
        return x


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., sr_ratio=1, drop=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = EfficientAttention(dim, num_heads=num_heads, sr_ratio=sr_ratio)
        self.norm2 = nn.LayerNorm(dim)
        hidden_dim = int(dim * mlp_ratio)
        self.ffn = FFN(dim, hidden_features=hidden_dim, drop=drop)

    def forward(self, x, H, W):
        x = x + self.attn(self.norm1(x), H, W)
        x = x + self.ffn(self.norm2(x), H, W)
        return x


class MixVisionTransformer(nn.Module):
    """使用 layers[i][0/1/2] 结构匹配 EasyCV 权重"""
    def __init__(self, in_chans=3, embed_dims=[32, 64, 160, 256], num_heads=[1, 2, 5, 8],
                 mlp_ratios=[4, 4, 4, 4], drop_rate=0., drop_path_rate=0.1,
                 depths=[2, 2, 2, 2], sr_ratios=[8, 4, 2, 1], num_stages=4):
        super().__init__()
        self.num_stages = num_stages
        self.layers = nn.ModuleList()
        for i in range(num_stages):
            patch_embed = OverlapPatchEmbed(
                patch_size=7 if i == 0 else 3,
                stride=4 if i == 0 else 2,
                in_chans=in_chans if i == 0 else embed_dims[i - 1],
                embed_dim=embed_dims[i]
            )
            blocks = nn.ModuleList([
                Block(
                    dim=embed_dims[i], num_heads=num_heads[i], mlp_ratio=mlp_ratios[i],
                    sr_ratio=sr_ratios[i], drop=drop_rate
                ) for _ in range(depths[i])
            ])
            norm = nn.LayerNorm(embed_dims[i])
            # EasyCV 权重格式: layers.{stage}.{0/1/2}
            stage = nn.ModuleList([patch_embed, blocks, norm])
            self.layers.append(stage)

    def forward(self, x):
        outs = []
        for i in range(self.num_stages):
            patch_embed, blocks, norm = self.layers[i]
            x, H, W = patch_embed(x)
            for block in blocks:
                x = block(x, H, W)
            x = norm(x)
            x = x.reshape(-1, H, W, x.shape[-1]).permute(0, 3, 1, 2).contiguous()
            outs.append(x)
        return outs


class SegformerHead(nn.Module):
    """Segformer Head，权重 key: convs.{i}.conv/bn, fusion_conv.conv/bn, conv_seg"""
    def __init__(self, in_channels=[32, 64, 160, 256], num_classes=172):
        super().__init__()
        self.convs = nn.ModuleList()
        for in_ch in in_channels:
            self.convs.append(nn.Sequential(
                nn.Conv2d(in_ch, 256, 1, bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True)
            ))
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(4 * 256, 256, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.dropout = nn.Dropout(0.1)
        self.conv_seg = nn.Conv2d(256, num_classes, 1)

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
        out = self.conv_seg(out)
        return out


class Segformer(nn.Module):
    def __init__(self, num_classes=172):
        super().__init__()
        self.backbone = MixVisionTransformer(
            in_chans=3,
            embed_dims=BACKBONE_EMBED_DIMS,
            num_heads=BACKBONE_NUM_HEADS,
            mlp_ratios=[4, 4, 4, 4],
            drop_rate=0.0,
            drop_path_rate=0.1,
            depths=[2, 2, 2, 2],
            sr_ratios=[8, 4, 2, 1],
            num_stages=4
        )
        self.decode_head = SegformerHead(in_channels=BACKBONE_EMBED_DIMS, num_classes=num_classes)

    def forward(self, x):
        feats = self.backbone(x)
        out = self.decode_head(feats)
        return out


# 模型类别数（权重是172类，包含171个COCO-Stuff类 + 1个背景/忽略类）
NUM_CLASSES = 172


# ============================== 模型加载 ==============================
def load_local_segformer():
    """加载本地 EasyCV 格式的 SegFormer 权重"""
    weight_path = WEIGHT_PATH
    if not os.path.isabs(weight_path):
        weight_path = os.path.join(SCRIPT_DIR, weight_path)
    print(f"[INFO] 加载本地 SegFormer-B0 权重: {weight_path}")

    model = Segformer(num_classes=NUM_CLASSES)

    # 加载 EasyCV checkpoint
    checkpoint = torch.load(weight_path, map_location='cpu', weights_only=False)
    state_dict = checkpoint.get('state_dict', checkpoint)

    print(f"[INFO] checkpoint state_dict keys count: {len(state_dict)}")

    # 将 EasyCV 权重 key 映射到自定义模型结构
    mapped_state_dict = {}
    attn_splits = {}  # 用于临时存储 attention in_proj 权重，后续拆分

    for k, v in state_dict.items():
        new_k = k
        # 1. decode_head conv/bn -> Sequential 0/1
        if 'decode_head.convs.' in k or 'decode_head.fusion_conv.' in k:
            if '.conv.' in k or k.endswith('.conv'):
                new_k = k.replace('.conv.', '.0.')
            elif '.bn.' in k or k.endswith('.bn'):
                new_k = k.replace('.bn.', '.1.')
        # 2. attention out_proj -> proj
        elif '.attn.attn.out_proj.' in k:
            new_k = k.replace('.attn.attn.out_proj.', '.attn.proj.')
        # 3. attention in_proj 需要拆分为 q 和 kv，先收集
        elif '.attn.attn.in_proj_' in k:
            attn_splits[k] = v
            continue
        mapped_state_dict[new_k] = v

    # 拆分 attention in_proj_weight/bias 为 q 和 kv
    for k, v in attn_splits.items():
        # k 形如: backbone.layers.0.1.0.attn.attn.in_proj_weight
        prefix = k.replace('.attn.attn.in_proj_weight', '.attn').replace('.attn.attn.in_proj_bias', '.attn')
        dim = v.shape[0] // 3  # in_proj 是 qkv 拼接，总维度为 3*dim
        if 'weight' in k:
            # v shape: (3*dim, dim)
            q_w, k_w, v_w = v[0:dim], v[dim:2*dim], v[2*dim:3*dim]
            mapped_state_dict[f"{prefix}.q.weight"] = q_w
            mapped_state_dict[f"{prefix}.kv.weight"] = torch.cat([k_w, v_w], dim=0)
        else:
            # v shape: (3*dim,)
            q_b, k_b, v_b = v[0:dim], v[dim:2*dim], v[2*dim:3*dim]
            mapped_state_dict[f"{prefix}.q.bias"] = q_b
            mapped_state_dict[f"{prefix}.kv.bias"] = torch.cat([k_b, v_b], dim=0)

    # 尝试直接加载
    missing, unexpected = model.load_state_dict(mapped_state_dict, strict=False)
    if missing:
        print(f"[WARN] 缺失的 key: {missing[:10]}")
    if unexpected:
        print(f"[WARN] 多余的 key: {unexpected[:10]}")

    model = model.to(DEVICE)
    # 启用 FP16 半精度推理以提高 DirectML 效率
    try:
        model = model.half()
        print("[INFO] 已启用 FP16 半精度推理")
    except Exception as e:
        print(f"[WARN] FP16 启用失败 ({e})，使用 FP32")
    model.eval()

    print("[INFO] 本地 SegFormer 模型加载成功")
    return model


# ============================== 预处理 ==============================
def preprocess_image(img_bgr, size=IMG_SIZE):
    """预处理：中心裁剪最大正方形 → resize 到 224，ImageNet 归一化"""
    h, w = img_bgr.shape[:2]
    # 中心裁剪最大正方形
    side = min(h, w)
    y0 = (h - side) // 2
    x0 = (w - side) // 2
    img = img_bgr[y0:y0 + side, x0:x0 + side]
    # resize 到目标尺寸
    img = cv2.resize(img, size, interpolation=cv2.INTER_LINEAR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32)

    # ImageNet 归一化
    mean = np.array([123.675, 116.28, 103.53])
    std = np.array([58.395, 57.12, 57.375])
    img = (img - mean) / std

    # HWC -> CHW
    img = img.transpose(2, 0, 1)
    img = torch.from_numpy(img).unsqueeze(0).float()
    return img, size


# ============================== 后处理 ==============================
def keep_boundary_connected(mask):
    """保留与图像任意边界（上/下/左/右）连通的区域"""
    if not np.any(mask):
        return mask.astype(bool)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), connectivity=8
    )
    valid_labels = set()
    for lbl in range(1, num_labels):
        lbl_mask = (labels == lbl)
        # 与任意一条边界连通即可
        if (np.any(lbl_mask[0, :]) or      # 顶部
            np.any(lbl_mask[-1, :]) or     # 底部
            np.any(lbl_mask[:, 0]) or      # 左侧
            np.any(lbl_mask[:, -1])):      # 右侧
            valid_labels.add(lbl)
    return np.isin(labels, list(valid_labels)) if valid_labels else np.zeros_like(mask, dtype=bool)


def process_and_save(args):
    pred_class, orig_img, base_name, output_root = args
    h, w = orig_img.shape[:2]

    # 调整 pred_class 到原图尺寸
    if pred_class.shape[:2] != (h, w):
        pred_class = cv2.resize(pred_class, (w, h), interpolation=cv2.INTER_NEAREST)

    # ===== 三种标记：clouds+fog / sky-other / 地面+其它 =====
    # clouds + fog 掩膜（不需要贴顶过滤）
    clouds_fog_mask = (pred_class == CLOUDS_ID) | (pred_class == FOG_ID)

    # sky-other 贴顶过滤（必须与图像顶部连通）
    sky_other_mask = keep_boundary_connected((pred_class == SKY_OTHER_ID))

    # 形态学闭运算
    clouds_fog_mask = cv2.morphologyEx(
        clouds_fog_mask.astype(np.uint8), cv2.MORPH_CLOSE, KERNEL
    ).astype(bool)
    sky_other_mask = cv2.morphologyEx(
        sky_other_mask.astype(np.uint8), cv2.MORPH_CLOSE, KERNEL
    ).astype(bool)

    # 底部 24 像素强制视为地面（不识别为天空，留白/留绿）
    BOTTOM_IGNORE = 24
    if h > BOTTOM_IGNORE:
        clouds_fog_mask[-BOTTOM_IGNORE:, :] = False
        sky_other_mask[-BOTTOM_IGNORE:, :] = False

    # 地面/其它 = 非天空
    sky_mask = clouds_fog_mask | sky_other_mask
    ground_mask = ~sky_mask

    # 生成掩膜图
    # clouds+fog = 200, sky-other = 255, 地面/其它 = 128
    filtered_seg = np.zeros((h, w), dtype=np.uint8)
    filtered_seg[clouds_fog_mask] = 200
    filtered_seg[sky_other_mask] = 255
    filtered_seg[ground_mask] = 128

    # 可视化叠加：用曲线（轮廓）标记边缘，不填充颜色
    vis_result = orig_img.copy()
    contour_color_map = {
        'clouds_fog': (0, 100, 255),    # 浅红
        'sky_other': (0, 0, 180),       # 深红
        'ground': (0, 255, 0),          # 绿色
    }

    def draw_contours_on_vis(vis_img, binary_mask, color, thickness=2):
        if not np.any(binary_mask):
            return
        contours, _ = cv2.findContours(
            binary_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(vis_img, contours, -1, color, thickness)

    draw_contours_on_vis(vis_result, clouds_fog_mask, contour_color_map['clouds_fog'])
    draw_contours_on_vis(vis_result, sky_other_mask, contour_color_map['sky_other'])
    draw_contours_on_vis(vis_result, ground_mask, contour_color_map['ground'])

    mask_path = os.path.join(output_root, f"{base_name}_mask.jpg")
    vis_path = os.path.join(output_root, f"{base_name}_sky1.jpg")
    cv2.imencode(".jpg", filtered_seg, [cv2.IMWRITE_JPEG_QUALITY, 95])[1].tofile(mask_path)
    cv2.imencode(".jpg", vis_result, [cv2.IMWRITE_JPEG_QUALITY, 95])[1].tofile(vis_path)


# ============================== 主流程 ==============================
def process_directory(model, executor, input_root):
    """递归扫描 input_root 下所有图片（含子文件夹），返回处理数量"""
    # 递归收集所有图片
    img_paths = []
    for ext in ['*.jpg']: #, '*.jpeg', '*.png', '*.bmp'
        img_paths.extend(glob.glob(os.path.join(input_root, '**', ext), recursive=True))
        img_paths.extend(glob.glob(os.path.join(input_root, '**', ext.upper()), recursive=True))
    img_paths = sorted(list(set(img_paths)))

    if not img_paths:
        print(f"[WARN] 未在 {input_root} 中找到图片")
        return 0

    print(f"\n[INFO] ===== 处理根目录: {input_root} =====")
    print(f"[INFO] 共 {len(img_paths)} 张图片待处理（含子文件夹）")

    futures = []
    processed_count = 0

    for i in range(0, len(img_paths), BATCH_SIZE):
        batch_paths = img_paths[i:i + BATCH_SIZE]
        batch_tensors = []
        orig_imgs = []
        base_names = []
        output_roots = []

        for img_path in batch_paths:
            # 中文路径兼容读取
            img_array = np.fromfile(img_path, dtype=np.uint8)
            orig_img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            if orig_img is None:
                print(f"[WARN] 无法读取: {img_path}")
                continue

            tensor, _ = preprocess_image(orig_img)
            batch_tensors.append(tensor)
            # 中心裁剪最大正方形后统一 resize 到 224，确保从后处理到保存全程都是 224x224
            h, w = orig_img.shape[:2]
            side = min(h, w)
            y0 = (h - side) // 2
            x0 = (w - side) // 2
            cropped_img = orig_img[y0:y0 + side, x0:x0 + side]
            cropped_img = cv2.resize(cropped_img, IMG_SIZE, interpolation=cv2.INTER_LINEAR)
            orig_imgs.append(cropped_img)
            base_name = os.path.splitext(os.path.basename(img_path))[0]
            base_names.append(base_name)

            # 根据相对路径推断类别和输出目录
            rel_path = os.path.relpath(img_path, input_root)
            parts = rel_path.split(os.sep)
            class_name = parts[0] if len(parts) > 1 else "unknown"
            # 输出目录: sky/{class}-sky-segformer-local/{子目录路径}
            output_root = os.path.join(OUTPUT_ROOT_BASE, f"{class_name}-sky-segformer-local")
            if len(parts) > 2:
                output_root = os.path.join(output_root, *parts[1:-1])
            output_roots.append(output_root)

        if not batch_tensors:
            continue

        try:
            # batch 推理（FP16）
            batch_input = torch.cat(batch_tensors, dim=0).to(DEVICE)
            if next(model.parameters()).dtype == torch.float16:
                batch_input = batch_input.half()
            with torch.no_grad():
                logits = model(batch_input)
                preds = logits.argmax(dim=1).cpu().numpy().astype(np.uint8)

            for b in range(len(orig_imgs)):
                os.makedirs(output_roots[b], exist_ok=True)
                task_args = (preds[b], orig_imgs[b], base_names[b], output_roots[b])
                futures.append(executor.submit(process_and_save, task_args))

            processed_count += len(orig_imgs)
            if processed_count % 10 == 0:
                print(f"[INFO] 已提交 {processed_count}/{len(img_paths)} 张...")

        except Exception as e:
            print(f"[WARN] batch 推理失败: {e}")

    print(f"[INFO] 推理完成，等待后处理与保存...")
    for future in futures:
        try:
            future.result()
        except Exception as e:
            print(f"[WARN] 后处理异常: {e}")

    print(f"[INFO] 处理完成，共 {processed_count} 张")
    return processed_count


def main():
    model = load_local_segformer()

    executor = ThreadPoolExecutor(max_workers=MAX_WORKERS)
    start_time = time.time()

    total_processed = 0
    for input_root in INPUT_ROOTS:
        total_processed += process_directory(model, executor, input_root)

    executor.shutdown()

    elapsed = time.time() - start_time
    fps = total_processed / elapsed if elapsed > 0 else 0
    print(f"\n[INFO] ===== 全部完成！共处理 {total_processed} 张，耗时 {elapsed:.1f}s ({fps:.2f} FPS) =====")


if __name__ == "__main__":
    main()

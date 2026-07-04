"""
ConvNeXt-Tiny + 三通道 Sky Mask 深层空间 Cross-Attention 天气分类训练脚本

目标:
    验证现代 Backbone + 深层三通道 Mask 融合是否能超过 DenseNet121+AFF 1通道基线。

三通道 Mask:
    通道0: clouds + fog    (mask 像素值 200)
    通道1: sky-other       (mask 像素值 255)
    通道2: 地面 / 其它      (mask 像素值 128)

核心结构:
    RGB -> ConvNeXt-Tiny 深层特征 [B,768,7,7]
    3通道 Sky Mask -> MaskEncoder 深层特征 [B,768,7,7]
    RGB 深层特征作为 Query，Mask 深层特征作为 Key/Value，做 Spatial Cross-Attention
    分类头输出 sunny/cloudy/rainy/snowy
"""

import os
import sys
import csv
import glob
import time
import copy
import random
import shutil
import subprocess
from datetime import datetime
from collections import defaultdict

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import models
from tqdm import tqdm

import albumentations as A
from albumentations.pytorch import ToTensorV2

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


#train_dir = "/home/jovyan/work/datasets/6a39ed934d7b489daf5f80a4-momodel/train"

DATA_ROOT = r"1cut"
#TEST_ROOT = r"weather224"
#TEST_SIZE = 800
CLASSES = ["sunny", "cloudy", "rainy", "snowy"]
CLASS_TO_IDX = {name: i for i, name in enumerate(CLASSES)}

IMG_SIZE = 224
BATCH_SIZE = 6
LR = 1e-4
WEIGHT_DECAY = 1e-4
ALPHA_CENTER = 0.01
FOCAL_GAMMA = 1
EPOCHS = 100
PATIENCE = 35
SEED = 42

MASK_ROOT = r"sky"
SKY_MASK_ROOT = r"sky"
OUTPUT_DIR = r"runs/convnext_crossmask2"

NUM_WORKERS = 6
PREFETCH_FACTOR = 2
USE_CHANNELS_LAST = True
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_dir(root_name):
    if os.path.isdir(root_name):
        return root_name
    p = os.path.join(SCRIPT_DIR, root_name)
    if os.path.isdir(p):
        return p
    raise FileNotFoundError(f"找不到目录: {root_name}")


def safe_imread_bgr(path):
    if path is None or not os.path.exists(path):
        return None
    try:
        arr = np.fromfile(path, dtype=np.uint8)
        return cv2.imdecode(arr, cv2.IMREAD_COLOR)
    except Exception:
        return None


def safe_imread_gray(path):
    if path is None or not os.path.exists(path):
        return None
    try:
        arr = np.fromfile(path, dtype=np.uint8)
        return cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
    except Exception:
        return None


def center_crop_square_bgr(img_bgr):
    h, w = img_bgr.shape[:2]
    side = min(h, w)
    y0 = (h - side) // 2
    x0 = (w - side) // 2
    return img_bgr[y0:y0 + side, x0:x0 + side]


def preprocess_rgb_and_mask(img_bgr, mask, img_size=IMG_SIZE):
    if img_bgr is None:
        img_rgb = np.zeros((img_size, img_size, 3), dtype=np.uint8)
        ih, iw, y0, x0, side = img_size, img_size, 0, 0, img_size
    else:
        ih, iw = img_bgr.shape[:2]
        side = min(ih, iw)
        y0 = (ih - side) // 2
        x0 = (iw - side) // 2
        img_crop = img_bgr[y0:y0 + side, x0:x0 + side]
        img_rgb = cv2.cvtColor(img_crop, cv2.COLOR_BGR2RGB)
        img_rgb = cv2.resize(img_rgb, (img_size, img_size), interpolation=cv2.INTER_AREA)

    if mask is None:
        mask = np.zeros((img_size, img_size), dtype=np.uint8)
    else:
        if mask.ndim == 3:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        if img_bgr is not None and mask.shape[:2] == (ih, iw):
            mask = mask[y0:y0 + side, x0:x0 + side]
        else:
            mh, mw = mask.shape[:2]
            mside = min(mh, mw)
            my0 = (mh - mside) // 2
            mx0 = (mw - mside) // 2
            mask = mask[my0:my0 + mside, mx0:mx0 + mside]
        mask = cv2.resize(mask, (img_size, img_size), interpolation=cv2.INTER_NEAREST)
    return img_rgb, mask


def find_existing_mask(img_path, class_name, sky_mask_root):
    base_name = os.path.splitext(os.path.basename(img_path))[0]
    mask_filename = f"{base_name}_mask.jpg"
    mask_base_dir = os.path.join(sky_mask_root, f"{class_name}-sky-segformer-local")
    if not os.path.isdir(mask_base_dir):
        return None
    direct = os.path.join(mask_base_dir, mask_filename)
    if os.path.exists(direct):
        return direct
    for root, _, files in os.walk(mask_base_dir):
        if mask_filename in files:
            return os.path.join(root, mask_filename)
    return None


class WeatherSkyDataset(Dataset):
    def __init__(self, img_dir, mask_root, transform=None):
        self.img_dir = img_dir
        self.mask_root = mask_root
        self.transform = transform
        self.samples = []
        for class_name, label in CLASS_TO_IDX.items():
            class_dir = os.path.join(img_dir, class_name)
            if not os.path.isdir(class_dir):
                continue
            for fname in os.listdir(class_dir):
                if fname.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                    self.samples.append({
                        "img_path": os.path.join(class_dir, fname),
                        "class_name": class_name,
                        "label": label,
                    })
        if not self.samples:
            raise RuntimeError(f"没有找到样本: {img_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        img_bgr = safe_imread_bgr(sample["img_path"])
        mask_path = find_existing_mask(sample["img_path"], sample["class_name"], SKY_MASK_ROOT)
        mask_gray = safe_imread_gray(mask_path)

        # 三通道 Mask：
        #   通道0: clouds + fog (mask == 200)
        #   通道1: sky-other     (mask == 255)
        #   通道2: 地面 / 其它    (mask == 128)
        img_rgb, mask_gray_resized = preprocess_rgb_and_mask(img_bgr, mask_gray, IMG_SIZE)
        if mask_gray_resized is not None:
            mask_3ch = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
            mask_3ch[..., 0] = (mask_gray_resized == 200) * 255
            mask_3ch[..., 1] = (mask_gray_resized == 255) * 255
            mask_3ch[..., 2] = (mask_gray_resized == 128) * 255
        else:
            mask_3ch = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)

        if self.transform:
            aug = self.transform(image=img_rgb, mask=mask_3ch)
            img_rgb = aug["image"]
            mask_3ch = aug["mask"]

        if not torch.is_tensor(mask_3ch):
            mask_3ch = torch.from_numpy(mask_3ch)
        # Albumentations ToTensorV2: (H,W,3) -> (3,H,W)
        if mask_3ch.ndim == 3 and mask_3ch.shape[-1] == 3:
            mask_3ch = mask_3ch.permute(2, 0, 1)
        elif mask_3ch.ndim == 2:
            mask_3ch = mask_3ch.unsqueeze(0).repeat(3, 1, 1)
        mask_3ch = mask_3ch.float()
        if mask_3ch.max() > 1.0:
            mask_3ch = mask_3ch / 255.0
        # 每个通道的面积比例
        sky_ratio = mask_3ch.mean(dim=(1, 2)).float()
        return img_rgb, mask_3ch, sky_ratio, torch.tensor(sample["label"], dtype=torch.long)


def build_transforms():
    train_tf = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=12, p=0.25, border_mode=cv2.BORDER_CONSTANT),
        A.ColorJitter(brightness=0.18, contrast=0.18, saturation=0.18, hue=0.06, p=0.45),
        A.GaussNoise(p=0.08),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])
    val_tf = A.Compose([
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])
    return train_tf, val_tf


class MaskEncoder(nn.Module):
    """将 3 通道 Mask（clouds+fog / sky-other / ground-other）编码到 ConvNeXt 深层空间尺度 [B,768,7,7]。"""
    def __init__(self, out_channels=768):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=4, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.Conv2d(128, 256, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.GELU(),
            nn.Conv2d(256, 512, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.GELU(),
            nn.Conv2d(512, out_channels, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )

    def forward(self, mask):
        return self.net(mask)


class SpatialCrossAttention(nn.Module):
    """深层空间交叉注意力：RGB 深层特征为 Query，Mask 深层特征为 Key/Value。"""
    def __init__(self, channels=768, num_heads=8, dropout=0.1):
        super().__init__()
        self.norm_q = nn.LayerNorm(channels)
        self.norm_kv = nn.LayerNorm(channels)
        self.attn = nn.MultiheadAttention(channels, num_heads, dropout=dropout, batch_first=True)
        self.gamma = nn.Parameter(torch.tensor(0.1))
        self.ffn = nn.Sequential(
            nn.LayerNorm(channels),
            nn.Linear(channels, channels * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(channels * 2, channels),
        )

    def forward(self, rgb_feat, mask_feat):
        b, c, h, w = rgb_feat.shape
        q = rgb_feat.flatten(2).transpose(1, 2)
        kv = mask_feat.flatten(2).transpose(1, 2)
        attn_out, _ = self.attn(self.norm_q(q), self.norm_kv(kv), self.norm_kv(kv), need_weights=False)
        x = q + self.gamma * attn_out
        x = x + self.gamma * self.ffn(x)
        return x.transpose(1, 2).reshape(b, c, h, w)


class ConvNeXtCrossMask2(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        try:
            weights = models.ConvNeXt_Tiny_Weights.DEFAULT
            backbone = models.convnext_tiny(weights=weights)
            print("[INFO] ConvNeXt-Tiny 使用 ImageNet 预训练权重")
        except Exception as e:
            print(f"[WARN] ConvNeXt 预训练权重不可用，改用随机初始化: {e}")
            backbone = models.convnext_tiny(weights=None)
        self.rgb_features = backbone.features
        self.mask_encoder = MaskEncoder(out_channels=768)
        self.cross_attn = SpatialCrossAttention(channels=768, num_heads=8, dropout=0.1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.LayerNorm(768),
            nn.Dropout(0.25),
            nn.Linear(768, num_classes),
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


class FocalLoss(nn.Module):
    def __init__(self, gamma=1):
        super().__init__()
        self.gamma = gamma
    def forward(self, inputs, targets):
        ce = F.cross_entropy(inputs, targets, reduction="none")
        pt = torch.exp(-ce)
        return (((1 - pt) ** self.gamma) * ce).mean()


class CenterLoss(nn.Module):
    def __init__(self, num_classes, feat_dim):
        super().__init__()
        self.centers = nn.Parameter(torch.randn(num_classes, feat_dim))
    def forward(self, x, labels):
        # 手动计算欧氏距离，避免 DML 上 torch.cdist 的兼容性问题
        # x: (B, feat_dim), centers: (C, feat_dim) -> dist: (B, C)
        dist = torch.sqrt(torch.sum((x.unsqueeze(1) - self.centers.unsqueeze(0)) ** 2, dim=-1) + 1e-12)
        classes = torch.arange(self.centers.size(0), device=x.device).long()
        mask = labels.unsqueeze(1).expand(x.size(0), self.centers.size(0)).eq(classes.unsqueeze(0)).float()
        return (dist * mask).clamp(min=1e-12).sum() / x.size(0)


def get_device():
    # 1. CUDA
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
        print(f"[INFO] 使用 CUDA: {torch.cuda.get_device_name(0)}")
        return torch.device("cuda:0"), True
    # 2. DirectML
    try:
        import torch_directml
        if torch_directml.is_available():
            dml = torch_directml.device()
            print(f"[INFO] 使用 DirectML: {torch_directml.device_name(0)}")
            return dml, False
    except ImportError:
        pass
    print("[WARN] 未检测到 CUDA/DirectML，使用 CPU")
    return torch.device("cpu"), False


def is_directml_device(device):
    return str(device).startswith("privateuseone") or "directml" in str(device).lower()


def make_loader(dataset, batch_size, shuffle, device):
    # DML/CPU 用 0 workers，避免多进程问题；CUDA 才用多 worker
    use_workers = (device.type == "cuda") and (NUM_WORKERS > 0)
    kwargs = {
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": NUM_WORKERS if use_workers else 0,
        "pin_memory": False,
    }
    if kwargs["num_workers"] > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = PREFETCH_FACTOR
    return DataLoader(dataset, **kwargs)


def plot_metrics(history, save_dir):
    if not history["train_acc"]:
        return
    epochs = range(1, len(history["train_acc"]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.plot(epochs, history["train_acc"], label="Train Acc")
    ax1.plot(epochs, history["val_acc"], label="Val Acc")
    ax1.legend(); ax1.grid(True, alpha=0.3)
    ax2.plot(epochs, history["train_loss"], label="Train Loss")
    ax2.plot(epochs, history["val_loss"], label="Val Loss")
    ax2.legend(); ax2.grid(True, alpha=0.3)
    os.makedirs(save_dir, exist_ok=True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "ConvNeXtCrossMask2_metrics.png"), dpi=200)
    plt.close()


def train():
    set_seed(SEED)
    device, use_amp = get_device()
    start_time_str = datetime.now().strftime("%m%d_%H%M")
    output_dir = os.path.join(SCRIPT_DIR, OUTPUT_DIR)
    os.makedirs(output_dir, exist_ok=True)

    data_path = resolve_dir(DATA_ROOT)
    train_tf, val_tf = build_transforms()

    # 从 DATA_ROOT 自动按类别分层拆分：90% 训练 / 10% 验证（训练:验证 = 9:1）
    full_train_aug = WeatherSkyDataset(data_path, os.path.join(SCRIPT_DIR, MASK_ROOT), transform=train_tf)
    full_val_plain = WeatherSkyDataset(data_path, os.path.join(SCRIPT_DIR, MASK_ROOT), transform=val_tf)

    rng = random.Random(SEED)
    class_to_indices = {i: [] for i in range(len(CLASSES))}
    for idx, sample in enumerate(full_train_aug.samples):
        class_to_indices[sample["label"]].append(idx)

    train_indices, val_indices = [], []
    for cls_idx, indices in class_to_indices.items():
        indices = indices[:]
        rng.shuffle(indices)
        split = int(len(indices) * 0.9)
        train_indices.extend(indices[:split])
        val_indices.extend(indices[split:])

    rng.shuffle(train_indices)
    rng.shuffle(val_indices)
    train_set = Subset(full_train_aug, train_indices)
    val_set = Subset(full_val_plain, val_indices)

    train_loader = make_loader(train_set, BATCH_SIZE, True, device)
    val_loader = make_loader(val_set, BATCH_SIZE, False, device)

    print(f"[INFO] 实验开始时间: {start_time_str}")
    print(f"[INFO] 数据源: {DATA_ROOT}，分层拆分 90% 训练 / 10% 验证（训练:验证=9:1）")
    print(f"[INFO] 数据: train={len(train_set)} val={len(val_set)} batch={BATCH_SIZE}")
    print(f"[INFO] 模型: ConvNeXt-Tiny + 深层 SkyMask Spatial Cross-Attention，weather224 推理")
    print(f"[INFO] 输出目录: {output_dir}")

    model = ConvNeXtCrossMask2(num_classes=len(CLASSES)).to(device)
    if device.type == "cuda" and USE_CHANNELS_LAST:
        model = model.to(memory_format=torch.channels_last)

    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)
    focal = FocalLoss(gamma=FOCAL_GAMMA)
    center = CenterLoss(len(CLASSES), 768).to(device)
    opt_center = optim.SGD(center.parameters(), lr=0.1)
    scaler = torch.amp.GradScaler("cuda") if use_amp and device.type == "cuda" else None

    best_acc, best_wts, best_epoch, trigger, last_epoch = 0.0, None, 0, 0, 0
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    for epoch in range(EPOCHS):
        last_epoch = epoch + 1
        model.train()
        running_loss, corrects = 0.0, 0
        for imgs, masks, ratios, labels in tqdm(train_loader, desc=f"[Train] Ep {epoch + 1}/{EPOCHS}"):
            imgs = imgs.to(device, non_blocking=True)
            if device.type == "cuda" and USE_CHANNELS_LAST:
                imgs = imgs.contiguous(memory_format=torch.channels_last)
            masks = masks.to(device, non_blocking=True)
            ratios = ratios.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            opt_center.zero_grad(set_to_none=True)

            if scaler is not None:
                with torch.amp.autocast("cuda"):
                    logits, feats = model(imgs, masks, ratios)
                    loss = focal(logits, labels) + ALPHA_CENTER * center(feats, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.step(opt_center)
                scaler.update()
            else:
                logits, feats = model(imgs, masks, ratios)
                loss = focal(logits, labels) + ALPHA_CENTER * center(feats, labels)
                loss.backward()
                optimizer.step()
                opt_center.step()

            running_loss += loss.item() * imgs.size(0)
            corrects += torch.sum(torch.argmax(logits, 1) == labels).item()

        model.eval()
        val_loss, val_corrects = 0.0, 0
        with torch.no_grad():
            for imgs, masks, ratios, labels in val_loader:
                imgs = imgs.to(device, non_blocking=True)
                if device.type == "cuda" and USE_CHANNELS_LAST:
                    imgs = imgs.contiguous(memory_format=torch.channels_last)
                masks = masks.to(device, non_blocking=True)
                ratios = ratios.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                if scaler is not None:
                    with torch.amp.autocast("cuda"):
                        logits, feats = model(imgs, masks, ratios)
                        loss = focal(logits, labels) + ALPHA_CENTER * center(feats, labels)
                else:
                    logits, feats = model(imgs, masks, ratios)
                    loss = focal(logits, labels) + ALPHA_CENTER * center(feats, labels)
                val_loss += loss.item() * imgs.size(0)
                val_corrects += torch.sum(torch.argmax(logits, 1) == labels).item()

        t_loss = running_loss / len(train_loader.dataset)
        t_acc = corrects / len(train_loader.dataset)
        v_loss = val_loss / len(val_loader.dataset)
        v_acc = val_corrects / len(val_loader.dataset)
        history["train_loss"].append(t_loss); history["val_loss"].append(v_loss)
        history["train_acc"].append(t_acc); history["val_acc"].append(v_acc)
        print(f"  \nTrain L:{t_loss:.4f} A:{t_acc:.4f} | Val L:{v_loss:.4f} A:{v_acc:.4f} | LR:{optimizer.param_groups[0]['lr']:.1e}")
        scheduler.step(v_acc)

        if v_acc > best_acc:
            best_acc = v_acc
            best_wts = copy.deepcopy(model.state_dict())
            best_epoch = epoch + 1
            trigger = 0
            save_name = f"ConvNeXtCrossMask2_best_ep{best_epoch:03d}_{start_time_str}_acc{best_acc:.4f}.pth"
            save_path = os.path.join(output_dir, save_name)
            torch.save({
                "model_state_dict": best_wts,
                "class_names": CLASSES,
                "fusion": "ConvNeXtTiny-DeepSpatialCrossAttention-Test90Split",
                "best_acc": best_acc,
                "best_epoch": best_epoch,
                "epoch": epoch + 1,
            }, save_path)
            # 删除本轮旧 best，节省服务器硬盘。
            for old in os.listdir(output_dir):
                if old.startswith("ConvNeXtCrossMask2_best_") and start_time_str in old and old != save_name and old.endswith(".pth"):
                    try:
                        os.remove(os.path.join(output_dir, old))
                        print(f"  🧹 已删除旧best: {old}")
                    except Exception as e:
                        print(f"  ⚠️ 删除旧best失败 {old}: {e}")
            print(f"  ✅ 新最佳模型! Val Acc: {best_acc:.4f} | 已保存: {save_path}")
        else:
            trigger += 1
            if trigger >= PATIENCE:
                print(f"  ⚠️ Early Stopping at Epoch {epoch + 1}")
                break

    if best_wts is not None:
        model.load_state_dict(best_wts)
    final_path = os.path.join(output_dir, f"ConvNeXtCrossMask2_final_ep{last_epoch:03d}_bestep{best_epoch:03d}_{start_time_str}_acc{best_acc:.4f}.pth")
    torch.save({
        "model_state_dict": model.state_dict(),
        "class_names": CLASSES,
        "fusion": "ConvNeXtTiny-DeepSpatialCrossAttention-Test90Split",
        "best_acc": best_acc,
        "best_epoch": best_epoch,
        "final_epoch": last_epoch,
    }, final_path)
    print(f"\n💾 最终模型已保存: {final_path} (Best Val Acc: {best_acc:.4f})")
    plot_metrics(history, output_dir)
    return best_acc


def run_inference_after_training():
    infer_script = os.path.join(SCRIPT_DIR, "ConvNeXtCrossMask2推理.py")
    if not os.path.exists(infer_script):
        print(f"[WARN] 未找到推理脚本: {infer_script}")
        return
    print("\n" + "=" * 70)
    print("🚀 训练结束，启动 ConvNeXtCrossMask2 推理程序 ...")
    print("=" * 70)
    subprocess.run([sys.executable, infer_script], cwd=SCRIPT_DIR, check=True)
    print("\n✅ ConvNeXtCrossMask2 推理程序执行成功。")


if __name__ == "__main__":
    best = train()
    print(f"\n🏆 ConvNeXtCrossMask2 训练完成! 最佳验证准确率: {best:.4f}")
    run_inference_after_training()

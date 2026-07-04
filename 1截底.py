import cv2
import numpy as np
from PIL import Image
from pathlib import Path

# ================= 配置区域 =================
TRAIN_DIR = Path(r'train')
OUTPUT_DIR = Path(r'1cut')
IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp'}

# 处理模式选择：
# 'crop_bottom'  -> 抹掉底部：把底部 24 像素涂成白色（保留上方内容，图片尺寸不变）
MODE = 'crop_bottom'

BOTTOM_PIXELS = 24


# ============================================


def load_image_cv2(path):
    pil_img = Image.open(str(path))
    if pil_img.mode == 'RGBA':
        img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGBA2BGRA)
    else:
        img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    return img


def save_image_cv2(img_bgr, path):
    if img_bgr.shape[2] == 4:
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGRA2RGBA)
    else:
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    pil_img.save(str(path))


def process_images():
    # 1. 创建输出目录
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 2. 收集图片
    images = [p for p in TRAIN_DIR.rglob('*') if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
    print(f'找到 {len(images)} 张图片，开始处理...')

    success_count = 0

    for img_path in images:
        try:
            # 计算相对路径，以便在输出目录中保持原有的文件夹结构
            rel_path = img_path.relative_to(TRAIN_DIR)
            save_path = OUTPUT_DIR / rel_path

            # 确保输出的子文件夹存在
            save_path.parent.mkdir(parents=True, exist_ok=True)

            # 3. 加载并处理图片
            img = load_image_cv2(img_path)
            img_h, img_w = img.shape[:2]

            if img_h <= BOTTOM_PIXELS:
                print(f'⚠️ 跳过 {img_path.name} (高度不足 {BOTTOM_PIXELS} 像素)')
                continue

            if MODE == 'crop_bottom':
                # 抹掉底部：将底部 24 像素涂成纯白色 [255, 255, 255]
                img[img_h - BOTTOM_PIXELS: img_h, :] = [255, 255, 255]



            else:
                print("❌ 未知的 MODE，请检查配置！")
                return

            # 4. 保存到新目录
            save_image_cv2(img, save_path)
            success_count += 1

        except Exception as e:
            print(f'❌ 处理失败 {img_path.name}: {e}')

    print(f'\n✅ 处理完成！成功保存了 {success_count} 张图片到: {OUTPUT_DIR}')


if __name__ == '__main__':
    process_images()
import os
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from network import get_model
from utils.utils import get_config
from torchvision import transforms
from PIL import Image
from dotenv import load_dotenv

# 加载.env文件
load_dotenv()

def load_image(image_path, input_resolution):
    """
    加载并预处理图片
    """
    image = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((input_resolution, input_resolution)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # ImageNet标准
    ])
    image = transform(image).unsqueeze(0)  # 增加batch维度
    return image

def predict(image, model):
    """
    使用模型进行分割推理
    """
    with torch.no_grad():
        model.eval()
        seg_output = model(image.cuda(), None, None)

        # 调整分割输出的尺寸
        mask = F.interpolate(seg_output, size=(image.shape[2], image.shape[3]), mode='bilinear', align_corners=False)

        # 输出 mask 的内容
        print("Mask Output:")
        print(mask)  # 打印 mask 变量的内容

        mask = mask.softmax(dim=1)
        preds = torch.argmax(mask, dim=1).cpu().numpy()
    return preds

def extract_parts(image_path, preds, output_dir, save_parts=False):
    """
    将 mask 中不同类别的部位分别提取并保存
    参数:
        - image_path: 输入图像路径
        - preds: 模型预测的分割结果
        - output_dir: 保存输出的目录
        - save_parts: 控制是否保存部位图片，默认 False
    """
    # 读取原图
    image = cv2.imread(image_path)
    h, w = image.shape[:2]

    # 将 mask 调整为与原图相同大小
    mask = cv2.resize(preds[0].astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)

    # 获取唯一类别编号，背景部分为0
    unique_parts = np.unique(mask)

    # 创建一个总和的 mask，用于保存整个脸部区域
    face_mask = np.zeros_like(mask)

    # 遍历 mask 中的所有类别，分别提取并保存，排除背景（part_id == 0）
    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.basename(image_path).split('.')[0]

    # 如果 save_parts 为 True，则保存部位图片
    if save_parts:
        for part_id in unique_parts:
            # 跳过背景部分
            if part_id == 0:
                continue

            # 创建单个部位 mask
            part_mask = (mask == part_id).astype(np.uint8)

            # 将部位 mask 应用到原图
            part = np.zeros_like(image)
            part[part_mask == 1] = image[part_mask == 1]

            # 保存部位图片
            output_path = os.path.join(output_dir, f"{base_name}_part_{part_id}.png")
            cv2.imwrite(output_path, part)
            print(f"部位 {part_id} 提取结果保存至: {output_path}")

    # 创建总和人脸部分的 mask（仅包含去除背景后的部分）
    face_mask = np.isin(mask, unique_parts[unique_parts != 0]).astype(np.uint8)

    # 创建并保存总和图
    face_image = np.zeros_like(image)
    face_image[face_mask == 1] = image[face_mask == 1]
    face_output_path = os.path.join(output_dir, f"{base_name}_face.png")
    cv2.imwrite(face_output_path, face_image)
    print(f"总和人脸部分提取结果保存至: {face_output_path}")

def load_model(args):
    """
    加载模型
    """
    model = get_model(args.backbone, args.input_resolution, args.model).cuda()
    model.eval()
    checkpoint = torch.load(args.model_path, weights_only=True)
    model.load_state_dict(checkpoint['state_dict_backbone'])
    return model

def process_image(image_path, model, input_resolution, output_dir):
    """
    对单张图片进行分割并保存不同部位和总和人脸部分
    """
    image = load_image(image_path, input_resolution)
    preds = predict(image, model)
    extract_parts(image_path, preds, output_dir)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_path", type=str, default="ckpts")
    parser.add_argument("--expt_name", type=str, default='convnext_celeba_512')
    parser.add_argument("--backbone", type=str, default='segface_celeb')
    parser.add_argument("--model", type=str, default='convnext_base')
    parser.add_argument("--input_resolution", type=int, default=512)
    parser.add_argument("--model_path", type=str)
    parser.add_argument("--image_path", type=str, default='test.png')  # 图片路径
    parser.add_argument("--output_dir", type=str, default='output')    # 指定保存路径

    args = parser.parse_args()

    # 模型路径
    args.ckpt_path = os.path.join(os.getenv('LOG_PATH'), args.ckpt_path)
    args.model_path = os.path.join(args.ckpt_path, args.expt_name, 'model_299.pt')

    # 加载模型
    model = load_model(args)

    # 处理图片并保存人脸各部分及总和人脸部分
    process_image(args.image_path, model, args.input_resolution, args.output_dir)

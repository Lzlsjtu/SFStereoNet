# face_processing.py
import os
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from .network import get_model
from torchvision import transforms
from PIL import Image
import time
import shutil

# Global debug flag
debug = False # 设置为True启用调试模式，设置为False禁用调试信息


def debug_print(message):
    """
    控制调试信息的打印
    """
    if debug:
        print(message)


def image_type(image, input_resolution):
    """
    加载并预处理图片
    """
    # debug_print(f"Loading image from {image_path}")
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)  # 转换为 PIL 图像

    transform = transforms.Compose([
        transforms.Resize((input_resolution, input_resolution)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # ImageNet标准
    ])
    image = transform(image).unsqueeze(0)  # 增加batch维度
    # debug_print(f"Image loaded and transformed to tensor with shape {image.shape}")
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
        mask = mask.softmax(dim=1)
        preds = torch.argmax(mask, dim=1).cpu().numpy()

    # debug_print(f"Prediction completed with mask shape {mask.shape}")
    return preds


def select_masks(image_path, preds, output_dir):
    """
    生成各分割区域的缩略图蒙太奇并让用户选择保留的索引
    参数:
        - image_path: 原图路径
        - preds: 模型预测的分割结果 (1, H, W)
        - output_dir: 输出目录
    返回:
        - selected_indices: 用户选择的索引列表
    """
    original_image = cv2.imread(image_path)
    h, w = original_image.shape[:2]

    # 调整mask尺寸与原图一致
    mask = cv2.resize(preds[0].astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
    unique_indices = np.unique(mask)
    unique_indices = unique_indices[unique_indices != 0]  # 排除背景

    if len(unique_indices) == 0:
        print("未检测到任何分割区域")
        return []

    # 为每个索引生成缩略图
    thumbnails = []
    for idx in unique_indices:
        # 提取当前索引的掩膜区域
        part_mask = (mask == idx).astype(np.uint8)
        # 提取原图对应区域，其他区域置黑
        part_image = cv2.bitwise_and(original_image, original_image, mask=part_mask)
        # 调整缩略图尺寸为100x100
        thumbnail = cv2.resize(part_image, (100, 100))
        # 添加索引文字标注
        cv2.putText(thumbnail, str(idx), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        thumbnails.append(thumbnail)

    # 生成缩略图蒙太奇
    cols = 5  # 每行显示5个缩略图
    rows = (len(thumbnails) + cols - 1) // cols
    montage = np.zeros((rows * 100, cols * 100, 3), dtype=np.uint8)

    for i, thumb in enumerate(thumbnails):
        row = i // cols
        col = i % cols
        montage[row * 100:(row + 1) * 100, col * 100:(col + 1) * 100] = thumb

    # 保存蒙太奇图像
    os.makedirs(output_dir, exist_ok=True)
    montage_path = os.path.join(output_dir, "mask_thumbnails.jpg")
    cv2.imwrite(montage_path, montage)
    print(f"分割区域缩略图已保存至: {montage_path}")

    # 获取用户输入
    while True:
        try:
            selected = input(f"可用索引: {unique_indices.tolist()}, 请输入保留的索引(逗号分隔): ")
            selected_indices = list(map(int, selected.strip().split(',')))
            # 过滤无效索引
            selected_indices = [idx for idx in selected_indices if idx in unique_indices]
            if len(selected_indices) == 0:
                print("未选择有效索引，请重新输入")
                continue
            return selected_indices
        except ValueError:
            print("输入格式错误，请使用逗号分隔的整数")

def load_model(args):
    """
    加载预训练模型
    """
    debug_print(f"加载模型: backbone={args.backbone}, resolution={args.input_resolution}")

    start_time = time.time()

    model = get_model(args.backbone, args.input_resolution, args.model).cuda()
    model.eval()
    checkpoint = torch.load(args.model_path, weights_only=True)
    model.load_state_dict(checkpoint['state_dict_backbone'])
    debug_print(f"模型加载自: {args.model_path}")

    end_time = time.time()
    debug_print(f"模型已加载，用时 {end_time - start_time:.2f} 秒")

    return model


def extract_parts1(image_path, preds, output_dir, save_parts=False, selected_indices=None):
    """
    提取并保存指定索引对应的分割区域，先扩展ROI，调整为正方形后裁剪，
    超出边界部分用黑色填充，然后resize为512*512保存，同时返回裁剪框左上角坐标。

    返回:
        - (crop_x_min, crop_y_min, crop_side, resized_crop): 左上角位置和边长，及resize后的图像
    """
    debug_print(f"[Start] 处理图像: {image_path}")
    image = cv2.imread(image_path)
    if image is None:
        debug_print(f"[Error] 无法读取图像: {image_path}")
        return None

    h, w = image.shape[:2]
    debug_print(f"[Info] 原图尺寸: {w}x{h}")

    mask = cv2.resize(preds[0].astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
    unique_parts = np.unique(mask)
    debug_print(f"[Info] 分割掩膜中找到的标签: {unique_parts.tolist()}")

    if selected_indices is None:
        selected_indices = unique_parts[unique_parts != 0].tolist()
    else:
        selected_indices = [idx for idx in selected_indices if idx in unique_parts]
    debug_print(f"[Info] 选定保留的标签索引: {selected_indices}")

    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.basename(image_path).split('.')[0]

    if save_parts:
        for part_id in unique_parts:
            if part_id == 0:
                continue
            part_mask = (mask == part_id).astype(np.uint8)
            part_image = np.zeros_like(image)
            part_image[part_mask == 1] = image[part_mask == 1]
            output_path = os.path.join(output_dir, f"{base_name}_part_{part_id}.bmp")
            cv2.imwrite(output_path, part_image)
            debug_print(f"[Save] 保存部位 {part_id} 至 {output_path}")

    face_mask = np.isin(mask, selected_indices).astype(np.uint8)
    coords = np.column_stack(np.where(face_mask == 1))

    if coords.size == 0:
        debug_print("[Warning] 未找到任何有效的分割区域。")
        return None

    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)
    debug_print(f"[Info] 原始掩膜区域: x=({x_min}-{x_max}), y=({y_min}-{y_max})")

    # 扩展为正方形裁剪框
    box_w = x_max - x_min + 1
    box_h = y_max - y_min + 1
    crop_side = max(box_w, box_h)
    center_x = x_min + box_w // 2
    center_y = y_min + box_h // 2

    crop_x_min = center_x - crop_side // 2
    crop_y_min = center_y - crop_side // 2
    crop_x_max = crop_x_min + crop_side
    crop_y_max = crop_y_min + crop_side
    debug_print(f"[Crop] 原始裁剪区域: 左上角=({crop_x_min}, {crop_y_min}), 边长={crop_side}")

    # 创建黑色背景图并将图像复制进去
    padded_crop = np.zeros((crop_side, crop_side, 3), dtype=np.uint8)

    x_start_src = max(0, crop_x_min)
    y_start_src = max(0, crop_y_min)
    x_end_src = min(crop_x_max, w)
    y_end_src = min(crop_y_max, h)

    x_start_dst = max(0, -crop_x_min)
    y_start_dst = max(0, -crop_y_min)
    x_end_dst = x_start_dst + (x_end_src - x_start_src)
    y_end_dst = y_start_dst + (y_end_src - y_start_src)

    # 拷贝图像有效区域到黑色背景中
    padded_crop[y_start_dst:y_end_dst, x_start_dst:x_end_dst] = image[y_start_src:y_end_src, x_start_src:x_end_src]

    resized_crop = cv2.resize(padded_crop, (512, 512), interpolation=cv2.INTER_LINEAR)

    save_path = os.path.join(output_dir, f"{base_name}_expanded_512.bmp")
    cv2.imwrite(save_path, resized_crop)
    debug_print(f"[Save] 裁剪区域保存至: {save_path}")

    return crop_x_min, crop_y_min, crop_side, resized_crop


def extract_parts2(image_path, output_dir, preds, crop_x_min, crop_y_min, crop_side, save_parts=False, selected_indices=None):
    """
    将 selected_indices 对应的掩膜按 crop_side/512 缩放倍数放大，嵌入 crop_side 方块背景，
    再嵌入原图中指定位置（支持负坐标），生成嵌入原图的面部图像。
    """
    debug_print(f"[Start] 分割处理: {image_path}")
    image = cv2.imread(image_path)
    if image is None:
        debug_print(f"[Error] 无法读取图像: {image_path}")
        return

    h, w = image.shape[:2]
    debug_print(f"[Info] 原图大小: {w}x{h}")

    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.basename(image_path).split('.')[0]

    raw_mask = preds[0].astype(np.uint8)
    raw_h, raw_w = raw_mask.shape[:2]
    scale = crop_side / 512  # 模型输出是512x512
    debug_print(f"[Info] 掩膜原尺寸: {raw_w}x{raw_h}, 缩放倍数: {scale:.3f}")

    full_mask = np.zeros((h, w), dtype=np.uint8)

    unique_parts = np.unique(raw_mask)
    if selected_indices is None:
        selected_indices = unique_parts[unique_parts != 0].tolist()
    else:
        selected_indices = [idx for idx in selected_indices if idx in unique_parts]

    debug_print(f"[Info] 有效标签索引: {selected_indices}")

    for part_id in selected_indices:
        debug_print(f"[Process] 处理标签: {part_id}")
        part_mask = (raw_mask == part_id).astype(np.uint8)

        # 缩放掩膜
        part_mask_resized = cv2.resize(part_mask, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
        ph, pw = part_mask_resized.shape
        debug_print(f"[Resize] 标签 {part_id} 放大后尺寸: {pw}x{ph}")

        # 创建背景方块
        part_canvas = np.zeros((crop_side, crop_side), dtype=np.uint8)
        offset_y = max((crop_side - ph) // 2, 0)
        offset_x = max((crop_side - pw) // 2, 0)
        part_canvas[offset_y:offset_y + ph, offset_x:offset_x + pw] = part_mask_resized[:crop_side - offset_y, :crop_side - offset_x]
        debug_print(f"[Canvas] 标签 {part_id} 嵌入位置: ({offset_x}, {offset_y})")

        # 粘贴位置
        x_start_dst = crop_x_min
        y_start_dst = crop_y_min
        x_end_dst = crop_x_min + crop_side
        y_end_dst = crop_y_min + crop_side

        debug_print(f"[Paste] 标签 {part_id} 原图目标区域: x=({x_start_dst}-{x_end_dst}), y=({y_start_dst}-{y_end_dst})")

        # 越界处理
        x_start_canvas = max(0, -x_start_dst)
        y_start_canvas = max(0, -y_start_dst)
        x_start_dst = max(0, x_start_dst)
        y_start_dst = max(0, y_start_dst)
        x_end_dst = min(w, x_end_dst)
        y_end_dst = min(h, y_end_dst)
        x_end_canvas = x_start_canvas + (x_end_dst - x_start_dst)
        y_end_canvas = y_start_canvas + (y_end_dst - y_start_dst)

        debug_print(f"[Clip] 标签 {part_id} 实际粘贴范围: "
                    f"原图=({x_start_dst}:{x_end_dst}, {y_start_dst}:{y_end_dst}) "
                    f"canvas=({x_start_canvas}:{x_end_canvas}, {y_start_canvas}:{y_end_canvas})")

        # 合并掩膜（不覆盖已有非零值）
        paste_region = part_canvas[y_start_canvas:y_end_canvas, x_start_canvas:x_end_canvas]
        target_region = full_mask[y_start_dst:y_end_dst, x_start_dst:x_end_dst]
        target_region[paste_region == 1] = part_id
        full_mask[y_start_dst:y_end_dst, x_start_dst:x_end_dst] = target_region

    # 保存每个部位（可选）
    if save_parts:
        for part_id in selected_indices:
            part_mask = (full_mask == part_id).astype(np.uint8)
            part_image = np.zeros_like(image)
            part_image[part_mask == 1] = image[part_mask == 1]
            output_path = os.path.join(output_dir, f"{base_name}_part_{part_id}.bmp")
            cv2.imwrite(output_path, part_image)
            debug_print(f"[Save] 单独部位标签 {part_id} 已保存至 {output_path}")

    # 合成整体图像
    face_mask = np.isin(full_mask, selected_indices).astype(np.uint8)
    face_image = np.zeros_like(image)
    face_image[face_mask == 1] = image[face_mask == 1]
    face_output_path = os.path.join(output_dir, f"{base_name}.bmp")
    cv2.imwrite(face_output_path, face_image)
    debug_print(f"[Save] 最终合成图像已保存至: {face_output_path}")


def SegFaces(image_path, model, input_resolution, output_dir, save_parts=False, Type="face"):
    """
    完整的脸部处理流程:
    1. 加载图像
    2. 模型第一次预测
    3. 根据Type设定选中区域或手动选择
    4. 提取第一次预测的方形区域以准备二次预测
    5. 在第一次预测的结果上再次进行预测（解决原图不是方形图的问题，从而提高精度）
    6. 保存二次预测结果
    """
    debug_print(f"开始处理图像: {image_path}")

    # 先读取图片，再转为推理所需类型
    image = Image.open(image_path).convert('RGB')
    image = image_type(image, input_resolution)
    preds1 = predict(image, model)

    # 根据模式选择部位或手动选择
    if Type == "face":
        selected_indices = [2] + list(range(6, 15))  # 纯脸部
    elif Type == "face_ears":
        selected_indices = [2] + list(range(4, 15))  # 脸 + 耳朵
    elif Type == "face_neck":
        selected_indices = [1, 2] + list(range(6, 15))  # 脸 + 颈部
    elif Type == "face_neck":
        selected_indices = [1, 2] + list(range(4, 15))  # 脸 + 耳朵 + 颈部
    elif Type == "full":
        selected_indices = list(range(1, 15))  # 全部
    elif Type == "manual":
        selected_indices = select_masks(image_path, preds1, output_dir)  # 手动选择
    else:
        raise ValueError(f"未知的Type类型: {Type}，可选值为 'face', 'face_neck', 'full', 'manual'")

    # 粗提取人脸区域，为下一次提取做准备
    crop_x_min, crop_y_min, crop_side, resized_crop = extract_parts1(image_path, preds1, output_dir, save_parts, selected_indices)

    # 进行二次推理提高精度
    resized_crop = image_type(resized_crop, input_resolution)
    preds2 = predict(resized_crop, model)

    # 将二次推理结果保存
    extract_parts2(image_path, output_dir, preds2, crop_x_min, crop_y_min, crop_side, save_parts, selected_indices)

def extract_and_Seg_Faces(args,Type="manual",saveparams=True):
    """
    加载模型并处理图像，提取人脸并进行分割
    """
    model = load_model(args)

    for viewpoint_folder in sorted(os.listdir(args.image_folder)):
        viewpoint_path = os.path.join(args.image_folder, viewpoint_folder)
        output_path = os.path.join(args.output_dir, viewpoint_folder)
        if not os.path.isdir(viewpoint_path):
            continue

        image_files = [f for f in os.listdir(viewpoint_path)
                       if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
        for image_file in image_files:
            image_path = os.path.join(viewpoint_path, image_file)
            start_time = time.time()
            SegFaces(image_path, model, args.input_resolution, output_path, args.save_parts,Type)
            end_time = time.time()
            debug_print(f"处理完成，用时 {end_time - start_time:.2f} 秒")

        # 复制 .json 文件
        for file in os.listdir(viewpoint_path):
            if file.lower().endswith('.json'):
                src_json_path = os.path.join(viewpoint_path, file)
                dst_json_path = os.path.join(output_path, file)
                shutil.copy(src_json_path, dst_json_path)
                debug_print(f"已复制 JSON 文件: {file} 到 {output_path}")

    debug_print("所有图片处理完成！")
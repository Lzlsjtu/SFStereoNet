import os
import cv2
import json
import numpy as np
import trimesh
from renderer import render_cvcam

# === 调试开关 ===
DEBUG = False

# === 设置参数 ===
root_dir = './facescape_trainset_001_100'
output_root = '../FacePics/Facescape'
os.makedirs(output_root, exist_ok=True)
if DEBUG: print(f"[INFO] 输出根目录创建或已存在: {output_root}")

# 相机参数
K = np.array([[2000, 0, 256],
              [0, 2000, 256],
              [0, 0, 1]], dtype=np.float64)
baseline = 40
Rt_left = np.array([[1, 0, 0, baseline / 2],
                    [0, -1, 0, 0],
                    [0, 0, -1, 1200]], dtype=np.float64)
Rt_right = np.array([[1, 0, 0, -baseline / 2],
                     [0, -1, 0, 0],
                     [0, 0, -1, 1200]], dtype=np.float64)

h, w = 512, 512

if DEBUG:
    print("[INFO] 相机参数配置完成")
    print("K:\n", K)
    print("Rt_left:\n", Rt_left)
    print("Rt_right:\n", Rt_right)

# === 遍历所有人物文件夹 ===
for i in range(1, 101):
    vp_folder = os.path.join(root_dir, str(i))
    if not os.path.isdir(vp_folder):
        if DEBUG: print(f"[WARNING] 找不到视角文件夹: {vp_folder}")
        continue

    if DEBUG: print(f"[INFO] 处理视角 viewpoint{i}（路径: {vp_folder}）")

    vp_folder = os.path.join(vp_folder, 'models_reg')

    for filename in os.listdir(vp_folder):
        if filename.endswith('.obj') and not filename.endswith('.obj.mtl'):
            name = filename[:-4]
            parts = name.split('_', 1)
            if len(parts) != 2:
                if DEBUG: print(f"[WARNING] 无效命名格式: {filename}")
                continue

            out_expr_dir = os.path.join(output_root, name)
            out_vp_dir = os.path.join(out_expr_dir, f'viewpoint{i}')
            os.makedirs(out_vp_dir, exist_ok=True)

            if DEBUG: print(f"[INFO] -> 表情: {name}, 创建目录: {out_vp_dir}")

            obj_path = os.path.join(vp_folder, filename)
            try:
                mesh = trimesh.load(obj_path, process=False)
                mesh.visual.material.diffuse = np.array([255, 255, 255, 255], dtype=np.uint8)
                if DEBUG: print(f"[DEBUG] 加载 mesh 成功: {filename}, 顶点数: {len(mesh.vertices)}")
            except Exception as e:
                print(f"[ERROR] 加载 mesh 失败: {obj_path}\n{e}")
                continue

            try:
                if DEBUG: print(f"[DEBUG] 开始渲染左视图: {filename}")
                _, left_img = render_cvcam(mesh, K, Rt_left, rend_size=(h, w), flat_shading=False)
                if DEBUG: print(f"[DEBUG] 左视图完成")
                if DEBUG: print(f"[DEBUG] 开始渲染右视图: {filename}")
                _, right_img = render_cvcam(mesh, K, Rt_right, rend_size=(h, w), flat_shading=False)
                if DEBUG: print(f"[DEBUG] 右视图完成")
            except Exception as e:
                print(f"[ERROR] 渲染失败: {obj_path}\n{e}")
                continue

            try:
                cv2.imwrite(os.path.join(out_vp_dir, '0.bmp'), left_img)
                cv2.imwrite(os.path.join(out_vp_dir, '1.bmp'), right_img)
                if DEBUG: print(f"[INFO] 图片保存成功: 0.bmp, 1.bmp")
            except Exception as e:
                print(f"[ERROR] 保存图片失败: {out_vp_dir}\n{e}")
                continue

            try:
                param_json = {
                    "0_K": K.tolist(),
                    "0_Rt": Rt_left.tolist(),
                    "0_distortion": [0, 0, 0, 0, 0],
                    "1_K": K.tolist(),
                    "1_Rt": Rt_right.tolist(),
                    "1_distortion": [0, 0, 0, 0, 0]
                }
                json_path = os.path.join(out_vp_dir, 'params.json')
                with open(json_path, 'w') as f:
                    json.dump(param_json, f, indent=2)
                if DEBUG: print(f"[INFO] 参数文件写入成功: {json_path}")
            except Exception as e:
                print(f"[ERROR] 写入参数文件失败: {json_path}\n{e}")

            if DEBUG: print(f"[✓] 完成处理: {name}/viewpoint{i}\n")

if DEBUG: print("========== 全部处理完成 ==========")

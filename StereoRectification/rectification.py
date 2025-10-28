import os
import json
import cv2
import numpy as np

def invert_rt(Rt):
    R = Rt[:, :3]
    t = Rt[:, 3:]
    R_inv = R.T
    t_inv = -R_inv @ t
    Rt_inv = np.hstack([R_inv, t_inv])
    return Rt_inv

def transform_points(points, Rt_inv):
    ones = np.ones((points.shape[0], 1))
    points_hom = np.hstack([points, ones])  # Nx4
    Rt4x4 = np.vstack([Rt_inv, np.array([[0, 0, 0, 1]])])  # 4x4
    points_world_hom = (Rt4x4 @ points_hom.T).T
    return points_world_hom[:, :3]

def find_image_with_suffix(folder, prefix):
    """在文件夹中查找指定编号的图像文件（例如 '1.*'），返回完整路径和后缀"""
    for fname in os.listdir(folder):
        name, ext = os.path.splitext(fname)
        if name == str(prefix) and ext.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
            return os.path.join(folder, fname), ext
    return None, None

def get_camera_params(image_folder, cam1_idx, cam2_idx):
    json_file = os.path.join(image_folder, 'params.json')
    with open(json_file, 'r') as f:
        camera_params = json.load(f)

    K1    = np.array(camera_params[f"{cam1_idx}_K"], dtype=np.float64)
    Rt1   = np.array(camera_params[f"{cam1_idx}_Rt"], dtype=np.float64)
    dist1 = np.array(camera_params[f"{cam1_idx}_distortion"], dtype=np.float64)

    K2    = np.array(camera_params[f"{cam2_idx}_K"], dtype=np.float64)
    Rt2   = np.array(camera_params[f"{cam2_idx}_Rt"], dtype=np.float64)
    dist2 = np.array(camera_params[f"{cam2_idx}_distortion"], dtype=np.float64)

    img1_path, ext1 = find_image_with_suffix(image_folder, cam1_idx)
    img2_path, ext2 = find_image_with_suffix(image_folder, cam2_idx)

    if not img1_path or not img2_path:
        raise FileNotFoundError(f"❌ 找不到编号为 {cam1_idx} 或 {cam2_idx} 的图像文件")

    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)

    return img1, img2, K1, K2, Rt1, Rt2, dist1, dist2, ext1, ext2


def draw_horizontal_lines(image, line_interval=50):
    h, w = image.shape[:2]
    for y in range(0, h, line_interval):
        cv2.line(image, (0, y), (w, y), (0, 255, 0), thickness=1)
    return image

def stereo_rectify_and_save(image_folder, output_folder, cam1_idx="1", cam2_idx="2", alpha=0.8, debug=False):
    """
    批量立体图像校正处理（融合版本）。

    参数:
        image_folder (str): 包含多个视点子文件夹的主目录
        output_folder (str): 输出路径，结构保持与输入一致，文件名添加 "_rectified"
        cam1_idx (str): 左图像编号（如 "1"）
        cam2_idx (str): 右图像编号（如 "2"）
        alpha (float): cv2.stereoRectify 的 alpha 参数
        debug (bool): 是否输出调试信息
    """
    for viewpoint in sorted(os.listdir(image_folder)):
        vp_path = os.path.join(image_folder, viewpoint)
        if not os.path.isdir(vp_path):
            continue

        if debug:
            print(f"[DEBUG] 处理视角文件夹: {vp_path}")

        left_path, ext1 = find_image_with_suffix(vp_path, cam1_idx)
        right_path, ext2 = find_image_with_suffix(vp_path, cam2_idx)

        if debug:
            print(f"[DEBUG] 左图路径: {left_path}")
            print(f"[DEBUG] 右图路径: {right_path}")

        if not left_path or not right_path:
            print(f"❌ 跳过 {viewpoint}，找不到以 {cam1_idx} 或 {cam2_idx} 开头的图像文件")
            continue

        output_vp_path = os.path.join(output_folder, viewpoint)
        os.makedirs(output_vp_path, exist_ok=True)

        try:
            _stereo_rectify_single_pair(
                input_dir=vp_path,
                output_dir=output_vp_path,
                cam1=os.path.basename(left_path),
                cam2=os.path.basename(right_path),
                alpha=alpha
            )
            print(f"✅ {viewpoint} 校正完成，输出至: {output_vp_path}")
        except Exception as e:
            print(f"❌ {viewpoint} 校正失败: {e}")
            if debug:
                import traceback
                traceback.print_exc()



# 创建 rectified1 和 rectified2 的白色掩膜（非黑色部分为白）
def create_white_mask(image):
    # 创建掩膜：图像中任一通道非零即视为非黑色像素
    mask = np.any(image != [0, 0, 0], axis=-1).astype(np.uint8) * 255
    return mask

def _stereo_rectify_single_pair(input_dir, output_dir, cam1, cam2, alpha=0.9):
    """
    单视点图像对的校正处理（供批处理内部调用）。
    """
    cam1_idx = os.path.splitext(cam1)[0]
    cam2_idx = os.path.splitext(cam2)[0]
    ext = os.path.splitext(cam1)[1]  # 假设左右图后缀一致

    img1, img2, K1, K2, Rt1, Rt2, dist1, dist2, ext1, ext2 = get_camera_params(input_dir, cam1_idx, cam2_idx)

    R1 = Rt1[:3, :3]
    t1 = Rt1[:3, 3].reshape(3, 1)
    R2 = Rt2[:3, :3]
    t2 = Rt2[:3, 3].reshape(3, 1)

    R = R2 @ R1.T
    T = t2 - R @ t1
    imageSize = img1.shape[1], img1.shape[0]

    R1_rect, R2_rect, P1, P2, Q, _, _ = cv2.stereoRectify(
        K1, dist1, K2, dist2, imageSize, R, T,
        flags=cv2.CALIB_ZERO_DISPARITY, alpha=alpha
    )

    map1_x, map1_y = cv2.initUndistortRectifyMap(K1, dist1, R1_rect, P1, imageSize, cv2.CV_32FC1)
    map2_x, map2_y = cv2.initUndistortRectifyMap(K2, dist2, R2_rect, P2, imageSize, cv2.CV_32FC1)

    rectified1 = cv2.remap(img1, map1_x, map1_y, interpolation=cv2.INTER_LINEAR)
    rectified2 = cv2.remap(img2, map2_x, map2_y, interpolation=cv2.INTER_LINEAR)

    concat_horizontal = cv2.hconcat([rectified1, rectified2])

    # 获取屏幕分辨率用于展示
    try:
        import pyautogui
        screen_width, screen_height = pyautogui.size()
    except:
        screen_width, screen_height = 1920, 1080

    target_width = int(screen_width * 0.75)
    target_height = int(screen_height * 0.75)
    h, w = concat_horizontal.shape[:2]
    scale = min(target_width / w, target_height / h)
    new_width, new_height = int(w * scale), int(h * scale)

    resized_concat = cv2.resize(concat_horizontal, (new_width, new_height))
    concat_with_lines = draw_horizontal_lines(resized_concat.copy())

    # 创建并保存掩膜
    rect1_mask = create_white_mask(rectified1)
    rect2_mask = create_white_mask(rectified2)

    cv2.imwrite(os.path.join(output_dir, f'{cam1_idx}_rectified_mask.bmp'), rect1_mask)
    cv2.imwrite(os.path.join(output_dir, f'{cam2_idx}_rectified_mask.bmp'), rect2_mask)

    cv2.imwrite(os.path.join(output_dir, f'{cam1_idx}_{cam2_idx}_rectified_concat_resized.jpg'), concat_with_lines)
    cv2.imwrite(os.path.join(output_dir, f'{cam1_idx}_rectified{ext}'), rectified1)
    cv2.imwrite(os.path.join(output_dir, f'{cam2_idx}_rectified{ext}'), rectified2)

    # 保存内参和基线
    rectified_intrinsics = P1[0:3, 0:3].tolist()
    baseline_distance = abs((P2[0, 3] - P1[0, 3]) / P1[0, 0])
    rectified_params = {
        "rectified_intrinsics": rectified_intrinsics,
        "baseline_distance": baseline_distance
    }

    output_json_path = os.path.join(output_dir, "K.json")
    with open(output_json_path, "w") as f:
        json.dump(rectified_params, f, indent=2)


# 使用示例
if __name__ == "__main__":
    stereo_rectify_and_save("../FacePics","../FacePics_rectified", cam1_idx=1, cam2_idx=2,alpha=0.9)

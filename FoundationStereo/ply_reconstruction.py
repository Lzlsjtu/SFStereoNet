import os
import sys
import argparse
import logging
import json
import torch
import imageio.v2 as imageio
import cv2
import numpy as np
import open3d as o3d
from omegaconf import OmegaConf
import matplotlib.pyplot as plt
from scipy.io import savemat
import time
# 导入自定义工具模块和主模型（请根据实际项目路径调整）
from core.utils.utils import InputPadder  # 图像填充工具类
from Utils import *  # 自定义工具函数，如：set_logging_format, set_seed, vis_disparity, depth2xyzmap, toOpen3dCloud 等
from core.foundation_stereo import FoundationStereo  # 主模型

# 获取项目的根目录
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# 添加到sys.path
sys.path.append(project_root)

# 现在就可以导入
from StereoRectification.rectification import get_camera_params
from StereoRectification.rectification import invert_rt
from StereoRectification.rectification import transform_points

def batch_process_all_viewpoints(
        facepics_dir,
        cam1_idx,
        cam2_idx,
        ckpt_dir,
        base_out_dir,
        scale=0.5,
        hiera=0,
        z_far=500,
        valid_iters=32,
        get_pc=1,
        remove_invisible=1,
        denoise_cloud=1,
        denoise_nb_points=3000,
        denoise_radius=1,
        debug=False  # ✅ 加入调试开关
):
    set_logging_format()
    if debug:
        logging.getLogger().setLevel(logging.DEBUG)  # ✅ 启用 DEBUG 日志
    else:
        logging.getLogger().setLevel(logging.INFO)  # ✅ 默认 INFO 日志

    set_seed(0)
    torch.autograd.set_grad_enabled(False)

    cfg_path = os.path.join(os.path.dirname(ckpt_dir), 'cfg.yaml')
    cfg = OmegaConf.load(cfg_path)
    args = OmegaConf.create(cfg)

    logging.info(f"加载预训练模型: {ckpt_dir}")
    start_time = time.time()

    model = FoundationStereo(args)
    ckpt = torch.load(ckpt_dir, map_location='cuda')
    model.load_state_dict(ckpt['model'])
    model.cuda()
    model.eval()

    end_time = time.time()
    logging.info(f"模型加载完成，用时 {end_time - start_time:.2f} 秒")


    for viewpoint_folder in sorted(os.listdir(facepics_dir)):
        viewpoint_path = os.path.join(facepics_dir, viewpoint_folder)
        if not os.path.isdir(viewpoint_path):
            continue

        mask_file = os.path.join(viewpoint_path, f"{cam1_idx}_rectified_mask.bmp")
        left_file = os.path.join(viewpoint_path, f"{cam1_idx}_rectified.bmp")
        right_file = os.path.join(viewpoint_path, f"{cam2_idx}_rectified.bmp")
        intrinsic_file = os.path.join(viewpoint_path, "K.json")  # 每个视点单独的内参文件

        if not os.path.isfile(left_file) and os.path.isfile(right_file) and os.path.isfile(intrinsic_file) and os.path.isfile(mask_file):
            logging.info(f"跳过 {viewpoint_folder}，缺少图像或内参文件")
            continue

        logging.info(f"处理视点 {viewpoint_folder}...")

        start_time = time.time()

        out_dir = os.path.join(base_out_dir, viewpoint_folder)

        os.makedirs(out_dir, exist_ok=True)
        img0 = imageio.imread(left_file)
        img1 = imageio.imread(right_file)

        if scale > 1:
            raise ValueError("缩放比例必须 <= 1")

        img0 = cv2.resize(img0, fx=scale, fy=scale, dsize=None)
        img1 = cv2.resize(img1, fx=scale, fy=scale, dsize=None)
        H, W = img0.shape[:2]
        img0_ori = img0.copy()

        img0 = torch.as_tensor(img0).cuda().float()[None].permute(0, 3, 1, 2)
        img1 = torch.as_tensor(img1).cuda().float()[None].permute(0, 3, 1, 2)

        padder = InputPadder(img0.shape, divis_by=32, force_square=False)
        img0, img1 = padder.pad(img0, img1)

        with torch.amp.autocast('cuda', enabled=True):
            if not hiera:
                disp = model.forward(img0, img1, iters=valid_iters, test_mode=True)
            else:
                disp = model.run_hierachical(img0, img1, iters=valid_iters, test_mode=True, small_ratio=0.5)

        disp = padder.unpad(disp.float())
        disp = disp.data.cpu().numpy().reshape(H, W)

        # 移除不可见区域的视差点
        if remove_invisible:
            yy, xx = np.meshgrid(np.arange(disp.shape[0]), np.arange(disp.shape[1]), indexing='ij')
            us_right = xx - disp
            invalid = us_right < 0
            disp[invalid] = np.inf

        # 生成并保存点云
        if get_pc:
            # 从 JSON 文件中读取相机内参和基线
            with open(intrinsic_file, 'r') as f:
                calib = json.load(f)
            K = np.array(calib["rectified_intrinsics"]).astype(np.float32)
            baseline = float(calib["baseline_distance"])

            # 根据缩放比例调整内参
            K[:2] *= scale

            # 计算深度图
            depth = K[0, 0] * baseline / disp

            # 将无效值（如负数或NaN）替换为0
            depth_map = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)

            # 假设 disp 是你的视差图，它和 depth_map 同尺寸
            mask = imageio.imread(mask_file)
            if mask.ndim == 3:
                mask = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)

            # Resize 掩膜到与 disp 相同尺寸
            mask = cv2.resize(mask, (disp.shape[1], disp.shape[0]), interpolation=cv2.INTER_NEAREST)
            valid_mask = mask > 127

            # ---------------- 向内收缩掩膜 n 个像素 ----------------
            num_erode_pixels = 2  # 掩膜向内收缩的像素数（即腐蚀迭代次数）
            kernel = np.ones((3, 3), np.uint8)  # 可以根据需要调整核大小
            eroded_mask = cv2.erode(valid_mask.astype(np.uint8), kernel, iterations=num_erode_pixels)
            # -----------------------------------------------------

            # 应用收缩后的掩膜，去除边缘杂点
            depth_map[eroded_mask == 0] = 0

            logging.debug(f"Min Depth: {np.min(depth_map)}, Max Depth: {np.max(depth_map)}")
            logging.debug(f"Depth Map - Shape: {depth_map.shape}, Data Type: {depth_map.dtype}")

            disp[eroded_mask == 0] = 0

            if debug:
                # 保存视差图数据
                np.save(os.path.join(out_dir, 'disparity.npy'), disp)
                savemat(os.path.join(out_dir, 'disparity.mat'), {'disparity': disp})

                # 保存深度图数据
                np.save(os.path.join(out_dir, 'depth_meter.npy'), depth_map)
                savemat(os.path.join(out_dir, 'depth_meter.mat'), {'depth': depth_map})

                plt.figure(figsize=(8, 8))
                plt.imshow(depth_map, cmap='viridis')
                plt.colorbar(label='Depth (meters)')
                plt.title('Depth Map Visualization')
                plt.show()

                plt.figure(figsize=(8, 8))
                plt.imshow(disp, cmap='magma')
                plt.colorbar(label='Disparity (pixels)')
                plt.title('Disparity Map Visualization')
                plt.show()

            vis = vis_disparity(disp)
            vis = np.concatenate([img0_ori, vis], axis=1)
            imageio.imwrite(os.path.join(out_dir, 'vis.png'), vis)
            logging.info(f"视差图结果已保存至 {out_dir}")

            # ========= 点云生成 =========
            xyz_map = depth2xyzmap(depth_map, K)

            # 调试1：检查深度转换后XYZ的有效性
            logging.info(f"XYZ map shape: {xyz_map.shape}, dtype: {xyz_map.dtype}")
            logging.info(f"Z轴范围: min={xyz_map[..., 2].min()}, max={xyz_map[..., 2].max()}")

            pcd = toOpen3dCloud(xyz_map.reshape(-1, 3), img0_ori.reshape(-1, 3))

            # 调试2：验证点云初始化结果
            assert len(pcd.points) > 0, "生成的点云为空，请检查depth2xyzmap或输入深度图"

            points = np.asarray(pcd.points)
            keep_mask = (points[:, 2] > 0) & (points[:, 2] <= z_far)
            keep_ids = np.arange(len(points))[keep_mask]

            # 调试3：过滤条件合理性检查
            logging.info(f"原始点数: {len(points)}, 保留点数: {len(keep_ids)}")
            if len(keep_ids) == 0:
                logging.error("所有点均被过滤，请检查z_far参数或深度单位是否匹配")

            pcd = pcd.select_by_index(keep_ids)
            assert len(pcd.points) > 0, "生成的点云为空，请检查"

            end_time = time.time()
            logging.info(f"点云已生成，用时 {end_time - start_time:.2f} 秒")

            # ========= 将点云转化到初始世界坐标系 =========
            # 替换路径中目录名
            originators_path = viewpoint_path.replace("FacePics_rectified", "FacePics_masked")
            _, _, _, _, Rt1, _, _, _, _, _ = get_camera_params(originators_path, cam1_idx, cam2_idx)
            Rt_inv = invert_rt(Rt1)

            points_cam = np.asarray(pcd.points)
            colors = np.asarray(pcd.colors)

            points_world = transform_points(points_cam, Rt_inv)

            pcd_world = o3d.geometry.PointCloud()
            pcd_world.points = o3d.utility.Vector3dVector(points_world)
            pcd_world.colors = o3d.utility.Vector3dVector(colors)

            o3d.io.write_point_cloud(os.path.join(out_dir, 'cloud.ply'), pcd_world)
            logging.info(f"点云已转为原坐标系下并保存至 {out_dir}/cloud.ply")


            # 点云去噪
            if denoise_cloud:
                start_time = time.time()

                logging.info("正在去噪点云...")
                cl, ind = pcd.remove_radius_outlier(nb_points=denoise_nb_points, radius=denoise_radius)
                logging.info(f"去噪前点数: {len(pcd.points)}, 去噪后点数: {len(ind)}")

                if len(ind) == 0:
                    logging.warning("去噪后点云为空，跳过保存与可视化")
                else:
                    inlier_cloud = pcd.select_by_index(ind)
                    o3d.io.write_point_cloud(os.path.join(out_dir, 'cloud_denoise.ply'), inlier_cloud)
                    pcd = inlier_cloud

                end_time = time.time()
                logging.info(f"点云已去噪，用时 {end_time - start_time:.2f} 秒")

            # # 可视化点云（窗口按 ESC 退出）
            # logging.info("可视化点云中，按 ESC 键退出")
            # vis_o3d = o3d.visualization.Visualizer()
            # vis_o3d.create_window()
            # vis_o3d.add_geometry(pcd)
            # vis_o3d.run()
            # vis_o3d.destroy_window()

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.realpath(__file__))
    facepics_dir = os.path.join(current_dir, "../FacePics_rectified")
    ckpt_dir = os.path.join(current_dir, "pretrained_models/23-51-11/model_best_bp2.pth")
    base_out_dir = os.path.join(current_dir, "../3DFaceReconstruction_result")

    batch_process_all_viewpoints(
        facepics_dir=facepics_dir,
        ckpt_dir=ckpt_dir,
        base_out_dir=base_out_dir,
        scale=0.5,
        hiera=0,
        z_far=500,
        valid_iters=32,
        get_pc=1,
        remove_invisible=1,
        denoise_cloud=1,
        denoise_nb_points=3000,
        denoise_radius=1
    )

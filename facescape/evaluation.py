import open3d as o3d
import trimesh
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

def evaluate_cloud_to_mesh_distance(
    cloud_path, mesh_path,
    output_dir="output",
    save_options=None,
    compute_chamfer=True
):
    """
    精配准点云到网格表面，计算点到面距离并输出多种误差指标。

    参数：
        cloud_path (str): 点云文件路径 (.ply)
        mesh_path (str): 网格模型路径 (.obj)
        output_dir (str): 输出文件夹
        save_options (set): 需要保存的内容，如
            {"aligned_cloud","colored_cloud","csv","hist","summary"}
        compute_chamfer (bool): 是否计算对称 Chamfer 距离

    返回：
        dict: 包含 mean, rms, std, percentiles, outlier_ratio,
              hausdorff, chamfer（若 compute_chamfer=True）等
    """
    if save_options is None:
        save_options = {"aligned_cloud","colored_cloud","csv","hist","summary"}

    os.makedirs(output_dir, exist_ok=True)

    # —— 加载
    cloud = o3d.io.read_point_cloud(cloud_path)
    mesh_o3d = o3d.io.read_triangle_mesh(mesh_path)
    mesh_o3d.compute_vertex_normals()

    # —— ICP
    icp = o3d.pipelines.registration.registration_icp(
        cloud,
        mesh_o3d.sample_points_uniformly(50000),
        0.05, np.identity(4),
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(
            max_iteration=20, relative_fitness=1e-6, relative_rmse=1e-6
        )
    )
    print("ICP RT 矩阵：\n", icp.transformation)
    cloud.transform(icp.transformation)
    pts = np.asarray(cloud.points)

    if "aligned_cloud" in save_options:
        o3d.io.write_point_cloud(
            os.path.join(output_dir,"aligned_cloud.ply"), cloud)

    # —— 点到面距离
    mesh_tm = trimesh.load(mesh_path, force='mesh')
    prox = trimesh.proximity.ProximityQuery(mesh_tm)
    d1 = np.abs(prox.signed_distance(pts))

    # —— 对称 Chamfer: 网格→点云
    chamfer = None
    if compute_chamfer:
        cloud_tree = o3d.geometry.KDTreeFlann(cloud)
        mesh_samples = mesh_o3d.sample_points_uniformly(50000)
        ms_pts = np.asarray(mesh_samples.points)
        d2 = []
        for p in ms_pts:
            _, idx, dist2 = cloud_tree.search_knn_vector_3d(p, 1)
            d2.append(np.sqrt(dist2[0]))
        d2 = np.array(d2)
        chamfer = 0.5*(d1.mean() + d2.mean())

    # —— 基本统计
    mean_er = d1.mean()
    rms_er  = np.sqrt((d1**2).mean())
    std_er  = d1.std()
    max_er  = d1.max()
    min_er  = d1.min()
    mse_er = (d1 ** 2).mean()

    # 百分位
    pct = {p: np.percentile(d1, p) for p in (50,90,95,99)}
    # 3σ 异常点比例
    thr = mean_er + 3*std_er
    outlier_ratio = np.sum(d1>thr)/len(d1)
    # Hausdorff（单向点云→网格）
    hausdorff = max_er

    # —— 可视化 & 导出
    if "colored_cloud" in save_options:
        norm = (d1-d1.min())/(d1.max()-d1.min()+1e-8)
        cloud.colors = o3d.utility.Vector3dVector(
            plt.get_cmap('jet')(norm)[:,:3])
        o3d.visualization.draw_geometries([cloud])
        o3d.io.write_point_cloud(
            os.path.join(output_dir,"cloud_color.ply"), cloud)

    if "csv" in save_options:
        df = pd.DataFrame({"X":pts[:,0],"Y":pts[:,1],"Z":pts[:,2],"Error":d1})
        df.to_csv(os.path.join(output_dir,"errors.csv"), index=False)

    if "hist" in save_options:
        import matplotlib
        matplotlib.rcParams['font.sans-serif'] = ['SimHei']
        matplotlib.rcParams['axes.unicode_minus'] = False
        plt.hist(d1, bins=1000, alpha=0.7)
        plt.title("误差直方图")
        plt.savefig(os.path.join(output_dir,"hist.png"))
        plt.show()

    # —— 输出 & summary
    stats = {
        "mean": mean_er,
        "mse": mse_er,
        "rms": rms_er,
        "std": std_er,
        "min": min_er,
        "max": max_er,
        "percentiles": pct,
        "outlier_ratio": outlier_ratio,
        "hausdorff": hausdorff,
        "chamfer": chamfer
    }

    print("\n误差统计：", stats)
    if "summary" in save_options:
        with open(os.path.join(output_dir,"summary.txt"),"w",encoding="utf-8") as f:
            for k,v in stats.items():
                f.write(f"{k}: {v}\n")

    return stats

import os

def batch_evaluate_all_faces(
        mesh_root,
        cloud_root,
        output_root,
        save_options={"summary", "csv", "hist"},
        compute_chamfer=True,
        subject_ids=None,              # None / set of int / (int, int)
        viewpoint_range=(1, 5)         # 限制 viewpoint 编号范围
):
    """
    遍历所有表情和人物，执行点云对网格的误差评估。

    参数：
        mesh_root: str，原始 FaceScape 网格根目录
        cloud_root: str，重建点云根目录（结构为 表情/viewpointX/cloud.ply）
        output_root: str，评估结果输出目录
        save_options: set，包含 "summary", "csv", "hist" 中任意组合
        compute_chamfer: bool，是否计算 chamfer 和 hausdorff 距离
        subject_ids: None / set of int / (int, int) 表示参与评估的人物编号
        viewpoint_range: tuple，(start, end)，仅评估 viewpoint 起止范围内的编号
    """

    os.makedirs(output_root, exist_ok=True)

    for expr_name in os.listdir(cloud_root):
        expr_path = os.path.join(cloud_root, expr_name)
        if not os.path.isdir(expr_path):
            continue

        for view_name in os.listdir(expr_path):
            if not view_name.startswith("viewpoint"):
                continue

            try:
                subject_id = int(view_name.replace("viewpoint", ""))
            except ValueError:
                continue

            # 限制 viewpoint 范围
            if not (viewpoint_range[0] <= subject_id <= viewpoint_range[1]):
                continue

            # 根据 subject_ids 过滤人物
            if subject_ids is not None:
                if isinstance(subject_ids, (tuple, list)) and len(subject_ids) == 2:
                    start_id, end_id = subject_ids
                    if not (start_id <= subject_id <= end_id):
                        continue
                elif isinstance(subject_ids, set):
                    if subject_id not in subject_ids:
                        continue
                else:
                    raise ValueError("subject_ids 应为 None、set 或 (start, end) 的 tuple")

            cloud_path = os.path.join(expr_path, view_name, "cloud.ply")
            if not os.path.isfile(cloud_path):
                print(f"[WARNING] 点云缺失: {cloud_path}")
                continue

            # mesh 路径中人物编号不补零，直接使用数字字符串
            subject_id_str = str(subject_id)
            mesh_path = os.path.join(
                mesh_root, subject_id_str, "models_reg", f"{expr_name}.obj"
            )

            if not os.path.isfile(mesh_path):
                print(f"[WARNING] 网格缺失: {mesh_path}")
                continue

            out_dir = os.path.join(output_root, expr_name, view_name)
            try:
                print(f"[INFO] 评估: 表情 {expr_name}, 人物 {subject_id}, 视角 {view_name}")
                evaluate_cloud_to_mesh_distance(
                    cloud_path=cloud_path,
                    mesh_path=mesh_path,
                    output_dir=out_dir,
                    save_options=save_options,
                    compute_chamfer=compute_chamfer
                )
            except Exception as e:
                print(f"[ERROR] 处理失败: {expr_name}/{view_name}\n{e}")


def summarize_evaluation_results(eval_root, output_excel):
    """
    汇总评估结果，将每个表情下所有人物的误差统计平均后输出为 Excel 表格。

    参数：
        eval_root: str，评估结果根目录（结构为 表情/人物/summary.txt）
        output_excel: str，输出的 Excel 文件路径
    """
    error_types = [
        "mean", "mse", "rms", "std", "min", "max",
        "outlier_ratio", "hausdorff", "chamfer"
    ]

    all_metrics = error_types

    results = {}

    for expr_name in sorted(os.listdir(eval_root)):
        expr_path = os.path.join(eval_root, expr_name)
        if not os.path.isdir(expr_path):
            continue

        subj_metrics = {key: [] for key in all_metrics}

        for subj_id in os.listdir(expr_path):
            summary_path = os.path.join(expr_path, subj_id, "summary.txt")
            if not os.path.isfile(summary_path):
                continue

            with open(summary_path, 'r', encoding='utf-8') as f:
                for line in f:
                    for key in error_types:
                        if line.startswith(f"{key}:"):
                            try:
                                val = float(line.split(":")[1].strip())
                                subj_metrics[key].append(val)
                            except:
                                pass

                    if line.startswith("percentiles:"):
                        try:
                            pct_part = line.split(":", 1)[1].strip().strip('{}')
                            pct_items = [kv.strip().split(":") for kv in pct_part.split(",")]
                            for k, v in pct_items:
                                pct_key = f"percentiles_{k}"
                                if pct_key in subj_metrics:
                                    subj_metrics[pct_key].append(float(v))
                        except:
                            pass

        expr_avg = {
            key: sum(vals) / len(vals) if vals else None
            for key, vals in subj_metrics.items()
        }

        results[expr_name] = expr_avg

    df = pd.DataFrame.from_dict(results, orient="index")
    df.loc["TotalMean"] = df.mean(numeric_only=True)
    df = df[all_metrics]
    df.to_excel(output_excel, index=True)

    print(f"[✓] Excel 文件已保存到: {output_excel}")


def main():
    # 数据路径
    mesh_root = "facescape_trainset_001_100"
    cloud_root = "../FacePics/FacePCLs"
    eval_output_root = "../FacePics/EvaluationResults"
    excel_path = "../FacePics/evaluation_summary_by_error_type.xlsx"

    # 步骤 1：批量评估所有点云和网格
    print("[STEP 1] 开始批量评估点云与网格误差...")
    batch_evaluate_all_faces(
        mesh_root=mesh_root,
        cloud_root=cloud_root,
        output_root=eval_output_root,
        # subject_ids=("001", "010"),  # 可选
        viewpoint_range=(1, 5)
    )

    # 步骤 2：统计每种表情的平均误差
    print("[STEP 2] 开始汇总误差结果到 Excel...")
    summarize_evaluation_results(
        eval_root=eval_output_root,
        output_excel=excel_path
    )

    print("[✓] 所有流程执行完毕。")

if __name__ == "__main__":
    main()

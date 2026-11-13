"""Stage C — anchor 配准(C.1 等价)+ 全对 pairwise ICP(C.2 等价)。

矩阵约定(与参考工程 stage_c2_pairwise.py 完全一致):
  A_ij = inv(T_bf_j) · T_bf_i           (T_{g_j←g_i})
  B_ij: p_lidar_j = B_ij · p_lidar_i     (ICP: source=cloud_i, target=cloud_j)
  手眼方程 A·X = X·B,X = T_flange_lidar

流程:
  1. anchor 对(可配,默认 capture_0020 居中位姿)FPFH+RANSAC + 多尺度 ICP;
     失败者走链式 fallback(借相邻可用 pose 合成)
  2. 欧拉约定自检:angle(A) vs angle(B) 共轭保角(约定错了立即中止)
  3. X0 = cv2 PARK(绝对 T_bf, inv(T_l0_li))
  4. 全 C(N,2) 对,旋转角 ∈ [rot_min, rot_max] 筛,B_pred=X0⁻¹AX0 当 ICP 初值
  5. gate: angle_diff < 1° ∧ fitness ≥ 0.20 ∧ rmse ≤ 100mm
     被拒 pair 也保留完整指标(D.2 免重跑重滤的前提)

并行说明:spawn 进程池(fork 会继承 open3d TBB 死锁,见 stage_a.py 注释);
worker 内按需懒加载点云并缓存(每 worker 峰值 ≈ 全部聚合云,~400MB)。

输出 analysis/C_planes_{lidar}/:
  icp_transforms.json / pairwise_icp.json / pairwise_icp_report.md
"""
from __future__ import annotations

import argparse
import json
import os
import time
from itertools import combinations
from multiprocessing import get_context
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

import bootref  # noqa: F401
import io_capture
from calib.geometry.transforms import rotation_angle_deg
from calib.io.report import write_markdown
from calib.registration.pipeline import register_pair
from calib.registration.icp import run_multiscale_icp

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ---- worker 进程全局(spawn 后各自独立) ----
_AGG_DIR: Path | None = None
_CLOUDS: dict[str, object] = {}
# 共享计数器:initializer 里给每个 worker 发唯一序号,用于 CPU 亲和性绑定
# (open3d/TBB 每次 C++ 调用会开满全部核,10 worker 不加限制就是 400+ 线程互踩)
_worker_seq = None


def _init_worker(agg_dir: str, seq_lock, cpus_per_worker: int, n_total_cpus: int) -> None:
    global _AGG_DIR, _worker_seq
    _AGG_DIR = Path(agg_dir)
    with seq_lock.get_lock():
        idx = seq_lock.value
        seq_lock.value += 1
    _worker_seq = idx
    cpus = [c for c in range(idx * cpus_per_worker, (idx + 1) * cpus_per_worker)
            if c < n_total_cpus]
    if cpus:
        try:
            os.sched_setaffinity(0, cpus)
        except OSError:
            pass


def _make_init_args(ctx, agg_dir: str, cpus_per_worker: int) -> tuple:
    seq_lock = ctx.Value("i", 0)  # spawn 下必须经 initargs 传入子进程
    return (agg_dir, seq_lock, cpus_per_worker, os.cpu_count())


def _cloud(name: str):
    if name not in _CLOUDS:
        import open3d as o3d
        _CLOUDS[name] = o3d.io.read_point_cloud(str(_AGG_DIR / f"{name}.pcd"))
    return _CLOUDS[name]


def angle_of_T_deg(T: np.ndarray) -> float:
    return rotation_angle_deg(T[:3, :3])


def predict_B(A: np.ndarray, X: np.ndarray) -> np.ndarray:
    return np.linalg.inv(X) @ A @ X


def _anchor_worker(job: tuple) -> tuple:
    """job = (name, anchor_name, T_init_guess, voxels, iters, ransac_voxel)"""
    import time as _time
    name, anchor_name, T_init, voxels, iters, ransac_voxel = job
    t0 = _time.time()
    reg = register_pair(
        _cloud(name), _cloud(anchor_name),
        T_init_guess=T_init,
        voxel_sizes=list(voxels), max_iter_per_scale=list(iters),
        ransac_voxel=ransac_voxel, use_global=True,
    )
    return (name, np.asarray(reg.T), float(reg.fitness), float(reg.inlier_rmse),
            _time.time() - t0)


def _pair_worker(job: tuple) -> tuple:
    """job = (i, j, name_i, name_j, B_pred, voxels, iters, dist_factors)"""
    i, j, name_i, name_j, B_pred, voxels, iters, mdfs = job
    T, fit, rmse = run_multiscale_icp(
        _cloud(name_i), _cloud(name_j), B_pred,
        voxel_sizes=list(voxels), max_iter_per_scale=list(iters),
        max_dist_factor=list(mdfs),
    )
    return i, j, np.asarray(T), float(fit), float(rmse)


def solve_park_anchor(T_bf_list: list[np.ndarray], T_l0_li_list: list[np.ndarray]) -> np.ndarray:
    """anchor 对 + OpenCV PARK 解 X0(照抄参考 solve_park_anchor)。"""
    import cv2
    R_g2b = [T[:3, :3] for T in T_bf_list]
    t_g2b = [T[:3, 3] for T in T_bf_list]
    R_t2c = [np.linalg.inv(T)[:3, :3] for T in T_l0_li_list]
    t_t2c = [np.linalg.inv(T)[:3, 3] for T in T_l0_li_list]
    R_x, t_x = cv2.calibrateHandEye(
        R_g2b, t_g2b, R_t2c, t_t2c, method=cv2.CALIB_HAND_EYE_PARK
    )
    X = np.eye(4)
    X[:3, :3] = R_x
    X[:3, 3] = t_x.ravel()
    return X


def main():
    ap = argparse.ArgumentParser(description="Stage C — anchor + pairwise ICP")
    ap.add_argument("--lidar", required=True, choices=io_capture.LIDARS)
    ap.add_argument("--data-root", type=Path, default=io_capture.DEFAULT_DATA_ROOT)
    ap.add_argument("--extract-dir", type=Path,
                    default=PROJECT_ROOT / "analysis" / "A_extract")
    ap.add_argument("--outdir", type=Path, default=None)
    ap.add_argument("--anchor", default="capture_0020")
    ap.add_argument("--convention", default="extrinsic",
                    choices=io_capture.CONVENTIONS,
                    help="欧拉角合成约定(用户确认 extrinsic;自检会同时评估两个)")
    # anchor 配准参数(参考 C.1 定稿)
    ap.add_argument("--anchor-voxels", default="0.10,0.05,0.02")
    ap.add_argument("--anchor-iters", default="50,30,20")
    ap.add_argument("--ransac-voxel", type=float, default=0.05)
    ap.add_argument("--anchor-fitness-gate", type=float, default=0.4)
    ap.add_argument("--anchor-rmse-gate-m", type=float, default=0.05)
    # pairwise 参数(参考 C.2 定稿)
    ap.add_argument("--voxels", default="0.10,0.10,0.10")
    ap.add_argument("--iters", default="60,60,80")
    ap.add_argument("--dist-factors", default="0.7,0.6,0.5")
    ap.add_argument("--rot-min-deg", type=float, default=12.0)
    ap.add_argument("--rot-max-deg", type=float, default=170.0)
    ap.add_argument("--angle-gate-deg", type=float, default=1.0)
    ap.add_argument("--fitness-gate", type=float, default=0.20)
    ap.add_argument("--rmse-gate-m", type=float, default=0.10)
    ap.add_argument("--max-pairs", type=int, default=0, help=">0 时只跑前 N 对(冒烟)")
    ap.add_argument("--force-anchor", action="store_true",
                    help="忽略已有 icp_transforms.json,强制重跑 anchor 配准")
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--cpus-per-worker", type=int, default=2,
                    help="每 worker 绑定的 CPU 数(含其 open3d 内部线程),0=不绑定")
    args = ap.parse_args()

    a_vox = [float(v) for v in args.anchor_voxels.split(",")]
    a_it = [int(v) for v in args.anchor_iters.split(",")]
    p_vox = [float(v) for v in args.voxels.split(",")]
    p_it = [int(v) for v in args.iters.split(",")]
    p_mdf = [float(v) for v in args.dist_factors.split(",")]
    assert len(a_vox) == len(a_it) and len(p_vox) == len(p_it) == len(p_mdf)

    out_dir = args.outdir or (PROJECT_ROOT / "analysis" / f"C_planes_{args.lidar}")
    out_dir.mkdir(parents=True, exist_ok=True)

    # === 位姿 ===
    names, T_bf, _ = io_capture.load_all_poses(args.data_root, convention=args.convention)
    T_bf_alt = io_capture.load_all_poses(args.data_root, convention="intrinsic")[1]
    agg_dir = args.extract_dir / "aggregated" / f"lidar_{args.lidar}"
    keep = [n for n in names if (agg_dir / f"{n}.pcd").is_file()]
    if args.anchor not in keep:
        print(f"[C] [warn] anchor {args.anchor} 不可用,改用 {keep[0]}")
        args.anchor = keep[0]
    order = [args.anchor] + [n for n in keep if n != args.anchor]  # idx 0 = anchor
    n_poses = len(order)
    print(f"[C] {args.lidar}: {n_poses} 个可用 pose,anchor={args.anchor}")

    T_bf_of = [T_bf[names.index(n)] for n in order]
    T_bf_alt_of = [T_bf_alt[names.index(n)] for n in order]

    # === 1. anchor 配准(并行;已有 icp_transforms.json 时复用) ===
    ctx = get_context("spawn")
    t0 = time.time()
    anchor_json = out_dir / "icp_transforms.json"
    anchor_results: dict[int, dict] = {0: {"status": "anchor", "T": np.eye(4),
                                           "fitness": None, "rmse_m": None,
                                           "strategy": "identity"}}
    if anchor_json.is_file() and not args.force_anchor:
        with open(anchor_json) as f:
            prev = json.load(f)
        if prev.get("anchor") == args.anchor and set(prev.get("transforms", {})) >= set(order):
            for i, n in enumerate(order):
                r = prev["transforms"][n]
                anchor_results[i] = {
                    "status": r["status"], "T": (np.eye(4) if r["T_lidar0_lidar_i"] is None
                                                 else np.array(r["T_lidar0_lidar_i"])),
                    "fitness": r["fitness"], "rmse_m": r["inlier_rmse_m"],
                    "strategy": r["strategy"],
                }
            print(f"\n[C] anchor: 复用 {anchor_json}(--force-anchor 强制重跑)")
        else:
            anchor_json = None  # sentinel: 需要重跑
    if len([r for r in anchor_results.values() if r["status"] in ("ok", "anchor")]) < 2:
        print(f"\n[C] anchor 配准(FPFH+RANSAC + ICP {a_vox}×{a_it}) ...")
        jobs = [
            (order[i], order[0], np.linalg.inv(T_bf_of[0]) @ T_bf_of[i],
             a_vox, a_it, args.ransac_voxel)
            for i in range(1, n_poses)
        ]
        init_args = _make_init_args(ctx, str(agg_dir), args.cpus_per_worker)
        with ProcessPoolExecutor(max_workers=args.jobs, mp_context=ctx,
                                 initializer=_init_worker, initargs=init_args) as ex:
            for name, T, fit, rmse, el in ex.map(_anchor_worker, jobs, chunksize=1):
                idx = order.index(name)
                ok = fit >= args.anchor_fitness_gate and rmse <= args.anchor_rmse_gate_m
                anchor_results[idx] = {
                    "status": "ok" if ok else "direct_failed", "T": T,
                    "fitness": fit, "rmse_m": rmse, "strategy": "direct_global",
                }
                print(f"  {name}: fit={fit:.3f} rmse={rmse*1000:5.1f}mm "
                      f"[{el:.0f}s] {'OK' if ok else 'FAIL'}", flush=True)

    # 链式 fallback:借相邻已 OK 的 pose 合成(串行,在父进程内)
    def _gate(fit, rmse):
        return fit >= args.anchor_fitness_gate and rmse <= args.anchor_rmse_gate_m

    for idx in range(1, n_poses):
        if anchor_results[idx]["status"] != "direct_failed":
            continue
        partner = max((j for j in range(1, n_poses)
                       if j != idx and anchor_results[j]["status"] in ("ok", "anchor")
                       and abs(j - idx) <= 3),
                      key=lambda j: -abs(j - idx), default=None)
        if partner is None:
            anchor_results[idx]["status"] = "failed"
            continue
        reg = register_pair(
            _cloud_in_parent(agg_dir, order[idx]), _cloud_in_parent(agg_dir, order[partner]),
            T_init_guess=None,
            voxel_sizes=a_vox, max_iter_per_scale=a_it,
            ransac_voxel=args.ransac_voxel, use_global=True,
        )
        if _gate(reg.fitness, reg.inlier_rmse):
            anchor_results[idx].update(
                status="ok",
                T=np.asarray(anchor_results[partner]["T"]) @ np.asarray(reg.T),
                fitness=float(reg.fitness), rmse_m=float(reg.inlier_rmse),
                strategy=f"chain_via_{order[partner]}")
            print(f"  {order[idx]}: OK chain via {order[partner]} "
                  f"(fit={reg.fitness:.3f} rmse={reg.inlier_rmse*1000:.1f}mm)")
        else:
            anchor_results[idx]["status"] = "failed"
    ok_idx = sorted(i for i, r in anchor_results.items() if r["status"] in ("ok", "anchor"))
    failed_names = [order[i] for i in anchor_results if anchor_results[i]["status"] == "failed"]
    print(f"[C] anchor: {len(ok_idx)}/{n_poses} OK ({time.time()-t0:.0f}s), failed: {failed_names}")

    # === 2. 欧拉约定自检(共轭保角) ===
    print(f"\n[C] 欧拉约定自检(angle(A) vs angle(B),仅 ang_A≥{args.rot_min_deg}° 的 anchor 对):")
    check = {}
    for conv_label in ("extrinsic", "intrinsic"):
        T_list = T_bf_of if conv_label == args.convention else T_bf_alt_of
        tag = "选用" if conv_label == args.convention else "对照"
        diffs = []
        for i in ok_idx:
            if i == 0:
                continue
            A_i_anchor = np.linalg.inv(T_list[0]) @ T_list[i]  # A_{i,anchor} = inv(T_anchor)·T_i
            ang_A = angle_of_T_deg(A_i_anchor)
            ang_B = angle_of_T_deg(anchor_results[i]["T"])
            if ang_A >= args.rot_min_deg:
                diffs.append(abs(ang_A - ang_B))
        med = float(np.median(diffs)) if diffs else float("nan")
        check[f"{conv_label}({tag})"] = (med, len(diffs))
        print(f"   {conv_label}({tag}): median|Δangle| = {med:.3f}° (n={len(diffs)})"
              f"{'' if med < 1 else '  ⚠️'}")

    chosen_med = check[f"{args.convention}(选用)"][0]
    other_conv = "intrinsic" if args.convention == "extrinsic" else "extrinsic"
    other_med = check[f"{other_conv}(对照)"][0]
    if chosen_med > 5.0:
        print(f"[C] ❌ 欧拉约定自检失败:选用 {args.convention} 的 median|Δangle|="
              f"{chosen_med:.2f}° > 5°(对照 {other_med:.2f}°)。位姿约定与数据不符,中止。")
        raise SystemExit(2)
    if not np.isnan(other_med) and other_med * 3 < chosen_med:
        print(f"[C] ⚠️ 对照约定({other_med:.3f}°)显著优于选用({chosen_med:.3f}°),"
              f"建议核查 --convention")

    # === 3. X0 = anchor PARK ===
    X0 = solve_park_anchor(
        [T_bf_of[i] for i in ok_idx], [anchor_results[i]["T"] for i in ok_idx]
    )
    print(f"\n[C] X0 (anchor PARK): t = {(X0[:3,3]*1000).round(1).tolist()} mm, "
          f"rot from I = {angle_of_T_deg(X0):.2f}°, |t| = {np.linalg.norm(X0[:3,3])*1000:.0f}mm")

    with open(out_dir / "icp_transforms.json", "w") as f:
        json.dump({
            "anchor": args.anchor,
            "euler_convention": args.convention,
            "angle_check": {k: {"median_deg": v[0], "n": v[1]} for k, v in check.items()},
            "transforms": {
                order[i]: {
                    "T_lidar0_lidar_i": (None if i == 0
                                         else np.asarray(anchor_results[i]["T"]).tolist()),
                    "status": anchor_results[i]["status"],
                    "strategy": anchor_results[i]["strategy"],
                    "fitness": anchor_results[i]["fitness"],
                    "inlier_rmse_m": anchor_results[i]["rmse_m"],
                } for i in range(n_poses)
            },
        }, f, indent=1, ensure_ascii=False)

    # === 4. 枚举 pair(只用 anchor OK 的 pose,照参考) ===
    all_pairs = []
    for i, j in combinations(ok_idx, 2):
        A_ij = np.linalg.inv(T_bf_of[j]) @ T_bf_of[i]
        ang_A = angle_of_T_deg(A_ij)
        if args.rot_min_deg <= ang_A <= args.rot_max_deg:
            all_pairs.append((i, j, A_ij, ang_A))
    print(f"\n[C] 候选对 (rot ∈ [{args.rot_min_deg},{args.rot_max_deg}]°): {len(all_pairs)}")
    if args.max_pairs > 0 and len(all_pairs) > args.max_pairs:
        all_pairs = all_pairs[:args.max_pairs]
        print(f"   [debug] 限制到前 {args.max_pairs} 对")

    # === 5. pairwise ICP(并行)+ gate ===
    print(f"[C] pairwise ICP: voxel={p_vox} iters={p_it} dist_factors={p_mdf}, "
          f"gate: Δang<{args.angle_gate_deg}° ∧ fit≥{args.fitness_gate} ∧ "
          f"rmse≤{args.rmse_gate_m*1000:.0f}mm")
    t0 = time.time()
    pair_records = []
    rejected = {"angle_gate": 0, "fitness_gate": 0, "rmse_gate": 0, "exception": 0}
    A_by_pair = {(i, j): A_ij for i, j, A_ij, _ in all_pairs}
    jobs = [(i, j, order[i], order[j], predict_B(A_by_pair[(i, j)], X0),
             p_vox, p_it, p_mdf) for i, j, _, _ in all_pairs]
    init_args = _make_init_args(ctx, str(agg_dir), args.cpus_per_worker)
    with ProcessPoolExecutor(max_workers=args.jobs, mp_context=ctx,
                             initializer=_init_worker, initargs=init_args) as ex:
        for k, r in enumerate(ex.map(_pair_worker, jobs, chunksize=4)):
            i, j, T_icp, fit, rmse = r
            A_ij = A_by_pair[(i, j)]
            B_ij = T_icp
            B_pred = predict_B(A_ij, X0)
            ang_A = angle_of_T_deg(A_ij)
            ang_B = angle_of_T_deg(B_ij)
            ang_diff = abs(ang_A - ang_B)
            if ang_diff > args.angle_gate_deg:
                status = "reject_angle"; rejected["angle_gate"] += 1
            elif fit < args.fitness_gate:
                status = "reject_fitness"; rejected["fitness_gate"] += 1
            elif rmse > args.rmse_gate_m:
                status = "reject_rmse"; rejected["rmse_gate"] += 1
            else:
                status = "ok"
            dT = np.linalg.inv(B_pred) @ B_ij
            pair_records.append({
                "i": i, "j": j, "name_i": order[i], "name_j": order[j],
                "angle_A_deg": ang_A, "angle_B_deg": ang_B,
                "angle_diff_deg": ang_diff,
                "fitness": fit, "rmse_mm": rmse * 1000,
                "A_ij": A_ij.tolist(), "B_ij": B_ij.tolist(),
                "B_pred_init": B_pred.tolist(),
                "correction_dt_mm": float(np.linalg.norm(dT[:3, 3]) * 1000),
                "status": status, "pass": "pairwise",
            })
            if (k + 1) % 100 == 0 or k == len(jobs) - 1:
                ok_ct = sum(1 for r2 in pair_records if r2["status"] == "ok")
                print(f"  [{k+1}/{len(jobs)}] ok={ok_ct} rejected={rejected} "
                      f"[{time.time()-t0:.0f}s]")

    ok_records = [r for r in pair_records if r["status"] == "ok"]
    print(f"\n[C] pairwise done in {time.time()-t0:.0f}s: ok={len(ok_records)}/{len(all_pairs)}, "
          f"rejected={rejected}")
    if ok_records:
        fits = np.array([r["fitness"] for r in ok_records])
        rmses = np.array([r["rmse_mm"] for r in ok_records])
        ad = np.array([r["angle_diff_deg"] for r in ok_records])
        print(f"  fitness: median={np.median(fits):.3f} min={fits.min():.3f}")
        print(f"  rmse mm: median={np.median(rmses):.1f} max={rmses.max():.1f}")
        print(f"  Δang° : median={np.median(ad):.3f} max={ad.max():.3f}")

    metadata = {
        "lidar": args.lidar, "anchor": args.anchor,
        "euler_convention": args.convention,
        "n_poses_total": n_poses, "n_poses_anchor_ok": len(ok_idx),
        "anchor_failed": failed_names,
        "anchor_voxels": a_vox, "anchor_iters": a_it,
        "pairwise_voxels": p_vox, "pairwise_iters": p_it,
        "pairwise_dist_factors": p_mdf,
        "rot_range_deg": [args.rot_min_deg, args.rot_max_deg],
        "angle_gate_deg": args.angle_gate_deg,
        "fitness_gate": args.fitness_gate, "rmse_gate_m": args.rmse_gate_m,
        "X0_t_mm": (X0[:3, 3] * 1000).tolist(),
        "matrix_convention": ("B_ij = T_{lidar_j ← lidar_i}: p_lidar_j = B_ij · p_lidar_i; "
                              "A_ij = inv(T_bf_j)·T_bf_i; A·X = X·B"),
        "n_candidate_pairs": len(all_pairs), "n_ok": len(ok_records),
        "rejected": rejected, "jobs": args.jobs,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(out_dir / "pairwise_icp.json", "w") as f:
        json.dump({"metadata": metadata, "X0": X0.tolist(), "pairs": pair_records},
                  f, indent=1, ensure_ascii=False)
    print(f"[C] wrote {out_dir/'pairwise_icp.json'}")

    lines = [
        f"# Stage C — {args.lidar}",
        "",
        f"- anchor: {args.anchor},欧拉约定 {args.convention}",
        f"- anchor 对: {len(ok_idx)}/{n_poses} OK(failed {failed_names})",
        "- 约定自检 median|Δangle|: "
        + ", ".join(f"{k}={v[0]:.3f}°" for k, v in check.items()),
        f"- X0 t = {(X0[:3,3]*1000).round(1).tolist()} mm",
        f"- 候选 pair {len(all_pairs)},通过 **{len(ok_records)}**,拒绝 {rejected}",
    ]
    write_markdown(out_dir / "pairwise_icp_report.md", lines)


def _cloud_in_parent(agg_dir: Path, name: str):
    """父进程内直接读聚合云(链式 fallback 用)。"""
    import open3d as o3d
    return o3d.io.read_point_cloud(str(agg_dir / f"{name}.pcd"))


if __name__ == "__main__":
    main()

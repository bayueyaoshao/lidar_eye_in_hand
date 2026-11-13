"""Stage D.2 — SE(3) Huber 加权求解 X + split-sample σ(金标准)。

纯点云解(无板联合),逻辑对齐参考工程 stage_d2_pairwise.py:
- 权重 w = fitness / rmse²(逆方差原理)
- 残差 r_k = logmap(A_k·X·B_k⁻¹·X⁻¹),Huber f_scale=3cm
- σ 三口径:split_sample_park(崩溃对照)/ by_pair(低估~5×,参考)/
  **by_pose(金标准,认这个)**
- gate 快速重滤:--fitness-gate/--rmse-gate-mm/--angle-gate-deg 直接从
  pairwise_icp.json(含 reject)重新过滤,不重跑 ICP

输出 analysis/D_calibration_{lidar}/:X_pairwise.{json,txt} + X_pairwise_report.md
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

import bootref  # noqa: F401
from calib.geometry.transforms import matrix_to_rpy_deg, rotation_angle_deg
from calib.handeye import solve_se3_huber, split_sample_sigma
from calib.io.report import write_markdown

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_pairwise(path: Path, fitness_gate=None, rmse_gate_mm=None, angle_gate_deg=None):
    with open(path) as f:
        data = json.load(f)
    if fitness_gate is None and rmse_gate_mm is None and angle_gate_deg is None:
        ok_pairs = [p for p in data["pairs"] if p["status"] == "ok"]
        print(f"[D] loaded {len(ok_pairs)} ok pairs (共 {len(data['pairs'])}) [c2 status]")
    else:
        fit_g = fitness_gate if fitness_gate is not None else -np.inf
        rmse_g = rmse_gate_mm if rmse_gate_mm is not None else np.inf
        ang_g = angle_gate_deg if angle_gate_deg is not None else np.inf
        ok_pairs = [p for p in data["pairs"]
                    if p["fitness"] >= fit_g and p["rmse_mm"] <= rmse_g
                    and p["angle_diff_deg"] <= ang_g]
        print(f"[D] loaded {len(ok_pairs)} pairs (共 {len(data['pairs'])}) "
              f"[gate: fit>={fit_g:.3f} rmse<={rmse_g:.1f}mm ang<={ang_g:.2f}°]")
    A_list = [np.array(p["A_ij"]) for p in ok_pairs]
    B_list = [np.array(p["B_ij"]) for p in ok_pairs]
    rmse_m = np.array([p["rmse_mm"] / 1000 for p in ok_pairs])
    fit = np.array([p["fitness"] for p in ok_pairs])
    weights = fit / np.maximum(rmse_m ** 2, 1e-9)
    return A_list, B_list, weights, ok_pairs


def _fmt_split(s: dict) -> str:
    return (f"σ_rot RMS={s['sigma_full_rot_rms_deg']:.4f}° (worst {s['sigma_full_rot_worst_deg']:.4f}°) | "
            f"σ_trans RMS={s['sigma_full_trans_rms_mm']:.3f}mm (worst {s['sigma_full_trans_worst_mm']:.3f}mm)")


def main():
    ap = argparse.ArgumentParser(description="Stage D.2 — SE(3) Huber 求解")
    ap.add_argument("--lidar", required=True, choices=("airy_202", "airy_205"))
    ap.add_argument("--pairwise-json", type=Path, default=None)
    ap.add_argument("--outdir", type=Path, default=None)
    ap.add_argument("--huber-f-scale", type=float, default=0.03)
    ap.add_argument("--n-splits", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fitness-gate", type=float, default=None)
    ap.add_argument("--rmse-gate-mm", type=float, default=None)
    ap.add_argument("--angle-gate-deg", type=float, default=None)
    args = ap.parse_args()

    pj = args.pairwise_json or (PROJECT_ROOT / "analysis" / f"C_planes_{args.lidar}" / "pairwise_icp.json")
    out_dir = args.outdir or (PROJECT_ROOT / "analysis" / f"D_calibration_{args.lidar}")
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(pj) as f:
        c2 = json.load(f)
    X_init = np.array(c2["X0"])
    A_list, B_list, weights, metas = load_pairwise(
        pj, args.fitness_gate, args.rmse_gate_mm, args.angle_gate_deg)
    n_pairs = len(A_list)
    print(f"[D] {args.lidar}: {n_pairs} pairs, weights median={np.median(weights):.3f}")
    print(f"[D] X_init (anchor PARK): t={(X_init[:3,3]*1000).round(1).tolist()} mm, "
          f"rpy={[round(x,2) for x in matrix_to_rpy_deg(X_init[:3,:3])]}°")

    # === SE(3) Huber ===
    t0 = time.time()
    result = solve_se3_huber(A_list, B_list, weights=weights, X_init=X_init,
                             huber_f_scale=args.huber_f_scale)
    X_se3 = result["X"]
    print(f"[D] SE(3) Huber done in {time.time()-t0:.1f}s (nfev={result['nfev']})")
    print(f"   t = {(X_se3[:3,3]*1000).round(2).tolist()} mm, |t| = {np.linalg.norm(X_se3[:3,3])*1000:.1f} mm")
    print(f"   rpy = {[round(x,3) for x in matrix_to_rpy_deg(X_se3[:3,:3])]}°")
    print(f"   rot resid: mean={result['rot_residual_deg']['mean']:.4f}° "
          f"median={result['rot_residual_deg']['median']:.4f}°")
    print(f"   trans resid: mean={result['trans_residual_mm']['mean']:.2f}mm "
          f"median={result['trans_residual_mm']['median']:.2f}mm")

    # === split-sample 三口径 ===
    pair_pose_idx = [(m["i"], m["j"]) for m in metas]
    print(f"\n[D] split-sample (n={args.n_splits}, seed={args.seed})")
    t0 = time.time()
    split_park = split_sample_sigma(A_list, B_list, weights=None, n_repeats=args.n_splits,
                                    seed=args.seed, method="park", use_se3_huber=False)
    print(f"   [A 崩溃对照 PARK]   {_fmt_split(split_park)}")
    split_by_pair = split_sample_sigma(A_list, B_list, weights=weights, n_repeats=args.n_splits,
                                       seed=args.seed, method="park", use_se3_huber=True,
                                       huber_f_scale=args.huber_f_scale, X_init=X_se3)
    print(f"   [B Huber 按pair↓]  {_fmt_split(split_by_pair)}")
    split_by_pose = split_sample_sigma(A_list, B_list, weights=weights, n_repeats=args.n_splits,
                                       seed=args.seed, method="park", use_se3_huber=True,
                                       huber_f_scale=args.huber_f_scale, X_init=X_se3,
                                       pair_pose_idx=pair_pose_idx)
    print(f"   [C Huber 按pose ✅] {_fmt_split(split_by_pose)}  [{time.time()-t0:.0f}s]")

    # === 落盘 ===
    out_json = {
        "metadata": {
            "lidar": args.lidar, "pairwise_source": str(pj), "n_pairs": n_pairs,
            "huber_f_scale": args.huber_f_scale,
            "n_splits": args.n_splits, "seed": args.seed,
            "gates": {"fitness": args.fitness_gate, "rmse_mm": args.rmse_gate_mm,
                      "angle_deg": args.angle_gate_deg},
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
        "X_init_anchor_park": X_init.tolist(),
        "X_se3_huber": X_se3.tolist(),
        "se3_huber_residuals": {"rot_deg": result["rot_residual_deg"],
                                "trans_mm": result["trans_residual_mm"]},
        "split_sample_park": {k: v for k, v in split_park.items() if k != "deltas"},
        "split_sample_huber_by_pair": {k: v for k, v in split_by_pair.items() if k != "deltas"},
        "split_sample_huber_by_pose": {k: v for k, v in split_by_pose.items() if k != "deltas"},
    }
    (out_dir / "X_pairwise.json").write_text(json.dumps(out_json, indent=1, ensure_ascii=False))
    np.savetxt(out_dir / "X_pairwise.txt", X_se3, fmt="%.8f",
               header=f"X_se3_huber 4x4 T_flange_lidar ({args.lidar})")

    dT = np.linalg.inv(X_init) @ X_se3
    dt = float(np.linalg.norm(dT[:3, 3]) * 1000)
    dR = float(rotation_angle_deg(dT[:3, :3]))
    sp = split_by_pose
    lines = [
        f"# Stage D.2 — {args.lidar} 手眼求解",
        "",
        f"- 输入: `{pj}`({n_pairs} pair)",
        f"- X_init(anchor PARK) t = {(X_init[:3,3]*1000).round(1).tolist()} mm",
        f"- **X_se3_huber t = {(X_se3[:3,3]*1000).round(2).tolist()} mm**, "
        f"rpy = {[round(x,3) for x in matrix_to_rpy_deg(X_se3[:3,:3])]}°",
        f"- X_init vs X_se3: Δt={dt:.1f}mm ΔR={dR:.4f}°",
        "",
        "## split-sample σ_full",
        "",
        "| 口径 | σ_rot RMS | σ_trans RMS | 说明 |",
        "|------|-----------|-------------|------|",
        f"| PARK | {split_park['sigma_full_rot_rms_deg']:.4f}° | "
        f"{split_park['sigma_full_trans_rms_mm']:.3f}mm | 崩溃对照 |",
        f"| Huber 按pair | {split_by_pair['sigma_full_rot_rms_deg']:.4f}° | "
        f"{split_by_pair['sigma_full_trans_rms_mm']:.3f}mm | 低估~5×,参考 |",
        f"| **Huber 按pose** | **{sp['sigma_full_rot_rms_deg']:.4f}°** | "
        f"**{sp['sigma_full_trans_rms_mm']:.3f}mm** | **金标准** |",
        "",
        f"金标准逐轴 — rot(°): {[f'{x:.4f}' for x in sp['sigma_full_rot_per_axis_deg']]}",
        f"金标准逐轴 — trans(mm): {[f'{x:.3f}' for x in sp['sigma_full_trans_per_axis_mm']]}",
        "",
        "## SE(3) Huber 残差",
        "",
        f"- rot: mean={result['rot_residual_deg']['mean']:.4f}° "
        f"median={result['rot_residual_deg']['median']:.4f}° max={result['rot_residual_deg']['max']:.4f}°",
        f"- trans: mean={result['trans_residual_mm']['mean']:.2f}mm "
        f"median={result['trans_residual_mm']['median']:.2f}mm max={result['trans_residual_mm']['max']:.2f}mm",
        "",
        "## 4x4 X = T_flange_lidar",
        "",
        "```",
        *[f"{row[0]:+.8f} {row[1]:+.8f} {row[2]:+.8f} {row[3]:+.8f}" for row in X_se3],
        "```",
    ]
    write_markdown(out_dir / "X_pairwise_report.md", lines)
    print(f"[D] wrote {out_dir}/X_pairwise.{{json,txt}} + report")


if __name__ == "__main__":
    main()

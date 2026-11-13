"""Stage A — 每 capture 聚合严格完整帧 → 每 pose 一个聚合点云。

输出(analysis/A_extract/):
- aggregated/lidar_{id}/capture_XXXX.pcd  (binary, xyz;~数十万点)
- frame_index_{id}.json                    (每帧分类明细)
- sanity_report_{id}.md

聚合管线(isfinite→r_max→SOR→voxel)复用参考工程
calib.io.pcd.aggregate_strict_complete;参数取参考工程 #202/#205 定稿值
(SOR 20/1.0, r_max 20m, voxel 5mm)。
"""
from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import bootref  # noqa: F401
import io_capture
from calib.io.pcd import aggregate_strict_complete

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _process_unit(u: dict) -> dict:
    import open3d as o3d

    frames = io_capture.classify_frames(
        io_capture.scan_lidar_frames(Path(u["lidar_dir"])), ratio=u["ratio"]
    )
    complete = [f for f in frames if f.status == "complete"]
    rec = {
        "lidar": u["lidar"],
        "capture": u["capture"],
        "n_frames": len(frames),
        "n_complete": len(complete),
        "n_points": 0,
        "out": None,
        "quality_ok": u.get("quality_ok"),
        "frames": {str(f.frame): [f.status, f.declared, f.actual] for f in frames},
    }
    if len(complete) < u["min_frames"]:
        rec["skip_reason"] = f"complete 帧数 {len(complete)} < 最低 {u['min_frames']}"
        return rec
    pts = aggregate_strict_complete(
        [f.path for f in complete],
        downsample_voxel_m=u["voxel"],
        sor_nb_neighbors=u["sor_nb"],
        sor_std_ratio=u["sor_std"],
        r_max_m=u["r_max"],
    )
    if len(pts) == 0:
        rec["skip_reason"] = "聚合后为空"
        return rec
    out = Path(u["outdir"]) / f"{u['capture']}.pcd"
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    o3d.io.write_point_cloud(str(out), pcd, write_ascii=False)
    rec.update(n_points=len(pts), out=str(out))
    return rec


def main():
    ap = argparse.ArgumentParser(description="Stage A — capture 帧聚合")
    ap.add_argument("--data-root", type=Path, default=io_capture.DEFAULT_DATA_ROOT)
    ap.add_argument("--outdir", type=Path, default=PROJECT_ROOT / "analysis" / "A_extract")
    ap.add_argument("--lidars", nargs="*", default=list(io_capture.LIDARS))
    ap.add_argument("--captures", nargs="*", default=None,
                    help="只处理这些 capture(调试),默认全部")
    ap.add_argument("--r-max-m", type=float, default=20.0)
    ap.add_argument("--sor-nb", type=int, default=20)
    ap.add_argument("--sor-std", type=float, default=1.0)
    ap.add_argument("--voxel", type=float, default=0.005)
    ap.add_argument("--ratio", type=float, default=0.9,
                    help="complete 判定:declared ≥ ratio×本 capture 最大 declared")
    ap.add_argument("--min-frames", type=int, default=5)
    ap.add_argument("--jobs", type=int, default=8)
    args = ap.parse_args()

    caps = args.captures or io_capture.list_captures(args.data_root)
    units = []
    for lidar in args.lidars:
        agg_dir = args.outdir / "aggregated" / f"lidar_{lidar}"
        agg_dir.mkdir(parents=True, exist_ok=True)
        for cap in caps:
            lidar_dir = args.data_root / cap / lidar
            if not lidar_dir.is_dir():
                continue
            meta = io_capture.load_capture_meta(args.data_root, cap)
            lmeta = meta.get("lidars", {}).get(lidar, {})
            if lmeta.get("error") is not None:
                continue
            units.append({
                "lidar": lidar, "capture": cap, "lidar_dir": str(lidar_dir),
                "outdir": str(agg_dir), "ratio": args.ratio, "voxel": args.voxel,
                "sor_nb": args.sor_nb, "sor_std": args.sor_std,
                "r_max": args.r_max_m, "min_frames": args.min_frames,
                "quality_ok": meta.get("quality_ok"),
            })
    print(f"[A] {len(caps)} capture × {len(args.lidars)} lidar → {len(units)} 个聚合单元")

    t0 = time.time()
    records = []
    # spawn(而非 fork):父进程 import open3d 会启动 TBB 线程池,
    # fork 出的子进程继承已锁的互斥量,首次 C++ 并行调用即死锁(futex_wait)。
    ctx = __import__("multiprocessing").get_context("spawn")
    with ProcessPoolExecutor(max_workers=args.jobs, mp_context=ctx) as ex:
        for rec in ex.map(_process_unit, units, chunksize=1):
            records.append(rec)
            tag = f"{rec['n_points']//1000}k pts" if rec["out"] else rec.get("skip_reason", "?")
            print(f"  {rec['lidar']} {rec['capture']}: {rec['n_complete']}/{rec['n_frames']} complete, {tag}")
    print(f"[A] done in {time.time()-t0:.0f}s")

    # 汇总落盘(按 lidar 分文件,避免参考工程那种两台覆盖同一文件的问题)
    for lidar in args.lidars:
        recs = [r for r in records if r["lidar"] == lidar]
        if not recs:
            continue
        with open(args.outdir / f"frame_index_{lidar}.json", "w") as f:
            json.dump({"units": recs}, f, indent=1, ensure_ascii=False)
        ok = [r for r in recs if r["out"]]
        n_pts = [r["n_points"] for r in ok]
        lines = [
            f"# Stage A sanity report — {lidar}",
            "",
            f"- 聚合参数: r_max={args.r_max_m}m, SOR({args.sor_nb},{args.sor_std}), "
            f"voxel={args.voxel}m, complete ratio={args.ratio}, min_frames={args.min_frames}",
            f"- 可用 capture: **{len(ok)} / {len(recs)}**",
            f"- 聚合点数: min={min(n_pts)}, median={sorted(n_pts)[len(n_pts)//2]}, max={max(n_pts)}",
            "",
            "| capture | complete/frames | points | skip_reason | quality_ok |",
            "|---------|-----------------|--------|-------------|------------|",
        ]
        for r in recs:
            lines.append(
                f"| {r['capture']} | {r['n_complete']}/{r['n_frames']} | "
                f"{r['n_points'] or '-'} | {r.get('skip_reason', '')} | {r['quality_ok']} |"
            )
        (args.outdir / f"sanity_report_{lidar}.md").write_text("\n".join(lines), encoding="utf-8")
        print(f"[A] wrote {args.outdir}/frame_index_{lidar}.json, sanity_report_{lidar}.md")


if __name__ == "__main__":
    main()

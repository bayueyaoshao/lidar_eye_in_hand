# 双雷达手眼标定结果(eye-in-hand, AX=XB)

- 数据: data/capture_0001..0063(62 个位姿 × 2 雷达,每位置 26-40 帧聚合)
- 方法: 严格完整帧聚合(r_max20m→SOR→voxel5mm) → anchor FPFH+RANSAC+多尺度ICP
  → 全对 pairwise ICP(X0 初值) → angle/fitness/rmse 三重 gate → SE(3) Huber 加权求解
- σ 评估: split-sample **按 pose 分半**(金标准,n=20,seed=0)
- 算法复用参考工程 `bolight_robot/RadarDAQ/code/ladar_in_head`(只读,未改动)
- 欧拉约定: values=[X,Y,Z(mm),Rx,Ry,Rz(°)],XYZ **外旋** R=Rz·Ry·Rx
  (自检:anchor 对旋转角守恒 |Δangle| 中位 202=0.106° / 205=0.142°,全部 <0.5°)

## 最终结果 X = T_flange_lidar

### airy_202

```
-0.491812  -0.869273  +0.049850  -0.261047
-0.034542  +0.076686  +0.996457  +3.141751
-0.870016  +0.488347  -0.067742  +0.596625
+0.000000  +0.000000  +0.000000  +1.000000
```
- t = [-261.05, 3141.75, 596.63] mm,|t| = 3208.5 mm
- rpy(XYZ外旋) = [97.897, 60.46, -175.982] °

### airy_205

```
-0.499326  +0.865710  +0.034924  -0.298917
+0.002046  +0.041487  -0.999137  -3.166292
-0.866412  -0.498823  -0.022487  +0.600450
+0.000000  +0.000000  +0.000000  +1.000000
```
- t = [-298.92, -3166.29, 600.45] mm,|t| = 3236.6 mm
- rpy(XYZ外旋) = [-92.581, 60.044, 179.765] °

## 精度 σ_full(金标准:Huber 按 pose 分半)

| 雷达 | pair | σ_rot RMS | σ_trans RMS | 逐轴 rot(°) | 逐轴 trans(mm) |
|------|------|-----------|-------------|-------------|----------------|
| airy_202 | 1060 | **0.0976°** | **9.190mm** | [0.0789, 0.0857, 0.1226] | [5.47, 5.69, 13.83] |
| airy_205 | 945 | **0.0980°** | **11.906mm** | [0.0682, 0.1169, 0.1023] | [7.87, 5.44, 18.27] |

参考对照(参考工程,#202 纯点云解): σ 0.0934° / 7.06mm —— 本结果同量级。
(by-pair 口径 0.021°/1.6mm 因 pose 耦合低估 ~5×,仅参考;PARK 口径崩溃,验证了参考工程『pair-wise 数据 OpenCV PARK 会翻转』的结论)

## SE(3) Huber 残差

| 雷达 | rot mean/median/max(°) | trans mean/median/max(mm) |
|------|------------------------|---------------------------|
| airy_202 | 0.1659/0.1477/0.4874 | 14.34/12.47/52.29 |
| airy_205 | 0.2213/0.2059/0.9824 | 18.55/18.00/74.91 |

## 配对质量(Stage C)

| 雷达 | anchor | anchor 对 | 候选对 | 通过 | 拒绝(fitness) | Δang 中位(°) | rmse 中位(mm) |
|------|--------|-----------|--------|------|----------------|--------------|---------------|
| airy_202 | capture_0020 | 62/62 | 1212 | **1060** | 152 | 0.061 | 36.2 |
| airy_205 | capture_0020 | 60/62 | 1091 | **945** | 146 | 0.085 | 36.5 |

- 零角度门拒绝(Δang 全部 <1°):每个通过 pair 的 ICP 旋转角与机械臂编码器旋转角守恒一致,
  同时验证了欧拉约定与数据链路正确性。

## 物理合理性交叉验证

- 两台独立解算的 |t| 分别为 3209mm 和 3237mm(4m 横梁两端,远离法兰 ~3.2m,一致)
- t_Y 近乎镜像:+3142mm(202)vs -3166mm(205)——背靠背安装的镜像特征
- t_Z 均为 ~597/600mm,安装高度一致
- 两台 σ_rot 几乎相同(0.098°),σ_trans 差异(9.2 vs 11.9mm)与 205 视场遮挡/重叠加权一致

## 使用方法

```bash
# 重复/复算全流程
python3 hecal/stage_a.py                                   # 帧聚合(~11min)
python3 hecal/stage_c2.py --lidar airy_202 --jobs 9        # anchor+pairwise(约1.5h)
python3 hecal/stage_c2.py --lidar airy_205 --jobs 9        # 复用 anchor 则只跑 pairwise
python3 hecal/stage_d2.py --lidar airy_202                 # 求解(~2min)
python3 hecal/stage_d2.py --lidar airy_205

# D.2 支持 gate 快速重滤(不重跑 ICP)
python3 hecal/stage_d2.py --lidar airy_202 --fitness-gate 0.25 --rmse-gate-mm 80
```

## 产物清单

- `analysis/A_extract/`: 聚合云 62×2 + sanity_report + frame_index
- `analysis/C_planes_{lidar}/`: icp_transforms.json(anchor 62 对), pairwise_icp.json(全部 pair 含被拒)
- `analysis/D_calibration_{lidar}/`: X_pairwise.{json,txt,report.md}

## 已知限制

- 本位姿集最大相对旋转 ~90°(参考工程达 170°),约束弱一档,σ_trans 略逊(9-12mm vs 7mm)
- 纯点云解(无标定板)。若需 <5mm,可补三块正交标定板后走参考工程的板角点联合解(--joint-board)
- capture.json quality_ok=False 的目录(本批次无)会被 Stage A 跳过
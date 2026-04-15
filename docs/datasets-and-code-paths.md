# 数据集与代码路径索引

本文档汇总相关**原始数据**、**处理/ GUI 工程**与**公共依赖**在本机上的路径，便于检索与配置。

## 数据来源

| 说明 | 路径 |
|------|------|
| 北京公路推车 | `D:\Code\dataset\BeiJingGongLuTuiChe` |
| 北京公路跑车 | `D:\Code\dataset\BeiJingGongLuPaoChe` |
| 余杭推车 | `D:\Code\dataset\YuHangTuiChe` |
| 海拉尔 | `D:\Code\dataset\hailaer` |

## 处理代码（应用 / GUI 工程）

| 说明 | 路径 |
|------|------|
| chapter5 模块2 | `D:\googleYun\30Code\chapter5_module2` |
| 海拉尔惯性 zAxisPro GUI | `D:\googleYun\30Code\HailaerInertial_zAxisPro_gui` |
| 余杭推车 GUI zAxisPro | `D:\googleYun\30Code\YuHangTuiChe_gui_zAxisPro` |
| 北京公路跑车 GUI zAxisPro | `D:\googleYun\30Code\BeiJingGongLuPaoChe_gui_zAxisPro` |
| 北京公路推车 zAxisPro GUI | `D:\googleYun\30Code\BeiJingGongLuTuiChe_zAxisPro_gui` |

## 公共代码（依赖 / 库）

| 说明 | 路径 |
|------|------|
| PSINS（230321） | `D:\googleYun\psins230321` |
|  Liu 公共代码 | `D:\googleYun\LiuCodeCommon` |

## 统一输出目录（仿真 / 批处理结果）

| 说明 | 路径 |
|------|------|
| 本机汇总输出根目录 | `D:\Code\dataset\output` |

运行 `ct_fgo_sim_main` 时，在 YAML 的 `outputpath` 中指定该目录下的子文件夹（程序会 `create_directories`），避免与原始数据混放。

## CT_FGO_SIM_sliding-window：余杭推车示例运行

**参考 YAML（chapter5 已整理配置，字段与数据切片一致）**

- 目录：`D:\googleYun\30Code\chapter5_module2\prepared_configs\zaxispro\YuHangTuiChe\`
- 示例：`...\20260122_121901_use__transformed1cut1\ct_fgo_sim.yaml`（GNSS/IMU 路径、`kf_interval_sec`、噪声与 NHC 等与之一致）

**本仓库已对齐的运行配置（仅将 `outputpath` 改到 `dataset\output`）**

- `config/run_yuhang_121901_dataset_output.yaml`
- 数据：`D:\Code\dataset\YuHangTuiChe\CT_FGO_use\20260122_121901_use\transformed1cut1\` 下的 `rtk_ct_fgo_sim.txt`、`imu_ct_fgo_sim.txt`
- 输出：`D:\Code\dataset\output\YuHangTuiChe\20260122_121901_use__transformed1cut1_zAxisPro\`（含 `trajectory_enu.txt`、`dense_trajectory_enu.txt`、`nominal_nav.txt`、`bias_nodes.txt`、`delta_estimates.txt`、`run_summary.txt`）

**命令示例（Release 可执行文件路径按本机 build 目录调整）**

```text
D:\Code\CT_FGO_SIM_sliding-window\build\Release\ct_fgo_sim_main.exe D:\Code\CT_FGO_SIM_sliding-window\config\run_yuhang_121901_dataset_output.yaml
```

**导航 vs RTK 经纬高曲线**

- 脚本：`CT_FGO_SIM_sliding-window/tools/plot_nav_vs_rtk_blh.py`
- 通过 `importlib` 加载 `D:\googleYun\30Code\chapter5_module2\run_chapter5_module2_kf_gins_pipeline.py`，复用其中 **WGS84 常数**（`WGS84_RA`、`WGS84_E1`、`RAD2DEG`）；经纬高由 `trajectory_enu.txt` + `run_summary.txt` 中的 `origin_blh_rad` 按与 `Earth::LocalToGlobal` 一致的 ECEF 链换算。
- 默认读上述输出目录，并写 `nav_vs_rtk_blh.png` 到同一输出文件夹；RTK 默认取 `run_summary.txt` 里的 `gnss_file`。

```text
python D:\Code\CT_FGO_SIM_sliding-window\tools\plot_nav_vs_rtk_blh.py --output-dir D:\Code\dataset\output\YuHangTuiChe\20260122_121901_use__transformed1cut1_zAxisPro
```

---

*路径为 Windows 本机绝对路径；若迁移机器或盘符，请同步更新本文档。*

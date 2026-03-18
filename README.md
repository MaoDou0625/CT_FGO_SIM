# CT_FGO_SIM

`CT_FGO_SIM` 是从原始 `CT_FGO` 中拆出来的最小连续时间惯导融合实验仓库。当前版本聚焦：

- 主 IMU + GNSS/RTK
- 连续时间样条轨迹
- 高精度地球模型
- 最小可运行的批处理与分析工具

当前目标不是一次性把轮速、NHC、多外参、多传感器全部带上，而是先把主链路做清楚、做稳定。

## 当前能力

- 读取标准化后的 IMU / RTK 文本数据
- 构建连续时间样条轨迹
- 加入 RTK 位置因子、IMU 惯导因子、偏置随机游走因子
- 用静态段完成粗对准
- 输出轨迹、偏置、误差图和 RMSE
- 支持 YuHangTuiChe 数据的批处理
- 支持按共同位置和按共同轨迹两种方式做高程重复性分析

## 数据格式

### RTK / GNSS

每行格式：

```text
time_s lat_rad lon_rad h_m
```

### IMU

每行格式：

```text
time_s gyro_x_radps gyro_y_radps gyro_z_radps accel_x_mps2 accel_y_mps2 accel_z_mps2
```

## 项目结构

```text
CT_FGO_SIM/
  apps/
  config/
  data/
  include/
  src/
  tools/
  README.md
  CMakeLists.txt
```

## 构建与运行

### 构建

```powershell
cmake -S D:\Code\CT_FGO_SIM -B D:\Code\CT_FGO_SIM\build
cmake --build D:\Code\CT_FGO_SIM\build --config Release
```

### 单组运行

```powershell
D:\Code\CT_FGO_SIM\build\Release\ct_fgo_sim_main.exe D:\Code\CT_FGO_SIM\config\minimal.yaml
```

## 工具脚本

下面这些 Python 脚本都是工具性脚本，用于数据准备、批处理、绘图和重复性分析。

### `tools/convert_yuhang_dataset.py`

作用：

- 将原始 `YuHangTuiChe` 数据转换成 `CT_FGO_SIM` 直接可读的标准格式

输入：

- 原始 `rtk_cut.txt`
- 原始 `imu_cut.txt`

输出：

- `rtk_ct_fgo_sim.txt`
- `imu_ct_fgo_sim.txt`
- `schema.md`

### `tools/run_yuhang_batch.py`

作用：

- 批量运行 `YuHangTuiChe` 下所有 `transformed1cut*` 分组
- 自动生成每组配置
- 自动调用主程序和误差绘图脚本

输出根目录：

- `D:\Code\dataset\YuHangTuiChe\ct_fgo_sim_results`

### `tools/plot_outputs.py`

作用：

- 对单组结果生成误差图和对比图
- 统计 `rmse_e/n/u/h/3d`
- 识别 IMU 静止到运动的起始时刻，并把时间标到误差图上

输出：

- `horizontal_compare.png`
- `horizontal_error.png`
- `height_compare.png`
- `height_error.png`
- `enu_error_timeseries.png`
- `metrics_summary.txt`

用法：

```powershell
python D:\Code\CT_FGO_SIM\tools\plot_outputs.py --output-dir D:\Code\dataset\YuHangTuiChe\ct_fgo_sim_results\20260122_145751_use\transformed1cut1
```

### `tools/analyze_yuhang_repeatability.py`

作用：

- 基于所有组结果，寻找“所有组共有的 RTK 位置”
- 统计这些共同位置上每组的 RTK 高程和导航高程
- 比较 RTK 与导航的高程重复性

注意：

- 这里的导航高程已经修正为绝对高程，不再直接使用局部 ENU 的 `Up`

输出目录：

- `D:\Code\dataset\YuHangTuiChe\ct_fgo_sim_results\repeatability_analysis`

主要输出：

- `common_position_group_heights.csv`
- `common_position_repeatability.csv`
- `repeatability_summary.txt`

### `tools/plot_repeatability_heights.py`

作用：

- 基于共同位置高程统计表，绘制：
  - 导航高程汇总图
  - RTK 高程汇总图

输出：

- `navigation_height_summary.png`
- `rtk_height_summary.png`

### `tools/analyze_yuhang_common_trajectory.py`

作用：

- 针对“同一路段重复跑，但起点终点不完全一致”的场景
- 先生成一条公共参考轨迹
- 再根据公共轨迹的共同起终点截取各组数据
- 按公共里程 `s` 分箱统计 RTK 和导航高程重复性

输出目录：

- `D:\Code\dataset\YuHangTuiChe\ct_fgo_sim_results\common_trajectory_analysis`

主要输出：

- `common_trajectory_summary.txt`
- `segment_coverage.csv`
- `reference_common_trajectory.csv`
- `trimmed_group_heights_by_s.csv`
- `repeatability_by_s_bin.csv`
- `navigation_height_vs_common_s.png`
- `rtk_height_vs_common_s.png`
- `height_repeatability_std_vs_common_s.png`

## 当前已验证的数据集

- `D:\Code\dataset\YuHangTuiChe\ct_fgo_sim_use`

当前已完成：

- 5 组数据批处理
- 单组误差分析
- 共同位置高程重复性分析
- 公共轨迹截取与按里程高程重复性分析

## 当前结论

按“公共轨迹 + 公共区间 + 1 m 里程分箱”的方案，5 组数据的高程重复性结果为：

- RTK 高程重复性均值标准差约 `0.0119 m`
- 导航高程重复性均值标准差约 `0.0126 m`

说明当前版本在这批数据上的高程重复性已经接近 RTK。

## 下一步

后续可以在新分支继续做：

- NHC 引入与测试
- 融合中心线替代单组参考轨迹
- 更稳健的轨迹投影和公共区间提取
- 时间偏差 `td` 放开估计
- 高程异常段鲁棒处理

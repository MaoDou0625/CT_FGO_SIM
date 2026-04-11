# CT 姿态可观测性增强方案：复用 KF-GINS 误差传播链

生成日期：2026-04-11

## 1. 目标

当前目标是完成“方案 1”：让 CT-FGO 像 KF-GINS 一样，通过 GNSS 位置观测和 INS 误差传播链间接修正姿态误差。

本轮原则：

- 尽量复用 `D:\Code\kf_gins_used_in_paper` 中已经验证过的 KF-GINS 内容。
- 不重新设计理论模型。
- 不优先新增 NHC、yaw course factor、姿态伪观测等额外观测。
- 先保证 CT 的误差状态定义、传播矩阵、GNSS 残差方向、状态反馈符号与 KF-GINS 一致。

## 2. 当前判断

检查后发现，CT 代码中已经存在一套类 KF 的 15 维误差状态传播链。

当前 CT 已有状态结构：

```text
delta_pos, delta_vel, delta_theta, delta_bg, delta_ba
```

当前 CT 已有机制：

```text
F 矩阵传播
Qd / sqrt_info
GNSS 位置误差因子
误差状态反馈到 nominal_nav
外迭代闭环
```

因此，优先问题不是“没有姿态可观测性传播链”，而是：

```text
CT 的误差状态符号、反馈方向、坐标约定或 GNSS 残差方向，可能与 KF-GINS 不完全一致。
```

如果这些定义不一致，即使 CT 中存在 `F(V,PHI)`、`F(PHI,BG)` 等块，GNSS 位置残差也可能无法有效、正确地修正姿态。

## 3. KF-GINS 中需要复用的核心内容

### 3.1 KF-GINS 状态索引

参考文件：

```text
D:\Code\kf_gins_used_in_paper\src\kf-gins\gi_engine.h
```

KF-GINS 的误差状态索引：

```text
P_ID   = 0
V_ID   = 3
PHI_ID = 6
BG_ID  = 9
BA_ID  = 12
SG_ID  = 15
SA_ID  = 18
```

CT 当前主要使用前 15 维：

```text
P, V, PHI, BG, BA
```

本轮建议不要引入比例因子 `SG/SA`，先对齐前 15 维。

### 3.2 KF-GINS 误差传播

参考文件：

```text
D:\Code\kf_gins_used_in_paper\src\kf-gins\gi_engine.cpp
```

重点参考函数：

```text
GIEngine::insPropagation()
GIEngine::EKFPredict()
```

需要逐项对照的矩阵块：

```text
F(P_ID, P_ID)
F(P_ID, V_ID)
F(V_ID, P_ID)
F(V_ID, V_ID)
F(V_ID, PHI_ID)
F(V_ID, BA_ID)
F(PHI_ID, P_ID)
F(PHI_ID, V_ID)
F(PHI_ID, PHI_ID)
F(PHI_ID, BG_ID)
G(V_ID, VRW_ID)
G(PHI_ID, ARW_ID)
G(BG_ID, BGSTD_ID)
G(BA_ID, BASTD_ID)
```

核心姿态可观测性通道来自：

```text
GNSS position residual
  -> position error
  -> velocity error
  -> attitude error
  -> gyro bias / accel bias
```

其中最关键的块是：

```text
F(P_ID, V_ID) = I
F(V_ID, PHI_ID) = skew(cbn * accel)
F(PHI_ID, BG_ID) = -cbn
F(V_ID, BA_ID) = cbn
```

如果这些块存在但符号、坐标或反馈方向不一致，位置观测无法稳定地修正姿态。

### 3.3 KF-GINS GNSS 更新

参考文件：

```text
D:\Code\kf_gins_used_in_paper\src\kf-gins\gi_engine.cpp
```

重点参考函数：

```text
GIEngine::gnssUpdate()
GIEngine::EKFUpdate()
```

KF-GINS 中 GNSS 位置新息：

```text
antenna_pos = pvacur_.pos + Dr_inv * pvacur_.att.cbn * options_.antlever
dz = Dr * (antenna_pos - gnssdata.blh)
```

观测矩阵：

```text
H(P_ID) = I
H(PHI_ID) = skew(pvacur_.att.cbn * options_.antlever)
```

当前 CT 若不使用杆臂，`H(PHI_ID)` 对 GNSS 单点位置残差可以为 0，但姿态仍应通过误差传播链间接可观。

### 3.4 KF-GINS 状态反馈

参考文件：

```text
D:\Code\kf_gins_used_in_paper\src\kf-gins\gi_engine.cpp
```

重点参考函数：

```text
GIEngine::stateFeedback()
```

KF-GINS 反馈逻辑：

```text
pos -= DRi(pos) * delta_r
vel -= delta_v
qbn = Exp(delta_phi) * qbn
gyrbias += delta_bg
accbias += delta_ba
dx = 0
```

CT 必须先明确 `delta_*` 的定义：

- 如果 CT 的 `delta_*` 表示 KF-GINS 中的误差状态 `dx`，反馈应采用 KF-GINS 的符号。
- 如果 CT 的 `delta_*` 表示直接修正量，则 GNSS 残差和传播方程必须整体使用“修正量”约定。

当前 CT 注入逻辑中存在需要重点核对的地方：

```text
nominal_state.q_nb = nominal_rot * Exp(delta_theta)
nominal_state.vel_ned += delta_vel
nominal_state.blh = LocalToGlobal(origin, nominal_local + delta_pos)
nominal_state.bg += delta_bg
nominal_state.ba += delta_ba
```

这和 KF-GINS 的 `pos -= delta_r`、`vel -= delta_v` 并不完全一致。这里是本轮最优先排查点。

## 4. CT 当前对应实现路径

### 4.1 CT 主流程

```text
D:\Code\CT_FGO_SIM_zAxisPro\src\core\system.cpp
```

包含：

```text
配置读取
初始对准
yaw feedback
问题构建
GNSS 因子添加
误差状态传播因子添加
外迭代
误差状态注入
结果输出
debug 输出
```

### 4.2 CT 系统配置与状态

```text
D:\Code\CT_FGO_SIM_zAxisPro\include\ct_fgo_sim\core\system.h
```

包含：

```text
SystemConfig
control_points_
delta_theta_nodes_
delta_vel_nodes_
delta_pos_nodes_
delta_bg_nodes_
delta_ba_nodes_
interval_cache_
initial_yaw_feedback 状态
外迭代 debug 状态
```

### 4.3 CT 误差状态传播因子

当前正在使用的区间传播因子：

```text
D:\Code\CT_FGO_SIM_zAxisPro\include\ct_fgo_sim\factors\error_state_interval_factor.h
```

其残差形式：

```text
xj - Phi * xi
```

其中状态顺序为：

```text
pos, vel, phi, bg, ba
```

### 4.4 CT 误差传播缓存

```text
D:\Code\CT_FGO_SIM_zAxisPro\include\ct_fgo_sim\navigation\interval_propagation.h
```

```text
D:\Code\CT_FGO_SIM_zAxisPro\src\navigation\interval_propagation.cpp
```

当前已经包含：

```text
BuildF()
BuildG()
BuildQc()
DiscretizeLinearSystem()
BuildIntervalPropagationCache()
```

这里是对齐 KF-GINS `GIEngine::insPropagation()` 的主要修改位置。

### 4.5 CT GNSS 位置误差状态因子

```text
D:\Code\CT_FGO_SIM_zAxisPro\include\ct_fgo_sim\factors\error_state_gnss_factor.h
```

当前残差形式：

```text
res = nominal_pos + delta_p - measured_pos
```

该残差方向必须与 CT 的 `delta_pos` 反馈方向统一。

### 4.6 CT 备用传播因子

```text
D:\Code\CT_FGO_SIM_zAxisPro\include\ct_fgo_sim\factors\error_state_process_factor.h
```

该文件也包含一套 `BuildF/BuildG/BuildQc/BuildPhi/BuildQd`，但当前主流程更依赖 `interval_propagation.cpp` 中的区间传播缓存。

建议不要同时维护两套传播公式。后续可以将公共传播矩阵逻辑抽成一个复用模块，避免两份实现漂移。

### 4.7 CT 静态对准与机械编排

```text
D:\Code\CT_FGO_SIM_zAxisPro\include\ct_fgo_sim\navigation\mechanization.h
```

```text
D:\Code\CT_FGO_SIM_zAxisPro\src\navigation\mechanization.cpp
```

用于：

```text
静态对准
初始姿态
初始 bg/ba
导航机械编排
```

## 5. 实施步骤

### 步骤 1：固定误差状态定义

先在 CT 文档和代码注释中明确：

```text
delta_pos, delta_vel, delta_theta, delta_bg, delta_ba
```

到底是：

```text
KF-GINS 风格误差状态 dx
```

还是：

```text
直接加到 nominal 上的修正量
```

建议采用 KF-GINS 风格误差状态，便于复用 KF-GINS。

### 步骤 2：按 KF-GINS 修正 CT 的反馈方向

若采用 KF-GINS 风格误差状态，则 CT 的反馈应对齐：

```text
pos_new = pos_old - delta_pos
vel_new = vel_old - delta_vel
q_new = Exp(delta_theta) * q_old
bg_new = bg_old + delta_bg
ba_new = ba_old + delta_ba
```

注意：

```text
pos 是局部 NED 或 BLH 增量时，需要使用当前 CT 的 Earth::GlobalToLocal / LocalToGlobal 对应处理。
```

当前 CT 里 `pos += delta_pos`、`vel += delta_vel` 是需要优先核对或改正的地方。

### 步骤 3：按 KF-GINS 核对 `BuildF`

对照：

```text
D:\Code\kf_gins_used_in_paper\src\kf-gins\gi_engine.cpp
```

修改或确认：

```text
D:\Code\CT_FGO_SIM_zAxisPro\src\navigation\interval_propagation.cpp
```

重点核对：

```text
NED 速度顺序是否一致：vn, ve, vd
BLH 顺序是否一致：lat, lon, h
cbn / cnb 是否一致
specific force 是否已扣除 ba
gyro 是否已扣除 bg
F(V,PHI) 符号是否与反馈方向配套
F(V,BA) 符号是否与反馈方向配套
F(PHI,BG) 符号是否与反馈方向配套
```

### 步骤 4：按 KF-GINS 核对 GNSS 因子

当前 CT：

```text
res = nominal_pos + delta_p - measured_pos
```

若 `delta_p` 改为 KF-GINS 误差状态，则 GNSS 因子有两种等价做法：

方案 A：保持 residual 为 `predicted - measured`，但预测位置写成：

```text
nominal_pos - delta_p
```

方案 B：保持预测为 `nominal_pos + delta_p`，但将 `delta_p` 解释为修正量，而不是 KF 误差状态。

建议采用方案 A，因为它与 KF-GINS 的 `dz = predicted - measured` 和 `pos -= delta_r` 一致。

### 步骤 5：外迭代后清零误差状态并重建传播缓存

CT 当前已有类似逻辑：

```text
InjectCurrentErrorStateIntoNominalTrajectory()
delta_* = 0
BuildIntervalPropagationCache()
```

需要保留。

### 步骤 6：增加诊断输出

建议输出：

```text
outer_iteration
gnss_residual_rms_before
gnss_residual_rms_after
max |delta_theta|
max |delta_vel|
max |delta_pos|
max |delta_bg|
max |delta_ba|
roll_slope_deg_per_s
pitch_slope_deg_per_s
yaw_slope_deg_per_s
```

用于确认：

```text
位置残差下降时，姿态误差状态是否同步被激活；
外迭代后首段 roll/pitch 斜率是否减小；
bg/ba 是否出现合理修正。
```

## 6. 验证用例

优先使用当前问题样本：

```text
YuHangTuiChe\20260122_121901_use\transformed1cut1
```

CT 输出路径：

```text
D:\googleYun\30Code\chapter5_module2\organized_data\YuHangTuiChe\zaxispro\output\20260122_121901_use\transformed1cut1
```

KF 输出路径：

```text
D:\googleYun\30Code\chapter5_module2\prepared_output\kf-gins\YuHangTuiChe\20260122_121901_use\transformed1cut1
```

当前静态对准约定：

```text
aligntime = 100.0 s
导航输出从静态对准结束后开始
```

验证标准：

```text
CT nominal_nav.txt 首时刻应约为 2487.931
KF nominal_nav.txt 首时刻应约为 2487.932
CT roll/pitch 初段斜率应明显减小
CT vs KF roll/pitch 差异不应继续出现明显单调漂移
```

## 7. CT 可执行文件版本整理

### 7.1 当前 CT_FGO_SIM_zAxisPro 下的 exe

真正的 `ct_fgo_sim_main.exe`：

```text
D:\Code\CT_FGO_SIM_zAxisPro\build\Debug\ct_fgo_sim_main.exe
D:\Code\CT_FGO_SIM_zAxisPro\build\Release\ct_fgo_sim_main.exe
D:\Code\CT_FGO_SIM_zAxisPro\build_reconfig\Release\ct_fgo_sim_main.exe
D:\Code\CT_FGO_SIM_zAxisPro\build_zaxis\Release\ct_fgo_sim_main.exe
D:\Code\CT_FGO_SIM_zAxisPro\build_zAxisPro\Release\ct_fgo_sim_main.exe
```

其中近期手动调试使用的是：

```text
D:\Code\CT_FGO_SIM_zAxisPro\build_reconfig\Release\ct_fgo_sim_main.exe
```

### 7.2 当前 CT_FGO_SIM 下的 exe

```text
D:\Code\CT_FGO_SIM\build\Debug\ct_fgo_sim_main.exe
D:\Code\CT_FGO_SIM\build\Release\ct_fgo_sim_main.exe
```

### 7.3 历史归档版本

```text
D:\Code\ct_fgo_versions\ct_fgo_sim_stable_20260320_final_3_gbd00aeb\build\Release\ct_fgo_sim_main.exe
D:\Code\ct_fgo_versions\ct_fgo_sim_stable_20260320_final_5_gedf0502_dirty\build\Release\ct_fgo_sim_main.exe
D:\Code\ct_fgo_versions\ct_fgo_sim_stable_20260320_final_6_g4bbe7a2_dirty\build\Release\ct_fgo_sim_main.exe
D:\Code\ct_fgo_versions\ct_fgo_sim_stable_20260320_final_dirty\build\Release\ct_fgo_sim_main.exe
D:\Code\ct_fgo_versions\ct_fgo_sim_zaxis_stable_20260320_final_1_gd337cd9_dirty\build\Release\ct_fgo_sim_main.exe
D:\Code\ct_fgo_versions\ct_fgo_sim_zaxis_stable_20260320_final_dirty\build\Release\ct_fgo_sim_main.exe
```

仅统计 `ct_fgo_sim_main.exe`，当前共发现 13 个。

不计入统计的文件包括：

```text
CompilerIdCXX.exe
attitude_priority_main.exe
```

## 8. Chapter 5 当前版本生成与调用机制

### 8.1 版本注册中心

```text
D:\googleYun\30Code\chapter5_module2\modules\chapter5_ct_fgo_registry.m
```

当前注册了 3 个逻辑版本：

```text
nhc
zaxispro
single-layer-param-laws
```

当前 exe 映射：

```text
nhc -> D:\Code\CT_FGO_SIM\build\Release\ct_fgo_sim_main.exe
zaxispro -> D:\Code\CT_FGO_SIM_zAxisPro\build\Release\ct_fgo_sim_main.exe
single-layer-param-laws -> D:\Code\ct_fgo_versions\ct_fgo_sim_stable_20260320_final_6_g4bbe7a2_dirty\build\Release\ct_fgo_sim_main.exe
```

注意：

```text
registry 中 zaxispro 当前不是 build_reconfig。
如果批量运行要使用最新调试版，需要把 registry 改为 build_reconfig，或将最新代码正式编译到 build\Release。
```

### 8.2 批量配置生成

脚本：

```text
D:\googleYun\30Code\chapter5_module2\modules\chapter5_ct_fgo_prepare_configs.m
```

输出配置目录：

```text
D:\googleYun\30Code\chapter5_module2\prepared_configs\<version>\<dataset>\<rel_slug>\ct_fgo_sim.yaml
```

输出结果目录：

```text
D:\googleYun\30Code\chapter5_module2\prepared_output\<version>\<dataset>\<relative_key>
```

### 8.3 批量运行缺失 CT 输出

入口 cmd：

```text
D:\googleYun\30Code\chapter5_module2\run_missing_outputs.cmd
```

当前命令只跑：

```text
VersionNames = zaxispro
```

实际 MATLAB 调用：

```text
run_chapter5_module2_run_missing_outputs('VersionNames', 'zaxispro')
```

再进入：

```text
D:\googleYun\30Code\chapter5_module2\run_chapter5_module2_run_missing_outputs.m
D:\googleYun\30Code\chapter5_module2\modules\chapter5_ct_fgo_run_missing_outputs.m
```

当前运行扫描目录：

```text
D:\googleYun\30Code\chapter5_module2\organized_data
```

这意味着当前有两套资产路径需要区分：

```text
prepared_configs / prepared_output
organized_data
```

KF-GINS 当前从 `prepared_configs` 读取 CT 配置生成 KF 配置；CT 的 `run_missing_outputs.cmd` 当前从 `organized_data` 扫描缺失输出。后续建议统一，否则 CT/KF 的起算时间、配置参数、输出目录可能继续不一致。

## 9. 推荐下一步

建议下一步只做最小闭环：

1. 将 `zaxispro` 的批量 exe 明确切到当前调试版或重新编译到正式 `build\Release`。
2. 按 KF-GINS 反馈定义统一 CT 的 `delta_*` 语义。
3. 修改 CT 的 GNSS 因子和 `InjectCurrentErrorStateIntoNominalTrajectory()`，保证残差方向和反馈方向一致。
4. 用 `YuHangTuiChe\20260122_121901_use\transformed1cut1` 单条验证。
5. 只在该单条通过后，再批量跑其余样本。

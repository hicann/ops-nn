# ApplyFtrl

> **SoC 扩展说明（务必先读）**
> 主线 `optim/apply_ftrl` 的 `ApplyFtrl` 仅提供 **ascend950（arch35 / DAV_3510，atvoss DAG）** kernel，开源仓内**无 ascend910b 原生 kernel**。
> 本实验算子（`experimental/optim/apply_ftrl/`）为 **ascend910b（Atlas A2/A3，DAV_2201）** 补一份开源的**原生 AscendC 经典 intrinsic kernel**，registry-invoke 方式自带 Kernel/Tiling，**复用同名 OpType `ApplyFtrl`**（端口结构与主线 proto 一致），与主线通过 `ENABLE_EXPERIMENTAL` 开关**互斥构建**，不改动 arch35 既有实现。
>
> **接口现状**：`ApplyFtrl` 是 **GE 图模式 / TensorFlow 算子**（对标 `tf.raw_ops.ApplyFtrl`），**无 CANN 已部署的 aclnn 接口、无 PyTorch 等价**。接口真值源为 [`op_graph/apply_ftrl_proto.h`](../../../optim/apply_ftrl/op_graph/apply_ftrl_proto.h)（GE IR `REG_OP(ApplyFtrl)`，复用主线）。

## 产品支持情况

| 产品 | 是否支持 | 本实验算子（experimental）覆盖 |
| :----------------------------------------- | :------:| :---------------------------- |
| <term>Ascend 950PR/Ascend 950DT</term> | √ | 由主线 `optim/apply_ftrl`（arch35）提供，**不在本实验范围** |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term> | √ | **本实验新增**（ascend910b / DAV_2201 原生 kernel） |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> | √ | **本实验新增**（ascend910b / DAV_2201 原生 kernel） |
| <term>Atlas 200I/500 A2 推理产品</term> | × | — |
| <term>Atlas 推理系列产品</term> | √ | 由主线提供 |
| <term>Atlas 训练系列产品</term> | √ | 由主线提供 |

> 上表反映 `ApplyFtrl` 算子整体（主线 + 本扩展）的产品支持。**本实验任务仅新增 ascend910b（Atlas A2/A3，DAV_2201）原生 AscendC kernel**；ascend950（arch35）路径由主线 `optim/apply_ftrl` 已提供，本任务不改动。

## 功能说明

- 算子功能：根据 FTRL-proximal（Follow-The-Regularized-Leader）方案**原地更新**参数张量 `var`，并同步**原地更新**梯度平方累加器 `accum` 与校正项 `linear`。功能对标 `tf.raw_ops.ApplyFtrl`，常用于带 L1/L2 正则的在线学习 / 大规模稀疏特征（推荐、广告 CTR）模型的参数更新。
- 计算公式（逐元素；下式右侧 `accum`/`linear`/`var` 为**更新前**值，`var` 的更新使用**更新后**的 `linear`，记为 $linear_t$）：

  $$
  accum\_new = accum + grad \times grad
  $$

  $$
  linear_t = linear + grad - \frac{accum\_new^{-lr\_power} - accum^{-lr\_power}}{lr} \times var
  $$

  $$
  quadratic = \frac{accum\_new^{-lr\_power}}{lr} + 2 \times l2
  $$

  $$
  var =
  \begin{cases}
  \dfrac{sign(linear_t) \times l1 - linear_t}{quadratic}, & |linear_t| > l1 \\[2ex]
  0, & |linear_t| \le l1
  \end{cases}
  $$

  $$
  accum = accum\_new
  $$

  其中 $sign(x) \in \{-1, 0, +1\}$（$x>0 \to +1$，$x<0 \to -1$，$x=0 \to 0$）；$var$、$accum$、$linear$ 三者均原地写回。
- 实现要点（ascend910b 原生 kernel）：
  - 逐元素独立计算，无跨元素/跨核归约 → 天然确定性。
  - **内部统一 fp32 计算**：bf16/fp16 输入在 kernel 入口 Cast 到 fp32 完成全部中间计算（含 `accum_new`、幂运算、sign + 软阈值、`quadratic`、除法），末尾 Cast 回原 dtype 输出；fp32 直接计算。
  - 幂运算 $a^{-lr\_power}$ 采用 `Exp(Muls(Ln(a), -lr\_power))` 分解（与 canndev `apply_ftrl_d.py` 的 `_pow = exp(index·ln(data))` 一致，要求底数 $a>0$）。
  - 三张量 `var`/`accum`/`linear` 均为 Ref Tensor：`var` 经声明输出端口写回，`accum`/`linear` 经各自 input ref 端口原地写回。

## 参数说明

<table style="table-layout: fixed; width: 1000px"><colgroup>
<col style="width: 110px">
<col style="width: 170px">
<col style="width: 360px">
<col style="width: 240px">
<col style="width: 90px">
</colgroup>
<thead>
  <tr>
    <th>参数名</th>
    <th>输入/输出/属性</th>
    <th>描述</th>
    <th>数据类型</th>
    <th>数据格式</th>
  </tr></thead>
<tbody>
  <tr><td>var</td><td>输入（Ref，原地更新）</td><td>待更新的参数张量，应来自 Variable。对应公式 var。shape 与 accum/linear/grad 一致。</td><td>BFLOAT16、FLOAT16、FLOAT</td><td>ND</td></tr>
  <tr><td>accum</td><td>输入（Ref，原地更新）</td><td>梯度平方累加器，应来自 Variable。对应公式 accum，要求各元素 ≥ 0。</td><td>BFLOAT16、FLOAT16、FLOAT</td><td>ND</td></tr>
  <tr><td>linear</td><td>输入（Ref，原地更新）</td><td>校正项，应来自 Variable。对应公式 linear。</td><td>BFLOAT16、FLOAT16、FLOAT</td><td>ND</td></tr>
  <tr><td>grad</td><td>输入</td><td>当前步梯度张量。对应公式 grad。</td><td>BFLOAT16、FLOAT16、FLOAT</td><td>ND</td></tr>
  <tr><td>lr</td><td>输入</td><td>学习率（缩放因子），标量（0-D 或 1 元素 1-D）。对应公式 lr，要求 ≠ 0。</td><td>BFLOAT16、FLOAT16、FLOAT</td><td>ND</td></tr>
  <tr><td>l1</td><td>输入</td><td>L1 正则化系数，标量。对应公式 l1。</td><td>BFLOAT16、FLOAT16、FLOAT</td><td>ND</td></tr>
  <tr><td>l2</td><td>输入</td><td>L2 正则化系数，标量。对应公式 l2。</td><td>BFLOAT16、FLOAT16、FLOAT</td><td>ND</td></tr>
  <tr><td>lr_power</td><td>输入</td><td>缩放因子的幂次，标量。对应公式 lr_power。</td><td>BFLOAT16、FLOAT16、FLOAT</td><td>ND</td></tr>
  <tr><td>use_locking</td><td>属性</td><td>是否使用锁机制保护更新操作，默认 False。仅支持 False。</td><td>Bool</td><td>-</td></tr>
  <tr><td>var</td><td>输出</td><td>更新后的参数张量，与输入 var 共享存储。算子仅声明此 1 个输出端口；accum、linear 经各自 input ref 端口原地更新。</td><td>BFLOAT16、FLOAT16、FLOAT</td><td>ND</td></tr>
</tbody></table>

## 支持的数据类型

支持 3 种浮点 dtype。**8 个输入与输出 dtype 必须严格一致**（无类型推导/提升，无混合精度入参）。

| 数据类型 | var/accum/linear/grad | lr/l1/l2/lr_power | 输出 var | 内部计算 dtype |
|---------|------------------------|-------------------|----------|----------------|
| FLOAT (fp32) | FLOAT | FLOAT | FLOAT | FLOAT（直接计算） |
| FLOAT16 (fp16) | FLOAT16 | FLOAT16 | FLOAT16 | **FLOAT32**（入口 Cast 升精度，末尾 Cast 回 fp16） |
| BFLOAT16 (bf16) | BF16 | BF16 | BF16 | **FLOAT32**（入口 Cast 升精度，末尾 Cast 回 bf16） |

## 约束说明

- **数据类型**：var/accum/linear/grad/lr/l1/l2/lr_power 与输出 var 的数据类型必须完全一致，取值 ∈ {BFLOAT16, FLOAT16, FLOAT}；无类型推导，无混合精度入参。
- **shape**：var/accum/linear/grad 的 shape 必须完全一致（逐元素，rank 0-8）；lr/l1/l2/lr_power 必须为 **0-D 或 1 元素 1-D** 标量，广播到张量 shape。张量之间不广播。
- **空 Tensor / 0 元素**：infershape 直通（`var_out.shape = var.shape`）；kernel 对 0 元素安全短路，无计算、不写越界。
- **属性**：use_locking 仅支持 False。
- **原地更新**：var、accum、linear 均为 Ref Tensor，算子执行后原地更新；调用方需将其视为可被算子修改的存储。
- **值域前置条件**（算子内**不做**运行时值域校验，由调用方保证）：`accum ≥ 0`、`lr ≠ 0`。
- **奇点行为（按 IEEE 传播，与 TensorFlow 原生 `ApplyFtrl` 一致，需调用方规避）**：
  - `accum_new = 0` 且 `lr_power > 0` 时，$accum\_new^{-lr\_power}$ 产生 Inf/NaN；
  - `quadratic = 0` 时，除法产生 Inf/NaN；
  - `lr = 0` 时，除以 0 产生 Inf/NaN；
  - `grad` 含 NaN 时，沿元素传播到 accum/linear 输出。
  - 软阈值边界：`|linear_t| ≤ l1` 时 `var = 0`（即使 quadratic 异常也短路为 0，与 Select 语义一致）。
- **精度标准**（浮点计算类社区标准，MERE/MARE 单标杆比对；MERE < Threshold 且 MARE < 10×Threshold）：fp32 Threshold 2⁻¹³ ≈ 1.22e-4、fp16 2⁻¹⁰ ≈ 9.77e-4、bf16 2⁻⁷ ≈ 7.81e-3。

## 调用说明

> `ApplyFtrl` **无 CANN 已部署的 aclnn 接口、无 PyTorch 等价**。可经 **GE 图模式** 下发，或对本实验 registry-invoke 自带 kernel 走 **kernel 直调（`<<<>>>`）** 调用。下表样例链接均指向本目录 `examples/`。

| 调用方式 | 是否支持 | 调用样例 | 说明 |
|---------|---------|---------|------|
| GE 图模式（静态/动态 shape） | √ | [test_geir_apply_ftrl.cpp](./examples/test_geir_apply_ftrl.cpp) | 通过[算子 IR](../../../optim/apply_ftrl/op_graph/apply_ftrl_proto.h) 构图方式调用 `ApplyFtrl`（按 `var → accum → linear → grad → lr → l1 → l2 → lr_power` 串接 8 个 `Data` 输入，scalar 输入 shape `{1}`，`set_attr_use_locking(false)`）。运行需将算子编译安装进 OPP 包。 |
| Kernel 直调（`<<<>>>`） | √（**本实验已在真实 NPU 验证通过**） | [test_kernel_direct_apply_ftrl.asc](./examples/test_kernel_direct_apply_ftrl.asc) | 直接复用本算子 `op_kernel/apply_ftrl_kernel.h` 的 `NsApplyFtrl::ApplyFtrl<T, PAD_TAIL, HAS_L1>` 计算类，经 `<<<>>>` 在 ascend910b 真实 NPU 上运行并与 CPU golden 比对。一键运行见 [examples/run.sh](./examples/run.sh)。 |
| aclnn 调用 | × | — | 无 CANN 已部署 aclnn 接口（ACLNNTYPE=`aclnn_exclude`）。 |
| torch_npu / torch.compile | × | — | 无 PyTorch 等价。 |
| TensorFlow 算子 | √（语义兼容） | — | 对标 `tf.raw_ops.ApplyFtrl`。 |

- aclnn 接口设计（派生原型，**非 CANN 已部署**）见 [docs/aclnnApplyFtrl.md](./docs/aclnnApplyFtrl.md)。
- 更完整的多 dtype / 多 shape 验证（含尾块、l1==0 快路、空 Tensor 等）见本算子 [tests/ut/op_kernel](./tests/ut/op_kernel)（CPU 孪生 tikicpulib 直跑主线 kernel）与 [tests/st/aclnnApplyFtrl/atk_ApplyFtrl.json](./tests/st/aclnnApplyFtrl/atk_ApplyFtrl.json)（ATK 真机用例集，全 dtype × shape × 标量 0-D/[1] 桶）。

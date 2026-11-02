# InplaceApplyKerasMomentum

## 产品支持情况

| 产品 | 是否支持 |
|:-----|:-------:|
| <term>Ascend 950PR&950DT系列产品</term> | √ |
| <term>Atlas A3系列产品</term>| x |
| <term>Atlas A2系列产品</term> | x |
| <term>Atlas 200I/500 A2推理产品</term> | × |
| <term>Atlas推理系列产品</term> | × |
| <term>Atlas训练系列产品</term> | × |

## 功能说明

- 算子功能：执行Keras Momentum优化器的单步参数更新。根据动量系数 momentum、学习率 lr 和梯度 grad 更新动量累积量 accum，并按标准模式或 Nesterov 模式原地更新权重参数 var（inplace 语义）。双输出版本：var 和 accum 均显式输出并 inplace 写回。对标 TensorFlow 中 `tf.raw_ops.ResourceApplyKerasMomentum` 接口的计算语义。

- 计算公式：

  $$
  accum_{new} = momentum \cdot accum - lr \cdot grad
  $$

  - 标准模式（use_nesterov = False）：

  $$
  var_{new} = var + accum_{new}
  $$

  - Nesterov 模式（use_nesterov = True）：

  $$
  var_{new} = var + momentum \cdot accum_{new} - lr \cdot grad
  $$

  其中 `momentum` 为动量系数，`lr` 为学习率，`grad` 为当前梯度，`accum` 为动量累积量，`var` 为待更新的权重参数。状态写回（inplace 语义）：`accum ← accum_new`，`var ← var_new`。

## 参数说明

| 参数名 | 输入/输出/属性 | 描述 | 数据类型 | 数据格式 |
|:-------|:--------------|:-----|:--------|:--------|
| var | 输入 | 待更新的权重参数，对应公式中的 var。来自 Variable。 | FLOAT16、FLOAT、BFLOAT16 | ND |
| accum | 输入 | 动量累积量，对应公式中的 accum。shape/dtype 必须与 var 一致。 | FLOAT16、FLOAT、BFLOAT16 | ND |
| lr | 输入 | 学习率，对应公式中的 lr。scalar Tensor。 | FLOAT | ND |
| grad | 输入 | 当前梯度 Tensor，对应公式中的 grad。shape/dtype 必须与 var 一致。 | FLOAT16、FLOAT、BFLOAT16 | ND |
| momentum | 输入 | 动量系数，对应公式中的 momentum。scalar Tensor。 | FLOAT | ND |
| var | 输出 | 更新后的 var Tensor，与输入 var 共享 Device 内存（inplace）。 | FLOAT16、FLOAT、BFLOAT16 | ND |
| accum | 输出 | 更新后的 accum Tensor，与输入 accum 共享 Device 内存（inplace）。 | FLOAT16、FLOAT、BFLOAT16 | ND |
| use_locking | 属性 | 是否加锁。NPU 单流执行忽略，保留属性兼容 TF。默认 False。 | Bool | - |
| use_nesterov | 属性 | 是否使用 Nesterov 动量。True 时切换 var 更新公式为 Nesterov 模式。默认 False。 | Bool | - |

## 约束说明

- var、accum、grad 必须具有相同的数据类型和形状，且 rank ≤ 8；超过 8 维时 Tiling 返回失败。
- var、accum、grad 仅支持连续的 ND 布局，不支持广播（broadcast）。
- lr、momentum 必须为 scalar Tensor 或 shape 为 `{1}` 的 Tensor，数据类型为 FLOAT；GE 图模式不支持 rank=0，请使用 `{1}` 构造。
- 空 Tensor（元素数为 0）合法：算子不更新数据，输出与输入保持一致。
- 输出 var、accum 与输入 var 同 shape（inplace 语义）。
- 支持通过 `-1`/`-2` 动态 shape/rank 标记构图：`-1` 表示动态维度，`-2` 表示动态 Rank。构图时 Data/输出 desc 使用动态标记，`RunGraph` 的输入 Tensor 必须携带实际 shape（例如 `[4,4]`），不能把 `-1`/`-2` 直接作为 feed shape；动态维度展开后仍需满足上述形状、dtype 与 rank 约束。
- 容量边界：var/accum/grad 的元素总数上限为 `4,294,967,295`（`UINT32_MAX`）；超过时 Tiling 返回失败。`4G 元素 = 2^32 = 4,294,967,296` 已超过上限，应被拒绝。
- 本算子不额外声明 `4GiB` 逻辑字节上限：`4GiB` 对 FP32 为 `2^30` 元素、对 FP16/BF16 为 `2^31` 元素，均在元素数上限内；能否执行还取决于设备 HBM 等物理容量。
- Tiling 内部将 `blockNum` 限制在 `65535`；需要更多块的超大 shape 通过增大每块 `blockFormer` 处理，不因块数拒绝 shape。
- 数值语义：FLOAT16/BFLOAT16 输入统一升 FP32 计算；BFLOAT16 的指数范围与 FP32 相同，中间量（如 `lr*grad`）可能在 FP32 溢出为 ±Inf，这是本算子声明的中间精度语义，不作为 FP64 数学真值。
- FLOAT16 的 var/accum 结果在 CAST_ROUND 回写前饱和到 `[-65504, 65504]`，溢出时得到 ±65504 而不是 ±Inf；BFLOAT16 与 FLOAT 不做该饱和。当前不保证 NaN/Inf 输入的数值行为，相关输入不在验证范围内。

## 拒绝条件与错误码

Tiling 阶段发现非法输入时返回 `GRAPH_FAILED`，并上报以下预定义错误码：

| 拒绝条件 | 错误码 |
|:---------|:-------|
| var/accum/grad 的 rank > 8 | `EZ0012` |
| var/accum/grad shape 不一致 | `EZ0010` |
| lr/momentum 既非 scalar 也非 shape `{1}` | `EZ0010` |
| var 的 storage format 不是 ND | `EZ0017` |
| var 的 dtype 不属于 FLOAT16/FLOAT/BFLOAT16，或 lr/momentum 不是 FLOAT | `EZ0019` |
| var 的元素总数小于 0 或大于 `4294967295` | `EZ0026` |

补充说明：

- 空 Tensor（元素数为 0）不进入拒绝流程，返回成功且不更新数据。
- 非连续 ND 输入不在本算子保证范围内，当前没有对应的运行时拒绝检查；调用方需保证输入为连续 ND 布局。

## 调用说明

| 调用方式 | 调用样例 | 说明 |
|:--------|:--------|:-----|
| GE 图模式 | [test_geir_inplace_apply_keras_momentum](examples/arch35/test_geir_inplace_apply_keras_momentum.cpp) | 通过[算子 IR](op_graph/inplace_apply_keras_momentum_proto.h) 构图方式调用 InplaceApplyKerasMomentum 算子。 |

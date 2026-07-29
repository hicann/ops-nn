# InplaceApplyKerasMomentum

## 产品支持情况

| 产品 | 是否支持 |
|:-----|:-------:|
| <term>Ascend 950PR/Ascend 950DT</term> | √ |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term> | √ |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> | √ |
| <term>Atlas 200I/500 A2 推理产品</term> | × |
| <term>Atlas 推理系列产品</term> | × |
| <term>Atlas 训练系列产品</term> | × |

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
| use_locking | 属性 | 是否加锁。NPU 单流执行忽略，保留属性兼容 TF。默认 False。 | Bool | - |
| use_nesterov | 属性 | 是否使用 Nesterov 动量。True 时切换 var 更新公式为 Nesterov 模式。默认 False。 | Bool | - |
| var | 输出 | 更新后的 var Tensor，与输入 var 共享 Device 内存（inplace）。 | FLOAT16、FLOAT、BFLOAT16 | ND |
| accum | 输出 | 更新后的 accum Tensor，与输入 accum 共享 Device 内存（inplace）。 | FLOAT16、FLOAT、BFLOAT16 | ND |

## 约束说明

- var、accum、grad 必须具有相同的数据类型和形状。
- lr、momentum 必须为 scalar Tensor，数据类型为 FLOAT。
- 输出 var、accum 与输入 var 同 shape（inplace 语义）。
- 不支持广播（broadcast）。

## 调用说明

| 调用方式 | 调用样例 | 说明 |
|:--------|:--------|:-----|
| GE 图模式 | [test_geir_inplace_apply_keras_momentum](examples/arch35/test_geir_inplace_apply_keras_momentum.cpp) | 通过[算子 IR](op_graph/inplace_apply_keras_momentum_proto.h) 构图方式调用 InplaceApplyKerasMomentum 算子。 |

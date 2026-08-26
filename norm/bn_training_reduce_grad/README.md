# BNTrainingReduceGrad

## 产品支持情况

|产品             |  是否支持  |
|:-------------------------|:----------:|
|  <term>Ascend 950PR/Ascend 950DT</term>   |     √    |
|  <term>Atlas A3训练系列产品/Atlas A3推理系列产品</term>   |     √    |
|  <term>Atlas A2训练系列产品/Atlas A2推理系列产品</term>     |     √    |
|  <term>Atlas 200I/500 A2推理产品</term>    |     √    |
|  <term>Atlas推理系列产品</term>    |     √    |
|  <term>Atlas训练系列产品</term>    |     √    |

## 功能说明

- 算子功能：批归一化训练反向的reduce-grad阶段（Batch Normalization Training Reduce Grad）。给定上层梯度`grads`、前向输入`x`、BNTrainingUpdateGrad产出的逐通道统计量`diff_scale`/`diff_offset`，以及`scale`/`batch_mean`/`batch_variance`，计算回传给`x`的梯度`y`。该算子在GE图内由`FusedBatchNormGrad`融合展开生成，与BNTrainingUpdateGrad配套使用。

- 计算公式：

  设grads的shape为 [N, C, R...]（dim0为N、dim1为C、后导维展平为R），num = N * R：

  $$
  sqrtVar = \sqrt{batch\_variance + ε}
  $$

  $$
  multiplier = {diff\_scale * (-{1\over num})\over sqrtVar}
  $$

  $$
  addend = {batch\_mean\over sqrtVar} * {diff\_scale\over num} + diff\_offset * (-{1\over num})
  $$

  $$
  mulScale = {scale\over sqrtVar}
  $$

  $$
  y = ((grads + multiplier * x) + addend) * mulScale
  $$

  `multiplier`/`addend`/`mulScale`均为长度C的向量，沿C维广播。fp16/bf16输入在算子内升fp32计算、单次舍入写回。

## 参数说明

| 参数名 | 输入/输出/属性 | 描述 | 数据类型 | 数据格式 |
|---|---|---|---|---|
| grads | 输入 | 上层回传的梯度张量。shape为 [N, C, R...]，支持 ≥2维，dim0为N、dim1为C、后导维展平为归一化轴R。不支持空tensor（各维必须为正数）。 | FLOAT32、FLOAT16、BFLOAT16 | ND |
| x | 输入 | 前向输入张量，shape与数据类型均与`grads`一致。 | FLOAT32、FLOAT16、BFLOAT16 | ND |
| diff_scale | 输入 | 逐通道统计量（BNTrainingUpdateGrad的scale梯度输出），shape为 [C]，元素数必须等于grads的dim1。 | FLOAT32 | ND |
| diff_offset | 输入 | 逐通道统计量（BNTrainingUpdateGrad的offset梯度输出），shape为 [C]，元素数必须等于grads的dim1。 | FLOAT32 | ND |
| scale | 输入 | 逐通道缩放因子，shape为 [C]，元素数必须等于grads的dim1。 | FLOAT32 | ND |
| batch_mean | 输入 | 前向逐通道均值，shape为 [C]，元素数必须等于grads的dim1。 | FLOAT32 | ND |
| batch_variance | 输入 | 前向逐通道方差，shape为 [C]，元素数必须等于grads的dim1。 | FLOAT32 | ND |
| epsilon | 可选属性 | 添加到batch_variance上再开方的小量，默认0.0001。 | FLOAT32 | - |
| y | 输出 | 回传给x的梯度，shape与数据类型均与`grads`一致。 | FLOAT32、FLOAT16、BFLOAT16 | ND |

## 约束说明

- `grads`与`x`的dtype、shape必须一致；五路统计量恒为FLOAT32且元素数等于C。
- 不支持空tensor（num = N*R为公式分母，任一维为0即除零，tiling阶段结构化拒绝）。
- `batch_variance + epsilon`必须为正数（开方定义域），算子不对非法统计量做额外守卫，与A2行为一致。

## 调用说明

| 调用方式 | 样例代码 | 说明 |
| --- | --- | --- |
| 图模式调用 | [test_geir_bn_training_reduce_grad.cpp](examples/arch35/test_geir_bn_training_reduce_grad.cpp) | 通过[算子IR](op_graph/bn_training_reduce_grad_proto.h)构图方式调用BNTrainingReduceGrad算子。 |

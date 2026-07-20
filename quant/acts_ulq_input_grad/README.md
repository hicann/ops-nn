# ActsULQInputGrad

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term>                       |    √     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    √     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>     |    √     |
| <term>Atlas 200I/500 A2 推理产品</term>                      |    ×     |
| <term>Atlas 推理系列产品</term>                              |    √     |
| <term>Atlas 训练系列产品</term>                              |    √     |

## 功能说明

- 算子功能：ActsULQInputGrad是ActsULQ（Activations Universal Linear Quantization，均匀线性量化模拟）的反向输入梯度算子，用于量化感知训练（QAT）场景。前向阶段基于clamp边界生成下界掩码clamp_min_mask与上界掩码clamp_max_mask，本算子据此对上游梯度y_grad做逐元素门控，得到输入梯度x_grad。

- 计算公式：

  $$
  x\_grad = y\_grad \times clamp\_min\_mask \times clamp\_max\_mask
  $$

  即仅当元素同时落在量化下界与上界之内（两掩码均为 1）时，梯度才被透传；否则被置零。

## 参数说明

| 参数名 | 输入/输出/属性 | 描述 | 数据类型 | 数据格式 |
|--------|---------------|------|----------|----------|
| y_grad | 输入 | 表示上游传入的梯度张量，对应公式中y_grad；shape需与clamp_min_mask、clamp_max_mask、x_grad完全一致。 | FLOAT、FLOAT16 | ND |
| clamp_min_mask | 输入 | 表示下界clamp掩码张量，对应公式中clamp_min_mask，取值 {0,1}；数据类型为bool或与y_grad相同的浮点；shape需与y_grad一致。 | BOOL、FLOAT16、FLOAT | ND |
| clamp_max_mask | 输入 | 表示上界clamp掩码张量，对应公式中clamp_max_mask，取值 {0,1}；数据类型必须与clamp_min_mask一致；shape需与y_grad一致。 | BOOL、FLOAT16、FLOAT | ND |
| x_grad | 输出 | 门控后的输入梯度张量，对应公式中x_grad；数据类型必须与y_grad一致；shape必须与y_grad一致。 | FLOAT、FLOAT16 | ND |

## 约束说明

- y_grad与x_grad的数据类型必须一致，仅支持FLOAT和FLOAT16。
- clamp_min_mask与clamp_max_mask的数据类型必须一致；浮点掩码须与y_grad同dtype，或为bool。合法的dtype组合仅 4 组：

  | y_grad | clamp_min_mask / clamp_max_mask | x_grad |
  |--------|---------------------------------|--------|
  | FLOAT16 | BOOL | FLOAT16 |
  | FLOAT16 | FLOAT16 | FLOAT16 |
  | FLOAT | BOOL | FLOAT |
  | FLOAT | FLOAT | FLOAT |

- 三个输入与输出的shape必须完全一致（无广播），支持空Tensor。

## 调用说明

| 调用方式 | 样例代码 | 说明 |
|----------|----------|------|
| 图模式 | [test_geir_acts_ulq_input_grad](./examples/test_geir_acts_ulq_input_grad.cpp) | 通过[算子IR](./op_graph/acts_ulq_input_grad_proto.h)构图方式调用ActsULQInputGrad算子。 |

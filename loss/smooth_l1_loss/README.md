# SmoothL1Loss

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------- | :------:|
|  <term>Ascend 950PR/Ascend 950DT</term>   |     √    |
|  <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>   |     √    |
|  <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>   |     √    |
|  <term>Atlas 200I/500 A2 推理产品</term>   |     ×    |
|  <term>Atlas 推理系列产品</term>   |     ×    |
|  <term>Atlas 训练系列产品</term>   |     ×    |

## 功能说明

- 算子功能：计算预测值与标签值之间的平滑 L1 损失（element-wise，无 reduction）。
- 计算公式：

  $$
  \text{loss}(z) = \begin{cases}
  \frac{0.5 \times z^2}{\sigma} & \text{if } |z| < \sigma \\
  |z| - 0.5 \times \sigma & \text{otherwise}
  \end{cases}
  $$

  其中 $z = \text{predict} - \text{label}$，$\sigma$ 为分段阈值（默认 1.0）。

## 参数说明

|参数名|输入/输出/属性|描述|数据类型|数据格式|
|-----|-----------|----|---------|------|
|predict|输入|表示预测值，即公式中`predict`。|BFLOAT16、FLOAT16、FLOAT32|ND|
|label|输入|表示真实标签值，即公式中`label`。数据类型与predict保持一致。|BFLOAT16、FLOAT16、FLOAT32|ND|
|loss|输出|表示平滑L1损失值输出，即公式中`loss`。|BFLOAT16、FLOAT16、FLOAT32|ND|
|sigma|属性|表示分段阈值，非负，默认1.0。|Float|-|

## 约束说明

- predict与label的shape和dtype必须一致。
- sigma必须为非负值。

## 调用说明

| 调用方式 | 调用样例 | 说明 |
|---------|---------|------|
| GE图模式 | [test_geir_smooth_l1_loss.cpp](examples/test_geir_smooth_l1_loss.cpp) | 通过GE图模式方式调用SmoothL1Loss算子，算子原型定义见[smooth_l1_loss_proto.h](op_graph/smooth_l1_loss_proto.h)。 |

## 参考资源

无

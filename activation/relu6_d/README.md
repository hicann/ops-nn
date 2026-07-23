# Relu6D

## 产品支持情况

|产品             |  是否支持  |
|:-------------------------|:----------:|
|  <term>Ascend 950PR/Ascend 950DT</term>   |     √    |
|  <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>   |     ×    |
|  <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>     |     ×    |
|  <term>Atlas 200I/500 A2 推理产品</term>    |     ×    |
|  <term>Atlas 推理系列产品</term>     |     ×    |
|  <term>Atlas 训练系列产品</term>    |     ×    |

## 功能说明

- 算子功能：带缩放系数的ReLU6激活函数（Relu6D），对输入张量逐元素先做下截断（负值置0）、再做上截断（限制到`6*scale`），输出值域为`[0, 6*scale]`。当`scale = 1.0`（默认）时等价于标准ReLU6。适用于MobileNet系列等轻量级视觉网络的激活层，限制激活值范围、提升低精度量化场景的数值稳定性。
- 计算公式：

$$
Relu6D(x) = \min(\max(x,\ 0),\ 6 \cdot scale)
$$

其中 $x$ 为输入张量，$scale$ 为阈值缩放系数（默认 $1.0$），$y$ 为输出张量，值域 $[0,\ 6 \cdot scale]$。

## 参数说明

| 参数名 | 输入/输出/属性 | 描述 | 数据类型 | 数据格式 |
|--------|-------------|------|---------|---------|
| x | 输入 | 输入特征张量，对应公式中x。 | FLOAT16、FLOAT、INT32、BFLOAT16 | ND |
| y | 输出 | 输出张量，对应公式中y，shape和dtype与输入x一致。 | FLOAT16、FLOAT、INT32、BFLOAT16 | ND |
| scale | 可选属性 | 阈值缩放系数，对应公式中scale，上界阈值 = 6*scale。默认值1.0（对应标准ReLU6）。 | FLOAT | - |

## 约束说明

- 输入x和输出y的shape必须相同。
- 最高支持8维张量。
- 支持动态shape与动态rank。
- scale约束：上界阈值为`6*scale`；`scale = 0`时上界为0，输出恒为0；`scale < 0`时上界`6*scale < 0`，输出恒等于`6*scale`（均为数学上合法的边界语义）。int32下阈值按截断向零。

## 调用说明

| 调用方式 | 样例代码 | 说明 |
|---------|---------|------|
| GE图模式 | [test_geir_relu6_d.cpp](examples/test_geir_relu6_d.cpp) | 通过[算子IR](op_graph/relu6_d_proto.h)构图方式调用Relu6D算子。 |

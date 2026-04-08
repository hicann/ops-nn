# SigmoidCrossEntropyWithLogits

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term>                             |    √     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    √     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> |    √     |
| <term>Atlas 200I/500 A2 推理产品</term>                      |    ×     |
| <term>Atlas 推理系列产品</term>                             |    √     |
| <term>Atlas 训练系列产品</term>                              |    √     |

## 功能说明

- 算子功能：计算输入 logits 与标签 target 之间的 Sigmoid Cross Entropy 损失。

- 计算公式：

  $$
  \text{loss} = \max(\text{predict}, 0) - \text{predict} \times \text{target} + \log(1 + \exp(-|\text{predict}|))
  $$

  其中：
  - predict 为输入的 logits
  - target 为标签值
  - loss 为计算得到的损失值

## 参数说明

| 参数名 | 输入/输出/属性 | 描述 | 数据类型 | 数据格式 |  
| ----- | ----- |----- |----- |----- |
| predict | 输入 | 预测值 logits。 | FLOAT16、FLOAT、BFLOAT16 | ND |
| target | 输入 | 标签值。 | FLOAT16、FLOAT、BFLOAT16 | ND |
| loss | 输出 | 损失值。shape和输入predict一致。 | FLOAT16、FLOAT、BFLOAT16 | ND |

## 约束说明

- predict 和 target 必须具有相同的数据类型和形状。
- 支持 FLOAT16、FLOAT、BFLOAT16 数据类型。

## 调用说明

| 调用方式 | 样例代码 | 说明 |
| ---------------- | --------------------------- | --------------------------------------------------- |
| 图模式 | - | 通过[算子IR](op_graph/sigmoid_cross_entropy_with_logits_proto.h)构图方式调用SigmoidCrossEntropyWithLogits算子。 |
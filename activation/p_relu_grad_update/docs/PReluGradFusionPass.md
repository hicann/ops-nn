# PReluGradFusionPass

## 融合模式

该融合将图中的PReluGrad算子拆分为PReluGradUpdate、PReluGradReduce两个算子：PReluGradUpdate基于dy、x、weight计算dx，并输出中间结果update；PReluGradReduce基于dy、x、weight和update计算da。融合前后图结构如下。

![](../../../docs/zh/figures/PReluGradFusionPass.png)

## 使用约束

- 结构约束：
  - 图中存在PReluGrad算子，输入为dy、x、weight，输出为dx、da。
- 数据类型约束：
  - dy、x、weight的数据类型需要保持一致，支持FLOAT16、FLOAT32、BFLOAT16。
- 数据格式和shape约束：
  - 输入数据格式为ND。
  - x的shape为2D\~8D。
  - weight为scalar或者1D张量，元素个数与x第2维（通道数）相同或者为1；也支持维数比x少一维的广播形式张量，其通道维与x第2维一致、其余维为1（如x为4D时weight可为[C,1,1]）。

## 支持的型号

<!-- npu="950" id1 -->
Ascend 950PR/Ascend 950DT
<!-- end id1 -->

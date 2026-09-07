# WeightQuantBatchMatmulV2TransposeFusionPass

## 融合模式

<!-- npu="950,A3,910b" id1 -->
融合模式一：将WeightQuantBatchMatmulV2算子x和/或weight输入前的Transpose/TransposeD节点从图中删除，并将转置信息打在算子的transpose_x和transpose_weight属性上。当weight输入连接Transpose节点时，同时将antiquant_scale和antiquant_offset输入前的Transpose/TransposeD节点从图中删除。如下图所示。

![](../../../docs/zh/figures/WeightQuantBatchMatmulV2TransposeFusionPass_1.png)

融合模式二：将WeightQuantBatchMatmulV2算子weight输入前的Transpose/TransposeD节点从图中删除，并将转置信息打在算子的transpose_weight属性上。同时将antiquant_scale和antiquant_offset输入前的简单Reshape节点从图中删除。如下图所示。

![](../../../docs/zh/figures/WeightQuantBatchMatmulV2TransposeFusionPass_2.png)

该融合模式支持的产品如下。

<!-- npu="910b" id2 -->
Atlas A2 训练系列产品/Atlas A2 推理系列产品
<!-- end id2 -->

<!-- npu="A3" id3 -->
Atlas A3 训练系列产品/Atlas A3 推理系列产品
<!-- end id3 -->

<!-- npu="950" id4 -->
Ascend 950PR/Ascend 950DT
<!-- end id4 -->
<!-- end id1 -->

## 使用约束

- x和weight输入至少有一个连接Transpose/TransposeD节点，否则不触发融合。
- 当weight节点连接Transpose节点时，才处理antiquant_scale和antiquant_offset所连接的Transpose/TransposeD或Reshape节点（融合模式一处理Transpose/TransposeD，融合模式二处理简单Reshape）。
- 融合模式二中，antiquant_scale和antiquant_offset的Reshape节点需为简单Reshape（输入shape某一维度为1）。
- 该融合规则不可关闭，关闭后会触发功能问题。
- x输入数据类型仅支持FLOAT16、BF16，weight输入数据类型支持INT4、INT8、INT32、FLOAT、FLOAT4_E2M1、FLOAT8_E4M3FN、HIFLOAT8，输出数据类型仅支持FLOAT16、BF16、INT8。
- x和weight的输入shape必须为2D。

# QuantBatchMatmulV4TransposeFusionPass

## 融合模式

<!-- npu="950" id1 -->
融合模式一：将QuantBatchMatmulV4算子x2和x2_scale输入前的Transpose/TransposeD节点从图中删除，并将转置信息打在QuantBatchMatmulV4算子的transpose_x2属性上。如下图所示。

![](../../../docs/zh/figures/QuantBatchMatmulV4TransposeFusionPass_1.png)

融合模式二：将QuantBatchMatmulV4算子x2输入前的Transpose/TransposeD节点和x2_scale输入前的Reshape节点从图中删除，并将转置信息打在QuantBatchMatmulV4算子的transpose_x2属性上。如下图所示。

![](../../../docs/zh/figures/QuantBatchMatmulV4TransposeFusionPass_2.png)

融合模式三：将QuantBatchMatmulV4算子x2输入前的Transpose/TransposeD节点和x2_scale输入前的Reshape节点（含Shape→Gather→Pack动态shape链）从图中删除，并将转置信息打在QuantBatchMatmulV4算子的transpose_x2属性上。如下图所示。

![](../../../docs/zh/figures/QuantBatchMatmulV4TransposeFusionPass_3.png)

>[!NOTE]说明
>该图融合仅支持Ascend 950PR/Ascend 950DT，不支持其他芯片型号。
<!-- end id1 -->

## 使用约束

- 该融合规则不可关闭。
- x2输入必须连接Transpose或TransposeD节点，否则不触发融合。
- x1和x2的输入shape必须为2D。
- x1数据类型仅支持FLOAT8_E4M3FN，x2数据类型仅支持FLOAT4_E2M1、FLOAT，输出数据类型仅支持BF16、FLOAT16。
- x2_scale输入前的Reshape节点需为简单Reshape（输入shape某一维度为1）或含Shape→Gather→Pack链的动态Reshape。

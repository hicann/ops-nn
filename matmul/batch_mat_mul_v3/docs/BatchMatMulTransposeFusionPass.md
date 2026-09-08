# BatchMatMulTransposeFusionPass

## 融合模式

该融合将MatMul输入侧仅交换最后两维的Transpose算子吸收进MatMul算子：删除Transpose节点，并将MatMul对应输入的转置属性取反（BatchMatMul/BatchMatMulV2翻转adj_x1/adj_x2，MatMul/MatMulV2翻转transpose_x1/transpose_x2）。

![](../../../docs/zh/figures/BatchMatMulTransposeFusionPass_1.png)

## 使用约束

- Transpose1和Transpose2可以同时存在，也可以只存在一个。
- Transpose类型只包括Transpose。
- Transpose节点仅对输入的最后两维进行交换，如Transpose节点的输入shape为[B,M,K]，输出shape为[B,K,M]。
- MatMul类型包括BatchMatMul/BatchMatMulV2/MatMul/MatMulV2。
- MatMul节点的输入dtype仅支持FLOAT16、FLOAT32、BFLOAT16。

## 支持的型号

<!-- npu="950" id1 -->
Ascend 950PR/Ascend 950DT
<!-- end id1 -->

# ZZMatMulTOQBMMV3FusionPass

## Fusion Patterns

This fusion converts MatMul/MatMulV2/MatMulV3 and BatchMatMul/BatchMatMulV2/BatchMatMulV3 into QuantBatchMatmulV3, inserting a constant scale of 1.0. The bias input, transpose attributes, and output y data type are preserved.

![Before and after converting MatMul to QuantBatchMatmulV3](../../../docs/en/figures/ZZMatMulTOQBMMV3FusionPass_1.png)

## Constraints

- MatMul types include MatMul, MatMulV2, MatMulV3, BatchMatMul, BatchMatMulV2, and BatchMatMulV3.
- The supported input data type is HIFLOAT8.
<!-- npu="950" id1 -->
- <term>Ascend 950PR&950DT products</term>: Support the combination of INT8 input and INT32 output.
<!-- end id1 -->

## Applicable Products

<!-- npu="950" id2 -->
- <term>Ascend 950PR&950DT products</term>
<!-- end id2 -->

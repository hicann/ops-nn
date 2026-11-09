# QuantBatchMatmulV4TransposeFusionPass

## Fusion Patterns

<!-- npu="950" id1 -->
Fusion mode 1: Delete the Transpose/TransposeD nodes before the x2 and x2_scale inputs of the QuantBatchMatmulV4 operator from the graph, and write the transpose information to the transpose_x2 attribute of the QuantBatchMatmulV4 operator. See the following figure.

![Pattern 1: Fusing Transpose nodes before weights and scale](../../../docs/en/figures/QuantBatchMatmulV4TransposeFusionPass_1.png)

Fusion mode 2: Delete the Transpose/TransposeD node before the x2 input of the QuantBatchMatmulV4 operator and the Reshape node before the x2_scale input from the graph, and write the transpose information to the transpose_x2 attribute of the QuantBatchMatmulV4 operator. See the following figure.

![Pattern 2: Fusing Transpose before weights and Reshape before scale](../../../docs/en/figures/QuantBatchMatmulV4TransposeFusionPass_2.png)

Fusion mode 3: Delete the Transpose/TransposeD node before the x2 input of the QuantBatchMatmulV4 operator and the Reshape node (including the Shape→Gather→Pack dynamic shape chain) before the x2_scale input from the graph, and write the transpose information to the transpose_x2 attribute of the QuantBatchMatmulV4 operator. See the following figure.

![Pattern 3: Fusing Transpose and Reshape with a dynamic-shape chain](../../../docs/en/figures/QuantBatchMatmulV4TransposeFusionPass_3.png)

> [!NOTE]
>
> This graph fusion supports only <term>Ascend 950PR&950DT products</term>.
<!-- end id1 -->

## Constraints

- This fusion rule cannot be disabled.
- The x2 input must be connected to a Transpose or TransposeD node. Otherwise, the fusion is not triggered.
- The input shapes of x1 and x2 must be 2D.
- x1 supports only FLOAT8_E4M3FN; x2 only FLOAT4_E2M1 and FLOAT32; and the output only BFLOAT16 and FLOAT16.
- The Reshape node before the x2_scale input must be a simple Reshape (a dimension of the input shape is 1) or a dynamic Reshape containing a Shape→Gather→Pack chain.

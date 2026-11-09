# QuantBatchMatmulV4ToV3FusionPass

## Fusion Patterns

This fusion replaces eligible `QuantBatchMatmulV4` nodes with `QuantBatchMatmulV3` nodes, reorders connections according to the two operators' input definitions, and preserves the output shape, data type, format, and related attributes. Quantized matrix multiplication is then performed by `QuantBatchMatmulV3`, as shown below.

![Converting QuantBatchMatmulV4 to QuantBatchMatmulV3](../../../docs/en/figures/QuantBatchMatmulV4ToV3FusionPass_1.png)

The figure shows the mapped inputs. Optional V3 inputs remain at their defaults when unconnected. Although `x2_scale` is optional in V4, it maps to the required V3 input scale, so an eligible graph must provide a valid `x2_scale`. The input mapping below uses indices from the operator IR definitions.

| QuantBatchMatmulV4 input | V4 index | QuantBatchMatmulV3 input | V3 index |
| ---- | ---- | ---- | ---- |
| x1 | 0 | x1 | 0 |
| x2 | 1 | x2 | 1 |
| bias | 2 | bias | 4 |
| `x1_scale` | 3 | `pertoken_scale` | 5 |
| `x2_scale` | 4 | scale | 2 |
| `x2_offset` | 7 | offset | 3 |

During conversion, the shape, data type, and format of each mapped input and the output `y` remain unchanged, and no nodes such as data type conversion are added. Therefore, the data type of each mapped input must also meet the requirements of the corresponding V3 input. Attribute handling is as follows.

| Attribute | Conversion Rule |
| ---- | ---- |
| dtype | The attribute value of V4 is passed. |
| `transpose_x1` | The attribute value of V4 is passed without inversion. |
| `transpose_x2` | The attribute value of V4 is passed without inversion. |
| `group_size` | If the V4 value is -1, it is changed to the V3 default value 0; other values remain unchanged. |

The `compute_type` attribute, which is unique to V4, is not passed to V3.

## Constraints

- This rule processes `QuantBatchMatmulV4` nodes in the graph.
- The matched structures of optional inputs include all inputs provided, or successive optional inputs absent starting from `bias` in the operator-defined input order. This does not mean that optional input combinations at arbitrary positions are supported. During conversion, connections are established only for mapped inputs whose shape, data type, and other information can be obtained.
- The input graph must provide the shape, data type, and other information of `x1`, `x2`, and output `y`. No conversion is performed if such information is missing.
- The input graph should fall within the quantization use cases that QuantBatchMatmulV3 can handle, and meet its data type, shape, and format constraints. See [QuantBatchMatmulV3 operator description](../../quant_batch_matmul_v3/README_en.md).
- `y_scale`, `x1_offset`, `y_offset`, and `x2_table` of V4 are not connected to the replaced V3. If the computation depends on these inputs, the graph is beyond the scope of equivalent conversion by this rule; do not rely on this rule to automatically intercept all such input combinations.

The following two use cases retain QuantBatchMatmulV4 without conversion. The conditions in the same row must be met simultaneously.

| Use cases for retaining V4 | Data type condition | shape condition |
| ---- | ---- | ---- |
| Native V4 case | x1 is `FLOAT8_E4M3FN`, x2 is `FLOAT4_E2M1` or FLOAT32, and y is BFLOAT16 or FLOAT16 | Retains V4 when these type conditions hold, with no additional shape conditions |
| Recognized A8W8 G-B case | x1 and x2 are INT8, both `x1_scale` and `x2_scale` are present as FLOAT32, and y is BFLOAT16 | x1 and `x1_scale` have the same original rank, and x2 and `x2_scale` have the same original rank |

The A8W8 G-B use case in the preceding table compares the number of dimensions only. It does not require equal dimension lengths, nor does it distinguish by the `group_size` attribute value. In addition to the conditions in the preceding table, this rule imposes no additional requirement that inputs be static shapes, and no additional restriction on input formats; the applicable shapes and formats must still meet the operator constraints of both V4 and V3.

## Applicable Products

The applicable products must satisfy the constraints in both the [QuantBatchMatmulV4 operator description](../README_en.md) and the [QuantBatchMatmulV3 operator description](../../quant_batch_matmul_v3/README_en.md). This pass does not perform chip model-specific branching. Whether the conversion is performed is determined by the preceding input conditions.

<!-- npu="910b" id1 -->
- <term>Atlas A2 products</term>
<!-- end id1 -->

<!-- npu="A3" id2 -->
- <term>Atlas A3 products</term>
<!-- end id2 -->

<!-- npu="950" id3 -->
- <term>Ascend 950PR&950DT products</term>
<!-- end id3 -->

# QuantBatchMatmulV3TransposeFusionPass

## Fusion Patterns

This fusion rule folds explicit transposes on the `x1` or `x2` input branch of `QuantBatchMatmulV3` into matrix multiplication, and can process both branches together. It recognizes `Transpose` and `TransposeD` nodes. Fusing the `x1` branch inverts the node's `transpose_x1` attribute; fusing the `x2` branch inverts `transpose_x2`. Inversion changes `false` to `true` or `true` to `false`.

In this document, "bypass" means connecting a conversion node's input directly to its successor so that the current branch no longer passes through that conversion node. Retain the bypassed node if other data consumers still use it; otherwise, remove it. The successor can be `QuantBatchMatmulV3` or a retained `Bitcast`.

The diagrams below assume that the [Input and Transpose Requirements](#input-and-transpose-requirements) and [Dimension and Platform Constraints](#dimension-and-platform-constraints) are satisfied. Variables used in the diagrams are defined below:

- **Input mapping**: The diagrams show `x1`, `x2`, and their two `scale` branches. `x1_scale` denotes input `pertoken_scale` (IR index 5), and `x2_scale` denotes input `scale` (IR index 2). These names are diagram aliases, not additional operator parameters or attributes.
- **Symbols**: `t1` and `t2` denote the `transpose_x1` and `transpose_x2` attributes of `QuantBatchMatmulV3` before fusion. `!` denotes logical negation.
- **Source tensors**: A source tensor with the same name before and after fusion denotes the same tensor.
- **Omitted items**: The diagrams omit `bias`, `offset`, transpose-permutation inputs, and target-shape inputs. Only connections relevant to fusion are shown.

### Fusing Only the x1 Transpose

When the `x1` and `x1_scale` branches each contain a transpose node, bypass both transpose nodes and connect their inputs directly to the `x1` and `pertoken_scale` ports of `QuantBatchMatmulV3`, respectively. Invert the node's `transpose_x1` attribute. Keep the `x2` and `scale` connections and the `transpose_x2` attribute unchanged.

![Fusing the x1 and x1_scale transposes with four independent input branches](../../../docs/en/figures/QuantBatchMatmulV3TransposeFusionPass_1.png)

### Fusing Only the x2 Transpose

When the `x2` and `x2_scale` branches each contain a transpose node, bypass both transpose nodes and connect their inputs directly to the `x2` and `scale` ports of `QuantBatchMatmulV3`, respectively. Invert the node's `transpose_x2` attribute. Keep the `x1` and `pertoken_scale` connections and the `transpose_x1` attribute unchanged.

![Fusing the x2 and x2_scale transposes with four independent input branches](../../../docs/en/figures/QuantBatchMatmulV3TransposeFusionPass_2.png)

### Fusing Transposes on Both Sides

When each of the `x1`, `x2`, `x1_scale`, and `x2_scale` branches contains a transpose node, and both data branches meet the constraints listed at the start of this section, bypass all four transpose nodes. Connect their inputs to the `x1`, `x2`, `pertoken_scale`, and `scale` ports of `QuantBatchMatmulV3`, respectively, and invert both `transpose_x1` and `transpose_x2`. Matching the illustrated structure alone is insufficient: the data type, dimension, and platform capability constraints must also be satisfied.

![Fusing the four transposes on the two data and two scale branches](../../../docs/en/figures/QuantBatchMatmulV3TransposeFusionPass_3.png)

### Reshape Fusion for Normal scale

When each data branch contains a transpose node and each `scale` branch contains a `Reshape` equivalent to the required transpose, bypass both transpose nodes and both `Reshape` nodes. Connect the four branches to their corresponding `QuantBatchMatmulV3` input ports and invert both `transpose_x1` and `transpose_x2`. For static normal `scale` requirements, see [Reshape Requirements for scale](#reshape-requirements-for-scale). The figure shows fusion on both sides; only the side meeting the constraints can also be fused.

![Fusing each normal scale Reshape with its corresponding data transpose](../../../docs/en/figures/QuantBatchMatmulV3TransposeFusionPass_4.png)

### Reshape Fusion for MX scale

For example, on the `x2` side, the data input is preceded by `Transpose` or `TransposeD`, `x2_scale` has type `FLOAT8_E8M0`, and its `Reshape` changes (1, N, 2) to (N, 1, 2). Bypass the data transpose and scale `Reshape`, connect their inputs to the `x2` and `scale` ports of `QuantBatchMatmulV3`, respectively, and invert `transpose_x2`. Keep the `x1` and `x1_scale` connections unchanged. Apply the same logic on the `x1` side when its corresponding conditions are met.

![MX inputs showing x1, x2, both scales, and a three-dimensional Reshape](../../../docs/en/figures/QuantBatchMatmulV3TransposeFusionPass_5.png)

### Transpose Fusion Retaining Bitcast

`Bitcast` reinterprets the data bits as the target data type. When each of the four branches follows "transpose node → `Bitcast` → `QuantBatchMatmulV3`", connect each transpose node's input directly to its following `Bitcast`. Retain all four `Bitcast` nodes and their connections to `QuantBatchMatmulV3`, and invert both `transpose_x1` and `transpose_x2`. The output data type of each `Bitcast` remains unchanged.

![Fusing the four input transposes while retaining their Bitcast nodes](../../../docs/en/figures/QuantBatchMatmulV3TransposeFusionPass_6.png)

The input branches do not all have to contain `Bitcast`. A transpose can connect to matrix multiplication directly or through one `Bitcast`; fusion across multiple consecutive `Bitcast` nodes is unsupported. On a `scale` branch, the node preceding `Bitcast` can also be a `Reshape` meeting the requirements below. For platform restrictions and the scope of dynamic `Reshape` support, see Constraints.

### Reshape Fusion for Dynamic scale

Treat the scenario as dynamic if `x1` or `x2` contains an unknown dimension. For example, on the `x2` side, bypass the data transpose and `x2_scale` `Reshape`, connect their inputs to the `x2` and `scale` ports of `QuantBatchMatmulV3`, respectively, and invert `transpose_x2`. Keep `x1` and `x1_scale` unchanged. Static `Reshape` dimension checks are not performed in dynamic scenarios, but the input graph must still ensure that the `Reshape` is equivalent to the required scale transpose.

![Four independent inputs in a dynamic scenario, with only the x2 transpose and Reshape fused](../../../docs/en/figures/QuantBatchMatmulV3TransposeFusionPass_7.png)

The scale conversion nodes shown in each figure are specific to that example. A conversion node is not required on every `scale` branch. Process a `scale` branch only when the corresponding data input's `Transpose` or `TransposeD` is fused; do not process `pertoken_scale` if it is unconnected. Retain a bypassed node if other data consumers still use it; otherwise, remove it from the graph.

## Constraints

### Input and Transpose Requirements

- The original shapes of `x1` and `x2` must each have at least two dimensions. At least one data input must have a preceding `Transpose` or `TransposeD`; otherwise, fusion is not triggered. A `Reshape` alone before a data input cannot trigger fusion.
- `x1` and `x2` each support `INT8`, `INT4`, `FLOAT8_E4M3FN`, `FLOAT8_E5M2`, `HIFLOAT8`, and `FLOAT4_E2M1`. The output supports `INT8`, `FLOAT16`, `BFLOAT16`, `INT32`, and `FLOAT32`. The actual combination must also satisfy the target product's operator constraints.
- Dynamic shapes are supported. If the shape of `x1` or `x2` contains an unknown dimension, the case is treated as dynamic.
- A data transpose must swap only the last two dimensions and preserve the order of all other dimensions, for example, `(B, M, K)` → `(B, K, M)`. Other permutations, such as swapping batch dimensions, are outside this rule's scope. The corresponding scale transformation must also preserve the correspondence between quantization coefficients and data. See the next section.
- `pertoken_scale` is optional. Its scale branch is not processed when unconnected.

### Restrictions for Bitcast Patterns

Some platforms support mixed INT8 and INT4 matrix multiplication inputs. On those platforms, if any of the four input branches matches a pattern in the table below, retain the entire `QuantBatchMatmulV3` node and all its input branches without performing this transpose fusion. This restriction applies even when the current node does not use mixed INT8 and INT4 inputs.

| Input branch | Connection pattern that triggers the restrictions above |
| ---- | ---- |
| `x1`, `x2`, or either of the two `scale` branches | `Transpose` or `TransposeD` → `Bitcast` → `QuantBatchMatmulV3` |
| Either of the two `scale` branches | `Reshape` that meets the fusion requirements in the next section → `Bitcast` → `QuantBatchMatmulV3` |

When only `Bitcast` is present and no preceding conversion nodes described above exist, fusion is not skipped due to this restriction.

There is also a recognition restriction when a dynamic `Reshape` is used in combination with `Bitcast`: if only the `scale` branch contains a "dynamic `Reshape` → `Bitcast` → matrix multiplication" structure and no other branch contains a recognizable structure with `Bitcast` described above, the dynamic `Reshape` is not guaranteed to be fused. Therefore, the dynamic `Reshape` examples in this document show only connections that do not go through `Bitcast`.

### Reshape Requirements for scale

The `Reshape` before the `scale` input must meet the conditions in the following table. In a static scenario, the input graph must also provide the shapes and data types of the `Reshape` input and output, and the corresponding `QuantBatchMatmulV3` input must have at least 2 dimensions.

| Scenario | Recognition Condition |
| ---- | ---- |
| Condition A: static general condition | At least one of the last two dimensions of the `shape` of the corresponding `QuantBatchMatmulV3` input is 1. This check does not restrict the data type of `scale`. |
| Condition B: static MX condition | The data type is `FLOAT8_E8M0`. The `Reshape` input and output are both 3-dimensional, with the first two dimensions swapped and at least one of them being 1. For example, (1, N, 2) changes to (N, 1, 2). |
| Dynamic `shape` | The preceding static shape conditions are not required, but the `Reshape` must still be equivalent to the required `scale` transpose. |

In a static scenario, meeting either Condition A or Condition B in the preceding table is sufficient. Condition B refers to swapping the first two dimensions of a 3-dimensional `FLOAT8_E8M0` `scale`. If Condition B is not met, the pattern can still be recognized through Condition A. For a dynamic scenario, follow the "Dynamic `shape`" row in the preceding table. In the MX example in the figure, the last dimension 2 comes from the quantization layout of that example and is not a fixed value additionally required by Condition B. After the shape conditions in the table are met, the `Reshape` must still be equivalent to the required `scale` transpose and must not change the correspondence between quantization coefficients and data. This rule does not separately delete nodes such as `Shape`, `Gather`, and `Pack` that are used to compute dynamic shapes.

### Dimension and Platform Constraints

`QuantBatchMatmulV3TransposeFusionPass` is the general transpose fusion rule, and `QuantBatchMatmulV3TransposeLimitFusionPass` is a supplementary rule for specific dimension scenarios. The two rules are registered separately. Each merges eligible explicit transposes into matrix multiplication computation. Fusion is implemented by adjusting input connections and inverting the corresponding transpose attributes of `QuantBatchMatmulV3`. They differ in the applicable products and dimension conditions. The Limit rule still provides transpose fusion processing for branches that meet its conditions even when the general rule is disabled.

Let `outer` denote the penultimate dimension of the original `shape` of the data input corresponding to `QuantBatchMatmulV3` before fusion, and `inner` denote the last dimension. Both rules check `x1` and `x2` respectively, and only the side that satisfies the conditions may be fused. The processing conditions of the products listed in this document are as follows. The actual fusion must also satisfy the preceding usage constraints, such as input, transpose node, and `scale` requirements.

| Product | General Rule | Limit Supplementary Rule |
| ---- | ---- | ---- |
| <term>Ascend 950PR&950DT products</term> | Neither `x1` nor `x2` is subject to this section's 65535-dimension threshold. Transpose fusion runs when the other constraints are met. | This supplementary rule is skipped; transpose fusion is handled by the regular rule. |
| <term>Atlas A2 products</term>, <term>Atlas A3 products</term> | `x1` requires `outer` ≤ 65535. `x2` requires `outer` ≤ 65535 or `FRACTAL_NZ` format. | Checks `x1` and `x2` independently and processes only branches with 0 < `outer` ≤ 65535 and `inner` > 65535. |

The dimension conditions of the Limit rule also apply to the `FRACTAL_NZ` format; they are not relaxed due to the format. If the `outer` or `inner` value involved in the check is the unknown value `-1`, the branch does not satisfy the dimension conditions of the Limit rule.

### Version Requirements

- The graph compiler version number must be no less than 90100000 at both compile time and run time.
- If the compile-time version does not meet the requirement, the two transpose rules are not enabled. If the run-time version does not meet the requirement, both rules keep the input graph unchanged.

## Applicable Products

See [QuantBatchMatmulV3 Operator Description](../README_en.md) for the data types and formats supported by the product. This rule determines fusion conditions based on platform instruction capabilities. Operator support for a product does not mean that the product supports all fusion modes of this rule.

<!-- npu="910b" id1 -->
- <term>Atlas A2 products</term>
<!-- end id1 -->

<!-- npu="A3" id2 -->
- <term>Atlas A3 products</term>
<!-- end id2 -->

<!-- npu="950" id3 -->
- <term>Ascend 950PR&950DT products</term>
<!-- end id3 -->

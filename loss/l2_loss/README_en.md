# L2Loss

English | [简体中文](./README.md)

## Applicable Products

|Product             |  Support  |
|:-------------------------|:----------:|
|  <term>Ascend 950PR&950DT products</term>   |     √    |
|  <term>Atlas A3 products</term>  |     √    |
|  <term>Atlas A2 products</term>     |     √    |
|  <term>Atlas 200I/500 A2 inference products</term>    |     ×    |
|  <term>Atlas inference products</term>     |     √    |
|  <term>Atlas training products</term>    |     √    |

## Function

- Operator function: Computes half of the sum of squares of all elements in the input x, that is, the L2 regularization loss. The output y is a scalar.

- Formula:

  $$
  y = \frac{1}{2} \sum_{i} x_i^2
  $$

  Here, $x$ is the input `x`, and $i$ traverses all elements in $x$.

- Computation implementation: To avoid intermediate overflow in the squaring operation, the implementation first multiplies the input by $\frac{1}{\sqrt{2}}$ and then accumulates the squares, which is mathematically equivalent to $\frac{1}{2}\sum_i x_i^2$. For FLOAT16 or BFLOAT16 inputs, the multiplication and accumulation are promoted to FLOAT32 on-chip and then written back as the input data type.

## Parameters

<table style="table-layout: fixed; width: 1576px">
<colgroup>
<col style="width: 170px">
<col style="width: 170px">
<col style="width: 200px">
<col style="width: 200px">
<col style="width: 170px">
</colgroup>
<thead>
<tr><th>Parameter</th><th>Input/Output/Attribute</th><th>Description</th><th>Data Type</th><th>Data Format</th></tr>
</thead>
<tbody>
<tr><td>x</td><td>Input</td><td>The input x in the formula, the tensor for which the L2 loss is computed. The total number of elements may be 0.</td><td>BFLOAT16, FLOAT16, FLOAT</td><td>ND</td></tr>
<tr><td>y</td><td>Output</td><td>The output y in the formula, a 0-dimensional scalar. Its data type is consistent with x.</td><td>BFLOAT16, FLOAT16, FLOAT</td><td>ND</td></tr>
</tbody>
</table>

## Constraints

- Data type and format: The data types of `x` and `y` must be consistent, and the data format supports only ND. This operator does not validate out-of-range data types or formats; the caller must ensure that the inputs are valid.
- Output shape: `y` is always a 0-dimensional scalar, regardless of the shape of `x`. The data of `y` is readable after graph execution completes.
- Input rank and per-dimension boundaries: Dynamic shape and dynamic rank are supported. The upper limit of the rank is determined by the shared reduce template and the UB capacity; the test cases in this repository cover rank 1 to 5. When the total number of elements of `x` is 0, the output is 0.
- Non-contiguous and in-place semantics: This operator does not provide non-contiguous tensor semantics. In GE graph mode, the contiguity of tensors is converted by the graph compiler. `y` is an independent output and does not support in-place reuse or memory overlap with `x`.
- Special values and precision: inf and nan in `x` propagate to the output following the IEEE 754 rules. Although FLOAT16 and BFLOAT16 accumulate in FLOAT32 on-chip, they are still limited by their own representation range when written back; large or extremely large values may overflow to inf.
- Determinism: The partitioning and accumulation order of the reduction are determined by tiling based on the input shape and the number of cores on the chip. Bitwise consistency across different shapes or chip configurations is not guaranteed.
- 0-dimensional input and 1-dimensional empty tensor: In the const-shape (cst) and dynamic (dyn) compilation modes of CANN 9.3.0, due to the compilation framework's code-generation limitation for 0-dimensional scalar output, rank-0 inputs and inputs with shape (0,) fail to compile. The binary (bin) mode is not affected.
- Invocation path: This operator supports only GE graph mode invocation. It does not provide the aclnn interface, nor PyTorch or TensorFlow end-to-end bindings. For details, see "Invocation Description".

## Invocation Description

| Invocation method | Sample code | Description |
| ---- | ---- | ---- |
| GE graph mode | [test_geir_l2_loss.cpp](examples/test_geir_l2_loss.cpp) | Calls the L2Loss operator by building a graph through the [operator IR](op_graph/l2_loss_proto.h). |

Current support status of other invocation paths:

- aclnn API: Not provided. This operator declares `ACLNNTYPE aclnn_exclude` in [CMakeLists.txt](CMakeLists.txt) to exclude aclnn interface generation, and there is no `op_api` implementation in the directory. Calling `aclnnL2Loss` returns "interface not found".
- PyTorch API: No end-to-end binding is provided.
- TensorFlow: [framework/l2_loss_tf_plugin.cpp](framework/l2_loss_tf_plugin.cpp) only provides TensorFlow 1.x (npu_bridge) style GE graph mapping registration, which maps the L2Loss node in a TensorFlow graph to this operator. It does not constitute a standalone end-to-end invocation path and depends on the corresponding TensorFlow and npu_bridge runtime environment.

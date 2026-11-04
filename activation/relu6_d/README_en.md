# Relu6D

English | [简体中文](./README.md)

## Applicable Products

| Product | Support |
| :--- | :---: |
| <term>Ascend 950PR&950DT products</term> | √ |
| <term>Atlas A3 products</term> | √ |
| <term>Atlas A2 products</term> | √ |
| <term>Atlas 200I/500 A2 inference products</term> | √ |
| <term>Atlas inference products</term> | √ |
| <term>Atlas training products</term> | √ |

## Function

- Operator function: Relu6D is the ReLU6 activation function with a scaling coefficient. It elementwise truncates the input tensor from below first (negative values become 0) and then truncates it from above (limits it to `6*scale`). When `scale ≥ 0`, the output value range is `[0, 6*scale]`; when `scale < 0`, the output always equals `6*scale`. When `scale = 1.0` (the default), it is equivalent to standard ReLU6. It suits activation layers of lightweight vision networks such as the MobileNet series, limiting the activation value range and improving numerical stability in low-precision quantization scenarios.
- Formula:

$$
Relu6D(x) = \min(\max(x,\ 0),\ 6 \cdot scale)
$$

In the formula, $x$ is the input tensor, $scale$ is the threshold scaling coefficient (default $1.0$), and $y$ is the output tensor. When `scale ≥ 0`, the value range is $[0,\ 6 \cdot scale]$; when `scale < 0`, the output is always $6 \cdot scale$.

## Parameters

| Parameter name | Input/Output/Attribute | Description | Data type | Data format |
|--------|-------------|------|---------|---------|
| x | Input | The input feature tensor, corresponding to x in the formula. | FLOAT16, FLOAT, INT32, BFLOAT16 | ND |
| y | Output | The output tensor, corresponding to y in the formula, with the same shape and dtype as input x. | FLOAT16, FLOAT, INT32, BFLOAT16 | ND |
| scale | Optional attribute | The threshold scaling coefficient, corresponding to scale in the formula; the upper threshold = 6*scale. Default value: 1.0 (corresponding to standard ReLU6). | FLOAT | - |

## Constraints

- The shapes of input x and output y must be the same.
- Up to 8-dimensional tensors are supported.
- Dynamic shapes and dynamic ranks are supported.
- Empty tensors return successfully directly: tiling sets `blockNum` to 0 and allocates no data processing resources, and both Kernel Init and Process return early without initiating GM reads or writes.
- Invalid range of the total element count: the total number of flattened elements of `x` (the shape product) overflows int64.
- scale constraints: the upper threshold is `6*scale`. When `scale = 0`, the upper threshold is 0 and the output is always 0. When `scale < 0`, the upper threshold `6*scale < 0` and the output always equals `6*scale` (both are mathematically valid boundary semantics). For int32, the threshold is truncated toward zero. Invalid ranges: `scale` is NaN/±Inf, or for int32 `6*scale` exceeds `[-2147483648, 2147483648)`.

## Invocation Description

| Invocation method | Sample code | Description |
|---------|---------|------|
| GE graph mode | [test_geir_relu6_d.cpp](examples/test_geir_relu6_d.cpp) | Calls the Relu6D operator by building the graph through the [operator IR](op_graph/relu6_d_proto.h). |

Compile and run the sample (execute in the repository root directory):

```bash
bash build.sh --run_example relu6_d graph
```

# ForeachAddcdivScalar

English | [简体中文](./README.md)

## Applicable Products

| Product | Support |
|:---|:---:|
| <term>Ascend 950PR&950DT products</term> | √ |
| <term>Atlas A3 products</term> | √ |
| <term>Atlas A2 products</term> | √ |
| <term>Atlas 200I/500 A2 inference products</term> | × |
| <term>Atlas inference products</term> | × |
| <term>Atlas training products</term> | × |
| <term>Kirin X90 processor series products</term> | √ |
| <term>Kirin 9030 processor series products</term> | √ |

## Function

- Operator function: Perform element-wise addition, multiplication, and division operations on multiple tensors. $x2_{i}$ and $x3_{i}$ are divided element-wise, the result is multiplied by scalar, and then the product is added to $x1_{i}$.
- Formula:

  $$
  x1 = [{x1_0}, {x1_1}, ... {x1_{n-1}}], x2 = [{x2_0}, {x2_1}, ... {x2_{n-1}}], x3 = [{x3_0}, {x3_1}, ... {x3_{n-1}}]\\
  y = [{y_0}, {y_1}, ... {y_{n-1}}]\\
  $$

  $$
  y_i = {x1}_{i}+ \frac{{x2}_{i}}{{x3}_{i}}\times{scalar} (i=0,1,...n-1)
  $$

## Parameters

<table style="undefined;table-layout: fixed; width: 1005px"><colgroup>
  <col style="width: 170px">
  <col style="width: 170px">
  <col style="width: 352px">
  <col style="width: 213px">
  <col style="width: 100px">
  </colgroup>
  <thead>
    <tr>
      <th>Parameter name</th>
      <th>Input/Output/Attribute</th>
      <th>Description</th>
      <th>Data type</th>
      <th>Data format</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>x1</td>
      <td>Input</td>
      <td>First input tensor list of the addition in the mixed operation, corresponding to `x1` in the formula. All tensors in this parameter have the same data type.</td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>x2</td>
      <td>Input</td>
      <td>First input tensor list of the division in the mixed operation, corresponding to `x2` in the formula. The data type, data format, and shape are the same as those of input parameter `x1`, and all tensors in this parameter have the same data type.</td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>x3</td>
      <td>Input</td>
      <td>Second input tensor list of the division in the mixed operation, corresponding to `x3` in the formula. The data type, data format, and shape are the same as those of input parameter `x1`, and all tensors in this parameter have the same data type.</td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>scalar</td>
      <td>Input</td>
      <td>Second input tensor of the multiplication in the mixed operation, corresponding to `scalar` in the formula. The number of elements is 1. The supported data types are FLOAT32 and FLOAT16, and the data type has a specific correspondence with the data type of input parameter `x1`: when the data type of `x1` is FLOAT32 or FLOAT16, the data type is the same as that of `x1`; when the data type of `x1` is BFLOAT16, the supported data type is FLOAT32.</td>
      <td>FLOAT32, FLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y</td>
      <td>Output</td>
      <td>Output tensor list of the mixed operation, corresponding to `y` in the formula. The data type and data format are the same as those of input parameter `x1`, and the shape is the same as that of `x1`. All tensors in this parameter have the same data type.</td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
  </tbody></table>

- <term>Kirin X90 processor series products</term>, <term>Kirin 9030 processor series products</term>: BFLOAT16 is not supported.

## Constraints

- The tensor lists `x1`, `x2`, and `x3` must contain the same number of tensors, which must be the same as the number of elements of `y`. A single tensor list contains no more than 256 tensors.
- Each tensor in `x1`, `x2`, `x3`, and `y` has no more than 8 dimensions and the same shape. `scalar` has no more than 8 dimensions and contains 1 element.
- The graph mode (GE/IR) path supports dynamic shape/rank marks: `-1` indicates a dynamic dimension, and `-2` indicates a dynamic rank. Host inference passes through the marks as they are. Specialization must be completed before execution, and the dimensions of each tensor must be concrete non-negative values (tiling returns a failure for inputs containing negative dimension marks).
- Non-contiguous output tensor support: aclnnForeachAddcdivScalar (V1) supports non-contiguous output tensors (the internal implementation automatically converts them to contiguous and writes them back); aclnnForeachAddcdivScalarV2 and graph mode (IR) do not support non-contiguous output tensors.

## Invocation Description

| Invocation method | Sample code | Description |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn API  | [test_aclnn_foreach_addcdiv_scalar](examples/test_aclnn_foreach_addcdiv_scalar.cpp) | Invoke the ForeachAddcdivScalar operator through the [aclnnForeachAddcdivScalar](docs/aclnnForeachAddcdivScalar_en.md) API. |
| aclnn API  | [test_aclnn_foreach_addcdiv_scalar_v2](examples/test_aclnn_foreach_addcdiv_scalar_v2.cpp) | Invoke the ForeachAddcdivScalar operator through the [aclnnForeachAddcdivScalarV2](docs/aclnnForeachAddcdivScalarV2_en.md) API. |
| GE graph mode | -  | Invoke the ForeachAddcdivScalar operator through graph construction using the [operator IR](op_graph/foreach_addcdiv_scalar_proto.h).         |

- Note: This repository delivers only the aclnn and graph mode paths in the preceding table. The E2E Torch/TF paths are not delivered in this repository. (If you need to invoke the operator through a framework, you can rely on external paths such as `torch._foreach_addcdiv` of torch_npu.)

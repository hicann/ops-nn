# FakeQuantAffineCachemask

English | [简体中文](./README.md)

## Applicable Products

| Product | Support |
| ---- | :----:|
|Ascend 950PR&950DT products|√|
|Atlas A3 products|√|
|Atlas A2 products|√|
|Atlas 200I/500 A2 inference products|×|
|Atlas inference products|×|
|Atlas training products|×|
|Kirin X90 processor series products|√|
|Kirin 9030 processor series products|√|

## Function

- Operator function: For the input data x, performs fake quantization on the input x along the specified axis using scale and zero_point, updates the value range of the fake quantization output according to quant_min and quant_max, and finally returns the result y and the corresponding position mask mask.
- Formula: According to the operator function, the operator first computes the temporary variable qval, and then computes out and mask.

  $$
  qval = Round(std::nearby\_int(x / scale) + zeroPoint)
  $$

  $$
  y = (Min(quantMax, Max(quantMin, qval)) - zeroPoint) * scale
  $$

  $$
  mask = (qval >= quantMin)   \&  (qval <= quantMax)
  $$

## Parameters

<table style="undefined;table-layout: fixed; width: 1250px"><colgroup>
  <col style="width: 150px">
  <col style="width: 150px">
  <col style="width: 500px">
  <col style="width: 250px">
  <col style="width: 200px">
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
      <td>x</td>
      <td>Input</td>
      <td>It is `x` in the formulas.</td>
      <td>FLOAT, FLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>scale</td>
      <td>Input</td>
      <td>It is `scale` in the formulas, representing the scaling coefficient of the fake quantization of the input. Only 1 dimension is supported for the shape.</td>
      <td>FLOAT, FLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>zero_point</td>
      <td>Input</td>
      <td>It is `zero_point` in the formulas, representing the zero-based parameter of the fake quantization of the input. Only 1 dimension is supported for the shape.</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>axis</td>
      <td>Attribute</td>
      <td>Represents the computation dimension.</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>quant_min</td>
      <td>Attribute</td>
      <td>Represents the minimum value of the input data after fake quantization.</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>quant_max</td>
      <td>Attribute</td>
      <td>Represents the maximum value of the input data after fake quantization.</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>y</td>
      <td>Output</td>
      <td>It is `y` in the formulas. The dimensions are consistent with those of the computation input `x`.</td>
      <td>FLOAT, FLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>mask</td>
      <td>Output</td>
      <td>It is `mask` in the formulas. The dimensions are consistent with those of the computation input `x`.</td>
      <td>BOOL</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## Constraints

None

## Invocation Description

| Invocation method   | Sample code           | Description                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn invocation  | [test_aclnn_fake_quant_per_channel_affine_cachemask.cpp](examples/test_aclnn_fake_quant_per_channel_affine_cachemask.cpp) | Invokes the FakeQuantAffineCachemask operator through the [aclnnFakeQuantPerChannelAffineCachemask](docs/aclnnFakeQuantPerChannelAffineCachemask_en.md) interface. |
| aclnn invocation  | [test_aclnn_fake_quant_per_tensor_affine_cachemask.cpp](examples/test_aclnn_fake_quant_per_tensor_affine_cachemask.cpp) | Invokes the FakeQuantAffineCachemask operator through the [aclnnFakeQuantPerTensorAffineCachemask](docs/aclnnFakeQuantPerTensorAffineCachemask_en.md) interface. |
| Graph-mode invocation | - | Builds the graph and invokes the FakeQuantAffineCachemask operator through the [operator IR](op_graph/fake_quant_affine_cachemask_proto.h).         |

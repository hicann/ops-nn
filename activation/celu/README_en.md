# Celu

English | [简体中文](./README.md)

## Applicable Products

|Product             |  Support  |
|:-------------------------|:----------:|
|  <term>Ascend 950PR&950DT products</term>   |     √    |
|  <term>Atlas A3 products</term>   |     √    |
|  <term>Atlas A2 products</term>     |     √    |
|  <term>Atlas 200I/500 A2 inference products</term>    |     √    |
|  <term>Atlas inference products</term>     |     √    |
|  <term>Atlas training products</term>    |     √    |

## Function

- **Operator function**: Celu (Continuously Differentiable Exponential Linear Unit) is a continuously differentiable activation function that scales positive inputs linearly by a coefficient and returns an exponentially decaying curve for negative inputs. It is compatible with the Celu operator of ONNX.
- **Formula**:

  $$
  y = \begin{cases}
  \alpha_3 \times x & \text{if } x \geq 0 \\
  \alpha_1 \times \left(\exp\left(\frac{x}{\alpha_2}\right) - 1\right) & \text{if } x < 0
  \end{cases}
  $$

  Here:
  - When `x >= 0`, the output is the linear scaling `alpha3 * x` (when the default `alpha3=1` applies, this is the identity `y=x`, consistent with ONNX Celu)
  - When `x < 0`, the output is the exponential decay form `alpha1 * (exp(x / alpha2) - 1)`
- **Precision protection**: the computation uses FP32 intermediate precision (FP16 inputs are first promoted to FP32), and the upper limit of the `exp` input is uniformly truncated to 87.0 to avoid FP32 exponential overflow.

## Parameters

<table style="table-layout: fixed; width: 100%"><colgroup>
<col style="width: 12%">
<col style="width: 12%">
<col style="width: 32%">
<col style="width: 24%">
<col style="width: 20%">
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
    <td>Input x in the formula, an ND tensor. For the rank restrictions, see Constraints.</td>
    <td>FLOAT16, FLOAT</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>alpha1</td>
    <td>Attribute</td>
    <td><ul><li>Scale coefficient of the exponential curve in the negative region, corresponding to alpha1 in the formula.</li><li>The default value is 1.0.</li></ul></td>
    <td>FLOAT</td>
    <td>-</td>
  </tr>
  <tr>
    <td>alpha2</td>
    <td>Attribute</td>
    <td><ul><li>Slope coefficient of the exponential curve in the negative region, corresponding to alpha2 in the formula, and it must be greater than 0.</li><li>The default value is 1.0.</li></ul></td>
    <td>FLOAT</td>
    <td>-</td>
  </tr>
  <tr>
    <td>alpha3</td>
    <td>Attribute</td>
    <td><ul><li>Linear scaling coefficient in the positive region, corresponding to alpha3 in the formula.</li><li>The default value is 1.0.</li></ul></td>
    <td>FLOAT</td>
    <td>-</td>
  </tr>
  <tr>
    <td>y</td>
    <td>Output</td>
    <td>Output y in the formula, a tensor with the same shape as the input x.</td>
    <td>FLOAT16, FLOAT</td>
    <td>ND</td>
  </tr>
</tbody></table>

## Constraints

- The supported data types of the input tensor are FLOAT16 and FLOAT, and the output has the same data type as the input.
- The format of the input tensor must be ND.
- The alpha2 attribute must be greater than 0; otherwise, division by zero or numerical instability may occur.
- The input supports 0-element empty tensors. On Ascend 950 (Arch35), the input rank must not exceed 8; when the rank is greater than 8, host tiling validation fails.

## Invocation Description

| Invocation method | Invocation sample | Description |
|---|---|---|
| Graph-mode invocation | [test_geir_celu](./examples/test_geir_celu.cpp) | Invokes the Celu operator by building a graph through the [operator IR](./op_graph/celu_proto.h). |

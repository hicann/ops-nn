# SoftplusV2Grad

English | [简体中文](./README.md)

## Applicable Products

|Product             |  Support  |
|:-------------------------|:----------:|
|  <term>Ascend 950PR&950DT products</term>   |     √    |
|  <term>Atlas A3 products</term>   |     √    |
|  <term>Atlas A2 products</term>     |     √    |
|  <term>Atlas 200I/500 A2 inference products</term>    |     ×    |
|  <term>Atlas inference products</term>     |     ×    |
|  <term>Atlas training products</term>    |     √    |

## Function

- Operator function: Backpropagation of [aclnnSoftplus](../softplus_v2/docs/aclnnSoftplus_en.md). Based on the forward input self and the upstream gradient gradOutput, it computes the gradient gradInput of the forward input.

- Formula:

  $$
  gradInput = gradOutput \cdot \begin{cases}
  \dfrac{1}{1+e^{(-\beta \cdot self)}}, & \beta \cdot self \le threshold \\
  1, & \beta \cdot self > threshold
  \end{cases}
  $$

  When $\beta \cdot self > threshold$, the gradient degenerates to 1.0 (that is, $gradInput = gradOutput$).

## Parameters

<table style="undefined;table-layout: fixed; width: 900px"><colgroup>
  <col style="width: 140px">
  <col style="width: 120px">
  <col style="width: 280px">
  <col style="width: 200px">
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
      <td>input_gradients</td>
      <td>Input</td>
      <td>Upstream gradient tensor, gradOutput in the formula. It supports 0D to 25D (kernel/GEIR; the original ACLNN rank≤8), has the same dtype as input_features, and satisfies a broadcast relationship.</td>
      <td>FLOAT, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>input_features</td>
      <td>Input</td>
      <td>Original input tensor of forward propagation, self in the formula. It supports 0D to 25D (kernel/GEIR; the original ACLNN rank≤8), has the same dtype as input_gradients, and satisfies a broadcast relationship.</td>
      <td>FLOAT, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>beta</td>
      <td>Optional attribute</td>
      <td>Supports FLOAT values, including ±Inf and NaN.</td>
      <td>FLOAT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>threshold</td>
      <td>Optional attribute</td>
      <td>Supports FLOAT values, including ±Inf and NaN.</td>
      <td>FLOAT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>output_backprops</td>
      <td>Output</td>
      <td>Gradient of the forward input, gradInput in the formula. The shape is the broadcast result of input_gradients and input_features, and the dtype is the same as the two inputs.</td>
      <td>FLOAT, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## Constraints

- In graph mode, the dtypes of input_gradients and input_features must be the same. The aclnn interface validates the dtype by platform; see the interface document for the supported range. Inputs are converted as needed before computation, and the output is converted to the data type of gradInput.
- input_gradients and input_features satisfy a broadcast relationship, and the shape of output_backprops is the broadcast result of the two (the aclnn interface requires gradOutput, self, and gradInput to have the same shape, and broadcasting is not supported).
- beta and threshold are Host-side scalar attributes, allowing ±Inf and NaN; the result may contain Inf or NaN.
- Graph mode supports 0D to 25D inputs, and the aclnn interface supports 0D to 8D inputs.
- Deterministic implementation by default.

## Invocation Description

| Invocation method | Invocation sample                                                                   | Description                                                             |
|--------------|------------------------------------------------------------------------|----------------------------------------------------------------|
| aclnn invocation | [test_aclnn_softplus_v2_grad](./examples/test_aclnn_softplus_v2_grad.cpp) | Calls the SoftplusV2Grad operator through the [aclnnSoftplusBackward](./docs/aclnnSoftplusBackward_en.md) interface.    |
| Graph-mode invocation | [test_geir_softplus_v2_grad](./examples/test_geir_softplus_v2_grad.cpp)   | Calls the SoftplusV2Grad operator by building a graph with the <a href="op_graph/softplus_v2_grad_proto.h">operator IR</a>. |

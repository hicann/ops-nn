# SeluGrad

English | [简体中文](./README.md)

## Applicable Products

|Product             |  Support  |
|:-------------------------|:----------:|
|  <term>Ascend 950PR&950DT products</term>   |     √    |
|  <term>Atlas A3 products</term>   |     √    |
|  <term>Atlas A2 products</term>     |     √    |
|  <term>Atlas 200I/500 A2 inference products</term>    |     ×    |
|  <term>Atlas inference products</term>     |     √    |
|  <term>Atlas training products</term>    |     √    |

## Function

- Operator function: computes the backward gradient of the SELU (Scaled Exponential Linear Unit) activation function for the input tensor.
- Formula:

  $$
  y = \begin{cases} \text{scale} \times \text{gradients}, & \text{outputs} \ge 0 \\ \text{gradients} \times (\text{outputs} + \text{scale} \times \alpha), & \text{outputs} < 0 \end{cases}
  $$

  where:

  - $\alpha = 1.6732632423543772848170429916717$
  - $\text{scale} = 1.0507009873554804934193349852946$
  - $\text{scale} \times \alpha = 1.7580993408473768599402175208123$

  Piecewise behavior:

  - When $\text{outputs} \ge 0$: $y = \text{scale} \times \text{gradients}$ (gradient of the linear region)
  - When $\text{outputs} < 0$: $y = \text{gradients} \times (\text{outputs} + \text{scale} \times \alpha)$ (gradient of the exponential saturation region)

## Parameters

<table style="table-layout: fixed; width: 1576px"><colgroup>
<col style="width: 170px">
<col style="width: 170px">
<col style="width: 200px">
<col style="width: 200px">
<col style="width: 170px">
<col style="width: 170px">
</colgroup>
<thead>
  <tr>
    <th>Parameter name</th>
    <th>Input/Output</th>
    <th>Description</th>
    <th>Data type</th>
    <th>Data format</th>
    <th>Dimensions (shape)</th>
  </tr></thead>
<tbody>
  <tr>
    <td>gradients</td>
    <td>Input</td>
    <td>Upstream gradient of backward propagation.</td>
    <td>FLOAT, FLOAT16, BFLOAT16, INT32, INT8, UINT8</td>
    <td>ND</td>
    <td>0-8</td>
  </tr>
  <tr>
    <td>outputs</td>
    <td>Input</td>
    <td>SELU forward output. Its shape must be exactly the same as gradients (broadcasting is not supported).</td>
    <td>FLOAT, FLOAT16, BFLOAT16, INT32, INT8, UINT8</td>
    <td>ND</td>
    <td>0-8</td>
  </tr>
  <tr>
    <td>y</td>
    <td>Output</td>
    <td>Backward gradient result, with a shape exactly the same as gradients/outputs.</td>
    <td>FLOAT, FLOAT16, BFLOAT16, INT32, INT8, UINT8</td>
    <td>ND</td>
    <td>0-8</td>
  </tr>
</tbody></table>

## Constraints

- The shapes of gradients and outputs must be exactly the same (broadcasting is not supported). An error is returned when the shapes are inconsistent.
- Ranks 0-8 are supported, including rank-0 scalars and empty tensors with zero-length dimensions.
- Deterministic computation: SeluGrad is a default deterministic implementation.

## Invocation Description

| Invocation method | Invocation sample                                                                   | Description                                                           |
|--------------|------------------------------------------------------------------------|--------------------------------------------------------------|
| aclnn invocation | [test_aclnn_selu_grad](./examples/arch35/test_aclnn_selu_grad.cpp) | Invokes the SeluGrad operator through the [aclnnSeluBackward](./docs/aclnnSeluBackward_en.md) interface. |
| GEIR invocation | [test_geir_selu_grad](./examples/arch35/test_geir_selu_grad.cpp) | Builds a graph using the [SeluGrad IR prototype](./op_graph/selu_grad_proto.h). |

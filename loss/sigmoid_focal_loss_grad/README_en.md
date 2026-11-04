# SigmoidFocalLossGrad

English | [简体中文](./README.md)

## Applicable Products

| Product | Support |
| :--- | :---: |
| <term>Ascend 950PR&950DT products</term> | √ |
| <term>Atlas A3 products</term>| √ |
| <term>Atlas A2 products</term> | √ |
| <term>Atlas 200I/500 A2 inference products</term> | × |
| <term>Atlas inference products</term> | × |
| <term>Atlas training products</term> | √ |

## Function

SigmoidFocalLossGrad computes the raw selector backward gradient of the Sigmoid Focal Loss with respect to the forward logits `pred`. It applies to the backward training process using Focal Loss, such as object detection. `target` is the complement of the forward dense target and cannot be directly used as an ordinary positive-class label of high-level frameworks.

Let `p=sigmoid(pred)`, `t=target`, and treat `weight` as all ones when it is absent. Then:

$$
dpos = \alpha\gamma p(1-p)^\gamma\log(p)-\alpha(1-p)^{\gamma+1}
$$

$$
dneg = \gamma(\alpha-1)p^\gamma(1-p)\log(1-p)+(1-\alpha)p^{\gamma+1}
$$

$$
grad = (dpos(1-t)+dneg\,t)\times dout\times weight
$$

When `reduction=mean`, the element-wise result is further divided by the number of elements of `pred`. `sum` and `none` do not scale the result. The output shapes are all the same as that of `pred`.

## Parameters

<table style="table-layout: fixed; width: 1576px">
<colgroup>
<col style="width: 170px">
<col style="width: 170px">
<col style="width: 200px">
<col style="width: 200px">
<col style="width: 170px">
</colgroup>
<thead><tr><th>Parameter name</th><th>Input/Output/Attribute</th><th>Description</th><th>Data type</th><th>Data format</th></tr></thead>
<tbody>
<tr><td>pred</td><td>Input</td><td>Forward logits, which corresponds to pred in the formula.</td><td>FLOAT16, FLOAT</td><td>ND</td></tr>
<tr><td>target</td><td>Input</td><td>Raw backward selector, which corresponds to t in the formula.</td><td>INT32</td><td>ND</td></tr>
<tr><td>dout</td><td>Input</td><td>Upstream gradient, which corresponds to dout in the formula.</td><td>FLOAT16, FLOAT</td><td>ND</td></tr>
<tr><td>weight</td><td>Optional input</td><td>Element-wise sample weight, which corresponds to weight in the formula. When it is absent, the value is equivalent to all ones.</td><td>FLOAT16, FLOAT</td><td>ND</td></tr>
<tr><td>grad</td><td>Output</td><td>Gradient of `pred`. Both the data type and shape follow `pred`.</td><td>FLOAT16, FLOAT</td><td>ND</td></tr>
<tr><td>alpha</td><td>Optional attribute</td><td>Class balancing weight, which corresponds to alpha in the formula. Default value: 0.25.</td><td>FLOAT</td><td>-</td></tr>
<tr><td>gamma</td><td>Optional attribute</td><td>Focusing exponent, which corresponds to gamma in the formula. Default value: 2.0.</td><td>FLOAT</td><td>-</td></tr>
<tr><td>reduction</td><td>Optional attribute</td><td>Result scaling method. The value can be mean, sum, or none. Default value: mean.</td><td>STRING</td><td>-</td></tr>
</tbody>
</table>

## Constraints

- `pred`, `target`, `dout`, `grad`, and `weight` (when present) must be non-empty 2D ND tensors with completely identical shapes. Broadcasting is not supported.
- `target` supports only INT32, and the business label semantics are 0 or 1. The IR and Kernel do not check device-side values element by element before running.
- `pred`, `dout`, and `weight` can use FLOAT16 or FLOAT respectively, and the data type of `grad` must be the same as that of `pred`.
- `alpha` and `gamma` must be finite numbers. `reduction` can only be mean, sum, or none.
- The Ascend 950 graph mode supports dynamic shapes (-1 for unknown dimensions) and does not support dynamic ranks (-2 for unknown ranks).

## Invocation Description

| Invocation method | Sample code | Description |
| :--- | :--- | :--- |
| GE graph mode | [test_geir_sigmoid_focal_loss_grad.cpp](examples/arch35/test_geir_sigmoid_focal_loss_grad.cpp) | Builds the operator graph with the [SigmoidFocalLossGrad IR definition](op_graph/sigmoid_focal_loss_grad_proto.h). |

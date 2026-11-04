# ForeachNonFiniteCheckAndUnscale

English | [简体中文](./README.md)

## Applicable Products

|Product             |  Support  |
|:-------------------------|:----------:|
|  <term>Ascend 950PR&950DT products</term>   |     √    |
|  <term>Atlas A3 products</term>   |     √    |
|  <term>Atlas A2 products</term>     |     √    |
|  <term>Atlas 200I/500 A2 inference products</term>    |     ×    |
|  <term>Atlas inference products</term>    |     √    |
|  <term>Atlas training products</term>    |     ×    |
|  <term>Kirin X90 processor series products</term> | √ |
|  <term>Kirin 9030 processor series products</term> | √ |

## Function

- Operator function: traverses all tensors in scaledGrads, checks whether Inf or NaN exists, sets foundInf to 1.0 if it exists, keeps the value of foundInf unchanged otherwise, and unscales all tensors in scaledGrads.

- Formula:
  $$
  foundInf = \begin{cases}1.0, & When\ (Inf \in  scaledGrads)\ or\ (NaN \in scaledGrads),\\
    foundInf, & otherwise.
  \end{cases}
  $$

  $$
   scaledGrads_i = {scaledGrads}_{i}*{invScale}.
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
      <td>scaled_grads</td>
      <td>Input/Output</td>
      <td>Indicates the input and output tensor list of the unscale computation, corresponding to `scaledGrads` in the formula. The maximum supported length is 256.</td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>found_inf</td>
      <td>Input</td>
      <td>Indicates the tensor used to mark whether Inf or NaN exists in input `scaled_grads`, corresponding to `foundInf` in the formula. It contains only one element. If a tensor with Inf or NaN exists in input `scaled_grads`, `found_inf` is set to 1.0; otherwise, no operation is performed on `found_inf`. Finally, all values in `scaled_grads` are multiplied by `inv_scale` and stored in `scaled_grads`.</td>
      <td>FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>inv_scale</td>
      <td>Input</td>
      <td>Indicates the tensor used for the unscale computation, corresponding to `invScale` in the formula. It contains only one element.</td>
      <td>FLOAT32</td>
      <td>ND</td>
    </tr>
  </tbody></table>

- <term>Kirin X90 processor series products</term> and <term>Kirin 9030 processor series products</term>: the data type of scaled_grads does not support BFLOAT16.

## Constraints

- The input/output of the inplace operation does not support non-contiguous tensors.

## Invocation Description

| Invocation method   | Sample code           | Description                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn interface  | [test_aclnn_foreach_non_finite_check_and_unscale](examples/test_aclnn_foreach_non_finite_check_and_unscale.cpp) | Invokes the ForeachNonFiniteCheckAndUnscale operator through the [aclnnForeachNonFiniteCheckAndUnscale](docs/aclnnForeachNonFiniteCheckAndUnscale_en.md) interface. |
| Graph mode | -  | Invokes the ForeachNonFiniteCheckAndUnscale operator through the [operator IR](op_graph/foreach_non_finite_check_and_unscale_proto.h) graph construction method.         |

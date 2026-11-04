# KlDivTargetLossGrad

English | [简体中文](./README.md)

## Applicable Products

|Product             |  Support  |
|:-------------------------|:----------:|
|  <term>Ascend 950PR&950DT products</term>   |     √    |
|  <term>Atlas A3 products</term>  |     ×    |
|  <term>Atlas A2 products</term>     |     ×    |
|  <term>Atlas 200I/500 A2 inference products</term>    |     ×    |
|  <term>Atlas inference products</term>     |     ×    |
|  <term>Atlas training products</term>    |     ×    |

## Function

- Operator function: Performs the target backward computation for the result of the [aclnnKlDiv](https://gitcode.com/cann/ops-math/blob/master/math/kl_div_v2/docs/aclnnKlDiv.md) API.

- Formula:

$$
out =
\begin{cases}
gradOutput * (log(target) - self + 1), ~~~~~~~~~~~~~~~~~~~~~~~~~logTarget=False \\
gradOutput * exp(target) * (target - self +1), ~~~~~~~~~~logTarget=True
\end{cases}
$$

## Parameters

<table style="undefined;table-layout: fixed; width: 1420px"><colgroup>
  <col style="width: 215px">
  <col style="width: 163px">
  <col style="width: 287px">
  <col style="width: 439px">
  <col style="width: 135px">
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
        <td>gradOutput</td>
        <td>Input</td>
        <td>Backward input gradient, gradOutput in the formula. Its shape needs to satisfy the broadcast relationship with target.</td>
        <td>FLOAT, FLOAT16, BFLOAT16</td>
        <td>ND</td>
      </tr>
      <tr>
        <td>self</td>
        <td>Input</td>
        <td>Input tensor, self in the formula. Its shape needs to satisfy the broadcast relationship with target.</td>
        <td>FLOAT, FLOAT16, BFLOAT16</td>
        <td>ND</td>
      </tr>
      <tr>
        <td>target</td>
        <td>Input</td>
        <td>Input tensor, target in the formula.</td>
        <td>FLOAT, FLOAT16, BFLOAT16</td>
        <td>ND</td>
      </tr>
      <tr>
        <td>reduction</td>
        <td>Attribute</td>
        <td>Specifies the reduction method applied to the output. 0(none) | 1(mean) | 2(sum) | 3(batchmean) are supported.
            <ul>
            <li>0(none): No reduction is applied</li>
            <li>1(mean): The sum of the output is divided by the number of elements in the output</li>
            <li>2(sum): The output is summed</li>
            <li>3(batchmean): The sum of the output is divided by the number of batches</li>
            </ul>
        </td>
        <td>STRING</td>
        <td>-</td>
      </tr>
      <tr>
        <td>logTarget</td>
        <td>Attribute</td>
        <td>Whether to convert target to the log space.</td>
        <td>BOOL</td>
        <td>-</td>
      </tr>
      <tr>
        <td>out</td>
        <td>Output</td>
        <td>Output loss gradient, out in the formula. Its shape is the same as target.</td>
        <td>FLOAT, FLOAT16, BFLOAT16</td>
        <td>ND</td>
      </tr>
    </tbody></table>

## Constraints

None

## Invocation Description

| Invocation method | Invocation sample                                                                   | Description                                                           |
|--------------|------------------------------------------------------------------------|--------------------------------------------------------------|
| aclnn invocation | [test_aclnn_kl_div_target_backward](./examples/test_aclnn_kl_div_target_backward.cpp) | Calls the KlDivTargetLossGrad operator through the [aclnnKlDivTargetBackward](./docs/aclnnKlDivTargetBackward_en.md) interface. |

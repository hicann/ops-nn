# LambApplyOptimizerAssign

English | [简体中文](./README.md)

## Applicable Products

|Product             |  Support  |
|:-------------------------|:----------:|
|  <term>Ascend 950PR&950DT products</term>   |     √    |
|  <term>Atlas A3 products</term>  |     √    |
|  <term>Atlas A2 products</term>     |     √    |
|  <term>Atlas 200I/500 A2 inference products</term>    |     ×    |
|  <term>Atlas inference products</term>    |     ×    |
|  <term>Atlas training products</term>    |     ×    |
|  <term>Kirin X90 processor series products</term> | × |
|  <term>Kirin 9030 processor series products</term> | × |

## Function

- Operator function: For a parameter in the model, performs the update computation of the LAMB optimizer after the Adam moment update and bias correction (including optional weight decay), and updates the first moment inputm and the second moment inputv in place.

- Formula:

  $next\_v = inputv \times mul2\_x + grad^2 \times mul3\_x$

  $next\_m = inputm \times mul0\_x + grad \times mul1\_x$

  $output0 = \frac{next\_m / (1 - mul0\_x^{steps})}{\sqrt{next\_v / (1 - mul2\_x^{steps})} + add2\_y} + input3 \times weight\_decay\_rate \times do\_use\_weight$

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
      <th>Input/Output</th>
      <th>Description</th>
      <th>Data type</th>
      <th>Data format</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>grad</td>
      <td>Input</td>
      <td>Empty tensors are supported. It is grad (gradient) in the formulas; the shape supports 0-8 dimensions and must satisfy the broadcast rules with other inputs.</td>
      <td>FLOAT16, FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>inputv</td>
      <td>Input</td>
      <td>Empty tensors are supported. It is inputv (second moment) in the formulas. <b>Updated in place; its shape is the output shape</b>. It must be exactly the same as inputm, and the broadcast result of all inputs must exactly equal this shape.</td>
      <td>FLOAT16, FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>inputm</td>
      <td>Input</td>
      <td>Empty tensors are supported. It is inputm (first moment) in the formulas. <b>Updated in place; its shape is the output shape</b>. It must be exactly the same as inputv, and the broadcast result of all inputs must exactly equal this shape.</td>
      <td>FLOAT16, FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>input3</td>
      <td>Input</td>
      <td>Empty tensors are supported. It is input3 (parameter involved in weight decay) in the formulas; the shape supports 0-8 dimensions and must satisfy the broadcast rules with other inputs.</td>
      <td>FLOAT16, FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>mul0_x</td>
      <td>Input</td>
      <td>Empty tensors are supported. It is mul0_x (beta1) in the formulas; the shape supports 0-8 dimensions and must satisfy the broadcast rules with other inputs.</td>
      <td>FLOAT16, FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>mul1_x</td>
      <td>Input</td>
      <td>Empty tensors are supported. It is mul1_x (1-beta1) in the formulas; the shape supports 0-8 dimensions and must satisfy the broadcast rules with other inputs.</td>
      <td>FLOAT16, FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>mul2_x</td>
      <td>Input</td>
      <td>Empty tensors are supported. It is mul2_x (beta2) in the formulas; the shape supports 0-8 dimensions and must satisfy the broadcast rules with other inputs.</td>
      <td>FLOAT16, FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>mul3_x</td>
      <td>Input</td>
      <td>Empty tensors are supported. It is mul3_x (1-beta2) in the formulas; the shape supports 0-8 dimensions and must satisfy the broadcast rules with other inputs.</td>
      <td>FLOAT16, FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>add2_y</td>
      <td>Input</td>
      <td>Empty tensors are supported. It is add2_y (epsilon) in the formulas; the shape supports 0-8 dimensions and must satisfy the broadcast rules with other inputs.</td>
      <td>FLOAT16, FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>steps</td>
      <td>Input</td>
      <td>Empty tensors are supported. It is steps (number of steps) in the formulas; the shape supports 0-8 dimensions and must satisfy the broadcast rules with other inputs.</td>
      <td>FLOAT16, FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>do_use_weight</td>
      <td>Input</td>
      <td>Empty tensors are supported. It is do_use_weight (whether to use weight decay) in the formulas; the shape supports 0-8 dimensions and must satisfy the broadcast rules with other inputs.</td>
      <td>FLOAT16, FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>weight_decay_rate</td>
      <td>Input</td>
      <td>Empty tensors are supported. It is weight_decay_rate (weight decay rate) in the formulas; the shape supports 0-8 dimensions and must satisfy the broadcast rules with other inputs.</td>
      <td>FLOAT16, FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>output0</td>
      <td>Output</td>
      <td>Empty tensors are supported. It is output0 (update) in the formulas; the shape is the same as those of inputv and inputm.</td>
      <td>FLOAT16, FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>inputv</td>
      <td>Output</td>
      <td>Empty tensors are supported. The updated inputv (second moment, updated in place); the shape is the same as that of the input inputv.</td>
      <td>FLOAT16, FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>inputm</td>
      <td>Output</td>
      <td>Empty tensors are supported. The updated inputm (first moment, updated in place); the shape is the same as that of the input inputm.</td>
      <td>FLOAT16, FLOAT</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## Constraints

- Shape constraints: the shapes of all inputs must pairwise satisfy the broadcast rules. `inputv` and `inputm` are momentum outputs updated in place; their shapes must be **exactly the same**, and the broadcast result of all inputs must exactly equal this shape (otherwise the in-place write-back goes out of bounds); the shapes of the three outputs all take this shape. Other inputs (including `grad`) can be broadcast to it.
- The number of dimensions of all inputs and outputs is 0~8. When all inputs are 0-dimensional, the shapes of the three outputs are (1,).

- The data types of all inputs must be the same, all FLOAT16 or all FLOAT.

## Invocation Description

| Invocation method | Invocation sample | Description |
|---|---|---|
| Graph-mode invocation | [test_geir_lamb_apply_optimizer_assign](./examples/test_geir_lamb_apply_optimizer_assign.cpp) | Builds the graph and invokes the LambApplyOptimizerAssign operator through the operator IR. |

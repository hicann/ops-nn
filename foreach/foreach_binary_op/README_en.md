# ForeachBinaryOp

English | [简体中文](./README.md)

## Applicable Products

| Product                                                         | Support |
| :----------------------------------------- | ------|
| <term>Ascend 950PR&950DT products</term>                             |    √     |
| <term>Atlas A3 products</term>     |    ×    |
| <term>Atlas A2 products</term> |    ×    |
| <term>Atlas 200I/500 A2 inference products</term>                      |    ×    |
| <term>Atlas inference products</term>                             |    ×    |
| <term>Atlas training products</term>                              |    ×    |

## Function

- Operator function: Performs a binary operation tensor by tensor and element by element on the two tensor lists `x1` and `x2`; the operation type is selected by the attribute `op_code`. This operator unifies multiple binary foreach operations into one operator, facilitating graph fusion.

- Formula:

  $$
  x1 = [{x1_0}, {x1_1}, ... {x1_{n-1}}],\ x2 = [{x2_0}, {x2_1}, ... {x2_{n-1}}],\ y = [{y_0}, {y_1}, ... {y_{n-1}}]
  $$

  $$
  y_i = x1_i \odot x2_i \quad (i=0,1,...,n-1)
  $$

  Where $\odot$ is determined by `op_code`:

  | op_code | Operation | Formula |
  |:---:|:---:|:---|
  | 0 | add | $y_i = x1_i + x2_i$ |
  | 1 | sub | $y_i = x1_i - x2_i$ |
  | 2 | mul | $y_i = x1_i \times x2_i$ |
  | 3 | div | $y_i = x1_i / x2_i$ |

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
      <td>Represents the first input tensor list of the binary operation, corresponding to `x1` in the formulas. The data types of all tensors in this parameter are consistent.</td>
      <td>FLOAT16, FLOAT, INT32, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>x2</td>
      <td>Input</td>
      <td>Represents the second input tensor list of the binary operation, corresponding to `x2` in the formulas. The data type, data format, and shape are consistent with those of the input `x1`, and the data types of all tensors in this parameter are consistent.</td>
      <td>FLOAT16, FLOAT, INT32, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>op_code</td>
      <td>Attribute</td>
      <td>Represents the type of the binary operation, corresponding to $\odot$ in the formulas. The value range is [0, 4): 0 represents add, 1 represents sub, 2 represents mul, and 3 represents div. When the value is 3 (div), elements of INT32 with a divisor of 0 produce a result of 0, and for floating point with a divisor of 0, inf or nan is produced according to IEEE semantics.</td>
      <td>INT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>y</td>
      <td>Output</td>
      <td>Represents the output tensor list of the binary operation, corresponding to `y` in the formulas. The data type, data format, and shape are consistent with those of the input `x1`, and the data types of all tensors in this parameter are consistent.</td>
      <td>FLOAT16, FLOAT, INT32, BFLOAT16</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## Constraints

- The three lists `x1`, `x2`, and `y` have the same number of tensors, and the tensors in one-to-one correspondence have the same shapes.
- The maximum number of tensors in a list is 256.
- Empty tensors are supported: no computation is performed when the total number of elements is 0.

## Invocation Description

| Invocation method   | Sample code           | Description                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| Graph-mode invocation | [test_geir_foreach_binary_op](examples/arch35/test_geir_foreach_binary_op.cpp) | Builds the graph and invokes the ForeachBinaryOp operator through the [operator IR](op_graph/foreach_binary_op_proto.h). |

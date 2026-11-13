# FusedMatMul

English | [简体中文](./README.md)

## Applicable Products

| Product                                                         | Support |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR&950DT products</term>                             |    √     |
| <term>Atlas A3 products</term>    |    ×     |
| <term>Atlas A2 products</term> |    ×     |
| <term>Atlas 200I/500 A2 inference products</term>                      |    ×     |
| <term>Atlas inference products</term>                             |    ×     |
| <term>Atlas training products</term>                              |    ×     |

## Function

- Operator function: fuses matrix multiplication with common vector computation.
- Formula:

  $$
  y = OP((x1 @ x2 + bias), x3)
  $$

  sigmoid operation:

  $$
  y = sigmoid(x1 @ x2 + bias) = \frac{1}{1 + \exp(-(x1 @ x2 + bias))}
  $$

  16cast32 operation:

  $$
  y = cast\_float32(x1 @ x2 + bias)
  $$

  When a non-default alpha or beta is passed through aclnnFusedMatmulV2 and fusedOpType is "add":

  $$
  y = alpha * (x1 @ x2) + beta * x3
  $$

## Parameters

<table style="undefined;table-layout: fixed; width: 1550px"><colgroup>
    <col style="width: 170px">
    <col style="width: 120px">
    <col style="width: 300px">
    <col style="width: 330px">
    <col style="width: 212px">
    <col style="width: 100px">
    <col style="width: 190px">
    <col style="width: 145px">
    </colgroup>
    <thead>
      <tr>
        <th>Parameter name</th>
        <th>Input/Output</th>
        <th>Description</th>
        <th>Usage instructions</th>
        <th>Data type</th>
        <th>Data format</th>
      </tr></thead>
    <tbody>
      <tr>
        <td>x1</td>
        <td>Input</td>
        <td>Input x1 in the formula.</td>
        <td><ul><li>The data type must satisfy the data type deduction rules with x2 (see <a href="../../docs/en/context/deduction_relationship.md" target="_blank">mutual deduction relationship</a>).</li></ul></td>
        <td>FLOAT16, BFLOAT16, FLOAT32</td>
        <td>ND</td>
      </tr>
      <tr>
        <td>x2</td>
        <td>Input</td>
        <td>Input x2 in the formula.</td>
        <td><ul><li>The data type must satisfy the data type deduction rules with x1 (see <a href="../../docs/en/context/deduction_relationship.md" target="_blank">mutual deduction relationship</a>).</li></ul></td>
        <td>The data type is consistent with x1</td>
        <td>ND</td>
      </tr>
      <tr>
        <td>bias</td>
        <td>Input</td>
        <td>Input bias in the formula.</td>
        <td><ul><li>This parameter takes effect only when fusedOpType is "", "16cast32", "relu", "add", "mul", "gelu_erf", "gelu_tanh", or "sigmoid". The shape must be (N,) or (1, N), where N is the last dimension of the logical shape of x2; pass a null pointer when bias is not used.</li></ul></td>
        <td>FLOAT16, BFLOAT16, FLOAT32</td>
        <td>ND</td>
      </tr>
      <tr>
        <td>x3</td>
        <td>Input</td>
        <td>Input x3 in the formula.</td>
        <td><ul><li>This parameter takes effect only when fusedOpType is "add" or "mul"; in other cases, just pass a null pointer.</li></ul></td>
        <td>The data type is consistent with x1</td>
        <td>ND</td>
      </tr>
      <tr>
        <td>alpha</td>
        <td>Attribute</td>
        <td>Optional coefficient used to scale the matrix multiplication result, with a default value of 1.0.</td>
        <td><ul><li>Set only by interfaces that support scaling coefficients; a non-default value is supported only for the scenario where fusedOpType is "add".</li></ul></td>
        <td>FLOAT</td>
        <td>-</td>
      </tr>
      <tr>
        <td>beta</td>
        <td>Attribute</td>
        <td>Optional coefficient used to scale x3, with a default value of 1.0.</td>
        <td><ul><li>Set only by interfaces that support scaling coefficients; a non-default value is supported only for the scenario where fusedOpType is "add".</li></ul></td>
        <td>FLOAT</td>
        <td>-</td>
      </tr>
      <tr>
        <td>fusedOpType</td>
        <td>Input</td>
        <td>Input OP in the formula.</td>
        <td><ul><li>The fusion mode value must be one of "" (meaning no fusion), "16cast32", "add", "mul", "gelu_erf", "gelu_tanh", "relu", and "sigmoid".</li><li>"scale_add" is for internal operator use only and does not support being passed directly by users; when scaling coefficients are used, this value may appear in run logs.</li></ul></td>
        <td>STRING</td>
        <td>-</td>
      </tr>
      <tr>
        <td>y</td>
        <td>Output</td>
        <td>Output y in the formula.</td>
        <td><ul><li>The data type must be consistent with the data type deduced from x1 and x2 (see <a href="../../docs/en/context/deduction_relationship.md" target="_blank">mutual deduction relationship</a>); when fusedOpType is "16cast32", the data type of y is fixed to FLOAT32.</li></ul></td>
        <td>FLOAT16, BFLOAT16, FLOAT32</td>
        <td>ND</td>
      </tr>
  </tbody></table>

## Constraints

- All fusion modes that support bias require the shape of bias to be (N,) or (1, N), where N is the last dimension of the logical shape of x2. The data type of bias must be consistent with that of the matrix inputs or be FLOAT32; when bias is passed, the K axis of the matrix multiplication must be greater than 0.
- For GELU (gelu_erf, gelu_tanh) and Sigmoid (sigmoid), the computation order is to first compute the matrix multiplication and add bias, and then perform the activation element-wise; when bias is not passed, no bias is added.
- When fusedOpType is "sigmoid", only the Ascend 950PR&950DT products are supported; x1, x2, and y must be two-dimensional, and their data types must be all FLOAT16 or all BFLOAT16. The shape of x1 is (M, K), the shape of x2 is (K, N), and the shape of y is (M, N); M, N, and K must all be greater than 0, and x3 must be passed as a null pointer.
- When fusedOpType is "sigmoid", alpha and beta can only be 1 or left unset, and empty tensors are not supported.

- When fusedOpType is "gelu_erf" or "gelu_tanh", the data types of x1 and x2 must be BFLOAT16 or FLOAT16; when fusedOpType is "" or "relu", the data types of x1 and x2 must be FLOAT32 (only the scenario with HFLOAT32 enabled is supported), BFLOAT16, or FLOAT16; when fusedOpType is "16cast32", the data types of x1 and x2 must be BFLOAT16 or FLOAT16; when fusedOpType is "add" or "mul", the data types of x1, x2, and x3 must be FLOAT32 (only the scenario with HFLOAT32 enabled is supported), BFLOAT16, or FLOAT16.
- When fusedOpType is "16cast32", the data type of the output y must be FLOAT32.
- When alpha or beta is a non-default value, only the three-dimensional non-transposed add scenario on the Ascend 950PR&950DT products is supported; bias and broadcast along the batch axis are not supported, and x1, x2, x3, and y must have the same FLOAT16, BFLOAT16 or FLOAT32 data type (FLOAT32 requires HF32 and runs in HF32 mode).

## Invocation Description

<table style="undefined;table-layout: fixed; width: 900px"><colgroup>
    <col style="width: 170px">
    <col style="width: 300px">
    <col style="width: 430px">
    </colgroup>
    <thead>
      <tr>
        <th>Invocation method</th>
        <th>Sample code</th>
        <th>Description</th>
      </tr></thead>
    <tbody>
      <tr>
        <td>aclnn interface</td>
        <td><a href="examples/arch35/test_aclnn_fused_mat_mul.cpp">test_aclnn_fused_mat_mul</a></td>
        <td>Invokes the FusedMatmul operator through the <a href="docs/aclnnFusedMatmul_en.md">aclnnFusedMatmul</a> and <a href="docs/aclnnFusedMatmulV2_en.md">aclnnFusedMatmulV2</a> interfaces. This sample uses the V1 interface.</td>
      </tr>
      <tr>
        <td>torch interface</td>
        <td><a href="torch_extension/fused_mat_mul.py">fused_matmul</a></td>
        <td>Invokes the FusedMatmul operator through the <a href="docs/torchapi_fused_matmul_en.md">fused_matmul</a> interface.</td>
      </tr>
  </tbody></table>

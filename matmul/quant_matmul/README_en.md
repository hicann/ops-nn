# QuantMatmul

English | [简体中文](./README.md)

## Applicable Products

| Product  | Support |
| ---- | :----: |
| Ascend 950PR&950DT products               |    ×     |
| Atlas A3 products |    √     |
| Atlas A2 products |    √     |
| Atlas 200I/500 A2 inference products               |    ×     |
| Atlas inference products                        |    ×     |
| Atlas training products                        |    ×     |

## Function

- Operator function: Performs quantized matrix multiplication.
- Formula:

$$
out = (x1@x2 + bias) * deqScale
$$

## Parameters

<table class="tg"><thead>
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
      <td>Input x1 in the formula.</td>
      <td>INT8</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>x2</td>
      <td>Input</td>
      <td>Input x2 in the formula.</td>
      <td>INT8</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>bias</td>
      <td>Input</td>
      <td>Input bias in the formula.</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>deqScale</td>
      <td>Input</td>
      <td>Quantization parameter, input deqScale in the formula.</td>
      <td>UINT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>adjX1</td>
      <td>Input</td>
      <td>Whether the input shape of x1 contains transpose.</td>
      <td>bool</td>
      <td>-</td>
    </tr>
    <tr>
      <td>adjX2</td>
      <td>Input</td>
      <td>Whether the input shape of x2 contains transpose.</td>
      <td>bool</td>
      <td>-</td>
    </tr>
    <tr>
      <td>out</td>
      <td>Output</td>
      <td>Output out in the formula.</td>
      <td>FLOAT16</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## Constraints

- Empty tensors are not supported.
- Contiguous tensors are supported. [Non-contiguous tensors](../../docs/en/context/non_contiguous_tensor.md) support only the transpose scenario.

## Invocation Description

| Invocation method   | Sample code           | Description                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn interface  | [test_aclnn_quant_matmul_v2](examples/test_aclnn_quant_matmul_v2.cpp) | Calls the QuantMatmul operator through <br>[aclnnQuantMatmul](docs/aclnnQuantMatmul_en.md)<br>[aclnnQuantMatmulV2](docs/aclnnQuantMatmulV2_en.md)<br>and other interfaces. |

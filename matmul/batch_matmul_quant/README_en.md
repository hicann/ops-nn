# BatchMatmulQuant

English | [简体中文](./README.md)

## Applicable Products

| Product | Support |
| ---- | :----: |
| Ascend 950PR&950DT products |    ×     |
| Atlas A3 products |    √     |
| Atlas A2 products |    √     |
| Atlas 200I/500 A2 inference products |    ×     |
| Atlas inference products |    ×     |
| Atlas training products |    ×     |

## Function

- Operator function:
Performs matrix multiplication where the dtype of the input tensor is float16 and the dtype of the output is int8.

- Formula:

  $$
  out = Quant(x1@x2 + bias)
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
      <td>FLOAT16</td>
      <td>-</td>
    </tr>
    <tr>
      <td>x2</td>
      <td>Input</td>
      <td>Input x2 in the formula.</td>
      <td>FLOAT16</td>
      <td>-</td>
    </tr>
    <tr>
      <td>bias</td>
      <td>Input</td>
      <td>Input bias in the formula.</td>
      <td>FLOAT16</td>
      <td>-</td>
    </tr>
    <tr>
      <td>quantParam</td>
      <td>Input</td>
      <td>Quantization parameter used by the hardware to complete the quantization computation, which can be obtained through the <a href="../../quant/trans_quant_param/docs/aclnnTransQuantParam_en.md">aclnnTransQuantParam</a> API.</td>
      <td>UINT64</td>
      <td>NC1HWC0</td>
    </tr>
    <tr>
      <td>transposeX1</td>
      <td>Input</td>
      <td>Describes whether x1 is transposed.</td>
      <td>bool</td>
      <td>-</td>
    </tr>
    <tr>
      <td>transposeX2</td>
      <td>Input</td>
      <td>Describes whether x2 is transposed.</td>
      <td>bool</td>
      <td>-</td>
    </tr>
    <tr>
      <td>out</td>
      <td>Output</td>
      <td>Output out in the formula.</td>
      <td>INT8</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## Constraints

- Empty tensors are not supported.
- Contiguous tensors are supported, and [non-contiguous tensors](../../docs/en/context/non_contiguous_tensor.md) support only the transpose scenario.

## Invocation Description

| Invocation method   | Sample code           | Description                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn API  | [test_aclnn_batch_matmul_quant](examples/test_aclnn_batch_matmul_quant.cpp) | Calls the BatchMatmulQuant operator through<br>[aclnnBatchMatmulQuant](docs/aclnnBatchMatmulQuant_en.md). |

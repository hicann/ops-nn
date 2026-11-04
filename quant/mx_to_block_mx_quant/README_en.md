# MxToBlockMxQuant

English | [简体中文](./README.md)

## Applicable Products

| Product                                                         | Support |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR&950DT products</term>                             |    √     |
| <term>Atlas A3 products</term>     |    ×     |
| <term>Atlas A2 products</term> |    ×     |
| <term>Atlas 200I/500 A2 inference products</term>                      |    ×     |
| <term>Atlas inference products</term>                             |    ×     |
| <term>Atlas training products</term>                              |    ×     |

## Function

- Operator function: Converts the FLOAT4 tensor obtained by quantization through `DynamicMxQuant` together with the FLOAT8_E8M0 scaling coefficients into the FLOAT8 block quantization format, and outputs the quantization scales along the -1 axis and the -2 axis.

- Formula:

$$
\begin{aligned}
scale_{fp8} &= max(mxscale_{fp4\_block}) / MAX\_OFFSET \\
offset &= mxscale_{fp4\_block} / scale_{fp8} \\
x_{fp8} &= x_{fp4} \times offset
\end{aligned}
$$

- $mxscale_{fp4\_block}$ is the FP8_E8M0 scaling coefficient provided by the input mxscale. $MAX\_OFFSET$ is the maximum offset between the input and output data types. $x_{fp4}$ is the quantized FLOAT4 tensor. $x_{fp8}$ is the converted FLOAT8 tensor.

- MAX_OFFSET reference table:

| Input     | Output       | MAX_OFFSET |
| -------- | ---------- | ---------- |
| FP4_E2M1 | FP8_E5M2   | 13         |
| FP4_E1M2 | FP8_E5M2   | 15         |
| FP4_E2M1 | FP8_E4M3FN | 6          |
| FP4_E1M2 | FP8_E4M3FN | 8          |

## Parameters

<table style="undefined;table-layout: fixed; width: 980px"><colgroup>
  <col style="width: 100px">
  <col style="width: 150px">
  <col style="width: 280px">
  <col style="width: 330px">
  <col style="width: 120px">
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
      <td>x</td>
      <td>Input</td>
      <td>Data to be quantized. It is the FLOAT4 tensor obtained by quantization through DynamicMxQuant.</td>
      <td>FLOAT4_E2M1, FLOAT4_E1M2</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>mxscale</td>
      <td>Input</td>
      <td>Quantization scale computed by calling DynamicMxQuant.</td>
      <td>FLOAT8_E8M0</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>dst_type</td>
      <td>Attribute</td>
      <td>Specifies the type of the output y after quantization.</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>y</td>
      <td>Output</td>
      <td>Corresponding result of the input x after quantization. Its shape is consistent with x.</td>
      <td>FLOAT8_E5M2, FLOAT8_E4M3FN</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>scale1</td>
      <td>Output</td>
      <td>Quantization scale corresponding to each group on the -1 axis.</td>
      <td>FLOAT8_E8M0</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>scale2</td>
      <td>Output</td>
      <td>Quantization scale corresponding to each group on the -2 axis. The output needs to be interleaved for every two rows of data.</td>
      <td>FLOAT8_E8M0</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## Constraints

- x supports only 2D or 3D input, the -2 axis is a multiple of 64, and the -1 axis is a multiple of 2. Non-contiguous tensors are not supported, and empty tensors are not supported.
- The shape constraints of x, mxscale, scale1, and scale2 are as follows:
  - rank(mxscale) = rank(x) + 1.
  - mxscale.shape[-2] = (Ceil(x.shape[-1] / 32) + 2 - 1) / 2.
  - mxscale.shape[-1] = 2.
  - The other dimensions are consistent with the input x.
- The shape constraints of the output scale1 are as follows:
  - rank(scale1) = rank(x) + 1.
  - scale1.shape[-2] = (Ceil(x.shape[-1] / 32) + 2 - 1) / 2.
  - scale1.shape[-1] = 2.
  - The other dimensions are consistent with the input x.
- The shape constraints of the output scale2 are as follows:
  - rank(scale2) = rank(x) + 1.
  - scale2.shape[-3] = ((Ceil(x.shape[-2] / 32) + 2 - 1) / 2) * 2 / 2.
  - scale2.shape[-2] = x.shape[-1].
  - scale2.shape[-1] = 2.
  - The other dimensions are consistent with the input x.
- dstType supports only {35, 36}, corresponding to {FLOAT8_E5M2, FLOAT8_E4M3FN}.

## Invocation Description

| Invocation method | Invocation sample                                                                   | Description                                                           |
|--------------|------------------------------------------------------------------------|--------------------------------------------------------------|
| aclnn invocation | [test_aclnn_mx_to_block_mx_quant](./examples/arch35/test_aclnn_mx_to_block_mx_quant.cpp) | Calls the MxToBlockMxQuant operator through the [aclnnMxToBlockMxQuant](./docs/aclnnMxToBlockMxQuant_en.md) interface. |
| PyTorch API | - | Calls the MxToBlockMxQuant operator through the [mx_to_block_mx_quant](./docs/torchapi_mx_to_block_mx_quant_en.md) interface. |

# AscendAntiQuant

English | [简体中文](./README.md)

## Applicable Products

| Product                                                         | Support |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR&950DT products</term>                       |    √     |
| <term>Atlas A3 products</term>     |    √     |
| <term>Atlas A2 products</term>     |    √     |
| <term>Atlas 200I/500 A2 inference products</term>                      |    ×     |
| <term>Atlas inference products</term>                              |    ×     |
| <term>Atlas training products</term>                              |    ×     |

## Function

- Operator function: Dequantizes the input `x` according to the attributes `scale` and `offset` (per-tensor scalar dequantization).

- Formula:
  - When `sqrt_mode` is false:

    $$
    y = cast\_to\_dst\_type((x + offset) * scale)
    $$

  - When `sqrt_mode` is true:

    $$
    y = cast\_to\_dst\_type((x + offset) * scale * scale)
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
      <td>x</td>
      <td>Input</td>
      <td><ul><li>Tensor of the operator input, which corresponds to x in the formula.</li><li>Empty tensors are not supported.</li><li>The shape is the same as that of the output y.</li></ul></td>
      <td>INT8, HIFLOAT8, FLOAT8_E5M2, FLOAT8_E4M3FN</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>scale</td>
      <td>Attribute</td>
      <td><ul><li>Scale value in the dequantization, which corresponds to scale in the formula.</li><li>Per-tensor scalar.</li></ul></td>
      <td>FLOAT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>offset</td>
      <td>Attribute</td>
      <td><ul><li>Offset value in the dequantization, which corresponds to offset in the formula.</li><li>Per-tensor scalar.</li></ul></td>
      <td>FLOAT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>dtype</td>
      <td>Optional attribute</td>
      <td><ul><li>Data type of the output.</li><li>Supported values are 1 and 0, which indicate FLOAT16 and FLOAT32 respectively (consistent with the ge::DataType enumeration).</li><li>Default value: FLOAT32 (ge::DT_FLOAT, that is, 0).</li><li>Must be consistent with the dtype of the output y.</li></ul></td>
      <td>INT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>sqrt_mode</td>
      <td>Optional attribute</td>
      <td><ul><li>Logic of scale participating in the computation, which corresponds to sqrt_mode in the formula.</li><li>Default value: false.</li></ul></td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
    <tr>
      <td>y</td>
      <td>Output</td>
      <td><ul><li>Computation output of the dequantization, which corresponds to y in the formula.</li><li>The shape is the same as that of the input x.</li></ul></td>
      <td>FLOAT16, FLOAT32</td>
      <td>ND</td>
    </tr>
  </tbody>
</table>

- <term>Ascend 950PR&950DT products</term>:
  - Supported combinations of (dtype of the input `x`, dtype of the output `y`):
    - (INT8, FLOAT16), (INT8, FLOAT32)
    - (HIFLOAT8, FLOAT16), (HIFLOAT8, FLOAT32)
    - (FLOAT8_E5M2, FLOAT16), (FLOAT8_E5M2, FLOAT32)
    - (FLOAT8_E4M3FN, FLOAT16), (FLOAT8_E4M3FN, FLOAT32)

## Constraints

- `scale` and `offset` are scalar attributes. Only per-tensor dequantization is supported.
- The shapes of the input `x` and output `y` must be completely identical. Empty tensors are not supported, and the size of each dimension must be greater than 0.
- The `dtype` attribute supports only FLOAT16 and FLOAT32, and must be consistent with the actual dtype of the output `y`.
- Only <term>Ascend 950PR&950DT products</term> are supported. Other product forms are not supported.
- Currently, no aclnn API is provided. Only graph-mode invocation is supported.

## Invocation Description

| Invocation method   | Sample code           | Description                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| Graph mode | [test_geir_ascend_anti_quant](examples/test_geir_ascend_anti_quant.cpp) | Calls the AscendAntiQuant operator through graph construction with GE IR (per-tensor scale/offset scalar attributes). |

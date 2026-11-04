# Dequantize

English | [简体中文](./README.md)

## Applicable Products

| Product | Support |
|:---|:---:|
| <term>Ascend 950PR&950DT products</term> | √ |
| <term>Atlas A3 products</term> | √ |
| <term>Atlas A2 products</term> | √ |
| <term>Atlas 200I/500 A2 inference products</term> | √ |
| <term>Atlas inference products</term> | √ |
| <term>Atlas training products</term> | √ |

## Function

- Operator function: Dequantize the quantized integer tensor and convert it back to a floating-point number.

- Formula:

  Three dequantization modes are supported, specified by the mode attribute:

  **MIN_COMBINED mode:**

  $$
  y = \frac{(x + bias) \times (max\_range - min\_range)}{2^n - 1} + min\_range
  $$

  Here $n$ is the number of bits of the input data type, and $bias = 2^{n-1}$ (signed types) or $0$ (unsigned types).

  **MIN_FIRST mode:**

  For INT8/UINT8:

  $$
  offset = min\_range + \frac{round(-min\_range \times (2^n - 1) / (max\_range - min\_range)) \times (max\_range - min\_range)}{2^n - 1}
  $$

  $$
  y = \frac{(x + bias) \times (max\_range - min\_range)}{2^n - 1} + min\_range - offset
  $$

  For INT32: the same as the MIN_COMBINED mode.

   **SCALED mode:**

   For signed types (INT8, INT32):

   $$
   y = x \times \frac{max\_range}{2^{n-1} - 1}
   $$

   For unsigned types (UINT8):

   $$
   y = x \times \frac{max\_range}{2^{n} - 1}
   $$

   Here $n$ is the number of bits of the input data type.

## Parameters

<table style="undefined;table-layout: fixed; width: 820px"><colgroup>
  <col style="width: 100px">
  <col style="width: 150px">
  <col style="width: 190px">
  <col style="width: 260px">
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
      <td>Quantized input tensor</td>
      <td>INT8, UINT8, INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>min_range</td>
      <td>Input</td>
      <td>Minimum value of the quantization range</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>max_range</td>
      <td>Input</td>
      <td>Maximum value of the quantization range</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>mode</td>
      <td>Attribute</td>
      <td>Dequantization mode. Options: MIN_COMBINED, MIN_FIRST, SCALED</td>
      <td>String</td>
      <td>-</td>
    </tr>
    <tr>
      <td>y</td>
      <td>Output</td>
       <td>Dequantized output tensor. The shape is the same as that of input x</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## Constraints

1. min_range and max_range must satisfy max_range > min_range.
2. In SCALED mode, max_range must be greater than 0.
3. min_range and max_range support broadcasting, and their shapes must be compatible with the shape of input x.

## Invocation Description

| Invocation method | Sample code | Description |
| ---------------- | --------------------------- | --------------------------------------------------- |
| Graph-mode invocation | [test_geir_dequantize.cpp](examples/test_geir_dequantize.cpp) | Invoke the Dequantize operator through graph construction using the [operator IR](op_graph/dequantize_proto.h). |

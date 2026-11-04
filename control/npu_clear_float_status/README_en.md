# NpuClearFloatStatus

English | [简体中文](./README.md)

## Applicable Products

| Product                                                         | Support |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR&950DT products</term>                     |     √    |
| <term>Atlas A3 products</term>   |    ×     |
| <term>Atlas A2 products</term>    |    ×     |
| <term>Atlas 200I/500 A2 inference products</term>                      |    √     |
| <term>Atlas inference products</term>                               |    √     |
| <term>Atlas training products</term>                               |    √     |

## Function

- Operator function: Clears the floating-point overflow status register of each AI core of the NPU, and the output is fixed to 8 float32 zero values.

- Formula:

$$
data = zeros(8, dtype=float32)
$$

## Parameters

<table><thead>
  <tr>
    <th>Parameter name</th>
    <th>Input/Output/Attribute</th>
    <th>Description</th>
    <th>Data type</th>
    <th>Data format</th>
  </tr></thead>
<tbody>
  <tr>
    <td>addr</td>
    <td>Input</td>
    <td>Address placeholder, with shape (8,); the data content does not participate in the computation.</td>
    <td>FLOAT</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>data</td>
    <td>Output</td>
    <td>Fixed output of 8 float32 zero values, with shape (8,).</td>
    <td>FLOAT</td>
    <td>ND</td>
  </tr>
</tbody>
</table>

## Constraints

- The data type of addr must be float32.
- The output data is fixed to 8 float32 zero values, regardless of the input data content.
- addr serves only as a placeholder for the operator input interface, and its data content does not participate in the computation.

## Invocation Description

<table><thead>
  <tr>
    <th>Invocation method</th>
    <th>Invocation sample</th>
    <th>Description</th>
  </tr></thead>
<tbody>
  <tr>
    <td>Graph-mode invocation</td>
    <td><a href="./examples/test_geir_npu_clear_float_status.cpp">test_geir_npu_clear_float_status</a></td>
    <td>See <a href="../../docs/zh/invocation/quick_op_invocation.md">operator invocation</a> to complete operator compilation and verification.</td>
  </tr>
</tbody>
</table>

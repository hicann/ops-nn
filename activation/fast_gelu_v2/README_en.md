# FastGeluV2

English | [简体中文](./README.md)

## Applicable Products

|Product             |  Support  |
|:-------------------------|:----------:|
|  <term>Ascend 950PR&950DT products</term>   |     √    |
|  <term>Atlas A3 products</term>   |     √    |
|  <term>Atlas A2 products</term>     |     √    |
|  <term>Atlas 200I/500 A2 inference products</term>    |     √    |
|  <term>Atlas inference products</term>     |     √    |
|  <term>Atlas training products</term>    |     √    |

## Function

- Operator function: Computes the FastGeluV2 activation function.
- Formula:

$$
FastGeluV2(x_i) = x_i \times (sgn(x_i) \times [-0.1444 \times (clip(0.7071 \times |x_i|,\ max=1.769) - 1.769)^2 + 0.5] + 0.5)
$$

where:

- x: Input tensor
- $sgn(x) = (x + 0.000000000001) / |x + 0.000000000001|$
- $clip(v, max)$ indicates limiting the value of v to within max

## Parameters

<table style="table-layout: auto; width: 100%">
<thead>
    <tr>
    <th style="white-space: nowrap">Parameter name</th>
    <th style="white-space: nowrap">Input/Output</th>
    <th style="white-space: nowrap">Description</th>
    <th style="white-space: nowrap">Data type</th>
    <th style="white-space: nowrap">Data format</th>
    </tr>
</thead>
<tbody>
    <tr>
    <td>x</td>
    <td>Input</td>
    <td>'x' in the formula, indicating the input tensor of the FastGeluV2 activation function.</td>
    <td>BFLOAT16, FLOAT16, FLOAT32</td>
    <td>ND</td>
    </tr>
    <tr>
    <td>y</td>
    <td>Output</td>
    <td>Indicates the output tensor of x computed by FastGeluV2.</td>
    <td>BFLOAT16, FLOAT16, FLOAT32</td>
    <td>ND</td>
    </tr>
</tbody></table>

## Constraints

None.

## Invocation Description

| Invocation method   | Sample code           | Description                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| Graph mode  | [test_geir_fast_gelu_v2.cpp](examples/arch35/test_geir_fast_gelu_v2.cpp) | Calls the FastGeluV2 operator in GE IR graph mode through the [operator IR](./op_graph/fast_gelu_v2_proto.h). |

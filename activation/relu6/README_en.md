# Relu6

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

- Operator function: the Relu6 activation function performs the activation computation element-wise on the input tensor and limits each element value to the `[0, 6]` interval.
- Formula:

$$
Relu6(x) = \min(\max(x, 0), 6)
$$

It is equivalent to the piecewise function:

$$
Relu6(x) =
\begin{cases}
0, & x < 0 \\
x, & 0 \le x \le 6 \\
6, & x > 6
\end{cases}
$$

Here:

- x: input tensor

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
    <td>'x' in the formula, denoting the input tensor of the Relu6 activation function.</td>
    <td>FLOAT16, FLOAT32, INT32, DT_BF16</td>
    <td>ND</td>
    </tr>
    <tr>
    <td>y</td>
    <td>Output</td>
    <td>Denotes the output tensor computed from x by Relu6, with the shape and dtype consistent with input x.</td>
    <td>FLOAT16, FLOAT32, INT32, DT_BF16</td>
    <td>ND</td>
    </tr>
</tbody></table>

## Constraints

- The shapes of input x and output y must be the same.
- Inputs support 1D to 8D tensors; 0-dimensional (scalar) inputs are not supported, and inputs with more than 8 dimensions are not supported either.

## Invocation Description

| Invocation method   | Sample code           | Description                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| Graph mode (GE IR)  | [test_geir_relu6.cpp](examples/test_geir_relu6.cpp) | Invokes the Relu6 operator in GE IR graph mode. |

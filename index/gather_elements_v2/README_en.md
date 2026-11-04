# GatherElementsV2

English | [简体中文](./README.md)

## Applicable Products

| Product                                                         | Support |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR&950DT products</term>                          |    ×     |
| <term>Atlas A3 products</term>    |    √     |
| <term>Atlas A2 products</term>     |    √     |
| <term>Atlas 200I/500 A2 inference products</term>                      |    ×     |
| <term>Atlas inference products</term>                             |    ×     |
| <term>Atlas training products</term>                              |    ×     |
|  <term>Kirin X90 processor series products</term> | √ |
|  <term>Kirin 9030 processor series products</term> | √ |

## Function

- Operator function: gathers data along the specified dimension dim of the input tensor.

- Formula:
  Given a tensor $x$, a dimension $d$, and an index tensor $index$, define n as the number of dimensions of $x$, $i_d$ as the index of dimension $d$, and $index_{i_d}$ as the $i_d$-th index value of the index tensor $index$ along dimension $d$. The gather function on the specified dimension d can be expressed by the following mathematical formula:

  $$
  gather(X,index,d)_{i_0,i_1,\cdots,i_{d-1},i_{d+1},\cdots,i_{n-1}} = x_{i_0,i_1,\cdots,i_{d-1},index_{i_d},i_{d+1},\cdots,i_{n-1}}
  $$

- Example:
  - Example 1:
    Assume the input tensor $x=\begin{bmatrix}1 & 2 & 3\\ 4 & 5 & 6\\ 7 & 8 & 9\end{bmatrix}$, the index tensor $index=\begin{bmatrix}0 & 2\\ 1 & 0\end{bmatrix}$, and $dim = 0$. Then the output tensor is $y=\begin{bmatrix}1 & 8\\ 4 & 2\end{bmatrix}$, and the computation process is as follows:

    $$
    \begin{aligned} y_{0,0}&=x_{index_{0,0}, 0}=x_{0,0}=1 \\
    y_{0,1}&=x_{index_{0,1}, 1}=x_{2,1}=8 \\
    y_{1,0}&=x_{index_{1,0}, 0}=x_{1,0}=4 \\
    y_{1,1}&=x_{index_{1,1}, 1}=x_{0,1}=2 \end{aligned}
    $$

  - Example 2:
    Assume the input tensor $x=\begin{bmatrix}1 & 2 & 3\\ 4 & 5 & 6\\ 7 & 8 & 9\end{bmatrix}$, the index tensor $index=\begin{bmatrix}0 & 2\\ 1 & 0\end{bmatrix}$, and $dim = 1$. Then the output tensor is $y=\begin{bmatrix}1 & 3\\ 5 & 4\end{bmatrix}$, and the computation process is as follows:

    $$
    \begin{aligned} y_{0,0}&=x_{0, index_{0,0}}=x_{0,0}=1 \\
    y_{0,1}&=x_{0, index_{0,1}}=x_{0,2}=3 \\
    y_{1,0}&=x_{1, index_{1,0}}=x_{1,1}=5 \\
    y_{1,1}&=x_{1, index_{1,1}}=x_{1,0}=4 \end{aligned}
    $$

## Parameters

<table style="undefined;table-layout: fixed; width: 1250px"><colgroup>
  <col style="width: 60px">
  <col style="width: 60px">
  <col style="width: 150px">
  <col style="width: 150px">
  <col style="width: 60px">
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
      <td>x in the formula.</td>
      <td>FLOAT, FLOAT16, BFLOAT16, INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>index</td>
      <td>Input</td>
      <td>index in the formula.</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y</td>
      <td>Output</td>
      <td>y in the formula.</td>
      <td>FLOAT, FLOAT16, BFLOAT16, INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>dim</td>
      <td>Optional attribute</td>
      <td><ul><li>d in the formula.</li><li>The default value is 0.</li></ul></td>
      <td>Int</td>
      <td>-</td>
    </tr>
  </tbody></table>

- Kirin X90/Kirin 9030 processor series products: BFLOAT16 is not supported for the data types of x and y.

## Constraints

None

## Invocation Description

| Invocation method | Invocation sample                                                                   | Description                                                           |
|--------------|------------------------------------------------------------------------|--------------------------------------------------------------|
| aclnn invocation | [test_aclnn_gather_elements_v2](./examples/test_aclnn_gather_elements_v2.cpp) | Invokes the GatherElementsV2 operator through the [aclnnGather](./docs/aclnnGather_en.md) interface. |

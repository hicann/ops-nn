# GeGluV2

English | [简体中文](./README.md)

## Applicable Products

|Product             |  Support  |
|:-------------------------|:----------:|
|  <term>Ascend 950PR&950DT products</term>   |     √    |
|  <term>Atlas A3 products</term>   |     √    |
|  <term>Atlas A2 products</term>     |     √    |
|  <term>Atlas 200I/500 A2 inference products</term>    |     ×    |
|  <term>Atlas inference products</term>     |     √    |
|  <term>Atlas training products</term>    |     ×    |
|  <term>kirin X90 processor series products</term>    |     √    |
|  <term>Kirin 9030 processor series products</term>    |     √    |

## Function

- Description: the Gaussian error linear unit activation gate function. Compared with aclnnGeGlu, it extends the function of setting the direction of the data block on which the activation function operates.
- Formula:
  If activateLeft is true, activate is performed on the left half of $self$

    $$
    out_{i}=GeGlu(self_{i}) = Gelu(A) \cdot B
    $$

  If activateLeft is false, activate is performed on the right half of $self$

    $$
    out_{i}=GeGlu(self_{i}) = A \cdot Gelu(B)
    $$

  Here $A$ denotes the left half of $self$, and $B$ denotes the right half of $self$.

## Parameters

<table style="undefined;table-layout: fixed; width: 919px"><colgroup>
  <col style="width: 130px">
  <col style="width: 144px">
  <col style="width: 273px">
  <col style="width: 256px">
  <col style="width: 116px">
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
      <td>self</td>
      <td>Input</td>
      <td>Input self in the formula.</td>
      <td>FLOAT, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>dim</td>
      <td>Optional attribute</td>
      <td>The set slice axis. The corresponding axis of self needs to be split in half, and the axis of self corresponding to dim must be an even number.</td>
      <td>INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>approximate</td>
      <td>Optional attribute</td>
      <td>Index of the activation function used by the GeGlu computation: 0 means "none", and 1 means "tanh".</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>activateLeft</td>
      <td>Attribute</td>
      <td>Direction of the data block on which the activation function operates: false means activate the right half, and true means activate the left half.</td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
    <tr>
      <td>out</td>
      <td>Output</td>
      <td>dx in the formula.</td>
      <td>FLOAT, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
     <td>outGelu</td>
      <td>Output</td>
      <td>outGelu in the formula.</td>
      <td>FLOAT, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
  </tbody></table>

- GeGlu does not include the activateLeft parameter.
- <term>Atlas inference products</term> and <term>Atlas training products</term>: the data types of self, out, and outGelu support FLOAT and FLOAT16.
- <term>kirin X90 processor series products</term> and <term>Kirin 9030 processor series products</term>: the data types of self, out, and outGelu do not support BFLOAT16.

## Constraints

None

## Invocation Description

| Invocation method | Invocation sample                                                                   | Description                                                             |
|--------------|------------------------------------------------------------------------|----------------------------------------------------------------|
| aclnn invocation | [test_aclnn_ge_glu](./examples/test_aclnn_ge_glu.cpp) | Calls the GeGluV2 operator through the [aclnnGeGlu](./docs/aclnnGeGlu_en.md) interface.    |
| aclnn invocation | [test_aclnn_ge_glu_v3](./examples/test_aclnn_ge_glu_v3.cpp) | Calls the GeGluV2 operator through the [aclnnGeGluV3](./docs/aclnnGeGluV3_en.md) interface.    |
| Graph-mode invocation | -  | Calls the GeGluV2 operator by constructing a graph through the [operator IR](./op_graph/ge_glu_v2_proto.h). |

# ClippedSwiglu

English | [简体中文](./README.md)

## Applicable Products

| Product             |  Support  |
|:-------------------------|:----------:|
|  <term>Ascend 950PR&950DT products</term>   |     √    |
|  <term>Atlas A3 products</term>   |     √    |
|  <term>Atlas A2 products</term>     |     √    |
|  <term>Atlas 200I/500 A2 inference products</term>    |     ×    |
|  <term>Atlas inference products</term>     |     ×    |
|  <term>Atlas training products</term>    |     ×    |

## Function

- Description: the Swish gated linear unit activation function with clipping, which implements the SwiGlu computation of x. Compared with the SwiGlu operator, this operator adds some input parameters: groupIndex, alpha, limit, bias, interleaved, and clamp_mode. These parameters support the variant SwiGlu used by the GPT-OSS model, the grouped scenario used by MoE models, and the scenario where some new models need to move the clamp operation to a later stage.

- Computation process:

  For a given input tensor x whose dimensions are [a,b,c,d,e,f,g...], the ClippedSwiglu operator performs the following computation:

  1. Merge the axes of x based on the input parameter dim. After the merge, the dimensions are [pre,cut,after]. The cut axis is the axis to be split into two tensors after the merge, and the split method is front-back splitting or odd-even splitting. pre and after can be 1. For example, when dim is 3, the dimensions of x after the merge are [a*b*c,d,e*f*g*...]. In addition, because the elements of the after axis are stored contiguously and the computation is element-wise, the cut axis and the after axis are merged, so the dimensions of x become [pre,cut].

  2. Filter the pre axis of x based on the input parameter group_index. The formulas are as follows:

     $$
     sum = \text{Sum}(group\_index)
     $$

     $$
     x = x[ : sum, : ]
     $$

     Here sum is the sum of all elements of group_index. When group_index is not provided, this step is skipped.

  3. Split x based on the input parameter interleaved. The formulas are as follows:

     When interleaved is true, it indicates odd-even splitting:

     $$
     A = x[ : , : : 2]
     $$

     $$
     B = x[ : , 1 : : 2]
     $$

     When interleaved is false, it indicates front-back splitting:

     $$
     h = x.shape[1] // 2
     $$

     $$
     A = x[ : , : h]
     $$

     $$
     B = x[ : , h : ]
     $$
  4. Perform the variant SwiGlu computation based on the input parameters alpha, limit, bias, and clamp_mode. The formulas are as follows:

     When clamp_mode is 0, the clamp operation is performed before silu:

     $$
     A = A.clamp(min=None, max=limit)
     $$

     $$
     B = B.clamp(min=-limit, max=limit)
     $$

     $$
     y\_glu = A * sigmoid(alpha * A)
     $$

     $$
     y = y\_glu * (B + bias)
     $$

     When clamp_mode is 1, the clamp operation is performed after silu:

     $$
     y\_glu = A * sigmoid(A)
     $$

     $$
     y\_glu = y\_glu.clamp(min=None, max=limit)
     $$

     $$
     B = B.clamp(min=-limit, max=limit)
     $$

     $$
     y = y\_glu * B
     $$

  5. Reshape the output tensor y so that its number of dimensions is the same as the number of dimensions of x before the axis merge. The size on the dim axis is half of that of x, and the other dimensions are the same as x.

## Parameters

<table style="undefined;table-layout: fixed; width: 970px"><colgroup>
  <col style="width: 181px">
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
      <td>x</td>
      <td>Input</td>
      <td>Input x in the formula. The number of dimensions must be greater than 0, and the dimension corresponding to the input parameter dim must be even.</td>
      <td>FLOAT, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>group_index</td>
      <td>Optional input</td>
      <td>Input group_index in the formula. It must be a 1D tensor. Each element must be an integer greater than or equal to 0, the sum of all elements must be greater than 0, and the sum must be less than or equal to the product of all dimensions before the dim axis in the shape of x (that is, the total number of rows). When dim=0, there is no dimension before the dim axis, the product is regarded as 1, and the sum of all elements must be greater than 0 and less than or equal to 1. The number of supported groups is determined by the UB space size.</td>
      <td>INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>dim</td>
      <td>Optional attribute</td>
      <td>Input dim in the formula, indicating the dimension index for merging and splitting x. The value range is [-x.dim(), x.dim()-1]. The default value is -1.</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>alpha</td>
      <td>Optional attribute</td>
      <td>Input alpha in the formula, indicating the parameter used by the variant SwiGlu. The default value is 1.702.</td>
      <td>FLOAT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>limit</td>
      <td>Optional attribute</td>
      <td>Input limit in the formula, indicating the threshold used by the variant SwiGlu. It must be greater than 0. The default value is 7.0.</td>
      <td>FLOAT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>bias</td>
      <td>Optional attribute</td>
      <td>Input bias in the formula, indicating the bias parameter used by the variant SwiGlu. The default value is 1.0.</td>
      <td>FLOAT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>interleaved</td>
      <td>Optional attribute</td>
      <td>Input interleaved in the formula. Setting it to true indicates odd-even splitting of x, and setting it to false indicates front-back splitting of x. The default value is true.</td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
    <tr>
      <td>clamp_mode</td>
      <td>Optional attribute</td>
      <td>Input clamp_mode in the formula. Setting it to 1 moves the clamp operation of A to after the silu activation, that is, clamps the result of silu(A); setting it to 0 means the clamp operation of A is performed before the silu activation. The default value is 0.</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>y</td>
      <td>Output</td>
      <td>Output y in the formula. The number of dimensions must be the same as the input x. It is half of x on the dimension corresponding to the input parameter dim and the same as x on the other dimensions.</td>
      <td>FLOAT, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
  </tbody></table>

- <term>Atlas A2 products/Atlas A3 products</term>:
  - When the attribute `clamp_mode` is an invalid value, the computation logic is the same as clamp_mode=0.

## Constraints

None.

## Invocation Description

| Invocation method | Invocation sample                                                                   | Description                                                             |
|--------------|------------------------------------------------------------------------|----------------------------------------------------------------|
| aclnn invocation | [test_aclnn_clipped_swiglu](./examples/test_aclnn_clipped_swiglu.cpp) | Invoke the ClippedSwiglu operator through the [aclnnClippedSwiglu](./docs/aclnnClippedSwiglu_en.md) interface.    |
| aclnn invocation | [test_aclnn_clipped_swiglu_v2](./examples/test_aclnn_clipped_swiglu_v2.cpp) | Invoke the ClippedSwiglu operator through the [aclnnClippedSwigluV2](./docs/aclnnClippedSwigluV2_en.md) interface.    |
| Graph-mode invocation | - | Invoke the ClippedSwiglu operator by constructing a graph through the [operator IR](./op_graph/clipped_swiglu_proto.h).    |

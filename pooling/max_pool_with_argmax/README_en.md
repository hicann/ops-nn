# MaxPoolWithArgmax

English | [简体中文](./README.md)

## Applicable Products

|Product             |  Support  |
|:-------------------------|:----------:|
| <term>Ascend 950PR&950DT products</term>                     |    √     |
| <term>Atlas A3 products</term>    |    x     |
| <term>Atlas A2 products</term>     |    x     |
| <term>Atlas 200I/500 A2 inference products</term>                     |    x     |
| <term>Atlas inference products</term>                             |    x     |
| <term>Atlas training products</term>                             |    x     |

## Function

- Description:
Performs the max pooling operation, and outputs both the pooled maximum values and the indices of the corresponding positions.

- Formula:
  - Formula for computing each element in the output tensor, using NCHW as an example:

    $$
    y(N_i, C_j, h, w) = \max\limits_{{k\in[0,k_{H}-1],m\in[0,k_{W}-1]}}x(N_i,C_j,stride[2]\times h + k, stride[3]\times w + m)
    $$

  - argmax outputs the index positions of the maximum values within the pooling window.

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
      <td>Input 4D tensor.</td>
      <td>FLOAT16, FLOAT, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>ksize</td>
      <td>Attribute</td>
      <td>Size of the pooling window, with a length of 4, and all array elements must be greater than 0.</td>
      <td>LIST_INT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>strides</td>
      <td>Attribute</td>
      <td>Stride of the sliding window, with a length of 4, and all array elements must be greater than 0.</td>
      <td>LIST_INT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>padding</td>
      <td>Attribute</td>
      <td>Specifies the padding mode, supporting "SAME" or "VALID".</td>
      <td>STRING</td>
      <td>-</td>
    </tr>
    <tr>
      <td>Targmax</td>
      <td>Attribute</td>
      <td>Specifies the data type of the argmax output, supporting int32(3) or int64(9). The default value is int64(9).</td>
      <td>INT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>include_batch_in_index</td>
      <td>Attribute</td>
      <td>Specifies whether the batch dimension is included when computing argmax indices. Currently, only false is supported.</td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
    <tr>
      <td>data_format</td>
      <td>Attribute</td>
      <td>Data format, supporting "NCHW" or "NHWC". The default value is "NHWC".</td>
      <td>STRING</td>
      <td>-</td>
    </tr>
    <tr>
      <td>nan_prop</td>
      <td>Attribute</td>
      <td>Specifies whether to process NAN values. When true, NAN participates in comparison; when false, NAN is ignored. The default value is false.</td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
    <tr>
      <td>y</td>
      <td>Output</td>
      <td>Pooled maximum value tensor, of the same type as the input x.</td>
      <td>FLOAT16, FLOAT, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>argmax</td>
      <td>Output</td>
      <td>Index positions corresponding to the maximum values.</td>
      <td>INT32, INT64</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## Constraints

- **Value range restrictions:**
  - x: a 4D tensor, supporting FLOAT16, FLOAT, and BFLOAT16 types, and supporting the NCHW and NHWC formats.
  - ksize: a list of length 4. The [NCHW] format requires ksize[0]=1 and ksize[1]=1; the [NHWC] format requires ksize[0]=1 and ksize[3]=1.
  - strides: a list of length 4. The [NCHW] format requires strides[0]=1 and strides[1]=1; the [NHWC] format requires strides[0]=1 and strides[3]=1.
  - padding: supports only the "SAME" or "VALID" mode.
    - SAME: padding makes the output shape equal to ceil(input shape/stride), and when the stride is 1, the output equals the input.
    - VALID: no padding; the window slides only over valid regions, and the output is smaller.
  - Targmax: supports only 3 (int32) or 9 (int64).
  - include_batch_in_index: currently supports only false, indicating that index computation does not include the batch dimension.
  - data_format: supports "NCHW" or "NHWC".
  - nan_prop: when true, NAN values participate in comparison and the maximum value can be NAN; when false, NAN values are ignored.

## Invocation Description

| Invocation method   | Sample code           | Description                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn mode interface | [test_aclnn_max_pool_with_argmax](./examples/arch35/test_geir_max_pool_with_argmax.cpp) | Calls the MaxPoolWithArgmax operator through the aclnn [aclnnMaxPool2dWithIndices](../max_pool3d_with_argmax_v2/docs/aclnnMaxPool2dWithIndices_en.md) method. |

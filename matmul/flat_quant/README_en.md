# FlatQuant

English | [简体中文](./README.md)

## Applicable Products

|Product             |  Support  |
|:-------------------------|:----------:|
|  <term>Ascend 950PR&950DT products</term>   |     √    |
|  <term>Atlas A3 products</term>  |     √    |
|  <term>Atlas A2 products</term>     |     √    |
|  <term>Atlas 200I/500 A2 inference products</term>    |     ×    |
|  <term>Atlas inference products</term>    |     ×    |
|  <term>Atlas training products</term>    |     ×    |

## Function

- Operator function: this fused operator performs two small matrix multiplications on the input matrix x at one time, that is, it right-multiplies the input matrix by kroneckerP2 and left-multiplies it by kroneckerP1, and then quantizes the matrix multiplication result. Currently, the pertoken and pergroup quantization modes are supported, corresponding to the INT4 and float4_e2m1 quantized output types respectively.

- Matrix multiplication formula:

  1. Right-multiply input x by kroneckerP2:

     $$
     x' = x @ kroneckerP2
     $$

  2. Left-multiply x' by kroneckerP1:

     $$
     x'' = kroneckerP1@x'
     $$

- Quantization computation method:

  pertoken quantization mode:

  1. Compute the maximum absolute value along dimension 0 of x'' and divide it by (7 / clipRatio) to compute the quantization factor for quantization into the INT4 format:

     $$
     quantScale = [max(abs(x''[0,:,:])),max(abs(x''[1,:,:])),...,max(abs(x''[M-1,:,:]))]/(7 / clipRatio)
     $$

  2. Compute the output out:

     $$
     out = x'' / quantScale
     $$

  pergroup quantization mode

  1. The shape of x'' after the matrix multiplication is [M,N1,N2]. Before computing mx_quantize in the pergroup quantization mode, reshape it into [M,N1*N2], denoted as x2.

  2. Group x2 along the second dimension by groupsize, which contains elements e0,e1...e31. Compute emax:

     $$
     emax = max(e0,e1....e31)
     $$

  3. Compute reduceMaxValue and sharedExp:

     $$
     reduceMaxValue = log2(reduceMax(x2),groupSize=32)
     $$

     $$
     sharedExp[M,N1*N2/32] = reduceMaxValue -emax
     $$

  4. Compute quantScale:

     $$
     quantScale[M,N1*N2/32] = 2 ^ {sharedExp[M,N1*N2/32]}
     $$

  5. Each groupsize shares one quantScale. Compute out:

     $$
     out = x2 / quantScale
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
      <td>Raw input data, corresponding to `x` in the formula. The shape is [M, N1, N2], where M does not exceed 262144, and N1 and N2 do not exceed 256. The data type of `out` is INT4, and N2 must be an even number.</td>
      <td>FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>kronecker_p1</td>
      <td>Input</td>
      <td>Input computation matrix 1, corresponding to `kroneckerP1` in the formula. The shape is [N1, N1], where N1 is consistent with the N1 dimension in `x`, and the data type is consistent with the data type of the input parameter `x`.</td>
      <td>FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>kronecker_p2</td>
      <td>Input</td>
      <td>Input computation matrix 2, corresponding to `kroneckerP2` in the formula. The shape is [N2, N2], where N2 is consistent with the N2 dimension in `x`, and the data type is consistent with the data type of the input parameter `x`.</td>
      <td>FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>group_list</td>
      <td>Optional input</td>
      <td>Denotes the quantization group size distribution of the input `x`. When group_list_type is 0 or 1, the shape is [G]; when group_list_type is 2, the shape is [G, 2]. G denotes the number of groups, and G must be less than or equal to 1024.</td>
      <td>INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>clip_ratio</td>
      <td>Optional attribute</td>
      <td><ul><li>Controls the clipping ratio of quantization. The input value range is (0, 1].</li><li>The default value is 1.</li></ul></td>
      <td>FLOAT32</td>
      <td>-</td>
    </tr>
    <tr>
      <td>dst_dtype</td>
      <td>Optional attribute</td>
      <td><ul><li>Controls the quantization mode. The value DT_FLOAT4_E2M1 indicates the pergroup quantization mode, and the value DT_INT32 indicates the pertoken quantization mode.</li><li>The default value is DT_INT32.</li></ul></td>
      <td>INT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>dst_type_max</td>
      <td>Optional attribute</td>
      <td><ul><li>Denotes the value of maxType, corresponding to Amax in the formula. The supported values are 0.0 and 6.0-12.0.</li><li>The value 0.0 indicates that this parameter is not used.</li><li>The value 6.0-12.0 indicates the maximum value of the target data type.</li><li>This value can be set only when the data type is FLOAT4_E2M1.</li><li>The default value is 0.0.</li></ul></td>
      <td>FLOAT32</td>
      <td>-</td>
    </tr>
    <tr>
      <td>group_list_type</td>
      <td>Optional attribute</td>
      <td><ul><li>Denotes the grouping mode of the groupList input. When group_list is not nullptr, the supported values are 0, 1, and 2.</li><li>The default value is 0.</li></ul></td>
      <td>INT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>out</td>
      <td>Output</td>
      <td>Output tensor, corresponding to `out` in the formula. When the data type is INT4, the shape is consistent with the input parameter `x`. When the data type is INT32, the shape is [M,N1,N2/8]. When the data type is FLOAT4_E2M1, the shape is [M,N1*N2].</td>
      <td>INT4, INT32, FLOAT4_E2M1</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>quant_scale</td>
      <td>Output</td>
      <td>Output quantization factor, corresponding to `quantScale` in the formula. When the output type is FLOAT, the shape is [M], where M is consistent with the M dimension in `x`. When the output type is FLOAT8_E8M0, the shape is [M,ceilDiv(N1*N2,64),2].</td>
      <td>FLOAT32, FLOAT8_E8M0</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## Constraints

- The parameters group_list and group_list_type must satisfy the following constraints:
  - <term>Ascend 950PR&950DT products</term>:
    - group_list supports only nullptr.
  - <term>Atlas A3 products</term>, <term>Atlas A2 products</term>:
    - When group_list is not nullptr:
      - group_list_type must take a value from [0, 1, 2].
      - When group_list_type is 0 or 1, the shape is [G]; when group_list_type is 2, the shape is [G, 2]. G denotes the number of groups, and G must be less than or equal to 1024.
      - The values of group_list must satisfy the following conditions; otherwise, whether the output meets expectations cannot be guaranteed:
        - When group_list_type is 0, group_list must be a non-negative, monotonically non-decreasing sequence, which represents the cumsum result (cumulative sum) of the size of each group after grouping; the last value must be less than or equal to the first dimension of the tensor in x.
        - When group_list_type is 1, group_list must be a non-negative sequence, which represents the size of each group after grouping; the sum of the values must be less than or equal to the first dimension of the tensor in x.
        - When group_list_type is 2, group_list must be a non-negative sequence, and the data layout is [[groupIdx0, groupSize0], [groupIdx1, groupSize1]...], where groupSize is the size of each group after grouping; the sum of the values in the second column must be less than or equal to the first dimension of the tensor in x.

## Invocation Description

| Invocation method   | Sample code           | Description                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn invocation  | [test_aclnn_flat_quant](examples/test_aclnn_flat_quant.cpp) | Invokes the FlatQuant operator through the [aclnnFlatQuant](docs/aclnnFlatQuant_en.md) API. |
| aclnn invocation  | [test_aclnn_flat_quant_v2](examples/test_aclnn_flat_quant_v2.cpp) | Invokes the FlatQuant operator through the [aclnnFlatQuantV2](docs/aclnnFlatQuantV2_en.md) API. |
| aclnn invocation  | [test_aclnn_flat_quant_v3](examples/test_aclnn_flat_quant_v3.cpp) | Invokes the FlatQuant operator through the [aclnnFlatQuantV3](docs/aclnnFlatQuantV3_en.md) API. |
| PyTorch API | - | Invokes the FlatQuant operator through the [flat_quant](./docs/torchapi_flat_quant_en.md) API. |
| Graph mode | -  | Invokes the FlatQuant operator by constructing the graph through the [operator IR](op_graph/flat_quant_proto.h).         |

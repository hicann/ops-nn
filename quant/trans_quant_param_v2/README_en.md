# TransQuantParamV2

English | [简体中文](./README.md)

## Applicable Products

| Product                                                         |  Support   |
| :----------------------------------------------------------- |:-------:|
| <term>Ascend 950PR&950DT products</term>                             |    √     |
| <term>Atlas A3 products</term>     |    √    |
| <term>Atlas A2 products</term> |    √    |
| <term>Atlas 200I/500 A2 inference products</term>                      |    ×    |
| <term>Atlas inference products</term>                             |    √    |
| <term>Atlas training products</term>                              |    ×    |
| <term>Kirin X90 processor series products</term> | √ |
| <term>Kirin 9030 processor series products</term> | √ |

## Function

- Operator function: Converts the data type of the quantization computation parameter scale from FLOAT32 to the UINT64 type required by the hardware.

- Formula:

  1. `out` is in 64-bit format and is initially 0.

  2. If `round_mode` is 1, `scale` is rounded bitwise to the highest 19 bits. If `round_mode` is 0, no processing is performed.

     $$
     scale = Round(scale)
     $$
  3. The highest 19 bits of `scale` are retained bitwise (that is, bits 13~31, with the lowest 13 bits cleared to zero), written into the lowest 32 bits of `out`, and bit 46 is set to 1.

     $$
     out = out\ |\ (scale\ \&\ 0xFFFFE000)\ |\ (1\ll46)
     $$

  4. Performs the subsequent computation according to the value of `offset`:

     - If `offset` does not exist, no subsequent computation is performed.
     - If `offset` exists:
       1. Processes the `offset` value into an int, with the range [-256, 255].

          $$
          offset = Max(Min(INT(Round(offset)),255),-256)
          $$

       2. Then retains 9 bits of `offset` bitwise and stores them in bits 37 to 45 of out.

          $$
          out = (out\ \&\ 0x4000FFFFFFFF)\ |\ ((offset\ \&\ 0x1FF)\ll37)
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
      <td>scale</td>
      <td>Input</td>
      <td>Quantization parameter, corresponding to `scale` in the formula. The shape is 1-dimensional (t,), where t = 1 or n, or 2-dimensional (1, n), where n is consistent with the shape n of the right matrix in the matmul computation. You must ensure that the scale data does not contain NaN or Inf.</td>
      <td>FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>offset</td>
      <td>Optional input</td>
      <td>Optional quantization parameter, corresponding to `offset` in the formula description. The shape is 1-dimensional (t,) or 2-dimensional (1, n), where t = 1 or n, and n is consistent with the shape n of the right matrix in the matmul computation. You must ensure that the offset data does not contain NaN or Inf.</td>
      <td>FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>round_mode</td>
      <td>Optional attribute</td>
      <td><ul><li>Round mode from FP32 to FP19 in quantization computation. Only the following values are supported: 0 (aclnn
      V3 compatible with V2), 1 (improves computation precision). It corresponds to `round_mode` in the formula description.</li><li>The default value is 0.</li></ul></td>
      <td>INT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>y</td>
      <td>Output</td>
      <td>Output result after quantization, corresponding to `out` in the formula. When the shape of the input scale is 1-dimensional, the shape of y is also 1-dimensional, and the shape size of this dimension is the maximum of the single-dimension shape sizes of scale and offset (if it is not nullptr). When the shape of the input scale is 2-dimensional, the dimensions and size of the shape of y are exactly the same as those of the input scale.</td>
      <td>UINT64</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## Constraints

- <term>Atlas inference products</term>, <term>Atlas A2 products</term>, <term>Atlas A3 products</term>, and <term>Ascend 950PR&950DT products</term>: The aclnn interface corresponding to this interface supports being used together with matmul operators (for example, [aclnnQuantMatmulV4](../../matmul/quant_batch_matmul_v3/docs/aclnnQuantMatmulV4_en.md)).
- <term>Atlas inference products</term>, <term>Atlas A2 products</term>, and <term>Atlas A3 products</term>: The aclnn interface corresponding to this interface does not support being used together with grouped matmul operators (for example, aclnnGroupedMatmulV4).
- The shapes of scale, offset, and y are described as follows:
  - When there is no offset, the y shape is the same as the scale shape.
    - If y is used as the input of a matmul operator, the supported shapes are 1-dimensional (1,) and (n,), or 2-dimensional (1, n), where n is consistent with the shape n of the right matrix (corresponding to parameter x2) in the matmul computation.
    - If y is used as the input of a grouped matmul operator, it is used only when the grouping mode is m-axis grouping (corresponding to groupType 0). The supported shapes are 1-dimensional (g,) or 2-dimensional (g, 1) and (g, n), where n is consistent with the shape n of the right matrix (corresponding to parameter x2) in the grouped matmul computation, and g is consistent with the number of groups (corresponding to the shape size of groupListOptional) in the grouped matmul computation.
  - When offset exists, it is used only as the input of matmul operators:
    - The supported shapes of offset, scale, and y are 1-dimensional (1,) and (n,), or 2-dimensional (1, n), where n is consistent with the shape n of the right matrix (corresponding to parameter x2) in the matmul computation.
    - When the shape of the input scale is 1-dimensional, the shape of y is also 1-dimensional, and the shape size is the maximum of the single-dimension shape sizes of scale and offset.
    - When the shape of the input scale is 2-dimensional, the dimensions and size of the shape of y are exactly the same as those of the input scale.

## Invocation Description

| Invocation method   | Sample code           | Description                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn invocation  | [test_aclnn_trans_quant_param_v2](examples/test_aclnn_trans_quant_param_v2.cpp) | Calls the TransQuantParamV2 operator through the [aclnnTransQuantParamV2](docs/aclnnTransQuantParamV2_en.md) interface. |
| aclnn invocation  | [test_aclnn_trans_quant_param_v3](examples/test_aclnn_trans_quant_param_v3.cpp) | Calls the TransQuantParamV2 operator through the [aclnnTransQuantParamV3](docs/aclnnTransQuantParamV3_en.md) interface. |
| Graph mode | [test_geir_trans_quant_param_v2](examples/test_geir_trans_quant_param_v2.cpp) | Calls the TransQuantParamV2 operator by building a graph through the [operator IR](op_graph/trans_quant_param_v2_proto.h).         |

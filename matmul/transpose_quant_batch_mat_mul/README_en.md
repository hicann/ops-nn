# TransposeQuantBatchMatMul

English | [简体中文](./README.md)

## Applicable Products

| Product                                                         |  Support   |
| :----------------------------------------------------------- |:-------:|
| <term>Ascend 950PR&950DT products</term>                             |    √     |
| <term>Atlas A3 products</term>    |    √    |
| <term>Atlas A2 products</term> |    √    |
|  <term>Atlas 200I/500 A2 inference products</term>    |     ×    |
|  <term>Atlas inference products</term>    |     ×    |
|  <term>Atlas training products</term>    |     ×    |

## Function

- Operator function: Performs the quantized matrix multiplication of tensor x1 and tensor x2, and supports the K-C, MX, and T-C [quantization modes](../../docs/en/context/quant_mode_introduction.md). Only 3-dimensional tensors are supported as inputs.

  Tensors support transposition, and the transpose sequence is specified by perm_x1, perm_x2, and perm_y. perm_x1 indicates the transpose sequence of tensor x1 and supports only [1, 0, 2]. perm_x2 indicates the transpose sequence of tensor x2 and supports [0, 1, 2] in K-C mode, and [0, 1, 2] and [0, 2, 1] in MX and T-C modes. perm_y indicates the transpose sequence of the matrix multiplication output matrix and supports only [1, 0, 2]. The dimension whose sequence value is 0 is the batch dimension, and the other two dimensions perform matrix multiplication.

  x1_scale and x2_scale are quantization scaling factors, whose shapes and data types vary with the product and quantization mode. bias is a reserved parameter and is not supported currently. For detailed constraints, see [Constraints](#constraints) or the [aclnnTransposeQuantBatchMatMul](docs/aclnnTransposeQuantBatchMatMul_en.md) API description.

- Formulas:

  After transposition, x1 is $[B, M, K]$ and x2 is $[B, K, N]$. The intermediate matrix multiplication result is $mm$, and the final output is $y$.

  1. K-C quantization mode:

     $$
     mm[b,m,n] = \sum_{k=0}^{K-1}(x1[b,m,k] \cdot s1[m,b]) \cdot (x2[b,k,n] \cdot s2[b,n])
     $$

     In the formula, $s1$ and $s2$ are the values of x1_scale and x2_scale broadcast according to the shapes described in Constraints.

  2. MX quantization mode (MXFP8 and MXFP4, group size 32):

     $$
     mm[b,m,n] = \sum_{k=0}^{K-1}(x1[b,m,k] \cdot s1[m,b,\lfloor k/64 \rfloor,(\lfloor k/32 \rfloor) \bmod 2]) \cdot (x2[b,k,n] \cdot s2[b,\lfloor k/64 \rfloor,n,(\lfloor k/32 \rfloor) \bmod 2])
     $$

     In the formula, $s1$ and $s2$ are FLOAT8_E8M0 scaling factors. The shape of $s1$ is $(M, B, \lceil K/64 \rceil, 2)$. When perm_x2 is [0, 1, 2], the shape of $s2$ is $(B, \lceil K/64 \rceil, N, 2)$ and the element is indexed by $[b, \lfloor k/64 \rfloor, n, (\lfloor k/32 \rfloor) \bmod 2]$; when perm_x2 is [0, 2, 1], the shape of $s2$ is $(B, N, \lceil K/64 \rceil, 2)$ and the element is indexed by $[b, n, \lfloor k/64 \rfloor, (\lfloor k/32 \rfloor) \bmod 2]$. The last dimension of size 2 indicates that each 64 K elements contain two 32-element group scaling factors.

  3. T-C quantization mode:

     $$
     mm[b,m,n] = \sum_{k=0}^{K-1}(x1[b,m,k] \cdot s1) \cdot (x2[b,k,n] \cdot s2[n])
     $$

     In the formula, $s1$ is x1_scale. When x1_scale is empty, $s1$ is 1.0; when its shape is (1,), $s1$ is the value of that element. $s2[n]$ is the per-channel scaling factor obtained by interpreting x2_scale in UINT64/INT64 bit mode, and the lower 32 bits of each element are in FLOAT32 bit mode.

  Finally, the intermediate matrix multiplication result is transposed according to perm_y to obtain the output:

  $$
  y = \text{permute}(mm, perm\_y)
  $$

- Example:
  Assume that the shape of x1 is [M, B, K], the shape of x2 is [B, K, N], x1_scale and x2_scale are not None, and batch_split_factor is 1; then the shape of the computed output y is [M, B, N].

## Parameters

<table class="tg" style="undefined;table-layout: fixed; width: 1034px"><colgroup>
<col style="width: 120px">
<col style="width: 123px">
<col style="width: 352px">
<col style="width: 320px">
<col style="width: 125px">
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
    <td>x1</td>
    <td>Input</td>
    <td>Left matrix in the matrix multiplication, corresponding to x1 in the formula.</td>
    <td>FLOAT8_E5M2, FLOAT8_E4M3FN, FLOAT4_E2M1, HIFLOAT8, INT8</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>x2</td>
    <td>Input</td>
    <td>Right matrix in the matrix multiplication, corresponding to x2 in the formula. x2 supports the ND and FRACTAL_NZ formats. The support scope of the FRACTAL_NZ format varies by product. For details, see Constraints.</td>
    <td>FLOAT8_E5M2, FLOAT8_E4M3FN, FLOAT4_E2M1, HIFLOAT8, INT8</td>
    <td>ND, FRACTAL_NZ</td>
  </tr>
  <tr>
    <td>bias</td>
    <td>Optional input</td>
    <td>Bias accumulated after the matrix multiplication. Reserved parameter, currently not supported.</td>
    <td>FLOAT16, BFLOAT16, FLOAT32</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>x1_scale</td>
    <td>Optional input</td>
    <td>Quantization scaling factor of x1, corresponding to x1_scale in the formula. Its shape varies by product and quantization mode. For details, see Constraints.</td>
    <td>FLOAT32, FLOAT8_E8M0, UINT64, INT64</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>x2_scale</td>
    <td>Optional input</td>
    <td>Quantization scaling factor of x2, corresponding to x2_scale in the formula. Its shape varies by product and quantization mode. For details, see Constraints.</td>
    <td>FLOAT32, FLOAT8_E8M0, UINT64, INT64</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>dtype</td>
    <td>Attribute</td>
    <td>Specifies the data type of the output matrix. This parameter is mandatory. The value 1 indicates FLOAT16, 27 indicates BFLOAT16, and 34 indicates HIFLOAT8 (supported only in T-C quantization mode).</td>
    <td>INT32</td>
    <td>-</td>
  </tr>
  <tr>
    <td>group_size</td>
    <td>Optional attribute</td>
    <td>Specifies the quantization group size. The default value is 0. It consists of group_size_m, group_size_n, and group_size_k. For details, see Constraints.</td>
    <td>INT64</td>
    <td>-</td>
  </tr>
  <tr>
    <td>perm_x1</td>
    <td>Optional attribute</td>
    <td>Transpose sequence of the first matrix in the matrix multiplication. Only [1, 0, 2] is supported, and the default value is [1, 0, 2].</td>
    <td>INT64</td>
    <td>-</td>
  </tr>
  <tr>
    <td>perm_x2</td>
    <td>Optional attribute</td>
    <td>Transpose sequence of the second matrix in the matrix multiplication. [0, 1, 2] and [0, 2, 1] are supported (K-C quantization mode supports only [0, 1, 2]), and the default value is [0, 1, 2].</td>
    <td>INT64</td>
    <td>-</td>
  </tr>
  <tr>
    <td>perm_y</td>
    <td>Optional attribute</td>
    <td>Transpose sequence of the output matrix of the matrix multiplication. Only [1, 0, 2] is supported, and the default value is [1, 0, 2].</td>
    <td>INT64</td>
    <td>-</td>
  </tr>
  <tr>
    <td>batch_split_factor</td>
    <td>Optional attribute</td>
    <td>Specifies the split size of the B dimension in the matrix multiplication output matrix. Currently only the value 1 is supported, and the default value is 1.</td>
    <td>INT32</td>
    <td>-</td>
  </tr>
  <tr>
    <td>y</td>
    <td>Output</td>
    <td>Computation result of the matrix multiplication, corresponding to y in the formula.</td>
    <td>FLOAT16, BFLOAT16, HIFLOAT8</td>
    <td>ND</td>
  </tr>
</tbody></table>

- <term>Ascend 950PR&950DT products</term>: x1 and y support only the ND format, and only the input x2 supports the FRACTAL_NZ format. When x2 is in NZ format, the operator must be invoked through [aclnnTransposeQuantBatchMatMulWeightNz](docs/aclnnTransposeQuantBatchMatMulWeightNz_en.md), and only the MXFP8, MXFP4, and HIFP8 quantization modes are supported.

## Constraints

- <term>Atlas A2 products</term>, <term>Atlas A3 products</term>: Only INT8 W8A8 K-C is supported. x1 and x2 are INT8, x1_scale and x2_scale are FLOAT32, and y is BF16. x1 is [M, B, K], x2 is [B, K, N], x1_scale is [M, B], x2_scale is [B, N], and y is [M, B, N]. perm_x1 and perm_y are [1, 0, 2], and perm_x2 is [0, 1, 2]. K is a multiple of 32, N is a multiple of 16, and batch_split_factor is 1. On A2, B*K must be less than 65536. x2 supports ND and FRACTAL_NZ.
- <term>Ascend 950PR&950DT products</term>:
    - perm_x1 and perm_y support only [1, 0, 2].
    - In the K-C quantization scenario, perm_x2 supports the input [0, 1, 2]; in the MX and T-C quantization scenarios, perm_x2 supports the input [0, 1, 2] or [0, 2, 1].
    - x1 and y support only the ND format; x2 supports the ND and FRACTAL_NZ formats. When x2 is in FRACTAL_NZ format, its storage shape is 5-dimensional (the logical shape is still 3-dimensional), only the MXFP8, MXFP4, and HIFP8 quantization modes are supported, and the operator must be invoked through [aclnnTransposeQuantBatchMatMulWeightNz](docs/aclnnTransposeQuantBatchMatMulWeightNz_en.md).
    - K-C [quantization mode](../../docs/en/context/quant_mode_introduction.md): K supports only 512, and N supports only 128. x1_scale and x2_scale are 1-dimensional, with x1_scale being (M,) and x2_scale being (N,). group_size supports only the configuration value 0, and other values do not take effect. The x1/x2 inputs support the two types FLOAT8_E5M2 and FLOAT8_E4M3FN, and x1_scale/x2_scale support only the FLOAT32 type.
    - MX [quantization mode](../../docs/en/context/quant_mode_introduction.md): The two data types MXFP8 and MXFP4 are supported. K supports only multiples of 64. x1_scale and x2_scale are 4-dimensional, with x1_scale being (M, B, K/64, 2) and x2_scale being (B, K/64, N, 2) or (B, N, K/64, 2). group_size_m and group_size_n of group_size support only 0 or 1, and group_size_k supports only 32. The x1/x2 inputs support the data types FLOAT8_E4M3FN and FLOAT4_E2M1, and x1_scale/x2_scale support only the FLOAT8_E8M0 type.
    - T-C [quantization mode](../../docs/en/context/quant_mode_introduction.md): Only static quantization is supported. x1_scale supports being empty or (1,). When it is not empty, only the UINT64/INT64 types are supported; x2_scale is 1-dimensional and is (N,), and it must be preprocessed through [trans_quant_param](../../quant/trans_quant_param_v2/docs/aclnnTransQuantParamV2_en.md) to be converted to the UINT64/INT64 types; x1/x2 support only the HIFLOAT8 type, and the group_size configuration does not take effect.
    - Constraints related to group_size:
        - It takes effect only in the MX quantization scenario.
        - The passed-in group_size is internally decomposed into group_size_m, group_size_n, and group_size_k according to the following formula. When group_size_m and group_size_n are 0, the API infers them based on the shape of scale, and group_size_k supports only 32. Principle: assume that group_size_m=0, which means that the quantization group value in the M direction is inferred by the API. The inference formula is group_size_m = M / scale_m (M must be divisible by scale_m), where M is the same as the M in the shape of x1, and scale_m is the same as the M in the shape of x1_scale.

        $$
        group\_size = group\_size\_k\ |\ group\_size\_n << 16\ |\ group\_size\_m << 32
        $$
    - Empty tensors are not supported.

## Invocation Description

| Invocation method   | Sample code           | Description                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn API  | [test_aclnn_transpose_quant_batch_mat_mul (A2/A3 INT8 W8A8)](examples/test_aclnn_transpose_quant_batch_mat_mul.cpp)<br>[test_aclnn_transpose_quant_batch_mat_mul (Ascend 950 FP8)](examples/arch35/test_aclnn_transpose_quant_batch_mat_mul.cpp) | Invoke the TransposeQuantBatchMatMul operator through<br>- [aclnnTransposeQuantBatchMatMul](docs/aclnnTransposeQuantBatchMatMul_en.md)<br>- [aclnnTransposeQuantBatchMatMulWeightNz](docs/aclnnTransposeQuantBatchMatMulWeightNz_en.md)<br>and other APIs.|
| torch API  | - | Invoke the TransposeQuantBatchMatMul operator through the [torchapi_transpose_quant_batch_mat_mul](docs/torchapi_transpose_quant_batch_mat_mul_en.md) API, supporting MX (MXFP8/MXFP4), FP8 (K-C), HIFP8, and INT8 W8A8 quantization modes.|

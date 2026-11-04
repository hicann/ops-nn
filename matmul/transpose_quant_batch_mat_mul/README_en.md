# TransposeQuantBatchMatMul

English | [简体中文](./README.md)

## Applicable Products

| Product                                                         |  Support   |
| :----------------------------------------------------------- |:-------:|
| <term>Ascend 950PR&950DT products</term>                             |    √     |
| <term>Atlas A3 products</term>    |    ×    |
| <term>Atlas A2 products</term> |    ×    |
|  <term>Atlas 200I/500 A2 inference products</term>    |     ×    |
|  <term>Atlas inference products</term>    |     ×    |
|  <term>Atlas training products</term>    |     ×    |

## Function

- Operator function: Performs the quantized matrix multiplication of tensor x1 and tensor x2, and supports the K-C, MX, and T-C [quantization modes](../../docs/en/context/quant_mode_introduction.md). Only 3-dimensional tensors are supported as inputs. Tensors support transposition, and the transpose sequence changes according to the passed-in sequence. permX1 represents the transpose sequence of tensor x1 and supports [1,0,2]; permX2 represents the transpose sequence of tensor x2. The K-C mode supports [0,1,2], and the MX and T-C modes support [0,1,2] and [0,2,1]; permY represents the transpose sequence of the matrix multiplication output matrix. Currently only [1,0,2] is supported. The dimension whose sequence value is 0 is the batch dimension, and the other two dimensions perform matrix multiplication. x1Scale and x2Scale represent the quantization coefficients of the output matrix; bias is a reserved parameter and is not supported currently. For detailed constraints, refer to the Constraints section or the invocation description document of [aclnnTransposeQuantBatchMatMul](docs/aclnnTransposeQuantBatchMatMul_en.md).

- Example:
  Assume that the shape of x1 is [M, B, K], the shape of x2 is [B, K, N], x1Scale and x2Scale are not None, and batchSplitFactor is 1; then the shape of the computed output out is [M, B, N].

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
    <td>Left matrix in the matrix multiplication.</td>
    <td>FLOAT8_E5M2, FLOAT8_E4M3FN, FLOAT4_E2M1, HIFLOAT8</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>x2</td>
    <td>Input</td>
    <td>Right matrix in the matrix multiplication.</td>
    <td>FLOAT8_E5M2, FLOAT8_E4M3FN, FLOAT4_E2M1, HIFLOAT8</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>bias</td>
    <td>Input</td>
    <td>Bias accumulated after the matrix multiplication. Reserved parameter.</td>
    <td>FLOAT32, FLOAT16, BF16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>x1Scale</td>
    <td>Input</td>
    <td>Scaling factor of the quantization parameters.</td>
    <td>FLOAT32, FLOAT8_E8M0, UINT64, INT64</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>x2Scale</td>
    <td>Input</td>
    <td>Scaling factor of the quantization parameters.</td>
    <td>FLOAT32, FLOAT8_E8M0, UINT64, INT64</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>dtype</td>
    <td>Input</td>
    <td>Specifies the data type of the output matrix.</td>
    <td>INT32</td>
    <td>-</td>
  </tr>
  <tr>
    <td>groupSize</td>
    <td>Input</td>
    <td>Specifies the quantization group size.</td>
    <td>INT64</td>
    <td>-</td>
  </tr>
  <tr>
    <td>permX1</td>
    <td>Input</td>
    <td>Transpose sequence of the first matrix in the matrix multiplication.</td>
    <td>INT64</td>
    <td>-</td>
  </tr>
  <tr>
    <td>permX2</td>
    <td>Input</td>
    <td>Transpose sequence of the second matrix in the matrix multiplication.</td>
    <td>INT64</td>
    <td>-</td>
  </tr>
  <tr>
    <td>permY</td>
    <td>Input</td>
    <td>Transpose sequence of the output matrix of the matrix multiplication.</td>
    <td>INT64</td>
    <td>-</td>
  </tr>
  <tr>
    <td>batchSplitFactor</td>
    <td>Input</td>
    <td>Specifies the split size of the B dimension in the matrix multiplication output matrix. Currently only the value 1 is supported.</td>
    <td>INT32</td>
    <td>-</td>
  </tr>
  <tr>
    <td>y</td>
    <td>Output</td>
    <td>Computation result of the matrix multiplication.</td>
    <td>FLOAT16, BFLOAT16, HIFLOAT8</td>
    <td>ND</td>
  </tr>
</tbody></table>

- Ascend 950PR&950DT products: Only the input x2 supports the FRACTAL_NZ format (when x2 is in NZ format, the computation goes through [aclnnTransposeQuantBatchMatMulWeightNz](docs/aclnnTransposeQuantBatchMatMulWeightNz_en.md), and the MXFP8 and HIFP8 quantization modes are supported).

## Constraints

- <term>Ascend 950PR&950DT products</term>:
    - permX1 and permY support [1, 0, 2].
    - In the K-C quantization scenario, permX2 supports the input [0, 1, 2]; in the MX and T-C quantization scenarios, permX2 supports the input [0, 1, 2] or [0, 2, 1].
    - K-C [quantization mode](../../docs/en/context/quant_mode_introduction.md): K supports only 512, and N supports only 128. x1Scale and x2Scale are 1-dimensional, with x1Scale being (M,) and x2Scale being (N,). group_size supports only the configuration value 0, and other values do not take effect. The x1/x2 inputs support the two types FLOAT8_E5M2 and FLOAT8_E4M3FN, and x1Scale/x2Scale support only the FLOAT32 type.
    - MX [quantization mode](../../docs/en/context/quant_mode_introduction.md): The two data types MXFP8 and MXFP4 are supported. K supports only multiples of 64. x1Scale and x2Scale are 4-dimensional, with x1Scale being (M, B, K/64, 2) and x2Scale being (B, K/64, N, 2) or (B, N, K/64, 2). groupSizeM and groupSizeN of group_size support only 0 or 1, and groupSizeK supports only 32. The x1/x2 inputs support the data types FLOAT8_E4M3FN and FLOAT4_E2M1, and x1Scale/x2Scale support only the FLOAT8_E8M0 type.
    - T-C [quantization mode](../../docs/en/context/quant_mode_introduction.md): Only static quantization is supported. x1Scale supports being empty or (1,). When it is not empty, only the UINT64/INT64 types are supported; x2Scale is 1-dimensional and is (N,), and it must be preprocessed through [trans_quant_param](../../quant/trans_quant_param_v2/docs/aclnnTransQuantParamV2_en.md) to be converted to the UINT64/INT64 types; x1/x2 support only the HIFLOAT8 type, and the group_size configuration does not take effect.
    - Constraints related to groupSize:
        - It takes effect only in the MX quantization scenario.
        - The passed-in groupSize is internally decomposed into groupSizeM, groupSizeN, and groupSizeK according to the following formula. When groupSizeM and groupSizeN are 0, the API infers them based on the shape of scale, and groupSizeK supports only 32. Principle: assume that groupSizeM=0, which means that the quantization group value in the M direction is inferred by the API. The inference formula is groupSizeM = M / scaleM (M must be divisible by scaleM), where M is the same as the M in the shape of x1, and scaleM is the same as the M in the shape of x1Scale.

        $$
        groupSize = groupSizeK | groupSizeN << 16 | groupSizeM << 32
        $$
    - Empty tensors are not supported.

## Invocation Description

| Invocation method   | Sample code           | Description                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn API  | [test_aclnn_quant_batch_mat_mul](examples/arch35/test_aclnn_transpose_quant_batch_mat_mul.cpp) | Invoke the TransposeQuantBatchMatMul operator through<br>- [aclnnTransposeQuantBatchMatMul](docs/aclnnTransposeQuantBatchMatMul_en.md)<br>- [aclnnTransposeQuantBatchMatMulWeightNz](docs/aclnnTransposeQuantBatchMatMulWeightNz_en.md)<br>and other APIs.|
| torch API  | [torchapi_transpose_quant_batch_mat_mul](docs/torchapi_transpose_quant_batch_mat_mul_en.md) | Invoke the TransposeQuantBatchMatMul operator through `cann_ops_nn.transpose_quant_batch_mat_mul`, supporting the MX (MXFP8/MXFP4), FP8 (K-C), and HIFP8 quantization modes.|

# TransQuantParamV2

English | [简体中文](./README.md)

## Applicable Products

| Product | Support |
| --- | :---: |
| <term>Ascend 950PR&950DT products</term> | √ |
| <term>Atlas A3 products</term> | √ |
| <term>Atlas A2 products</term> | √ |
| <term>Atlas 200I/500 A2 inference products</term> | × |
| <term>Atlas inference products</term> | √ |
| <term>Atlas training products</term> | × |
| <term>Kirin X90 processor products</term> | √ |
| <term>Kirin 9030 processor products</term> | √ |

## Function

- Function: Encodes FLOAT32 quantization scale factors and optional offsets into the UINT64 quantization parameters required by the hardware for operators such as matrix multiplication. The output contains the floating-point bit pattern of scale, the integer encoding of offset, and a flag bit.

- Interface versions: Both [aclnnTransQuantParamV2](docs/aclnnTransQuantParamV2_en.md) and [aclnnTransQuantParamV3](docs/aclnnTransQuantParamV3_en.md) call this operator. V2 always uses round_mode=0. V3 adds the roundMode parameter to select round_mode=0 or 1.

- Encoding flow: The following steps describe the calculation of a single output element. Bit positions start at 0 for the least significant bit.

  1. Process the bit pattern of scale. When round_mode=0, directly preserve the highest 19 bits of FLOAT32. When round_mode=1, first round to 19-bit floating-point precision using Round to Nearest, ties to Even (RNE), and then encode the result. These 19 bits comprise 1 sign bit, 8 exponent bits, and 10 mantissa bits.

     $$
     b=\begin{cases}
     \operatorname{Bits}_{32}(scale), & \mathrm{round\_mode}=0 \\
     \operatorname{Bits}_{32}(\operatorname{RoundFP19}(scale)), & \mathrm{round\_mode}=1
     \end{cases}
     $$

     Here, $\operatorname{Bits}_{32}$ reinterprets FLOAT32 bitwise as UINT32. $\operatorname{RoundFP19}$ denotes rounding to the 19-bit precision described above, with the result still represented in the FLOAT32 bit layout. RNE rounds to the nearest value and resolves exact ties to the value whose least significant retained mantissa bit is 0.

  2. Write bits 13–31 of scale to the corresponding positions in y, clear the lowest 13 bits, and set bit 46 of y to 1 to obtain the intermediate encoding $y_0$. All shifts and bitwise OR operations below use UINT64.

     $$
     y_0=\left(\operatorname{UINT64}(b)\mathbin{\&}\mathtt{0xFFFFE000}\right)
     \mathbin{|}(1\ll46)
     $$

  3. If offset is omitted, $y=y_0$. Otherwise, convert offset to INT32 using RNE, clamp the result to $[-256,255]$, and write its lowest 9 bits to bits 37–45 of y.

     $$
     q=\min\left(255,\max\left(-256,\operatorname{RNE}_{\mathrm{INT32}}(offset)\right)\right)
     $$

     $$
     y=y_0\mathbin{|}\left[\left(\operatorname{UINT64}(q)\mathbin{\&}\mathtt{0x1FF}\right)\ll37\right]
     $$

     Here, $q$ is the processed offset. For negative values, retain the lowest 9 bits of the two's complement representation. round_mode controls only scale processing; offset always uses RNE.

## Parameters

| Parameter | Input/Output/Attribute | Description | Data Type | Data Format |
| --- | --- | --- | --- | --- |
| scale | Input | Quantization scale factor, corresponding to scale in the formula. Supports one- or two-dimensional tensors. For shapes and usage scenarios, see [Constraints](#constraints). | FLOAT32 | ND |
| offset | Optional input | Quantization offset, corresponding to offset in the formula. Omitted by default, meaning no offset is encoded. If provided, supports one- or two-dimensional tensors. | FLOAT32 | ND |
| y | Output | Encoded quantization parameters, corresponding to y in the formula. For the output shape, see [Constraints](#constraints). | UINT64 | ND |
| round_mode | Optional attribute | <ul><li>Encoding mode for scale: 0 discards the lowest 13 bits; 1 rounds to 19-bit floating-point precision using RNE before encoding.</li><li>Defaults to 0. Only 0 and 1 are supported.</li></ul> | INT64 | - |

## Constraints

- scale and any provided offset must be nonempty tensors containing no NaN or Inf.
- When round_mode=1, subnormal scale values are treated as zeros with their signs preserved.
- When used with matrix multiplication operators, scale and any provided offset support shapes `(1,)`, `(n,)`, or `(1, n)`, where n is the number of columns of the right matrix used in the multiplication. This operator can be used with operators such as [QuantBatchMatmulV3](../../matmul/quant_batch_matmul_v3/README_en.md).
- Output shape inference:
  - If offset is omitted, the shape of y matches that of scale.
  - If offset is provided, y uses the shape of scale, except when offset is two-dimensional and scale is one-dimensional, in which case y uses the shape of offset.
  - <term>Ascend 950PR&950DT products</term>: when both scale and offset are one-dimensional, the length of y is the maximum of their lengths.
- <term>Ascend 950PR&950DT products</term>: when used with grouped matrix multiplication, only grouping along the M dimension is supported, and offset must be omitted. scale and y support shapes `(g,)`, `(g, 1)`, or `(g, n)`, where g is the group count and n is the number of columns of the right matrix used in the multiplication.
- <term>Atlas A3 products</term>, <term>Atlas A2 products</term>, and <term>Atlas inference products</term>: use with grouped matrix multiplication is unsupported.
- For aclnn input combinations and layout requirements, see [aclnnTransQuantParamV2](docs/aclnnTransQuantParamV2_en.md#constraints) and [aclnnTransQuantParamV3](docs/aclnnTransQuantParamV3_en.md#constraints).

## Invocation Description

| Invocation method | Sample code | Description |
| --- | --- | --- |
| aclnn API | [test_aclnn_trans_quant_param_v2](examples/test_aclnn_trans_quant_param_v2.cpp) | Calls [aclnnTransQuantParamV2](docs/aclnnTransQuantParamV2_en.md) with round_mode=0. |
| aclnn API | [test_aclnn_trans_quant_param_v3](examples/test_aclnn_trans_quant_param_v3.cpp) | Calls [aclnnTransQuantParamV3](docs/aclnnTransQuantParamV3_en.md). The example uses roundMode=1. |
| GE graph mode | [test_geir_trans_quant_param_v2](examples/test_geir_trans_quant_param_v2.cpp) | Builds and invokes a graph using the [TransQuantParamV2 operator IR](op_graph/trans_quant_param_v2_proto.h). The example uses round_mode=0. |

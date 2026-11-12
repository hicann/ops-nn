# transpose_quant_batch_mat_mul

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-22T19:12:58.207Z -->

## Applicable Products

<!-- npu="950" id1 -->
- <term>Ascend 950PR&950DT products</term>: Supported
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3 products</term>: Supported
<!-- end id2 -->
<!-- npu="910b" id3 -->
- <term>Atlas A2 products</term>: Supported
<!-- end id3 -->
<!-- npu="310b" id4 -->
- <term>Atlas 200I/500 A2 inference products</term>: Not supported
<!-- end id4 -->
<!-- npu="310p" id5 -->
- <term>Atlas inference products</term>: Not supported
<!-- end id5 -->
<!-- npu="910" id6 -->
- <term>Atlas training products</term>: Not supported
<!-- end id6 -->

## Function

- Description: Performs [MX quantization](../../../docs/en/context/quant_mode_introduction.md) matrix multiplication on tensor x1 and tensor x2. The bottom layer encapsulates aclnnTransposeQuantBatchMatMul; when x2 is in FRACTAL_NZ format, it encapsulates aclnnTransposeQuantBatchMatMulWeightNz. On Ascend 950 this page describes the MX quantization mode (MXFP8 and MXFP4). On Atlas A2 and Atlas A3, only INT8 W8A8 K-C is supported: x1 and x2 are torch.int8, x1_scale is torch.float32 with shape (M, B), x2_scale is torch.float32 with shape (B, N), and y is torch.bfloat16. K is a multiple of 32, N is a multiple of 16, perm_x2 is only [0, 1, 2], batch_split_factor is 1, and dtype is only 27. On A2, B*K must be less than 65536. For K-C and T-C quantization modes on Ascend 950, use the aclnn API.
- Formula (taking each slice along the batch dimension as an example):

  $$
  y[m, n] = \sum_{j=0}^{K/32-1} \left(\left(\sum_{k=0}^{31} x1[m, j \times 32 + k] \cdot x2[j \times 32 + k, n]\right) \cdot x1Scale[m, j] \cdot x2Scale[j, n]\right)
  $$

  Where K is the K-axis length of the matrix multiplication, x1Scale and x2Scale are MX quantization scale factors encoded in FLOAT8_E8M0, and the intermediate results of the matrix multiplication are scaled and accumulated in groups of 32 elements along the K axis.

- Example: Assume that the shape of x1 is (M, B, K), the shape of x2 is (B, K, N), and the shape of output y is (M, B, N).

## Function Prototype

```python
cann_ops_nn.transpose_quant_batch_mat_mul(
    x1,
    x2,
    *,
    dtype,
    bias=None,
    x1_scale=None,
    x2_scale=None,
    group_sizes=None,
    perm_x1=None,
    perm_x2=None,
    perm_y=None,
    batch_split_factor=None,
    x1_dtype=None,
    x2_dtype=None,
    x1_scale_dtype=None,
    x2_scale_dtype=None,
) -> Tensor
```

## Mandatory

| Name | Type | Type | Description | Data Type | Dimension (Shape) |
| --- | --- | --- | --- | --- | --- |
| x1 | Tensor | Required | Left matrix in the matrix multiplication, with shape (M, B, K). In MXFP4 use cases, the last dimension of the tensor is the physical length K/2 after FP4 packing. | torch.float8_e4m3fn; in MXFP4 use cases, the actual storage type (such as torch.uint8), specified as FLOAT4_E2M1 through x1_dtype | 3 dimensions, (M, B, K) |
| x2 | Tensor | Required | Right matrix in the matrix multiplication, with the same data type and K-axis length as x1. The shape is (B, K, N) when perm_x2 is [0, 1, 2]; (B, N, K) when perm_x2 is [0, 2, 1]. In MXFP4 use cases, the last dimension of the tensor is the physical length (N/2 or K/2) after FP4 packing. | Same as x1 | 3 dimensions |
| dtype | int | Required | Data type enumeration value of output y: 1 indicates torch.float16, and 27 indicates torch.bfloat16. | int64 | - |
| bias | Tensor | Optional | Bias added after the matrix multiplication. Reserved parameter, currently not supported. You must pass in None. | - | - |
| x1_scale | Tensor | Required | MX quantization scale factors of x1. | torch.float8_e8m0fnu | 4 dimensions, (M, B, K/64, 2) |
| x2_scale | Tensor | Required | MX quantization scale factors of x2. | torch.float8_e8m0fnu | 4 dimensions; (B, K/64, N, 2) when perm_x2 is [0, 1, 2], and (B, N, K/64, 2) when perm_x2 is [0, 2, 1] |
| group_sizes | List[int] | Required | Quantization group sizes [groupSizeM, groupSizeN, groupSizeK], each element with a value range of [0, 65535]. In MX quantization use cases, groupSizeM and groupSizeN only support 0 or 1 (when the value is 0, the API infers it according to the shape of scale), and groupSizeK only supports 32. | int64 | - |
| perm_x1 | List[int] | Optional | Transpose order of x1. Only [1, 0, 2] is supported. Default value: [1, 0, 2]. | int64 | - |
| perm_x2 | List[int] | Optional | Transpose order of x2. [0, 1, 2] and [0, 2, 1] are supported. Default value: [0, 1, 2]. | int64 | - |
| perm_y | List[int] | Optional | Transpose order of the output matrix. Only [1, 0, 2] is supported. Default value: [1, 0, 2]. | int64 | - |
| batch_split_factor | int | Optional | Split size of dimension B of the output matrix. Currently only the value 1 is supported. Default value: 1. | int64 | - |
| x1_dtype | int | Optional | Data type enumeration value of x1. When not passed, it is auto-inferred according to the data type of x1. In MXFP4 use cases, this parameter must be passed in and set to torch_npu.float4_e2m1fn_x2. | int64 | - |
| x2_dtype | int | Optional | Data type enumeration value of x2. When not passed, it is auto-inferred according to the data type of x2. In MXFP4 use cases, this parameter must be passed in and set to torch_npu.float4_e2m1fn_x2. | int64 | - |
| x1_scale_dtype | int | Optional | Data type enumeration value of x1_scale. When not passed, it is auto-inferred according to the data type of x1_scale. | int64 | - |
| x2_scale_dtype | int | Optional | Data type enumeration value of x2_scale. When not passed, it is auto-inferred according to the data type of x2_scale. | int64 | - |

## Return Values

| Output | Output Type | Description | Data Type | Dimension (Shape) |
| --- | --- | --- | --- | --- |
| y | Tensor | Output of the MX quantization matrix multiplication. | torch.float16 when dtype is 1, torch.bfloat16 when dtype is 27 | (M, B, N) |

## Constraints

- This API currently supports calls in single-operator mode.
- Only the MX quantization mode is supported on Ascend 950: x1, x2, x1_scale, and x2_scale must be NPU tensors, and x1_scale and x2_scale are required inputs.
- <term>Atlas A2 products</term>, <term>Atlas A3 products</term>: Only INT8 W8A8 K-C is supported. x1 and x2 are torch.int8 with shapes (M, B, K) and (B, K, N). x1_scale and x2_scale are torch.float32 with shapes (M, B) and (B, N). y is torch.bfloat16 and dtype is only 27. K is a multiple of 32, N is a multiple of 16, perm_x2 is only [0, 1, 2], and batch_split_factor is 1. On A2, B*K must be less than 65536. bias must be None. When x2 is FRACTAL_NZ, the WeightNz API is used.
- x1 and x2 must have the same data type: torch.float8_e4m3fn in MXFP8 use cases and torch.uint8 in MXFP4 use cases.
- x1_scale and x2_scale must have the data type torch.float8_e8m0fnu.
- Only 3D tensors are supported: the shape of x1 is (M, B, K), the shape of x2 is (B, K, N) or (B, N, K), the batch and K dimensions of x1 and x2 must be the same, and batch dimension broadcasting is not supported.
- In MX quantization use cases, K only supports multiples of 64.
- The shape of x1_scale must be (M, B, K/64, 2); the shape of x2_scale must be (B, K/64, N, 2) when perm_x2 is [0, 1, 2] and (B, N, K/64, 2) when perm_x2 is [0, 2, 1]; the last dimension must be 2.
- group_sizes must be explicitly passed in, groupSizeM and groupSizeN must be 0 or 1, and groupSizeK must be 32.
- perm_x1 only supports [1, 0, 2]; perm_x2 supports [0, 1, 2] and [0, 2, 1]; perm_y only supports [1, 0, 2].
- batch_split_factor currently supports only the value 1.
- bias is a reserved parameter and is currently not supported.
- Empty tensors are not supported.
- In MXFP4 mode, data is stored with two FLOAT4_E2M1 values packed together (the last dimension of the tensor is the physical length, which is half of the logical length): the last dimension of x1 is K/2; the last dimension of x2 is N/2 when perm_x2 is [0, 1, 2] and K/2 when perm_x2 is [0, 2, 1]. In this case, the FP4 type cannot be automatically inferred from the actual storage type of the tensor (such as torch.uint8). Specify torch_npu.float4_e2m1fn_x2 through x1_dtype and x2_dtype.
- Only x2 supports the FRACTAL_NZ format (only in MX quantization mode).

## Deterministic Computation

Defaults to a deterministic computation.

## Example

- Single-operator mode call (eager)

  - MXFP8 use case example:

    ```python
    import torch
    import torch_npu
    import cann_ops_nn

    M, B, K, N = 64, 16, 128, 256
    x1 = torch.randn(M, B, K).to(torch.float8_e4m3fn).npu()
    x2 = torch.randn(B, K, N).to(torch.float8_e4m3fn).npu()
    x1_scale = torch.ones(M, B, K // 64, 2, dtype=torch.float8_e8m0fnu).npu()
    x2_scale = torch.ones(B, K // 64, N, 2, dtype=torch.float8_e8m0fnu).npu()

    y = cann_ops_nn.transpose_quant_batch_mat_mul(
        x1,
        x2,
        dtype=27,
        x1_scale=x1_scale,
        x2_scale=x2_scale,
        group_sizes=[1, 1, 32],
    )
    print(y.shape, y.dtype)
    ```

  - MXFP4 use case example:

    ```python
    import torch
    import torch_npu
    import cann_ops_nn

    M, B, K, N = 64, 16, 128, 256
    # FP4 is stored in a packed format of two FLOAT4_E2M1 values, and the last dimension of the tensor is the physical length (half of the logical length).
    x1 = torch.randint(0, 256, (M, B, K // 2), dtype=torch.uint8).npu()
    x2 = torch.randint(0, 256, (B, K, N // 2), dtype=torch.uint8).npu()
    x1_scale = torch.ones(M, B, K // 64, 2, dtype=torch.float8_e8m0fnu).npu()
    x2_scale = torch.ones(B, K // 64, N, 2, dtype=torch.float8_e8m0fnu).npu()

    y = cann_ops_nn.transpose_quant_batch_mat_mul(
        x1,
        x2,
        dtype=27,
        x1_scale=x1_scale,
        x2_scale=x2_scale,
        group_sizes=[1, 1, 32],
        x1_dtype=torch_npu.float4_e2m1fn_x2,
        x2_dtype=torch_npu.float4_e2m1fn_x2,
    )
    print(y.shape, y.dtype)
    ```

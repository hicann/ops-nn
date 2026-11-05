# transpose_batch_mat_mul

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

- Description: Performs matrix multiplication on 3D tensors x1 and x2. The bottom layer encapsulates `aclnnTransposeBatchMatMul`; when x2 is in FRACTAL_NZ format, it encapsulates `aclnnTransposeBatchMatMulWeightNz`. x1/x2 are permuted according to `perm_x1`/`perm_x2`, and the output is permuted by `perm_y`.
- Example: when x1 has shape (M, B, K) and x2 has shape (B, K, N):
  - scale is None and batch_split_factor is 1: y has shape (M, B, N) and the same dtype as x1;
  - scale is not None and batch_split_factor is 1: y has shape (M, 1, B * N) and dtype INT8;
  - scale is None and batch_split_factor is greater than 1: y has shape (batch_split_factor, M, B * N / batch_split_factor).

## Function Prototype

```python
cann_ops_nn.transpose_batch_mat_mul(
    x1,
    x2,
    *,
    bias=None,
    scale=None,
    perm_x1=None,
    perm_x2=None,
    perm_y=None,
    enable_hf32=False,
    batch_split_factor=None,
) -> Tensor
```

## Parameter Description

| Parameter | Type | Optional/Required | Description | Data Type | Shape |
| --- | --- | --- | --- | --- | --- |
| x1 | Tensor | Required | Left matrix of the matrix multiplication. The default `perm_x1=[1, 0, 2]` interprets x1 as (M, B, K). | torch.float16 / torch.bfloat16 / torch.float32 | 3D |
| x2 | Tensor | Required | Right matrix of the matrix multiplication. The default `perm_x2=[0, 1, 2]` interprets x2 as (B, K, N). FRACTAL_NZ (WeightNZ) is supported; only FLOAT16 and BFLOAT16 are supported in this format. | Same as x1 | 3D |
| bias | Tensor | Optional | Bias. Reserved and not supported currently. | - | - |
| scale | Tensor | Optional | Output quantization scale. When provided, it enables FLOAT16 input with INT8 output. The shape must be (B * N,). The scale must be converted from FLOAT32 to INT64/UINT64 by `aclnnTransQuantParamV2` before passing in. | torch.int64 / torch.uint64 | 1D, (B * N,) |
| perm_x1 | List[int] | Optional | Permutation of x1. Supports [0, 1, 2] and [1, 0, 2]. Default is [1, 0, 2]. | int64 | - |
| perm_x2 | List[int] | Optional | Permutation of x2. Supports [0, 1, 2] and [0, 2, 1]. Default is [0, 1, 2]. | int64 | - |
| perm_y | List[int] | Optional | Permutation of the output. Only [1, 0, 2] is supported. Default is [1, 0, 2]. | int64 | - |
| enable_hf32 | bool | Optional | Whether to enable HF32 computation. Only effective for FLOAT32 input. It is mapped to aclnn `cubeMathType` (True=3/USE_HF32, False=0/KEEP_DTYPE). | bool | - |
| batch_split_factor | int | Optional | Split factor of the output B dimension. The value range is [1, B], and B must be divisible by it. Default is 1. It must be 1 when scale is provided. | int64 | - |

## Return Value Description

| Output | Type | Description | Data Type | Shape |
| --- | --- | --- | --- | --- |
| y | Tensor | Output of the matrix multiplication. | Same as x1 when scale is None; torch.int8 when scale is provided | (M, B, N) when scale is None and batch_split_factor is 1; (M, 1, B * N) when scale is provided; (batch_split_factor, M, B * N / batch_split_factor) when batch_split_factor > 1 |

## Constraints

- Only 3D tensors are supported. The batch axis and K axis of x1/x2 must match; batch broadcasting is not supported.
- x1/x2 must have the same dtype, which must be FLOAT16, BFLOAT16, or FLOAT32.
- perm_x1 supports [0, 1, 2] and [1, 0, 2]; perm_x2 supports [0, 1, 2] and [0, 2, 1]; perm_y only supports [1, 0, 2].
- scale is only supported for FLOAT16 input with INT8 output. When scale is provided, batch_split_factor must be 1.
- bias is reserved and not supported currently. Pass None.
- x2 supports FRACTAL_NZ (WeightNZ), which can be created by `torch_npu.npu_format_cast(x2, 29)`. In this format, x1/x2 only support FLOAT16 and BFLOAT16.
- Empty tensors are not supported.
- Both eager mode and TorchAir graph mode are supported. In graph mode, the GE Converter converts the call to the `TransposeBatchMatMul` graph operator.

## Deterministic Computation

Deterministic computation is supported by default.

## Examples

- Eager mode without scale:

  ```python
  import torch
  import torch_npu
  import cann_ops_nn

  M, B, K, N = 64, 16, 512, 128
  x1 = torch.randn(M, B, K, dtype=torch.float16).npu()
  x2 = torch.randn(B, K, N, dtype=torch.float16).npu()

  y = cann_ops_nn.transpose_batch_mat_mul(x1, x2)
  print(y.shape, y.dtype)  # torch.Size([64, 16, 128]) torch.float16
  ```

- Eager mode with scale (FLOAT16 input, INT8 output):

  ```python
  import torch
  import torch_npu
  import cann_ops_nn

  M, B, K, N = 64, 16, 512, 128
  x1 = torch.randn(M, B, K, dtype=torch.float16).npu()
  x2 = torch.randn(B, K, N, dtype=torch.float16).npu()
  # The scale must be converted from FLOAT32 to INT64/UINT64 by aclnnTransQuantParamV2.
  scale = torch.ones(B * N, dtype=torch.int64).npu()

  y = cann_ops_nn.transpose_batch_mat_mul(x1, x2, scale=scale)
  print(y.shape, y.dtype)  # torch.Size([64, 1, 2048]) torch.int8
  ```

- batch_split_factor example:

  ```python
  import torch
  import torch_npu
  import cann_ops_nn

  M, B, K, N = 64, 16, 512, 128
  x1 = torch.randn(M, B, K, dtype=torch.float16).npu()
  x2 = torch.randn(B, K, N, dtype=torch.float16).npu()
  y = cann_ops_nn.transpose_batch_mat_mul(x1, x2, batch_split_factor=2)
  print(y.shape)  # torch.Size([2, 64, 1024])
  ```

- x2 in FRACTAL_NZ (WeightNZ):

  ```python
  import torch
  import torch_npu
  import cann_ops_nn

  M, B, K, N = 64, 16, 512, 128
  x1 = torch.randn(M, B, K, dtype=torch.float16).npu()
  x2 = torch.randn(B, K, N, dtype=torch.float16).npu()
  x2_nz = torch_npu.npu_format_cast(x2, 29)  # 29 is the NZ format
  y = cann_ops_nn.transpose_batch_mat_mul(x1, x2_nz)
  print(y.shape, y.dtype)
  ```

- TorchAir graph mode:

  ```python
  import torch
  import torch_npu
  import cann_ops_nn

  @torch.compile(backend="npu")
  def func(x1, x2):
      return cann_ops_nn.transpose_batch_mat_mul(x1, x2)

  M, B, K, N = 64, 16, 512, 128
  x1 = torch.randn(M, B, K, dtype=torch.float16).npu()
  x2 = torch.randn(B, K, N, dtype=torch.float16).npu()
  y = func(x1, x2)
  print(y.shape, y.dtype)
  ```

# fused_matmul

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-22T18:05:58.235Z -->

## Applicable Products

<!-- npu="950" id1 -->
- <term>Ascend 950PR&950DT products</term>: Supported
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3 products</term>: Not supported
<!-- end id2 -->
<!-- npu="910b" id3 -->
- <term>Atlas A2 products</term>: Not supported
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

- Description: fuses matrix multiplication with general vector computation. The underlying implementation encapsulates aclnnFusedMatmulV2.
- Formula:

  $$
  y = OP((x @ x2 + bias), x3)
  $$

  The computations supported by fused_op_type are as follows:

  - "": $y = x @ x2 + bias$.
  - "add": $y = (x @ x2 + bias) + x3$.
  - "mul": $y = (x @ x2 + bias) * x3$.
  - "gelu_erf": $y = gelu\_erf(x @ x2 + bias)$.
  - "gelu_tanh": $y = gelu\_tanh(x @ x2 + bias)$.
  - "relu": $y = relu(x @ x2 + bias)$.
  - "16cast32": $y = cast\_float32(x @ x2 + bias)$.

  When fused_op_type="add" and alpha or beta is not 1:

  $$
  y = alpha * (x @ x2) + beta * x3
  $$

  This use case does not support bias. The operator internally completes the computation in scale_add fusion mode.

## Function Prototype

```python
cann_ops_nn.fused_matmul(
    x,
    x2,
    *,
    bias=None,
    x3=None,
    alpha=None,
    beta=None,
    fused_op_type="",
) -> Tensor
```

## Mandatory

| Name | Type | Type | Description | Data Type | Dimension (Shape) |
| --- | --- | --- | --- | --- | --- |
| x | Tensor | Required | First input matrix of the matrix multiplication, with the last two dimensions (M, K). | torch.float16, torch.bfloat16, torch.float32 | Supports 2-6 dimensions when fused_op_type is "" or "relu"; 2-3 dimensions when it is "add" or "mul"; 2 dimensions for other values. |
| x2 | Tensor | Required | Second input matrix of the matrix multiplication, with the last two dimensions (K, N). The data type and the length of dimension K must be the same as those of x. | Same as x | Supports 2-6 dimensions when fused_op_type is "" or "relu"; 2-3 dimensions when it is "add" or "mul"; 2 dimensions for other values. |
| bias | Tensor | Optional | Bias of the matrix multiplication. Takes effect only when fused_op_type is "", "16cast32", "relu", "add", "mul", "gelu_erf", or "gelu_tanh". | torch.float16, torch.bfloat16, torch.float32 | 1-2 dimensions |
| x3 | Tensor | Optional | Input matrix of the fused operation. Required when fused_op_type is "add" or "mul"; must be None in other use cases. | Same as x | 2-3 dimensions |
| alpha | float | Optional | Scaling coefficient of the matrix multiplication result. Defaults to 1.0 when None is passed in; the value passed in is converted to FP32 internally by the API. | - | - |
| beta | float | Optional | Scaling coefficient of x3. Defaults to 1.0 when None is passed in; the value passed in is converted to FP32 internally by the API. | - | - |
| fused_op_type | str | Optional | Fusion mode. Supported values are "", "16cast32", "add", "mul", "gelu_erf", "gelu_tanh", and "relu". The default value is "". "scale_add" is reserved for internal operator use only and does not support user input. | - | - |

## Returns

| Output | Output Type | Description | Data Type | Dimension (Shape) |
| --- | --- | --- | --- | --- |
| y | Tensor | Output of the fused matrix multiplication. | torch.float32 when fused_op_type is "16cast32", otherwise the same as x | (..., M, N) |

## Constraints

- GELU (gelu_erf and gelu_tanh) supports an optional bias. The computation order is to first perform matrix multiplication, add the bias, and then apply GELU. The data type of bias must be the same as that of the matrix input or FLOAT32, and the shape must be (N,) or (1, N). When bias is passed in, the K axis of matrix multiplication must be greater than 0.

- This API currently supports invocation in single-operator mode.
- x, x2, bias, and x3 must be NPU tensors. None can be passed in for optional tensors.
- x and x2 must have the same data type, and their shapes must satisfy the matrix multiplication relationship. In multi-dimensional use cases, the batch dimensions of x and x2 must be the same, and broadcasting along the batch axis is not supported.
- When x is torch.float16 or torch.bfloat16, the data type of bias must be the same as that of x or torch.float32. When x is torch.float32, bias must be torch.float32.
- When fused_op_type is "add" or "mul", the data type of x3 must be the same as that of x.
- When fused_op_type is "add" or "mul", in BMM (3D) use cases, x3 supports 2-3 dimensions. A 2D x3 can be broadcast as a matrix for a 3D output, and the batch axis of a 3D x3 must be the same as that of y or 1.
- When alpha or beta is not 1, only the 3D non-transposed use case with fused_op_type="add" is supported. Broadcasting of bias and the batch axis is not supported. x, x2, x3, and y must be of the same data type: torch.float16, torch.bfloat16 or torch.float32. The float32 scenario requires HF32 (cube_math_type=USE_HF32 or allow_hf32=True).
- The string length of fused_op_type cannot exceed 100.
- torch_npu.npu.matmul.cube_math_type has the highest priority. If the value is not None, the API directly uses the Cube computation mode set by it, and torch.npu.matmul.allow_hf32 does not take effect. If the value is None, the API selects USE_HF32 or KEEP_DTYPE based on torch.npu.matmul.allow_hf32.

## Deterministic Computation

Defaults to a deterministic computation.

## Example

- Calling in single-operator mode (eager)

  ```python
  import torch
  import torch_npu
  import cann_ops_nn

  x = torch.randn((32, 21, 9), dtype=torch.bfloat16).npu()
  x2 = torch.randn((32, 9, 16), dtype=torch.bfloat16).npu()
  x3 = torch.randn((32, 21, 16), dtype=torch.bfloat16).npu()

  y = cann_ops_nn.fused_matmul(
      x,
      x2,
      x3=x3,
      alpha=1.5,
      beta=0.5,
      fused_op_type="add",
  )
  print(y.shape, y.dtype)
  ```

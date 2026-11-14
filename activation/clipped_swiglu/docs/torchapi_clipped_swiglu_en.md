# clipped_swiglu

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-20T08:47:47.123Z -->

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

- Description:

  The clipped Swish gated linear unit (ClippedSwiGLU) activation function. Compared with standard SwiGLU, this API adds parameters such as `group_index`, `alpha`, `limit`, `bias`, `interleaved`, and `clamp_mode`. These parameters support the SwiGLU variant used by the GPT-OSS model, the grouped use case of MoE models, and the use case where some new models require moving the clamp operation to after the silu activation.

- Formula:

  For a given input tensor `x` with dimensions `[a, b, c, d, e, f, g, ...]`, `clipped_swiglu` performs the following computation:

  1. Merge the axes of `x` based on the input parameter `dim`. After merging, the dimensions are `[pre, cut, after]`. The `cut` axis is the axis to be split into two tensors after merging. The axis can be split in two modes: front-back split or odd-even split. `pre` and `after` can be 1. For example, when `dim` is 3, the dimensions of `x` after merging are `[a*b*c, d, e*f*g*...]`. Since the elements of the `after` axis are stored contiguously and the computation is element-wise, the `cut` axis and the `after` axis are merged, and the dimensions of `x` become `[pre, cut]`.

  2. Filter the `pre` axis of `x` based on the input parameter `group_index`:

     $$
     sum = \text{Sum}(group\_index)
     $$

     $$
     x = x[ : sum, : ]
     $$

     Here, `sum` denotes the sum of all elements in `group_index`. If `group_index` is not provided, skip this step.

  3. Split `x` based on the input parameter `interleaved`:

     If `interleaved=True` (odd-even split):

     $$
     A = x[ : , ::2], \quad B = x[ : , 1::2]
     $$

     If `interleaved=False` (front-back split):

     $$
     h = x.shape[1] // 2
     $$

     $$
     A = x[ : , : h], \quad B = x[ : , h : ]
     $$

  4. Perform Variant SwiGlu computation based on `alpha`, `limit`, `bias`, and `clamp_mode`:

     When `clamp_mode=0` (the clamp operation is performed before silu):

     $$
     A = A.clamp(min=\text{None}, max=limit)
     $$

     $$
     B = B.clamp(min=-limit, max=limit)
     $$

     $$
     y\_glu = A \cdot sigmoid(\alpha \cdot A)
     $$

     $$
     y = y\_glu \cdot (B + bias)
     $$

     When `clamp_mode=1` (the clamp operation is performed after silu; requires underlying support for the V2 API):

     $$
     y\_glu = A \cdot sigmoid(A)
     $$

     $$
     y\_glu = y\_glu.clamp(min=\text{None}, max=limit)
     $$

     $$
     B = B.clamp(min=-limit, max=limit)
     $$

     $$
     y = y\_glu \cdot B
     $$

  5. The number of dimensions of the reshaped output tensor `y` is the same as that of `x` before axis merging. The size along the `dim` axis is half of that of `x`, and the other dimensions are the same as those of `x`.

## Function Prototype

```python
cann_ops_nn.clipped_swiglu(
    x,
    group_index=None,
    *,
    dim=-1,
    alpha=1.702,
    limit=7.0,
    bias=1.0,
    interleaved=True,
    clamp_mode=0,
) -> Tensor
```

## Parameters

| Name | Type | Type | Description | Data Type | Dimension (Shape) |
| --- | --- | --- | --- | --- | --- |
| `x` | Tensor | Required | Input `x` in the formula. The size along the `dim` dimension must be even. | `torch.float16`, `torch.bfloat16`, `torch.float32` | 1-8 dimensions |
| `group_index` | Tensor | Optional | `group_index` in the formula. The `i`-th element indicates the number of batches of `x` to be processed by the `i`-th group. Passing `None` indicates no grouping. | `torch.int64` | 1 dimension. Elements must be greater than or equal to 0. The number of supported groups is determined by the UB space size. |
| `dim` | int | Optional | Dimension index for merging and splitting `x`. Value range: `[-x.dim(), x.dim()-1]`. Default Value: `-1`. | - | - |
| `alpha` | float | Optional | Scaling parameter of the SwiGlu Variant. Recommended Value: `1.702`. Default Value: `1.702`. | - | - |
| `limit` | float | Optional | Threshold of the SwiGlu Variant. The value must be greater than 0. Recommended Value: `7.0`. Default Value: `7.0`. | - | - |
| `bias` | float | Optional | Bias parameter of the SwiGlu Variant. Recommended Value: `1.0`. Default Value: `1.0`. | - | - |
| `interleaved` | bool | Optional | Method of splitting `x`. `True` indicates odd-even splitting, and `False` indicates front-back splitting. Default Value: `True`. | - | - |
| `clamp_mode` | int | Optional | Order of the clamp operation and the silu operation. `0` indicates clamp before silu, and `1` indicates clamp after silu. Default Value: `0`. | - | - |

## Return Values

| Name | Type | Description | Data Type | Dimension (Shape) |
| --- | --- | --- | --- | --- |
| `y` | Tensor | ClippedSwiglu activation result. | Same as `x` | Same as `x`, but the dimension corresponding to `dim` is `x.shape[dim] // 2` |

## Constraints

- This API supports calls in single-operator mode and TorchAir graph mode.
- `x` and `group_index` must both be NPU tensors; optional tensors can be passed as `None`.
- `clamp_mode` supports only the values `0` or `1`; if the `aclnnClippedSwigluV2` kernel is not registered on the target chip, `aclnnClippedSwiglu` is called, and `clamp_mode=1` is ignored and executed as `clamp_mode=0`.

## Deterministic Computation

Defaults to a deterministic computation.

## Example

- Call in single-operator mode:

  ```python
  import torch
  import torch_npu
  import cann_ops_nn

  x = torch.randn(32, 128, dtype=torch.float16).npu()
  group_index = torch.randint(1, 10, (2, ), dtype=torch.int64).npu()

  y = cann_ops_nn.clipped_swiglu(x, group_index=group_index, dim=-1, alpha=1.702, limit=7.0, bias=1.0, interleaved=True, clamp_mode=1)

  print("y:============", y.shape, y.cpu())
  ```

- Call in graph mode (torchair):

  ```python
  import torch
  import torch_npu
  import torchair
  import cann_ops_nn

  npu_backend = "npu"

  class NetModel(torch.nn.Module):
      def __init__(self):
          super().__init__()

      def forward(
          self,
          x,
          group_index,
          dim,
          alpha,
          limit,
          bias,
          interleaved,
          clamp_mode
      ):
          return cann_ops_nn.clipped_swiglu(
              x, group_index=group_index, dim=dim, alpha=alpha, limit=limit, bias=bias, interleaved=interleaved, clamp_mode=clamp_mode
          )

  def clipped_swiglu_test():
      x = torch.randn(32, 128, dtype=torch.float16).npu()
      group_index = torch.randint(1, 10, (2, ), dtype=torch.int64).npu()

      model = NetModel()
      config = torchair.CompilerConfig()
      npu_backend = torchair.get_npu_backend(compiler_config=config)
      model = torch.compile(model, backend=npu_backend, dynamic=True, fullgraph=True)

      y = model(x, group_index=group_index, dim=-1, alpha=1.702, limit=7.0, bias=1.0, interleaved=True, clamp_mode=1)
      print("y:============", y.shape, y.cpu())

  if __name__ == "__main__":
      clipped_swiglu_test()
  ```

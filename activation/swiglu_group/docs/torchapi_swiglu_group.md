# cann_ops_nn.swiglu_group

## 产品支持情况

- <term>Ascend 950PR/Ascend 950DT</term>：支持
- <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：不支持
- <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：不支持
- <term>Atlas 200I/500 A2 推理产品</term>：不支持
- <term>Atlas 推理系列产品</term>：不支持
- <term>Atlas 训练系列产品</term>：不支持

## 功能说明

- 接口功能：

  对输入 `x` 执行 SwiGLU 激活，支持可选的逐 token 权重和 count 模式分组。底层封装 `aclnnSwigluGroup`。

- 计算公式：

  令输入 `x` 最后一维大小为 `D`，`A = x[..., :D/2]`，`B = x[..., D/2:]`。

  $$
  y = silu(A) \times B = \frac{A}{1 + e^{-A}} \times B
  $$

  当 `clamp_limit > 0` 时，先对 `A` 和 `B` 做截断；当 `weight` 非空时，对结果逐 token 乘以 `weight`。

## 函数原型

```python
torch.ops.cann_ops_nn.swiglu_group(
    x,
    *,
    weight=None,
    group_index=None,
    clamp_limit=-1.0,
) -> Tensor
```

## 参数说明

| 参数名 | 参数类型 | 可选/必选 | 描述 | 数据类型 | 维度(shape) |
| --- | --- | --- | --- | --- | --- |
| `x` | Tensor | 必选 | SwiGLU 输入，最后一维会被均分为两部分。 | `torch.float16`、`torch.bfloat16`、`torch.float32` | 1-8维，最后一维为偶数 |
| `weight` | Tensor | 可选 | 逐 token 权重，非空时乘到 SwiGLU 结果上。 | `torch.float32` | 元素个数等于 `x` 除最后一维外的元素个数 |
| `group_index` | Tensor | 可选 | count 模式分组 token 数。 | `torch.int64` | 1维 |
| `clamp_limit` | float | 可选 | 激活前截断阈值，默认 `-1.0` 表示不启用截断。 | - | - |

## 返回值说明

| 参数名 | 参数类型 | 描述 | 数据类型 | 维度(shape) |
| --- | --- | --- | --- | --- |
| `y` | Tensor | SwiGLU 激活结果。 | 与 `x` 相同 | 与 `x` 相同，但最后一维为 `x.shape[-1] // 2` |

## 约束说明

- 该接口支持单算子模式和 TorchAir 图模式调用。
- `x`、`weight`、`group_index` 均需为 NPU Tensor；可选 Tensor 可以传 `None`。
- `x` 的最后一维必须为偶数。
- 不支持空 Tensor 和非连续 Tensor。
- `group_index` 表示每个 group 的 token 数，所有元素需大于等于 0。

## 确定性计算

默认支持确定性计算。

## 调用说明

- 单算子模式调用：

  ```python
  import torch
  import torch_npu
  import cann_ops_nn.ops

  x = torch.randn(8, 512, dtype=torch.float16).npu()
  weight = torch.ones(8, dtype=torch.float32).npu()
  group_index = torch.tensor([4, 4], dtype=torch.int64).npu()

  y = torch.ops.cann_ops_nn.swiglu_group(
      x, weight=weight, group_index=group_index, clamp_limit=-1.0)
  print(y.shape, y.dtype)
  ```

- 图模式（torchair）调用：

  ```python
  import torch
  import torch_npu
  import torchair
  import cann_ops_nn.ops

  class Model(torch.nn.Module):
      def forward(self, x):
          return torch.ops.cann_ops_nn.swiglu_group(x, clamp_limit=-1.0)

  model = torch.compile(Model().npu(), backend=torchair.get_npu_backend(), dynamic=False)
  x = torch.randn(8, 512, dtype=torch.float16).npu()
  y = model(x)
  ```

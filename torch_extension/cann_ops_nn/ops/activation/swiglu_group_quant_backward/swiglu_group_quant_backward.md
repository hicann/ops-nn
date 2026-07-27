# cann_ops_nn.swiglu_group_quant_backward

## 产品支持情况

- <term>Ascend 950PR/Ascend 950DT</term>：支持
- <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：不支持
- <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：不支持
- <term>Atlas 200I/500 A2 推理产品</term>：不支持
- <term>Atlas 推理系列产品</term>：不支持
- <term>Atlas 训练系列产品</term>：不支持

## 功能说明

- 接口功能：

  `SwigluGroupQuantGrad` 是 `SwigluGroupQuant` 算子的反向梯度计算算子。根据量化输出 `y` 的梯度 `grad_y`、前向输入 `x` 以及可选的前向辅助张量，反向计算输入 `x` 的梯度 `grad_x` 和可选权重 `weight` 的梯度 `grad_weight`。底层封装 `aclnnSwigluGroupQuantGrad`。

- 计算流程：

  1. 接收前向量化输出的梯度 `grad_y` 和前向输入 `x`。
  2. 根据前向 SwiGLU 的输入分段和量化参数，沿量化路径和 SwiGLU 激活路径反向传播梯度。
  3. 输出 `grad_x`（与 `x` 同 shape/dtype）和 `grad_weight`（与 `weight` 同 shape，若 `weight` 未提供则为空 Tensor）。

## 函数原型

```python
cann_ops_nn.swiglu_group_quant_backward(
    grad_y,
    x,
    *,
    weight=None,
    y_origin=None,
    group_index=None,
    clamp_limit=-1.0,
) -> (Tensor, Tensor)
```

## 参数说明

| 参数名 | 参数类型 | 可选/必选 | 描述 | 数据类型 | 维度(shape) |
| --- | --- | --- | --- | --- | --- |
| `grad_y` | Tensor | 必选 | 梯度输入张量，即前向量化输出 `y` 的梯度。 | `torch.float16`、`torch.bfloat16`、`torch.float32` | ND，与前向输出 `y` 相同 |
| `x` | Tensor | 必选 | 前向计算的输入张量。 | `torch.float16`、`torch.bfloat16`、`torch.float32` | ND，与前向输入 `x` 相同 |
| `weight` | Tensor | 可选 | 前向计算使用的 topk 权重张量。 | `torch.float32` | ND，与前向 `weight` 相同 |
| `y_origin` | Tensor | 可选 | 前向量化前的 SwiGLU 输出张量。 | `torch.float16`、`torch.bfloat16`、`torch.float32` | ND，与前向 `y_origin` 相同 |
| `group_index` | Tensor | 可选 | 前向动态量化使用的分组索引张量。 | `torch.int64` | ND，与前向 `group_index` 相同 |
| `clamp_limit` | float | 可选 | 梯度掩码的截断值，默认 `-1.0` 表示不截断。设为正值时在激活前对 SwiGLU 输入做截断。必须为 `-1.0` 或大于 `0`，且需与前向 `clamp_limit` 一致。 | - | - |

## 返回值说明

| 参数名 | 参数类型 | 描述 | 数据类型 | 维度(shape) |
| --- | --- | --- | --- | --- |
| `grad_x` | Tensor | 输入 `x` 的梯度，类型与 `x` 相同。 | 与 `x` 相同 | ND，与 `x` 相同 |
| `grad_weight` | Tensor | 权重 `weight` 的梯度。若 `weight` 未提供则返回 shape 为 `[0]` 的空 Tensor。 | `torch.float32` | ND，与 `weight` 相同，或 `[0]` |

## 约束说明

- 该接口支持单算子模式和 TorchAir 图模式调用。
- `grad_y`、`x` 均需为 NPU Tensor；可选 Tensor 可以传 `None`。
- 输入 `x` 的 rank 必须大于 0，最后一维必须为偶数。
- `clamp_limit` 不启用时使用默认值 `-1.0`；启用时必须大于 `0`，且需与前向 `clamp_limit` 保持一致。
- 不支持空 Tensor 和非连续 Tensor。

## 确定性计算

默认支持确定性计算。

## 调用说明

- 单算子模式调用：

  ```python
  import torch
  import torch_npu
  import cann_ops_nn.ops

  grad_y = torch.randn(8, 256, dtype=torch.float16).npu()
  x = torch.randn(8, 512, dtype=torch.float16).npu()
  weight = torch.randn(256, dtype=torch.float32).npu()

  grad_x, grad_weight = cann_ops_nn.swiglu_group_quant_backward(
      grad_y, x, weight=weight, clamp_limit=-1.0
  )
  ```

- 图模式（torchair）调用：

  ```python
  import torch
  import torch_npu
  import torchair
  import cann_ops_nn.ops

  class Model(torch.nn.Module):
      def forward(self, grad_y, x, weight):
          grad_x, grad_weight = cann_ops_nn.swiglu_group_quant_backward(
              grad_y, x, weight=weight
          )
          return grad_x, grad_weight

  model = torch.compile(Model().npu(), backend=torchair.get_npu_backend(), dynamic=False)
  grad_y = torch.randn(8, 256, dtype=torch.float16).npu()
  x = torch.randn(8, 512, dtype=torch.float16).npu()
  weight = torch.randn(256, dtype=torch.float32).npu()
  grad_x, grad_weight = model(grad_y, x, weight)
  ```

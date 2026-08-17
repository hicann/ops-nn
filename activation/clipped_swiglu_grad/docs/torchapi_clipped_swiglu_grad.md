# clipped_swiglu_grad

## 产品支持情况

<!-- npu="950" id1 -->
- <term>Ascend 950PR/Ascend 950DT</term>：支持
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：支持
<!-- end id2 -->
<!-- npu="910b" id3 -->
- <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：支持
<!-- end id3 -->
<!-- npu="310b" id4 -->
- <term>Atlas 200I/500 A2 推理产品</term>：不支持
<!-- end id4 -->
<!-- npu="310p" id5 -->
- <term>Atlas推理系列产品</term>：不支持
<!-- end id5 -->
<!-- npu="910" id6 -->
- <term>Atlas训练系列产品</term>：不支持
<!-- end id6 -->

## 功能说明

- 接口功能：

  `ClippedSwigluGrad`是`ClippedSwiglu`算子的反向梯度计算算子。根据正向输出`y`的梯度`grad_y`、前向输入`x`以及可选的分组索引`group_index`，反向计算输入`x`的梯度`grad_x`。底层封装`aclnnClippedSwigluGrad`。

- 计算流程：

  1. 将`x`基于参数`dim`进行合轴，合轴后维度为`[pre, dim2H]`；`grad_y`合轴后维度为`[pre, dim2H // 2]`。
  2. 根据`group_index`，对`x`的`pre`轴进行过滤处理，仅前`group`行参与计算，其余行梯度为 0。
  3. 根据`interleaved`，对`x`进行切分，得到`x0`和`x1`两部分（与正向一致）。
  4. 重算正向中间量：
     - $\mathbf{x}_0' = \min(\mathbf{x}_0, \text{limit})$
     - $\mathbf{x}_1' = \min(\max(\mathbf{x}_1, -\text{limit}), \text{limit})$
     - $s = \sigma(\alpha \times \mathbf{x}_0') = \frac{1}{1 + e^{-\alpha \times \mathbf{x}_0'}}$
  5. 计算反向梯度：
     - $\text{mask}_{x_0} = (\mathbf{x}_0 \le \text{limit})$
     - $\text{mask}_{x_1} = (-\text{limit} \le \mathbf{x}_1 \le \text{limit})$
     - $\mathbf{grad}_{x_0} = \text{grad\_y} \times (\mathbf{x}_1' + \text{bias}) \times s \times (1 + \alpha \times \mathbf{x}_0' \times (1 - s)) \times \text{mask}_{x_0}$
     - $\mathbf{grad}_{x_1} = \text{grad\_y} \times \mathbf{x}_0' \times s \times \text{mask}_{x_1}$
  6. 将`grad_x0`、`grad_x1`散回`dx`（与切分方式对应），重塑输出维度与`x`一致。

## 函数原型

```python
cann_ops_nn.clipped_swiglu_grad(
    grad_y,
    x,
    *,
    group_index=None,
    dim=-1,
    alpha=1.702,
    limit=7.0,
    bias=1.0,
    interleaved=True,
    clamp_mode=0,
) -> Tensor
```

## 参数说明

| 参数名 | 参数类型 | 可选/必选 | 描述 | 数据类型 | 维度(shape) |
| --- | --- | --- | --- | --- | --- |
| `grad_y` | Tensor | 必选 | 正向输出`y`的梯度，`dim`维为`x`的一半。 | `torch.float16`、`torch.bfloat16`、`torch.float32` | 与`x`相同，但`dim`维为`x`对应维度的一半 |
| `x` | Tensor | 必选 | 正向输入，用于重算正向中间量。 | `torch.float16`、`torch.bfloat16`、`torch.float32` | 1-8维，`dim`维大小为偶数 |
| `group_index` | Tensor | 可选 | 分组索引，第`i`个元素表示第`i`组处理的batch数量。 | `torch.int64` | 1维，元素个数不超过 8192 |
| `dim` | int | 可选 | 合轴及切分的维度序号，取值范围`[-x.dim(), x.dim()-1]`，默认`-1`。 | - | - |
| `alpha` | float | 可选 | SwiGLU激活系数，默认`1.702`。 | - | - |
| `limit` | float | 可选 | 截断门限值，必须大于 0，默认`7.0`。 | - | - |
| `bias` | float | 可选 | 线性计算偏差，默认`1.0`。 | - | - |
| `interleaved` | bool | 可选 | 切分方式。`true`=奇偶切分，`false`=前后切分，默认`true`。 | - | - |
| `clamp_mode` | int | 可选 | clamp位置控制，默认`0`。当前仅支持`0`（clamp在silu之前），后续版本将支持`1`（clamp移至silu之后）。 | - | - |

## 返回值说明

| 参数名 | 参数类型 | 描述 | 数据类型 | 维度(shape) |
| --- | --- | --- | --- | --- |
| `grad_x` | Tensor | 输入`x`的梯度。 | 与`x`相同 | 与`x`相同 |

## 约束说明

- 该接口支持单算子模式和TorchAir图模式调用。
- `grad_y`、`x`均需为NPU Tensor；`group_index`可传`None`。
- `grad_y`与`x`的数据类型必须一致。
- `x`的rank必须大于 0，`dim`维大小必须为偶数。
- `grad_y`的`dim`维大小必须等于`x`对应维度的一半，其余维度与`x`一致。
- `limit`必须大于 0。
- `clamp_mode`当前仅支持取值`0`。
- 不支持空Tensor和非连续Tensor。

## 确定性计算

默认支持确定性计算。

## 调用说明

- 单算子模式调用：

  ```python
  import torch
  import torch_npu
  import cann_ops_nn.ops

  x = torch.randn(8, 512, dtype=torch.float16).npu()
  grad_y = torch.randn(8, 256, dtype=torch.float16).npu()

  grad_x = torch.ops.cann_ops_nn.clipped_swiglu_grad(
      grad_y, x, dim=-1, alpha=1.702, limit=7.0, bias=1.0, interleaved=True)
  print(grad_x.shape, grad_x.dtype)
  ```

- 图模式（torchair）调用：

  ```python
  import torch
  import torch_npu
  import torchair
  import cann_ops_nn.ops

  class Model(torch.nn.Module):
      def forward(self, grad_y, x):
          return torch.ops.cann_ops_nn.clipped_swiglu_grad(grad_y, x)

  model = torch.compile(Model().npu(), backend=torchair.get_npu_backend(), dynamic=False)
  grad_y = torch.randn(8, 256, dtype=torch.float16).npu()
  x = torch.randn(8, 512, dtype=torch.float16).npu()
  grad_x = model(grad_y, x)
  ```

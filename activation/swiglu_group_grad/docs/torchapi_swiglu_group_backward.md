# swiglu_group_backward

## 产品支持情况

<!-- npu="950" id1 -->
- <term>Ascend 950PR/Ascend 950DT</term>：支持
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：不支持
<!-- end id2 -->
<!-- npu="910b" id3 -->
- <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：不支持
<!-- end id3 -->
<!-- npu="310b" id4 -->
- <term>Atlas 200I/500 A2 推理产品</term>：不支持
<!-- end id4 -->
<!-- npu="310p" id5 -->
- <term>Atlas 推理系列产品</term>：不支持
<!-- end id5 -->
<!-- npu="910" id6 -->
- <term>Atlas 训练系列产品</term>：不支持
<!-- end id6 -->

## 功能说明

- 接口功能：

  完成ClampedSwiglu激活函数的反向梯度计算。从上游梯度`grad_output`和前向输入`x`重算clamp mask与sigmoid，输出`grad_x`与可选 `grad_weight`。底层调用`aclnnSwigluGroupGrad`。

- 计算公式：

  前向分解：`x`按hidden维劈半得到gate(`g`)和up(`u`)；可选clamp产生`g̃ = min(c, g)`、`ũ = clip(u, −c, c)`；`SiLU(g̃) = g̃·σ(g̃)`；`y = SiLU(g̃)·ũ·w_t`。

  $$
  silu'(g̃) = s + f − f·s
  $$

  $$
  dg = grad\_output \cdot silu'(g̃) \cdot ũ \cdot w_t \cdot I(g < c) \cdot m_r
  $$

  $$
  du = grad\_output \cdot f \cdot w_t \cdot I(−c < u < c) \cdot m_r
  $$

  $$
  grad\_weight = \Sigma(grad\_output \cdot y\_origin) \text{ along hidden dim}
  $$

  其中`I`为开区间指示函数（边界值时 mask=0），`m_r`为group_index mask，`w_t`为weight的broadcast。

## 函数原型

```python
torch.ops.cann_ops_nn.swiglu_group_backward(
    grad_output,
    x,
    *,
    weight=None,
    y_origin=None,
    group_index=None,
    clamp_limit=-1.0,
) -> (Tensor, Tensor?)
```

## 参数说明

| 参数名 | 参数类型 | 可选/必选 | 描述 | 数据类型 | 维度(shape) |
| --- | --- | --- | --- | --- | --- |
| `grad_output`| Tensor | 必选 | 上游梯度。 | `torch.float16`、`torch.bfloat16`、`torch.float32`| 1-8维 |
| `x`| Tensor | 必选 | 前向输入，包含 gate 和 up 分支。 | `torch.float16`、`torch.bfloat16`、`torch.float32`| 1-8维，最后一维为 `2 * grad_output.shape[-1]`|
| `weight`| Tensor | 可选 | MoE top-k路由权重，缺省视作全1。 | `torch.float32`| 元素个数等于`grad_output`除最后一维外的元素个数 |
| `y_origin`| Tensor | 可选 | 前向输出 y，weight存在时y已乘该权重。 | 与`grad_output`相同 | 最后一维与`grad_output`相同，非尾轴元素个数等于`grad_output`非尾轴元素个数 |
| `group_index`| Tensor | 可选 | 各分组token数量索引，缺省视作全部行有效。 | `torch.int64`| 1维 |
| `clamp_limit`| float | 可选 | 截断门限，默认`-1.0`表示不clamp，启用时必须大于0。 | - | - |

## 返回值说明

| 参数名 | 参数类型 | 描述 | 数据类型 | 维度(shape) |
| --- | --- | --- | --- | --- |
|`grad_x`| Tensor | x的梯度。 | 与`grad_output`相同 | 与`x`相同 |
| `grad_weight`| Tensor | weight的梯度，仅weight和y_origin同时提供时返回。 | `torch.float32`| 与`weight`相同 |

## 约束说明

- 该接口支持单算子模式和TorchAir图模式调用。
- `grad_output`、`x`、`weight`、`y_origin`、`group_index`均需为NPU Tensor；可选Tensor可以传`None`。
- `weight`和`y_origin`必须同时提供或同时为空。
- `grad_output`与`x`的非尾轴元素总数相等，`x.shape[-1]`必须等于`2 * grad_output.shape[-1]`。
- `group_index`非空时必须是一维非空Tensor（G > 0）。
- `clamp_limit`为`-1.0`或大于`0.0`，`-1.0`表示不clamp。

## 确定性计算

默认支持确定性计算。

## 调用示例

- 单算子模式调用：

  ```python
  import torch
  import torch_npu
  import cann_ops_nn.ops

  grad_output = torch.randn(8, 128, dtype=torch.float16).npu()
  x = torch.randn(8, 256, dtype=torch.float16).npu()
  weight = torch.ones(8, 1, dtype=torch.float32).npu()
  y_origin = torch.randn(8, 128, dtype=torch.float16).npu()
  group_index = torch.tensor([8], dtype=torch.int64).npu()

  grad_x, grad_weight = torch.ops.cann_ops_nn.swiglu_group_backward(
      grad_output, x, weight=weight, y_origin=y_origin,
      group_index=group_index, clamp_limit=7.0)
  print(grad_x.shape, grad_x.dtype)
  print(grad_weight.shape, grad_weight.dtype)
  ```

- 图模式（torchair）调用：

  ```python
  import torch
  import torch_npu
  import torchair
  import cann_ops_nn.ops

  class Model(torch.nn.Module):
      def forward(self, grad_output, x):
          grad_x, _ = torch.ops.cann_ops_nn.swiglu_group_backward(
              grad_output, x, clamp_limit=-1.0)
          return grad_x

  model = torch.compile(Model().npu(), backend=torchair.get_npu_backend(), dynamic=False)
  grad_output = torch.randn(8, 128, dtype=torch.float16).npu()
  x = torch.randn(8, 256, dtype=torch.float16).npu()
  grad_x = model(grad_output, x)
  ```

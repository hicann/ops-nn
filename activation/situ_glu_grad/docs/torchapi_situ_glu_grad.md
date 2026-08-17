# situ_glu_grad

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

- <term>Atlas 推理系列产品</term>：不支持

<!-- end id5 -->

<!-- npu="910" id6 -->

- <term>Atlas 训练系列产品</term>：不支持

<!-- end id6 -->

## 功能说明

- 接口功能：

  [situ_glu](../../situ_glu/docs/torchapi_situ_glu.md)的反向梯度算子。给定前向输出梯度`grad_y`与前向输入`x`，按SiTU门控线性单元公式对`x`求导，返回输入梯度`grad_x`。底层封装`aclnnSituGluGrad`。

- 计算公式：

  对给定的输入张量`x`，基于输入参数`dim`进行合轴，合轴后维度为`[pre, cut]`，其中`cut`必须为偶数。令`h = cut // 2`。已知前向输出梯度$\text{grad\_y}$（shape与前向输出`y`一致，即`dim`轴为`x`的一半），按以下步骤计算输入梯度`grad_x`：

  1. 根据`activate_left`对`x`进行前后切分得到`gate`与`up`（与前向一致，各shape为`[..., h]`）：

     - `activate_left=True`（默认）：`gate = x[..., :h]`，`up = x[..., h:]`
     - `activate_left=False`：`gate = x[..., h:]`，`up = x[..., :h]`

  2. 计算前向中间量：

     $$
     t_g = \tanh\left(\frac{gate}{\beta}\right),\quad s_g = \text{sigmoid}(gate),\quad \text{situ\_a} = \beta \cdot t_g \cdot s_g
     $$

     当`linear_beta > 0`时：

     $$
     t_u = \tanh\left(\frac{up}{linear\_beta}\right),\quad up' = linear\_beta \cdot t_u
     $$

     当`linear_beta <= 0`时，$up' = up$。前向输出为$y = \text{situ\_a} \cdot up'$。

  3. 乘积法则：

     $$
     \text{grad\_situ\_a} = \text{grad\_y} \cdot up'
     $$

     $$
     \text{grad\_up'} = \text{grad\_y} \cdot \text{situ\_a}
     $$

  4. gate梯度：

     $$
     \text{grad\_gate} = \text{grad\_situ\_a} \cdot s_g \cdot \left((1 - t_g^2) + \beta \cdot t_g \cdot (1 - s_g)\right)
     $$

  5. up梯度：

     - `linear_beta > 0`时：$\text{grad\_up} = \text{grad\_up'} \cdot (1 - t_u^2)$
     - `linear_beta <= 0`时：$\text{grad\_up} = \text{grad\_up'}$

  6. 按`activate_left`将`grad_gate`、`grad_up`拼接为`grad_x`（shape与`x`一致）：

     - `activate_left=True`：$\text{grad\_x} = \text{concat}([\text{grad\_gate}, \text{grad\_up}], \text{dim})$
     - `activate_left=False`：$\text{grad\_x} = \text{concat}([\text{grad\_up}, \text{grad\_gate}], \text{dim})$

## 函数原型

```python
cann_ops_nn.situ_glu_grad(
    grad_y,
    x,
    *,
    dim=-1,
    beta=1.0,
    linear_beta=0.0,
    activate_left=True
) -> Tensor
```

## 参数说明

| 参数名 | 参数类型 | 可选/必选 | 描述 | 数据类型 | 维度(shape) |
| --- | --- | --- | --- | --- | --- |
| `grad_y` | Tensor | 必选 | 前向输出`y`的梯度。 | float16、float32、bfloat16 | 与`x`一致，但`dim`维度大小为`x.shape[dim] // 2` |
| `x` | Tensor | 必选 | 前向输入，`dim`维度会被均分为gate与up两部分。`grad_y`与`x`数据类型需一致。 | float16、float32、bfloat16 | 1-8维，`dim`维度为偶数 |
| `dim` | int | 可选 | 切分维度，取值范围`[-x.dim(), x.dim()-1]`，默认`-1`。需与前向`situ_glu`一致。 | - | - |
| `beta` | float | 可选 | SiTU门控部分的缩放系数，需与前向`situ_glu`保持一致，默认`1.0`。 | - | - |
| `linear_beta` | float | 可选 | up路径线性tanh的缩放系数，需与前向`situ_glu`保持一致。默认`0.0`。 | - | - |
| `activate_left` | bool | 可选 | 切分`x`时gate是否为前半部分，需与前向`situ_glu`保持一致。默认`True`。 | - | - |

## 返回值说明

| 参数名 | 参数类型 | 描述 | 数据类型 | 维度(shape) |
| --- | --- | --- | --- | --- |
| `grad_x` | Tensor | 输入`x`的梯度。 | 与`x`一致 | 与`x`完全一致 |

## 约束说明

- 该接口支持单算子模式和TorchAir图模式调用。
- `dim`非last维时，若尾部维度组合不满足半行32B对齐，会回退到Long-H路径，功能正确但性能略降。
- `beta`、`linear_beta`、`activate_left`、`dim`需与前向`situ_glu`调用时的取值一致，否则梯度结果不正确。

## 确定性计算

- 默认支持确定性计算。

## 调用示例

- 单算子模式调用：

  ```python
  import torch
  import torch_npu
  import cann_ops_nn

  x = torch.randn(2, 32, dtype=torch.float32).npu()
  grad_y = torch.randn(2, 16, dtype=torch.float32).npu()

  # 与前向参数一致的调用
  grad_x = cann_ops_nn.situ_glu_grad(
      grad_y, x, dim=-1, beta=1.0, linear_beta=0.0, activate_left=True
  )
  print(grad_x.shape, grad_x.dtype)  # torch.Size([2, 32]) torch.float32
  ```

- TorchAir图模式调用：

  ```python
  import torch
  import torch_npu
  import torchair
  import cann_ops_nn

  class Model(torch.nn.Module):
      def forward(self, grad_y, x):
          return cann_ops_nn.situ_glu_grad(
              grad_y, x, dim=-1, beta=1.0, linear_beta=0.0, activate_left=True
          )

  model = torch.compile(Model().npu(), backend=torchair.get_npu_backend(), dynamic=False)
  x = torch.randn(2, 32, dtype=torch.float32).npu()
  grad_y = torch.randn(2, 16, dtype=torch.float32).npu()
  grad_x = model(grad_y, x)
  ```

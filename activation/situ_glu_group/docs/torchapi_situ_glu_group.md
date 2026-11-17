# situ_glu_group

## 产品支持情况

<!-- npu="950" id1 -->

- <term>Ascend 950PR&950DT系列产品</term>：支持

<!-- end id1 -->

<!-- npu="A3" id2 -->

- <term>Atlas A3系列产品</term>：不支持

<!-- end id2 -->

<!-- npu="910b" id3 -->

- <term>Atlas A2系列产品</term>：不支持

<!-- end id3 -->

<!-- npu="310b" id4 -->

- <term>Atlas 200I/500 A2推理产品</term>：不支持

<!-- end id4 -->

<!-- npu="310p" id5 -->

- <term>Atlas推理系列产品</term>：不支持

<!-- end id5 -->

<!-- npu="910" id6 -->

- <term>Atlas训练系列产品</term>：不支持

<!-- end id6 -->

## 功能说明

- 接口功能：

  SiTU门控线性单元（SiTU Gated Linear Unit）激活函数。对输入张量`x`沿指定维度`dim`切分为门控（gate）与上路径（up）两半，按SiTU公式计算输出。底层封装`aclnnSituGluGroup`。

- 计算公式：

  对给定的输入张量`x`，按以下步骤计算SiTU输出`y`：

  1. 基于`dim`进行合轴，合轴后维度为`[pre, cut]`，其中`cut`必须为偶数。令`h = cut // 2`。

  2. 根据`activate_left`对`x`进行前后切分：

     - `activate_left=True`（默认）：`gate = x[..., :h]`，`up = x[..., h:]`
     - `activate_left=False`：`gate = x[..., h:]`，`up = x[..., :h]`

  3. 根据`beta`、`linear_beta`进行SiTU计算：

     $$
     situ\_a = \beta \cdot \tanh\left(\frac{gate}{\beta}\right) \cdot \text{sigmoid}(gate)
     $$

     当`linear_beta > 0`时：

     $$
     up = linear\_beta \cdot \tanh\left(\frac{up}{linear\_beta}\right)
     $$

     当`linear_beta <= 0`时，`up`直接透传。

  4. 输出：

     $$
     y = situ\_a \cdot up
     $$

  5. 可选：当传入`weight`时，对每一行（`dim`轴之前各维的索引）逐行乘对应标量：

     $$
     y[i] = y[i] \cdot weight[i]
     $$

  6. 可选：当传入`group_index`时，仅前`realBs`行参与计算并写出，`realBs = min(sum(group_index), 行数)`。

  7. 输出张量`y`的维度与`x`一致，`dim`轴上的大小为`x`的一半，其他维度与`x`相同。

## 函数原型

```python
cann_ops_nn.situ_glu_group(
    x,
    *,
    weight=None,
    group_index=None,
    dim=-1,
    beta=1.0,
    linear_beta=0.0,
    activate_left=True
) -> Tensor
```

## 参数说明

| 参数名 | 参数类型 | 可选/必选 | 描述 | 数据类型 | 维度(shape) |
| --- | --- | --- | --- | --- | --- |
| `x` | Tensor | 必选 | SiTU输入，`dim`维度会被均分为门控（gate）与上路径（up）两部分。 | float16、float32、bfloat16 | 1-8维，`dim`维度为偶数 |
| `weight` | Tensor | 可选 | 逐行权重标量，乘到SiTU激活结果上。元素个数等于`x`在`dim`轴之前各维的乘积；`dim=-1`时等于除最后一维外各维的乘积。 | float32 | 1-8维 |
| `group_index` | Tensor | 可选 | count模式的分组token数，1维非空，元素值必须大于等于0且`sum(group_index) > 0`。传入时仅前`min(sum(group_index), 行数)`行参与计算并写出，其余行不写出（内容未定义），调用方需自行掩码。 | int64 | 1维 |
| `dim` | int | 可选 | 切分维度，取值范围`[-x.dim(), x.dim()-1]`，默认`-1`。 | - | - |
| `beta` | float | 可选 | SiTU门控部分的缩放系数，控制tanh非线性强度。必须为非零值，传入0会返回参数校验失败，默认`1.0`。 | - | - |
| `linear_beta` | float | 可选 | up路径线性tanh的缩放系数。大于0时对up施加有界化变换；小于等于0时up直接透传。默认`0.0`。 | - | - |
| `activate_left` | bool | 可选 | 切分`x`时gate是否为前半部分。`True`表示gate为前半、up为后半；`False`表示gate为后半、up为前半。默认`True`。 | - | - |

## 返回值说明

| 参数名 | 参数类型 | 描述 | 数据类型 | 维度(shape) |
| --- | --- | --- | --- | --- |
| `y` | Tensor | SiTU激活结果。 | 与`x`一致 | 与`x`相同，但`dim`维度大小为`x.shape[dim] // 2` |

## 约束说明

- 该接口支持训练、推理场景下使用。
- 该接口支持单算子模式和TorchAir图模式调用。
- `beta`必须为非零值，传入0会返回参数校验失败。
- `weight`、`group_index`为可选输入，元素个数/形状约束见参数说明。
- `group_index`传入时，元素值必须大于等于0且`sum(group_index) > 0`；仅前realBs行写出，其余行内容未定义，调用方需自行掩码。

## 确定性计算

- 默认支持确定性计算。

## 调用示例

- 单算子模式调用：

  ```python
  import torch
  import torch_npu
  import cann_ops_nn

  x = torch.randn(2, 32, dtype=torch.float32).npu()

  # 默认参数调用
  y = cann_ops_nn.situ_glu_group(x)
  print(y.shape, y.dtype)  # torch.Size([2, 16]) torch.float32

  # 指定全部参数
  y = cann_ops_nn.situ_glu_group(x, dim=-1, beta=1.0, linear_beta=0.0, activate_left=True)
  print(y.shape, y.dtype)

  # 传入可选输入 weight / group_index
  weight = torch.rand(x.shape[0], dtype=torch.float32).npu()
  group_index = torch.tensor([2], dtype=torch.int64).npu()
  y = cann_ops_nn.situ_glu_group(x, weight=weight, group_index=group_index)
  print(y.shape, y.dtype)
  ```

- TorchAir图模式调用：

  ```python
  import torch
  import torch_npu
  import torchair
  import cann_ops_nn

  class Model(torch.nn.Module):
      def forward(self, x, weight, group_index):
          return cann_ops_nn.situ_glu_group(
              x,
              weight=weight,
              group_index=group_index,
              dim=-1,
              beta=1.0,
              linear_beta=0.0,
              activate_left=True,
          )

  model = torch.compile(Model().npu(), backend=torchair.get_npu_backend(), dynamic=False)
  x = torch.randn(2, 32, dtype=torch.float32).npu()
  weight = torch.rand(x.shape[0], dtype=torch.float32).npu()
  group_index = torch.tensor([2], dtype=torch.int64).npu()
  y = model(x, weight, group_index)
  print(y.shape, y.dtype)
  ```

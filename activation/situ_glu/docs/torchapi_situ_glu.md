# situ_glu

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

  SiTU门控线性单元（SiTU Gated Linear Unit）激活函数。对输入张量`x`沿指定维度`dim`切分为门控（gate）与上路径（up）两半，按SiTU公式计算输出。底层封装`aclnnSituGlu`。

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

  5. 输出张量`y`的维度与`x`一致，`dim`轴上的大小为`x`的一半，其他维度与`x`相同。

## 函数原型

```python
cann_ops_nn.situ_glu(
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
| `x` | Tensor | 必选 | SiTU输入，`dim`维度会被均分为门控（gate）与上路径（up）两部分。 | float16、float32、bfloat16 | 1-8维，`dim`维度为偶数 |
| `dim` | int | 可选 | 切分维度，取值范围`[-x.dim(), x.dim()-1]`，默认`-1`。 | - | - |
| `beta` | float | 可选 | SiTU门控部分的缩放系数，控制tanh非线性强度。建议为非零正值，默认`1.0`。 | - | - |
| `linear_beta` | float | 可选 | up路径线性tanh的缩放系数。大于0时对up施加有界化变换；小于等于0时up直接透传。默认`0.0`。 | - | - |
| `activate_left` | bool | 可选 | 切分`x`时gate是否为前半部分。`True`表示gate为前半、up为后半；`False`表示gate为后半、up为前半。默认`True`。 | - | - |

## 返回值说明

| 参数名 | 参数类型 | 描述 | 数据类型 | 维度(shape) |
| --- | --- | --- | --- | --- |
| `y` | Tensor | SiTU激活结果。 | 与`x`一致 | 与`x`相同，但`dim`维度大小为`x.shape[dim] // 2` |

## 约束说明

- 该接口支持训练、推理场景下使用。
- 该接口支持单算子模式和TorchAir图模式调用。
- `dim`非last维时，若尾部维度组合不满足半行32B对齐，会回退到Long-H路径，功能正确但性能略降。

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
  y = cann_ops_nn.situ_glu(x)
  print(y.shape, y.dtype)  # torch.Size([2, 16]) torch.float32

  # 指定全部参数
  y = cann_ops_nn.situ_glu(x, dim=-1, beta=1.0, linear_beta=0.0, activate_left=True)
  print(y.shape, y.dtype)
  ```

- TorchAir图模式调用：

  ```python
  import torch
  import torch_npu
  import torchair
  import cann_ops_nn

  class Model(torch.nn.Module):
      def forward(self, x):
          return cann_ops_nn.situ_glu(x, dim=-1, beta=1.0)

  model = torch.compile(Model().npu(), backend=torchair.get_npu_backend(), dynamic=False)
  x = torch.randn(2, 32, dtype=torch.float32).npu()
  y = model(x)
  ```

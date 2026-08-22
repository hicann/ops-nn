# clipped_swiglu

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

  带截断的 Swish 门控线性单元（ClippedSwiGLU）激活函数。相较于标准 SwiGLU，新增 `group_index`、`alpha`、`limit`、`bias`、`interleaved`、`clamp_mode` 等参数，用于支持 GPT-OSS 模型使用的变体 SwiGlu、MoE 模型的分组场景以及部分新模型需要将clamp操作后移至silu激活之后的场景。

- 计算公式：

  对给定的输入张量 `x`，其维度为 `[a, b, c, d, e, f, g, ...]`，`clipped_swiglu` 进行以下计算：

  1. 将 `x` 基于输入参数 `dim` 进行合轴，合轴后维度为 `[pre, cut, after]`。其中 `cut` 轴为合轴之后需要切分为两个张量的轴，切分方式分为前后切分或者奇偶切分；`pre`、`after` 可以等于 1。例如当 `dim` 为 3 时，合轴后 `x` 的维度为 `[a*b*c, d, e*f*g*...]`。由于 `after` 轴元素连续存放且计算为逐元素的，将 `cut` 轴与 `after` 轴合并，得到 `x` 的维度为 `[pre, cut]`。

  2. 根据输入参数 `group_index`，对 `x` 的 `pre` 轴进行过滤处理：

     $$
     sum = \text{Sum}(group\_index)
     $$

     $$
     x = x[ : sum, : ]
     $$

     其中 `sum` 表示 `group_index` 所有元素之和。当不输入 `group_index` 时，跳过该步骤。

  3. 根据输入参数 `interleaved`，对 `x` 进行切分：

     当 `interleaved=True`（奇偶切分）：

     $$
     A = x[ : , ::2], \quad B = x[ : , 1::2]
     $$

     当 `interleaved=False`（前后切分）：

     $$
     h = x.shape[1] // 2
     $$

     $$
     A = x[ : , : h], \quad B = x[ : , h : ]
     $$

  4. 根据 `alpha`、`limit`、`bias`、`clamp_mode` 进行变体 SwiGlu 计算：

     当 `clamp_mode=0`（clamp 操作在 silu 之前）：

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

     当 `clamp_mode=1`（clamp 操作在 silu 之后，需底层支持 V2 接口）：

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

  5. 重塑输出张量 `y` 的维度数量与合轴前的 `x` 一致，`dim` 轴上的大小为 `x` 的一半，其他维度与 `x` 相同。

## 函数原型

```python
cann_ops_nn.clipped_swiglu(
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
| `x` | Tensor | 必选 | 公式中的输入 `x`，在 `dim` 对应维度上必须为偶数。 | `torch.float16`、`torch.bfloat16`、`torch.float32` | 1-8 维 |
| `group_index` | Tensor | 可选 | 公式中的 `group_index`。第 `i` 个元素代表第 `i` 组需要处理 `x` 的 batch 数量。传入 `None` 表示不分组。 | `torch.int64` | 1 维，长度不超过 8192，元素需大于等于 0 |
| `dim` | int | 可选 | 对 `x` 进行合轴以及切分的维度序号，取值范围 `[-x.dim(), x.dim()-1]`。默认值 `-1`。 | - | - |
| `alpha` | float | 可选 | 变体 SwiGlu 的缩放参数，建议值 `1.702`。默认值 `1.702`。 | - | - |
| `limit` | float | 可选 | 变体 SwiGlu 的门限值，必须大于 0，建议值 `7.0`。默认值 `7.0`。 | - | - |
| `bias` | float | 可选 | 变体 SwiGlu 的偏差参数，建议值 `1.0`。默认值 `1.0`。 | - | - |
| `interleaved` | bool | 可选 | 切分 `x` 的方式。`True` 表示奇偶切分，`False` 表示前后切分。默认值 `True`。 | - | - |
| `clamp_mode` | int | 可选 | clamp 操作与 silu 操作的先后顺序。`0` 表示 clamp 在 silu 之前，`1` 表示 clamp 在 silu 之后。默认值 `0`。 | - | - |

## 返回值说明

| 参数名 | 参数类型 | 描述 | 数据类型 | 维度(shape) |
| --- | --- | --- | --- | --- |
| `y` | Tensor | ClippedSwiglu 激活结果。 | 与 `x` 相同 | 与 `x` 相同，但 `dim` 对应维度上为 `x.shape[dim] // 2` |

## 约束说明

- 该接口支持单算子模式和 TorchAir 图模式调用。
- `x`、`group_index` 均需为 NPU Tensor；可选 Tensor 可以传 `None`。
- `clamp_mode` 仅支持取值 `0` 或 `1`；当目标芯片未注册 `aclnnClippedSwigluV2` 内核时，会调用 `aclnnClippedSwiglu`，传入 `clamp_mode=1` 会被忽略并按 `clamp_mode=0` 执行。

## 确定性计算

默认支持确定性计算。

## 调用示例

- 单算子模式调用：

  ```python
  import torch
  import torch_npu
  import cann_ops_nn

  x = torch.randn(32, 128, dtype=torch.float16).npu()
  group_index = torch.randint(1, 10, (2, ), dtype=torch.int64).npu()

  y = cann_ops_nn.clipped_swiglu(x, group_index=group_index, dim=-1, alpha=1.702, limit=7.0, bias=1.0, interleaved=True, clamp_mode=1)

  print("y:============", y.shape, y.cpu())
  ```

- 图模式（torchair）调用：

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

# swiglu_backward_group_quant_with_dual_axis

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

  `swiglu_backward_group_quant_with_dual_axis`是融合算子的PyTorch接口，完成带可选Clamp和weight的SwiGLU反向计算，以及沿最后一维和倒数第二维的动态MX量化。底层封装`aclnnSwigluBackwardGroupQuantWithDualAxis`。

- 返回值：

  - 非group场景和不带weight的group场景返回`y1`、`scale1`、`y2`和`scale2`。
  - group场景成对传入`weight`和`y_origin`时，额外返回`grad_weight`。

- 计算约定：

  `x` 的 shape 为 `[T, 2H]`，`grad_y` 的 shape 为 `[T, H]`。`group_index` 采用 cumsum 模式，表示每个 group 的累计行终点。

## 函数原型

```python
cann_ops_nn.swiglu_backward_group_quant_with_dual_axis(
    grad_y,
    x,
    *,
    weight=None,
    y_origin=None,
    group_index=None,
    clamp_limit=-1.0,
    alpha=1.0,
    bias=0.0,
    quant_mode=1,
    dst_type=36,
) -> List[torch.Tensor]
```

## 参数说明

|参数名|参数类型|可选/必选|描述|数据类型|维度(shape)|
|:---|:---|:---|:---|:---|:---|
|`grad_y`|Tensor|必选|SwiGLU 输出的反向梯度。|`torch.float16`、`torch.bfloat16`|`[T, H]`|
|`x`|Tensor|必选|SwiGLU 前向输入，最后一维被均分为两部分。|`torch.float16`、`torch.bfloat16`|`[T, 2H]`|
|`weight`|Tensor|可选|每个 token 的权重，仅 group 场景支持。|`torch.float16`、`torch.bfloat16`、`torch.float32`|`[T]`|
|`y_origin`|Tensor|可选|SwiGLU 前向输出的原始值，与 `weight` 成对传入时用于计算 `grad_weight`。|与 `x` 相同|`[T, H]`|
|`group_index`|Tensor|可选|group 的累计行终点，采用 cumsum 模式。|`torch.int64`|`[G]`|
|`clamp_limit`|float|可选|Clamp 阈值，默认 `-1.0` 表示不启用 Clamp。|—|—|
|`alpha`|float|可选|SwiGLU 反向计算中的 alpha 系数，默认 `1.0`。|—|—|
|`bias`|float|可选|SwiGLU 反向计算中的 bias 系数，默认 `0.0`。|—|—|
|`quant_mode`|int|可选|量化模式，当前仅支持 `1`。|—|—|
|`dst_type`|int|可选|目标 FP8 类型，默认 `36`。|—|—|

### dst_type 编码说明

|`dst_type`|对应类型|输出 Tensor 数据类型|
|:---:|:---|:---|
|`35`|FLOAT8_E5M2|`torch.float8_e5m2`|
|`36`|FLOAT8_E4M3FN|`torch.float8_e4m3fn`|

## 返回值说明

|参数名|参数类型|描述|数据类型|维度(shape)|
|:---|:---|:---|:---|:---|
|`y1`|Tensor|沿最后一维的 MX 量化结果。|`torch.float8_e4m3fn` 或 `torch.float8_e5m2`|与 `x` 相同|
|`scale1`|Tensor|沿最后一维的 E8M0 缩放因子。|`torch.float8_e8m0fnu`|`[T, ceil(2H/64), 2]`|
|`y2`|Tensor|沿倒数第二维的 MX 量化结果。|`torch.float8_e4m3fn` 或 `torch.float8_e5m2`|与 `x` 相同|
|`scale2`|Tensor|沿倒数第二维的 E8M0 缩放因子。|`torch.float8_e8m0fnu`|非 group 为 `[ceil(T/64), 2H, 2]`；group 为 `[floor(T/64)+G, 2H, 2]`|
|`grad_weight`|Tensor|`weight` 的梯度，仅在同时传入 `weight` 和 `y_origin` 时返回。|与 `weight` 相同|`[T]`|

其中，非 group 场景或未传入 `weight` 的 group 场景返回前四个 Tensor；只有传入 `weight` 和 `y_origin` 时返回第五个 Tensor。

## 约束说明

- 该接口支持单算子模式和ACLGraph图模式调用。
- `grad_y` 和 `x` 必须为 NPU Tensor，且分别为二维 `[T, H]` 和 `[T, 2H]`；`x` 的最后一维必须为 64 的整数倍。
- 非 group 场景不传入 `group_index`、`weight` 和 `y_origin`；group 场景必须传入 `group_index`。
- `group_index` 采用 cumsum 模式，必须严格递增且最后一个值等于 `T`。
- `weight` 仅支持 group 场景；`weight` 和 `y_origin` 必须同时传入或同时不传入。
- `quant_mode` 仅支持 `1`；`dst_type` 仅支持 `35` 和 `36`。
- `clamp_limit` 仅支持 `-1.0` 或大于 0 的值，`alpha` 必须大于 0。
- 支持非连续 Tensor，算子原型会将输入转换为连续布局；不支持空 Tensor。
- 不支持 `interleaved`、`dim` 等未定义参数，也不支持 Batch 一致性配置。

## 确定性计算

默认支持确定性计算。

## 调用说明

- 单算子模式调用：

  ```python
  import torch
  import torch_npu
  import cann_ops_nn.ops

  torch.npu.set_device(0)
  grad_y = torch.randn(10240, 3072, dtype=torch.float16).npu()
  x = torch.randn(10240, 6144, dtype=torch.float16).npu()
  group_index = torch.tensor([2560, 5120, 7680, 10240], dtype=torch.int64).npu()
  weight = torch.randn(10240, dtype=torch.float32).npu()
  y_origin = torch.randn_like(grad_y)

  y1, scale1, y2, scale2, grad_weight = torch.ops.cann_ops_nn.swiglu_backward_group_quant_with_dual_axis(
      grad_y,
      x,
      weight=weight,
      y_origin=y_origin,
      group_index=group_index,
      clamp_limit=7.0,
      alpha=1.0,
      bias=0.0,
      quant_mode=1,
      dst_type=36,
  )
  ```

- 图模式调用：

  ```python
  import torch
  import torch_npu
  import cann_ops_nn.ops

  class Model(torch.nn.Module):
      def forward(self, grad_y, x):
          return torch.ops.cann_ops_nn.swiglu_backward_group_quant_with_dual_axis(grad_y, x, quant_mode=1, dst_type=36)

  torch.npu.set_device(0)
  model = torch.compile(
      Model().npu(),
      backend="npugraph_ex",
      dynamic=False,
      fullgraph=True,
      options={"force_eager": True},
  )
  grad_y = torch.randn(128, 32, dtype=torch.float16).npu()
  x = torch.randn(128, 64, dtype=torch.float16).npu()
  y1, scale1, y2, scale2 = model(grad_y, x)
  ```

# swiglu_group_quant_with_dual_axis

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

  融合Clipped-SwiGLU激活、可选逐token `weight`乘法与**两个方向**的MX FP8量化，
  一次调用返回五路结果。底层封装`aclnnSwigluGroupQuantWithDualAxis`。

- 计算流程：

  1. 将`x`的最后一维前后均分为`A`、`B`两部分。`clamp_limit > 0`时令
     `A' = min(A, limit)`、`B' = clamp(B, -limit, limit)`；`clamp_limit = -1.0`时令`A'=A`、`B'=B`。
     计算变体SwiGLU：`F = A' * sigmoid(alpha * A') * (B' + bias)`。
     当`alpha=1.0`、`bias=0.0`且`clamp_limit=-1.0`时退化为标准SwiGLU。
  2. 激活结果先舍入回`x.dtype`，记为`F_T = R_T(F)`。当`weight`非空时，逐token加权并再次舍入：
     `u = R_T(F_T * weight[t])`；否则`u = F_T`。
  3. 第一路沿末维（-1轴）量化：每个token行内每32个元素共享一个尺度，输出`y1`、`mxscale1`。
  4. 第二路沿行方向（-2轴）量化：`group_index`非空时按分组边界，组内每32行、每列共享一个尺度；
     `group_index`为空时按全体行成块，输出`y2`、`mxscale2`。
  5. 当`output_origin=True`时，额外返回乘weight前的激活`y_origin`；否则返回shape为`[0]`的占位Tensor。

> 第一路的量化数值与`swiglu_group_quant(..., quant_mode=5)` **逐字节一致**（两者复用同一MX kernel），
> 因此训练侧（本接口）与推理侧（单轴`quant_mode=5`）的第一路结果对齐。

## 函数原型

```python
cann_ops_nn.swiglu_group_quant_with_dual_axis(
    x,
    weight=None,
    group_index=None,
    *,
    dst_type=292,
    quant_mode=1,
    clamp_limit=-1.0,
    output_origin=False,
    alpha=1.0,
    bias=0.0,
) -> (Tensor, Tensor, Tensor, Tensor, Tensor)
```

## 参数说明

| 参数名 | 参数类型 | 可选/必选 | 描述 | 数据类型 | 维度(shape) |
| --- | --- | --- | --- | --- | --- |
| `x` | Tensor | 必选 | SwiGLU输入，最后一维前后均分为A、B两部分。支持非连续Tensor。 | `torch.float16`、`torch.bfloat16` | 2维`[T, 2H]`，`2H >= 64`且能被64整除 |
| `weight` | Tensor | 可选 | 逐token权重；非空时乘到激活结果上，加权后舍入回`x`的dtype再量化。**仅group场景支持**。支持非连续Tensor。 | `torch.float16`、`torch.bfloat16`、`torch.float32` | 1-8维，元素个数等于`T` |
| `group_index` | Tensor | 可选 | **累积组端索引（cumsum语义，元素值本身即组端行索引）**，仅用于第二路分组量化，不承担输出截断。元素非负、非递减、末元素等于`T`；相邻相等表示空组。不传表示non-group。支持非连续Tensor。 | `torch.int64` | 1维`[G]`，非空 |
| `dst_type` | int | 可选 | 目标量化类型。Torch编码：`291`=`float8_e5m2`、`292`=`float8_e4m3fn`（默认`292`）；实现同时兼容ACLNN/GE编码`35`=`FLOAT8_E5M2`、`36`=`FLOAT8_E4M3FN`。 | - | - |
| `quant_mode` | int | 可选 | 量化模式，**仅支持`1`**（双轴MX FP8）。 | - | - |
| `clamp_limit` | float | 可选 | Clipped-SwiGLU的clamp门限，默认`-1.0`表示不启用；启用时必须为有限正数。 | - | - |
| `output_origin` | bool | 可选 | 是否返回乘weight前的激活`y_origin`，默认`False`。 | - | - |
| `alpha` | float | 可选 | Sigmoid输入缩放系数，默认`1.0`，必须为有限正数。 | - | - |
| `bias` | float | 可选 | 线性分支偏置，默认`0.0`，必须为有限数。 | - | - |

### quant_mode 与 dst_type

本接口仅支持`quant_mode=1`（双轴MX FP8），对应的输出类型如下：

| `quant_mode` | 含义 | `dst_type`支持值 | `y1`/`y2`的torch dtype | `mxscale1`/`mxscale2`的torch dtype |
| --- | --- | --- | --- | --- |
| `1` | 双轴MX FP8 | `35`、`36`、`291`、`292` | `torch.float8_e5m2`或`torch.float8_e4m3fn` | `torch.float8_e8m0fnu` |

#### dst_type 编码说明

| `dst_type` | 对应类型 | 来源 | 下发到GE/ACL的类型 | 适用`quant_mode` |
| --- | --- | --- | --- | --- |
| `291` | `torch_npu.float8_e5m2`，语义同`torch.float8_e5m2` | torch_npu扩展dtype编码 | `DT_FLOAT8_E5M2` / `ACL_FLOAT8_E5M2` | `1` |
| `292` | `torch_npu.float8_e4m3fn`，语义同`torch.float8_e4m3fn` | torch_npu扩展dtype编码 | `DT_FLOAT8_E4M3FN` / `ACL_FLOAT8_E4M3FN` | `1` |
| `35` | `torch_npu.float8_e5m2`，语义同`torch.float8_e5m2` | ACL/GE dtype编码 | `DT_FLOAT8_E5M2` / `ACL_FLOAT8_E5M2` | `1` |
| `36` | `torch_npu.float8_e4m3fn`，语义同`torch.float8_e4m3fn` | ACL/GE dtype编码 | `DT_FLOAT8_E4M3FN` / `ACL_FLOAT8_E4M3FN` | `1` |

说明：`291`/`292`为torch_npu扩展dtype编码，`35`/`36`为ACLNN/GE dtype编码，两者语义一致、可互换使用。与单轴`swiglu_group_quant`不同，本接口不接受PyTorch原生dtype编码（`23`/`24`），也不涉及FP4（`296`/`297`）与HiFloat8（`290`）编码。graph_convert会把上表中的torch dtype编码转换为GE dtype attr，ACLNN路径会把对应编码转换为`aclDataType`后调用底层接口。

## 返回值说明

| 参数名 | 参数类型 | 描述 | 数据类型 | 维度(shape) |
| --- | --- | --- | --- | --- |
| `y1` | Tensor | 第一路（-1轴）量化输出。 | 与`dst_type`一致（`float8_e5m2` / `float8_e4m3fn`） | `[T, H]` |
| `mxscale1` | Tensor | 第一路量化尺度；最后一维最多存放相邻的2个32-element block scale。未使用的padding位置无有效值保证。 | `torch.float8_e8m0fnu` | `[T, ceil(ceil(H/32)/2), 2]` |
| `y2` | Tensor | 第二路（-2轴）量化输出。 | 与`dst_type`一致 | `[T, H]` |
| `mxscale2` | Tensor | 第二路量化尺度；最后一维slot0/slot1最多对应两个相邻的32-row block scale。group场景可能包含为分组边界保留的物理位置；未使用的预留槽位无有效值保证。 | `torch.float8_e8m0fnu` | group：`[floor(T/64) + G, H, 2]`；non-group：`[ceil(T/64), H, 2]` |
| `y_origin` | Tensor | 乘weight前的激活（含clamp/alpha/bias效果）。 | 与`x`一致 | `output_origin=True`时为`[T, H]`，否则为`[0]`占位 |

其中`H = x.shape[-1] / 2`，`G = group_index.numel()`。`mxscale1`/`mxscale2`中未使用的padding或预留位置不属于有效结果，调用方不应依赖其内容。

## 约束说明

- 该接口支持单算子模式和TorchAir图模式调用。
- `x`、`weight`、`group_index`需为NPU Tensor；可选Tensor可以传`None`。
- `x`仅支持二维`[T, 2H]`，`T > 0`，`2H >= 64`且能被64整除。
- `quant_mode`仅支持`1`；`dst_type`支持Torch编码`291`（`float8_e5m2`）、`292`（`float8_e4m3fn`），并兼容ACLNN/GE编码`35`、`36`。
- `weight`仅在提供`group_index`时允许，元素数必须等于`T`；non-group场景传`weight`将被拒绝。
- `group_index`的元素值在设备执行时校验：非法值（负值、递减、越界、末元素不等于`T`）
  会触发设备执行异常，异步调用可能在后续`torch.npu.synchronize()`时才报告失败；
  失败后所有输出均不可使用。参数校验阶段只校验dtype、维数与元素个数，不保证元素值合法。
- `clamp_limit = -1.0`关闭Clamp，否则必须为有限正数；`alpha`必须为有限正数，`bias`必须有限。
- 该接口仅支持前向，不支持autograd。
- 默认支持确定性计算。
- `x`、`weight`、`group_index`均支持非连续Tensor。

## 确定性计算

默认支持确定性计算。

## 调用说明

- 单算子模式调用：

  ```python
  import torch
  import torch_npu
  import cann_ops_nn

  x = torch.randn(8, 512, dtype=torch.float16).npu()
  y1, mx_scale1, y2, mx_scale2, y_origin = cann_ops_nn.swiglu_group_quant_with_dual_axis(
      x,
      dst_type=292,
      clamp_limit=7.0,
      alpha=1.702,
      bias=1.0,
      output_origin=True,
  )
  ```

- group场景（提供累积组端索引与逐token权重）：

  ```python
  import torch
  import torch_npu
  import cann_ops_nn

  x = torch.randn(128, 768, dtype=torch.bfloat16).npu()
  group_index = torch.tensor([32, 64, 96, 128], dtype=torch.int64).npu()
  weight = torch.rand(128, dtype=torch.float32).npu()

  y1, mx_scale1, y2, mx_scale2, y_origin = cann_ops_nn.swiglu_group_quant_with_dual_axis(
      x,
      weight=weight,
      group_index=group_index,
      dst_type=292,
      clamp_limit=7.0,
      alpha=1.702,
      bias=1.0,
  )
  ```

- 图模式（torchair）调用：

  ```python
  import torch
  import torch_npu
  import torchair
  import cann_ops_nn

  class Model(torch.nn.Module):
      def forward(self, x, group_index):
          y1, mx_scale1, y2, mx_scale2, _ = cann_ops_nn.swiglu_group_quant_with_dual_axis(
              x,
              group_index=group_index,
              dst_type=292,
              clamp_limit=7.0,
              alpha=1.702,
              bias=1.0,
          )
          return y1, mx_scale1, y2, mx_scale2

  model = torch.compile(Model().npu(), backend=torchair.get_npu_backend(), dynamic=False)
  x = torch.randn(128, 768, dtype=torch.bfloat16).npu()
  group_index = torch.tensor([32, 64, 96, 128], dtype=torch.int64).npu()
  y1, mx_scale1, y2, mx_scale2 = model(x, group_index)
  ```

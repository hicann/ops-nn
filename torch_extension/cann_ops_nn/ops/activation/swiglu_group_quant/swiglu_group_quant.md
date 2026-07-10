# cann_ops_nn.swiglu_group_quant

## 产品支持情况

- <term>Ascend 950PR/Ascend 950DT</term>：支持
- <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：不支持
- <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：不支持
- <term>Atlas 200I/500 A2 推理产品</term>：不支持
- <term>Atlas 推理系列产品</term>：不支持
- <term>Atlas 训练系列产品</term>：不支持

## 功能说明

- 接口功能：

  对输入 `x` 执行 SwiGLU 激活后进行分组低比特量化，支持 Block FP8、MX FP8、MX FP4、HiFloat8 静态量化和 HiFloat8 动态量化。底层封装 `aclnnSwigluGroupQuant`。

- 计算流程：

  1. 将 `x` 的最后一维均分为 `A`、`B` 两部分，计算 `y_origin = silu(A) * B`。
  2. 当 `weight` 非空时，对 `y_origin` 逐 token 乘以 `weight`。
  3. 按 `quant_mode` 对 `y_origin` 量化，输出 `y` 和 `y_scale`。
  4. 当 `output_origin=True` 时，额外返回量化前的 `y_origin`；否则返回 shape 为 `[0]` 的占位 Tensor。

## 函数原型

```python
torch.ops.cann_ops_nn.swiglu_group_quant(
    x,
    *,
    weight=None,
    group_index=None,
    scale=None,
    dst_type=291,
    quant_mode=0,
    block_size=0,
    round_scale=False,
    clamp_limit=-1.0,
    dst_type_max=15.0,
    output_origin=False,
) -> (Tensor, Tensor, Tensor)
```

## 参数说明

| 参数名 | 参数类型 | 可选/必选 | 描述 | 数据类型 | 维度(shape) |
| --- | --- | --- | --- | --- | --- |
| `x` | Tensor | 必选 | SwiGLU 输入，最后一维会被均分为两部分。 | `torch.float16`、`torch.bfloat16`、`torch.float32` | 1-7维 |
| `weight` | Tensor | 可选 | 逐 token 权重，非空时乘到量化前结果上。 | `torch.float32` | 1维，元素个数等于 `x` 除最后一维外的元素个数 |
| `group_index` | Tensor | 可选 | count 模式分组 token 数。 | `torch.int64` | 1维 |
| `scale` | Tensor | 可选 | HiFloat8 静态量化使用的 scale。 | `torch.float32` | 1维 |
| `dst_type` | int | 可选 | 目标量化类型的 torch dtype 编码，默认 `291`。 | - | - |
| `quant_mode` | int | 可选 | 量化模式，支持 `0`、`1`、`2`、`3`。 | - | - |
| `block_size` | int | 可选 | 量化块大小，`0` 表示使用模式默认值。 | - | - |
| `round_scale` | bool | 可选 | MX 量化是否将 scale 舍入为 2 的幂。 | - | - |
| `clamp_limit` | float | 可选 | SwiGLU 计算前的截断阈值，默认 `-1.0` 表示不启用截断。 | - | - |
| `dst_type_max` | float | 可选 | HiFloat8 动态量化计算 scale 时使用的最大有限值。 | - | - |
| `output_origin` | bool | 可选 | 是否返回量化前的 SwiGLU 结果。 | - | - |

### quant_mode 与 dst_type

| `quant_mode` | 含义 | `dst_type` 支持值 | `y` 的 torch dtype | `y_scale` 的 torch dtype |
| --- | --- | --- | --- | --- |
| `0` | Block FP8 | `23`/`291` 表示 `float8_e5m2`；`24`/`292` 表示 `float8_e4m3fn` | `torch.float8_e5m2` 或 `torch.float8_e4m3fn` | `torch.float32` |
| `1` | MX FP8 / MX FP4 | `23`、`24`、`291`、`292`、`296`、`297` | FP8 使用 torch FP8 dtype；FP4 在 eager/ACL Graph 路径使用 `torch.uint8` 打包存储。TorchAir GE 图模式中，`296` 输出类型为 `torch.float4_e2m1fn_x2`，`297` 输出类型为 `torch.uint8` | `torch.float8_e8m0fnu` |
| `2` | HiFloat8 静态量化 | 建议使用 `290` 表示 HiFloat8；当前 torch 接口实际下发 HIFLOAT8 | `torch.uint8` | `torch.float32`，shape 为 `[0]` |
| `3` | HiFloat8 动态量化 | 建议使用 `290` 表示 HiFloat8；当前 torch 接口实际下发 HIFLOAT8 | `torch.uint8` | `torch.float32` |

#### dst_type 编码说明

| `dst_type` | 对应类型 | 来源 | 下发到 GE/ACL 的类型 | 适用 `quant_mode` |
| --- | --- | --- | --- | --- |
| `23` | `torch.float8_e5m2` | PyTorch 原生 dtype int 值 | `DT_FLOAT8_E5M2` / `ACL_FLOAT8_E5M2` | `0`、`1` |
| `24` | `torch.float8_e4m3fn` | PyTorch 原生 dtype int 值 | `DT_FLOAT8_E4M3FN` / `ACL_FLOAT8_E4M3FN` | `0`、`1` |
| `291` | `torch_npu.float8_e5m2`，语义同 `torch.float8_e5m2` | torch_npu 扩展 dtype 编码 | `DT_FLOAT8_E5M2` / `ACL_FLOAT8_E5M2` | `0`、`1` |
| `292` | `torch_npu.float8_e4m3fn`，语义同 `torch.float8_e4m3fn` | torch_npu 扩展 dtype 编码 | `DT_FLOAT8_E4M3FN` / `ACL_FLOAT8_E4M3FN` | `0`、`1` |
| `290` | `torch_npu.hifloat8` | torch_npu 扩展 dtype 编码 | `DT_HIFLOAT8` / `ACL_HIFLOAT8` | `2`、`3` |
| `296` | `torch_npu.float4_e2m1fn_x2` | torch_npu 扩展 dtype 编码 | `DT_FLOAT4_E2M1` / `ACL_FLOAT4_E2M1` | `1` |
| `297` | `torch_npu.float4_e1m2fn_x2` | torch_npu 扩展 dtype 编码 | `DT_FLOAT4_E1M2` / `ACL_FLOAT4_E1M2` | `1` |

说明：`quant_mode=2/3` 为 HiFloat8 模式，实际下发为 `DT_HIFLOAT8` / `ACL_HIFLOAT8`。graph_convert 会把上表中的 torch dtype 编码转换为 GE dtype attr，ACLNN 路径会把对应编码转换为 `aclDataType` 后调用底层接口。

## 返回值说明

| 参数名 | 参数类型 | 描述 | 数据类型 | 维度(shape) |
| --- | --- | --- | --- | --- |
| `y` | Tensor | 量化输出。 | 见 `quant_mode 与 dst_type` | FP8/HiFloat8 为 `x.shape[:-1] + [D/2]`；FP4 为 `x.shape[:-1] + [ceil((D/2)/2)]`，在 `D` 可被 256 整除时等价于 `x.shape[:-1] + [D/4]` |
| `y_scale` | Tensor | 量化 scale 输出。 | 见 `quant_mode 与 dst_type` | `quant_mode=0` 为 `x.shape[:-1] + [ceil((D/2)/128)]`；`quant_mode=1` 为 `x.shape[:-1] + [ceil(ceil((D/2)/32)/2), 2]`；`quant_mode=2` 为 `[0]`；`quant_mode=3` 为 `group_index.shape` 或 `[1]` |
| `y_origin` | Tensor | 量化前 SwiGLU 结果或占位 Tensor。 | 与 `x` 相同 | `output_origin=True` 时为 `x.shape[:-1] + [D/2]`，否则为 `[0]` |

其中 `D = x.shape[-1]`。

## 约束说明

- 该接口支持单算子模式和 TorchAir 图模式调用。
- `x`、`weight`、`group_index`、`scale` 均需为 NPU Tensor；可选 Tensor 可以传 `None`。
- 输入 `x` 的 rank 必须大于 0，最后一维 `D` 必须大于等于 256 且能被 256 整除。
- `dst_type` 支持 FP8、FP4 和 HiFloat8 对应的 torch dtype 编码，详见 `dst_type 编码说明`。
- `quant_mode=0` 时仅支持 FP8 输出，`dst_type` 支持 `23`、`24`、`291`、`292`，`block_size` 支持 `0` 或 `128`。
- `quant_mode=1` 时支持 FP8/FP4 输出，`dst_type` 支持 `23`、`24`、`291`、`292`、`296`、`297`，`block_size` 支持 `0` 或 `32`，`round_scale` 必须为 `True`。
- `quant_mode=2` 时支持 HiFloat8 静态量化输出，需传入 `scale`，`dst_type`、`block_size` 和 `round_scale` 不生效，实际下发 HiFloat8。
- `quant_mode=3` 时支持 HiFloat8 动态量化输出，不使用 `scale`，`dst_type`、`block_size` 和 `round_scale` 不生效，实际下发 HiFloat8。
- `dst_type` 为 `296` 或 `297`，即 `FLOAT4_E2M1` 或 `FLOAT4_E1M2` 时，必须使用 `quant_mode=1`。
- `y_scale` 的数据类型必须与 `quant_mode` 匹配：Block FP8 为 `torch.float32`，MX 为 `torch.float8_e8m0fnu`，HiFloat8 为 `torch.float32`。
- `quant_mode=3` 时，`group_index` 可用于 MoE 场景的分组动态量化；`y_scale` 的 shape 为 `group_index.shape`，未传 `group_index` 时为 `[1]`。
- `clamp_limit` 不启用时使用默认占位值 `-1.0`；启用时必须大于 0。
- quant_mode=0和quant_mode=1时，output_origin仅支持False。
- 不支持空 Tensor 和非连续 Tensor。

## 确定性计算

默认支持确定性计算。

## 调用说明

- 单算子模式调用：

  ```python
  import torch
  import torch_npu
  import cann_ops_nn.ops

  x = torch.randn(8, 512, dtype=torch.float16).npu()
  y, y_scale, y_origin = torch.ops.cann_ops_nn.swiglu_group_quant(
      x,
      dst_type=291,
      quant_mode=0,
      block_size=128,
  )
  ```

  MX FP4 模式：

  ```python
  y, y_scale, y_origin = torch.ops.cann_ops_nn.swiglu_group_quant(
      x,
      dst_type=296,
      quant_mode=1,
      block_size=32,
      round_scale=True,
  )
  ```

- 图模式（torchair）调用：

  ```python
  import torch
  import torch_npu
  import torchair
  import cann_ops_nn.ops

  class Model(torch.nn.Module):
      def forward(self, x):
          y, y_scale, _ = torch.ops.cann_ops_nn.swiglu_group_quant(
              x,
              dst_type=291,
              quant_mode=0,
              block_size=128,
          )
          return y, y_scale

  model = torch.compile(Model().npu(), backend=torchair.get_npu_backend(), dynamic=False)
  x = torch.randn(8, 512, dtype=torch.float16).npu()
  y, y_scale = model(x)
  ```

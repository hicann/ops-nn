# swiglu_group_quant

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

  对输入`x`执行SwiGLU激活后进行分组低比特量化，支持Block FP8、MX FP8、HiFloat8静态量化和HiFloat8动态量化。`quant_mode=5`底层封装`aclnnSwigluGroupQuantV2`，其余模式封装`aclnnSwigluGroupQuant`。

- 计算流程：

  1. 将`x`的最后一维均分为`A`、`B`两部分，计算SwiGLU结果；mode 5使用`A * sigmoid(alpha * A) * (B + bias)`。
  2. 当`weight`非空时，对SwiGLU结果逐token乘以`weight`，得到量化输入。
  3. 按`quant_mode`对加权后的结果量化，输出`y`和`y_scale`。
  4. 当`output_origin=True`时，额外返回weight加权前的SwiGLU结果`y_origin`；否则返回shape为`[0]`的占位Tensor。

## 函数原型

```python
cann_ops_nn.swiglu_group_quant(
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
    alpha=1.0,
    bias=0.0,
) -> (Tensor, Tensor, Tensor)
```

## 参数说明

| 参数名 | 参数类型 | 可选/必选 | 描述 | 数据类型 | 维度(shape) |
| --- | --- | --- | --- | --- | --- |
| `x` | Tensor | 必选 | SwiGLU输入，最后一维会被均分为两部分。 | `torch.float16`、`torch.bfloat16`（quant_mode为2/3时额外支持`torch.float32`） | quant_mode=5仅支持2维；其他模式为2-8维（quant_mode为1时为2-7维） |
| `weight` | Tensor | 可选 | 逐token权重，非空时乘到SwiGLU结果上，加权后的结果用于量化。 | quant_mode=5支持`torch.float16`、`torch.bfloat16`、`torch.float32`，其他模式仅支持`torch.float32` | 1-8维，元素个数等于`x`除最后一维外的元素个数 |
| `group_index` | Tensor | 可选 | count模式分组token数。 | `torch.int64` | 1维 |
| `scale` | Tensor | 可选 | HiFloat8静态量化使用的scale，仅quant mode为2时使用。 | `torch.float32` | 1维 |
| `dst_type` | int | 可选 | 目标量化类型的torch dtype编码，默认`291`。 | - | - |
| `quant_mode` | int | 可选 | 量化模式，支持`0`、`1`、`2`、`3`、`5`。其中`1`为原MX公式，`5`为MX V2公式。 | - | - |
| `block_size` | int | 可选 | 量化块大小，`0`表示使用模式默认值。 | - | - |
| `round_scale` | bool | 可选 | MX量化是否将scale舍入为2的幂。 | - | - |
| `clamp_limit` | float | 可选 | SwiGLU计算前的截断阈值，默认`-1.0`表示不启用截断。 | - | - |
| `dst_type_max` | float | 可选 | HiFloat8动态量化计算scale时使用的最大有限值。 | - | - |
| `output_origin` | bool | 可选 | 是否返回weight加权前的SwiGLU结果。 | - | - |
| `alpha` | float | 可选 | Sigmoid输入的缩放系数，默认值为`1.0`，仅`quant_mode=5`生效。 | - | - |
| `bias` | float | 可选 | SwiGLU第二分支的加法偏置，默认值为`0.0`，仅`quant_mode=5`生效。 | - | - |

### quant_mode 与 dst_type

| `quant_mode` | 含义 | `dst_type`支持值 | `y`的torch dtype | `y_scale`的torch dtype |
| --- | --- | --- | --- | --- |
| `0` | Block FP8 | `23`/`291`表示`float8_e5m2`；`24`/`292`表示`float8_e4m3fn` | `torch.float8_e5m2`或`torch.float8_e4m3fn` | `torch.float32` |
| `1` | MX FP8 | `23`、`24`、`291`、`292` | `torch.float8_e5m2`或`torch.float8_e4m3fn` | `torch.float8_e8m0fnu` |
| `2` | HiFloat8静态量化 | `290`表示HiFloat8 | `torch.uint8` | `torch.float32`，shape为`[0]` |
| `3` | HiFloat8动态量化 |  `290`表示HiFloat8 | `torch.uint8` | `torch.float32` |
| `5` | MX V2 FP8 | `23`、`24`、`291`、`292` | `torch.float8_e5m2`或`torch.float8_e4m3fn` | `torch.float8_e8m0fnu` |

#### dst_type 编码说明

| `dst_type` | 对应类型 | 来源 | 下发到GE/ACL的类型 | 适用`quant_mode` |
| --- | --- | --- | --- | --- |
| `23` | `torch.float8_e5m2` | PyTorch原生dtype int值 | `DT_FLOAT8_E5M2` / `ACL_FLOAT8_E5M2` | `0`、`1`、`5` |
| `24` | `torch.float8_e4m3fn` | PyTorch原生dtype int值 | `DT_FLOAT8_E4M3FN` / `ACL_FLOAT8_E4M3FN` | `0`、`1`、`5` |
| `291` | `torch_npu.float8_e5m2`，语义同`torch.float8_e5m2` | torch_npu扩展dtype编码 | `DT_FLOAT8_E5M2` / `ACL_FLOAT8_E5M2` | `0`、`1`、`5` |
| `292` | `torch_npu.float8_e4m3fn`，语义同`torch.float8_e4m3fn` | torch_npu扩展dtype编码 | `DT_FLOAT8_E4M3FN` / `ACL_FLOAT8_E4M3FN` | `0`、`1`、`5` |
| `290` | `torch_npu.hifloat8` | torch_npu扩展dtype编码 | `DT_HIFLOAT8` / `ACL_HIFLOAT8` | `2`、`3` |

说明：`quant_mode=2/3`为HiFloat8模式，实际下发为`DT_HIFLOAT8`/`ACL_HIFLOAT8`。graph_convert会把上表中的torch dtype编码转换为GE dtype attr，ACLNN路径会把对应编码转换为`aclDataType`后调用底层接口。

## 返回值说明

| 参数名 | 参数类型 | 描述 | 数据类型 | 维度(shape) |
| --- | --- | --- | --- | --- |
| `y` | Tensor | 量化输出。 | 参见`quant_mode 与 dst_type` | `x.shape[:-1] + [D/2]` |
| `y_scale` | Tensor | 量化scale输出。 | 参见`quant_mode 与 dst_type` | `quant_mode=0`为`x.shape[:-1] + [ceil((D/2)/128)]`；`quant_mode=1/5`为`x.shape[:-1] + [ceil(ceil((D/2)/32)/2), 2]`；`quant_mode=2`为`[0]`；`quant_mode=3`为`group_index.shape`或`[1]` |
| `y_origin` | Tensor | weight加权前的SwiGLU结果或占位Tensor。 | 与`x`相同 | `output_origin=True`时为`x.shape[:-1] + [D/2]`，否则为`[0]` |

其中`D = x.shape[-1]`。

## 约束说明

- 该接口支持单算子模式和TorchAir图模式调用。
- `x`、`weight`、`group_index`、`scale`均需为NPU Tensor；可选Tensor可以传`None`。
- quant_mode=5时输入`x`仅支持二维`[T,D]`，高于二维直接报错；其他模式为2-8维（quant_mode为1时为2-7维）。quant_mode=5时最后一维`D`必须大于等于64且能被64整除；其他模式仍要求`D`大于等于256且能被256整除。
- `dst_type`支持FP8和HiFloat8对应的torch dtype编码，详见`dst_type 编码说明`。
- `quant_mode=0`时仅支持FP8输出，`dst_type`支持`23`、`24`、`291`、`292`，`block_size`支持`0`或`128`。
- `quant_mode=1`时支持FP8输出，`dst_type`支持`23`、`24`、`291`、`292`，`block_size`支持`0`或`32`，`round_scale`必须为`True`。
- `quant_mode=5`时使用MX V2量化公式，仅支持FP8输出；`block_size`支持`0`或`32`，`round_scale`必须为`True`，不支持`group_index`和`scale`输入。
- `quant_mode=2`时支持HiFloat8静态量化输出，需传入`scale`，`dst_type`、`block_size`和`round_scale`不生效，实际下发HiFloat8。
- `quant_mode=3`时支持HiFloat8动态量化输出，不使用`scale`，`dst_type`、`block_size`和`round_scale`不生效，实际下发HiFloat8。
- `y_scale`的数据类型必须与`quant_mode`匹配：Block FP8为`torch.float32`，MX为`torch.float8_e8m0fnu`，HiFloat8为`torch.float32`。
- `quant_mode=3`时，`group_index`可用于MoE场景的分组动态量化；`y_scale`的shape为`group_index.shape`，未传`group_index`时为`[1]`。
- `clamp_limit`不启用时使用默认占位值`-1.0`；启用时必须大于0。`quant_mode=5`时还必须为有限值。
- 所有支持的`quant_mode`均可通过`output_origin=True`输出weight加权前的`y_origin`。
- group_index中的元素值须大于等于0。
- quant_mode为0、1时，x、weight支持为空Tensor，group_index不支持单独为空Tensor；quant_mode为2、3、5时不支持空Tensor。
- quant_mode=5的Torch接口支持非连续`x`和`weight`；其他模式不支持非连续Tensor。

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

# cla_gate_quant

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

  CLA（Cross-Layer Attention）两路 head-wise gate 加权合并与双轴动态块量化的融合算子，
  基于 Torch Extension 方式注册，底层封装 `aclnnClaGateQuant`。
  先对 Global/CLA 分支与 Local/SWA 分支的 Attention 输出分别施加 Sigmoid 门控并加权合并，
  再将合并结果 reshape 为 `[T, K]`（`K = N × D`），在 K 方向（`[1,32]` block，row-wise）
  与 T 方向（`[32,1]` block，col-wise）做基于块的动态量化，输出低精度 FP8/FP4 张量与
  对应的 E8M0 缩放因子。

- 计算流程：

  1. 两路 head-wise Sigmoid 门控：`s_g = sigmoid(global_gate_logits)`，`s_l = sigmoid(local_gate_logits)`；
  2. gate 融合：`merged = s_g * global_attn + s_l * local_attn`；
  3. 量化前逻辑矩阵：`X = reshape(merged, [T, K])`，`K = N × D`；
  4. row-wise 量化（K 方向，`[1,32]` block）得到 `row_data` 与 `row_scale`；
  5. `dual_axis_flag=True` 时额外做 col-wise 量化（T 方向，`[32,1]` block）得到
     `col_data` 与 `col_scale`；`dual_axis_flag=False` 时 `col_data`/`col_scale` 返回空 Tensor。

## 函数原型

```python
cann_ops_nn.cla_gate_quant(
    global_attn,
    local_attn,
    global_gate_logits,
    local_gate_logits,
    *,
    dst_type=None,
    round_mode="rint",
    scale_alg=1,
    input_attn_layout="TND",
    dual_axis_flag=False,
) -> (Tensor, Tensor, Tensor, Tensor)
```

## 参数说明

| 参数名 | 参数类型 | 可选/必选 | 描述 | 数据类型 | 维度(shape) |
| --- | --- | --- | --- | --- | --- |
| `global_attn` | Tensor | 必选 | Global/CLA 分支 Attention 输出 `O_g`。 | `torch.float16`、`torch.bfloat16` | 3 维 `[T, N, D]` |
| `local_attn` | Tensor | 必选 | Local/SWA 分支 Attention 输出 `O_l`，shape 与 `global_attn` 一致。 | `torch.float16`、`torch.bfloat16` | 3 维 `[T, N, D]` |
| `global_gate_logits` | Tensor | 必选 | Global gate 的 Sigmoid 前值 `z_g`，dtype 与 `global_attn` 一致。 | `torch.float16`、`torch.bfloat16` | 2 维 `[T, N]` |
| `local_gate_logits` | Tensor | 必选 | Local gate 的 Sigmoid 前值 `z_l`，shape 与 `global_gate_logits` 一致。 | `torch.float16`、`torch.bfloat16` | 2 维 `[T, N]` |
| `dst_type` | int | 可选 | 目标量化类型的 torch dtype 编码，见下方 `dst_type 编码说明`；传 `None` 等价于 `24`(`torch.float8_e4m3fn`)。支持传入torch.dtype类型。 | - | - |
| `round_mode` | str | 可选 | 舍入模式。FP8 仅支持 `"rint"`；FP4 支持 `"rint"`、`"round"`、`"floor"`。默认 `"rint"`。 | - | - |
| `scale_alg` | int | 可选 | scale 计算方法，`1` 为 cuBLAS、`0` 为 OCP；FP4 仅支持 `0`。默认 `1`。 | - | - |
| `input_attn_layout` | str | 可选 | 输入 `global_attn`/`local_attn` 的排布格式，当前仅支持 `"TND"`。默认 `"TND"`。 | - | - |
| `dual_axis_flag` | bool | 可选 | `True` 为双轴量化（同时输出 row/col 两套结果）；`False` 为单轴量化（仅输出 row-wise，col 输出为空 Tensor）。默认 `False`。 | - | - |

### dst_type 编码说明

| `dst_type` | 对应类型 | 来源 | 下发到 GE/ACL 的类型 |
| --- | --- | --- | --- |
| `23` | `torch.float8_e5m2` | PyTorch 原生 dtype int 值 | `DT_FLOAT8_E5M2` / `ACL_FLOAT8_E5M2`(35) |
| `24` | `torch.float8_e4m3fn` | PyTorch 原生 dtype int 值 | `DT_FLOAT8_E4M3FN` / `ACL_FLOAT8_E4M3FN`(36) |
| `296` | `torch_npu.float4_e2m1fn_x2` | torch_npu 扩展 dtype 编码 | `DT_FLOAT4_E2M1` / `ACL_FLOAT4_E2M1`(40) |
| `297` | `torch_npu.float4_e1m2fn_x2` | torch_npu 扩展 dtype 编码 | `DT_FLOAT4_E1M2` / `ACL_FLOAT4_E1M2`(41) |

说明：`None` 归一化为 `24`（`torch.float8_e4m3fn`）。图模式（TorchAir）路径由
`convert_cla_gate_quant` 把上表的 torch dtype 编码转换为 GE dtype attr，ACLNN 路径转换为
对应 `aclDataType` 后调用底层接口。

## 返回值说明

| 参数名 | 参数类型 | 描述 | 数据类型 | 维度(shape) |
| --- | --- | --- | --- | --- |
| `row_data` | Tensor | Row-wise（K 方向）量化数据 `P_i`。 | FP8 为 `torch.float8_e5m2`/`torch.float8_e4m3fn`；FP4 在 torch 侧以 `torch.uint8` 打包存储（2 个 FP4 值占 1 字节） | FP8 为 `[T, K]`；FP4 为 `[T, K/2]` |
| `row_scale` | Tensor | Row-wise 每个 `[1,32]` 分组的 E8M0 缩放因子，两两一组存最后一维，偶数 pad 补 0。 | `torch.uint8`（E8M0 原始值） | `[T, ceil(K/64), 2]` |
| `col_data` | Tensor | Col-wise（T 方向）量化数据 `P_j`；`dual_axis_flag=False` 时为空 Tensor。 | 同 `row_data` | 双轴：同 `row_data`；单轴：`[0]` |
| `col_scale` | Tensor | Col-wise 每个 `[32,1]` 分组的 E8M0 缩放因子，偶数 pad 补 0；`dual_axis_flag=False` 时为空 Tensor。 | `torch.uint8`（E8M0 原始值） | 双轴：`[ceil(T/64), K, 2]`；单轴：`[0]` |

其中 `T = global_attn.shape[0]`，`N = global_attn.shape[1]`，`D = global_attn.shape[2]`，`K = N × D`。

## 约束说明

- 该接口支持单算子模式和 TorchAir 图模式调用。
- `global_attn`、`local_attn`、`global_gate_logits`、`local_gate_logits` 均需为 NPU Tensor。
- `global_attn`/`local_attn` 必须为 3 维张量且 shape 一致，为 `[T, N, D]`；`N ∈ [1,128]`，
  `D ∈ {128, 256}`。
- `global_gate_logits`/`local_gate_logits` 必须为 2 维张量 `[T, N]`，dtype 与 `global_attn`/`local_attn` 一致。
- 当 `dst_type` 为 FP4（`296`/`297`）时，`K = N × D` 必须可被 4 整除，且 `scale_alg` 必须为 `0`。
- FP8 输出类型（`23`/`24`）仅支持 `round_mode="rint"`。
- `dual_axis_flag=False` 时 `col_data`/`col_scale` 为空 Tensor，底层接口不访问该输出；
  row 输出与双轴模式逐比特一致。
- `round_mode` 必须是 `"rint"`、`"round"`、`"floor"` 之一；`scale_alg` 必须是 `0` 或 `1`。
- `input_attn_layout` 仅支持 `"TND"`。

## 确定性/Batch一致性

- 确定性计算
  - 默认支持确定性计算：分核方式固定、核内归约顺序与量化块划分固定，同输入同输出。

- Batch一致性说明：
  - <term>Ascend 950PR/Ascend 950DT</term>：单轴默认Batch一致性实现，双轴在T轴=64倍数场景中默认支持，否则不支持。

## 调用说明

- 单算子模式调用（双轴 FP8 量化）：

  ```python
  import torch
  import torch_npu
  import cann_ops_nn

  t, n, d = 8, 24, 256
  global_attn = torch.randn(t, n, d, dtype=torch.bfloat16).npu()
  local_attn = torch.randn(t, n, d, dtype=torch.bfloat16).npu()
  global_gate_logits = torch.randn(t, n, dtype=torch.bfloat16).npu()
  local_gate_logits = torch.randn(t, n, dtype=torch.bfloat16).npu()

  row_data, row_scale, col_data, col_scale = cann_ops_nn.cla_gate_quant(
      global_attn,
      local_attn,
      global_gate_logits,
      local_gate_logits,
      dst_type=24,          # torch.float8_e4m3fn
      round_mode="rint",
      scale_alg=1,
      input_attn_layout="TND",
      dual_axis_flag=True,
  )
  ```

- 单轴 FP4 量化（`col_data`/`col_scale` 为空 Tensor）：

  ```python
  row_data, row_scale, col_data, col_scale = cann_ops_nn.cla_gate_quant(
      global_attn,
      local_attn,
      global_gate_logits,
      local_gate_logits,
      dst_type=296,         # torch_npu.float4_e2m1fn_x2
      round_mode="rint",
      scale_alg=0,
      input_attn_layout="TND",
      dual_axis_flag=False,
  )
  ```

- 图模式（TorchAir）调用：

  ```python
  import torch
  import torch_npu
  import torchair
  import cann_ops_nn

  class Model(torch.nn.Module):
      def forward(self, global_attn, local_attn, global_gate_logits, local_gate_logits):
          row_data, row_scale, _, _ = cann_ops_nn.cla_gate_quant(
              global_attn,
              local_attn,
              global_gate_logits,
              local_gate_logits,
              dst_type=24,
              dual_axis_flag=False,
          )
          return row_data, row_scale

  model = torch.compile(Model().npu(), backend=torchair.get_npu_backend(), dynamic=False)
  model(global_attn, local_attn, global_gate_logits, local_gate_logits)
  ```

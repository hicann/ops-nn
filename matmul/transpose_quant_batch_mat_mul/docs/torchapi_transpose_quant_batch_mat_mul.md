# transpose_quant_batch_mat_mul

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

- 接口功能：完成张量x1与张量x2的量化矩阵乘计算，底层封装aclnnTransposeQuantBatchMatMul；当x2为FRACTAL_NZ格式时，封装aclnnTransposeQuantBatchMatMulWeightNz（仅MXFP8场景）。该接口支持三种量化模式：**MX量化**（MXFP8与MXFP4）、**FP8量化**（K-C场景）、**HIFP8量化**。
- 三种量化模式说明：

  | 量化模式 | x1/x2数据类型 | x1_scale | x2_scale | K/N约束 |
  | --- | --- | --- | --- | --- |
  | MX | torch.float8_e4m3fn（MXFP8）或torch.uint8（MXFP4，x1_dtype/x2_dtype指定为float4_e2m1fn_x2） | torch.float8_e8m0fnu，4维 | torch.float8_e8m0fnu，4维 | K为64的倍数，N无限制 |
  | FP8（K-C） | torch.float8_e4m3fn与torch.float8_e5m2（可任意组合） | torch.float32，1维(M,) | torch.float32，1维(N,) | K=512，N=128 |
  | HIFP8 | torch.uint8（hifloat8存储，x1_dtype/x2_dtype指定为torch_npu.hifloat8） | 不传（None） | torch.uint64或torch.int64（按位解释，低32位为float32位模式），1维(N,)，per-channel FIXPIPE反量化缩放因子 | K、N无限制，均为正数 |

- 计算公式（以batch维的每个切片为例）：

  - MX量化：

    $$
    y[m, n] = \sum_{j=0}^{K/32-1} \left(\left(\sum_{k=0}^{31} x1[m, j \times 32 + k] \cdot x2[j \times 32 + k, n]\right) \cdot x1Scale[m, j] \cdot x2Scale[j, n]\right)
    $$

    其中K为矩阵乘的K轴长度，x1Scale、x2Scale为torch.float8_e8m0fnu编码的MX量化缩放因子，矩阵乘中间结果按K轴每32个元素一组进行缩放累加。

  - FP8（K-C）量化：

    $$
    y[m, n] = \left(\sum_{k=0}^{K-1} x1[m, k] \cdot x2[k, n]\right) \cdot x1Scale[m] \cdot x2Scale[n]
    $$

  - HIFP8量化：

    $$
    y[m, n] = \left(\sum_{k=0}^{K-1} x1[m, k] \cdot x2[k, n]\right) \cdot x2Scale[n]
    $$

    其中x1、x2为HIFLOAT8数据，x2Scale为FIXPIPE per-channel 反量化缩放因子。

- 示例：假设x1的shape是(M, B, K)，x2的shape是(B, K, N)，输出y的shape是(M, B, N)。

## 函数原型

```python
cann_ops_nn.transpose_quant_batch_mat_mul(
    x1,
    x2,
    *,
    dtype,
    bias=None,
    x1_scale=None,
    x2_scale=None,
    group_sizes=None,
    perm_x1=None,
    perm_x2=None,
    perm_y=None,
    batch_split_factor=None,
    x1_dtype=None,
    x2_dtype=None,
    x1_scale_dtype=None,
    x2_scale_dtype=None,
) -> Tensor
```

## 参数说明

| 参数名 | 参数类型 | 可选/必选 | 描述 | 数据类型 | 维度(shape) |
| --- | --- | --- | --- | --- | --- |
| x1 | Tensor | 必选 | 矩阵乘运算中的左矩阵，shape为(M, B, K)。MXFP4场景Tensor最后一维为FP4拼包后的物理长度K/2；HIFP8场景存储类型为torch.uint8，通过x1_dtype指定为HIFLOAT8。 | torch.float8_e4m3fn/torch.float8_e5m2；MXFP4、HIFP8场景为实际存储类型（如torch.uint8），并通过x1_dtype指定逻辑类型 | 3维，(M, B, K) |
| x2 | Tensor | 必选 | 矩阵乘运算中的右矩阵，数据类型与x1一致，K轴长度与x1一致。perm_x2为[0, 1, 2]时shape为(B, K, N)；perm_x2为[0, 2, 1]时shape为(B, N, K)。MXFP4场景Tensor最后一维为FP4拼包后的物理长度（N/2或K/2）。 | 同x1 | 3维 |
| dtype | int | 必选 | 输出y的数据类型枚举值：1表示torch.float16，27表示torch.bfloat16，34表示HIFLOAT8（仅T-C量化支持，torch侧以torch.uint8承载）。 | int64 | - |
| bias | Tensor | 可选 | 矩阵乘运算后累加的偏置。预留参数，当前暂不支持，必须传入None。 | - | - |
| x1_scale | Tensor | 可选 | x1的量化缩放因子。MX场景必选；FP8（K-C）场景必选，为per-token缩放因子；HIFP8场景不参与计算，传入None。 | MX：torch.float8_e8m0fnu；FP8：torch.float32 | MX：4维，(M, B, K/64, 2)；FP8：1维，(M,) |
| x2_scale | Tensor | 可选 | x2的量化缩放因子。MX场景必选；FP8（K-C）场景必选，为per-channel缩放因子；HIFP8场景必选，为per-channel FIXPIPE缩放因子。 | MX：torch.float8_e8m0fnu；FP8：torch.float32；HIFP8：torch.uint64或torch.int64（按位解释为uint64） | MX：4维，perm_x2为[0, 1, 2]时为(B, K/64, N, 2)，perm_x2为[0, 2, 1]时为(B, N, K/64, 2)；FP8：1维，(N,)；HIFP8：1维，(N,) |
| group_sizes | List[int] | 可选 | 量化分组大小[groupSizeM, groupSizeN, groupSizeK]，每个元素取值范围为[0, 65535]。MX量化场景groupSizeM和groupSizeN仅支持0或1（取值为0时由接口根据scale的shape推断），groupSizeK仅支持32；FP8（K-C）场景固定传[0, 0, 0]；HIFP8场景传[0, 0, 0]。 | int64 | - |
| perm_x1 | List[int] | 可选 | x1的转置序列，仅支持[1, 0, 2]，默认值[1, 0, 2]。 | int64 | - |
| perm_x2 | List[int] | 可选 | x2的转置序列，支持[0, 1, 2]和[0, 2, 1]，默认值[0, 1, 2]。FP8（K-C）量化模式仅支持[0, 1, 2]。 | int64 | - |
| perm_y | List[int] | 可选 | 输出矩阵的转置序列，仅支持[1, 0, 2]，默认值[1, 0, 2]。 | int64 | - |
| batch_split_factor | int | 可选 | 输出矩阵B维的切分大小，当前仅支持取值1，默认值1。 | int64 | - |
| x1_dtype | int | 可选 | x1的数据类型枚举值，不传入时根据x1的数据类型自动推导。MXFP4场景必须传入该参数指定为torch_npu.float4_e2m1fn_x2；HIFP8场景必须传入该参数指定为torch_npu.hifloat8。 | int64 | - |
| x2_dtype | int | 可选 | x2的数据类型枚举值，不传入时根据x2的数据类型自动推导。MXFP4场景必须传入该参数指定为torch_npu.float4_e2m1fn_x2；HIFP8场景必须传入该参数指定为torch_npu.hifloat8。 | int64 | - |
| x1_scale_dtype | int | 可选 | x1_scale的数据类型枚举值，不传入时根据x1_scale的数据类型自动推导。 | int64 | - |
| x2_scale_dtype | int | 可选 | x2_scale的数据类型枚举值，不传入时根据x2_scale的数据类型自动推导。 | int64 | - |

## 返回值说明

| 输出名 | 输出类型 | 描述 | 数据类型 | 维度(shape) |
| --- | --- | --- | --- | --- |
| y | Tensor | 量化矩阵乘的输出。dtype为34时输出HIFLOAT8（仅T-C量化支持），结果以torch.uint8承载，需用户自行按HIFLOAT8解析。 | dtype为1时为torch.float16，为27时为torch.bfloat16，为34时为torch.uint8 | (M, B, N) |

## 约束说明

- 该接口当前支持单算子模式调用。
- x1、x2必须是NPU Tensor；x1_scale、x2_scale按量化模式要求传入（见下表）。
- 仅支持3维Tensor：x1的shape为(M, B, K)，x2的shape为(B, K, N)或(B, N, K)，x1与x2的batch轴和K轴必须一致，不支持batch轴广播。
- perm_x1仅支持[1, 0, 2]；perm_y仅支持[1, 0, 2]；perm_x2支持[0, 1, 2]和[0, 2, 1]（FP8（K-C）量化模式仅支持[0, 1, 2]）。
- batch_split_factor当前仅支持取值1。
- bias为预留参数，当前暂不支持。
- 不支持空Tensor。
- dtype为34时表示HIFLOAT8输出（仅T-C量化支持），torch侧以torch.uint8承载，需用户自行按HIFLOAT8解析。

各量化模式的差异化约束：

- **MX量化模式**：
  - x1与x2的数据类型必须一致：MXFP8场景为torch.float8_e4m3fn；MXFP4场景用torch.uint8表示。
  - x1_scale与x2_scale的数据类型必须为torch.float8_e8m0fnu。
  - K仅支持64的倍数。
  - x1_scale的shape必须为(M, B, K/64, 2)；x2_scale的shape在perm_x2为[0, 1, 2]时必须为(B, K/64, N, 2)，在perm_x2为[0, 2, 1]时必须为(B, N, K/64, 2)，最后一维必须为2。
  - group_sizes必须显式传入，且groupSizeM、groupSizeN取值为0或1，groupSizeK取值为32。
  - MXFP4场景数据按两个FLOAT4_E2M1拼包存储（Tensor最后一维为物理长度，即逻辑长度的一半）：x1的最后一维为K/2；x2在perm_x2为[0, 1, 2]时最后一维为N/2，在perm_x2为[0, 2, 1]时最后一维为K/2。此时Tensor实际存储类型（如torch.uint8）无法自动推导出FP4类型，必须通过x1_dtype、x2_dtype指定为torch_npu.float4_e2m1fn_x2。
  - 仅x2支持FRACTAL_NZ格式（仅MXFP8）。
- **FP8（K-C）量化模式**：
  - x1与x2的数据类型支持torch.float8_e4m3fn与torch.float8_e5m2，可任意组合。
  - x1_scale的数据类型必须为torch.float32，shape为(M,)；x2_scale的数据类型必须为torch.float32，shape为(N,)。
  - K仅支持512，N仅支持128。
  - group_sizes传[0, 0, 0]。
- **HIFP8量化模式**：
  - x1与x2的存储类型为torch.uint8，必须通过x1_dtype、x2_dtype指定为torch_npu.hifloat8。
  - x2_scale的数据类型必须为torch.uint64或torch.int64，shape为(N,)；两者均按64位**位模式**解释，int64不是数值语义。其每个元素的**低32位存放torch.float32的位模式**（1.0编码为0x000000003F800000），高32位无意义。不要传入数值语义的整数（例如scale=2就传2），否则2会被按位重解释为float32的denormal位模式，导致结果错误且无告警。
  - x1_scale传入None，该参数不参与计算。
  - K、N无特殊对齐约束（均为正数即可），perm_x2支持[0, 1, 2]和[0, 2, 1]。
  - group_sizes传[0, 0, 0]。

## 确定性计算

默认支持确定性计算。

## 调用示例

- 单算子模式调用（eager）

  - MXFP8场景示例：

    ```python
    import torch
    import torch_npu
    import cann_ops_nn

    M, B, K, N = 64, 16, 128, 256
    x1 = torch.randn(M, B, K).to(torch.float8_e4m3fn).npu()
    x2 = torch.randn(B, K, N).to(torch.float8_e4m3fn).npu()
    x1_scale = torch.ones(M, B, K // 64, 2, dtype=torch.float8_e8m0fnu).npu()
    x2_scale = torch.ones(B, K // 64, N, 2, dtype=torch.float8_e8m0fnu).npu()

    y = cann_ops_nn.transpose_quant_batch_mat_mul(
        x1,
        x2,
        dtype=27,
        x1_scale=x1_scale,
        x2_scale=x2_scale,
        group_sizes=[1, 1, 32],
    )
    print(y.shape, y.dtype)
    ```

  - MXFP4场景示例：

    ```python
    import torch
    import torch_npu
    import cann_ops_nn

    M, B, K, N = 64, 16, 128, 256
    # FP4以torch_npu.float4_e2m1fn_x2格式存储（两个FP4元素拼包），Tensor最后一维为物理长度（逻辑长度的一半）
    x1 = torch.randint(0, 256, (M, B, K // 2), dtype=torch.uint8).npu()
    x2 = torch.randint(0, 256, (B, K, N // 2), dtype=torch.uint8).npu()
    x1_scale = torch.ones(M, B, K // 64, 2, dtype=torch.float8_e8m0fnu).npu()
    x2_scale = torch.ones(B, K // 64, N, 2, dtype=torch.float8_e8m0fnu).npu()

    y = cann_ops_nn.transpose_quant_batch_mat_mul(
        x1,
        x2,
        dtype=27,
        x1_scale=x1_scale,
        x2_scale=x2_scale,
        group_sizes=[1, 1, 32],
        x1_dtype=torch_npu.float4_e2m1fn_x2,
        x2_dtype=torch_npu.float4_e2m1fn_x2,
    )
    print(y.shape, y.dtype)
    ```

  - FP8（K-C）场景示例：

    ```python
    import torch
    import torch_npu
    import cann_ops_nn

    M, B, K, N = 64, 2, 512, 128
    x1 = torch.randn(M, B, K).to(torch.float8_e4m3fn).npu()
    x2 = torch.randn(B, K, N).to(torch.float8_e5m2).npu()  # e4m3与e5m2可任意组合
    x1_scale = torch.rand(M).to(torch.float32).npu()       # per-token (M,)
    x2_scale = torch.rand(N).to(torch.float32).npu()       # per-channel (N,)

    y = cann_ops_nn.transpose_quant_batch_mat_mul(
        x1,
        x2,
        dtype=1,
        x1_scale=x1_scale,
        x2_scale=x2_scale,
        group_sizes=[0, 0, 0],
    )
    print(y.shape, y.dtype)
    ```

  - HIFP8场景示例：

    ```python
    import numpy as np
    import torch
    import torch_npu
    import cann_ops_nn

    M, B, K, N = 64, 16, 128, 96
    # HIFLOAT8按uint8存储，通过x1_dtype/x2_dtype指定逻辑类型
    x1 = torch.randint(0, 128, (M, B, K), dtype=torch.uint8).npu()
    x2 = torch.randint(0, 128, (B, K, N), dtype=torch.uint8).npu()
    # x2_scale为64位 per-channel编码：每个元素的低32位存放float32位模式（1.0=0x000000003F800000）
    # 可传torch.uint64或torch.int64，两者均按位解释，不是数值语义（不要直接传2这类整数当scale）
    f32_bits = np.full(N, np.float32(1.0).view(np.uint32), dtype=np.uint32)
    x2_scale = torch.from_numpy(f32_bits.astype(np.uint64)).npu()
    # int64构造方式（与上面uint64位模式等价）：
    # x2_scale = torch.from_numpy(f32_bits.astype(np.uint64).view(np.int64)).npu()

    y = cann_ops_nn.transpose_quant_batch_mat_mul(
        x1,
        x2,
        dtype=1,
        x2_scale=x2_scale,
        group_sizes=[0, 0, 0],
        x1_dtype=torch_npu.hifloat8,
        x2_dtype=torch_npu.hifloat8,
    )
    print(y.shape, y.dtype)
    ```

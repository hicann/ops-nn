# QuantBatchMatmulV4

## 产品支持情况

| 产品 | 是否支持 |
| --- | :---: |
| <term>Ascend 950PR&950DT系列产品</term> | √ |
| <term>Atlas A3系列产品</term> | √ |
| <term>Atlas A2系列产品</term> | √ |
| <term>Atlas 200I/500 A2推理产品</term> | × |
| <term>Atlas推理系列产品</term> | × |
| <term>Atlas训练系列产品</term> | × |
| <term>KirinX90处理器系列产品</term> | √ |
| <term>Kirin9030处理器系列产品</term> | × |

## 功能说明

- 算子功能：完成量化矩阵乘计算，支持输入缩放、偏置累加以及不同输入精度的组合。

- 版本演进：相较[QuantBatchMatmulV3](../quant_batch_matmul_v3/README.md)，扩展了x1_scale、x2_scale、y_scale及各阶段offset等参数，并支持8位输入与4位权重的混合精度计算。

- 公式约定：x1、x2表示按transpose_x1、transpose_x2解释后的矩阵，形状分别为(M, K)、(K, N)，结果y的形状为(M, N)。公式中的缩放因子按量化分组广播，带上标j的项表示第j个K方向分组，j从0取到$\lceil K/g_K\rceil-1$，$g_K$为K方向分组大小；UINT64、INT64保存编码后的量化参数，使用时须解码。未提供bias或offset时按0处理，未参与计算的scale按1处理。

- 计算公式：

  - <term>Atlas A2系列产品</term>、<term>Atlas A3系列产品</term>：
    - x1为INT8，x2为INT4，x1_scale为FLOAT32，x2_scale为UINT64，y_offset为FLOAT32，y为FLOAT16、BFLOAT16：

      $$
      y = ((x1 @ (x2*\mathrm{x2\_scale})) + \mathrm{y\_offset}) * \mathrm{x1\_scale}
      $$

    - 无x1_scale、无bias：

      $$
      y = x1@x2 * \mathrm{x2\_scale} + \mathrm{x2\_offset}
      $$

    - bias为INT32：

      $$
      y = (x1@x2 + bias) * \mathrm{x2\_scale} + \mathrm{x2\_offset}
      $$

    - bias为BFLOAT16、FLOAT32（此场景无x2_offset）：

      $$
      y = x1@x2 * \mathrm{x2\_scale} + bias
      $$

    - 有x1_scale、无bias：

      $$
      y = x1@x2 * \mathrm{x2\_scale} * \mathrm{x1\_scale}
      $$

    - 有x1_scale，bias为INT32（此场景无x2_offset）：

      $$
      y = (x1@x2 + bias) * \mathrm{x2\_scale} * \mathrm{x1\_scale}
      $$

    - 有x1_scale，bias为BFLOAT16、FLOAT16、FLOAT32（此场景无x2_offset）：

      $$
      y = x1@x2 * \mathrm{x2\_scale} * \mathrm{x1\_scale} + bias
      $$

    - x1、x2为INT8，x1_scale、x2_scale为FLOAT32，bias为FLOAT32，y为FLOAT16、BFLOAT16（逐token对分组量化）：

      $$
      y = \sum_j (x1^{(j)} @ x2^{(j)}) * \mathrm{x1\_scale}^{(j)} * \mathrm{x2\_scale}^{(j)} + bias
      $$

    - x1、x2为INT4，x1_scale、x2_scale为FLOAT32，x2_offset为FLOAT16，y为FLOAT16、BFLOAT16（逐token对分组非对称量化）：

      $$
      y = \sum_j (x1^{(j)} @ (x2^{(j)} - \mathrm{x2\_offset}^{(j)})) * \mathrm{x1\_scale}^{(j)} * \mathrm{x2\_scale}^{(j)}
      $$

  - <term>Ascend 950PR&950DT系列产品</term>：

    - x1、x2为FLOAT8_E4M3FN、FLOAT8_E5M2、HIFLOAT8，无x1_scale，x2_scale为INT64、UINT64，无x2_offset，可选参数bias的dtype为FLOAT32，y为FLOAT16、BFLOAT16、FLOAT32：

      $$
      y = (x1@x2 + bias) * \mathrm{x2\_scale}
      $$

    - MX（Microscaling，微缩放）[量化模式](../../docs/zh/context/quant_mode_introduction.md)中，x1、x2为FLOAT4_E2M1、FLOAT8_E4M3FN、FLOAT8_E5M2，x1_scale为FLOAT8_E8M0，x2_scale为FLOAT8_E8M0，无x2_offset，可选参数bias的dtype为FLOAT32：

      $$
      y = (x1* \mathrm{x1\_scale})@(x2* \mathrm{x2\_scale}) + bias
      $$

    - x1、x2为FLOAT8_E4M3FN、FLOAT8_E5M2、HIFLOAT8，x1_scale为FLOAT32，x2_scale为FLOAT32，无x2_offset，可选参数bias的dtype为FLOAT32：

      $$
      y = (x1@x2 + bias) * \mathrm{x2\_scale} * \mathrm{x1\_scale}
      $$

    - 在G-B和B-B[量化模式](../../docs/zh/context/quant_mode_introduction.md)中，x1、x2为FLOAT8_E4M3FN、FLOAT8_E5M2、HIFLOAT8，x1_scale为FLOAT32，x2_scale为FLOAT32，无x2_offset，无bias，当x1为(a0, a1)，x2为(b0, b1)时，x1_scale为(ceil(a0 / 128), ceil(a1 / 128))或(a0, ceil(a1 / 128))，x2_scale为(ceil(b0 / 128), ceil(b1 / 128))：

      $$
      y_{pq} = \sum_{r=0}^{\lceil K/g_K\rceil-1} (x1_{pr}@x2_{rq}*(\mathrm{x1\_scale}_{pr}*\mathrm{x2\_scale}_{rq}))
      $$

      其中，p、q为M、N方向的分块索引，r为K方向的分组索引，$g_K=128$。各scale按对应块内的行、列广播。

    - x1为FLOAT8_E4M3FN，x2为FLOAT4_E2M1，x1_scale为FLOAT8_E8M0，x2_scale为FLOAT8_E8M0，无x1_offset，无x2_offset，可选参数bias的dtype为BFLOAT16、FLOAT16，y为BFLOAT16、FLOAT16：

      $$
      y = (x1 * \mathrm{x1\_scale})@(x2 * \mathrm{x2\_scale}) + bias
      $$

    - x1为FLOAT8_E4M3FN，x2为FLOAT4_E2M1，无x1_scale，x2_scale为BFLOAT16、FLOAT16，无x1_offset，无x2_offset，无bias，y_scale为UINT64，y为BFLOAT16、FLOAT16：

      $$
      y = (x1@(x2 * \mathrm{x2\_scale})) * \mathrm{y\_scale}
      $$

  - <term>KirinX90处理器系列产品</term>：

    - x1为INT8，x2为INT4（FRACTAL_Z格式），x1_scale为FLOAT32，x2_scale为UINT64，可选参数bias的数据类型为FLOAT16，y为FLOAT16。按K方向分组缩放并累加，未提供bias时取0：

      $$
      y = \left[\sum_j (x1^{(j)} @ x2^{(j)}) * \mathrm{x2\_scale}^{(j)}\right] * \mathrm{x1\_scale} + bias
      $$

## 参数说明

| 参数名 | 输入/输出/属性 | 描述 | 数据类型 | 数据格式 |
| --- | --- | --- | --- | --- |
| x1 | 输入 | 矩阵乘的左矩阵。 | INT8、INT4、HIFLOAT8、FLOAT8_E5M2、FLOAT8_E4M3FN、FLOAT4_E2M1 | ND |
| x2 | 输入 | 矩阵乘的右矩阵。FLOAT32用于承载特定预处理后的FLOAT4_E2M1权重，不表示普通FLOAT32矩阵乘。 | INT8、INT4、HIFLOAT8、FLOAT8_E5M2、FLOAT8_E4M3FN、FLOAT4_E2M1、FLOAT32 | ND、FRACTAL_NZ、FRACTAL_NZ_C0_32、FRACTAL_NZ_C0_4、FRACTAL_Z |
| bias | 可选输入 | 矩阵乘累加的偏置；未提供时不加偏置。累加位置由量化模式决定。 | INT32、BFLOAT16、FLOAT16、FLOAT32 | ND |
| x1_scale | 可选输入 | 左矩阵的量化缩放因子；逐token及分组量化场景须提供。不使用独立左矩阵缩放因子时可省略。 | BFLOAT16、FLOAT16、FLOAT32、FLOAT8_E8M0 | ND |
| x2_scale | 可选输入 | 右矩阵的量化缩放因子，当前支持的量化计算场景均须提供。 | BFLOAT16、FLOAT16、FLOAT32、UINT64、INT64、FLOAT8_E8M0 | ND |
| y_scale | 可选输入 | 矩阵乘后的量化缩放因子；需要输出缩放的场景须提供，不执行该缩放时省略。 | UINT64 | ND |
| x1_offset | 可选输入 | 左矩阵的量化偏置预留参数，当前不支持传入。 | BFLOAT16、FLOAT16、FLOAT32 | ND |
| x2_offset | 可选输入 | 右矩阵的量化偏置；非对称量化场景提供对应的偏置，不使用时省略。 | BFLOAT16、FLOAT16、FLOAT32 | ND |
| y_offset | 可选输入 | 矩阵乘后的偏移修正参数，在对应量化场景提供；不使用时省略。 | BFLOAT16、FLOAT16、FLOAT32 | ND |
| x2_table | 可选输入 | 权重量化查找表预留参数，当前不支持传入。 | INT8 | ND |
| y | 输出 | 矩阵乘及量化处理后的结果，数据类型由dtype指定。 | BFLOAT16、FLOAT16、INT8、FLOAT32 | ND |
| dtype | 属性 | 输出y的数据类型枚举，支持0（FLOAT32）、1（FLOAT16）、2（INT8）、27（BFLOAT16），须满足对应产品及输入组合的要求。 | INT64 | - |
| compute_type | 可选属性 | <ul><li>计算类型，支持-1（自动选择）或4（FLOAT8与FLOAT4混合精度计算）。</li><li>默认值为-1。</li></ul> | INT64 | - |
| transpose_x1 | 可选属性 | <ul><li>是否转置x1的最后两维：false表示不转置，true表示转置。</li><li>默认值为false，具体支持组合见<a href="#约束说明">约束说明</a>。</li></ul> | BOOL | - |
| transpose_x2 | 可选属性 | <ul><li>是否转置x2的最后两维：false表示不转置，true表示转置。</li><li>默认值为false，具体支持组合见<a href="#约束说明">约束说明</a>。</li></ul> | BOOL | - |
| group_size | 可选属性 | <ul><li>量化分组大小或M、N、K方向的分组编码；取值由量化模式决定，见<a href="#约束说明">约束说明</a>。</li><li>默认值为-1；分组量化须按对应场景设置。</li></ul> | INT64 | - |

## 约束说明

- 不支持空Tensor。
- group_size须与量化模式匹配。M、N、K方向的分组编码方式为`group_size_k | (group_size_n << 16) | (group_size_m << 32)`，G-B和B-B模式分别按`[1, 128, 128]`和`[128, 128, 128]`编码。
- <term>Atlas A3系列产品</term>、<term>Atlas A2系列产品</term>：INT8输入与INT4权重的逐通道量化场景group_size为0，分组量化场景为256。
- <term>Ascend 950PR&950DT系列产品</term>：FLOAT8输入与FLOAT4权重的ND场景group_size为32。
- 输入和输出支持以下数据类型组合：

  - <term>Atlas A2系列产品</term>、<term>Atlas A3系列产品</term>：

    | x1 | x2 | x1_scale | x2_scale | x2_offset | y_scale | bias | y_offset | y |
    | --- | --- | --- | --- | --- | --- | --- | --- | --- |
    | INT8 | INT4 | FLOAT32 | UINT64 | 不传入 | 不传入 | 不传入 | FLOAT32 | FLOAT16、BFLOAT16 |
    | INT8 | INT8 | 不传入 | UINT64、INT64 | 不传入 | 不传入 | 不传入或INT32 | 不传入 | FLOAT16 |
    | INT8 | INT8 | 不传入 | UINT64、INT64 | 不传入或FLOAT32 | 不传入 | 不传入或INT32 | 不传入 | INT8 |
    | INT8 | INT8 | 不传入或FLOAT32 | BFLOAT16 | 不传入 | 不传入 | 不传入或INT32、BFLOAT16、FLOAT32 | 不传入 | BFLOAT16 |
    | INT8 | INT8 | FLOAT32 | FLOAT32 | 不传入 | 不传入 | 不传入或INT32、FLOAT16、FLOAT32 | 不传入 | FLOAT16 |
    | INT8 | INT8 | 不传入 | FLOAT32、BFLOAT16 | 不传入 | 不传入 | 不传入或INT32 | 不传入 | FLOAT16 |
    | INT8 | INT8 | FLOAT32 | FLOAT32 | 不传入 | 不传入 | FLOAT32 | 不传入 | BFLOAT16 |
    | INT4 | INT4 | FLOAT32 | FLOAT32、BFLOAT16 | 不传入 | 不传入 | 不传入或INT32、BFLOAT16、FLOAT32 | 不传入 | BFLOAT16 |
    | INT4 | INT4 | FLOAT32 | FLOAT32 | 不传入 | 不传入 | 不传入或INT32、FLOAT16、FLOAT32 | 不传入 | FLOAT16 |
    | INT4 | INT4 | FLOAT32 | FLOAT32 | FLOAT16 | 不传入 | 不传入 | 不传入 | BFLOAT16 |

  - <term>Ascend 950PR&950DT系列产品</term>：

    | x1 | x2 | x1_scale | x2_scale | x2_offset | y_scale | bias | y |
    | --- | --- | --- | --- | --- | --- | --- | --- |
    | INT8 | INT8 | 不传入 | UINT64、INT64 | 不传入 | 不传入 | 不传入或INT32 | FLOAT16、BFLOAT16 |
    | INT8 | INT8 | 不传入 | UINT64、INT64 | 不传入或FLOAT32 | 不传入 | 不传入或INT32 | INT8 |
    | INT8 | INT8 | 不传入或FLOAT32 | FLOAT32、BFLOAT16 | 不传入 | 不传入 | 不传入或INT32、FLOAT32、BFLOAT16 | BFLOAT16 |
    | INT8 | INT8 | FLOAT32 | FLOAT32 | 不传入 | 不传入 | 不传入或INT32、FLOAT32、FLOAT16 | FLOAT16 |
    | FLOAT8_E4M3FN、FLOAT8_E5M2 | FLOAT8_E4M3FN、FLOAT8_E5M2 | 不传入 | UINT64、INT64 | 不传入 | 不传入 | 不传入或FLOAT32 | FLOAT16、BFLOAT16、FLOAT32 |
    | HIFLOAT8 | HIFLOAT8 | 不传入 | UINT64、INT64 | 不传入 | 不传入 | 不传入或FLOAT32 | FLOAT16、BFLOAT16、FLOAT32 |
    | FLOAT8_E4M3FN、FLOAT8_E5M2 | FLOAT8_E4M3FN、FLOAT8_E5M2 | FLOAT32 | FLOAT32 | 不传入 | 不传入 | 不传入或FLOAT32 | FLOAT16、BFLOAT16、FLOAT32 |
    | HIFLOAT8 | HIFLOAT8 | FLOAT32 | FLOAT32 | 不传入 | 不传入 | 不传入或FLOAT32 | FLOAT16、BFLOAT16、FLOAT32 |
    | FLOAT4_E2M1 | FLOAT4_E2M1 | FLOAT8_E8M0 | FLOAT8_E8M0 | 不传入 | 不传入 | 不传入或FLOAT32 | FLOAT16、BFLOAT16、FLOAT32 |
    | FLOAT8_E4M3FN、FLOAT8_E5M2 | FLOAT8_E4M3FN、FLOAT8_E5M2 | FLOAT8_E8M0 | FLOAT8_E8M0 | 不传入 | 不传入 | 不传入或FLOAT32 | FLOAT16、BFLOAT16、FLOAT32 |
    | FLOAT8_E4M3FN | FLOAT4_E2M1 | FLOAT8_E8M0 | FLOAT8_E8M0 | 不传入 | 不传入 | 不传入或BFLOAT16、FLOAT16 | BFLOAT16、FLOAT16 |
    | FLOAT8_E4M3FN | FLOAT4_E2M1 | 不传入 | BFLOAT16、FLOAT16 | 不传入 | UINT64 | 不传入 | BFLOAT16、FLOAT16 |

  - <term>KirinX90处理器系列产品</term>：

    | x1 | x2 | x1_scale | x2_scale | x2_offset | y_scale | bias | y_offset | y |
    | --- | --- | --- | --- | --- | --- | --- | --- | --- |
    | INT8 | INT4 | FLOAT32 | UINT64 | 不传入 | 不传入 | 不传入或FLOAT16 | 不传入 | FLOAT16 |

    额外约束：

    - x2为FRACTAL_Z格式，存储形状须为`[K/64, N/16, 16, 8]`。
    - transpose_x1、transpose_x2仅支持false。
    - group_size仅支持128或256。
    - 对齐约束：`K % 64 == 0`、`N % 16 == 0`、`K % group_size == 0`。
    - x2_scale的形状须为`[ceil(K/group_size), N]`，其中ceil表示向上取整。
    - 分组数`GT=ceil(K/group_size)`不超过960。

## 调用说明

- <term>Atlas A3系列产品</term>、<term>Atlas A2系列产品</term>：

  | 调用方式 | 样例代码 | 说明 |
  | --- | --- | --- |
  | aclnn API | [test_aclnn_quant_matmul_v5_at2_at3](examples/test_aclnn_quant_matmul_v5_at2_at3.cpp) | 通过[aclnnQuantMatmulV5](docs/aclnnQuantMatmulV5.md)调用，展示INT8输入与4位权重的量化计算。 |

- <term>Ascend 950PR&950DT系列产品</term>：

  | 调用方式 | 样例代码 | 说明 |
  | --- | --- | --- |
  | aclnn API | [test_aclnn_quant_matmul_v5_f8f4_nd](examples/arch35/test_aclnn_quant_matmul_v5_f8f4_nd.cpp) | 通过[aclnnQuantMatmulV5](docs/aclnnQuantMatmulV5.md)调用，展示FLOAT8与FLOAT4混合精度计算。 |
  | aclnn API | [test_aclnn_quant_matmul_weight_nz_f8mxf4](../quant_batch_matmul_v3/examples/arch35/test_aclnn_quant_matmul_weight_nz_f8mxf4.cpp) | 通过[aclnnQuantMatmulWeightNz](../quant_batch_matmul_v3/docs/aclnnQuantMatmulWeightNz.md)调用，展示使用FRACTAL_NZ权重的FLOAT8与FLOAT4混合精度计算。 |
  | GE图模式 | [test_geir_quant_batch_matmul_v4](examples/arch35/test_geir_quant_batch_matmul_v4.cpp) | 通过[QuantBatchMatmulV4算子IR](op_graph/quant_batch_matmul_v4_proto.h)构图调用。 |

aclnnQuantMatmulV5会根据输入组合调用QuantBatchMatmulV3或QuantBatchMatmulV4，接口参数及布局要求见[aclnnQuantMatmulV5](docs/aclnnQuantMatmulV5.md)。

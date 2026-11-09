# QuantBatchMatmulV3

## 产品支持情况

| 产品 | 是否支持 |
| --- | :---: |
| <term>Ascend 950PR&950DT系列产品</term> | √ |
| <term>Atlas A3系列产品</term> | √ |
| <term>Atlas A2系列产品</term> | √ |
| <term>Atlas 200I/500 A2推理产品</term> | × |
| <term>Atlas推理系列产品</term> | √ |
| <term>Atlas训练系列产品</term> | × |
| <term>KirinX90处理器系列产品</term> | √ |
| <term>Kirin9030处理器系列产品</term> | √ |

## 功能说明

- 算子功能：完成量化矩阵乘计算，支持缩放、累加偏置和输出类型转换，输入的逻辑维数为2～6。支持的逐张量、逐token、逐通道及分组量化模式随产品和数据类型而异，模式定义见[量化模式介绍](../../docs/zh/context/quant_mode_introduction.md)。

- 版本关系：QuantBatchMatmulV3是算子名称，与aclnn接口的版本号分别定义。[aclnnQuantMatmulV4](docs/aclnnQuantMatmulV4.md)相较[aclnnQuantMatmulV3](docs/aclnnQuantMatmulV3.md)增加了左矩阵的pertoken_scale输入；[aclnnQuantMatmulV5](../quant_batch_matmul_v4/docs/aclnnQuantMatmulV5.md)提供更多量化参数，并按输入组合调用本算子或[QuantBatchMatmulV4](../quant_batch_matmul_v4/README.md)。

- 计算公式：以下省略batch下标，$X_1$、$X_2$表示按transpose_x1、transpose_x2解释后的矩阵，形状分别为$(M,K)$、$(K,N)$。M、N为输出矩阵的行数、列数，K为归约维度；$S_1$、$S_2$分别对应pertoken_scale、scale。未提供pertoken_scale、bias、offset时，分别按1、0、0处理。

  - 非分组量化、无bias：

    $$
    y=(X_1@X_2)\odot S_2\odot S_1+offset
    $$

  - INT8或INT4输入且bias为INT32：

    $$
    y=(X_1@X_2+bias)\odot S_2\odot S_1+offset
    $$

  - INT8或INT4输入且bias为BFLOAT16、FLOAT16或FLOAT32：

    $$
    y=(X_1@X_2)\odot S_2\odot S_1+bias
    $$

  - FLOAT8_E4M3FN、FLOAT8_E5M2或HIFLOAT8的非分组量化，bias为FLOAT32：

    $$
    y=(X_1@X_2+bias)\odot S_2\odot S_1
    $$

  - INT32输出：直接输出整数累加结果，不执行缩放。

    $$
    y=X_1@X_2+bias
    $$

  - MX（Microscaling，微缩放）、G-B和B-B分组量化：按K轴分组计算，再累加各组结果。

    $$
    y_{m,n}=\sum_{j=0}^{L-1}\left[
    \left(\sum_{t=0}^{g_j-1}X_{1,m,jg_K+t}X_{2,jg_K+t,n}\right)
    S_{1,m,j}S_{2,j,n}\right]+bias_n
    $$

    其中，$g_K$为K方向分组大小，$L=\lceil K/g_K\rceil$，$g_j=\min(g_K,K-jg_K)$；$S_1$、$S_2$表示按对应量化分组广播后的scale。G-B和B-B模式不支持bias，公式中取$bias_n=0$。

  其中，$@$表示矩阵乘，$\odot$表示按量化参数形状广播的逐元素乘。offset仅用于支持的INT8输出组合，不能与pertoken_scale同时使用。上述公式表示量化计算关系，实际结果还需转换为dtype指定的数据类型。

## 参数说明

| 参数名 | 输入/输出/属性 | 描述 | 数据类型 | 数据格式 |
| --- | --- | --- | --- | --- |
| x1 | 输入 | 矩阵乘的左矩阵，对应公式中的X₁。 | INT8、INT4、HIFLOAT8、FLOAT8_E5M2、FLOAT8_E4M3FN、FLOAT4_E2M1 | ND、FRACTAL_NZ |
| x2 | 输入 | 矩阵乘的右矩阵，对应公式中的X₂。 | INT8、INT4、HIFLOAT8、FLOAT8_E5M2、FLOAT8_E4M3FN、FLOAT4_E2M1 | ND、FRACTAL_NZ |
| scale | 输入 | 右矩阵的量化缩放参数。UINT64、INT64类型存储编码后的量化参数，须按解码后的scale参与公式计算。 | UINT64、FLOAT32、INT64、BFLOAT16、FLOAT8_E8M0 | ND、FRACTAL_NZ |
| offset | 可选输入 | 输出量化的偏置因子；未提供时按0处理，仅用于支持的INT8输出组合。 | FLOAT32 | ND |
| bias | 可选输入 | 矩阵乘的累加偏置；未提供时不加偏置。累加位置由输入及bias的数据类型决定。 | INT32、BFLOAT16、FLOAT16、FLOAT32 | ND |
| pertoken_scale | 可选输入 | 左矩阵的量化缩放参数；未提供时按1处理。逐token及分组量化场景须提供对应的缩放参数。 | FLOAT32、FLOAT8_E8M0 | ND |
| y | 输出 | 矩阵乘及量化处理后的结果，数据类型由dtype指定。 | FLOAT16、INT8、BFLOAT16、INT32、FLOAT32 | ND、FRACTAL_NZ |
| dtype | 属性 | 输出y的数据类型枚举，支持0（FLOAT32）、1（FLOAT16）、2（INT8）、3（INT32）、27（BFLOAT16），具体取值受产品及输入组合限制。 | INT64 | - |
| transpose_x1 | 可选属性 | <ul><li>是否转置x1的最后两维：false表示不转置，true表示转置。</li><li>默认值为false。</li></ul> | BOOL | - |
| transpose_x2 | 可选属性 | <ul><li>是否转置x2的最后两维：false表示不转置，true表示转置。</li><li>默认值为false。</li></ul> | BOOL | - |
| group_size | 可选属性 | <ul><li>M、N、K方向量化分组大小的编码值，各分量占16位；支持的分组组合见<a href="#约束说明">约束说明</a>。</li><li>默认值为0。</li></ul> | INT64 | - |

- <term>Atlas A3系列产品</term>、<term>Atlas A2系列产品</term>：x1、x2不支持HIFLOAT8、FLOAT8_E5M2、FLOAT8_E4M3FN、FLOAT4_E2M1；scale不支持FLOAT8_E8M0；pertoken_scale仅支持FLOAT32；y不支持FLOAT32。
- <term>Atlas推理系列产品</term>：x1、x2仅支持INT8；scale支持UINT64、INT64，逐token场景还支持FLOAT32；bias支持INT32，逐token场景还支持FLOAT32；pertoken_scale仅支持FLOAT32；y支持FLOAT16、INT8，逐token场景仅支持FLOAT16。逐token场景的y使用FRACTAL_NZ，其余场景使用ND。
- <term>KirinX90处理器系列产品</term>、<term>Kirin9030处理器系列产品</term>：x1、x2仅支持INT8，x1使用ND，x2使用FRACTAL_NZ；scale仅支持UINT64、INT64；bias仅支持INT32；pertoken_scale仅支持FLOAT32；y仅支持FLOAT16、INT8。
- <term>Ascend 950PR&950DT系列产品</term>：scale的FRACTAL_NZ格式仅用于MX全量化的ScaleNZ场景，要求x2也使用FRACTAL_NZ，且x1、x2类型相同，为FLOAT8_E4M3FN或FLOAT4_E2M1。布局和转换要求见[ScaleNZ约束](docs/aclnnQuantMatmulWeightNz.md#scalenz约束)。其他产品的scale使用ND。

## 约束说明

- 输入矩阵的K维必须一致，batch维须满足[广播关系](../../docs/zh/context/broadcast_relationship.md)。具体形状和产品限制见[算子IR说明](docs/ascendir_QuantBatchMatmulV3.md)。
- INT4输入仅支持二维，transpose_x1须为false，输入的最内维须为偶数。
- group_size按`group_size_k | (group_size_n << 16) | (group_size_m << 32)`编码。分组分量为0时由输入形状推导，推导时对应维度须满足整除要求。MX、G-B、B-B模式的最终分组大小分别为`[1, 1, 32]`、`[1, 128, 128]`、`[128, 128, 128]`；其他量化模式的group_size为0。
- MX模式下，pertoken_scale、scale均须提供且类型为FLOAT8_E8M0。FLOAT4_E2M1输入的最内维须为偶数，且K须大于2。
- <term>Atlas A3系列产品</term>、<term>Atlas A2系列产品</term>：输入矩阵逻辑形状的最后一维不能超过65535。
- <term>Ascend 950PR&950DT系列产品</term>：x2使用ND时支持M=0或N=0并输出空Tensor；x2使用FRACTAL_NZ时支持M=0并输出空Tensor。其他场景不支持维度大小为0的输入。

## 调用说明

| 调用方式 | 样例代码 | 说明 |
| --- | --- | --- |
| GE图模式 | [test_geir_quant_batch_matmul_v3](examples/test_geir_quant_batch_matmul_v3.cpp) | 通过[QuantBatchMatmulV3算子IR](docs/ascendir_QuantBatchMatmulV3.md)构图调用。 |
| aclnn API | [test_aclnn_quant_matmul_v3_at](examples/test_aclnn_quant_matmul_v3_at.cpp) | 通过[aclnnQuantMatmulV3](docs/aclnnQuantMatmulV3.md)调用。 |
| aclnn API | [test_aclnn_quant_matmul_v4_at2_at3](examples/test_aclnn_quant_matmul_v4_at2_at3.cpp) | 通过[aclnnQuantMatmulV4](docs/aclnnQuantMatmulV4.md)调用，支持左矩阵逐token缩放。 |
| aclnn API | [test_aclnn_quant_matmul_weight_nz](examples/test_aclnn_quant_matmul_weight_nz.cpp) | 通过[aclnnQuantMatmulWeightNz](docs/aclnnQuantMatmulWeightNz.md)调用，使用FRACTAL_NZ格式的权重。 |
| aclnn API | [test_aclnn_quant_matmul_v5_hif8hif8](../quant_batch_matmul_v4/examples/arch35/test_aclnn_quant_matmul_v5_hif8hif8.cpp) | 通过[aclnnQuantMatmulV5](../quant_batch_matmul_v4/docs/aclnnQuantMatmulV5.md)调用，样例展示由QuantBatchMatmulV3执行的HIFLOAT8量化场景。 |

aclnnQuantMatmulV3、aclnnQuantMatmulV4已在接口声明中标记为待废弃，迁移说明见对应接口文档。

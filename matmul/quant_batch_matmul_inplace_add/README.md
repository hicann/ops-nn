# QuantBatchMatmulInplaceAdd

## 产品支持情况

| 产品 | 是否支持 |
| --- | :---: |
| <term>Ascend 950PR&950DT系列产品</term> | √ |
| <term>Atlas A3系列产品</term> | × |
| <term>Atlas A2系列产品</term> | × |
| <term>Atlas 200I/500 A2推理产品</term> | × |
| <term>Atlas推理系列产品</term> | × |
| <term>Atlas训练系列产品</term> | × |

## 功能说明

- 算子功能：融合量化矩阵乘和原地加法，将矩阵乘结果累加到已有的y中，适用于微批次（micro-batch）训练中的梯度累积。

- 计算公式：支持MX（Microscaling，微缩放）量化和HIFLOAT8的T-T（pertensor-pertensor，逐张量对逐张量）量化，模式说明见[量化模式介绍](../../docs/zh/context/quant_mode_introduction.md)。

  以下公式中的$X_1$、$X_2$表示按转置属性解释后的矩阵，形状分别为$(M,K)$、$(K,N)$。M、N为结果矩阵的行数、列数，K为归约维度；$Y_{\mathrm{in}}$、$Y_{\mathrm{out}}$分别表示原地累加前后的y。

  - MX量化：

    $$
    Y_{\mathrm{out},m,n}=Y_{\mathrm{in},m,n}+
    \sum_{j=0}^{L-1}\left[
    \left(\sum_{t=0}^{g_j-1}X_{1,m,jg_K+t}X_{2,jg_K+t,n}\right)
    S_{1,m,j}S_{2,j,n}\right]
    $$

    其中，$g_K=32$，$L=\lceil K/g_K\rceil$，$g_j=\min(g_K,K-jg_K)$。$S_1$、$S_2$为对应K分组的输入scale，最后一组可以不足32个元素。

  - HIFLOAT8的T-T量化：

    $$
    Y_{\mathrm{out},m,n}=Y_{\mathrm{in},m,n}+
    \left(\sum_{k=0}^{K-1}X_{1,m,k}X_{2,k,n}\right)S_1S_2
    $$

    其中，$S_1$、$S_2$分别对应x1_scale、x2_scale，均为形状为$(1,)$的缩放因子。

## 参数说明

| 参数名 | 输入/输出/属性 | 描述 | 数据类型 | 数据格式 |
| --- | --- | --- | --- | --- |
| x1 | 输入 | 矩阵乘的左矩阵，对应公式中的X₁。 | FLOAT8_E4M3FN、FLOAT8_E5M2、HIFLOAT8、FLOAT4_E2M1 | ND |
| x2 | 输入 | 矩阵乘的右矩阵，对应公式中的X₂。 | FLOAT8_E4M3FN、FLOAT8_E5M2、HIFLOAT8、FLOAT4_E2M1 | ND |
| x2_scale | 输入 | x2的量化缩放因子，对应公式中的S₂。MX量化使用FLOAT8_E8M0，HIFLOAT8的T-T量化使用FLOAT32。 | FLOAT8_E8M0、FLOAT32 | ND |
| y | 输入 | 参与原地累加的已有结果，对应公式中的Y_in。 | FLOAT32 | ND |
| x1_scale | 可选输入 | x1的量化缩放因子，对应公式中的S₁。当前支持的MX和HIFLOAT8的T-T量化场景均须提供，数据类型与x2_scale一致。 | FLOAT8_E8M0、FLOAT32 | ND |
| y | 输出 | 原地累加后的结果，对应公式中的Y_out，与输入y共用存储。 | FLOAT32 | ND |
| transpose_x1 | 可选属性 | <ul><li>是否转置x1的最后两维：false表示不转置，true表示转置。</li><li>默认值为false；当前支持的计算场景须设为true。</li></ul> | BOOL | - |
| transpose_x2 | 可选属性 | <ul><li>是否转置x2的最后两维：false表示不转置，true表示转置。</li><li>默认值为false；当前支持的计算场景须保持false。</li></ul> | BOOL | - |
| group_size | 可选属性 | <ul><li>M、N、K方向量化分组大小的编码值，各分量占16位；支持的分组组合见<a href="#约束说明">约束说明</a>。</li><li>默认值为0。</li></ul> | INT64 | - |

## 约束说明

- 当前仅支持transpose_x1=true且transpose_x2=false。输入x1、x2的逻辑形状分别为`(K, M)`、`(K, N)`，y为`(M, N)`。
- group_size按`group_size_k | (group_size_n << 16) | (group_size_m << 32)`编码。分量为0时由对应输入及scale的形状推导；推导后须满足对应量化模式的分组要求。
- MX量化：
  - x1、x2可分别为FLOAT8_E4M3FN或FLOAT8_E5M2，也可同时为FLOAT4_E2M1；不支持FP4与FP8混合输入，不支持FLOAT4_E1M2。
  - x1_scale、x2_scale均为FLOAT8_E8M0，形状分别为`(ceil(K/64), M, 2)`、`(ceil(K/64), N, 2)`，其中ceil表示向上取整。
  - 最终分组大小为`[group_size_m, group_size_n, group_size_k]=[1, 1, 32]`。可显式设置group_size=4295032864；设置为32时，M、N方向的分组大小由形状推导。
  - MXFP4场景下，M、N须为非负偶数。x1、x2采用4位打包存储，每字节保存两个FLOAT4_E2M1元素；连续输入对应的字节存储形状分别为`(K, M/2)`、`(K, N/2)`。
  - MXFP4场景下，当M、N、K均大于0时，还须同时满足$M\times K>2$、$N\times K>2$。
- HIFLOAT8的T-T量化：
  - x1、x2均为HIFLOAT8，x1_scale、x2_scale均为FLOAT32且形状为`(1,)`。
  - group_size须为0。

## 调用说明

| 调用方式 | 样例代码 | 说明 |
| --- | --- | --- |
| aclnn API | [MXFP8样例](examples/arch35/test_aclnn_quant_batch_matmul_inplace_add_mxfp8.cpp) | 通过[aclnnQuantBatchMatmulInplaceAdd](docs/aclnnQuantBatchMatmulInplaceAdd.md)调用，展示MXFP8量化。 |
| aclnn API | [MXFP4样例](examples/arch35/test_aclnn_quant_batch_matmul_inplace_add_mxfp4.cpp) | 通过[aclnnQuantBatchMatmulInplaceAdd](docs/aclnnQuantBatchMatmulInplaceAdd.md)调用，展示打包存储的FLOAT4_E2M1输入。 |
| aclnn API | [HIFLOAT8的T-T样例](examples/arch35/test_aclnn_quant_batch_matmul_inplace_add_TT.cpp) | 通过[aclnnQuantBatchMatmulInplaceAdd](docs/aclnnQuantBatchMatmulInplaceAdd.md)调用，展示逐张量量化。 |
| GE图模式 | - | 通过[QuantBatchMatmulInplaceAdd算子IR](op_graph/quant_batch_matmul_inplace_add_proto.h)构图调用。 |

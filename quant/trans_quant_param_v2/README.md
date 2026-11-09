# TransQuantParamV2

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

- 算子功能：将FLOAT32类型的量化缩放因子scale及可选偏置offset编码为硬件所需的UINT64量化参数，供矩阵乘等算子使用，输出包含scale的浮点位模式、offset的整数编码及标志位。

- 版本关系：[aclnnTransQuantParamV2](docs/aclnnTransQuantParamV2.md)和[aclnnTransQuantParamV3](docs/aclnnTransQuantParamV3.md)均调用本算子。V2接口固定使用round_mode=0；V3接口增加roundMode参数，可选择round_mode=0或1。

- 编码流程：以下描述单个输出元素的计算过程，位编号从最低位0开始。

  1. 处理scale的位模式。round_mode=0时直接保留FLOAT32的高19位；round_mode=1时先按最近偶数舍入模式（RNE，Round to Nearest, ties to Even）保留19位浮点精度，再编码。19位由1位符号、8位指数和10位尾数组成。

     $$
     b=\begin{cases}
     \operatorname{Bits}_{32}(scale), & \mathrm{round\_mode}=0 \\
     \operatorname{Bits}_{32}(\operatorname{RoundFP19}(scale)), & \mathrm{round\_mode}=1
     \end{cases}
     $$

     其中，$\operatorname{Bits}_{32}$表示将FLOAT32按位解释为UINT32；$\operatorname{RoundFP19}$表示上述19位精度舍入，结果仍按FLOAT32位布局表示。RNE表示舍入到最近值，恰好位于中点时取保留尾数最低位为0的值。

  2. 将scale的第13～31位写入y的对应位置，低13位清零，并将y的第46位置1，得到中间编码$y_0$。以下移位和按位或运算均在UINT64中进行。

     $$
     y_0=\left(\operatorname{UINT64}(b)\mathbin{\&}\mathtt{0xFFFFE000}\right)
     \mathbin{|}(1\ll46)
     $$

  3. 未传入offset时，$y=y_0$。传入offset时，先按RNE转换为INT32，再将结果限制到$[-256,255]$，取低9位写入y的第37～45位。

     $$
     q=\min\left(255,\max\left(-256,\operatorname{RNE}_{\mathrm{INT32}}(offset)\right)\right)
     $$

     $$
     y=y_0\mathbin{|}\left[\left(\operatorname{UINT64}(q)\mathbin{\&}\mathtt{0x1FF}\right)\ll37\right]
     $$

     其中，$q$为处理后的offset，负数按补码保留低9位。round_mode仅控制scale的处理方式，offset始终使用RNE。

## 参数说明

| 参数名 | 输入/输出/属性 | 描述 | 数据类型 | 数据格式 |
| --- | --- | --- | --- | --- |
| scale | 输入 | 量化缩放因子，对应公式中的scale。支持一维或二维Tensor，具体形状及使用场景见[约束说明](#约束说明)。 | FLOAT32 | ND |
| offset | 可选输入 | 量化偏置，对应公式中的offset。默认不传入，表示不编码偏置；传入时支持一维或二维Tensor。 | FLOAT32 | ND |
| y | 输出 | 编码后的量化参数，对应公式中的y。输出形状见[约束说明](#约束说明)。 | UINT64 | ND |
| round_mode | 可选属性 | <ul><li>scale的编码模式：0表示截断低13位，1表示按RNE保留19位浮点精度后编码。</li><li>默认值为0，仅支持0或1。</li></ul> | INT64 | - |

## 约束说明

- scale及传入的offset不支持空Tensor，数据中不能包含NaN或Inf。
- round_mode=1时，scale的非规格化浮点数按保留符号的0处理。
- 与矩阵乘算子配套使用时，scale及传入的offset支持形状`(1,)`、`(n,)`或`(1, n)`，n为矩阵乘右矩阵参与计算的列数。可与[QuantBatchMatmulV3](../../matmul/quant_batch_matmul_v3/README.md)等算子配套使用。
- 输出形状推导：
  - 未传入offset时，y的形状与scale一致。
  - 传入offset时，y沿用scale的形状；当offset为二维、scale为一维时，y采用offset的形状。
  - <term>Ascend 950PR&950DT系列产品</term>：当scale和offset均为一维时，y的长度为二者长度的最大值。
- <term>Ascend 950PR&950DT系列产品</term>：与分组矩阵乘算子配套时，仅支持M轴分组，且不传入offset。scale及y的形状支持`(g,)`、`(g, 1)`或`(g, n)`，g为分组数，n为右矩阵参与计算的列数。
- <term>Atlas A3系列产品</term>、<term>Atlas A2系列产品</term>、<term>Atlas推理系列产品</term>：不支持与分组矩阵乘算子配套使用。
- aclnn接口的输入组合及布局要求分别见[aclnnTransQuantParamV2](docs/aclnnTransQuantParamV2.md#约束说明)和[aclnnTransQuantParamV3](docs/aclnnTransQuantParamV3.md#约束说明)。

## 调用说明

| 调用方式 | 样例代码 | 说明 |
| --- | --- | --- |
| aclnn API | [test_aclnn_trans_quant_param_v2](examples/test_aclnn_trans_quant_param_v2.cpp) | 通过[aclnnTransQuantParamV2](docs/aclnnTransQuantParamV2.md)调用，使用round_mode=0。 |
| aclnn API | [test_aclnn_trans_quant_param_v3](examples/test_aclnn_trans_quant_param_v3.cpp) | 通过[aclnnTransQuantParamV3](docs/aclnnTransQuantParamV3.md)调用，样例使用roundMode=1。 |
| GE图模式 | [test_geir_trans_quant_param_v2](examples/test_geir_trans_quant_param_v2.cpp) | 通过[TransQuantParamV2算子IR](op_graph/trans_quant_param_v2_proto.h)构图调用，样例使用round_mode=0。 |

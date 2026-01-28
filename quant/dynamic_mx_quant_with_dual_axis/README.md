# DynamicMxQuantWithDualAxis

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term>                             |    √     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    ×     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> |    ×     |
| <term>Atlas 200I/500 A2 推理产品</term>                      |    ×     |
| <term>Atlas 推理系列产品</term>                             |    ×     |
| <term>Atlas 训练系列产品</term>                              |    ×     |
| <term>Atlas 200/300/500 推理产品</term>                      |    ×     |

## 功能说明

- 算子功能：在-1轴和-2轴上同时进行目的数据类型为FLOAT4类、FLOAT8类的MX量化。在给定的-1轴和-2轴上，每32个数，计算出这两组数对应的量化尺度mxscale1、mxscale2作为输出mxscale1Out、mxscale2Out的对应部分，然后分别对两组数所有元素除以对应的mxscale1或mxscale2，根据round_mode转换到对应的dstType，得到量化结果y1和y2分别作为输出y1Out和y2Out的对应部分。

- 计算公式：
  - 当前只支持scaleAlg=0，即OCP实现：
  - 将输入x在-1轴上按照32个数进行分组，一组32个数 $\{\{V_i\}_{i=1}^{32}\}$ 量化为 $\{mxscale1, \{P_i\}_{i=1}^{32}\}$

    $$
    shared\_exp = floor(log_2(max_i(|V_i|))) - emax
    $$

    $$
    mxscale1 = 2^{shared\_exp}
    $$

    $$
    P_i = cast\_to\_dst\_type(V_i/mxscale1, round\_mode), \space i\space from\space 1\space to\space 32
    $$

  - 同时，将输入x在-2轴上按照32个数进行分组，一组32个数 $\{\{V_j\}_{j=1}^{32}\}$ 量化为 $\{mxscale2, \{P_j\}_{j=1}^{32}\}$

    $$
    shared\_exp = floor(log_2(max_j(|V_j|))) - emax
    $$

    $$
    mxscale2 = 2^{shared\_exp}
    $$

    $$
    P_j = cast\_to\_dst\_type(V_j/mxscale2, round\_mode), \space j\space from\space 1\space to\space 32
    $$

  - -1轴​量化后的 $P_{i}$ 按对应的 $V_{i}$ 的位置组成输出y1Out，mxscale1按对应的-1轴维度上的分组组成输出mxscale1Out。-2轴​量化后的 $P_{j}$ 按对应的 $V_{j}$ 的位置组成输出y2Out，mxscale2按对应的-2轴维度上的分组组成输出mxscale2Out。

  - emax: 对应数据类型的最大正则数的指数位。

    |   DataType    | emax |
    | :-----------: | :--: |
    |  FLOAT4_E2M1  |  2   |
    |  FLOAT4_E1M2  |  0   |
    | FLOAT8_E4M3FN |  8   |
    |  FLOAT8_E5M2  |  15  |

## 参数说明

  | 参数名 | 输入/输出/属性 | 描述 | 数据类型 | 数据格式 |  
  | ----- | ----- |----- |----- |----- |
  | x (aclTensor\*) | 输入 | 表示输入x，对应公式中$V_i$。<br>目的类型为FLOAT4_E2M1、FLOAT4_E1M2时，x的最后一维必须是偶数。 | FLOAT16、BFLOAT16 | ND |
  | roundModeOptional (char\*)  | 可选属性 | 表示数据转换的模式，对应公式中的round_mode。<br>当dstType为40/41，对应输出y1Out和y2Out的数据类型为FLOAT4_E2M1/FLOAT4_E1M2时，支持{"rint", "floor", "round"}；<br>当dstType为35/36，对应输出y1Out和y2Out数据类型为FLOAT8_E5M2/FLOAT8_E4M3FN时，仅支持{"rint"}；<br>传入空指针时，采用"rint"模式。 | STRING | - |
  | dstType (int64_t) | 可选属性 | 表示指定数据转换后y1Out和y2Out的类型，对应公式中的DType。<br>输入范围为{35, 36, 40, 41}，分别对应输出y1Out和y2Out的数据类型为{35:FLOAT8_E5M2, 36:FLOAT8_E4M3FN, 40:FLOAT4_E2M1, 41:FLOAT4_E1M2} | INT64 | - |
  | scaleAlg (int64_t) | 可选属性 | 表示mxscale1Out和mxscale2Out的计算方法。<br>当前仅支持取值0。后续支持取值0/1，0代表OCP实现，1代表cuBLAS实现。 | INT64 | - |
  | y1Out (aclTensor\*) | 输出 | 表示输入x量化-1轴后的对应结果，对应公式中的$P_i$。<br>shape和输入x一致。 | FLOAT4_E2M1、FLOAT4_E1M2、FLOAT8_E4M3FN、FLOAT8_E5M2 | ND |
  | mxscale1Out (aclTensor*) | 输出 | 表示-1轴每个分组对应的量化尺度，对应公式中的mxscale1。<br>shape为x的-1轴的值除以32向上取整，并对其进行偶数pad，pad填充值为0。 | FLOAT8_E8M0 | ND |
  | y2Out (aclTensor\*) | 输出 | 表示输入x量化-2轴后的对应结果，对应公式中的$P_j$。<br>shape和输入x一致。 |  FLOAT4_E2M1、FLOAT4_E1M2、FLOAT8_E4M3FN、FLOAT8_E5M2 | ND |
  | mxscale2Out (aclTensor*) | 输出 | 表示-2轴每个分组对应的量化尺度，对应公式中的mxscale2。 <br>shape为x的-2轴的值除以32向上取整，并对其进行偶数pad，pad填充值为0； <br>mxscale2Out输出需要对每两行数据进行交织处理。| FLOAT8_E8M0 | ND |

## 约束说明

 - 关于x、mxscale1Out、mxscale2Out的shape约束说明如下：
    - x的维度应该大于等于2。
    - rank(mxscale1Out) = rank(x) + 1。
    - rank(mxscale2Out) = rank(x) + 1。
    - mxscale1Out.shape[-2] = (ceil(x.shape[-1] / 32) + 2 - 1) / 2。
    - mxscale2Out.shape[-3] = (ceil(x.shape[-2] / 32) + 2 - 1) / 2。
    - mxscale1Out.shape[-1] = 2。
    - mxscale2Out.shape[-1] = 2。
    - 其他维度与输入x一致。
    - 举例：输入x的shape为[B, M, N]，目的数据类型为FP8类时，对应的y1和y2的shape为[B, M, N]，mxscale1的shape为[B, M, (ceil(N/32)+2-1)/2, 2]，mxscale2的shape为[B, (ceil(M/32)+2-1)/2, N, 2]。

## 调用说明

| 调用方式   | 样例代码           | 说明                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn接口  | [test_aclnn_dynamic_mx_quant_with_dual_axis](examples/arch35/test_aclnn_dynamic_mx_quant_with_dual_axis.cpp) | 通过[aclnnDynamicMxQuantWithDualAxis](docs/aclnnDynamicMxQuantWithDualAxis.md)接口方式调用DynamicMxQuantWithDualAxis算子。 |
| 图模式 | -  | 通过[算子IR](op_graph/dynamic_mx_quant_with_dual_axis_proto.h)构图方式调用DynamicMxQuantWithDualAxis算子。         |
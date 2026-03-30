# DynamicBlockMxQuant

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term>                             |    √     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    ×     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> |    ×     |
| <term>Atlas 200I/500 A2 推理产品</term>                      |    ×     |
| <term>Atlas 推理系列产品</term>                             |    ×     |
| <term>Atlas 训练系列产品</term>                              |    ×     |

## 功能说明

- 算子功能：对输入变量，以数据块（32\*32）为基本块进行MX量化转换为目的数据类型。在每个基本块中，根据scale_alg的取值采取不同的scale算法计算出当前块对应的量化参数scale（1\*1），将其广播为scale1（32\*1）和scale2（1\*32）输出。同时对基本块中的每一个数除以scale，根据round_mode转换到对应的dst_type，得到量化结果y。

- 计算公式：
    - 场景1，当scaleAlg为0时：
      - 将输入x以数据块（32\*32）为基本块进行分组，一个数据块的数 $\{\{V_i\}_{i=1}^{32*32}\}$ 量化为 $\{scale, \{P_i\}_{i=1}^{32*32}\}$

        $$
        shared\_exp = floor(log_2(max_i(|V_i|))) - emax
        $$

        $$
        scale = 2^{shared\_exp}
        $$

        $$
        P_i = cast\_to\_dst\_type(V_i/scale, round\_mode), \space i\space from\space 1\space to\space 32*32
        $$

      - 同时将 scale（1\*1）广播为scale1（32\*1）和scale2（1\*32）作为输出scale1和scale2，​量化后的 $P_{i}$ 按对应的 $V_{i}$ 的位置组成输出y。

      - emax: 对应数据类型的最大正则数的指数位。

        |   DataType    | emax |
        | :-----------: | :--: |
        |  FLOAT4_E2M1  |  2   |
        |  FLOAT4_E1M2  |  0   |
        | FLOAT8_E4M3FN |  8   |
        |  FLOAT8_E5M2  |  15  |

    - 场景2，当scaleAlg为2时，只涉及FLOAT4_E2M1类型：
      - 将长向量按块分，每块长度为k，对每块单独计算一个块缩放因子$S_{fp32}^b$，再把块内所有元素用同一个$S_{fp32}^b$映射到目标低精度类型FP8。如果最后一块不足k个元素，把缺失值视为0，按照完整块处理。
      - 找到该块中数值的最大绝对值:
      $$
      Amax(D_{fp32}^b)=max(\{|d_{i}|\}_{i=1}^{k})
      $$
      - 引入新的属性 dst_type_max。dst_type_max类型为float，默认值为0。默认输出时代表max_type为目标数据类型的最大值，如果传入其他数值，则需要按照传入的数值计算scale，有效值当前支持0.0、6.0和7.0，只支持在FLOAT4_E2M1场景设置该值。
      - 将FP32映射到目标数据类型FLOAT4_E2M1可表示的范围内。
      $$
      S_{fp32}^b = \frac{Amax(D_{fp32}^b)}{dst\_type\_max}
      $$
      - 将块缩放因子$S_{fp32}^b$转换为FP8格式下可表示的缩放值$S_{ue8m0}^b$
      - 从块的浮点缩放因子$S_{fp32}^b$中提取无偏指数$E_{int}^b$和尾数$M_{fixp}^b$
      - 为保证量化时不溢出，对指数进行向上取整，且在FP8可表示的范围内：
      $$
      E_{int}^b = \begin{cases} E_{int}^b + 1, & \text{如果} S_{fp32}^b \text{为正规数，且} E_{int}^b < 254 \text{且} M_{fixp}^b > 0 \\ E_{int}^b, & \text{其余情况} \end{cases}
      $$
      - 计算块缩放因子：$S_{ue8m0}^b=2^{E_{int}^b}$
      - 计算块转换因子：$R_{fp32}^b=\frac{1}{fp32(S_{ue8m0}^b)}$
      - 应用到量化的最终步骤，对于每个块内元素，$d^i = DType(d_{fp32}^i \cdot R_{fp32}^n)$，最终输出的量化结果是$\left(S^b, [d^i]_{i=1}^k\right)$，其中$S^b$代表块的缩放因子，这里指$S_{ue8m0}^b$，$[d^i]_{i=1}^k$代表块内量化后的数据。

## 参数说明

  | 参数名 | 输入/输出/属性 | 描述 | 数据类型 | 数据格式 |  
  | ----- | ----- |----- |----- |----- |
  | x | 输入 | 表示输入x，对应公式中$V_i$。<br>当dst_type为FLOAT4_E2M1、FLOAT4_E1M2时，x的最后一维必须是偶数。 | FLOAT16、BFLOAT16 | ND |
  | round_mode  | 可选属性 | 表示数据转换的模式，对应公式中的round_mode。<br>当dst_type为40/41，对应输出y的数据类型为FLOAT4_E2M1/FLOAT4_E1M2时，支持{"rint", "floor", "round"}；<br>当dst_type为35/36，对应输出y的数据类型为FLOAT8_E5M2/FLOAT8_E4M3FN时，仅支持{"rint"}；<br>传入空指针时，采用"rint"模式。 | STRING | - |
  | dst_type | 输入 | 表示指定数据转换后y的类型。<br>输入范围为{35, 36, 40, 41}，分别对应输出y1和y2的数据类型为{35:FLOAT8_E5M2, 36:FLOAT8_E4M3FN, 40:FLOAT4_E2M1, 41:FLOAT4_E1M2} | INT64 | - |
  | scale_alg | 输入 | 表示scale的计算方法。<br>当前仅支持取值0和2，分别代表OCP Microscaling Formats (Mx) Specification和Dynamic Dtype Range实现。 | INT64 | - |
  | dst_type_max | 输入 | 表示max_type为目标数据类型的最大值。<br>在scale_alg=2，dst_type为FLOAT4_E2M1时生效，需要按照传入的数值计算scale。<br>有效值当前支持取值为0.0/6.0/7.0，只支持在FLOAT4_E2M1场景设置该值。 | FLOAT | - |
  | y | 输出 | 表示输入x量化后的对应结果，对应公式中的$P_i$。<br>shape和输入x一致。 | FLOAT4_E2M1、FLOAT4_E1M2、FLOAT8_E4M3FN、FLOAT8_E5M2 | ND |
  | scale1 | 输出 | 表示-1轴每个分组对应的量化尺度，对应公式中的scale广播的scale1。<br>shape为x的-1轴的值除以32向上取整，并对其进行偶数pad，pad填充值为0。 | FLOAT8_E8M0 | ND |
  | scale2 | 输出 | 表示-2轴每个分组对应的量化尺度，对应公式中的scale广播的scale2。 <br>shape为x的-2轴的值除以32向上取整，并对其进行偶数pad，pad填充值为0； <br>scale2输出需要对每两行数据进行交织处理。| FLOAT8_E8M0 | ND |

## 约束说明

 - 关于x、scale1、scale2的shape约束说明如下：
    - x的维度应该为2~3。
    - rank(scale1) = rank(x) + 1。
    - rank(scale2) = rank(x) + 1。
    - scale1.shape[-2] = (ceil(x.shape[-1] / 32) + 2 - 1) / 2。
    - scale2.shape[-3] = (ceil(x.shape[-2] / 32) + 2 - 1) / 2。
    - scale1.shape[-1] = 2。
    - scale2.shape[-1] = 2。
    - 其他维度与输入x一致。
    - 举例：输入x的shape为[B, M, N]，目的数据类型为FP8类时，对应的y的shape为[B, M, N]，scale1的shape为[B, M, ceil(N/64), 2]，scale2的shape为[B, ceil(M/64), N, 2]。

## 调用说明

| 调用方式   | 样例代码           | 说明                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn接口  | [test_aclnn_dynamic_block_mx_quant](examples/arch35/test_aclnn_dynamic_block_mx_quant.cpp) | 通过[aclnnDynamicBlockMxQuant](docs/aclnnDynamicBlockMxQuant.md)接口方式调用DynamicBlockMxQuant算子。 |
| 图模式 | -  | 通过[算子IR](op_graph/dynamic_block_mx_quant_proto.h)构图方式调用DynamicBlockMxQuant算子。         |

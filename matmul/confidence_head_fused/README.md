# ConfidenceHeadFused

## 产品支持情况

| 产品 | 是否支持 |
| :--- | :---: |
| <term>Ascend 950PR&950DT系列产品</term> | √ |
| <term>Atlas A3系列产品</term> | × |
| <term>Atlas A2系列产品</term> | × |
| <term>Atlas 200I/500 A2推理产品</term> | × |
| <term>Atlas推理系列产品</term> | × |
| <term>Atlas训练系列产品</term> | × |

## 功能说明

- 算子功能：将hidden与可选的markov_emb沿最后一维逻辑拼接，与weight做矩阵向量乘，加上可选的bias，经temperature缩放、clamp_low与clamp_high截断和Sigmoid计算后，输出每行的置信度。矩阵乘累加、后处理及输出的数据类型为FLOAT32。
- 计算公式：

  以下以三维输入为例；二维输入省略batch下标$b$。$K_1$为hidden的最后一维长度。markov_emb缺省时$K_2=0$，否则$K_2$为markov_emb的最后一维长度。weight的长度为$K_1+K_2$。

  1. 沿最后一维逻辑拼接后，计算矩阵向量乘并加偏置：

     $$
     z_{b,n}=\sum_{k=0}^{K_1-1}\mathrm{hidden}_{b,n,k}\,\mathrm{weight}_k
     +\sum_{k=0}^{K_2-1}\mathrm{markov\_emb}_{b,n,k}\,\mathrm{weight}_{K_1+k}+p_n
     $$

     $K_2=0$时第二项为$0$。bias缺省时$p_n=0$；传入bias时$p_n=\mathrm{bias}_n$，各batch共用同一个$p_n$。

  2. 按温度缩放，得到指数函数的输入：

     $$
     T=\begin{cases}
     1, & \mathrm{temperature}=\mathrm{None}\text{或}\mathrm{temperature}[0]=0,\\
     \mathrm{temperature}[0], & \text{其他情况},
     \end{cases}
     \qquad u_{b,n}=-\frac{z_{b,n}}{T}
     $$

  3. 先按clamp_low处理下限，再按clamp_high处理上限，最后计算Sigmoid。设$C=\operatorname{FP32}(\mathrm{INT32\_MAX})=2147483648.0$：

     $$
     \begin{aligned}
     L&=\begin{cases}
     -C, & \mathrm{clamp\_low}=\mathrm{None},\\
     \operatorname{FP32}(\mathrm{clamp\_low}), & \text{其他情况},
     \end{cases}\\
     H&=\begin{cases}
     C, & \mathrm{clamp\_high}=\mathrm{None},\\
     \operatorname{FP32}(\mathrm{clamp\_high}), & \text{其他情况}
     \end{cases}
     \end{aligned}
     $$

     $$
     \begin{aligned}
     v_{b,n}&=\begin{cases}
     u_{b,n}, & L=-C,\\
     \max(u_{b,n},L), & L\ne-C,
     \end{cases}\\
     w_{b,n}&=\begin{cases}
     v_{b,n}, & H=C,\\
     \min(v_{b,n},H), & H\ne C,
     \end{cases}\\
     \mathrm{out}_{b,n}&=\frac{1}{1+\exp(w_{b,n})}
     \end{aligned}
     $$

     显式传入的clamp_low等于$-C$或clamp_high等于$C$时，算子也会跳过对应的截断。

## 参数说明

| 参数名 | 输入/输出/属性 | 描述 | 数据类型 | 数据格式 |
| --- | --- | --- | --- | --- |
| hidden | 输入 | 特征，shape为 [N,K1] 或 [B,N,K1]。 | FLOAT16、BFLOAT16 | ND |
| weight | 输入 | 所有行共用的投影权重，shape为 [K1+K2]，类型与hidden一致。 | FLOAT16、BFLOAT16 | ND |
| markov_emb | 可选输入 | shape为 [N,K2] 或 [B,N,K2]，前导维度、类型与hidden一致。缺省时K2=0。 | FLOAT16、BFLOAT16 | ND |
| temperature | 可选输入 | shape为 [1]；缺省或当前值为正负零时按1计算。 | FLOAT32 | ND |
| bias | 可选输入 | shape为 [N]，在batch间广播。 | FLOAT32 | ND |
| clamp_low | 可选属性 | 指数输入下限；未传入或转为FLOAT32后等于-2147483648.0时不做下限截断。 | Float | - |
| clamp_high | 可选属性 | 指数输入上限；未传入或转为FLOAT32后等于2147483648.0时不做上限截断。 | Float | - |
| out | 输出 | shape为 [N] 或 [B,N]。 | FLOAT32 | ND |

## 约束说明

- 支持合法基础格式和非连续Tensor，不支持FRACTAL_NZ。
- hidden、weight、markov_emb的数据类型须同为FLOAT16或同为BFLOAT16；temperature、bias和out为FLOAT32。weight的长度为K1+K2，markov_emb的前导维度与hidden一致，bias的shape为[N]，out的shape与hidden去掉最后一维后的shape一致。
- 各维度、B*N和K1+K2须位于[0,UINT32_MAX]。二维输入按B=1理解。
- K1+K2=0时仍执行bias、temperature、可选截断和Sigmoid计算；B=0或N=0时输出为空。

## 调用说明

- <term>Ascend 950PR&950DT系列产品</term>：

  | 调用方式 | 样例代码 | 说明 |
  | --- | --- | --- |
  | aclnn接口 | [test_aclnn_confidence_head_fused](examples/arch35/test_aclnn_confidence_head_fused.cpp) | 通过<br>[aclnnConfidenceHeadFused](docs/aclnnConfidenceHeadFused.md)<br>接口调用ConfidenceHeadFused算子。 |
  | PyTorch API | [调用示例](docs/torchapi_confidence_head_fused.md#调用示例) | 通过<br>[confidence_head_fused](docs/torchapi_confidence_head_fused.md)<br>接口调用ConfidenceHeadFused算子。 |

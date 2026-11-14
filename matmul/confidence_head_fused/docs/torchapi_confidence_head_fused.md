# confidence_head_fused

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

- 接口功能：将hidden与可选的markov_emb沿最后一维逻辑拼接，与weight做矩阵向量乘，加上可选的bias，经temperature缩放、可选截断和Sigmoid计算后，输出每行的置信度。矩阵乘累加、后处理及输出的数据类型为FLOAT32。
- 计算公式：

  以下以三维输入为例；二维输入省略batch下标$b$。$K_1$为hidden的最后一维长度。markov_emb为None时$K_2=0$，否则$K_2$为markov_emb的最后一维长度。weight的长度为$K_1+K_2$。

  1. 沿最后一维逻辑拼接后，计算矩阵向量乘并加偏置：

     $$
     z_{b,n}=\sum_{k=0}^{K_1-1}\mathrm{hidden}_{b,n,k}\,\mathrm{weight}_k
     +\sum_{k=0}^{K_2-1}\mathrm{markov\_emb}_{b,n,k}\,\mathrm{weight}_{K_1+k}+p_n
     $$

     $K_2=0$时第二项为0。bias为None时$p_n=0$；传入bias时$p_n=\mathrm{bias}_n$，并在各batch共用。

  2. 按温度缩放，得到指数函数的输入：

     $$
     T=\begin{cases}
     1, & \mathrm{temperature}=\mathrm{None}\text{或}\mathrm{temperature}[0]=0,\\
     \mathrm{temperature}[0], & \text{其他情况},
     \end{cases}
     \qquad u_{b,n}=-\frac{z_{b,n}}{T}
     $$

  3. 先处理下限、再处理上限，最后计算Sigmoid：

     $$
     \begin{aligned}
     v_{b,n}&=\begin{cases}u_{b,n},&L=-C,\\\max(u_{b,n},L),&L\ne-C,\end{cases}\\
     w_{b,n}&=\begin{cases}v_{b,n},&H=C,\\\min(v_{b,n},H),&H\ne C,\end{cases}\\
     \mathrm{out}_{b,n}&=\frac{1}{1+\exp(w_{b,n})}
     \end{aligned}
     $$

     $C=\operatorname{FP32}(\mathrm{INT32\_MAX})=2147483648.0$。clamp_low为None时$L=-C$；传入clamp_low时$L$为其FLOAT32值。clamp_high为None时$H=C$；传入clamp_high时$H$为其FLOAT32值。显式传入约定值也会跳过对应的截断。

## 函数原型

```python
cann_ops_nn.confidence_head_fused(
    hidden,
    weight,
    *,
    markov_emb=None,
    temperature=None,
    bias=None,
    clamp_low=None,
    clamp_high=None,
) -> Tensor
```

## 参数说明

| 参数名 | 参数类型 | 可选/必选 | 描述 | 数据类型 | 维度（shape） |
| --- | --- | --- | --- | --- | --- |
| hidden | Tensor | 必选 | 待投影的特征，最后一维为K1。 | torch.float16、torch.bfloat16 | (N, K1) 或 (B, N, K1) |
| weight | Tensor | 必选 | 所有行与batch共用的一维投影权重，长度为K1+K2。 | 与hidden一致 | (K1+K2,) |
| markov_emb | Tensor | 可选 | 沿最后一维与hidden逻辑拼接；前导维度与hidden一致。默认值为None，此时K2=0。 | 与hidden一致 | (N, K2) 或 (B, N, K2) |
| temperature | Tensor | 可选 | 温度系数，用于确定公式中的T。为None或当前值为正负零时按1计算。 | torch.float32 | (1,) |
| bias | Tensor | 可选 | 偏置，用于确定公式中的p_n。bias为None时不加偏置；传入bias时在各batch按N维共用。 | torch.float32 | (N,) |
| clamp_low | float | 可选 | 指数输入下限，对应公式中的L。clamp_low为None或转换为FLOAT32后等于-2147483648.0时不做下限截断。 | - | - |
| clamp_high | float | 可选 | 指数输入上限，对应公式中的H。clamp_high为None或转换为FLOAT32后等于2147483648.0时不做上限截断。 | - | - |

## 返回值说明

| 输出名 | 输出类型 | 描述 | 数据类型 | 维度(shape) |
| --- | --- | --- | --- | --- |
| out | Tensor | 每行的置信度，shape为hidden去除最后一维。 | torch.float32 | (N,) 或 (B, N) |

## 约束说明

- 该接口支持Torch单算子模式调用，所有Tensor须在同一NPU设备。
- 支持合法基础格式和非连续Tensor，不支持FRACTAL_NZ。
- hidden、weight、markov_emb的数据类型须同为FLOAT16或同为BFLOAT16；temperature、bias和out为FLOAT32。weight的长度为K1+K2，markov_emb的前导维度与hidden一致，bias的shape为[N]，out的shape与hidden去掉最后一维后的shape一致。
- 各维度、B*N和K1+K2须位于[0,UINT32_MAX]。二维输入按B=1理解。
- K1+K2=0时仍执行bias、temperature、可选截断和Sigmoid计算；B=0或N=0时输出为空。

## 确定性计算

默认支持确定性计算。

## 调用示例

- 单算子模式调用（eager）

  ```python
  import torch
  import torch_npu
  import cann_ops_nn

  hidden = torch.randn((2, 129, 33), dtype=torch.float16).npu()
  weight = torch.randn((40,), dtype=torch.float16).npu()
  markov_emb = torch.randn((2, 129, 7), dtype=torch.float16).npu()
  temperature = torch.tensor([2.0], dtype=torch.float32).npu()
  bias = torch.randn((129,), dtype=torch.float32).npu()

  out = cann_ops_nn.confidence_head_fused(
      hidden,
      weight,
      markov_emb=markov_emb,
      temperature=temperature,
      bias=bias,
      clamp_low=-60.0,
      clamp_high=60.0,
  )
  print(out.shape, out.dtype)
  ```

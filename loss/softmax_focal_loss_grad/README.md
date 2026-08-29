# SoftmaxFocalLossGrad

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term>                       |    √     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    √     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>     |    √     |
| <term>Atlas 200I/500 A2 推理产品</term>                      |    ×     |
| <term>Atlas 推理系列产品</term>                              |    ×     |
| <term>Atlas 训练系列产品</term>                              |    √     |

## 功能说明

- 算子功能：计算SoftmaxFocalLoss的反向梯度。输入pred为已完成softmax的概率，输出对pred的梯度。

- 计算公式：

  设pred为$p$（shape为$[N, C]$），target为$t$，dout为$\text{d}$，weight为$w$（缺省时按全1处理），对第$n$个样本先求四个行标量：

  $$
  \text{WF}_n = \sum_c \alpha (1-p_{n,c})^{\gamma} t_{n,c}, \quad
  \text{WB}_n = \sum_c \alpha (1-p_{n,c})^{\gamma-1} t_{n,c}
  $$

  $$
  \text{CE}_n = \sum_c -\log(p_{n,c}) t_{n,c} w_{n,c}, \quad
  \text{W}_n = \sum_c w_{n,c} t_{n,c}
  $$

  再逐元素组合：

  $$
  \text{dce}_{n,c} = p_{n,c}\text{W}_n - t_{n,c}w_{n,c}
  $$

  $$
  \text{dwf}_{n,c} = -\gamma \cdot p_{n,c} \cdot \big[(\text{WF}_n - \text{WB}_n) + \alpha (1-p_{n,c})^{\gamma-1} t_{n,c}\big]
  $$

  $$
  \text{grad}_{n,c} = \big(\text{dce}_{n,c}\text{WF}_n + \text{dwf}_{n,c}\text{CE}_n\big) \cdot \text{d}_{n,c} \cdot \text{coef}
  $$

  其中：

  - $N$为batch数，$C$为类别数，归约轴为最后一维。
  - $\alpha$、$\gamma$分别为属性alpha、gamma。
  - reduction取"mean"时$\text{coef} = 1/(N \times C)$，取"none"或"sum"时$\text{coef} = 1$。
  - $(1-p)^{\gamma}$按$\exp(\gamma \log(1-p))$求值。

## 参数说明

| 参数名 | 输入/输出/属性 | 描述                                                         | 数据类型        | 数据格式 |
| ------ | -------------- | ------------------------------------------------------------ | --------------- | -------- |
| pred   | 输入           | 前级softmax输出的概率，对应公式中的$p$，取值应落在(0, 1)开区间。 | FLOAT16、FLOAT  | ND       |
| target | 输入           | one-hot标签，对应公式中的$t$，shape与pred一致。               | INT32           | ND       |
| dout   | 输入           | 上游传入的梯度，对应公式中的$\text{d}$，shape与数据类型均与pred一致。 | FLOAT16、FLOAT  | ND       |
| weight | 可选输入       | 逐元素权重，对应公式中的$w$，shape与数据类型均与pred一致。不传入时按全1处理，该行为仅<term>Ascend 950PR/Ascend 950DT</term>支持，其余产品需显式传入weight。 | FLOAT16、FLOAT  | ND       |
| alpha  | 属性           | 调制因子的权重系数，对应公式中的$\alpha$，缺省值为0.25。      | FLOAT           | -        |
| gamma  | 属性           | 调制因子的指数，对应公式中的$\gamma$，缺省值为2.0。           | FLOAT           | -        |
| reduction | 属性        | 取值为"none"、"mean"、"sum"之一（大小写不敏感），缺省值为"mean"，传入其他取值会报错。取"mean"时梯度乘以$1/\text{numel(pred)}$，"none"与"sum"不缩放。 | STRING          | -        |
| grad   | 输出           | pred的梯度，shape与数据类型均与pred一致。                     | FLOAT16、FLOAT  | ND       |

## 约束说明

- pred、dout、weight、grad的数据类型保持一致；target的数据类型固定为INT32。
- target、dout、weight的shape必须与pred一致。
- pred各维长度必须大于0，不支持空Tensor。
- 归约轴固定为最后一维。
- gamma小于1且pred趋近1时，$(1-p)^{\gamma-1}$发散，该行为与算法定义一致，不做拦截。

## 调用说明

| 调用方式 | 样例代码 | 说明 |
| -------- | -------- | ---- |
| 图模式调用 | [test_geir_softmax_focal_loss_grad.cpp](examples/test_geir_softmax_focal_loss_grad.cpp) | 通过[算子IR](./op_graph/softmax_focal_loss_grad_proto.h)构图方式调用SoftmaxFocalLossGrad算子。 |

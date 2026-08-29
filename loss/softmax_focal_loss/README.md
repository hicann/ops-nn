# SoftmaxFocalLoss

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term>                       |    √     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    √     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>     |    √     |
| <term>Atlas 200I/500 A2 推理产品</term>                      |    √     |
| <term>Atlas 推理系列产品</term>                              |    √     |
| <term>Atlas 训练系列产品</term>                              |    √     |

## 功能说明

- 算子功能：计算多分类场景下的Softmax Focal Loss。Focal Loss在交叉熵的基础上引入调制因子$(1-p)^\gamma$，降低易分样本的损失权重，使训练更关注难分样本。输入pred为已完成softmax的概率，本算子不再内置softmax。

- 计算公式：

  设pred为$p$（shape为$[N, C]$），target为$t$（与$p$同shape的one-hot标签），weight为$w$（与$p$同shape，缺省时按全1处理），对第$n$个样本：

  $$
  \text{CE}_n = \sum_{c=0}^{C-1} -\log(p_{n,c}) \cdot t_{n,c} \cdot w_{n,c}
  $$

  $$
  \text{FW}_n = \sum_{c=0}^{C-1} \alpha \cdot \exp\big(\gamma \cdot \log(1 - p_{n,c})\big) \cdot t_{n,c}
  $$

  $$
  y_{n,c} = \text{CE}_n \cdot \text{FW}_n
  $$

  其中：

  - $N$为batch数，$C$为类别数，归约轴为最后一维。
  - $\alpha$、$\gamma$分别为属性alpha、gamma。
  - $y$与$p$同shape，同一行内各元素取值相同。

## 参数说明

| 参数名 | 输入/输出/属性 | 描述                                                         | 数据类型        | 数据格式 |
| ------ | -------------- | ------------------------------------------------------------ | --------------- | -------- |
| pred   | 输入           | 前级softmax输出的概率，对应公式中的$p$，取值应落在(0, 1)开区间。 | FLOAT16、FLOAT  | ND       |
| target | 输入           | one-hot标签，对应公式中的$t$，shape与pred一致。               | INT32           | ND       |
| weight | 可选输入       | 逐元素权重，对应公式中的$w$，shape与pred一致；不传入时按全1处理。数据类型可与pred不同。 | FLOAT16、FLOAT  | ND       |
| gamma  | 属性           | 调制因子的指数，对应公式中的$\gamma$，缺省值为2.0。           | FLOAT           | -        |
| alpha  | 属性           | 调制因子的权重系数，对应公式中的$\alpha$，缺省值为0.25。      | FLOAT           | -        |
| reduction | 属性        | 缺省值为"none"，当前也仅支持"none"，传入其他取值会报错。 | STRING          | -        |
| y      | 输出           | 损失值，shape与pred一致，同一行内各元素取值相同。数据类型与pred一致。 | FLOAT16、FLOAT  | ND       |

## 约束说明

- pred与y的数据类型保持一致；weight的数据类型可独立于pred取FLOAT16或FLOAT。
- target、weight的shape必须与pred一致，target的数据类型固定为INT32。
- pred各维长度必须大于0，不支持空Tensor。
- target应为one-hot编码：本算子不校验target的取值，非one-hot时其数值会作为权重直接参与行内求和，结果不再是Focal Loss的定义值。
- 归约轴固定为最后一维：pred的最后一维为类别维，其余维为样本维。<term>Ascend 950PR/Ascend 950DT</term>支持任意rank≥1的pred，其余产品仅支持二维形式(batch_size, num_classes)。
- reduction属性当前仅支持"none"：本算子的输出shape恒等于pred，无法承载"mean"/"sum"所需的标量结果，故传入"none"以外的取值（大小写不敏感）直接报错。缺省值取"none"，不显式传入该属性时按"none"处理。

## 调用说明

| 调用方式 | 样例代码 | 说明 |
| -------- | -------- | ---- |
| 图模式调用 | [test_geir_softmax_focal_loss.cpp](examples/test_geir_softmax_focal_loss.cpp) | 通过[算子IR](./op_graph/softmax_focal_loss_proto.h)构图方式调用SoftmaxFocalLoss算子。 |

# KlDivLossGrad

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR&950DT系列产品</term>                      |    √     |
| <term>Atlas A3系列产品</term>                                |    √     |
| <term>Atlas A2系列产品</term>                                |    √     |
| <term>Atlas 200I/500 A2推理产品</term>                       |    ×     |
| <term>Atlas推理系列产品</term>                               |    ×     |
| <term>Atlas训练系列产品</term>                               |    ×     |

## 功能说明

- 算子功能：计算KL散度损失对输入input的梯度。

- 计算公式：

  - 当log_target取值为false时：

    $$
    y = -target \times grad
    $$

  - 当log_target取值为true时：

    $$
    y = -e^{target} \times grad
    $$

  其中：

  - grad为反向传播的梯度输入。
  - target为真实的标签。
  - y为计算输出的梯度。
  - log_target表示target是否已处于log空间。

- reduction缩放说明：

  - 'none'：不做缩放。
  - 'sum'：不做缩放。
  - 'mean'：输出除以output中的元素总数。
  - 'batchmean'：输出除以batch的个数，即input第0维的大小。

## 参数说明

| 参数名     | 输入/输出/属性 | 描述                                                         | 数据类型                 | 数据格式 |
| :--------- | :------------- | :----------------------------------------------------------- | :----------------------- | :------- |
| grad       | 输入           | 反向传播的梯度输入，对应公式中的grad。                         | BFLOAT16、FLOAT16、FLOAT32    | ND       |
| input      | 输入           | 输入张量，计算梯度的目标，对应公式中的input。                  | BFLOAT16、FLOAT16、FLOAT32    | ND       |
| target     | 输入           | 真实的标签，对应公式中的target。                               | BFLOAT16、FLOAT16、FLOAT32    | ND       |
| y          | 输出           | 计算输出的梯度，对应公式中的y，shape与input相同。              | BFLOAT16、FLOAT16、FLOAT32    | ND       |
| reduction  | 可选属性       | 指定要应用到输出的缩减方式，默认值为"mean"，支持"none"、"mean"、"sum"、"batchmean"。 | STRING                   | -        |
| log_target | 可选属性       | 指定是否对target进行log空间转换，默认值为false。               | BOOL                     | -        |

## 约束说明

- input与target的shape需满足broadcast关系，grad的shape需能broadcast到input与target broadcast后的结果形状。
- grad、input、target、y的数据类型需保持一致。
- reduction取值为"mean"时，input不能为空Tensor，即input各维的长度均不能为0。
- reduction取值为"batchmean"时，input第0维的长度必须大于0。
- log_target取值为true时，计算公式包含exp(target)，当target取值过大时exp(target)会溢出，输出结果可能为inf或nan。

## 调用说明

| 调用方式  | 样例代码                                                     | 说明                                                         |
| :-------- | :----------------------------------------------------------- | :----------------------------------------------------------- |
| aclnn API | [test_aclnn_kl_div_loss_grad.cpp](examples/test_aclnn_kl_div_loss_grad.cpp) | 通过[aclnnKlDivBackward](docs/aclnnKlDivBackward.md)接口方式调用KlDivLossGrad算子。 |
| GE图模式  | -                                                            | 通过[算子IR](op_graph/kl_div_loss_grad_proto.h)构图方式调用KlDivLossGrad算子。 |

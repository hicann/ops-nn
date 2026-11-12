# L1LossGrad

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR&950DT系列产品</term>                      |    √     |
| <term>Atlas A3系列产品</term>                                |    √     |
| <term>Atlas A2系列产品</term>                                |    √     |
| <term>Atlas 200I/500 A2推理产品</term>                       |    ×     |
| <term>Atlas推理系列产品</term>                               |    √     |
| <term>Atlas训练系列产品</term>                               |    √     |

## 功能说明

- 算子功能：计算L1损失（平均绝对误差）对输入predict的梯度。

- 计算公式：

  $$
  y = sign(predict - label) \times grads
  $$

  其中sign(x)表示x的符号函数：
  $$
  sign(x) =\begin{cases}
  1,& x>0 \\
  0,& x=0 \\
  -1,& x<0
  \end{cases}
  $$

  reduction取自前向损失，指定损失的归约方式：'mean'将每个梯度元素除以元素总数；'none'和'sum'不做额外缩放。三种模式下输出y均保留grads、predict、label广播后的形状。

## 参数说明

| 参数名    | 输入/输出/属性 | 描述                                                         | 数据类型                 | 数据格式 |
| :-------- | :------------- | :----------------------------------------------------------- | :----------------------- | :------- |
| grads     | 输入           | 梯度反向输入，公式中的grads。                                 | BFLOAT16、FLOAT16、FLOAT32 | ND       |
| predict   | 输入           | 预测值，公式中的predict。                                     | BFLOAT16、FLOAT16、FLOAT32 | ND       |
| label     | 输入           | 真实标签，公式中的label。                                     | BFLOAT16、FLOAT16、FLOAT32 | ND       |
| y         | 输出           | 计算输出的梯度，公式中的y，shape为grads、predict、label广播后的结果。 | BFLOAT16、FLOAT16、FLOAT32 | ND       |
| reduction | 可选属性       | 指定要应用的缩减方式，默认值为"mean"，取值为"none"、"mean"、"sum"。 | STRING                   | -        |

## 约束说明

- 输入grads、predict、label与输出y的数据类型必须一致。
- 输入grads、predict、label的shape需满足broadcast关系，输出y的shape为三者broadcast之后的结果。
- 数据格式仅支持ND。
- 输入包含inf或nan时的输出行为未定义，本算子不做保证。

## 调用说明

| 调用方式  | 样例代码                                                     | 说明                                                         |
| :-------- | :----------------------------------------------------------- | :----------------------------------------------------------- |
| aclnn API | [test_aclnn_l1_loss_grad.cpp](examples/test_aclnn_l1_loss_grad.cpp) | 通过[aclnnL1LossBackward](docs/aclnnL1LossBackward.md)接口方式调用L1LossGrad算子。 |
| GE图模式  | -                                                            | 通过[算子IR](op_graph/l1_loss_grad_proto.h)（REG_OP(L1LossGrad)）构图方式调用L1LossGrad算子。 |

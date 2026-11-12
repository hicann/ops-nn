# SmoothL1LossGradV2

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

- 算子功能：计算Smooth L1损失对输入predict的梯度。

- 计算公式：

  Smooth L1损失的反向传播可以通过求导计算。对于|x-y|<sigma的情况，其导数为：
  $$
  \frac{\partial SmoothL1Loss(x,y)}{\partial x} = \frac{x-y}{sigma}
  $$

  对于|x-y|≥sigma的情况，其导数为：
  $$
  \frac{\partial SmoothL1Loss(x,y)}{\partial x} = sign(x - y)
  $$

  其中x为predict，y为label；sigma为L1与L2损失的分界阈值，该值必须是非负的。sign(x)表示x的符号函数：
  $$
  sign(x) =\begin{cases}
  1,& x>0 \\
  0,& x=0 \\
  -1,& x<0
  \end{cases}
  $$

## 参数说明

| 参数名    | 输入/输出/属性 | 描述                                                         | 数据类型                 | 数据格式 |
| :-------- | :------------- | :----------------------------------------------------------- | :----------------------- | :------- |
| predict   | 输入           | 公式中的x，预测值。                                           | BFLOAT16、FLOAT16、FLOAT32 | ND       |
| label     | 输入           | 公式中的y，真实标签。                                         | BFLOAT16、FLOAT16、FLOAT32 | ND       |
| dout      | 输入           | 梯度反向输入。                                                | BFLOAT16、FLOAT16、FLOAT32 | ND       |
| gradient  | 输出           | 计算输出的梯度，shape为predict、label、dout做broadcast后的结果。 | BFLOAT16、FLOAT16、FLOAT32 | ND       |
| sigma     | 可选属性       | L1与L2损失的分界阈值，默认值为1.0，必须是非负的。              | FLOAT32                  | -        |
| reduction | 可选属性       | 指定要应用的缩减方式，默认值为"mean"，取值为"none"、"mean"、"sum"。对反向梯度：'mean'将每个梯度元素除以元素总数；'none'和'sum'不做额外缩放。 | STRING                   | -        |

## 约束说明

- 输入predict、label、dout与输出gradient的数据类型必须一致。
- 输入predict、label、dout的shape必须满足broadcast关系，输出gradient的shape为三者broadcast之后的结果。
- 各输入/输出Tensor的维度（rank）上限为8；rank-0标量输入按{1}归一处理。
- 数据格式仅支持ND。
- FLOAT16、BFLOAT16输入在算子内部提升至FLOAT32计算后回写为输出数据类型。
- 输入包含NaN/Inf时，输出按浮点语义传播。

## 调用说明

| 调用方式  | 样例代码                                                     | 说明                                                         |
| :-------- | :----------------------------------------------------------- | :----------------------------------------------------------- |
| aclnn API | [test_aclnn_smooth_l1_loss_backward.cpp](examples/arch35/test_aclnn_smooth_l1_loss_backward.cpp) | 通过[aclnnSmoothL1LossBackward](docs/aclnnSmoothL1LossBackward.md)接口方式调用SmoothL1LossGradV2算子。 |
| GE图模式  | -                                                            | 通过[算子IR](op_graph/smooth_l1_loss_grad_v2_proto.h)（REG_OP(SmoothL1LossGradV2)）构图方式调用SmoothL1LossGradV2算子。 |

# SmoothL1LossGrad

## 产品支持情况

| 产品 | 是否支持 |
| :-------------------------|:----------:|
|  <term>Ascend 950PR/Ascend 950DT</term>   |     √    |
|  <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>   |     √    |
|  <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>     |     √    |
|  <term>Atlas 200I/500 A2 推理产品</term>    |     ×    |
|  <term>Atlas 推理系列产品</term>     |     ×    |
|  <term>Atlas 训练系列产品</term>    |     ×    |

## 功能说明

- 算子功能：计算SmoothL1Loss的反向传播梯度。reduction='none'，仅输出逐元素梯度；mean/sum归约由前向SmoothL1Loss承担。
- 计算公式：

  $$
  gradient = clamp\left(\frac{predict - label}{sigma},\ -1,\ 1\right) \times gradOutput
  $$

  其中sigma等价于PyTorch `smooth_l1_loss`的`beta`参数，控制L1与L2之间的平滑过渡阈值。当差值`predict - label = 0`时`gradient = 0`。

## 参数说明

|参数名|输入/输出/属性|描述|数据类型|数据格式|
|:-----|:-----------|:----|:---------|:------|
|predict|输入|预测值，对应公式中predict。|FLOAT16、FLOAT、BFLOAT16|ND|
|label|输入|目标值，对应公式中label。数据类型与predict保持一致。|FLOAT16、FLOAT、BFLOAT16|ND|
|gradOutput|输入|上游梯度，对应公式中gradOutput。数据类型与predict保持一致。|FLOAT16、FLOAT、BFLOAT16|ND|
|gradient|输出|输出梯度，对应公式中gradient。数据类型与predict保持一致。|FLOAT16、FLOAT、BFLOAT16|ND|
|sigma|属性|Smooth L1平滑阈值，对应公式中sigma，等价PyTorch的beta。取值需大于0，默认1.0。|FLOAT|-|

## 约束说明

- 确定性计算：算子为纯逐元素运算，默认确定性实现。
- predict、label、gradOutput三者的数据类型必须一致，且均为FLOAT16、FLOAT、BFLOAT16之一；gradient的数据类型与predict一致。
- predict、label、gradOutput、gradient四者的shape必须一致（当前不支持broadcast），支持0-8维。
- sigma取值必须大于0（默认1.0），不实现sigma≤0降级为L1 Loss backward的语义。

## 调用说明

| 调用方式 | 调用样例 | 说明 |
| ---------------- | --------------------------- |-------------------------------------------------------------------------|
| GE图模式  | [test_geir_smooth_l1_loss_grad.cpp](examples/arch35/test_geir_smooth_l1_loss_grad.cpp)   | 通过[算子IR](op_graph/smooth_l1_loss_grad_proto.h)构图方式调用SmoothL1LossGrad算子。 |

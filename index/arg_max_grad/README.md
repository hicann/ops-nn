# ArgMaxGrad

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

- 算子功能：ArgMax的反向。沿`dimension`指定的轴，把`updates`写入`indices`指定的位置，其余位置保留`var`的原值。

- 计算公式：

  设$var$的shape为$(d_0, ..., d_{m-1})$，$a$为归一化后的`dimension`，$D=d_a$。$indices$与$updates$的shape为$var$的形状把第$a$维置1，$k$为元素在第$a$维上的下标，则：

  $$
  y[i_0,...,k,...,i_{m-1}] = \begin{cases} updates[i_0,...,0,...,i_{m-1}], & k = indices[i_0,...,0,...,i_{m-1}] \\ var[i_0,...,k,...,i_{m-1}], & \text{其他} \end{cases}
  $$

  其中：

  - `dimension`为负数时按$dimension + m$归一化。
  - `indices`的取值不在$[0, D)$范围内时，该位置在整条轴上都不命中，输出保留`var`的原值。
  - 本算子只做条件选择，不涉及任何算术运算，`var`与`updates`的值按位原样搬运。

## 参数说明

| 参数名 | 输入/输出/属性 | 描述                                                         | 数据类型        | 数据格式 |
| ------ | -------------- | ------------------------------------------------------------ | --------------- | -------- |
| var    | 输入           | 被写基底，输出`y`与其同形同数据类型。                          | FLOAT16、FLOAT、INT32、INT8 | ND |
| indices | 输入          | 沿`dimension`轴的目标下标，shape为`var`的形状把第`dimension`维置1。 | INT32     | ND |
| updates | 输入          | 要写入的值，shape与`indices`一致，数据类型与`var`一致。         | FLOAT16、FLOAT、INT32、INT8 | ND |
| y      | 输出           | 命中位置取`updates`，其余保留`var`，shape与数据类型均与`var`一致。 | FLOAT16、FLOAT、INT32、INT8 | ND |
| dimension | 属性        | 必选，指定沿哪个轴比较，取值范围为$[-m, m-1]$，$m$为`var`的维度数。 | INT       | -    |

## 约束说明

- `indices`与`updates`的shape必须完全一致。
- `indices`与`updates`沿`dimension`轴的长度必须为1，即不支持PyTorch`scatter`的一般形态：`var`的shape为$[2,3,4,5]$、`dimension`为2时，`indices`与`updates`只能是$[2,3,1,5]$（等价$[2,3,5]$），不支持$[2,3,2,5]$。
- `updates`的数据类型必须与`var`一致；`indices`的数据类型固定为INT32。
- 元素在`dimension`轴上的下标由算子内部按`dimension`与`var`的shape生成，无需用户构造，也不作为输入传入。

## 调用说明

| 调用方式 | 样例代码 | 说明 |
| -------- | -------- | ---- |
| 图模式调用 | [test_geir_arg_max_grad.cpp](examples/test_geir_arg_max_grad.cpp) | 通过[算子IR](./op_graph/arg_max_grad_proto.h)构图方式调用ArgMaxGrad算子。 |

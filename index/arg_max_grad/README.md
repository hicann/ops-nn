# ArgMaxGrad

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR&950DT系列产品</term>                       |    √     |
| <term>Atlas A3系列产品</term>    |    √     |
| <term>Atlas A2系列产品</term>     |    √     |
| <term>Atlas 200I/500 A2推理产品</term>                      |    ×     |
| <term>Atlas推理系列产品</term>                              |    ×     |
| <term>Atlas训练系列产品</term>                              |    √     |

## 功能说明

- 算子功能：ArgMax的反向。沿`dimension`指定的轴，把`updates`写入`indices`指定的位置，其余位置保留`var`的原值。

- 计算公式：

  设$var$的shape为$(d_0, ..., d_{m-1})$，$a$为归一化后的`dimension`，$D=d_a$。$indices$与$updates$的shape为$var$去掉第$a$维后的形状（下式以该维长度为1的方式描述广播），$k$为元素在第$a$维上的下标，则：

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
| indices | 输入          | 沿`dimension`轴的目标下标，shape为`var`去掉第`dimension`维后的形状（`rank(var)==1`时为$(1,)$）。 | INT32     | ND |
| updates | 输入          | 要写入的值，shape与`indices`一致，数据类型与`var`一致。         | FLOAT16、FLOAT、INT32、INT8 | ND |
| y      | 输出           | 命中位置取`updates`，其余保留`var`，shape与数据类型均与`var`一致。 | FLOAT16、FLOAT、INT32、INT8 | ND |
| dimension | 属性        | 必选，指定沿哪个轴比较，取值范围为$[-m, m-1]$，$m$为`var`的维度数。 | INT       | -    |

## 约束说明

- 记$var$的维度数为$m$、归一化后的`dimension`为$a$。$m \ge 2$时，`indices`与`updates`的维度数均为$m-1$，shape均为$(d_0,\ ...,\ d_{a-1},\ d_{a+1},\ ...,\ d_{m-1})$；$m=1$时，二者shape均为$(1,)$。
- `indices`与`updates`的shape必须完全一致。
- 不支持保留`dimension`轴（即该轴长度为1，如$[2,3,1,5]$）或该轴长度大于1（如$[2,3,2,5]$）的输入。
- `updates`的数据类型必须与`var`一致；`indices`的数据类型固定为INT32。
- `var`的rank须不小于1，无上界（rank为0时拒收）。
- 被选轴长度$D=var.shape[dimension]$不超过INT32_MAX（轴下标与`indices`均以int32表示）。
- 元素在`dimension`轴上的下标由算子内部按`dimension`与`var`的shape生成，无需用户构造，也不作为输入传入。

## 调用说明

| 调用方式 | 样例代码 | 说明 |
| -------- | -------- | ---- |
| 图模式调用 | [test_geir_arg_max_grad.cpp](examples/test_geir_arg_max_grad.cpp) | 通过[算子IR](./op_graph/arg_max_grad_proto.h)构图方式调用ArgMaxGrad算子。 |

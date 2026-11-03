# SGD

## 产品支持情况

| 产品 | 是否支持 |
| :--- | :---: |
| <term>Ascend 950PR&950DT系列产品</term> | √ |
| <term>Atlas A3系列产品</term> | √ |
| <term>Atlas A2系列产品</term> | √ |
| <term>Atlas 200I/500 A2推理产品</term> | √ |
| <term>Atlas推理系列产品</term> | √ |
| <term>Atlas训练系列产品</term> | √ |

## 功能说明

- 算子功能：按照带动量的随机梯度下降（Stochastic Gradient Descent，SGD）算法更新权重，
  并原地更新动量累积量和首步标记。
- 计算公式：

  记$d$为`dampening`、$wd$为`weight_decay`、$lr$为`learning_rate[0]`、
  $m$为`momentum[0]`。首先计算：

  $$
  grad =
  \begin{cases}
  gradient + parameters \times wd, & wd \neq 0 \\
  gradient, & wd = 0
  \end{cases}
  $$

  $$
  accum_t = accum \times m + grad
  $$

  当$d \neq 0$时：

  $$
  accum_t = accum_t - grad \times (1 - stat) \times d
  $$

  权重更新为：

  $$
  parameters_{out} =
  \begin{cases}
  parameters - (grad + accum_t \times m) \times lr, & nesterov = true \\
  parameters - accum_t \times lr, & nesterov = false
  \end{cases}
  $$

  当$m \neq 0$时，将`accum`更新为`accum_t`并将`stat`更新为0；当$m=0$时，
  `accum`和`stat`保持输入值。

## 参数说明

| 参数名 | 输入/输出/属性 | 描述 | 数据类型 | 数据格式 |
| --- | --- | --- | --- | --- |
| parameters | 输入 | 待更新的权重，必须为可写Tensor。 | FLOAT、FLOAT16、BFLOAT16 | NC1HWC0、NDC1HWC0、ND、FRACTAL_Z、FRACTAL_Z_3D |
| gradient | 输入 | 梯度，shape、数据类型和数据格式与`parameters`一致。 | FLOAT、FLOAT16、BFLOAT16 | NC1HWC0、NDC1HWC0、ND、FRACTAL_Z、FRACTAL_Z_3D |
| learning_rate | 输入 | 学习率，元素个数必须为1，数据类型与`parameters`一致。 | FLOAT、FLOAT16、BFLOAT16 | ND |
| accum | 输入 | 动量累积量，必须为可写Tensor；shape、数据类型和数据格式与`parameters`一致。 | FLOAT、FLOAT16、BFLOAT16 | NC1HWC0、NDC1HWC0、ND、FRACTAL_Z、FRACTAL_Z_3D |
| momentum | 输入 | 动量因子，元素个数必须为1，数据类型与`parameters`一致。 | FLOAT、FLOAT16、BFLOAT16 | ND |
| stat | 输入 | 逐元素首步标记，取值1表示不施加阻尼；必须为可写Tensor，shape、数据类型和数据格式与`parameters`一致。 | FLOAT、FLOAT16、BFLOAT16 | NC1HWC0、NDC1HWC0、ND、FRACTAL_Z、FRACTAL_Z_3D |
| parameters | 输出 | 更新后的权重，shape、数据类型和数据格式与输入`parameters`保持一致。 | FLOAT、FLOAT16、BFLOAT16 | NC1HWC0、NDC1HWC0、ND、FRACTAL_Z、FRACTAL_Z_3D |
| dampening | 可选属性 | 动量阻尼系数，默认值为0.0；当`nesterov=true`时必须为0。 | FLOAT | - |
| weight_decay | 可选属性 | 权重衰减系数，默认值为0.0，取值必须大于或等于0。 | FLOAT | - |
| nesterov | 可选属性 | 是否启用Nesterov动量，默认值为false。 | BOOL | - |

## 约束说明

- 图上仅声明一个`parameters`输出，但`parameters`、`accum`和`stat`均可能被原地更新，
  调用方必须将这三个输入作为可写Tensor。权重更新值应从`parameters`输出读取。
- `momentum=0`仅阻止`accum`和`stat`回写，不跳过权重更新中的动量乘法；当`accum`含Inf时，`0×Inf`产生的NaN会传播到权重结果。
- <term>Ascend 950PR&950DT系列产品</term>：
  - 数据类型：各输入、输出支持FLOAT、FLOAT16、BFLOAT16。
  - 数据格式：静态shape和动态shape的各输入、输出均为ND。
  - GE图输入按ND连续布局提供数据；转置、切片等非连续视图需由调用方或框架先转换为连续Tensor。
  - 动态shape能力：支持动态shape和动态rank。
  - 输入shape约束：`parameters`的rank为1～8，不支持空Tensor。
  - 中间计算精度：使用FLOAT进行中间计算，FLOAT16、BFLOAT16结果按就近偶数舍入转换回输入数据类型。
- <term>Atlas A3系列产品</term>、<term>Atlas A2系列产品</term>：
  - 数据类型：各输入、输出支持FLOAT、FLOAT16、BFLOAT16。
  - 静态shape数据格式：`parameters`、`gradient`、`accum`、`stat`及输出支持NC1HWC0、NDC1HWC0、ND、FRACTAL_Z、FRACTAL_Z_3D；`learning_rate`和`momentum`为ND。
  - 动态shape能力：不支持动态shape。
  - 输入shape约束：`parameters`的rank为1～8，不支持空Tensor。
- <term>Atlas 200I/500 A2推理产品</term>、<term>Atlas推理系列产品</term>、<term>Atlas训练系列产品</term>：
  - 数据类型：各输入、输出支持FLOAT、FLOAT16。
  - 静态shape数据格式：`parameters`、`gradient`、`accum`、`stat`及输出支持NC1HWC0、NDC1HWC0、ND、FRACTAL_Z、FRACTAL_Z_3D；`learning_rate`和`momentum`为ND。
  - 动态shape能力：不支持动态shape。
  - 输入shape约束：`parameters`的rank为1～8，不支持空Tensor。

## 调用说明

本算子不提供aclnn单算子接口，通过图引擎（Graph Engine，GE）图模式调用。

| 调用方式 | 样例代码 | 说明 |
| --- | --- | --- |
| GE图模式 | [test_geir_sgd.cpp](examples/test_geir_sgd.cpp) | 通过[SGD算子IR](op_graph/sgd_proto.h)构图并调用算子。 |

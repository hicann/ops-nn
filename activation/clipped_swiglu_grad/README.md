# ClippedSwigluGrad

## 产品支持情况

|产品             |  是否支持  |
|:-------------------------|:----------:|
|  <term>Ascend 950PR/Ascend 950DT</term>   |     √    |
|  <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>   |     √    |
|  <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>     |     √    |
|  <term>Atlas 200I/500 A2 推理产品</term>    |     ×    |
|  <term>Atlas 推理系列产品</term>     |     ×    |
|  <term>Atlas 训练系列产品</term>    |     ×    |

## 功能说明

- 接口功能：aclnnClippedSwiglu的反向算子，计算输入x的梯度`gradXOut`。

- 计算公式：

  对给定的正向输入张量x（维度为[a,b,c,d,e,f,g…]）和输出梯度gradY，ClippedSwigluGrad对其进行以下计算：

  1. 将x基于输入参数dim进行合轴，合轴后维度为[pre, dim2H]；gradY合轴后维度为[pre, dim2H//2]。

  2. 根据输入参数group_index，对x的pre轴进行过滤处理，仅前group行参与计算，其余行梯度为0。

  3. 根据输入参数interleaved，对x进行切分，得到a和b两部分（与正向一致）。

  4. 重算正向中间量：
     - A = clamp(a, max=limit)
     - B = clamp(b, -limit, limit)
     - s = sigmoid(alpha * A)

  5. 计算反向梯度：
     - maskX0 = (a <= limit)
     - maskX1 = (-limit <= b <= limit)
     - da = gradY * (B + bias) * s * (1 + alpha * A * (1 - s)) * maskX0
     - db = gradY * A * s * maskX1

  6. 将da、db散回dx：
     - interleaved为true时：dx[..., ::2] = da，dx[..., 1::2] = db
     - interleaved为false时：dx[..., :h] = da，dx[..., h:] = db

  7. 重塑输出张量grad_x的维度与x一致。

## 参数说明

|参数名|输入/输出/属性|描述|数据类型|数据格式|
|---|---|---|---|---|
|grad_y|输入|正向输出y的梯度。维度在dim轴上为x的一半，其他维度与x一致。|FLOAT、FLOAT16、BFLOAT16|ND|
|x|输入|正向输入x。|FLOAT、FLOAT16、BFLOAT16|ND|
|group_index|可选输入|分组索引。1维，元素个数不超过8192。|INT64|-|
|dim|可选属性|合轴以及切分的维度序号。取值范围[-x.dim(), x.dim()-1]。默认-1。|INT64|-|
|alpha|可选属性|SwiGLU激活系数。默认1.702。|FLOAT|-|
|limit|可选属性|截断门限值。必须大于0。默认7.0。|FLOAT|-|
|bias|可选属性|线性计算偏差。默认1.0。|FLOAT|-|
|interleaved|可选属性|切分方式。true=奇偶切分，false=前后切分。默认true。|BOOL|-|
|grad_x|输出|x的梯度，与x同shape。|FLOAT、FLOAT16、BFLOAT16|ND|

## 约束说明

无。

## 调用说明

| 调用方式 | 调用样例 | 说明 |
|---|---|---|
| aclnn调用 | [test_aclnn_clipped_swiglu_grad](./examples/test_aclnn_clipped_swiglu_grad.cpp) | 通过[aclnnClippedSwigluGrad](./docs/aclnnClippedSwigluGrad.md)接口方式调用ClippedSwigluGrad算子。 |

# INTrainingReduceV2

## 产品支持情况

| 产品 | 是否支持 |
| :----------------------------------------- | :------: |
| <term>Ascend 950PR&950DT系列产品</term> | √ |
| <term>Atlas A3系列产品</term> | √ |
| <term>Atlas A2系列产品</term> | √ |
| <term>Atlas 200I/500 A2推理产品</term> | √ |
| <term>Atlas推理系列产品</term> | √ |
| <term>Atlas训练系列产品</term> | √ |

## 功能说明

- 算子功能：作为实例归一化（Instance Normalization）训练前向的规约阶段，
  对每个实例通道的空间维分别计算元素和`sum`及平方和`square_sum`，并保留规约维度。
  本算子与INTrainingUpdateV2配合使用。
- 计算公式：

  $$
  sum_{n,c} = \sum_{r \in R} x_{n,c,r}
  $$

  $$
  square\_sum_{n,c} = \sum_{r \in R} x_{n,c,r}^{2}
  $$

  其中，$R$表示空间维集合；输出为原始和，不进行均值缩放。
  FLOAT16输入在乘法和累加前转换为FLOAT，两个输出的数据类型均为FLOAT。

## 参数说明

| 参数名 | 输入/输出/属性 | 描述 | 数据类型 | 数据格式 |
| --- | --- | --- | --- | --- |
| x | 输入 | 待规约的输入Tensor，对应公式中的`x`。 | FLOAT、FLOAT16 | NC1HWC0、NDC1HWC0、ND、NCHW、NHWC、NCDHW、NDHWC |
| sum | 输出 | `x`沿空间维求和的结果，保留实例维和通道维，规约维大小为1。 | FLOAT | NC1HWC0、NDC1HWC0、ND |
| square_sum | 输出 | `x`沿空间维求平方和的结果，shape与`sum`一致。 | FLOAT | NC1HWC0、NDC1HWC0、ND |

### 产品差异说明

| 产品 | 静态shape能力 | 动态shape能力 | shape/rank及格式映射 |
| --- | --- | --- | --- |
| <term>Ascend 950PR&950DT系列产品</term> | `x`为NCHW、NCDHW、ND、NHWC或NDHWC，两个输出为ND | 支持上述格式的动态shape及ND动态rank；NHWC/NDHWC输入保留对应格式，NCHW/NCDHW/ND输入使用ND存储，两个输出均为ND | NCHW输入为`[N,C,H,W]`，输出为`[N,C,1,1]`；NCDHW输入为`[N,C,D,H,W]`，输出为`[N,C,1,1,1]`；ND输入的rank为2～8，前两维为N、C，其余维规约为1；NHWC输入为`[N,H,W,C]`，输出为`[N,1,1,C]`；NDHWC输入为`[N,D,H,W,C]`，输出为`[N,1,1,1,C]` |
| <term>Atlas A3系列产品</term><br><term>Atlas A2系列产品</term><br><term>Atlas 200I/500 A2推理产品</term><br><term>Atlas推理系列产品</term><br><term>Atlas训练系列产品</term> | 4维NCHW/NHWC原始格式映射为NC1HWC0，5维NCDHW/NDHWC原始格式映射为NDC1HWC0；输入、输出存储格式一致 | 对应型号已注册的动态shape和动态rank路径使用NC1HWC0或NDC1HWC0存储，格式映射与静态shape一致 | 原始输入为4维或5维；逻辑输出保留N和C，H、W或D、H、W为1；存储格式由图引擎映射 |

## 约束说明

- NCHW/NHWC输入的逻辑shape为4维，空间维为H、W；NCDHW/NDHWC输入为5维，空间维为D、H、W。
- <term>Ascend 950PR&950DT系列产品</term>的ND输入rank为2～8，dim0和dim1分别为实例维和通道维，dim2及之后的维度为空间维。
- <term>Ascend 950PR&950DT系列产品</term>要求`x`的各维度均大于0，不支持空Tensor。
- <term>Ascend 950PR&950DT系列产品</term>要求输入元素总数（各维度大小的乘积）不超过`2^63-1`。
- <term>Ascend 950PR&950DT系列产品</term>的NHWC/NDHWC输入要求通道行字节数`C*sizeof(x.dtype)`不超过`2^40-1`，即FLOAT16的`C`不超过`2^39-1`，FLOAT的`C`不超过`2^38-1`。

## 调用说明

本算子不提供aclnn单算子接口，通过图引擎（Graph Engine，GE）图模式调用。

| 调用方式 | 调用样例 | 说明 |
| --- | --- | --- |
| GE图模式 | [test_geir_in_training_reduce_v2](examples/arch35/test_geir_in_training_reduce_v2.cpp) | 通过[INTrainingReduceV2算子IR](op_graph/in_training_reduce_v2_proto.h)构图并调用算子。 |

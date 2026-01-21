# LpNormV2

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term>                             |    √     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    √     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> |    √     |
| <term>Atlas 200I/500 A2 推理产品</term>                      |    √     |
| <term>Atlas 推理系列产品</term>                             |    √     |
| <term>Atlas 训练系列产品</term>                              |    √     |


## 功能说明

- 接口功能：返回给定张量的矩阵范数或者向量范数。

- 计算公式：支持1/2范数、无穷范数以及其他`p`为float类型的范数计算。
  - 1-范数：  
  
    $$
    \Vert x \Vert = \sum_{i=1}^{N}{\vert x_i \vert}
    $$

  - 2-范数（默认值）：  

    $$
    \Vert x \Vert_2 = (\sum_{i=1}^{N}{\vert x_i \vert^2})^{\frac{1}{2}}
    $$

  - 无穷范数：
  
    $$
    \Vert x \Vert_\infty = \max\limits_{i}{\vert x_i \vert}
    $$
  
    $$
    \Vert x \Vert_{-\infty} = \min\limits_{i}{\vert x_i \vert}
    $$

  - p范数：
  
    $$
    \Vert x \Vert_p = (\sum_{i=1}^{N}{\vert x_i \vert^p})^{\frac{1}{p}}
    $$

## 参数说明

  - self（aclTensor*，计算输入）：公式中的`x`，Device侧的aclTensor。shape支持0-8维，支持[非连续的Tensor](../../../docs/zh/context/非连续的Tensor.md)，[数据格式](../../../docs/zh/context/数据格式.md)支持ND。
    - <term>Atlas 训练系列产品</term>、<term>Atlas 推理系列产品</term>、<term>Atlas 200I/500 A2 推理产品</term>：数据类型支持FLOAT、FLOAT16。
    - <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>、<term>Ascend 950PR/Ascend 950DT</term>：数据类型支持FLOAT、FLOAT16、BFLOAT16。

  - pScalar（aclScalar*，计算输入）：表示范数的类型，公式中的`p`。支持1/2范数、无穷范数以及其他为FLOAT类型的范数计算，数据类型支持FLOAT。

  - dim（aclIntArray*，计算输入）：计算self范数的维度，支持[-N, N-1]，且dims中的元素不能重复，N为self的维度，支持负数，数据类型支持INT。

  - keepdim（bool，计算输入）：决定输出张量是否保留dim参数指定的轴，Host侧的bool常量。

  - relType（aclDataType，计算输入）：预留参数，暂不支持使用。

  - out（aclTensor*，计算输出）：Device侧的aclTensor。shape支持0-8维。若keepdim为true，除dim指定维度上的size为1以外，其余维度的shape需要与self保持一致；若keepdim为false，reduce轴的维度不保留，其余维度shape需要与self一致。支持[非连续的Tensor](../../../docs/zh/context/非连续的Tensor.md)，[数据格式](../../../docs/zh/context/数据格式.md)支持ND。
    - <term>Atlas 训练系列产品</term>、<term>Atlas 推理系列产品</term>、<term>Atlas 200I/500 A2 推理产品</term>：数据类型支持FLOAT、FLOAT16。
    - <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>、<term>Ascend 950PR/Ascend 950DT</term>：数据类型支持FLOAT、FLOAT16、BFLOAT16。

## 约束说明

默认非确定性计算，支持配置开启。

## 调用示例

| 调用方式   | 样例代码           | 说明                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn接口 | [test_aclnn_norm](./examples/test_aclnn_norm.cpp) | 通过[aclnnNorm](docs/aclnnNorm.md)接口方式调用LpNormV2算子。 |
| aclnn接口 | [test_aclnn_linalg_vector_norm](./examples/test_aclnn_LinalgVectorNorm.cpp) | 通过[aclnnLinalgVectorNorm](docs/aclnnLinalgVectorNorm.md)接口方式调用LpNormV2算子。 |
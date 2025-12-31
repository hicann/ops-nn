# ScatterElements

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term>                             |      √     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    √     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> |    √     |
## 功能说明

算子功能：返回输入张量中的唯一元素。

## 参数说明
  - self（aclTensor*，计算输入）：Device侧的aclTensor，[数据格式](../../../docs/zh/context/数据格式.md)支持ND，维度不大于8。
    - <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>、<term>Ascend 950PR/Ascend 950DT</term>：数据类型支持BOOL、FLOAT、FLOAT16、DOUBLE、UINT8、INT8、UINT16、INT16、INT32、UINT32、UINT64、INT64、BFLOAT16。
  
  - sorted（bool，计算输入）: 表示是否对 valueOut 按升序进行排序。true时排序，false时乱序。
  - returnInverse（bool，计算输入）: 表示是否返回输入数据中各个元素在 valueOut 中的下标。
  - valueOut（aclTensor*，计算输出）: Device侧的aclTensor，第一个输出张量，输入张量中的唯一元素，[数据格式](../../../docs/zh/context/数据格式.md)支持ND。
    - <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>、<term>Ascend 950PR/Ascend 950DT</term>：数据类型支持BOOL、FLOAT、FLOAT16、DOUBLE、UINT8、INT8、UINT16、INT16、INT32、UINT32、UINT64、INT64、BFLOAT16。
  - inverseOut（aclTensor*，计算输出）: 第二个输出张量，当returnInverse为True时有意义，返回self中各元素在valueOut中出现的位置下标，数据类型支持INT64，shape与self保持一致。

## 约束说明

无

## 调用说明

| 调用方式   | 样例代码           | 说明                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn接口 | [test_aclnn_unique](examples/test_aclnn_unique.cpp) | 通过[aclnnUnique](docs/aclnnUnique.md)接口方式调用ScatterElements算子。 |
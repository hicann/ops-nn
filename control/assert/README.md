# Assert

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term>                         |    √     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    √     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> |    √     |
| <term>Atlas 200I/500 A2 推理产品</term>                      |    √     |
| <term>Atlas 推理系列产品 </term>                             |    √     |
| <term>Atlas 训练系列产品</term>                              |    √     |

## 功能说明

- 算子功能：断言给定条件为 true，如果输入张量 input_condition 判定为 false，则打印 input_data 中的张量列表。

## 参数说明

| 参数名             | 输入/输出/属性 | 描述                                                         | 数据类型                                                                             | 数据格式 |
|-----------------|----------|------------------------------------------------------------|----------------------------------------------------------------------------------|------|
| input_condition | 输入       | 输入Tensor，要检查的条件。                                           | BOOL                                                                             | ND   |
| input_data      | 输入       | 动态输入Tensor（数量可变），表示当条件为 false 时要打印的 Tensor 集合。             | FLOAT、FLOAT16、INT8、INT16、INT32、INT64、UINT8、UINT16、UINT32、UINT64、BOOL、DOUBLE、STRING | ND   |
| summarize       | 可选属性     | • 可选整型，指定打印每个 Tensor 的前 summarize 个元素。<br>• 默认值为 3。 | INT                                                                              | -    |

## 约束说明

无

## 调用说明

| 调用方式 | 调用样例                                                         | 说明                                                 |
|--------------|--------------------------------------------------------------|----------------------------------------------------|
| 图模式调用 | [test_geir_assert](./examples/test_geir_assert.cpp) | 通过[算子IR](./op_graph/assert_proto.h)构图方式调用Assert算子。 |



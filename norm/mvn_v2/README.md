# MVNV2

## 产品支持情况

| 产品 | 是否支持 |
| ---- | :----: |
| Ascend 950PR/Ascend 950DT | √ |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √ |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | √ |
| Atlas 200I/500 A2 推理产品 | × |
| Atlas 推理系列产品 | × |
| Atlas 训练系列产品 | × |

## 功能说明

- 算子功能：对ND输入张量沿指定axes计算均值与标准差，执行均值方差归一化，使输出在规约轴上均值为0、方差近似为1。

- 计算公式：

$$
y = \frac{x - mean}{\sqrt{var} + eps}
$$

## 参数说明

| 参数名 | 输入/输出/属性 | 描述 | 数据类型 | 数据格式 |
| :--- | :--- | :--- | :--- | :--- |
| x | 输入 | ND输入张量，对应公式中x。 | FLOAT16、FLOAT | ND |
| y | 输出 | 归一化输出张量，shape与dtype均与x相同，对应公式中y。 | FLOAT16、FLOAT | ND |
| eps | 可选属性 | 防除零小常数，加在std上（std + eps），默认1e-9。 | FLOAT | - |
| axes | 可选属性 | 规约轴列表，元素须在[0, rank(x))内，默认取[0, 2, 3]中的有效轴。 | INT64 | - |

## 约束说明

- 输入x支持1-D至8-D ND张量。
- axes元素须在[0, rank(x))内，越界会被tiling阶段拒绝。
- 空Tensor输入直接返回同shape的空Tensor输出，不执行kernel计算。
- 数据类型仅支持FLOAT16与FLOAT，输出数据类型与输入一致。

## 调用说明

| 调用方式 | 调用样例 | 说明 |
| :--- | :--- | :--- |
| GE图模式 | [test_geir_mvn_v2](./examples/test_geir_mvn_v2.cpp) | 通过算子IR定义[op_graph/mvn_v2_proto.h](./op_graph/mvn_v2_proto.h)接入Graph Engine图模式。 |

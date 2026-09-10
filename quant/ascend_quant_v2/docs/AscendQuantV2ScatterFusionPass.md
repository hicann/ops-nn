# AscendQuantV2ScatterFusionPass

## 融合模式

该融合将符合下图左侧图结构的AscendQuantV2、Scatter两个小算子，融合成下图右侧的QuantUpdateScatter算子。其中Scatter的updates输入为AscendQuantV2的输出，AscendQuantV2的输入（x、scale，以及可选的offset）与Scatter的输入（var、indices）一起，作为融合后QuantUpdateScatter算子的输入。

![](../../../docs/zh/figures/AscendQuantV2ScatterFusionPass_1.png)

## 使用约束

- AscendQuantV2的输出个数为1，该输出作为Scatter的updates输入，输出数据类型支持INT8、HIFLOAT8、FLOAT8\_E5M2、FLOAT8\_E4M3FN。
- AscendQuantV2的属性sqrt\_mode为false，round\_mode为"round"，Scatter节点reduce属性为"update"。
- AscendQuantV2的输入个数为2\~3个；第三个输入（offset）存在时，数据类型必须为BFLOAT16或INT32。
- AscendQuantV2的input0必须为FLOAT16或者BFLOAT16，input1必须为BFLOAT16或者为FLOAT32。
- Scatter的属性axis不能为-1和0。

## 支持的型号

<!-- npu="950" id1 -->
Ascend 950PR/Ascend 950DT
<!-- end id1 -->

# AntiQuantMatMulFusionPass

## 融合模式

<!-- npu="950,910b" id1 -->
该融合将AscendAntiQuant（及可选的Add、Mul）与MatMul/BatchMatMul融合为WeightQuantBatchMatmulV2算子。通过常量折叠，在编译期预计算fp16的antiquant_scale和antiquant_offset，消除运行期的AntiQuant+Add+Mul计算。支持以下三种融合模式。

融合模式一：AscendAntiQuant + Add + Mul + MatMul场景。AscendAntiQuant的输出经过Add（加常量offset）和Mul（乘常量scale）后进入MatMul。融合后Add的常量offset和Mul的常量scale与AscendAntiQuant的scale/offset属性折叠为antiquant_scale和antiquant_offset。如下图所示。

![](../../../docs/zh/figures/AntiQuantMatMulFusionPass_1.png)

融合模式二：AscendAntiQuant + Mul + MatMul场景。AscendAntiQuant的输出经过Mul（乘以常量scale）后进入MatMul，不存在Add节点。融合后Mul的常量scale与AscendAntiQuant的scale属性折叠为antiquant_scale，antiquant_offset默认为0。如下图所示。

![](../../../docs/zh/figures/AntiQuantMatMulFusionPass_2.png)

融合模式三：AscendAntiQuant + MatMul场景。AscendAntiQuant的输出直接进入MatMul，不存在Add和Mul节点。融合后antiquant_scale默认为1，antiquant_offset默认为0。如下图所示。

![](../../../docs/zh/figures/AntiQuantMatMulFusionPass_3.png)

>[!NOTE]说明
>常量折叠公式：antiquant_scale = scale_data * anti_scale，antiquant_offset = offset_data / anti_scale + anti_offset。其中anti_scale和anti_offset为AscendAntiQuant的属性，scale_data和offset_data分别为Mul和Add的常量输入。当不存在Add节点时， antiquant_offset默认为0。当不存在Mul节点时，antiquant_scale默认为1。

该融合模式支持的产品如下。

<!-- npu="910b" id2 -->
Atlas A2 训练系列产品/Atlas A2 推理系列产品
<!-- end id2 -->

<!-- npu="950" id3 -->
Ascend 950PR/Ascend 950DT
<!-- end id3 -->
<!-- end id1 -->

## 使用约束

- 支持的MatMul算子类型：MatMul、MatMulV2、BatchMatMul、BatchMatMulV2。
- AscendAntiQuant、Add、Mul节点的输出只能连接到一个下游节点，否则不触发融合。
- Add节点的常量输入和Mul节点的常量输入必须为Const节点。
- MatMul输入x和weight的数据类型必须为FLOAT16，输出数据类型必须为FLOAT16。
- AscendAntiQuant输入数据类型必须为INT8，输出数据类型必须为FLOAT16。
- 输入shape必须为2D，不支持动态shape。
- Mul的常量scale元素数仅支持1或N（weight的N维度），Add的常量offset元素数仅支持1或N。
<!-- npu="910b" -->
- 在Atlas A2 训练系列产品/Atlas A2 推理系列产品场景下，需满足shape准入条件：M<=64、K>=5120、N>=5120，且(K,N)不能为(5120,10240)或(10240,5120)。
<!-- end -->
<!-- npu="950" -->
- 在Ascend 950PR/Ascend 950DT场景下，无shape准入限制。
<!-- end -->
- MatMul的transpose_x1/transpose_x2（或BatchMatMul的adj_x1/adj_x2）属性会传递到融合后的WeightQuantBatchMatmulV2的transpose_x/transpose_weight属性。
- 支持可选的bias输入。

# MatMulReshapeBiasAddFusionPass

## 融合模式

该融合规则将MatMul/MatMulV2 → Reshape → BiasAdd/Add结构融合为带bias输入的MatMul/MatMulV2（has_bias=true）：删除BiasAdd/Add节点，bias改为直连MatMul/MatMulV2，Reshape节点保留并移至MatMul/MatMulV2之后，由Reshape输出直连下游节点。

![](../../../docs/zh/figures/MatMulReshapeBiasAddFusionPass_1.png)

## 使用约束

- MatMul/MatMulV2仅支持两个输入。
- MatMul节点与Reshape节点的输出都只能连接一个下游算子。
- BiasAdd/Add的输入必须是一个bias和一个Reshape的输出。
- bias维度：BiasAdd分支的bias须为一维，总长度等于Reshape输出最后一维；Add分支的bias可为一维或多维，各维须与Reshape输出尾部对应维度一致（不支持broadcast），且总长度等于MatMul输出的尾轴大小。
- Reshape节点无法拆分MatMul节点输出的尾轴。
- MatMul的输出维度为2，且不支持空Tensor。
- Reshape的输出dtype须与BiasAdd/Add的输出dtype一致。
- 仅支持静态图模式。

## 支持的型号

<!-- npu="910b" id1 -->

Atlas A2 训练系列产品/Atlas A2 推理系列产品

<!-- end id1 -->

<!-- npu="A3" id2 -->

Atlas A3 训练系列产品/Atlas A3 推理系列产品

<!-- end id2 -->

<!-- npu="950" id3 -->

Ascend 950PR/Ascend 950DT

<!-- end id3 -->

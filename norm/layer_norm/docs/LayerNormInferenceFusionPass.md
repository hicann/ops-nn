# LayerNormInferenceFusionPass

## 融合模式

该融合规则将ReduceMean、SquaredDifference、Add、Rsqrt、Mul、Sub等小算子组合识别并融合为LayerNorm算子。

- 场景一：

  ![](../../../docs/zh/figures/LayerNormInferenceFusionPass_1.png)

  **融合为**

  ![](../../../docs/zh/figures/LayerNormInferenceFusionPass_2.png)

## 使用约束

- 不支持中间节点被其他分支复用的场景。
- 不支持动态shape场景。
- 4个输入参数的限制：
  - 入参InputTensor，数据格式支持ND、NCHW、NHWC，shape维度大于等于1，且shape的最后一维的值不能为1。
  - 入参Const\_0，必须是1D，且shape的长度为1。
  - 入参Const\_1，必须是1D，且shape的长度等于InputTensor shape的最后一维长度。
  - 入参Const\_2，必须是1D，且shape的长度等于InputTensor shape的最后一维长度。

- 输入ReduceMean的限制：
  - 两个ReduceMean的axes参数必须一致，且axes参数与InputTensor维度的最后一维保持一致。
  - 两个ReduceMean的keep\_dims参数都必须是true。

- 输入Sub的限制：
  - 第一个参数必须是Const\_2。
  - 第二个参数必须是ReduceMean\_0与Mul\_0相乘的结果，不能调换位置。

- 数据类型限制：数据类型仅支持FLOAT32、FLOAT16。

## 支持的型号

<!-- npu="950" id1 -->
Ascend 950PR/Ascend 950DT
<!-- end id1 -->

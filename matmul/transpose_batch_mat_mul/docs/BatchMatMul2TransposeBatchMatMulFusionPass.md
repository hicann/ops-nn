# BatchMatMul2TransposeBatchMatMulFusionPass

## 融合模式

**模式一：**

当Transpose（可选）、BatchMatMul、Transpose这几个节点按下图所示顺序连接时，可融合为TransposeBatchMatMul算子节点。

![](../../../docs/zh/figures/BatchMatMul2TransposeBatchMatMulFusionPass_1.png)

**模式二：**

当BatchMatMul/BatchMatMulV2节点的x1输入shape为\[B2,B1,1,K\]，x2输入shape为\[1,B1,K,N\]/\[1,B1,N,K\]时，融合为TransposeBatchMatMul算子：x1、x2先分别经Reshape合轴为\[B2,B1,K\]、\[B1,K,N\]/\[B1,N,K\]，经TransposeBatchMatMul计算得到\[B2,B1,N\]，再经Reshape恢复为\[B2,B1,1,N\]。

![](../../../docs/zh/figures/BatchMatMul2TransposeBatchMatMulFusionPass_2.png)

**模式三：**

当Transpose（可选）、Transpose（可选）、BatchMatMul、Transpose、Reshape、Reshape、Transpose这几个节点按下图所示顺序连接时，可融合为TransposeBatchMatMul算子节点。

![](../../../docs/zh/figures/BatchMatMul2TransposeBatchMatMulFusionPass_3.png)

## 使用约束

**模式一和模式三：**

- 输入x1、x2与输出out（输入与输出要对应）支持的数据类型：BFLOAT16、FLOAT16、FLOAT32。
- 模式三中out的输出为\[B1,M,B/B1\*N\]，B%B1=0。
- BatchMatMul/BatchMatMulV2节点的adj\_x1、adj\_x2属性必须均为false。
- 仅支持静态图模式。
- 不支持bias输入。
  <!-- npu="910b" id1 -->
- <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：x1输入支持\[B,M,K\]或者\[M,B,K\]，x2输入只支持\[B,K,N\]，对应的TransposeBatchMatMul的属性为perm\_x1=\[0,1,2\]/\[1,0,2\]，perm\_x2=\[0,1,2\]。
  <!-- end id1 -->

  <!-- npu="910b" id2 -->
- <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：当输入数据类型为BFLOAT16、FLOAT16时，k和n向128对齐，需要满足B\*K<65536或者B\*K\>=65536，k<65536。
  <!-- end id2 -->

  <!-- npu="910b" id3 -->
- <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：当输入数据类型为FLOAT32且输入的transpose节点不为空时，无对齐限制，但仍需满足B\*K<65536。
  <!-- end id3 -->

  <!-- npu="A3" id4 -->
- <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：x1输入支持\[B,M,K\]或者\[M,B,K\]，x2输入只支持\[B,K,N\]，对应的TransposeBatchMatMul的属性为perm\_x1=\[0,1,2\]/\[1,0,2\]，perm\_x2=\[0,1,2\]。
  <!-- end id4 -->

  <!-- npu="A3" id5 -->
- <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：当输入数据类型为BFLOAT16、FLOAT16时，k和n向128对齐，需要满足B\*K<65536或者B\*K\>=65536，k<65536。
  <!-- end id5 -->

  <!-- npu="A3" id6 -->
- <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：当输入数据类型为FLOAT32且输入的transpose节点不为空时，无对齐限制，但仍需满足B\*K<65536。
  <!-- end id6 -->

  <!-- npu="950" id7 -->
- <term>Ascend 950PR/Ascend 950DT</term>：x1的输入为\[B,M,K\]或者\[M,B,K\]，x2的输入为\[B,K,N\]或者\[B,N,K\]，对应的TransposeBatchMatMul的属性为perm\_x1=\[0,1,2\]/\[1,0,2\]，perm\_x2=\[0,1,2\]/\[0,2,1\]。
  <!-- end id7 -->

  <!-- npu="950" id8 -->
- <term>Ascend 950PR/Ascend 950DT</term>：当输入shape可走matmultomul或iterbatch优化模板时不融合。
  <!-- end id8 -->

**模式二：**

- x1的输入为\[B2,B1,1,K\]，x2的输入为\[1,B1,K,N\]或者\[1,B1,N,K\]，对应的BatchMatMul的属性为adj\_x1=false，adj\_x2=false/true。
- 输入x1、x2与输出out（输入与输出要对应）支持的数据类型：FLOAT32。
  <!-- npu="910b" id9 -->
- <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：不支持开启HF32。
  <!-- end id9 -->

  <!-- npu="A3" id10 -->
- <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：不支持开启HF32。
  <!-- end id10 -->

- 仅支持静态图模式。
- 不支持bias输入。
  <!-- npu="910b" id11 -->
- <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：输入需要满足B1\*K<65536。
  <!-- end id11 -->

  <!-- npu="A3" id12 -->
- <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：输入需要满足B1\*K<65536。
  <!-- end id12 -->

## 支持的型号

<!-- npu="910b" id13 -->
Atlas A2 训练系列产品/Atlas A2 推理系列产品
<!-- end id13 -->

<!-- npu="A3" id14 -->
Atlas A3 训练系列产品/Atlas A3 推理系列产品
<!-- end id14 -->

<!-- npu="950" id15 -->
Ascend 950PR/Ascend 950DT
<!-- end id15 -->

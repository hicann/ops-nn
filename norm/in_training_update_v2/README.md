# INTrainingUpdateV2

## 产品支持情况

| 产品 | 是否支持 |
| :--- | :---: |
| <term>Ascend 950PR&950DT系列产品</term> | √ |
| <term>Atlas A3系列产品</term> | √ |
| <term>Atlas A2系列产品</term> | √ |
| <term>Atlas 200I/500 A2推理产品</term> | √ |
| <term>Atlas推理系列产品</term> | √ |
| <term>Atlas训练系列产品</term> | √ |

## 功能说明

- 算子功能：作为实例归一化训练融合流程的更新阶段，与INTrainingReduceV2配合使用。算子根据`sum`和`square_sum`计算当前批次每个实例、每个通道的统计量，对`x`归一化，可选执行仿射变换和滑动统计量更新。
- 以下公式描述算子的公开数学语义；可选输入的触发规则和各产品动态Shape能力见“产品差异说明”和“约束说明”。
- 设$R=H\times W$，当前均值、偏置方差和无偏方差分别为：

  $$
  \mu={sum\over R},\qquad
  v_b=\max\left({square\_sum\over R}-\mu^2,0\right)
  $$

  $$
  v_u=\begin{cases}
  0,&R=1\\
  v_b{R\over R-1},&R>1
  \end{cases}
  $$

- `gamma`和`beta`均提供时：

  $$
  y={x-\mu\over\sqrt{v_b+\epsilon}}\times gamma+beta
  $$

- `gamma`或`beta`至少一个未提供时，两者均不参与计算：

  $$
  y={x-\mu\over\sqrt{v_b+\epsilon}}
  $$

- `mean`和`variance`均提供时：

  $$
  batch\_mean=momentum\times\mu+(1-momentum)\times mean
  $$

  $$
  batch\_variance=momentum\times v_u+(1-momentum)\times variance
  $$

- `mean`或`variance`至少一个未提供时，两者均不参与更新，输出当前统计量：

  $$
  batch\_mean=\mu,\qquad batch\_variance=v_u
  $$

## 参数说明

<table style="table-layout: fixed"><colgroup>
<col style="width: 150px">
<col style="width: 150px">
<col style="width: 410px">
<col style="width: 180px">
<col style="width: 150px">
</colgroup>
<thead><tr>
<th>参数名</th><th>输入/输出/属性</th><th>描述</th><th>数据类型</th><th>数据格式</th>
</tr></thead>
<tbody>
<tr><td>x</td><td>输入</td><td>待归一化的4维张量。</td><td>FLOAT16、FLOAT</td><td>NCHW、NHWC</td></tr>
<tr><td>sum</td><td>输入</td><td>INTrainingReduceV2输出的4维空间维求和结果，H、W均为1。</td><td>FLOAT</td><td>NCHW、NHWC</td></tr>
<tr><td>square_sum</td><td>输入</td><td>INTrainingReduceV2输出的4维空间维平方和，H、W均为1。</td><td>FLOAT</td><td>NCHW、NHWC</td></tr>
<tr><td>gamma</td><td>可选输入</td><td>4维仿射缩放参数，H、W均为1。</td><td>FLOAT</td><td>NCHW、NHWC</td></tr>
<tr><td>beta</td><td>可选输入</td><td>4维仿射偏移参数，H、W均为1。</td><td>FLOAT</td><td>NCHW、NHWC</td></tr>
<tr><td>mean</td><td>可选输入</td><td>4维待更新滑动均值，H、W均为1。</td><td>FLOAT</td><td>NCHW、NHWC</td></tr>
<tr><td>variance</td><td>可选输入</td><td>4维待更新滑动方差，H、W均为1。</td><td>FLOAT</td><td>NCHW、NHWC</td></tr>
<tr><td>momentum</td><td>可选属性</td><td>统计量更新权重，默认值为0.1。</td><td>FLOAT</td><td>-</td></tr>
<tr><td>epsilon</td><td>可选属性</td><td>加到偏置方差上的数值稳定项，默认值为0.00001。</td><td>FLOAT</td><td>-</td></tr>
<tr><td>y</td><td>输出</td><td>归一化结果，shape、数据类型和数据格式与x一致。</td><td>FLOAT16、FLOAT</td><td>NCHW、NHWC</td></tr>
<tr><td>batch_mean</td><td>输出</td><td>更新后的均值，shape与sum一致。</td><td>FLOAT</td><td>NCHW、NHWC</td></tr>
<tr><td>batch_variance</td><td>输出</td><td>更新后的方差，shape与sum一致。</td><td>FLOAT</td><td>NCHW、NHWC</td></tr>
</tbody></table>

### 产品差异说明

- <term>Ascend 950PR&950DT系列产品</term>：支持4维NCHW、NHWC；支持固定4维下的动态维度，不支持动态秩。
- <term>Atlas A3系列产品</term>、<term>Atlas A2系列产品</term>、<term>Atlas 200I/500 A2推理产品</term>、<term>Atlas推理系列产品</term>和<term>Atlas训练系列产品</term>：支持4维NCHW、NHWC。

## 约束说明

### 所有支持产品的公开约束

- 所有输入均为4维张量，仅支持NCHW和NHWC公开逻辑格式。`x`和`y`的数据类型为FLOAT16或FLOAT，其余输入和输出为FLOAT。
- `sum`、`square_sum`、`gamma`、`beta`、`mean`、`variance`以及两个统计输出的H、W均为1；四个可选输入在公开原型中分别为可选。

### 仅<term>Ascend 950PR&950DT系列产品</term>的补充约束

- 固定4维下支持动态维度，不支持动态秩。
- NCHW下空间维为`x`的H、W轴；NHWC下规则等价。
- `sum`和`square_sum`始终必须与`x`使用相同逻辑格式，且shape精确为[N,C,1,1]或[N,1,1,C]。
- `gamma`和`beta`均提供时，两者必须与`x`使用相同逻辑格式，各自支持G=1或G=N，G可不同，C必须与`x`一致。
- `mean`和`variance`均提供时，两者必须与`x`使用相同逻辑格式，且shape与`sum`一致。
- 四个可选输入可独立缺席。仅当`gamma`和`beta`均提供时启用仿射分支；仅当`mean`和`variance`均提供时启用滑动更新分支。半对输入可传入，但孤立输入不被读取；孤立输入仍校验FP32、4D、NCHW/NHWC和空间维为1，不校验其G/N/C与`x`的耦合关系。
- N=0或C=0时按空Tensor处理；N和C均大于0时，H和W必须大于0。
- R=1时无偏方差定义为0。
- Shape派生的`N*C`必须可由有符号64位整数表示；N和C均大于0的非空输入中，`H*W`、`N*C*H*W`以及输入和统计张量的字节长度也必须可由有符号64位整数表示。实际可用规模还受设备内存和框架通用上限约束。
- NHWC下二维数据搬运的GM字节stride，即每个通道分块对应的`(C-currentC)*sizeof(x)`，不得超过$2^{40}-1$字节；实现会将通道分块控制在不超过64个元素，并按最小分块检查最坏stride。

## 调用说明

| 调用方式 | 调用样例 | 说明 |
| --- | --- | --- |
| GE图模式 | [test_geir_in_training_update_v2](./examples/arch35/test_geir_in_training_update_v2.cpp) | 通过[算子IR](./op_graph/in_training_update_v2_proto.h)构图，在<term>Ascend 950PR&950DT系列产品</term>上执行。 |

本算子是实例归一化训练融合流程的内部图节点，不提供同名aclnn单算子接口。

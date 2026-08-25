# BN3DTrainingReduce

## 产品支持情况

|产品             |  是否支持  |
|:-------------------------|:----------:|
|  <term>Ascend 950PR/Ascend 950DT</term>   |     √    |
|  <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>   |     √    |
|  <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>     |     √    |
|  <term>Atlas 200I/500 A2 推理产品</term>    |     √    |
|  <term>Atlas 推理系列产品</term>    |     √    |
|  <term>Atlas 训练系列产品</term>    |     √    |

## 功能说明

- 算子功能：BN3DTrainingReduce是三维Batch Normalization（批归一化）训练前向的reduce（规约）阶段算子，与BN3DTrainingUpdate配对使用。对输入`x`的每个有效逻辑通道，在批次轴N与全部空间轴D、H、W上分别求和与求平方和，输出统计量`sum`（Σx）与`square_sum`（Σx²），仅保留通道轴C。规约跨N进行、只保留C，区别于[INTrainingReduceV2](../in_training_reduce_v2/README.md)仅沿空间轴规约、保留N与C。输出为原始和，不做1/R缩放，均值与方差由下游`BN3DTrainingUpdate`阶段计算。
- 计算公式（以5D NCDHW为例）：

  $$
  sum_{c} = \sum_{n=0}^{N-1} \sum_{d=0}^{D-1} \sum_{h=0}^{H-1} \sum_{w=0}^{W-1} x_{(n,c,d,h,w)}
  $$

  $$
  squareSum_{c} = \sum_{n=0}^{N-1} \sum_{d=0}^{D-1} \sum_{h=0}^{H-1} \sum_{w=0}^{W-1} x_{(n,c,d,h,w)}^2
  $$

  其中$x$为输入特征图，$sum$、$squareSum$为per-C的统计量。`square_sum`是逐元素平方后求和，不是$(\sum x)^2$。

## 参数说明

<table style="undefined;table-layout: fixed; width: 1005px"><colgroup>
  <col style="width: 170px">
  <col style="width: 170px">
  <col style="width: 352px">
  <col style="width: 213px">
  <col style="width: 100px">
  </colgroup>
  <thead>
    <tr>
      <th>参数名</th>
      <th>输入/输出/属性</th>
      <th>描述</th>
      <th>数据类型</th>
      <th>数据格式</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>x</td>
      <td>输入</td>
      <td><ul><li>表示需要规约的输入特征图，对应公式中的<code>x</code>。</li><li>跨N轴与全部空间轴规约，仅C轴保留。</li><li>各数据格式支持的shape维度数不同，详见<a href="#格式与shape支持矩阵">格式与shape支持矩阵</a>。</li></ul></td>
      <td>FLOAT16、FLOAT32、BFLOAT16</td>
      <td>NCDHW/NDHWC</td>
    </tr>
    <tr>
      <td>sum</td>
      <td>输出</td>
      <td><ul><li>输入<code>x</code>的通道轴为0时为空Tensor。</li><li>表示对x求和的结果，对应公式中的<code>sum</code>。</li><li>数据类型固定为FLOAT32，与输入x的dtype无关。</li><li>数据格式与输入x保持一致。</li><li>逻辑shape为[C]。</li></ul></td>
      <td>FLOAT32</td>
      <td>NCDHW/NDHWC</td>
    </tr>
    <tr>
      <td>square_sum</td>
      <td>输出</td>
      <td><ul><li>输入<code>x</code>的通道轴为0时为空Tensor。</li><li>表示对x求平方和的结果，对应公式中的<code>squareSum</code>。</li><li>数据类型、shape、数据格式与输出sum保持一致。</li></ul></td>
      <td>FLOAT32</td>
      <td>NCDHW/NDHWC</td>
    </tr>
  </tbody></table>

- <term>Ascend 950PR/Ascend 950DT</term>：`x`的数据类型支持FLOAT16、FLOAT32、BFLOAT16；数据格式支持NCDHW、NDHWC。
- <term>Atlas 训练系列产品</term>：`x`的数据类型支持FLOAT16、FLOAT32，不支持BFLOAT16；数据格式支持NDC1HWC0、NCHW。
- <term>Atlas 推理系列产品</term>：`x`的数据类型支持FLOAT16、FLOAT32，不支持BFLOAT16；数据格式支持NDC1HWC0、NCDHW。
- <term>Atlas 200I/500 A2 推理产品</term>：`x`的数据类型支持FLOAT16、FLOAT32，不支持BFLOAT16；数据格式仅支持NDC1HWC0。

### 格式与shape支持矩阵

在<term>Ascend 950PR/Ascend 950DT</term>上，输入`x`支持的数据格式、shape维度数及规约轴如下。

| 数据格式 | 输入shape维度数 | 输入shape | 通道轴 | 规约轴 | 输出shape |
| :--- | :---: | :--- | :--- | :--- | :--- |
| NCDHW | 2~5 | [N, C, ...] | 第2维 | 除第2维外的全部轴 | [C] |
| NDHWC | 仅5 | [N, D, H, W, C] | 第5维 | N、D、H、W | [C] |

输入`x`的数据格式和shape维度数必须同时满足上表要求，否则报错。

## 约束说明

- 空Tensor：仅支持通道轴大小为0；通道轴不为0时，N轴和各空间规约轴大小必须大于0。
- 支持动态shape与动态rank：图编译期允许shape、rank未知，执行期以具体shape重新校验，具体shape不允许含负值维度。rank未知时，原始格式为NCDHW、NDHWC的输出shape为[-1]。
- 输入含NaN、Inf时，按IEEE 754语义传播到对应通道的两个输出；有限值平方或累加溢出FLOAT32时，允许输出Inf。

## 调用说明

本算子是图模式内部算子，仅提供算子IR构图入口，不提供aclnn对外接口。

| 调用方式   | 调用样例           | 说明                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| 图模式调用  | [test_geir_bn3d_training_reduce](examples/arch35/test_geir_bn3d_training_reduce.cpp) | 通过[算子IR](op_graph/bn3d_training_reduce_proto.h)构图方式调用BN3DTrainingReduce算子。 |

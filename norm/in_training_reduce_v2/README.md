# INTrainingReduceV2

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

- 算子功能：INTrainingReduceV2是Instance Normalization（实例归一化）训练前向的reduce（规约）阶段算子，与InTrainingUpdateV2配对使用。对每个实例通道 (n, c)，在其空间维度（4D NCHW的H、W；5D NCDHW的D、H、W）上分别求和与求平方和，输出统计量`sum`（Σx）与`square_sum`（Σx²），规约轴保留（keepdims）。规约仅沿空间轴，N与C保留（区别于[BatchNorm](../batch_norm_v3/README.md)沿N规约）；输出为原始和，不做1/R缩放（均值/方差由下游`InTrainingUpdateV2`阶段计算）。
- 计算公式（以4D NCHW为例，5D NCDHW沿D、H、W规约同理）：

  $$
  sum_{(n,c)} = \sum_{h=0}^{H-1} \sum_{w=0}^{W-1} x_{(n,c,h,w)}
  $$

  $$
  squareSum_{(n,c)} = \sum_{h=0}^{H-1} \sum_{w=0}^{W-1} x_{(n,c,h,w)}^2
  $$

  其中 $x$ 为输入特征图，$sum$、$squareSum$ 为per-(N,C) 的统计量（保留N、C，空间轴规约为1）。

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
      <td><ul><li>表示需要规约的输入特征图，对应公式中的`x`。</li><li>仅沿空间轴规约，N、C轴保留。</li><li>NCHW：shape支持4D。</li><li>NCDHW：shape支持5D。</li><li>ND：shape支持2~8D，其中第2维固定为C轴。</li></ul></td>
      <td>FLOAT32、FLOAT16</td>
      <td>NCHW/NCDHW/ND</td>
    </tr>
    <tr>
      <td>sum</td>
      <td>输出</td>
      <td><ul><li>表示对x沿空间轴求和的结果，对应公式中的`sum`。</li><li>数据类型固定为FLOAT32，与输入x的dtype无关。</li><li>shape维度数与输入x一致，N、C与输入x保持一致，空间（规约）轴置1，即 [N, C, 1, 1]（4D）或 [N, C, 1, 1, 1]（5D）。</li></ul></td>
      <td>FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>square_sum</td>
      <td>输出</td>
      <td><ul><li>表示对x沿空间轴求平方和的结果，对应公式中的`squareSum`。</li><li>数据类型、shape与输出sum保持一致。</li></ul></td>
      <td>FLOAT32</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## 约束说明

- 输出`sum`、`square_sum`的数据类型恒为FLOAT32，与输入`x`的dtype无关；FLOAT16输入全程提升FLOAT32计算与累加。
- 输出为原始和（Σx、Σx²），不做1/R缩放；均值/方差由下游`InTrainingUpdateV2`计算。
- 输出`sum`、`square_sum`的空间（规约）轴大小为1，N、C与输入`x`一致。
- 不支持空tensor：输入`x`的**任意轴为0**（含空间规约轴）均判非法，在Host Tiling阶段返回失败。
- 本算子不提供aclnn单算子接口，仅支持GE图模式调用。

## 调用说明

| 调用方式   | 样例代码           | 说明                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| 图模式接口  | [test_geir_in_training_reduce_v2](examples/arch35/test_geir_in_training_reduce_v2.cpp) | 通过GE图模式构建INTrainingReduceV2算子图并执行RunGraph验证。 |

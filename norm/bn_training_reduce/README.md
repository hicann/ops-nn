# BNTrainingReduce

## 产品支持情况

| 产品 | 是否支持 |
| :----------------------------------------- | :------:|
| <term>Ascend 950PR&950DT系列产品</term> | √ |
| <term>Atlas A3系列产品</term> | √ |
| <term>Atlas A2系列产品</term> | √ |
| <term>Atlas 200I/500 A2推理产品</term> | √ |
| <term>Atlas推理系列产品</term> | √ |
| <term>Atlas训练系列产品</term> | √ |

## 功能说明

- 算子功能：保留输入的C轴并归约其余所有轴，输出每个通道的元素和与平方和。
- 计算公式：

  $$
  sum_c = \sum_{n,h,w}x_{n,c,h,w}
  $$

  $$
  square\_sum_c = \sum_{n,h,w}x_{n,c,h,w}^{2}
  $$

上述公式以4维NCHW为例；其他支持的Rank和数据格式均保留C轴并归约其余所有轴，NCDHW场景还包括D轴。

## 参数说明

<table style="table-layout: fixed; width: 1576px"><colgroup>
<col style="width: 170px">
<col style="width: 170px">
<col style="width: 520px">
<col style="width: 300px">
<col style="width: 170px">
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
    <td>待统计的输入张量，对应公式中的x。</td>
    <td>FLOAT16、BFLOAT16、FLOAT</td>
    <td>NCHW、NHWC、NCDHW</td>
  </tr>
  <tr>
    <td>sum</td>
    <td>输出</td>
    <td>沿C轴以外的所有轴归约得到的每通道元素和，对应公式中的sum。</td>
    <td>FLOAT</td>
    <td>与x相同</td>
  </tr>
  <tr>
    <td>square_sum</td>
    <td>输出</td>
    <td>沿C轴以外的所有轴归约得到的每通道平方和，对应公式中的square_sum。</td>
    <td>FLOAT</td>
    <td>与x相同</td>
  </tr>
</tbody></table>

## 约束说明

- <term>Ascend 950PR&950DT系列产品</term>：
  - x支持NCHW格式的2～4维Tensor、NHWC格式的4维Tensor和NCDHW格式的5维Tensor。
  - sum和square_sum的shape均为一维[C]，其中C为x的通道维大小，数据格式与x相同。
  - 图模式支持动态shape和动态rank。
  - 支持空Tensor。C = 0时输出为空Tensor；C > 0且归约集合为空时，两个输出均为shape [C]的全零Tensor。

## 调用说明

<table><thead>
  <tr>
    <th>调用方式</th>
    <th>调用样例</th>
    <th>说明</th>
  </tr></thead>
<tbody>
  <tr>
    <td>aclnn调用</td>
    <td><a href="./examples/test_aclnn_batch_norm_reduce.cpp">test_aclnn_batch_norm_reduce</a></td>
    <td>通过<a href="./docs/aclnnBatchNormReduce.md">aclnnBatchNormReduce</a>两段式接口调用，产品支持范围和参数约束以该API文档为准。</td>
  </tr>
  <tr>
    <td>图模式调用</td>
    <td><a href="./examples/test_geir_bn_training_reduce.cpp">test_geir_bn_training_reduce</a></td>
    <td>通过<a href="./op_graph/bn_training_reduce_proto.h">BNTrainingReduce算子IR</a>构图并调用。</td>
  </tr>
</tbody></table>

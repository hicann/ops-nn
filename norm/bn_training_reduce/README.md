# BNTrainingReduce

## 产品支持情况

| 产品 | 是否支持 |
| :----------------------------------------- | :------:|
| <term>Ascend 950PR/Ascend 950DT</term> | √ |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term> | √ |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> | √ |
| <term>Atlas 200I/500 A2 推理产品</term> | × |
| <term>Atlas 推理系列产品</term> | × |
| <term>Atlas 训练系列产品</term> | × |

## 功能说明

- 算子功能：对四维NCHW或NHWC输入的N、H、W轴执行归约，输出每个通道的元素和与平方和。
- 计算公式：

  $$
  sum_c = \sum_{n,h,w}x_{n,c,h,w}
  $$

  $$
  square\_sum_c = \sum_{n,h,w}x_{n,c,h,w}^{2}
  $$

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
    <td>待统计的四维训练激活。</td>
    <td>FLOAT16、BFLOAT16、FLOAT</td>
    <td>NCHW、NHWC</td>
  </tr>
  <tr>
    <td>sum</td>
    <td>输出</td>
    <td>沿N、H、W轴归约得到的每通道元素和。</td>
    <td>FLOAT</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>square_sum</td>
    <td>输出</td>
    <td>沿N、H、W轴归约得到的每通道平方和。</td>
    <td>FLOAT</td>
    <td>ND</td>
  </tr>
</tbody></table>

## 约束说明

- x必须为四维NCHW或NHWC张量，通道轴分别为第1维或第3维。
- sum和square_sum必须为一维ND张量，长度等于x的C维，数据类型固定为FLOAT。
- FLOAT16和BFLOAT16输入按FLOAT精度执行平方与累加。
- 算子无属性，支持空Tensor；归约集合为空时输出为零或空向量。

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
    <td>通过aclnnBatchNormReduce两段式接口调用BNTrainingReduce算子。</td>
  </tr>
  <tr>
    <td>图模式调用</td>
    <td><a href="./examples/test_geir_bn_training_reduce.cpp">test_geir_bn_training_reduce</a></td>
    <td>通过算子IR构图并调用BNTrainingReduce算子。</td>
  </tr>
</tbody></table>

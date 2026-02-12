# CtcLossV2

## 产品支持情况

|产品             |  是否支持  |
|:-------------------------|:----------:|
|  <term>Ascend 950PR/Ascend 950DT</term>   |     √    |
|  <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>   |     √    |
|  <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>     |     √    |
|  <term>Atlas 200I/500 A2 推理产品</term>    |     ×    |
|  <term>Atlas 推理系列产品</term>     |     ×    |
|  <term>Atlas 训练系列产品</term>    |     ×    |

## 功能说明

- 算子功能：计算连接时序分类损失值。

- 计算公式：
  定义$y_{k}^{t}$表示在时刻$t$时真实字符为$k$的概率。（一般地，$y_{k}^{t}$是经过softmax之后的输出矩阵中的一个元素）。将字符集$L^{'}$可以构成的所有序列的集合称为$L^{'T}$，将$L^{'T}$中的任意一个序列称为路径，并标记为$π$。$π$的分布为公式(1)：

  $$
  p(π|x)=\prod_{t=1}^{T}y^{t}_{π_{t}} , \forall π \in L'^{T}. \tag{1}
  $$

  定义多对一(many to one)映射B: $L^{'T} \to L^{\leq T}$，通过映射B计算得到$l \in L^{\leq T}$的条件概率，等于对应于$l$的所有可能路径的概率之和，公式(2):

  $$
  p(l|x)=\sum_{π \in B^{-1}(l)}p(π|x).\tag{2}
  $$

  将找到使$p(l|x)$值最大的$l$的路径的任务称为解码，公式(3)：

  $$
  h(x)=^{arg \  max}_{l \in L^{ \leq T}} \ p(l|x).\tag{3}
  $$

  当zeroInfinity为True时

  $$
  h(x)=\begin{cases}0,&h(x) == Inf \text{ or } h(x) == -Inf \\h(x),&\text { else }\end{cases}
  $$

## 参数说明

<table style="undefined;table-layout: fixed; width: 1480px"><colgroup>
    <col style="width: 177px">
    <col style="width: 120px">
    <col style="width: 273px">
    <col style="width: 292px">
    <col style="width: 152px">
    <col style="width: 110px">
    <col style="width: 151px">
    <col style="width: 145px">
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
      <td>log_probs</td>
      <td>输入</td>
      <td>表示输出的对数概率，公式中的y。 </td>
      <td>FLOAT16、FLOAT、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>target</td>
      <td>输入</td>
      <td>表示包含目标序列的标签，公式中的π。</td>
      <td>INT64、INT32、BOOL、FLOAT、FLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>input_lengths</td>
      <td>输入</td>
      <td>表示输入序列的实际长度，公式中的T为inputLengths中的元素。</td>
      <td>int64_t</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>target_lengths</td>
      <td>输入</td>
      <td>表示目标序列的实际长度，公式中的l的长度为targetLengths中的元素。</td>
      <td>int64_t</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>neg_log_likelihood</td>
      <td>输出</td>
      <td>公式中的输入reduction，指定要应用到输出的缩减，支持 0('none') | 1('mean') | 2('sum')。'none' 表示不应用缩减，'mean' 表示输出的总和将除以输出中的元素数，'sum' 表示输出将被求和。</td>
      <td>int64_t</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>log_alpha</td>
      <td>输出</td>
      <td>表示输入到目标的可能跟踪的概率，公式中的p(l|x)</td>
      <td>与log_probs一致</td>
      <td>ND</td>
    </tr>
  </tbody></table>


## 约束说明

无

## 调用说明

| 调用方式 | 调用样例                                                                   | 说明                                                             |
|--------------|------------------------------------------------------------------------|----------------------------------------------------------------|
| aclnn调用 | [test_aclnn_ctc_loss_v2](./examples/arch35/test_aclnn_ctc_loss_v2.cpp) | 通过[aclnnCtcLoss](./docs/aclnnCtcLoss.md)接口方式调用CtcLossV2算子。    |
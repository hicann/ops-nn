# CtcLossV2Grad

## 产品支持情况

|产品             |  是否支持  |
|:-------------------------|:----------:|
|  <term>Ascend 950PR/Ascend 950DT</term>   |     √    |
|  <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>   |     √    |
|  <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>     |     √    |
|  <term>Atlas 200I/500 A2 推理产品</term>    |     ×    |
|  <term>Atlas 推理系列产品</term>     |     √    |
|  <term>Atlas 训练系列产品</term>    |     √    |

## 功能说明

- 算子功能：计算连接时序分类损失值的反向。
- 计算公式：
定义$y_{k}^{t}$表示在时刻$t$时真实字符为$k$的概率。(一般地，$y_{k}^{t}$是经过softmax之后的输出矩阵中的一个元素)。将字符集$L^{'}$可以构成的所有序列的集合称为$L^{'T}$，将$L^{'T}$中的任意一个序列称为路径，并标记为$π$。$π$的分布为公式(1)：

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
  h(x)=\begin{cases}0,&h(x) == inf \text{ or } h(x) == -inf \\h(x),&\text { else }\end{cases}
  $$

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
      <td>grad_out</td>
      <td>输入</td>
      <td>梯度更新系数。</td>
      <td>FLOAT16、FLOAT32、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>log_probs</td>
      <td>输入</td>
      <td>输出的概率对数。</td>
      <td>FLOAT16、FLOAT32、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>targets</td>
      <td>输入</td>
      <td>目标序列。</td>
      <td>INT32、INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>input_lengths</td>
      <td>输入</td>
      <td>输入序列的长度。</td>
      <td>INT32、INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>target_lengths</td>
      <td>输入</td>
      <td>目标序列的长度。</td>
      <td>INT32、INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>neg_log_likelihood</td>
      <td>输入</td>
      <td>相对于每个节点可微分的损失值。</td>
      <td>FLOAT16、FLOAT32、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>log_alpha</td>
      <td>输入</td>
      <td>输入序列追踪到目标序列的概率。</td>
      <td>FLOAT16、FLOAT32、BFLOAT16</td>
      <td>ND</td>
    <tr>
      <td>blank</td>
      <td>属性</td>
      <td>blank字符的位置。</td>
      <td>INT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>reduction</td>
      <td>属性</td>
      <td>指定计算输出的归约方式。</td>
      <td>STRING</td>
      <td>-</td>
    </tr>
    <tr>
      <td>zero_infinity</td>
      <td>属性</td>
      <td>是否为零无限损失和相关梯度。</td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
    <tr>
      <td>grad</td>
      <td>输出</td>
      <td>ctcloss的损失梯度。</td>
      <td>FLOAT16、FLOAT32、BFLOAT16</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## 约束说明

* grad_out的元素数量必须为 N。
* log_probs的shape为 (T, N, C)，？其中 T >= max( input_lengths )。
* targets的shape为 (N, S)或者 ( sum( target_lengths ),)。
* input_lengths的元素数量必须为 N。
* target_lengths的元素数量必须为 N。
* neg_log_likelihood的shape必须为1维 (N)。
* log_alpha的shape必须为3维 (N, T, X)，其中 X = ( 2 * S + 1 )。
* reduction的取值范围为{'none', 'mean', 'sum'}。

## 调用说明

| 调用方式   | 样例代码           | 说明                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn接口 | [test_aclnn_ctc_loss_backward](examples/arch35/test_aclnn_ctc_loss_backward.cpp) | 通过[aclnnCtcLossBackward](docs/aclnnCtcLossBackward.md)接口方式调用CtclossV2Grad算子。 |
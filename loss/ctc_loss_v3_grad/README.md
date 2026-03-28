# CtcLossV3Grad

## 产品支持情况

|产品             |  是否支持  |
|:-------------------------|:----------:|
|  <term>Ascend 950PR/Ascend 950DT</term>   |     ×    |
|  <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>   |     √    |
|  <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>     |     √    |
|  <term>Atlas 200I/500 A2 推理产品</term>    |     ×    |
|  <term>Atlas 推理系列产品</term>     |     ×    |
|  <term>Atlas 训练系列产品</term>    |     ×    |

## 功能说明

- 算子功能：计算连续（未分段）时间序列与目标序列之间的损失的反向。
- 计算公式：

  $$
  O^{ML}(S,N_w) = - \sum_{x,z\in S} {\ln{p(z|x)}}
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
      <td>FLOAT16、FLOAT32、BFLOAT16、DOUBLE</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>log_probs</td>
      <td>输入</td>
      <td>输入的每个时刻每个token的概率。</td>
      <td>FLOAT16、FLOAT32、BFLOAT16、DOUBLE</td>
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
      <td>FLOAT16、FLOAT32、BFLOAT16、DOUBLE</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>log_alpha</td>
      <td>输入</td>
      <td>输入序列追踪到目标序列的概率。</td>
      <td>FLOAT16、FLOAT32、BFLOAT16、DOUBLE</td>
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
      <td>ctcloss的梯度。</td>
      <td>FLOAT16、FLOAT32、BFLOAT16、DOUBLE</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## 约束说明

* grad_out的元素数量必须为N。
* log_probs的shape为(T, N, C)，其中T>=max(input_lengths)。
* targets的shape为(N, S)或者(sum(target_lengths),)，其中S>=max(target_lengths)，值为[0, C)。
* input_lengths的元素数量必须为N，值为[0, T]。
* target_lengths的元素数量必须为N，值为[1, S]。
* neg_log_likelihood的shape必须为1维(N)。
* log_alpha的shape必须为3维(N, T, X)，其中X=((2 * S + 1) + 7) / 8 * 8。
* blank的值域为[0, C - 1]。
* reduction的取值范围为{'none', 'mean', 'sum'}。

## 调用说明

| 调用方式   | 样例代码           | 说明                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn接口 | [test_aclnn_ctc_loss_v3_grad](examples/test_aclnn_ctc_loss_v3_grad.cpp) | 通过[aclnnCtcLossBackward](../ctc_loss_v2_grad/docs/aclnnCtcLossBackward.md)接口方式调用CtclossV3Grad算子。 |
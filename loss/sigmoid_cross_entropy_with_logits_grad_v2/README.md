# SigmoidCrossEntropyWithLogitsGradV2

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

将输入`self`执行`logits`计算，将得到的值与标签值`target`一起进行`BCELoss`的反向传播计算。

## 参数说明

<table class="tg" style="undefined;table-layout: fixed; width: 940px"><colgroup>
<col style="width: 138px">
<col style="width: 121px">
<col style="width: 280px">
<col style="width: 280px">
<col style="width: 121px">
</colgroup>
<thead>
  <tr>
    <th class="tg-0lax">参数名</th>
    <th class="tg-0lax">输入/输出/属性</th>
    <th class="tg-0lax">描述</th>
    <th class="tg-0lax">数据类型</th>
    <th class="tg-0lax">数据格式</th>
  </tr></thead>
<tbody>
  <tr>
    <td class="tg-0lax">gradOutput</td>
    <td class="tg-0lax">输入</td>
    <td class="tg-0lax">网络反向传播前一步的梯度值。</td>
    <td class="tg-0lax">FLOAT16、FLOAT、BFLOAT16</td>
    <td class="tg-0lax">ND</td>
  </tr>
  <tr>
    <td class="tg-0lax">self</td>
    <td class="tg-0lax">输入</td>
    <td class="tg-0lax">网络正向前一层的计算结果。</td>
    <td class="tg-0lax">FLOAT16、FLOAT、BFLOAT16</td>
    <td class="tg-0lax">ND</td>
  </tr>
  <tr>
    <td class="tg-0lax">target</td>
    <td class="tg-0lax">输入</td>
    <td class="tg-0lax">样本的标签值，取值范围为0~1。</td>
    <td class="tg-0lax">与self保持一致</td>
    <td class="tg-0lax">ND</td>
  </tr>
  <tr>
    <td class="tg-0lax">weightOptional</td>
    <td class="tg-0lax">输入</td>
    <td class="tg-0lax">二分交叉熵权重，shape可以broadcast到self的shape</td>
    <td class="tg-0lax">与self保持一致</td>
    <td class="tg-0lax">ND</td>
  </tr>
  <tr>
    <td class="tg-0lax">posWeightOptional</td>
    <td class="tg-0lax">输入</td>
    <td class="tg-0lax">正类的权重。</td>
    <td class="tg-0lax">与self保持一致</td>
    <td class="tg-0lax">ND</td>
  </tr>
  <tr>
    <td class="tg-0lax">reduction</td>
    <td class="tg-0lax">输入</td>
    <td class="tg-0lax">表示对二元交叉熵反向求梯度计算结果做的reduce操作。</td>
    <td class="tg-0lax">INT64</td>
    <td class="tg-0lax">-</td>
  </tr>
  <tr>
    <td class="tg-0lax">out</td>
    <td class="tg-0lax">输出</td>
    <td class="tg-0lax">存储梯度计算结果。</td>
    <td class="tg-0lax">与self保持一致</td>
    <td class="tg-0lax">与self保持一致</td>
  </tr>
</tbody></table>

## 约束说明

- 确定性计算：

    - aclnnBinaryCrossEntropyWithLogitsBackward默认确定性实现。

## 调用说明

| 调用方式   | 样例代码           | 说明                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn接口  | [test_aclnn_sigmoid_cross_entropy_with_logits_grad_v2.cpp](examples/test_aclnn_sigmoid_cross_entropy_with_logits_grad_v2.cpp) | 通过[aclnnBinaryCrossEntropyWithLogitsBackward](docs/aclnnBinaryCrossEntropyWithLogitsBackward.md)接口方式调用SigmoidCrossEntropyWithLogitsGradV2算子。 |
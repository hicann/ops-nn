# SigmoidCrossEntropyWithLogitsGrad

## 产品支持情况

|产品             |  是否支持  |
|:-------------------------|:----------:|
|  <term>Ascend 950PR/Ascend 950DT</term>   |     √    |
|  <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>   |     √    |
|  <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>   |     √    |
|  <term>Atlas 200I/500 A2 推理产品</term>   |     ×    |
|  <term>Atlas 推理系列产品</term>   |     ×    |
|  <term>Atlas 训练系列产品</term>   |     ×    |

## 功能说明

- 算子功能：Sigmoid与二元交叉熵（BCELoss）融合损失的反向梯度计算。将输入`predict`经`sigmoid`计算，与标签`target`相减后乘以上游梯度`d_out`，得到关于`predict`的梯度。
- 计算公式：

  $$
  gradient = (sigmoid(predict) - target) \cdot d_out
  $$

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
    <th class="tg-0lax">输入/输出</th>
    <th class="tg-0lax">描述</th>
    <th class="tg-0lax">数据类型</th>
    <th class="tg-0lax">数据格式</th>
  </tr></thead>
<tbody>
  <tr>
    <td class="tg-0lax">predict</td>
    <td class="tg-0lax">输入</td>
    <td class="tg-0lax">网络正向前一层的计算结果（sigmoid前的logits），决定输出的shape和dtype。</td>
    <td class="tg-0lax">FLOAT16、FLOAT、BFLOAT16</td>
    <td class="tg-0lax">ND</td>
  </tr>
  <tr>
    <td class="tg-0lax">target</td>
    <td class="tg-0lax">输入</td>
    <td class="tg-0lax">样本的标签值，shape与predict保持一致。</td>
    <td class="tg-0lax">与predict保持一致</td>
    <td class="tg-0lax">ND</td>
  </tr>
  <tr>
    <td class="tg-0lax">d_out</td>
    <td class="tg-0lax">输入</td>
    <td class="tg-0lax">网络反向传播前一步的梯度值，shape可broadcast到predict的shape。</td>
    <td class="tg-0lax">与predict保持一致</td>
    <td class="tg-0lax">ND</td>
  </tr>
  <tr>
    <td class="tg-0lax">gradient</td>
    <td class="tg-0lax">输出</td>
    <td class="tg-0lax">存储梯度计算结果，shape与dtype与predict保持一致。</td>
    <td class="tg-0lax">与predict保持一致</td>
    <td class="tg-0lax">ND</td>
  </tr>
</tbody></table>

## 约束说明

- 确定性计算：SigmoidCrossEntropyWithLogitsGrad默认确定性实现。

## 调用说明

| 调用方式   | 样例代码           | 说明                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| 图模式(GE IR)  | [test_geir_sigmoid_cross_entropy_with_logits_grad.cpp](examples/test_geir_sigmoid_cross_entropy_with_logits_grad.cpp) | 通过GE图模式方式调用SigmoidCrossEntropyWithLogitsGrad算子，算子原型定义见[sigmoid_cross_entropy_with_logits_grad_proto.h](op_graph/sigmoid_cross_entropy_with_logits_grad_proto.h)。 |

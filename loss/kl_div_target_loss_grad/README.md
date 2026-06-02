# KlDivTargetLossGrad

## 产品支持情况

|产品             |  是否支持  |
|:-------------------------|:----------:|
|  <term>Ascend 950PR/Ascend 950DT</term>   |     √    |
|  <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>   |     ×    |
|  <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>     |     ×    |
|  <term>Atlas 200I/500 A2 推理产品</term>    |     ×    |
|  <term>Atlas 推理系列产品</term>     |     ×    |
|  <term>Atlas 训练系列产品</term>    |     ×    |

## 功能说明

- 算子功能：进行[aclnnKlDiv](https://gitcode.com/cann/ops-math/blob/master/math/kl_div_v2/docs/aclnnKlDiv.md) api的结果的target反向计算。

- 计算公式：

$$
out = 
\begin{cases}
gradOutput * (log(target) - self + 1), ~~~~~~~~~~~~~~~~~~~~~~~~~logTarget=False \\
gradOutput * exp(target) * (target - self +1), ~~~~~~~~~~logTarget=True
\end{cases}
$$

## 参数说明

<table style="undefined;table-layout: fixed; width: 1420px"><colgroup>
  <col style="width: 215px">
  <col style="width: 163px">
  <col style="width: 287px">
  <col style="width: 439px">
  <col style="width: 135px">
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
        <td>gradOutput</td>
        <td>输入</td>
        <td>梯度反向输入，公式中的gradOutput，shape需要与target满足broadcast关系。</td>
        <td>FLOAT、FLOAT16、BFLOAT16</td>
        <td>ND</td>
      </tr>
      <tr>
        <td>self</td>
        <td>输入</td>
        <td>输入张量，公式中的self，shape需要与target满足broadcast关系。</td>
        <td>FLOAT、FLOAT16、BFLOAT16</td>
        <td>ND</td>
      </tr>
      <tr>
        <td>target</td>
        <td>输入</td>
        <td>输入张量，公式中的target。</td>
        <td>FLOAT、FLOAT16、BFLOAT16</td>
        <td>ND</td>
      </tr>
      <tr>
        <td>reduction</td>
        <td>属性</td>
        <td>指定要应用到输出的缩减方式，支持0(none) | 1(mean) | 2(sum) | 3(batchmean)。
            <ul>
            <li>0(none)：不应用缩减</li>
            <li>1(mean)：输出的总和将除以输出中的元素数</li>
            <li>2(sum)：输出将被求和</li>
            <li>3(batchmean)：输出的总和将除以batch的个数</li>
            </ul>
        </td>
        <td>STRING</td>
        <td>-</td>
      </tr>
      <tr>
        <td>logTarget</td>
        <td>属性</td>
        <td>是否对target进行log空间转换。</td>
        <td>BOOL</td>
        <td>-</td>
      </tr>
      <tr>
        <td>out</td>
        <td>输出</td>
        <td>输出的损失梯度，公式中的out，shape与target相同。</td>
        <td>FLOAT、FLOAT16、BFLOAT16</td>
        <td>ND</td>
      </tr>
    </tbody></table>

## 约束说明

无

## 调用说明

| 调用方式 | 调用样例                                                                   | 说明                                                           |
|--------------|------------------------------------------------------------------------|--------------------------------------------------------------|
| aclnn调用 | [test_aclnn_kl_div_target_backward](./examples/test_aclnn_kl_div_target_backward.cpp) | 通过[aclnnKlDivTargetBackward](./docs/aclnnKlDivTargetBackward.md)接口方式调用KlDivTargetLossGrad算子。 |

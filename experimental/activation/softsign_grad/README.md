# SoftsignGrad

## 产品支持情况

| 产品 | 是否支持 |
| :----------------------------------------- | :------:|
| <term>Ascend 950PR/Ascend 950DT</term> | √ |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term> | × |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> | × |
| <term>Atlas 200I/500 A2 推理产品</term> | × |
| <term>Atlas 推理系列产品</term> | × |
| <term>Atlas 训练系列产品</term> | × |

## 功能说明

- 算子功能：完成Softsign激活函数的反向梯度计算。给定上游梯度gradients和前向输入特征features，按元素计算输出梯度。
- 计算公式：

  $$
  output_i = \frac{gradients_i}{(1 + |features_i|)^2}
  $$

  其中：
  - $gradients$ 为上游梯度张量。
  - $features$ 为Softsign前向函数的输入特征张量。
  - $output$ 为计算得到的梯度输出张量。

## 参数说明

<table style="table-layout: fixed; width: 1576px"><colgroup>
<col style="width: 170px">
<col style="width: 170px">
<col style="width: 300px">
<col style="width: 200px">
<col style="width: 170px">
</colgroup>
<thead>
  <tr>
    <th>参数名</th>
    <th>输入/输出</th>
    <th>描述</th>
    <th>数据类型</th>
    <th>数据格式</th>
  </tr></thead>
<tbody>
  <tr>
    <td>gradients</td>
    <td>输入</td>
    <td>上游梯度张量，对应公式中的gradients。支持0-8维tensor，支持空tensor和非连续tensor。shape需要与features的shape完全相同。</td>
    <td>FLOAT、FLOAT16、BFLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>features</td>
    <td>输入</td>
    <td>Softsign前向函数的输入特征张量，对应公式中的features。支持0-8维tensor，支持空tensor和非连续tensor。shape需要与gradients的shape完全相同。</td>
    <td>FLOAT、FLOAT16、BFLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>output</td>
    <td>输出</td>
    <td>计算得到的梯度输出张量，对应公式中的output。shape与输入相同，数据类型与输入一致。</td>
    <td>FLOAT、FLOAT16、BFLOAT16</td>
    <td>ND</td>
  </tr>
</tbody></table>

## 约束说明

- gradients和features的数据类型必须一致，且仅支持FLOAT、FLOAT16、BFLOAT16。
- gradients和features的shape必须完全相同，不支持广播。
- FP16/BF16输入在内部会提升至FP32精度计算，结果再转换回原始精度。
- aclnnSoftsignBackward默认确定性实现。

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
    <td><a href="examples/test_aclnn_softsign_grad.cpp">test_aclnn_softsign_grad</a></td>
    <td rowspan="2">参见<a href="docs/aclnnSoftsignBackward.md">aclnnSoftsignBackward接口文档</a>了解接口详情。</td>
  </tr>
  <tr>
    <td>图模式调用</td>
    <td><a href="examples/test_geir_softsign_grad.cpp">test_geir_softsign_grad</a></td>
  </tr>
</tbody></table>

## 参考资源

* [aclnnSoftsignBackward接口文档](docs/aclnnSoftsignBackward.md)

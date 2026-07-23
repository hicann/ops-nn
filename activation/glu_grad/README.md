# GLUGrad

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

- 接口功能：GLU的反向梯度计算。将输入张量self沿着指定的维度dim平均分成两个张量a和b，根据gradOut计算梯度。
- 计算公式：

  $$
  \frac{\partial GLU(a,b)}{\partial(a,b)}=cat(\sigma(b),\sigma(b) \otimes a \otimes (1-\sigma(b)))
  $$

  数学计算表达式：
  假设输出的GLUGrad有两部分组成：out=[a_grad, b_grad]，则：
  - sig_b=sigmoid(b)
  - **a_grad**=grad_out*sig_b
  - **b_grad**=a_grad*(a - a*sig_b)

  其中：grad_out为梯度输出，a表示的是输入张量self根据指定dim进行均分后的前部分张量，b表示后半部分张量。

## 参数说明

<table style="undefined;table-layout: fixed; width: 980px"><colgroup>
  <col style="width: 100px">
  <col style="width: 150px">
  <col style="width: 280px">
  <col style="width: 330px">
  <col style="width: 120px">
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
      <td>梯度输出张量。</td>
      <td>DOUBLE、FLOAT、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>self</td>
      <td>输入</td>
      <td>前向传播的输入张量。</td>
      <td>DOUBLE、FLOAT、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>dim</td>
      <td>属性</td>
      <td>表示要拆分输入self的维度。</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>out</td>
      <td>输出</td>
      <td>梯度输出，shape与self一致。</td>
      <td>DOUBLE、FLOAT、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## 约束说明

无

## 调用说明

| 调用方式 | 调用样例                                                                   | 说明                                                           |
|--------------|------------------------------------------------------------------------|--------------------------------------------------------------|
| aclnn调用 | [test_aclnn_glu_backward](examples/test_aclnn_glu_backward.cpp) | 通过[aclnnGluBackward](docs/aclnnGluBackward.md)接口方式调用GluGrad算子。 |

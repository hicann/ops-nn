# SwiGluGrad

## 产品支持情况

|产品             |  是否支持  |
|:-------------------------|:----------:|
|  <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>   |     √    |
|  <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>     |     √    |

## 功能说明
- 算子功能：完成[aclnnSwiGlu](../swi_glu/docs/aclnnSwiGlu.md)的反向计算，完成x的SwiGlu反向梯度计算。

- 计算公式：
  <p style="text-align: center">
  out<sub>A</sub> = yGrad<sub>i</sub>*[sigmoid(A)+sigmoid(A)*(1-sigmoid(A)*A)]*B
  </p>
  <p style="text-align: center">
  out<sub>B</sub> = yGrad<sub>i</sub>*sigmoid(A)*A
  </p>
  其中，A表示输入x根据指定dim维度一分为二后的前部分张量，B表示x均分后的后半部分张量。out<sub>A</sub>和out<sub>B</sub>合并为out。

## 参数说明

<table style="undefined;table-layout: fixed; width: 970px"><colgroup>
  <col style="width: 181px">
  <col style="width: 144px">
  <col style="width: 273px">
  <col style="width: 256px">
  <col style="width: 116px">
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
      <td>yGrad</td>
      <td>输入</td>
      <td>公式中的输入yGrad<sub>i</sub>。</td>
      <td>BFLOAT16、FLOAT16、FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>x</td>
      <td>输入</td>
      <td>公式中的输入A和B<sub>i</sub>。</td>
      <td>BFLOAT16、FLOAT16、FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
    <tr>
      <td>dim</td>
      <td>属性</td>
      <td><ul><li>需要进行切分的维度序号。</li><li>取值范围为[-x.dim(), x.dim()-1]。</li></ul></td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>out</td>
      <td>输出</td>
      <td>公式中的out<sub>A</sub>和out<sub>B</sub>的合并。</td>
      <td>BFLOAT16、FLOAT16、FLOAT32</td>
      <td>ND</td>
    </tr>

  </tbody></table>


## 约束说明

无

## 调用说明

| 调用方式 | 调用样例                                                                   | 说明                                                             |
|--------------|------------------------------------------------------------------------|----------------------------------------------------------------|
| aclnn调用 | [test_aclnn_swi_glu_grad](./examples/test_aclnn_swi_glu_grad.cpp) | 通过[aclnnSwigluGrad](./docs/aclnnSwiGluGrad.md)接口方式调用SwiGluGrad算子。    |
| 图模式调用 | -   | 通过[算子IR](./op_graph/swi_glu_grad_proto.h)构图方式调用SwiGluGrad算子。 |




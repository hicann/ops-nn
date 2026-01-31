# MaxPool3D

## 产品支持情况

|产品             |  是否支持  |
|:-------------------------|:----------:|
| <term>Ascend 950PR/Ascend 950DT</term>                     |    √     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    √     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>     |    √     |
| <term>Atlas 200I/500 A2 推理产品</term>                     |    x     |
| <term>Atlas 推理系列产品</term>                             |    √     |
| <term>Atlas 训练系列产品</term>                             |    √     |

## 功能说明

- 接口功能：
对于输入信号的输入通道，提供3维最大池化（Max pooling）操作。
- 计算公式：
  - output tensor中每个元素的计算公式：

    $$
    out(N_i, C_j, d, h, w) = \max\limits_{{k\in[0,k_{D}-1],m\in[0,k_{H}-1],n\in[0,k_{W}-1]}}input(N_i,C_j,stride[0]\times d + k, stride[1]\times h + m, stride[2]\times w + n)
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
      <td>x</td>
      <td>输入</td>
      <td>输入的张量。</td>
      <td>FLOAT16、FLOAT、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>ksize</td>
      <td>属性</td>
      <td>最大池化的窗口大小。</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>strides</td>
      <td>属性</td>
      <td>窗口移动的步长。</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>padding</td>
      <td>属性</td>
      <td>指定padding的模式。</td>
      <td>STRING</td>
      <td>-</td>
    </tr>
    <tr>
      <td>pads</td>
      <td>属性</td>
      <td>每一条边补充的层数。</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>dilations</td>
      <td>属性</td>
      <td>控制窗口中元素的步幅。</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>ceilMode</td>
      <td>属性</td>
      <td>计算输出形状的取整模式。</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>data_fromat</td>
      <td>属性</td>
      <td>支持的数据格式</td>
      <td>STRING</td>
      <td>-</td>
    </tr>
    <tr>
      <td>y</td>
      <td>输出</td>
      <td>输出的张量。</td>
      <td>FLOAT16、FLOAT、BFLOAT16</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## 约束说明
- **值域限制说明：**
  - ksize：数组长度必须为5，且N和C维度对应的值必须为1。
  - strides：数组长度必须为5，且N和C维度对应的值必须为1。
  - padding：只支持三种模式：“SAME”、“VALID”、“CALCULATED”。
  - pads：该参数仅在padding模式为“CALCULATED”时生效。
  - dilations：数组长度必须为5，且N和C维度对应的值必须为1。
  - ceilMode：取值为0时，代表False，向下取整；非0值时，代表True，向上取整，该参数仅在padding模式为“CALCULATED”时生效。


## 调用说明

| 调用方式   | 样例代码           | 说明                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| 图模式接口 | [test_max_pool_3d](examples/test_max_pool_3d.cpp) | 通过IR[MaxPool3D](./op_graph/max_pool3_d_proto.h)构图方式调用MaxPool3D算子。 |
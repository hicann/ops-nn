# ThresholdGradV2D

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

- 算子功能：完成threshold正向操作的反向传播梯度计算。当threshold==0时等价于ReluGrad。

- 计算公式：

  $$
      out_i =
      \begin{cases}
      gradOutput_i, \quad self_i > threshold\\
      0,  \quad self_i \leq threshold
      \end{cases}
  $$

## 参数说明

<table style="undefined;table-layout: fixed; width: 1430px">
  <colgroup>
    <col style="width: 127px">
    <col style="width: 120px">
    <col style="width: 273px">
    <col style="width: 292px">
    <col style="width: 152px">
  </colgroup>
  <thead>
    <tr>
      <th>参数名</th>
      <th>输入/输出/属性</th>
      <th>描述</th>
      <th>数据类型</th>
      <th>数据格式</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>gradOutput</td>
      <td>输入</td>
      <td>上游梯度，与self广播兼容。</td>
      <td>FLOAT16、FLOAT、BF16、INT32、INT8、UINT8</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>self</td>
      <td>输入</td>
      <td>正向输入，决定门控掩码。</td>
      <td>FLOAT16、FLOAT、BF16、INT32、INT8、UINT8</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>threshold</td>
      <td>属性</td>
      <td>阈值标量，OPTIONAL，默认1.0。threshold==0时等价ReluGrad。</td>
      <td>FLOAT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>out</td>
      <td>输出</td>
      <td>反向梯度，dtype与self一致，shape为广播后形状。</td>
      <td>FLOAT16、FLOAT、BF16、INT32、INT8、UINT8</td>
      <td>ND</td>
    </tr>
  </tbody>
</table>

## 约束说明

- gradOutput、self、out三者dtype一致。
- threshold==0走ReluGrad路径（regbase上额外支持INT64）。

## 调用说明

| 调用方式   | 调用样例                                                | 说明                            |
| :--------- | :------------------------------------------------------ | :------------------------------ |
| aclnn调用 | [test_aclnn_threshold_backward.cpp](./examples/test_aclnn_threshold_backward.cpp) | 通过[aclnnThresholdBackward](./docs/aclnnThresholdBackward.md)接口方式调用threshold_grad_v2_d算子。 |

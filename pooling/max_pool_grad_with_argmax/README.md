# MaxPoolGradWithArgmax

## 产品支持情况

|产品             |  是否支持  |
|:-------------------------|:----------:|
|  <term>Ascend 950PR/Ascend 950DT</term>                   |     √    |
|  <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>   |     ×    |
|  <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>   |     ×    |
|  <term>Atlas 200I/500 A2 推理产品</term>                   |     ×    |
|  <term>Atlas 推理系列产品</term>                           |     ×    |
|  <term>Atlas 训练系列产品</term>                           |     ×    |

## 功能说明

- 算子功能：正向最大池化MaxPoolWithArgmax的反向梯度计算。

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
      <td>x</td>
      <td>输入</td>
      <td>输入x，形状为[NCHW]或[NHWC]</td>
      <td>FLOAT16、BFLOAT16、FLOAT</td>
      <td>NCHW、NHWC</td>
    </tr>
    <tr>
      <td>grad</td>
      <td>输入</td>
      <td>梯度Tensor，形状为[NCHW]或[NHWC]</td>
      <td>FLOAT16、BFLOAT16、FLOAT</td>
      <td>NCHW、NHWC</td>
    </tr>
    <tr>
      <td>argmax</td>
      <td>输入</td>
      <td>正向最大池化输出的索引，形状与grad相同</td>
      <td>INT32、INT64</td>
      <td>NCHW、NHWC</td>
    </tr>
    <tr>
      <td>ksize</td>
      <td>属性（必选）</td>
      <td>池化窗口大小，长度为4的列表。NCHW格式时ksize[0]=1且ksize[1]=1；NHWC格式时ksize[0]=1且ksize[3]=1</td>
      <td>ListInt</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>strides</td>
      <td>属性（必选）</td>
      <td>窗口移动步长，长度为4的列表。NCHW格式时strides[0]=1且strides[1]=1；NHWC格式时strides[0]=1且strides[3]=1</td>
      <td>ListInt</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>padding</td>
      <td>属性（必选）</td>
      <td>填充算法，取值为"SAME"或"VALID"</td>
      <td>String</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>include_batch_in_index</td>
      <td>属性（可选）</td>
      <td>是否在计算argmax索引时包含batch维度。当前仅支持false。默认值：false</td>
      <td>Bool</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>data_format</td>
      <td>属性（可选）</td>
      <td>数据布局格式，取值为"NHWC"或"NCHW"。默认值："NHWC"</td>
      <td>String</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y</td>
      <td>输出</td>
      <td>输出梯度，形状与x相同</td>
      <td>FLOAT16、BFLOAT16、FLOAT</td>
      <td>NCHW、NHWC</td>
    </tr>
  </tbody></table>

## 约束说明

- include_batch_in_index：当前仅支持false

## 调用说明

| 调用方式   | 样例代码           | 说明                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| 图模式接口 | [test_geir_max_pool_grad_with_argmax](examples/arch35/test_geir_max_pool_grad_with_argmax.cpp) | 通过IR [MaxPoolGradWithArgmax](./op_graph/max_pool_grad_with_argmax_proto.h)构图方式调用MaxPoolGradWithArgmax算子。 |
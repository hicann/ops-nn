# MaxPoolWithArgmax

## 产品支持情况

|产品             |  是否支持  |
|:-------------------------|:----------:|
| <term>Ascend 950PR/Ascend 950DT</term>                     |    √     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    x     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>     |    x     |
| <term>Atlas 200I/500 A2 推理产品</term>                     |    x     |
| <term>Atlas 推理系列产品</term>                             |    x     |
| <term>Atlas 训练系列产品</term>                             |    x     |

## 功能说明

- 接口功能：
执行最大池化操作，同时输出池化后的最大值和对应位置的索引。

- 计算公式：
  - output tensor中每个元素的计算公式，以 NCHW 为例：
    
    $$
    y(N_i, C_j, h, w) = \max\limits_{{k\in[0,k_{H}-1],m\in[0,k_{W}-1]}}x(N_i,C_j,stride[2]\times h + k, stride[3]\times w + m)
    $$
    
  - argmax输出最大值在池化窗口中的索引位置。

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
      <td>输入的4D张量。</td>
      <td>FLOAT16、FLOAT、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>ksize</td>
      <td>属性</td>
      <td>池化窗口的大小，长度为4，且数组元素必须都大于0。</td>
      <td>LIST_INT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>strides</td>
      <td>属性</td>
      <td>滑动窗口的步长，长度为4，且数组元素必须都大于0。</td>
      <td>LIST_INT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>padding</td>
      <td>属性</td>
      <td>指定padding的模式，支持"SAME"或"VALID"。</td>
      <td>STRING</td>
      <td>-</td>
    </tr>
    <tr>
      <td>Targmax</td>
      <td>属性</td>
      <td>指定argmax输出的数据类型，支持int32(3)或int64(9)。默认int64(9)。</td>
      <td>INT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>include_batch_in_index</td>
      <td>属性</td>
      <td>计算argmax索引时是否包含batch维度。目前仅支持false。</td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
    <tr>
      <td>data_format</td>
      <td>属性</td>
      <td>数据格式，支持"NCHW"或"NHWC"。默认"NHWC"。</td>
      <td>STRING</td>
      <td>-</td>
    </tr>
    <tr>
      <td>nan_prop</td>
      <td>属性</td>
      <td>是否处理NAN值。true时NAN参与比较，false时忽略NAN。默认false。</td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
    <tr>
      <td>y</td>
      <td>输出</td>
      <td>池化后的最大值张量，与输入x类型相同。</td>
      <td>FLOAT16、FLOAT、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>argmax</td>
      <td>输出</td>
      <td>最大值对应的索引位置。</td>
      <td>INT32、INT64</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## 约束说明

- **值域限制说明：**
  - x：4D张量，支持FLOAT16、FLOAT、BFLOAT16类型，支持NCHW和NHWC格式。
  - ksize：长度为4的列表，[NCHW]格式要求ksize[0]=1和ksize[1]=1；[NHWC]格式要求ksize[0]=1和ksize[3]=1。
  - strides：长度为4的列表，[NCHW]格式要求strides[0]=1和strides[1]=1；[NHWC]格式要求strides[0]=1和strides[3]=1。
  - padding：只支持"SAME"或"VALID"模式。
    - SAME：填充使输出形状等于ceil(输入形状/步长)，当步长为1时输出等于输入。
    - VALID：不填充，仅在有有效区域滑动，输出较小。
  - Targmax：只支持3(int32)或9(int64)。
  - include_batch_in_index：目前仅支持false，表示索引计算不包含batch维度。
  - data_format：支持"NCHW"或"NHWC"。
  - nan_prop：true时NAN值参与比较，最大值可以是NAN；false时忽略NAN值。

## 调用说明

| 调用方式   | 样例代码           | 说明                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn模式接口 | [test_aclnn_max_pool_with_argmax](examples/test_aclnn_max_pool_with_argmax.cpp) | 通过aclnn[aclnnMaxPool2dWithIndices](../max_pool3d_with_argmax_v2/docs/aclnnMaxPool2dWithIndices.md)方式调用MaxPoolWithArgmax算子。 |

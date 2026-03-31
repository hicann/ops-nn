# LeakyReluGrad

## 产品支持情况

|产品             |  是否支持  |
|:-------------------------|:----------:|
|  <term>Ascend 950PR/Ascend 950DT</term>   |     √    |
|  <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>   |     √    |
|  <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>     |     √    |
|  <term>Atlas 200I/500 A2 推理产品</term>    |     ×    |
|  <term>Atlas 推理系列产品</term>    |     √    |
|  <term>Atlas 训练系列产品</term>    |     √    |

## 功能说明

接口功能：[aclnnLeakyRelu](../leaky_relu/docs/aclnnLeakyRelu&aclnnInplaceLeakyRelu.md)激活函数反向。
计算公式：

$$
output =
\begin{cases}
gradOutput, &if\ self \gt 0 \\
gradOutput*negativeSlope, &if\ self \le 0
\end{cases}
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
      <td>gradOutput</td>
      <td>输入</td>
      <td>表示梯度。</td>
      <td>FLOAT、FLOAT16、BFLOAT16、DOUBLE</td>
      <td>ND</td>
    </tr>
      <tr>
      <td>self</td>
      <td>输入</td>
      <td>表示特性。</td>
      <td>FLOAT、FLOAT16、BFLOAT16、DOUBLE</td>
      <td>ND</td>
    </tr>
     <tr>
      <td>negativeSlope</td>
      <td>输入</td>
      <td>表示self < 0时的斜率。</td>
      <td>FLOAT、FLOAT16、DOUBLE、INT32、INT64、INT8、BOOL、INT16、UINT8、BFLOAT16</td>
      <td>-</td>
    </tr>
     <tr>
      <td>selfIsResult</td>
      <td>输入</td>
      <td>selfIsResult为true时，negativeSlope不可以是负数。</td>
      <td>-</td>
      <td>-</td>
    </tr>
      <tr>
      <td>out</td>
      <td>输出</td>
      <td>表示计算输出。</td>
      <td>FLOAT、FLOAT16、BFLOAT16、DOUBLE</td>
      <td>ND</td>
    </tr>
  </tbody>
  </table>

- <term>Atlas 推理系列产品</term>、<term>Atlas 训练系列产品</term>：gradOutput、self、和out的数据类型支持FLOAT、FLOAT16、DOUBLE。negativeSlope的数据类型支持FLOAT、FLOAT16、DOUBLE、INT32、INT64、INT8、BOOL、INT16、UINT8。

## 约束说明

无

## 调用说明

| 调用方式   | 样例代码                                                                         | 说明                                                                               |
| ---------------- |------------------------------------------------------------------------------|----------------------------------------------------------------------------------|
| aclnn接口  | [test_aclnn_leaky_relu_backward.cpp](examples/test_aclnn_leaky_relu_backward.cpp) | 通过[aclnnLeakyReluBackward](docs/aclnnLeakyReluBackward.md)接口方式调用LeakyReluGrad算子。 |

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

  <table style="undefined;table-layout: fixed; width: 1410px"><colgroup>
  <col style="width: 111px">
  <col style="width: 115px">
  <col style="width: 220px">
  <col style="width: 300px">
  <col style="width: 177px">
  <col style="width: 104px">
  <col style="width: 238px">
  <col style="width: 145px">
  </colgroup>
  <thead>
    <tr>
      <th>参数名</th>
      <th>输入/输出</th>
      <th>描述</th>
      <th>使用说明</th>
      <th>数据类型</th>
      <th>数据格式</th>
      <th>维度(shape)</th>
      <th>非连续Tensor</th>
    </tr></thead>
  <tbody>
      <tr>
      <td>gradOutput</td>
      <td>输入</td>
      <td>表示梯度。</td>
      <td><ul><li>支持空Tensor。</li><li>数据类型与self的数据类型需满足数据类型推导规则（参见<a href="../../../docs/zh/context/互推导关系.md" target="_blank">互推导关系</a>）。</li><li>shape需要与self满足<a href="../../../docs/zh/context/broadcast关系.md" target="_blank">broadcast关系</a>。</li></ul></td>
      <td>FLOAT、FLOAT16、BFLOAT16、DOUBLE</td>
      <td>ND</td>
      <td>0-8</td>
      <td>√</td>
    </tr>
      <tr>
      <td>self</td>
      <td>输入</td>
      <td>表示特性。</td>
      <td><ul><li>支持空Tensor。</li><li>数据类型与gradOutput的数据类型需满足数据类型推导规则（参见<a href="../../../docs/zh/context/互转换关系.md" target="_blank">互转换关系</a>）。</li><li>shape需要与gradOutput满足<a href="../../../docs/zh/context/broadcast关系.md" target="_blank">broadcast关系</a>。</li></ul></td>
      <td>FLOAT、FLOAT16、BFLOAT16、DOUBLE</td>
      <td>ND</td>
      <td>0-8</td>
      <td>√</td>
    </tr>
     <tr>
      <td>negativeSlope</td>
      <td>输入</td>
      <td>表示self < 0时的斜率。</td>
      <td>-</td>
      <td>FLOAT、FLOAT16、DOUBLE、INT32、INT64、INT8、BOOL、INT16、UINT8、BFLOAT16</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
     <tr>
      <td>selfIsResult</td>
      <td>输入</td>
      <td>-</td>
      <td>selfIsResult为true时，negativeSlope不可以是负数。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
      <tr>
      <td>out</td>
      <td>输出</td>
      <td>表示计算输出。</td>
      <td><ul><li>不需要额外申请空间，数据类型需要是gradOutput与self推导之后可转换的数据类型（参见<a href="../../../docs/zh/context/互转换关系.md" target="_blank">互转换关系</a>）。</li><li>其他数据类型（INT8、UINT8、INT16、UINT16、INT32、UINT32、INT64、UINT64、BOOL、COMPLEX64、COMPLEX128）通过自动cast能力支持，但会额外申请空间。</li></ul></td>
      <td>FLOAT、FLOAT16、BFLOAT16、DOUBLE</td>
      <td>ND</td>
      <td>0-8</td>
      <td>√</td>
    </tr>
     <tr>
      <td>workspaceSize</td>
      <td>输出</td>
      <td>返回需要在Device侧申请的workspace大小。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
      <tr>
      <td>executor</td>
      <td>输出</td>
      <td>返回op执行器，包含了算子计算流程。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
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
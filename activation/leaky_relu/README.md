# LeakyRelu

## 产品支持情况

|产品             |  是否支持  |
|:-------------------------|:----------:|
|  <term>Ascend 950PR/Ascend 950DT</term>   |     √    |
|  <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>   |     √    |
|  <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>     |     √    |

## 功能说明

- 算子功能：激活函数，用于解决Relu函数在输入小于0时输出为0的问题，避免神经元无法更新参数。
- 计算公式：

  $$
  out = max(0,self) + negativeSlope * min(0,self)
  $$

## 参数说明

  <table style="undefined;table-layout: fixed; width: 1310px"><colgroup>
  <col style="width: 111px">
  <col style="width: 115px">
  <col style="width: 220px">
  <col style="width: 177px">
  <col style="width: 104px">
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
      <td>self</td>
      <td>输入</td>
      <td>待进行LeakyRelu激活函数的入参，公式中的self。</td>
      <td>FLOAT、FLOAT16、BFLOAT16、DOUBLE</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>negativeSlope</td>
      <td>输入</td>
      <td>表示self < 0时的斜率，公式中的negativeSlope。</td>
      <td>FLOAT</td>
      <td>-</td>
    </tr>
      <tr>
      <td>out</td>
      <td>输出</td>
      <td>待进行LeakyRelu激活函数的出参。</td>
      <td>FLOAT、FLOAT16、BFLOAT16、DOUBLE</td>
      <td>ND</td>
    </tr>
  </tbody>
  </table>

## 约束说明

无

## 调用说明

| 调用方式   | 样例代码                                                                         | 说明                                                                                 |
| ---------------- |------------------------------------------------------------------------------|------------------------------------------------------------------------------------|
| aclnn接口  | [test_aclnn_leaky_relu.cpp](examples/test_aclnn_leaky_relu.cpp) | 通过[aclnnLeakyRelu&aclnnInplaceLeakyRelu](docs/aclnnLeakyRelu&aclnnInplaceLeakyRelu.md)接口方式调用LeakyRelu算子。 |
| aclnn接口  | [test_aclnn_inplace_leaky_relu.cpp](examples/test_aclnn_inplace_leaky_relu.cpp) | 通过[aclnnLeakyRelu&aclnnInplaceLeakyRelu](docs/aclnnLeakyRelu&aclnnInplaceLeakyRelu.md)接口方式调用LeakyRelu算子。 |
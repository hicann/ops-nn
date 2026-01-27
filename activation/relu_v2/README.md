# ReluV2

## 产品支持情况

|产品             |  是否支持  |
|:-------------------------|:----------:|
|  <term>Ascend 950PR/Ascend 950DT</term>   |     √    |
|  <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>   |     √    |
|  <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>     |     √    |
|  <term>Atlas 200I/500 A2 推理产品</term>    |     ×   |
|  <term>Atlas 推理系列产品</term>    |     √    |
|  <term>Atlas 训练系列产品</term>    |     √    |

## 功能说明

- 算子功能：激活函数，返回与输入tensor shape相同的tensor，tensor中value大于等于0时，取该value，小于0，取0。
- 计算公式：
$$
relu_v2(self) = \begin{cases} self, & self\gt 0 \\ 0, & self\le 0 \end{cases}
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
      <td>待进行ReluV2计算的入参，公式中的输入self。</td>
      <td>FLOAT、FLOAT16、BFLOAT16、INT8、INT32、UINT8</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y</td>
      <td>输出</td>
      <td>进行ReluV2计算后的输出，公式中的relu_v2(self)。</td>
      <td>FLOAT、FLOAT16、INT8、INT32、INT64、BFLOAT16、UINT8</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>mask</td>
      <td>输出</td>
      <td>进行ReluV2计算后的输出，当输入的值大于0时，该值为1，当输入的值小于等于0时，该值为0。</td>
      <td>UINT1</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## 约束说明

无

## 调用说明

| 调用方式 | 调用样例                                                                   | 说明                                                           |
|--------------|------------------------------------------------------------------------|--------------------------------------------------------------|
| 图模式调用 | [test_geir_relu_v2](./examples/test_geir_relu_v2.cpp)   | 通过[算子IR](./op_graph/relu_v2_proto.h)构图方式调用ReluV2算子。 |
# ThresholdV2

## 产品支持情况

| 产品　　　　　　　　　　　　　　　　　　　　　　　　　　 | 是否支持 |
| :---------------------------------------------------------| :--------:|
| <term>Ascend 950PR/Ascend 950DT</term>　　　　　　　　　 | √　　　　|
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term> | √　　　　|
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> | √　　　　|
| <term>Atlas 200I/500 A2 推理产品</term>　　　　　　　　　| ×　　　　|
| <term>Atlas 推理系列产品</term>　　　　　　　　　　　　　| ×　　　　|
| <term>Atlas 训练系列产品</term>　　　　　　　　　　　　　| √　　　　|

## 功能说明

- 算子功能：对输入x进行阈值操作。当x中的elements大于threshold时，返回elements；否则，返回value。

- 计算公式：

$$
out(x) = \begin{cases} x, & x\gt threshold \\ value, & otherwise \end{cases}
$$

## 参数说明

<table style="undefined;table-layout: fixed; width: 820px"><colgroup>
  <col style="width: 100px">
  <col style="width: 150px">
  <col style="width: 190px">
  <col style="width: 260px">
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
      <td>对应公式中的x</td>
      <td>BFLOAT16、FLOAT16、FLOAT、INT8、INT32、INT64、UINT8</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>threshold</td>
      <td>输入</td>
      <td>对应公式中的threshold</td>
      <td>BFLOAT16、FLOAT16、FLOAT、INT8、INT32、INT64、UINT8</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>value</td>
      <td>输入</td>
      <td>对应公式中的value</td>
      <td>BFLOAT16、FLOAT16、FLOAT、INT8、INT32、INT64、UINT8</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y</td>
      <td>输出</td>
      <td>公式中的输出张量</td>
      <td>BFLOAT16、FLOAT16、FLOAT、INT8、INT32、INT64、UINT8</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## 调用说明

| 调用方式　| 样例代码　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　| 说明　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　 |
| -----------| -----------------------------------------------------------------------------| ----------------------------------------------------------------------------------------------------------|
| 图模式　　| [test_geir_threshold_v2.cpp](./examples/test_geir_threshold_v2.cpp)　　　　 | 通过[算子IR](./op_graph/threshold_v2_proto.h)构图方式调用ThresholdV2算子。　　　　　　　　　　　　　　　 |
| aclnn调用 | [test_aclnn_threshold](../threshold/examples/test_aclnn_threshold.cpp)　　　　　　　　 | 通过[aclnnThreshold](../threshold/docs/aclnnThreshold&aclnnInplaceThreshold.md)接口方式调用ThresholdV2算子。　　　　|
| aclnn调用 | [test_aclnn_inplace_threshold](../threshold/examples/test_aclnn_inplace_threshold.cpp) | 通过[aclnnInplaceThreshold](../threshold/docs/aclnnThreshold&aclnnInplaceThreshold.md)接口方式调用ThresholdV2算子。 |

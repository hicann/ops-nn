# RepeatInterleaveGrad

##  产品支持情况

| 产品 | 是否支持 |
| ---- | :----:|
|Atlas A3 训练系列产品/Atlas A3 推理系列产品|√|
|Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件|√|
 

## 功能说明

- 算子功能：算子RepeatInterleave的反向, 将y_grad tensor的axis维度按repeats进行ReduceSum。

## 参数说明

<table style="undefined;table-layout: fixed; width: 1250px"><colgroup>
  <col style="width: 60px">
  <col style="width: 60px">
  <col style="width: 310px">
  <col style="width: 150px">
  <col style="width: 60px">
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
      <td>y_grad</td>
      <td>输入</td>
      <td>表示待被ReduceSum的输入tensor。</td>
      <td>FLOAT、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>repeats</td>
      <td>输入</td>
      <td>表示重复的次数。</td>
      <td>INT32、INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>x_grad</td>
      <td>输出</td>
      <td>表示ReduceSum完成的输出tensor。</td>
      <td>FLOAT、INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>axis</td>
      <td>可选属性</td>
      <td><ul><li>表示ReduceSum作用的维度。取值范围为[-n, n)，其中n为y_grad的维度。</li><li>默认值为-1。</li></td>
      <td>int64_t</td>
      <td>-</td>
    </tr>
  </tbody></table>

## 约束说明

无

## 调用说明

| 调用方式 | 调用样例                                                                   | 说明                                                           |
|--------------|------------------------------------------------------------------------|--------------------------------------------------------------|
| aclnn调用 | [test_aclnn_repeat_interleave_grad](./examples/test_aclnn_repeat_interleave_grad.cpp) | 通过[aclnnRepeatInterleaveGrad](./docs/aclnnRepeatInterleaveGrad.md)接口方式调用RepeatInterleaveGrad算子。 |
| 图模式调用 | - | 通过[算子IR](./op_graph/repeat_interleave_grad_proto.h)构图方式调用RepeatInterleaveGrad算子。 |

# SoftmaxV2

[📄 查看源码](https://gitcode.com/cann/ops-nn/tree/master/activation/softmax_v2)

## 产品支持情况

| 产品                                           | 是否支持 |
|:---------------------------------------------| :------: |
| <term>Ascend 950PR/Ascend 950DT</term>                             |    √     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    √     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> |    √     |
| <term>Atlas 200I/500 A2 推理产品</term>    |     √    |
| <term>Atlas 推理系列产品 </term>    |     √    |
| <term>Atlas 训练系列产品</term>    |     √    |
## 功能说明

- 算子功能：对输入张量计算Softmax值。

- 计算公式：

  $$
  Softmax\left ( {{x}_{i}} \right )= {\frac {{e}^{{x}_{i}}} {\sum ^{n-1}_{j=0} {{e}^{{x}_{j}}}}}
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
      <td>待进行Softmax计算的入参，公式中的x。</td>
      <td>FLOAT,FLOAT16,BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>axes</td>
      <td>输入</td>
      <td>需要做Softmax的轴。只支持1维，默认-1。</td>
      <td>INT32_ARRAY</td>
      <td>-</td>
    </tr>
    <tr>
      <td>half_to_float</td>
      <td>输入</td>
      <td>可选的布尔值。此参数决定当输入数据类型为float16时，是否将输出数据类型转换为float32。默认值为false。值为false时，输入输出数据类型要求一致。如果为true且输入数据类型为float16，则输出数据类型应为float32。</td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
    <tr>
      <td>y</td>
      <td>输出</td>
      <td>公式中的输出张量</td>
      <td>FLOAT,FLOAT16,BFLOAT16</td>
      <td>ND</td>
    </tr>
  </tbody></table>


## 调用说明

| 调用方式   | 样例代码           | 说明                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn接口 | [test_aclnn_softmax_v2](examples/test_aclnn_softmax_v2.cpp) | 通过[aclnnSoftmax](docs/aclnnSoftmax.md)接口方式调用SoftmaxV2算子。 |
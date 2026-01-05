# Quantize


## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term>                             |    √     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    √     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> |    √     |

## 功能说明

- 算子功能：对输入张量x进行量化处理。

- 计算公式：

  $$
  out=round((x/scales)+zeroPoints)
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
      <td>公式中的x</td>
      <td>FLOAT、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>scales</td>
      <td>输入</td>
      <td>公式中的scales</td>
      <td>FLOAT、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>zero_points</td>
      <td>输入</td>
      <td>公式中的zeroPoints</td>
      <td>FLOAT、INT32、INT8、UINT8、BFLOAT16</td>
      <td>-</td>
    </tr>
    <tr>
      <td>y</td>
      <td>输出</td>
      <td>公式中的输出张量out</td>
      <td>INT8、UINT8、INT32、HIFLOAT8、FLOAT8_E5M2、FLOAT8_E4M3FN</td>
      <td>ND</td>
    </tr>
  </tbody></table>


## 调用说明

| 调用方式   | 样例代码           | 说明                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn接口 | [test_aclnn_quantize.cpp](examples/test_aclnn_quantize.cpp) | 通过[aclnnQuantize.md](docs/aclnnQuantize.md)接口方式调用算子。 |

# FastGelu

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR/Ascend 950DT9</term>                             |    √     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    √      |

## 功能说明

- 算子功能：
  输入fast_gelu的计算结果。
- 计算公式：
 $$ FastGelu(x_i) = \frac {x_i} {1 + \exp(-1.702 * \left| x_i \right|)} * \exp(0851 * (x_i - \left| x_i \right|)) $$
 
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
      <td>输入一个张量。</td>
      <td>DT_BF16、FLOAT16、DT_FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y</td>
      <td>输出</td>
      <td>输出一个张量。</td>
      <td>DT_BF16、FLOAT16、DT_FLOAT</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## 约束说明

无

## 调用说明
| 调用方式 | 调用样例                                                                   | 说明                                                           |
|--------------|------------------------------------------------------------------------|--------------------------------------------------------------|
| aclnn调用 | [test_aclnn_fast_gelu](examples/arch35/test_aclnn_fast_gelu.cpp) | 通过[aclnnFastGeluGetWorkspaceSize](docs/aclnnFastGelu.md)接口方式调用fast_gelu算子。 |
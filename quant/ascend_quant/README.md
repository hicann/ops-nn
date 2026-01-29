# AscendQuant

## 产品支持情况

|产品             |  是否支持  |
|:-------------------------|:----------:|
|  <term>Ascend 950PR/Ascend 950DT</term>   |     √    |
|  <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>   |     ×    |
|  <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>     |     ×    |
|  <term>Atlas 200I/500 A2 推理产品</term>    |     ×    |
|  <term>Atlas 推理系列产品</term>    |     ×    |
|  <term>Atlas 训练系列产品</term>    |     ×    |

## 功能说明

- 算子功能：对输入x进行量化操作，且scale和offset的size需要是x的最后一维或1。
- 计算公式：
  - sqrtMode为false时，计算公式为:
    
    $$
    y = round((x * scale) + offset)
    $$

  - sqrtMode为true时，计算公式为:
    
    $$
    y = round((x * scale * scale) + offset)
    $$

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
      <td>待进行ascend_quant计算的入参，公式中的x。</td>
      <td>FLOAT、FLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>scale</td>
      <td>输入</td>
      <td>待进行ascend_quant计算的入参，公式中的scale。</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>offset</td>
      <td>输入</td>
      <td>待进行ascend_quant计算的入参，公式中的offset。</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>sqrt_mode</td>
      <td>输入</td>
      <td>待进行ascend_quant计算的入参，公式中的sqrt_mode。</td>
      <td>BOOL</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>round_mode</td>
      <td>输入</td>
      <td>待进行ascend_quant计算的入参，公式中的round_mode。</td>
      <td>STRING</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>dst_type</td>
      <td>输入</td>
      <td>待进行ascend_quant计算的入参，公式中的dst_type。</td>
      <td>INT8</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y</td>
      <td>输出</td>
      <td>待进行ascend_quant计算的出参，公式中的y。</td>
      <td>INT8、INT4、HIFLOAT8</td>
      <td>ND</td>
    </tr>
  </tbody></table>

- Ascend 950PR/Ascend 950DT </term>：数据类型支持FLOAT32、FLOAT16。

## 约束说明

无

## 调用说明

| 调用方式   | 样例代码           | 说明                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| 图模式 | -  | 通过[算子IR](op_graph/ascend_quant_proto.h)构图方式调用AscendQuant算子。         |
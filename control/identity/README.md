# Identity

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term>                             |    √     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    √     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> |    √     |
| <term>Atlas 200I/500 A2 推理产品</term>                      |    ×     |
| <term>Atlas 推理系列产品 </term>                             |    √     |
| <term>Atlas 训练系列产品</term>                              |    √     |

## 功能说明

返回一个与输入张量x具有相同形状和内容的张量y。

## 参数说明

<table style="undefined;table-layout: fixed; width: 996px"><colgroup>
  <col style="width: 102px">
  <col style="width: 168px">
  <col style="width: 203px">
  <col style="width: 403px">
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
      <td>输入张量。</td>
      <td>FLOAT, FLOAT16, INT8, INT16, UINT16, UINT8,
        INT32, INT64, UINT32, UINT64, BOOL, DOUBLE, STRING, BFLOAT16</td>
      <td>ND</td>
    </tr>    
    <tr>
      <td>y</td>
      <td>输出</td>
      <td>与输入张量x具有相同形状和内容的张量</td>
      <td>FLOAT, FLOAT16, INT8, INT16, UINT16, UINT8,
        INT32, INT64, UINT32, UINT64, BOOL, DOUBLE, STRING, BFLOAT16</td>
      <td>ND</td>
    </tr>

  </tbody></table>

## 约束说明

无

## 调用说明

| 调用方式   | 样例代码           | 说明                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| 图模式调用 | [test_geir_identity](./examples/test_geir_identity.cpp)   | 通过[算子IR](./op_graph/identity_proto.h)构图方式调用identity算子。 |
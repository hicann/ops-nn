# EmbeddingHashTableLookupOrInsert

## 产品支持情况

|产品             |  是否支持  |
|:-------------------------|:----------:|
| <term>Ascend 950PR/Ascend 950DT</term> |√|
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    ✗     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> |    ✗     |

## 功能说明

- 算子功能：根据key值查看table中是否存在key；如果存在则不插入value值，并且导出key当前位置上的值；如果不存在则对对key进行hash，找到位置后插入value。

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
      <td>table_handles</td>
      <td>输入</td>
      <td>输入hash表handle句柄，里面包含了hash表的表头地址等。</td>
      <td>INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>keys</td>
      <td>输入</td>
      <td>查询key序列。</td>
      <td>INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>values</td>
      <td>输入</td>
      <td>查询key如果已存在返回的对应位置上的value序列。</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## 约束说明

无

## 调用说明

| 调用方式   | 样例代码           | 说明                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
# EmbeddingHashTableExport

## 产品支持情况

|产品             |  是否支持  |
|:-------------------------|:----------:|
| <term>Ascend 950PR/Ascend 950DT</term> |√|
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    ✗     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> |    ✗     |
| <term>Atlas 200I/500 A2 推理产品</term> |      ✗     |
| <term>Atlas 推理系列产品</term> |      ✗     |
| <term>Atlas 训练系列产品</term> |      ✗     |

## 功能说明

- 算子功能：导出整个table表。

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
      <td>table_sizes</td>
      <td>输入</td>
      <td>table数量。</td>
      <td>INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>embedding_dims</td>
      <td>输入</td>
      <td>hash表桶深度。</td>
      <td>INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>bucket_sizes</td>
      <td>输入</td>
      <td>hash表桶数量。</td>
      <td>INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>keys</td>
      <td>输出</td>
      <td>插入key序列。</td>
      <td>INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>counters</td>
      <td>输出</td>
      <td>导出key数量。</td>
      <td>UINT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>filter_flags</td>
      <td>输出</td>
      <td>准入标志。</td>
      <td>UINT8</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>values</td>
      <td>输出</td>
      <td>导出key对应的value序列。</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## 约束说明

无

## 调用说明

| 调用方式   | 样例代码           | 说明                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| 无 | 无 | 无 |

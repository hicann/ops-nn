# InitEmbeddingHashTable

## 产品支持情况

|产品             |  是否支持  |
|:-------------------------|:----------:|
| <term>Ascend 950PR/Ascend 950DT</term> |√|
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    ✗     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> |    ✗     |

## 功能说明

- 算子功能：初始化hash表。

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
      <td>sampled_values</td>
      <td>输入</td>
      <td>外部生成随机数，用于填充value值，initializer_mode为random时生效。</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>bucket_size</td>
      <td>输入属性</td>
      <td>hash表桶数量。</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>embedding_dim</td>
      <td>输入属性</td>
      <td>hash表桶深度。</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>initializer_mode</td>
      <td>输入属性</td>
      <td>控制初始化的hash表中填充的默认值，random填充sampled_values里面的值，const填充constant_value属性值。</td>
      <td>STRING</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>constant_value</td>
      <td>输入属性</td>
      <td>hash表的初始填充值，initializer_mode为const时生效。</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## 约束说明

无

## 调用说明

| 调用方式   | 样例代码           | 说明                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
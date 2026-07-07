# EmbeddingHashTableLookupOrInsert

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
      <td>table_handle</td>
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
      <td>输出</td>
      <td>查询key如果已存在返回的对应位置上的value序列。</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>bucket_size</td>
      <td>输入属性</td>
      <td>hash表桶数量。</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>embedding_dim</td>
      <td>输入属性</td>
      <td>hash表桶深度。</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>filter_mode</td>
      <td>输入属性</td>
      <td>准入过滤模式，默认no_filter（不过滤）。</td>
      <td>STRING</td>
      <td>-</td>
    </tr>
    <tr>
      <td>filter_freq</td>
      <td>输入属性</td>
      <td>准入频次阈值，filter_mode生效时使用，默认0。</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>default_key_or_value</td>
      <td>输入属性</td>
      <td>是否使用默认key或value，默认false。</td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
    <tr>
      <td>default_key</td>
      <td>输入属性</td>
      <td>默认key值，默认0。</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>default_value</td>
      <td>输入属性</td>
      <td>默认value值，默认0.0。</td>
      <td>FLOAT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>filter_key_flag</td>
      <td>输入属性</td>
      <td>是否启用filter_key过滤，默认false。</td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
    <tr>
      <td>filter_key</td>
      <td>输入属性</td>
      <td>需要过滤的key值，默认-1。</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
  </tbody></table>

## 约束说明

无

## 调用说明

| 调用方式   | 样例代码           | 说明                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| 无 | 无 | 无 |

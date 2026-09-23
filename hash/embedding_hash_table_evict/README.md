# EmbeddingHashTableEvict

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

- 算子功能：根据输入 key 在 hash 表中查找对应桶并标记为 evicted，同时重置计数并按指定模式重新初始化 value。

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
      <td>输入 hash 表 handle 句柄，里面包含 hash 表表头地址等信息。</td>
      <td>INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>keys</td>
      <td>输入</td>
      <td>待驱逐的 key 序列。</td>
      <td>INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>sampled_values</td>
      <td>输入</td>
      <td>外部生成随机数，用于 init_mode 为 random 时填充 value。</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>table_cap</td>
      <td>输入属性</td>
      <td>hash 表容量。</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>embedding_dim</td>
      <td>输入属性</td>
      <td>hash 表桶深度。</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>init_mode</td>
      <td>输入属性</td>
      <td>驱逐后 value 重新初始化模式，可取 random 或 constant，默认 constant。</td>
      <td>STRING</td>
      <td>-</td>
    </tr>
    <tr>
      <td>const_val</td>
      <td>输入属性</td>
      <td>init_mode 为 constant 时使用的填充值，默认 0.0。</td>
      <td>FLOAT</td>
      <td>-</td>
    </tr>
  </tbody></table>

## 约束说明

无

## 调用说明

| 调用方式   | 样例代码           | 说明                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| 无 | 无 | 无 |

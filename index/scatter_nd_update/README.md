# ScatterNdUpdate

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term>                          |   √   |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    √    |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>     |    √     |
| <term>Atlas 200I/500 A2 推理产品</term>                      |    ×     |
| <term>Atlas 推理系列产品 </term>                             |    ×     |
| <term>Atlas 训练系列产品</term>                              |    ×   |

## 功能说明

- 算子功能：根据indices在给定变量内，将updates应用于变量的单个值或切片。
- 计算公式：ref[indices[k]]=updates[k]
    - ref是一个维度为 P的张量Tensor；
    - indices是一个维度为 Q的整型张量Tensor；
    - indices的shape一定是[d0,..., d_Q-2, K]，此处0<K<=P；
    - updates 是一个维度为Q−1+P−K 的张量：[d_0, ..., d_{Q-2}, ref.shape[K], ..., ref.shape[P-1]]

## 参数说明

<table style="undefined;table-layout: fixed; width: 1576px"><colgroup>
  <col style="width: 170px">
  <col style="width: 170px">
  <col style="width: 310px">
  <col style="width: 212px">
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
      <td>var</td>
      <td>输入</td>
      <td>表示一个待被更新的张量, 等同于公式中的`ref`</td>
      <td>DT_INT8、DT_INT64、DT_BOOL、DT_FLOAT16、DT_BF16、DT_FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>indices</td>
      <td>输入</td>
      <td>一个索引张量，索引到公式中的`ref`</td>
      <td>INT32、INT64。</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>updates</td>
      <td>输入</td>
      <td>使用此张量来更新var张量，必须跟输入var张量保持一样的数据类型</td>
      <td>DT_INT8、DT_INT64、DT_BOOL、DT_FLOAT16、DT_BF16、DT_FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>var</td>
      <td>输出</td>
      <td>表示更新后的张量</td>
      <td>DT_INT8、DT_INT64、DT_BOOL、DT_FLOAT16、DT_BF16、DT_FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>use_locking</td>
      <td>可选属性</td>
      <td>可选属性，默认值为`false`, 如果为`true`, 这次操作将会被一个lock保护</td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
    </tr>
  </tbody></table>

## 约束说明

- 确定性计算：
  - aclnnScatterNdUpdate默认确定性实现。

- 输入shape限制：
  - indices至少是2维，其最后1维的大小不能超过varRef的维度大小。
  - 假设indices最后1维的大小是a，则updates的shape等于indices除最后1维外的shape加上varRef除前a维外的shape。举例：varRef的shape是(4, 5, 6)，indices的shape是(3, 2)，则updates的shape必须是(3, 6)。

## 调用说明

| 调用方式   | 样例代码           | 说明                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn接口 | [test_aclnn_scatter_nd_update](examples/test_aclnn_scatter_nd_update.cpp) | 通过[aclnnScatterNdUpdate](docs/aclnnScatterNdUpdate.md)接口方式调用ScatterNdUpdate算子。 |
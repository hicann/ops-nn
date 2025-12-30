# EmbeddingDenseGradV2

##  产品支持情况

| 产品 | 是否支持 |
| ---- | :----:|
|  <term>Ascend 950PR/Ascend 950DT</term>|√|
|  <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>   |     ×    |
|  <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>     |     √    |

## 功能说明

 - 算子功能：实现Embedding的反向计算，将相同索引indices对应grad的一行累加到out上。

## 参数说明

  <table style="undefined;table-layout: fixed; width: 1250px"><colgroup>
    <col style="width: 50px">
    <col style="width: 60px">
    <col style="width: 250px">
    <col style="width: 150px">
    <col style="width: 60px">
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
        <td>grad</td>
        <td>输入</td>
        <td>表示数据的原始梯度。</td>
        <td>FLOAT、FLOAT16、BFLOAT16</td>
        <td>ND</td>
      </tr>
      <tr>
        <td>sort_indices</td>
        <td>输入</td>
        <td>表示grad输入对应的索引值。</td>
        <td>INT32、INT64</td>
        <td>ND</td>
      </tr>
      <tr>
        <td>out</td>
        <td>输出</td>
        <td>表示梯度求和的结果输出。</td>
        <td>FLOAT、FLOAT16、BFLOAT16</td>
        <td>ND</td>
      </tr>
      <tr>
        <td>num_weights</td>
        <td>属性</td>
        <td>表示输出tensor的首轴大小。</td>
        <td>Int</td>
        <td>-</td>
      </tr>
      <tr>
        <td>padding_idx</td>
        <td>可选属性</td>
        <td><ul><li>将输出tensor中第paddingIdx行填充成0，如果paddingIdx为负数则不进行处理。</li><li>默认值为-1。</li></ul></td>
        <td>Int</td>
        <td>-</td>
      </tr>
      <tr>
        <td>scale_grad_by_freq</td>
        <td>可选属性</td>
        <td><ul><li>根据单词出现的频率，是否对梯度进行缩放。</li><li>默认值为false。</li></ul></td>
        <td>Bool</td>
        <td>-</td>
      </tr>
    </tbody></table>

  - <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>、<term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：grad不支持BFLOAT16、FLOAT16，indices不支持INT64。

## 约束说明

- Atlas A3 训练系列产品/Atlas A3 推理系列产品、Atlas A2 训练系列产品/Atlas A2 推理系列产品：
  - 在参数shape超过以下限制时，输出无法保证高精度，若开启了确定性计算，也无法保证高性能。
    - grad合轴成二维shape后，第一个维度超过INT32_MAX(2147483647)。
    - num_weights超过INT32_MAX(2147483647)。
  - sort_indices合轴后维度超过INT32_INF(2139095040)时，无法保证高性能。

## 调用说明

| 调用方式 | 调用样例                                                                   | 说明                                                           |
|--------------|------------------------------------------------------------------------|--------------------------------------------------------------|
| aclnn调用 | [test_aclnn_embedding_dense_grad_v2](./examples/test_aclnn_embedding_dense_grad_v2.cpp) | 通过[aclnnEmbeddingDenseBackward](./docs/aclnnEmbeddingDenseBackward.md)接口方式调用EmbeddingDenseGradV2算子。 |
| 图模式调用 | [test_geir_embedding_dense_grad_v2](./examples/op_graph/test_geir_embedding_dense_grad_v2.cpp)   | 通过[算子IR](./op_graph/embedding_dense_grad_v2_proto.h)构图方式调用EmbeddingDenseGradV2算子。 |

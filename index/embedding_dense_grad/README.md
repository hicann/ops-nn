# EmbeddingDenseBackward

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term>                             |    √     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    √     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> |    √     |
| <term>Atlas 200I/500 A2 推理产品</term>                      |    ×     |
| <term>Atlas 推理系列产品 </term>                             |    ×     |
| <term>Atlas 训练系列产品</term>                              |    √     |

## 功能说明

  实现[Embedding]的反向计算, 将相同索引`indices`对应grad的一行累加到out上。

## 参数说明

  <table style="undefined;table-layout: fixed; width: 1250px"><colgroup>
  <col style="width: 183px">
  <col style="width: 120px">
  <col style="width: 265px">
  <col style="width: 197px">
  <col style="width: 114px">
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
      <td>计算输入</td>
      <td>数据的原始梯度，Device侧的aclTensor，支持维度2-8维，除尾轴外合轴后shape与indices合轴后shape相同；支持非连续的Tensor。</td>
      <td>BFLOAT16、FLOAT16、FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>indices</td>
      <td>计算输入</td>
      <td>grad输入对应的索引值，Device侧的aclTensor，取值范围为[0, numWeights)，支持维度1-8维，支持非连续的Tensor。</td>
      <td>FLOAT、FLOAT16、DOUBLE、INT32、INT64、INT16、INT8、UINT8、BOOL</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>numWeights</td>
      <td>计算输入</td>
      <td>输出tensor的首轴大小（uint64_t）。</td>
      <td>UINT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>paddingIdx</td>
      <td>计算输入</td>
      <td>将输出tensor中第paddingIdx行填充成0；若paddingIdx为负数则不进行处理（uint64_t）。</td>
      <td>UINT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>scaleGradByFreq</td>
      <td>计算输入</td>
      <td>是否根据单词出现频率对梯度进行缩放。true：按词频缩放；false：不处理。</td>
      <td>Bool</td>
      <td>-</td>
    </tr>
    <tr>
      <td>out</td>
      <td>计算输出</td>
      <td>梯度求和的结果输出，Device侧的aclTensor，维度为2维，首轴大小为numWeights，尾轴大小与grad尾轴相同，仅支持ND格式。</td>
      <td>BFLOAT16、FLOAT16、FLOAT</td>
      <td>ND</td>
    </tr>
  </tbody></table>

  - <term>Atlas 训练系列产品</term>：grad和out参数数据类型不支持BFLOAT16。

## 约束说明
- <term>Atlas 训练系列产品</term>：
  - 对于scale为true的场景，设定grad最后一维为embeddingDim，其大小超出指定范围时会被拦截报错。其合理范围如下：
    - indices为int32时，需满足
    $$
    embeddingDim < \frac{180192 - countsSize * 4}{36}
    $$
    - indices为int64时，需满足
    $$
    embeddingDim < \frac{180192 - countsSize * 8}{20}
    $$
    - 其中，countsSize的公式如下，coreNum代表AI处理器核数：
    $$
    countsSize = numWeights / coreNum + numWeights \% coreNum
    $$
- <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：
  - 在参数shape超过以下限制时，输出无法保证高精度，若开启了确定性计算，也无法保证高性能
    - grad合轴成二维shape后，第一个维度超过INT32_MAX(2147483647)
    - numWeights超过INT32_MAX(2147483647)
  - indices合轴后维度超过INT32_INF(2139095040)时，无法保证高性能
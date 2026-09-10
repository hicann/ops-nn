# ScatterList简介

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term>                          |    √  |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    √    |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>     |    √     |
| <term>Atlas 200I/500 A2 推理产品</term>                      |    ×     |
| <term>Atlas 推理系列产品</term>                             |    ×     |
| <term>Atlas 训练系列产品</term>                              |    ×   |
|  <term>Kirin X90 处理器系列产品</term> | √ |
|  <term>Kirin 9030 处理器系列产品</term> | √ |

## 功能说明

- 算子功能：将稀疏矩阵更新应用到变量引用中。

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
      <td>不支持空Tensor。表示待被更新的张量列表，Device侧的aclTensor，为动态输入（tensor list），原地更新。列表中每个张量的shape需相同，每个张量≥1维，数据类型需与updates一致。</td>
      <td>DT_INT8、DT_INT16、DT_INT32、DT_UINT8、DT_UINT16、DT_UINT32、DT_FLOAT16、DT_BF16、DT_FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>indice</td>
      <td>输入</td>
      <td>不支持空Tensor。表示写入位置张量，Device侧的aclTensor。shape支持1维或2维：1维时shape为(B,)，只给出写入起始位置；2维时shape为(B, 2)，给出写入起始位置与长度。B为var列表中的张量个数。数据类型为INT32或INT64。</td>
      <td>INT32、INT64。</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>updates</td>
      <td>输入</td>
      <td>不支持空Tensor。表示需要更新到var上的张量，Device侧的aclTensor。updates≥2维，第一维大小等于var列表中的张量个数，axis轴的大小不大于var对应轴，其余维度与var一致，数据类型需与var一致。</td>
      <td>DT_INT8、DT_INT16、DT_INT32、DT_UINT8、DT_UINT16、DT_UINT32、DT_FLOAT16、DT_BF16、DT_FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>mask</td>
      <td>输入</td>
      <td>不支持空Tensor。表示需要更新数据的掩码，Device侧的aclTensor，可选输入。shape支持1维，第一维大小等于var列表中的张量个数，数据类型为UINT8。取值仅支持0和1，其余取值行为未定义。</td>
      <td>DT_UINT8</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>var</td>
      <td>输出</td>
      <td>不支持空Tensor。表示更新后的张量列表，Device侧的aclTensor，与输入var共享内存，shape、数据类型均与输入var一致。</td>
      <td>DT_INT8、DT_INT16、DT_INT32、DT_UINT8、DT_UINT16、DT_UINT32、DT_FLOAT16、DT_BF16、DT_FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>reduce</td>
      <td>属性</td>
      <td>HOST侧的字符串，选择应用的reduction操作，当前仅支持取值"update"。</td>
      <td>string</td>
      <td>-</td>
    </tr>
    <tr>
      <td>axis</td>
      <td>属性</td>
      <td>用来scatter的维度。归一化（负数按updates维度数折算）后的取值必须落在开区间(0, updates维度数)内，即不能指向第0维，也不能越界。</td>
      <td>int64_t</td>
      <td>-</td>
    </tr>
  </tbody></table>

- Kirin X90/Kirin 9030 处理器系列产品：不支持BFLOAT16。

## 约束说明

- indice语义与取值范围（取值由调用方保证，算子不做校验）：indice给出的是**写入起始位置**，不是散射目标下标。写入是从该位置开始的一段连续区间，需保证整段不越出axis轴。
  - 1维：`indice[i]`为第i个张量的写入起始位置，写入长度为updates在axis轴上的大小。
  - 2维：`indice[i][0]`为起始位置，`indice[i][1]`为写入长度，长度不应超过updates在axis轴上的大小。

- mask取值范围（调用方保证，算子不做校验）：取值仅支持0和1，0表示该列表元素不执行写入，1表示执行写入，其余取值行为未定义。

## 调用说明

| 调用方式 | 样例代码 | 说明 |
| --- | --- | --- |
| aclnn接口 | [aclnnScatterList](docs/aclnnScatterList.md) | 通过[aclnnScatterList](docs/aclnnScatterList.md)接口方式调用ScatterList算子，调用示例见接口文档。 |

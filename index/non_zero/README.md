# NonZero

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term>                             |    √     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    √     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> |    √     |

## 功能说明

算子功能：找出`self`中非零元素的位置，设self的维度为D，self中非零元素的个数为N，则返回`out`的shape为N * D，每一列表示一个非零元素的位置坐标。
## 参数说明

<table style="undefined;table-layout: fixed; width: 980px"><colgroup>
  <col style="width: 100px">
  <col style="width: 150px">
  <col style="width: 280px">
  <col style="width: 330px">
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
      <td>表示待计算的目标张量，Device侧的aclTensor。</td>
      <td>FLOAT、FLOAT16、DOUBLE、INT32、INT64、INT16、INT8、UINT8、BOOL、BFLOAT16、UINT16、UINT32、UINT64。</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y</td>
      <td>输出</td>
      <td>计算输出，Device侧的aclTensor。</td>
      <td>INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>transpose</td>
      <td>属性</td>
      <td>标识输出是否需要转置。</td>
      <td>BOOL</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## 约束说明

- 确定性计算：
  - aclnnNonzero默认确定性实现。

- 由于硬件资源限制，输出索引要求在int32精度范围内。所以输入的某一维度不能超过int32的表示范围。
- 接口的输出size需要按照最大的输出size申请（即全部非0的场景，实测输出size不能超过2G），同时该接口使用的workspace也比较大，需要关注是否超过device内存大小。

## 调用说明

| 调用方式 | 调用样例                                                                   | 说明                                                           |
|--------------|------------------------------------------------------------------------|--------------------------------------------------------------|
| aclnn调用 | [test_aclnn_non_zero](./examples/test_aclnn_non_zero.cpp) | 通过[aclnnNonzero](./docs/aclnnNonzero.md)接口方式调用Nonzero算子。 |
| aclnn调用 | [test_aclnn_non_zero_v2](./examples/test_aclnn_non_zero_v2.cpp) | 通过[aclnnNonzeroV2](./docs/aclnnNonzeroV2.md)接口方式调用NonzeroV2算子。 |
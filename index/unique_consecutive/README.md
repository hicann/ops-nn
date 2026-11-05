# UniqueConsecutive

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR&950DT系列产品</term>                          |    √  |
| <term>Atlas A3系列产品</term>    |    ×    |
| <term>Atlas A2系列产品</term>     |    ×     |
| <term>Atlas 200I/500 A2推理产品</term>                      |    ×     |
| <term>Atlas推理系列产品</term>                             |    ×     |
| <term>Atlas训练系列产品</term>                              |    ×   |

## 功能说明

- 算子功能：去除每一个元素后的重复元素。当dim不为空时，去除对应维度上的每一个张量后的重复张量。

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
      <td>待进行UniqueConsecutive去重计算的入参。</td>
      <td>BF16、FLOAT16、FLOAT、INT8、INT16、INT32、INT64、UINT8、UINT16、UINT32、UINT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y</td>
      <td>输出</td>
      <td>元素消除连续重复后的输出。</td>
      <td>BF16、FLOAT16、FLOAT、INT8、INT16、INT32、INT64、UINT8、UINT16、UINT32、UINT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>idx</td>
      <td>输出</td>
      <td>返回self中各元素在输出y中的位置，return_idx为True时生效。</td>
      <td>INT32、INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>count</td>
      <td>输出</td>
      <td>返回输出y中各元素重复次数，return_counts为True时生效。</td>
      <td>INT32、INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>return_idx</td>
      <td>属性</td>
      <td>是否返回self中各元素在输出y中的位置，默认值为false。</td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
    <tr>
      <td>return_counts</td>
      <td>属性</td>
      <td>是否返回输出y中各元素重复次数，默认值为false。</td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
    <tr>
      <td>axis</td>
      <td>属性</td>
      <td>去重维度，默认值为1000，表示不指定维度（展平模式，将输入展平后去重）。</td>
      <td>INT32</td>
      <td>-</td>
    </tr>
    <tr>
      <td>out_idx</td>
      <td>属性</td>
      <td>指定输出idx和count的数据类型，仅支持INT32、INT64，默认值为INT64。</td>
      <td>INT32、INT64</td>
      <td>-</td>
    </tr>
  </tbody></table>

## 约束说明

- axis（aclnn接口对应参数dim）：AI Core路径仅支持取值1000（展平模式，将输入展平后去重），AI CPU路径支持按指定维度去重。
- return_idx（aclnn接口对应参数returnInverse）：AI Core路径仅支持false，AI CPU路径支持true。
- 输出idx与count的数据类型仅支持INT32、INT64，由属性out_idx指定，默认为INT64，且两者数据类型须一致。
- 算子为确定性实现，相同输入在相同环境下计算结果确定。

## 调用说明

| 调用方式 | 样例代码                                                                   | 说明                                                           |
|--------------|------------------------------------------------------------------------|--------------------------------------------------------------|
| aclnn API | [test_aclnn_unique_consecutive](./examples/test_aclnn_unique_consecutive.cpp) | 通过[aclnnUniqueConsecutive](./docs/aclnnUniqueConsecutive.md)接口方式调用UniqueConsecutive算子。 |
| GE图模式 | - | 通过[算子IR](./op_graph/unique_consecutive_proto.h)构图方式调用UniqueConsecutive算子。 |

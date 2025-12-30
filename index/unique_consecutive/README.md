# UniqueConsecutive

##  产品支持情况

| 产品 | 是否支持 |
| ---- | :----:|
| <term>Ascend 950PR/Ascend 950DT</term> |√|

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
      <td>return_idx</td>
      <td>输入属性</td>
      <td>是否返回self中各元素在输出y中的位置。</td>
      <td>BOOL</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>return_counts</td>
      <td>输入属性</td>
      <td>是否返回输出y中各元素重复次数。</td>
      <td>BOOL</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>axis</td>
      <td>输入属性</td>
      <td>去重维度。</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>out_idx</td>
      <td>输入属性</td>
      <td>指定输出idx和count的类型，默认INT64。</td>
      <td>INT32</td>
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
  </tbody></table>

## 约束说明

无

## 调用说明

| 调用方式 | 调用样例                                                                   | 说明                                                           |
|--------------|------------------------------------------------------------------------|--------------------------------------------------------------|
| aclnn调用 | [test_aclnn_unique_consecutive](./examples/test_aclnn_unique_consecutive.cpp) | 通过[aclnnUniqueConsecutive](./docs/aclnnUniqueConsecutive.md)接口方式调用UniqueConsecutive算子。 |

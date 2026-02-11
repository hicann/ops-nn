# index_put_with_sort_v2

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term>                       |    √     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    ×     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> |    ×     |
| <term>Atlas 200I/500 A2 推理产品</term>                      |    ×     |
| <term>Atlas 推理系列产品</term>                              |    ×     |
| <term>Atlas 训练系列产品</term>                              |    ×      |

## 功能说明

- 算子功能：执行scatter操作，根据pos_idx的值循环对应的values，累加/替换到linear_index指向的self的位置。

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
      <td>self</td>
      <td>输入/输出</td>
      <td>累加/替换操作的目的张量</td>
      <td>INT64、INT32、FLOAT、FLOAT16、BFLOAT16、DT_BF16、INT8、UINT8、BOOL</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>linear_index</td>
      <td>输入</td>
      <td>索引Tensor，指向self相应位置</td>
      <td>INT64、INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>pos_idx</td>
      <td>输入</td>
      <td>索引Tensor，指向values相应位置</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>values</td>
      <td>输入</td>
      <td>累加/替换操作的源张量</td>
      <td>INT64、INT32、FLOAT、FLOAT16、BFLOAT16、DT_BF16、INT8、UINT8、BOOL</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## 约束说明

无

## 调用说明

| 调用方式 | 样例代码                                                                   | 说明                                                           |
|--------------|------------------------------------------------------------------------|--------------------------------------------------------------|
| aclnn调用 | [IndexPutWithSortV2](../index_put_v2/op_api/aclnn_index_put_impl.cpp) | 通过[DeterministicProcess](../index_put_v2/docs/aclnnIndexPutImpl.md)接口方式调用index_put_with_sort_v2算子。 |

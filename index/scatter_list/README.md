# ScatterList简介

##  产品支持情况

| 产品 | 是否支持 |
| ---- | :----:|
|Atlas A3 训练系列产品/Atlas A3 推理系列产品|√|
|Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件|√|

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
      <td>表示待被更新的张量，Device侧的aclTensor。</td>
      <td>DT_INT8、DT_INT16、DT_INT32、DT_UINT8、DT_UINT16、DT_UINT32、DT_FLOAT16、DT_BF16、DT_FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>indice</td>
      <td>输入</td>
      <td>表示待更新的索引张量，Device侧的aclTensor。</td>
      <td>INT32、INT64。</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>updates</td>
      <td>输入</td>
      <td>表示需要更新到var上的张量，Device侧的aclTensor。</td>
      <td>DT_INT8、DT_INT16、DT_INT32、DT_UINT8、DT_UINT16、DT_UINT32、DT_FLOAT16、DT_BF16、DT_FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>mask</td>
      <td>输入</td>
      <td>表示需要更新数据的掩码，Device侧的aclTensor。</td>
      <td>DT_UINT8</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>var</td>
      <td>输出</td>
      <td>表示更新后的张量，Device侧的aclTensor。</td>
      <td>FLOAT32、FLOAT16、DOUBLE、BFLOAT16、INT8、INT16、INT32、INT64、UINT8、BOOL</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>reduce</td>
      <td>属性</td>
      <td>HOST侧的字符串，选择应用的reduction操作</td>
      <td>string</td>
      <td>-</td>
    </tr>
    <tr>
      <td>axis</td>
      <td>属性</td>
      <td>用来scatter的维度。</td>
      <td>int64_t</td>
      <td>-</td>
    </tr>
  </tbody></table>


## 约束说明

无

## 调用说明

| 调用方式 | 调用样例                                                                   | 说明                                                           |
|--------------|------------------------------------------------------------------------|--------------------------------------------------------------|
| 图模式调用 | [test_geir_scatter_list](./examples/test_geir_scatter_list.cpp)   | 通过[算子IR](./op_graph/scatter_list_proto.h)构图方式调用ScatterList算子。 |

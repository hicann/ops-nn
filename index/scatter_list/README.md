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
      <td>self</td>
      <td>输入</td>
      <td>公式中的`self`，Device侧的aclTensor。</td>
      <td>UINT8、INT8、INT16、INT32、INT64、BOOL、FLOAT16、FLOAT32、DOUBLE、COMPLEX64、COMPLEX128、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>index</td>
      <td>输入</td>
      <td>公式中的`index`，Device侧的aclTensor。</td>
      <td>INT32、INT64。</td>
      <td>ND</td>
    </tr>
      <td>src</td>
      <td>输入</td>
      <td>公式中的`src`，Device侧的aclTensor。</td>
      <td>UINT8、INT8、INT16、INT32、INT64、BOOL、FLOAT16、FLOAT32、DOUBLE、COMPLEX64、COMPLEX128、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>dim</td>
      <td>输入</td>
      <td>Host侧的整型，选择应用的reduction操作。</td>
      <td>int64_t</td>
      <td>-</td>
    <tr>
      <td>out</td>
      <td>输出</td>
      <td>待进行abs计算的出参，公式中的out_i。</td>
      <td>FLOAT、FLOAT16、DOUBLE、BFLOAT16、INT8、INT16、INT32、INT64、UINT8、BOOL</td>
      <td>ND</td>
    </tr>
  </tbody></table>


## 约束说明

无

<!--## 调用说明-->

<!--| 调用方式 | 调用样例                                                                   | 说明                                                           |
|--------------|------------------------------------------------------------------------|--------------------------------------------------------------|-->
<!--| aclnn调用 | [test_aclnn_scatter_list](./examples/test_aclnn_scatter_list.cpp) | 通过[aclnnScatterList](./docs/aclnnScatterList.md)接口方式调用ScatterList算子。 |-->
<!--| 图模式调用 | [test_geir_scatter_list](./examples/test_geir_scatter_list.cpp)   | 通过[算子IR](./op_graph/scatter_list_proto.h)构图方式调用ScatterList算子。 |-->

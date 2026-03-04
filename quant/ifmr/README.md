# IFMR

##  产品支持情况

| 产品 | 是否支持 |
| ---- | :----:|
|Ascend 950PR/Ascend 950DT|√|
|Atlas A3 训练系列产品/Atlas A3 推理系列产品|×|
|Atlas A2 训练系列产品/Atlas A2 推理系列产品|×|
|Atlas 200I/500 A2 推理产品|×|
|Atlas 推理系列产品|×|
|Atlas 训练系列产品|×|

## 功能说明

输入特征图重建的量化方法。

## 参数说明

<table style="undefined;table-layout: fixed; width: 980px"><colgroup>
  <col style="width: 150px">
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
      <td>data</td>
      <td>输入</td>
      <td>特征图的张量。</td>
      <td>FLOAT、FLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>data_min</td>
      <td>输入</td>
      <td>特征图最小值的张量。</td>
      <td>FLOAT、FLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>data_max</td>
      <td>输入</td>
      <td>特征图最大值的张量。</td>
      <td>FLOAT、FLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>cumsum</td>
      <td>输入</td>
      <td>数据的累积求和的张量。</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>min_percentile</td>
      <td>属性</td>
      <td>最小初始化百分位数。</td>
      <td>FLOAT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>max_percentile</td>
      <td>属性</td>
      <td>最大初始化百分位数。</td>
      <td>FLOAT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>search_range</td>
      <td>属性</td>
      <td>搜索范围。</td>
      <td>LISTFLOAT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>search_step</td>
      <td>属性</td>
      <td>搜索步长。</td>
      <td>FLOAT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>with_offset</td>
      <td>属性</td>
      <td>是否使用偏移量。</td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
    <tr>
      <td>quant_bits</td>
      <td>属性</td>
      <td>量化位数，可选属性，默认值为8。</td>
      <td>INT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>scale</td>
      <td>输出</td>
      <td>最优尺度。</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>offset</td>
      <td>输出</td>
      <td>最优偏移量。</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## 约束说明

无

## 调用说明

| 调用方式   | 样例代码           | 说明                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| 图模式 | [test_geir_ifmr](./examples/test_geir_ifmr.cpp)  | 通过[算子IR](./op_graph/ifmr_proto.h)构图方式调用IFMR算子。         |
# DynamicQuantUpdateScatter

##  产品支持情况

| 产品 | 是否支持 |
| ---- | :----:|
|Atlas A3 训练系列产品/Atlas A3 推理系列产品|×|
|Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件|√|
|Kirin X90 处理器系列产品|√|
|Kirin 9030 处理器系列产品|√|

## 功能说明

- 算子功能：融合DynamicQuant+scatter+scatter为DynamicQuantUpdateScatter算子提升性能。

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
      <td>输入/输出</td>
      <td>待更新的tensor。</td>
      <td>INT8</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>var_scale</td>
      <td>输入/输出</td>
      <td>量化的scale因子，待更新的tensor。</td>
      <td>FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>indices</td>
      <td>输入</td>
      <td>表示更新位置。</td>
      <td>INT32、INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>updates</td>
      <td>输入</td>
      <td>表示更新数据</td>
      <td>BFLOAT16、FLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>smooth_scales</td>
      <td>输入</td>
      <td>代表DynamicQuant的smoothScales。</td>
      <td>BFLOAT16、FLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>axis</td>
      <td>属性</td>
      <td>scatter轴。只支持-2。</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>reduce</td>
      <td>属性</td>
      <td>shape与var_scale一致。</td>
      <td>与var_scale一致。</td>
      <td>-</td>
    </tr>
  </tbody></table>

## 约束说明

1. indices的维数只能是1维或者2维，如果是2维，其第2维的大小必须是2。
2. updates的维数与var、var_scale的维数一样，其第1维的大小等于indices的第1维的大小，且var不大于的第1维的大小，其axis轴的大小不大于var的axis轴的大小。
3. var和var_scale维度一致。
4. smooth_scales 为1维且大小和var[-1]一致。
5. reduce当前只支持‘update’，即更新操作。
6. 尾轴需要32B对齐。
7. indices映射的scatter数据段不能重合，若重合则因为多核并发原因将导致多次执行结果不一样。

## 调用说明

| 调用方式 | 调用样例                                                                   | 说明                                                           |
|--------------|------------------------------------------------------------------------|--------------------------------------------------------------|
| 图模式调用 | [test_geir_dynamic_quant_update_scatter](./examples/test_geir_dynamic_quant_update_scatter.cpp)   | 通过[算子IR](./op_graph/dynamic_quant_update_scatter_proto.h)构图方式调用DynamicQuantUpdateScatter算子。 |

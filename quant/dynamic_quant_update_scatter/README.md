# DynamicQuantUpdateScatter

## 产品支持情况

| 产品 | 是否支持 |
| ---- | :----:|
|Ascend 950PR/Ascend 950DT|√|
|Atlas A3 训练系列产品/Atlas A3 推理系列产品|×|
|Atlas A2 训练系列产品/Atlas A2 推理系列产品|√|
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
      <td>scatter轴。支持负数取值（按var维数归一化），归一化后必须是内层轴（不能取第0维或最后一维）。</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>reduce</td>
      <td>属性</td>
      <td>更新模式。支持update；none和空字符串作为兼容取值，语义与update一致。</td>
      <td>STRING</td>
      <td>-</td>
    </tr>
  </tbody></table>

- Kirin X90/Kirin 9030 处理器系列产品:不支持BFLOAT16。

## 约束说明

1. indices的维数只能是1维或者2维，如果是2维，其第2维的大小必须是2。
2. updates的维数与var的维数一致；其第1维的大小等于indices的第1维的大小，且不大于var的第1维的大小；其axis轴的大小不大于var的axis轴的大小；除第1维和axis轴外，其余各维的大小必须与var完全一致。
3. var_scale与var的维数一致，除最后一维外各维大小必须与var相同，且最后一维的大小必须为1（即每个量化行对应一个scale，var_scale的元素数等于var的元素数除以var最后一维的大小）。
4. smooth_scales为1维且大小和var[-1]一致，其数据类型必须与updates的数据类型一致。
5. reduce支持‘update’；为兼容历史调用，‘none’和空字符串同样执行更新操作。
6. 尾轴需要32B对齐：var与updates在axis轴之后各维的乘积（合并尾轴）按INT8元素个数计必须是32的倍数，且两者相等。
7. indices映射的scatter数据段不能重合，若重合则因为多核并发原因将导致多次执行结果不一样。
8. axis支持负数取值，按“axis + var维数”归一化；归一化后必须是内层轴（不能取第0维或最后一维），因此var的维数必须大于等于3。
9. 各输入shape的每一维大小必须为正数（不支持空tensor）。

## 调用说明

| 调用方式 | 调用样例                                                                   | 说明                                                           |
|--------------|------------------------------------------------------------------------|--------------------------------------------------------------|
| 图模式调用 | [test_geir_dynamic_quant_update_scatter](./examples/test_geir_dynamic_quant_update_scatter.cpp)   | 通过[算子IR](./op_graph/dynamic_quant_update_scatter_proto.h)构图方式调用DynamicQuantUpdateScatter算子。 |

# DynamicQuantUpdateScatterV2

##  产品支持情况

| 产品 | 是否支持 |
| ---- | :----:|
|Atlas A3 训练系列产品/Atlas A3 推理系列产品|×|
|Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件|√|
 

## 功能说明

- 算子功能：将DynamicQuantV2和ScatterUpdate单算子自动融合为DynamicQuantUpdateScatterV2融合算子，以实现INT4类型的非对称量化。
- 过程描述：原始数据x首先经过DynamicQuantV2算子处理，将其转化为INT4类型，并输出对应的缩放因子scale和偏移量offset。随后，三个不同的ScatterUpdate以插入索引indices、三个初始数据（var，var_scale和var_offset）和DynamicQuantV2算子的三个输出（即量化后的数据，scale和offset）作为输入按指定位置执行写入更新操作，最终得到三个原地更新输出var，var_scale和var_offset，分别对应量化更新后的数据及其对应的量化参数。

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
      <td>x</td>
      <td>输入</td>
      <td>量化输入，对应过程描述中的"x"</td>
      <td>FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>indices</td>
      <td>输入</td>
      <td>量化数据更新索引，对应过程描述中的"indices"</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>var</td>
      <td>输入/输出</td>
      <td>需要更新的量化结果，对应过程描述中的"var"</td>
      <td>INT4</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>var_scale</td>
      <td>输入/输出</td>
      <td>需要更新的量化scale因子，对应过程描述中的"var_scale"</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>var_offset</td>
      <td>输入/输出</td>
      <td>需要更新的量化offset因子，对应过程描述中的"var_offset"</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## 约束说明

- 量化方式支持非对称量化，量化数据类型支持INT4。
- 量化不支持smooth_scale输入。
- INT4量化情况下，输入x的尾轴要能被2整除。
- DynamicQuantV2的output0为INT4类型，output1为FLOAT类型，output2为FLOAT类型。
- DynamicQuantV2的输入dtype必须为FLOAT16或者BFLOAT16。input1如果存在，且input2如果不存在，input1的shape必须是1维，且等于input0的最后一维；若input2存在，input1是两维，第一维大小是专家数，不超过1024，第二维大小等于input0的最后一维。

## 调用说明

| 调用方式 | 调用样例                                                                   | 说明                                                           |
|--------------|------------------------------------------------------------------------|--------------------------------------------------------------|
| 图模式调用 | [test_geir_dynamic_quant_update_scatter_v2](./examples/test_geir_dynamic_quant_update_scatter_v2.cpp)   | 通过[算子IR](./op_graph/dynamic_quant_update_scatter_v2_proto.h)构图方式调用DynamicQuantUpdateScatterV2算子。 |

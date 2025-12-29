# DynamicMxQuant

##  产品支持情况

| 产品 | 是否支持 |
| ---- | :----:|
|昇腾910_95 AI处理器|√|
|Atlas A3 训练系列产品/Atlas A3 推理系列产品|×|
|Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件|×|
|Atlas 200I/500 A2推理产品|×|
|Atlas 推理系列产品|×|
|Atlas 训练系列产品|×|
|Atlas 200/300/500 推理产品|×|

## 功能说明

- 算子功能：在给定的轴axis上，根据每blocksize个数，计算出这组数对应的量化尺度mxscale作为输出mxscaleOut的对应部分，然后对这组数每一个除以mxscale，根据round_mode转换到对应的dstType，得到量化结果y作为输出yOut的对应部分。在dstType为FLOAT8_E4M3FN、FLOAT8_E5M2时，根据scaleAlg的取值来指定计算mxscale的不同算法。

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
      <td>待量化数据。</td>
      <td>FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>axis</td>
      <td>输入属性</td>
      <td>量化发生的轴</td>
      <td>INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>round_mode</td>
      <td>输入属性</td>
      <td>数据转换的模式</td>
      <td>INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>dst_type</td>
      <td>输入属性</td>
      <td>指定数据转换后yOut的类型</td>
      <td>INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>blocksize</td>
      <td>输入属性</td>
      <td>每次量化的元素个数</td>
      <td>INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>scale_alg</td>
      <td>输入属性</td>
      <td>mxscale的计算方法</td>
      <td>INT64</td>
      <td>ND</td>
    </tr>
    </tr>
      <td>y</td>
      <td>输出</td>
      <td>输入x量化后的对应结果</td>
      <td>FLOAT4_E2M1、FLOAT4_E1M2、FLOAT8_E4M3FN、FLOAT8_E5M2</td>
      <td>ND</td>
    </tr>
    </tr>
      <td>mxscale</td>
      <td>输出</td>
      <td>每个分组对应的量化尺度</td>
      <td>FLOAT8_E8M0</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## 约束说明

无

## 调用说明

| 调用方式 | 调用样例                                                                   | 说明                                                           |
|--------------|------------------------------------------------------------------------|--------------------------------------------------------------|
| aclnn调用 | [test_aclnn_dynamic_mx_quant](./examples/test_aclnn_dynamic_mx_quant.cpp) | 通过[aclnnDynamicMxQuant](./docs/aclnnDynamicMxQuant.md)接口方式调用DynamicMxQuant算子。 |

# DynamicMxQuant

##  产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term>                             |    √     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    ×     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> |    ×     |
| <term>Atlas 200I/500 A2 推理产品</term>                      |    ×     |
| <term>Atlas 推理系列产品</term>                             |    ×     |
| <term>Atlas 训练系列产品</term>                              |    ×     |

## 功能说明

- 算子功能：在给定的轴axis上，根据每blocksize个数，计算出这组数对应的量化尺度mxscale，然后对这组数每一个除以mxscale，根据round_mode转换到对应的dst_type，得到量化结果y。在dst_type为FLOAT8_E4M3FN、FLOAT8_E5M2时，根据scale_alg的取值来指定计算mxscale的不同算法。

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
      <td>输入</td>
      <td>量化发生的轴</td>
      <td>INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>round_mode</td>
      <td>可选属性</td>
      <td>数据转换的模式</td>
      <td>STRING</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>dst_type</td>
      <td>输入</td>
      <td>指定数据转换后y的类型</td>
      <td>INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>blocksize</td>
      <td>输入</td>
      <td>每次量化的元素个数</td>
      <td>INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>scale_alg</td>
      <td>输入</td>
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

- 关于x、mxscale的shape约束说明如下：
  - rank(mxscale) = rank(x) + 1。
  - axis_change = axis if axis >= 0 else axis + rank(x)。
  - mxscale.shape[axis_change] = (ceil(x.shape[axis] / blocksize) + 2 - 1) / 2。
  - mxscale.shape[-1] = 2。
  - 其他维度与输入x一致。

## 调用说明

| 调用方式 | 调用样例                                                                   | 说明                                                           |
|--------------|------------------------------------------------------------------------|--------------------------------------------------------------|
| aclnn调用 | [test_aclnn_dynamic_mx_quant](./examples/test_aclnn_dynamic_mx_quant.cpp) | 通过[aclnnDynamicMxQuant](./docs/aclnnDynamicMxQuant.md)接口方式调用DynamicMxQuant算子。 |
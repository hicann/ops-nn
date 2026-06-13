# AntiMxQuant

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term>                             |    √     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    ×     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> |    ×     |
| <term>Atlas 200I/500 A2 推理产品</term>                      |    ×     |
| <term>Atlas 推理系列产品</term>                             |    ×     |
| <term>Atlas 训练系列产品</term>                              |    ×     |

## 功能说明

- 算子功能：将调用`DynamicMxQuant`量化得到的FLOAT4/FLOAT8的Tensor反量化为FLOAT16/BFLOAT16/FLOAT32格式。

- 反量化公式：

  $$
  X_{dq} = X_q \times 2^{sf - bias}
  $$

  - 其中$sf$是缩放因子，由输入mxscale提供；$bias$是指数位的偏移，对于FLOAT8_E8M0格式，$bias=127$；$X_q$是量化得到的FLOAT4/FLOAT8张量；$X_{dq}$是反量化得到的FLOAT16/BFLOAT16/FLOAT32张量。

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
      <td>待反量化的数据。由DynamicMxQuant量化得到的FLOAT4/FLOAT8张量。</td>
      <td>FLOAT4_E2M1、FLOAT4_E1M2、FLOAT8_E4M3FN、FLOAT8_E5M2</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>mxscale</td>
      <td>输入</td>
      <td>调用DynamicMxQuant计算得到的量化尺度。</td>
      <td>FLOAT8_E8M0</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>axis</td>
      <td>属性</td>
      <td>指定反量化轴。</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>dst_type</td>
      <td>属性</td>
      <td>指定反量化后输出y的类型。</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>y</td>
      <td>输出</td>
      <td>输入x反量化后的对应结果。shape与x一致。</td>
      <td>FLOAT16、BFLOAT16、FLOAT32</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## 约束说明

- 关于x、mxscale的shape约束说明如下：
  - 如果输入x的数据类型是float4_e2m1或float4_e1m2，x.shape[-1]必须是偶数。
  - axis_change = axis if axis >= 0 else axis + rank(x)。
  - mxscale.shape[axis_change] = (ceil(x.shape[axis] / 32) + 2 - 1) / 2。
  - mxscale.shape[-1] = 2。
  - rank(mxscale) = rank(x) + 1。
  - 其它维度与输入x一致。

## 调用说明

| 调用方式 | 调用样例                                                                   | 说明                                                           |
|--------------|------------------------------------------------------------------------|--------------------------------------------------------------|
| aclnn调用 | [test_aclnn_anti_mx_quant](./examples/arch35/test_aclnn_anti_mx_quant.cpp) | 通过[aclnnAntiMxQuant](./docs/aclnnAntiMxQuant.md)接口方式调用AntiMxQuant算子。 |

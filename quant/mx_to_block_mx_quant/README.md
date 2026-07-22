# MxToBlockMxQuant

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

- 算子功能：将调用`DynamicMxQuant`量化得到的FLOAT4的Tensor结合FLOAT8_E8M0缩放系数，转换为FLOAT8分块量化格式，同时输出-1轴和-2轴方向的量化尺度。

- 计算公式：

$$
\begin{aligned}
scale_{fp8} &= max(mxscale_{fp4\_block}) / MAX\_OFFSET \\
offset &= mxscale_{fp4\_block} / scale_{fp8} \\
x_{fp8} &= x_{fp4} \times offset
\end{aligned}
$$

- 其中$mxscale_{fp4\_block}$是输入mxscale提供的FP8_E8M0缩放系数；$MAX\_OFFSET$是输入和输出数据类型之间的最大偏移量；$x_{fp4}$是量化得到的FLOAT4张量；$x_{fp8}$是转换得到的FLOAT8张量。

- MAX_OFFSET对照表：

| 输入     | 输出       | MAX_OFFSET |
| -------- | ---------- | ---------- |
| FP4_E2M1 | FP8_E5M2   | 13         |
| FP4_E1M2 | FP8_E5M2   | 15         |
| FP4_E2M1 | FP8_E4M3FN | 6          |
| FP4_E1M2 | FP8_E4M3FN | 8          |

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
      <td>待量化的数据。由DynamicMxQuant量化得到的FLOAT4张量。</td>
      <td>FLOAT4_E2M1、FLOAT4_E1M2</td>
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
      <td>dst_type</td>
      <td>属性</td>
      <td>指定量化后输出y的类型。</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>y</td>
      <td>输出</td>
      <td>输入x量化后的对应结果。shape与x一致。</td>
      <td>FLOAT8_E5M2、FLOAT8_E4M3FN</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>scale1</td>
      <td>输出</td>
      <td>表示-1轴每个分组对应的量化尺度。</td>
      <td>FLOAT8_E8M0</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>scale2</td>
      <td>输出</td>
      <td>表示-2轴每个分组对应的量化尺度。输出需要对每两行数据进行交织处理。</td>
      <td>FLOAT8_E8M0</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## 约束说明

- x只支持2维或3维输入，且-2轴是64的倍数, -1轴是2的倍数,不支持非连续Tensor,不支持空Tensor。
- 关于x、mxscale、scale1、scale2的shape约束说明如下：
  - rank(mxscale) = rank(x) + 1。
  - mxscale.shape[-2] = (Ceil(x.shape[-1], 32) + 2 - 1) / 2。
  - mxscale.shape[-1] = 2。
  - 其它维度与输入x一致。
- 关于输出scale1的shape约束说明如下：
  - rank(scale1) = rank(x) + 1。
  - scale1.shape[-2] = (Ceil(x.shape[-1], 32) + 2 - 1) / 2。
  - scale1.shape[-1] = 2。
  - 其它维度和输入x保持一致。
- 关于输出scale2的shape约束说明如下：
  - rank(scale2) = rank(x) + 1。
  - scale2.shape[-3] = ((Ceil(x.shape[-2], 32) + 2 - 1) / 2) * 2 / 2。
  - scale2.shape[-2] = x.shape[-1]。
  - scale2.shape[-1] = 2。
  - 其它维度和输入x保持一致。
- dstType仅支持{35, 36}，对应{FLOAT8_E5M2, FLOAT8_E4M3FN}。

## 调用说明

| 调用方式 | 调用样例                                                                   | 说明                                                           |
|--------------|------------------------------------------------------------------------|--------------------------------------------------------------|
| aclnn调用 | [test_aclnn_mx_to_block_mx_quant](./examples/arch35/test_aclnn_mx_to_block_mx_quant.cpp) | 通过[aclnnMxToBlockMxQuant](./docs/aclnnMxToBlockMxQuant.md)接口方式调用MxToBlockMxQuant算子。 |

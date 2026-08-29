# BNInfer

## 产品支持情况

| 产品                                                     | 是否支持 |
| :------------------------------------------------------- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term>                   |    √     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term> |    √     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> |    √     |
| <term>Atlas 200I/500 A2 推理产品</term>                  |    √     |
| <term>Atlas 推理系列产品</term>                          |    √     |
| <term>Atlas 训练系列产品</term>                          |    √     |

## 功能说明

- 算子功能：在推理场景下，使用给定的均值`mean`和方差`variance`对输入`x`进行批归一化，得到输出`y`。

- 计算公式：

  $$
  y = scale \times \frac{x - mean}{\sqrt{variance + epsilon}} + offset
  $$

  `scale`、`offset`、`mean`和`variance`均为一维张量，长度等于输入`x`的通道数。

## 参数说明

<table style="undefined;table-layout: fixed; width: 1005px"><colgroup>
  <col style="width: 170px">
  <col style="width: 170px">
  <col style="width: 352px">
  <col style="width: 213px">
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
  <td>待归一化的输入张量。不同数据格式对应的通道维见约束说明。</td>
      <td>FLOAT、FLOAT16、BFLOAT16</td>
      <td>NC1HWC0、ND、NCHW、NCDHW、NHWC、NDHWC</td>
    </tr>
    <tr>
      <td>scale</td>
      <td>输入</td>
      <td>缩放参数，一维张量，shape为[C]。</td>
      <td>FLOAT</td>
      <td>ND、NC1HWC0、NCHW、NCDHW、NHWC</td>
    </tr>
    <tr>
      <td>offset</td>
      <td>输入</td>
      <td>偏置参数，一维张量，shape为[C]。</td>
      <td>FLOAT</td>
      <td>ND、NC1HWC0、NCHW、NCDHW、NHWC</td>
    </tr>
    <tr>
      <td>mean</td>
      <td>输入</td>
      <td>推理时使用的均值，一维张量，shape为[C]。</td>
      <td>FLOAT</td>
      <td>ND、NC1HWC0、NCHW、NCDHW、NHWC</td>
    </tr>
    <tr>
      <td>variance</td>
      <td>输入</td>
      <td>推理时使用的方差，一维张量，shape为[C]。</td>
      <td>FLOAT</td>
      <td>ND、NC1HWC0、NCHW、NCDHW、NHWC</td>
    </tr>
    <tr>
      <td>epsilon</td>
      <td>必选属性</td>
      <td>添加到方差中的数值稳定性常数，用于避免除零。</td>
      <td>FLOAT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>y</td>
      <td>输出</td>
      <td>归一化后的输出张量，数据类型、数据格式和shape与x一致。</td>
      <td>FLOAT、FLOAT16、BFLOAT16</td>
      <td>NC1HWC0、ND、NCHW、NCDHW、NHWC、NDHWC</td>
    </tr>
  </tbody></table>

## 约束说明

- 参数表中的数据格式为各产品及shape模式所支持格式的并集，具体支持范围以本节的产品说明为准。
- scale、offset、mean和variance的数据类型必须为FLOAT，且均为shape为[C]的一维张量，其中C为x的通道数。
- <term>Ascend 950PR/Ascend 950DT</term>：
  - x和y支持ND、NCHW、NCDHW、NHWC和NDHWC格式，不支持NC1HWC0格式。
  - scale、offset、mean和variance仅支持ND格式。
  - ND格式下，x的rank不小于2，通道维为第1维；NCHW和NCDHW格式下，通道维为C维；NHWC和NDHWC格式下，通道维为最后一维。
  - NCHW和NHWC格式下，x必须为4维；NCDHW和NDHWC格式下，x必须为5维。
  - 不支持空Tensor，x的所有维度大小均必须大于0。
- <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>、<term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：
  - 静态shape场景下，x、scale、offset、mean、variance和y支持NC1HWC0和NCDHW格式。
  - 动态shape场景下，x、scale、offset、mean、variance和y支持NC1HWC0、NCHW、NHWC和NCDHW格式。
  - x和y不支持ND和NDHWC格式。
- <term>Atlas 200I/500 A2 推理产品</term>、<term>Atlas 推理系列产品</term>、<term>Atlas 训练系列产品</term>：
  - x和y不支持BFLOAT16。
- 本算子支持GE图模式和TensorFlow Parser调用，不提供公开的aclnn接口。

## 调用说明

| 调用方式 | 样例代码 | 说明 |
| ---------------- | --------------------------- | --------------------------------------------------- |
| 图模式调用 | [test_geir_bn_infer](./examples/test_geir_bn_infer.cpp) | 通过[算子IR](./op_graph/bn_infer_proto.h)构图方式调用BNInfer算子。 |

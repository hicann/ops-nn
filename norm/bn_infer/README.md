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

- 算子功能：推理场景下对输入张量x执行批量归一化，使用已给定的均值mean和方差variance计算输出y。

- 计算公式：

  $$
  y = scale \times \frac{x - mean}{\sqrt{variance + epsilon}} + offset
  $$

  scale、offset、mean、variance均为一维张量，长度等于输入x的通道维。

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
  <td>被归一化的输入张量。A2/A3兼容声明包含NC1HWC0、NULL及其未知形状映射；Ascend950物理执行路径按950实现收窄为ND、NCHW、NCDHW、NHWC和NDHWC，未提供NC1HWC0和NULL路径。ND格式下通道维为第1维，NCHW/NCDHW格式下通道维为C维，NHWC/NDHWC格式下通道维为最后一维。</td>
      <td>FLOAT、FLOAT16、BFLOAT16</td>
      <td>ND、NCHW、NCDHW、NHWC、NDHWC</td>
    </tr>
    <tr>
      <td>scale</td>
      <td>输入</td>
      <td>缩放参数，一维张量，shape为(C)。</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>offset</td>
      <td>输入</td>
      <td>偏置参数，一维张量，shape为(C)。</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>mean</td>
      <td>输入</td>
      <td>推理使用的均值，一维张量，shape为(C)。</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>variance</td>
      <td>输入</td>
      <td>推理使用的方差，一维张量，shape为(C)。</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>epsilon</td>
      <td>必选属性</td>
      <td>添加到方差中的小值，用于避免除0。</td>
      <td>FLOAT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>y</td>
      <td>输出</td>
      <td>归一化后的输出张量，声明格式、数据类型和shape与x一致；Ascend950物理执行格式与x相同。</td>
      <td>FLOAT、FLOAT16、BFLOAT16</td>
      <td>ND、NCHW、NCDHW、NHWC、NDHWC</td>
    </tr>
  </tbody></table>

## 约束说明

- scale、offset、mean、variance的数据类型必须为FLOAT。
- scale、offset、mean、variance必须为一维张量，长度必须等于x的通道维大小。
- ND格式下x维度数必须大于等于2；NCHW和NHWC格式下x必须为4维；NCDHW和NDHWC格式下x必须为5维。
- Ascend950物理执行路径不支持x的任一参与维度为0的空Tensor；host tiling返回`GRAPH_FAILED`，不会进入kernel。
- A2/A3原型契约保留NC1HWC0和NULL；Ascend950 AscendC物理执行面按950实现收窄为ND、NCHW、NCDHW、NHWC和NDHWC。由于950 Vector kernel/tiling未实现NC1HWC0存储布局，且已知形状NULL无法映射到950物理路径，host tiling在kernel前结构化拒绝这两类格式；该差异通过950独立OpDef/config和tiling隔离，不修改A2/A3实现。
- 本算子为GE图内部算子，不提供公开aclnnBNInfer接口；aclnn/torch单算子接口不涉及。

## 调用说明

| 调用方式 | 样例代码 | 说明 |
| ---------------- | --------------------------- | --------------------------------------------------- |
| 图模式调用 | [test_geir_bn_infer](./examples/test_geir_bn_infer.cpp) | 通过[算子IR](./op_graph/bn_infer_proto.h)构图方式调用BNInfer算子。 |

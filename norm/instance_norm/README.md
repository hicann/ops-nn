# InstanceNorm

## 产品支持情况

|产品             |  是否支持  |
|:-------------------------|:----------:|
|  <term>Ascend 950PR/Ascend 950DT</term>   |     √    |
|  <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>   |     ×    |
|  <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>     |     ×    |
|  <term>Atlas 200I/500 A2 推理产品</term>    |     ×    |
|  <term>Atlas 推理系列产品</term>    |     ×    |
|  <term>Atlas 训练系列产品</term>    |     ×    |


## 功能说明

- 算子功能：用于执行Instance Normalization（实例归一化）操作。与[BatchNorm](../batch_norm_v3/README.md)相比，InstanceNorm在每个样本的实例上进行归一化，而不是在整个批次上进行归一化，这使得该函数更适合处理图像等数据。
- 计算公式：

  $$
  y = {{x-E(x)}\over\sqrt {Var(x) + ε}} * γ + β
  $$

## 参数说明

<table style="undefined;table-layout: fixed; width: 1005px"><colgroup>
  <col style="width: 170px">
  <col style="width: 170px">
  <col style="width: 372px">
  <col style="width: 213px">
  <col style="width: 105px">
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
      <td><ul><li>支持空Tensor，仅支持reduce轴输入为0，不支持N轴和C轴输入为0。</li><li>表示进行InstanceNorm计算的输入数据，对应公式中的`x`。</li><li>实际数据格式由参数data_format决定。</li><li>NCHW/NHWC：shape支持4D。</li><li>NCDHW/NDHWC：shape支持5D。</li><li>ND：shape支持2~8D，其中第2维固定为C轴。</li></ul></td>
      <td>FLOAT32、FLOAT16、BFLOAT16</td>
      <td>NCHW/NHWC/NCDHW/NDHWC/ND</td>
    </tr>
    <tr>
      <td>gamma</td>
      <td>输入</td>
      <td><ul><li>不支持空Tensor。</li><li>一个1D张量，表示进行InstanceNorm计算的缩放因子（权重），对应公式中的`γ`。</li><li>数据类型默认与x一致；若不一致，则显示设为FLOAT32。</li><li>shape与输入x的C轴保持一致。</li></ul></td>
      <td>FLOAT32、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>beta</td>
      <td>输入</td>
      <td><ul><li>不支持空Tensor。</li><li>一个1D张量，表示进行InstanceNorm计算的偏置，对应公式中的`β`。</li><li>数据类型、shape与输入gamma保持一致。</li></ul></td>
      <td>FLOAT32、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>data_format</td>
      <td>可选属性</td>
      <td><ul><li>指定输入x的数据格式。</li><li>默认值为NDHWC。</li><li>参数data_format仅支持"NCHW"、"NHWC"、"NCDHW"、"NDHWC"、"ND"。</li></ul></td>
      <td>STRING</td>
      <td>-</td>
    </tr>
    <tr>
      <td>epsilon</td>
      <td>可选属性</td>
      <td><ul><li>表示添加到方差中的值，以避免出现除以零的情况。对应公式中的`ε`。</li><li>默认值为1e-6f。</li></ul></td>
      <td>FLOAT32</td>
      <td>-</td>
    </tr>
    <tr>
      <td>y</td>
      <td>输出</td>
      <td><ul><li>支持空Tensor。</li><li>表示InstanceNorm的输出结果，对应公式中的`y`。</li><li>shape、数据类型和数据格式与输入x的保持一致。</li></ul></td>
      <td>FLOAT32、FLOAT16、BFLOAT16</td>
      <td>NCHW/NHWC/NCDHW/NDHWC/ND</td>
    </tr>
    <tr>
      <td>mean</td>
      <td>输出</td>
      <td><ul><li>不支持空Tensor。</li><li>表示InstanceNorm的均值，对应公式中的`E(x)`。</li><li>数据类型与gamma保持一致。</li><li>shape与输入x满足<a href="../../docs/zh/context/broadcast关系.md">broadcast关系</a>（shape维度数与输入x一致，非规约轴shape大小与输入x保持一致，规约轴shape大小为1）。</li></ul></td>
      <td>FLOAT32、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>variance</td>
      <td>输出</td>
      <td><ul><li>不支持空Tensor。</li><li>表示InstanceNorm的方差，对应公式中的`Var(x)`。</li><li>数据类型、shape与输入mean保持一致。</li></ul></td>
      <td>FLOAT32、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
  </tbody>
  </table>

## 约束说明

无

## 调用说明

| 调用方式 | 调用样例                                                                   | 说明                                                           |
|--------------|------------------------------------------------------------------------|--------------------------------------------------------------|
| 图模式调用 | [test_geir_instance_norm](./examples/test_geir_instance_norm.cpp) | 通过[算子IR](op_graph/instance_norm_proto.h)构图方式调用InstanceNorm算子。 |
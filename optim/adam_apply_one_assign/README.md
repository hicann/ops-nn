# AdamApplyOneAssign

## 产品支持情况

| 产品                                              | 是否支持 |
|:------------------------------------------------| :------: |
| <term>Ascend 950PR/Ascend 950DT</term>          |    √     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>    |    √     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>    |    √     |
| <term>Atlas 200I/500 A2 推理产品</term>             |    √     |
| <term>Atlas 推理系列产品</term>                       |    ×     |
| <term>Atlas 训练系列产品</term>                       |    √     |

## 功能说明

- 算子功能：执行Adam优化器的一次参数更新，10输入3输出，支持broadcast。输出端口名 input1、input2、input3 与输入同名（GE inplace别名），框架将输出内存别名到输入内存，实现原地更新；输出shape分别与对应的输入相同。

- 计算公式：

$$
out\_input1 = input1 × mul2\_x + input0² × mul3\_x
$$

$$
out\_input2 = mul0\_x × input2 + mul1\_x × input0
$$

$$
out\_input3 = input3 - out\_input2 / (sqrt(out\_input1) + add2\_y) × input4
$$

- 支持broadcast：所有输入shape需满足broadcast关系，广播后的最大值shape须与输入input1、input2、input3的shape相同，输出shape与对应的输入相同。

## 参数说明

<table style="undefined;table-layout: fixed; width: 1576px"><colgroup>
<col style="width: 170px">
<col style="width: 170px">
<col style="width: 200px">
<col style="width: 200px">
<col style="width: 170px">
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
    <td>input0</td>
    <td>输入</td>
    <td>公式中输入张量input0。</td>
    <td>FLOAT、FLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>input1</td>
    <td>输入</td>
    <td>公式中输入张量input1。与图输出端口input1共享GM地址（inplace更新）。</td>
    <td>FLOAT、FLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>input2</td>
    <td>输入</td>
    <td>公式中输入张量input2。与图输出端口input2共享GM地址（inplace更新）。</td>
    <td>FLOAT、FLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>input3</td>
    <td>输入</td>
    <td>公式中输入张量input3。与图输出端口input3共享GM地址（inplace更新）。</td>
    <td>FLOAT、FLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>input4</td>
    <td>输入</td>
    <td>公式中输入张量input4。</td>
    <td>FLOAT、FLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>mul0_x</td>
    <td>输入</td>
    <td>公式中输入张量mul0_x。</td>
    <td>FLOAT、FLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>mul1_x</td>
    <td>输入</td>
    <td>公式中输入张量mul1_x。</td>
    <td>FLOAT、FLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>mul2_x</td>
    <td>输入</td>
    <td>公式中输入张量mul2_x。</td>
    <td>FLOAT、FLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>mul3_x</td>
    <td>输入</td>
    <td>公式中输入张量mul3_x。</td>
    <td>FLOAT、FLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>add2_y</td>
    <td>输入</td>
    <td>公式中输入张量add2_y。</td>
    <td>FLOAT、FLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>input1</td>
    <td>输出</td>
    <td>更新后的张量，公式中的out_input1。与输入input1共享Device内存（inplace别名），shape/dtype与输入input1完全相同。</td>
    <td>FLOAT、FLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>input2</td>
    <td>输出</td>
    <td>更新后的张量，公式中的out_input2。与输入input2共享Device内存（inplace别名），shape/dtype与输入input2完全相同。</td>
    <td>FLOAT、FLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>input3</td>
    <td>输出</td>
    <td>更新后的张量，公式中的out_input3。与输入input3共享Device内存（inplace别名），shape/dtype与输入input3完全相同。</td>
    <td>FLOAT、FLOAT16</td>
    <td>ND</td>
  </tr>
</tbody></table>

## 约束说明

- 输入shape维度最大为8。
- 所有输入和输出的数据类型必须一致。
- 输出input1、input2、input3与输入input1、input2、input3同名（GE同名端口=inplace别名），shape分别与对应的输入相同，调用方必须把这三个输入视为可写。

## 调用说明

| 调用方式 | 调用样例                                                                   | 说明                                                           |
|--------------|------------------------------------------------------------------------|--------------------------------------------------------------|
| 图模式调用 | [test_geir_adam_apply_one_assign](./examples/test_geir_adam_apply_one_assign.cpp)   | 通过算子IR构图方式调用AdamApplyOneAssign算子。 |

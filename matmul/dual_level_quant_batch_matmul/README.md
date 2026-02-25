# DualLevelQuantBatchMatmul

## 产品支持情况

| 产品                                                         |  是否支持   |
| :----------------------------------------------------------- |:-------:|
| <term>Ascend 950PR/Ascend 950DT</term>                             |    √    |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    ×    |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> |    ×    |
| <term>Atlas 200I/500 A2 推理产品</term>                      |    ×    |
| <term>Atlas 推理系列产品 </term>                             |    ×    |
| <term>Atlas 训练系列产品</term>                              |    ×    |

## 功能说明

- 算子功能：完成二级量化mxfp4的矩阵乘计算
- 计算公式
    - x1和x2为FLOAT4_E2M1，x1Levl0Scale和x2Levl0Scale为FLOAT32，x1Levl1Scale和x2Levl1Scale为FLOAT8_E8M0，out为FLOAT16/BFLOAT16, bias为可选参数类型为FLOAT32：

      $$
      out =\sum_{i}^{level0GroupSize} x1Levl0Scale @ x2Levl0Scale \sum_{ij}^{level1GroupSize} ((x1Levl1Scale @ x1_{ij})@ (x2Levl1Scale @ x2_{ij})) + bias
      $$

## 参数说明

<table style="undefined;table-layout: fixed; width: 1166px"><colgroup>
<col style="width: 150px">
<col style="width: 150px">
<col style="width: 430px">
<col style="width: 212px">
<col style="width: 144px">
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
    <td>x1</td>
    <td>输入</td>
    <td>矩阵乘运算中的左矩阵。</td>
    <td>FLOAT4_E2M1</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>x2</td>
    <td>输入</td>
    <td>矩阵乘运算中的右矩阵。</td>
    <td>FLOAT4_E2M1</td>
    <td>FRACTAL_NZ</td>
  </tr>
  <tr>
    <td>x1_level0_scale</td>
    <td>输入</td>
    <td>矩阵乘计算时，x1的一级量化参数的缩放因子，对应公式的x1Level0Scale。</td>
    <td>FLOAT32</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>x1_level1_scale</td>
    <td>输入</td>
    <td>矩阵乘计算时，x1的二级量化参数的缩放因子，对应公式的x1Level1Scale。</td>
    <td>FLOAT8_E8M0</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>x2_level0_scale</td>
    <td>输入</td>
    <td>矩阵乘计算时，x2的一级量化参数的缩放因子，对应公式的x2Level0Scale。</td>
    <td>FLOAT32</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>x2_level1_scale</td>
    <td>输入</td>
    <td>矩阵乘计算时，x2的二级量化参数的缩放因子，对应公式的x2Level1Scale。</td>
    <td>FLOAT8_E8M0</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>bias</td>
    <td>输入</td>
    <td>矩阵乘运算后累加的偏置，对应公式中的bias。</td>
    <td>FLOAT32</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>y</td>
    <td>输出</td>
    <td>矩阵乘运算的计算结果。</td>
    <td>BFLOAT16, FLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>dtype</td>
    <td>属性</td>
    <td>矩阵乘运算计算结果的输出类型。</td>
    <td>INT64</td>
    <td>-</td>
  </tr>
  <tr>
    <td>transpose_x1</td>
    <td>属性</td>
    <td>矩阵乘运算中的左矩阵x1是否转置</td>
    <td>BOOL</td>
    <td>-</td>
  </tr>
  <tr>
    <td>transpose_x2</td>
    <td>属性</td>
    <td>矩阵乘运算中的右矩阵x2是否转置。</td>
    <td>BOOL</td>
    <td>-</td>
  </tr>
  <tr>
    <td>level0_group_size</td>
    <td>属性</td>
    <td>一级量化groupsize的大小，对应公式中的level0GroupSize</td>
    <td>INT64</td>
    <td>-</td>
  </tr>
  <tr>
    <td>level1_group_size</td>
    <td>属性</td>
    <td>二级量化groupsize的大小，对应公式中的level1GroupSize</td>
    <td>INT64</td>
    <td>-</td>
  </tr>
</tbody></table>

## 约束说明

- 不支持空tensor。
- 支持连续tensor，[非连续tensor](../../docs/zh/context/非连续的Tensor.md)只支持转置场景。

## 调用说明

| 调用方式      | 调用样例                 | 说明                                                         |
|--------------|-------------------------|--------------------------------------------------------------|
| aclnn调用 | [test_aclnn_dual_level_quant_matmul_weight_nz](examples/arch35/test_aclnn_dual_level_quant_matmul_weight_nz.cpp) | 通过[aclnnDualLevelQuantMatmulWeightNz](docs/aclnnDualLevelQuantMatmulWeightNz.md)接口方式调用DualLevelQuantBatchMatmul算子。 |

# FlatQuant

## 产品支持情况

|产品             |  是否支持  |
|:-------------------------|:----------:|
|  <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>   |     √    |
|  <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>     |     √    |

## 功能说明

- 算子功能：该融合算子为输入矩阵x一次进行两次小矩阵乘法，即右乘输入矩阵kroneckerP2，左乘输入矩阵kroneckerP1，然后针对矩阵乘的结果进行per-token量化处理。

- 计算公式：
  
  1.输入x右乘kroneckerP2：
  
    $$
    x' = x @ kroneckerP2
    $$

  2.kroneckerP1左乘x'：

    $$
    x'' = kroneckerP1@x'
    $$
  
  3.沿着x''的0维计算最大绝对值并除以(7 / clipRatio)以计算需量化为INT4格式的量化因子：

    $$
    quantScale = [max(abs(x''[0,:,:])),max(abs(x''[1,:,:])),...,max(abs(x''[K,:,:]))]/(7 / clipRatio)
    $$
  
  4.计算输出的out：
  
    $$
    out = x'' / quantScale
    $$

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
      <td>输入的原始数据，对应公式中的`x`。shape为[K, M, N]，其中，K不超过262144，M和N不超过256。`out`的数据类型为INT4，N必须是偶数。</td>
      <td>FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>kronecker_p1</td>
      <td>输入</td>
      <td>输入的计算矩阵1，对应公式中的`kroneckerP1`。shape为[M, M]，M与`x`中M维一致，数据类型与入参`x`的数据类型一致。</td>
      <td>FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>kronecker_p2</td>
      <td>输入</td>
      <td>输入的计算矩阵2，对应公式中的`kroneckerP2`。shape为[N, N]，N与`x`中N维一致，数据类型与入参`x`的数据类型一致。</td>
      <td>FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>clip_ratio</td>
      <td>可选属性</td>
      <td><ul><li>用于控制量化的裁剪比例，输入数据范围为(0, 1]。</li><li>默认值为1。</li></ul></td>
      <td>FLOAT32</td>
      <td>-</td>
    </tr>
    <tr>
      <td>out</td>
      <td>输出</td>
      <td>输出张量，对应公式中的`out`。数据类型为INT4时，shape与入参`x`一致。</td>
      <td>INT4</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>quant_scale</td>
      <td>输出</td>
      <td>输出的量化因子，对应公式中的`quantScale`。shape为[K]，K与`x`中K维一致。</td>
      <td>FLOAT32</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## 约束说明

无

## 调用说明

| 调用方式   | 样例代码           | 说明                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn接口  | [test_aclnn_flat_quant](examples/test_aclnn_flat_quant.cpp) | 通过[aclnnFlatQuant](docs/aclnnFlatQuant.md)接口方式调用FlatQuant算子。 |
| 图模式 | -  | 通过[算子IR](op_graph/flat_quant_proto.h)构图方式调用FlatQuant算子。         |
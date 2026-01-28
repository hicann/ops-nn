# DynamicDualLevelMxQuant

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term>                             |    √     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    ×     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品 </term> |    ×     |
| <term>Atlas 200I/500 A2 推理产品</term>                      |    ×     |
| <term>Atlas 推理系列产品 </term>                             |    ×     |
| <term>Atlas 训练系列产品</term>                              |    ×     |

## 功能说明

- 算子功能：目的数据类型为FLOAT4类的MX量化。只对尾轴进行量化，前面所有的轴都合轴处理，通过给定的level0_block_size将输入划分成多个数据块，对每个数据块进行一级量化，输出量化尺度level0_scale；然后将一级量化的结果作为新的输入，并通过给定的level1_block_size将其划分成多个数据块，对每个数据块进行二级量化，输出量化尺度level1_scale，根据round_mode进行数据类型的转换，得到量化结果y。

- 计算公式：
  - 将输入x在尾轴上按$k_0$ = level0_block_size个数分组，一组$k_0$个数  $\{\{x_i\}_{i=1}^{k_0}\}$ 动态量化为 $\{level0_scale, \{temp_i\}_{i=1}^{k_0}\}$, $k_0$ = level0_block_size，然后将temp在尾轴上按$k_1$ = level1_block_size个数分组，一组$k_1$个数  $\{\{temp_i\}_{i=1}^{k_1}\}$ 动态量化为 $\{level1_scale, \{y_i\}_{i=1}^{k_1}\}$, $k_1$ = level1_block_size

  $$
  input\_max_i = max_i(abs(x_i))
  $$

  $$
  level0\_scale_i = input\_max_i / (FP4\_E2M1\_MAX)
  $$

  $$
  temp_i = cast\_to\_x\_type(x_i / level0\_scale_i), \space i\space from\space 1\space to\space level0\_block\_size
  $$

  $$
  shared\_exp_i = floor(log_2(max_i(|temp_i|))) - emax
  $$

  $$
  level1\_scale_i = 2^{shared\_exp_i}
  $$

  $$
  y_i = cast\_to\_FP4\_E2M1(temp_i/level1\_scale_i, round\_mode), \space i\space from\space 1\space to\space level1\_block\_size
  $$

  - ​量化后的 $y_{i}$ 按对应的 $x_{i}$ 的位置组成输出y，$level0\_scale_i$按尾轴对应的分组组成输出$level0\_scale$，$level1\_scale_i$按尾轴对应的分组组成输出$level1\_scale$。

  - max_i代表求第i个分组中的最大值

  - emax: 对应数据类型的最大正则数的指数位。

      |   DataType    | emax |
      | :-----------: | :--: |
      |  FLOAT4_E2M1  |  2   |

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
      <td><ul><li>表示算子输入的Tensor，对应公式中<em>x</em><sub>i</sub>；</li><li>shape支持1-7维，最后一维必须是偶数。</td>
      <td>FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>smooth_scales</td>
      <td>可选输入</td>
      <td>当前该功能暂不支持，只支持输入为nullptr。</td>
      <td>和输入x一致</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>round_mode</td>
      <td>可选属性</td>
      <td><ul><li>表示数据转换的模式，对应公式中的round_mode；</li><li>支持{"rint", "round", "floor"}；</li><li> 默认值为"rint"。</td>
      <td>STRING</td>
      <td>-</td>
    </tr>
    <tr>
      <td>level0_block_size</td>
      <td>可选属性</td>
      <td><ul><li>表示第一级量化的block_size，对应公式中的level0_block_size；</li><li>输入范围为{512}。</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>level1_block_size</td>
      <td>可选属性</td>
      <td><ul><li>表示第二级量化的block_size，对应公式中的level1_block_size；</li><li>输入范围为{32}。</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>y</td>
      <td>输出</td>
      <td><ul><li>表示输入x量化后的对应结果，对应公式中的<em>y</em><sub>i</sub>；</li><li>shape和输入x一致。</td>
      <td>FLOAT4_E2M1</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>level0_scale</td>
      <td>输出</td>
      <td><ul><li>表示第一级量化的scale，对应公式中的level0_scale；</li><li>shape在尾轴上的值，为x尾轴的值除以level0_block_size向上取整</td>
      <td>FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>level1_scale</td>
      <td>输出</td>
      <td><ul><li>表示第二级量化的scale，对应公式中的level1_scale；</li><li>shape的大小为x的dim + 1；</li><li> shape在最后两轴的值为((ceil(x.shape[-1] / level1_block_size) + 2 - 1) / 2, 2)，并对其进行偶数pad，pad填充值为0。</td>
      <td>FLOAT8_E8M0</td>
      <td>ND</td>
    </tr>
  </tbody>
</table>

## 约束说明

- 关于x、level0_scale、level1_scale的shape约束说明如下：
    - rank(level1_scale) = rank(x) + 1。
    - level0_scale.shape[-1] = ceil(x.shape[-1] / level0_block_size)。
    - level1_scale.shape[-2] = (ceil(x.shape[-1] / level1_block_size) + 2 - 1) / 2。
    - level1_scale.shape[-1] = 2。
    - 其他维度与输入x一致。

## 调用说明

| 调用方式   | 样例代码           | 说明                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn接口  | [test_aclnn_dynamic_dual_level_mx_quant](examples/arch35/test_aclnn_dynamic_dual_level_mx_quant.cpp) | 通过[aclnnDynamicDualLevelMxQuant](docs/aclnnDynamicDualLevelMxQuant.md)接口方式调用DynamicDualLevelMxQuant算子。 |
| 图模式 | -  | 通过[算子IR](op_graph/dynamic_dual_level_mx_quant_proto.h)构图方式调用DynamicDualLevelMxQuant算子。         |
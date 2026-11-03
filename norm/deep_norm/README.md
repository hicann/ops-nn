# DeepNorm

## 产品支持情况

| 产品                                    | 是否支持 |
| :-------------------------------------- | :------: |
| <term>Ascend 950PR&950DT系列产品</term> |    √    |
| <term>Atlas A3系列产品</term>           |    √    |
| <term>Atlas A2系列产品</term>           |    √    |
| <term>Atlas 200I/500 A2推理产品</term>  |    ×    |
| <term>Atlas推理系列产品</term>          |    √    |
| <term>Atlas训练系列产品</term>          |    ×    |
| <term>Kirin X90处理器系列产品</term>    |    √    |
| <term>Kirin 9030处理器系列产品</term>   |    √    |

## 功能说明

- 算子功能：对输入张量x的元素进行深度归一化，通过计算其均值和标准差，将每个元素标准化为具有零均值和单位方差的输出张量。具体地说，先将输入张量`x`乘以权重`alpha`后与`gx`相加，得到中间张量`x'`；再沿归一化维度计算`x'`的均值`mean`和标准差倒数`rstd`，完成归一化，并通过`gamma`缩放、`beta`偏置得到输出张量`y`。算子同时输出`mean`、`rstd`和`y`。
- 计算公式：

  $$
  \begin{aligned}
  x_i^{\prime} &= \mathrm{alpha} \cdot x_i + \mathrm{gx}_i \\
  \mathrm{mean} &= \frac{1}{n}\sum_{i=1}^{n}x_i^{\prime} \\
  \mathrm{rstd} &= \frac{1}{\sqrt{\frac{1}{n}\sum_{i=1}^{n}(x_i^{\prime} - \mathrm{mean})^2 + \mathrm{epsilon}}} \\
  y_i &= (x_i^{\prime} - \mathrm{mean}) \cdot \mathrm{rstd} \cdot \mathrm{gamma}_i + \mathrm{beta}_i
  \end{aligned}
  $$

  其中，归一化维度为`x`中与`gamma`的shape对应的尾部维度，`n`为这些维度的大小之积。上述公式针对固定前导维度索引的一组数据，`i`为该组归一化维度展平后的元素索引；`gamma`、`beta`在不同组之间共享。`x'`为计算过程中的中间张量，`mean`和`rstd`为每组数据的统计量，`y`为最终输出，`epsilon`为加到方差上的数值稳定项。

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
      <td>不支持空Tensor。输入数据，通常为神经网络的中间层输出，公式中的输入`x`。shape支持2-8维度。</td>
      <td>FLOAT32、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>gx</td>
      <td>输入</td>
      <td>不支持空Tensor。与加权后的`x`逐元素相加的输入数据，对应公式中的`gx`。数据类型与输入`x`一致，shape与输入`x`一致。</td>
      <td>FLOAT32、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>beta</td>
      <td>输入</td>
      <td>不支持空Tensor。偏置参数，用于调整归一化后的输出，公式中的输入`beta`。数据类型与输入`x`一致，维度数与输入`gamma`一致，shape与输入`x`的后几维（需要norm的维度）一致。</td>
      <td>FLOAT32、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>gamma</td>
      <td>输入</td>
      <td>不支持空Tensor。缩放参数，用于调整归一化后的输出，公式中的输入`gamma`。数据类型与输入`x`一致，维度数取值范围为[1, 7]且必须小于输入`x`的维度数，shape与输入`x`的后几维（需要norm的维度）一致。</td>
      <td>FLOAT32、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>alpha</td>
      <td>可选属性</td>
      <td>权重参数，用于调整输入x的权重，公式中的输入`alpha`。默认值为0.3f。</td>
      <td>FLOAT32</td>
      <td>-</td>
    </tr>
    <tr>
      <td>epsilon</td>
      <td>可选属性</td>
      <td>计算`rstd`时加到`x'`的方差上的数值稳定项，用于防止除0错误，对应公式中的`epsilon`。默认值为1e-06f。</td>
      <td>FLOAT32</td>
      <td>-</td>
    </tr>
    <tr>
      <td>mean</td>
      <td>输出</td>
      <td>不支持空Tensor。中间张量`x'`沿归一化维度计算得到的均值，对应公式中的输出`mean`，用于计算`rstd`和`y`。shape与输入`x`满足broadcast关系（前几维（不需要norm的维度）与输入`x`一致，其余维度为1）。</td>
      <td>FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>rstd</td>
      <td>输出</td>
      <td>不支持空Tensor。中间张量`x'`沿归一化维度计算方差、加上`epsilon`后开平方得到的倒数，对应公式中的输出`rstd`，用于计算`y`。shape与输入`x`满足broadcast关系（前几维（不需要norm的维度）与输入`x`一致，其余维度为1）。</td>
      <td>FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y</td>
      <td>输出</td>
      <td>不支持空Tensor。中间张量`x'`经归一化、`gamma`缩放和`beta`偏置后的结果，对应公式中的输出`y`。数据类型与输入`x`一致，shape与输入`x`一致。</td>
      <td>FLOAT32、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
  </tbody></table>

- <term>Atlas推理系列产品</term>：所有输入参数和输出参数y的数据类型不支持BFLOAT16。
- <term>Kirin X90处理器系列产品</term>、<term>Kirin 9030处理器系列产品</term>：所有输入参数和输出参数y的数据类型不支持BFLOAT16。

## 约束说明

- `x` 维度数支持 2-8，`gamma` 维度数支持 1-7，且必须小于 `x` 的维度数。
- `x`、`gx`、`beta`、`gamma`、`y` 的 dtype 必须一致，只支持 `FLOAT32`、`FLOAT16`、`BFLOAT16`；`mean`、`rstd` 固定为 `FLOAT32`。
- `gx`、`y` 的 shape 必须与 `x` 一致；`beta`、`gamma` 的 shape 必须与 `x` 的尾部维一致。
- `mean`、`rstd` 的 shape 必须与 `x` 的前导维一致，归一化对应的尾部维必须全部为 `1`，且 `mean` 与 `rstd` 的 shape 必须一致。
- `x` 的各维度必须大于 0，tiling 侧会对 `numRow/numColAlign` 以及 shape 乘积做范围检查，超出 `uint32` 的输入会被拒绝。
- arch35 tiling 会根据 reduce 维长度和 UB 空间在 full-load / partial-load 两条路径间切换；partial-load 场景需要额外 workspace。

## 调用说明

| 调用方式  | 样例代码                                                 | 说明                                                                |
| --------- | -------------------------------------------------------- | ------------------------------------------------------------------- |
| aclnn接口 | [test_aclnn_deep_norm](examples/test_aclnn_deep_norm.cpp) | 通过[aclnnDeepNorm](docs/aclnnDeepNorm.md)接口方式调用DeepNorm算子。 |
| 图模式    | -                                                        | 通过[算子IR](op_graph/deep_norm_proto.h)构图方式调用DeepNorm算子。   |

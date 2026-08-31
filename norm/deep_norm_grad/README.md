# DeepNormGrad

## 产品支持情况

|产品             |  是否支持  |
|:-------------------------|:----------:|
|  <term>Ascend 950PR/Ascend 950DT</term>   |     √    |
|  <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>   |     √    |
|  <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>     |     √    |
|  <term>Atlas 200I/500 A2 推理产品</term>    |     ×    |
|  <term>Atlas 推理系列产品</term>    |     √    |
|  <term>Atlas 训练系列产品</term>    |     ×    |

## 功能说明

- 算子功能：[DeepNorm](../deep_norm/README.md)的反向传播，完成张量x、张量gx、张量gamma的梯度计算，以及张量dy的求和计算。

- 计算公式：

  $$
  dgx_i = tmpone_i * rstd + dvar * tmptwo_i + dmean
  $$

  $$
  dx_i = alpha * {dgx}_i
  $$

  $$
  dbeta = \sum_{i=1}^{N} dy_i
  $$

  $$
  dgamma =  \sum_{i=1}^{N} dy_i * rstd * {tmptwo}_i
  $$

  其中：

  $$
  oneDiv=-1/SizeOf(gamma)
  $$

  $$
  tmpone_i = dy_i * gamma
  $$

  $$
  tmptwo_i = alpha * x_i + {gx}_i - mean
  $$

  $$
  dvar = (oneDiv) * \sum_{i=1}^{N} {tmpone}_i * {tmptwo}_i * {rstd}^3
  $$

  $$
  dmean = (oneDiv) * \sum_{i=1}^{N} {tmpone}_i * rstd
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
      <td>dy</td>
      <td>输入</td>
      <td>主要的grad输入，对应公式中的`dy`。</td>
      <td>FLOAT32、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>x</td>
      <td>输入</td>
      <td>为正向融合算子的输入x，对应公式中的`x`。</td>
      <td>FLOAT32、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>gx</td>
      <td>输入</td>
      <td>为正向融合算子的输入gx，对应公式中的`gx`。</td>
      <td>FLOAT32、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>gamma</td>
      <td>输入</td>
      <td>前向传播的缩放参数，对应公式中的`gamma`。</td>
      <td>FLOAT32、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>mean</td>
      <td>输入</td>
      <td>表示正向输入x、gx之和的均值，对应公式中的`mean`。</td>
      <td>FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>rstd</td>
      <td>输入</td>
      <td>表示正向输入x、gx之和的rstd，对应公式中的`rstd`。</td>
      <td>FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>alpha</td>
      <td>可选属性</td>
      <td><ul><li>含义与deepnorm正向输入alpha相同，deepnorm输入x维度的乘数权重参数，公式中的输入`alpha`。</li><li>默认值为0.3f。</li></ul></td>
      <td>FLOAT32</td>
      <td>-</td>
    </tr>
    <tr>
      <td>dx</td>
      <td>输出</td>
      <td>计算输出的梯度，用于更新输入数据x的梯度，对应公式中的`dx`。</td>
      <td>FLOAT32、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>dgx</td>
      <td>输出</td>
      <td>计算输出的梯度，用于更新输入数据gx的梯度，对应公式中的`dgx`。</td>
      <td>FLOAT32、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>dbeta</td>
      <td>输出</td>
      <td>计算输出的梯度，用于更新偏置参数的梯度，对应公式中的`dbeta`。</td>
      <td>FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>dgamma</td>
      <td>输出</td>
      <td>计算输出的梯度，用于更新缩放参数的梯度。对应公式中的`dgamma`。</td>
      <td>FLOAT32</td>
      <td>ND</td>
    </tr>
  </tbody></table>

  - <term>Atlas 推理系列产品</term>：输入参数x、gx、gamma和输出参数dx、dgx的数据类型不支持BFLOAT16。

## 约束说明

- `dy`、`x`、`gx`、`dx`、`dgx` 的 shape 必须完全一致，且 `dy` 的各维度都必须大于 0。
- `mean`、`rstd` 的 rank 必须与 `dy` 一致，前导维与 `dy` 对齐，归一化对应的尾部维必须全部为 `1`，并且 `mean` 与 `rstd` 的 shape 必须一致。
- `gamma` 的 rank 必须小于 `dy` 的 rank，且其各维度必须与 `dy` 的尾部维一一匹配；`dbeta`、`dgamma` 的 shape 也必须与 `gamma` 一致。
- `dy/x/gx/gamma/dx/dgx` 的 dtype 必须一致，只支持 `FLOAT32`、`FLOAT16`、`BFLOAT16`；`mean/rstd/dbeta/dgamma` 只支持 `FLOAT32`。
- tiling 侧会对 shape 乘积和字节跨度做溢出检查，不满足约束时会直接拒绝输入。
- 当 `numCols < 500 && numRows >= 1024` 时，算子走 small-D tiling 分支，并要求额外 workspace；其余场景走普通分支。

## 调用说明

| 调用方式   | 样例代码           | 说明                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn接口  | [test_aclnn_deep_norm_grad](examples/test_aclnn_deep_norm_grad.cpp) | 通过[aclnnDeepNormGrad](docs/aclnnDeepNormGrad.md)接口方式调用DeepNormGrad算子。 |
| 图模式 | -  | 通过[算子IR](op_graph/deep_norm_grad_proto.h)构图方式调用DeepNormGrad算子。         |

# InstanceNormGrad

## 产品支持情况

|产品             |  是否支持  |
|:-------------------------|:----------:|
|  <term>Ascend 950PR/Ascend 950DT</term>   |     √    |
|  <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>   |     √    |
|  <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>     |     √    |
|  <term>Atlas 200I/500 A2 推理产品</term>    |     √    |
|  <term>Atlas 推理系列产品</term>    |     √    |
|  <term>Atlas 训练系列产品</term>    |     √    |

## 功能说明

- 算子功能：Instance Normalization的反向传播。给定上游梯度`dy`与前向的`x`、`variance`、`mean`、`gamma`，计算`pd_x`、`pd_gamma`、`pd_beta`。`pd_x`逐`(N, C)`在空间维`(D, H, W)`上归约；`pd_gamma`、`pd_beta`在空间维之外再对`N`归约，只保留`C`维。
- 计算公式（`variance`为原始方差，`ε`为编译期常量`1e-6`，`rstd = (variance + ε)^(-1/2)`，`m = D*H*W`，`R = {D,H,W}`，`x_hat = (x - mean) * rstd`）：

  $$
  pd\_xl = dy \cdot gamma
  $$

  $$
  pd\_var = \sum_{R} \left( -0.5 \cdot pd\_xl \cdot (x - mean) \cdot (variance + \varepsilon)^{-3/2} \right)
  $$

  $$
  pd\_mean = \sum_{R} \left( -1.0 \cdot pd\_xl \cdot rstd \right)
  $$

  $$
  pd\_x = pd\_xl \cdot rstd + pd\_var \cdot \frac{2}{m} \cdot (x - mean) + pd\_mean \cdot \frac{1}{m}
  $$

  $$
  pd\_gamma = \sum_{N,R} (dy \cdot x\_hat), \quad pd\_beta = \sum_{N,R} (dy)
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
      <td>表示反向传回的梯度，对应公式中的`dy`。shape与数据类型与入参`x`一致。</td>
      <td>FLOAT16、FLOAT32</td>
      <td>NDHWC</td>
    </tr>
    <tr>
      <td>x</td>
      <td>输入</td>
      <td>表示正向算子的输入，对应公式中的`x`。</td>
      <td>FLOAT16、FLOAT32</td>
      <td>NDHWC</td>
    </tr>
    <tr>
      <td>variance</td>
      <td>输入</td>
      <td>表示每个instance的方差，对应公式中的`variance`。数据类型与入参`dy`一致。</td>
      <td>FLOAT16、FLOAT32</td>
      <td>NDHWC</td>
    </tr>
    <tr>
      <td>mean</td>
      <td>输入</td>
      <td>表示每个instance的均值，对应公式中的`mean`。shape与数据类型与入参`variance`一致。</td>
      <td>FLOAT16、FLOAT32</td>
      <td>NDHWC</td>
    </tr>
    <tr>
      <td>gamma</td>
      <td>输入</td>
      <td>表示标准化过程中的缩放张量，对应公式中的`gamma`。数据类型与入参`dy`一致。</td>
      <td>FLOAT16、FLOAT32</td>
      <td>NDHWC</td>
    </tr>
    <tr>
      <td>pd_x</td>
      <td>输出</td>
      <td>表示对`x`的梯度，对应公式中的`pd_x`。shape与数据类型与入参`x`一致。</td>
      <td>FLOAT16、FLOAT32</td>
      <td>NDHWC</td>
    </tr>
    <tr>
      <td>pd_gamma</td>
      <td>输出</td>
      <td>表示对`gamma`的梯度，对应公式中的`pd_gamma`。shape与数据类型与入参`gamma`一致。</td>
      <td>FLOAT16、FLOAT32</td>
      <td>NDHWC</td>
    </tr>
    <tr>
      <td>pd_beta</td>
      <td>输出</td>
      <td>表示对`beta`的梯度，对应公式中的`pd_beta`。shape与数据类型与入参`gamma`一致。</td>
      <td>FLOAT16、FLOAT32</td>
      <td>NDHWC</td>
    </tr>
  </tbody></table>

## 约束说明

- 确定性说明：确定性实现。
- `x`为空Tensor时，`pd_x`同为空Tensor，`pd_gamma`、`pd_beta`的所有元素均为0。

## 调用说明

| 调用方式   | 样例代码           | 说明                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| 图模式 | [test_geir_instance_norm_grad](examples/test_geir_instance_norm_grad.cpp) | 通过[算子IR](op_graph/instance_norm_grad_proto.h)构图方式调用InstanceNormGrad算子。 |

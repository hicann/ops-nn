# BatchNormalizationGrad

## 产品支持情况

| 产品 | 是否支持 |
| ---- | :----:|
|Ascend 950PR/Ascend 950DT|×|
|Atlas A3 训练系列产品/Atlas A3 推理系列产品|×|
|Atlas A2 训练系列产品/Atlas A2 推理系列产品|√|
|Atlas 200I/500 A2 推理产品|×|
|Atlas 推理系列产品|×|
|Atlas 训练系列产品|×|

## 功能说明

- 算子功能：完成 BatchNormalization 反向梯度计算。

- 计算公式：

$$
grad\_input = \frac{weight \cdot save\_invstd}{\sqrt{m}} \cdot (m \cdot grad\_output - sum(grad\_output) - xhat \cdot sum(grad\_output \cdot xhat))
$$

$$
grad\_weight = sum(grad\_output \cdot xhat)
$$

$$
grad\_bias = sum(grad\_output)
$$

其中 $xhat = (input - save\_mean) \cdot save\_invstd$，$m = N \times H \times W \times ...$（每个通道的元素数）。

## 参数说明

<table style="undefined;table-layout: fixed; width: 980px"><colgroup>
  <col style="width: 100px">
  <col style="width: 150px">
  <col style="width: 280px">
  <col style="width: 330px">
  <col style="width: 120px">
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
      <td>grad_output</td>
      <td>输入</td>
      <td>反向传播输入的梯度。</td>
      <td>FLOAT、FLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>input</td>
      <td>输入</td>
      <td>前向传播的输入特征图。</td>
      <td>FLOAT、FLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>weight</td>
      <td>输入</td>
      <td>BN 层的 gamma 权重参数。</td>
      <td>FLOAT、FLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>bias</td>
      <td>输入</td>
      <td>BN 层的 beta 偏置参数（未参与梯度计算，保留接口兼容性）。</td>
      <td>FLOAT、FLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>save_mean</td>
      <td>输入</td>
      <td>前向传播保存的均值。</td>
      <td>FLOAT、FLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>save_invstd</td>
      <td>输入</td>
      <td>前向传播保存的标准差倒数。</td>
      <td>FLOAT、FLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>epsilon</td>
      <td>属性</td>
      <td>BN 层 epsilon 参数。默认值为 1e-5。</td>
      <td>FLOAT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>grad_input</td>
      <td>输出</td>
      <td>输入特征图的梯度。</td>
      <td>FLOAT、FLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>grad_weight</td>
      <td>输出</td>
      <td>权重参数的梯度。</td>
      <td>FLOAT、FLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>grad_bias</td>
      <td>输出</td>
      <td>偏置参数的梯度。</td>
      <td>FLOAT、FLOAT16</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## 约束说明

- 输入张量维度至少为 2 维（N, C, ...）。
- 支持多核并行计算（按通道维度划分）。
- save_mean、save_invstd、weight、bias、grad_weight、grad_bias 的第一维大小等于通道数 C。

## 调用说明

<table><thead>
  <tr>
    <th>调用方式</th>
    <th>调用样例</th>
    <th>说明</th>
  </tr></thead>
<tbody>
  <tr>
    <td>aclnn调用</td>
    <td><a href="./examples/test_aclnn_batch_normalization_grad.cpp">test_aclnn_batch_normalization_grad</a></td>
    <td>参见算子调用完成算子编译和验证。</td>
  </tr>
</tbody>
</table>

# LayerNormalizationGrad

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

- 算子功能：完成 LayerNormalization 梯度计算。

- 计算公式：

$$
x\_hat[i] = (x[i] - mean[i]) * rstd[i]
$$

$$
dxhat[i] = dy[i] * gamma
$$

$$
ds[i] = sum(dxhat[i] * x\_hat[i])
$$

$$
db[i] = sum(dxhat[i])
$$

$$
dx[i] = rstd[i] * (dxhat[i] - (db[i] + ds[i] * x\_hat[i]) / D)
$$

$$
dgamma = sum(dy * x\_hat, axis=0)
$$

$$
dbeta = sum(dy, axis=0)
$$

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
      <td>dy</td>
      <td>输入</td>
      <td>上游梯度，形状为 [N, D]。</td>
      <td>FLOAT、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>x</td>
      <td>输入</td>
      <td>前向输入，形状为 [N, D]。</td>
      <td>FLOAT、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>gamma</td>
      <td>输入</td>
      <td>缩放参数，形状为 [D]。</td>
      <td>FLOAT、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>mean</td>
      <td>输入</td>
      <td>前向保存的均值，形状为 [N]。</td>
      <td>FLOAT、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>rstd</td>
      <td>输入</td>
      <td>前向保存的 1/sqrt(var+eps)，形状为 [N]。</td>
      <td>FLOAT、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>dx</td>
      <td>输出</td>
      <td>对输入的梯度，形状为 [N, D]。</td>
      <td>FLOAT、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>dgamma</td>
      <td>输出</td>
      <td>对 gamma 的梯度，形状为 [D]。</td>
      <td>FLOAT、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>dbeta</td>
      <td>输出</td>
      <td>对 beta 的梯度，形状为 [D]。</td>
      <td>FLOAT、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## 约束说明

无

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
    <td><a href="./examples/test_aclnn_layer_normalization_grad.cpp">test_aclnn_layer_normalization_grad</a></td>
    <td>参见算子调用完成算子编译和验证。</td>
  </tr>
</tbody>
</table>

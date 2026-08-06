# BNInferGrad

## 产品支持情况

| 产品 | 是否支持 |
| :----------------------------------------- | :------:|
| <term>Ascend 950PR/Ascend 950DT</term> | √ |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term> | √ |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> | √ |
| <term>Atlas 200I/500 A2 推理产品</term> | × |
| <term>Atlas 推理系列产品</term> | √ |
| <term>Atlas 训练系列产品</term> | √ |

## 功能说明

- 算子功能：计算Batch Normalization推理模式下输入的反向梯度。
- 计算公式：

  $$
  x\_backprop = grads \times \frac{scale}{\sqrt{batch\_variance + epsilon}}
  $$

## 参数说明

<table style="table-layout: fixed; width: 1576px"><colgroup>
<col style="width: 170px">
<col style="width: 170px">
<col style="width: 520px">
<col style="width: 300px">
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
    <td>grads</td>
    <td>输入</td>
    <td>上游梯度，支持2D至5D张量。</td>
    <td>FLOAT16、FLOAT、BFLOAT16</td>
    <td>NHWC、NCHW、NC1HWC0</td>
  </tr>
  <tr>
    <td>scale</td>
    <td>输入</td>
    <td>缩放参数，必须为一维张量，长度等于grads的C维。</td>
    <td>FLOAT</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>batch_variance</td>
    <td>输入</td>
    <td>推理阶段使用的方差，必须为一维张量，长度等于grads的C维。</td>
    <td>FLOAT</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>epsilon</td>
    <td>可选属性</td>
    <td>防止除零的小常数，默认值为0.0001。</td>
    <td>FLOAT</td>
    <td>-</td>
  </tr>
  <tr>
    <td>x_backprop</td>
    <td>输出</td>
    <td>输入梯度，shape和数据类型与grads一致。</td>
    <td>FLOAT16、FLOAT、BFLOAT16</td>
    <td>NHWC、NCHW、NC1HWC0</td>
  </tr>
</tbody></table>

## 约束说明

- grads的逻辑C维（NCHW的第二维、NHWC的最后一维）必须等于scale和batch_variance的长度。
- scale和batch_variance仅支持FLOAT。
- 原型中grads和x_backprop的逻辑格式支持NHWC、NCHW或NC1HWC0；scale和batch_variance为一维ND。
- Ascend950 Vector分支的grads和x_backprop仅支持NCHW和NHWC；不支持NC1HWC0。
- x_backprop的shape、数据类型和数据格式与grads一致。
- epsilon仅限定为FLOAT标量，无额外范围、枚举或步长约束。

## 调用说明

<table><thead>
  <tr>
    <th>调用方式</th>
    <th>调用样例</th>
    <th>说明</th>
  </tr></thead>
<tbody>
  <tr>
    <td>图模式调用</td>
    <td><a href="./examples/test_geir_bn_infer_grad.cpp">test_geir_bn_infer_grad</a></td>
    <td>通过<a href="./op_graph/bn_infer_grad_proto.h">算子IR</a>构图并调用BNInferGrad算子。</td>
  </tr>
</tbody></table>

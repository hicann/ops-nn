# GeluGradV2

## 产品支持情况


| 产品                                        | 是否支持 |
| ------------------------------------------- | :------: |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 |    √    |

## 功能说明

- 算子功能：求geluv2函数梯度。
- 计算公式：

  Gelu正向（其中x可以为标量或者Tensor）：

  $$
  Gelu(x)=x \cdot \Phi(x)=x/2 \cdot [1+erf(x/\sqrt{2})]
  $$

  其中erf的计算公式为：

  $$
  erf(x)=\frac{2}{\sqrt \pi}\sum^{\infty}_{n=0}{\frac{(-1)^n \cdot x^{2n+1}}{n! \cdot (2n+1)}}
  $$

  gradInput和gradOutput的关系可以表示为：

  $$
  gradInput = gradOutput \cdot (\frac{1}{2}+\frac{1}{2} \cdot erf(\frac{x}{\sqrt2})+\frac{x}{\sqrt{2\pi}} \cdot e^{-\frac{x^2}{2}})
  $$

  Gelu近似计算公式为：

  $$
  Gelu(x)=0.5x(1+tanh(\sqrt{2/\pi}(x+0.044715x^3)))
  $$

## 参数说明

<table style="undefined;table-layout: fixed; width: 820px"><colgroup>
  <col style="width: 100px">
  <col style="width: 150px">
  <col style="width: 190px">
  <col style="width: 260px">
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
      <td>x</td>
      <td>输入</td>
      <td>反向传播梯度</td>
      <td>FLOAT、FLOAT16、BFLOAT16</td>
      <td>ND、FRACTAL_NZ,NC1HWC0</td>
    </tr>
    <tr>
      <td>dy</td>
      <td>输入</td>
      <td>与"x"具有相同的类型、格式和形状。</td>
      <td>FLOAT、FLOAT16、BFLOAT16</td>
      <td>ND、FRACTAL_NZ,NC1HWC0</td>
    </tr>
    <tr>
      <td>approximate</td>
      <td>属性</td>
      <td>可选</td>
      <td>str</td>
      <td></td>
    </tr>
    <tr>
      <td>z</td>
      <td>输出</td>
      <td>公式中的输出张量</td>
      <td>FLOAT、FLOAT16、BFLOAT16</td>
      <td>ND、FRACTAL_NZ,NC1HWC0</td>
    </tr>
  </tbody></table>

## 调用说明


| 调用方式  | 样例代码                                                                | 说明                                                                                        |
| --------- | ----------------------------------------------------------------------- | ------------------------------------------------------------------------------------------- |
| Aclnn模式 | [test_aclnn_gelu_backward_v2](examples/test_aclnn_gelu_backward_v2.cpp) | 通过Aclnn接口调用GeluGradV2算子[aclnnGeluBackwardV2接口文档](docs/aclnnGeluBackwardV2.md)。 |

### 约束说明

无

## 贡献说明


| 贡献者      | 贡献方     | 贡献算子   | 贡献时间 | 贡献内容                 |
| ----------- | ---------- | ---------- | -------- | ------------------------ |
| ilovescrapy | 个人开发者 | GeluGradV2 | 2026     | GeluGradV2算子适配开源仓 |

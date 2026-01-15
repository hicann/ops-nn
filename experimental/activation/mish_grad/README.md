# MishGrad

## 产品支持情况


| 产品                                                               | 是否支持 |
| ------------------------------------------------------------------ | :------: |
| Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件 |    √    |

## 功能说明

- 算子功能：计算mish的梯度。
- 计算公式：

tanhx输入时：

$$
xgrad=grad*(tanhx-x*(1-tan^{2}x)*e^{x})/(1+e^{x})
$$

tanhx不输入时：

$$
y1=1.0 / ((1.0 + e^{x})^{2} + 1.0)
$$

$$
xgrad = grad * (4 * x * e^{x} * (e^{x} + 1.0) * y_{1} ^{2} - 2.0 * y_{1} + 1.0)
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
      <td>grad</td>
      <td>输入</td>
      <td>待进行mishgrad计算的入参，公式中的grad。</td>
      <td>FLOAT、FLOAT16、BFLOAT16</td>
      <td>ND、NCHW、NHWC、NC1HWC0、FRACTAL_Z、FRACTAL_NZ</td>
    </tr>
    <tr>
      <td>x</td>
      <td>输入</td>
      <td>待进行mishgrad计算的入参，公式中的x。</td>
      <td>FLOAT、FLOAT16、BFLOAT16</td>
      <td>ND、NCHW、NHWC、NC1HWC0、FRACTAL_Z、FRACTAL_NZ</td>
    </tr>
    <tr>
      <td>tanhx</td>
      <td>输入</td>
      <td>待进行mishgrad计算的可选入参，公式中的tanhx。</td>
      <td>FLOAT、FLOAT16、BFLOAT16</td>
      <td>ND、NCHW、NHWC、NC1HWC0、FRACTAL_Z、FRACTAL_NZ</td>
    </tr>
    <tr>
      <td>x_grad</td>
      <td>输出</td>
      <td>待进行mishgrad计算的出参，公式中的xgrad。</td>
      <td>FLOAT、FLOAT16、BFLOAT16</td>
      <td>ND、NCHW、NHWC、NC1HWC0、FRACTAL_Z、FRACTAL_NZ</td>
    </tr>
  </tbody></table>

## 约束说明

无

## 调用说明


| 调用方式  | 调用样例                                                    | 说明                                                                   |
| --------- | ----------------------------------------------------------- | ---------------------------------------------------------------------- |
| aclnn调用 | [test_aclnn_mish_grad](./examples/test_aclnn_mish_grad.cpp) | 通过[aclnnMishBackward](./docs/aclnnMishBackward.md)接口方式调用MishGrad算子。 |

## 贡献说明

| 贡献者 | 贡献方 | 贡献算子 | 贡献时间 | 贡献内容 |
| ---- | ---- | ---- | ---- | ---- |
| ilovescrapy | 个人开发者 | MishGrad | 2025/12/23 | MishGrad算子适配开源仓 |
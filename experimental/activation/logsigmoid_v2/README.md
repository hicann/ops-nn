# LogsigmoidV2

## 产品支持情况

| 产品 | 是否支持 |
| ---- | :----:|
| Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件 | √ |

## 功能说明

- 算子功能：计算输入张量的 LogSigmoid 值，即对每个元素计算 \( \log(\frac{1}{1 + e^{-x}}) \)。

- 计算公式：

\[ \text{out}_i = \log\left(\frac{1}{1 + e^{-x_i}}\right) \]

## 参数说明

&lt;table style="undefined;table-layout: fixed; width: 980px"&gt;&lt;colgroup&gt;
  &lt;col style="width: 100px"&gt;
  &lt;col style="width: 150px"&gt;
  &lt;col style="width: 280px"&gt;
  &lt;col style="width: 330px"&gt;
  &lt;col style="width: 120px"&gt;
  &lt;/colgroup&gt;
  &lt;thead&gt;
    &lt;tr&gt;
      &lt;th&gt;参数名&lt;/th&gt;
      &lt;th&gt;输入/输出/属性&lt;/th&gt;
      &lt;th&gt;描述&lt;/th&gt;
      &lt;th&gt;数据类型&lt;/th&gt;
      &lt;th&gt;数据格式&lt;/th&gt;
    &lt;/tr&gt;&lt;/thead&gt;
  &lt;tbody&gt;
    &lt;tr&gt;
      &lt;td&gt;x&lt;/td&gt;
      &lt;td&gt;输入&lt;/td&gt;
      &lt;td&gt;待进行 LogsigmoidV2 计算的输入张量。&lt;/td&gt;
      &lt;td&gt;FLOAT、FLOAT16&lt;/td&gt;
      &lt;td&gt;ND&lt;/td&gt;
    &lt;/tr&gt;  
    &lt;tr&gt;
      &lt;td&gt;z&lt;/td&gt;
      &lt;td&gt;输出&lt;/td&gt;
      &lt;td&gt;LogsigmoidV2 计算后的输出张量。&lt;/td&gt;
      &lt;td&gt;FLOAT、FLOAT16&lt;/td&gt;
      &lt;td&gt;ND&lt;/td&gt;
    &lt;/tr&gt;
  &lt;/tbody&gt;&lt;/table&gt;

## 约束说明

无

## 调用说明

| 调用方式 | 调用样例 | 说明 |
|--------------|------------------------------------------------------------------------|--------------------------------------------------------------|
| aclnn调用 | [test_aclnn_logsigmoid_v2.cpp](./examples/test_aclnn_logsigmoid_v2.cpp) | 通过 [test_aclnn_logsigmoid_v2](./docs/test_aclnn_logsigmoid_v2.md) 接口方式调用 LogsigmoidV2 算子。 |

## 贡献说明

| 贡献者 | 贡献方 | 贡献算子 | 贡献时间 | 贡献内容 |
| ---- | ---- | ---- | ---- | ---- |
| Li Wen | 个人开发者 | 2025/11/25 | LogsigmoidV2 算子适配开源仓 |

# ThresholdGradV2D

## 产品支持情况


| 产品                                                               | 是否支持 |
| ------------------------------------------------------------------ | :------: |
| Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件 |    √    |

## 功能说明

- 算子功能：求thresholdv2d函数梯度。
- 计算公式：

$$
 y = 
 	 \begin{cases} 
 	 input\_ gradient, & input\_ feature > threshold \\
 	 0, & input\_ feature <= threshold
 	 \end{cases}
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
      <td>input_gradient</td>
      <td>输入</td>
      <td>反向传播梯度</td>
      <td>FLOAT、FLOAT16、BFLOAT16、INT32、INT8、UINT8</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>input_feature</td>
      <td>输入</td>
      <td>与"input_gradient"具有相同的类型、格式和形状。</td>
      <td>FLOAT、FLOAT16、BFLOAT16、INT32、INT8、UINT8</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>threshold</td>
      <td>属性</td>
      <td></td>
      <td>FLOAT</td>
      <td></td>
    </tr>
    <tr>  
      <td>output_backprops</td>
      <td>输出</td>
      <td>公式中的输出张量</td>
      <td>FLOAT、FLOAT16、BFLOAT16、INT32、INT8、UINT8</td>
      <td>ND</td>
    </tr>
  </tbody></table>



## 调用说明

| 调用方式   | 样例代码           | 说明                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| Aclnn模式 | [test_aclnn_threshold_backward.cpp](./examples/test_aclnn_threshold_backward.cpp)  | 通过Aclnn接口调用ThresholdGradV2D算子。         |


## 约束说明

无

## 贡献说明


| 贡献者      | 贡献方     | 贡献算子  | 贡献时间   | 贡献内容                |
| ----------- | ---------- | --------- | ---------- | ----------------------- |
| ilovescrapy | 个人开发者 | ThresholdGradV2D | 2026/3/23 | ThresholdGradV2D算子适配开源仓 |

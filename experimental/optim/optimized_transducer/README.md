# OptimizedTransducer

## 产品支持情况

| 产品 | 是否支持 |
| ---- | :----:|
|Atlas A2 训练系列产品/Atlas 800I A2 推理产品|√|

## 功能说明

- 算子功能：完成计算输入音频序列与输出文本序列之间的损失。

- 计算公式：
给定模型输出logits、目标序列targets和长度信息，RNN-T损失计算负对数似然：
$$\text{Loss} = -\frac{1}{B} \sum_{b=1}^{B} \log P(\mathbf{y}_b | \mathbf{x}_b)$$

其中对齐概率通过对所有可能对齐路径求和得到：

$$
P(\mathbf{y} | \mathbf{x}) = \sum_{\mathbf{a} \in \mathcal{A}(\mathbf{x}, \mathbf{y})} \prod_{t=1}^{T+U} P(a_t | \mathbf{x}, a_{1:t-1})$$
每个时间步的发射概率通过softmax计算：
$$P(k | t, u) = \frac{\exp(\text{logits}[t, u, k] / \tau)}{\sum_{k'=0}^{V-1} \exp(\text{logits}[t, u, k'] / \tau)}$$

前向计算使用动态规划的前向算法计算所有对齐路径的总概率：
$$\alpha(t, u) = \alpha(t - 1, u) \cdot P(\text{blank}|t - 1, u) + \alpha(t, u - 1) \cdot P(y_u|t, u - 1)$$
反向传播梯度通过前向后向算法传播：
$$
\frac{\partial \text{Loss}}{\partial \text{logits}[t, u, k]} = \begin{cases} P(k|t, u) \cdot (\beta_{\text{blank}} - 1) & \text{if } k = \text{blank} \\ P(k|t, u) \cdot (\beta_{\text{label}} - 1) & \text{if } k = y_u \\ P(k|t, u) \cdot \beta_{\text{other}} & \text{otherwise} \end{cases}$$

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
      <td>logits</td>
      <td>输入</td>
      <td>输入序列</td>
      <td>FLOAT32、FLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>targets</td>
      <td>输入</td>
      <td>目标序列</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>logit_lengths</td>
      <td>输入</td>
      <td>每个输入序列的长度</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>target_lengths</td>
      <td>输入</td>
      <td>每个目标序列的长度</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>loss</td>
      <td>输出</td>
      <td>损失结果</td>
      <td>FLOAT32、FLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>grad</td>
      <td>输出</td>
      <td>输入logits的梯度结果</td>
      <td>FLOAT32、FLOAT16</td>
      <td>ND</td>
    </tr>
    <tr> 
      <td>blank</td> 
      <td>属性</td> 
      <td>blank的索引位置，默认为-1</td> 
      <td>INT</td> 
      <td>-</td> 
    </tr>
    <tr> 
      <td>clamp</td> 
      <td>属性</td> 
      <td>梯度的范围，默认为-1表示不限制梯度</td> 
      <td>DOUBLE</td> 
      <td>-</td> 
    </tr>
    <tr> 
      <td>fused_log_softmax</td> 
      <td>属性</td> 
      <td>输入是否经过log_softmax，默认为true</td> 
      <td>BOOL</td> 
      <td>-</td> 
    </tr>
    <tr> 
      <td>requires_grad</td> 
      <td>属性</td> 
      <td>是否需要计算梯度，默认为true</td> 
      <td>BOOL</td> 
      <td>-</td> 
    </tr>
  </tbody></table>

## 约束说明

logits的输入shape需要保证(V + max(T, U)) * 3大小的数据不会超过UB空间

## 调用说明

| 调用方式 | 调用样例                                                                   | 说明                                                           |
|--------------|------------------------------------------------------------------------|--------------------------------------------------------------|
| aclnn调用 | [test_aclnn_optimized_transducer](./examples/test_aclnn_optimized_transducer.cpp) | 通过aclnnOptimizedTransducer接口方式调用OptimizedTransducer算子。 |

## 贡献说明

| 贡献者 | 贡献方 | 贡献算子 | 贡献时间 | 贡献内容 |
| ---- | ---- | ---- | ---- | ---- |
| HustAga | 个人开发者 | OptimizedTransducer | 2025/12/30 | OptimizedTransducer算子适配开源仓 |

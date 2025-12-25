# CrossEntropyLossGrad

## 产品支持情况

|产品             |  是否支持  |
|:-------------------------|:----------:|
|  <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>   |     √    |
|  <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>     |     √    |

## 功能说明

- **算子功能**：aclnnCrossEntropyLoss的反向传播。
- **计算公式**：

  $$
  ignoreMask_{target(t)}=\begin{cases}
  1, &target(t) ≠ ignoreIndex \\
  0, &target(t) = ignoreIndex
  \end{cases}
  $$

  $$
  smoothLossGrad=\begin{cases}
  grad / sum(weight_{target}* ignoreMask) * labelSmoothing / C, &reduction = mean \\
  grad * labelSmoothing / C, &reduction = sum \\
  grad * labelSmoothing / C, &reduction = none
  \end{cases}
  $$

  $$
  lossOutGrad=\begin{cases}
  grad * (1-labelSmoothing) / sum(weight_{target}* ignoreMask) * ignoreMask, &reduction = mean \\
  grad * (1-labelSmoothing) * ignoreMask, &reduction = sum \\
  grad * (1-labelSmoothing) * ignoreMask, &reduction = none
  \end{cases}
  $$

  $$
  nllLossGrad = lossOutGrad * weight_{target}
  $$

  $$
  logSoftmaxGradLossOutSubPart = exp(logProb) * nllLossGrad
  $$

  $$
  predictionsGradLossOut_{ij}=\begin{cases}
  nllLossGrad_i, & j=target(i)  \\
  0, & j ≠ target(i) 
  \end{cases}
  $$

  $$
  predictionsGradLossOut = logSoftmaxGradLossOutSubPart - predictionsGradLossOut
  $$

  $$
  smoothLossGrad = smoothLossGrad * ignoreMask
  $$

  $$
  logSoftmaxGradSmoothLoss = smoothLossGrad * weight
  $$

  $$
  predictionsGradSmoothLoss = exp(logProb) * sum(logSoftmaxGradSmoothLoss) - logSoftmaxGradSmoothLoss
  $$

  不涉及zloss场景输出：

  $$
  xGrad_{out} = predictionsGradLossOut + predictionsGradSmoothLoss
  $$

  zloss场景：

  $$
  gradZ=\begin{cases}
  grad + gradZloss, & 传入gradZloss  \\
  grad, & 不传gradZloss
  \end{cases}
  $$

  $$
  zlossGrad=\begin{cases}
  gradZ / sum(ignoreMask), & &reduction = mean  \\
  gradZ, & &reduction = sum \\
  gradZ, & &reduction = none
  \end{cases}
  $$

  $$
  lseGrad = 2 * lseSquareScaleForZloss * lseForZloss * ignoreMask * zlossGrad
  $$

  $$
  zlossOutputGrad = exp(logProb) * lseGrad
  $$

  zloss场景输出：

  $$
  xGrad_{out} = xGrad_{out} + zlossOutputGrad
  $$

## 参数说明

<table style="undefined;table-layout: fixed; width: 1576px"><colgroup>
  <col style="width: 170px">
  <col style="width: 170px">
  <col style="width: 312px">
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
      <td>grad</td>
      <td>输入</td>
      <td>正向输出loss的梯度，公式中的grad。</td>
      <td>FLOAT16、BFLOAT16、FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>logProb</td>
      <td>输入</td>
      <td>正向输入的logSoftmax计算结果，公式中的logProb。</td>
      <td>FLOAT16、BFLOAT16、FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>target</td>
      <td>输入</td>
      <td>类索引，公式中的target。</td>
      <td>INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>weight</td>
      <td>可选输入</td>
      <td><li>表示为每个类别指定的缩放权重，公式中的weight。<li>默认为全1。</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>gradZloss</td>
      <td>可选输入</td>
      <td><li>正向输出zloss的梯度，公式中的gradZloss。<li>当前暂不支持。</td>
      <td>FLOAT16、BFLOAT16、FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>lseForZloss</td>
      <td>可选输入</td>
      <td><li>zloss相关输入，如果lse_square_scale_for_zloss非0，正向额外输出的lse_for_zloss中间结果给反向用于计算lse，公式中的lseForZloss。<li>当前暂不支持。</td>
      <td>FLOAT16、BFLOAT16、FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>reduction</td>
      <td>可选属性</td>
      <td><li>指定要应用于输出的归约方式。<li>默认值为“mean”。</td>
      <td>STRING</td>
      <td>-</td>
    </tr>
    <tr>
      <td>ignoreIndex</td>
      <td>可选属性</td>
      <td><li>指定忽略不影响输入梯度的目标值。<li>默认值为-100。</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>labelSmoothing</td>
      <td>可选属性</td>
      <td><li>表示计算loss时的平滑量。<li>当前仅支持输入0.0。</td>
      <td>DOUBLE</td>
      <td>-</td>
    </tr>
    <tr>
      <td>lseSquareScaleForZloss</td>
      <td>可选属性</td>
      <td><li>表示zloss计算所需的scale。<li>当前暂不支持。</td>
      <td>DOUBLE</td>
      <td>-</td>
    </tr>
    <tr>
      <td>out</td>
      <td>输出</td>
      <td>梯度计算结果，对应公式中的out。</td>
      <td>FLOAT16、BFLOAT16、FLOAT32</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## 约束说明

  - target仅支持类标签索引，不支持概率输入。
  - gradLoss、logProb、gradZlossOptional、lseForZlossOptional、xGradOut数据类型需保持一致。
  - 当前暂不支持zloss功能，传入相关输入，即gradZlossOptional、lseForZlossOptional、lseSquareScaleForZloss，不会生效。

## 调用说明

| 调用方式   | 样例代码           | 说明                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn接口  | [test_aclnn_cross_entropy_loss_grad](examples/test_aclnn_cross_entropy_loss_grad.cpp) | 通过[aclnnCrossEntropyLossGrad](docs/aclnnCrossEntropyLossGrad.md)接口方式调用CrossEntropyLossGrad算子。 |
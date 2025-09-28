# CrossEntropyLoss

## 产品支持情况

|产品             |  是否支持  |
|:-------------------------|:----------:|
|  <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>   |     √    |
|  <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>     |     √    |

## 功能说明

- 算子功能：计算输入的交叉熵损失。
- 计算表达式：
  
  reductionOptional = mean时，交叉熵损失loss的计算公式为：
  $$
  l_n = -weight_{y_n}*log\frac{exp(x_{n,y_n})}{\sum_{c=1}^Cexp(x_{n,c})}*1\{y_n\ !=\ ignoreIndex \}
  $$
  $$
  loss=\begin{cases}\sum_{n=1}^N\frac{1}{\sum_{n=1}^Nweight_{y_n}*1\{y_n\ !=\ ignoreIndex \}}l_n,&\text{if reductionOptional = ‘mean’} \\\sum_{n=1}^Nl_n,&\text {if reductionOptional = ‘sum’ }\\\{l_0,l_1,...,l_n\},&\text{if reductionOptional = ‘None’ }\end{cases}
  $$
  log\_prob计算公式为：
  $$
  lse_n = log*\sum_{c=1}^{C}exp(x_{n,c})
  $$
  $$
  logProb_{n,c} = x_{n,c} - lse_n
  $$
  zloss计算公式为：
  $$
  zloss_n = lseSquareScaleForZloss * （lse_n）^2 
  $$
  其中，N为batch数，C为标签数。

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
      <td>input</td>
      <td>输入</td>
      <td>表示输入，公式中的`input`。</td>
      <td>FLOAT16、BFLOAT16、FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>target</td>
      <td>输入</td>
      <td>表示标签，公式中的`y`。</td>
      <td>INT64，INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>weightOptional</td>
      <td>可选输入</td>
      <td><li>表示为每个类别指定的缩放权重，公式中的`weight`。<li>默认为全1。</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>reductionOptional</td>
      <td>可选属性</td>
      <td><li>表示loss的归约方式。<li>默认值为“mean”。</td>
      <td>STRING</td>
      <td>-</td>
    </tr>
    <tr>
      <td>ignoreIndex</td>
      <td>可选属性</td>
      <td><li>指定忽略的标签。<li>默认值为-100。</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>labelSmoothing</td>
      <td>可选属性</td>
      <td><li>表示计算loss时的平滑量。<li>默认值为0。</td>
      <td>DOUBLE</td>
      <td>-</td>
    </tr>
    <tr>
      <td>lseSquareScaleForZloss</td>
      <td>可选属性</td>
      <td><li>表示zloss计算所需的scale。<li>当前仅支持传入0。</td>
      <td>DOUBLE</td>
      <td>-</td>
    </tr>
    <tr>
      <td>returnZloss</td>
      <td>可选属性</td>
      <td><li>控制是否返回zloss输出。Host侧的布尔值。需要输出zLoss时传入True，否则传入False。<li>当前仅支持传入False。</td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
    <tr>
      <td>lossOut</td>
      <td>输出</td>
      <td>表示输出损失，对应公式中的`loss`。</td>
      <td>FLOAT16、BFLOAT16、FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>logProbOut</td>
      <td>输出</td>
      <td>输出给反向计算的输出，对应公式中的`logProb`。</td>
      <td>FLOAT16、BFLOAT16、FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>zlossOut</td>
      <td>输出</td>
      <td><li>表示辅助损失，对应公式中的`zlossOut`。<li>当前暂不支持。</td>
      <td>FLOAT16、BFLOAT16、FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>lseForZlossOut</td>
      <td>输出</td>
      <td><li>表示zloss场景输出给反向的Tensor，lseSquareScaleForZloss为0时输出为None，对应公式中的`lse`。<li>当前暂不支持。</td>
      <td>FLOAT16、BFLOAT16、FLOAT32</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## 约束说明

  - target仅支持类标签索引，不支持概率输入。
  - 当前暂不支持zloss相关功能。lseSquareScaleForZloss、returnZloss仅支持传入nullptr。
  - input第零维N需满足N<200000。

## 调用说明

| 调用方式   | 样例代码           | 说明                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn接口  | [test_aclnn_cross_entropy_loss](examples/test_aclnn_cross_entropy_loss.cpp) | 通过[aclnnCrossEntropyLoss](docs/aclnnCrossEntropyLoss.md)接口方式调用CrossEntropyLoss算子。 |
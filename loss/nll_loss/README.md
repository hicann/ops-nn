# NllLoss

  ## 产品支持情况

|产品             |  是否支持  |
|:-------------------------|:----------:|
|  <term>Ascend 950PR/Ascend 950DT</term>   |     √    |
|  <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>   |     √    |
|  <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>     |     √    |
|  <term>Atlas 200I/500 A2 推理产品</term>    |     ×    |
|  <term>Atlas 推理系列产品</term>     |     √    |
|  <term>Atlas 训练系列产品</term>    |     √    |

  ## 功能说明

  - 接口功能：计算负对数似然损失值。

  - 计算公式：

    当`reduction`为`none`时，

    $$
    \ell(x, y) = L = \{l_1,\dots,l_N\}^\top, \quad
    l_n = - w_{y_n} x_{n,y_n}, \quad
    w_{c} = \text{weight}[c] \cdot \mathbb{1}\{c \not=   \text{ignoreIndex}\},
    $$

    其中$x$是self, $y$是target, $w$是weight，$N$是batch的大小. 如果`reduction`不是`'none'`, 那么：

    $$
    \ell(x, y) = \begin{cases}
        \sum_{n=1}^N \frac{1}{\sum_{n=1}^N w_{y_n}} l_n, &
        \text{if reduction} = \text{`mean';}\\
        \sum_{n=1}^N l_n,  &
        \text{if reduction} = \text{`sum'.}
    \end{cases}
    $$

    $$
    totalWeight = \sum_{n=1}^N w_{y_n}
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
        <td>x</td>
        <td>输入</td>
        <td>待进行nll_loss计算的入参，公式中的x。</td>
        <td>FLOAT、FLOAT16、BFLOAT16</td>
        <td>ND</td>
      </tr>
      <tr>
        <td>target</td>
        <td>输入</td>
        <td>待进行nll_loss计算的入参，公式中的target。</td>
        <td>INT32、INT64、UINT8</td>
        <td>ND</td>
      </tr>
      <tr>
        <td>weight</td>
        <td>输入</td>
        <td>待进行nll_loss计算的入参，公式中的weight。</td>
        <td>FLOAT、FLOAT16、BFLOAT16</td>
        <td>ND</td>
      </tr>
      <tr>
        <td>reduction</td>
        <td>输入</td>
        <td>待进行nll_loss计算的入参，公式中的reduction。</td>
        <td>STRING</td>
        <td>ND</td>
      </tr>
      <tr>
        <td>ignore_index</td>
        <td>输入</td>
        <td>待进行nll_loss计算的入参，公式中的ignoreIndex。</td>
        <td>INT32</td>
        <td>ND</td>
      </tr>
      <tr>
        <td>y</td>
        <td>输出</td>
        <td>待进行nll_loss计算的出参，公式中的L。</td>
        <td>FLOAT、FLOAT16、BFLOAT16</td>
        <td>ND</td>
      </tr>
    </tbody></table>

  ## 约束说明

  无

  ## 调用说明

  | 调用方式 | 调用样例                                                                   | 说明                                                           |
  |--------------|------------------------------------------------------------------------|--------------------------------------------------------------|
  | aclnn调用 | [test_aclnn_nll_loss](./examples/test_aclnn_nll_loss.cpp) | 通过[aclnnNLLLoss](./docs/aclnnNLLLoss.md)接口方式调用NLLLoss算子。 |
  | aclnn调用 | [test_aclnn_nll_loss_2d](./examples/test_aclnn_nll_loss_2d.cpp) | 通过[aclnnNLLLoss2d](./docs/aclnnNLLLoss2d.md)接口方式调用NLLLoss算子。 |
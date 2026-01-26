# GroupNormSiluQunat

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term>                             |    x     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    √     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> |    √     |
| <term>Atlas 200I/500 A2 推理产品</term>                      |    ×     |
| <term>Atlas 推理系列产品</term>                             |    x     |
| <term>Atlas 训练系列产品</term>                              |    ×     |


## 功能说明

- 接口功能：计算输入self的组归一化，输出均值meanOut，标准差的倒数rstdOut，以及对silu的输出结果进行量化的结果out。

- 计算公式：
  - **GroupNorm:**
  记 $E[x] = \bar{x}$代表$x$的均值，$Var[x] = \frac{1}{n} * \sum_{i=1}^n(x_i - E[x])^2$代表$x$的方差，则
  $$
  \left\{
  \begin{array} {rcl}
  groupNormOut& &= \frac{x - E[x]}{\sqrt{Var[x] + eps}} * \gamma + \beta \\
  meanOut& &= E[x]\\
  rstdOut& &= \frac{1}{\sqrt{Var[x] + eps}}\\
  \end{array}
  \right.
  $$
  - **Silu:**
  $$
  siluOut = \frac{groupNormOut}{1+e^{-groupNormOut}}
  $$

  - **Quant:**
  $$
  out = round(siluOut / quantScale)
  $$

      
## 参数说明

  <table style="undefined;table-layout: fixed; width: 1550px"><colgroup>
    <col style="width: 187px">
    <col style="width: 121px">
    <col style="width: 287px">
    <col style="width: 387px">
    <col style="width: 187px">
    <col style="width: 187px">
    <col style="width: 187px">
    <col style="width: 146px">
    </colgroup>
    <thead>
    <tr>
        <th>参数名</th>
        <th>输入/输出</th>
        <th>描述</th>
        <th>使用说明</th>
        <th>数据类型</th>
        <th>数据格式</th>
        <th>维度(shape)</th>
        <th>非连续Tensor</th>
    </tr></thead>
    <tbody>
    <tr>
        <td>self</td>
        <td>输入</td>
        <td>计算公式中的x。</td>
        <td>-</td>
        <td>FLOAT16、BFLOAT16</td>
        <td>ND</td>
        <td>2-8，其中第0维为N，第1维为C</td>
        <td>√</td>
    </tr>
    <tr>
        <td>gamma</td>
        <td>输入</td>
        <td>公式中的γ。</td>
        <td>数据类型与self保持一致，元素数量需与输入self的第1维度保持相同。</td>
        <td>FLOAT16、BFLOAT16</td>
        <td>ND</td>
        <td>1</td>
        <td>√</td>
    </tr>
    <tr>
        <td>beta</td>
        <td>输入</td>
        <td>公式中的β。</td>
        <td>数据类型与self保持一致，元素数量需与输入self的第1维度保持相同。</td>
        <td>FLOAT16、BFLOAT16</td>
        <td>ND</td>
        <td>1</td>
        <td>√</td>
    </tr>
    <tr>
        <td>quantScale</td>
        <td>输入</td>
        <td>公式中的quantScale。</td>
        <td>元素数量需为1或与输入self的第1维度保持相同。</td>
        <td>FLOAT32</td>
        <td>ND</td>
        <td>1</td>
        <td>√</td>
    </tr>
    <tr>
        <td>group</td>
        <td>输入</td>
        <td>表示将输入self的第1维度分为group组。</td>
        <td>group需可以整除self的第一维度</td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>eps</td>
        <td>输入</td>
        <td>公式中的eps。</td>
        <td>eps需要大于0</td>
        <td>DOUBLE</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>activateSilu</td>
        <td>输入</td>
        <td>是否开启silu计算。</td>
        <td>当前仅支持开启。</td>
        <td>BOOL</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>out</td>
        <td>输出</td>
        <td>量化后的结果，公式中的out。</td>
        <td>-</td>
        <td>INT8</td>
        <td>ND</td>
        <td>与self一致</td>
        <td>x</td>
    </tr>
    <tr>
        <td>meanOut</td>
        <td>输出</td>
        <td>公式中的meanOut。</td>
        <td>数据类型与self保持一致，shape中N与self的第0维度保持一致。</td>
        <td>FLOAT16、BFLOAT16</td>
        <td>ND</td>
        <td>(N, group)</td>
        <td>x</td>
    </tr>
    <tr>
        <td>rstdOut</td>
        <td>输出</td>
        <td>公式中的rstdOut。</td>
        <td>数据类型与self保持一致，shape中N与self的第0维度保持一致。</td>
        <td>FLOAT16、BFLOAT16</td>
        <td>ND</td>
        <td>(N, group)</td>  
        <td>x</td>
    </tr>
    <tr>
        <td>workspaceSize</td>
        <td>输出</td>
        <td>返回需要在Device侧申请的workspace大小。</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>executor</td>
        <td>输出</td>
        <td>返回op执行器，包含了算子计算流程。</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    </tbody></table>

## 约束说明

无

## 调用说明

| 调用方式   | 样例代码           | 说明                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn接口  | [test_aclnn_group_norm_silu_quant.cpp](examples/test_aclnn_group_norm_silu_quant.cpp) | 通过[aclnnGroupNormSiluQuant.md](docs/aclnnGroupNormSiluQuant.md)接口方式调用GroupNormSiluQuant算子。 |
# EluGradV2

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR/Ascend 950DT AI处理器</term>                             |    √     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    √     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> |    √     |



## 功能说明

- 算子功能：[aclnnElu](../../elu/docs/aclnnElu&aclnnInplaceElu.md)激活函数的反向计算，输出ELU激活函数正向输入的梯度。

- 计算公式：$x$是selfOrResult中的某个元素。

    - 当isResult是True时：

      $$
      gradInput = gradOutput *
      \begin{cases}
      scale, \quad x > 0\\
      inputScale \ast (x + \alpha \ast scale),  \quad x \leq 0
      \end{cases}
      $$

    - 当isResult是False时：

      $$
      gradInput = gradOutput *
      \begin{cases}
      scale, \quad x > 0\\
      inputScale \ast \alpha \ast scale \ast exp(x \ast inputScale), \quad x \leq 0
      \end{cases}
      $$


## 参数说明

  <table style="undefined;table-layout: fixed; width: 1370px"><colgroup>
  <col style="width: 171px">
  <col style="width: 115px">
  <col style="width: 220px">
  <col style="width: 200px">
  <col style="width: 177px">
  <col style="width: 104px">
  <col style="width: 238px">
  <col style="width: 145px">
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
      <td>gradOutput</td>
      <td>输入</td>
      <td>表示ELU激活函数正向输出的梯度，公式中的gradInput。</td>
      <td>支持空Tensor。</td>
      <td>FLOAT、FLOAT16、BFLOAT16</td>
      <td>ND</td>
      <td>0-8</td>
      <td>√</td>
    </tr>
      <tr>
      <td>alpha</td>
      <td>输入</td>
      <td>表示ELU激活函数的激活系数，公式中的\alpha。</td>
      <td><ul><li>如果isResult为true，\alpha必须大于等于0。</li><li>数据类型需要是可转换为FLOAT的数据类型（参见<a href="../../../docs/zh/context/互转换关系.md" target="_blank">互转换关系</a>）。</li></ul></td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
     <tr>
      <td>scale</td>
      <td>输入</td>
      <td>表示ELU激活函数的缩放系数，公式中的scale。</td>
      <td>数据类型需要是可转换为FLOAT的数据类型（参见<a href="../../../docs/zh/context/互转换关系.md" target="_blank">互转换关系</a>）。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
     <tr>
      <td>inputScale</td>
      <td>输入</td>
      <td>表示ELU激活函数的输入的缩放系数，公式中的inputScale。</td>
      <td>数据类型需要是可转换为FLOAT的数据类型（参见<a href="../../../docs/zh/context/互转换关系.md" target="_blank">互转换关系</a>）。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
      <tr>
      <td>isResult</td>
      <td>输入</td>
      <td>表示传给ELU反向计算的输入是否是ELU正向的输出。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
       <tr>
      <td>selfOrResult</td>
      <td>输入</td>
      <td><ul><li>当isResult为True时，表示ELU激活函数正向的输出。</li><li>当isResult为False时，表示ELU激活函数正向的输入。</li></ul></td>
      <td><ul><li>数据类型需要与gradOutput一致。</li><li>shape需要与gradOutput的shape一致。</li><li>支持空Tensor。</li></ul></td>
      <td>FLOAT、FLOAT16、BFLOAT16</td>
      <td>ND</td>
      <td>0-8</td>
      <td>√</td>
    </tr>
       <tr>
      <td>gradInput</td>
      <td>输出</td>
      <td>表示ELU激活函数正向输入的梯度，即对输入进行求导后的结果，公式中的gradOutput。</td>
      <td><ul><li>数据类型需要是gradOutput推导之后可转换的数据类型（参见<a href="../../../docs/zh/context/互转换关系.md" target="_blank">互转换关系</a>）。</li><li>shape需要和gradOutput的shape一致。</li></ul></td>
      <td>FLOAT、FLOAT16、BFLOAT16</td>
      <td>ND</td>
      <td>0-8</td>
      <td>√</td>
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
  </tbody>
  </table>

## 约束说明
- 确定性计算：
    - aclnnEluBackward默认确定性实现。

## 调用说明

| 调用方式   | 样例代码           | 说明                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn接口 | [test_aclnn_elu_backward](examples/arch35/test_aclnn_elu_backward.cpp) | 通过[aclnnEluBackward](docs/aclnnEluBackward.md)接口方式调用算子。 |
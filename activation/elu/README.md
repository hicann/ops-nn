# Elu

  ## 产品支持情况

  |产品             |  是否支持  |
  |:-------------------------|:----------:|
  |  <term>Ascend 950PR/Ascend 950DT</term>   |     √    |
  |  <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>   |     √    |
  |  <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>     |     √    |

  ## 功能说明

  - 算子功能: 对输入张量self中的每个元素x调用指数线性单元激活函数ELU，并将得到的结果存入输出张量out中。
  - 计算公式：

    $$
    ELU(x) =
    \begin{cases}
    scale \ast x, \quad x > 0\\
    \alpha \ast scale \ast (exp(x \ast inputScale)-1), \quad x \leq 0
    \end{cases}
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
        <td>待进行elu计算的入参，公式中的x。</td>
        <td>FLOAT、FLOAT16、BFLOAT16</td>
        <td>ND</td>
      </tr>
      <tr>
        <td>alpha</td>
        <td>输入</td>
        <td>待进行elu计算的入参，公式中的α。</td>
        <td>FLOAT</td>
        <td>ND</td>
      </tr>
      <tr>
        <td>scale</td>
        <td>输入</td>
        <td>待进行elu计算的入参，公式中的scale。</td>
        <td>FLOAT</td>
        <td>ND</td>
      </tr>
      <tr>
        <td>input_scale</td>
        <td>输入</td>
        <td>待进行elu计算的入参，公式中的inputScale。</td>
        <td>FLOAT</td>
        <td>ND</td>
      </tr>
      <tr>
        <td>y</td>
        <td>输出</td>
        <td>待进行elu计算的出参，公式中的out。</td>
        <td>FLOAT、FLOAT16、BFLOAT16</td>
        <td>ND</td>
      </tr>
    </tbody></table>

  ## 约束说明

  无

  ## 调用说明

  | 调用方式 | 调用样例                                                                   | 说明                                                           |
  |--------------|------------------------------------------------------------------------|--------------------------------------------------------------|
  | aclnn调用 | [test_aclnn_elu](./examples/test_aclnn_elu.cpp) | 通过[aclnnElu&aclnnInplaceElu](./docs/aclnnElu&aclnnInplaceElu.md)接口方式调用Elu算子。 |
  | aclnn调用 | [test_aclnn_inplace_elu](./examples/test_aclnn_inplace_elu.cpp) | 通过[aclnnElu&aclnnInplaceElu](./docs/aclnnElu&aclnnInplaceElu.md)接口方式调用Elu算子。 |
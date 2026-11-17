# SituGluGroup

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------- | ------|
| <term>Ascend 950PR&950DT系列产品</term>                             |    √     |
| <term>Atlas A3系列产品</term>     |    ×     |
| <term>Atlas A2系列产品</term> |    ×     |
| <term>Atlas 200I/500 A2推理产品</term>                      |    ×    |
| <term>Atlas推理系列产品</term>                             |    ×     |
| <term>Atlas训练系列产品</term>                              |    ×    |

## 功能说明

- 算子功能：SiTU门控线性单元（SiTU Gated Linear Unit）激活函数。对输入张量x沿指定维度切分为门控（gate）与上路径（up）两半，按SiTU公式计算输出。

- 计算公式或计算流程：

  对给定的输入张量x，其维度为[a, b, c, d, e, f, g, ...]，算子SituGluGroup对其进行以下计算：

  1. 将x基于输入参数dim进行合轴，合轴后维度为[pre, cut]，其中cut必须为偶数。令h = cut // 2。

  2. 根据输入参数activate_left对x进行前后切分：

     - activate_left为true时（默认）：

     $$
     gate = x[:, :h], \quad up = x[:, h:]
     $$

     - activate_left为false时：

     $$
     gate = x[:, h:], \quad up = x[:, :h]
     $$

  3. 根据输入参数beta、linear_beta进行SiTU计算：

     $$
     situ\_a = \beta \cdot \tanh\left(\frac{gate}{\beta}\right) \cdot \text{sigmoid}(gate)
     $$

     当linear_beta > 0时：

     $$
     up = linear\_beta \cdot \tanh\left(\frac{up}{linear\_beta}\right)
     $$

  4. 输出：

     $$
     y = situ\_a \cdot up
     $$

  5. 可选：当传入weight时，对每一行（dim轴之前各维的索引）逐行乘对应标量：

     $$
     y[i] = y[i] \cdot weight[i]
     $$

  6. 可选：当传入group_index时，仅前realBs行参与计算并写出，realBs = min(sum(group_index), 行数)。

  7. 重塑输出张量y的维度数量与合轴前的x的维度数量一致，dim轴上的大小为x的一半，其他维度与x相同。

## 参数说明

<table style="undefined;table-layout: fixed; width: 1576px"><colgroup>
  <col style="width: 170px">
  <col style="width: 170px">
  <col style="width: 200px">
  <col style="width: 200px">
  <col style="width: 170px">
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
      <td>公式中的输入x，dim维度会被均分为门控（gate）与上路径（up）两部分。</td>
      <td>FLOAT、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>dim</td>
      <td>属性</td>
      <td><ul><li>切分维度。</li><li>取值范围为[-x.dim(), x.dim()-1]。</li><li>默认值为-1。</li></ul></td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>beta</td>
      <td>属性</td>
      <td><ul><li>SiTU门控部分的缩放系数，控制tanh非线性强度。</li><li>必须为非零值，传入0会返回参数校验失败。</li><li>默认值为1.0。</li></ul></td>
      <td>FLOAT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>linear_beta</td>
      <td>属性</td>
      <td><ul><li>up路径线性tanh的缩放系数。</li><li>大于0时对up施加有界化变换，小于等于0时up直接透传。</li><li>默认值为0.0。</li></ul></td>
      <td>FLOAT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>activate_left</td>
      <td>属性</td>
      <td><ul><li>切分x时gate是否为前半部分。</li><li>true表示gate为前半、up为后半；false表示gate为后半、up为前半。</li><li>默认值为true。</li></ul></td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
    <tr>
      <td>weight</td>
      <td>可选输入</td>
      <td><ul><li>逐行权重标量，乘到SiTU激活结果上。</li><li>元素个数等于x在dim轴之前各维的乘积；dim为-1时等于除最后一维外各维的乘积。</li></ul></td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>group_index</td>
      <td>可选输入</td>
      <td><ul><li>count模式的分组token数。1维非空Tensor，元素值必须大于等于0，且sum(group_index)必须大于0。</li><li>传入时仅前min(sum(group_index), 行数)行参与计算并写出，其余行不写出（内容未定义），调用方需自行掩码。</li></ul></td>
      <td>INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y</td>
      <td>输出</td>
      <td>公式中的y，SiTU激活结果，dim维度大小为x的一半。</td>
      <td>FLOAT、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## 约束说明

- 支持非连续Tensor。
- weight与group_index为可选输入。
- group_index传入时，元素值必须大于等于0且sum(group_index)必须大于0；仅前realBs行写出，其余行内容未定义，调用方需自行掩码。
- beta必须为非零值，传入0会返回参数校验失败。

## 调用说明

| 调用方式   | 样例代码           | 说明                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn API  | [test_aclnn_situ_glu_group](./examples/test_aclnn_situ_glu_group.cpp) | 通过[aclnnSituGluGroup](./docs/aclnnSituGluGroup.md)接口调用SituGluGroup算子（含可选输入weight/group_index）。 |
| GE图模式 | -  | 通过[算子IR](./op_graph/situ_glu_group_proto.h)构图方式调用SituGluGroup算子。         |
| PyTorch API |  -  | 通过[situ_glu_group](./docs/torchapi_situ_glu_group.md)接口调用SituGluGroup算子。         |

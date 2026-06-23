# GroupNorm

## 产品支持情况

|产品             |  是否支持  |
|:-------------------------|:----------:|
|  <term>Ascend 950PR/Ascend 950DT</term>   |     ×    |
|  <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>   |     √    |
|  <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>     |     √    |
|  <term>Atlas 200I/500 A2 推理产品</term>    |     ×    |
|  <term>Atlas 推理系列产品</term>    |     ×    |
|  <term>Atlas 训练系列产品</term>    |     ×    |

## 功能说明

- 接口功能: 对输入张量self按group维度进行归一化计算，并可选使用gamma和beta对归一化结果做仿射变换，将结果写入out，同时输出每个样本每个group的均值meanOut和倒标准差rstdOut。
- 计算公式：

  $$
  mean_{n,g} = \frac{1}{m}\sum_{i \in group(g)} x_i
  $$

  $$
  rstd_{n,g} = \frac{1}{\sqrt{\frac{1}{m}\sum_{i \in group(g)}(x_i - mean_{n,g})^2 + eps}}
  $$

  $$
  y_i = (x_i - mean_{n,g}) \times rstd_{n,g} \times gamma_c + beta_c
  $$

  当gamma或beta为空时，分别按gamma为1、beta为0处理。

- **参数说明：**

  <table style="undefined;table-layout: fixed; width: 1300px"><colgroup>
  <col style="width: 101px">
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
      <td>self</td>
      <td>输入</td>
      <td>表示GroupNorm的输入张量，公式中的x。</td>
      <td>shape需满足rank为2-8，且第0维等于N、第1维等于C；第2维及之后元素个数乘积等于HxW。</td>
      <td>FLOAT、FLOAT16、BFLOAT16</td>
      <td>ND</td>
      <td>2-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>gamma</td>
      <td>可选输入</td>
      <td>表示GroupNorm仿射变换的缩放系数，公式中的gamma。</td>
      <td>可为空；不为空时shape需为[C]，数据类型需与self一致。</td>
      <td>FLOAT、FLOAT16、BFLOAT16</td>
      <td>ND</td>
      <td>1</td>
      <td>√</td>
    </tr>
    <tr>
      <td>beta</td>
      <td>可选输入</td>
      <td>表示GroupNorm仿射变换的偏置系数，公式中的beta。</td>
      <td>可为空；不为空时shape需为[C]，数据类型需与self一致。</td>
      <td>FLOAT、FLOAT16、BFLOAT16</td>
      <td>ND</td>
      <td>1</td>
      <td>√</td>
    </tr>
    <tr>
      <td>N</td>
      <td>输入</td>
      <td>表示输入张量的batch维大小。</td>
      <td>需等于self.shape[0]。</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>C</td>
      <td>输入</td>
      <td>表示输入张量的channel维大小。</td>
      <td>需等于self.shape[1]，且能被group整除。</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>HxW</td>
      <td>输入</td>
      <td>表示输入张量从第2维开始的元素个数乘积。</td>
      <td>需等于self.shape[2] * ... * self.shape[rank-1]。</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>group</td>
      <td>输入</td>
      <td>表示GroupNorm的分组数。</td>
      <td>需大于0，且C % group == 0。</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>eps</td>
      <td>输入</td>
      <td>表示归一化计算中加到方差上的数值稳定项。</td>
      <td>需大于0。</td>
      <td>DOUBLE</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>out</td>
      <td>输出</td>
      <td>表示GroupNorm归一化并仿射变换后的输出。</td>
      <td>shape和数据类型需与self一致。</td>
      <td>FLOAT、FLOAT16、BFLOAT16</td>
      <td>ND</td>
      <td>2-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>meanOut</td>
      <td>输出</td>
      <td>表示每个样本每个group的均值。</td>
      <td>shape需为[N, group]，数据类型需与self一致。</td>
      <td>FLOAT、FLOAT16、BFLOAT16</td>
      <td>ND</td>
      <td>2</td>
      <td>√</td>
    </tr>
    <tr>
      <td>rstdOut</td>
      <td>输出</td>
      <td>表示每个样本每个group的倒标准差。</td>
      <td>shape需为[N, group]，数据类型需与self一致。</td>
      <td>FLOAT、FLOAT16、BFLOAT16</td>
      <td>ND</td>
      <td>2</td>
      <td>√</td>
    </tr>
  </tbody>
  </table>

## 约束说明

- self的rank需为2-8，且N、C、HxW需与self的实际shape一致。
- group需大于0，且C需能被group整除。
- eps需大于0。
- gamma、beta为可选输入；不为空时shape均需为[C]，数据类型需与self一致。
- out、meanOut、rstdOut为必选输出，数据类型需与self一致；out的shape需与self一致，meanOut和rstdOut的shape需为[N, group]。
- op_host tiling路径不支持N、C或HxW为0的输入。

## 调用说明

| 调用方式 | 调用样例                                                                   | 说明                                                           |
|--------------|------------------------------------------------------------------------|--------------------------------------------------------------|
| aclnn调用 | [test_aclnn_group_norm](./tests/ut/op_api/test_aclnn_group_norm.cpp) | 通过aclnnGroupNormGetWorkspaceSize和aclnnGroupNorm接口方式调用GroupNorm算子。 |

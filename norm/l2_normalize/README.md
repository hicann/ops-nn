# L2Normalize

## 产品支持情况

| 产品 | 是否支持 |
| :--- | :---: |
| <term>Ascend 950PR&950DT系列产品</term> | √ |
| <term>Atlas A3系列产品</term> | √ |
| <term>Atlas A2系列产品</term> | √ |
| <term>Atlas 200I/500 A2推理产品</term> | √ |
| <term>Atlas推理系列产品</term> | √ |
| <term>Atlas训练系列产品</term> | √ |

## 功能说明

- 算子功能：对输入张量`x`沿`axis`指定的轴列表做L2归一化，输出`y`与`x`同shape、同dtype。
- 计算公式：

  $$
  s = \sum_{i \in axis} x_i^2 \quad (keepdims)
  $$

  $$
  y = x / \sqrt{\max(s, eps)}
  $$

  其中`eps`为平方和的下限，先max后sqrt；当`eps > 0`时可避免全零/极小向量除零。
  公开契约不限制`eps`的符号，`eps <= 0`时按上述公式直接计算；`eps`为非有限值（`inf`/`nan`）时行为不定义，调用方应传入有限值。

## 参数说明

<table style="table-layout: fixed; width: 1005px"><colgroup>
  <col style="width: 170px">
  <col style="width: 170px">
  <col style="width: 352px">
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
      <td>x</td>
      <td>输入</td>
      <td>表示被归一化的输入特征张量，即公式中的`x`。</td>
      <td>FLOAT16、FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y</td>
      <td>输出</td>
      <td>表示L2归一化结果张量，即公式中的`y`，shape和数据类型与`x`一致。</td>
      <td>FLOAT16、FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>axis</td>
      <td>可选属性</td>
      <td>表示归一化轴列表，即公式中的`axis`，支持多轴联合归约与负索引折算，默认值为`{}`；省略或传入空列表时不做跨元素归约，逐元素计算。</td>
      <td>LIST_INT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>eps</td>
      <td>可选属性</td>
      <td>表示平方和的下限，即公式中的`eps`，默认值为1e-4。</td>
      <td>FLOAT</td>
      <td>-</td>
    </tr>
  </tbody></table>

### 产品差异说明

各支持产品的公开GEIR接口保持一致：`x`和`y`支持FLOAT16、FLOAT32和ND格式，`y`与`x`的shape和数据类型一致，运行时实际rank为1-8；`axis`和`eps`的语义与默认值也一致。各产品的具体能力和契约边界如下。

| 产品 | 实现与调用通路 | 静态shape能力 | 动态shape能力 | shape/rank及扩展场景 |
| :--- | :--- | :--- | :--- | :--- |
| <term>Ascend 950PR&950DT系列产品</term> | 由本仓库的<term>Ascend 950PR&950DT系列产品</term>实现提供能力，通过GE图模式调用。 | FLOAT16、FLOAT32，ND→ND；支持1-8维具体shape，输出与输入同shape、同dtype。 | ND→ND；图推导支持未知维（`-1`）和未知rank（`{-2}`），执行前需解析为1-8维具体shape。 | 不支持0维标量；支持某维为0的空Tensor及非连续Tensor。 |
| <term>Atlas A3系列产品</term><br><term>Atlas A2系列产品</term><br><term>Atlas 200I/500 A2推理产品</term><br><term>Atlas推理系列产品</term><br><term>Atlas训练系列产品</term> | 由CANN存量实现提供能力，通过GE图模式调用。 | FLOAT16、FLOAT32，ND→ND；支持1-8维具体shape，输出与输入同shape、同dtype。 | ND→ND；CANN存量注册支持动态shape，运行时实际rank仍需满足1-8维的公开契约。 | 不支持0维标量；存量公开契约未明确承诺空Tensor、未知rank和非连续Tensor，本文档不将这些场景作为该类产品的对外能力。 |

## 约束说明

- `x`、`y`仅支持FLOAT16、FLOAT32（数据类型一致），数据格式仅支持ND，维度为1-8，`y.shape = x.shape`，不支持0维标量输入。
- `axis`省略或为空列表时不归约，此时逐元素计算`y = x / sqrt(max(x * x, eps))`；非空时每个元素的取值范围为`[-rank(x), rank(x))`，支持负索引折算，折算到同一维的重复项按一个归约轴处理。
- 在<term>Ascend 950PR&950DT系列产品</term>上，支持空Tensor（某维为0时输出为同维空Tensor）、动态shape（某维为`-1`）、未知维度数（shape的dims为`{-2}`，由图编译阶段推导实际rank）与非连续Tensor。
- 在<term>Ascend 950PR&950DT系列产品</term>上，本算子为非原地实现，执行后输入`x`保持不变；调用方应保证`y`与`x`底层内存不重叠。
- 在<term>Ascend 950PR&950DT系列产品</term>上，FLOAT16输入在算子内部提升为FLOAT32计算后回转FLOAT16输出。

## 调用说明

| 调用方式 | 调用样例 | 说明 | 支持产品 |
|---------|----|------|---------|
| GE图模式 | [test_geir_l2_normalize](examples/test_geir_l2_normalize.cpp) | 通过[算子IR](op_graph/l2_normalize_proto.h)构图方式调用L2Normalize算子。 | 全部支持产品 |

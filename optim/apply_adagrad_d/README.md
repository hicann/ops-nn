# ApplyAdagradD

## 产品支持情况

|产品             |  是否支持  |
|:-------------------------|:----------:|
|  <term>Ascend 950PR&950DT系列产品</term>   |     √    |
|  <term>Atlas A3系列产品</term>  |     √    |
|  <term>Atlas A2系列产品</term>     |     √    |
|  <term>Atlas 200I/500 A2推理产品</term>    |     ×    |
|  <term>Atlas推理系列产品</term>     |     √    |
|  <term>Atlas训练系列产品</term>    |     √    |

## 功能说明

- 算子功能：ApplyAdagradD 实现自适应梯度算法的核心更新操作，主要用于在优化器中更新模型参数。

计算公式：

$$
\begin{aligned}
accum &+= grad \times grad \quad (\text{if update\_slots=true})\\
var &= var - lr \times grad \times (1 / \sqrt{accum})
\end{aligned}
$$

## 参数说明

**约束说明**：
- 输入张量var、accum和grad必须具有相同的形状和数据类型。
- 输入张量lr必须是标量。

<table style="undefined;table-layout: fixed; width: 820px"><colgroup>
  <col style="width: 100px">
  <col style="width: 150px">
  <col style="width: 190px">
  <col style="width: 260px">
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
      <td>var</td>
      <td>输入</td>
      <td>可变张量。</td>
      <td>FLOAT、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>accum</td>
      <td>输入</td>
      <td>可变张量。与"var"具有相同类型。</td>
      <td>FLOAT、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>lr</td>
      <td>输入</td>
      <td>学习率标量。与"var"具有相同类型。</td>
      <td>FLOAT、FLOAT16、BFLOAT16</td>
      <td>标量</td>
    </tr>
    <tr>
      <td>grad</td>
      <td>输入</td>
      <td>梯度张量。与"var"具有相同类型。</td>
      <td>FLOAT、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>var</td>
      <td>输出</td>
      <td>可变张量。</td>
      <td>FLOAT、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>accum</td>
      <td>输出</td>
      <td>可变张量。与"var"具有相同类型。</td>
      <td>FLOAT、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>update_slots</td>
      <td>属性</td>
      <td>布尔值，默认为true。如果为true，则更新accum张量；否则accum输出等于accum输入。</td>
      <td>Bool</td>
      <td>-</td>
    </tr>
    <tr>
      <td>use_locking</td>
      <td>属性</td>
      <td>布尔值，默认为false。如果为true，则使用锁保护var和accum张量的更新；否则行为未定义，但可能减少竞争。</td>
      <td>Bool</td>
      <td>-</td>
    </tr>
  </tbody></table>

## 调用说明

| 调用方式   | 样例代码           | 说明                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| 图模式 | [test_geir_apply_adagrad_d.cpp](examples/test_geir_apply_adagrad_d.cpp)  | 通过[算子IR](op_graph/apply_adagrad_d_proto.h)构图方式调用ApplyAdagradD算子。         |

## 适用场景

- ApplyAdagradD适用于稀疏梯度场景，特别是在自然语言处理和推荐系统中。
- Adagrad优化器会为每个参数维护独立的学习率，适合处理稀疏数据。
- 由于累积平方梯度持续增长，学习率会单调递减，因此不适合长时间训练的场景。

## 精度与性能

- **精度**：算子支持FLOAT16、BFLOAT16和FLOAT32数据类型。在使用FLOAT16/BFLOAT16时，内部计算使用FLOAT32进行累加和除法运算以保持精度，最后再转换回原始类型输出。
- **性能**：算子基于AscendC ElementwiseSch调度框架实现，支持自动tiling和向量化计算，可充分利用NPU算力。实际性能取决于张量大小、数据类型和硬件平台。

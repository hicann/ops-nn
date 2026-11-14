# LpLoss

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

- 算子功能：计算输入predict和目标label中每个元素之间的平均绝对误差（Mean Absolute Error，简称MAE）。reduction指定要应用到输出的缩减，支持"none"、"mean"、"sum"。"none"表示不应用缩减，"mean"表示输出的总和将除以输出中的元素数，"sum"表示输出将被求和。

- 计算公式：

  当`reduction`为`none`时

  $$
  \ell(x, y) = L = \{l_1,\dots,l_N\}^\top, \quad
  l_n = \left| x_n - y_n \right|,
  $$

  其中$x$是predict，$y$是label，$N$是batch的大小。如果`reduction`不是`"none"`，那么

  $$
  \ell(x, y) =
  \begin{cases}
  \operatorname{mean}(L), &  \text{if reduction} = \text{'mean';}\\
  \operatorname{sum}(L),  &  \text{if reduction} = \text{'sum'.}
  \end{cases}
  $$

## 参数说明

本章描述算子（LpLoss）定义文件`op_host/lp_loss_def.cpp`中的参数，即GE图模式下的IR层口径。aclnn API（aclnnL1Loss）的参数口径与本表存在差异，详见[aclnnL1Loss](docs/aclnnL1Loss.md)与下文“约束说明”中的层级口径对应表。

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
      <td>predict</td>
      <td>输入</td>
      <td>公式中的输入$x$，即预测值。shape支持0-8维，支持动态shape（-1）与动态rank（-2）。</td>
      <td>FLOAT16、BFLOAT16、FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>label</td>
      <td>输入</td>
      <td>公式中的输入$y$，即真实标签。数据类型与shape必须与predict完全相同，IR层不做数据类型推导与broadcast。</td>
      <td>FLOAT16、BFLOAT16、FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>p</td>
      <td>属性</td>
      <td><ul><li>计算loss的阶数，当前仅支持p=1（即平均绝对误差）。</li><li>必选属性，无默认值。</li></ul></td>
      <td>INT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>reduction</td>
      <td>可选属性</td>
      <td><ul><li>指定要应用到输出的缩减，公式中的`reduction`，支持"none"、"mean"、"sum"。</li><li>默认值为"mean"。</li></ul></td>
      <td>String</td>
      <td>-</td>
    </tr>
    <tr>
      <td>y</td>
      <td>输出</td>
      <td>公式中的输出。数据类型与predict一致。reduction为"none"时shape与predict一致；reduction为"mean"或"sum"时为0维tensor（标量）。</td>
      <td>FLOAT16、BFLOAT16、FLOAT</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## 约束说明

- LpLoss中p为计算loss的阶数，只支持p=1，对应的aclnn API名称为aclnnL1Loss。

- 层级口径对应表：本算子存在GE图模式（IR/Kernel层）与aclnn直调（aclnnL1Loss）两条通路，两层的参数命名与数据类型域不同，使用时请勿混用。

<table style="undefined;table-layout: fixed; width: 980px"><colgroup>
  <col style="width: 160px">
  <col style="width: 200px">
  <col style="width: 300px">
  <col style="width: 320px">
  </colgroup>
  <thead>
    <tr>
      <th>对比项</th>
      <th>IR/Kernel层（LpLoss）</th>
      <th>aclnn层（aclnnL1Loss）</th>
      <th>说明</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>参数命名</td>
      <td>predict、label、y</td>
      <td>self、target、out</td>
      <td>同一参数的两套命名，一一对应。</td>
    </tr>
    <tr>
      <td>数据类型域</td>
      <td>FLOAT16、BFLOAT16、FLOAT</td>
      <td>推导后支持FLOAT、FLOAT16、INT64，以及BFLOAT16（按产品）</td>
      <td>INT64由aclnn层的小算子组合实现，不经LpLoss kernel；IR/Kernel层不支持INT64。两层均不支持COMPLEX64/COMPLEX128。</td>
    </tr>
    <tr>
      <td>输入间关系</td>
      <td>label的dtype与shape必须与predict完全相同</td>
      <td>self与target满足数据类型推导规则与broadcast规则</td>
      <td>推导与broadcast由aclnn层在调用kernel前完成。</td>
    </tr>
    <tr>
      <td>reduction类型</td>
      <td>String："none"/"mean"/"sum"，默认"mean"</td>
      <td>int64_t枚举：0('none')/1('mean')/2('sum')</td>
      <td>属性省略时，IR层按默认值"mean"生效。</td>
    </tr>
    <tr>
      <td>非连续Tensor</td>
      <td>不支持，输入须为连续ND tensor</td>
      <td>支持</td>
      <td>Kernel将由predict推导的访问视图同样套用于label，故IR/Kernel层要求两输入布局一致；aclnn层在调用kernel前已做连续化。</td>
    </tr>
  </tbody></table>

- 产品间数据类型差异：
    - <term>Ascend 950PR&950DT系列产品</term>、<term>Atlas A3系列产品</term>、<term>Atlas A2系列产品</term>：IR/Kernel层支持FLOAT16、BFLOAT16、FLOAT；aclnn层推导后支持FLOAT、FLOAT16、BFLOAT16、INT64。
    - <term>Atlas推理系列产品</term>、<term>Atlas训练系列产品</term>：aclnn层推导后支持FLOAT、FLOAT16、INT64，**不支持BFLOAT16**（传入BFLOAT16将返回161002）。

- 数据类型与shape的拒绝义务：上表声明的数据类型域、数据格式域与维度范围为算子的完整支持范围。超出声明范围的输入（未枚举的数据类型、非ND格式、维度大于8）应被拒绝并返回错误；调用方不应依赖未声明组合的行为。

- 维度范围：predict与label的shape支持0-8维（含0维标量）。

- 空Tensor行为（输入含0元素时）：
    - reduction为"none"（aclnn层为0）：输出空tensor。
    - reduction为"mean"（aclnn层为1）：输出NaN。注意该结果与“平均值”的直觉不同，源于元素数为0时的除零语义。
    - reduction为"sum"（aclnn层为2）：输出0。

- 输入输出重叠：不支持输入与输出的内存重叠（别名）。predict、label与y须为彼此独立的内存，算子不提供原地（in-place）计算语义，也不对重叠做检测；传入重叠内存的行为未定义。

- 动态shape：算子注册声明支持动态shape与动态rank，shape中-1表示该维度大小未知、-2表示rank未知，编译期按动态分支处理。

## 调用说明

| 调用方式   | 样例代码           | 说明                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn API  | [test_aclnn_l1_loss.cpp](examples/test_aclnn_l1_loss.cpp) | 通过[aclnnL1Loss](docs/aclnnL1Loss.md)接口方式调用LpLoss算子。 |
| GE图模式  | - | 通过[算子IR](op_graph/lp_loss_proto.h)构图方式调用LpLoss算子。 |

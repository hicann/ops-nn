# aclnnUnionSparseGen

简体中文 | [English](./aclnnUnionSparseGen_en.md)

## 产品支持情况

<!-- npu="950" id1 -->
- <term>Ascend 950PR&950DT系列产品</term>：支持
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3系列产品</term>：支持
<!-- end id2 -->
<!-- npu="910b" id3 -->
- <term>Atlas A2系列产品</term>：支持
<!-- end id3 -->
<!-- npu="310b" id4 -->
- <term>Atlas 200I/500 A2推理产品</term>：不支持
<!-- end id4 -->
<!-- npu="310p" id5 -->
- <term>Atlas推理系列产品</term>：不支持
<!-- end id5 -->
<!-- npu="910" id6 -->
- <term>Atlas训练系列产品</term>：不支持
<!-- end id6 -->

## 功能说明

将同一请求内最多`unionSize`行query显式选中的KV块合成一份升序去重列表，同时输出并集块数，以及逐KV token记录query选择关系的位掩码。算子不执行attention计算；residualBlockMode=1 在并集前补入不完整可见尾块。

设输入`sparseBlockIdx`的shape为`[H,T,K]`。H为稀疏head数，T为Q存储行总数，K为每行块编号容量；B为请求数，$W=\text{unionSize}$，$Y=\text{blockShape}[1]$。请求b的有效Q/KV长度分别为$Q_b$、$KV_b$，由长度输入按约束说明中的优先级确定。

**1. 按请求分组。** 每个请求单独按W行分组，不跨请求。请求b的第j组包含$n=\min(W,Q_b-jW)$个有效query，组内第i行在请求内的位置为$q_i=jW+i$。令$A_b$为该请求的Q存储起点，则输入行号为$r_i=A_b+q_i$。

$$
G_b=\left\lceil\frac{Q_b}{W}\right\rceil,\qquad
F_b=\sum_{a=0}^{b-1}G_a,\qquad
g=F_b+j,\qquad G=\sum_{b=0}^{B-1}G_b
$$

其中$0\le j<G_b$、$0\le i<n$；g是第二输出维中的全局组号。Q有效长度为0的请求不产生输出组。

**2. 计算合法块并集。** 令$L_b=\lceil KV_b/Y\rceil$，并按`sparseBlockCount`读取每行的有效前缀。只将$0\le d<L_b$的逻辑块编号纳入集合，重复编号只保留一次：

$$
c_i=\min\bigl(K,\max(0,\text{sparseBlockCount}[h,r_i])\bigr)
$$

$$
S_i=\{\,d=\text{sparseBlockIdx}[h,r_i,k]\mid 0\le k<c_i,\ 0\le d<L_b\,\}
$$

residual=0 保留合法显式块。residual=1 按 query 选择块：显式块仅保留完整位于可见范围内的块（d < floor(max(V,0)/Y)），并在 V>0 且 V%Y!=0 时加入不完整尾块 floor(V/Y)，然后合并去重；尾块同时参与 unionSparseIdx、unionSparseCount 和 exactMask。maskType=1 时 V=KV-Q+query局部位置+1，maskType=0 时 V 为该请求实际有效 KV 长度。完整尾块不自动补入，已选择的尾块不会重复计数。 Q=3、KV=5、Y=4、空显式列表时，非因果 residual=1 输出块 [1]、count=1、有效掩码 [7,0,0,0]；因果 residual=1 输出块 [0,1]、count=2、有效掩码 [1,1,1,0,4,0,0,0]。

$$
D=\bigcup_{i=0}^{n-1}\widehat S_i,\qquad
(d_0,\ldots,d_{C-1})=\operatorname{sort}(D),\qquad C=|D|
$$

$\text{unionSparseCount}[h,g]=C$，$\text{unionSparseIdx}[h,g,r]=d_r$（$0\le r<C$）。块编号相对于请求自身的KV起点，不是全局token位置。

**3. 保留每个query的选择关系。** 一个INT32中的第i位对应组内query i。对同一块，将选择它的query位按位或；这称为该块的归属位（membership）：

令 $\widehat S_i$ 为按上述 residual 规则得到的 query 选择集合，输出并集 $D=\bigcup_i\widehat S_i$；掩码保留各 query 对这些块的选择关系。

$$
\operatorname{membership}(d)=\bigvee_{i=0}^{n-1}\left(2^i[d\in \widehat S_i]\right)
$$

方括号表示条件成立取1，否则取0；$\bigvee$表示按位或。重复选择不会增加权重，也不会改变其他query的位。

**4. 展开token掩码。** 对输出块$d_r$中的第t个token（$0\le t<Y$），其请求内KV位置为$p=Yd_r+t$。两种mask分别为：

$$
C_0(i,p)=1,\qquad
C_1(i,p)=[p<KV_b-Q_b+q_i+1]
$$

$$
\text{exactMask}[h,g,Yr+t]
=\bigvee_{i=0}^{n-1}
\left(2^i[d_r\in \widehat S_i][0\le p<KV_b]C_m(i,p)\right),\qquad m=\text{maskType}
$$

`maskType=0`保留query选择关系与有效KV边界，不是把mask全置为1。`maskType=1`再应用右下对齐因果约束：请求内query i可见的KV token数为$KV_b-Q_b+q_i+1$；该值不为正时，没有可见token。

residual=0 时合法显式块即使因果掩码全零也保留；residual=1 按上述可见范围筛选显式块并补尾，因此 maskType 和 residual 可以影响并集及 count。

**5. 输出形状与容量。**

图模式的 InferShape 只读取 Host 上的 shape 和属性，不访问长度 Tensor 数据，也不执行 Host/Device 同步。令输入为 `[H,T,K]`（即 `[kvN,totalqBlockNum,maxKvBlockNum]`），输出为 `[H,G,U]`、`[H,G]`、`[H,G,U*Y]`：

- `totalq_union_num>0` 时 `G=totalq_union_num`；否则 `G=(T+7*B)//8`。
- `max_logical_blocks>0` 时 `U=max_logical_blocks`；否则 `U=(K+residual)*union_size`，不再按 KV 逻辑域裁剪容量。
- B 由长度 Tensor 的 shape 推导；定长 Q 使用 `ceil(T/ceil(maxQSeqlen/block_shape[0]))`。当前 blockX=1，允许最后一槽不足 maxQ；maxQ=0 时只允许 T=0，并沿用空 batch/已传 KV 长度 Tensor 的 B 规则。
- 实际有效组数仍为 `sum(ceil(effectiveQ/8))`。额外的输出组填 `idx=-1,count=0,mask=0`。调用方必须保证 G 覆盖全部实际有效组；G 不足不保证报错，可能导致部分有效组未被处理。
- U 必须覆盖每组实际并集。kernel 检查到实际并集超过 U 后触发设备错误，而不是静默截断该并集。错误可能在执行或流同步时暴露，不能保证在 `aclnnUnionSparseGenGetWorkspaceSize` 阶段拒绝。该检查不校验 G，也不代表所有输出容量不足都会自动报错。
- 默认 `U=(K+residual)*union_size` 在 residual=1 时为每个 query 额外预留一个尾块候选；maskType=0 也沿用该安全上界。

`maxLogicalBlocks` 支持 `-1` 或 `[1, INT32_MAX]`：正值作为排序及成员表规划上限 L，须覆盖有效 KV 逻辑域；`-1` 时，变长 KV 使用 `INT32_MAX`，定长 KV 使用 `max(1, ceil(maxKvSeqlen/Y))`。Host 不回读 Device 长度。

`totalqUnionNum=-1` 使用输出 G，其他值必须等于 `unionSparseIdx.shape[1]`。ACLNN 按 `U=min(unionSize*(K+residualBlockMode),L)` 校验输出 `[H,G,U]`、`[H,G]`、`[H,G,U*Y]`，不匹配时在 GetWorkspaceSize 阶段报错，G=0 也执行校验。

U 是存储容量。非法块编号（负数或超出请求有效 KV 逻辑域）会被过滤。

Torch 适配层保持原有计算：`G=sum(ceil(effectiveQ/8))`，`L=max(1,max(ceil(effectiveKV/Y)))`，`U=min(8*(K+residual),L)`；它传入 G/L 并预分配该 U 的输出，ACLNN 按这些输出执行。Torch 的低层与 Meta 入口仍要求显式 G/L，尚未切换到上述 GE 自动形状规则。CPU 标杆仍计算精确有效输出；图模式测试只对其结果补齐到 G/U，并检查没有丢弃有效块。

## 函数原型

每个算子分为[两段式接口](../../../docs/zh/context/two_phase_api.md)，必须先调用“aclnnUnionSparseGenGetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnUnionSparseGen”接口执行计算。

```cpp
aclnnStatus aclnnUnionSparseGenGetWorkspaceSize(
  const aclTensor* sparseBlockIdx,
  const aclTensor* sparseBlockCount,
  const aclTensor* cuSqlenQOptional,
  const aclTensor* cuSqlenKvOptional,
  const aclTensor* seqUsedQOptional,
  const aclTensor* seqUsedKvOptional,
  const aclIntArray* blockShape,
  int64_t          unionSize,
  int64_t          maskType,
  int64_t          maxQSeqlen,
  int64_t          maxKvSeqlen,
  int64_t          layoutSparsePattern,
  bool             isInputIdxSorted,
  int64_t          residualBlockMode,
  int64_t          totalqUnionNum,
  int64_t          maxLogicalBlocks,
  aclTensor*       unionSparseIdx,
  aclTensor*       unionSparseCount,
  aclTensor*       exactMask,
  uint64_t*        workspaceSize,
  aclOpExecutor**  executor);
```

```cpp
aclnnStatus aclnnUnionSparseGen(
  void*          workspace,
  uint64_t       workspaceSize,
  aclOpExecutor* executor,
  aclrtStream    stream);
```

## aclnnUnionSparseGenGetWorkspaceSize

- **参数说明**：

  <table style="table-layout: fixed; width: 1550px">
  <colgroup>
    <col style="width: 180px">
    <col style="width: 120px">
    <col style="width: 280px">
    <col style="width: 320px">
    <col style="width: 250px">
    <col style="width: 120px">
    <col style="width: 140px">
    <col style="width: 140px">
  </colgroup>
  <thead>
    <tr>
      <th style="white-space: nowrap">参数名</th>
      <th>输入/输出</th>
      <th>描述</th>
      <th>使用说明</th>
      <th>数据类型</th>
      <th>数据格式</th>
      <th>维度(shape)</th>
      <th>非连续Tensor</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td style="white-space: nowrap">sparseBlockIdx（aclTensor*）</td>
      <td>输入</td>
      <td>每个query选中的逻辑KV块索引。</td>
      <td>逻辑块ID；每行有效前缀之外不参与计算。</td>
      <td>INT32</td>
      <td>ND</td>
      <td>[H,T,K]</td>
      <td>√</td>
    </tr>
    <tr>
      <td style="white-space: nowrap">sparseBlockCount（aclTensor*）</td>
      <td>输入</td>
      <td>每个query的有效索引数量。</td>
      <td>每行有效前缀长度，兼容按[0,K]夹取。</td>
      <td>INT32</td>
      <td>ND</td>
      <td>[H,T]</td>
      <td>√</td>
    </tr>
    <tr>
      <td style="white-space: nowrap">cuSqlenQOptional（aclTensor*）</td>
      <td>可选输入</td>
      <td>各请求query存储长度的前缀和；前缀模式条件必传。</td>
      <td>nullptr选择固定槽位，否则按Q前缀定位。</td>
      <td>INT64</td>
      <td>ND</td>
      <td>[B+1]</td>
      <td>√</td>
    </tr>
    <tr>
      <td style="white-space: nowrap">cuSqlenKvOptional（aclTensor*）</td>
      <td>可选输入</td>
      <td>各请求KV存储长度的前缀和。</td>
      <td>KV存储前缀；与Q侧模式独立。</td>
      <td>INT64</td>
      <td>ND</td>
      <td>[B+1]</td>
      <td>√</td>
    </tr>
    <tr>
      <td style="white-space: nowrap">seqUsedQOptional（aclTensor*）</td>
      <td>可选输入</td>
      <td>各请求的有效query长度。</td>
      <td>非前缀的有效Q长度；非空时优先于Q前缀差分。</td>
      <td>INT32</td>
      <td>ND</td>
      <td>[B]</td>
      <td>√</td>
    </tr>
    <tr>
      <td style="white-space: nowrap">seqUsedKvOptional（aclTensor*）</td>
      <td>可选输入</td>
      <td>各请求的有效KV长度。</td>
      <td>非前缀的有效KV长度；非空时优先于KV前缀差分。</td>
      <td>INT32</td>
      <td>ND</td>
      <td>[B]</td>
      <td>√</td>
    </tr>
    <tr>
      <td style="white-space: nowrap">blockShape（aclIntArray*）</td>
      <td>输入</td>
      <td>Q、KV块粒度组成的整数数组。</td>
      <td>元素顺序为[X,Y]，分别表示Q和KV块粒度。</td>
      <td>INT64</td>
      <td>-</td>
      <td>[2]</td>
      <td>-</td>
    </tr>
    <tr>
      <td style="white-space: nowrap">unionSize（int64_t）</td>
      <td>输入</td>
      <td>每个并集组包含的query行数，对应公式中的W。</td>
      <td>每个请求独立分组。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td style="white-space: nowrap">maskType（int64_t）</td>
      <td>输入</td>
      <td>掩码规则整数枚举。</td>
      <td>0表示非因果，1表示右下对齐causal；见计算公式。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td style="white-space: nowrap">maxQSeqlen（int64_t）</td>
      <td>输入</td>
      <td>无Q长度Tensor时的定长Q长度，否则忽略。</td>
      <td>语义及使用条件见约束说明。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td style="white-space: nowrap">maxKvSeqlen（int64_t）</td>
      <td>输入</td>
      <td>无KV长度Tensor时的定长KV长度，否则忽略。</td>
      <td>语义及使用条件见约束说明。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td style="white-space: nowrap">layoutSparsePattern（int64_t）</td>
      <td>输入</td>
      <td>稀疏模式tensor的布局。</td>
      <td>稀疏模式布局，见shape列。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td style="white-space: nowrap">isInputIdxSorted（bool）</td>
      <td>输入</td>
      <td>输入索引的排序状态。</td>
      <td>表示有效前缀按非降序排列。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td style="white-space: nowrap">residualBlockMode（int64_t）</td>
      <td>输入</td>
      <td>不完整KV尾块的处理方式。</td>
      <td>0使用显式选择；1在并集前补入不完整可见尾块，影响索引、count和掩码。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td style="white-space: nowrap">totalqUnionNum（int64_t）</td>
      <td>输入</td>
      <td>输出组数 G 的校验值。</td>
      <td>-1 使用输出 shape；其他值必须等于 unionSparseIdx.shape[1]。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td style="white-space: nowrap">maxLogicalBlocks（int64_t）</td>
      <td>输入</td>
      <td>排序及成员表资源规划的逻辑 KV 块上限 L。</td>
      <td>-1 自动推导；正值须在 [1, INT32_MAX] 内并覆盖有效 KV 逻辑域。U=min(unionSize*(K+residualBlockMode),L)。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td style="white-space: nowrap">unionSparseIdx（aclTensor*）</td>
      <td>输出</td>
      <td>各组的升序去重KV块索引。</td>
      <td>升序去重结果；未使用位置填-1。</td>
      <td>INT32</td>
      <td>ND</td>
      <td>[H,G,U]</td>
      <td>√</td>
    </tr>
    <tr>
      <td style="white-space: nowrap">unionSparseCount（aclTensor*）</td>
      <td>输出</td>
      <td>各组的实际并集块数。</td>
      <td>实际并集块数。</td>
      <td>INT32</td>
      <td>ND</td>
      <td>[H,G]</td>
      <td>√</td>
    </tr>
    <tr>
      <td style="white-space: nowrap">exactMask（aclTensor*）</td>
      <td>输出</td>
      <td>各并集块展开后的逐token查询位掩码，对应公式中的exactMask。</td>
      <td>bit i对应组内query i；未使用位置和缺失query位为0。</td>
      <td>INT32</td>
      <td>ND</td>
      <td>[H,G,U×Y]</td>
      <td>√</td>
    </tr>
    <tr>
      <td style="white-space: nowrap">workspaceSize（uint64_t*）</td>
      <td>输出</td>
      <td>返回需要在Device侧申请的workspace大小。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td style="white-space: nowrap">executor（aclOpExecutor**）</td>
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

- **返回值：**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn_return_code.md)。

  第一段接口完成入参校验，出现以下场景时报错：

  <table style="table-layout: fixed; width: 1100px">
  <colgroup>
    <col style="width: 300px">
    <col style="width: 150px">
    <col style="width: 650px">
  </colgroup>
  <thead>
    <tr>
      <th style="white-space: nowrap">返回值</th>
      <th>错误码</th>
      <th>描述</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td style="white-space: nowrap">ACLNN_ERR_PARAM_NULLPTR</td>
      <td>161001</td>
      <td>sparseBlockIdx、sparseBlockCount、unionSparseIdx、unionSparseCount、exactMask、blockShape、workspaceSize或executor存在空指针。</td>
    </tr>
    <tr>
      <td style="white-space: nowrap">ACLNN_ERR_PARAM_INVALID</td>
      <td>161002</td>
      <td>输入或输出的数据类型、数据格式、shape或属性取值不满足约束说明；长度输入组合、固定Q槽宽与T的关系、输出几何或INT64字节范围不合法。</td>
    </tr>
  </tbody>
  </table>

## aclnnUnionSparseGen

- **参数说明**：

  <table style="table-layout: fixed; width: 1100px">
  <colgroup>
    <col style="width: 200px">
    <col style="width: 130px">
    <col style="width: 770px">
  </colgroup>
  <thead>
    <tr>
      <th style="white-space: nowrap">参数名</th>
      <th>输入/输出</th>
      <th>描述</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td style="white-space: nowrap">workspace</td>
      <td>输入</td>
      <td>在Device侧申请的workspace内存地址。</td>
    </tr>
    <tr>
      <td style="white-space: nowrap">workspaceSize</td>
      <td>输入</td>
      <td>在Device侧申请的workspace大小，由第一段接口aclnnUnionSparseGenGetWorkspaceSize获取。</td>
    </tr>
    <tr>
      <td style="white-space: nowrap">executor</td>
      <td>输入</td>
      <td>op执行器，包含了算子计算流程。</td>
    </tr>
    <tr>
      <td style="white-space: nowrap">stream</td>
      <td>输入</td>
      <td>指定执行任务的Stream。</td>
    </tr>
  </tbody>
  </table>

- **返回值：**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn_return_code.md)。

## 约束说明

- 仅支持blockShape=[1,Y], Y∈{4,8,16,128}、unionSize=8、maskType∈{0,1}、layoutSparsePattern=4、isInputIdxSorted=true/false、residualBlockMode=0/1；不开放窗口或Union消费mask模式。blockShape非空且恰有两个元素。
- cu前缀为INT64，其余输入、输出Tensor为INT32，均位于同一Device，数据格式为ND。K>0，维度非负，输入输出的维度及INT32字节范围均须能由INT64表示。
- 支持非连续输入，ACLNN在计算前将其转连续；支持各元素存储位置不重叠的非连续输出，按输出的shape、stride和storageOffset回写。调用方须提供合法的stride及足够的底层存储。
- sparseBlockIdx为[H,T,K]，sparseBlockCount为[H,T]。isInputIdxSorted=true时每行有效前缀须非降序；false允许无序，两者均允许重复。块ID相对于请求的KV起点；负ID及超过该请求有效KV逻辑域的ID忽略。residual=0 保留合法显式块。residual=1 按 query 选择块：显式块仅保留完整位于可见范围内的块（d < floor(max(V,0)/Y)），并在 V>0 且 V%Y!=0 时加入不完整尾块 floor(V/Y)，然后合并去重；尾块同时参与 unionSparseIdx、unionSparseCount 和 exactMask。maskType=1 时 V=KV-Q+query局部位置+1，maskType=0 时 V 为该请求实际有效 KV 长度。完整尾块不自动补入，已选择的尾块不会重复计数。
- cuSqlenQOptional非空时为[B+1]，从0开始、非降序、末项为T；请求存储起点为cuQ[b]，Q容量为相邻差分。seqUsedQOptional非空时为[B]，其值须在对应Q容量内。

Q、KV 独立选择长度来源：seqUsed 优先，其次为 cu 相邻差分；对应 cu、seqUsed 任意一个非空时，max 的任意 int64 值均被忽略。两者均为空时才使用 max 作为每个请求的定长有效长度，要求 max≥0。

| cuX | seqUsedX | 有效长度 | maxX |
|---|---|---|---|
| 传入 | 传入 | seqUsedX[b]，不得超过 cuX 差分容量 | 完全忽略 |
| 传入 | nullptr | cuX[b+1]−cuX[b] | 完全忽略 |
| nullptr | 传入 | seqUsedX[b] | 完全忽略 |
| nullptr | nullptr | Q=min(maxQ,剩余Q行)，KV=maxKV | 定长槽宽/长度，必须非负 |

Q 有 cu 时通过前缀定位。Q 只有 seqUsed 时，B 由其长度确定，要求 T%B=0，slot=T/B，请求 b 从 b×slot 开始，seqUsedQ[b]≤slot。Q 两个长度 Tensor 都为空时，slot=maxQ，B=ceil(T/slot)，请求 b 从 b×slot 开始，有效Q=min(slot,T-b×slot)，允许最后一槽不足 slot。KV 只有 seqUsed 时不受 Q slot 限制。所有已传长度 Tensor 必须具有相同 B；cu 形状为 [B+1]，seqUsed 为 [B]。四个 Tensor 均为空时，正 maxQ 通过 ceil(T/maxQ) 确定 B，最后不足一槽也计为一个 batch；maxQ=0 时必须 T=0，并按空 batch 处理。有 KV 长度 Tensor 时可确定非空 B，允许 maxQ=0、T=0。B=0 时 T=0、实际组数为0；显式提供额外输出组时全部填充为空。定长 KV 要求 ceil(maxKV/Y)≤INT32_MAX；不要求 KV≥Q。默认 max=-1 仅在被忽略时合法。

- ACLNN 的 G、L 和 U 按上述规则确定，支持 G=0 空输出；空输出也校验属性与 shape。
- 容量由公式自动确定，不能静默截断。residual=0 时residual=0 时两种mask类型下并集和count相同；尾部越界token及padding mask均为0。位掩码采用INT32位模式；本轮W8仅使用低8位，不开放W32。
- aclnnUnionSparseGen默认确定性实现。

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/compile_and_run_sample.md)。

<term>Ascend 950PR&950DT系列产品</term>、<term>Atlas A3系列产品</term>、<term>Atlas A2系列产品</term>：

```cpp
#include <cstdio>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_union_sparse_gen.h"

#define CHECK_RET(cond, return_expr) \
  do {                               \
    if (!(cond)) {                   \
      return_expr;                   \
    }                                \
  } while (0)

#define LOG_PRINT(message, ...)       \
  do {                                \
    printf(message, ##__VA_ARGS__);    \
  } while (0)

int64_t GetShapeSize(const std::vector<int64_t>& shape) {
  int64_t shapeSize = 1;
  for (auto i : shape) {
    shapeSize *= i;
  }
  return shapeSize;
}

int Init(int32_t deviceId, aclrtStream* stream) {
  auto ret = aclInit(nullptr);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
  ret = aclrtSetDevice(deviceId);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
  ret = aclrtCreateStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
  return 0;
}

template <typename T>
int CreateAclTensor(const std::vector<T>& hostData, const std::vector<int64_t>& shape,
                    void** deviceAddr, aclDataType dataType, aclTensor** tensor) {
  auto size = GetShapeSize(shape) * sizeof(T);
  auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);

  ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

  std::vector<int64_t> strides(shape.size(), 1);
  for (int64_t i = static_cast<int64_t>(shape.size()) - 2; i >= 0; i--) {
    strides[i] = shape[i + 1] * strides[i + 1];
  }
  *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0,
                           ACL_FORMAT_ND, shape.data(), shape.size(), *deviceAddr);
  CHECK_RET(*tensor != nullptr, LOG_PRINT("aclCreateTensor failed.\n"); return 1);
  return 0;
}

int main(int argc, char**) {
  // 无参数时按cu前缀定位；传任意参数（例如 ./example fixed）时运行固定槽位调用。
  const bool fixedSlots = argc > 1;
  // 1. device/stream初始化。
  int32_t deviceId = 0;
  aclrtStream stream = nullptr;
  auto ret = Init(deviceId, &stream);
  CHECK_RET(ret == 0, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  // 2. 构造输入与输出：H=1，Q=2，KV=128，K=1，Y=128，W=8，G=1，Lmax=1，U=1。
  std::vector<int64_t> sparseShape = {1, 2, 1};
  std::vector<int64_t> countShape = {1, 2};
  std::vector<int64_t> lengthShape = {2};
  std::vector<int64_t> usedShape = {1};
  std::vector<int64_t> unionShape = {1, 1, 1};
  std::vector<int64_t> unionCountShape = {1, 1};
  std::vector<int64_t> maskShape = {1, 1, 128};
  std::vector<int32_t> sparseHostData = {0, 0};
  std::vector<int32_t> countHostData = {1, 1};
  std::vector<int64_t> cuQHostData = {0, 2};
  std::vector<int64_t> cuKvHostData = {0, 128};
  std::vector<int32_t> seqQHostData = {2};
  std::vector<int32_t> seqKvHostData = {128};
  std::vector<int32_t> unionHostData(1, 0);
  std::vector<int32_t> unionCountHostData(1, 0);
  std::vector<int32_t> maskHostData(128, 0);

  void* sparseDeviceAddr = nullptr;
  void* countDeviceAddr = nullptr;
  void* cuQDeviceAddr = nullptr;
  void* cuKvDeviceAddr = nullptr;
  void* seqQDeviceAddr = nullptr;
  void* seqKvDeviceAddr = nullptr;
  void* unionDeviceAddr = nullptr;
  void* unionCountDeviceAddr = nullptr;
  void* maskDeviceAddr = nullptr;
  aclTensor* sparseBlockIdx = nullptr;
  aclTensor* sparseBlockCount = nullptr;
  aclTensor* cuSqlenQ = nullptr;
  aclTensor* cuSqlenKv = nullptr;
  aclTensor* seqUsedQ = nullptr;
  aclTensor* seqUsedKv = nullptr;
  aclTensor* unionSparseIdx = nullptr;
  aclTensor* unionSparseCount = nullptr;
  aclTensor* exactMask = nullptr;

  ret = CreateAclTensor(sparseHostData, sparseShape, &sparseDeviceAddr, ACL_INT32, &sparseBlockIdx);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(countHostData, countShape, &countDeviceAddr, ACL_INT32, &sparseBlockCount);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(cuQHostData, lengthShape, &cuQDeviceAddr, ACL_INT64, &cuSqlenQ);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(cuKvHostData, lengthShape, &cuKvDeviceAddr, ACL_INT64, &cuSqlenKv);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(seqQHostData, usedShape, &seqQDeviceAddr, ACL_INT32, &seqUsedQ);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(seqKvHostData, usedShape, &seqKvDeviceAddr, ACL_INT32, &seqUsedKv);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(unionHostData, unionShape, &unionDeviceAddr, ACL_INT32, &unionSparseIdx);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(unionCountHostData, unionCountShape, &unionCountDeviceAddr,
                        ACL_INT32, &unionSparseCount);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(maskHostData, maskShape, &maskDeviceAddr, ACL_INT32, &exactMask);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 3. 调用两段式接口；maskType=1表示causal。
  // 固定槽位不传cuQ/cuKV，使用seqUsedQ/seqUsedKv，Q槽宽由T/B确定，maxQSeqlen被忽略。
  // 前缀模式和固定槽位模式的有效Q/KV相同，三个输出也相同。
  const int64_t shapeValues[] = {1, 128};
  auto* blockShape = aclCreateIntArray(shapeValues, 2);
  CHECK_RET(blockShape != nullptr, return 1);
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor = nullptr;
  ret = aclnnUnionSparseGenGetWorkspaceSize(
      sparseBlockIdx, sparseBlockCount, fixedSlots ? nullptr : cuSqlenQ,
      fixedSlots ? nullptr : cuSqlenKv, seqUsedQ, seqUsedKv,
      blockShape, 8, int64_t{1}, fixedSlots ? 2 : -1, -1, 4, true, 0, 1, 1,
      unionSparseIdx, unionSparseCount, exactMask, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("aclnnUnionSparseGenGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);

  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
  }
  ret = aclnnUnionSparseGen(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnUnionSparseGen failed. ERROR: %d\n", ret); return ret);

  // 4. 同步执行结果并将输出拷贝到Host。
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
  aclDestroyIntArray(blockShape);
  ret = aclrtMemcpy(unionHostData.data(), sizeof(int32_t), unionDeviceAddr,
                    sizeof(int32_t), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = aclrtMemcpy(unionCountHostData.data(), sizeof(int32_t), unionCountDeviceAddr,
                    sizeof(int32_t), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = aclrtMemcpy(maskHostData.data(), maskHostData.size() * sizeof(int32_t), maskDeviceAddr,
                    maskHostData.size() * sizeof(int32_t), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  LOG_PRINT("unionSparseIdx=[%d], unionSparseCount=[%d]\n", unionHostData[0], unionCountHostData[0]);
  LOG_PRINT("exactMask first=%d, last=%d\n", maskHostData.front(), maskHostData.back());
  // 预期输出：unionSparseIdx=[0]，unionSparseCount=[1]，exactMask前127项为3，最后一项为2。

  bool correct = unionHostData[0] == 0 && unionCountHostData[0] == 1;
  for (size_t i = 0; i < maskHostData.size(); ++i) {
    correct = correct && maskHostData[i] == (i + 1 == maskHostData.size() ? 2 : 3);
  }

  // 5. 释放资源。
  aclDestroyTensor(sparseBlockIdx);
  aclDestroyTensor(sparseBlockCount);
  aclDestroyTensor(cuSqlenQ);
  aclDestroyTensor(cuSqlenKv);
  aclDestroyTensor(seqUsedQ);
  aclDestroyTensor(seqUsedKv);
  aclDestroyTensor(unionSparseIdx);
  aclDestroyTensor(unionSparseCount);
  aclDestroyTensor(exactMask);
  aclrtFree(sparseDeviceAddr);
  aclrtFree(countDeviceAddr);
  aclrtFree(cuQDeviceAddr);
  aclrtFree(cuKvDeviceAddr);
  aclrtFree(seqQDeviceAddr);
  aclrtFree(seqKvDeviceAddr);
  aclrtFree(unionDeviceAddr);
  aclrtFree(unionCountDeviceAddr);
  aclrtFree(maskDeviceAddr);
  if (workspaceAddr != nullptr) {
    aclrtFree(workspaceAddr);
  }
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();
  return correct ? 0 : 1;
}
```

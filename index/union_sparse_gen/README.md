# UnionSparseGen

简体中文 | [English](./README_en.md)

## 产品支持情况

| 产品 | 是否支持 |
|---|:---:|
| <term>Ascend 960PR&960DT系列产品</term> | × |
| <term>Ascend 950PR&950DT系列产品</term> | √ |
| <term>Atlas A3系列产品</term> | √ |
| <term>Atlas A2系列产品</term> | √ |
| <term>Atlas 200I/500 A2推理产品</term> | × |
| <term>Atlas推理系列产品</term> | × |
| <term>Atlas训练系列产品</term> | × |

## 功能说明

将同一请求内相邻query显式选择的逻辑KV块合并为升序去重并集，输出并集索引、块数及逐token位掩码，保留每个query原有的选择关系。

## 参数说明

下表使用算子原型中的参数名和Tensor类型。ACLNN的驼峰参数名、Torch公共入口的长度Tensor转换约定见各自接口文档。

| 参数名 | 输入/输出/属性 | 描述 | 数据类型 | 数据格式 |
|---|---|---|---|---|
| sparse_block_idx | 输入 | [H,T,K]，每行显式选择的逻辑KV块ID。 | INT32 | ND |
| sparse_block_count | 输入 | [H,T]，每行有效前缀长度。 | INT32 | ND |
| cu_sqlen_q | 可选输入 | [B+1]，Q存储位置前缀和。 | INT64 | ND |
| cu_sqlen_kv | 可选输入 | [B+1]，KV存储长度前缀和。 | INT64 | ND |
| seq_used_q | 可选输入 | [B]，各请求的有效Q长度。 | INT32 | ND |
| seq_used_kv | 可选输入 | [B]，各请求的有效KV长度。 | INT32 | ND |
| union_sparse_idx | 输出 | [H,G,U]，升序去重并集，padding为-1。 | INT32 | ND |
| union_sparse_count | 输出 | [H,G]，各组实际并集块数。 | INT32 | ND |
| exact_mask | 输出 | [H,G,U×Y]，逐token的query位掩码，padding为0。 | INT32 | ND |
| block_shape | 属性 | Q/KV块粒度，默认[1,128]，支持[1,Y]，Y∈{4,8,16,128}。 | ListInt | - |
| union_size | 属性 | 每组query数，默认且仅支持8。 | INT64 | - |
| mask_type | 属性 | 0为非因果，1为右下对齐causal，默认0。 | INT64 | - |
| max_q_seqlen | 属性 | 默认-1；有任意Q长度Tensor时忽略，否则为非负定长Q长度。 | INT64 | - |
| max_kv_seqlen | 属性 | 默认-1；有任意KV长度Tensor时忽略，否则为非负定长KV长度。 | INT64 | - |
| layout_sparse_pattern | 属性 | 稀疏模式布局，默认且仅支持4。 | INT64 | - |
| is_input_idx_sorted | 属性 | 有效前缀有序承诺，默认true；false允许无序输入。 | BOOL | - |
| residual_block_mode | 属性 | 默认0；1补入不完整可见尾块，参与索引、count及掩码。 | INT64 | - |
| totalq_union_num | 属性 | G 提示；tiling 中 -1 使用输出 G，其他值必须与输出 G 相等。 | INT64 | - |
| max_logical_blocks | 属性 | 排序/成员表上限 L：-1 自动推导，正值须在 [1, INT32_MAX] 内；图模式正值同时作为 U。 | INT64 | - |

## 约束说明

- 支持[1,Y]块（Y∈{4,8,16,128}）、8行分组、有序/无序输入、mask类型0/1与residual模式0/1。输入允许重复。

Q、KV 独立选择长度来源：seqUsed 优先，其次为 cu 相邻差分；对应 cu、seqUsed 任意一个非空时，max 的任意 int64 值均被忽略。两者均为空时才使用 max 作为每个请求的定长有效长度，要求 max≥0。

| cuX | seqUsedX | 有效长度 | maxX |
|---|---|---|---|
| 传入 | 传入 | seqUsedX[b]，不得超过 cuX 差分容量 | 完全忽略 |
| 传入 | nullptr | cuX[b+1]−cuX[b] | 完全忽略 |
| nullptr | 传入 | seqUsedX[b] | 完全忽略 |
| nullptr | nullptr | Q=min(maxQ,剩余Q行)，KV=maxKV | 定长槽宽/长度，必须非负 |

Q 有 cu 时通过前缀定位。Q 只有 seqUsed 时，B 由其长度确定，要求 T%B=0，slot=T/B，请求 b 从 b×slot 开始，seqUsedQ[b]≤slot。Q 两个长度 Tensor 都为空时，slot=maxQ，B=ceil(T/slot)，请求 b 从 b×slot 开始，有效Q=min(slot,T-b×slot)，允许最后一槽不足 slot。KV 只有 seqUsed 时不受 Q slot 限制。所有已传长度 Tensor 必须具有相同 B；cu 形状为 [B+1]，seqUsed 为 [B]。四个 Tensor 均为空时，正 maxQ 通过 ceil(T/maxQ) 确定 B，最后不足一槽也计为一个 batch；maxQ=0 时必须 T=0，并按空 batch 处理。有 KV 长度 Tensor 时可确定非空 B，允许 maxQ=0、T=0。B=0 时 T=0、实际组数为0；显式提供额外输出组时全部填充为空。定长 KV 要求 ceil(maxKV/Y)≤INT32_MAX；不要求 KV≥Q。默认 max=-1 仅在被忽略时合法。

## 输出形状

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

- idx/count及输出为INT32，cu前缀为INT64。调用方提供合法的 Device 长度和足够的输出容量，接口不回读 Device 长度。

## 调用说明

| 调用方式 | 样例代码 | 说明 |
|---|---|---|
| aclnn API | [test_aclnn_union_sparse_gen](./examples/test_aclnn_union_sparse_gen.cpp) | 通过[aclnnUnionSparseGen](./docs/aclnnUnionSparseGen.md)调用。 |
| PyTorch扩展 | - | 通过[Torch接口](./docs/torchapi_union_sparse_gen.md)调用。 |

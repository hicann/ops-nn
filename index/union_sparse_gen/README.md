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
| exact_mask | 输出 | [H,G,U×128]，逐token的query位掩码，padding为0。 | INT32 | ND |
| block_shape | 属性 | Q/KV块粒度，默认且仅支持[1,128]。 | ListInt | - |
| union_size | 属性 | 每组query数，默认且仅支持8。 | INT64 | - |
| mask_type | 属性 | 0为非因果，1为右下对齐causal，默认0。 | INT64 | - |
| max_q_seqlen | 属性 | 默认-1；正值为有效Q上界，无cuQ时还指定槽宽。 | INT64 | - |
| max_kv_seqlen | 属性 | 默认-1；不替代或约束实际KV长度。 | INT64 | - |
| layout_sparse_pattern | 属性 | 稀疏模式布局，默认且仅支持4。 | INT64 | - |
| is_input_idx_sorted | 属性 | 有效前缀有序承诺，默认且仅支持true。 | BOOL | - |
| residual_block_mode | 属性 | 默认且仅支持0，按显式列表处理。 | INT64 | - |
| totalq_union_num | 属性 | 总分组数G，低层调用必传，公共Python自动计算。 | INT64 | - |
| max_logical_blocks | 属性 | 最大逻辑块数Lmax，低层必传，公共Python自动计算。 | INT64 | - |

## 约束说明

- 仅支持[1,128]块、8行分组、有序显式列表和mask类型0/1。输入允许重复，不隐式补尾部块。
- Q/KV分别按cu是否传入解析。无cuQ时seqUsedQ与正maxQ必传，T=B×maxQ；无cuKV时seqUsedKV必传。
- 有效长度优先seqUsed，否则取前缀差分；等宽槽位的有效长度仍可不同。不新增Q/KV布局属性。
- G=Σceil(effectiveQ/8)，Lmax=max(1,max ceil(effectiveKV/128))，U=min(8K,Lmax)，分组不跨请求。
- idx/count及输出为INT32，cu前缀为INT64。低层调用方提供合法的Device长度与G/Lmax，接口不回读Device长度。

## 调用说明

| 调用方式 | 样例代码 | 说明 |
|---|---|---|
| aclnn API | [test_aclnn_union_sparse_gen](./examples/test_aclnn_union_sparse_gen.cpp) | 通过[aclnnUnionSparseGen](./docs/aclnnUnionSparseGen.md)调用。 |
| PyTorch扩展 | - | 通过[Torch接口](./docs/torchapi_union_sparse_gen.md)调用。 |

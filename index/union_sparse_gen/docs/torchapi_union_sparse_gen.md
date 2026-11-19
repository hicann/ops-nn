# union_sparse_gen

简体中文 | [English](./torchapi_union_sparse_gen_en.md)

## 功能说明

生成升序去重的块并集、并集块数和逐token的query位掩码。mask保留query选择关系与有效KV边界；mask_type=1时再应用右下对齐causal。不执行attention。H为KV头数，T为Q存储行总数，B为请求数；块ID相对于各请求的KV起点，不是全局token位置。

## 函数原型

```python
def union_sparse_gen(
    sparse_block_idx, sparse_block_count,
    cu_sqlen_q=None, cu_sqlen_kv=None,
    seq_used_q=None, seq_used_kv=None,
    *, block_shape=(1, 128), union_size=8, mask_type=0,
    max_q_seqlen=-1, max_kv_seqlen=-1,
    layout_sparse_pattern=4, is_input_idx_sorted=True,
    residual_block_mode=0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    ...
```

选择性构建包导出`cann_ops_nn_custom.union_sparse_gen`，完整包导出`cann_ops_nn.union_sparse_gen`。示例统一使用别名`ops`。

## 参数说明

| 参数 | 说明 |
|---|---|
| sparse_block_idx | [H,T,K]，每行显式选择的逻辑KV块ID。NPU上的INT32 Tensor。 |
| sparse_block_count | [H,T]，每行有效前缀长度。NPU上的INT32 Tensor。 |
| cu_sqlen_q | [B+1]，Q存储位置前缀和。CPU上的INT32/INT64 Tensor或None。 |
| cu_sqlen_kv | [B+1]，KV存储长度前缀和。CPU上的INT32/INT64 Tensor或None。 |
| seq_used_q | [B]，各请求的有效Q长度。CPU上的INT32/INT64 Tensor或None。 |
| seq_used_kv | [B]，各请求的有效KV长度。CPU上的INT32/INT64 Tensor或None。 |
| block_shape | Q/KV块粒度，默认[1,128]，支持[1,Y]，Y∈{4,8,16,128}。 |
| union_size | 每组query数，默认且仅支持8。 |
| mask_type | 0为非因果，1为右下对齐causal，默认0。 |
| max_q_seqlen | 默认-1；有任意Q长度Tensor时忽略，否则为非负定长Q长度。 |
| max_kv_seqlen | 默认-1；有任意KV长度Tensor时忽略，否则为非负定长KV长度。 |
| layout_sparse_pattern | 稀疏模式布局，默认且仅支持4。 |
| is_input_idx_sorted | 有效前缀有序承诺，默认true；false允许无序输入。 |
| residual_block_mode | 默认0；1补入不完整可见尾块，参与索引、count及掩码。 |

## 返回值

按顺序返回三个NPU INT32 Tensor：

```text
union_sparse_idx   [H,G,U]
union_sparse_count [H,G]
exact_mask         [H,G,U×Y]
G = Σ_b ceil(effectiveQ_b/8)
Lmax = max(1, max_b ceil(effectiveKV_b/Y))
U = min(8*(K+residual_block_mode), Lmax)
```

索引升序去重，padding为-1；mask的padding及越界KV token为0。第i位对应组内query i。非因果模式仍保留逐query选择关系，并非全1。

## 约束说明

- 仅支持block_shape=[1,Y]（Y∈{4,8,16,128}）、union_size=8、mask_type=0/1、layout_sparse_pattern=4、sorted=true/false、residual=0/1。
- is_input_idx_sorted=true时输入有效前缀须非降序，false允许无序，两者均允许重复，长度由sparse_block_count指定；非法块ID忽略。residual=0 保留合法显式块。residual=1 沿用旧版逐 query 选择规则：显式块仅保留完整位于可见范围内的块（d < floor(max(V,0)/Y)），并在 V>0 且 V%Y!=0 时加入不完整尾块 floor(V/Y)，然后合并去重；尾块同时参与 unionSparseIdx、unionSparseCount 和 exactMask。maskType=1 时 V=KV-Q+query局部位置+1，maskType=0 时 V 为该请求实际有效 KV 长度。完整尾块不自动补入，已选择的尾块不会重复计数。

Q、KV 独立选择长度来源：seqUsed 优先，其次为 cu 相邻差分；对应 cu、seqUsed 任意一个非空时，max 的任意 int64 值均被忽略。两者均为空时才使用 max 作为每个请求的定长有效长度，要求 max≥0。

| cuX | seqUsedX | 有效长度 | maxX |
|---|---|---|---|
| 传入 | 传入 | seqUsedX[b]，不得超过 cuX 差分容量 | 完全忽略 |
| 传入 | nullptr | cuX[b+1]−cuX[b] | 完全忽略 |
| nullptr | 传入 | seqUsedX[b] | 完全忽略 |
| nullptr | nullptr | KV 为 maxKV；Q 为 min(maxQ,T-b×maxQ) | 定长槽宽/长度，必须非负 |

Q 有 cu 时通过前缀定位。Q 只有 seqUsed 时，B 由其长度确定，要求 T%B=0，slot=T/B，请求 b 从 b×slot 开始，seqUsedQ[b]≤slot。Q 两个长度 Tensor 都为空时，slot=maxQ，B=ceil(T/maxQ)，有效Q=min(maxQ,T-b×maxQ)，允许最后一槽不足 maxQ。例如 T=10、maxQ=6 时 Q 长度为 [6,4]。KV 只有 seqUsed 时不受 Q slot 限制。所有已传长度 Tensor 必须具有相同 B；cu 形状为 [B+1]，seqUsed 为 [B]，定长 Q 时 KV 描述符的 B 必须等于 ceil(T/maxQ)。四个 Tensor 均为空时，正 maxQ 通过 ceil(T/maxQ) 确定 B；maxQ=0 时必须 T=0，并按空 batch 处理。有 KV 长度 Tensor 时可确定非空 B，允许 maxQ=0、T=0。B=0 时 T=G=0。定长 KV 要求 ceil(maxKV/Y)≤INT32_MAX；不要求 KV≥Q。默认 max=-1 仅在被忽略时合法。

- CPU长度校验后，cu转为NPU INT64，seqUsed转为NPU INT32。不新增Q/KV布局参数。
- 公共入口根据CPU长度计算G/Lmax/U并分配输出，调用成本包含校验和长度下发；首次调用可能JIT编译C++桥接，需可用的C++编译器与ninja；缺少ninja时，在当前Python环境执行`python -m pip install ninja`。

## 低层接口

```python
torch.ops.cann_ops_nn.union_sparse_gen_launch(
    sparse_block_idx, sparse_block_count,
    cu_sqlen_q, cu_sqlen_kv, seq_used_q, seq_used_kv,
    block_shape, union_size, mask_type, max_q_seqlen, max_kv_seqlen,
    layout_sparse_pattern, is_input_idx_sorted, residual_block_mode,
    totalq_union_num, max_logical_blocks,
)
```

所有非空Tensor输入均须位于同一NPU；cu必须INT64，其余输入INT32。调用方传入准确的G/Lmax并保证Device长度值合法；低层接口不回读长度。G=0表示实际空任务，不表示自动推导；Fake/Meta仅使用传入几何，不计算长度内容。容量始终自动计算。

低层`totalq_union_num`与`max_logical_blocks`的默认值均为`-1`，用于检测参数遗漏，不表示自动推导。省略任意一个参数或传入负G、非正Lmax时，在输出分配和kernel启动前报错；Lmax还须不超过INT32_MAX。公共`union_sparse_gen`入口仍自动计算并显式传入准确几何，调用方式不变。

直接构造 GE 节点时，InferShape 仅使用 Host shape 和属性，不读取长度 Tensor 数据或同步 Device。totalq_union_num>0 时直接使用 G，否则 G=(T+7*B)//8；max_logical_blocks>0 时直接使用容量 U，否则 U=8*(K+residual)。B 由长度描述符推导，定长 Q 使用 ceil(T/maxQ)，允许末槽不足。ACLNN 直调以输出 shape 为准；Torch 的预分配规则保持上述 G/L/U 约定。

## 调用示例

安装Torch扩展及其匹配的ACLNN算子包后：

```python
import torch
import torch_npu
import cann_ops_nn_custom as ops

idx = torch.tensor([[[0], [0]]], dtype=torch.int32, device="npu:0")
count = torch.ones((1, 2), dtype=torch.int32, device="npu:0")
cu_q = torch.tensor([0, 2], dtype=torch.int64)
cu_kv = torch.tensor([0, 128], dtype=torch.int64)
union_idx, union_count, mask = ops.union_sparse_gen(
    idx, count, cu_q, cu_kv, mask_type=1,
)
assert union_idx.shape == (1, 1, 1)
assert union_count.cpu().tolist() == [[1]]
assert torch.all(mask[0, 0, :127] == 3)
assert mask[0, 0, 127].item() == 2
```

C接口及精确可见性公式见[aclnnUnionSparseGen](./aclnnUnionSparseGen.md)。

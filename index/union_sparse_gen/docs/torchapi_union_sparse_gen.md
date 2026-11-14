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
| block_shape | Q/KV块粒度，默认且仅支持[1,128]。 |
| union_size | 每组query数，默认且仅支持8。 |
| mask_type | 0为非因果，1为右下对齐causal，默认0。 |
| max_q_seqlen | 默认-1；正值为有效Q上界，无cuQ时还指定槽宽。 |
| max_kv_seqlen | 默认-1；不替代或约束实际KV长度。 |
| layout_sparse_pattern | 稀疏模式布局，默认且仅支持4。 |
| is_input_idx_sorted | 有效前缀有序承诺，默认且仅支持true。 |
| residual_block_mode | 默认且仅支持0，按显式列表处理。 |

## 返回值

按顺序返回三个NPU INT32 Tensor：

```text
union_sparse_idx   [H,G,U]
union_sparse_count [H,G]
exact_mask         [H,G,U×128]
G = Σ_b ceil(effectiveQ_b/8)
Lmax = max(1, max_b ceil(effectiveKV_b/128))
U = min(8K, Lmax)
```

索引升序去重，padding为-1；mask的padding及越界KV token为0。第i位对应组内query i。非因果模式仍保留逐query选择关系，并非全1。

## 约束说明

- 仅支持block_shape=[1,128]、union_size=8、mask_type=0/1、layout_sparse_pattern=4、sorted=true、residual=0。
- 输入有效前缀非降序，允许重复，长度由sparse_block_count指定；非法块ID忽略，causal后全零的合法块仍保留。
- Q/KV分别按cu是否存在解析。有cuQ时按前缀定位，有效Q优先seqUsedQ，否则取前缀差分；无cuQ时seqUsedQ和正maxQ必传，T=B×maxQ，请求b从b×maxQ开始。
- 有效KV优先seqUsedKV，否则取cuKV差分；无cuKV时seqUsedKV必传。seqUsed不得超过对应前缀容量。
- 两个max须≥-1；正maxQ约束有效Q，无cuQ时还须≤INT32_MAX。maxKV不替代或限制实际KV长度。
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

直接构造GE节点时，这两个属性可省略，默认`-1`表示由长度Tensor自动推导缺失的几何值：`G=Σceil(effectiveQ[b]/8)`，`Lmax=max(1,max_b ceil(effectiveKV[b]/128))`。`seq_used`存在时优先使用其值，否则使用对应前缀差分；固定Q槽位仍须提供`seq_used_q`和正数`max_q_seqlen`。推导在精确输出分配前完成；长度值仅在NPU时存在运行时值依赖开销。该能力不改变ACLNN或低层Torch的显式几何要求。显式几何的Host解析分支不主动读取长度值；框架层的值依赖调度由GE决定。

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

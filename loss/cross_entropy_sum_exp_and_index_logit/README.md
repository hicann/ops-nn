# CrossEntropySumExpAndIndexLogit

## 产品支持情况

| 产品 | 是否支持 |
|:----------------------------|:-----------:|
| <term>Ascend 950PR/Ascend 950DT</term> | √ |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term> | × |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> | × |
| <term>Atlas 200I/500 A2 推理产品</term> | × |
| <term>Atlas 推理系列产品</term> | × |
| <term>Atlas 训练系列产品</term> | × |

## 功能说明

- 算子功能：面向vocab并行（Tensor Parallel）场景的CrossEntropy本地计算融合算子。对按vocab维切分后的TP rank本地logits，在`all_reduce(MAX)`之后、`predicted_logits` / `sum_exp_logits`的`all_reduce(SUM)`之前，一次性完成logits平移（减全局最大值）、target越界mask判定、本地target offset计算、target对应logit的gather，以及`exp`与沿vocab维的本地求和，降低小算子launch、GM读写和中间tensor materialization开销。面向超大词表大模型训练场景。

- 计算公式：设当前rank本地vocab shard为`vocab_parallel_logits`，全局最大logit为`global_logits_max`，vocab分片范围为`[vocab_start_index, vocab_end_index)`。对每个token `i`、每个本地vocab位置`j`：

  **1. target mask：**

  $$
  target\_mask[i] = \begin{cases} 1, & target[i] < vocab\_start\_index \text{ 或 } target[i] \geq vocab\_end\_index \\ 0, & \text{otherwise} \end{cases}
  $$

  **2. 本地target offset：**

  $$
  target\_offset[i] = \begin{cases} 0, & target\_mask[i] = 1 \\ target[i] - vocab\_start\_index, & target\_mask[i] = 0 \end{cases}
  $$

  **3. predicted logit gather（logits平移）：**

  $$
  predicted\_logits[i] = \begin{cases} 0, & target\_mask[i] = 1 \\ vocab\_parallel\_logits[i, target\_offset[i]] - global\_logits\_max[i], & target\_mask[i] = 0 \end{cases}
  $$

  **4. 指数计算：**

  $$
  exp\_logits[i, j] = \exp\big(vocab\_parallel\_logits[i, j] - global\_logits\_max[i]\big)
  $$

  **5. 本地求和：**

  $$
  sum\_exp\_logits[i] = \sum_{j=0}^{V\_local-1} exp\_logits[i, j]
  $$

- 特殊处理说明：中间计算强制使用FLOAT（`BFLOAT16`输入自动升精度），exp操作数已减去全局最大值以抑制上溢，下溢按FLOAT自然返回0。本算子不涉及除法，无需除零保护。`sum_exp_logits`沿V_local维的累加在单核内完成（无跨核规约、无AtomicAdd），累加顺序固定，相同输入多次调用结果一致，为确定性实现。

## 参数说明

<table style="table-layout: fixed; width: 100%">
  <thead>
    <tr>
      <th style="white-space: nowrap">参数名</th>
      <th style="white-space: nowrap">输入/输出/属性</th>
      <th style="white-space: nowrap">描述</th>
      <th style="white-space: nowrap">数据类型</th>
      <th style="white-space: nowrap">数据格式</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>vocab_parallel_logits</td>
      <td>输入</td>
      <td>当前TP rank的本地vocab shard logits，对应公式中的vocab_parallel_logits。支持非连续Tensor。</td>
      <td>FLOAT、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>target</td>
      <td>输入</td>
      <td>全局vocab索引，对应公式中的target。取值为非负整数。支持非连续Tensor。</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>global_logits_max</td>
      <td>输入</td>
      <td>all_reduce(MAX) 后得到的全局最大logit，对应公式中的global_logits_max。数据类型需与vocab_parallel_logits一致。支持非连续Tensor。</td>
      <td>同vocab_parallel_logits</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>vocab_start_index</td>
      <td>属性</td>
      <td>当前rank vocab分片起始索引（全局），对应公式中的vocab_start_index。需满足vocab_end_index &gt; vocab_start_index。</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>vocab_end_index</td>
      <td>属性</td>
      <td>当前rank vocab分片结束索引（全局），对应公式中的vocab_end_index。需满足vocab_end_index - vocab_start_index == vocab_parallel_logits.size(-1)。</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>predicted_logits</td>
      <td>输出</td>
      <td>target对应的logit减去全局最大值的结果，对应公式中的predicted_logits。target不在当前rank分片内时对应位置为0。</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>sum_exp_logits</td>
      <td>输出</td>
      <td>本地exp(logits - global_max) 沿最后一维的求和，对应公式中的sum_exp_logits。</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>exp_logits</td>
      <td>输出</td>
      <td>exp(vocab_parallel_logits - global_logits_max)，对应公式中的exp_logits。</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>target_offset</td>
      <td>输出</td>
      <td>target - vocab_start_index，对应公式中的target_offset。target不在当前rank分片内时置0。</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>target_mask</td>
      <td>输出</td>
      <td>vocab越界掩码，1表示target不在当前rank分片内、0表示在内，对应公式中的target_mask。</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
  </tbody>
</table>

## 约束说明

- `vocab_parallel_logits`仅支持二维`[N, V_local]`或三维`[S, B, V_local]`（此时`N = S * B`），`shape[:-1]`需与`target.shape`、`global_logits_max.shape`完全一致。
- V_local对齐约束：BFLOAT16输入时V_local需为16的倍数，FLOAT输入时V_local需为8的倍数（保证UB 32字节对齐）。

### 规格约束

| 规格项 | 规格 | 规格说明 |
|:--- |:--- |:--- |
| N | 1~32K | prod(target.shape)，即展平后token总数。 |
| V_local | 16~200K | 当前rank本地vocab分片长度；BFLOAT16需为16的倍数、FLOAT需为8的倍数。 |

### 典型值

| 规格项 | 典型值 |
|:--- |:--- |
| N | 1024/2048/4096 |
| V_local | 37888（推荐） |

## 调用说明

| 调用方式 | 样例代码 | 说明 |
|:--- |:--- |:--- |
| aclnn接口 | [test_aclnn_cross_entropy_sum_exp_and_index_logit.cpp](examples/arch35/test_aclnn_cross_entropy_sum_exp_and_index_logit.cpp) | 通过 [aclnnCrossEntropySumExpAndIndexLogit](docs/aclnnCrossEntropySumExpAndIndexLogit.md) 接口方式调用算子。 |
| GE图模式 | [test_geir_cross_entropy_sum_exp_and_index_logit.cpp](examples/arch35/test_geir_cross_entropy_sum_exp_and_index_logit.cpp) | 通过 [算子IR](op_graph/cross_entropy_sum_exp_and_index_logit_proto.h) 构图方式调用算子。 |
| PyTorch API | - | 通过 [torch.ops.cann_ops_nn.cross_entropy_sum_exp_and_index_logit](docs/torchapi_cross_entropy_sum_exp_and_index_logit.md) 接口调用算子。 |

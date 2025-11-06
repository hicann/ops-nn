# KvRmsNormRopeCache

## 产品支持情况

|产品             |  是否支持  |
|:-------------------------|:----------:|
|  <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>   |     ×    |
|  <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>     |     ×    |

## 功能说明

- 算子功能：对输入张量(kv)的尾轴，拆分出左半边用于rms_norm计算，右半边用于rope计算，再将计算结果分别scatter到两块cache中。
- 计算公式：

  (1) interleaveRope:

  $$
  x=kv[...,Dv:]
  $$

  $$
  x1=x[...,::2]
  $$

  $$
  x2=x[...,1::2]
  $$

  $$
  x\_part1=torch.cat((x1,x2),dim=-1)
  $$

  $$
  x\_part2=torch.cat((-x2,x1),dim=-1)
  $$

  $$
  y=x\_part1*cos+x\_part2*sin
  $$

  (2) rmsNorm:

  $$
  x=kv[...,:Dv]
  $$

  $$
  square\_x=x*x
  $$

  $$
  mean\_square\_x=square\_x.mean(dim=-1,keepdim=True)
  $$

  $$
  rms=torch.sqrt(mean\_square\_x+epsilon)
  $$

  $$
  y=(x/rms)*gamma
  $$
## 参数说明

<table style="undefined;table-layout: fixed; width: 1576px"><colgroup>
  <col style="width: 170px">
  <col style="width: 170px">
  <col style="width: 312px">
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
      <td>kv</td>
      <td>输入</td>
      <td>用于切分出rms_norm计算所需数据Dv和rope计算所需数据Dk的输入数据，对应公式中的`kv`。</td>
      <td>FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>gamma</td>
      <td>输入</td>
      <td>用于rms_norm计算的输入数据，对应公式中的`gamma`。</td>
      <td>FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>cos</td>
      <td>输入</td>
      <td>用于rope计算的输入数据，对输入张量进行余弦变换，对应公式中的`cos`。</td>
      <td>FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>sin</td>
      <td>输入</td>
      <td>用于rope计算的输入数据，对输入张量进行正弦变换，对应公式中的`sin`。</td>
      <td>FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>epsilon</td>
      <td>可选属性</td>
      <td><ul><li>用于防止rms_norm计算除0错误，对应公式中的eps。</li><li>默认值为1e-5。</li></ul></td>
      <td>FLOAT32</td>
      <td>-</td>
    </tr>
    <tr>
      <td>index</td>
      <td>输入</td>
      <td>用于指定写入cache的具体索引位置。</td>
      <td>INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>kCacheRef</td>
      <td>输入/输出</td>
      <td>提前申请的cache，输入输出同地址复用。</td>
      <td>FLOAT16、BFLOAT16、INT8</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>ckvCacheRef</td>
      <td>输入/输出</td>
      <td>提前申请的cache，输入输出同地址复用。</td>
      <td>FLOAT16、BFLOAT16、INT8</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>kRopeScaleOptional</td>
      <td>可选属性</td>
      <td>当kCacheRef数据类型为INT8时需要此输入参数。</td>
      <td>FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>ckvScaleOptional</td>
      <td>可选属性</td>
      <td>当ckvCacheRef数据类型为INT8时需要此输入参数。</td>
      <td>FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>kRopeOffsetOptional</td>
      <td>可选属性</td>
      <td>当kCacheRef数据类型为INT8且对应的kRopeScaleOptional输入存在并量化场景为非对称量化时，需要此参数输入。</td>
      <td>FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>cKvOffsetOptional</td>
      <td>可选属性</td>
      <td>当ckvCacheRef数据类型为INT8且对应的ckvScaleOptional输入存在并量化场景为非对称量化时，需要此参数输入。</td>
      <td>FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>cacheModeOptional</td>
      <td>可选属性</td>
      <td>cache格式的选择标记。类型有Norm、PA、PA_BNSD、PA_NZ、PA_BLK_BNSD、PA_BLK_NZ。</td>
      <td>CHAR*</td>
      <td>-</td>
    </tr>
    <tr>
      <td>isOutputKv</td>
      <td>可选属性</td>
      <td>kRopeOut和cKvOut输出控制标记。</td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
    <tr>
      <td>kRopeOut</td>
      <td>输出</td>
      <td>rope计算结果，对应interleaveRope计算公式中的`y`。</td>
      <td>FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>cKvOut</td>
      <td>输出</td>
      <td>rms_norm计算结果，对应rmsNorm计算公式中的`y`。</td>
      <td>FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>

  </tbody></table>

## 约束说明

  * 输入shape限制：
      * kv为四维张量，shape为[Bkv,N,Skv,D]，Bkv为输入kv的batch size，Skv为输入kv的sequence length，大小由用户输入场景决定，无明确限制。
      * N为输入kv的head number。此算子与DeepSeekV3网络结构强相关，仅支持N=1的场景，不存在N非1的场景。
      * D为输入kv的head dim。rms_norm计算所需数据Dv和rope计算所需数据Dk由输入kv的D切分而来。故Dk、Dv大小需满足Dk+Dv=D。同时，Dk需满足rope规则。根据rope规则，Dk为偶数。
      * 若cacheModeOptional为PA场景（cacheModeOptional为PA、PA_BNSD、PA_NZ、PA_BLK_BNSD、PA_BLK_NZ），其shape[BlockNum,BlockSize,N,Dk]中BlockSize需32B对齐。
  * 其他限制：
      * 对于index，当cacheModeOptional为Norm时，shape为2维[Bkv,Skv]，要求index的value值范围为[-1,Scache)。不同的Bkv下，value数值可以重复。
      * 当cacheModeOptional为PA_BNSD、PA_NZ时，shape为1维[Bkv * Skv]，要求index的value值范围为[-1,BlockNum * BlockSize)。value数值不能重复。
      * 当cacheModeOptional为PA_BLK_BNSD、PA_BLK_NZ时，shape为1维[Bkv * ceil_div(Skv,BlockSize)]，要求index的value的数值范围为[-1,BlockNum * BlockSize)。value/BlockSize的值不能重复。

## 调用说明

| 调用方式   | 样例代码           | 说明                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn接口  | [test_aclnn_kv_rms_norm_rope_cache](examples/test_aclnn_kv_rms_norm_rope_cache.cpp) | 通过[aclnnKvRmsNormRopeCache](docs/aclnnKvRmsNormRopeCache.md)接口方式调用KvRmsNormRopeCache算子。 |
| 图模式 | [test_geir_kv_rms_norm_rope_cache](examples/test_geir_kv_rms_norm_rope_cache.cpp)  | 通过[算子IR](op_graph/kv_rms_norm_rope_cache_proto.h)构图方式调用KvRmsNormRopeCache算子。         |
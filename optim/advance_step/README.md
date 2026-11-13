# AdvanceStep

## 产品支持情况

|产品             |  是否支持  |
|:-------------------------|:----------:|
|  <term>Ascend 950PR&950DT系列产品</term>   |     √    |
|  <term>Atlas A3系列产品</term>  |     √    |
|  <term>Atlas A2系列产品</term>     |     √    |
|  <term>Atlas 200I/500 A2推理产品</term>    |     ×    |
|  <term>Atlas推理系列产品</term>     |     ×    |
|  <term>Atlas训练系列产品</term>    |     ×    |

## 功能说明

- 算子功能：

  vLLM是一个高性能的LLM推理和服务框架，专注于优化大规模语言模型的推理效率。它的核心特点包括PageAttention和高效内存管理。advance_step算子的主要作用是推进推理步骤，即在每个生成步骤中更新模型的状态并生成新的inputTokens、inputPositions、seqLens和slotMapping，为vLLM的推理提升效率。

- 计算公式：

  以下公式仅适用于 **V1 路径**（可选输入specToken/acceptedNum未传入或仅传入其一时，numSeqs > numQueries）。V2 投机采样路径（可选输入同时传入）的输出语义不同，见 [aclnnAdvanceStepV2](./docs/aclnnAdvanceStepV2.md) 的功能说明。

  记 query 行序号 $i \in [0, numQueries)$，$pos_i$ 为更新前的 $seqLens[i]$，$blockTablesFlat$ 为 $blockTables$ 按行展开的一维视角：

  $$
  blockTablesStride = blockTables.stride(0)
  $$

  $$
  inputTokens[i] = sampledTokenIds[i, 0]
  $$

  $$
  inputPositions[i] = pos_i
  $$

  $$
  seqLens[i] = pos_i + 1
  $$

  $$
  slotMapping[i] = blockTablesFlat[blockTablesStride * i + \lfloor pos_i / blockSize \rfloor] * blockSize + (pos_i \% blockSize)
  $$

  V1 路径（可选输入未传入，numSeqs > numQueries）额外执行 padding 行全量清零：对所有 $j \in [numQueries, numSeqs)$ 写入 $inputTokens[j]=0$、$inputPositions[j]=0$、$slotMapping[j]=-1$（$seqLens$ 的 padding 行保留输入原值）。

## 参数说明

<table style="undefined;table-layout: fixed; width: 1576px"><colgroup>
  <col style="width: 170px">
  <col style="width: 170px">
  <col style="width: 310px">
  <col style="width: 212px">
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
      <td>inputTokens</td>
      <td>输入</td>
      <td>公式中的输入inputTokens；算子以原地（inplace）方式将更新结果写回该张量。</td>
      <td>INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>sampledTokenIds</td>
      <td>输入</td>
      <td>公式中的输入sampledTokenIds。</td>
      <td>INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>inputPositions</td>
      <td>输入</td>
      <td>公式中的输入inputPositions；算子以原地（inplace）方式将更新结果写回该张量。</td>
      <td>INT64</td>
      <td>ND</td>
    </tr>
     <tr>
      <td>seqLens</td>
      <td>输入</td>
      <td>公式中的输入seqLens；算子以原地（inplace）方式将更新结果写回该张量。</td>
      <td>INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>slotMapping</td>
      <td>输入</td>
      <td>公式中的输入slotMapping；算子以原地（inplace）方式将更新结果写回该张量。</td>
      <td>INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>blockTables</td>
      <td>输入</td>
      <td>公式中的输入blockTables，2-D张量 [numSeqs, blockTablesStride]，第二维为每序列的block表容量。</td>
      <td>INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>specToken</td>
      <td>可选输入</td>
      <td>投机采样路径（V2）的可选输入，shape为[numSeqs, specNum]；与acceptedNum须同时传入或同时不传入，传入空Tensor视为未传入（回落V1路径）。</td>
      <td>INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>acceptedNum</td>
      <td>可选输入</td>
      <td>投机采样路径（V2）的可选输入，每序列接受的投机token数量，shape为[numSeqs]；与specToken须同时传入或同时不传入，传入空Tensor视为未传入（回落V1路径）。</td>
      <td>INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>numSeqs</td>
      <td>属性</td>
      <td><ul><li>记录输入的seq数量。</li><li>取值范围是大于0且小于200000000的整数。</li><li>V1路径（可选输入未传入）：大于numQueries，且等于inputTokens等1-D输入的长度。</li><li>V2路径（可选输入同时传入）：等于numQueries。</li></ul></td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>numQueries</td>
      <td>属性</td>
      <td><ul><li>记录输入的Query的数量，大小与sampledTokenIds第一维的长度一致。</li><li>取值范围是大于0的正整数。</li></ul></td>
      <td>INT64</td>
      <td>-</td>
    </tr>
      <tr>
      <td>blockSize</td>
      <td>属性</td>
      <td><ul><li>每个block的大小。</li><li>取值范围是大于0的正整数。</li></ul></td>
      <td>INT64</td>
      <td>-</td>
    </tr>
  </tbody></table>

## 约束说明

### Host侧校验约束（编译/tiling阶段校验，不满足时报错终止）

- 数据类型：所有输入、输出仅支持INT64。
- 数据格式：所有输入仅支持ND格式。
- 属性约束：numSeqs、numQueries、blockSize必须为大于0的整数，且numSeqs必须小于200000000。
- 形状约束（spec_token与accepted_num均未传入，V1路径）：
  - inputTokens、inputPositions、seqLens、slotMapping、blockTables的第一维长度必须等于numSeqs。
  - sampledTokenIds的shape必须为[numQueries, 1]。
  - numSeqs必须大于numQueries。
- 形状约束（spec_token与accepted_num同时传入，V2路径）：
  - inputTokens、inputPositions、seqLens、slotMapping的shape必须为[numSeqs * (1 + specNum)]，specNum为spec_token第二维长度且必须大于0。
  - sampledTokenIds的shape必须为[numSeqs, (1 + specNum)]，spec_token的shape必须为[numSeqs, specNum]，accepted_num的shape必须为[numSeqs]。
  - blockTables的shape必须为[numSeqs, blockTablesStride]，第二维长度blockTablesStride必须大于0。
  - numSeqs必须等于numQueries。
  - spec_token与accepted_num须同时传入或同时不传入（传入空Tensor视为未传入）；仅传入其一或均未传入时走V1路径，spec_token/accepted_num不参与计算。spec_token为空Tensor（specNum为0）时不会走V2路径。无投机解码场景可评估使用aclnnAdvanceStep，但V1路径要求numSeqs大于numQueries，与V2路径（numSeqs等于numQueries）的约束不同，不能直接等价替换。
- 所有必选输入不支持空Tensor（上述形状约束已隐含，第一维不小于1）。

### 用户保证约束（涉及数据内容或内存布局，host侧无法校验，由调用方保证，违反时可能导致计算错误或越界访问）

- 所有输入Tensor必须连续，不支持非连续Tensor。
- **入参默认值域（host侧与kernel侧均不校验数据值，须由用户保证）**：输入数据值须为符合业务语义的非负整数（token ID、序列长度、block编号、物理槽位等，0为合法值），token ID按INT64非负全域取值（fp32仅用于V2路径-1哨兵检索，fp32不可精确表示的非-1值不会误判为哨兵）；V2路径sampledTokenIds还支持-1哨兵值（表示无接受token）。inputTokens、slotMapping为原地输出张量，其输入值不参与计算（V1路径下[numQueries, numSeqs)中未被padding标记覆盖的行保留输入原值）。
- blockTables的第二维长度必须大于seqLens中最大值除以blockSize，保证block表能覆盖所有序列的最大block索引；V2路径须计入acceptedNum与投机槽位贡献，即满足 pos + acceptedNum + 1 < blockTablesStride × blockSize（pos为更新前的inputPositions值）。
- V2路径中间计算以int32参与运算（位置/序列长度/槽位类数值须在int32表示范围内）；blockSize与blockTablesStride小于2^31已由tiling校验。
- 值域违约（如负的位置值或超出blockTables覆盖范围的位置）属契约外输入：查表列号越出本行blockTables覆盖范围（q = pos/blockSize ∉ [0, blockTablesStride)）时，V1/V2路径均防御性写0——块号与余数同时屏蔽，不发生GM越界读。V1路径该行为已定义为公开契约（越界行slotMapping防御性写0、其余输出按公式对输入原值计算，golden同步镜像，契约外值域用例可精度比对）；V2路径位置值超出int32表示域时，kernel中间计算（int32管线）与golden（int64计算）存在表示差异，该子域不保证值一致。

### 图模式（GEIR）能力说明

GE图模式支持动态shape（-1）与动态rank（-2）编译执行，覆盖样例见 [test_geir_advance_step_dynamic.cpp](./examples/test_geir_advance_step_dynamic.cpp)（V1/V2 × 动态Shape(-1)/动态Rank(-2) 四场景）。

## 调用说明

| 调用方式 | 样例代码                                                                   | 说明                                                             |
|--------------|------------------------------------------------------------------------|----------------------------------------------------------------|
| aclnn API | [test_aclnn_advance_step](./examples/test_aclnn_advance_step.cpp) | 通过[aclnnAdvanceStep](./docs/aclnnAdvanceStep.md)接口方式调用AdvanceStep算子。    |
| aclnn API | [test_aclnn_advance_step_v2](./examples/test_aclnn_advance_step_v2.cpp) | 通过[aclnnAdvanceStepV2](./docs/aclnnAdvanceStepV2.md)接口方式调用AdvanceStep算子。    |
| GE图模式 | [test_geir_advance_step_dynamic](./examples/test_geir_advance_step_dynamic.cpp) | 通过[算子IR](./op_graph/advance_step_proto.h)构图方式调用AdvanceStep算子（含动态shape/rank样例）。 |

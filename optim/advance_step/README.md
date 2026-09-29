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

  $$
  blockIdx是当前代码被执行的核的index。
  $$

  $$
  blockTablesStride = blockTables.stride(0)
  $$

  $$
  inputTokens[blockIdx] = sampledTokenIds[blockIdx]
  $$

  $$
  inputPositions[blockIdx] = seqLens[blockIdx]
  $$

  $$
  seqLens[blockIdx] = seqLens[blockIdx] + 1
  $$

  $$
  slotMapping[blockIdx] = (blockTables[blockIdx] + blockTablesStride * blockIdx) * blockSize + (seqLens[blockIdx] \% blockSize)
  $$

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
      <td>输入/输出</td>
      <td>公式中的输入/输出inputTokens。</td>
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
      <td>输入/输出</td>
      <td>公式中的输入/输出inputPositions。</td>
      <td>INT64</td>
      <td>ND</td>
    </tr>
     <tr>
      <td>seqLens</td>
      <td>输入/输出</td>
      <td>公式中的输入/输出seqLens。</td>
      <td>INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>slotMapping</td>
      <td>输入/输出</td>
      <td>公式中的输入/输出slotMapping。</td>
      <td>INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>blockTables</td>
      <td>输入</td>
      <td>公式中的输入blockTables。</td>
      <td>INT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>numSeqs</td>
      <td>属性</td>
      <td><ul><li>记录输入的seq数量，大小与seqLens的长度一致。</li><li>取值范围是大于0的正整数。numSeqs的值大于输入numQueries的值。</li></ul></td>
      <td>INT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>numQueries</td>
      <td>属性</td>
      <td><ul><li>记录输入的Query的数量，大小与sampledTokenIds第一维的长度一致。</li><li>取值范围是大于0的正整数。</li></ul></td>
      <td>INT</td>
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
  - inputTokens、inputPositions、seqLens、slotMapping的shape必须为[numSeqs * (1 + specNum)]，specNum为spec_token第二维长度。
  - sampledTokenIds的shape必须为[numSeqs, (1 + specNum)]，spec_token的shape必须为[numSeqs, specNum]，accepted_num的shape必须为[numSeqs]。
  - blockTables的shape必须为[numSeqs, blockTablesStride]，第二维长度blockTablesStride必须大于0。
  - spec_token与accepted_num须同时传入或同时不传入（传入空Tensor视为未传入）；仅传入其一时走V1路径，spec_token/accepted_num不参与计算。
- 所有必选输入不支持空Tensor（上述形状约束已隐含，第一维不小于1）。

### 用户保证约束（涉及数据内容或内存布局，host侧无法校验，由调用方保证，违反时可能导致计算错误或越界访问）

- 所有输入Tensor必须连续，不支持非连续Tensor。
- inputTokens、sampledTokenIds、inputPositions、seqLens、slotMapping、blockTables的数据值须为符合业务语义的有效值（正整数，如token ID、序列长度、block编号、物理槽位等）。
- blockTables的第二维长度必须大于seqLens中最大值除以blockSize，保证block表能覆盖所有序列的最大block索引。

## 调用说明

| 调用方式 | 调用样例                                                                   | 说明                                                             |
|--------------|------------------------------------------------------------------------|----------------------------------------------------------------|
| aclnn调用 | [test_aclnn_advance_step](./examples/test_aclnn_advance_step.cpp) | 通过[aclnnAdvanceStep](./docs/aclnnAdvanceStep.md)接口方式调用AdvanceStep算子。    |
| aclnn调用 | [test_aclnn_advance_step_v2](./examples/test_aclnn_advance_step_v2.cpp) | 通过[aclnnAdvanceStepV2](./docs/aclnnAdvanceStepV2.md)接口方式调用AdvanceStep算子。    |
| 图模式调用 | -   | 通过[算子IR](./op_graph/advance_step_proto.h)构图方式调用AdvanceStep算子。 |

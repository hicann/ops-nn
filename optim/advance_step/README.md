# AdvanceStep

## 产品支持情况

|产品             |  是否支持  |
|:-------------------------|:----------:|
|  <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>   |     √    |
|  <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>     |     √    |

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

无  

## 调用说明

| 调用方式 | 调用样例                                                                   | 说明                                                             |
|--------------|------------------------------------------------------------------------|----------------------------------------------------------------|
| aclnn调用 | - | 通过[aclnnAdvanceStep](./docs/aclnnAdvanceStep.md)接口方式调用AdvanceStep算子。    |
| 图模式调用 | -   | 通过[算子IR](./op_graph/advance_step_proto.h)构图方式调用AdvanceStep算子。 |
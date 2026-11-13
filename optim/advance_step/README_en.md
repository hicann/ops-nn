# AdvanceStep

English | [简体中文](./README.md)

## Applicable Products

|Product             |  Support  |
|:-------------------------|:----------:|
|  <term>Ascend 950PR&950DT products</term>   |     √    |
|  <term>Atlas A3 products</term>  |     √    |
|  <term>Atlas A2 products</term>     |     √    |
|  <term>Atlas 200I/500 A2 inference products</term>    |     ×    |
|  <term>Atlas inference products</term>     |     ×    |
|  <term>Atlas training products</term>    |     ×    |

## Function

- Operator function:

  vLLM is a high-performance LLM inference and serving framework that focuses on optimizing the inference efficiency of large language models. Its core features include PageAttention and efficient memory management. The main role of the advance_step operator is to advance the inference step, that is, to update the model state at each generation step and generate new inputTokens, inputPositions, seqLens, and slotMapping, improving the inference efficiency of vLLM.

- Formula:

  The following formulas apply only to the **V1 path** (when neither or only one of the optional inputs specToken/acceptedNum is passed, numSeqs > numQueries). The V2 speculative-sampling path (both optional inputs passed) has different output semantics; see the function description in [aclnnAdvanceStepV2](./docs/aclnnAdvanceStepV2_en.md).

  Let $i \in [0, numQueries)$ be the query row index, $pos_i$ the pre-update $seqLens[i]$, and $blockTablesFlat$ the row-flattened one-dimensional view of $blockTables$:

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

  The V1 path (optional inputs not passed, numSeqs > numQueries) additionally clears all padding rows: for every $j \in [numQueries, numSeqs)$ it writes $inputTokens[j]=0$, $inputPositions[j]=0$, and $slotMapping[j]=-1$ (the padding rows of $seqLens$ keep their input values).

## Parameters

<table style="undefined;table-layout: fixed; width: 1576px"><colgroup>
  <col style="width: 170px">
  <col style="width: 170px">
  <col style="width: 310px">
  <col style="width: 212px">
  <col style="width: 100px">
  </colgroup>
  <thead>
    <tr>
      <th>Parameter name</th>
      <th>Input/Output/Attribute</th>
      <th>Description</th>
      <th>Data type</th>
      <th>Data format</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>inputTokens</td>
      <td>Input/Output</td>
      <td>Input/Output inputTokens in the formula.</td>
      <td>INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>sampledTokenIds</td>
      <td>Input</td>
      <td>Input sampledTokenIds in the formula.</td>
      <td>INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>inputPositions</td>
      <td>Input/Output</td>
      <td>Input/Output inputPositions in the formula.</td>
      <td>INT64</td>
      <td>ND</td>
    </tr>
     <tr>
      <td>seqLens</td>
      <td>Input/Output</td>
      <td>Input/Output seqLens in the formula.</td>
      <td>INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>slotMapping</td>
      <td>Input/Output</td>
      <td>Input/Output slotMapping in the formula.</td>
      <td>INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>blockTables</td>
      <td>Input</td>
      <td>Input blockTables in the formula.</td>
      <td>INT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>numSeqs</td>
      <td>Attribute</td>
      <td><ul><li>Records the number of input seqs, and its size is consistent with the length of seqLens.</li><li>The value range is positive integers greater than 0.</li><li>In the V1 path, numSeqs is greater than numQueries; in the V2 path, numSeqs is equal to numQueries.</li></ul></td>
      <td>INT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>numQueries</td>
      <td>Attribute</td>
      <td><ul><li>Records the number of input Queries, and its size is consistent with the length of the first dimension of sampledTokenIds.</li><li>The value range is positive integers greater than 0.</li></ul></td>
      <td>INT</td>
      <td>-</td>
    </tr>
      <tr>
      <td>blockSize</td>
      <td>Attribute</td>
      <td><ul><li>Size of each block.</li><li>The value range is positive integers greater than 0.</li></ul></td>
      <td>INT64</td>
      <td>-</td>
    </tr>
  </tbody></table>

## Constraints

### Host-side validation constraints (validated at the compilation/tiling stage; if not satisfied, an error is reported and execution terminates)

- Data types: all inputs and outputs support only INT64.
- Data formats: all inputs support only the ND format.
- Attribute constraints: numSeqs, numQueries, and blockSize must be integers greater than 0, and numSeqs must be less than 200000000.
- Shape constraints (neither spec_token nor accepted_num is passed, V1 path):
  - The first-dimension length of inputTokens, inputPositions, seqLens, slotMapping, and blockTables must equal numSeqs.
  - The shape of sampledTokenIds must be [numQueries, 1].
  - numSeqs must be greater than numQueries.
- Shape constraints (both spec_token and accepted_num are passed, V2 path):
  - The shape of inputTokens, inputPositions, seqLens, and slotMapping must be [numSeqs * (1 + specNum)], where specNum is the length of the second dimension of spec_token and must be greater than 0.
  - The shape of sampledTokenIds must be [numSeqs, (1 + specNum)], the shape of spec_token must be [numSeqs, specNum], and the shape of accepted_num must be [numSeqs].
  - The shape of blockTables must be [numSeqs, blockTablesStride], and the second-dimension length blockTablesStride must be greater than 0.
  - numSeqs must be equal to numQueries.
  - spec_token and accepted_num must be passed together or omitted together (passing an empty tensor counts as not passing); when only one of them is passed or neither is passed, the V1 path is used, and spec_token/accepted_num do not participate in the computation. When spec_token is an empty tensor (specNum is 0), the V2 path is not used. For the no-speculative-decoding scenario, evaluate aclnnAdvanceStep, but note that the V1 path requires numSeqs to be greater than numQueries, which differs from the V2 path constraint (numSeqs equals numQueries); it is not a direct equivalent replacement.
- All mandatory inputs do not support empty tensors (implied by the shape constraints above: the first dimension is not less than 1).

### User guarantee constraints (involving data content or memory layout; the host side cannot validate them, and the caller guarantees them. Violations may cause computation errors or out-of-bounds access)

- All input tensors must be contiguous, and non-contiguous tensors are not supported.
- The data values of inputTokens, sampledTokenIds, inputPositions, seqLens, slotMapping, and blockTables must be valid values consistent with the business semantics (non-negative integers, such as token IDs, sequence lengths, block numbers, and physical slots; 0 is a legal value).
- The second-dimension length of blockTables must be greater than the maximum value in seqLens divided by blockSize, ensuring that the block table can cover the maximum block index of all sequences.
- Out-of-range violations (such as negative positions or positions beyond the blockTables coverage) are out-of-contract inputs: when the lookup column index falls outside the row-local blockTables coverage ($q = pos/blockSize \notin [0, blockTablesStride)$), both the V1 and V2 paths defensively write 0 — both the block number and the remainder are masked, and no out-of-bounds GM read occurs. This behavior is defined as the public contract for the V1 path (out-of-range rows get slotMapping=0 while the other outputs are computed from the input values by the formulas, mirrored by golden; out-of-contract value-range cases can be accuracy-compared). For the V2 path, when position values exceed the int32 representable range, the kernel (int32 pipeline) and golden (int64 computation) have representation differences; value consistency is not guaranteed in that subdomain.

## Invocation Description

| Invocation method | Invocation sample                                                                   | Description                                                             |
|--------------|------------------------------------------------------------------------|----------------------------------------------------------------|
| aclnn invocation | [test_aclnn_advance_step](./examples/test_aclnn_advance_step.cpp) | Invokes the AdvanceStep operator through the [aclnnAdvanceStep](./docs/aclnnAdvanceStep_en.md) interface.    |
| aclnn invocation | [test_aclnn_advance_step_v2](./examples/test_aclnn_advance_step_v2.cpp) | Invokes the AdvanceStep operator through the [aclnnAdvanceStepV2](./docs/aclnnAdvanceStepV2_en.md) interface.    |
| Graph-mode invocation | -   | Invokes the AdvanceStep operator by building a graph through the [operator IR](./op_graph/advance_step_proto.h). |

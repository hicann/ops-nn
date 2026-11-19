# aclnnUnionSparseGen

English | [简体中文](./aclnnUnionSparseGen.md)

## Applicable Products

<!-- npu="950" id1 -->
- <term>Ascend 950PR&950DT products</term>: Supported
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3 products</term>: Supported
<!-- end id2 -->
<!-- npu="910b" id3 -->
- <term>Atlas A2 products</term>: Supported
<!-- end id3 -->
<!-- npu="310b" id4 -->
- <term>Atlas 200I/500 A2 inference products</term>: Not supported
<!-- end id4 -->
<!-- npu="310p" id5 -->
- <term>Atlas inference products</term>: Not supported
<!-- end id5 -->
<!-- npu="910" id6 -->
- <term>Atlas training products</term>: Not supported
<!-- end id6 -->

## Function

Builds a sorted unique union of the explicit KV block selections from at most `unionSize` adjacent query rows in one request. It also returns the union count and a per-KV-token bit mask that preserves each query's selections. It does not compute attention. residualBlockMode=1 inserts incomplete visible tails before union.

Let `sparseBlockIdx` have shape `[H,T,K]`: H is the number of sparse heads, T is the number of stored Q rows, and K is the block capacity per row. B is the number of requests, $W=\text{unionSize}$, and $Y=\text{blockShape}[1]$. The effective Q/KV lengths of request b are $Q_b$ and $KV_b$, determined by the length-input precedence in Constraints.

**1. Group queries within each request.** Request b is split into groups of at most W rows. Its group j has $n=\min(W,Q_b-jW)$ valid queries. Query i in this group has request-local position $q_i=jW+i$. If $A_b$ is the request's Q storage start, its input row is $r_i=A_b+q_i$.

$$
G_b=\left\lceil\frac{Q_b}{W}\right\rceil,\qquad
F_b=\sum_{a=0}^{b-1}G_a,\qquad
g=F_b+j,\qquad G=\sum_{b=0}^{B-1}G_b
$$

Here $0\le j<G_b$, $0\le i<n$, and g is the global group index in the second output dimension. A request with zero effective Q length produces no output group.

**2. Compute the union of legal blocks.** Let $L_b=\lceil KV_b/Y\rceil$. Read only the valid prefix specified by `sparseBlockCount`, retain logical block IDs in $[0,L_b)$, and remove duplicates:

$$
c_i=\min\bigl(K,\max(0,\text{sparseBlockCount}[h,r_i])\bigr)
$$

$$
S_i=\{\,d=\text{sparseBlockIdx}[h,r_i,k]\mid 0\le k<c_i,\ 0\le d<L_b\,\}
$$

Mode 0 retains legal explicit blocks. Mode 1 restores the per-query selection rule: retain explicit blocks fully within the visible range (d < floor(max(V,0)/Y)), add floor(V/Y) when V>0 and V%Y!=0, then merge and deduplicate. Added tails participate in unionSparseIdx, unionSparseCount and exactMask. V is KV-Q+localQuery+1 for maskType=1 and the effective KV length for maskType=0. Complete tails are not automatically added; existing tails are not counted twice. For Q=3, KV=5, Y=4 and empty lists, noncausal residual=1 returns [1], count=1 and mask [7,0,0,0]; causal residual=1 returns [0,1], count=2 and mask [1,1,1,0,4,0,0,0].

$$
D=\bigcup_{i=0}^{n-1}\widehat S_i,\qquad
(d_0,\ldots,d_{C-1})=\operatorname{sort}(D),\qquad C=|D|
$$

Then $\text{unionSparseCount}[h,g]=C$ and $\text{unionSparseIdx}[h,g,r]=d_r$ for $0\le r<C$. Block IDs are relative to the request's KV start, not global token positions.

**3. Preserve each query's selections.** Bit i of an INT32 corresponds to query i within the group. Combine the bits of all queries selecting a block using bitwise OR; this is the block's membership:

Let $\widehat S_i$ be the query selection after the residual rule above. The output union is $D=\bigcup_i\widehat S_i$; membership preserves each query's selections.

$$
\operatorname{membership}(d)=\bigvee_{i=0}^{n-1}\left(2^i[d\in \widehat S_i]\right)
$$

Square brackets evaluate to 1 when the condition is true and 0 otherwise; $\bigvee$ denotes bitwise OR. Repeated selections do not add weight or alter another query's bit.

**4. Expand token masks.** Token t of output block $d_r$, with $0\le t<Y$, has request-local KV position $p=Yd_r+t$. The two mask modes are:

$$
C_0(i,p)=1,\qquad
C_1(i,p)=[p<KV_b-Q_b+q_i+1]
$$

$$
\text{exactMask}[h,g,Yr+t]
=\bigvee_{i=0}^{n-1}
\left(2^i[d_r\in \widehat S_i][0\le p<KV_b]C_m(i,p)\right),\qquad m=\text{maskType}
$$

`maskType=0` preserves query membership and the effective KV boundary; it does not fill the mask with ones. `maskType=1` additionally applies bottom-right causal alignment: query i can see the first $KV_b-Q_b+q_i+1$ KV tokens. A nonpositive boundary leaves no visible tokens.

Mode 0 retains legal explicit blocks even with zero causal masks. Mode 1 filters explicit blocks by visibility and adds partial tails, so maskType and residual can affect the union and count.

**5. Output shape and capacity.**

Graph InferShape reads only host shape metadata and attributes, with no length-tensor data access or host/device synchronization. For input `[H,T,K]` (`[kvN,totalqBlockNum,maxKvBlockNum]`), outputs are `[H,G,U]`, `[H,G]`, and `[H,G,U*Y]`:

- Positive `totalq_union_num` is used as G; otherwise G=(T+7*B)//8.
- Positive `max_logical_blocks` is used as U; otherwise U=(K+residual)*union_size. Capacity is not clipped to the KV logical domain.
- B comes from length-tensor shapes. Fixed Q uses ceil(T/ceil(maxQSeqlen/block_shape[0])). With supported blockX=1, the final fixed slot may contain fewer than maxQ rows. A zero maxQ requires T=0 and retains the empty-batch/present-KV-tensor batch rules.
- The actual group count remains sum(ceil(effectiveQ/8)). Extra output groups contain idx=-1, count=0, mask=0. The caller must ensure G covers all actual groups; insufficient G is not guaranteed to raise an error and may leave valid groups unprocessed.
- U must cover every group's actual union. When the kernel detects that an actual union exceeds U, it triggers a device error instead of silently truncating that union. The error may surface during execution or stream synchronization; rejection at `aclnnUnionSparseGenGetWorkspaceSize` is not guaranteed. This check does not validate G or guarantee errors for every form of insufficient output capacity.
- The default U=(K+residual)*union_size reserves one additional tail candidate per query when residual=1; maskType=0 uses the same safe bound.

Direct ACLNN calls use G/U from `unionSparseIdx.shape[1:3]`, ignoring the two scalar shape hints `totalqUnionNum/maxLogicalBlocks`, and require matching count/mask output shapes. U is storage capacity, not an upper bound on valid block IDs. Invalid block IDs (negative or outside the request's effective KV logical domain) are filtered out rather than reported as capacity errors.

The Torch adapter retains its existing formulas: G=sum(ceil(effectiveQ/8)), L=max(1,max(ceil(effectiveKV/Y))), and U=min(8*(K+residual),L). It passes G/L and allocates outputs with that U, which ACLNN respects. Its low-level and Meta entries still require explicit G/L and do not implement GE's automatic shape rule. The CPU golden retains exact effective outputs; graph tests pad those outputs to G/U and check that no valid blocks are discarded.

## Function Prototype

This is a [two-phase API](../../../docs/en/context/two_phase_api.md). Call GetWorkspaceSize first, then execute the returned plan.

```cpp
aclnnStatus aclnnUnionSparseGenGetWorkspaceSize(
  const aclTensor* sparseBlockIdx,
  const aclTensor* sparseBlockCount,
  const aclTensor* cuSqlenQOptional,
  const aclTensor* cuSqlenKvOptional,
  const aclTensor* seqUsedQOptional,
  const aclTensor* seqUsedKvOptional,
  const aclIntArray* blockShape,
  int64_t          unionSize,
  int64_t          maskType,
  int64_t          maxQSeqlen,
  int64_t          maxKvSeqlen,
  int64_t          layoutSparsePattern,
  bool             isInputIdxSorted,
  int64_t          residualBlockMode,
  int64_t          totalqUnionNum,
  int64_t          maxLogicalBlocks,
  aclTensor*       unionSparseIdx,
  aclTensor*       unionSparseCount,
  aclTensor*       exactMask,
  uint64_t*        workspaceSize,
  aclOpExecutor**  executor);
```

```cpp
aclnnStatus aclnnUnionSparseGen(
  void*          workspace,
  uint64_t       workspaceSize,
  aclOpExecutor* executor,
  aclrtStream    stream);
```

## aclnnUnionSparseGenGetWorkspaceSize

### Parameters

<table>
<thead><tr><th>Parameter</th><th>Input/Output</th><th>Description</th><th>Usage</th><th>Data type</th><th>Format</th><th>Shape</th><th>Noncontiguous Tensor</th></tr></thead>
<tbody>
<tr><td>sparseBlockIdx</td><td>Input</td><td>Logical block IDs [H,T,K].</td><td>See Constraints.</td><td>INT32</td><td>ND</td><td>[H,T,K]</td><td>√</td></tr>
<tr><td>sparseBlockCount</td><td>Input</td><td>Valid input prefix lengths.</td><td>See Constraints.</td><td>INT32</td><td>ND</td><td>[H,T]</td><td>√</td></tr>
<tr><td>cuSqlenQOptional</td><td>Input</td><td>Optional Q storage prefix sum.</td><td>See Constraints.</td><td>INT64</td><td>ND</td><td>[B+1]</td><td>√</td></tr>
<tr><td>cuSqlenKvOptional</td><td>Input</td><td>Optional KV storage prefix sum.</td><td>See Constraints.</td><td>INT64</td><td>ND</td><td>[B+1]</td><td>√</td></tr>
<tr><td>seqUsedQOptional</td><td>Input</td><td>Optional effective Q lengths.</td><td>See Constraints.</td><td>INT32</td><td>ND</td><td>[B]</td><td>√</td></tr>
<tr><td>seqUsedKvOptional</td><td>Input</td><td>Optional effective KV lengths.</td><td>See Constraints.</td><td>INT32</td><td>ND</td><td>[B]</td><td>√</td></tr>
<tr><td>blockShape</td><td>Input</td><td>Q/KV block sizes, as a two-element aclIntArray.</td><td>See Constraints.</td><td>INT64</td><td>-</td><td>[2]</td><td>-</td></tr>
<tr><td>unionSize</td><td>Input</td><td>Number of queries per group.</td><td>See Constraints.</td><td>-</td><td>-</td><td>-</td><td>-</td></tr>
<tr><td>maskType</td><td>Input</td><td>0: noncausal; 1: bottom-right causal.</td><td>See Constraints.</td><td>-</td><td>-</td><td>-</td><td>-</td></tr>
<tr><td>maxQSeqlen</td><td>Input</td><td>Fixed Q length when both Q length tensors are absent; otherwise ignored.</td><td>See Constraints.</td><td>-</td><td>-</td><td>-</td><td>-</td></tr>
<tr><td>maxKvSeqlen</td><td>Input</td><td>Fixed KV length when both KV length tensors are absent; otherwise ignored.</td><td>See Constraints.</td><td>-</td><td>-</td><td>-</td><td>-</td></tr>
<tr><td>layoutSparsePattern</td><td>Input</td><td>Sparse pattern layout.</td><td>See Constraints.</td><td>-</td><td>-</td><td>-</td><td>-</td></tr>
<tr><td>isInputIdxSorted</td><td>Input</td><td>Promises sorted valid prefixes.</td><td>See Constraints.</td><td>-</td><td>-</td><td>-</td><td>-</td></tr>
<tr><td>residualBlockMode</td><td>Input</td><td>Explicit-only mode (0) or incomplete visible-tail mode (1).</td><td>See Constraints.</td><td>-</td><td>-</td><td>-</td><td>-</td></tr>
<tr><td>totalqUnionNum</td><td>Input</td><td>Graph group-count hint; ignored by direct ACLNN.</td><td>See Constraints.</td><td>-</td><td>-</td><td>-</td><td>-</td></tr>
<tr><td>maxLogicalBlocks</td><td>Input</td><td>Graph capacity hint; ignored by direct ACLNN.</td><td>See Constraints.</td><td>-</td><td>-</td><td>-</td><td>-</td></tr>
<tr><td>unionSparseIdx</td><td>Output</td><td>Sorted union indices, padded with -1.</td><td>See Constraints.</td><td>INT32</td><td>ND</td><td>[H,G,U]</td><td>√</td></tr>
<tr><td>unionSparseCount</td><td>Output</td><td>Number of union blocks.</td><td>See Constraints.</td><td>INT32</td><td>ND</td><td>[H,G]</td><td>√</td></tr>
<tr><td>exactMask</td><td>Output</td><td>Token-wise query membership bit mask, padded with 0.</td><td>See Constraints.</td><td>INT32</td><td>ND</td><td>[H,G,U×Y]</td><td>√</td></tr>
<tr><td>workspaceSize</td><td>Output</td><td>Required workspace bytes, returned through uint64_t*.</td><td>See Constraints.</td><td>-</td><td>-</td><td>-</td><td>-</td></tr>
<tr><td>executor</td><td>Output</td><td>Execution plan returned through aclOpExecutor**.</td><td>See Constraints.</td><td>-</td><td>-</td><td>-</td><td>-</td></tr>
</tbody></table>

The four Optional tensor pointers may be nullptr according to the length combinations in Constraints. Direct ACLNN takes G/U from the output tensor shapes.

### Return Value

| Return value | Code | Description |
|---|---|---|
| ACLNN_SUCCESS | 0 | The execution plan was created successfully. |
| ACLNN_ERR_PARAM_NULLPTR | 161001 | A required input/output, blockShape, workspaceSize or executor pointer is null. |
| ACLNN_ERR_PARAM_INVALID | 161002 | Types, formats, shapes, attributes, length combinations, fixed-slot geometry or byte extents violate the constraints. |

## aclnnUnionSparseGen

| Parameter | Input/Output | Description | Usage | Data type | Format | Shape | Noncontiguous Tensor |
|---|---|---|---|---|---|---|---|
| workspace | Input | Device workspace allocated according to the first phase. | May be null when no workspace is required. | - | - | - | - |
| workspaceSize | Input | Workspace bytes returned by the first phase. | Returned workspace size. | - | - | - | - |
| executor | Input | Executor returned by the first phase. | Non-null. | - | - | - | - |
| stream | Input | ACL device stream used for execution. | Valid ACL stream. | - | - | - | - |

Returns ACLNN_SUCCESS on successful submission, otherwise an ACLNN error code.

## Constraints

- Only blockShape=[1,Y], Y∈{4,8,16,128}, unionSize=8, maskType=0/1, layoutSparsePattern=4, isInputIdxSorted=true/false and residualBlockMode=0/1 are supported. blockShape is non-null and contains exactly two values.
- cu prefixes are INT64; all other input/output tensors are INT32 and reside on the same device. The data format is ND. K is positive; dimensions are nonnegative; all byte extents must fit INT64 without overflow.
- Noncontiguous inputs are made contiguous before computation. Noncontiguous outputs with non-overlapping elements are supported: results are copied back according to the output shape, strides and storage offset. Callers must provide valid strides and sufficient underlying storage.
- Each sparseBlockIdx row has a valid prefix bounded by sparseBlockCount in [0,K]. The prefix must be nondecreasing when isInputIdxSorted=true; false accepts unsorted input. Both allow duplicates. Block IDs are relative to each request's KV sequence. Negative and out-of-domain block IDs are ignored. Mode 0 retains legal explicit blocks. Mode 1 restores the per-query selection rule: retain explicit blocks fully within the visible range (d < floor(max(V,0)/Y)), add floor(V/Y) when V>0 and V%Y!=0, then merge and deduplicate. Added tails participate in unionSparseIdx, unionSparseCount and exactMask. V is KV-Q+localQuery+1 for maskType=1 and the effective KV length for maskType=0. Complete tails are not automatically added; existing tails are not counted twice.
- With cuQ, its shape is [B+1], values start at zero and are nondecreasing, and its last value equals T. It supplies storage offsets and capacities. Effective Q prefers seqUsedQ, otherwise adjacent cuQ differences.

Q and KV select length sources independently: seqUsed takes precedence over adjacent cu differences. If either tensor for a family is present, every int64 value of its max attribute is ignored. Only when both tensors are absent does max specify that family's fixed effective length, and max must be nonnegative.

| cuX | seqUsedX | Effective length | maxX |
|---|---|---|---|
| Present | Present | seqUsedX[b], bounded by the cuX difference | Ignored |
| Present | nullptr | cuX[b+1]−cuX[b] | Ignored |
| nullptr | Present | seqUsedX[b] | Ignored |
| nullptr | nullptr | maxX | Nonnegative fixed length |

With cuQ, Q rows use prefix offsets. With only seqUsedQ, B is its length, T must be divisible by B, slot=T/B, request b starts at b×slot, and seqUsedQ[b] must not exceed slot. Without either Q tensor, slot=maxQ, B=ceil(T/slot), request b starts at b×slot, and effectiveQ=min(slot,T-b×slot); the final slot may be partial. KV with only seqUsedKV has no Q-slot bound. Every present length tensor must describe the same B: prefixes have [B+1] elements and used lengths have [B]. When all four tensors are absent, positive maxQ determines B=ceil(T/maxQ), counting a partial final slot as one batch; maxQ=0 requires T=0 and denotes an empty batch. Present KV length tensors may instead establish a nonempty B with maxQ=0 and T=0. B=0 requires T=0 and has no actual groups; explicitly allocated extra groups are empty. Fixed KV requires ceil(maxKV/Y)≤INT32_MAX. KV need not exceed Q. The default max=-1 is legal only when ignored.

- Direct ACLNN ignores both scalar shape hints and uses matching preallocated output shapes with sufficient G/U capacity. G=0 outputs are supported.
- Callers guarantee valid device length contents and agreement with G/Lmax; ACLNN does not read back device lengths. The public Python wrapper validates CPU length values before transfer.
- aclnnUnionSparseGen defaults to a deterministic implementation. Only the lower eight query bits are used in this release. Out-of-range tail tokens and padding have zero masks.

## Example

See [compiling and running an example](../../../docs/en/context/compile_and_run_sample.md). The operator package must match the device and CANN installation.

```cpp
/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <cstdio>
#include <vector>

#include "acl/acl.h"
#include "aclnnop/aclnn_union_sparse_gen.h"

#define CHECK_RET(cond, expr) \
    do {                      \
        if (!(cond)) {        \
            expr;             \
        }                     \
    } while (0)

namespace {
int64_t ShapeSize(const std::vector<int64_t>& shape)
{
    int64_t size = 1;
    for (const int64_t dim : shape) {
        size *= dim;
    }
    return size;
}

template <typename T>
int CreateTensor(const std::vector<T>& host, const std::vector<int64_t>& shape, void** device, aclTensor** tensor)
{
    const auto bytes = static_cast<size_t>(ShapeSize(shape)) * sizeof(T);
    auto ret = aclrtMalloc(device, bytes, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = aclrtMemcpy(*device, bytes, host.data(), bytes, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = static_cast<int64_t>(shape.size()) - 2; i >= 0; --i) {
        strides[static_cast<size_t>(i)] = shape[static_cast<size_t>(i + 1)] * strides[static_cast<size_t>(i + 1)];
    }
    *tensor = aclCreateTensor(shape.data(), shape.size(), sizeof(T) == 8 ? ACL_INT64 : ACL_INT32, strides.data(), 0,
                              ACL_FORMAT_ND, shape.data(), shape.size(), *device);
    return *tensor == nullptr ? ACL_ERROR_FAILURE : ACL_SUCCESS;
}

int Run(int64_t blockY, bool fixedSlots)
{
    aclrtStream stream = nullptr;
    CHECK_RET(aclInit(nullptr) == ACL_SUCCESS, return 1);
    CHECK_RET(aclrtSetDevice(0) == ACL_SUCCESS, return 2);
    CHECK_RET(aclrtCreateStream(&stream) == ACL_SUCCESS, return 3);

    const std::vector<int64_t> sparseShape = {1, 3, 2};
    const std::vector<int64_t> countShape = {1, 3};
    const std::vector<int64_t> lengthShape = {2};
    const std::vector<int64_t> seqShape = {1};
    const std::vector<int64_t> unionShape = {1, 1, 1};
    const std::vector<int64_t> unionCountShape = {1, 1};
    const std::vector<int64_t> maskShape = {1, 1, blockY};
    const std::vector<int32_t> sparseHost = {0, -1, 0, 1, 1, -1};
    const std::vector<int32_t> countHost = {1, 2, 1};
    const std::vector<int64_t> qHost = {0, 3};
    const std::vector<int64_t> kvHost = {0, static_cast<int32_t>(blockY + 2)};
    const std::vector<int32_t> seqQHost = {2};
    const std::vector<int32_t> seqKvHost = {static_cast<int32_t>(blockY)};
    std::vector<int32_t> unionHost(ShapeSize(unionShape), 0);
    std::vector<int32_t> unionCountHost(ShapeSize(unionCountShape), 0);
    std::vector<int32_t> maskHost(ShapeSize(maskShape), 0);

    void* sparseDevice = nullptr;
    void* countDevice = nullptr;
    void* qDevice = nullptr;
    void* kvDevice = nullptr;
    void* seqQDevice = nullptr;
    void* seqKvDevice = nullptr;
    void* unionDevice = nullptr;
    void* unionCountDevice = nullptr;
    void* maskDevice = nullptr;
    aclTensor* sparse = nullptr;
    aclTensor* count = nullptr;
    aclTensor* qLengths = nullptr;
    aclTensor* kvLengths = nullptr;
    aclTensor* seqQ = nullptr;
    aclTensor* seqKv = nullptr;
    aclTensor* unionIdx = nullptr;
    aclTensor* unionCount = nullptr;
    aclTensor* exactMask = nullptr;
    CHECK_RET(CreateTensor(sparseHost, sparseShape, &sparseDevice, &sparse) == ACL_SUCCESS, return 4);
    CHECK_RET(CreateTensor(countHost, countShape, &countDevice, &count) == ACL_SUCCESS, return 4);
    CHECK_RET(CreateTensor(qHost, lengthShape, &qDevice, &qLengths) == ACL_SUCCESS, return 4);
    CHECK_RET(CreateTensor(kvHost, lengthShape, &kvDevice, &kvLengths) == ACL_SUCCESS, return 4);
    CHECK_RET(CreateTensor(seqQHost, seqShape, &seqQDevice, &seqQ) == ACL_SUCCESS, return 4);
    CHECK_RET(CreateTensor(seqKvHost, seqShape, &seqKvDevice, &seqKv) == ACL_SUCCESS, return 4);
    CHECK_RET(CreateTensor(unionHost, unionShape, &unionDevice, &unionIdx) == ACL_SUCCESS, return 4);
    CHECK_RET(CreateTensor(unionCountHost, unionCountShape, &unionCountDevice, &unionCount) == ACL_SUCCESS, return 4);
    CHECK_RET(CreateTensor(maskHost, maskShape, &maskDevice, &exactMask) == ACL_SUCCESS, return 4);

    const int64_t shapeData[] = {1, blockY};
    auto* blockShape = aclCreateIntArray(shapeData, 2);
    CHECK_RET(blockShape != nullptr, return 4);
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    // Prefix mode and fixed-slot mode have the same effective Q=2 and KV=128.
    // Without cuQ/cuKV, seqQ/seqKv supply effective lengths; the stored Q slot width is T/B and maxQSeqlen is ignored.
    auto ret = aclnnUnionSparseGenGetWorkspaceSize(
        sparse, count, fixedSlots ? nullptr : qLengths, fixedSlots ? nullptr : kvLengths, seqQ, seqKv, blockShape, 8,
        int64_t{1}, fixedSlots ? 3 : -1, -1, 4, true, 0, 1, 1, unionIdx, unionCount,
                                                   exactMask, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, return 5);
    void* workspace = nullptr;
    if (workspaceSize > 0) {
        CHECK_RET(aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST) == ACL_SUCCESS, return 6);
    }
    CHECK_RET(aclnnUnionSparseGen(workspace, workspaceSize, executor, stream) == ACL_SUCCESS, return 7);
    CHECK_RET(aclrtSynchronizeStream(stream) == ACL_SUCCESS, return 8);
    CHECK_RET(aclrtMemcpy(unionHost.data(), unionHost.size() * sizeof(int32_t), unionDevice,
                          unionHost.size() * sizeof(int32_t), ACL_MEMCPY_DEVICE_TO_HOST) == ACL_SUCCESS,
              return 9);
    CHECK_RET(aclrtMemcpy(unionCountHost.data(), unionCountHost.size() * sizeof(int32_t), unionCountDevice,
                          unionCountHost.size() * sizeof(int32_t), ACL_MEMCPY_DEVICE_TO_HOST) == ACL_SUCCESS,
              return 9);
    CHECK_RET(aclrtMemcpy(maskHost.data(), maskHost.size() * sizeof(int32_t), maskDevice,
                          maskHost.size() * sizeof(int32_t), ACL_MEMCPY_DEVICE_TO_HOST) == ACL_SUCCESS,
              return 9);
    printf("observed union_count=%d union_prefix=[%d] mask_prefix=[%d]\n", unionCountHost[0], unionHost[0],
           maskHost[0]);

    bool ok = unionCountHost[0] == 1 && unionHost[0] == 0;
    for (int64_t i = 0; i < blockY - 1; ++i) {
        ok = ok && maskHost[static_cast<size_t>(i)] == 3;
    }
    ok = ok && maskHost[static_cast<size_t>(blockY - 1)] == 2;

    aclDestroyTensor(sparse);
    aclDestroyTensor(count);
    aclDestroyTensor(qLengths);
    aclDestroyTensor(kvLengths);
    aclDestroyTensor(seqQ);
    aclDestroyTensor(seqKv);
    aclDestroyTensor(unionIdx);
    aclDestroyTensor(unionCount);
    aclDestroyTensor(exactMask);
    aclrtFree(sparseDevice);
    aclrtFree(countDevice);
    aclrtFree(qDevice);
    aclrtFree(kvDevice);
    aclrtFree(seqQDevice);
    aclrtFree(seqKvDevice);
    aclrtFree(unionDevice);
    aclrtFree(unionCountDevice);
    aclrtFree(maskDevice);
    if (workspace != nullptr) {
        aclrtFree(workspace);
    }
    aclDestroyIntArray(blockShape);
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();
    if (!ok) {
        printf("UnionSparseGen mismatch\n");
        return 10;
    }
    printf("UnionSparseGen OK: union_count=%d, union=[%d], mask_prefix=[%d]\n", unionCountHost[0], unionHost[0],
           maskHost[0]);
    return 0;
}
} // namespace

// With no argument, use prefixes. Run ./example fixed to use fixed slots instead.
int main(int argc, char**) { return Run(128, argc > 1); }

```

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

Builds a sorted unique union of the explicit KV block selections from at most `unionSize` adjacent query rows in one request. It also returns the union count and a per-KV-token bit mask that preserves each query's selections. It does not compute attention or append unselected blocks.

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

$$
D=\bigcup_{i=0}^{n-1}S_i,\qquad
(d_0,\ldots,d_{C-1})=\operatorname{sort}(D),\qquad C=|D|
$$

Then $\text{unionSparseCount}[h,g]=C$ and $\text{unionSparseIdx}[h,g,r]=d_r$ for $0\le r<C$. Block IDs are relative to the request's KV start, not global token positions.

**3. Preserve each query's selections.** Bit i of an INT32 corresponds to query i within the group. Combine the bits of all queries selecting a block using bitwise OR; this is the block's membership:

$$
\operatorname{membership}(d)=\bigvee_{i=0}^{n-1}\left(2^i[d\in S_i]\right)
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
\left(2^i[d_r\in S_i][0\le p<KV_b]C_m(i,p)\right),\qquad m=\text{maskType}
$$

`maskType=0` preserves query membership and the effective KV boundary; it does not fill the mask with ones. `maskType=1` additionally applies bottom-right causal alignment: query i can see the first $KV_b-Q_b+q_i+1$ KV tokens. A nonpositive boundary leaves no visible tokens.

The mask mode does not change the union or count. A legal selected block stays in the output list even when all of its mask values are zero.

**5. Derive output capacity and padding.** Capacity depends on effective lengths and input capacity, not the actual overlap between block lists:

$$
L_{\max}=\max\bigl(\{1\}\cup\{L_b\mid 0\le b<B\}\bigr),\qquad
U=\min(WK,L_{\max})
$$

| Output | Shape | Padding |
|---|---|---|
| unionSparseIdx | $[H,G,U]$ | Entries at or after C are -1. |
| unionSparseCount | $[H,G]$ | Stores the actual union count C. |
| exactMask | $[H,G,U\times Y]$ | The Y columns of unused blocks are zero, as are out-of-range tokens in the KV tail block. |

For Q3/KV259 and three input lists `[0,2]`, `[0]`, `[1,2]`, the output list is `[0,1,2]`, the count is 3, and the block membership values are 3, 4, and 5. Block 2 covers KV positions 256 through 383; only its first three tokens are valid:

| Mask mode | First three columns of block 2 | Remaining 125 columns |
|---|---|---|
| 0 (noncausal) | `[5,5,5]` | All zero. |
| 1 (bottom-right causal) | `[5,4,4]` | All zero. |

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
<tr><td>maxQSeqlen</td><td>Input</td><td>Effective Q upper bound or fixed Q slot width.</td><td>See Constraints.</td><td>-</td><td>-</td><td>-</td><td>-</td></tr>
<tr><td>maxKvSeqlen</td><td>Input</td><td>Reserved maximum; does not supply effective KV lengths.</td><td>See Constraints.</td><td>-</td><td>-</td><td>-</td><td>-</td></tr>
<tr><td>layoutSparsePattern</td><td>Input</td><td>Sparse pattern layout.</td><td>See Constraints.</td><td>-</td><td>-</td><td>-</td><td>-</td></tr>
<tr><td>isInputIdxSorted</td><td>Input</td><td>Promises sorted valid prefixes.</td><td>See Constraints.</td><td>-</td><td>-</td><td>-</td><td>-</td></tr>
<tr><td>residualBlockMode</td><td>Input</td><td>Explicit selected-block mode.</td><td>See Constraints.</td><td>-</td><td>-</td><td>-</td><td>-</td></tr>
<tr><td>totalqUnionNum</td><td>Input</td><td>Exact total query-group count G.</td><td>See Constraints.</td><td>-</td><td>-</td><td>-</td><td>-</td></tr>
<tr><td>maxLogicalBlocks</td><td>Input</td><td>Exact maximum logical block count Lmax.</td><td>See Constraints.</td><td>-</td><td>-</td><td>-</td><td>-</td></tr>
<tr><td>unionSparseIdx</td><td>Output</td><td>Sorted union indices, padded with -1.</td><td>See Constraints.</td><td>INT32</td><td>ND</td><td>[H,G,U]</td><td>√</td></tr>
<tr><td>unionSparseCount</td><td>Output</td><td>Number of union blocks.</td><td>See Constraints.</td><td>INT32</td><td>ND</td><td>[H,G]</td><td>√</td></tr>
<tr><td>exactMask</td><td>Output</td><td>Token-wise query membership bit mask, padded with 0.</td><td>See Constraints.</td><td>INT32</td><td>ND</td><td>[H,G,U×128]</td><td>√</td></tr>
<tr><td>workspaceSize</td><td>Output</td><td>Required workspace bytes, returned through uint64_t*.</td><td>See Constraints.</td><td>-</td><td>-</td><td>-</td><td>-</td></tr>
<tr><td>executor</td><td>Output</td><td>Execution plan returned through aclOpExecutor**.</td><td>See Constraints.</td><td>-</td><td>-</td><td>-</td><td>-</td></tr>
</tbody></table>

The four Optional tensor pointers may be nullptr according to the length combinations in Constraints. Output capacity U is derived from K, unionSize and maxLogicalBlocks.

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

- Only blockShape=[1,128], unionSize=8, maskType=0/1, layoutSparsePattern=4, isInputIdxSorted=true and residualBlockMode=0 are supported. blockShape is non-null and contains exactly two values.
- cu prefixes are INT64; all other input/output tensors are INT32 and reside on the same device. The data format is ND. K is positive; dimensions are nonnegative; all byte extents must fit INT64 without overflow.
- Noncontiguous inputs are made contiguous before computation. Noncontiguous outputs with non-overlapping elements are supported: results are copied back according to the output shape, strides and storage offset. Callers must provide valid strides and sufficient underlying storage.
- Each sparseBlockIdx row has a nondecreasing valid prefix bounded by sparseBlockCount in [0,K]. Duplicates are allowed. Block IDs are relative to each request's KV sequence. Negative and out-of-domain block IDs are ignored. No implicit block is added.
- With cuQ, its shape is [B+1], values start at zero and are nondecreasing, and its last value equals T. It supplies storage offsets and capacities. Effective Q prefers seqUsedQ, otherwise adjacent cuQ differences.
- Without cuQ, seqUsedQ [B] and positive maxQSeqlen are required. maxQSeqlen must not exceed INT32_MAX, T=B×maxQSeqlen, and request b starts at b×maxQSeqlen. Effective Q must lie in [0,maxQSeqlen]. Fixed storage does not require equal effective lengths. For B=0, T=G=0.
- KV is independent of cuQ presence. Effective KV prefers seqUsedKV [B], otherwise adjacent cuKV differences. cuKV, when present, is INT64 [B+1], starts at zero and is nondecreasing. Without cuKV, nonnegative seqUsedKV is required.
- seqUsed values cannot exceed their corresponding prefix capacities. Both max attributes must be at least -1. Positive maxQSeqlen also bounds effective Q in prefix mode. maxKvSeqlen neither supplies nor limits effective KV. KV_b need not be greater than or equal to Q_b.
- totalqUnionNum equals G, maxLogicalBlocks equals Lmax in [1,INT32_MAX], and all three preallocated outputs match the formulas above. Capacity cannot be silently truncated. Output G=0 is supported.
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
    // Without cuQ/cuKV, seqQ/seqKv supply effective lengths; maxQSeqlen=3 is the stored Q slot width.
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

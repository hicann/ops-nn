/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <algorithm>
#include <cstdint>

#include "register/op_impl_registry.h"
#include "exe_graph/runtime/tiling_context.h"
#include "op_common/op_host/util/platform_util.h"
#include "op_common/log/log.h"
#include "../../op_kernel/arch35/global_lp_pool_tiling_key.h"
#include "../../op_kernel/arch35/global_lp_pool_tiling_data.h"
#include "global_lp_pool_tiling_arch35.h"

namespace gert {
namespace tiling {

namespace {

// ─── Design Constants (arch35-specific) ───
// NOTE (per reduction paradigm): cacheLineSize MUST come from the platform
// interface (Ops::Base::GetCacheLineSize) — hard-coding is forbidden.
// On Ascend950 it is 256 B → fp32 = 64 elements, fp16/bf16 = 128 elements.
static constexpr int64_t CACHE_BUF_UB_SIZE = 16384; // 16 KB for cache tree
static constexpr int64_t INNER_A_PROD = 1;          // single A axis: no inner A bundle
static constexpr int64_t INNER_R_PROD = 1;          // single R axis: no inner R bundle

// UB config per reduction paradigm:
//   P_pre = 2 (preInBuf + preReduceResult), P_pre_ext = 1 (preReduceResultTail,
//   which doubles as the Reduce API sharedTmpBuffer), P_post = 1 (outBuf)
//   Double buffer is NOT enabled.
static constexpr int64_t P_PRE = 2;
static constexpr int64_t P_PRE_EXT = 1;
static constexpr int64_t P_POST = 1;

// ─── Math helpers ───
inline int64_t CeilDiv(int64_t a, int64_t b) { return (b == 0) ? 0 : (a + b - 1) / b; }

inline int64_t CeilAlign(int64_t a, int64_t align) { return CeilDiv(a, align) * align; }

inline int64_t FloorAlign(int64_t a, int64_t align) { return (align == 0) ? 0 : (a / align) * align; }

inline int64_t GetDtypeSize(ge::DataType dtype)
{
    switch (dtype) {
        case ge::DT_FLOAT:
            return 4;
        case ge::DT_FLOAT16:
            return 2;
        case ge::DT_BF16:
            return 2;
        default:
            return 0;
    }
}

struct TilingCtx {
    // Platform info (from Ops::Base) — all 5 via interfaces, never hard-coded
    uint64_t ubSize;
    uint32_t coreNum;
    uint32_t blockSize;
    uint32_t cacheLineSize; // Ops::Base::GetCacheLineSize → 256 on Ascend950

    // Input info
    ge::DataType dtype;
    int64_t dtypeSize;
    int64_t maxDtypeSize; // max(sizeof(D_T), sizeof(float)) — byte capacity sizing
    int64_t dimNum;
    int64_t shape[5];
    float p;

    // Empty tensor flags (set by FusedAxis)
    bool isEmptyA = false; // A==0 → EMPTY_A
    bool isEmptyR = false; // R==0 && A>0 → EMPTY_R

    // After FusedAxis: AR pattern (axisNum=2)
    int64_t A;       // axisShape[0] = N*C
    int64_t R;       // axisShape[1] = H*W or D0*D1*D2
    int64_t strideA; // axisStride[0] = R (stride between A elements in GM)
    int64_t strideR; // axisStride[1] = 1

    // After CalcTilingInfo
    int64_t aUbFactor;
    int64_t aUbFactorAlign;
    int64_t rUbFactor;
    int64_t rUbFactorAlign;
    int64_t unit;
    int64_t aTotal;
    int64_t preReduceUbSize;
    int64_t postReduceUbSize;
    int64_t tmpBufUbSize;
    int64_t aSplitChunkCnt;
    int64_t rLoopCntTotal;
    int64_t aBigCoreLoopCnt;
    int64_t aSmallCoreLoopCnt;
    int32_t aBigCoreCnt;
    int32_t usedCoreNum;
    int64_t cacheLevelStride; // = CeilAlign(aUbFactorAlign, 8)
};

// ─── 1. GetPlatformInfo ───
ge::graphStatus GetPlatformInfoFromCtx(gert::TilingContext* ctx, TilingCtx& tc)
{
    tc.ubSize = Ops::Base::GetUbSize(ctx);
    if (tc.ubSize == 0) {
        return ge::GRAPH_FAILED;
    }
    tc.coreNum = Ops::Base::GetAivCoreNum(ctx);
    if (tc.coreNum == 0) {
        return ge::GRAPH_FAILED;
    }
    tc.blockSize = Ops::Base::GetUbBlockSize(ctx);
    tc.cacheLineSize = Ops::Base::GetCacheLineSize(ctx);
    if (tc.cacheLineSize == 0) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

// ─── 2. GetShapeAttrsInfo ───
ge::graphStatus GetShapeAttrsInfo(gert::TilingContext* ctx, TilingCtx& tc)
{
    auto inputDesc = ctx->GetInputDesc(0);
    if (inputDesc == nullptr) {
        return ge::GRAPH_FAILED;
    }
    tc.dtype = inputDesc->GetDataType();
    tc.dtypeSize = GetDtypeSize(tc.dtype);
    if (tc.dtypeSize == 0) {
        return ge::GRAPH_FAILED;
    }
    // maxDtypeSize: byte-capacity sizing for buffers (they are allocated wide
    // enough for fp32 so pre/post slots can be reused across the fp32 compute
    // domain). bsElem (element rounding) uses sizeof(D_T),
    // maxDtypeSize (byte capacity) uses max(sizeof(D_T), sizeof(float)) — never mix.
    tc.maxDtypeSize = std::max(tc.dtypeSize, static_cast<int64_t>(sizeof(float)));

    auto inputShape = ctx->GetInputShape(0);
    if (inputShape == nullptr) {
        return ge::GRAPH_FAILED;
    }
    const auto& storageShape = inputShape->GetStorageShape();
    tc.dimNum = storageShape.GetDimNum();
    if (tc.dimNum != 4 && tc.dimNum != 5) {
        return ge::GRAPH_FAILED;
    }
    for (int64_t i = 0; i < tc.dimNum; ++i) {
        tc.shape[i] = storageShape.GetDim(static_cast<size_t>(i));
    }

    auto attrs = ctx->GetAttrs();
    if (attrs == nullptr) {
        return ge::GRAPH_FAILED;
    }
    const float* pPtr = attrs->GetAttrPointer<float>(0);
    if (pPtr == nullptr) {
        return ge::GRAPH_FAILED;
    }
    tc.p = *pPtr;

    return ge::GRAPH_SUCCESS;
}

// ─── 3. FusedAxis: merge consecutive A and R axes → AR (tail R) ───
ge::graphStatus FusedAxis(TilingCtx& tc)
{
    // A = N * C (merge consecutive non-reduction axes)
    // R = H * W (4D) or D0 * D1 * D2 (5D) (merge consecutive reduction axes)
    int64_t n = tc.shape[0];
    int64_t c = tc.shape[1];
    tc.A = n * c;

    int64_t r = 1;
    for (int64_t i = 2; i < tc.dimNum; ++i) {
        r *= tc.shape[i];
    }
    tc.R = r;

    // Detect empty tensor
    if (tc.A == 0) {
        tc.isEmptyA = true;
        return ge::GRAPH_SUCCESS; // handled in ComputeTiling empty branch
    }
    if (tc.R == 0) {
        tc.isEmptyR = true;       // A > 0, R == 0
        return ge::GRAPH_SUCCESS; // handled in ComputeTiling empty branch
    }

    // axisStride[0] = product of R dims = R (stride between A elements in GM)
    // axisStride[1] = 1 (innermost R stride)
    tc.strideA = tc.R;
    tc.strideR = 1;

    return ge::GRAPH_SUCCESS;
}

// ─── 4a. ComputeAUbFactor (Step 1) ───
//   cachelineTmp = FloorAlign(cacheLineSize / sizeof(D_T), bsElem)   // block aligned
//   aUbFactor    = min(cachelineTmp, totalA)
// Fractional-climb is trivial here: the operator is AR (single A axis, single R
// axis, tail-R), so the A split point is always LastA.
// tail-R never needs `aUbFactorAlign` — use aUbFactor directly.
ge::graphStatus ComputeAUbFactor(TilingCtx& tc)
{
    const int64_t bsElem = static_cast<int64_t>(tc.blockSize) / tc.dtypeSize;
    const int64_t cachelineTmp = FloorAlign(static_cast<int64_t>(tc.cacheLineSize) / tc.dtypeSize, bsElem);
    if (cachelineTmp <= 0) {
        return ge::GRAPH_FAILED;
    }
    tc.aUbFactor = std::min(cachelineTmp, tc.A);
    // tail-R: A is not burst tail, no CeilAlign needed
    tc.aUbFactorAlign = tc.aUbFactor;
    return ge::GRAPH_SUCCESS;
}

// ─── 4b. ComputeRUbFactor (Step 2) ───
//   maxDtypeSize  = max(sizeof(D_T), sizeof(float))
//   aUnit         = aUbFactor × innerAProdAlign
//   postBufSize   = CeilAlign(aUnit × maxDtypeSize, blockSize)
//   aOnlyBytes    = P_post × postBufSize
//   bytesPerRElem = (P_pre + P_pre_ext) × aUnit × maxDtypeSize
//   r_i_max       = (ubAvailable − aOnlyBytes) / bytesPerRElem
// rUbFactorAlign: burst-tail R + split axis == LastR →
//   multi-chunk: FloorAlign then rUbFactorAlign = rUbFactor
//   whole-axis : CeilAlign, and if it no longer fits the budget, fall back to
//                FloorAlign multi-chunk
ge::graphStatus ComputeRUbFactor(TilingCtx& tc)
{
    int64_t ubAvailable = static_cast<int64_t>(tc.ubSize) - CACHE_BUF_UB_SIZE;
    if (ubAvailable <= 0) {
        return ge::GRAPH_FAILED;
    }

    const int64_t aUnit = tc.aUbFactorAlign; // × innerAProdAlign (=1)
    const int64_t postBufSize = CeilAlign(aUnit * tc.maxDtypeSize, static_cast<int64_t>(tc.blockSize));
    const int64_t aOnlyBytes = P_POST * postBufSize;
    const int64_t bytesPerRElem = (P_PRE + P_PRE_EXT) * aUnit * tc.maxDtypeSize;
    if (bytesPerRElem <= 0) {
        return ge::GRAPH_FAILED;
    }
    int64_t rIMax = (ubAvailable - aOnlyBytes) / bytesPerRElem;
    if (rIMax <= 0) {
        return ge::GRAPH_FAILED;
    }

    tc.rUbFactor = std::min(rIMax, tc.R);

    const int64_t bsElem = static_cast<int64_t>(tc.blockSize) / tc.dtypeSize;
    if (tc.rUbFactor < tc.R) {
        // multi-chunk: burst tail axis needs block alignment
        tc.rUbFactor = FloorAlign(tc.rUbFactor, bsElem);
        if (tc.rUbFactor == 0) {
            return ge::GRAPH_FAILED;
        }
        tc.rUbFactorAlign = tc.rUbFactor;
    } else {
        // whole axis: CeilAlign, then verify it still fits the budget
        tc.rUbFactorAlign = CeilAlign(tc.rUbFactor, bsElem);
        if (tc.rUbFactorAlign <= rIMax) {
            // keep rUbFactor == R (whole axis, padded row stride)
        } else {
            // padding pushed it over budget → fall back to multi-chunk
            tc.rUbFactor = FloorAlign(tc.rUbFactor, bsElem);
            if (tc.rUbFactor == 0) {
                return ge::GRAPH_FAILED;
            }
            tc.rUbFactorAlign = tc.rUbFactor;
        }
    }

    return ge::GRAPH_SUCCESS;
}

// ─── 4c. ExpandAIfRFullyLoaded (Step 3) ───
//   aUnitMax = ubAvailable / (maxDtypeSize × ((P_pre+P_pre_ext) × rPaddedElems + P_post))
//   aUnitMax = min(aUnitMax, ∏A)                    // all A axes
//   aUnitMax = min(aUnitMax, cacheBufUbSize / sizeof(float))   // ★ cacheBuf hard clamp
// tail-R: NO FloorAlign after the clamp (only tail-A needs it — the A axis is
//         not the burst tail here, so a non-aligned aUbFactor is fine).
ge::graphStatus ExpandAIfRFullyLoaded(TilingCtx& tc)
{
    if (tc.rUbFactor != tc.R) {
        return ge::GRAPH_SUCCESS;
    }

    const int64_t ubAvailable = static_cast<int64_t>(tc.ubSize) - CACHE_BUF_UB_SIZE;
    const int64_t rPaddedElems = tc.rUbFactorAlign; // × innerRProdAlign (=1)
    const int64_t coeff = tc.maxDtypeSize * ((P_PRE + P_PRE_EXT) * rPaddedElems + P_POST);
    if (coeff <= 0) {
        return ge::GRAPH_FAILED;
    }

    int64_t aUnitMax = ubAvailable / coeff;
    aUnitMax = std::min(aUnitMax, tc.A);
    // ★ cacheBuf hard clamp: when R is fully resident cacheCount == 1, aUnitMax
    //   must not exceed the cacheBuf capacity, otherwise the bisection tree
    //   overflows.
    aUnitMax = std::min(aUnitMax, CACHE_BUF_UB_SIZE / static_cast<int64_t>(sizeof(float)));

    if (aUnitMax > tc.aUbFactor) {
        tc.aUbFactor = aUnitMax;
        tc.aUbFactorAlign = tc.aUbFactor; // tail-R: A is not the burst tail
    }

    return ge::GRAPH_SUCCESS;
}

// ─── cacheBuf capacity: 不设运行时校验 ───
// cacheBuf 容量约束由构造保证：
//   cacheCount × CeilAlign(aUnit, 8) × sizeof(float) ≤ cacheBufUbSize 恒成立
//   —— fp32 恒满足（aUnit ≤ cachelineTmp=64 → 最多 64 层）；
//   b16 需 rLoopCntTotal ≥ 2^32 才违反，实际不可达。
// R 全驻留（cacheCount == 1）的情形已由 Step 3 的
// `aUnitMax ≤ cacheBufUbSize / sizeof(float)` 约束（见 ExpandAIfRFullyLoaded）。

// ─── 4d. ComputeUbSizes ───
//   maxDtypeSize   = max(sizeof(D_T), sizeof(float))
//   aUnit          = aUbFactor × innerAProdAlign
//   postBufSize    = CeilAlign(aUnit × maxDtypeSize, blockSize)
//   preBufSize     = aUnit × rUbFactorAlign × innerRProdAlign × maxDtypeSize
//   cacheBufUbSize = 16 × 1024
// NOTE: the paradigm defines NO VL padding term. Buffers are sized with
// maxDtypeSize (fp32 width) so pre/post slots can be reused across the fp32
// compute domain; that headroom is what absorbs the full-VL LoadAlign over-read.
void ComputeUbSizes(TilingCtx& tc)
{
    tc.unit = tc.aUbFactorAlign * tc.rUbFactorAlign;
    tc.aTotal = tc.aUbFactorAlign;

    // preBufSize is a 2D dense buffer: [rows × rowstride] × elemBytes
    tc.preReduceUbSize = tc.unit * tc.maxDtypeSize;
    tc.postReduceUbSize = CeilAlign(tc.aTotal * tc.maxDtypeSize, static_cast<int64_t>(tc.blockSize));
    // All P_pre + P_pre_ext pre-phase slots share preBufSize (same physical size).
    tc.tmpBufUbSize = tc.preReduceUbSize;

    // cacheLevelStride: 32B-aligned per-layer stride for cache tree (fp32 lanes)
    //   When aTotal=1, must CeilAlign to 8 to satisfy VF LoadAlign 32B-alignment
    static constexpr int64_t BLOCK_F32 = 32 / static_cast<int64_t>(sizeof(float)); // = 8
    tc.cacheLevelStride = CeilAlign(tc.aTotal, BLOCK_F32);
}

// ─── 4e. ComputeFusedALoopSplit ───
void ComputeFusedALoopSplit(TilingCtx& tc)
{
    tc.aSplitChunkCnt = CeilDiv(tc.A, tc.aUbFactor);
    tc.rLoopCntTotal = CeilDiv(tc.R, tc.rUbFactor);

    int64_t aLoopCntTotal = tc.aSplitChunkCnt;
    int64_t cn = static_cast<int64_t>(tc.coreNum);
    tc.aSmallCoreLoopCnt = aLoopCntTotal / cn;
    tc.aBigCoreCnt = static_cast<int32_t>(aLoopCntTotal % cn);
    tc.aBigCoreLoopCnt = tc.aSmallCoreLoopCnt + (tc.aBigCoreCnt > 0 ? 1 : 0);
    tc.usedCoreNum = (tc.aSmallCoreLoopCnt > 0) ? static_cast<int32_t>(cn) : tc.aBigCoreCnt;
}

// ─── 5. FillAndLogTilingData ───
ge::graphStatus FillAndLogTilingData(gert::TilingContext* ctx, const TilingCtx& tc)
{
    auto tilingData = ctx->GetTilingData<GlobalLpPoolTilingData>();
    if (tilingData == nullptr) {
        return ge::GRAPH_FAILED;
    }

    // Pattern: axisNum=2, AR
    tilingData->axisNum = 2;
    tilingData->axisShape[0] = tc.A;
    tilingData->axisShape[1] = tc.R;
    for (int32_t i = 2; i < 9; ++i) {
        tilingData->axisShape[i] = 1;
    }
    tilingData->axisStride[0] = tc.strideA;
    tilingData->axisStride[1] = tc.strideR;
    for (int32_t i = 2; i < 9; ++i) {
        tilingData->axisStride[i] = 0;
    }

    // Multi-core split
    tilingData->aLoopCntTotal = tc.aSplitChunkCnt;
    tilingData->aSplitChunkCnt = tc.aSplitChunkCnt;
    tilingData->aBigCoreLoopCnt = tc.aBigCoreLoopCnt;
    tilingData->aSmallCoreLoopCnt = tc.aSmallCoreLoopCnt;
    tilingData->aBigCoreCnt = tc.aBigCoreCnt;
    tilingData->usedCoreNum = tc.usedCoreNum;

    // UB split
    tilingData->aSplitAxisIdx = 0;
    tilingData->rSplitAxisIdx = 1;
    tilingData->aUbFactor = tc.aUbFactor;
    tilingData->aUbFactorAlign = tc.aUbFactorAlign;
    tilingData->rUbFactor = tc.rUbFactor;
    tilingData->rUbFactorAlign = tc.rUbFactorAlign;
    tilingData->innerAProd = INNER_A_PROD;
    tilingData->innerAProdAlign = INNER_A_PROD;
    tilingData->innerRProd = INNER_R_PROD;
    tilingData->innerRProdAlign = INNER_R_PROD;

    // R loop
    tilingData->rLoopCntTotal = tc.rLoopCntTotal;

    // UB buffer sizes (bytes, block-aligned)
    tilingData->preReduceUbSize = tc.preReduceUbSize;
    tilingData->postReduceUbSize = tc.postReduceUbSize;
    tilingData->tmpBufUbSize = tc.tmpBufUbSize;
    tilingData->cacheBufUbSize = CACHE_BUF_UB_SIZE;
    tilingData->cacheLevelStride = tc.cacheLevelStride;
    tilingData->is_fast_path = 1; // always fast path for this op

    // Operator-specific
    tilingData->p = tc.p;
    tilingData->invP = (tc.p != 0.0f) ? (1.0f / tc.p) : 0.0f;

    OP_LOGD(ctx->GetNodeName(),
            "TILING: A=%ld R=%ld aUb=%ld/%ld rUb=%ld/%ld "
            "cores=%d used=%d preUb=%ld postUb=%ld tmpUb=%ld cacheLvl=%ld p=%.1f invP=%.3f",
            tc.A, tc.R, tc.aUbFactor, tc.aUbFactorAlign, tc.rUbFactor, tc.rUbFactorAlign, (int)tc.coreNum,
            tc.usedCoreNum, tc.preReduceUbSize, tc.postReduceUbSize, tc.tmpBufUbSize, tc.cacheLevelStride, tc.p,
            (tc.p != 0.0f) ? (1.0f / tc.p) : 0.0f);

    return ge::GRAPH_SUCCESS;
}

// ─── FillEmpty* : EMPTY_A / EMPTY_R tiling data fillers ───
// FillEmptyTilingCommon fills the fields shared by both empty paths;
// FillEmptyATilingData / FillEmptyRTilingData append the EMPTY_A / EMPTY_R
// specific fields respectively.
static void FillEmptyTilingCommon(gert::TilingContext* ctx, TilingCtx& tc)
{
    auto td = ctx->GetTilingData<GlobalLpPoolTilingData>();
    if (td == nullptr)
        return;
    td->axisNum = 2;
    td->axisShape[0] = tc.A;
    td->axisShape[1] = tc.R;
    for (int32_t i = 2; i < 9; ++i)
        td->axisShape[i] = 1;
    td->axisStride[0] = (tc.R > 0) ? tc.R : 1;
    td->axisStride[1] = 1;
    for (int32_t i = 2; i < 9; ++i)
        td->axisStride[i] = 0;
    td->aSplitAxisIdx = 0;
    td->rSplitAxisIdx = 1;
    td->innerAProd = INNER_A_PROD;
    td->innerAProdAlign = INNER_A_PROD;
    td->innerRProd = INNER_R_PROD;
    td->innerRProdAlign = INNER_R_PROD;
    td->p = tc.p;
    td->invP = (tc.p != 0.0f) ? (1.0f / tc.p) : 0.0f;
}

static void FillEmptyATilingData(gert::TilingContext* ctx, TilingCtx& tc)
{
    FillEmptyTilingCommon(ctx, tc);
    auto td = ctx->GetTilingData<GlobalLpPoolTilingData>();
    if (td == nullptr)
        return;
    td->usedCoreNum = 0;
    td->aLoopCntTotal = 0;
    td->aSplitChunkCnt = 0;
    td->aBigCoreLoopCnt = 0;
    td->aSmallCoreLoopCnt = 0;
    td->aBigCoreCnt = 0;
    td->aUbFactor = 0;
    td->aUbFactorAlign = 0;
    td->rUbFactor = 0;
    td->rUbFactorAlign = 0;
    td->rLoopCntTotal = 0;
    td->preReduceUbSize = 0;
    td->postReduceUbSize = 0;
    td->tmpBufUbSize = 0;
    td->cacheBufUbSize = CACHE_BUF_UB_SIZE;
    td->cacheLevelStride = 8;
    td->is_fast_path = 1;
}

static void FillEmptyRTilingData(gert::TilingContext* ctx, TilingCtx& tc)
{
    FillEmptyTilingCommon(ctx, tc);
    auto td = ctx->GetTilingData<GlobalLpPoolTilingData>();
    if (td == nullptr)
        return;
    int64_t totalA = tc.A;
    // Same cachelineTmp rule as the normal path (platform cacheLineSize, never hard-coded)
    const int64_t bsElem = static_cast<int64_t>(tc.blockSize) / tc.dtypeSize;
    const int64_t maxInitElem = FloorAlign(static_cast<int64_t>(tc.cacheLineSize) / tc.dtypeSize, bsElem);
    int64_t atmp = std::min(maxInitElem, CeilDiv(totalA, static_cast<int64_t>(tc.coreNum)));
    int64_t aUb = std::min(atmp, totalA);
    int64_t aChunks = CeilDiv(totalA, aUb);
    int64_t cn = static_cast<int64_t>(tc.coreNum);
    int64_t smallLoop = aChunks / cn;
    int32_t bigCnt = static_cast<int32_t>(aChunks % cn);
    int64_t bigLoop = smallLoop + (bigCnt > 0 ? 1 : 0);
    int32_t usedCore = (smallLoop > 0) ? static_cast<int32_t>(cn) : bigCnt;
    int64_t postUb = CeilAlign(aUb * tc.maxDtypeSize, static_cast<int64_t>(tc.blockSize));
    static constexpr int64_t BLOCK_F32 = 32 / static_cast<int64_t>(sizeof(float));
    td->usedCoreNum = usedCore;
    td->aLoopCntTotal = aChunks;
    td->aSplitChunkCnt = aChunks;
    td->aBigCoreLoopCnt = bigLoop;
    td->aSmallCoreLoopCnt = smallLoop;
    td->aBigCoreCnt = bigCnt;
    td->aUbFactor = aUb;
    td->aUbFactorAlign = aUb;
    td->rUbFactor = 0;
    td->rUbFactorAlign = 0;
    td->rLoopCntTotal = 0;
    td->preReduceUbSize = postUb;
    td->postReduceUbSize = postUb;
    td->tmpBufUbSize = 0;
    td->cacheBufUbSize = CACHE_BUF_UB_SIZE;
    td->cacheLevelStride = CeilAlign(aUb, BLOCK_F32);
}

} // anonymous namespace

// ─── ComputeNormalTiling: extracted from ComputeTiling for size control ──
static ge::graphStatus ComputeNormalTiling(gert::TilingContext* ctx, TilingCtx& tc)
{
    if (ComputeAUbFactor(tc) != ge::GRAPH_SUCCESS)
        return ge::GRAPH_FAILED;
    if (ComputeRUbFactor(tc) != ge::GRAPH_SUCCESS)
        return ge::GRAPH_FAILED;
    if (ExpandAIfRFullyLoaded(tc) != ge::GRAPH_SUCCESS)
        return ge::GRAPH_FAILED;
    ComputeUbSizes(tc);
    ComputeFusedALoopSplit(tc);
    // No cacheBuf capacity re-check needed: the constraint holds by construction
    // (cacheCount × CeilAlign(aUnit,8) × 4 ≤ 16KB 恒成立).
    if (FillAndLogTilingData(ctx, tc) != ge::GRAPH_SUCCESS)
        return ge::GRAPH_FAILED;
    const uint64_t tilingKey = GET_TPL_TILING_KEY(0, 1);
    if (ctx->SetTilingKey(tilingKey) != ge::GRAPH_SUCCESS)
        return ge::GRAPH_FAILED;
    if (ctx->SetBlockDim(static_cast<uint32_t>(tc.usedCoreNum)) != ge::GRAPH_SUCCESS)
        return ge::GRAPH_FAILED;
    return ge::GRAPH_SUCCESS;
}

// ─── ComputeEmptyTiling: EMPTY_A or EMPTY_R path ──
static ge::graphStatus ComputeEmptyTiling(gert::TilingContext* ctx, TilingCtx& tc)
{
    if (tc.isEmptyA) {
        FillEmptyATilingData(ctx, tc);
    } else {
        FillEmptyRTilingData(ctx, tc);
    }
    const uint64_t tilingKey = GET_TPL_TILING_KEY(1, 0);
    if (ctx->SetTilingKey(tilingKey) != ge::GRAPH_SUCCESS)
        return ge::GRAPH_FAILED;
    auto td = ctx->GetTilingData<GlobalLpPoolTilingData>();
    uint32_t blockDim = (td != nullptr) ? static_cast<uint32_t>(td->usedCoreNum) : 0;
    // 空 tensor 场景若使用核数为 0，也需要 SetBlockDim(1)，
    //    kernel 侧通过 usedCoreNum=0 短路处理。
    if (blockDim == 0) {
        blockDim = 1;
    }
    if (ctx->SetBlockDim(blockDim) != ge::GRAPH_SUCCESS)
        return ge::GRAPH_FAILED;
    return ge::GRAPH_SUCCESS;
}

// ─── 6. SetWorkspaceSize ───
//   即使不需要使用 workspace，也需要显式设置
//   ws[0] = sysWorkspaceSize
// Base template uses no user workspace, so the total is just the framework's
// libapi workspace. (Group template would add usrWorkspaceBytes.)
static void SetWorkspaceSize(gert::TilingContext* ctx)
{
    size_t* ws = ctx->GetWorkspaceSizes(1);
    OP_CHECK_IF(ws == nullptr, OP_LOGE("GlobalLpPool", "workspace ptr is nullptr"), return);
    ws[0] = Ops::Base::GetWorkspaceSize(ctx);
}

// ─── Main entry: ComputeTiling ───
ge::graphStatus GlobalLpPoolTilingFunc::ComputeTiling(gert::TilingContext* ctx)
{
    if (ctx == nullptr)
        return ge::GRAPH_FAILED;
    TilingCtx tc;
    SetWorkspaceSize(ctx);
    if (GetPlatformInfoFromCtx(ctx, tc) != ge::GRAPH_SUCCESS)
        return ge::GRAPH_FAILED;
    if (GetShapeAttrsInfo(ctx, tc) != ge::GRAPH_SUCCESS)
        return ge::GRAPH_FAILED;
    if (FusedAxis(tc) != ge::GRAPH_SUCCESS)
        return ge::GRAPH_FAILED;
    if (tc.isEmptyA || tc.isEmptyR)
        return ComputeEmptyTiling(ctx, tc);
    return ComputeNormalTiling(ctx, tc);
}

} // namespace tiling

static ge::graphStatus TilingParseForGlobalLpPool([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

struct GlobalLpPoolCompileInfo {}; // 必须定义, 入图场景依赖

} // namespace gert

IMPL_OP_OPTILING(GlobalLpPool)
    .Tiling(gert::tiling::GlobalLpPoolTilingFunc::ComputeTiling)
    .TilingParse<gert::GlobalLpPoolCompileInfo>(gert::TilingParseForGlobalLpPool);

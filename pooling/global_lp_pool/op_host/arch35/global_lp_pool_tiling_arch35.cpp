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
// MAX_INNER_A_INIT = cacheLineSize on Ascend950/arch35 = 512 bytes.
// Used to compute max initial A elements: MAX_INNER_A_INIT_ELEM = 512 / dtypeSize.
static constexpr int64_t MAX_INNER_A_INIT = 512;
static constexpr int64_t CACHE_BUF_UB_SIZE = 16384; // 16 KB for cache tree
static constexpr int64_t INNER_A_PROD = 1;          // single A axis: no inner A bundle
static constexpr int64_t INNER_R_PROD = 1;          // single R axis: no inner R bundle

// UB config per design §6.1
static constexpr int64_t N_PRE_REUSE = 1;    // 1 preIn reuse slot
static constexpr int64_t N_PRE_PARALLEL = 0; // 0 parallel slots
static constexpr int64_t N_OUT = 1;          // 1 output buffer
static constexpr int64_t N_POST_EXTRA = 0;   // 0 extra post buffers
static constexpr int64_t DB = 2;             // double buffer depth for preIn

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
    // Platform info (from Ops::Base)
    uint64_t ubSize;
    uint32_t coreNum;
    uint32_t blockSize;

    // Input info
    ge::DataType dtype;
    int64_t dtypeSize;
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

    // Detect empty tensor (ref: reduce-normal-pattern-design §1.1)
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

// ─── 4a. CalcAUbFactor (Step 1) ───
ge::graphStatus CalcAUbFactor(TilingCtx& tc)
{
    int64_t totalA = tc.A;
    // MAX_INNER_A_INIT_ELEM = MAX_INNER_A_INIT / dtypeSize
    int64_t maxInitElem = MAX_INNER_A_INIT / tc.dtypeSize;
    int64_t atmp = std::min(maxInitElem, CeilDiv(totalA, static_cast<int64_t>(tc.coreNum)));
    tc.aUbFactor = std::min(atmp, totalA);
    // tail-R: A is not burst tail, no CeilAlign needed
    tc.aUbFactorAlign = tc.aUbFactor;
    return ge::GRAPH_SUCCESS;
}

// ─── 4b. CalcRFromBudget (Step 2) ───
ge::graphStatus CalcRFromBudget(TilingCtx& tc)
{
    int64_t ubAvailable = static_cast<int64_t>(tc.ubSize) - CACHE_BUF_UB_SIZE;
    if (ubAvailable <= 0) {
        return ge::GRAPH_FAILED;
    }

    int64_t aUnit = tc.aUbFactorAlign; // aUbFactorAlign × innerAProdAlign (=1)
    int64_t bytesPerRElem = (DB * N_PRE_REUSE + N_PRE_PARALLEL) * tc.dtypeSize +
                            2 * static_cast<int64_t>(sizeof(float));
    int64_t aOnlyBytes = (N_OUT + N_POST_EXTRA) * tc.dtypeSize * tc.aUbFactorAlign; // padded A size
    int64_t rIMax = (ubAvailable - aOnlyBytes) / (aUnit * bytesPerRElem);
    if (rIMax <= 0) {
        return ge::GRAPH_FAILED;
    }

    tc.rUbFactor = std::min(rIMax, tc.R);

    // If multi-chunk: FloorAlign for burst tail alignment
    int64_t bsElem = static_cast<int64_t>(tc.blockSize) / tc.dtypeSize;
    if (tc.rUbFactor < tc.R) {
        tc.rUbFactor = FloorAlign(tc.rUbFactor, bsElem);
        if (tc.rUbFactor == 0) {
            return ge::GRAPH_FAILED;
        }
    }

    // rSplit==LastR && isTailR: rUbFactorAlign must be CeilAlign'd
    tc.rUbFactorAlign = CeilAlign(tc.rUbFactor, bsElem);

    return ge::GRAPH_SUCCESS;
}

// ─── 4c. TryExpandA (Step 3) ───
ge::graphStatus TryExpandA(TilingCtx& tc)
{
    if (tc.rUbFactor != tc.R) {
        return ge::GRAPH_SUCCESS;
    }

    int64_t ubAvailable = static_cast<int64_t>(tc.ubSize) - CACHE_BUF_UB_SIZE;
    int64_t totalA = tc.A;
    int64_t bytesPerRElem = (DB * N_PRE_REUSE + N_PRE_PARALLEL) * tc.dtypeSize +
                            2 * static_cast<int64_t>(sizeof(float));

    // Solve aUbFactor from UB budget (R fully loaded, fixed rUbFactorAlign)
    int64_t atmpNew = ubAvailable / (bytesPerRElem * tc.rUbFactorAlign + (N_OUT + N_POST_EXTRA) * tc.dtypeSize);
    atmpNew = std::min(atmpNew, totalA);
    atmpNew = std::min(atmpNew, CACHE_BUF_UB_SIZE / static_cast<int64_t>(sizeof(float)));

    if (atmpNew > tc.aUbFactor) {
        // tail-R (isTailR=true) + aSplit==LastA → A is NOT burst tail
        // → aUbFactor is chunk count (not burst tail axis), no FloorAlign needed
        //   (ref: reduce-normal-pattern-design §5.3: "非 burst-tail 切分轴不 FloorAlign")
        tc.aUbFactor = atmpNew; // not FloorAlign'd for tail-R
        // aUbFactorAlign = aUbFactor (tail-R, A not burst tail, no CeilAlign)
        tc.aUbFactorAlign = tc.aUbFactor;
    }

    return ge::GRAPH_SUCCESS;
}

// ─── Helper: Shrink aUbFactor and recalc multi-core split ─────────────
// Extracted from CheckCacheBufConstraint to reduce function size.
// When the cache tree exceeds cacheBufUbSize, this shrinks aUbFactor
// and recalculates all dependent buffer sizes and multi-core split.
// Returns false if even 1 block won't fit (→ group template).
static bool ShrinkAUbAndRecalc(TilingCtx& tc, int64_t totalLayers, int64_t cacheBudget)
{
    int64_t bsF32 = 32 / static_cast<int64_t>(sizeof(float));
    int64_t maxAElems = cacheBudget / (totalLayers * static_cast<int64_t>(sizeof(float)));
    maxAElems = FloorAlign(maxAElems, bsF32);
    if (maxAElems <= 0) {
        return false;
    }
    tc.aUbFactor = std::min(maxAElems, tc.A);
    tc.aUbFactorAlign = tc.aUbFactor;
    tc.unit = tc.aUbFactorAlign * tc.rUbFactorAlign;
    tc.aTotal = tc.aUbFactorAlign;
    static constexpr int64_t VL_BYTES_B16 = 128;
    static constexpr int64_t VL_BYTES_F32 = 256;
    tc.preReduceUbSize = CeilAlign(tc.unit * tc.dtypeSize, static_cast<int64_t>(tc.blockSize)) + VL_BYTES_B16;
    tc.postReduceUbSize = CeilAlign(tc.aTotal * tc.dtypeSize, static_cast<int64_t>(tc.blockSize));
    tc.tmpBufUbSize = CeilAlign(tc.unit * static_cast<int64_t>(sizeof(float)), static_cast<int64_t>(tc.blockSize)) +
                      VL_BYTES_F32;
    static constexpr int64_t BLOCK_F32 = 32 / static_cast<int64_t>(sizeof(float));
    tc.cacheLevelStride = CeilAlign(tc.aTotal, BLOCK_F32);
    tc.aSplitChunkCnt = CeilDiv(tc.A, tc.aUbFactor);
    int64_t aLoopCntTotal = tc.aSplitChunkCnt;
    int64_t cn = static_cast<int64_t>(tc.coreNum);
    tc.aSmallCoreLoopCnt = aLoopCntTotal / cn;
    tc.aBigCoreCnt = static_cast<int32_t>(aLoopCntTotal % cn);
    tc.aBigCoreLoopCnt = tc.aSmallCoreLoopCnt + (tc.aBigCoreCnt > 0 ? 1 : 0);
    tc.usedCoreNum = (tc.aSmallCoreLoopCnt > 0) ? static_cast<int32_t>(cn) : tc.aBigCoreCnt;
    return true;
}

// ─── 4c-bis. CheckCacheBufConstraint (needs_bisection=true) ───
ge::graphStatus CheckCacheBufConstraint(TilingCtx& tc)
{
    if (tc.rLoopCntTotal <= 1) {
        return ge::GRAPH_SUCCESS;
    }
    // Host-side FindNearestPower2 and CalLog2 (no AscendC intrinsics)
    auto hpFindNearestPower2 = [](int64_t v) -> int64_t {
        if (v <= 0)
            return 0;
        if (v <= 2)
            return 1;
        if (v <= 4)
            return 2;
        int64_t n = v - 1, p = 0;
        while (n > 0) {
            n >>= 1;
            p++;
        }
        return static_cast<int64_t>(1) << (p - 1);
    };
    auto hpCalLog2 = [](int64_t v) -> int64_t {
        int64_t r = 0;
        while (v > 1) {
            v >>= 1;
            r++;
        }
        return r;
    };
    int64_t bisectionPos = hpFindNearestPower2(tc.rLoopCntTotal);
    int64_t cacheCount = hpCalLog2(bisectionPos) + 1;
    int64_t totalLayers = cacheCount + 1;
    int64_t perLayerBytes = tc.cacheLevelStride * static_cast<int64_t>(sizeof(float));
    if (totalLayers * perLayerBytes > static_cast<int64_t>(CACHE_BUF_UB_SIZE)) {
        if (!ShrinkAUbAndRecalc(tc, totalLayers, CACHE_BUF_UB_SIZE)) {
            return ge::GRAPH_FAILED;
        }
    }
    return ge::GRAPH_SUCCESS;
}

// ─── 4d. CalcBufferSizes ───
void CalcBufferSizes(TilingCtx& tc)
{
    tc.unit = tc.aUbFactorAlign * tc.rUbFactorAlign;
    tc.aTotal = tc.aUbFactorAlign;

    // ─── VF LoadAlign over-read guard ────────────────────────────────
    // Reg::LoadAlign<float> always accesses 256B (64 fp32 lanes) and
    // DIST_UNPACK_B16 accesses 128B (64 half lanes) regardless of mask.
    // If totalF32 < 64, the last VF iteration LoadAligns beyond the
    // allocated buffer → VEC_ERROR.
    //
    // Fix: add 1 VL of safety padding to preIn (+128B for DIST_UNPACK_B16)
    // and tmpBuf (+256B for fp32 LoadAlign) so the last LoadAlign stays
    // within the allocated region. Does NOT change rUbFactorAlign (row stride),
    // so data layout is unaffected.
    static constexpr int64_t VL_BYTES_F32 = 256; // 1 VL for fp32 LoadAlign
    static constexpr int64_t VL_BYTES_B16 = 128; // 1 VL/2 for DIST_UNPACK_B16
    tc.preReduceUbSize = CeilAlign(tc.unit * tc.dtypeSize, static_cast<int64_t>(tc.blockSize)) + VL_BYTES_B16;
    tc.postReduceUbSize = CeilAlign(tc.aTotal * tc.dtypeSize, static_cast<int64_t>(tc.blockSize));
    tc.tmpBufUbSize = CeilAlign(tc.unit * static_cast<int64_t>(sizeof(float)), static_cast<int64_t>(tc.blockSize)) +
                      VL_BYTES_F32;

    // cacheLevelStride: 32B-aligned per-layer stride for cache tree (fp32 lanes)
    //   When aTotal=1, must CeilAlign to 8 to satisfy VF LoadAlign 32B-alignment
    static constexpr int64_t BLOCK_F32 = 32 / static_cast<int64_t>(sizeof(float)); // = 8
    tc.cacheLevelStride = CeilAlign(tc.aTotal, BLOCK_F32);
}

// ─── 4e. CalcMultiCoreSplit ───
void CalcMultiCoreSplit(TilingCtx& tc)
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

// ─── 5. SetAndPrintTilingData ───
ge::graphStatus SetAndPrintTilingData(gert::TilingContext* ctx, const TilingCtx& tc)
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
    int64_t maxInitElem = MAX_INNER_A_INIT / tc.dtypeSize;
    int64_t atmp = std::min(maxInitElem, CeilDiv(totalA, static_cast<int64_t>(tc.coreNum)));
    int64_t aUb = std::min(atmp, totalA);
    int64_t aChunks = CeilDiv(totalA, aUb);
    int64_t cn = static_cast<int64_t>(tc.coreNum);
    int64_t smallLoop = aChunks / cn;
    int32_t bigCnt = static_cast<int32_t>(aChunks % cn);
    int64_t bigLoop = smallLoop + (bigCnt > 0 ? 1 : 0);
    int32_t usedCore = (smallLoop > 0) ? static_cast<int32_t>(cn) : bigCnt;
    int64_t postUb = CeilAlign(aUb * tc.dtypeSize, static_cast<int64_t>(tc.blockSize));
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
    if (CalcAUbFactor(tc) != ge::GRAPH_SUCCESS)
        return ge::GRAPH_FAILED;
    if (CalcRFromBudget(tc) != ge::GRAPH_SUCCESS)
        return ge::GRAPH_FAILED;
    if (TryExpandA(tc) != ge::GRAPH_SUCCESS)
        return ge::GRAPH_FAILED;
    CalcBufferSizes(tc);
    CalcMultiCoreSplit(tc);
    if (CheckCacheBufConstraint(tc) != ge::GRAPH_SUCCESS)
        return ge::GRAPH_FAILED;
    if (SetAndPrintTilingData(ctx, tc) != ge::GRAPH_SUCCESS)
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
    if (ctx->SetBlockDim(blockDim) != ge::GRAPH_SUCCESS)
        return ge::GRAPH_FAILED;
    return ge::GRAPH_SUCCESS;
}

// ─── Main entry: ComputeTiling ───
ge::graphStatus GlobalLpPoolTilingFunc::ComputeTiling(gert::TilingContext* ctx)
{
    if (ctx == nullptr)
        return ge::GRAPH_FAILED;
    TilingCtx tc;
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

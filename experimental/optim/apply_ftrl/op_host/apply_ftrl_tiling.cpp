/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * NOTE: Portions of this code were AI-generated and have been
 * technically reviewed for functional accuracy and security
 */

/*!
 * \file apply_ftrl_tiling.cpp
 * \brief ApplyFtrl tiling (ascend910b / DAV_2201, single Elementwise strategy).
 *
 * Strategy (DESIGN s3.3):
 *   1. Multi-core: split total elements evenly across AI vector cores,
 *      blockFactor = CeilAlign(CeilDiv(total, coreNum), ubBlockSize) so
 *      neighbouring cores' GM regions never overlap (32B DMA granularity).
 *   2. UB: reserve SELECT_TMP_BYTES (8KB, Select mode1/2 framework tmp, MED-1)
 *      first, then size the per-tile element count by a conservative
 *      UB_FP32_SLOTS budget, capped at TILE_ELEM_NUM_TARGET (2048) and aligned
 *      DOWN to 64 elements (256B, MED-2) so CompareScalar/Select satisfy the
 *      256B count rule.
 *   3. TilingKey via ASCENDC_TPL_SEL_PARAM(dTypeVar, padTail, hasL1).  HAS_L1 is
 *      forced to 1 (host cannot read the Device GM scalar l1 at tiling time); the
 *      kernel takes a runtime fast-path when l1 == 0.  The HAS_L1=0 binary is
 *      still produced so UT can drive it directly.
 */

#include "register/op_def_registry.h"
#include "op_common/log/log.h"
#include "op_common/op_host/util/math_util.h"
#include "op_common/op_host/util/platform_util.h"
#include "../op_kernel/apply_ftrl_tiling_data.h"
#include "../op_kernel/apply_ftrl_tiling_key.h"

namespace optiling {

using Ops::Base::CeilAlign;
using Ops::Base::CeilDiv;
using Ops::Base::FloorAlign;
using Ops::Base::FloorDiv;
using Ops::Base::GetUbBlockSize;

// User workspace (GM scratch) request, in bytes.  This is a UB-only Elementwise
// kernel: all compute stays in UB and the kernel entry's `GM_ADDR workspace`
// formal is never touched by Init/Process, so the op needs NO user workspace ->
// 0.  (Any system-reserved workspace baseline is appended by the runtime/
// framework independently of this user request; it is a separate concept.  The
// 16MB SYS_WORKSPACE figure in mainline arch35/baseline tiling does not apply to
// this ascend910b registry-invoke implementation -- see DESIGN s3.8 / MED-A.)
constexpr uint32_t USER_WORKSPACE_SIZE = 0U;
constexpr int64_t TYPE_SIZE_FP32 = 4; // internal compute dtype is fp32.
// CompareScalar / Select count must be 256B aligned -> 256B / sizeof(fp32) = 64.
constexpr int64_t kCmpAlignElem = 256 / TYPE_SIZE_FP32;
// Per-tile target element count (256B aligned: 2048 = 32 * 64).
constexpr int64_t TILE_ELEM_NUM_TARGET = 2048;
// Select mode1/2 reserves an 8KB UB framework tmp region (MED-1).
constexpr int64_t SELECT_TMP_BYTES = 8192;
// Conservative per-element fp32-equivalent UB slots.  MUST stay in sync with the
// kernel-side constant `kUbFp32Slots` in op_kernel/apply_ftrl_kernel.h.  The fp32 path
// (tightest) uses ~20.25 slots/elem (IN 8*4 + OUT 6*4 + scratch 24 + mask 1 = 81B
// / 4B); 24 keeps headroom.  Update both together when changing the UB layout.
constexpr int64_t UB_FP32_SLOTS = 24;
// Adaptive block-dim floor (PERF, small/medium shapes): minimum elements assigned
// to one AI vector core.  The plain even split CeilAlign(CeilDiv(total, coreNum), ..)
// spawns up to coreNum cores even for tiny tensors, so every core pays the fixed
// per-core overhead (kernel launch + the four scalar GM LoadScalar reads + buffer
// init + multi-core scheduling) while doing trivial compute -> the head overhead
// dominates and the kernel runs no faster (often slower) than on few cores.  Raising
// blockFactor to this floor reduces usedCoreNum for small/medium shapes so each core
// has enough work to amortise that fixed cost; the FTRL math, dtype and the three
// in-place outputs are unchanged (purely a multi-core split decision -> no numerical
// effect).  Empirically tuned on ascend910b (DAV_2201) via an msprof-op block-dim
// sweep (kernel Task Duration, warm-up=10, launch-count=10, fp32):
//   N=1024   32c -> 2c   4.76us -> 3.48us  (-27%)
//   N=4096   40c -> 8c   6.02us -> 3.86us  (-36%)
//   N=16384  40c -> 32c  5.94us -> ~5.5us  (modest)
// A balanced 512 elem/core split is the measured sweet spot for the smallest shapes.
// Large shapes are unaffected: once totalElements >= MIN_ELEM_PER_CORE * coreNum the
// plain blockFactor already exceeds the floor, so all cores are used exactly as before
// (no regression on >=64K / >=1M shapes; verified by the perf re-test).
constexpr int64_t MIN_ELEM_PER_CORE = 512;

// var / accum / linear / grad input indices.
constexpr size_t kIdxVar = 0;
constexpr size_t kIdxAccum = 1;
constexpr size_t kIdxLinear = 2;
constexpr size_t kIdxGrad = 3;

static const gert::Shape g_vec_1_shape = {1};

static inline const gert::Shape EnsureNotScalar(const gert::Shape& in_shape)
{
    if (in_shape.GetDimNum() == 0) {
        return g_vec_1_shape;
    }
    return in_shape;
}

static bool ShapeEqual(const gert::Shape& a, const gert::Shape& b)
{
    if (a.GetDimNum() != b.GetDimNum()) {
        return false;
    }
    for (size_t i = 0; i < a.GetDimNum(); ++i) {
        if (a.GetDim(i) != b.GetDim(i)) {
            return false;
        }
    }
    return true;
}

static ge::graphStatus GetPlatformInfo(gert::TilingContext* context, uint64_t& ubSize, int64_t& coreNum)
{
    fe::PlatFormInfos* platformInfoPtr = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfoPtr);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    coreNum = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF(coreNum == 0, OP_LOGE(context, "coreNum is 0"), return ge::GRAPH_FAILED);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    OP_CHECK_IF(ubSize == 0, OP_LOGE(context, "ubSize is 0"), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus GetShapeInfo(gert::TilingContext* context, int64_t& totalElements, ge::DataType& dataType)
{
    // Input 0 = var; its shape is canonical (var/accum/linear/grad must match).
    auto inputVar = context->GetInputShape(kIdxVar);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputVar);
    auto inputAccum = context->GetInputShape(kIdxAccum);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputAccum);
    auto inputLinear = context->GetInputShape(kIdxLinear);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputLinear);
    auto inputGrad = context->GetInputShape(kIdxGrad);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputGrad);

    const gert::Shape& varShape = inputVar->GetStorageShape();
    // MED-3 defensive check: the four element-wise tensors must share one shape.
    OP_CHECK_IF(!ShapeEqual(varShape, inputAccum->GetStorageShape()) ||
                    !ShapeEqual(varShape, inputLinear->GetStorageShape()) ||
                    !ShapeEqual(varShape, inputGrad->GetStorageShape()),
                OP_LOGE(context, "ApplyFtrl: var/accum/linear/grad shapes must be identical"), return ge::GRAPH_FAILED);

    totalElements = EnsureNotScalar(varShape).GetShapeSize();

    auto inputDesc = context->GetInputDesc(kIdxVar);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputDesc);
    dataType = inputDesc->GetDataType();
    OP_CHECK_IF(dataType != ge::DT_BF16 && dataType != ge::DT_FLOAT16 && dataType != ge::DT_FLOAT,
                OP_LOGE(context, "ApplyFtrl: only bf16/fp16/fp32 supported, got %d", static_cast<int>(dataType)),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus GetWorkspaceSize(gert::TilingContext* context)
{
    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, currentWorkspace);
    currentWorkspace[0] = USER_WORKSPACE_SIZE;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InitTilingData(gert::TilingContext* context, ApplyFtrlTilingData*& tiling)
{
    tiling = context->GetTilingData<ApplyFtrlTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
    OP_CHECK_IF(memset_s(tiling, sizeof(ApplyFtrlTilingData), 0, sizeof(ApplyFtrlTilingData)) != EOK,
                OP_LOGE(context, "set tiling data error"), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus SetApplyFtrlTilingKey(gert::TilingContext* context, int64_t totalElements, int64_t ubBlockSize,
                                             ge::DataType dataType)
{
    if (ubBlockSize == 0) {
        OP_LOGE(context, "ubBlockSize is 0");
        return ge::GRAPH_FAILED;
    }
    if (ubBlockSize < 0) {
        OP_LOGE(context, "invalid ubBlockSize=%ld", ubBlockSize);
        return ge::GRAPH_FAILED;
    }

    uint32_t dTypeVar = static_cast<uint32_t>(dataType);
    uint32_t padTail = ((totalElements % ubBlockSize) != 0) ? 1U : 0U;
    uint32_t hasL1 = 1U;
    ASCENDC_TPL_SEL_PARAM(context, dTypeVar, padTail, hasL1);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus SetEmptyTensorTiling(gert::TilingContext* context, ApplyFtrlTilingData* tiling,
                                            ge::DataType dataType)
{
    // Empty tensor: single idle core; pick the simplest binary (PAD_TAIL=0,HAS_L1=1).
    tiling->blockFactor = 0;
    tiling->ubFactor = 0;
    context->SetBlockDim(1);
    return SetApplyFtrlTilingKey(context, 0, 1, dataType);
}

static int64_t CalcBlockFactor(int64_t totalElements, int64_t coreNum, int64_t ubBlockSize)
{
    // Multi-core split: ceil-aligned to DMA granularity.
    int64_t blockFactor = CeilAlign(CeilDiv(totalElements, coreNum), ubBlockSize);
    // Adaptive block-dim (PERF): floor blockFactor at MIN_ELEM_PER_CORE (DMA-aligned)
    // so small/medium shapes use fewer cores and amortise the fixed per-core head overhead.
    int64_t minBlockFactor = CeilAlign(MIN_ELEM_PER_CORE, ubBlockSize);
    return (blockFactor < minBlockFactor) ? minBlockFactor : blockFactor;
}

static ge::graphStatus CalcUbFactor(gert::TilingContext* context, uint64_t ubSize, int64_t blockFactor,
                                    int64_t& ubFactor)
{
    // UB split: reserve the 8KB Select tmp first (MED-1), then budget by fp32 slots,
    // and align DOWN to 64 elements (256B, MED-2).
    int64_t usableUb = static_cast<int64_t>(ubSize) - SELECT_TMP_BYTES;
    OP_CHECK_IF(usableUb <= 0, OP_LOGE(context, "UB too small after reserving Select tmp: %ld", usableUb),
                return ge::GRAPH_FAILED);
    int64_t ubCapacityElem = FloorAlign(FloorDiv(usableUb / TYPE_SIZE_FP32, UB_FP32_SLOTS), kCmpAlignElem);

    ubFactor = (TILE_ELEM_NUM_TARGET < ubCapacityElem) ? TILE_ELEM_NUM_TARGET : ubCapacityElem;
    // Cap by per-core workload to avoid over-allocating UB for small shapes.
    if (ubFactor > blockFactor) {
        ubFactor = blockFactor;
    }
    ubFactor = FloorAlign(ubFactor, kCmpAlignElem);
    // Floor at one 256B compute block so CompareScalar/Select alignment + tail
    // round-up (alignedNum) stay within the per-tile scratch buffers.
    if (ubFactor < kCmpAlignElem) {
        ubFactor = kCmpAlignElem;
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus ApplyFtrlTilingFunc(gert::TilingContext* context)
{
    uint64_t ubSize = 0;
    int64_t coreNum = 0;
    OP_CHECK_IF(GetPlatformInfo(context, ubSize, coreNum) != ge::GRAPH_SUCCESS,
                OP_LOGE(context, "GetPlatformInfo error"), return ge::GRAPH_FAILED);

    int64_t totalElements = 0;
    ge::DataType dataType = ge::DT_FLOAT16;
    OP_CHECK_IF(GetShapeInfo(context, totalElements, dataType) != ge::GRAPH_SUCCESS,
                OP_LOGE(context, "GetShapeInfo error"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(GetWorkspaceSize(context) != ge::GRAPH_SUCCESS, OP_LOGE(context, "GetWorkspaceSize error"),
                return ge::GRAPH_FAILED);

    ApplyFtrlTilingData* tiling = nullptr;
    OP_CHECK_IF(InitTilingData(context, tiling) != ge::GRAPH_SUCCESS, OP_LOGE(context, "InitTilingData error"),
                return ge::GRAPH_FAILED);
    tiling->totalElements = totalElements;
    if (totalElements == 0) {
        return SetEmptyTensorTiling(context, tiling, dataType);
    }

    // ubBlockSize = 32B / sizeof(T): fp32 -> 8, bf16/fp16 -> 16 elements.
    int64_t ubBlockSize = Ops::Base::GetUbBlockSize(context);
    if (ubBlockSize <= 0) {
        OP_LOGE(context, "invalid ubBlockSize=%ld", ubBlockSize);
        return ge::GRAPH_FAILED;
    }

    int64_t blockFactor = CalcBlockFactor(totalElements, coreNum, ubBlockSize);
    int64_t ubFactor = 0;
    OP_CHECK_IF(CalcUbFactor(context, ubSize, blockFactor, ubFactor) != ge::GRAPH_SUCCESS,
                OP_LOGE(context, "CalcUbFactor error"), return ge::GRAPH_FAILED);

    tiling->blockFactor = blockFactor;
    tiling->ubFactor = ubFactor;
    context->SetBlockDim(CeilDiv(totalElements, blockFactor));
    return SetApplyFtrlTilingKey(context, totalElements, ubBlockSize, dataType);
}

static ge::graphStatus TilingParseForApplyFtrl([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

struct ApplyFtrlCompileInfo {};

IMPL_OP_OPTILING(ApplyFtrl).Tiling(ApplyFtrlTilingFunc).TilingParse<ApplyFtrlCompileInfo>(TilingParseForApplyFtrl);

} // namespace optiling

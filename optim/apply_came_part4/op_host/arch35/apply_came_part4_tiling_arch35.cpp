/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file apply_came_part4_tiling_arch35.cpp
 * \brief ApplyCamePart4 tiling implementation (arch35)
 *
 * Aligned to canndev Tiling4ApplyCamePart4, with three independent splits:
 *   - R phase:   elementwise over n (len of r)
 *   - C phase:   elementwise over m (len of c)
 *   - Param phase: cores over n, inner 2D tile loops (rRcNumPerLoop x cRcNumPerLoop)
 *
 * Deviations from canndev:
 *   1. Tail-core loop count uses floor + remainder (rTailNum / rNumTailPerLoop and
 *      rTailNum % rNumTailPerLoop). canndev computes ceil and a subtraction that goes
 *      negative when rTailNum is not a multiple of rNumTailPerLoop, which makes the
 *      kernel copy a garbage (negative-cast) count. Only affects shapes where the tail
 *      core holds more than one UB-sized chunk; the fixed formulas are identical to
 *      canndev's whenever canndev's are non-negative.
 *   2. The ConfusionTranspose tiling struct is dropped (kernel always uses the
 *      per-row Muls path for r*c).
 *
 * Workspace layout (user part): [0, totalCoreNum*32) reserved for multi-core sync,
 * then 32B for the sum_r reduction slot (only used when optional input sum_r is absent).
 */

#include <cstdint>
#include <string>

#include "register/op_def_registry.h"
#include "op_common/log/log.h"
#include "op_common/op_host/util/math_util.h"
#include "op_common/op_host/util/platform_util.h"
#include "../../op_kernel/arch35/apply_came_part4_tiling_data.h"
#include "../../op_kernel/arch35/apply_came_part4_tiling_key.h"

namespace optiling {

using Ops::Base::CeilDiv;

namespace {
constexpr int64_t ONE_BLK_SIZE = 32;
constexpr int64_t RC_INPUT_UB_BYTES = 255 * 256; // per-chunk UB budget for the R/C phase, from canndev
constexpr int64_t CALC_SIZE = 256;               // bytes per vector instruction chunk, from canndev
constexpr int64_t RATIO_LOOP = 8;                // rows-per-loop divisor for the param phase, from canndev
constexpr size_t SYS_WORKSPACE_SIZE = 16 * 1024 * 1024;

constexpr size_t kIdxInParam = 0;
constexpr size_t kIdxInM = 1;
constexpr size_t kIdxInR = 2;
constexpr size_t kIdxInC = 3;
} // namespace

static ge::graphStatus GetPlatformInfo(gert::TilingContext* context, int64_t& coreNum)
{
    fe::PlatFormInfos* platformInfoPtr = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfoPtr);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    coreNum = ascendcPlatform.GetCoreNumAiv();
    if (coreNum <= 0) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "coreNum", std::to_string(coreNum).c_str(),
                                              "failed to get aiv core num");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

// shape contract: param/m are [n, m], r is [n], c is [m] (checked on the last dims)
static ge::graphStatus CheckParamsShape(gert::TilingContext* context)
{
    auto paramShape = context->GetInputShape(kIdxInParam);
    auto mShape = context->GetInputShape(kIdxInM);
    auto rShape = context->GetInputShape(kIdxInR);
    auto cShape = context->GetInputShape(kIdxInC);
    OP_CHECK_NULL_WITH_CONTEXT(context, paramShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, mShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, rShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, cShape);

    auto paramStorage = paramShape->GetStorageShape();
    auto mStorage = mShape->GetStorageShape();
    auto rStorage = rShape->GetStorageShape();
    auto cStorage = cShape->GetStorageShape();

    size_t paramDimNum = paramStorage.GetDimNum();
    size_t mDimNum = mStorage.GetDimNum();
    size_t rDimNum = rStorage.GetDimNum();
    size_t cDimNum = cStorage.GetDimNum();

    if (paramDimNum != 2 || mDimNum != 2) {
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context->GetNodeName(), "param_in,m",
                                              std::to_string(paramDimNum) + "D," + std::to_string(mDimNum) + "D",
                                              "param_in and m must both be 2D.");
        return ge::GRAPH_FAILED;
    }
    if (rDimNum != 1 || cDimNum != 1) {
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context->GetNodeName(), "r_in,c_in",
                                              std::to_string(rDimNum) + "D," + std::to_string(cDimNum) + "D",
                                              "r_in and c_in must both be 1D.");
        return ge::GRAPH_FAILED;
    }
    if (paramStorage.GetDim(0) != mStorage.GetDim(0) || paramStorage.GetDim(1) != mStorage.GetDim(1)) {
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context->GetNodeName(), "m", Ops::Base::ToString(mStorage).c_str(),
                                              "m shape must equal param_in shape.");
        return ge::GRAPH_FAILED;
    }
    if (paramStorage.GetDim(1) != cStorage.GetDim(0)) {
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context->GetNodeName(), "c_in", Ops::Base::ToString(cStorage).c_str(),
                                              "c_in dim0 must equal param_in dim1.");
        return ge::GRAPH_FAILED;
    }
    if (paramStorage.GetDim(0) != rStorage.GetDim(0)) {
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context->GetNodeName(), "r_in", Ops::Base::ToString(rStorage).c_str(),
                                              "r_in dim0 must equal param_in dim0.");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

// R phase split over n (canndev Tiling4CalcR, with the tail-core floor/mod fix)
static void Tiling4CalcR(int32_t typeSize, int64_t totalCoreNum, ApplyCamePart4TilingData* tiling)
{
    int64_t numPerBlock = ONE_BLK_SIZE / typeSize;
    int64_t n = tiling->n;
    // elements per core
    int64_t rNumPerCore = (n / totalCoreNum + numPerBlock - 1) / numPerBlock * numPerBlock;
    rNumPerCore = rNumPerCore > numPerBlock ? rNumPerCore : numPerBlock;
    // cores to use
    int64_t rCoreNumToUse = (n + rNumPerCore - 1) / rNumPerCore;
    rCoreNumToUse = rCoreNumToUse < totalCoreNum ? rCoreNumToUse : totalCoreNum;
    // elements on the tail core
    int64_t rTailNum = n - (rCoreNumToUse - 1) * rNumPerCore;
    // per-loop elements: non-tail core
    int64_t ubElements = (RC_INPUT_UB_BYTES - RC_INPUT_UB_BYTES % ONE_BLK_SIZE) / typeSize;
    int64_t rNumPerLoop = ubElements < rNumPerCore ? ubElements : rNumPerCore;
    int64_t rLoopCount = CeilDiv(rNumPerCore, rNumPerLoop);
    // per-loop elements: tail core (floor full loops + unaligned remainder, see file header)
    int64_t rNumTailPerLoop = ubElements < rTailNum ? ubElements : rTailNum;
    int64_t rLoopCountTailCore = rTailNum / rNumTailPerLoop;
    int64_t rNumTailLoopLast = rTailNum % rNumTailPerLoop;

    tiling->rNumPerCore = rNumPerCore;
    tiling->rCoreNumToUse = rCoreNumToUse;
    tiling->rNumPerLoop = rNumPerLoop;
    tiling->rLoopCount = rLoopCount;
    tiling->rNumTailPerLoop = rNumTailPerLoop;
    tiling->rLoopCountTailCore = rLoopCountTailCore;
    tiling->rNumTailLoopLast = rNumTailLoopLast;
}

// C phase split over m (canndev Tiling4CalcC, with the tail-core floor/mod fix)
static void Tiling4CalcC(int32_t typeSize, int64_t totalCoreNum, ApplyCamePart4TilingData* tiling)
{
    int64_t numPerBlock = ONE_BLK_SIZE / typeSize;
    int64_t m = tiling->m;
    int64_t cNumPerCore = (m / totalCoreNum + numPerBlock - 1) / numPerBlock * numPerBlock;
    cNumPerCore = cNumPerCore > numPerBlock ? cNumPerCore : numPerBlock;
    int64_t cCoreNumToUse = (m + cNumPerCore - 1) / cNumPerCore;
    cCoreNumToUse = cCoreNumToUse < totalCoreNum ? cCoreNumToUse : totalCoreNum;
    int64_t cTailNum = m - (cCoreNumToUse - 1) * cNumPerCore;
    int64_t ubElements = (RC_INPUT_UB_BYTES - RC_INPUT_UB_BYTES % ONE_BLK_SIZE) / typeSize;
    int64_t cNumPerLoop = ubElements < cNumPerCore ? ubElements : cNumPerCore;
    int64_t cLoopCount = CeilDiv(cNumPerCore, cNumPerLoop);
    int64_t cNumTailPerLoop = ubElements < cTailNum ? ubElements : cTailNum;
    int64_t cLoopCountTailCore = cTailNum / cNumTailPerLoop;
    int64_t cNumTailLoopLast = cTailNum % cNumTailPerLoop;

    tiling->cNumPerCore = cNumPerCore;
    tiling->cCoreNumToUse = cCoreNumToUse;
    tiling->cNumPerLoop = cNumPerLoop;
    tiling->cLoopCount = cLoopCount;
    tiling->cNumTailPerLoop = cNumTailPerLoop;
    tiling->cLoopCountTailCore = cLoopCountTailCore;
    tiling->cNumTailLoopLast = cNumTailLoopLast;
}

// Param phase split (canndev Tiling4CalcRc): cores over n, 2D tile loops inside
static void Tiling4CalcRc(int32_t typeSize, int64_t totalCoreNum, ApplyCamePart4TilingData* tiling)
{
    int64_t leastNumPerCore = ONE_BLK_SIZE / typeSize;
    leastNumPerCore = leastNumPerCore > 0 ? leastNumPerCore : 1;
    int64_t n = tiling->n;
    int64_t tmpCoreNumToUse = totalCoreNum < n ? totalCoreNum : n;
    int64_t tmpNumPerCore = CeilDiv(n, tmpCoreNumToUse);
    int64_t numPerCore = CeilDiv(tmpNumPerCore, leastNumPerCore) * leastNumPerCore;
    int64_t coreNumToUse = CeilDiv(n, numPerCore);
    int64_t numOnTailCore = n - (coreNumToUse - 1) * numPerCore;

    tiling->rRcNumPerCore = numPerCore;
    tiling->rRcCoreNumToUse = coreNumToUse;
    tiling->rRcNumOnTailCore = numOnTailCore;

    // rows per tile = (elements per instruction) / RATIO_LOOP; columns per tile = elements per instruction
    int64_t numPerInstruct = CALC_SIZE / typeSize;
    int64_t rRcNumPerLoop = numPerInstruct / RATIO_LOOP;
    tiling->rRcNumPerLoop = rRcNumPerLoop;
    tiling->rRcLoopCount = CeilDiv(tiling->rRcNumPerCore, rRcNumPerLoop);
    tiling->rRcNumTailLoop = tiling->rRcNumPerCore - (tiling->rRcLoopCount - 1) * rRcNumPerLoop;
    tiling->rRcLoopCountTailCore = CeilDiv(tiling->rRcNumOnTailCore, rRcNumPerLoop);
    tiling->rRcNumTailLoopTailCore = tiling->rRcNumOnTailCore - (tiling->rRcLoopCountTailCore - 1) * rRcNumPerLoop;

    tiling->cRcNumPerLoop = numPerInstruct;
    tiling->cRcLoopCount = CeilDiv(tiling->m, tiling->cRcNumPerLoop);
    tiling->cRcNumTailLoop = tiling->m - (tiling->cRcLoopCount - 1) * tiling->cRcNumPerLoop;
}

static int64_t GetMaxCoreNumToUse(const ApplyCamePart4TilingData* tiling)
{
    int64_t tmpMax = tiling->rCoreNumToUse > tiling->cCoreNumToUse ? tiling->rCoreNumToUse : tiling->cCoreNumToUse;
    return tmpMax > tiling->rRcCoreNumToUse ? tmpMax : tiling->rRcCoreNumToUse;
}

static int64_t GetHandleMax(const ApplyCamePart4TilingData* tiling)
{
    int64_t tmpMax1 = tiling->rNumPerLoop > tiling->cNumPerLoop ? tiling->rNumPerLoop : tiling->cNumPerLoop;
    int64_t tmpMax2 = tiling->rNumTailPerLoop > tiling->cNumTailPerLoop ? tiling->rNumTailPerLoop :
                                                                          tiling->cNumTailPerLoop;
    return tmpMax1 > tmpMax2 ? tmpMax1 : tmpMax2;
}

static ge::graphStatus GetAndCheckTypeSize(gert::TilingContext* context, int32_t& typeSize, uint32_t& dTypeX)
{
    auto paramDesc = context->GetInputDesc(kIdxInParam);
    OP_CHECK_NULL_WITH_CONTEXT(context, paramDesc);
    ge::DataType dataType = paramDesc->GetDataType();
    typeSize = ge::GetSizeByDataType(dataType);
    if (typeSize <= 0) {
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(context->GetNodeName(), "param_in", Ops::Base::ToString(dataType).c_str(),
                                              "param_in dtype must be float32/float16/bfloat16.");
        return ge::GRAPH_FAILED;
    }
    dTypeX = static_cast<uint32_t>(dataType);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus Tiling4ApplyCamePart4(gert::TilingContext* context)
{
    OP_LOGD(context->GetNodeName(), "Tiling4ApplyCamePart4 running begin");

    if (CheckParamsShape(context) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    int64_t totalCoreNum = 0;
    if (GetPlatformInfo(context, totalCoreNum) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    auto rShape = context->GetInputShape(kIdxInR);
    auto cShape = context->GetInputShape(kIdxInC);
    OP_CHECK_NULL_WITH_CONTEXT(context, rShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, cShape);
    int64_t n = rShape->GetStorageShape().GetDim(0);
    int64_t m = cShape->GetStorageShape().GetDim(0);

    int32_t typeSize = 0;
    uint32_t dTypeX = 0;
    if (GetAndCheckTypeSize(context, typeSize, dTypeX) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    // empty tensor: nothing to compute; zero the tiling data so the kernel-side
    // n/m guard reads deterministic zeros instead of an uninitialized buffer
    if (n <= 0 || m <= 0) {
        ApplyCamePart4TilingData* tiling = context->GetTilingData<ApplyCamePart4TilingData>();
        OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
        if (memset_s(tiling, sizeof(*tiling), 0, sizeof(*tiling)) != EOK) {
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "tilingData", "memset_s",
                                                  "failed to initialize tiling data");
            return ge::GRAPH_FAILED;
        }
        context->SetBlockDim(1);
        ASCENDC_TPL_SEL_PARAM(context, dTypeX);
        return ge::GRAPH_SUCCESS;
    }

    ApplyCamePart4TilingData* tiling = context->GetTilingData<ApplyCamePart4TilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
    if (memset_s(tiling, sizeof(*tiling), 0, sizeof(*tiling)) != EOK) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "tilingData", "memset_s",
                                              "failed to initialize tiling data");
        return ge::GRAPH_FAILED;
    }
    tiling->n = n;
    tiling->m = m;
    tiling->totalCoreNum = totalCoreNum;

    Tiling4CalcR(typeSize, totalCoreNum, tiling);
    Tiling4CalcC(typeSize, totalCoreNum, tiling);
    Tiling4CalcRc(typeSize, totalCoreNum, tiling);
    tiling->handleMax = GetHandleMax(tiling);

    // workspace: [0, totalCoreNum*32) sync flags + 32B sum_r slot, plus system workspace
    size_t userSize = static_cast<size_t>(totalCoreNum) * 32 + 32;
    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, currentWorkspace);
    currentWorkspace[0] = userSize + SYS_WORKSPACE_SIZE;

    context->SetBlockDim(GetMaxCoreNumToUse(tiling));
    // kernel uses SyncAll and atomic add: batch schedule mode + atomic clean
    context->SetScheduleMode(1);
    context->SetNeedAtomic(true);

    ASCENDC_TPL_SEL_PARAM(context, dTypeX);
    OP_LOGD(context->GetNodeName(), "Tiling4ApplyCamePart4 running end");
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParse4ApplyCamePart4([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

struct ApplyCamePart4CompileInfo {};

IMPL_OP_OPTILING(ApplyCamePart4)
    .Tiling(Tiling4ApplyCamePart4)
    .TilingParse<ApplyCamePart4CompileInfo>(TilingParse4ApplyCamePart4);

} // namespace optiling

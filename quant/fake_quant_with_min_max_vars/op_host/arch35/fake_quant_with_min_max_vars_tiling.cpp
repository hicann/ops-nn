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
 * \file fake_quant_with_min_max_vars_tiling.cpp
 * \brief Tiling implementation — PT cacheLine block cutting (static-quant-tiling.md §5.1)
 *
 * Tiling computes only work distribution (multi-core + UB tile).
 * Quantization parameters (scale, nmin, nmax) are NOT in TilingData —
 * kernel loads min/max from GM inputs and computes nudge params on-device.
 *
 * Algorithm (PT template):
 *   1. SelectMode = PER_TENSOR (always)
 *   2. MergeInputShape → (1, N, 1)
 *   3. PT cacheLine block cutting (blockAxis=1)
 *   4. UB baseLen = min(blockBase, maxBase)
 *   5. Attrs (numBits/narrowRange) in TilingData for kernel nudge computation
 */

#include <cmath>
#include <algorithm>
#include "register/op_def_registry.h"
#include "op_common/log/log.h"
#include "op_common/op_host/util/math_util.h"
#include "op_common/op_host/util/platform_util.h"
#include "../../op_kernel/arch35/fake_quant_with_min_max_vars_tiling_data.h"
#include "../../op_kernel/arch35/fake_quant_with_min_max_vars_tiling_key.h"

namespace optiling {

using Ops::Base::CeilDiv;
using Ops::Base::CeilAlign;
using Ops::Base::FloorDiv;
using Ops::Base::FloorAlign;

// DAV_3510 constants
constexpr int64_t CACHE_LINE = 128;          // bytes
constexpr int64_t TYPE_SIZE = 4;             // sizeof(float)
constexpr int64_t ELEM_PER_CACHE_LINE = CACHE_LINE / TYPE_SIZE;  // 32
constexpr int64_t UB_RESERVED = 2048;        // reserved bytes (stack/event/SPR)
constexpr int64_t QUEUE_SCALE_FIXED = 160;  // inQueueScale(32×2 DB) + inQueueOffset(32×2 DB) + bufNudge(32)
constexpr size_t WORKSPACE_NUM = 1;

// UB budget for PT fake-quant with double buffer + VF MicroAPI:
//   inQueueX:        baseLen * 4 * BUFF_NUM(2) = 8*baseLen
//   outQueueY:       baseLen * 4 * BUFF_NUM(2) = 8*baseLen
//   inQueueScale:    32 * 2 = 64
//   inQueueOffset:   32 * 2 = 64
//   calcInt32Buf:    baseLen * 4 = 4*baseLen (single buf, not used in VF path but allocated)
//   bufNudge:        32 (single buf, 5+ nudge scalars)
//   total = 20*baseLen + 160 + reserved
constexpr int64_t BYTES_PER_ELEM = 20;       // (8+8+4) for x, y, int32temp

static ge::graphStatus GetPlatformInfo(gert::TilingContext* context, uint64_t* ubSize, int64_t* coreNum)
{
    fe::PlatFormInfos* platformInfoPtr = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfoPtr);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    *coreNum = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF(*coreNum == 0, OP_LOGE(context, "coreNum is 0"), return ge::GRAPH_FAILED);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, *ubSize);
    OP_CHECK_IF(*ubSize == 0, OP_LOGE(context, "ubSize is 0"), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus GetShapeAttrsInfo(
    gert::TilingContext* context,
    int64_t* totalLength,
    int64_t* numBits,
    bool* narrowRange)
{
    auto inputX = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputX);
    auto shapeX = inputX->GetStorageShape();
    *totalLength = shapeX.GetShapeSize();

    // attrs: [0]=num_bits(Int64), [1]=narrow_range(Bool)
    auto attrs = context->GetAttrs();
    if (attrs != nullptr) {
        auto numBitsPtr = attrs->GetInt(0);
        *numBits = (numBitsPtr != nullptr) ? *numBitsPtr : 8;
        auto narrowRangePtr = attrs->GetBool(1);
        *narrowRange = (narrowRangePtr != nullptr) ? *narrowRangePtr : false;
    } else {
        *numBits = 8;
        *narrowRange = false;
    }

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus GetWorkspaceSize(gert::TilingContext* context)
{
    size_t* currentWorkspace = context->GetWorkspaceSizes(WORKSPACE_NUM);
    OP_CHECK_NULL_WITH_CONTEXT(context, currentWorkspace);
    currentWorkspace[0] = 0U;  // no workspace needed
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus FakeQuantWithMinMaxVarsTilingFunc(gert::TilingContext* context)
{
    OP_LOGD(context, "test xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx");
    // 1. Get platform info
    uint64_t ubSize;
    int64_t coreNum;
    OP_CHECK_IF(
        GetPlatformInfo(context, &ubSize, &coreNum) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "GetPlatformInfo error"), return ge::GRAPH_FAILED);

    // 2. Get shape and attrs (no min/max scalar loading — kernel loads from GM)
    int64_t totalLength;
    int64_t numBits;
    bool narrowRange;
    OP_CHECK_IF(
        GetShapeAttrsInfo(context, &totalLength, &numBits, &narrowRange) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "GetShapeAttrsInfo error"), return ge::GRAPH_FAILED);

    // 3. Workspace size
    OP_CHECK_IF(
        GetWorkspaceSize(context) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "GetWorkspaceSize error"), return ge::GRAPH_FAILED);

    // 4. SelectMode = PER_TENSOR, MergeInputShape → (1, N, 1)
    int64_t N = totalLength;

    // 5. Allocate and zero TilingData
    FakeQuantWithMinMaxVarsTilingData* tiling = context->GetTilingData<FakeQuantWithMinMaxVarsTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
    OP_CHECK_IF(
        memset_s(tiling, sizeof(FakeQuantWithMinMaxVarsTilingData), 0,
                 sizeof(FakeQuantWithMinMaxVarsTilingData)) != EOK,
        OP_LOGE(context, "set tiling data error"), return ge::GRAPH_FAILED);

    tiling->dim0 = 1;
    tiling->dim1 = N;
    tiling->dim2 = 1;
    tiling->totalLength = static_cast<uint64_t>(N);
    tiling->numBits = static_cast<uint32_t>(numBits);
    tiling->narrowRange = narrowRange;

    // 6. Empty tensor fast path
    if (N == 0) {
        tiling->baseLen = 0;
        tiling->blockFactor = 0;
        tiling->blockTailFactor = 0;
        tiling->numCore = 1;
        context->SetBlockDim(1);
        // TilingKey: MODE=0(PT), BUFFER_MODE=1, ROUND_MODE=1(RINT), HAS_ZP=0
        uint64_t key = GET_TPL_TILING_KEY(0, 1, 1, 0);
        OP_LOGD(context, "test xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx %ul", key);
        ASCENDC_TPL_SEL_PARAM(context, 0, 1, 1, 0);
        return ge::GRAPH_SUCCESS;
    }

    // 7. PT cacheLine block cutting (static-quant-tiling.md §5.1)
    int64_t cacheLineNum = CeilDiv(N, ELEM_PER_CACHE_LINE);
    int64_t idealPerCore = std::max(static_cast<int64_t>(1), CeilDiv(cacheLineNum, coreNum));
    int64_t actCore = CeilDiv(cacheLineNum, idealPerCore);
    if (actCore < 1) actCore = 1;
    if (actCore > coreNum) actCore = coreNum;

    int64_t cacheLinePerCore = CeilDiv(cacheLineNum, actCore);
    int64_t blockFactor = cacheLinePerCore * ELEM_PER_CACHE_LINE;
    if (blockFactor > N) blockFactor = N;
    int64_t blockTailFactor = N - blockFactor * (actCore - 1);
    if (blockTailFactor <= 0) {
        blockTailFactor = blockFactor;
    }

    tiling->numCore = static_cast<int32_t>(actCore);
    tiling->blockAxis = 1;   // PT: always dim1
    tiling->ubAxis = 1;      // PT: always dim1
    tiling->blockUnion = 1;  // PT: always 1
    tiling->blockFactor = static_cast<int32_t>(blockFactor);
    tiling->blockTailFactor = static_cast<int32_t>(blockTailFactor);
    tiling->baseN = 1;       // PT: always 1 row

    // 8. UB budget
    int64_t availableUb = static_cast<int64_t>(ubSize);
    int64_t maxBase = (availableUb - UB_RESERVED - QUEUE_SCALE_FIXED) / BYTES_PER_ELEM;
    if (maxBase < ELEM_PER_CACHE_LINE) {
        maxBase = ELEM_PER_CACHE_LINE;
    }
    maxBase = FloorAlign(maxBase, ELEM_PER_CACHE_LINE);

    int64_t blockBase = CeilAlign(blockFactor, ELEM_PER_CACHE_LINE);
    int64_t baseLen = (blockBase < maxBase) ? blockBase : maxBase;
    if (baseLen < ELEM_PER_CACHE_LINE) {
        baseLen = ELEM_PER_CACHE_LINE;
    }

    tiling->baseLen = static_cast<int32_t>(baseLen);
    tiling->hasZeroPoint = false;   // no explicit zp input
    tiling->axis = 0;               // unused for PT
    tiling->roundMode = 1;          // 1=CAST_RINT (round-half-to-even)
    tiling->sqrtMode = 0;           // no sqrt

    // 9. Set block dim
    context->SetBlockDim(static_cast<uint32_t>(actCore));

    // 10. TilingKey: MODE=0(PT), BUFFER_MODE=1(double), ROUND_MODE=1(RINT), HAS_ZP=0
    ASCENDC_TPL_SEL_PARAM(context, static_cast<uint32_t>(0), static_cast<uint32_t>(1), static_cast<uint32_t>(1), static_cast<uint32_t>(0));

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForFakeQuantWithMinMaxVars([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

struct FakeQuantWithMinMaxVarsCompileInfo {};

IMPL_OP_OPTILING(FakeQuantWithMinMaxVars)
    .Tiling(FakeQuantWithMinMaxVarsTilingFunc)
    .TilingParse<FakeQuantWithMinMaxVarsCompileInfo>(TilingParseForFakeQuantWithMinMaxVars);

}  // namespace optiling
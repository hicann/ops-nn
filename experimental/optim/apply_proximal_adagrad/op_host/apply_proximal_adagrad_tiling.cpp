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
 * \file apply_proximal_adagrad_tiling.cpp
 * \brief ApplyProximalAdagrad tiling (arch35).
 *
 * Tiling strategy (iteration-1 skeleton, single TilingKey path):
 *   1. Multi-core: split total elements evenly across AI vector cores.
 *      blockFactor = CeilAlign(CeilDiv(total, coreNum), ubBlockSize)
 *      so neighbouring cores do not trample each other's GM output.
 *   2. UB: allocate the per-tile size (ubFactor) based on an 8-buffer layout
 *      that covers var / accum / grad IN queues (x 2 double-buffered),
 *      var / accum OUT queues (x 2 double-buffered), and a few tmp scratch
 *      buffers (eta / prox / thresh / scale / sign) kept in float32.
 *      Total co-resident float32 tensors per tile ~= 3*2 (in) + 2*2 (out) +
 *      4 (tmp) = 14 buffers.  ubFactor = floor(UB / (14 * 4B)) aligned down
 *      to 32-byte granularity; an absolute ceiling of 2048 elements keeps
 *      the first iteration deterministic (see DESIGN s5.2, TILE_ELEM_NUM).
 */

#include "register/op_def_registry.h"
#include "op_common/log/log.h"
#include "op_common/op_host/util/math_util.h"
#include "op_common/op_host/util/platform_util.h"
#include "../op_kernel/apply_proximal_adagrad_tiling_data.h"
#include "../op_kernel/apply_proximal_adagrad_tiling_key.h"

namespace optiling {

using Ops::Base::CeilDiv;
using Ops::Base::CeilAlign;
using Ops::Base::FloorAlign;
using Ops::Base::FloorDiv;
using Ops::Base::GetUbBlockSize;

constexpr uint32_t WS_SYS_SIZE = 0U;
constexpr int64_t TYPE_SIZE = 4;              // sizeof(float) -- fp32 only.
// Per-tile target element count (float32 * 2048 = 8KB).
// Iteration-1 uses this conservative tile; later iterations may make it
// platform-derived.
constexpr int64_t TILE_ELEM_NUM_TARGET = 2048;
// Co-resident fp32 UB tensors per tile: var/accum/grad IN (double buffer) +
// var/accum OUT (double buffer) + 4 tmp (eta / prox / thresh / scale).
// SUG-002: This MUST stay in sync with the kernel-side constant
// `kUbResidentFp32TensorCount` defined at the top of
// `op_kernel/apply_proximal_adagrad.h`. Any change to the kernel's UB
// buffer layout (adding/removing TBuf or TQue, changing double-buffer depth)
// MUST update both constants together; see the kernel-side static_assert and
// breakdown comment.
constexpr int64_t UB_BUFFER_COUNT = 14;

static const gert::Shape g_vec_1_shape = {1};

static inline const gert::Shape EnsureNotScalar(const gert::Shape& in_shape)
{
    if (in_shape.GetDimNum() == 0) {
        return g_vec_1_shape;
    }
    return in_shape;
}

static ge::graphStatus GetPlatformInfo(gert::TilingContext* context,
                                       uint64_t& ubSize, int64_t& coreNum)
{
    fe::PlatFormInfos* platformInfoPtr = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfoPtr);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    coreNum = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF(coreNum == 0, OP_LOGE(context, "coreNum is 0"),
                return ge::GRAPH_FAILED);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    OP_CHECK_IF(ubSize == 0, OP_LOGE(context, "ubSize is 0"),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus GetShapeInfo(gert::TilingContext* context,
                                    int64_t& totalElements,
                                    ge::DataType& dataType)
{
    // Input 0 = var; its shape is canonical (var/accum/grad must match).
    auto inputVar = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputVar);
    auto varShape = EnsureNotScalar(inputVar->GetStorageShape());
    totalElements = varShape.GetShapeSize();

    auto inputDesc = context->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputDesc);
    dataType = inputDesc->GetDataType();
    OP_CHECK_IF(dataType != ge::DT_FLOAT,
                OP_LOGE(context,
                        "ApplyProximalAdagrad: only float32 is supported, got %d",
                        static_cast<int>(dataType)),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus GetWorkspaceSize(gert::TilingContext* context)
{
    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, currentWorkspace);
    currentWorkspace[0] = WS_SYS_SIZE;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus ApplyProximalAdagradTilingFunc(gert::TilingContext* context)
{
    // 1. Platform info
    uint64_t ubSize = 0;
    int64_t coreNum = 0;
    OP_CHECK_IF(GetPlatformInfo(context, ubSize, coreNum) != ge::GRAPH_SUCCESS,
                OP_LOGE(context, "GetPlatformInfo error"),
                return ge::GRAPH_FAILED);

    // 2. Shape / dtype info
    int64_t totalElements = 0;
    ge::DataType dataType = ge::DT_FLOAT;
    OP_CHECK_IF(GetShapeInfo(context, totalElements, dataType) != ge::GRAPH_SUCCESS,
                OP_LOGE(context, "GetShapeInfo error"),
                return ge::GRAPH_FAILED);

    // 3. Workspace
    OP_CHECK_IF(GetWorkspaceSize(context) != ge::GRAPH_SUCCESS,
                OP_LOGE(context, "GetWorkspaceSize error"),
                return ge::GRAPH_FAILED);

    // 4. Fill TilingData
    ApplyProximalAdagradTilingData* tiling =
        context->GetTilingData<ApplyProximalAdagradTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
    OP_CHECK_IF(
        memset_s(tiling, sizeof(ApplyProximalAdagradTilingData), 0,
                 sizeof(ApplyProximalAdagradTilingData)) != EOK,
        OP_LOGE(context, "set tiling data error"),
        return ge::GRAPH_FAILED);

    tiling->totalElements = totalElements;

    // Empty tensor: run a single idle core so the launcher still succeeds.
    if (totalElements == 0) {
        tiling->blockFactor = 0;
        tiling->ubFactor = 0;
        context->SetBlockDim(1);
        uint32_t dTypeVar = static_cast<uint32_t>(dataType);
        // Pick the simplest binary (PAD_TAIL=0, HAS_L1=1) for the empty path -
        // Process() short-circuits before doing any compute.
        uint32_t padTail = 0U;
        uint32_t hasL1 = 1U;
        ASCENDC_TPL_SEL_PARAM(context, dTypeVar, padTail, hasL1);
        return ge::GRAPH_SUCCESS;
    }

    // ubBlockSize = 32B / sizeof(T), with T = fp32 -> 8 elements.
    int64_t ubBlockSize = Ops::Base::GetUbBlockSize(context);
    OP_CHECK_IF(ubBlockSize <= 0,
                OP_LOGE(context, "invalid ubBlockSize=%ld", ubBlockSize),
                return ge::GRAPH_FAILED);

    // Multi-core split: ceil-aligned to DMA granularity.
    int64_t blockFactor = CeilAlign(CeilDiv(totalElements, coreNum), ubBlockSize);
    int64_t usedCoreNum = CeilDiv(totalElements, blockFactor);

    // UB split.  Cap by platform UB, target TILE_ELEM_NUM_TARGET (2048).
    int64_t ubCapacityElem =
        FloorAlign(FloorDiv(static_cast<int64_t>(ubSize) / TYPE_SIZE,
                            UB_BUFFER_COUNT),
                   ubBlockSize);
    OP_CHECK_IF(ubCapacityElem <= 0,
                OP_LOGE(context, "UB too small: ubCapacityElem=%ld",
                        ubCapacityElem),
                return ge::GRAPH_FAILED);

    int64_t ubFactor = (TILE_ELEM_NUM_TARGET < ubCapacityElem)
                           ? TILE_ELEM_NUM_TARGET
                           : ubCapacityElem;

    // Also cap by blockFactor so a single core does not allocate more UB
    // space than it will ever use.
    if (ubFactor > blockFactor) {
        ubFactor = FloorAlign(blockFactor, ubBlockSize);
        if (ubFactor <= 0) {
            ubFactor = ubBlockSize;
        }
    }

    tiling->blockFactor = blockFactor;
    tiling->ubFactor = ubFactor;

    context->SetBlockDim(usedCoreNum);

    // 5. TilingKey via ASCENDC_TPL_SEL_PARAM (template-argument mechanism).
    //    Iteration-2: derive PAD_TAIL from shape alignment.  HAS_L1 cannot be
    //    determined from host-side tiling without a Host<->Device sync (lr/l1/l2
    //    are aclTensor inputs whose values live in Device GM), so we default it
    //    to 1 and rely on the kernel to take a runtime fast-path when l1 == 0.
    //    The HAS_L1 = 0 binary is still produced so UT can drive TilingKey
    //    10003 directly (and a future L0-API hint can flip this from host).
    uint32_t dTypeVar = static_cast<uint32_t>(dataType);
    uint32_t padTail = ((totalElements % ubBlockSize) != 0) ? 1U : 0U;
    uint32_t hasL1 = 1U;
    ASCENDC_TPL_SEL_PARAM(context, dTypeVar, padTail, hasL1);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForApplyProximalAdagrad(
    [[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

struct ApplyProximalAdagradCompileInfo {};

IMPL_OP_OPTILING(ApplyProximalAdagrad)
    .Tiling(ApplyProximalAdagradTilingFunc)
    .TilingParse<ApplyProximalAdagradCompileInfo>(TilingParseForApplyProximalAdagrad);

} // namespace optiling

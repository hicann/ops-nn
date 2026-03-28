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

/**
 * \file hard_sigmoid_grad_v2_tiling.cpp
 * \brief HardSigmoidGradV2 tiling implementation (arch32, Ascend910B)
 *
 * Tiling strategy:
 *   - Multi-core: split total elements evenly across AI Cores
 *   - UB split: 5 tensors (2 input + 2 mask + 1 output) x buffer count
 *   - Double buffer when totalNum > 1024
 *
 * TilingKey parameters (via ASCENDC_TPL_SEL_PARAM):
 *   - dTypeX: data type enum value
 *   - useDoubleBuffer: 0 or 1
 */

#include "register/op_def_registry.h"
#include "op_common/log/log.h"
#include "op_common/op_host/util/math_util.h"
#include "op_common/op_host/util/platform_util.h"
#include "../op_kernel/hard_sigmoid_grad_v2_tiling_data.h"
#include "../op_kernel/hard_sigmoid_grad_v2_tiling_key.h"

namespace optiling {

using Ops::Base::CeilDiv;
using Ops::Base::FloorDiv;
using Ops::Base::FloorAlign;
using Ops::Base::GetUbBlockSize;

constexpr uint32_t WS_SYS_SIZE = 0U;
// Double buffer threshold: enable double buffer when element count > this value
constexpr int64_t MIN_SPLIT_THRESHOLD = 1024;

static const gert::Shape g_vec_1_shape = {1};

static inline const gert::Shape EnsureNotScalar(const gert::Shape& in_shape) {
    if (in_shape.GetDimNum() == 0) {
        return g_vec_1_shape;
    }
    return in_shape;
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

static ge::graphStatus GetShapeAttrsInfo(gert::TilingContext* context, int64_t& totalNum, ge::DataType& dataType)
{
    auto inputGradOutput = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputGradOutput);
    auto gradOutputShape = EnsureNotScalar(inputGradOutput->GetStorageShape());

    auto inputSelf = context->GetInputShape(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputSelf);
    auto selfShape = EnsureNotScalar(inputSelf->GetStorageShape());

    // Shape validation: grad_output and self must have same shape
    OP_CHECK_IF(
        gradOutputShape.GetShapeSize() != selfShape.GetShapeSize(),
        OP_LOGE(context, "HardSigmoidGradV2: grad_output and self shape size mismatch: grad=%ld, self=%ld",
                gradOutputShape.GetShapeSize(), selfShape.GetShapeSize()),
        return ge::GRAPH_FAILED);

    totalNum = gradOutputShape.GetShapeSize();

    // dtype validation
    const std::set<ge::DataType> supportedDtype = {ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16};
    auto inputDesc = context->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputDesc);
    dataType = inputDesc->GetDataType();
    if (supportedDtype.count(dataType) == 0) {
        OP_LOGE(context, "HardSigmoidGradV2: unsupported dtype");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus GetWorkspaceSize(gert::TilingContext* context)
{
    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, currentWorkspace);
    currentWorkspace[0] = WS_SYS_SIZE;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus HardSigmoidGradV2TilingFunc(gert::TilingContext* context)
{
    // 1. Get platform info
    uint64_t ubSize;
    int64_t coreNum;
    OP_CHECK_IF(
        GetPlatformInfo(context, ubSize, coreNum) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "GetPlatformInfo error"),
        return ge::GRAPH_FAILED);

    // 2. Get shape and attribute info
    int64_t totalNum;
    ge::DataType dataType;
    OP_CHECK_IF(
        GetShapeAttrsInfo(context, totalNum, dataType) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "GetShapeAttrsInfo error"),
        return ge::GRAPH_FAILED);

    // 3. Get workspace size
    OP_CHECK_IF(
        GetWorkspaceSize(context) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "GetWorkspaceSize error"),
        return ge::GRAPH_FAILED);

    // 4. Compute TilingData
    HardSigmoidGradV2TilingData* tiling = context->GetTilingData<HardSigmoidGradV2TilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
    OP_CHECK_IF(
        memset_s(tiling, sizeof(HardSigmoidGradV2TilingData), 0, sizeof(HardSigmoidGradV2TilingData)) != EOK,
        OP_LOGE(context, "set tiling data error"), return ge::GRAPH_FAILED);

    // Multi-core split
    tiling->totalNum = totalNum;

    // Empty tensor protection: totalNum==0 would cause CeilDiv divide-by-zero
    if (totalNum == 0) {
        tiling->blockFactor = 0;
        tiling->ubFactor = 0;
        context->SetBlockDim(1);
        uint32_t dTypeX = static_cast<uint32_t>(dataType);
        uint64_t useDoubleBuffer = 0;
        ASCENDC_TPL_SEL_PARAM(context, dTypeX, useDoubleBuffer);
        return ge::GRAPH_SUCCESS;
    }

    tiling->blockFactor = CeilDiv(totalNum, coreNum);
    int64_t usedCoreNum = CeilDiv(totalNum, tiling->blockFactor);

    // UB split:
    // half/float path: 3 T-sized queued buffers (2 input + 1 output) + 2 small mask buffers
    // bf16 path: 3 bf16-sized queued buffers + 2 small mask buffers + 2 float temp buffers
    // Conservative approach: count equivalent T-sized buffer slots
    // For half/float: 5 T-sized slots (3 queued + 2 mask ~= 2 small but round up)
    // For bf16: 3 bf16 slots + 2 float slots (2x size) = 3*2 + 2*4 = 14 bytes per element -> ~7 half-slots
    int64_t typeSize = (dataType == ge::DT_FLOAT) ? 4 : 2;
    uint64_t useDoubleBuffer = (totalNum > MIN_SPLIT_THRESHOLD) ? 1 : 0;
    int64_t bufferNum = useDoubleBuffer ? 2 : 1;
    // Queued buffers: 3 T-sized x bufferNum
    // Mask buffers: 2 x (ubFactor/8) bytes (small, ~0.125 per element)
    // bf16 extra: 2 float buffers x 4 bytes per element (not queued, no x bufferNum)
    int64_t queuedBytes;  // per element
    if (dataType == ge::DT_BF16) {
        // 3 bf16 queued (2 bytes each) x bufferNum + 2 float temp (4 bytes each) + mask overhead
        queuedBytes = 3 * 2 * bufferNum + 2 * 4 + 1;  // +1 for mask bytes
    } else {
        // 3 T queued x bufferNum + mask overhead
        queuedBytes = 3 * typeSize * bufferNum + 1;
    }
    int64_t ubBlockSize = GetUbBlockSize(context);
    // Compares Level 2 requires count*sizeof(ComputeT) % 256 == 0.
    // bf16 computes in float (4 bytes), half/float compute in their own type.
    int64_t computeTypeSize = (dataType == ge::DT_BF16 || dataType == ge::DT_FLOAT) ? 4 : 2;
    int64_t computeAlignment = 256 / computeTypeSize;  // 64 for float/bf16, 128 for fp16
    int64_t alignment = std::max(ubBlockSize, computeAlignment);
    tiling->ubFactor = FloorAlign(static_cast<int64_t>(ubSize) / queuedBytes, alignment);

    context->SetBlockDim(usedCoreNum);

    // 5. Set TilingKey
    // Parameter order matches ASCENDC_TPL_ARGS_DECL in hard_sigmoid_grad_v2_tiling_key.h
    uint32_t dTypeX = static_cast<uint32_t>(dataType);
    ASCENDC_TPL_SEL_PARAM(context, dTypeX, useDoubleBuffer);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForHardSigmoidGradV2([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

struct HardSigmoidGradV2CompileInfo {};

IMPL_OP_OPTILING(HardSigmoidGradV2)
    .Tiling(HardSigmoidGradV2TilingFunc)
    .TilingParse<HardSigmoidGradV2CompileInfo>(TilingParseForHardSigmoidGradV2);

} // namespace optiling

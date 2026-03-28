/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * \file celu_v3_tiling.cpp
 * \brief CeluV3 tiling implementation (arch32)
 *
 * Tiling strategy:
 *   1. Multi-core: divide total elements evenly across AI Cores
 *   2. UB: divide per-core elements into UB-sized chunks
 *   3. Buffer layout: inputQueue(1 buf) + outputQueue(1 buf) + tmpBuf1 + tmpBuf2
 *      - float32: ubDivisor = 4 (2*sizeof(float) + 2*sizeof(float) = 16 bytes per elem, 4 floats)
 *      - float16/bf16: ubDivisor = 6 (2*sizeof(half) + 2*sizeof(float) = 12 bytes per elem, 6 halfs)
 */

#include "register/op_def_registry.h"
#include "op_common/log/log.h"
#include "op_common/op_host/util/math_util.h"
#include "op_common/op_host/util/platform_util.h"
#include "../op_kernel/celu_v3_tiling_data.h"
#include "../op_kernel/celu_v3_tiling_key.h"

namespace optiling {

using Ops::Base::CeilDiv;
using Ops::Base::FloorDiv;
using Ops::Base::FloorAlign;
using Ops::Base::GetUbBlockSize;

constexpr uint32_t WS_SYS_SIZE = 0U;

static const gert::Shape g_vec_1_shape = {1};

static inline const gert::Shape EnsureNotScalar(const gert::Shape& in_shape)
{
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

static ge::graphStatus GetShapeAttrsInfo(gert::TilingContext* context, int64_t& totalElements,
                                         ge::DataType& dataType, float& alphaVal)
{
    // Get input shape
    auto inputSelf = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputSelf);
    auto inputShape = EnsureNotScalar(inputSelf->GetStorageShape());
    totalElements = inputShape.GetShapeSize();

    // Get dtype
    auto inputDesc = context->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputDesc);
    dataType = inputDesc->GetDataType();
    const std::set<ge::DataType> supportedDtype = {ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16};
    if (supportedDtype.count(dataType) == 0) {
        OP_LOGE(context, "CeluV3: unsupported dtype %d", static_cast<int>(dataType));
        return ge::GRAPH_FAILED;
    }

    // Get alpha attribute
    const float* alphaPtr = context->GetAttrs()->GetAttrPointer<float>(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, alphaPtr);
    alphaVal = *alphaPtr;
    if (alphaVal == 0.0f) {
        OP_LOGE(context, "CeluV3: alpha cannot be 0");
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

static ge::graphStatus CeluV3TilingFunc(gert::TilingContext* context)
{
    // 1. Get platform info
    uint64_t ubSize = 0;
    int64_t coreNum = 0;
    OP_CHECK_IF(
        GetPlatformInfo(context, ubSize, coreNum) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "GetPlatformInfo error"), return ge::GRAPH_FAILED);

    // 2. Get shape, attrs info
    int64_t totalElements = 0;
    ge::DataType dataType = ge::DT_FLOAT;
    float alphaVal = 0.0f;
    OP_CHECK_IF(
        GetShapeAttrsInfo(context, totalElements, dataType, alphaVal) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "GetShapeAttrsInfo error"), return ge::GRAPH_FAILED);

    // 3. Get workspace size
    OP_CHECK_IF(
        GetWorkspaceSize(context) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "GetWorkspaceSize error"), return ge::GRAPH_FAILED);

    // 4. Compute tiling parameters
    CeluV3TilingData* tiling = context->GetTilingData<CeluV3TilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
    OP_CHECK_IF(
        memset_s(tiling, sizeof(CeluV3TilingData), 0, sizeof(CeluV3TilingData)) != EOK,
        OP_LOGE(context, "set tiling data error"), return ge::GRAPH_FAILED);

    // Determine element size for UB computation
    int64_t typeSize = sizeof(float); // 4 bytes
    if (dataType == ge::DT_FLOAT16 || dataType == ge::DT_BF16) {
        typeSize = 2; // 2 bytes for half/bf16
    }

    int64_t ubBlockSize = 32 / typeSize; // 32-byte alignment in elements

    // Handle empty tensor (totalElements=0): set blockDim=1, kernel will early-return
    if (totalElements == 0) {
        tiling->totalElements = 0;
        tiling->blockFactor = 0;
        tiling->ubFactor = 0;
        tiling->alphaVal = alphaVal;
        tiling->invAlpha = 1.0f / alphaVal;
        context->SetBlockDim(1);
        uint32_t dTypeX = static_cast<uint32_t>(dataType);
        ASCENDC_TPL_SEL_PARAM(context, dTypeX);
        return ge::GRAPH_SUCCESS;
    }

    // Multi-core split
    int64_t blockFactor = CeilDiv(totalElements, coreNum);
    // Align blockFactor to ubBlockSize
    blockFactor = ((blockFactor + ubBlockSize - 1) / ubBlockSize) * ubBlockSize;
    int64_t usedCoreNum = CeilDiv(totalElements, blockFactor);

    // UB split
    // Buffer layout:
    //   inputQueue(1 buf): ubFactor * typeSize
    //   outputQueue(1 buf): ubFactor * typeSize
    //   tmpBuf1: ubFactor * sizeof(float) = ubFactor * 4
    //   tmpBuf2: ubFactor * sizeof(float) = ubFactor * 4
    //
    // For float32: total = ubFactor * (4 + 4 + 4 + 4) = ubFactor * 16 = ubFactor * 4 * sizeof(float)
    //   ubDivisor = 4 (in units of sizeof(T))
    // For float16/bf16: total = ubFactor * (2 + 2 + 4 + 4) = ubFactor * 12 = ubFactor * 6 * sizeof(T)
    //   ubDivisor = 6 (in units of sizeof(T))
    int64_t ubDivisor;
    if (dataType == ge::DT_FLOAT) {
        ubDivisor = 4;
    } else {
        ubDivisor = 6;
    }

    int64_t ubFactor = FloorAlign(
        FloorDiv(static_cast<int64_t>(ubSize) / typeSize, ubDivisor),
        ubBlockSize);

    tiling->totalElements = totalElements;
    tiling->blockFactor = blockFactor;
    tiling->ubFactor = ubFactor;
    tiling->alphaVal = alphaVal;
    tiling->invAlpha = 1.0f / alphaVal;

    context->SetBlockDim(usedCoreNum);

    // 5. Set TilingKey using ASCENDC_TPL_SEL_PARAM
    uint32_t dTypeX = static_cast<uint32_t>(dataType);
    ASCENDC_TPL_SEL_PARAM(context, dTypeX);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForCeluV3([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

struct CeluV3CompileInfo {};

IMPL_OP_OPTILING(CeluV3).Tiling(CeluV3TilingFunc).TilingParse<CeluV3CompileInfo>(TilingParseForCeluV3);

} // namespace optiling

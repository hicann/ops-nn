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
 * \file selu_tiling.cpp
 * \brief Selu tiling implementation (arch32)
 *
 * Tiling strategy:
 *   1. Multi-core: divide total elements evenly across AI Cores
 *   2. UB: divide per-core elements into UB-sized chunks
 *   3. Buffer layout: inputQueue(1 buf) + outputQueue(1 buf) + tmpBuf1 + tmpBuf2
 *
 * ubDivisor by dtype (in units of sizeof(T)):
 *   FLOAT32:  (2*4 + 2*4) / 4 = 4
 *   FLOAT16:  (2*2 + 2*2) / 2 = 4
 *   BFLOAT16: (2*2 + 2*4) / 2 = 6
 *   INT32:    (2*4 + 2*4) / 4 = 4
 *   INT8:     (2*1 + 2*4) / 1 = 10
 */

#include "register/op_def_registry.h"
#include "op_common/log/log.h"
#include "op_common/op_host/util/math_util.h"
#include "op_common/op_host/util/platform_util.h"
#include "../op_kernel/selu_tiling_data.h"
#include "../op_kernel/selu_tiling_key.h"

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

static ge::graphStatus GetShapeInfo(gert::TilingContext* context, int64_t& totalElements,
                                    ge::DataType& dataType)
{
    // Get input shape
    auto inputX = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputX);
    auto inputShape = EnsureNotScalar(inputX->GetStorageShape());
    totalElements = inputShape.GetShapeSize();

    // Get dtype
    auto inputDesc = context->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputDesc);
    dataType = inputDesc->GetDataType();
    const std::set<ge::DataType> supportedDtype = {
        ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16, ge::DT_INT32, ge::DT_INT8
    };
    if (supportedDtype.count(dataType) == 0) {
        OP_LOGE(context, "Selu: unsupported dtype %d", static_cast<int>(dataType));
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

static ge::graphStatus SeluTilingFunc(gert::TilingContext* context)
{
    // 1. Get platform info
    uint64_t ubSize = 0;
    int64_t coreNum = 0;
    OP_CHECK_IF(
        GetPlatformInfo(context, ubSize, coreNum) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "GetPlatformInfo error"), return ge::GRAPH_FAILED);

    // 2. Get shape info
    int64_t totalElements = 0;
    ge::DataType dataType = ge::DT_FLOAT;
    OP_CHECK_IF(
        GetShapeInfo(context, totalElements, dataType) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "GetShapeInfo error"), return ge::GRAPH_FAILED);

    // 3. Get workspace size
    OP_CHECK_IF(
        GetWorkspaceSize(context) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "GetWorkspaceSize error"), return ge::GRAPH_FAILED);

    // 4. Compute tiling parameters
    SeluTilingData* tiling = context->GetTilingData<SeluTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
    OP_CHECK_IF(
        memset_s(tiling, sizeof(SeluTilingData), 0, sizeof(SeluTilingData)) != EOK,
        OP_LOGE(context, "set tiling data error"), return ge::GRAPH_FAILED);

    // Determine typeSize and computeTypeSize based on dtype
    int64_t typeSize = 4;        // default: sizeof(float)
    int64_t computeTypeSize = 4; // default: sizeof(float)
    switch (dataType) {
        case ge::DT_FLOAT:
            typeSize = 4;
            computeTypeSize = 4;
            break;
        case ge::DT_FLOAT16:
            typeSize = 2;
            computeTypeSize = 2; // FP16 direct computation
            break;
        case ge::DT_BF16:
            typeSize = 2;
            computeTypeSize = 4; // Cast to FP32
            break;
        case ge::DT_INT32:
            typeSize = 4;
            computeTypeSize = 4; // Cast to FP32
            break;
        case ge::DT_INT8:
            typeSize = 1;
            computeTypeSize = 4; // Cast to FP32 for precision
            break;
        default:
            OP_LOGE(context, "Selu: unexpected dtype %d", static_cast<int>(dataType));
            return ge::GRAPH_FAILED;
    }

    int64_t ubBlockSize = 32 / typeSize; // 32-byte alignment in elements

    // Handle empty tensor (totalElements=0): set blockDim=1, kernel will early-return
    if (totalElements == 0) {
        tiling->totalElements = 0;
        tiling->blockFactor = 0;
        tiling->ubFactor = 0;
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
    // Buffer layout per element:
    //   inputQueue:  typeSize bytes
    //   outputQueue: typeSize bytes
    //   tmpBuf1:     computeTypeSize bytes
    //   tmpBuf2:     computeTypeSize bytes
    //   Total = 2*typeSize + 2*computeTypeSize bytes per element
    //   ubDivisor = (2*typeSize + 2*computeTypeSize) / typeSize
    int64_t ubDivisor = (2 * typeSize + 2 * computeTypeSize) / typeSize;

    int64_t ubFactor = FloorAlign(
        FloorDiv(static_cast<int64_t>(ubSize) / typeSize, ubDivisor),
        ubBlockSize);
    OP_CHECK_IF(ubFactor <= 0, OP_LOGE(context, "Selu: ubFactor is %ld, UB too small", ubFactor),
                return ge::GRAPH_FAILED);

    tiling->totalElements = totalElements;
    tiling->blockFactor = blockFactor;
    tiling->ubFactor = ubFactor;

    context->SetBlockDim(usedCoreNum);

    // 5. Set TilingKey using ASCENDC_TPL_SEL_PARAM
    uint32_t dTypeX = static_cast<uint32_t>(dataType);
    ASCENDC_TPL_SEL_PARAM(context, dTypeX);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForSelu([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

struct SeluCompileInfo {};

IMPL_OP_OPTILING(Selu).Tiling(SeluTilingFunc).TilingParse<SeluCompileInfo>(TilingParseForSelu);

} // namespace optiling

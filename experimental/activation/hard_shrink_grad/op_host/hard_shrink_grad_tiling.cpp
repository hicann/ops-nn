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
 * \file hard_shrink_grad_tiling.cpp
 * \brief HardShrinkGrad Tiling implementation for arch32 (Ascend910B)
 *
 * Tiling flow:
 *   1. Get platform info (coreNum, ubSize)
 *   2. Get input shape info (totalNum, dataType)
 *   3. Get attribute (lambd)
 *   4. Compute split parameters (blockFactor, ubFactor)
 *   5. Set BlockDim, TilingKey
 */

#include "register/op_def_registry.h"
#include "op_common/log/log.h"
#include "op_common/op_host/util/math_util.h"
#include "op_common/op_host/util/platform_util.h"
#include "../op_kernel/hard_shrink_grad_tiling_data.h"
#include "../op_kernel/hard_shrink_grad_tiling_key.h"

namespace optiling {

using Ops::Base::CeilDiv;
using Ops::Base::FloorDiv;
using Ops::Base::FloorAlign;
using Ops::Base::GetUbBlockSize;

constexpr uint32_t WS_SYS_SIZE = 0U;
// Double buffer threshold: enable double buffer when totalNum > this value
constexpr int64_t MIN_SPLIT_THRESHOLD = 1024;

static const gert::Shape g_vec_1_shape = {1};

static inline const gert::Shape EnsureNotScalar(const gert::Shape& inShape)
{
    if (inShape.GetDimNum() == 0) {
        return g_vec_1_shape;
    }
    return inShape;
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

static ge::graphStatus GetShapeAttrsInfo(gert::TilingContext* context, int64_t& totalNum,
                                          ge::DataType& dataType, float& lambd)
{
    // Get input shape info (grad_output)
    auto inputGrad = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputGrad);
    auto inputShapeGrad = EnsureNotScalar(inputGrad->GetStorageShape());

    // Get input shape info (self)
    auto inputSelf = context->GetInputShape(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputSelf);
    auto inputShapeSelf = EnsureNotScalar(inputSelf->GetStorageShape());

    // Shape validation: grad_output and self must have the same shape
    OP_CHECK_IF(
        inputShapeGrad.GetShapeSize() != inputShapeSelf.GetShapeSize(),
        OP_LOGE(context, "HardShrinkGrad: grad_output and self shape size mismatch: grad=%ld, self=%ld",
                inputShapeGrad.GetShapeSize(), inputShapeSelf.GetShapeSize()),
        return ge::GRAPH_FAILED);

    totalNum = inputShapeGrad.GetShapeSize();

    // Get data type
    const std::set<ge::DataType> supportedDtype = {ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16};
    auto inputDesc = context->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputDesc);
    dataType = inputDesc->GetDataType();
    if (supportedDtype.count(dataType) == 0) {
        OP_LOGE(context, "HardShrinkGrad: unsupported dtype");
        return ge::GRAPH_FAILED;
    }

    // Get lambd attribute (default = 0.5)
    auto lambdAttr = context->GetAttrs()->GetAttrPointer<float>(0);
    if (lambdAttr != nullptr) {
        lambd = *lambdAttr;
    } else {
        lambd = 0.5f;
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

static ge::graphStatus HardShrinkGradTilingFunc(gert::TilingContext* context)
{
    // 1. Get platform info
    uint64_t ubSize;
    int64_t coreNum;
    OP_CHECK_IF(
        GetPlatformInfo(context, ubSize, coreNum) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "GetPlatformInfo error"),
        return ge::GRAPH_FAILED);

    // 2. Get shape, attribute info
    int64_t totalNum;
    ge::DataType dataType;
    float lambd;
    OP_CHECK_IF(
        GetShapeAttrsInfo(context, totalNum, dataType, lambd) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "GetShapeAttrsInfo error"),
        return ge::GRAPH_FAILED);

    // 3. Get workspace size
    OP_CHECK_IF(
        GetWorkspaceSize(context) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "GetWorkspaceSize error"),
        return ge::GRAPH_FAILED);

    // Handle empty tensor
    if (totalNum == 0) {
        context->SetBlockDim(1);
        HardShrinkGradTilingData* tiling = context->GetTilingData<HardShrinkGradTilingData>();
        OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
        memset_s(tiling, sizeof(HardShrinkGradTilingData), 0, sizeof(HardShrinkGradTilingData));
        tiling->lambd = lambd;
        uint32_t dTypeVal = static_cast<uint32_t>(dataType);
        uint64_t useDoubleBuffer = 0;
        ASCENDC_TPL_SEL_PARAM(context, dTypeVal, useDoubleBuffer);
        return ge::GRAPH_SUCCESS;
    }

    // 4. Set tiling data
    HardShrinkGradTilingData* tiling = context->GetTilingData<HardShrinkGradTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
    OP_CHECK_IF(
        memset_s(tiling, sizeof(HardShrinkGradTilingData), 0, sizeof(HardShrinkGradTilingData)) != EOK,
        OP_LOGE(context, "set tiling data error"),
        return ge::GRAPH_FAILED);

    // Multi-core split: distribute elements evenly across cores
    tiling->totalNum = totalNum;
    tiling->blockFactor = CeilDiv(totalNum, coreNum);
    int64_t usedCoreNum = CeilDiv(totalNum, tiling->blockFactor);
    tiling->lambd = lambd;

    // UB split calculation
    // Determine buffer mode
    uint64_t useDoubleBuffer = (totalNum > MIN_SPLIT_THRESHOLD) ? 1 : 0;
    int64_t bufNum = useDoubleBuffer ? 2 : 1;

    // Calculate total bytes per element for UB allocation.
    // For fp32 (Direct path, no Cast):
    //   Queued buffers (fp32): (2 inputs + 1 output) * bufNum * 4
    //   Temp buffers (fp32, single-buffered): abs + lambd + zero = 3 * 4
    //   Total = (3 * bufNum + 3) * 4
    //
    // For fp16 (CastFp32 path):
    //   Queued buffers (fp16): (2 inputs + 1 output) * bufNum * 2
    //   Temp buffers (fp32, single-buffered): gradFp32, selfFp32, outFp32, abs, lambd, zero = 6 * 4
    //   Total = 3 * bufNum * 2 + 6 * 4
    int64_t bytesPerElement;
    if (dataType == ge::DT_FLOAT) {
        bytesPerElement = (3 * bufNum + 3) * 4;
    } else {
        // fp16 and bf16: Cast to fp32 for compute (sizeof(T)=2 for both)
        bytesPerElement = 3 * bufNum * 2 + 6 * 4;
    }

    int64_t ubBlockSize = GetUbBlockSize(context);

    // Reserve space for cmpMask (256 bytes minimum, bit-mask)
    int64_t cmpMaskReserve = 256;
    int64_t availableUbSize = static_cast<int64_t>(ubSize) - cmpMaskReserve;

    int64_t ubFactorRaw = availableUbSize / bytesPerElement;
    tiling->ubFactor = FloorAlign(ubFactorRaw, ubBlockSize);

    // Ensure ubFactor satisfies Compare 256-byte alignment constraint
    // For fp16 path, Compare operates on fp32 tensors, so alignment is 256/4 = 64 elements
    // For fp32 path, alignment is 256/4 = 64 elements
    int64_t cmpAlignElements = 64;  // Compare operates on fp32 in both paths
    tiling->ubFactor = FloorAlign(tiling->ubFactor, cmpAlignElements);

    context->SetBlockDim(usedCoreNum);

    // 5. Set TilingKey
    // Parameter order matches ASCENDC_TPL_ARGS_DECL in hard_shrink_grad_tiling_key.h
    uint32_t dTypeVal = static_cast<uint32_t>(dataType);
    ASCENDC_TPL_SEL_PARAM(context, dTypeVal, useDoubleBuffer);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForHardShrinkGrad([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

struct HardShrinkGradCompileInfo {};

IMPL_OP_OPTILING(HardShrinkGrad)
    .Tiling(HardShrinkGradTilingFunc)
    .TilingParse<HardShrinkGradCompileInfo>(TilingParseForHardShrinkGrad);

} // namespace optiling

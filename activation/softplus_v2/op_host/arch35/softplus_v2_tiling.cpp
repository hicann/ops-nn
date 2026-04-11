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
 * \file softplus_v2_tiling.cpp
 * \brief SoftplusV2 Host-side Tiling implementation (arch35)
 */

#include <limits>
#include "register/op_def_registry.h"
#include "op_common/log/log.h"
#include "op_common/op_host/util/math_util.h"
#include "op_common/op_host/util/platform_util.h"
#include "../../op_kernel/arch35/softplus_v2_tiling_data.h"
#include "../../op_kernel/arch35/softplus_v2_tiling_key.h"

namespace optiling {

using Ops::Base::CeilDiv;
using Ops::Base::FloorDiv;
using Ops::Base::FloorAlign;

constexpr uint32_t WS_SYS_SIZE = 0U;
constexpr uint64_t UB_SIZE_BYTES = 192ULL * 1024; // 192KB

static const gert::Shape g_vec_1_shape = {1};

static inline float SafeInvBeta(float beta)
{
    if (beta == 0.0f) {
        return std::numeric_limits<float>::infinity();
    }
    return 1.0f / beta;
}

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

static ge::graphStatus GetWorkspaceSize(gert::TilingContext* context)
{
    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, currentWorkspace);
    currentWorkspace[0] = WS_SYS_SIZE;
    return ge::GRAPH_SUCCESS;
}

static uint32_t ComputeFp32TileLength(uint64_t ubSize)
{
    // Compute tileLength for FP32 path:
    //   5 main buffers (2 inQ + 2 outQ + 1 tmpBuf) * tileLength * 4
    //   + cmpMask: ceil(tileLength/8) aligned to 256B
    //   + 8KB reserved for Select mode2 internal temp
    //   total <= ubSize
    constexpr uint32_t TYPE_SIZE = 4;         // sizeof(float)
    constexpr uint32_t BUFFER_NUM = 5;        // 2(in) + 2(out) + 1(tmp)
    constexpr uint32_t CMP_ALIGN = 256;       // cmpMask 256B alignment
    constexpr uint32_t ELEM_ALIGN = 64;       // Compare requires 256B = 64 fp32 elements
    constexpr uint64_t SELECT_RESERVE = 8ULL * 1024; // 8KB for Select mode2

    if (ubSize <= SELECT_RESERVE) {
        return ELEM_ALIGN; // minimum tile size if UB is too small
    }
    uint64_t availableSize = ubSize - SELECT_RESERVE;
    // maxTile * (BUFFER_NUM * TYPE_SIZE) + ceil(maxTile/8) aligned 256 <= availableSize
    // Approximate: 20 * maxTile + maxTile/8 <= availableSize
    uint32_t maxTile = static_cast<uint32_t>((availableSize * 8ULL) / (BUFFER_NUM * TYPE_SIZE * 8 + 1));
    maxTile = (maxTile / ELEM_ALIGN) * ELEM_ALIGN;
    if (maxTile == 0) {
        return ELEM_ALIGN;
    }

    // Verify
    uint32_t cmpMaskSize = ((maxTile / 8 + CMP_ALIGN - 1) / CMP_ALIGN) * CMP_ALIGN;
    uint64_t totalBytes = static_cast<uint64_t>(BUFFER_NUM) * maxTile * TYPE_SIZE + cmpMaskSize + SELECT_RESERVE;
    while (totalBytes > ubSize && maxTile > ELEM_ALIGN) {
        maxTile -= ELEM_ALIGN;
        cmpMaskSize = ((maxTile / 8 + CMP_ALIGN - 1) / CMP_ALIGN) * CMP_ALIGN;
        totalBytes = static_cast<uint64_t>(BUFFER_NUM) * maxTile * TYPE_SIZE + cmpMaskSize + SELECT_RESERVE;
    }

    return maxTile;
}

static uint32_t ComputeFp16TileLength(uint64_t ubSize)
{
    // Compute tileLength for FP16/BF16 path:
    //   4 main buffers: 2*tileLength*2(inQ) + 2*tileLength*2(outQ) + tileLength*4(castBuf) + tileLength*4(tmpBuf)
    //   = 16 * tileLength
    //   + cmpMask: ceil(tileLength/8) aligned 256B
    //   + 8KB reserved for Select mode2 internal temp
    //   total <= ubSize
    constexpr uint32_t BYTES_PER_ELEM = 16;   // 4*2(in) + 4*2(out) + 4(cast) + 4(tmp)
    constexpr uint32_t CMP_ALIGN = 256;
    constexpr uint32_t ELEM_ALIGN = 64;       // Compare 256B alignment (fp32 compute)
    constexpr uint64_t SELECT_RESERVE = 8ULL * 1024; // 8KB for Select mode2

    if (ubSize <= SELECT_RESERVE) {
        return ELEM_ALIGN; // minimum tile size if UB is too small
    }
    uint64_t availableSize = ubSize - SELECT_RESERVE;
    uint32_t maxTile = static_cast<uint32_t>((availableSize * 8ULL) / (BYTES_PER_ELEM * 8 + 1));
    maxTile = (maxTile / ELEM_ALIGN) * ELEM_ALIGN;
    if (maxTile == 0) {
        return ELEM_ALIGN;
    }

    uint32_t cmpMaskSize = ((maxTile / 8 + CMP_ALIGN - 1) / CMP_ALIGN) * CMP_ALIGN;
    uint64_t totalBytes = static_cast<uint64_t>(BYTES_PER_ELEM) * maxTile + cmpMaskSize + SELECT_RESERVE;
    while (totalBytes > ubSize && maxTile > ELEM_ALIGN) {
        maxTile -= ELEM_ALIGN;
        cmpMaskSize = ((maxTile / 8 + CMP_ALIGN - 1) / CMP_ALIGN) * CMP_ALIGN;
        totalBytes = static_cast<uint64_t>(BYTES_PER_ELEM) * maxTile + cmpMaskSize + SELECT_RESERVE;
    }

    return maxTile;
}

static ge::graphStatus GetInputInfo(gert::TilingContext* context,
                                    int64_t& totalLength, ge::DataType& dataType,
                                    float& beta, float& threshold)
{
    auto inputX = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputX);
    auto inputShapeX = EnsureNotScalar(inputX->GetStorageShape());
    totalLength = inputShapeX.GetShapeSize();

    auto inputDesc = context->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputDesc);
    dataType = inputDesc->GetDataType();

    const gert::RuntimeAttrs *attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    beta = (attrs->GetAttrPointer<float>(0) != nullptr) ? *(attrs->GetAttrPointer<float>(0)) : 1.0f;
    threshold = (attrs->GetAttrPointer<float>(1) != nullptr) ? *(attrs->GetAttrPointer<float>(1)) : 20.0f;

    // beta=0 is allowed: invBeta = 1/0 = +inf, output = inf (aligned with PyTorch)
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus HandleEmptyTensor(gert::TilingContext* context,
                                         ge::DataType dataType, float beta, float threshold)
{
    OP_CHECK_IF(
        GetWorkspaceSize(context) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "GetWorkspaceSize error"), return ge::GRAPH_FAILED);
    SoftplusV2TilingData* tiling = context->GetTilingData<SoftplusV2TilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
    OP_CHECK_IF(
        memset_s(tiling, sizeof(SoftplusV2TilingData), 0, sizeof(SoftplusV2TilingData)) != EOK,
        OP_LOGE(context, "set tiling data error"), return ge::GRAPH_FAILED);
    tiling->beta = beta;
    tiling->threshold = threshold;
    tiling->invBeta = SafeInvBeta(beta);
    context->SetBlockDim(1);
    uint32_t dTypeX = static_cast<uint32_t>(dataType);
    ASCENDC_TPL_SEL_PARAM(context, dTypeX);
    return ge::GRAPH_SUCCESS;
}

static uint32_t ComputeTileLength(ge::DataType dataType, uint64_t ubSize)
{
    if (dataType == ge::DT_FLOAT) {
        return ComputeFp32TileLength(ubSize);
    }
    return ComputeFp16TileLength(ubSize);
}

static int64_t ComputeUsedCoreNum(int64_t totalLength, ge::DataType dataType, int64_t coreNum)
{
    constexpr int64_t MIN_BYTES_PER_CORE = 2048; // 2KB
    uint32_t elemSize = (dataType == ge::DT_FLOAT) ? 4 : 2;
    int64_t totalBytes = totalLength * elemSize;
    int64_t maxCores = CeilDiv(totalBytes, MIN_BYTES_PER_CORE);
    int64_t effectiveCoreNum = std::min(maxCores, coreNum);
    if (effectiveCoreNum < 1) {
        effectiveCoreNum = 1;
    }
    int64_t blockFactor = CeilDiv(totalLength, effectiveCoreNum);
    return CeilDiv(totalLength, blockFactor);
}

static ge::graphStatus FillTilingData(gert::TilingContext* context,
                                      int64_t totalLength, int64_t usedCoreNum,
                                      uint32_t tileLength, ge::DataType dataType,
                                      float beta, float threshold)
{
    SoftplusV2TilingData* tiling = context->GetTilingData<SoftplusV2TilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
    OP_CHECK_IF(
        memset_s(tiling, sizeof(SoftplusV2TilingData), 0, sizeof(SoftplusV2TilingData)) != EOK,
        OP_LOGE(context, "set tiling data error"), return ge::GRAPH_FAILED);

    int64_t effectiveCoreNum = usedCoreNum;
    if (effectiveCoreNum < 1) {
        effectiveCoreNum = 1;
    }
    tiling->totalLength = totalLength;
    tiling->blockFactor = CeilDiv(totalLength, effectiveCoreNum);
    tiling->tileLength = static_cast<int64_t>(tileLength);
    tiling->beta = beta;
    tiling->threshold = threshold;
    tiling->invBeta = SafeInvBeta(beta);

    context->SetBlockDim(usedCoreNum);

    OP_LOGI(context, "totalLength:%ld blockFactor:%d tileLength:%ld beta:%f threshold:%f invBeta:%f",
        tiling->totalLength, tiling->blockFactor, tiling->tileLength, tiling->beta, tiling->threshold, tiling->invBeta);

    uint32_t dTypeX = static_cast<uint32_t>(dataType);
    ASCENDC_TPL_SEL_PARAM(context, dTypeX);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus SoftplusV2TilingFunc(gert::TilingContext* context)
{
    // 1. Platform info
    uint64_t ubSize;
    int64_t coreNum;
    OP_CHECK_IF(
        GetPlatformInfo(context, ubSize, coreNum) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "GetPlatformInfo error"), return ge::GRAPH_FAILED);

    // 2. Input info (shape, dtype, attrs)
    int64_t totalLength;
    ge::DataType dataType;
    float beta, threshold;
    OP_CHECK_IF(
        GetInputInfo(context, totalLength, dataType, beta, threshold) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "GetInputInfo error"), return ge::GRAPH_FAILED);

    // 3. Empty tensor early return
    if (totalLength == 0) {
        return HandleEmptyTensor(context, dataType, beta, threshold);
    }

    // 4. Workspace
    OP_CHECK_IF(
        GetWorkspaceSize(context) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "GetWorkspaceSize error"), return ge::GRAPH_FAILED);

    // 5. Compute tileLength and multi-core split
    uint32_t tileLength = ComputeTileLength(dataType, ubSize);
    int64_t usedCoreNum = ComputeUsedCoreNum(totalLength, dataType, coreNum);

    // 6. Fill TilingData and set BlockDim/TilingKey
    return FillTilingData(context, totalLength, usedCoreNum, tileLength, dataType, beta, threshold);
}

static ge::graphStatus TilingParseForSoftplusV2([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

struct SoftplusV2CompileInfo {};

IMPL_OP_OPTILING(SoftplusV2)
    .Tiling(SoftplusV2TilingFunc)
    .TilingParse<SoftplusV2CompileInfo>(TilingParseForSoftplusV2);

} // namespace optiling

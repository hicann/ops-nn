/**
 * This file is part of the OpenBOAT project at Harbin Institute of Technology (HIT)
 * and is contributed to the CANN Open Software.
 *
 * Copyright (c) 2026 AISS Group, Harbin Institute of Technology (HIT).
 * All Rights Reserved.
 *
 * Authors (accounts):
 * - Shi Xiangyang <@shi-xiangyang225>
 * - Su Tonghua <@sutonghua>
 *
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file relu_grad_v3_tiling.cpp
 * \brief ReluGradV3算子的tiling(分块)策略实现，支持多核切分和多数据类型
 */

#include "log/log.h"
#include "util/math_util.h"
#include "op_host/util/platform_util.h"
#include "op_host/tiling_util.h"
#include "op_host/tiling_templates_registry.h"
#include "../op_kernel/relu_grad_v3_tiling_data.h"
#include "../op_kernel/relu_grad_v3_tiling_key.h"
#include <algorithm>
#include <array>

namespace optiling {

constexpr uint32_t WS_SYS_SIZE = 0U;
constexpr uint32_t BUFFER_NUM = 2;
constexpr uint32_t BLOCK_SIZE = 256; // api需要
constexpr uint32_t CORE_ALIGN_BYTES = BLOCK_SIZE;
// Select模式需要预留8KB UB临时空间
constexpr uint32_t SELECT_RESERVED_UB = 8 * 1024;

static const gert::Shape g_vec_1_shape = {1};

struct ReluGradV3CompileInfo {};

inline const gert::Shape& EnsureNotScalar(const gert::Shape& in_shape)
{
    if (in_shape.IsScalar()) {
        return g_vec_1_shape;
    }
    return in_shape;
}

static int64_t GetTypeSize(ge::DataType dtype)
{
    if (dtype == ge::DT_INT8 || dtype == ge::DT_UINT8) {
        return 1;
    }
    if (dtype == ge::DT_FLOAT16 || dtype == ge::DT_BF16) {
        return 2;
    }
    return 4;
}

static ge::graphStatus GetPlatformInfo(gert::TilingContext* context, uint64_t& ubSize, int64_t& coreNum)
{
    fe::PlatFormInfos* platformInfoPtr = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfoPtr);

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);

    coreNum = ascendcPlatform.GetCoreNum();
    OP_CHECK_IF(coreNum == 0, OP_LOGE(context, "coreNum is 0"), return ge::GRAPH_FAILED);

    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    OP_CHECK_IF(ubSize == 0, OP_LOGE(context, "ubSize is 0"), return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

static uint64_t GetShapeSize(const std::array<uint64_t, 8>& shape, size_t dimNum)
{
    uint64_t size = 1;
    for (size_t i = 0; i < dimNum; ++i) {
        size *= shape[i];
    }
    return size;
}

static ge::graphStatus BuildBroadcastInfo(gert::TilingContext* context, const gert::Shape& xShape,
                                          const gert::Shape& yShape, std::array<uint64_t, 8>& outShape,
                                          std::array<uint64_t, 8>& xStrides, std::array<uint64_t, 8>& yStrides,
                                          uint64_t& dimNum, uint64_t& xElementNum, uint64_t& yElementNum,
                                          bool& broadcastMode)
{
    size_t xDimNum = xShape.IsScalar() ? 1 : xShape.GetDimNum();
    size_t yDimNum = yShape.IsScalar() ? 1 : yShape.GetDimNum();
    size_t outDimNum = std::max(xDimNum, yDimNum);
    OP_CHECK_IF(outDimNum > 8, OP_LOGE(context, "broadcast dim num exceeds 8"), return ge::GRAPH_FAILED);

    std::array<uint64_t, 8> xAlignedShape = {1, 1, 1, 1, 1, 1, 1, 1};
    std::array<uint64_t, 8> yAlignedShape = {1, 1, 1, 1, 1, 1, 1, 1};
    outShape = {1, 1, 1, 1, 1, 1, 1, 1};
    xStrides = {0, 0, 0, 0, 0, 0, 0, 0};
    yStrides = {0, 0, 0, 0, 0, 0, 0, 0};

    for (size_t i = 0; i < outDimNum; ++i) {
        uint64_t xDim = 1;
        uint64_t yDim = 1;
        if (i >= outDimNum - xDimNum) {
            xDim = static_cast<uint64_t>(xShape.IsScalar() ? 1 : xShape.GetDim(i - (outDimNum - xDimNum)));
        }
        if (i >= outDimNum - yDimNum) {
            yDim = static_cast<uint64_t>(yShape.IsScalar() ? 1 : yShape.GetDim(i - (outDimNum - yDimNum)));
        }
        OP_CHECK_IF(xDim != yDim && xDim != 1 && yDim != 1, OP_LOGE(context, "input shapes are not broadcastable"),
                    return ge::GRAPH_FAILED);
        xAlignedShape[i] = xDim;
        yAlignedShape[i] = yDim;
        outShape[i] = std::max(xDim, yDim);
    }

    uint64_t xStride = 1;
    uint64_t yStride = 1;
    for (int64_t i = static_cast<int64_t>(outDimNum) - 1; i >= 0; --i) {
        xStrides[i] = (xAlignedShape[i] == 1 && outShape[i] != 1) ? 0 : xStride;
        yStrides[i] = (yAlignedShape[i] == 1 && outShape[i] != 1) ? 0 : yStride;
        xStride *= xAlignedShape[i];
        yStride *= yAlignedShape[i];
    }

    dimNum = static_cast<uint64_t>(outDimNum);
    xElementNum = GetShapeSize(xAlignedShape, outDimNum);
    yElementNum = GetShapeSize(yAlignedShape, outDimNum);
    broadcastMode = (xElementNum != GetShapeSize(outShape, outDimNum)) ||
                    (yElementNum != GetShapeSize(outShape, outDimNum));
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GetShapeAttrsInfo(gert::TilingContext* context, int64_t& totalIdx, ge::DataType& dataType,
                                  std::array<uint64_t, 8>& outShape, std::array<uint64_t, 8>& xStrides,
                                  std::array<uint64_t, 8>& yStrides, uint64_t& dimNum, uint64_t& xElementNum,
                                  uint64_t& yElementNum, bool& broadcastMode)
{
    auto inputX = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputX);
    auto inputShapeX = EnsureNotScalar(inputX->GetStorageShape());

    auto inputY = context->GetInputShape(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputY);
    auto inputShapeY = EnsureNotScalar(inputY->GetStorageShape());

    auto outZ = context->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, outZ);
    (void)EnsureNotScalar(outZ->GetStorageShape());

    OP_CHECK_IF(BuildBroadcastInfo(context, inputShapeX, inputShapeY, outShape, xStrides, yStrides, dimNum, xElementNum,
                                   yElementNum, broadcastMode) != ge::GRAPH_SUCCESS,
                OP_LOGE(context, "BuildBroadcastInfo error"), return ge::GRAPH_FAILED);
    totalIdx = static_cast<int64_t>(GetShapeSize(outShape, dimNum));

    const std::set<ge::DataType> supportedDtype = {ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16,
                                                   ge::DT_INT32, ge::DT_UINT8,   ge::DT_INT8};
    auto inputDesc = context->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputDesc);
    auto inputYDesc = context->GetInputDesc(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputYDesc);
    auto outputDesc = context->GetOutputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, outputDesc);
    dataType = inputDesc->GetDataType();

    if (supportedDtype.count(dataType) == 0) {
        OP_LOGE(context, "invalid dtype");
        return ge::GRAPH_FAILED;
    }
    if (inputYDesc->GetDataType() != dataType || outputDesc->GetDataType() != dataType) {
        OP_LOGE(context, "x, y and z dtype must be same");
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GetWorkspaceSize(gert::TilingContext* context)
{
    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, currentWorkspace);
    currentWorkspace[0] = WS_SYS_SIZE;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus ReluGradV3TilingFunc(gert::TilingContext* context)
{
    uint64_t ubSize;
    int64_t coreNum;
    OP_CHECK_IF(GetPlatformInfo(context, ubSize, coreNum) != ge::GRAPH_SUCCESS,
                OP_LOGE(context, "GetPlatformInfo error"), return ge::GRAPH_FAILED);

    int64_t totalIdx;
    ge::DataType dataType;
    std::array<uint64_t, 8> outShape = {0};
    std::array<uint64_t, 8> xStrides = {0};
    std::array<uint64_t, 8> yStrides = {0};
    uint64_t dimNum = 0;
    uint64_t xElementNum = 0;
    uint64_t yElementNum = 0;
    bool broadcastMode = false;

    OP_CHECK_IF(GetShapeAttrsInfo(context, totalIdx, dataType, outShape, xStrides, yStrides, dimNum, xElementNum,
                                  yElementNum, broadcastMode) != ge::GRAPH_SUCCESS,
                OP_LOGE(context, "GetShapeAttrsInfo error"), return ge::GRAPH_FAILED);

    OP_CHECK_IF(GetWorkspaceSize(context) != ge::GRAPH_SUCCESS, OP_LOGE(context, "GetWorkspaceSize error"),
                return ge::GRAPH_FAILED);

    ReluGradV3TilingData* tiling = context->GetTilingData<ReluGradV3TilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);

    OP_CHECK_IF(memset_s(tiling, sizeof(ReluGradV3TilingData), 0, sizeof(ReluGradV3TilingData)) != EOK,
                OP_LOGE(context, "set tiling data error"), return ge::GRAPH_FAILED);

    // 总数据量
    int64_t totalLength = totalIdx;
    tiling->totalLength = static_cast<uint64_t>(totalLength);

    // 根据数据类型获取元素字节大小
    int64_t typeSize = GetTypeSize(dataType);
    OP_CHECK_IF(typeSize <= 0, OP_LOGE(context, "Unsupported dtype, type size is invalid"), return ge::GRAPH_FAILED);

    int64_t availableUbSize = static_cast<int64_t>(ubSize > SELECT_RESERVED_UB ? ubSize - SELECT_RESERVED_UB : ubSize);
    // TQue storage is double buffered; selectors and cast temporaries are not.
    int64_t bytesPerElement = BUFFER_NUM * 2 * typeSize;
    if (dataType == ge::DT_INT32) {
        bytesPerElement = BUFFER_NUM * 3 * typeSize;
    } else if (dataType == ge::DT_FLOAT) {
        bytesPerElement += sizeof(uint8_t);
    } else if (dataType == ge::DT_FLOAT16) {
        bytesPerElement = BUFFER_NUM * 3 * typeSize + sizeof(uint8_t);
    } else if (dataType == ge::DT_BF16) {
        bytesPerElement += sizeof(uint8_t) + 3 * sizeof(float);
    } else if (dataType == ge::DT_INT8 || dataType == ge::DT_UINT8) {
        bytesPerElement += 2 * sizeof(uint16_t);
    }
    int64_t tileAlignElements = BLOCK_SIZE / typeSize;
    int64_t ubPartDataNum = (availableUbSize / bytesPerElement / tileAlignElements) * tileAlignElements;
    OP_CHECK_IF(ubPartDataNum <= 0, OP_LOGE(context, "UB is too small"), return ge::GRAPH_FAILED);
    tiling->ubPartDataNum = static_cast<uint64_t>(ubPartDataNum);

    // Keep every inter-core GM boundary aligned to the vector block size. The
    // final core owns the only unaligned tail, so rounded DataCopy blocks from
    // one core cannot overwrite data produced by the next core.
    int64_t maxUsefulCores = Ops::Base::CeilDiv(totalLength, ubPartDataNum);
    if (maxUsefulCores == 0) {
        maxUsefulCores = 1;
    }
    int64_t usedCoreNum = std::min(static_cast<int64_t>(coreNum), maxUsefulCores);
    if (usedCoreNum <= 0) {
        usedCoreNum = 1;
    }
    int64_t coreAlignElements = CORE_ALIGN_BYTES / typeSize;
    int64_t averageCoreDataNum = Ops::Base::CeilDiv(totalLength, usedCoreNum);
    int64_t alignedCoreDataNum = Ops::Base::CeilDiv(averageCoreDataNum, coreAlignElements) * coreAlignElements;
    usedCoreNum = Ops::Base::CeilDiv(totalLength, alignedCoreDataNum);
    int64_t bigCoreDataNum = alignedCoreDataNum;
    int64_t smallCoreDataNum = totalLength - (usedCoreNum - 1) * alignedCoreDataNum;
    int64_t tailBlockNum = usedCoreNum - 1;
    if (smallCoreDataNum == bigCoreDataNum) {
        tailBlockNum = 0;
    }
    if (usedCoreNum == 1) {
        bigCoreDataNum = 0;
        tailBlockNum = 0;
    }

    tiling->bigCoreDataNum = static_cast<uint64_t>(bigCoreDataNum);
    tiling->smallCoreDataNum = static_cast<uint64_t>(smallCoreDataNum);
    tiling->tailBlockNum = static_cast<uint64_t>(tailBlockNum);
    tiling->broadcastMode = broadcastMode ? 1U : 0U;
    tiling->dimNum = dimNum;
    tiling->xElementNum = xElementNum;
    tiling->yElementNum = yElementNum;
    for (size_t i = 0; i < 8; ++i) {
        tiling->outShape[i] = outShape[i];
        tiling->xStrides[i] = xStrides[i];
        tiling->yStrides[i] = yStrides[i];
    }

    // 计算循环次数和尾数据量
    tiling->bigCoreLoopNum = (bigCoreDataNum > 0) ?
                                 static_cast<uint64_t>(Ops::Base::CeilDiv(bigCoreDataNum, ubPartDataNum)) :
                                 0;
    tiling->bigCoreTailDataNum = (bigCoreDataNum > 0) ?
                                     static_cast<uint64_t>(bigCoreDataNum -
                                                           (static_cast<int64_t>(tiling->bigCoreLoopNum) - 1) *
                                                               ubPartDataNum) :
                                     0;
    tiling->smallCoreLoopNum = static_cast<uint64_t>(Ops::Base::CeilDiv(smallCoreDataNum, ubPartDataNum));
    tiling->smallCoreTailDataNum = static_cast<uint64_t>(
        smallCoreDataNum - (static_cast<int64_t>(tiling->smallCoreLoopNum) - 1) * ubPartDataNum);

    context->SetBlockDim(static_cast<uint32_t>(usedCoreNum));
    uint64_t tilingKeyMode = static_cast<uint64_t>(broadcastMode ? RELU_GRAD_V3_TPL_SCH_MODE_BROADCAST :
                                                                   RELU_GRAD_V3_TPL_SCH_MODE_NORMAL);
    context->SetTilingKey(GET_TPL_TILING_KEY(tilingKeyMode));

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForReluGradV3([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(ReluGradV3).Tiling(ReluGradV3TilingFunc).TilingParse<ReluGradV3CompileInfo>(TilingParseForReluGradV3);
} // namespace optiling

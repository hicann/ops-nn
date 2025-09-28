/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file adaptive_avg_pool3d_tiling.cpp
 * \brief
 *
 */

#include "adaptive_avg_pool3d_tiling.h"
#include <sstream>

namespace {
constexpr size_t X_INDEX = 0;
constexpr size_t Y_INDEX = 0;
constexpr size_t OUTPUT_SIZE_INDEX = 0;
constexpr size_t X_DIMS = 5;
constexpr size_t DIM0 = 0;
constexpr size_t DIM1 = 1;
constexpr size_t DIM2 = 2;
constexpr size_t DIMC = 4;
constexpr size_t OUTPUT_SIZE_DIMS = 3;

constexpr uint32_t RESERVED_UB_SIZE = 10U * 1024U;
constexpr uint32_t INDEX_BUF_SIZE = 3U * 1024U;
constexpr uint32_t INDEX_BUF_NUM = 6;

constexpr int32_t BF16_DTYPE_KEY = 0;
constexpr int32_t FP16_DTYPE_KEY = 1;
constexpr int32_t FP32_DTYPE_KEY = 2;

constexpr int32_t MODE_SPLIT_C = 1;
constexpr int32_t MODE_SPLIT_W = 2;
constexpr int32_t MODE_MULTI_W = 3;

constexpr int64_t BLOCK_SIZE = 32;
constexpr int64_t MAX_INPUT_TILE_NUM = 4095;
constexpr int64_t MAX_INPUT_W_NUM = 4095;

struct TilingParams {
    uint64_t dimC = 0;
    uint64_t dimN = 0;
    uint64_t cTileLength = 0;
    uint64_t inD = 0;
    uint64_t inH = 0;
    uint64_t inW = 0;
    uint64_t outD = 0;
    uint64_t outH = 0;
    uint64_t outW = 0;
    uint64_t formerLength = 0;
    uint64_t formerNum = 0;
    uint64_t tailLength = 0;
    uint64_t tailNum = 0;
    uint64_t indexBufLen = 0;
    uint64_t windowWNum = 0;
    uint64_t maxWindowWLength = 0;
    uint64_t inputTileNum = 0;
    uint64_t atomicAddNum = 0;

    uint32_t ubSize = 0;
    uint32_t coreNum = 0;
    int32_t dataTypeKey = 0;
};

} // namespace

namespace optiling {
static void ComputeCoreTilingStrategy(TilingParams& params, int32_t& usedCoreNum)
{
    uint64_t outputNum = params.dimN * params.outD * params.outH * params.outW;
    if (outputNum < params.coreNum) {
        params.formerNum = outputNum;
        params.tailNum = 0UL;
        params.formerLength = 1UL;
        params.tailLength = 0UL;
        usedCoreNum = static_cast<int32_t>(outputNum);
    } else if (outputNum % static_cast<uint64_t>(params.coreNum) == 0UL) {
        params.formerNum = params.coreNum;
        params.tailNum = 0UL;
        params.formerLength = outputNum / params.coreNum;
        params.tailLength = 0UL;
        usedCoreNum = static_cast<int32_t>(params.coreNum);
    } else {
        params.formerNum = outputNum % params.coreNum;
        params.tailNum = params.coreNum - params.formerNum;
        params.formerLength = outputNum / params.coreNum + 1UL;
        params.tailLength = outputNum / params.coreNum;
        usedCoreNum = static_cast<int32_t>(params.coreNum);
    }
}

static void ComputeUBTilingStrategy(TilingParams& params, int32_t& mode)
{
    int32_t dataTypeSize = params.dataTypeKey == FP32_DTYPE_KEY ? 4 : 2;
    int32_t needCast = params.dataTypeKey == FP32_DTYPE_KEY ? 0 : 1;

    params.maxWindowWLength = (params.outW + params.inW + params.outW - 1UL) / params.outW;

    uint64_t alignNum = static_cast<uint64_t>(BLOCK_SIZE) / dataTypeSize;
    uint64_t tileLen =
        static_cast<uint64_t>(params.ubSize) /
        (2UL * static_cast<uint64_t>(dataTypeSize) + sizeof(float) * static_cast<uint64_t>(1 + needCast)) / alignNum *
        alignNum;
    uint64_t alignC = (params.dimC + alignNum - 1UL) / alignNum * alignNum;

    uint64_t doubleC = 2UL * alignC;
    if (doubleC > tileLen) {
        mode = MODE_SPLIT_C;
        params.cTileLength = alignC > tileLen ? tileLen : alignC;
        uint64_t tileTailLen = params.dimC % params.cTileLength;
        params.atomicAddNum = (tileTailLen < alignNum) && (tileTailLen != 0UL) ? 1UL : 0UL;
        return;
    }

    if (params.dimC < alignNum) {
        params.atomicAddNum = (alignC - 1UL) / params.dimC;
    }

    uint64_t inputTileNum = (params.ubSize / alignC - static_cast<uint64_t>(dataTypeSize) -
                             sizeof(float) * static_cast<uint64_t>(1 + needCast)) /
                            static_cast<uint64_t>(dataTypeSize);
    if (inputTileNum < params.maxWindowWLength) {
        mode = MODE_SPLIT_W;
        params.inputTileNum = inputTileNum < static_cast<uint64_t>(MAX_INPUT_TILE_NUM) ?
                                  inputTileNum :
                                  static_cast<uint64_t>(MAX_INPUT_TILE_NUM);
        return;
    }

    uint64_t windowWNum = (params.ubSize / alignC - sizeof(float) * needCast) /
                          ((params.maxWindowWLength + 1UL) * static_cast<uint64_t>(dataTypeSize) + sizeof(float));

    mode = MODE_MULTI_W;
    windowWNum = windowWNum * params.maxWindowWLength < static_cast<uint64_t>(MAX_INPUT_W_NUM) ?
                     windowWNum :
                     static_cast<uint64_t>(MAX_INPUT_W_NUM) / params.maxWindowWLength;
    params.windowWNum = windowWNum < params.outW ? windowWNum : params.outW;

    if (windowWNum == 0UL) {
        mode = MODE_SPLIT_C;
        params.cTileLength = alignC > tileLen ? tileLen : alignC;
    }
}

static void SetTiling(const TilingParams& params, AdaptiveAvgPool3dTilingData& tiling)
{
    tiling.set_dimC(params.dimC);
    tiling.set_cTileLength(params.cTileLength);
    tiling.set_inD(params.inD);
    tiling.set_inH(params.inH);
    tiling.set_inW(params.inW);
    tiling.set_outD(params.outD);
    tiling.set_outH(params.outH);
    tiling.set_outW(params.outW);
    tiling.set_formerLength(params.formerLength);
    tiling.set_formerNum(params.formerNum);
    tiling.set_tailLength(params.tailLength);
    tiling.set_tailNum(params.tailNum);
    tiling.set_indexBufLen(params.indexBufLen);
    tiling.set_windowWNum(params.windowWNum);
    tiling.set_maxWindowWLength(params.maxWindowWLength);
    tiling.set_inputTileNum(params.inputTileNum);
    tiling.set_atomicAddNum(params.atomicAddNum);
}

static void PrintTiling(
    const gert::TilingContext* context, AdaptiveAvgPool3dTilingData& tiling, uint32_t tilingKey, int32_t usedCoreNum)
{
    auto nodeName = context->GetNodeName();
    OP_LOGD(nodeName, "tilingKey:          %d.", tilingKey);
    OP_LOGD(nodeName, "usedCoreNum:        %d.", usedCoreNum);
    OP_LOGD(nodeName, "dimC:               %ld.", tiling.get_dimC());
    OP_LOGD(nodeName, "cTileLength:        %ld.", tiling.get_cTileLength());
    OP_LOGD(nodeName, "inD:                %ld.", tiling.get_inD());
    OP_LOGD(nodeName, "inH:                %ld.", tiling.get_inH());
    OP_LOGD(nodeName, "inW:                %ld.", tiling.get_inW());
    OP_LOGD(nodeName, "outD:               %ld.", tiling.get_outD());
    OP_LOGD(nodeName, "outH:               %ld.", tiling.get_outH());
    OP_LOGD(nodeName, "outW:               %ld.", tiling.get_outW());
    OP_LOGD(nodeName, "formerLength:       %ld.", tiling.get_formerLength());
    OP_LOGD(nodeName, "formerNum:          %ld.", tiling.get_formerNum());
    OP_LOGD(nodeName, "tailLength:         %ld.", tiling.get_tailLength());
    OP_LOGD(nodeName, "tailNum:            %ld.", tiling.get_tailNum());
    OP_LOGD(nodeName, "indexBufLen:        %ld.", tiling.get_indexBufLen());
    OP_LOGD(nodeName, "windowWNum:         %ld.", tiling.get_windowWNum());
    OP_LOGD(nodeName, "maxWindowWLength:   %ld.", tiling.get_maxWindowWLength());
    OP_LOGD(nodeName, "inputTileNum:       %ld.", tiling.get_inputTileNum());
    OP_LOGD(nodeName, "atomicAddNum:       %ld.", tiling.get_atomicAddNum());
}

static bool GetDataTypeKey(ge::DataType dataType, int32_t& dataTypeKey)
{
    switch (dataType) {
        case ge::DT_FLOAT16:
            dataTypeKey = FP16_DTYPE_KEY;
            break;
        case ge::DT_BF16:
            dataTypeKey = BF16_DTYPE_KEY;
            break;
        case ge::DT_FLOAT:
            dataTypeKey = FP32_DTYPE_KEY;
            break;
        default:
            return false;
    }

    return true;
}

static ge::graphStatus KernelTiling(
    gert::TilingContext* context, const AdaptiveAvgPool3dCompileInfo* compileInfo, TilingParams& params)
{
    auto nodeName = context->GetNodeName();
    OP_LOGD(nodeName, "Tiling start.");

    params.coreNum = static_cast<uint32_t>(compileInfo->totalCoreNum);
    params.ubSize = static_cast<uint32_t>(compileInfo->ubSizePlatForm) - RESERVED_UB_SIZE - INDEX_BUF_SIZE;
    ;

    int32_t usedCoreNum = 0;
    ComputeCoreTilingStrategy(params, usedCoreNum);

    int32_t modeKey = MODE_SPLIT_C;
    ComputeUBTilingStrategy(params, modeKey);

    AdaptiveAvgPool3dTilingData tiling;
    SetTiling(params, tiling);

    uint32_t tilingKey = static_cast<uint32_t>(modeKey) * 10U + static_cast<uint32_t>(params.dataTypeKey);
    context->SetTilingKey(tilingKey);
    context->SetBlockDim(usedCoreNum);

    PrintTiling(context, tiling, tilingKey, usedCoreNum);

    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

    auto sysWorkspaceSize = compileInfo->sysWorkspaceSize;
    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, currentWorkspace);
    currentWorkspace[0] = sysWorkspaceSize;

    OP_LOGD(nodeName, "Tiling end.");

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus Tiling4AdaptiveAvgPool3d(gert::TilingContext* context)
{
    auto nodeName = context->GetNodeName();
    OP_LOGD(nodeName, "Tiling4AdaptiveAvgPool3d start.");

    auto compileInfo = static_cast<const AdaptiveAvgPool3dCompileInfo*>(context->GetCompileInfo());

    const gert::Shape xShape = context->GetInputShape(X_INDEX)->GetStorageShape();
    OP_CHECK_IF(
        xShape.GetDimNum() != X_DIMS, OP_LOGE(nodeName, "Check x shape failed, the dims of x not equal 5."),
        return ge::GRAPH_FAILED);

    auto dataType = context->GetInputDesc(X_INDEX)->GetDataType();
    int32_t dataTypeKey = FP32_DTYPE_KEY;
    OP_CHECK_IF(
        GetDataTypeKey(dataType, dataTypeKey) == false,
        OP_LOGE(nodeName, "The dtype of input must be in [float32, float16, bfloat16]."), return ge::GRAPH_FAILED);

    auto attrPtr = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrPtr);
    auto outputSizePtr = attrPtr->GetAttrPointer<gert::ContinuousVector>(OUTPUT_SIZE_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, outputSizePtr);
    OP_CHECK_IF(
        outputSizePtr->GetSize() != OUTPUT_SIZE_DIMS,
        OP_LOGE(nodeName, "Check output_size failed, the size of output_size not equal 3."), return ge::GRAPH_FAILED);
    const int64_t* outputSize = static_cast<const int64_t*>(outputSizePtr->GetData());
    OP_CHECK_IF(
        outputSize[DIM0] <= 0 || outputSize[DIM1] <= 0 || outputSize[DIM2] <= 0,
        OP_LOGE(nodeName, "Check output_size failed, the value of output_size should be greater than 0."),
        return ge::GRAPH_FAILED);

    TilingParams params;
    params.dimN = xShape.GetDim(DIM0);
    params.inD = xShape.GetDim(DIM0 + 1);
    params.inH = xShape.GetDim(DIM1 + 1);
    params.inW = xShape.GetDim(DIM2 + 1);
    params.dimC = xShape.GetDim(DIMC);
    params.outD = static_cast<uint64_t>(outputSize[DIM0]);
    params.outH = static_cast<uint64_t>(outputSize[DIM1]);
    params.outW = static_cast<uint64_t>(outputSize[DIM2]);
    params.indexBufLen = static_cast<uint64_t>(INDEX_BUF_SIZE) / static_cast<uint64_t>(INDEX_BUF_NUM) / sizeof(int64_t);
    params.dataTypeKey = dataTypeKey;

    ge::graphStatus tilingStatus = KernelTiling(context, compileInfo, params);

    OP_LOGD(nodeName, "Tiling4AdaptiveAvgPool3d end.");
    return tilingStatus;
}

static ge::graphStatus TilingPrepare4AdaptiveAvgPool3d(gert::TilingParseContext* context)
{
    auto nodeName = context->GetNodeName();
    OP_LOGD(nodeName, "TilingPrepare4AdaptiveAvgPool3d start.");
    auto compileInfo = context->GetCompiledInfo<AdaptiveAvgPool3dCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);
    auto platformInfo = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    compileInfo->totalCoreNum = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF(
        (compileInfo->totalCoreNum <= 0), OP_LOGE(nodeName, "Failed to get core num."), return ge::GRAPH_FAILED);
    compileInfo->sysWorkspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();
    uint64_t ubSizePlatForm;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatForm);
    compileInfo->ubSizePlatForm = static_cast<int64_t>(ubSizePlatForm);
    OP_CHECK_IF(
        (compileInfo->ubSizePlatForm <= 0), OP_LOGE(nodeName, "Failed to get ub size."), return ge::GRAPH_FAILED);
    OP_LOGD(nodeName, "TilingPrepare4AdaptiveAvgPool3d end.");

    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(AdaptiveAvgPool3d)
    .Tiling(Tiling4AdaptiveAvgPool3d)
    .TilingParse<AdaptiveAvgPool3dCompileInfo>(TilingPrepare4AdaptiveAvgPool3d);

} // namespace optiling

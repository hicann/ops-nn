/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
*/

/*!
 * \file adaptive_avg_pool3d_grad_tiling.cpp
 * \brief
 *
 */

#include "adaptive_avg_pool3d_grad_tiling.h"
#include <iostream>

namespace optiling {
constexpr int64_t D_DIM = 0;
constexpr int64_t H_DIM = 1;
constexpr int64_t W_DIM = 2;
constexpr int64_t RESERVED_UB = 3L * 1024L;
constexpr int64_t INDEX_USE_UB = 3L * 1024L;
constexpr int64_t ALIGN_SIZE = 32;
constexpr int64_t BF_FP16_DSIZE = 2;
constexpr int64_t FP32_DSIZE = 4;

constexpr int64_t Y_GRAD_DIMS = 4;
constexpr int64_t X_DIMS_4 = 4;
constexpr int64_t X_DIMS_5 = 5;

constexpr int64_t FP32_DTYPE_KEY = 0;
constexpr int64_t FP16_DTYPE_KEY = 1;
constexpr int64_t BF16_DTYPE_KEY = 2;

constexpr int64_t NC_SMALL_KEY = 0;
constexpr int64_t NC_LARGE_KEY = 1;
constexpr int64_t DTYPE_KEY_WEIGHT = 10;

constexpr size_t DIM0 = 0;
constexpr size_t DIM1 = 1;
constexpr size_t DIM2 = 2;
constexpr size_t DIM3 = 3;
constexpr size_t DIM4 = 4;

class AdaptiveAvgPool3dGradTiling
{
public:
    explicit AdaptiveAvgPool3dGradTiling(gert::TilingContext* tilingContext) : context(tilingContext) {};
    ge::graphStatus Init();
    ge::graphStatus RunKernelTiling();
    void PrintTilingData();
    void CalTilingKey(uint32_t ubSize);
    bool GetDataTypeKey(ge::DataType dataType);

private:
    AdaptiveAvgPool3dGradTilingData TilingData;
    gert::TilingContext* context = nullptr;
    int64_t sysWorkspaceSize = 16L * 1024L * 1024L;
    int64_t useWorkspaceSize = 0;
    int64_t blockBytes = 32;
    int64_t dataAlign = 4;
    int64_t yGradDSize = 4;
    uint32_t coreNum = 48;
    int64_t dOut = 1;
    int64_t hOut = 1;
    int64_t wOut = 1;
    int64_t dIn = 1;
    int64_t hIn = 1;
    int64_t wIn = 1;
    int64_t ncNum = 1;
    int64_t ncAlign = 1;
    int64_t yGradNum = 0;
    int64_t taskCoreUsed = 0;

    int64_t yNumPerCalc = 0;
    int64_t taskNum = 0;
    int64_t taskNumPerCore = 0;
    int64_t taskNumLastCore = 0;
    int64_t indexCalcCoreUsed = 0;
    int64_t indexNumPerCore = 0;
    int64_t indexNumLastCore = 0;
    int64_t kW = 0;
    int64_t kH = 0;
    int64_t kD = 0;
    int64_t tilingKey = 0;

    int64_t perCalcSize = 0;
    int64_t ncSliceNum = 0;
    int64_t ncAlignSliceLength = 0;
    int64_t ncAlignSliceTail = 0;

    int64_t isAtomicAdd = 1;
    int64_t deterministicFlag = 0;
};

ge::graphStatus AdaptiveAvgPool3dGradTiling::Init()
{
    OP_LOGD(context, "AdaptiveAvgPool3dGrad tiling starts running");
    auto nodeName = context->GetNodeName();

    auto compileInfo = static_cast<const AdaptiveAvgPool3dGradCompileInfo*>(context->GetCompileInfo());
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);
    uint32_t totalUBSize = static_cast<uint32_t>(compileInfo->ubSizePlatForm);
    uint32_t ubSize = totalUBSize - static_cast<uint32_t>(RESERVED_UB) - static_cast<uint32_t>(INDEX_USE_UB);
    sysWorkspaceSize = compileInfo->sysWorkspaceSize;
    coreNum = compileInfo->coreNum;
    deterministicFlag = context->GetDeterministic() == 1 ? 1 : 0;
    if (deterministicFlag == 1) {
        coreNum = 1U;
    }
    auto const yGradShape = context->GetInputShape(0)->GetStorageShape();
    OP_CHECK_IF(
        yGradShape.GetDimNum() != Y_GRAD_DIMS,
        OP_LOGE(nodeName, "Check yGrad shape failed, the dims of yGrad not equal 4."), return ge::GRAPH_FAILED);

    auto const xShapeVal = context->GetInputShape(1)->GetStorageShape();
    OP_CHECK_IF(
        (xShapeVal.GetDimNum() != X_DIMS_4 && xShapeVal.GetDimNum() != X_DIMS_5),
        OP_LOGE(nodeName, "Check yGrad shape failed, the dims of yGrad not equal 4 or 5."), return ge::GRAPH_FAILED);

    auto const yGradDtype = context->GetInputDesc(0)->GetDataType();
    OP_CHECK_IF(
        GetDataTypeKey(yGradDtype) == false,
        OP_LOGE(nodeName, "The dtype of input must be in [float32, float16, bfloat16]."), return ge::GRAPH_FAILED);

    dOut = yGradShape.GetDim(DIM0);
    hOut = yGradShape.GetDim(DIM1);
    wOut = yGradShape.GetDim(DIM2);
    ncNum = yGradShape.GetDim(DIM3);
    if (xShapeVal.GetDimNum() == X_DIMS_4) {
        dIn = xShapeVal.GetDim(DIM1);
        hIn = xShapeVal.GetDim(DIM2);
        wIn = xShapeVal.GetDim(DIM3);
    } else if (xShapeVal.GetDimNum() == X_DIMS_5) {
        dIn = xShapeVal.GetDim(DIM2);
        hIn = xShapeVal.GetDim(DIM3);
        wIn = xShapeVal.GetDim(DIM4);
    }
    OP_CHECK_IF(
        dOut <= 0 || hOut <= 0 || wOut <= 0 || ncNum <= 0,
        OP_LOGE(nodeName, "Check yGradShape failed, the value of yGradShape should be greater than 0."),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        dIn <= 0 || hIn <= 0 || wIn <= 0,
        OP_LOGE(nodeName, "Check xShape failed, the value of xShape should be greater than 0."),
        return ge::GRAPH_FAILED);
    ncAlign = (ncNum + dataAlign - 1) / dataAlign * dataAlign;
    CalTilingKey(ubSize);
    PrintTilingData();
    OP_LOGD(nodeName, "AdaptiveAvgPool3dGrad tiling end running");
    return ge::GRAPH_SUCCESS;
}

void AdaptiveAvgPool3dGradTiling::CalTilingKey(uint32_t ubSize)
{
    int64_t ALIGN_NUM = ALIGN_SIZE / yGradDSize;
    if (dIn % dOut == 0 && hIn % hOut == 0 && wIn % wOut == 0) {
        isAtomicAdd = 0;
    }
    if (isAtomicAdd == 1 && yGradDSize == BF_FP16_DSIZE) {
        useWorkspaceSize = ncNum * dIn * hIn * wIn * FP32_DSIZE;
    }
    if (yGradDSize == BF_FP16_DSIZE) {
        perCalcSize = ncAlign * yGradDSize + ncAlign * FP32_DSIZE + ncAlign * yGradDSize + ncAlign * FP32_DSIZE;
    } else {
        perCalcSize = ncAlign * yGradDSize + ncAlign * yGradDSize;
    }

    if (perCalcSize < static_cast<int64_t>(ubSize)) {
        taskNum = dOut * hOut * wOut;
        yNumPerCalc = static_cast<int64_t>(ubSize) / perCalcSize;
        tilingKey = tilingKey * DTYPE_KEY_WEIGHT + NC_SMALL_KEY;
        ncAlignSliceLength = ncAlign;
        ncAlignSliceTail = ncAlign;
        ncSliceNum = 1;
    } else {
        tilingKey = tilingKey * DTYPE_KEY_WEIGHT + NC_LARGE_KEY;
        OP_CHECK_IF(ubSize == 0, OP_LOGE(context, "ubSize is 0."), return);
        ncSliceNum = (perCalcSize - 1 + static_cast<int64_t>(ubSize)) / static_cast<int64_t>(ubSize);
        ncAlignSliceLength = ncAlign / ncSliceNum / ALIGN_NUM * ALIGN_NUM;
        ncSliceNum = (ncNum - 1 + ncAlignSliceLength) / ncAlignSliceLength;
        ncAlignSliceTail = ncNum - ncAlignSliceLength * (ncSliceNum - 1);
        taskNum = dOut * hOut * wOut * ncSliceNum;
        yNumPerCalc = 1;
    }

    taskCoreUsed = taskNum > static_cast<int64_t>(coreNum) ? static_cast<int64_t>(coreNum) : taskNum;
    taskNumPerCore = (taskNum - 1 + taskCoreUsed) / taskCoreUsed;
    taskCoreUsed = (taskNum - 1 + taskNumPerCore) / taskNumPerCore;
    taskNumLastCore = taskNum - taskNumPerCore * (taskCoreUsed - 1);
    yNumPerCalc = yNumPerCalc > taskNumPerCore ? taskNumPerCore : yNumPerCalc;
    context->SetTilingKey(tilingKey);
}

ge::graphStatus AdaptiveAvgPool3dGradTiling::RunKernelTiling()
{
    context->SetBlockDim(coreNum);
    TilingData.set_ncNum(ncNum);
    TilingData.set_dIn(dIn);
    TilingData.set_hIn(hIn);
    TilingData.set_wIn(wIn);
    TilingData.set_dOut(dOut);
    TilingData.set_hOut(hOut);
    TilingData.set_wOut(wOut);
    TilingData.set_taskCoreUsed(taskCoreUsed);
    TilingData.set_taskNumPerCore(taskNumPerCore);
    TilingData.set_taskNumLastCore(taskNumLastCore);
    TilingData.set_yNumPerCalc(yNumPerCalc);
    TilingData.set_ncSliceNum(ncSliceNum);
    TilingData.set_ncAlignSliceLength(ncAlignSliceLength);
    TilingData.set_ncAlignSliceTail(ncAlignSliceTail);
    TilingData.set_isAtomicAdd(isAtomicAdd);
    TilingData.set_deterministicFlag(deterministicFlag);
    size_t* workspaces = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, workspaces);
    workspaces[0] = static_cast<size_t>(useWorkspaceSize) + static_cast<size_t>(sysWorkspaceSize) +
                    static_cast<size_t>(sysWorkspaceSize);
    TilingData.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(TilingData.GetDataSize());
    return ge::GRAPH_SUCCESS;
}

bool AdaptiveAvgPool3dGradTiling::GetDataTypeKey(ge::DataType dataType)
{
    switch (dataType) {
        case ge::DT_FLOAT16:
            yGradDSize = BF_FP16_DSIZE;
            tilingKey = FP16_DTYPE_KEY;
            dataAlign = blockBytes / yGradDSize;
            break;
        case ge::DT_BF16:
            yGradDSize = BF_FP16_DSIZE;
            tilingKey = BF16_DTYPE_KEY;
            dataAlign = blockBytes / yGradDSize;
            break;
        case ge::DT_FLOAT:
            yGradDSize = FP32_DSIZE;
            tilingKey = FP32_DTYPE_KEY;
            dataAlign = blockBytes / yGradDSize;
            break;
        default:
            return false;
    }
    return true;
}

void AdaptiveAvgPool3dGradTiling::PrintTilingData()
{
    OP_LOGD(context, "Start printing");
    OP_LOGD(context, "ncNum is %lu.", ncNum);
    OP_LOGD(context, "dIn is %lu.", dIn);
    OP_LOGD(context, "hIn is %lu.", hIn);
    OP_LOGD(context, "wIn is %lu.", wIn);
    OP_LOGD(context, "dOut is %lu.", dOut);
    OP_LOGD(context, "hOut is %lu.", hOut);
    OP_LOGD(context, "wOut is %lu.", wOut);
    OP_LOGD(context, "taskCoreUsed is %lu.", taskCoreUsed);
    OP_LOGD(context, "taskNumPerCore is %lu.", taskNumPerCore);
    OP_LOGD(context, "taskNumLastCore is %lu.", taskNumLastCore);
    OP_LOGD(context, "yNumPerCalc is %lu.", yNumPerCalc);
    OP_LOGD(context, "ncSliceNum is %lu.", ncSliceNum);
    OP_LOGD(context, "ncAlignSliceLength is %lu.", ncAlignSliceLength);
    OP_LOGD(context, "ncAlignSliceTail is %lu.", ncAlignSliceTail);
    OP_LOGD(context, "isAtomicAdd is %lu.", isAtomicAdd);
    OP_LOGD(context, "deterministicFlag is %lu.", deterministicFlag);
    OP_LOGD(context, "End printing");
}

ge::graphStatus TilingPrepare4AdaptiveAvgPool3dGrad(gert::TilingParseContext* context)
{
    auto nodeName = context->GetNodeName();
    OP_LOGD(nodeName, "TilingPrepare4AdaptiveAvgPool3dGrad start.");
    auto compileInfo = context->GetCompiledInfo<AdaptiveAvgPool3dGradCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);
    auto platformInfo = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    compileInfo->coreNum = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF((compileInfo->coreNum <= 0), OP_LOGE(nodeName, "Failed to get core num."), return ge::GRAPH_FAILED);
    compileInfo->sysWorkspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();
    uint64_t ubSizePlatForm;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatForm);
    compileInfo->ubSizePlatForm = static_cast<int64_t>(ubSizePlatForm);
    OP_CHECK_IF(
        (compileInfo->ubSizePlatForm <= 0), OP_LOGE(nodeName, "Failed to get ub size."), return ge::GRAPH_FAILED);
    OP_LOGD(nodeName, "TilingPrepare4AdaptiveAvgPool3dGrad end.");

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingFunc4AdaptiveAvgPool3dGrad(gert::TilingContext* context)
{
    AdaptiveAvgPool3dGradTiling tilingObject(context);
    if (tilingObject.Init() != ge::GRAPH_SUCCESS) {
        OP_LOGE(context, "Init failed!");
        return ge::GRAPH_FAILED;
    }
    return tilingObject.RunKernelTiling();
}

IMPL_OP_OPTILING(AdaptiveAvgPool3dGrad)
    .Tiling(TilingFunc4AdaptiveAvgPool3dGrad)
    .TilingParse<AdaptiveAvgPool3dGradCompileInfo>(TilingPrepare4AdaptiveAvgPool3dGrad);
} // namespace optiling

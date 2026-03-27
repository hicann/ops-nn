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
 * \file adaptive_avg_pool3d_para_pool_tiling.cpp
 * \brief
 */

#include <cstdint>
#include "adaptive_avg_pool3d_para_tiling.h"

constexpr uint64_t KERNEL_SIZE_LIMIT = 128;
constexpr uint64_t RESERVE_UB_SIZE = 0;
constexpr uint64_t MAX_UB_BUFFER_NUM = 2;
constexpr uint64_t INT32_MAX_VALUE = 2147483647UL;
constexpr uint64_t DOUBLE = 2;
constexpr uint64_t TRANS_ADDR_LEN = 16;

namespace optiling {

bool AdaptiveAvgPool3dParaPoolTiling::IsCapable()
{
    OP_TILING_CHECK(
        GetAndCheckDataFormat() != ge::GRAPH_SUCCESS,
        VECTOR_INNER_ERR_REPORT_TILIING(context_, "GetAndCheckDataFormat fail."), return ge::GRAPH_FAILED);
    if (input_.dataFormat != ge::Format::FORMAT_NCDHW) {
        OP_LOGD(context_->GetNodeName(), "AdaptiveAvgPool3dParaPoolTiling only support attr data_format NCDHW");
        return false;
    }
    if (ge::GetSizeByDataType(input_.xDtype) == 0) {
        OP_LOGE(context_->GetNodeName(), "Get xDtype size is 0, not support");
        return false;
    }
    avgComptuteInfo_.vfLen = Ops::Base::GetVRegSize(context_) / ge::GetSizeByDataType(input_.xDtype);
    avgComptuteInfo_.alignNum = Ops::Base::GetUbBlockSize(context_) / ge::GetSizeByDataType(input_.xDtype);
    avgComptuteInfo_.availableUbSize = input_.ubSize - RESERVE_UB_SIZE;
    avgComptuteInfo_.ncFactor = avgComptuteInfo_.vfLen;
    avgComptuteInfo_.woFactor = 1;
    avgComptuteInfo_.hoFactor = 1;
    avgComptuteInfo_.doFactor = 1;

    avgComptuteInfo_.kernelDMax = CalKernelSizeOneDimMax(input_.dIn, input_.dOut);
    avgComptuteInfo_.kernelHMax = CalKernelSizeOneDimMax(input_.hIn, input_.hOut);
    avgComptuteInfo_.kernelWMax = CalKernelSizeOneDimMax(input_.wIn, input_.wOut);

    bool isKernelSizeMeet =
        (avgComptuteInfo_.kernelDMax * avgComptuteInfo_.kernelHMax * avgComptuteInfo_.kernelWMax < KERNEL_SIZE_LIMIT);
    bool isNcLenEnough = input_.nIn * input_.cIn >= (avgComptuteInfo_.vfLen / DOUBLE);
    /* 计算只处理一个窗口占用的UB */
    auto occupyUbSize = CalOccupySize();
    bool isUbSizeEnough = (occupyUbSize <= avgComptuteInfo_.availableUbSize);
    bool isCapable = isKernelSizeMeet && isNcLenEnough && isUbSizeEnough;
    OP_LOGD(
        context_->GetNodeName(), "AdaptiveAvgPool3dParaPoolTiling IsCapable check: %s", isCapable ? "true" : "false");
    return isCapable;
}

void AdaptiveAvgPool3dParaPoolTiling::CalMaxUbSplitSize()
{
    auto doNum = avgComptuteInfo_.doFactor;
    auto hoNum = avgComptuteInfo_.hoFactor;
    auto woNum = avgComptuteInfo_.woFactor;
    auto woNumAlign = Ops::Base::CeilAlign(woNum, avgComptuteInfo_.alignNum);

    auto wiDataLen = avgComptuteInfo_.woFactor * avgComptuteInfo_.kernelWMax;
    auto hiDataLen = avgComptuteInfo_.hoFactor * avgComptuteInfo_.kernelHMax;
    auto diDataLen = avgComptuteInfo_.doFactor * avgComptuteInfo_.kernelDMax;
    auto wiDataLenAlign = Ops::Base::CeilAlign(wiDataLen, avgComptuteInfo_.alignNum);

    auto maxD = std::max(doNum, diDataLen);
    auto maxH = std::max(hoNum, hiDataLen);
    auto maxW = std::max(woNumAlign, wiDataLenAlign);
    /* 转置接口需要按16对齐 */
    auto maxDhw = Ops::Base::CeilAlign(maxD * maxH * maxW, TRANS_ADDR_LEN);
    avgComptuteInfo_.maxInputSize = maxDhw * avgComptuteInfo_.ncFactor;
    avgComptuteInfo_.maxDimOut = std::max({doNum, hoNum, woNum});
}

void AdaptiveAvgPool3dParaPoolTiling::CalUbBlockFactor()
{
    avgComptuteInfo_.doOuter = Ops::Base::CeilDiv(input_.dOut, avgComptuteInfo_.doFactor);
    avgComptuteInfo_.doTail = input_.dOut - (avgComptuteInfo_.doOuter - 1) * avgComptuteInfo_.doFactor;
    avgComptuteInfo_.hoOuter = Ops::Base::CeilDiv(input_.hOut, avgComptuteInfo_.hoFactor);
    avgComptuteInfo_.hoTail = input_.hOut - (avgComptuteInfo_.hoOuter - 1) * avgComptuteInfo_.hoFactor;
    avgComptuteInfo_.woOuter = Ops::Base::CeilDiv(input_.wOut, avgComptuteInfo_.woFactor);
    avgComptuteInfo_.woTail = input_.wOut - (avgComptuteInfo_.woOuter - 1) * avgComptuteInfo_.woFactor;
    avgComptuteInfo_.ncOuter = Ops::Base::CeilDiv(input_.nIn * input_.cIn, avgComptuteInfo_.ncFactor);
    avgComptuteInfo_.ncTail = input_.nIn * input_.cIn - (avgComptuteInfo_.ncOuter - 1) * avgComptuteInfo_.ncFactor;

    /* 总共的UB块 */
    avgComptuteInfo_.totalOuter = avgComptuteInfo_.ncOuter * avgComptuteInfo_.woOuter * avgComptuteInfo_.hoOuter * avgComptuteInfo_.doOuter;
    avgComptuteInfo_.blockFactor = Ops::Base::CeilDiv(avgComptuteInfo_.totalOuter, input_.coreNum);
    avgComptuteInfo_.useCoreNum = Ops::Base::CeilDiv(avgComptuteInfo_.totalOuter, avgComptuteInfo_.blockFactor);
    avgComptuteInfo_.blockTail = avgComptuteInfo_.totalOuter - (avgComptuteInfo_.useCoreNum - 1) * avgComptuteInfo_.blockFactor;
}

/*
* inputQue:    vl * diDataLen * hiDataLen * wiDataLenAlign
* avgQue:      diDataLen * hiDataLen * wiDataLenAlign * vl,
               hoNum * woNumAlign * diDataLen * vl,
               vl * doNum * hoNum * woNumAlign
* avgTransQue:   woNumAlign * diDataLen * hiDataLen * vl
                 doNum * hoNum * woNumAlign * vl
* startIdxBuf:   大小为maxDimOut
* kerSizeBuf:    各轴对应的factor大小
*/
uint64_t AdaptiveAvgPool3dParaPoolTiling::CalOccupySize()
{
    CalMaxUbSplitSize();
    uint64_t dataBlock = Ops::Base::GetUbBlockSize(context_);
    auto occupySize = avgComptuteInfo_.maxInputSize * ge::GetSizeByDataType(input_.xDtype) +
                      avgComptuteInfo_.maxInputSize * ge::GetSizeByDataType(ge::DT_FLOAT) * MAX_UB_BUFFER_NUM +
                      Ops::Base::CeilAlign(avgComptuteInfo_.maxDimOut * ge::GetSizeByDataType(ge::DT_INT32), dataBlock) +
                      Ops::Base::CeilAlign(avgComptuteInfo_.doFactor * ge::GetSizeByDataType(ge::DT_INT32), dataBlock) +
                      Ops::Base::CeilAlign(avgComptuteInfo_.hoFactor * ge::GetSizeByDataType(ge::DT_INT32), dataBlock) +
                      Ops::Base::CeilAlign(avgComptuteInfo_.woFactor * ge::GetSizeByDataType(ge::DT_INT32), dataBlock);
    return occupySize;
}

void AdaptiveAvgPool3dParaPoolTiling::BinarySearch(uint64_t& initFactor)
{
    if (initFactor <= 1) {
        return;
    }
    uint64_t left = 1;
    uint64_t bestSplit = 1;
    uint64_t right = initFactor;

    while (left <= right) {
        uint64_t mid = left + (right - left) / DOUBLE;
        initFactor = mid;
        if (CalOccupySize() < avgComptuteInfo_.availableUbSize && initFactor > 1) {
            bestSplit = mid;
            left = mid + 1;
        } else {
            right = mid - 1;
        }
    }
    initFactor = bestSplit;
}

void AdaptiveAvgPool3dParaPoolTiling::SearchOuterSingle(uint64_t& initFactor)
{
    if (initFactor <= 1) {
        return;
    }
    do {
        uint64_t lastBlockFactor = avgComptuteInfo_.blockFactor;
        initFactor -= 1;
        CalUbBlockFactor();
        if (avgComptuteInfo_.blockFactor > lastBlockFactor) {
            initFactor += 1;
            break;
        }
    } while (avgComptuteInfo_.useCoreNum < input_.coreNum && initFactor > 1);
}

ge::graphStatus AdaptiveAvgPool3dParaPoolTiling::SearchUbFactor()
{
    OP_LOGD(context_->GetNodeName(), "AdaptiveAvgPool3dParaPoolTiling search ubfactor start.");
    if (CalOccupySize() < avgComptuteInfo_.availableUbSize) {
        return ge::GRAPH_SUCCESS;
    }
    BinarySearch(avgComptuteInfo_.doFactor);
    if (CalOccupySize() < avgComptuteInfo_.availableUbSize) {
        return ge::GRAPH_SUCCESS;
    }
    BinarySearch(avgComptuteInfo_.hoFactor);
    if (CalOccupySize() < avgComptuteInfo_.availableUbSize) {
        return ge::GRAPH_SUCCESS;
    }
    BinarySearch(avgComptuteInfo_.woFactor);

    OP_LOGD(
        context_->GetNodeName(), "doFactor = %lu, hoFactor = %lu, woFactor = %lu", avgComptuteInfo_.doFactor,
        avgComptuteInfo_.hoFactor, avgComptuteInfo_.woFactor);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AdaptiveAvgPool3dParaPoolTiling::SearchOuter()
{
    OP_LOGD(context_->GetNodeName(), "AdaptiveAvgPool3dParaPoolTiling search outer start.");
    if (avgComptuteInfo_.useCoreNum == input_.coreNum) {
        return ge::GRAPH_SUCCESS;
    }
    SearchOuterSingle(avgComptuteInfo_.doFactor);
    if (avgComptuteInfo_.useCoreNum == input_.coreNum) {
        return ge::GRAPH_SUCCESS;
    }
    SearchOuterSingle(avgComptuteInfo_.hoFactor);
    if (avgComptuteInfo_.useCoreNum == input_.coreNum) {
        return ge::GRAPH_SUCCESS;
    }
    SearchOuterSingle(avgComptuteInfo_.woFactor);

    OP_LOGD(
        context_->GetNodeName(), "doFactor = %lu, hoFactor = %lu, woFactor = %lu", avgComptuteInfo_.doFactor,
        avgComptuteInfo_.hoFactor, avgComptuteInfo_.woFactor);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AdaptiveAvgPool3dParaPoolTiling::InitUbFactor()
{
    OP_LOGD(context_->GetNodeName(), "AdaptiveAvgPool3dParaPoolTiling init ubfactor start.");
    auto kernelD = avgComptuteInfo_.kernelDMax;
    auto kernelH = avgComptuteInfo_.kernelHMax;
    auto kernelW = avgComptuteInfo_.kernelWMax;
    OP_CHECK_IF(
        (kernelW <= 0 || kernelH <= 0 || kernelD <= 0),
        OP_LOGE(context_->GetNodeName(), "Kernel size <= 0, not support"), return ge::GRAPH_FAILED);

    avgComptuteInfo_.ncFactor = avgComptuteInfo_.vfLen;
    avgComptuteInfo_.woFactor = input_.wOut;
    avgComptuteInfo_.hoFactor = input_.hOut;
    avgComptuteInfo_.doFactor = input_.dOut;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AdaptiveAvgPool3dParaPoolTiling::DoOpTiling()
{
    OP_LOGD(context_->GetNodeName(), "AdaptiveAvgPool3dParaPoolTiling DoOpTiling start.");
    OP_CHECK_IF(
        InitUbFactor() != ge::GRAPH_SUCCESS,
        OP_LOGE(context_->GetNodeName(), "AdaptiveAvgPool3dParaPoolTiling init ubfactor failed"),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        SearchUbFactor() != ge::GRAPH_SUCCESS,
        OP_LOGE(context_->GetNodeName(), "AdaptiveAvgPool3dParaPoolTiling search ubfactor failed"),
        return ge::GRAPH_FAILED);

    CalUbBlockFactor();
    OP_CHECK_IF(
        SearchOuter() != ge::GRAPH_SUCCESS,
        OP_LOGE(context_->GetNodeName(), "AdaptiveAvgPool3dParaPoolTiling search outer failed"),
        return ge::GRAPH_FAILED);

    CalUbBlockFactor();
    CalMaxUbSplitSize();

    SetTilingData();
    return ge::GRAPH_SUCCESS;
}

void AdaptiveAvgPool3dParaPoolTiling::SetTilingData()
{
    AdaptivePool3DTiling::AdaptivePool3dParaKernelTilingData* tilingData =
        context_->GetTilingData<AdaptivePool3dParaKernelTilingData>();
    OP_CHECK_IF(tilingData == nullptr, OP_LOGE(context_->GetNodeName(), "tilingData is null"), return);

    tilingData->useCoreNum = avgComptuteInfo_.useCoreNum;
    tilingData->dIn = input_.dIn;
    tilingData->hIn = input_.hIn;
    tilingData->wIn = input_.wIn;
    tilingData->dOut = input_.dOut;
    tilingData->hOut = input_.hOut;
    tilingData->wOut = input_.wOut;
    tilingData->blockFactor = avgComptuteInfo_.blockFactor;
    tilingData->blockTail = avgComptuteInfo_.blockTail;
    tilingData->ncFactor = avgComptuteInfo_.ncFactor;
    tilingData->doFactor = avgComptuteInfo_.doFactor;
    tilingData->hoFactor = avgComptuteInfo_.hoFactor;
    tilingData->woFactor = avgComptuteInfo_.woFactor;
    tilingData->ncOuter = avgComptuteInfo_.ncOuter;
    tilingData->doOuter = avgComptuteInfo_.doOuter;
    tilingData->hoOuter = avgComptuteInfo_.hoOuter;
    tilingData->woOuter = avgComptuteInfo_.woOuter;
    tilingData->ncTail = avgComptuteInfo_.ncTail;
    tilingData->doTail = avgComptuteInfo_.doTail;
    tilingData->hoTail = avgComptuteInfo_.hoTail;
    tilingData->woTail = avgComptuteInfo_.woTail;
    tilingData->maxInputSize = avgComptuteInfo_.maxInputSize;
    tilingData->maxDimOut = avgComptuteInfo_.maxDimOut;
}

void AdaptiveAvgPool3dParaPoolTiling::PrintTilingData() const
{
    std::ostringstream info;
    info << "nc: " << input_.nIn * input_.cIn;
    info << ", useCoreNum: " << avgComptuteInfo_.useCoreNum;
    info << ", dInDim: " << input_.dIn;
    info << ", hInDim: " << input_.hIn;
    info << ", wInDim: " << input_.wIn;
    info << ", dOutDim: " << input_.dOut;
    info << ", hOutDim: " << input_.hOut;
    info << ", wOutDim: " << input_.wOut;
    info << ", blockFactor: " << avgComptuteInfo_.blockFactor;
    info << ", blockTail: " << avgComptuteInfo_.blockTail;
    info << ", ncFactor: " << avgComptuteInfo_.ncFactor;
    info << ", doFactor: " << avgComptuteInfo_.doFactor;
    info << ", hoFactor: " << avgComptuteInfo_.hoFactor;
    info << ", woFactor: " << avgComptuteInfo_.woFactor;
    info << ", doOuter: " << avgComptuteInfo_.doOuter;
    info << ", hoOuter: " << avgComptuteInfo_.hoOuter;
    info << ", woOuter: " << avgComptuteInfo_.woOuter;
    info << ", ncOuter: " << avgComptuteInfo_.ncOuter;
    info << ", ncTail: " << avgComptuteInfo_.ncTail;
    info << ", doTail: " << avgComptuteInfo_.doTail;
    info << ", hoTail: " << avgComptuteInfo_.hoTail;
    info << ", woTail: " << avgComptuteInfo_.woTail;
    info << ", maxInputSize: " << avgComptuteInfo_.maxInputSize;
    info << std::endl;

    OP_LOGI(context_->GetNodeName(), "%s", info.str().c_str());
}

uint64_t AdaptiveAvgPool3dParaPoolTiling::GetTilingKey() const
{
    OP_LOGD(context_->GetNodeName(), "AdaptiveAvgPool3dParaPoolTiling GetTilingKey start.");
    int64_t maxIdxValue = std::max({input_.dIn * input_.dOut, input_.hIn * input_.hOut, input_.wIn * input_.wOut});
    uint64_t idxTypeMode = static_cast<uint64_t>(maxIdxValue) < INT32_MAX_VALUE ? TPL_INT32_UINT32 : TPL_INT64_UINT64;

    return GET_TPL_TILING_KEY(TPL_MODE_0, idxTypeMode, TPL_MULTI_MODE_0, TPL_DATA_FORMAT_MODE_0);
}

ge::graphStatus AdaptiveAvgPool3dParaPoolTiling::PostTiling()
{
    OP_LOGD(context_->GetNodeName(), "AdaptiveAvgPool3dParaPoolTiling PostTiling start.");
    context_->SetBlockDim(avgComptuteInfo_.useCoreNum);
    return ge::GRAPH_SUCCESS;
}

REGISTER_OPS_TILING_TEMPLATE(AdaptiveAvgPool3d, AdaptiveAvgPool3dParaPoolTiling, 0);
} // namespace optiling

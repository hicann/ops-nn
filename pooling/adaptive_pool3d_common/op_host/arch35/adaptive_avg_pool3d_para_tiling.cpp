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

constexpr uint64_t KERNEL_SIZE_LIMIT = 350;
constexpr uint64_t RESERVE_UB_SIZE = 0;
constexpr uint64_t MAX_UB_BUFFER_NUM = 2;
constexpr uint64_t INT32_MAX_VALUE = 2147483647UL;
constexpr uint64_t DOUBLE = 2;
constexpr uint64_t TRANS_ADDR_LEN = 16;
constexpr uint64_t UB_UTIL_RATE = 224 * 1024;

namespace optiling {

bool AdaptiveAvgPool3dParaPoolTiling::IsCapable()
{
    OP_TILING_CHECK(GetAndCheckDataFormat() != ge::GRAPH_SUCCESS,
                    VECTOR_INNER_ERR_REPORT_TILIING(context_, "GetAndCheckDataFormat fail."), return ge::GRAPH_FAILED);
    if (input_.dataFormat != ge::Format::FORMAT_NCDHW) {
        OP_LOGD(context_->GetNodeName(), "AdaptiveAvgPool3dParaPoolTiling only support attr data_format NCDHW");
        return false;
    }
    if (ge::GetSizeByDataType(input_.xDtype) == 0) {
        OP_LOGE_FOR_INVALID_DTYPE("AdaptiveAvgPool3d", "x", "unknown/unsupported", "[DT_FLOAT, DT_FLOAT16, DT_BF16]");
        return false;
    }
    avgComputeInfo_.vfLen = Ops::Base::GetVRegSize(context_) / ge::GetSizeByDataType(input_.xDtype);
    avgComputeInfo_.alignNum = Ops::Base::GetUbBlockSize(context_) / ge::GetSizeByDataType(input_.xDtype);
    avgComputeInfo_.availableUbSize = input_.ubSize - RESERVE_UB_SIZE;
    avgComputeInfo_.ncFactor = avgComputeInfo_.vfLen;
    avgComputeInfo_.woFactor = 1;
    avgComputeInfo_.hoFactor = 1;
    avgComputeInfo_.doFactor = 1;

    avgComputeInfo_.kernelDMax = CalKernelSizeOneDimMax(input_.dIn, input_.dOut);
    avgComputeInfo_.kernelHMax = CalKernelSizeOneDimMax(input_.hIn, input_.hOut);
    avgComputeInfo_.kernelWMax = CalKernelSizeOneDimMax(input_.wIn, input_.wOut);

    bool isKernelSizeMeet = (avgComputeInfo_.kernelDMax * avgComputeInfo_.kernelHMax * avgComputeInfo_.kernelWMax <
                             KERNEL_SIZE_LIMIT);
    bool isNcLenEnough = input_.nIn * input_.cIn >= (avgComputeInfo_.vfLen / DOUBLE);
    /* 计算只处理一个窗口占用的UB */
    auto occupyUbSize = CalOccupySize();
    bool isUbSizeEnough = (occupyUbSize <= avgComputeInfo_.availableUbSize);

    /* 数据量足够大时，UB使用率不能低于阈值 */
    avgComputeInfo_.woFactor = input_.wOut;
    avgComputeInfo_.hoFactor = input_.hOut;
    avgComputeInfo_.doFactor = input_.dOut;
    auto outSize = CalOccupySize();
    bool ubUseEnough = outSize >= UB_UTIL_RATE;
    DoTilingForUbFactor();
    occupyUbSize = CalOccupySize();
    ubUseEnough = ubUseEnough ? occupyUbSize >= UB_UTIL_RATE : true;

    auto wiDataLen = avgComputeInfo_.woFactor * avgComputeInfo_.kernelWMax;
    ubUseEnough = ubUseEnough && (wiDataLen % avgComputeInfo_.alignNum) > (avgComputeInfo_.alignNum / DOUBLE);
    uint64_t kprod = avgComputeInfo_.kernelDMax * avgComputeInfo_.kernelHMax * avgComputeInfo_.kernelWMax;
    bool isWOutDegenerateVectorized = (input_.wOut == 1) && kprod >= avgComputeInfo_.vfLen &&
                                      input_.wIn >= avgComputeInfo_.alignNum;
    ubUseEnough = ubUseEnough || isWOutDegenerateVectorized;

    bool isCapable = isKernelSizeMeet && isNcLenEnough && isUbSizeEnough && ubUseEnough;
    OP_LOGD(context_->GetNodeName(), "AdaptiveAvgPool3dParaPoolTiling IsCapable check: %s",
            isCapable ? "true" : "false");
    return isCapable;
}

void AdaptiveAvgPool3dParaPoolTiling::CalMaxUbSplitSize()
{
    auto doNum = avgComputeInfo_.doFactor;
    auto hoNum = avgComputeInfo_.hoFactor;
    auto woNum = avgComputeInfo_.woFactor;
    auto woNumAlign = Ops::Base::CeilAlign(woNum, avgComputeInfo_.alignNum);

    auto wiDataLen = avgComputeInfo_.woFactor * avgComputeInfo_.kernelWMax;
    auto hiDataLen = avgComputeInfo_.hoFactor * avgComputeInfo_.kernelHMax;
    auto diDataLen = avgComputeInfo_.doFactor * avgComputeInfo_.kernelDMax;
    auto wiDataLenAlign = Ops::Base::CeilAlign(wiDataLen, avgComputeInfo_.alignNum);

    auto maxD = std::max(doNum, diDataLen);
    auto maxH = std::max(hoNum, hiDataLen);
    auto maxW = std::max(woNumAlign, wiDataLenAlign);
    /* 转置接口需要按16对齐 */
    auto maxDhw = Ops::Base::CeilAlign(maxD * maxH * maxW, TRANS_ADDR_LEN);
    avgComputeInfo_.maxInputSize = maxDhw * avgComputeInfo_.ncFactor;
    avgComputeInfo_.maxDimOut = std::max({doNum, hoNum, woNum});
}

void AdaptiveAvgPool3dParaPoolTiling::CalUbBlockFactor()
{
    avgComputeInfo_.doOuter = Ops::Base::CeilDiv(input_.dOut, avgComputeInfo_.doFactor);
    avgComputeInfo_.doTail = input_.dOut - (avgComputeInfo_.doOuter - 1) * avgComputeInfo_.doFactor;
    avgComputeInfo_.hoOuter = Ops::Base::CeilDiv(input_.hOut, avgComputeInfo_.hoFactor);
    avgComputeInfo_.hoTail = input_.hOut - (avgComputeInfo_.hoOuter - 1) * avgComputeInfo_.hoFactor;
    avgComputeInfo_.woOuter = Ops::Base::CeilDiv(input_.wOut, avgComputeInfo_.woFactor);
    avgComputeInfo_.woTail = input_.wOut - (avgComputeInfo_.woOuter - 1) * avgComputeInfo_.woFactor;
    avgComputeInfo_.ncOuter = Ops::Base::CeilDiv(input_.nIn * input_.cIn, avgComputeInfo_.ncFactor);
    avgComputeInfo_.ncTail = input_.nIn * input_.cIn - (avgComputeInfo_.ncOuter - 1) * avgComputeInfo_.ncFactor;

    /* 总共的UB块 */
    avgComputeInfo_.totalOuter = avgComputeInfo_.ncOuter * avgComputeInfo_.woOuter * avgComputeInfo_.hoOuter *
                                 avgComputeInfo_.doOuter;
    avgComputeInfo_.blockFactor = Ops::Base::CeilDiv(avgComputeInfo_.totalOuter, input_.coreNum);
    avgComputeInfo_.useCoreNum = Ops::Base::CeilDiv(avgComputeInfo_.totalOuter, avgComputeInfo_.blockFactor);
    avgComputeInfo_.blockTail = avgComputeInfo_.totalOuter -
                                (avgComputeInfo_.useCoreNum - 1) * avgComputeInfo_.blockFactor;
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
    auto occupySize = avgComputeInfo_.maxInputSize * ge::GetSizeByDataType(input_.xDtype) +
                      avgComputeInfo_.maxInputSize * ge::GetSizeByDataType(ge::DT_FLOAT) * MAX_UB_BUFFER_NUM +
                      Ops::Base::CeilAlign(avgComputeInfo_.maxDimOut * ge::GetSizeByDataType(ge::DT_INT32), dataBlock) +
                      Ops::Base::CeilAlign(avgComputeInfo_.doFactor * ge::GetSizeByDataType(ge::DT_INT32), dataBlock) +
                      Ops::Base::CeilAlign(avgComputeInfo_.hoFactor * ge::GetSizeByDataType(ge::DT_INT32), dataBlock) +
                      Ops::Base::CeilAlign(avgComputeInfo_.woFactor * ge::GetSizeByDataType(ge::DT_INT32), dataBlock);
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
        if (CalOccupySize() < avgComputeInfo_.availableUbSize && initFactor > 1) {
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
        uint64_t lastBlockFactor = avgComputeInfo_.blockFactor;
        initFactor -= 1;
        CalUbBlockFactor();
        if (avgComputeInfo_.blockFactor > lastBlockFactor) {
            initFactor += 1;
            break;
        }
    } while (avgComputeInfo_.useCoreNum < input_.coreNum && initFactor > 1);
}

ge::graphStatus AdaptiveAvgPool3dParaPoolTiling::SearchUbFactor()
{
    OP_LOGD(context_->GetNodeName(), "AdaptiveAvgPool3dParaPoolTiling search ubfactor start.");
    if (CalOccupySize() < avgComputeInfo_.availableUbSize) {
        return ge::GRAPH_SUCCESS;
    }
    BinarySearch(avgComputeInfo_.doFactor);
    if (CalOccupySize() < avgComputeInfo_.availableUbSize) {
        return ge::GRAPH_SUCCESS;
    }
    BinarySearch(avgComputeInfo_.hoFactor);
    if (CalOccupySize() < avgComputeInfo_.availableUbSize) {
        return ge::GRAPH_SUCCESS;
    }
    BinarySearch(avgComputeInfo_.woFactor);

    OP_LOGD(context_->GetNodeName(), "doFactor = %lu, hoFactor = %lu, woFactor = %lu", avgComputeInfo_.doFactor,
            avgComputeInfo_.hoFactor, avgComputeInfo_.woFactor);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AdaptiveAvgPool3dParaPoolTiling::SearchOuter()
{
    OP_LOGD(context_->GetNodeName(), "AdaptiveAvgPool3dParaPoolTiling search outer start.");
    if (avgComputeInfo_.useCoreNum == input_.coreNum) {
        return ge::GRAPH_SUCCESS;
    }
    SearchOuterSingle(avgComputeInfo_.doFactor);
    if (avgComputeInfo_.useCoreNum == input_.coreNum) {
        return ge::GRAPH_SUCCESS;
    }
    SearchOuterSingle(avgComputeInfo_.hoFactor);
    if (avgComputeInfo_.useCoreNum == input_.coreNum) {
        return ge::GRAPH_SUCCESS;
    }
    SearchOuterSingle(avgComputeInfo_.woFactor);

    OP_LOGD(context_->GetNodeName(), "doFactor = %lu, hoFactor = %lu, woFactor = %lu", avgComputeInfo_.doFactor,
            avgComputeInfo_.hoFactor, avgComputeInfo_.woFactor);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AdaptiveAvgPool3dParaPoolTiling::InitUbFactor()
{
    OP_LOGD(context_->GetNodeName(), "AdaptiveAvgPool3dParaPoolTiling init ubfactor start.");
    auto kernelD = avgComputeInfo_.kernelDMax;
    auto kernelH = avgComputeInfo_.kernelHMax;
    auto kernelW = avgComputeInfo_.kernelWMax;
    if (kernelW <= 0 || kernelH <= 0 || kernelD <= 0) {
        OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
            "AdaptiveAvgPool3d", "kernelD, kernelH, kernelW",
            (std::to_string(kernelD) + ", " + std::to_string(kernelH) + ", " + std::to_string(kernelW)).c_str(),
            "kernel size must be > 0");
        return ge::GRAPH_FAILED;
    }

    avgComputeInfo_.ncFactor = avgComputeInfo_.vfLen;
    avgComputeInfo_.woFactor = input_.wOut;
    avgComputeInfo_.hoFactor = input_.hOut;
    avgComputeInfo_.doFactor = input_.dOut;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AdaptiveAvgPool3dParaPoolTiling::DoTilingForUbFactor()
{
    OP_CHECK_IF(InitUbFactor() != ge::GRAPH_SUCCESS,
                OP_LOGE(context_->GetNodeName(), "AdaptiveAvgPool3dParaPoolTiling init ubfactor failed"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(SearchUbFactor() != ge::GRAPH_SUCCESS,
                OP_LOGE(context_->GetNodeName(), "AdaptiveAvgPool3dParaPoolTiling search ubfactor failed"),
                return ge::GRAPH_FAILED);

    CalUbBlockFactor();
    OP_CHECK_IF(SearchOuter() != ge::GRAPH_SUCCESS,
                OP_LOGE(context_->GetNodeName(), "AdaptiveAvgPool3dParaPoolTiling search outer failed"),
                return ge::GRAPH_FAILED);

    CalUbBlockFactor();
    CalMaxUbSplitSize();
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AdaptiveAvgPool3dParaPoolTiling::DoOpTiling()
{
    OP_LOGD(context_->GetNodeName(), "AdaptiveAvgPool3dParaPoolTiling DoOpTiling start.");
    SetTilingData();
    PrintTilingData();
    return ge::GRAPH_SUCCESS;
}

void AdaptiveAvgPool3dParaPoolTiling::SetTilingData()
{
    AdaptivePool3DTiling::AdaptivePool3dParaKernelTilingData*
        tilingData = context_->GetTilingData<AdaptivePool3dParaKernelTilingData>();
    OP_CHECK_IF(tilingData == nullptr, OP_LOGE(context_->GetNodeName(), "tilingData is null"), return);

    tilingData->useCoreNum = avgComputeInfo_.useCoreNum;
    tilingData->dIn = input_.dIn;
    tilingData->hIn = input_.hIn;
    tilingData->wIn = input_.wIn;
    tilingData->dOut = input_.dOut;
    tilingData->hOut = input_.hOut;
    tilingData->wOut = input_.wOut;
    tilingData->blockFactor = avgComputeInfo_.blockFactor;
    tilingData->blockTail = avgComputeInfo_.blockTail;
    tilingData->ncFactor = avgComputeInfo_.ncFactor;
    tilingData->doFactor = avgComputeInfo_.doFactor;
    tilingData->hoFactor = avgComputeInfo_.hoFactor;
    tilingData->woFactor = avgComputeInfo_.woFactor;
    tilingData->ncOuter = avgComputeInfo_.ncOuter;
    tilingData->doOuter = avgComputeInfo_.doOuter;
    tilingData->hoOuter = avgComputeInfo_.hoOuter;
    tilingData->woOuter = avgComputeInfo_.woOuter;
    tilingData->ncTail = avgComputeInfo_.ncTail;
    tilingData->doTail = avgComputeInfo_.doTail;
    tilingData->hoTail = avgComputeInfo_.hoTail;
    tilingData->woTail = avgComputeInfo_.woTail;
    tilingData->maxInputSize = avgComputeInfo_.maxInputSize;
    tilingData->maxDimOut = avgComputeInfo_.maxDimOut;
}

void AdaptiveAvgPool3dParaPoolTiling::PrintTilingData() const
{
    std::ostringstream info;
    info << "nc: " << input_.nIn * input_.cIn;
    info << ", useCoreNum: " << avgComputeInfo_.useCoreNum;
    info << ", dInDim: " << input_.dIn;
    info << ", hInDim: " << input_.hIn;
    info << ", wInDim: " << input_.wIn;
    info << ", dOutDim: " << input_.dOut;
    info << ", hOutDim: " << input_.hOut;
    info << ", wOutDim: " << input_.wOut;
    info << ", blockFactor: " << avgComputeInfo_.blockFactor;
    info << ", blockTail: " << avgComputeInfo_.blockTail;
    info << ", ncFactor: " << avgComputeInfo_.ncFactor;
    info << ", doFactor: " << avgComputeInfo_.doFactor;
    info << ", hoFactor: " << avgComputeInfo_.hoFactor;
    info << ", woFactor: " << avgComputeInfo_.woFactor;
    info << ", doOuter: " << avgComputeInfo_.doOuter;
    info << ", hoOuter: " << avgComputeInfo_.hoOuter;
    info << ", woOuter: " << avgComputeInfo_.woOuter;
    info << ", ncOuter: " << avgComputeInfo_.ncOuter;
    info << ", ncTail: " << avgComputeInfo_.ncTail;
    info << ", doTail: " << avgComputeInfo_.doTail;
    info << ", hoTail: " << avgComputeInfo_.hoTail;
    info << ", woTail: " << avgComputeInfo_.woTail;
    info << ", maxInputSize: " << avgComputeInfo_.maxInputSize;
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
    context_->SetBlockDim(avgComputeInfo_.useCoreNum);
    return ge::GRAPH_SUCCESS;
}

REGISTER_OPS_TILING_TEMPLATE(AdaptiveAvgPool3d, AdaptiveAvgPool3dParaPoolTiling, 0);
} // namespace optiling

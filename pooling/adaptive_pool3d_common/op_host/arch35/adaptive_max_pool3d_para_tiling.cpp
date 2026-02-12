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
 * \file adaptive_max_pool3d_parall_pool_tiling.cpp
 * \brief
 */

#include <cstdint>
#include "adaptive_max_pool3d_tiling.h"
#include "adaptive_max_pool3d_para_tiling.h"

constexpr uint64_t KERNEL_SIZE_LIMIT = 128;
constexpr uint64_t RESERVE_UB_SIZE = 0;
constexpr uint64_t MAX_UB_BUFFER_NUM = 3;
constexpr uint64_t INT32_MAX_VALUE = 2147483647UL;
constexpr uint64_t CAL_KER_THRESHOLD = 10000;
constexpr uint64_t DOUBLE = 2;
constexpr uint64_t TRANS_ADDR_LEN = 16;

namespace optiling {

bool AdaptiveMaxPool3dParaPoolTiling::IsCapable()
{
    if (ge::GetSizeByDataType(input_.xDtype) == 0) {
        OP_LOGE(context_->GetNodeName(), "Get xDtype size is 0, not support");
        return false;
    }
    computeInfo_.vfLen = Ops::Base::GetVRegSize(context_) / ge::GetSizeByDataType(input_.xDtype);
    computeInfo_.alignNum = Ops::Base::GetUbBlockSize(context_) / ge::GetSizeByDataType(input_.xDtype);
    computeInfo_.availableUbSize = input_.ubSize - RESERVE_UB_SIZE;
    computeInfo_.ncFactor = computeInfo_.vfLen;
    computeInfo_.doFactor = 1;
    computeInfo_.hoFactor = 1;
    computeInfo_.woFactor = 1;

    computeInfo_.kernelDMax = CalKernelMax(input_.dIn, input_.dOut);
    computeInfo_.kernelHMax = CalKernelMax(input_.hIn, input_.hOut);
    computeInfo_.kernelWMax = CalKernelMax(input_.wIn, input_.wOut);

    bool isKernelSizeMeet =
        (computeInfo_.kernelDMax * computeInfo_.kernelHMax * computeInfo_.kernelWMax < KERNEL_SIZE_LIMIT);
    bool isIndexSizeMeet = (input_.dIn * input_.hIn * input_.wIn < INT32_MAX_VALUE);
    bool isNcLenEnough = input_.nIn * input_.cIn >= (computeInfo_.vfLen / DOUBLE);
    /* 计算只处理一个窗口占用的UB */
    auto occupyUbSize = CalOccupySize();
    bool isUbSizeEnough = (occupyUbSize <= computeInfo_.availableUbSize);
    bool isCapable = isKernelSizeMeet && isIndexSizeMeet && isNcLenEnough && isUbSizeEnough;
    OP_LOGD(
        context_->GetNodeName(), "AdaptiveMaxPool3dParaPoolTiling IsCapable check: %s", isCapable ? "true" : "false");
    return isCapable;
}

uint64_t AdaptiveMaxPool3dParaPoolTiling::CalKernelMax(uint64_t inputSize, uint64_t outputSize)
{
    if (outputSize == 0) {
        OP_LOGE(context_->GetNodeName(), "outputSize is 0, not support");
        return 0;
    }
    uint64_t kernelSize = 1;
    if (outputSize > CAL_KER_THRESHOLD) {
        return (inputSize + outputSize - 1) / outputSize + 1; // 防止计算时间过长
    }
    for (uint64_t i = 0; i < outputSize; i++) {
        auto kernelLeft = (i * inputSize) / outputSize;
        auto kernelRight = ((i + 1) * inputSize + outputSize - 1) / outputSize;
        auto kernelCurrent = kernelRight - kernelLeft;
        kernelSize = kernelCurrent > kernelSize ? kernelCurrent : kernelSize;
    }
    return kernelSize;
}

void AdaptiveMaxPool3dParaPoolTiling::CalMaxUbSplitSize()
{
    auto doNum = computeInfo_.doFactor;
    auto hoNum = computeInfo_.hoFactor;
    auto woNum = computeInfo_.woFactor;
    auto woNumAlign = Ops::Base::CeilAlign(woNum, computeInfo_.alignNum);

    auto diDataLen = computeInfo_.doFactor * computeInfo_.kernelDMax;
    auto hiDataLen = computeInfo_.hoFactor * computeInfo_.kernelHMax;
    auto wiDataLen = computeInfo_.woFactor * computeInfo_.kernelWMax;
    auto wiDataLenAlign = Ops::Base::CeilAlign(wiDataLen, computeInfo_.alignNum);

    auto maxD = std::max(doNum, diDataLen);
    auto maxH = std::max(hoNum, hiDataLen);
    auto maxW = std::max(woNumAlign, wiDataLenAlign);
    /* 转置接口需要按16对齐 */
    auto maxDhw = Ops::Base::CeilAlign(maxD * maxH * maxW, TRANS_ADDR_LEN);
    computeInfo_.maxInputSize = maxDhw * computeInfo_.ncFactor;
}

void AdaptiveMaxPool3dParaPoolTiling::CalUbBlockFactor()
{
    computeInfo_.doOuter = Ops::Base::CeilDiv(input_.dOut, computeInfo_.doFactor);
    computeInfo_.doTail = input_.dOut - (computeInfo_.doOuter - 1) * computeInfo_.doFactor;
    computeInfo_.hoOuter = Ops::Base::CeilDiv(input_.hOut, computeInfo_.hoFactor);
    computeInfo_.hoTail = input_.hOut - (computeInfo_.hoOuter - 1) * computeInfo_.hoFactor;
    computeInfo_.woOuter = Ops::Base::CeilDiv(input_.wOut, computeInfo_.woFactor);
    computeInfo_.woTail = input_.wOut - (computeInfo_.woOuter - 1) * computeInfo_.woFactor;
    computeInfo_.ncOuter = Ops::Base::CeilDiv(input_.nIn * input_.cIn, computeInfo_.ncFactor);
    computeInfo_.ncTail = input_.nIn * input_.cIn - (computeInfo_.ncOuter - 1) * computeInfo_.ncFactor;

    /* 总共的UB块 */
    computeInfo_.totalOuter = computeInfo_.ncOuter * computeInfo_.woOuter * computeInfo_.hoOuter * computeInfo_.doOuter;
    computeInfo_.blockFactor = Ops::Base::CeilDiv(computeInfo_.totalOuter, input_.coreNum);
    computeInfo_.useCoreNum = Ops::Base::CeilDiv(computeInfo_.totalOuter, computeInfo_.blockFactor);
    computeInfo_.blockTail = computeInfo_.totalOuter - (computeInfo_.useCoreNum - 1) * computeInfo_.blockFactor;
}

/*
* inputQue:    vl * diDataLen * hiDataLen * wiDataLenAlign
* maxQue:      diDataLen * hiDataLen * wiDataLenAlign * vl,
               hoNum * woNumAlign * diDataLen * vl,
               vl * doNum * hoNum * woNumAlign
* maxTransQue:   woNumAlign * diDataLen * hiDataLen * vl
                 doNum * hoNum * woNumAlign * vl
* maxInidceQue:      元素个数等于maxQue, sizeof(int32_t)
* maxInidceTransQue: 元素个数等于maxBuffer, sizeof(indicesDtype)
*/
uint64_t AdaptiveMaxPool3dParaPoolTiling::CalOccupySize()
{
    CalMaxUbSplitSize();
    auto occupySize = computeInfo_.maxInputSize * ge::GetSizeByDataType(input_.xDtype) * MAX_UB_BUFFER_NUM +
                      computeInfo_.maxInputSize * ge::GetSizeByDataType(ge::DT_INT32) +
                      computeInfo_.maxInputSize * ge::GetSizeByDataType(input_.indicesDtype);
    return occupySize;
}

void AdaptiveMaxPool3dParaPoolTiling::BinarySearch(uint64_t& initFactor)
{
    if (initFactor <= 1) {
        return;
    }
    uint64_t left = 1;
    uint64_t right = initFactor;
    uint64_t bestSplit = 1;

    while (left <= right) {
        uint64_t mid = left + (right - left) / DOUBLE;
        initFactor = mid;
        auto occupyUbSize = CalOccupySize();
        if (occupyUbSize < computeInfo_.availableUbSize && initFactor > 1) {
            bestSplit = mid;
            left = mid + 1;
        } else {
            right = mid - 1;
        }
    }
    initFactor = bestSplit;
}

void AdaptiveMaxPool3dParaPoolTiling::SearchOuterSingle(uint64_t& initFactor)
{
    if (initFactor <= 1) {
        return;
    }
    do {
        uint64_t lastBlockFactor = computeInfo_.blockFactor;
        initFactor -= 1;
        CalUbBlockFactor();
        if (computeInfo_.blockFactor > lastBlockFactor) {
            initFactor += 1;
            break;
        }
    } while (computeInfo_.useCoreNum < input_.coreNum && initFactor > 1);
}

ge::graphStatus AdaptiveMaxPool3dParaPoolTiling::SearchUbFactor()
{
    OP_LOGD(context_->GetNodeName(), "AdaptiveMaxPool3dParaPoolTiling search ubfactor start.");
    if (CalOccupySize() < computeInfo_.availableUbSize) {
        return ge::GRAPH_SUCCESS;
    }
    BinarySearch(computeInfo_.doFactor);
    if (CalOccupySize() < computeInfo_.availableUbSize) {
        return ge::GRAPH_SUCCESS;
    }
    BinarySearch(computeInfo_.hoFactor);
    if (CalOccupySize() < computeInfo_.availableUbSize) {
        return ge::GRAPH_SUCCESS;
    }
    BinarySearch(computeInfo_.woFactor);

    OP_LOGD(
        context_->GetNodeName(), "doFactor = %lu, hoFactor = %lu, woFactor = %lu", computeInfo_.doFactor,
        computeInfo_.hoFactor, computeInfo_.woFactor);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AdaptiveMaxPool3dParaPoolTiling::SearchOuter()
{
    OP_LOGD(context_->GetNodeName(), "AdaptiveMaxPool3dParaPoolTiling search outer start.");
    if (computeInfo_.useCoreNum == input_.coreNum) {
        return ge::GRAPH_SUCCESS;
    }
    SearchOuterSingle(computeInfo_.doFactor);
    if (computeInfo_.useCoreNum == input_.coreNum) {
        return ge::GRAPH_SUCCESS;
    }
    SearchOuterSingle(computeInfo_.hoFactor);
    if (computeInfo_.useCoreNum == input_.coreNum) {
        return ge::GRAPH_SUCCESS;
    }
    SearchOuterSingle(computeInfo_.woFactor);

    OP_LOGD(
        context_->GetNodeName(), "doFactor = %lu, hoFactor = %lu, woFactor = %lu", computeInfo_.doFactor,
        computeInfo_.hoFactor, computeInfo_.woFactor);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AdaptiveMaxPool3dParaPoolTiling::InitUbFactor()
{
    OP_LOGD(context_->GetNodeName(), "AdaptiveMaxPool3dParaPoolTiling init ubfactor start.");
    auto kernelD = computeInfo_.kernelDMax;
    auto kernelH = computeInfo_.kernelHMax;
    auto kernelW = computeInfo_.kernelWMax;
    OP_CHECK_IF(
        (kernelW <= 0 || kernelH <= 0 || kernelD <= 0),
        OP_LOGE(context_->GetNodeName(), "Kernel size <= 0, not support"), return ge::GRAPH_FAILED);

    computeInfo_.ncFactor = computeInfo_.vfLen;
    computeInfo_.woFactor = input_.wOut;
    computeInfo_.hoFactor = input_.hOut;
    computeInfo_.doFactor = input_.dOut;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AdaptiveMaxPool3dParaPoolTiling::DoOpTiling()
{
    OP_LOGD(context_->GetNodeName(), "AdaptiveMaxPool3dParaPoolTiling DoOpTiling start.");
    OP_CHECK_IF(
        InitUbFactor() != ge::GRAPH_SUCCESS,
        OP_LOGE(context_->GetNodeName(), "AdaptiveMaxPool3dParaPoolTiling init ubfactor failed"),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        SearchUbFactor() != ge::GRAPH_SUCCESS,
        OP_LOGE(context_->GetNodeName(), "AdaptiveMaxPool3dParaPoolTiling search ubfactor failed"),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        SearchOuter() != ge::GRAPH_SUCCESS,
        OP_LOGE(context_->GetNodeName(), "AdaptiveMaxPool3dParaPoolTiling search ubfactor failed"),
        return ge::GRAPH_FAILED);

    CalUbBlockFactor();
    CalMaxUbSplitSize();

    SetTilingData();
    return ge::GRAPH_SUCCESS;
}

void AdaptiveMaxPool3dParaPoolTiling::SetTilingData()
{
    AdaptivePool3DTiling::AdaptivePool3dParaKernelTilingData* tilingData =
        context_->GetTilingData<AdaptivePool3dParaKernelTilingData>();

    tilingData->dIn = input_.dIn;
    tilingData->hIn = input_.hIn;
    tilingData->wIn = input_.wIn;
    tilingData->dOut = input_.dOut;
    tilingData->hOut = input_.hOut;
    tilingData->wOut = input_.wOut;
    tilingData->useCoreNum = computeInfo_.useCoreNum;
    tilingData->blockFactor = computeInfo_.blockFactor;
    tilingData->blockTail = computeInfo_.blockTail;
    tilingData->ncFactor = computeInfo_.ncFactor;
    tilingData->doFactor = computeInfo_.doFactor;
    tilingData->hoFactor = computeInfo_.hoFactor;
    tilingData->woFactor = computeInfo_.woFactor;
    tilingData->ncOuter = computeInfo_.ncOuter;
    tilingData->doOuter = computeInfo_.doOuter;
    tilingData->hoOuter = computeInfo_.hoOuter;
    tilingData->woOuter = computeInfo_.woOuter;
    tilingData->ncTail = computeInfo_.ncTail;
    tilingData->doTail = computeInfo_.doTail;
    tilingData->hoTail = computeInfo_.hoTail;
    tilingData->woTail = computeInfo_.woTail;
    tilingData->maxInputSize = computeInfo_.maxInputSize;
}

void AdaptiveMaxPool3dParaPoolTiling::PrintTilingData() const
{
    std::ostringstream info;
    info << "nc: " << input_.nIn * input_.cIn;
    info << ", dInDim: " << input_.dIn;
    info << ", hInDim: " << input_.hIn;
    info << ", wInDim: " << input_.wIn;
    info << ", dOutDim: " << input_.dOut;
    info << ", hOutDim: " << input_.hOut;
    info << ", wOutDim: " << input_.wOut;
    info << ", useCoreNum: " << computeInfo_.useCoreNum;
    info << ", blockFactor: " << computeInfo_.blockFactor;
    info << ", blockTail: " << computeInfo_.blockTail;
    info << ", ncFactor: " << computeInfo_.ncFactor;
    info << ", doFactor: " << computeInfo_.doFactor;
    info << ", hoFactor: " << computeInfo_.hoFactor;
    info << ", woFactor: " << computeInfo_.woFactor;
    info << ", ncOuter: " << computeInfo_.ncOuter;
    info << ", doOuter: " << computeInfo_.doOuter;
    info << ", hoOuter: " << computeInfo_.hoOuter;
    info << ", woOuter: " << computeInfo_.woOuter;
    info << ", ncTail: " << computeInfo_.ncTail;
    info << ", doTail: " << computeInfo_.doTail;
    info << ", hoTail: " << computeInfo_.hoTail;
    info << ", woTail: " << computeInfo_.woTail;
    info << ", maxInputSize: " << computeInfo_.maxInputSize;
    info << std::endl;

    OP_LOGI(context_->GetNodeName(), "%s", info.str().c_str());
}

uint64_t AdaptiveMaxPool3dParaPoolTiling::GetTilingKey() const
{
    OP_LOGD(context_->GetNodeName(), "AdaptiveMaxPool3dParaPoolTiling GetTilingKey start.");
    return GET_TPL_TILING_KEY(TPL_MODE_0, TPL_DYTPE_0, TPL_MULTI_MODE_0);
}

ge::graphStatus AdaptiveMaxPool3dParaPoolTiling::PostTiling()
{
    OP_LOGD(context_->GetNodeName(), "AdaptiveMaxPool3dParaPoolTiling PostTiling start.");
    context_->SetBlockDim(computeInfo_.useCoreNum);
    return ge::GRAPH_SUCCESS;
}

REGISTER_OPS_TILING_TEMPLATE(AdaptiveMaxPool3d, AdaptiveMaxPool3dParaPoolTiling, 0);
} // namespace optiling

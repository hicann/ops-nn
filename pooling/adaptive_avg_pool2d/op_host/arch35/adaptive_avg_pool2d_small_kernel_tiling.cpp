/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file adaptive_avg_pool2d_small_kernel_tiling.cpp
 * \brief
 */

#include <cstdint>
#include "adaptive_avg_pool2d_small_kernel_tiling.h"

constexpr uint64_t KERNEL_SIZE_LIMIT = 128;
constexpr uint64_t RESERVE_UB_SIZE = 0;
constexpr uint64_t DOUBLE = 2;
constexpr uint64_t TRANS_ADDR_LEN = 16;
constexpr uint64_t INT32_MAX_VALUE = 2147483647UL;

namespace optiling {

bool AdaptiveAvgPool2dSmallKernelTiling::IsCapable()
{
    computeInfo_.xDtypeSize = ge::GetSizeByDataType(input_.xDtype);
    if (computeInfo_.xDtypeSize == 0) {
        OP_LOGE(context_->GetNodeName(), "Get xDtype size is 0, not support");
        return false;
    }
    computeInfo_.vfLen = Ops::Base::GetVRegSize(context_) / computeInfo_.xDtypeSize;
    computeInfo_.alignNum = Ops::Base::GetUbBlockSize(context_) / computeInfo_.xDtypeSize;
    computeInfo_.availableUbSize = input_.ubSize - RESERVE_UB_SIZE;
    computeInfo_.ncFactor = computeInfo_.vfLen;
    computeInfo_.hoFactor = 1;
    computeInfo_.woFactor = 1;

    computeInfo_.kernelHMax = CalKernelSizeOneDimMax(input_.hIn, input_.hOut);
    computeInfo_.kernelWMax = CalKernelSizeOneDimMax(input_.wIn, input_.wOut);

    bool isKernelSizeMeet =
        (computeInfo_.kernelHMax * computeInfo_.kernelWMax < KERNEL_SIZE_LIMIT);
    bool isNcLenEnough = input_.nIn * input_.cIn >= (computeInfo_.vfLen / DOUBLE);
    /* 计算只处理一个窗口占用的UB */
    bool isCapable = isKernelSizeMeet && isNcLenEnough && IsMeetUbSize();
    OP_LOGI(
        context_->GetNodeName(), "AdaptiveAvgPool2dSmallKernelTiling IsCapable check: %s", isCapable ? "true" : "false");
    return isCapable;
}

void AdaptiveAvgPool2dSmallKernelTiling::CalMaxUbSplitSize()
{
    uint64_t hoNum = computeInfo_.hoFactor;
    uint64_t woNum = computeInfo_.woFactor;
    uint64_t woNumAlign = Ops::Base::CeilAlign(woNum, computeInfo_.alignNum);

    uint64_t hiDataLen = computeInfo_.hoFactor * computeInfo_.kernelHMax;
    uint64_t wiDataLen = computeInfo_.woFactor * computeInfo_.kernelWMax;
    uint64_t wiDataLenAlign = Ops::Base::CeilAlign(wiDataLen, computeInfo_.alignNum);

    computeInfo_.inputQueSize = computeInfo_.ncFactor * hiDataLen * wiDataLenAlign * computeInfo_.xDtypeSize;
    uint64_t outTransAlign = Ops::Base::CeilAlign(hoNum * woNumAlign, TRANS_ADDR_LEN);
    computeInfo_.resQue1Size = std::max(hiDataLen * wiDataLenAlign, outTransAlign) * computeInfo_.ncFactor * sizeof(float);
    computeInfo_.resQue2Size = std::max(woNumAlign * hiDataLen, outTransAlign) * computeInfo_.ncFactor * sizeof(float);

    computeInfo_.maxDimOut = std::max(hoNum, woNum);
}

void AdaptiveAvgPool2dSmallKernelTiling::CalUbBlockFactor()
{
    computeInfo_.hoOuter = Ops::Base::CeilDiv(input_.hOut, computeInfo_.hoFactor);
    computeInfo_.hoTail = input_.hOut - (computeInfo_.hoOuter - 1) * computeInfo_.hoFactor;
    computeInfo_.woOuter = Ops::Base::CeilDiv(input_.wOut, computeInfo_.woFactor);
    computeInfo_.woTail = input_.wOut - (computeInfo_.woOuter - 1) * computeInfo_.woFactor;
    computeInfo_.ncOuter = Ops::Base::CeilDiv(input_.nIn * input_.cIn, computeInfo_.ncFactor);
    computeInfo_.ncTail = input_.nIn * input_.cIn - (computeInfo_.ncOuter - 1) * computeInfo_.ncFactor;

    /* 总共的UB块 */
    computeInfo_.totalOuter = computeInfo_.ncOuter * computeInfo_.woOuter * computeInfo_.hoOuter;
    computeInfo_.blockFactor = Ops::Base::CeilDiv(computeInfo_.totalOuter, input_.coreNum);
    computeInfo_.useCoreNum = Ops::Base::CeilDiv(computeInfo_.totalOuter, computeInfo_.blockFactor);
    computeInfo_.blockTail = computeInfo_.totalOuter - (computeInfo_.useCoreNum - 1) * computeInfo_.blockFactor;
}

/*
* inputQue:    vl * hiDataLen * wiDataLenAlign
* resQue:      hiDataLen * wiDataLenAlign * vl,
               hoNum * woNumAlign * vl
* resTransQue:   woNumAlign * hiDataLen * vl
                 vl * hoNum * woNumAlign
* startIdxBuf:   大小为maxDimOut
* kerSizeBuf:    各轴对应的factor大小
*/
bool AdaptiveAvgPool2dSmallKernelTiling::IsMeetUbSize()
{
    CalMaxUbSplitSize();
    uint64_t dataBlock = Ops::Base::GetUbBlockSize(context_);
    uint64_t occupySize = computeInfo_.inputQueSize + computeInfo_.resQue1Size + computeInfo_.resQue2Size +
                      Ops::Base::CeilAlign(computeInfo_.maxDimOut * sizeof(int32_t), dataBlock) +
                      Ops::Base::CeilAlign(computeInfo_.hoFactor * sizeof(int32_t), dataBlock) +
                      Ops::Base::CeilAlign(computeInfo_.woFactor * sizeof(int32_t), dataBlock);
    return occupySize <= computeInfo_.availableUbSize;
}

void AdaptiveAvgPool2dSmallKernelTiling::BinarySearch(uint64_t& initFactor)
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
        if (IsMeetUbSize() && initFactor > 1) {
            bestSplit = mid;
            left = mid + 1;
        } else {
            right = mid - 1;
        }
    }
    initFactor = bestSplit;
}

void AdaptiveAvgPool2dSmallKernelTiling::SearchOuterSingle(uint64_t& initFactor)
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

ge::graphStatus AdaptiveAvgPool2dSmallKernelTiling::SearchUbFactor()
{
    if (IsMeetUbSize()) {
        return ge::GRAPH_SUCCESS;
    }
    BinarySearch(computeInfo_.hoFactor);
    if (IsMeetUbSize()) {
        return ge::GRAPH_SUCCESS;
    }
    BinarySearch(computeInfo_.woFactor);

    OP_LOGI(
        context_->GetNodeName(), "hoFactor = %lu, woFactor = %lu", computeInfo_.hoFactor, computeInfo_.woFactor);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AdaptiveAvgPool2dSmallKernelTiling::SearchOuter()
{
    CalUbBlockFactor();
    if (computeInfo_.useCoreNum == input_.coreNum) {
        return ge::GRAPH_SUCCESS;
    }
    SearchOuterSingle(computeInfo_.hoFactor);
    CalUbBlockFactor();
    if (computeInfo_.useCoreNum == input_.coreNum) {
        return ge::GRAPH_SUCCESS;
    }
    SearchOuterSingle(computeInfo_.woFactor);

    OP_LOGI(
        context_->GetNodeName(), "hoFactor = %lu, woFactor = %lu", computeInfo_.hoFactor, computeInfo_.woFactor);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AdaptiveAvgPool2dSmallKernelTiling::InitUbFactor()
{
    uint64_t kernelH = computeInfo_.kernelHMax;
    uint64_t kernelW = computeInfo_.kernelWMax;
    OP_CHECK_IF(
        (kernelW <= 0 || kernelH <= 0),
        OP_LOGE(context_->GetNodeName(), "Kernel size <= 0, not support"), return ge::GRAPH_FAILED);

    computeInfo_.ncFactor = computeInfo_.vfLen;
    computeInfo_.woFactor = input_.wOut;
    computeInfo_.hoFactor = input_.hOut;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AdaptiveAvgPool2dSmallKernelTiling::DoOpTiling()
{
    OP_LOGI(context_->GetNodeName(), "AdaptiveAvgPool2dSmallKernelTiling DoOpTiling start.");
    OP_CHECK_IF(
        InitUbFactor() != ge::GRAPH_SUCCESS,
        OP_LOGE(context_->GetNodeName(), "AdaptiveAvgPool2dSmallKernelTiling InitUbFactor failed"),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        SearchUbFactor() != ge::GRAPH_SUCCESS,
        OP_LOGE(context_->GetNodeName(), "AdaptiveAvgPool2dSmallKernelTiling SearchUbFactor failed"),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        SearchOuter() != ge::GRAPH_SUCCESS,
        OP_LOGE(context_->GetNodeName(), "AdaptiveAvgPool2dSmallKernelTiling SearchOuter failed"),
        return ge::GRAPH_FAILED);

    CalUbBlockFactor();
    CalMaxUbSplitSize();

    OP_CHECK_IF(
        SetTilingData() != ge::GRAPH_SUCCESS,
        OP_LOGE(context_->GetNodeName(), "AdaptiveAvgPool2dSmallKernelTiling SetTilingData failed"),
        return ge::GRAPH_FAILED);
    PrintTilingData();
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AdaptiveAvgPool2dSmallKernelTiling::SetTilingData()
{
    AdaptiveAvgPool2dOp::AdaptivePool2dSmallKernelTilingData* tilingData =
        context_->GetTilingData<AdaptivePool2dSmallKernelTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context_, tilingData);

    tilingData->hIn = static_cast<int64_t>(input_.hIn);
    tilingData->wIn = static_cast<int64_t>(input_.wIn);
    tilingData->hOut = static_cast<int64_t>(input_.hOut);
    tilingData->wOut = static_cast<int64_t>(input_.wOut);
    tilingData->useCoreNum = static_cast<int64_t>(computeInfo_.useCoreNum);
    tilingData->blockFactor = static_cast<int64_t>(computeInfo_.blockFactor);
    tilingData->blockTail = static_cast<int64_t>(computeInfo_.blockTail);
    tilingData->ncFactor = static_cast<int64_t>(computeInfo_.ncFactor);
    tilingData->hoFactor = static_cast<int64_t>(computeInfo_.hoFactor);
    tilingData->woFactor = static_cast<int64_t>(computeInfo_.woFactor);
    tilingData->ncOuter = static_cast<int64_t>(computeInfo_.ncOuter);
    tilingData->hoOuter = static_cast<int64_t>(computeInfo_.hoOuter);
    tilingData->woOuter = static_cast<int64_t>(computeInfo_.woOuter);
    tilingData->ncTail = static_cast<int64_t>(computeInfo_.ncTail);
    tilingData->hoTail = static_cast<int64_t>(computeInfo_.hoTail);
    tilingData->woTail = static_cast<int64_t>(computeInfo_.woTail);
    tilingData->inputQueSize = static_cast<int64_t>(computeInfo_.inputQueSize);
    tilingData->resQue1Size = static_cast<int64_t>(computeInfo_.resQue1Size);
    tilingData->resQue2Size = static_cast<int64_t>(computeInfo_.resQue2Size);
    tilingData->maxDimOut = static_cast<int64_t>(computeInfo_.maxDimOut);
    return ge::GRAPH_SUCCESS;
}

void AdaptiveAvgPool2dSmallKernelTiling::PrintTilingData() const
{
    std::ostringstream info;
    info << "nc: " << input_.nIn * input_.cIn;
    info << ", hInDim: " << input_.hIn;
    info << ", wInDim: " << input_.wIn;
    info << ", hOutDim: " << input_.hOut;
    info << ", wOutDim: " << input_.wOut;
    info << ", useCoreNum: " << computeInfo_.useCoreNum;
    info << ", blockFactor: " << computeInfo_.blockFactor;
    info << ", blockTail: " << computeInfo_.blockTail;
    info << ", ncFactor: " << computeInfo_.ncFactor;
    info << ", hoFactor: " << computeInfo_.hoFactor;
    info << ", woFactor: " << computeInfo_.woFactor;
    info << ", ncOuter: " << computeInfo_.ncOuter;
    info << ", hoOuter: " << computeInfo_.hoOuter;
    info << ", woOuter: " << computeInfo_.woOuter;
    info << ", ncTail: " << computeInfo_.ncTail;
    info << ", hoTail: " << computeInfo_.hoTail;
    info << ", woTail: " << computeInfo_.woTail;
    info << ", inputQueSize: " << computeInfo_.inputQueSize;
    info << ", resQue1Size: " << computeInfo_.resQue1Size;
    info << ", resQue2Size: " << computeInfo_.resQue2Size;
    info << ", maxDimOut: " << computeInfo_.maxDimOut;

    OP_LOGI(context_->GetNodeName(), "%s", info.str().c_str());
}

uint64_t AdaptiveAvgPool2dSmallKernelTiling::GetTilingKey() const
{
    int64_t maxIdxValue = std::max(input_.hIn * input_.hOut, input_.wIn * input_.wOut);
    uint64_t idxTypeMode = static_cast<uint64_t>(maxIdxValue) < INT32_MAX_VALUE ? TPL_INT32_UINT32 : TPL_INT64_UINT64;
    return GET_TPL_TILING_KEY(TPL_SMALL_KERNEL, idxTypeMode);
}

ge::graphStatus AdaptiveAvgPool2dSmallKernelTiling::PostTiling()
{
    context_->SetBlockDim(computeInfo_.useCoreNum);
    return ge::GRAPH_SUCCESS;
}

REGISTER_OPS_TILING_TEMPLATE(AdaptiveAvgPool2d, AdaptiveAvgPool2dSmallKernelTiling, 0);
} // namespace optiling

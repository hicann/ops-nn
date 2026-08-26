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
 * \file adaptive_avg_pool3d_gather_trans_tiling.cpp
 * \brief
 */
#include <algorithm>
#include <cstdint>
#include "adaptive_avg_pool3d_gather_trans_tiling.h"

constexpr uint64_t GT_KERNEL_SIZE_LIMIT = 6;
constexpr uint64_t GT_RESERVE_UB_SIZE = 0;
constexpr uint64_t GT_DOUBLE = 2;
constexpr uint64_t GT_NC_SIZE_LIMIT = 6000;
constexpr uint64_t GT_MAX_BLOCK_COUNT = 4095;
constexpr int16_t GT_B16_IDX_MAX = 65535;
constexpr uint64_t GT_W_FULL_GATHER_DEGRADE_KSIZE = 3;

namespace optiling {

uint64_t AdaptiveAvgPool3dGatherTransTiling::CalOccupySize(uint64_t ncBatch) const
{
    uint64_t dtypeSize = ge::GetSizeByDataType(input_.xDtype);
    uint64_t fp32Size = ge::GetSizeByDataType(ge::DT_FLOAT);
    uint64_t int32Size = ge::GetSizeByDataType(ge::DT_INT32);
    uint64_t blockSize = Ops::Base::GetUbBlockSize(context_);
    uint64_t ncTotal = ncBatch * gtInfo_.ncFactor;
    uint64_t spatialBlock = gtInfo_.maxDInBlock * gtInfo_.inHW;
    uint64_t outBlock = gtInfo_.maxDoBlock * gtInfo_.outHW;
    uint64_t inQueSize = Ops::Base::CeilAlign(ncTotal * spatialBlock * dtypeSize, blockSize);
    uint64_t transBufSize = Ops::Base::CeilAlign(ncBatch * spatialBlock * gtInfo_.vfLen * dtypeSize, blockSize);
    if (input_.xDtype != ge::DT_FLOAT) {
        transBufSize = std::max(transBufSize, Ops::Base::CeilAlign(ncTotal * outBlock * dtypeSize, blockSize));
    }
    uint64_t yTransBufSize = Ops::Base::CeilAlign(ncTotal * outBlock * fp32Size, blockSize);
    uint64_t indexBufSize = Ops::Base::CeilAlign(gtInfo_.vfLen * int32Size, blockSize);
    uint64_t shareBufSize = std::max(inQueSize, yTransBufSize);
    uint64_t allBufSize = shareBufSize + transBufSize + indexBufSize * GT_DOUBLE;
    return allBufSize;
}

void AdaptiveAvgPool3dGatherTransTiling::CalMaxDInBlock(uint64_t doFactor)
{
    uint64_t dIn = input_.dIn;
    uint64_t dOut = input_.dOut;
    uint64_t doOuter = Ops::Base::CeilDiv(dOut, doFactor);
    uint64_t maxDInBlock = 0;
    uint64_t maxDoBlock = 0;
    for (uint64_t b = 0; b < doOuter; ++b) {
        uint64_t odStart = b * doFactor;
        uint64_t odEnd = std::min(odStart + doFactor, dOut);
        uint64_t dLo = odStart * dIn / dOut;
        uint64_t dHi = (odEnd * dIn + dOut - 1) / dOut;
        maxDInBlock = std::max(maxDInBlock, dHi - dLo);
        maxDoBlock = std::max(maxDoBlock, odEnd - odStart);
    }
    gtInfo_.maxDInBlock = maxDInBlock;
    gtInfo_.maxDoBlock = maxDoBlock;
}

void AdaptiveAvgPool3dGatherTransTiling::SetDoBlockInfo(uint64_t doFactor)
{
    gtInfo_.doFactor = doFactor;
    gtInfo_.doOuter = Ops::Base::CeilDiv(input_.dOut, doFactor);
    gtInfo_.doTail = input_.dOut - (gtInfo_.doOuter - 1) * doFactor;
    CalMaxDInBlock(doFactor);
}

bool AdaptiveAvgPool3dGatherTransTiling::IsIndexInRange() const
{
    if (input_.xDtype == ge::DT_FLOAT) {
        return true;
    }
    uint64_t maxIdx = (gtInfo_.ncFactor - 1) * gtInfo_.maxDInBlock * gtInfo_.inHW;
    return maxIdx <= GT_B16_IDX_MAX;
}

void AdaptiveAvgPool3dGatherTransTiling::CalDoFactor()
{
    uint64_t doFactor = input_.dOut;
    SetDoBlockInfo(doFactor);
    while (doFactor > 1 && (CalOccupySize(1) > gtInfo_.availableUbSize || !IsIndexInRange())) {
        --doFactor;
        SetDoBlockInfo(doFactor);
    }
}

void AdaptiveAvgPool3dGatherTransTiling::CalBlockFactor()
{
    uint64_t nc = input_.nIn * input_.cIn;
    gtInfo_.ncOuter = Ops::Base::CeilDiv(nc, gtInfo_.ncFactor);
    gtInfo_.ncTail = nc - (gtInfo_.ncOuter - 1) * gtInfo_.ncFactor;
    uint64_t hi = Ops::Base::CeilDiv(gtInfo_.ncOuter, input_.coreNum);
    if (gtInfo_.doOuter > 1) {
        hi = std::min(hi, GT_MAX_BLOCK_COUNT / gtInfo_.ncFactor);
    }
    hi = std::max(hi, static_cast<uint64_t>(1));
    uint64_t lo = 1;
    while (lo < hi) {
        uint64_t mid = lo + (hi - lo + 1) / GT_DOUBLE;
        if (CalOccupySize(mid) <= gtInfo_.availableUbSize) {
            lo = mid;
        } else {
            hi = mid - 1;
        }
    }
    gtInfo_.ncBatch = lo;
    gtInfo_.tileNum = Ops::Base::CeilDiv(gtInfo_.ncOuter, gtInfo_.ncBatch);
    uint64_t totalTasks = gtInfo_.doOuter * gtInfo_.tileNum;
    gtInfo_.blockFactor = Ops::Base::CeilDiv(totalTasks, input_.coreNum);
    gtInfo_.useCoreNum = Ops::Base::CeilDiv(totalTasks, gtInfo_.blockFactor);
    gtInfo_.blockTail = totalTasks - (gtInfo_.useCoreNum - 1) * gtInfo_.blockFactor;
}

bool AdaptiveAvgPool3dGatherTransTiling::IsCapable()
{
    OP_TILING_CHECK(GetAndCheckDataFormat() != ge::GRAPH_SUCCESS,
                    VECTOR_INNER_ERR_REPORT_TILIING(context_, "GetAndCheckDataFormat fail."), return ge::GRAPH_FAILED);
    if (input_.dataFormat != ge::Format::FORMAT_NCDHW) {
        OP_LOGD(context_->GetNodeName(), "AdaptiveAvgPool3dGatherTransTiling only support attr data_format NCDHW");
        return false;
    }
    if (ge::GetSizeByDataType(input_.xDtype) == 0) {
        OP_LOGE_FOR_INVALID_DTYPE("AdaptiveAvgPool3d", "x", "unknown/unsupported", "[DT_FLOAT, DT_FLOAT16, DT_BF16]");
        return false;
    }
    uint64_t dtypeSize = ge::GetSizeByDataType(input_.xDtype);
    gtInfo_.vfLen = Ops::Base::GetVRegSize(context_) / dtypeSize;
    gtInfo_.ncFactor = gtInfo_.vfLen;
    gtInfo_.availableUbSize = input_.ubSize - GT_RESERVE_UB_SIZE;
    gtInfo_.spatialIn = input_.dIn * input_.hIn * input_.wIn;
    gtInfo_.outDHW = input_.dOut * input_.hOut * input_.wOut;
    gtInfo_.inHW = input_.hIn * input_.wIn;
    gtInfo_.outHW = input_.hOut * input_.wOut;
    gtInfo_.kernelDMax = CalKernelSizeOneDimMax(input_.dIn, input_.dOut);
    gtInfo_.kernelHMax = CalKernelSizeOneDimMax(input_.hIn, input_.hOut);
    gtInfo_.kernelWMax = CalKernelSizeOneDimMax(input_.wIn, input_.wOut);
    bool isKernelSizeMeet = (gtInfo_.kernelDMax * gtInfo_.kernelHMax * gtInfo_.kernelWMax <= GT_KERNEL_SIZE_LIMIT);
    bool isNcLenEnough = input_.nIn * input_.cIn >= GT_NC_SIZE_LIMIT;
    CalDoFactor();
    bool isUbSizeEnough = (CalOccupySize(1) <= gtInfo_.availableUbSize) && IsIndexInRange();
    bool isWDimFullGatherDegrade = (input_.dIn == input_.dOut) && (input_.hIn == input_.hOut) && (input_.wOut == 1) &&
                                   (gtInfo_.kernelWMax >= GT_W_FULL_GATHER_DEGRADE_KSIZE);
    bool isCapable = isKernelSizeMeet && isNcLenEnough && isUbSizeEnough && !isWDimFullGatherDegrade;
    OP_LOGD(context_->GetNodeName(), "AdaptiveAvgPool3dGatherTransTiling IsCapable check: %s",
            isCapable ? "true" : "false");
    return isCapable;
}

ge::graphStatus AdaptiveAvgPool3dGatherTransTiling::DoOpTiling()
{
    OP_LOGD(context_->GetNodeName(), "AdaptiveAvgPool3dGatherTransTiling DoOpTiling start.");

    CalBlockFactor();
    SetTilingData();
    PrintTilingData();

    return ge::GRAPH_SUCCESS;
}

void AdaptiveAvgPool3dGatherTransTiling::SetTilingData()
{
    AdaptivePool3DTiling::AdaptivePool3dGatherTransTilingData*
        tilingData = context_->GetTilingData<AdaptivePool3dGatherTransTilingData>();
    OP_CHECK_IF(tilingData == nullptr, OP_LOGE(context_->GetNodeName(), "tilingData is null"), return);
    tilingData->dIn = input_.dIn;
    tilingData->hIn = input_.hIn;
    tilingData->wIn = input_.wIn;
    tilingData->dOut = input_.dOut;
    tilingData->hOut = input_.hOut;
    tilingData->wOut = input_.wOut;
    tilingData->useCoreNum = gtInfo_.useCoreNum;
    tilingData->blockFactor = gtInfo_.blockFactor;
    tilingData->blockTail = gtInfo_.blockTail;
    tilingData->ncFactor = gtInfo_.ncFactor;
    tilingData->ncOuter = gtInfo_.ncOuter;
    tilingData->ncTail = gtInfo_.ncTail;
    tilingData->ncBatch = gtInfo_.ncBatch;
    tilingData->doFactor = gtInfo_.doFactor;
    tilingData->doOuter = gtInfo_.doOuter;
    tilingData->doTail = gtInfo_.doTail;
    tilingData->maxDInBlock = gtInfo_.maxDInBlock;
    tilingData->maxDoBlock = gtInfo_.maxDoBlock;
}

void AdaptiveAvgPool3dGatherTransTiling::PrintTilingData() const
{
    std::ostringstream info;
    info << "nc: " << input_.nIn * input_.cIn;
    info << ", useCoreNum: " << gtInfo_.useCoreNum;
    info << ", dInDim: " << input_.dIn;
    info << ", hInDim: " << input_.hIn;
    info << ", wInDim: " << input_.wIn;
    info << ", dOutDim: " << input_.dOut;
    info << ", hOutDim: " << input_.hOut;
    info << ", wOutDim: " << input_.wOut;
    info << ", blockFactor: " << gtInfo_.blockFactor;
    info << ", blockTail: " << gtInfo_.blockTail;
    info << ", ncFactor: " << gtInfo_.ncFactor;
    info << ", ncOuter: " << gtInfo_.ncOuter;
    info << ", ncTail: " << gtInfo_.ncTail;
    info << ", ncBatch: " << gtInfo_.ncBatch;
    info << ", doFactor: " << gtInfo_.doFactor;
    info << ", doOuter: " << gtInfo_.doOuter;
    info << ", doTail: " << gtInfo_.doTail;
    info << ", maxDInBlock: " << gtInfo_.maxDInBlock;
    info << ", maxDoBlock: " << gtInfo_.maxDoBlock;
    info << ", tileNum: " << gtInfo_.tileNum;
    info << std::endl;
    OP_LOGI(context_->GetNodeName(), "%s", info.str().c_str());
}

uint64_t AdaptiveAvgPool3dGatherTransTiling::GetTilingKey() const
{
    OP_LOGD(context_->GetNodeName(), "AdaptiveAvgPool3dGatherTransTiling GetTilingKey start.");

    return GET_TPL_TILING_KEY(TPL_MODE_0, TPL_DTYPE_0, TPL_MULTI_MODE_1, TPL_DATA_FORMAT_MODE_0);
}

ge::graphStatus AdaptiveAvgPool3dGatherTransTiling::PostTiling()
{
    OP_LOGD(context_->GetNodeName(), "AdaptiveAvgPool3dGatherTransTiling PostTiling start.");
    context_->SetBlockDim(gtInfo_.useCoreNum);
    return ge::GRAPH_SUCCESS;
}
REGISTER_OPS_TILING_TEMPLATE(AdaptiveAvgPool3d, AdaptiveAvgPool3dGatherTransTiling, 1);
} // namespace optiling

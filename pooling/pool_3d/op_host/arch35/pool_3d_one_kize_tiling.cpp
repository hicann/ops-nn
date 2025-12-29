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
 * \file pool_3d_one_kize_tiling.cpp
 * \brief
 */
#include "op_util.h"

#include "pool_tiling_templates_registry.h"
#include "pool_3d_one_kize_tiling.h"
#include "op_common/op_host/util/platform_util.h"
#include "util/math_util.h"

namespace optiling
{
static constexpr int64_t UB_RESVERVED_SIZE = 1024;
static constexpr uint64_t ONE_KSIZE_TILING_KEY = 100001;
static constexpr int64_t DOUBLE_BUFFER = 2;
static constexpr int64_t COEXISTING_QUANTITY = 2;
static constexpr int64_t DIGIT_TWO = 2;

static constexpr int64_t BLOCK_SPLIT_THREHOLD = 1024;

static constexpr int64_t MAX_CHANNEL = 256;
static constexpr int32_t FORMAT_NCDHW = 0;
static constexpr int32_t FORMAT_NDHWC = 1;

void Pool3DOneKsizeTiling::InitializationVars()
{
    oneBlockNum_ = Ops::Base::GetUbBlockSize(context_) / inputData_.dtypeSize;
    availableUb_ = static_cast<int64_t>(ubSize_ - UB_RESVERVED_SIZE) / inputData_.dtypeSize;
    availableUb_ = availableUb_ / DOUBLE_BUFFER / COEXISTING_QUANTITY;
    availableUb_ -= oneBlockNum_;
    if (inputData_.inputFormat == ge::Format::FORMAT_NDHWC && inputData_.channels != 1) {
        availableUb_ = availableUb_ / inputData_.channels;
        dataFormat_ = FORMAT_NDHWC;
    } else {
        dataFormat_ = FORMAT_NCDHW;
    }
}

bool Pool3DOneKsizeTiling::IsCapable()
{
    if (inputData_.kernelSize[D_DIM] != 1 || inputData_.kernelSize[H_DIM] != 1 || inputData_.kernelSize[W_DIM] != 1) {
        return false;
    }
    if (inputData_.inputFormat == ge::Format::FORMAT_NDHWC && inputData_.channels * inputData_.dtypeSize > MAX_CHANNEL) {
        return false;
    }
    InitializationVars();
    return true;
}

uint64_t Pool3DOneKsizeTiling::GetTilingKey() const
{
    uint64_t tilingKey = ONE_KSIZE_TILING_KEY;
    return tilingKey;
}

ge::graphStatus Pool3DOneKsizeTiling::DoLibApiTiling()
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus Pool3DOneKsizeTiling::GetWorkspaceSize()
{
    uint32_t sysWorkspace = WS_SYS_SIZE;
    size_t* currentWorkspace = context_->GetWorkspaceSizes(1);
    OPS_CHECK_NULL_WITH_CONTEXT(context_, currentWorkspace);
    currentWorkspace[0] = sysWorkspace;

    return ge::GRAPH_SUCCESS;
}

void Pool3DOneKsizeTiling::DoUBTilingSingle()
{
    // d*h*w全载
    if (inputData_.outShape[D_DIM] * inputData_.outShape[W_DIM] * inputData_.outShape[H_DIM] <= availableUb_) {
        int64_t maxNfactor = availableUb_ / (inputData_.outShape[D_DIM] * inputData_.outShape[W_DIM] * inputData_.outShape[H_DIM]);
        ubFactorN_ = std::min(inputData_.batches, maxNfactor);
        outUbFactorD_ = inputData_.outShape[D_DIM];
        outUbFactorH_ = inputData_.outShape[H_DIM];
        outUbFactorW_ = inputData_.outShape[W_DIM];
        nLoop_ = (inputData_.batches + ubFactorN_ - 1) / ubFactorN_;
        hLoop_ = 1;
        wLoop_ = 1;
        dLoop_ = 1;
        return;
    }
    // h*w全载
    if (inputData_.outShape[W_DIM] * inputData_.outShape[H_DIM] <= availableUb_) {
        int64_t maxDfactor = availableUb_ / (inputData_.outShape[W_DIM] * inputData_.outShape[H_DIM]);
        ubFactorN_ = 1;
        outUbFactorD_ = std::min(inputData_.outShape[D_DIM], maxDfactor);
        outUbFactorH_ = inputData_.outShape[H_DIM];
        outUbFactorW_ = inputData_.outShape[W_DIM];
        nLoop_ = inputData_.batches;
        dLoop_ = (inputData_.outShape[D_DIM] + outUbFactorD_ - 1) / outUbFactorD_;
        hLoop_ = 1;
        wLoop_ = 1;
        return;
    }    
    // w全载
    if (inputData_.outShape[W_DIM] <= availableUb_) {
        int64_t maxHfactor = availableUb_ / inputData_.outShape[W_DIM];
        ubFactorN_ = 1;
        outUbFactorD_ = 1;
        outUbFactorH_ = std::min(inputData_.outShape[H_DIM], maxHfactor);
        outUbFactorW_ = inputData_.outShape[W_DIM];
        nLoop_ = inputData_.batches;
        dLoop_ = inputData_.outShape[D_DIM];
        hLoop_ = (inputData_.outShape[H_DIM] + outUbFactorH_ - 1) / outUbFactorH_;
        wLoop_ = 1;
        return;
    }
    // w不全载
    ubFactorN_ = 1;
    outUbFactorD_ = 1;
    outUbFactorH_ = 1;
    outUbFactorW_ = availableUb_;
    nLoop_ = inputData_.batches;
    dLoop_ = inputData_.outShape[D_DIM];
    hLoop_ = inputData_.outShape[H_DIM];
    wLoop_ = (inputData_.outShape[W_DIM] + outUbFactorW_ - 1) / outUbFactorW_;
    return;
}

void Pool3DOneKsizeTiling::DoUBTiling()
{
    int64_t ubThres = BLOCK_SPLIT_THREHOLD / inputData_.dtypeSize / inputData_.channels;
    do {
        DoUBTilingSingle();
        if (nLoop_ * dLoop_ * hLoop_ * wLoop_ >= static_cast<int64_t>(coreNum_) || isZero_) {
            break;
        }
        availableUb_ /= DIGIT_TWO;
    } while (availableUb_ > ubThres);
}

void Pool3DOneKsizeTiling::DoBlockTiling()
{
    int64_t totalLoop = nLoop_ * dLoop_ * hLoop_ * wLoop_;
    blockFactor_ = totalLoop / static_cast<int64_t>(coreNum_) ;
    blockTail_ = totalLoop - blockFactor_ * static_cast<int64_t>(coreNum_) ;
    usedCoreNum_ = blockFactor_ == 0 ? blockTail_ : static_cast<int64_t>(coreNum_) ;
    inUbSize_ = Ops::Base::CeilAlign(ubFactorN_ * outUbFactorD_ * outUbFactorH_ * outUbFactorW_ * inputData_.channels, oneBlockNum_);
}

void Pool3DOneKsizeTiling::CalcDivisor()
{
    if (inputData_.divisorOverride != 0L) {
        divisor_ = inputData_.divisorOverride;
    } else {
        divisor_ = 1;
    }
}

void Pool3DOneKsizeTiling::SetTilingData()
{
    tilingData_.set_inUbSize(inUbSize_);
    tilingData_.set_dataFormat(dataFormat_);
    tilingData_.set_blockFactor(blockFactor_);
    tilingData_.set_blockTail(blockTail_);
    tilingData_.set_ubFactorN(ubFactorN_);
    tilingData_.set_outUbFactorD(outUbFactorD_);
    tilingData_.set_outUbFactorH(outUbFactorH_);
    tilingData_.set_outUbFactorW(outUbFactorW_);
    tilingData_.set_nLoop(nLoop_);
    tilingData_.set_dLoop(dLoop_);
    tilingData_.set_hLoop(hLoop_);
    tilingData_.set_wLoop(wLoop_);
    tilingData_.set_channel(inputData_.channels);
    tilingData_.set_dInDim(inputData_.inputShape[D_DIM]);
    tilingData_.set_hInDim(inputData_.inputShape[H_DIM]);
    tilingData_.set_wInDim(inputData_.inputShape[W_DIM]);
    tilingData_.set_nOutDim(inputData_.batches);
    tilingData_.set_dOutDim(inputData_.outShape[D_DIM]);
    tilingData_.set_hOutDim(inputData_.outShape[H_DIM]);
    tilingData_.set_wOutDim(inputData_.outShape[W_DIM]);
    tilingData_.set_sD(inputData_.stride[D_DIM]);
    tilingData_.set_sH(inputData_.stride[H_DIM]);
    tilingData_.set_sW(inputData_.stride[W_DIM]);
    tilingData_.set_divisor(divisor_);
}

ge::graphStatus Pool3DOneKsizeTiling::DoOpTiling()
{
    DoUBTiling();
    DoBlockTiling();
    CalcDivisor();
    SetTilingData();
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus Pool3DOneKsizeTiling::PostTiling()
{
    context_->SetBlockDim(usedCoreNum_);
    if (tilingData_.GetDataSize() > context_->GetRawTilingData()->GetCapacity()) {
        return ge::GRAPH_FAILED;
    }

    tilingData_.SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(tilingData_.GetDataSize());
    return ge::GRAPH_SUCCESS;
}

void Pool3DOneKsizeTiling::DumpTilingInfo()
{
    std::string str;
    str += " inUbSize:" + std::to_string(tilingData_.get_inUbSize());
    str += " dataFormat:" + std::to_string(tilingData_.get_dataFormat());
    str += " blockFactor:" + std::to_string(tilingData_.get_blockFactor());
    str += " blockTail:" + std::to_string(tilingData_.get_blockTail());
    str += " ubFactorN:" + std::to_string(tilingData_.get_ubFactorN());
    str += " outUbFactorD:" + std::to_string(tilingData_.get_outUbFactorD());
    str += " outUbFactorH:" + std::to_string(tilingData_.get_outUbFactorH());
    str += " outUbFactorW:" + std::to_string(tilingData_.get_outUbFactorW());
    str += " nLoop:" + std::to_string(tilingData_.get_nLoop());
    str += " dLoop:" + std::to_string(tilingData_.get_dLoop());
    str += " hLoop:" + std::to_string(tilingData_.get_hLoop());
    str += " wLoop:" + std::to_string(tilingData_.get_wLoop());
    str += " channel:" + std::to_string(tilingData_.get_channel());
    str += " dInDim:" + std::to_string(tilingData_.get_dInDim());
    str += " hInDim:" + std::to_string(tilingData_.get_hInDim());
    str += " wInDim:" + std::to_string(tilingData_.get_wInDim());
    str += " nOutDim:" + std::to_string(tilingData_.get_nOutDim());
    str += " dOutDim:" + std::to_string(tilingData_.get_dOutDim());
    str += " hOutDim:" + std::to_string(tilingData_.get_hOutDim());
    str += " wOutDim:" + std::to_string(tilingData_.get_wOutDim());
    str += " sD:" + std::to_string(tilingData_.get_sD());
    str += " sH:" + std::to_string(tilingData_.get_sH());
    str += " sW:" + std::to_string(tilingData_.get_sW());
    str += " divisor:" + std::to_string(tilingData_.get_divisor());
    OP_LOGI(context_, "%s", str.c_str());
}

//////////////////////////////// AvgPool3DOneKsizeTiling /////////////////////////////////
ge::graphStatus AvgPool3DOneKsizeTiling::GetPlatformInfo()
{
    return GetAvgPool3DPlatformInfo(context_, ubSize_, coreNum_);
}

ge::graphStatus AvgPool3DOneKsizeTiling::GetShapeAttrsInfo()
{
    return GetAvgPool3DShapeAttrsInfo(context_, inputData_);
}

REGISTER_POOL_TILING_TEMPLATE("AvgPool3D", AvgPool3DOneKsizeTiling, 0);

//////////////////////////////// MaxPool3DOneKsizeTiling /////////////////////////////////
ge::graphStatus MaxPool3DOneKsizeTiling::GetPlatformInfo()
{
    return GetMaxPool3DPlatformInfo(context_, ubSize_, coreNum_);
}

ge::graphStatus MaxPool3DOneKsizeTiling::GetShapeAttrsInfo()
{
    return GetMaxPool3DShapeAttrsInfo(context_, inputData_);
}

REGISTER_OPS_POOL_TILING_TEMPLATE(MaxPool3D, MaxPool3DOneKsizeTiling, 0);

}  // namespace optiling

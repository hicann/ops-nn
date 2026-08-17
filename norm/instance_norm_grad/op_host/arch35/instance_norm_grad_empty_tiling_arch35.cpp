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
 * \file instance_norm_grad_empty_tiling_arch35.cpp
 * \brief Empty-tensor tiling for InstanceNormGrad (tilingKey 500). Only pd_gamma/pd_beta are
 *        zeroed along the C axis; pd_x is empty. Mirrors group_norm_grad_empty_tiling structure.
 */
#include "instance_norm_grad_empty_tiling_arch35.h"
#include "op_host/tiling_templates_registry.h"

namespace optiling {
namespace {
constexpr uint32_t MIN_WORKSPACE_SIZE = 16 * 1024 * 1024;
constexpr uint16_t INPUT_IDX_GAMMA = 4;
constexpr uint32_t EMPTY_TENSOR_KEY = 500;
constexpr uint32_t CONST_TWO = 2;
constexpr uint32_t BYTES_OF_FLOAT = 4;
constexpr uint32_t BUFFER_NUM = 2;
constexpr int64_t MAX_CORE_COLS = 8000;
} // namespace

ge::graphStatus InstanceNormGradEmptyTiling::GetPlatformInfo()
{
    auto platformInfo = context_->GetPlatformInfo();
    if (platformInfo != nullptr) {
        auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
        aivCoreNum_ = ascendcPlatform.GetCoreNumAiv();
        uint64_t ubSizePlatform = 0;
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatform);
        ubSize_ = ubSizePlatform;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus InstanceNormGradEmptyTiling::GetShapeAttrsInfo()
{
    auto gammaShapePtr = context_->GetInputShape(INPUT_IDX_GAMMA);
    OP_CHECK_NULL_WITH_CONTEXT(context_, gammaShapePtr);
    auto gammaShape = gammaShapePtr->GetStorageShape();
    cols_ = 1;
    for (size_t i = 0; i < gammaShape.GetDimNum(); i++) {
        cols_ *= gammaShape.GetDim(i);
    }
    return ge::GRAPH_SUCCESS;
}

bool InstanceNormGradEmptyTiling::IsCapable() { return true; }

uint64_t InstanceNormGradEmptyTiling::NearestLowerPowerOfTwo(int32_t tmp)
{
    int64_t power = 0;
    uint64_t powerOfTwoValue = 0;
    while (power <= tmp) {
        powerOfTwoValue += 1;
        power = std::pow(CONST_TWO, powerOfTwoValue);
    }
    return powerOfTwoValue - 1;
}

ge::graphStatus InstanceNormGradEmptyTiling::CalcuTilingData()
{
    if (ubSize_ >= BUFFER_NUM * (colsPerCoreDG_ * BYTES_OF_FLOAT)) {
        coreUbBlockCount_ = 0;
        tailUbCols_ = colsPerCoreDG_;
        lastCoreBlockCount_ = 0;
        lastCoreTailUbCols_ = colsLastCoreDG_;
    } else {
        int64_t maxRowsNumDG = ubSize_ / (BUFFER_NUM * BYTES_OF_FLOAT);
        colsPerUBDG_ = std::pow(CONST_TWO, NearestLowerPowerOfTwo(maxRowsNumDG));
        OP_CHECK_IF(maxRowsNumDG <= 0, OP_LOGE(context_->GetNodeName(), "maxRowsNumDG is neg: %ld.", maxRowsNumDG),
                    return ge::GRAPH_FAILED);
        if (colsPerUBDG_ == 0) {
            OP_LOGE(context_->GetNodeName(), "colsPerUBDG_ is zero, cannot perform division.");
            return ge::GRAPH_FAILED;
        }
        coreUbBlockCount_ = (colsPerCoreDG_ + colsPerUBDG_ - 1) / colsPerUBDG_ - 1;
        tailUbCols_ = colsPerCoreDG_ - colsPerUBDG_ * coreUbBlockCount_;
        if (colsPerUBDG_ > colsLastCoreDG_) {
            lastCoreBlockCount_ = 0;
            lastCoreTailUbCols_ = colsLastCoreDG_;
        } else {
            lastCoreBlockCount_ = (colsLastCoreDG_ + colsPerUBDG_ - 1) / colsPerUBDG_ - 1;
            lastCoreTailUbCols_ = colsLastCoreDG_ - colsPerUBDG_ * lastCoreBlockCount_;
        }
    }
    return ge::GRAPH_SUCCESS;
}

void InstanceNormGradEmptyTiling::CalcUsedCoreNumGamma()
{
    if (cols_ == 0) {
        // C 轴为 0:pd_gamma/pd_beta 同样为空,没有任何元素要写回。让所有核在 Init 里就退出,
        // 否则 colsPerUB_ = 0 会走到 InitBuffer(0)/Duplicate(count=0)/DataCopyPad(blockLen=0)。
        usedCoreNumDG_ = 0;
        return;
    }
    if (cols_ <= MAX_CORE_COLS) {
        usedCoreNumDG_ = 1;
        colsPerCoreDG_ = cols_;
        colsLastCoreDG_ = cols_;
        coreUbBlockCount_ = 0;
        tailUbCols_ = colsPerCoreDG_;
        colsPerUBDG_ = colsPerCoreDG_;
        lastCoreBlockCount_ = 0;
        lastCoreTailUbCols_ = colsLastCoreDG_;
    } else {
        usedCoreNumDG_ = aivCoreNum_;
        colsPerCoreDG_ = Ops::Base::CeilDiv(cols_, usedCoreNumDG_);
        colsLastCoreDG_ = cols_ - colsPerCoreDG_ * (usedCoreNumDG_ - 1);
        if (CalcuTilingData() == ge::GRAPH_FAILED) {
            OP_LOGE(context_->GetNodeName(), "CalcuTilingData failed.");
        }
    }
}

ge::graphStatus InstanceNormGradEmptyTiling::DoOpTiling()
{
    CalcUsedCoreNumGamma();
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus InstanceNormGradEmptyTiling::DoLibApiTiling() { return ge::GRAPH_SUCCESS; }

uint64_t InstanceNormGradEmptyTiling::GetTilingKey() const { return EMPTY_TENSOR_KEY; }

ge::graphStatus InstanceNormGradEmptyTiling::GetWorkspaceSize()
{
    workSpaceSize_ = MIN_WORKSPACE_SIZE;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus InstanceNormGradEmptyTiling::PostTiling()
{
    SetTilingData();
    context_->SetBlockDim(aivCoreNum_);
    size_t* currentWorkspace = context_->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, currentWorkspace);
    currentWorkspace[0] = workSpaceSize_;
    tilingData.SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(tilingData.GetDataSize());
    return ge::GRAPH_SUCCESS;
}

void InstanceNormGradEmptyTiling::SetTilingData()
{
    tilingData.set_colsPerUBDG(colsPerUBDG_);
    tilingData.set_tailUbCols(tailUbCols_);
    tilingData.set_coreUbBlockCount(coreUbBlockCount_);
    tilingData.set_lastCoreBlockCount(lastCoreBlockCount_);
    tilingData.set_lastCoreTailUbCols(lastCoreTailUbCols_);
    tilingData.set_usedCoreNumDG(usedCoreNumDG_);
    tilingData.set_colsPerCoreDG(colsPerCoreDG_);
    tilingData.set_colsLastCoreDG(colsLastCoreDG_);
}
} // namespace optiling

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
 * \file add_rms_norm_dynamic_quant_tiling_arch35.cpp
 * \brief
 */

#include "add_rms_norm_dynamic_quant_tiling_arch35.h"
#include "norm/norm_common/op_host/norm_tiling_check_common.h"

namespace optiling {
using namespace NormCheck;

static constexpr uint32_t TWO = 2;
static constexpr uint64_t BUFFER_NUM = 2;
static constexpr uint64_t FLOATBYTESIZE = 4;
static constexpr uint32_t MIN_WORKSPACE_SIZE = 16 * 1024 * 1024;

static constexpr uint64_t PER_CORE_MAX_SIZE = 4096;

const std::string OP_NAME = "AddRmsNormDynamicQuant";

ge::graphStatus AddRmsNormDynamicQuantEmptyTiling::SetInputParams()
{
    OP_LOGD(nodeName_.c_str(), "Enter AddRmsNormDynamicQuantEmptyTiling SetInputParams.");
    // Set input dim
    const gert::StorageShape* x1Shape = context_->GetInputShape(X1_INDEX);
    auto x1InputShape = x1Shape->GetStorageShape();
    const gert::StorageShape* gammaShape = context_->GetInputShape(GAMMA_INDEX);
    auto gammaInputShape = gammaShape->GetStorageShape();

    size_t x1DimNum = x1InputShape.GetDimNum();
    size_t gammaDimNum = gammaInputShape.GetDimNum();
    uint64_t numM = 1;
    uint64_t numN = 0;
    for (size_t i = 0; i < x1DimNum - gammaDimNum; i++) {
        numM *= x1InputShape.GetDim(i);
    }

    numM_ = numM;
    numN_ = numN;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AddRmsNormDynamicQuantEmptyTiling::GetPlatformInfo()
{
    OP_LOGD(nodeName_.c_str(), "Enter AddRmsNormDynamicQuantEmptyTiling GetPlatformInfo.");
    auto platformInfo = context_->GetPlatformInfo();
    if (platformInfo != nullptr) {
        auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
        aivCoreNum_ = ascendcPlatform.GetCoreNumAiv();
        uint64_t ubSizePlatform;
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatform);
        ubSize_ = ubSizePlatform;
    }
    return ge::GRAPH_SUCCESS;
}

bool AddRmsNormDynamicQuantEmptyTiling::IsCapable() { return true; }

ge::graphStatus AddRmsNormDynamicQuantEmptyTiling::DoLibApiTiling() { return ge::GRAPH_SUCCESS; }

ge::graphStatus AddRmsNormDynamicQuantEmptyTiling::GetWorkspaceSize()
{
    workspaceSize_ = MIN_WORKSPACE_SIZE;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AddRmsNormDynamicQuantEmptyTiling::PostTiling()
{
    context_->SetBlockDim(usedCoreNum_);
    size_t* currentWorkspace = context_->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, currentWorkspace);
    currentWorkspace[0] = workspaceSize_;
    auto rawTilingData = context_->GetRawTilingData();
    OP_CHECK_IF(sizeof(tilingData_) > rawTilingData->GetCapacity(),
                OP_LOGE(context_->GetNodeName(), "actual tiling data size %zu > context tiling data size %zu",
                        sizeof(tilingData_), rawTilingData->GetCapacity()),
                return ge::GRAPH_FAILED);
    auto capSize = rawTilingData->GetCapacity();
    void* ptrData = rawTilingData->GetData();
    OP_CHECK_NULL_WITH_CONTEXT(context_, ptrData);
    void* ptrStruct = static_cast<void*>(&tilingData_);
    OP_CHECK_NULL_WITH_CONTEXT(context_, ptrStruct);
    OP_CHECK_IF(memcpy_s(ptrData, capSize, ptrStruct, sizeof(tilingData_)) != 0,
                OP_LOGE(context_->GetNodeName(), "Set tiling data is failed!"), return ge::GRAPH_FAILED);
    rawTilingData->SetDataSize(sizeof(tilingData_));
    return ge::GRAPH_SUCCESS;
}

uint64_t AddRmsNormDynamicQuantEmptyTiling::GetTilingKey() const
{
    AddRmsNormDynamicQuantTilingKey tilingKey;
    tilingKey.SetComputeMode(ComputeMode::REDUCE_EMPTY, Y3Mode::NO_Y3, Y4Mode::NO_Y4);
    return tilingKey.GetTilingKey();
}

uint64_t AddRmsNormDynamicQuantEmptyTiling::NearestLowerPowerOfTwo(uint64_t tmp)
{
    uint64_t power = 0;
    uint64_t powerofTwoValue = 0;
    while (power <= tmp) {
        powerofTwoValue += 1;
        power = std::pow(TWO, powerofTwoValue);
    }
    return powerofTwoValue - 1;
}

void AddRmsNormDynamicQuantEmptyTiling::CalcTilingData()
{
    if (ubSize_ >= (BUFFER_NUM * (mPerCore_ * FLOATBYTESIZE))) {
        coreUbBlockCount_ = 0;
        mTailUb_ = mPerCore_;
        lastCoreBlockCount_ = 0;
        mlastCoreTailUb_ = mLastCore_;
    } else {
        uint64_t maxRowsNum_ = ubSize_ / (BUFFER_NUM * FLOATBYTESIZE);
        mPerUB_ = std::pow(TWO, NearestLowerPowerOfTwo(maxRowsNum_));
        coreUbBlockCount_ = (mPerCore_ + mPerUB_ - 1) / mPerUB_ - 1;
        mTailUb_ = mPerCore_ - mPerUB_ * coreUbBlockCount_;
        if (mPerUB_ > mLastCore_) {
            lastCoreBlockCount_ = 0;
            mlastCoreTailUb_ = mLastCore_;
        } else {
            lastCoreBlockCount_ = (mLastCore_ + mPerUB_ - 1) / mPerUB_ - 1;
            mlastCoreTailUb_ = mLastCore_ - mPerUB_ * lastCoreBlockCount_;
        }
    }
}

void AddRmsNormDynamicQuantEmptyTiling::CalcUsedCoreNum()
{
    if (numM_ <= PER_CORE_MAX_SIZE) {
        usedCoreNum_ = 1;
        mPerCore_ = numM_;
        mLastCore_ = numM_;
        coreUbBlockCount_ = 0;
        mTailUb_ = mPerCore_;
        mPerUB_ = mPerCore_;
        lastCoreBlockCount_ = 0;
        mlastCoreTailUb_ = mLastCore_;
    } else {
        usedCoreNum_ = aivCoreNum_;
        mPerCore_ = Ops::Base::CeilDiv(numM_, usedCoreNum_);
        mPerUB_ = mPerCore_;
        mLastCore_ = numM_ - mPerCore_ * (usedCoreNum_ - 1);
        CalcTilingData();
    }
}

void AddRmsNormDynamicQuantEmptyTiling::LogTilingResult() { OP_LOGI(OP_NAME, "numN: %ld, numM: %ld", numN_, numM_); }

ge::graphStatus AddRmsNormDynamicQuantEmptyTiling::DoOpTiling()
{
    // split cores
    CalcUsedCoreNum();

    tilingData_.mPerUB = mPerUB_;
    tilingData_.mTailUb = mTailUb_;
    tilingData_.coreUbBlockCount = coreUbBlockCount_;
    tilingData_.lastCoreBlockCount = lastCoreBlockCount_;
    tilingData_.mlastCoreTailUb = mlastCoreTailUb_;
    tilingData_.ubSize = ubSize_;
    tilingData_.numM = numM_;
    tilingData_.usedCoreNum = usedCoreNum_;
    tilingData_.mPerCore = mPerCore_;
    tilingData_.mLastCore = mLastCore_;
    tilingData_.hasSmoothScale1 = hasSmoothScale1_;
    tilingData_.hasSmoothScale2 = hasSmoothScale2_;
    tilingData_.outQuant1Flag = outQuant1Flag_;
    tilingData_.outQuant2Flag = outQuant2Flag_;
    LogTilingResult();
    return ge::GRAPH_SUCCESS;
}

} // namespace optiling

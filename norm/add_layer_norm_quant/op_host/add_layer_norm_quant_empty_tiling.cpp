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
 * \file add_layer_norm_quant_empty_tiling.cpp
 * \brief
 */
#include "op_common/op_host/util/platform_util.h"
#include "util/math_util.h"
#include "add_layer_norm_quant_tiling.h"

namespace optiling {
static constexpr uint32_t CONST_TWO = 2;
static constexpr uint32_t BYTES_OF_FLOAT = 4;
static constexpr uint32_t KERNEL_BUFFER_NUM = 2;
static constexpr uint32_t MIN_WORKSPACE_SIZE = 16 * 1024 * 1024;
static constexpr int64_t MAX_SIZE_PER_CORE = 4096;
static constexpr int64_t EMPTY_TENSOR_KEY = 9;

ge::graphStatus AddLayerNormQuantEmptyTiling::GetPlatformInfo()
{
    auto platformInfo = context_->GetPlatformInfo(); // 判断条件修改
    if (platformInfo != nullptr) {
        auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
        aivCoreNum_ = ascendcPlatform.GetCoreNumAiv();
        uint64_t ubSizePlatform;
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatform);
        ubSize_ = ubSizePlatform;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AddLayerNormQuantEmptyTiling::GetAttrs()
{
    const gert::RuntimeAttrs* attrs = context_->GetAttrs();

    const char* qms = attrs->GetAttrPointer<char>(QUANT_MODE_IDX);
    OP_CHECK_IF(nullptr == qms, OP_LOGE("GetAttrs", "Get required attr quantMode failed, tiling failed"), return false);
    std::string quantModeStr = qms;
    isDyn_ = (quantModeStr == "dynamic");
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AddLayerNormQuantEmptyTiling::CheckShapeAllPositive(gert::Shape& shape)
{
    for (size_t i = 0; i < shape.GetDimNum(); i++) {
        OP_CHECK_IF(
            shape.GetDim(i) < 0,
            OP_LOGE(
                context_->GetNodeName(), "Dim %lu of input should be positive, but actual %ld.", i, shape.GetDim(i)),
            return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AddLayerNormQuantEmptyTiling::DoTiling()
{
    auto ret = GetShapeAttrsInfo();
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }

    ret = GetAttrs();
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }

    ret = GetPlatformInfo();
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }

    ret = DoOpTiling();
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }

    ret = GetWorkspaceSize();
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }
    ret = PostTiling();
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }
    context_->SetTilingKey(GetTilingKey());
    
    return ge::GRAPH_SUCCESS;
}

void AddLayerNormQuantEmptyTiling::CalcRowsAndCols(gert::Shape& xShape, gert::Shape& gammaShape)
{
    rows_ = 1;
    cols_ = 0;
    for (size_t i = 0; i < xShape.GetDimNum() - gammaShape.GetDimNum(); i++) {
        rows_ *= xShape.GetDim(i);
    }
}

ge::graphStatus AddLayerNormQuantEmptyTiling::CheckInputsShape()
{
    // Check Shape Not NULL
    const gert::StorageShape* x1Shape = this->context_->GetInputShape(X1_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(this->context_, x1Shape);
    auto x1StorageShape = x1Shape->GetStorageShape();

    const gert::StorageShape* x2Shape = this->context_->GetInputShape(X2_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(this->context_, x2Shape);
    auto x2StorageShape = x2Shape->GetStorageShape();

    const gert::StorageShape* gammaShape = this->context_->GetInputShape(GAMMA_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(this->context_, gammaShape);
    auto gammaStorageShape = gammaShape->GetStorageShape();
   
    const gert::StorageShape* betaShape = this->context_->GetInputShape(BETA_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(this->context_, betaShape);
    auto betaStorageShape = betaShape->GetStorageShape();
 
    const gert::StorageShape* y1Shape = this->context_->GetOutputShape(Y1_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(this->context_, y1Shape);
    auto y1StorageShape = y1Shape->GetStorageShape();

    const gert::StorageShape* xShape = this->context_->GetOutputShape(X_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(this->context_, xShape);
    auto xStorageShape = xShape->GetStorageShape();

    size_t elewiseDimNum = x1StorageShape.GetDimNum();
    size_t weightDimNum = gammaStorageShape.GetDimNum();
    OP_CHECK_IF(
        (0 == elewiseDimNum || 0 == weightDimNum),
        OP_LOGW(this->context_->GetNodeName(), "Got x1/gamma is zero dim tensor, tiling FAILED."), return ge::GRAPH_FAILED);

    bool elewiseShapeEqual = ((*x1Shape) == (*x2Shape)) && ((*x1Shape) == (*xShape)) && ((*x1Shape) == (*y1Shape));
    bool weightShapeEqual = ((*gammaShape) == (*betaShape));
    OP_CHECK_IF(
        !(elewiseShapeEqual && weightShapeEqual),
        OP_LOGW(
            this->context_->GetNodeName(),
            "Got x1/x2/y1/x shape not equat OR gamma/beta Shape not equal, tiling FAILED."),
        return ge::GRAPH_FAILED);

    // calculate the number of valid rows.
    CalcRowsAndCols(xStorageShape, gammaStorageShape);

    return ge::GRAPH_SUCCESS;
}


ge::graphStatus AddLayerNormQuantEmptyTiling::GetShapeAttrsInfo()
{
    if (context_ == nullptr) {
        OP_LOGE(context_->GetNodeName(), "Tiling context is null");
        return ge::GRAPH_FAILED;
    }

    // check inputs shape
    if (CheckInputsShape() != ge::GRAPH_SUCCESS) {
        OP_LOGE(context_->GetNodeName(), "Inputs shape invalid.");
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

uint64_t AddLayerNormQuantEmptyTiling::GetTilingKey() const
{
    return tilingKey_;
}

ge::graphStatus AddLayerNormQuantEmptyTiling::PostTiling()
{
    context_->SetBlockDim(usedCoreNum_);
    size_t* currentWorkspace = context_->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, currentWorkspace);
    currentWorkspace[0] = workspaceSize_;
    tilingData_.SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(tilingData_.GetDataSize());
    OP_LOGD(context_->GetNodeName(), "Tiling end.");
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AddLayerNormQuantEmptyTiling::GetWorkspaceSize()
{
    workspaceSize_ = MIN_WORKSPACE_SIZE;
    tilingData_.set_workspaceSize(workspaceSize_);
    return ge::GRAPH_SUCCESS;
}

uint64_t AddLayerNormQuantEmptyTiling::NearestLowerPowerOfTwo(int32_t tmp) {
    int64_t power = 0;
    uint64_t powerOfTwoValue = 0;
    while (power <= tmp) {
        powerOfTwoValue += 1;
        power = std::pow(CONST_TWO, powerOfTwoValue);
    }
    return powerOfTwoValue - 1;
}

ge::graphStatus AddLayerNormQuantEmptyTiling::CalcuTilingData() {
    if(static_cast<int64_t>(ubSize_) >= KERNEL_BUFFER_NUM * (rowsPerCore_ * BYTES_OF_FLOAT)){
        numBlocks_ = 0;
        lastBlockSize_ = rowsPerCore_;
        numBlocksLastCore_ = 0;
        sizeLastCore_ = rowsLastCore_;
    } else {
        uint64_t maxOutputSize = ubSize_ / (KERNEL_BUFFER_NUM * BYTES_OF_FLOAT);
        OP_CHECK_IF(maxOutputSize < 0, OP_LOGE(context_->GetNodeName(), 
            "The maxOutputSize size is neg: %ld.", maxOutputSize), return ge::GRAPH_FAILED);
        blockSize_ = std::pow(CONST_TWO, NearestLowerPowerOfTwo(maxOutputSize));
        numBlocks_ = (rowsPerCore_ + blockSize_ - 1) / blockSize_ - 1;
        lastBlockSize_ = rowsPerCore_ - blockSize_ * numBlocks_;
        if (blockSize_ < rowsLastCore_) {
            numBlocksLastCore_ = (rowsLastCore_ + blockSize_ - 1) / blockSize_ - 1;
            sizeLastCore_ = rowsLastCore_ - blockSize_ * numBlocksLastCore_;
        } else {
            numBlocksLastCore_ = 0;
            sizeLastCore_ = rowsLastCore_;
        }
    }
    return ge::GRAPH_SUCCESS;
}

void AddLayerNormQuantEmptyTiling::CalcUsedCoreNums() {
    if (this->rows_ <= MAX_SIZE_PER_CORE) {
        // 单核计算
        usedCoreNum_ = 1;
        rowsPerCore_ = rows_;
        rowsLastCore_ = rows_;
        numBlocks_ = 0;
        lastBlockSize_ = rowsPerCore_;
        blockSize_ = rowsLastCore_;
        numBlocksLastCore_ = 0;
        sizeLastCore_ = rowsLastCore_;
    } else {
        // 多核计算
        ge::graphStatus ret = ge::GRAPH_SUCCESS;
        usedCoreNum_ = aivCoreNum_;
        rowsPerCore_ = Ops::Base::CeilDiv(rows_, usedCoreNum_);
        blockSize_ = rowsPerCore_;
        rowsLastCore_ = rows_ - rowsPerCore_ * (usedCoreNum_ - 1);
        ret = CalcuTilingData();
    }
}

void AddLayerNormQuantEmptyTiling::LogTilingResult()
{
    OP_LOGW(context_->GetNodeName(), "rows: %ld, cols_: %ld", rows_, cols_);
}

ge::graphStatus AddLayerNormQuantEmptyTiling::DoOpTiling()
{
    tilingKey_ = EMPTY_TENSOR_KEY;
    // split cores
    CalcUsedCoreNums();
    
    auto scale2 = context_->GetOptionalInputTensor(SCALE2_IDX);
    if(scale2 != nullptr) {
        isDualQuant_ = 1;
    }
    tilingData_.set_isDualQuant(isDualQuant_);
    tilingData_.set_isDyn(isDyn_);
    tilingData_.set_rows(rows_);
    tilingData_.set_cols(cols_);
    tilingData_.set_usedCoreNum(usedCoreNum_);
    tilingData_.set_rowsPerCore(rowsPerCore_);
    tilingData_.set_rowsLastCore(rowsLastCore_);
    tilingData_.set_ubSize(ubSize_);
    tilingData_.set_blockSize(blockSize_);
    tilingData_.set_numBlocks(numBlocks_);
    tilingData_.set_lastBlockSize(lastBlockSize_);
    tilingData_.set_numBlocksLastCore(numBlocksLastCore_);
    tilingData_.set_sizeLastCore(sizeLastCore_);
    LogTilingResult();
    return ge::GRAPH_SUCCESS;
}

}  // namespace optiling
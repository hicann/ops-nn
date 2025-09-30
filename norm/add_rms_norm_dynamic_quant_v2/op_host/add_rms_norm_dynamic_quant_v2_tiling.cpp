/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file add_rms_norm_dynamic_quant_v2_tiling.cpp
 * \brief
 */
#include "add_rms_norm_dynamic_quant_v2_tiling.h"

namespace optiling {

constexpr int X1_IDX = 0;
constexpr int X2_IDX = 1;
constexpr int GAMMA_IDX = 2;
constexpr int SMOOTH1_IDX = 3;
constexpr int SMOOTH2_IDX = 4;

constexpr int Y1_IDX = 0;
constexpr int Y2_IDX = 1;
constexpr int Y3_IDX = 2;
constexpr int Y4_IDX = 3;
constexpr int X_IDX = 4;
constexpr int SCALE1_IDX = 5;
constexpr int SCALE2_IDX = 6;

constexpr int EPS_IDX = 0;

constexpr uint64_t USR_WORKSPACE_SIZE_910B = 1;

constexpr uint32_t SIZEOF_B16 = 2;
constexpr uint32_t NUM_RMSNORM_STATS = 2;
constexpr uint32_t BLOCK_SIZE = 32;
constexpr uint64_t ROW_FACTOR = 128;
constexpr uint64_t UB_RESERVED_BYTE = 768;
constexpr uint32_t MAX_ROW_STEP = 16;

constexpr uint32_t UB_TILING_POLICY_NORMAL = 1;
constexpr uint32_t UB_TILING_POLICY_SINGLE_ROW = 2;
constexpr uint32_t UB_TILING_POLICY_SLICE_D = 3;

constexpr uint32_t SLICE_COL_LEN = 8864;

bool CheckOptionalShapeExistingV2(const gert::StorageShape* smoothShape)
{
    OP_CHECK_IF(
        nullptr == smoothShape, OP_LOGD("CheckOptionalShapeExistingV2", "Get nullptr smoothShape"), return false);
    int64_t smoothShapeSize = smoothShape->GetOriginShape().GetShapeSize();
    OP_CHECK_IF(
        (smoothShapeSize <= 0),OP_LOGD("CheckOptionalShapeExistingV2", "Get empty smoothShape"), return false);
    return true;
}

void AddRmsNormDynamicQuantV2TilingHelper::SetTilingDataAndTilingKeyAndWorkSpace(
    AddRmsNormDynamicQuantV2TilingData* tiling)
{
    context_->SetBlockDim(this->useCore_);
    tiling->set_useCore(this->useCore_);
    tiling->set_numFirstDim(this->numFirstDim_);
    tiling->set_numLastDim(this->numLastDim_);
    tiling->set_numLastDimAligned(this->numLastDimAligned_);
    tiling->set_firstDimPerCore(this->firstDimPerCore_);
    tiling->set_firstDimPerCoreTail(this->firstDimPerCoreTail_);
    tiling->set_firstDimPerLoop(this->firstDimPerLoop_);
    tiling->set_lastDimSliceLen(this->lastDimSliceLen_);
    tiling->set_lastDimLoopNum(this->lastDimLoopNum_);
    tiling->set_lastDimSliceLenTail(this->lastDimSliceLenTail_);
    tiling->set_smoothNum(this->smoothNum_);
    tiling->set_epsilon(this->eps_);
    tiling->set_avgFactor(this->avgFactor_);

    uint32_t tilingKey = 0;
    size_t usrSize = USR_WORKSPACE_SIZE_910B;

    if (this->ubTilingPolicy_ == UB_TILING_POLICY::NORMAL) {
        tilingKey += UB_TILING_POLICY_NORMAL;
    } else if (this->ubTilingPolicy_ == UB_TILING_POLICY::SINGLE_ROW) {
        tilingKey += UB_TILING_POLICY_SINGLE_ROW;
    } else {
        tilingKey += UB_TILING_POLICY_SLICE_D;
        size_t workspaceRowsNum = (this->smoothNum_ == 0) ? 1 : this->smoothNum_;
        usrSize = this->useCore_ * this->numLastDim_ * sizeof(float) * workspaceRowsNum;
    }

    context_->SetTilingKey(tilingKey);

    tiling->SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(tiling->GetDataSize());

    // set workspace
    size_t* currentWorkspace = context_->GetWorkspaceSizes(1);
    currentWorkspace[0] = this->sysWorkspaceSize_ + usrSize;

    OP_LOGI(
        "SetTilingDataAndTilingKeyAndWorkSpace", "Tilingdata useCore_: %lu, smoothNum_: %u", this->useCore_,
        this->smoothNum_);
    OP_LOGI(
        "SetTilingDataAndTilingKeyAndWorkSpace", "Tilingdata N: %lu, D:%lu, DAligned: %lu", numFirstDim_, numLastDim_,
        numLastDimAligned_);
    OP_LOGI(
        "SetTilingDataAndTilingKeyAndWorkSpace", "Tilingdata firstDimPerCore_: %lu, firstDimPerCoreTail_: %lu",
        firstDimPerCore_, firstDimPerCoreTail_);
    OP_LOGI("SetTilingDataAndTilingKeyAndWorkSpace", "Tilingdata firstDimPerLoop_: %lu", firstDimPerLoop_);
    OP_LOGI(
        "SetTilingDataAndTilingKeyAndWorkSpace",
        "Tilingdata lastDimSliceLen_: %lu, lastDimLoopNum_: %lu, lastDimSliceLenTail_: %lu", lastDimSliceLen_,
        lastDimLoopNum_, lastDimSliceLenTail_);
    OP_LOGI("SetTilingDataAndTilingKeyAndWorkSpace", "Tilingdata eps_: %f, avgFactor_: %f", eps_, avgFactor_);
    OP_LOGI(
        "SetTilingDataAndTilingKeyAndWorkSpace", "Tilingdata tilingKey = %u, usr Workspace: %zu", tilingKey, usrSize);
}

bool AddRmsNormDynamicQuantV2TilingHelper::DoTiling()
{
    OP_CHECK_IF(
        (nullptr == context_), OP_LOGE("AddRmsNormDynamicQuantV2Tiling", "Helper context_ get nullptr, return failed."),
        return false);
    OP_CHECK_IF(!GetBaseInfo(), OP_LOGE(context_->GetNodeName(), "GetBaseInfo falied, return false"), return false);
    OP_CHECK_IF(!GetShapeInfo(), OP_LOGE(context_->GetNodeName(), "GetShapeInfo falied, return false"), return false);
    OP_CHECK_IF(!DoBlockTiling(), OP_LOGE(context_->GetNodeName(), "DoBlockTiling falied, return false"), return false);
    OP_CHECK_IF(!DoUbTiling(), OP_LOGE(context_->GetNodeName(), "DoUbTiling falied, return false"), return false);
    return true;
}

bool AddRmsNormDynamicQuantV2TilingHelper::DoBlockTiling()
{
    // Block Tiling, Cut N
    this->firstDimPerCore_ = Ops::Base::CeilDiv(this->numFirstDim_, this->socCoreNums_);
    this->useCore_ = Ops::Base::CeilDiv(this->numFirstDim_, this->firstDimPerCore_);
    this->firstDimPerCore_ = Ops::Base::CeilDiv(this->numFirstDim_, this->useCore_);
    this->firstDimPerCoreTail_ = this->numFirstDim_ - this->firstDimPerCore_ * (this->useCore_ - 1);
    OP_LOGI(
        "DoBlockTiling", "BlockTiling Factor: useCore_: %lu, firstDimPerCore_: %lu, firstDimPerCoreTail_: %lu",
        this->useCore_, this->firstDimPerCore_, this->firstDimPerCoreTail_);
    return true;
}

bool AddRmsNormDynamicQuantV2TilingHelper::GetBaseInfo()
{
    auto platformInfo = context_->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context_, platformInfo);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    this->socCoreNums_ = ascendcPlatform.GetCoreNumAiv();
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, this->ubSize_);
    this->sysWorkspaceSize_ = ascendcPlatform.GetLibApiWorkSpaceSize();

    auto attrs = context_->GetAttrs();
    OP_CHECK_IF(nullptr == attrs, OP_LOGE(context_->GetNodeName(), "Get attrs nullptr, return false."), return false);

    const float* epsPtr = attrs->GetFloat(EPS_IDX);
    if (epsPtr != nullptr) {
        this->eps_ = *epsPtr;
    }
    OP_CHECK_IF(
        this->eps_ <= 0, OP_LOGE(context_->GetNodeName(), "Epsilon less or equal than zero, please check."),
        return false);
    OP_CHECK_IF(
        (this->ubSize_ <= 0), OP_LOGE(context_->GetNodeName(), "ubSize less or equal than zero, please check."),
        return false);
    OP_CHECK_IF(
        (this->socCoreNums_ <= 0),
        OP_LOGE(context_->GetNodeName(), "socCoreNums_ less or equal than zero, please check."), return false);

    OP_LOGI(
        "GetBaseInfo", "socCoreNum: %lu, ubSize: %lu, sysWorkspaceSize: %lu, epsilon: %f", this->socCoreNums_,
        this->ubSize_, this->sysWorkspaceSize_, this->eps_);
    return true;
}

bool AddRmsNormDynamicQuantV2TilingHelper::GetShapeInfo()
{
    OP_CHECK_IF(
        CheckInputOutputShape() == false, OP_LOGE(context_->GetNodeName(), "Check tensor shape failed."), return false);
    // no fp32 allowed here
    this->dtSize_ = SIZEOF_B16;
    auto xShape = context_->GetInputShape(X1_IDX)->GetStorageShape();
    auto gammaShape = context_->GetInputShape(GAMMA_IDX)->GetStorageShape();
    size_t xDimNum = xShape.GetDimNum();
    size_t gammaDimNum = gammaShape.GetDimNum();

    const gert::StorageShape* smooth1Shape = this->context_->GetOptionalInputShape(SMOOTH1_IDX);
    const gert::StorageShape* smooth2Shape = this->context_->GetOptionalInputShape(SMOOTH2_IDX);
    bool smooth1Exist = CheckOptionalShapeExistingV2(smooth1Shape);
    bool smooth2Exist = CheckOptionalShapeExistingV2(smooth2Shape);

    this->smoothNum_ += (smooth1Exist) ? 1 : 0;
    this->smoothNum_ += (smooth2Exist) ? 1 : 0;

    OP_CHECK_IF(
        (!smooth1Exist) && (smooth2Exist),
        OP_LOGE(context_->GetNodeName(), "Smooth2 exist but smooth1 not exist, bad input."), return false);
    OP_CHECK_IF(
        (smooth1Exist && smooth1Shape->GetStorageShape() != gammaShape),
        OP_LOGE(context_->GetNodeName(), "GammaShape is not same to smooth1Shape."), return false);
    OP_CHECK_IF(
        (smooth2Exist && smooth2Shape->GetStorageShape() != gammaShape),
        OP_LOGE(context_->GetNodeName(), "GammaShape is not same to smooth2Shape."), return false);

    uint64_t numRow = 1;
    uint64_t numCol = 1;
    for (size_t i = 0; i < xDimNum - gammaDimNum; i++) {
        numRow *= xShape.GetDim(i);
    }
    for (size_t i = 0; i < gammaDimNum; i++) {
        numCol *= gammaShape.GetDim(i);
    }
    this->numFirstDim_ = numRow;
    this->numLastDim_ = numCol;
    this->numLastDimAligned_ = Ops::Base::CeilDiv(numCol, static_cast<uint64_t>(BLOCK_SIZE)) * BLOCK_SIZE;
    this->avgFactor_ = 1.0 / ((float)this->numLastDim_);

    OP_LOGI("GetShapeInfo", "[N, D] = [%lu, %lu]", this->numFirstDim_, this->numLastDim_);
    OP_LOGI("GetShapeInfo", "dtSize_=%lu, avgFactor_=%f", this->dtSize_, this->avgFactor_);
    return true;
}

bool AddRmsNormDynamicQuantV2TilingHelper::DoUbTiling()
{
    OP_CHECK_IF(CheckUbNormalTiling(), OP_LOGI(context_->GetNodeName(), "Ub Tiling: Normal."), return true);
    OP_CHECK_IF(CheckUbSingleRowTiling(), OP_LOGI(context_->GetNodeName(), "Ub Tiling: SingleRow."), return true);
    OP_CHECK_IF(CheckUbSliceDTiling(), OP_LOGI(context_->GetNodeName(), "Ub Tiling: SliceD."), return true);
    return false;
}

bool AddRmsNormDynamicQuantV2TilingHelper::CheckUbNormalTiling()
{
    // 3 weights tensor required.
    int64_t ubConst = this->numLastDimAligned_ * this->dtSize_ * 3 + UB_RESERVED_BYTE;
    int64_t ubAvaliable = this->ubSize_ - ubConst;
    // 2 rows for tmpBuffer.
    int64_t coexistingRowsNum = 2 * (this->dtSize_) + 2 * (this->dtSize_) + 1 * sizeof(float) + 1 * sizeof(float);
    // 2 buffers for out_scale.
    int64_t rowCommons = coexistingRowsNum * this->numLastDimAligned_ + 2 * sizeof(float);
    int64_t rowStep = ubAvaliable / rowCommons;
    bool ret = (rowStep >= 1);
    OP_LOGI(
        this->context_->GetNodeName(),
        "CheckUbNormalTiling, ret:%d, ubConst: %ld, ubAvaliable=%ld, coexistingRowsNum: %ld, rowStep: %ld, "
        "rowCommons: %ld",
        ret, ubConst, ubAvaliable, coexistingRowsNum, rowStep, rowCommons);
    if (ret) {
        // No mutilN now. max RowStep = 16
        this->firstDimPerLoop_ = (rowStep <= MAX_ROW_STEP) ? rowStep : MAX_ROW_STEP;
        this->lastDimSliceLen_ = this->numLastDimAligned_;
        this->lastDimLoopNum_ = 1;
        this->lastDimSliceLenTail_ = 0;
        this->ubTilingPolicy_ = UB_TILING_POLICY::NORMAL;
    }
    return ret;
}

bool AddRmsNormDynamicQuantV2TilingHelper::CheckUbSingleRowTiling()
{
    // 2 tmp buffer, 2 rows copy in and 1 rows copy out
    int64_t ubRequired = ((2 + 1 + 1) * this->dtSize_ + 2 * sizeof(float)) * this->numLastDimAligned_;
    ubRequired = ubRequired + NUM_RMSNORM_STATS * ROW_FACTOR * sizeof(float);
    bool ret = (((int64_t)this->ubSize_) >= ubRequired);
    OP_LOGI(this->context_->GetNodeName(), "CheckUbSingleRowTiling, ret:%d, ubRequired: %ld", ret, ubRequired);
    if (ret) {
        this->firstDimPerLoop_ = 1;
        this->lastDimSliceLen_ = this->numLastDimAligned_;
        this->lastDimLoopNum_ = 1;
        this->lastDimSliceLenTail_ = 0;
        this->ubTilingPolicy_ = UB_TILING_POLICY::SINGLE_ROW;
    }
    return ret;
}

bool AddRmsNormDynamicQuantV2TilingHelper::CheckUbSliceDTiling()
{
    OP_LOGI(this->context_->GetNodeName(), "CheckUbSliceDTiling success. Compute tiling by yourself.");
    this->ubTilingPolicy_ = UB_TILING_POLICY::SLICE_D;
    this->firstDimPerLoop_ = 1;
    this->lastDimSliceLen_ = SLICE_COL_LEN;
    this->lastDimSliceLenTail_ = (this->numLastDim_ % this->lastDimSliceLen_ == 0) ?
                                     this->lastDimSliceLen_ :
                                     this->numLastDim_ % this->lastDimSliceLen_;
    this->lastDimLoopNum_ = (this->numLastDim_ - this->lastDimSliceLenTail_) / this->lastDimSliceLen_;
    return true;
}

ge::graphStatus Tiling4AddRmsNormDynamicQuantV2(gert::TilingContext* context)
{
    OP_LOGI(context, "Enter Tiling4AddRmsNormDynamicQuantV2");

    AddRmsNormDynamicQuantV2TilingData tiling;

    AddRmsNormDynamicQuantV2TilingHelper instanceNormV3TilingHelper(context);
    bool status = instanceNormV3TilingHelper.DoTiling();
    OP_CHECK_IF(!status, OP_LOGE(context, "DoTiling Failed, return Failed."), return ge::GRAPH_FAILED);

    instanceNormV3TilingHelper.SetTilingDataAndTilingKeyAndWorkSpace(&tiling);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingPrepare4AddRmsNormDynamicQuantV2(gert::TilingParseContext* context)
{
    OP_LOGD(context, "Enter TilingPrepare4AddRmsNormDynamicQuantV2");
    return ge::GRAPH_SUCCESS;
}

bool AddRmsNormDynamicQuantV2TilingHelper::CheckInputOutputShape()
{
    // Check Shape Not NULL
    const gert::StorageShape* x1Shape = this->context_->GetInputShape(X1_IDX);
    const gert::StorageShape* x2Shape = this->context_->GetInputShape(X2_IDX);
    const gert::StorageShape* gammaShape = this->context_->GetInputShape(GAMMA_IDX);

    const gert::StorageShape* y1Shape = this->context_->GetOutputShape(Y1_IDX);
    const gert::StorageShape* y2Shape = this->context_->GetOutputShape(Y2_IDX);
    const gert::StorageShape* y3Shape = this->context_->GetOutputShape(Y3_IDX);
    const gert::StorageShape* y4Shape = this->context_->GetOutputShape(Y4_IDX);
    const gert::StorageShape* xShape = this->context_->GetOutputShape(X_IDX);
    const gert::StorageShape* scale1Shape = this->context_->GetOutputShape(SCALE1_IDX);
    const gert::StorageShape* scale2Shape = this->context_->GetOutputShape(SCALE2_IDX);

    OP_CHECK_NULL_WITH_CONTEXT(this->context_, x1Shape);
    OP_CHECK_NULL_WITH_CONTEXT(this->context_, x2Shape);
    OP_CHECK_NULL_WITH_CONTEXT(this->context_, gammaShape);
    OP_CHECK_NULL_WITH_CONTEXT(this->context_, y1Shape);
    OP_CHECK_NULL_WITH_CONTEXT(this->context_, y2Shape);
    OP_CHECK_NULL_WITH_CONTEXT(this->context_, y3Shape);
    OP_CHECK_NULL_WITH_CONTEXT(this->context_, y4Shape);
    OP_CHECK_NULL_WITH_CONTEXT(this->context_, xShape);
    OP_CHECK_NULL_WITH_CONTEXT(this->context_, scale1Shape);
    OP_CHECK_NULL_WITH_CONTEXT(this->context_, scale2Shape);

    // Check Shape relations
    size_t x1DimNum = x1Shape->GetStorageShape().GetDimNum();
    size_t x2DimNum = x2Shape->GetStorageShape().GetDimNum();
    size_t gammaDimNum = gammaShape->GetStorageShape().GetDimNum();
    size_t y1DimNum = y1Shape->GetStorageShape().GetDimNum();
    size_t y2DimNum = y2Shape->GetStorageShape().GetDimNum();
    size_t y3DimNum = y3Shape->GetStorageShape().GetDimNum();
    size_t y4DimNum = y4Shape->GetStorageShape().GetDimNum();
    size_t xDimNum = xShape->GetStorageShape().GetDimNum();
    size_t scale1DimNum = scale1Shape->GetStorageShape().GetDimNum();
    size_t scale2DimNum = scale2Shape->GetStorageShape().GetDimNum();

    OP_LOGI(
        this->context_->GetNodeName(),
        "ShapeDim info: x1.dim=%zu, x2.dim=%zu, gamma.dim=%zu, y1.dim=%zu, y2.dim=%zu, y3.dim=%zu, y4.dim=%zu, "
        "x.dim=%zu, scale1.dim=%zu, "
        "scale2.dim=%zu",
        x1DimNum, x2DimNum, gammaDimNum, y1DimNum, y2DimNum, y3DimNum, y4DimNum, xDimNum, scale1DimNum, scale2DimNum);

    bool hasZeroDimTensor = x1DimNum <= 0 || x2DimNum <= 0 || gammaDimNum <= 0;
    OP_CHECK_IF(
        (hasZeroDimTensor),
        OP_LOGE(
            this->context_->GetNodeName(),
            "Input x1/x2/y1//x/scale1DimNum shape invaild, dim num should not be smaller or equal to zero."),
        return false);
    OP_CHECK_IF(
        ((x1DimNum != x2DimNum)),
        OP_LOGE(this->context_->GetNodeName(), "Input x1/x2 shape dims not equal. Tiling failed. "), return false);
    OP_CHECK_IF(
        ((gammaDimNum != 1)), OP_LOGE(this->context_->GetNodeName(), "gamma shape dims not equal to 1. Tiling failed."),
        return false);
    return true;
}

IMPL_OP_OPTILING(AddRmsNormDynamicQuantV2)
    .Tiling(Tiling4AddRmsNormDynamicQuantV2)
    .TilingParse<AddRmsNormDynamicQuantV2CompileInfo>(TilingPrepare4AddRmsNormDynamicQuantV2);
} // namespace optiling
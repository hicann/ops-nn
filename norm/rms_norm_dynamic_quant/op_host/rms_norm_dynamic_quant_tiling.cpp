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
 * \file rms_norm_dynamic_quant_tiling.cpp
 * \brief
 */
#include "rms_norm_dynamic_quant_tiling.h"

namespace optiling {

constexpr int X_IDX = 0;
constexpr int GAMMA_IDX = 1;
constexpr int SMOOTH_IDX = 2;
constexpr int BETA_IDX = 3;

constexpr int Y_IDX = 0;
constexpr int SCALE_IDX = 1;

constexpr int SHAPEDIM_OF_GAMMA = 1;

constexpr int NUM_WITH_BETA = 3;
constexpr int NUM_WITHOUT_BETA = 2;

constexpr int EPS_IDX = 0;

constexpr uint64_t USR_WORKSPACE_SIZE_910B = 1;

constexpr uint32_t SIZEOF_B16 = 2;
constexpr uint32_t BLOCK_SIZE = 32;
constexpr uint64_t ROW_FACTOR = 128;
constexpr uint64_t UB_RESERVED_BYTE = 768;
constexpr uint32_t MAX_ROW_STEP = 16;
constexpr uint32_t INT4_ALIGN_SIZE = 64;

constexpr uint32_t UB_TILING_POLICY_NORMAL = 1;
constexpr uint32_t UB_TILING_POLICY_SINGLE_ROW = 2;
constexpr uint32_t UB_TILING_POLICY_SLICE_D = 3;

constexpr uint32_t SLICE_COL_LEN = 8864;
constexpr uint32_t SLICE_COL_LEN_INT4 = 8832;

constexpr int32_t INT_NEGATIVE_ONE = -1;
constexpr int32_t INT_ZERO = 0;
constexpr int32_t INT_ONE = 1;
constexpr int32_t INT_TWO = 2;

template <typename T>
static inline T CeilDiv(T num, T rnd)
{
    return (((rnd) == 0) ? 0 : (((num) + (rnd)-1) / (rnd)));
}

template <typename T>
static inline T CeilAlign(T num, T rnd)
{
    return (((rnd) == 0) ? 0 : (((num) + (rnd)-1) / (rnd)) * (rnd));
}

static bool CheckOptionalShapeExisting(const gert::StorageShape* smoothShape)
{
    OP_CHECK_IF(nullptr == smoothShape, OP_LOGD("CheckOptionalShapeExisting", "Get nullptr smoothShape"), return false);
    int64_t smoothShapeSize = smoothShape->GetOriginShape().GetShapeSize();
    OP_CHECK_IF((smoothShapeSize <= 0), OP_LOGD("CheckOptionalShapeExisting", "Get empty smoothShape"), return false);
    return true;
}

static bool CheckOptionalBetaExisting(const gert::StorageShape* betaShape)
{
    OP_CHECK_IF(nullptr == betaShape, OP_LOGD("CheckOptionalBetaExisting", "Get nullptr betaShape"), return false);
    int64_t betaShapeSize = betaShape->GetOriginShape().GetShapeSize();
    OP_CHECK_IF((betaShapeSize <= 0), OP_LOGD("CheckOptionalBetaExisting", "Get empty betaShape"), return false);
    return true;
}

static size_t GetworkspaceRowsNum(int32_t outQuant1Flag, uint32_t smoothNum1_)
{
    if (outQuant1Flag == INT_NEGATIVE_ONE) {
        return (smoothNum1_ == INT_ZERO) ? INT_ONE : INT_TWO;
    }
    return (outQuant1Flag == INT_ONE) ? INT_TWO : INT_ONE;
}

void RmsNormDynamicQuantTilingHelper::SetTilingDataAndTilingKeyAndWorkSpace(RmsNormDynamicQuantTilingData* tiling)
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
    tiling->set_smoothNum1(this->smoothNum1_);
    tiling->set_epsilon(this->eps_);
    tiling->set_outQuant1Flag(this->outQuant1Flag);
    tiling->set_avgFactor(this->avgFactor_);
    tiling->set_betaFlag(this->betaFlag_);
    uint32_t tilingKey = 0;
    size_t usrSize = USR_WORKSPACE_SIZE_910B;

    if (this->ubTilingPolicy_ == UB_TILING_POLICY::NORMAL) {
        tilingKey += UB_TILING_POLICY_NORMAL;
    } else if (this->ubTilingPolicy_ == UB_TILING_POLICY::SINGLE_ROW) {
        tilingKey += UB_TILING_POLICY_SINGLE_ROW;
    } else {
        tilingKey += UB_TILING_POLICY_SLICE_D;
        size_t workspaceRowsNum = GetworkspaceRowsNum(this->outQuant1Flag, this->smoothNum1_);
        usrSize = this->useCore_ * this->numLastDim_ * sizeof(float) * workspaceRowsNum;
    }

    context_->SetTilingKey(tilingKey);

    tiling->SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(tiling->GetDataSize());

    // set workspace
    size_t* currentWorkspace = context_->GetWorkspaceSizes(1);
    currentWorkspace[0] = this->sysWorkspaceSize_ + usrSize;

    OP_LOGI("SetTilingDataAndTilingKeyAndWorkSpace",
            "useCore=%lu smooth=%u N=%lu D=%lu DAligned=%lu "
            "firstPerCore=%lu firstPerCoreTail=%lu firstPerLoop=%lu "
            "sliceLen=%lu loopNum=%lu sliceTail=%lu eps=%f avg=%f tilingKey=%u wsSize=%zu",
            this->useCore_, this->smoothNum1_, numFirstDim_, numLastDim_, numLastDimAligned_, firstDimPerCore_,
            firstDimPerCoreTail_, firstDimPerLoop_, lastDimSliceLen_, lastDimLoopNum_, lastDimSliceLenTail_, eps_,
            avgFactor_, tilingKey, usrSize);
}

bool RmsNormDynamicQuantTilingHelper::DoTiling()
{
    OP_CHECK_IF((nullptr == context_),
                OP_LOGE("RmsNormDynamicQuantTiling", "Helper context_ get nullptr, return failed."), return false);
    // 1. platform + attrs
    OP_CHECK_IF(!GetBaseInfo(), OP_LOGE(context_->GetNodeName(), "GetBaseInfo failed, return false"), return false);
    // 2. validate dtypes + shapes + compute shape parameters
    OP_CHECK_IF(!GetShapeInfo(), OP_LOGE(context_->GetNodeName(), "GetShapeInfo failed, return false"), return false);
    // 3. compute tiling
    OP_CHECK_IF(!DoBlockTiling(), OP_LOGE(context_->GetNodeName(), "DoBlockTiling failed, return false"), return false);
    OP_CHECK_IF(!DoUbTiling(), OP_LOGE(context_->GetNodeName(), "DoUbTiling failed, return false"), return false);
    return true;
}

bool RmsNormDynamicQuantTilingHelper::DoBlockTiling()
{
    // Block Tiling, Cut N
    this->firstDimPerCore_ = CeilDiv(this->numFirstDim_, this->socCoreNums_);
    this->useCore_ = CeilDiv(this->numFirstDim_, this->firstDimPerCore_);
    this->firstDimPerCore_ = CeilDiv(this->numFirstDim_, this->useCore_);
    this->firstDimPerCoreTail_ = this->numFirstDim_ - this->firstDimPerCore_ * (this->useCore_ - 1);
    OP_LOGI("DoBlockTiling", "BlockTiling Factor: useCore_: %lu, firstDimPerCore_: %lu, firstDimPerCoreTail_: %lu",
            this->useCore_, this->firstDimPerCore_, this->firstDimPerCoreTail_);
    return true;
}

bool RmsNormDynamicQuantTilingHelper::InitializePlatformInfo()
{
    auto platformInfo = context_->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context_, platformInfo);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    this->socCoreNums_ = ascendcPlatform.GetCoreNumAiv();
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, this->ubSize_);
    this->sysWorkspaceSize_ = ascendcPlatform.GetLibApiWorkSpaceSize();
    return true;
}

bool RmsNormDynamicQuantTilingHelper::GetBaseInfo()
{
    if (!InitializePlatformInfo()) {
        return false;
    }

    auto attrs = context_->GetAttrs();
    OP_CHECK_IF(nullptr == attrs, OP_LOGE(context_->GetNodeName(), "Get attrs nullptr, return false."), return false);

    const float* epsPtr = attrs->GetFloat(EPS_IDX);
    if (epsPtr != nullptr) {
        this->eps_ = *epsPtr;
    }

    this->outQuant1Flag = -1; // auto mode: quant enabled when smooth_scales present
    if (!ValidateBaseParameters()) {
        return false;
    }
    OP_LOGI("GetBaseInfo", "socCoreNum: %lu, ubSize: %lu, sysWorkspaceSize: %lu, epsilon: %f", this->socCoreNums_,
            this->ubSize_, this->sysWorkspaceSize_, this->eps_);

    return true;
}

bool RmsNormDynamicQuantTilingHelper::ValidateBaseParameters()
{
    OP_CHECK_IF(this->eps_ <= 0,
                OP_LOGE_FOR_INVALID_VALUE(context_->GetNodeName(), "epsilon", std::to_string(this->eps_).c_str(),
                                          "greater than 0"),
                return false);
    OP_CHECK_IF((this->ubSize_ <= 0), OP_LOGE(context_->GetNodeName(), "ubSize less or equal than zero, please check."),
                return false);
    OP_CHECK_IF((this->socCoreNums_ <= 0),
                OP_LOGE(context_->GetNodeName(), "socCoreNums_ less or equal than zero, please check."), return false);

    return true;
}

static ge::graphStatus CheckDtypeVaild(ge::DataType& srcDtype, std::vector<ge::DataType>& supportDtypeList)
{
    for (const auto& supportedDtype : supportDtypeList) {
        if (supportedDtype == srcDtype) {
            return ge::GRAPH_SUCCESS;
        }
    }
    return ge::GRAPH_FAILED;
}

bool RmsNormDynamicQuantTilingHelper::ValidateDtypes()
{
    auto xDataType = context_->GetInputDesc(X_IDX)->GetDataType();
    std::vector<ge::DataType> supportedInDtypes = {ge::DataType::DT_FLOAT16, ge::DataType::DT_BF16};
    if (ge::GRAPH_SUCCESS != CheckDtypeVaild(xDataType, supportedInDtypes)) {
        OP_LOGE_FOR_INVALID_DTYPE(context_->GetNodeName(), "x", Ops::Base::ToString(xDataType).c_str(),
                                  "FLOAT16 or BFLOAT16");
        return false;
    }

    // gamma/smooth/beta dtypes must match x
    auto gammaDataType = context_->GetInputDesc(GAMMA_IDX)->GetDataType();
    if (gammaDataType != xDataType) {
        std::string dtypeMsg = Ops::Base::ToString(xDataType) + " and " + Ops::Base::ToString(gammaDataType);
        OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(context_->GetNodeName(), "x and gamma", dtypeMsg.c_str(),
                                               "The dtypes of input x and gamma must be the same");
        return false;
    }
    if (this->smoothNum1_) {
        auto smoothDataType = context_->GetInputDesc(SMOOTH_IDX)->GetDataType();
        if (smoothDataType != xDataType) {
            std::string dtypeMsg = Ops::Base::ToString(xDataType) + " and " + Ops::Base::ToString(smoothDataType);
            OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(context_->GetNodeName(), "x and smooth_scales", dtypeMsg.c_str(),
                                                   "The dtypes of input x and smooth_scales must be the same");
            return false;
        }
    }
    if (this->betaFlag_) {
        auto betaDataType = context_->GetInputDesc(BETA_IDX)->GetDataType();
        if (betaDataType != xDataType) {
            std::string dtypeMsg = Ops::Base::ToString(xDataType) + " and " + Ops::Base::ToString(betaDataType);
            OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(context_->GetNodeName(), "x and beta", dtypeMsg.c_str(),
                                                   "The dtypes of input x and beta must be the same");
            return false;
        }
    }

    // output y dtype
    auto yDataType = context_->GetOutputDesc(Y_IDX)->GetDataType();
    if (yDataType != ge::DataType::DT_INT8) {
        OP_LOGE_FOR_INVALID_DTYPE(context_->GetNodeName(), "y", Ops::Base::ToString(yDataType).c_str(), "INT8");
        return false;
    }

    // output scale dtype
    auto scaleDataType = context_->GetOutputDesc(SCALE_IDX)->GetDataType();
    if (scaleDataType != ge::DataType::DT_FLOAT) {
        OP_LOGE_FOR_INVALID_DTYPE(context_->GetNodeName(), "scale", Ops::Base::ToString(scaleDataType).c_str(),
                                  "FLOAT32");
        return false;
    }

    return true;
}

bool RmsNormDynamicQuantTilingHelper::CheckInputShapes()
{
    const gert::StorageShape* xShape = this->context_->GetInputShape(X_IDX);
    const gert::StorageShape* gammaShape = this->context_->GetInputShape(GAMMA_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(this->context_, xShape);
    OP_CHECK_NULL_WITH_CONTEXT(this->context_, gammaShape);

    auto xStorage = xShape->GetStorageShape();
    auto gammaStorage = gammaShape->GetStorageShape();
    size_t xDimNum = xStorage.GetDimNum();
    size_t gammaDimNum = gammaStorage.GetDimNum();

    if (xStorage.GetShapeSize() == 0) {
        OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(context_->GetNodeName(), "x", "0",
                                                  "Input x cannnot be an empty tensor");
        return false;
    }

    if (xDimNum <= 0) {
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(context_->GetNodeName(), "x", std::to_string(xDimNum).c_str(),
                                                 "The shape dim of input x must be greater than 0");
        return false;
    }
    if (gammaDimNum != 1) {
        OP_LOGE_FOR_INVALID_SHAPEDIM(context_->GetNodeName(), "gamma", std::to_string(gammaDimNum).c_str(), "1D");
        return false;
    }
    if (xStorage.GetDim(xDimNum - 1) != gammaStorage.GetDim(0)) {
        std::string shapeMsg = Ops::Base::ToString(gammaStorage) + " and " + Ops::Base::ToString(xStorage);
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
            context_->GetNodeName(), "gamma and x", shapeMsg.c_str(),
            "The shape of input gamma must be the same as the shape consisting of the last axis of input x");
        return false;
    }

    // 可选输入存在性及 shape 一致性
    const gert::StorageShape* smoothShape = this->context_->GetOptionalInputShape(SMOOTH_IDX);
    const gert::StorageShape* betaShape = this->context_->GetOptionalInputShape(BETA_IDX);
    this->smoothNum1_ = CheckOptionalShapeExisting(smoothShape) ? 1 : 0;
    this->betaFlag_ = CheckOptionalBetaExisting(betaShape) ? 1 : 0;

    if (this->smoothNum1_ && smoothShape->GetStorageShape() != gammaStorage) {
        std::string shapeMsg = Ops::Base::ToString(gammaStorage) + " and " +
                               Ops::Base::ToString(smoothShape->GetStorageShape());
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(context_->GetNodeName(), "gamma and smooth_scales", shapeMsg.c_str(),
                                               "The shapes of input gamma and smooth_scales must be the same");
        return false;
    }
    if (this->betaFlag_ && betaShape->GetStorageShape() != gammaStorage) {
        std::string shapeMsg = Ops::Base::ToString(gammaStorage) + " and " +
                               Ops::Base::ToString(betaShape->GetStorageShape());
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(context_->GetNodeName(), "gamma and beta", shapeMsg.c_str(),
                                               "The shapes of input gamma and beta must be the same");
        return false;
    }
    return true;
}

bool RmsNormDynamicQuantTilingHelper::CheckOutputShapes()
{
    const gert::StorageShape* yShape = this->context_->GetOutputShape(Y_IDX);
    const gert::StorageShape* scaleShape = this->context_->GetOutputShape(SCALE_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(this->context_, yShape);
    OP_CHECK_NULL_WITH_CONTEXT(this->context_, scaleShape);

    auto xStorage = this->context_->GetInputShape(X_IDX)->GetStorageShape();
    auto yStorage = yShape->GetStorageShape();
    auto scaleStorage = scaleShape->GetStorageShape();

    if (yStorage != xStorage) {
        std::string shapeMsg = Ops::Base::ToString(yStorage) + " and " + Ops::Base::ToString(xStorage);
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(context_->GetNodeName(), "y and x", shapeMsg.c_str(),
                                               "The shapes of output y and input x must be the same");
        return false;
    }
    size_t expectedScaleDim = xStorage.GetDimNum() - SHAPEDIM_OF_GAMMA;
    if (scaleStorage.GetDimNum() != expectedScaleDim) {
        std::string shapeDimMsg = std::to_string(scaleStorage.GetDimNum()) + " and " + std::to_string(expectedScaleDim);
        OP_LOGE_FOR_INVALID_SHAPEDIMS_WITH_REASON(
            context_->GetNodeName(), "scale", shapeDimMsg.c_str(),
            "The shape dim of output scale must be one less than that of input x");
        return false;
    }
    for (int i = 0; i < expectedScaleDim; ++i) {
        if (scaleStorage.GetDim(i) != xStorage.GetDim(i)) {
            std::string shapeMsg = Ops::Base::ToString(scaleStorage) + " and " + Ops::Base::ToString(xStorage);
            OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                context_->GetNodeName(), "scale and x", shapeMsg.c_str(),
                "The shape of output scale must be the same as the shape of input x excluding the last dimension");
        }
    }
    return true;
}

bool RmsNormDynamicQuantTilingHelper::ValidateShapes()
{
    if (!CheckInputShapes()) {
        return false;
    }
    if (!CheckOutputShapes()) {
        return false;
    }
    return true;
}

bool RmsNormDynamicQuantTilingHelper::CalculateShapeParameters()
{
    // 设置数据类型大小
    this->dtSize_ = SIZEOF_B16;

    // 获取输入形状
    auto xShape = context_->GetInputShape(X_IDX)->GetStorageShape();
    auto gammaShape = context_->GetInputShape(GAMMA_IDX)->GetStorageShape();
    size_t xDimNum = xShape.GetDimNum();
    size_t gammaDimNum = gammaShape.GetDimNum();

    // 计算numRow和numCol
    uint64_t numRow = 1;
    uint64_t numCol = 1;
    for (size_t i = 0; i < xDimNum - gammaDimNum; i++) {
        numRow *= xShape.GetDim(i);
    }
    for (size_t i = 0; i < gammaDimNum; i++) {
        numCol *= gammaShape.GetDim(i);
    }

    // 设置对齐大小和目标类型
    this->numFirstDim_ = numRow;
    this->numLastDim_ = numCol;
    auto yDataType = context_->GetOutputDesc(Y_IDX)->GetDataType();
    uint32_t alignSize = yDataType == ge::DT_INT4 ? INT4_ALIGN_SIZE : BLOCK_SIZE;
    this->dstType_ = static_cast<uint32_t>(yDataType);
    this->numLastDimAligned_ = CeilDiv(numCol, static_cast<uint64_t>(alignSize)) * static_cast<uint64_t>(alignSize);

    // 计算平均因子
    this->avgFactor_ = 1.0 / ((float)this->numLastDim_);

    return true;
}

bool RmsNormDynamicQuantTilingHelper::GetShapeInfo()
{
    // 1. 校验shape
    if (!ValidateShapes()) {
        return false;
    }

    // 2. 校验dtype
    if (!ValidateDtypes()) {
        return false;
    }

    // 3. 计算形状参数
    if (!CalculateShapeParameters()) {
        return false;
    }

    // 打印日志
    OP_LOGI("GetShapeInfo", "[N,D]=[%lu,%lu] dtSize=%lu avgFactor=%f", numFirstDim_, numLastDim_, dtSize_, avgFactor_);
    return true;
}

bool RmsNormDynamicQuantTilingHelper::DoUbTiling()
{
    OP_CHECK_IF(CheckUbNormalTiling(), OP_LOGI(context_->GetNodeName(), "Ub Tiling: Normal."), return true);
    OP_CHECK_IF(CheckUbSingleRowTiling(), OP_LOGI(context_->GetNodeName(), "Ub Tiling: SingleRow."), return true);
    OP_CHECK_IF(CheckUbSliceDTiling(), OP_LOGI(context_->GetNodeName(), "Ub Tiling: SliceD."), return true);
    return false;
}

bool RmsNormDynamicQuantTilingHelper::CheckUbNormalTiling()
{
    // 3 weights tensor required.
    int64_t ubConst = 0;
    if (this->betaFlag_ == 1) {
        ubConst = this->numLastDimAligned_ * this->dtSize_ * NUM_WITH_BETA + UB_RESERVED_BYTE;
    } else {
        ubConst = this->numLastDimAligned_ * this->dtSize_ * NUM_WITHOUT_BETA + UB_RESERVED_BYTE;
    }
    int64_t ubAvaliable = this->ubSize_ - ubConst;
    // 2 rows for tmpBuffer.
    int64_t coexistingRowsNum = 2 * (this->dtSize_) + 2 * (this->dtSize_) + 1 * sizeof(float) + 1 * sizeof(float);
    // 2 buffers for out_scale.
    int64_t rowCommons = coexistingRowsNum * this->numLastDimAligned_ + 2 * sizeof(float);
    int64_t rowStep = ubAvaliable / rowCommons;
    bool ret = (rowStep >= 1);
    OP_LOGI(this->context_->GetNodeName(),
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

bool RmsNormDynamicQuantTilingHelper::CheckUbSingleRowTiling()
{
    // 2 tmp buffer, 2 rows copy in and 1 rows copy out
    int64_t ubRequired = ((2 + 1 + 1) * this->dtSize_ + 2 * sizeof(float)) * this->numLastDimAligned_;
    ubRequired = ubRequired + 2L * ROW_FACTOR * sizeof(float);
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

bool RmsNormDynamicQuantTilingHelper::CheckUbSliceDTiling()
{
    OP_LOGI(this->context_->GetNodeName(), "CheckUbSliceDTiling success. Compute tiling by yourself.");
    this->ubTilingPolicy_ = UB_TILING_POLICY::SLICE_D;
    this->firstDimPerLoop_ = 1;
    if (this->dstType_ == ge::DT_INT4) {
        this->lastDimSliceLen_ = SLICE_COL_LEN_INT4;
    } else {
        this->lastDimSliceLen_ = SLICE_COL_LEN;
    }
    this->lastDimSliceLenTail_ = (this->numLastDim_ % this->lastDimSliceLen_ == 0) ?
                                     this->lastDimSliceLen_ :
                                     this->numLastDim_ % this->lastDimSliceLen_;
    this->lastDimLoopNum_ = (this->numLastDim_ - this->lastDimSliceLenTail_) / this->lastDimSliceLen_;
    return true;
}

ge::graphStatus Tiling4RmsNormDynamicQuant(gert::TilingContext* context)
{
    OP_CHECK_IF(nullptr == context, OP_LOGE("RmsNormDynamicQuant", "Context is null"), return ge::GRAPH_FAILED);
    OP_LOGI(context->GetNodeName(), "Enter Tiling4RmsNormDynamicQuant");
    auto ptrCompileInfo = reinterpret_cast<const RmsNormDynamicQuantCompileInfo*>(context->GetCompileInfo());
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    platform_ascendc::SocVersion curSocVersion = (ptrCompileInfo) == nullptr ? ascendcPlatform.GetSocVersion() :
                                                                               ptrCompileInfo->curSocVersion;
    RmsNormDynamicQuantTilingData tiling;
    RmsNormDynamicQuantTilingHelper instanceNormV3TilingHelper(context);
    bool status = instanceNormV3TilingHelper.DoTiling();
    OP_CHECK_IF(!status, OP_LOGE(context->GetNodeName(), "DoTiling Failed, return Failed."), return ge::GRAPH_FAILED);
    instanceNormV3TilingHelper.SetTilingDataAndTilingKeyAndWorkSpace(&tiling);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingPrepare4RmsNormDynamicQuant(gert::TilingParseContext* context)
{
    OP_CHECK_IF(nullptr == context, OP_LOGE("RmsNormDynamicQuant", "Context is null"), return ge::GRAPH_FAILED);
    OP_LOGD(context->GetNodeName(), "Enter TilingPrepare4RmsNormDynamicQuant.");
    fe::PlatFormInfos* platformInfoPtr = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfoPtr);

    auto compileInfoPtr = context->GetCompiledInfo<RmsNormDynamicQuantCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfoPtr);

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    compileInfoPtr->curSocVersion = ascendcPlatform.GetSocVersion();
    compileInfoPtr->totalCoreNum = ascendcPlatform.GetCoreNumAiv();
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, compileInfoPtr->maxUbSize);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(RmsNormDynamicQuant)
    .Tiling(Tiling4RmsNormDynamicQuant)
    .TilingParse<RmsNormDynamicQuantCompileInfo>(TilingPrepare4RmsNormDynamicQuant);

} // namespace optiling
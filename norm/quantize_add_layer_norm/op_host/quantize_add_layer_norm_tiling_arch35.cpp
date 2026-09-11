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
 * \file quantize_add_layer_norm_tiling_arch35.cpp
 * \brief ascend950 (arch35 / regbase) tiling for QuantizeAddLayerNorm.
 *        Mirrors add_layer_norm_quant_tiling_arch35.cpp, static-quant & single-path only
 *        (no dual scales/zero_points, no dynamic-quant).
 */
#include "quantize_add_layer_norm_tiling.h"

namespace optiling {
constexpr int32_t CONST_2 = 2;
constexpr int32_t CONST_4 = 4;
constexpr int32_t CONST_8 = 8;
constexpr int32_t CONST_16 = 16;
constexpr int32_t CONST_32 = 32;
constexpr uint64_t KERNEL_BUFFER_NUM = 2;

constexpr uint32_t BLOCK_SIZE = 32;
constexpr uint64_t UB_RESERVED_BYTE = 256;
constexpr int32_t MAX_ROW_STEP = 255;

constexpr uint32_t TILING_REGBASE_PREFIX = 8000;
// full-load: 0, welford: 100
constexpr uint32_t TILING_WELFORD = 100;
// no bias: 0, bias elewise: 1, bias brc: 2
constexpr uint32_t TILING_BIAS_ELEWISE = 1;
constexpr uint32_t TILING_BIAS_BRC = 2;

// quant mode bits: mul_mode = 0 (default), per_channel(div) = 10, per_tensor(scalar mul) = 20
constexpr uint32_t TILING_DIV_MODE = 10;
constexpr uint32_t TILING_PER_TENSOR_MODE = 20;

constexpr size_t DEFAULT_WORKSPACE_SIZE = 32;

// axis magic numbers (same as 910b tiling)
constexpr int64_t AXIS_VALUE_FOR_PER_TENSOR = 65535;
constexpr int64_t AXIS_VALUE_FOR_MUL_DIV_MODE = -65535;

static int64_t inline FindFloorPowerTwo(int64_t num)
{
    int64_t n = num;
    n |= n >> 1;
    n |= n >> CONST_2;
    n |= n >> CONST_4;
    n |= n >> CONST_8;
    n |= n >> CONST_16;
    n |= n >> CONST_32;
    n = (n + 1) >> 1;
    n = (n >= num) ? n / CONST_2 : n;
    return n;
}

static inline bool HasZeroDim(const gert::StorageShape* s)
{
    if (nullptr == s) {
        return false;
    }
    gert::Shape shape = s->GetStorageShape();
    for (size_t i = 0; i < shape.GetDimNum(); i++) {
        OP_CHECK_IF((shape.GetDim(i) == 0), OP_LOGW("HasZeroDim", "Got 0 dim in shape."), return true);
    }
    return (shape.GetShapeSize() <= 0);
}

void QuantizeAddLayerNormRegbaseTiling::ComputeBinaryAddVars()
{
    int64_t vcaddNum = this->binaryAddNum_ / this->vlFp32_;
    if (vcaddNum <= this->vlFp32_) {
        this->binaryAddK_ = 0;
        this->binaryAddLastNum_ = vcaddNum;
    } else {
        this->binaryAddK_ = 0;
        int64_t curBinaryAddNum = 1;
        while (curBinaryAddNum < (vcaddNum / this->vlFp32_)) {
            this->binaryAddK_++;
            curBinaryAddNum *= CONST_2;
        }
        this->binaryAddLastNum_ = this->vlFp32_;
    }
    OP_LOGW("ComputeBinaryAddVars", "binaryAddNum:%ld, binaryAddK:%ld, binaryAddLastNum:%ld", this->binaryAddNum_,
            this->binaryAddK_, this->binaryAddLastNum_);
}

void QuantizeAddLayerNormRegbaseTiling::ApplyFullLoadTilingResult(int64_t rowStep, int64_t binaryAddNum)
{
    this->rowsPerLoop_ = (rowStep <= this->rowsPerCore_) ? rowStep : this->rowsPerCore_;
    this->rowsPerLoop_ = (this->rowsPerLoop_ <= MAX_ROW_STEP) ? this->rowsPerLoop_ : MAX_ROW_STEP;
    this->colsPerLoop_ = this->cols_;
    this->colsLoopCount_ = 1;
    this->colsTail_ = this->colsPerLoop_;

    this->binaryAddNum_ = binaryAddNum;
    ComputeBinaryAddVars();
    this->ubTilingPolicy_ = UB_TILING_POLICY::FULL_LOAD;
}

void QuantizeAddLayerNormRegbaseTiling::SetTilingDataAndTilingKeyAndWorkSpace(
    QuantizeAddLayerNormRegbaseTilingData* tiling)
{
    tiling->set_rowsPerCore(this->rowsPerCore_);
    tiling->set_rowsPerTailCore(this->rowsPerTailCore_);
    tiling->set_rowsPerLoop(this->rowsPerLoop_);
    tiling->set_cols(this->cols_);
    tiling->set_colsPerLoop(this->colsPerLoop_);
    tiling->set_colsLoopCount(this->colsLoopCount_);
    tiling->set_colsTail(this->colsTail_);
    tiling->set_binaryAddNum(this->binaryAddNum_);
    tiling->set_binaryAddK(this->binaryAddK_);
    tiling->set_binaryAddLastNum(this->binaryAddLastNum_);
    tiling->set_eps(this->eps_);
    tiling->set_outputX(this->needOutputX_);

    uint32_t tilingKey = TILING_REGBASE_PREFIX;
    if (this->ubTilingPolicy_ == UB_TILING_POLICY::WELFORD) {
        tilingKey += TILING_WELFORD;
    }
    if (biasType_ == BIAS_TYPE::ELEWISE_BIAS) {
        tilingKey += TILING_BIAS_ELEWISE;
    } else if (biasType_ == BIAS_TYPE::BROADCAST_BIAS) {
        tilingKey += TILING_BIAS_BRC;
    }
    if (quantMode_ == QUANT_MODE::PER_CHANNEL) {
        tilingKey += TILING_DIV_MODE;
    } else if (quantMode_ == QUANT_MODE::PER_TENSOR) {
        tilingKey += TILING_PER_TENSOR_MODE;
    } // MUL_MODE: +0

    // static quant: fixed small workspace, no per-token outScale workspace, no SyncAll
    size_t usrWorkspaceSize = DEFAULT_WORKSPACE_SIZE;

    context_->SetTilingKey(tilingKey);
    context_->SetBlockDim(this->usedCoreNum_);
    tiling->SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(tiling->GetDataSize());

    size_t* currentWorkspace = context_->GetWorkspaceSizes(1);
    currentWorkspace[0] = usrWorkspaceSize;

    OP_LOGI("SetTilingDataAndTilingKeyAndWorkSpace", "Tilingdata tilingKey = %u, usr Workspace: %zu", tilingKey,
            usrWorkspaceSize);
    OP_LOGI("SetTilingData",
            "usedCoreNum:%u, vlFp32:%u, rowsPerCore:%ld, rowsPerTailCore:%ld, rowsPerLoop:%ld, "
            "cols:%ld, colsPerLoop:%ld, colsLoopCount:%ld, colsTail:%ld",
            this->usedCoreNum_, this->vlFp32_, this->rowsPerCore_, this->rowsPerTailCore_, this->rowsPerLoop_,
            this->cols_, this->colsPerLoop_, this->colsLoopCount_, this->colsTail_);
}

bool QuantizeAddLayerNormRegbaseTiling::DoTiling()
{
    OP_CHECK_IF((nullptr == context_),
                OP_LOGE("QuantizeAddLayerNormRegbaseTiling", "Helper context_ get nullptr, return failed."),
                return false);
    OP_CHECK_IF(!GetBaseInfo(), OP_LOGE(context_->GetNodeName(), "GetBaseInfo failed, return false"), return false);
    OP_CHECK_IF(!GetShapeInfo(), OP_LOGE(context_->GetNodeName(), "GetShapeInfo failed, return false"), return false);
    OP_CHECK_IF(!CheckDtype(), OP_LOGE(context_->GetNodeName(), "CheckDtype failed, return false"), return false);
    OP_CHECK_IF(!DoBlockTiling(), OP_LOGE(context_->GetNodeName(), "DoBlockTiling failed, return false"), return false);
    OP_CHECK_IF(!DoUbTiling(), OP_LOGE(context_->GetNodeName(), "DoUbTiling failed, return false"), return false);

    QuantizeAddLayerNormRegbaseTilingData tiling;
    SetTilingDataAndTilingKeyAndWorkSpace(&tiling);
    OP_LOGW(context_->GetNodeName(), "Finish DoTiling");
    return true;
}

bool QuantizeAddLayerNormRegbaseTiling::DoBlockTiling()
{
    // Block Tiling, Cut N (rows / M axis)
    this->rowsPerCore_ = Ops::Base::CeilDiv(this->rows_, static_cast<int64_t>(this->aivCoreNum_));
    this->usedCoreNum_ = Ops::Base::CeilDiv(this->rows_, this->rowsPerCore_);
    this->rowsPerCore_ = Ops::Base::CeilDiv(this->rows_, static_cast<int64_t>(this->usedCoreNum_));
    this->rowsPerTailCore_ = this->rows_ - this->rowsPerCore_ * (this->usedCoreNum_ - 1);
    OP_LOGW("DoBlockTiling", "usedCoreNum: %u, rowsPerCore: %ld, rowsPerTailCore: %ld", this->usedCoreNum_,
            this->rowsPerCore_, this->rowsPerTailCore_);
    return true;
}

bool QuantizeAddLayerNormRegbaseTiling::GetPlatformInfo()
{
    auto compileInfo = reinterpret_cast<const QuantizeAddLayerNormCompileInfo*>(context_->GetCompileInfo());
    if (compileInfo != nullptr) {
        this->ubSize_ = compileInfo->ubSize_;
        this->aivCoreNum_ = compileInfo->aivCoreNum_;
        this->blockSize_ = compileInfo->blockSize_;
        this->vecRegSize_ = compileInfo->vecRegSize_;
        this->sysWorkspaceSize_ = compileInfo->sysWorkspaceSize_;
    } else {
        auto platformInfo = this->context_->GetPlatformInfo();
        OP_CHECK_IF(nullptr == platformInfo, OP_LOGE(context_->GetNodeName(), "platform info is null"), return false);
        auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, this->ubSize_);
        this->aivCoreNum_ = ascendcPlatform.GetCoreNumAiv();
        this->blockSize_ = Ops::Base::GetUbBlockSize(this->context_);
        this->vecRegSize_ = Ops::Base::GetVRegSize(this->context_);
        this->sysWorkspaceSize_ = ascendcPlatform.GetLibApiWorkSpaceSize();
    }
    this->vlFp32_ = this->vecRegSize_ / sizeof(float);

    OP_CHECK_IF((this->ubSize_ <= 0),
                OP_LOGE(context_->GetNodeName(), "ubSize_ less or equal than zero, please check."), return false);
    OP_CHECK_IF((this->aivCoreNum_ <= 0),
                OP_LOGE(context_->GetNodeName(), "socCoreNums_ less or equal than zero, please check."), return false);

    OP_LOGW("GetPlatformInfo", "aivCoreNum: %u, ubSize: %lu, blockSize: %u, vecRegSize: %u", this->aivCoreNum_,
            this->ubSize_, this->blockSize_, this->vecRegSize_);
    return true;
}

bool QuantizeAddLayerNormRegbaseTiling::GetAttrs()
{
    auto attrs = context_->GetAttrs();
    OP_CHECK_IF(attrs == nullptr, OP_LOGE(context_->GetNodeName(), "Get attrs nullptr, return false."), return false);

    // attr[1] = axis (magic: 65535 -> per_tensor, -65535 -> mul_mode, else -> per_channel div)
    int64_t axis = GetOptionalAttr<int64_t>(attrs, AXIS_IDX, -1);
    if (axis == AXIS_VALUE_FOR_PER_TENSOR) {
        this->quantMode_ = QUANT_MODE::PER_TENSOR;
    } else if (axis == AXIS_VALUE_FOR_MUL_DIV_MODE) {
        this->quantMode_ = QUANT_MODE::MUL_MODE;
    } else {
        this->quantMode_ = QUANT_MODE::PER_CHANNEL;
    }

    this->eps_ = GetOptionalAttr<float>(attrs, EPS_IDX, (float)1e-5);
    this->needOutputX_ = GetOptionalAttr<bool>(attrs, X_OUT_ATTR_IDX, false);

    OP_CHECK_IF(this->eps_ <= 0,
                OP_LOGE_FOR_INVALID_VALUE(context_->GetNodeName(), "epsilon", std::to_string(this->eps_).c_str(),
                                          "greater than zero"),
                return false);

    OP_LOGW("GetAttrs", "axis=%ld, quantMode=%d, eps=%f, xOut=%d", axis, static_cast<int>(this->quantMode_), this->eps_,
            this->needOutputX_);
    return true;
}

bool QuantizeAddLayerNormRegbaseTiling::GetBaseInfo()
{
    OP_CHECK_IF(!GetPlatformInfo(), OP_LOGE(context_->GetNodeName(), "GetPlatformInfo failed, return false"),
                return false);
    OP_CHECK_IF(!GetAttrs(), OP_LOGE(context_->GetNodeName(), "GetAttrs failed, return false"), return false);
    return true;
}

bool QuantizeAddLayerNormRegbaseTiling::GetShapeInfo()
{
    OP_CHECK_IF((!CheckTensorAndAttr()), OP_LOGE(context_->GetNodeName(), "Check tensor shape and attr failed."),
                return false);
    OP_CHECK_IF((!CheckOptionalTensor()), OP_LOGE(context_->GetNodeName(), "Check optional tensor shape failed."),
                return false);
    this->dataTypeX1_ = context_->GetInputTensor(X1_IDX)->GetDataType();
    this->dtSizeX1_ = GetSizeByDataType(this->dataTypeX1_);
    auto xShape = context_->GetInputShape(X1_IDX)->GetStorageShape();
    auto gammaShape = context_->GetInputShape(GAMMA_IDX)->GetStorageShape();
    auto biasShape = context_->GetInputShape(BIAS_IDX)->GetStorageShape();
    size_t xDimNum = xShape.GetDimNum();
    size_t gammaDimNum = gammaShape.GetDimNum();
    // weights = gamma + beta + (bias if 1D)
    this->weightTensorNums_ = 2;
    size_t biasDimNum = biasShape.GetDimNum();
    this->biasType_ = (biasDimNum == gammaDimNum) ? BIAS_TYPE::BROADCAST_BIAS : BIAS_TYPE::ELEWISE_BIAS;
    this->weightTensorNums_ += (biasDimNum == 1) ? 1 : 0;

    uint64_t numRow = 1;
    for (size_t i = 0; i < xDimNum - gammaDimNum; i++) {
        numRow *= xShape.GetDim(i);
    }
    uint64_t numCol = 1;
    for (size_t i = 0; i < gammaDimNum; i++) {
        numCol *= gammaShape.GetDim(i);
    }
    this->rows_ = numRow;
    this->cols_ = numCol;
    this->colsAligned_ = Ops::Base::CeilDiv(this->cols_, static_cast<int64_t>(BLOCK_SIZE)) *
                         BLOCK_SIZE; // 32 element aligned
    this->avgFactor_ = 1.0f / (static_cast<float>(this->cols_));

    // static quant: quant tensors = scales(always) + zero_points(if exist)
    this->quantTensorNums_ = 1 + (this->offsetExist_ ? 1 : 0);

    OP_LOGW("GetShapeInfo",
            "[M, N] = [%ld, %ld], dtSizeX1=%lu, dtSizeScale=%lu, avgFactor_=%f, quantTensorNums=%ld, "
            "weightTensorNums=%ld, biasType=%d",
            this->rows_, this->cols_, this->dtSizeX1_, this->dtSizeScale_, this->avgFactor_, this->quantTensorNums_,
            this->weightTensorNums_, static_cast<int>(this->biasType_));
    return true;
}

bool QuantizeAddLayerNormRegbaseTiling::DoUbTiling()
{
    this->bufferNum_ = CONST_2;
    OP_CHECK_IF(CheckStcQuantFullLoadTiling(), OP_LOGW(context_->GetNodeName(), "Ub Tiling: FullLoad."), return true);
    OP_CHECK_IF(CheckStcQuantWelfordTiling(), OP_LOGW(context_->GetNodeName(), "Ub Tiling: WelFord."), return true);
    return false;
}

bool QuantizeAddLayerNormRegbaseTiling::CheckStcQuantFullLoadTiling()
{
    int64_t blkFp32Nums = BLOCK_SIZE / sizeof(float);
    int64_t tmpBinaryAddNum = (this->cols_ > this->vlFp32_) ? FindFloorPowerTwo(this->cols_) : this->vlFp32_;

    int64_t binaryAddUbSize = Ops::Base::CeilDiv((tmpBinaryAddNum / this->vlFp32_), blkFp32Nums) * blkFp32Nums *
                              sizeof(float);
    int64_t quantBufSize = this->colsAligned_ * this->quantTensorNums_ * this->dtSizeScale_;
    int64_t weightBufSize = this->colsAligned_ * this->weightTensorNums_ * this->dtSizeX1_;

    int64_t ubAvaliable = static_cast<int64_t>(this->ubSize_) -
                          (binaryAddUbSize + quantBufSize + weightBufSize + UB_RESERVED_BYTE);
    // COUNT(x1,x2)=2, COUNT(y)=1, COUNT(xOut)=1
    int64_t inOutCols = (this->dtSizeX1_ * 2 + sizeof(int8_t) * 1 + this->dtSizeX1_ * 1) * this->bufferNum_ *
                        this->colsAligned_;
    inOutCols += (this->biasType_ == BIAS_TYPE::ELEWISE_BIAS) ?
                     (this->dtSizeX1_ * 1 * this->bufferNum_ * this->colsAligned_) :
                     0;
    int64_t tmpBufsCols = sizeof(float) * this->colsAligned_ + sizeof(float) * 1 + sizeof(float) * 1;
    int64_t rowFactor = inOutCols + tmpBufsCols;
    int64_t rowStep = ubAvaliable / rowFactor;

    bool ret = (rowStep >= 1);
    OP_LOGW("CheckStcQuantFullLoadTiling",
            "ubAvaliable=%ld, binaryAddUbSize: %ld, quantBufSize: %ld, weightBufSize: %ld", ubAvaliable,
            binaryAddUbSize, quantBufSize, weightBufSize);
    OP_LOGW("CheckStcQuantFullLoadTiling", "inOutCols=%ld, tmpBufsCols=%ld, rowFactor=%ld, rowStep: %ld, ret: %d",
            inOutCols, tmpBufsCols, rowFactor, rowStep, ret);
    if (ret) {
        ApplyFullLoadTilingResult(rowStep, tmpBinaryAddNum);
    }
    return ret;
}

bool QuantizeAddLayerNormRegbaseTiling::CheckStcQuantWelfordTiling()
{
    int64_t quantSliceNums = this->bufferNum_ * this->dtSizeScale_ * this->quantTensorNums_;
    int64_t weightSliceNums = this->bufferNum_ * this->dtSizeX1_ * this->weightTensorNums_;
    // COUNT(x1,x2)=2, COUNT(xOut)=1, COUNT(y)=1
    int64_t elewiseSliceNums = this->bufferNum_ * this->dtSizeX1_ * (2 + 1) + this->bufferNum_ * sizeof(int8_t) * 1;
    elewiseSliceNums += (this->biasType_ == BIAS_TYPE::ELEWISE_BIAS) ? (this->bufferNum_ * this->dtSizeX1_ * 1) : 0;
    // COUNT(meanBuf,varBuf)=2, binaryAddBuf <= colSliceLen
    int64_t tmpSliceNums = sizeof(float) * (2 + 1);

    int64_t ubAvaliable = static_cast<int64_t>(this->ubSize_) - UB_RESERVED_BYTE;

    this->colsPerLoop_ = ubAvaliable / (quantSliceNums + weightSliceNums + elewiseSliceNums + tmpSliceNums);
    this->colsPerLoop_ = this->colsPerLoop_ / this->vlFp32_ * this->vlFp32_;
    OP_CHECK_IF((this->colsPerLoop_ <= 0), OP_LOGE(context_->GetNodeName(), "Welford colsPerLoop <= 0, unsupported."),
                return false);
    this->colsLoopCount_ = Ops::Base::CeilDiv(this->cols_, this->colsPerLoop_);

    // try to use aligned welford finalize process for better perf
    this->colsPerLoop_ = (this->cols_ % this->colsLoopCount_ == 0) ? (this->cols_ / this->colsLoopCount_) :
                                                                     this->colsPerLoop_;

    this->colsTail_ = this->cols_ % this->colsPerLoop_;
    this->colsTail_ = (this->colsTail_ == 0) ? this->colsPerLoop_ : this->colsTail_;

    this->binaryAddNum_ = (this->colsPerLoop_ > this->vlFp32_) ? FindFloorPowerTwo(this->colsPerLoop_) : this->vlFp32_;
    ComputeBinaryAddVars();
    this->ubTilingPolicy_ = UB_TILING_POLICY::WELFORD;
    return true;
}

bool QuantizeAddLayerNormRegbaseTiling::CheckTensorAndAttr()
{
    const gert::StorageShape* x1Shape = this->context_->GetInputShape(X1_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(this->context_, x1Shape);
    const gert::StorageShape* x2Shape = this->context_->GetInputShape(X2_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(this->context_, x2Shape);
    const gert::StorageShape* gammaShape = this->context_->GetInputShape(GAMMA_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(this->context_, gammaShape);
    const gert::StorageShape* betaShape = this->context_->GetInputShape(BETA_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(this->context_, betaShape);
    const gert::StorageShape* biasShape = this->context_->GetInputShape(BIAS_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(this->context_, biasShape);

    const gert::StorageShape* yShape = this->context_->GetOutputShape(Y_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(this->context_, yShape);
    const gert::StorageShape* xShape = this->context_->GetOutputShape(X_OUT_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(this->context_, xShape);

    size_t elewiseDimNum = x1Shape->GetStorageShape().GetDimNum();
    size_t weightDimNum = gammaShape->GetStorageShape().GetDimNum();
    OP_CHECK_IF((0 == elewiseDimNum || 0 == weightDimNum),
                OP_LOGW(this->context_->GetNodeName(), "Got x1/gamma is zero dim tensor, tiling FAILED."),
                return false);

    bool elewiseShapeEqual = ((*x1Shape) == (*x2Shape)) && ((*x1Shape) == (*xShape)) && ((*x1Shape) == (*yShape));
    bool weightShapeEqual = ((*gammaShape) == (*betaShape));
    OP_CHECK_IF(!(elewiseShapeEqual && weightShapeEqual),
                OP_LOGW(this->context_->GetNodeName(),
                        "Got x1/x2/y/x shape not equal OR gamma/beta shape not equal, tiling FAILED."),
                return false);

    OP_CHECK_IF(HasZeroDim(x1Shape),
                OP_LOGE_FOR_INVALID_SHAPESIZE(context_->GetNodeName(), "x1", "0", "greater than 0"), return false);
    OP_CHECK_IF(HasZeroDim(gammaShape),
                OP_LOGE_FOR_INVALID_SHAPESIZE(context_->GetNodeName(), "gamma", "0", "greater than 0"), return false);
    return true;
}

static inline bool checkOptionalShape(const size_t x1DimNum, const gert::StorageShape* gammaShape,
                                      const gert::StorageShape* optionalShape)
{
    if (*optionalShape == *gammaShape) {
        return true;
    }
    auto optionalDimNum = optionalShape->GetStorageShape().GetDimNum();
    if (optionalDimNum != x1DimNum) {
        return false;
    }
    auto gammaDimNum = gammaShape->GetStorageShape().GetDimNum();
    auto diffDimNum = optionalDimNum - gammaDimNum;
    for (size_t i = 0; i < diffDimNum; ++i) {
        if (optionalShape->GetStorageShape().GetDim(i) != 1) {
            return false;
        }
    }
    for (size_t i = 0; i < gammaDimNum; ++i) {
        if (optionalShape->GetStorageShape().GetDim(i + diffDimNum) != gammaShape->GetStorageShape().GetDim(i)) {
            return false;
        }
    }
    return true;
}

bool QuantizeAddLayerNormRegbaseTiling::CheckOptionalTensor()
{
    const gert::StorageShape* x1Shape = context_->GetInputShape(X1_IDX);
    const gert::StorageShape* gammaShape = context_->GetInputShape(GAMMA_IDX);
    size_t elewiseDimNum = x1Shape->GetStorageShape().GetDimNum();

    // bias is required, shape must equal x1 or gamma
    const gert::StorageShape* biasShape = context_->GetInputShape(BIAS_IDX);
    bool invalidBias = ((*biasShape) != (*x1Shape)) && ((*biasShape) != (*gammaShape));
    OP_CHECK_IF(invalidBias,
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context_->GetNodeName(), "bias",
                                                      Ops::Base::ToString(biasShape->GetStorageShape()).c_str(),
                                                      "The shape of bias should be equal to the shape of x1 or gamma"),
                return false);

    const gert::StorageShape* scaleShape = context_->GetInputShape(SCALE_IDX);
    const gert::StorageShape* offsetShape = context_->GetOptionalInputShape(ZERO_POINT_IDX);
    this->scaleExist_ = true; // scales is a required input
    this->offsetExist_ = (nullptr != offsetShape);

    // scales: per_channel / mul_mode -> shape equals gamma; per_tensor -> scalar (shape size <= 1)
    bool scaleShapeOk = checkOptionalShape(elewiseDimNum, gammaShape, scaleShape) ||
                        (scaleShape->GetStorageShape().GetShapeSize() <= 1);
    OP_CHECK_IF(
        (!scaleShapeOk),
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(context_->GetNodeName(), "scales and gamma",
                                               (Ops::Base::ToString(scaleShape->GetStorageShape()) + " and " +
                                                Ops::Base::ToString(gammaShape->GetStorageShape()))
                                                   .c_str(),
                                               "The shape of scales should equal gamma or be scalar for per_tensor"),
        return false);
    // zero_points: per_channel / mul_mode -> shape equals gamma; per_tensor -> scalar (shape size <= 1)
    bool offsetShapeOk = !this->offsetExist_ || checkOptionalShape(elewiseDimNum, gammaShape, offsetShape) ||
                         (offsetShape->GetStorageShape().GetShapeSize() <= 1);
    OP_CHECK_IF((!offsetShapeOk),
                OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                    context_->GetNodeName(), "zero_points and gamma",
                    (Ops::Base::ToString(offsetShape->GetStorageShape()) + " and " +
                     Ops::Base::ToString(gammaShape->GetStorageShape()))
                        .c_str(),
                    "The shape of zero_points should equal gamma or be scalar for per_tensor"),
                return false);

    const gert::Tensor* scaleTensor = context_->GetInputTensor(SCALE_IDX);
    this->dataTypeScale_ = (nullptr != scaleTensor) ? scaleTensor->GetDataType() : ge::DataType::DT_FLOAT;
    this->dtSizeScale_ = GetSizeByDataType(this->dataTypeScale_);
    return true;
}

bool QuantizeAddLayerNormRegbaseTiling::CheckOptionalInputDtype(ge::DataType x1Dtype)
{
    auto checkDt = [x1Dtype](ge::DataType dt, const char* name, gert::TilingContext* ctx) -> bool {
        if (dt != x1Dtype && dt != ge::DT_FLOAT) {
            OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(ctx->GetNodeName(), name, Ops::Base::ToString(dt).c_str(),
                                                  (std::string(name) + " dtype must match x1 or be fp32").c_str());
            return false;
        }
        return true;
    };
    // scales is a required input
    auto scaleDesc = context_->GetInputDesc(SCALE_IDX);
    OP_CHECK_IF(scaleDesc == nullptr, OP_LOGE(context_->GetNodeName(), "scales desc is nullptr"), return false);
    if (!checkDt(scaleDesc->GetDataType(), "scales", context_)) {
        return false;
    }
    // zero_points is optional
    if (this->offsetExist_) {
        auto offsetDesc = context_->GetOptionalInputDesc(ZERO_POINT_IDX);
        OP_CHECK_IF(offsetDesc == nullptr, OP_LOGE(context_->GetNodeName(), "zero_points desc is nullptr"),
                    return false);
        if (!checkDt(offsetDesc->GetDataType(), "zero_points", context_)) {
            return false;
        }
    }
    return true;
}

bool QuantizeAddLayerNormRegbaseTiling::CheckDtype()
{
    OP_LOGD(context_->GetNodeName(), "Enter CheckDtype.");
    auto x1Desc = context_->GetInputDesc(X1_IDX);
    OP_CHECK_IF(x1Desc == nullptr, OP_LOGE(context_->GetNodeName(), "x1 desc is nullptr"), return false);
    ge::DataType x1Dtype = x1Desc->GetDataType();
    auto x2Desc = context_->GetInputDesc(X2_IDX);
    OP_CHECK_IF(x2Desc == nullptr, OP_LOGE(context_->GetNodeName(), "x2 desc is nullptr"), return false);
    ge::DataType x2Dtype = x2Desc->GetDataType();
    auto gammaDesc = context_->GetInputDesc(GAMMA_IDX);
    OP_CHECK_IF(gammaDesc == nullptr, OP_LOGE(context_->GetNodeName(), "gamma desc is nullptr"), return false);
    ge::DataType gammaDtype = gammaDesc->GetDataType();
    auto betaDesc = context_->GetInputDesc(BETA_IDX);
    OP_CHECK_IF(betaDesc == nullptr, OP_LOGE(context_->GetNodeName(), "beta desc is nullptr"), return false);
    ge::DataType betaDtype = betaDesc->GetDataType();

    if (x1Dtype != ge::DT_FLOAT16 && x1Dtype != ge::DT_BF16 && x1Dtype != ge::DT_FLOAT) {
        OP_LOGE(context_->GetNodeName(), "Unsupported x1 dtype: %s", Ops::Base::ToString(x1Dtype).c_str());
        return false;
    }
    if (x1Dtype != x2Dtype || x1Dtype != gammaDtype || x1Dtype != betaDtype) {
        OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(
            context_->GetNodeName(), "x1, x2, gamma and beta",
            (Ops::Base::ToString(x1Dtype) + ", " + Ops::Base::ToString(x2Dtype) + ", " +
             Ops::Base::ToString(gammaDtype) + " and " + Ops::Base::ToString(betaDtype))
                .c_str(),
            "x1, x2, gamma and beta must have the same dtype");
        return false;
    }

    // bias is required, must match x1
    auto biasDesc = context_->GetInputDesc(BIAS_IDX);
    OP_CHECK_IF(biasDesc == nullptr, OP_LOGE(context_->GetNodeName(), "bias desc is nullptr"), return false);
    ge::DataType biasDtype = biasDesc->GetDataType();
    if (biasDtype != x1Dtype) {
        OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(
            context_->GetNodeName(), "x1 and bias",
            (Ops::Base::ToString(x1Dtype) + " and " + Ops::Base::ToString(biasDtype)).c_str(),
            "bias must have the same dtype as x1");
        return false;
    }

    if (!CheckOptionalInputDtype(x1Dtype)) {
        return false;
    }
    if (!CheckOutputDtype(x1Dtype)) {
        return false;
    }
    return true;
}

bool QuantizeAddLayerNormRegbaseTiling::CheckOutputDtype(ge::DataType x1Dtype)
{
    auto yDesc = context_->GetOutputDesc(Y_IDX);
    OP_CHECK_IF(yDesc == nullptr, OP_LOGE(context_->GetNodeName(), "y desc is nullptr"), return false);
    if (yDesc->GetDataType() != ge::DT_INT8) {
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
            context_->GetNodeName(), "y", Ops::Base::ToString(yDesc->GetDataType()).c_str(), "y dtype must be int8");
        return false;
    }
    auto xDesc = context_->GetOutputDesc(X_OUT_IDX);
    OP_CHECK_IF(xDesc == nullptr, OP_LOGE(context_->GetNodeName(), "x desc is nullptr"), return false);
    if (xDesc->GetDataType() != x1Dtype) {
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(context_->GetNodeName(), "x",
                                              Ops::Base::ToString(xDesc->GetDataType()).c_str(),
                                              "x dtype must be the same as x1 dtype");
        return false;
    }
    return true;
}

} // namespace optiling

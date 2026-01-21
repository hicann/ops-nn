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
 * \file add_layer_norm_quant_tiling_arch35.cpp
 * \brief
 */
#include "add_layer_norm_quant_tiling.h"

namespace optiling {
constexpr int32_t CONST_2 = 2;
constexpr int32_t CONST_4 = 4;
constexpr int32_t CONST_8 = 8;
constexpr int32_t CONST_16 = 16;
constexpr int32_t CONST_32 = 32;
constexpr uint64_t KERNEL_BUFFER_NUM = 2;
constexpr uint64_t USR_WORKSPACE_SIZE_910B = 1;

constexpr uint32_t BLOCK_SIZE = 32;
constexpr uint64_t UB_RESERVED_BYTE = 256;
constexpr int32_t MAX_ROW_STEP = 255;

constexpr uint32_t TILING_910_95_PREFIX = 8000;
// full-load: 000, welford: 100
constexpr uint32_t TILING_WELFORD = 100;
// no bias: 0, bias elewise: 1, bias brc: 2
constexpr uint32_t TILING_BIAS_ELEWISE = 1;
constexpr uint32_t TILING_BIAS_BRC = 2;

constexpr uint32_t TILING_DIV_MODE = 10;
constexpr uint32_t TILING_DYN_MODE = 20;

constexpr int64_t QUANT_OUT_NUMS_SOLE = 1;
constexpr int64_t QUANT_OUT_NUMS_DUAL = 2;
constexpr size_t DEFAULT_WORKSPACE_SIZE = 32;

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

static inline int64_t ConditionalIncreaseOne(int64_t value, bool cond)
{
    if (cond) {
        return (value + 1);
    } else {
        return value;
    }
}

static inline bool HasZeroDim(const gert::StorageShape* s)
{
    if (nullptr == s) {
        return false;
    }
    gert::Shape shape = s->GetStorageShape();
    for (size_t i = 0; i < shape.GetDimNum(); i++) {
        OP_CHECK_IF((shape.GetDim(i) == 0), OP_LOGW("HasZeroDim", "Got 0 dim in shape."), return false);
    }
    return (shape.GetShapeSize() <= 0);
}

void AddLayerNormQuantRegbaseTiling::ComputeBinaryAddVars()
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
    OP_LOGW(
        "ComputeBinaryAddVars", "binaryAddNum:%ld, binaryAddK:%ld, binaryAddLastNum:%ld", this->binaryAddNum_,
        this->binaryAddK_, this->binaryAddLastNum_);
}

void AddLayerNormQuantRegbaseTiling::SetTilingDataAndTilingKeyAndWorkSpace(AddLayerNormQuantRegbaseTilingData* tiling)
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

    uint32_t tilingKey = TILING_910_95_PREFIX;
    if (this->ubTilingPolicy_ == UB_TILING_POLICY::WELFORD) {
        tilingKey += TILING_WELFORD;
    }

    if (biasType_ == BIAS_TYPE::ELEWISE_BIAS) {
        tilingKey += TILING_BIAS_ELEWISE;
        OP_LOGW("GetShapeInfo", "AddLayerNorm ELEWISE_BIAS");
    } else if (biasType_ == BIAS_TYPE::BROADCAST_BIAS) {
        tilingKey += TILING_BIAS_BRC;
        OP_LOGW("GetShapeInfo", "AddLayerNorm BROADCAST_BIAS");
    } else {
        OP_LOGW("GetShapeInfo", "AddLayerNorm NO_BIAS");
    }

    size_t usrWorkspaceSize = DEFAULT_WORKSPACE_SIZE;
    if (isDynamicQuant_) {
        tilingKey += TILING_DYN_MODE;
        if (this->ubTilingPolicy_ == UB_TILING_POLICY::WELFORD) {
            usrWorkspaceSize = this->sysWorkspaceSize_ +
                      this->usedCoreNum_ * this->bufferNum_ * this->outQuantNums_ * this->cols_ * sizeof(float);
        }
    } else if (divMode_) {
        tilingKey += TILING_DIV_MODE;
    }

    // deal with small N case:
    if (1 == rowsPerCore_ && isDynamicQuant_) {
        // every core need 128 Bytes
        usrWorkspaceSize += static_cast<size_t>(usedCoreNum_* 128U * 2U);
        if (this->ubTilingPolicy_ != UB_TILING_POLICY::WELFORD) {
            usrWorkspaceSize += this->sysWorkspaceSize_;
        }
    }

    context_->SetTilingKey(tilingKey);
    context_->SetBlockDim(this->usedCoreNum_);
    tiling->SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(tiling->GetDataSize());

    // set workspace
    size_t* currentWorkspace = context_->GetWorkspaceSizes(1);
    currentWorkspace[0] = usrWorkspaceSize;

    OP_LOGI(
        "SetTilingDataAndTilingKeyAndWorkSpace", "Tilingdata tilingKey = %u, usr Workspace: %zu", tilingKey, usrWorkspaceSize);
    OP_LOGI(
        "SetTilingData", "blockSize: %u, usedCoreNum:%u, vlFp32:%u, rowsPerCore:%ld, rowsPerTailCore:%ld",
        this->blockSize_, this->usedCoreNum_, this->vlFp32_, this->rowsPerCore_, this->rowsPerTailCore_);
    OP_LOGI(
        "SetTilingData", "rowsPerLoop: %ld, cols:%ld, colsPerLoop:%ld, colsLoopCount:%ld, colsTail:%ld",
        this->rowsPerLoop_, this->cols_, this->colsPerLoop_, this->colsLoopCount_, this->colsTail_);
    OP_LOGI(
        "SetTilingData", "binaryAddNum: %ld, binaryAddK:%ld, binaryAddLastNum:%ld", this->binaryAddNum_,
        this->binaryAddK_, this->binaryAddLastNum_);
    OP_LOGI("SetTilingData", "eps: %f, needOutputX:%d", this->eps_, this->needOutputX_);
}

bool AddLayerNormQuantRegbaseTiling::DoTiling()
{
    OP_CHECK_IF(
        (nullptr == context_), OP_LOGE("AddLayerNormQuantRegbaseTiling", "Helper context_ get nullptr, return failed."),
        return false);
    OP_CHECK_IF(!GetBaseInfo(), OP_LOGE(context_->GetNodeName(), "GetBaseInfo falied, return false"), return false);
    OP_CHECK_IF(!GetShapeInfo(), OP_LOGE(context_->GetNodeName(), "GetShapeInfo falied, return false"), return false);
    OP_CHECK_IF(!DoBlockTiling(), OP_LOGE(context_->GetNodeName(), "DoBlockTiling falied, return false"), return false);
    OP_CHECK_IF(!DoUbTiling(), OP_LOGE(context_->GetNodeName(), "DoUbTiling falied, return false"), return false);
    OP_LOGW(context_->GetNodeName(), "Finish DoTiling");
    return true;
}

bool AddLayerNormQuantRegbaseTiling::DoBlockTiling()
{
    // Block Tiling, Cut N
    this->rowsPerCore_ = Ops::Base::CeilDiv(this->rows_, static_cast<int64_t>(this->aivCoreNum_));
    this->usedCoreNum_ = Ops::Base::CeilDiv(this->rows_, this->rowsPerCore_);
    this->rowsPerCore_ = Ops::Base::CeilDiv(this->rows_, static_cast<int64_t>(this->usedCoreNum_));
    this->rowsPerTailCore_ = this->rows_ - this->rowsPerCore_ * (this->usedCoreNum_ - 1);
    OP_LOGW(
        "DoBlockTiling", "BlockTiling Factor: usedCoreNum: %u, rowsPerCore: %ld, rowsPerTailCore: %ld",
        this->usedCoreNum_, this->rowsPerCore_, this->rowsPerTailCore_);
    return true;
}

bool AddLayerNormQuantRegbaseTiling::GetPlatformInfo()
{
    auto compileInfo = reinterpret_cast<const AddLayerNormQuantCompileInfo*>(context_->GetCompileInfo());
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

    OP_CHECK_IF(
        (this->ubSize_ <= 0), OP_LOGE(context_->GetNodeName(), "ubSize_ less or equal than zero, please check."),
        return false);
    OP_CHECK_IF(
        (this->aivCoreNum_ <= 0),
        OP_LOGE(context_->GetNodeName(), "socCoreNums_ less or equal than zero, please check."), return false);

    OP_LOGW(
        "GetPlatformInfo", "aivCoreNum: %u, ubSize: %lu, blockSize: %u, vecRegSize: %u", this->aivCoreNum_,
        this->ubSize_, this->blockSize_, this->vecRegSize_);
    return true;
}

bool AddLayerNormQuantRegbaseTiling::GetAttrs()
{
    auto attrs = context_->GetAttrs();
    OP_CHECK_IF(attrs == nullptr, OP_LOGE(context_->GetNodeName(), "Get attrs nullptr, return false."), return false);

    const char* qms = attrs->GetAttrPointer<char>(QUANT_MODE_IDX);
    OP_CHECK_IF(
        nullptr == qms, OP_LOGE(context_->GetNodeName(), "Get required attr quantMode failed, tiling failed"),
        return false);
    std::string quantModeStr = qms;
    this->isDynamicQuant_ = (quantModeStr == "dynamic");

    this->eps_ = GetOptionalAttr<float>(attrs, EPS_IDX, (float)1e-5);
    this->needOutputX_ = GetOptionalAttr<bool>(attrs, X_OUT_IDX, false);
    this->divMode_ = GetOptionalAttr<bool>(attrs, DIV_MODE_IDX, true);

    OP_CHECK_IF(
        this->eps_ <= 0, OP_LOGE(context_->GetNodeName(), "Epsilon less or equal than zero, please check."),
        return false);
    OP_CHECK_IF(
        ((quantModeStr != "dynamic") && (quantModeStr != "static")),
        OP_LOGE(
            context_->GetNodeName(), "QuantMode can be 'dynamic' or 'static', but got %s, place check.",
            quantModeStr.c_str()),
        return false);

    OP_LOGW(
        "GetAttrs", "quantModeStr=%s, eps=%f, xOut=%d, divMode=%d, isDynamicQuant=%d", quantModeStr.c_str(), this->eps_,
        this->needOutputX_, this->divMode_, this->isDynamicQuant_);
    return true;
}

bool AddLayerNormQuantRegbaseTiling::GetBaseInfo()
{
    OP_CHECK_IF(
        !GetPlatformInfo(), OP_LOGE(context_->GetNodeName(), "GetPlatformInfo falied, return false"), return false);
    OP_CHECK_IF(!GetAttrs(), OP_LOGE(context_->GetNodeName(), "GetAttrs falied, return false"), return false);
    return true;
}

bool AddLayerNormQuantRegbaseTiling::GetShapeInfo()
{
    OP_CHECK_IF(
        (!CheckTensorAndAttr()), OP_LOGE(context_->GetNodeName(), "Check tensor shape and attr failed."), return false);
    OP_CHECK_IF(
        (!CheckOptionalTensor()), OP_LOGE(context_->GetNodeName(), "Check optional tensor shape failed."),
        return false);
    this->dataTypeX1_ = context_->GetInputTensor(X1_IDX)->GetDataType();
    this->dtSizeX1_ = GetSizeByDataType(this->dataTypeX1_);
    auto xShape = context_->GetInputShape(X1_IDX)->GetStorageShape();
    auto gammaShape = context_->GetInputShape(GAMMA_IDX)->GetStorageShape();
    auto biasShape = context_->GetOptionalInputShape(BIAS_IDX);
    size_t xDimNum = xShape.GetDimNum();
    size_t gammaDimNum = gammaShape.GetDimNum();
    // LayerNorm have COUNT(bias,gamma) = 2 weights.
    this->weightTensorNums_ = 2;
    if (nullptr == biasShape) {
        this->biasType_ = BIAS_TYPE::NO_BIAS;
    } else {
        size_t biasDimNum = biasShape->GetStorageShape().GetDimNum();
        this->biasType_ = (biasDimNum == gammaDimNum) ? BIAS_TYPE::BROADCAST_BIAS : BIAS_TYPE::ELEWISE_BIAS;
        this->weightTensorNums_ += (biasDimNum == 1) ? 1 : 0;
    }

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
    this->colsAligned_ =
        Ops::Base::CeilDiv(this->cols_, static_cast<int64_t>(BLOCK_SIZE)) * BLOCK_SIZE; // 32 element aligned
    this->avgFactor_ = 1.0f / (static_cast<float>(this->cols_));

    OP_LOGW(
        "GetShapeInfo", "[M, N] = [%ld, %ld], dtSizeX1=%lu, avgFactor_=%f", this->rows_, this->cols_, this->dtSizeX1_,
        this->avgFactor_);

    if (this->isDynamicQuant_) {
        OP_CHECK_IF(
            !GetShapeInfoDynamicQuant(), OP_LOGE(context_->GetNodeName(), "Check tensor shape failed."), return false);
    } else {
        OP_CHECK_IF(
            !GetShapeInfoStaticQuant(), OP_LOGE(context_->GetNodeName(), "Check tensor shape failed."), return false);
    }
    return true;
}

bool AddLayerNormQuantRegbaseTiling::GetShapeInfoStaticQuant()
{
    OP_CHECK_IF(
        !(this->scale1Exist_), OP_LOGE(context_->GetNodeName(), "Static Quant required at least one scale."),
        return false);

    this->quantTensorNums_ = 1;

    /*
        s1  s2  z1  z2      ok
        T   T   T   T       T
        T   T   T   F       F
        T   T   F   T       F
        T   T   F   F       T
        T   F   T   T       F
        T   F   T   F       T
        T   F   F   T       F
        T   F   F   F       T
        F   *   *   *       F
        (s2 && z1 && z2) | (s2 && !z1 && !z2) | (!s2 && z1 && !z2) | (!s2 && !z1 && !z2)
    */
    bool validComb = (this->scale2Exist_ && this->offset1Exist_ && this->offset2Exist_) ||
                     (this->scale2Exist_ && !this->offset1Exist_ && !this->offset2Exist_) ||
                     (!this->scale2Exist_ && this->offset1Exist_ && !this->offset2Exist_) ||
                     (!this->scale2Exist_ && !this->offset1Exist_ && !this->offset2Exist_);
    OP_CHECK_IF(
        !validComb,
        OP_LOGE(
            context_->GetNodeName(), "Bad scale_offset comb, get [s1, s2, z1, z2] = [1, %d, %d, %d]",
            this->scale2Exist_, this->offset1Exist_, this->offset2Exist_),
        return false);

    this->quantTensorNums_ = ConditionalIncreaseOne(this->quantTensorNums_, this->scale2Exist_);
    this->quantTensorNums_ = ConditionalIncreaseOne(this->quantTensorNums_, this->offset1Exist_);
    this->quantTensorNums_ = ConditionalIncreaseOne(this->quantTensorNums_, this->offset2Exist_);
    this->outQuantNums_ = (this->scale2Exist_) ? QUANT_OUT_NUMS_DUAL : QUANT_OUT_NUMS_SOLE;

    OP_LOGW(
        "GetShapeInfoStaticQuant", "quantTensorNums=%ld, outQuantNums: %ld", this->quantTensorNums_,
        this->outQuantNums_);
    return true;
}

bool AddLayerNormQuantRegbaseTiling::GetShapeInfoDynamicQuant()
{
    this->quantTensorNums_ = 0;

    OP_CHECK_IF(
        (this->offset1Exist_ || this->offset2Exist_),
        OP_LOGE(context_->GetNodeName(), "Asymmetric quant not support in DynamicQuant yet."), return false);

    /*
        s1  s2  z1  z2      ok
        T   T   F   F       T
        T   F   F   F       T
        F   T   F   T       F
        F   F   F   F       T
        *   *   T   T       F
        (s1 && s2) || (s1 && !s2) || (!s1 && !s2)
    */
    bool validComb = (this->scale1Exist_ && this->scale2Exist_) || (this->scale1Exist_ && !this->scale2Exist_) ||
                     (!this->scale1Exist_ && !this->scale2Exist_);
    OP_CHECK_IF(
        !validComb,
        OP_LOGE(
            context_->GetNodeName(), "Bad scale_offset comb, get [s1, s2] = [%d, %d]", this->scale1Exist_,
            this->scale2Exist_),
        return false);

    this->quantTensorNums_ = ConditionalIncreaseOne(this->quantTensorNums_, this->scale1Exist_);
    this->quantTensorNums_ = ConditionalIncreaseOne(this->quantTensorNums_, this->scale2Exist_);
    this->outQuantNums_ = (this->scale2Exist_) ? QUANT_OUT_NUMS_DUAL : QUANT_OUT_NUMS_SOLE;

    OP_LOGW(
        "GetShapeInfoDynamicQuant", "quantTensorNums=%ld, outQuantNums: %ld", this->quantTensorNums_,
        this->outQuantNums_);
    return true;
}

bool AddLayerNormQuantRegbaseTiling::DoUbTiling()
{
    this->bufferNum_ = CONST_2;
    if (this->isDynamicQuant_) {
        OP_CHECK_IF(
            CheckDynQuantFullLoadTiling(), OP_LOGW(context_->GetNodeName(), "Ub Tiling: FullLoad."), return true);
        OP_CHECK_IF(CheckDynQuantWelfordTiling(), OP_LOGW(context_->GetNodeName(), "Ub Tiling: WelFord."), return true);
        return false;
    } else {
        OP_CHECK_IF(
            CheckStcQuantFullLoadTiling(), OP_LOGW(context_->GetNodeName(), "Ub Tiling: FullLoad."), return true);
        OP_CHECK_IF(CheckStcQuantWelfordTiling(), OP_LOGW(context_->GetNodeName(), "Ub Tiling: WelFord."), return true);
        return false;
    }
}

bool AddLayerNormQuantRegbaseTiling::CheckDynQuantFullLoadTiling()
{
    int64_t blkFp32Nums = BLOCK_SIZE / sizeof(float);
    int64_t tmpBinaryAddNum = (this->cols_ > this->vlFp32_) ? FindFloorPowerTwo(this->cols_) : this->vlFp32_;

    int64_t binaryAddUbSize =
        Ops::Base::CeilDiv((tmpBinaryAddNum / this->vlFp32_), blkFp32Nums) * blkFp32Nums * sizeof(float);
    int64_t quantBufSize = this->colsAligned_ * this->quantTensorNums_ * this->dtSizeScale_;
    int64_t weightBufSize = this->colsAligned_ * this->weightTensorNums_ * this->dtSizeX1_;

    int64_t ubAvaliable =
        static_cast<int64_t>(this->ubSize_) - (binaryAddUbSize + quantBufSize + weightBufSize + UB_RESERVED_BYTE);

    // ele queues
    int64_t x1x2InCols = this->dtSizeX1_ * 1 + this->dtSizeX1_ * 1;
    int64_t y1y2OutCols = sizeof(int8_t) * this->outQuantNums_;
    int64_t xOutCols = this->dtSizeX1_ * 1;
    int64_t biasInCols = (this->biasType_ == BIAS_TYPE::ELEWISE_BIAS) ? (this->dtSizeX1_ * 1) : 0;
    int64_t inOutCols = (x1x2InCols + y1y2OutCols + xOutCols + biasInCols) * this->bufferNum_ * this->colsAligned_;
    // reduce queues
    int64_t outScalePoints = this->bufferNum_ * sizeof(float) * this->outQuantNums_;
    // Tbufs
    int64_t tmpXBufCols = sizeof(float) * this->colsAligned_ * this->outQuantNums_;
    int64_t tmpMeanRstdPoints = sizeof(float) * 1 + sizeof(float) * 1;

    // meanBuf and rstdBuf
    int64_t rowFactor = inOutCols + outScalePoints + tmpXBufCols + tmpMeanRstdPoints;
    int64_t rowStep = ubAvaliable / rowFactor;

    bool ret = (rowStep >= 1);
    OP_LOGW(
        "CheckDynQuantFullLoadTiling", "x1x2InCols=%ld, y1y2OutCols: %ld, xOutCols: %ld, biasInCols: %ld", x1x2InCols,
        y1y2OutCols, xOutCols, biasInCols);
    OP_LOGW(
        "CheckDynQuantFullLoadTiling", "outScalePoints=%ld, tmpXBufCols: %ld, tmpMeanRstdPoints: %ld", outScalePoints,
        tmpXBufCols, tmpMeanRstdPoints);

    OP_LOGW(
        "CheckDynQuantFullLoadTiling", "ubAvaliable=%ld, binaryAddUbSize: %ld, quantBufSize: %ld, weightBufSize: %ld",
        ubAvaliable, binaryAddUbSize, quantBufSize, weightBufSize);
    OP_LOGW(
        "CheckDynQuantFullLoadTiling", "inOutCols=%ld, rowFactor=%ld, rowStep: %ld, ret: %d", inOutCols, rowFactor,
        rowStep, ret);
    if (ret) {
        this->rowsPerLoop_ = (rowStep <= this->rowsPerCore_) ? rowStep : this->rowsPerCore_;
        this->rowsPerLoop_ = (this->rowsPerLoop_ <= MAX_ROW_STEP) ? this->rowsPerLoop_ : MAX_ROW_STEP;
        this->colsPerLoop_ = this->cols_;
        this->colsLoopCount_ = 1;
        this->colsTail_ = this->colsPerLoop_;

        this->binaryAddNum_ = tmpBinaryAddNum;
        ComputeBinaryAddVars();
        this->ubTilingPolicy_ = UB_TILING_POLICY::FULL_LOAD;
    }
    return ret;
}

bool AddLayerNormQuantRegbaseTiling::CheckDynQuantWelfordTiling()
{
    int64_t quantSliceNums = this->bufferNum_ * this->dtSizeScale_ * this->quantTensorNums_;
    int64_t weightSliceNums = this->bufferNum_ * this->dtSizeX1_ * this->weightTensorNums_;
    // COUNT(x1,x2) = 2, COUNT(y1,y2) = 2, COUNT(x,) = 1
    int64_t elewiseSliceNums =
        this->bufferNum_ * this->dtSizeX1_ * (2 + 1) + this->bufferNum_ * sizeof(int8_t) * this->outQuantNums_;
    elewiseSliceNums += (this->biasType_ == BIAS_TYPE::ELEWISE_BIAS) ? (this->bufferNum_ * this->dtSizeX1_ * 1) : 0;
    // COUNT(meanBuf,varBuf) = 2, binaryAddBuf <= colSliceLen.
    int64_t tmpSliceNums = sizeof(float) * (this->bufferNum_ * this->outQuantNums_ + 1);

    // COUNT(tmpMean, tmpRstd, tmpMax1, tmpMax2) = 4
    int64_t constTmpBufSize = this->bufferNum_ * this->outQuantNums_ * BLOCK_SIZE + BLOCK_SIZE * 4;
    int64_t ubAvaliable = static_cast<int64_t>(this->ubSize_) - UB_RESERVED_BYTE - constTmpBufSize;

    this->colsPerLoop_ = ubAvaliable / (quantSliceNums + weightSliceNums + elewiseSliceNums + tmpSliceNums);
    this->colsPerLoop_ = this->colsPerLoop_ / this->vlFp32_ * this->vlFp32_;
    this->colsLoopCount_ = Ops::Base::CeilDiv(this->cols_, this->colsPerLoop_);

    // try to use aligned welford finalize process for better prof...
    this->colsPerLoop_ =
        (this->cols_ % this->colsLoopCount_ == 0) ? (this->cols_ / this->colsLoopCount_) : this->colsPerLoop_;

    this->colsTail_ = this->cols_ % this->colsPerLoop_;
    this->colsTail_ = (this->colsTail_ == 0) ? this->colsPerLoop_ : this->colsTail_;

    this->binaryAddNum_ = (this->colsPerLoop_ > this->vlFp32_) ? FindFloorPowerTwo(this->colsPerLoop_) : this->vlFp32_;
    ComputeBinaryAddVars();
    this->ubTilingPolicy_ = UB_TILING_POLICY::WELFORD;

    OP_LOGW(
        "CheckDynQuantWelfordTiling",
        "quantSliceNums=%ld, weightSliceNums: %ld, elewiseSliceNums: %ld, tmpSliceNums: %ld", quantSliceNums,
        weightSliceNums, elewiseSliceNums, tmpSliceNums);
    OP_LOGW("CheckDynQuantWelfordTiling", "ubAvaliable=%ld", ubAvaliable);
    return true;
}

bool AddLayerNormQuantRegbaseTiling::CheckStcQuantFullLoadTiling()
{
    int64_t blkFp32Nums = BLOCK_SIZE / sizeof(float);
    int64_t tmpBinaryAddNum = (this->cols_ > this->vlFp32_) ? FindFloorPowerTwo(this->cols_) : this->vlFp32_;

    int64_t binaryAddUbSize =
        Ops::Base::CeilDiv((tmpBinaryAddNum / this->vlFp32_), blkFp32Nums) * blkFp32Nums * sizeof(float);
    int64_t quantBufSize = this->colsAligned_ * this->quantTensorNums_ * this->dtSizeScale_;
    int64_t weightBufSize = this->colsAligned_ * this->weightTensorNums_ * this->dtSizeX1_;

    int64_t ubAvaliable =
        static_cast<int64_t>(this->ubSize_) - (binaryAddUbSize + quantBufSize + weightBufSize + UB_RESERVED_BYTE);
    // COUNT(x1,x2) = 2, COUNT(y1,y2) = 2, COUNT(x,) = 1
    int64_t inOutCols =
        (this->dtSizeX1_ * 2 + sizeof(int8_t) * 2 + this->dtSizeX1_ * 1) * this->bufferNum_ * this->colsAligned_;
    inOutCols += (this->biasType_ == BIAS_TYPE::ELEWISE_BIAS) ?
                     (this->dtSizeX1_ * 1 * this->bufferNum_ * this->colsAligned_) :
                     0;
    int64_t tmpBufsCols = sizeof(float) * this->colsAligned_ + sizeof(float) * 1 + sizeof(float) * 1;
    int64_t rowFactor = inOutCols + tmpBufsCols;
    int64_t rowStep = ubAvaliable / rowFactor;

    bool ret = (rowStep >= 1);
    OP_LOGW(
        "CheckStcQuantFullLoadTiling", "ubAvaliable=%ld, binaryAddUbSize: %ld, quantBufSize: %ld, weightBufSize: %ld",
        ubAvaliable, binaryAddUbSize, quantBufSize, weightBufSize);
    OP_LOGW(
        "CheckStcQuantFullLoadTiling", "inOutCols=%ld, tmpBufsCols=%ld, rowFactor=%ld, rowStep: %ld, ret: %d",
        inOutCols, tmpBufsCols, rowFactor, rowStep, ret);
    if (ret) {
        this->rowsPerLoop_ = (rowStep <= this->rowsPerCore_) ? rowStep : this->rowsPerCore_;
        this->rowsPerLoop_ = (this->rowsPerLoop_ <= MAX_ROW_STEP) ? this->rowsPerLoop_ : MAX_ROW_STEP;
        this->colsPerLoop_ = this->cols_;
        this->colsLoopCount_ = 1;
        this->colsTail_ = this->colsPerLoop_;

        this->binaryAddNum_ = tmpBinaryAddNum;
        ComputeBinaryAddVars();
        this->ubTilingPolicy_ = UB_TILING_POLICY::FULL_LOAD;
    }
    return ret;
}

bool AddLayerNormQuantRegbaseTiling::CheckStcQuantWelfordTiling()
{
    int64_t quantSliceNums = this->bufferNum_ * this->dtSizeScale_ * this->quantTensorNums_;
    int64_t weightSliceNums = this->bufferNum_ * this->dtSizeX1_ * this->weightTensorNums_;
    // COUNT(x1,x2) = 2, COUNT(y1,y2) = 2, COUNT(x,) = 1
    int64_t elewiseSliceNums = this->bufferNum_ * this->dtSizeX1_ * (2 + 1) + this->bufferNum_ * sizeof(int8_t) * 2;
    elewiseSliceNums += (this->biasType_ == BIAS_TYPE::ELEWISE_BIAS) ? (this->bufferNum_ * this->dtSizeX1_ * 1) : 0;
    // COUNT(meanBuf,varBuf) = 2, binaryAddBuf <= colSliceLen.
    int64_t tmpSliceNums = sizeof(float) * (2 + 1);

    int64_t ubAvaliable = static_cast<int64_t>(this->ubSize_) - UB_RESERVED_BYTE;

    this->colsPerLoop_ = ubAvaliable / (quantSliceNums + weightSliceNums + elewiseSliceNums + tmpSliceNums);
    this->colsPerLoop_ = this->colsPerLoop_ / this->vlFp32_ * this->vlFp32_;
    this->colsLoopCount_ = Ops::Base::CeilDiv(this->cols_, this->colsPerLoop_);

    // try to use aligned welford finalize process for better prof...
    this->colsPerLoop_ =
        (this->cols_ % this->colsLoopCount_ == 0) ? (this->cols_ / this->colsLoopCount_) : this->colsPerLoop_;

    this->colsTail_ = this->cols_ % this->colsPerLoop_;
    this->colsTail_ = (this->colsTail_ == 0) ? this->colsPerLoop_ : this->colsTail_;

    this->binaryAddNum_ = (this->colsPerLoop_ > this->vlFp32_) ? FindFloorPowerTwo(this->colsPerLoop_) : this->vlFp32_;
    ComputeBinaryAddVars();
    this->ubTilingPolicy_ = UB_TILING_POLICY::WELFORD;
    return true;
}

bool AddLayerNormQuantRegbaseTiling::CheckTensorAndAttr()
{
    // Check Shape Not NULL
    const gert::StorageShape* x1Shape = this->context_->GetInputShape(X1_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(this->context_, x1Shape);
    const gert::StorageShape* x2Shape = this->context_->GetInputShape(X2_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(this->context_, x2Shape);
    const gert::StorageShape* gammaShape = this->context_->GetInputShape(GAMMA_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(this->context_, gammaShape);
    const gert::StorageShape* betaShape = this->context_->GetInputShape(BETA_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(this->context_, betaShape);

    const gert::StorageShape* y1Shape = this->context_->GetOutputShape(Y1_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(this->context_, y1Shape);
    const gert::StorageShape* xShape = this->context_->GetOutputShape(X_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(this->context_, xShape);

    size_t elewiseDimNum = x1Shape->GetStorageShape().GetDimNum();
    size_t weightDimNum = gammaShape->GetStorageShape().GetDimNum();
    OP_CHECK_IF(
        (0 == elewiseDimNum || 0 == weightDimNum),
        OP_LOGW(this->context_->GetNodeName(), "Got x1/gamma is zero dim tensor, tiling FAILED."), return false);

    bool elewiseShapeEqual = ((*x1Shape) == (*x2Shape)) && ((*x1Shape) == (*xShape)) && ((*x1Shape) == (*y1Shape));
    bool weightShapeEqual = ((*gammaShape) == (*betaShape));
    OP_CHECK_IF(
        !(elewiseShapeEqual && weightShapeEqual),
        OP_LOGW(
            this->context_->GetNodeName(),
            "Got x1/x2/y1/x shape not equat OR gamma/beta Shape not equal, tiling FAILED."),
        return false);

    OP_CHECK_IF(HasZeroDim(x1Shape), OP_LOGE("CheckTensor", "Got x1 shapeSize = 0."), return false);
    OP_CHECK_IF(HasZeroDim(gammaShape), OP_LOGE("CheckTensor", "Got gamma shapeSize = 0."), return false);
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
    for (size_t  i = 0; i < diffDimNum; ++i) {
        if (optionalShape->GetStorageShape().GetDim(i) != 1) {
            return false;
        }
    }
    for (size_t  i = 0; i < gammaDimNum; ++i) {
        if (optionalShape->GetStorageShape().GetDim(i + diffDimNum) != gammaShape->GetStorageShape().GetDim(i)) {
            return false;
        }
    }
    return true;
}

bool AddLayerNormQuantRegbaseTiling::CheckOptionalTensor()
{
    const gert::StorageShape* x1Shape = this->context_->GetInputShape(X1_IDX);
    const gert::StorageShape* gammaShape = this->context_->GetInputShape(GAMMA_IDX);
    size_t elewiseDimNum = x1Shape->GetStorageShape().GetDimNum();
    size_t weightDimNum = gammaShape->GetStorageShape().GetDimNum();

    // bias shape should equal to x1 or gamma if exist.
    const gert::StorageShape* biasShape = context_->GetOptionalInputShape(BIAS_IDX);
    bool invalidBias = (nullptr != biasShape) && ((*biasShape) != (*x1Shape)) && ((*biasShape) != (*gammaShape));
    OP_CHECK_IF(
        invalidBias,
        OP_LOGE("CheckOptionalTensor", "Bias Shape neither equal to x1 shape nor gamma shape, tiling failed."),
        return false);

    const gert::StorageShape* scale1Shape = context_->GetOptionalInputShape(SCALE1_IDX);
    const gert::StorageShape* scale2Shape = context_->GetOptionalInputShape(SCALE2_IDX);
    const gert::StorageShape* offset1Shape = context_->GetOptionalInputShape(ZERO_POINT1_IDX);
    const gert::StorageShape* offset2Shape = context_->GetOptionalInputShape(ZERO_POINT2_IDX);

    this->scale1Exist_ = (nullptr != scale1Shape);
    this->scale2Exist_ = (nullptr != scale2Shape);
    this->offset1Exist_ = (nullptr != offset1Shape);
    this->offset2Exist_ = (nullptr != offset2Shape);

    OP_CHECK_IF(
        ((this->scale1Exist_) && (!checkOptionalShape(elewiseDimNum, gammaShape, scale1Shape))),
        OP_LOGE("CheckOptionalTensor", "Scale1 exist but its shape not equal to gamma shape, tiling failed."),
        return false);
    OP_CHECK_IF(
        ((this->scale2Exist_) && (!checkOptionalShape(elewiseDimNum, gammaShape, scale2Shape))),
        OP_LOGE("CheckOptionalTensor", "Scale2 exist but its shape not equal to gamma shape, tiling failed."),
        return false);
    OP_CHECK_IF(
        ((this->offset1Exist_) && (!checkOptionalShape(elewiseDimNum, gammaShape, offset1Shape))),
        OP_LOGE("CheckOptionalTensor", "ZeroPoints1 exist but its shape not equal to gamma shape, tiling failed."),
        return false);
    OP_CHECK_IF(
        ((this->offset2Exist_) && (!checkOptionalShape(elewiseDimNum, gammaShape, offset2Shape))),
        OP_LOGE("CheckOptionalTensor", "ZeroPoints2 exist but its shape not equal to gamma shape, tiling failed."),
        return false);

    const gert::Tensor* scale1Tensor = context_->GetOptionalInputTensor(SCALE1_IDX);
    this->dataTypeScale_ = (nullptr != scale1Tensor) ? scale1Tensor->GetDataType() : ge::DataType::DT_FLOAT;
    this->dtSizeScale_ = GetSizeByDataType(this->dataTypeScale_);

    const gert::StorageShape* y2Shape = this->context_->GetOutputShape(Y2_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(this->context_, y2Shape);
    OP_CHECK_IF(
        ((this->scale2Exist_) && (*y2Shape) != (*x1Shape)),
        OP_LOGE("CheckOptionalTensor", "y2 exist but its shape not equal to x1 shape, tiling failed."), return false);

    const gert::StorageShape* outScale1Shape = this->context_->GetOutputShape(OUT_SCALE1_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(this->context_, outScale1Shape);
    const gert::StorageShape* outScale2Shape = this->context_->GetOutputShape(OUT_SCALE2_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(this->context_, outScale2Shape);
    size_t outScaleDimNum = outScale1Shape->GetStorageShape().GetDimNum();
    OP_LOGI("CheckOptionalTensor", "outScaleDimNum=%zu", outScaleDimNum);
    if (this->isDynamicQuant_) {
        OP_CHECK_IF(
            (outScaleDimNum != elewiseDimNum - weightDimNum),
            OP_LOGE("CheckOptionalTensor",
                    "The dim num of x1 %zu should be equal to the sum of the dim num of weight %zu and outScale1 %zu.",
                    elewiseDimNum, weightDimNum, outScaleDimNum),
            return false);
        OP_CHECK_IF(
            (this->scale2Exist_) && ((*outScale1Shape) != (*outScale2Shape)),
            OP_LOGE("CheckOptionalTensor", "Got different outScale1 and outScale2, tiling Failed."), return false);

        for (size_t i = 0; i < outScaleDimNum; i++) {
            OP_CHECK_IF(
                (outScale1Shape->GetStorageShape().GetDim(i) != x1Shape->GetStorageShape().GetDim(i)),
                OP_LOGE("CheckOptionalTensor", "y2 exist but its shape not equal to x1 shape, tiling failed."),
                return false);
        }
    }
    return true;
}

} // namespace optiling
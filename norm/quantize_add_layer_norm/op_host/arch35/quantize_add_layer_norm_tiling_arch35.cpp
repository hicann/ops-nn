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
#include "quantize_add_layer_norm_tiling_arch35.h"
#include "../../op_kernel/arch35/quantize_add_layer_norm_tiling_key.h"
#include <cinttypes>
#include <cmath>
#include <limits>
#include <vector>
#include "securec.h"

namespace optiling {
// Framework-contract failures use EZ1008 instead of falling back to the internal EZ9999 code.
static void ReportTilingStageError(const char* opName, const std::string& reason)
{
    const std::vector<const char*> msgKey = {"opName", "reason"};
    const std::vector<const char*> msgValue = {opName != nullptr ? opName : "QuantizeAddLayerNorm", reason.c_str()};
    REPORT_PREDEFINED_ERR_MSG("EZ1008", msgKey, msgValue);
}

constexpr int32_t BINARY_REDUCTION_RADIX = 2;
constexpr int32_t BIT_SPREAD_SHIFT_2 = 2;
constexpr int32_t BIT_SPREAD_SHIFT_4 = 4;
constexpr int32_t BIT_SPREAD_SHIFT_8 = 8;
constexpr int32_t BIT_SPREAD_SHIFT_16 = 16;
constexpr int32_t BIT_SPREAD_SHIFT_32 = 32;
constexpr uint64_t KERNEL_BUFFER_NUM = 1;

constexpr uint32_t BLOCK_SIZE = 32;
constexpr uint64_t UB_RESERVED_BYTE = 256;
constexpr int32_t MAX_ROW_STEP = 255;
constexpr int64_t MAX_FULL_LOAD_BYTES_PER_COLUMN = 32;
constexpr int64_t BASE_WEIGHT_TENSOR_COUNT = 2;
constexpr int64_t RESIDUAL_INPUT_COUNT = 2;
constexpr int64_t MEAN_VAR_BUFFER_COUNT = 2;

constexpr size_t DEFAULT_WORKSPACE_SIZE = 32;

// axis magic numbers (same as 910b tiling)
constexpr int64_t AXIS_VALUE_FOR_PER_TENSOR = 65535;
constexpr int64_t AXIS_VALUE_FOR_MUL_DIV_MODE = -65535;

static int64_t inline FindFloorPowerTwo(int64_t num)
{
    if (num <= 1) {
        return 0;
    }
    uint64_t n = static_cast<uint64_t>(num - 1);
    n |= n >> 1;
    n |= n >> BIT_SPREAD_SHIFT_2;
    n |= n >> BIT_SPREAD_SHIFT_4;
    n |= n >> BIT_SPREAD_SHIFT_8;
    n |= n >> BIT_SPREAD_SHIFT_16;
    n |= n >> BIT_SPREAD_SHIFT_32;
    return static_cast<int64_t>((n + 1) >> 1);
}

static bool CheckedMulNonNegative(int64_t lhs, int64_t rhs, int64_t& result)
{
    if (lhs < 0 || rhs < 0 || (lhs != 0 && rhs > std::numeric_limits<int64_t>::max() / lhs)) {
        return false;
    }
    result = lhs * rhs;
    return true;
}

static inline bool HasZeroDim(const gert::StorageShape* s)
{
    if (s == nullptr) {
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
            curBinaryAddNum *= BINARY_REDUCTION_RADIX;
        }
        this->binaryAddLastNum_ = this->vlFp32_;
    }
    OP_LOGW(context_, "binaryAddNum:%" PRId64 ", binaryAddK:%" PRId64 ", binaryAddLastNum:%" PRId64,
            this->binaryAddNum_, this->binaryAddK_, this->binaryAddLastNum_);
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
    this->ubTilingPolicy_ = UbTilingPolicy::FULL_LOAD;
}

bool QuantizeAddLayerNormRegbaseTiling::SetTilingDataAndTilingKeyAndWorkSpace(
    QuantizeAddLayerNormRegbaseTilingData* tiling)
{
    OP_CHECK_IF(tiling == nullptr, OP_LOGE(context_->GetNodeName(), "tiling data is nullptr"), return false);
    tiling->rowsPerCore = this->rowsPerCore_;
    tiling->rowsPerTailCore = this->rowsPerTailCore_;
    tiling->rowsPerLoop = this->rowsPerLoop_;
    tiling->cols = this->cols_;
    tiling->colsPerLoop = this->colsPerLoop_;
    tiling->colsLoopCount = this->colsLoopCount_;
    tiling->colsTail = this->colsTail_;
    tiling->binaryAddNum = this->binaryAddNum_;
    tiling->binaryAddK = this->binaryAddK_;
    tiling->binaryAddLastNum = this->binaryAddLastNum_;
    tiling->eps = this->eps_;
    tiling->outputX = static_cast<uint32_t>(this->needOutputX_);

    uint64_t tilingKey = GET_TPL_TILING_KEY(static_cast<uint64_t>(this->ubTilingPolicy_ == UbTilingPolicy::WELFORD),
                                            static_cast<uint64_t>(this->biasType_ == BiasType::BROADCAST_BIAS),
                                            static_cast<uint64_t>(this->quantMode_), 0, 0);

    // static quant: fixed small workspace, no per-token outScale workspace, no SyncAll
    size_t usrWorkspaceSize = DEFAULT_WORKSPACE_SIZE;

    auto rawTilingData = context_->GetRawTilingData();
    OP_CHECK_IF(rawTilingData == nullptr, OP_LOGE(context_->GetNodeName(), "raw tiling data is nullptr"), return false);
    OP_CHECK_IF(rawTilingData->GetData() == nullptr, OP_LOGE(context_->GetNodeName(), "raw tiling buffer is nullptr"),
                return false);
    constexpr size_t tilingDataSize = sizeof(QuantizeAddLayerNormRegbaseTilingData);
    OP_CHECK_IF(rawTilingData->GetCapacity() < tilingDataSize,
                OP_LOGE(context_->GetNodeName(), "raw tiling capacity %zu is smaller than required size %zu",
                        rawTilingData->GetCapacity(), tilingDataSize),
                return false);
    size_t* currentWorkspace = context_->GetWorkspaceSizes(1);
    OP_CHECK_IF(currentWorkspace == nullptr, OP_LOGE(context_->GetNodeName(), "workspace output is nullptr"),
                return false);

    ge::graphStatus tilingKeyStatus = context_->SetTilingKey(tilingKey);
    OP_CHECK_IF(tilingKeyStatus != ge::GRAPH_SUCCESS,
                OP_LOGE(context_->GetNodeName(), "SetTilingKey failed for key=%" PRIu64 ", status=%u.", tilingKey,
                        static_cast<uint32_t>(tilingKeyStatus)),
                return false);
    ge::graphStatus blockDimStatus = context_->SetBlockDim(this->usedCoreNum_);
    OP_CHECK_IF(blockDimStatus != ge::GRAPH_SUCCESS,
                OP_LOGE(context_->GetNodeName(), "SetBlockDim failed for blockDim=%u, status=%u.", this->usedCoreNum_,
                        static_cast<uint32_t>(blockDimStatus)),
                return false);
    errno_t copyStatus = memcpy_s(rawTilingData->GetData(), rawTilingData->GetCapacity(), tiling, tilingDataSize);
    OP_CHECK_IF(copyStatus != EOK,
                OP_LOGE(context_->GetNodeName(),
                        "Failed to serialize tiling data: status=%d, requiredBytes=%zu, availableBytes=%zu.",
                        copyStatus, tilingDataSize, rawTilingData->GetCapacity()),
                return false);
    rawTilingData->SetDataSize(tilingDataSize);
    currentWorkspace[0] = usrWorkspaceSize;

    OP_LOGI(context_, "Tiling data: tilingKey=%" PRIu64 ", userWorkspace=%zu bytes.", tilingKey, usrWorkspaceSize);
    OP_LOGI(context_,
            "usedCoreNum:%u, vlFp32:%u, rowsPerCore:%" PRId64 ", rowsPerTailCore:%" PRId64 ", rowsPerLoop:%" PRId64
            ", cols:%" PRId64 ", colsPerLoop:%" PRId64 ", colsLoopCount:%" PRId64 ", colsTail:%" PRId64,
            this->usedCoreNum_, this->vlFp32_, this->rowsPerCore_, this->rowsPerTailCore_, this->rowsPerLoop_,
            this->cols_, this->colsPerLoop_, this->colsLoopCount_, this->colsTail_);
    return true;
}

bool QuantizeAddLayerNormRegbaseTiling::DoTiling()
{
    OP_CHECK_IF(context_ == nullptr,
                ReportTilingStageError("QuantizeAddLayerNorm",
                                       "the TilingContext is null; check the tiling framework invocation"),
                return false);
    OP_CHECK_IF(!GetBaseInfo(), OP_LOGE_WITHOUT_REPORT(context_, "GetBaseInfo failed."), return false);
    OP_CHECK_IF(!GetShapeInfo(), OP_LOGE_WITHOUT_REPORT(context_, "GetShapeInfo failed."), return false);
    OP_CHECK_IF(!CheckDtype(), OP_LOGE_WITHOUT_REPORT(context_, "CheckDtype failed."), return false);
    OP_CHECK_IF(!DoBlockTiling(), OP_LOGE_WITHOUT_REPORT(context_, "DoBlockTiling failed."), return false);
    OP_CHECK_IF(!DoUbTiling(), OP_LOGE_WITHOUT_REPORT(context_, "DoUbTiling failed."), return false);

    QuantizeAddLayerNormRegbaseTilingData tiling{};
    OP_CHECK_IF(!SetTilingDataAndTilingKeyAndWorkSpace(&tiling),
                OP_LOGE_WITHOUT_REPORT(context_, "Set tiling outputs failed."), return false);
    OP_LOGW(context_->GetNodeName(), "DoTiling completed.");
    return true;
}

bool QuantizeAddLayerNormRegbaseTiling::DoBlockTiling()
{
    // Block Tiling, Cut N (rows / M axis)
    this->rowsPerCore_ = Ops::Base::CeilDiv(this->rows_, static_cast<int64_t>(this->aivCoreNum_));
    this->usedCoreNum_ = Ops::Base::CeilDiv(this->rows_, this->rowsPerCore_);
    this->rowsPerCore_ = Ops::Base::CeilDiv(this->rows_, static_cast<int64_t>(this->usedCoreNum_));
    this->rowsPerTailCore_ = this->rows_ - this->rowsPerCore_ * (this->usedCoreNum_ - 1);
    OP_LOGW(context_, "usedCoreNum: %u, rowsPerCore: %" PRId64 ", rowsPerTailCore: %" PRId64, this->usedCoreNum_,
            this->rowsPerCore_, this->rowsPerTailCore_);
    return true;
}

bool QuantizeAddLayerNormRegbaseTiling::GetPlatformInfo()
{
    auto platformInfo = this->context_->GetPlatformInfo();
    OP_CHECK_IF(
        platformInfo == nullptr,
        ReportTilingStageError(
            context_->GetNodeName(),
            "GetPlatformInfo returned null; check the TilingContext construction and the platform registration"),
        return false);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, this->ubSize_);
    this->aivCoreNum_ = ascendcPlatform.GetCoreNumAiv();
    this->vecRegSize_ = ascendcPlatform.GetVecRegLen();
    this->vlFp32_ = this->vecRegSize_ / sizeof(float);

    OP_CHECK_IF(
        this->ubSize_ == 0,
        ReportTilingStageError(context_->GetNodeName(),
                               "GetCoreMemSize(UB) returned 0; check the platform info, SoC and OPP installation"),
        return false);
    OP_CHECK_IF(this->ubSize_ > static_cast<uint64_t>(std::numeric_limits<int64_t>::max()),
                OP_LOGE(context_, "UB size %" PRIu64 " exceeds the supported int64 range.", this->ubSize_),
                return false);
    OP_CHECK_IF(
        this->aivCoreNum_ == 0,
        ReportTilingStageError(context_->GetNodeName(), "GetCoreNumAiv returned 0; check the platform info and SoC"),
        return false);
    OP_CHECK_IF(
        this->vecRegSize_ == 0 || this->vlFp32_ == 0,
        ReportTilingStageError(context_->GetNodeName(), "GetVecRegLen returned 0; check the platform info and SoC"),
        return false);

    OP_LOGW(context_, "aivCoreNum: %u, ubSize: %" PRIu64 ", vecRegSize: %u", this->aivCoreNum_, this->ubSize_,
            this->vecRegSize_);
    return true;
}

bool QuantizeAddLayerNormRegbaseTiling::GetAttrs()
{
    auto attrs = context_->GetAttrs();
    OP_CHECK_IF(
        attrs == nullptr,
        ReportTilingStageError(context_->GetNodeName(),
                               "GetShapeAttrsInfo: GetAttrs() returned null; check the TilingContext construction"),
        return false);

    const int64_t* dtype = attrs->GetInt(DTYPE_IDX);
    OP_CHECK_IF(dtype == nullptr, OP_LOGE(context_->GetNodeName(), "Failed to get the required dtype attribute."),
                return false);
    OP_CHECK_IF(*dtype != static_cast<int64_t>(ge::DT_INT8),
                OP_LOGE_FOR_INVALID_VALUE(context_->GetNodeName(), "dtype", std::to_string(*dtype).c_str(), "DT_INT8"),
                return false);

    // Keep the legacy arch22 compatibility mapping for internal callers.
    int64_t axis = GetOptionalAttr<int64_t>(attrs, AXIS_IDX, -1);
    if (axis == AXIS_VALUE_FOR_PER_TENSOR) {
        this->quantMode_ = QuantMode::PER_TENSOR;
    } else if (axis == AXIS_VALUE_FOR_MUL_DIV_MODE) {
        this->quantMode_ = QuantMode::MUL_MODE;
    } else {
        this->quantMode_ = QuantMode::PER_CHANNEL;
    }

    this->eps_ = GetOptionalAttr<float>(attrs, EPS_IDX, (float)1e-5);
    this->needOutputX_ = GetOptionalAttr<bool>(attrs, X_OUT_ATTR_IDX, false);

    OP_CHECK_IF(this->eps_ <= 0.0F,
                OP_LOGE_FOR_INVALID_VALUE(context_->GetNodeName(), "epsilon", std::to_string(this->eps_).c_str(),
                                          "greater than zero"),
                return false);
    OP_CHECK_IF(!std::isfinite(this->eps_),
                OP_LOGE_FOR_INVALID_VALUE(context_->GetNodeName(), "epsilon", std::to_string(this->eps_).c_str(),
                                          "a finite value"),
                return false);

    OP_LOGW(context_, "axis=%" PRId64 ", quantMode=%d, eps=%f, xOut=%d", axis, static_cast<int>(this->quantMode_),
            this->eps_, this->needOutputX_);
    return true;
}

bool QuantizeAddLayerNormRegbaseTiling::GetBaseInfo()
{
    OP_CHECK_IF(!GetPlatformInfo(), OP_LOGE_WITHOUT_REPORT(context_, "GetPlatformInfo failed."), return false);
    OP_CHECK_IF(!GetAttrs(), OP_LOGE_WITHOUT_REPORT(context_, "GetAttrs failed."), return false);
    return true;
}

bool QuantizeAddLayerNormRegbaseTiling::GetShapeInfo()
{
    OP_CHECK_IF(!CheckTensorAndAttr(), OP_LOGE_WITHOUT_REPORT(context_, "Check tensor shape and attr failed."),
                return false);
    OP_CHECK_IF(!CheckOptionalTensor(), OP_LOGE_WITHOUT_REPORT(context_, "Check optional tensor shape failed."),
                return false);
    auto x1Desc = context_->GetInputDesc(X1_IDX);
    OP_CHECK_IF(x1Desc == nullptr, OP_LOGE(context_->GetNodeName(), "x1 desc is nullptr"), return false);
    this->dataTypeX1_ = x1Desc->GetDataType();
    this->dtSizeX1_ = GetSizeByDataType(this->dataTypeX1_);
    auto xShape = context_->GetInputShape(X1_IDX)->GetStorageShape();
    auto gammaShape = context_->GetInputShape(GAMMA_IDX)->GetStorageShape();
    auto biasShape = context_->GetInputShape(BIAS_IDX)->GetStorageShape();
    size_t xDimNum = xShape.GetDimNum();
    size_t gammaDimNum = gammaShape.GetDimNum();
    // weights = gamma + beta + (bias if 1D)
    this->weightTensorNums_ = BASE_WEIGHT_TENSOR_COUNT;
    size_t biasDimNum = biasShape.GetDimNum();
    this->biasType_ = (biasDimNum == gammaDimNum) ? BiasType::BROADCAST_BIAS : BiasType::ELEMENTWISE_BIAS;
    this->weightTensorNums_ += (biasDimNum == 1) ? 1 : 0;

    int64_t numRow = 1;
    for (size_t i = 0; i < xDimNum - gammaDimNum; i++) {
        int64_t nextNumRow = 0;
        OP_CHECK_IF(!CheckedMulNonNegative(numRow, xShape.GetDim(i), nextNumRow) || nextNumRow == 0,
                    OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                        context_->GetNodeName(), "x1", Ops::Base::ToString(xShape).c_str(),
                        "all dimensions and the row product must be positive and fit in int64"),
                    return false);
        numRow = nextNumRow;
    }
    int64_t numCol = 1;
    for (size_t i = 0; i < gammaDimNum; i++) {
        int64_t nextNumCol = 0;
        OP_CHECK_IF(!CheckedMulNonNegative(numCol, gammaShape.GetDim(i), nextNumCol) || nextNumCol == 0,
                    OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                        context_->GetNodeName(), "gamma", Ops::Base::ToString(gammaShape).c_str(),
                        "all dimensions and the column product must be positive and fit in int64"),
                    return false);
        numCol = nextNumCol;
    }
    this->rows_ = numRow;
    this->cols_ = numCol;
    int64_t alignedUnits = Ops::Base::CeilDiv(this->cols_, static_cast<int64_t>(this->vlFp32_));
    OP_CHECK_IF(
        !CheckedMulNonNegative(alignedUnits, static_cast<int64_t>(this->vlFp32_), this->colsAligned_),
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context_->GetNodeName(), "gamma", Ops::Base::ToString(gammaShape).c_str(),
                                              "the aligned column count must fit in int64"),
        return false);
    this->avgFactor_ = 1.0f / (static_cast<float>(this->cols_));

    // static quant: quant tensors = scales(always) + zero_points(if exist)
    this->quantTensorNums_ = 1 + (this->offsetExist_ ? 1 : 0);

    OP_LOGW(context_,
            "[M, N] = [%" PRId64 ", %" PRId64 "], dtSizeX1=%" PRId64 ", dtSizeScale=%" PRId64
            ", avgFactor_=%f, quantTensorNums=%" PRId64 ", "
            "weightTensorNums=%" PRId64 ", biasType=%d",
            this->rows_, this->cols_, this->dtSizeX1_, this->dtSizeScale_, this->avgFactor_, this->quantTensorNums_,
            this->weightTensorNums_, static_cast<int>(this->biasType_));
    return true;
}

bool QuantizeAddLayerNormRegbaseTiling::DoUbTiling()
{
    this->bufferNum_ = KERNEL_BUFFER_NUM;
    OP_CHECK_IF(CheckStcQuantFullLoadTiling(), OP_LOGW(context_->GetNodeName(), "Ub Tiling: FullLoad."), return true);
    OP_CHECK_IF(CheckStcQuantWelfordTiling(), OP_LOGW(context_->GetNodeName(), "Ub Tiling: WelFord."), return true);
    return false;
}

bool QuantizeAddLayerNormRegbaseTiling::CheckStcQuantFullLoadTiling()
{
    OP_CHECK_IF(this->colsAligned_ <= 0 || static_cast<uint64_t>(this->colsAligned_) > this->ubSize_ ||
                    this->colsAligned_ > std::numeric_limits<int64_t>::max() / MAX_FULL_LOAD_BYTES_PER_COLUMN,
                OP_LOGW(context_, "FullLoad columns do not fit in UB."), return false);
    int64_t blkFp32Nums = BLOCK_SIZE / sizeof(float);
    int64_t tmpBinaryAddNum = (this->cols_ > this->vlFp32_) ? FindFloorPowerTwo(this->cols_) : this->vlFp32_;

    int64_t binaryAddUbSize = Ops::Base::CeilDiv((tmpBinaryAddNum / this->vlFp32_), blkFp32Nums) * blkFp32Nums *
                              sizeof(float);
    int64_t quantBufSize = this->colsAligned_ * this->quantTensorNums_ * this->dtSizeScale_;
    int64_t weightBufSize = this->colsAligned_ * this->weightTensorNums_ * this->dtSizeX1_;

    int64_t ubAvailable = static_cast<int64_t>(this->ubSize_) -
                          (binaryAddUbSize + quantBufSize + weightBufSize + UB_RESERVED_BYTE);
    // Inputs are x1/x2; outputs are y/x.
    int64_t inOutCols = (this->dtSizeX1_ * RESIDUAL_INPUT_COUNT + sizeof(int8_t) + this->dtSizeX1_) * this->bufferNum_ *
                        this->colsAligned_;
    inOutCols += (this->biasType_ == BiasType::ELEMENTWISE_BIAS) ?
                     (this->dtSizeX1_ * 1 * this->bufferNum_ * this->colsAligned_) :
                     0;
    int64_t tmpBufsCols = sizeof(float) * this->colsAligned_ + sizeof(float) * 1 + sizeof(float) * 1;
    int64_t rowFactor = inOutCols + tmpBufsCols;
    int64_t rowStep = ubAvailable / rowFactor;

    bool ret = (rowStep >= 1);
    OP_LOGW(context_,
            "ubAvailable=%" PRId64 ", binaryAddUbSize: %" PRId64 ", quantBufSize: %" PRId64 ", weightBufSize: %" PRId64,
            ubAvailable, binaryAddUbSize, quantBufSize, weightBufSize);
    OP_LOGW(context_,
            "inOutCols=%" PRId64 ", tmpBufsCols=%" PRId64 ", rowFactor=%" PRId64 ", rowStep: %" PRId64 ", ret: %d",
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
    // Inputs are x1/x2; outputs are y/x.
    int64_t elewiseSliceNums = this->bufferNum_ * this->dtSizeX1_ * (RESIDUAL_INPUT_COUNT + 1) +
                               this->bufferNum_ * sizeof(int8_t);
    elewiseSliceNums += (this->biasType_ == BiasType::ELEMENTWISE_BIAS) ? (this->bufferNum_ * this->dtSizeX1_ * 1) : 0;
    // Temporary buffers hold mean, variance, and binary reduction data.
    int64_t tmpSliceNums = sizeof(float) * (MEAN_VAR_BUFFER_COUNT + 1);

    int64_t ubAvailable = static_cast<int64_t>(this->ubSize_) - UB_RESERVED_BYTE;
    int64_t bytesPerColumn = quantSliceNums + weightSliceNums + elewiseSliceNums + tmpSliceNums;
    this->colsPerLoop_ = ubAvailable / bytesPerColumn;
    this->colsPerLoop_ = this->colsPerLoop_ / this->vlFp32_ * this->vlFp32_;
    OP_CHECK_IF(
        (this->colsPerLoop_ <= 0),
        OP_LOGE(context_,
                "Welford UB capacity is insufficient: cols=%" PRId64 ", vlFp32=%u, xDtype=%d, "
                "scaleDtype=%d, bytesPerColumn=%" PRId64 ", minimumRequiredBytes=%" PRId64 ", availableBytes=%" PRId64
                ", reservedBytes=%" PRId64 ". Check platform support or reduce the normalized dimension.",
                this->cols_, this->vlFp32_, static_cast<int>(this->dataTypeX1_), static_cast<int>(this->dataTypeScale_),
                bytesPerColumn, bytesPerColumn * this->vlFp32_, ubAvailable, static_cast<int64_t>(UB_RESERVED_BYTE)),
        return false);
    this->colsLoopCount_ = Ops::Base::CeilDiv(this->cols_, this->colsPerLoop_);

    // Try to use the aligned Welford finalize process for better performance.
    int64_t equalSliceCols = this->cols_ / this->colsLoopCount_;
    if (this->cols_ % this->colsLoopCount_ == 0 && equalSliceCols % this->vlFp32_ == 0) {
        this->colsPerLoop_ = equalSliceCols;
    }

    this->colsTail_ = this->cols_ % this->colsPerLoop_;
    this->colsTail_ = (this->colsTail_ == 0) ? this->colsPerLoop_ : this->colsTail_;

    this->binaryAddNum_ = (this->colsPerLoop_ > this->vlFp32_) ? FindFloorPowerTwo(this->colsPerLoop_) : this->vlFp32_;
    ComputeBinaryAddVars();
    this->ubTilingPolicy_ = UbTilingPolicy::WELFORD;
    return true;
}

bool QuantizeAddLayerNormRegbaseTiling::CheckTensorAndAttr()
{
    const gert::StorageShape* x1Shape = this->context_->GetInputShape(X1_IDX);
    OP_CHECK_IF(x1Shape == nullptr, OP_LOGE(this->context_->GetNodeName(), "x1 shape is nullptr"), return false);
    const gert::StorageShape* x2Shape = this->context_->GetInputShape(X2_IDX);
    OP_CHECK_IF(x2Shape == nullptr, OP_LOGE(this->context_->GetNodeName(), "x2 shape is nullptr"), return false);
    const gert::StorageShape* gammaShape = this->context_->GetInputShape(GAMMA_IDX);
    OP_CHECK_IF(gammaShape == nullptr, OP_LOGE(this->context_->GetNodeName(), "gamma shape is nullptr"), return false);
    const gert::StorageShape* betaShape = this->context_->GetInputShape(BETA_IDX);
    OP_CHECK_IF(betaShape == nullptr, OP_LOGE(this->context_->GetNodeName(), "beta shape is nullptr"), return false);
    const gert::StorageShape* biasShape = this->context_->GetInputShape(BIAS_IDX);
    OP_CHECK_IF(biasShape == nullptr, OP_LOGE(this->context_->GetNodeName(), "bias shape is nullptr"), return false);

    const gert::StorageShape* yShape = this->context_->GetOutputShape(Y_IDX);
    OP_CHECK_IF(yShape == nullptr, OP_LOGE(this->context_->GetNodeName(), "y shape is nullptr"), return false);
    const gert::StorageShape* xShape = this->context_->GetOutputShape(X_OUT_IDX);
    OP_CHECK_IF(xShape == nullptr, OP_LOGE(this->context_->GetNodeName(), "x output shape is nullptr"), return false);

    size_t elewiseDimNum = x1Shape->GetStorageShape().GetDimNum();
    size_t weightDimNum = gammaShape->GetStorageShape().GetDimNum();
    OP_CHECK_IF(elewiseDimNum == 0,
                OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(this->context_->GetNodeName(), "x1", "0",
                                                         "x1 rank must be at least 1"),
                return false);
    OP_CHECK_IF(weightDimNum != 1,
                OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(this->context_->GetNodeName(), "gamma",
                                                         std::to_string(weightDimNum).c_str(), "gamma rank must be 1"),
                return false);
    OP_CHECK_IF(x1Shape->GetStorageShape().GetDim(elewiseDimNum - 1) != gammaShape->GetStorageShape().GetDim(0),
                OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(this->context_->GetNodeName(), "x1 and gamma",
                                                       (Ops::Base::ToString(x1Shape->GetStorageShape()) + " and " +
                                                        Ops::Base::ToString(gammaShape->GetStorageShape()))
                                                           .c_str(),
                                                       "gamma must match the last dimension of x1"),
                return false);

    bool elewiseShapeEqual = ((*x1Shape) == (*x2Shape)) && ((*x1Shape) == (*xShape)) && ((*x1Shape) == (*yShape));
    OP_CHECK_IF(!elewiseShapeEqual,
                OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(this->context_->GetNodeName(), "x1, x2, y and x",
                                                       (Ops::Base::ToString(x1Shape->GetStorageShape()) + ", " +
                                                        Ops::Base::ToString(x2Shape->GetStorageShape()) + ", " +
                                                        Ops::Base::ToString(yShape->GetStorageShape()) + " and " +
                                                        Ops::Base::ToString(xShape->GetStorageShape()))
                                                           .c_str(),
                                                       "x1, x2, y and x shapes must match"),
                return false);
    OP_CHECK_IF((*gammaShape) != (*betaShape),
                OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(this->context_->GetNodeName(), "gamma and beta",
                                                       (Ops::Base::ToString(gammaShape->GetStorageShape()) + " and " +
                                                        Ops::Base::ToString(betaShape->GetStorageShape()))
                                                           .c_str(),
                                                       "gamma and beta shapes must match"),
                return false);

    OP_CHECK_IF(HasZeroDim(x1Shape),
                OP_LOGE_FOR_INVALID_SHAPESIZE(context_->GetNodeName(), "x1", "0", "greater than 0"), return false);
    OP_CHECK_IF(HasZeroDim(gammaShape),
                OP_LOGE_FOR_INVALID_SHAPESIZE(context_->GetNodeName(), "gamma", "0", "greater than 0"), return false);
    return true;
}

static inline bool CheckOptionalShape(const size_t x1DimNum, const gert::StorageShape* gammaShape,
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
    OP_CHECK_IF(x1Shape == nullptr || gammaShape == nullptr,
                OP_LOGE(context_->GetNodeName(), "x1 or gamma shape is nullptr"), return false);
    size_t elewiseDimNum = x1Shape->GetStorageShape().GetDimNum();

    // bias is required, shape must equal x1 or gamma
    const gert::StorageShape* biasShape = context_->GetInputShape(BIAS_IDX);
    OP_CHECK_IF(biasShape == nullptr, OP_LOGE(context_->GetNodeName(), "bias shape is nullptr"), return false);
    bool invalidBias = ((*biasShape) != (*x1Shape)) && ((*biasShape) != (*gammaShape));
    OP_CHECK_IF(invalidBias,
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context_->GetNodeName(), "bias",
                                                      Ops::Base::ToString(biasShape->GetStorageShape()).c_str(),
                                                      "The shape of bias should be equal to the shape of x1 or gamma"),
                return false);

    const gert::StorageShape* scaleShape = context_->GetInputShape(SCALE_IDX);
    const gert::StorageShape* offsetShape = context_->GetOptionalInputShape(ZERO_POINT_IDX);
    OP_CHECK_IF(scaleShape == nullptr, OP_LOGE(context_->GetNodeName(), "scales shape is nullptr"), return false);
    this->offsetExist_ = (offsetShape != nullptr);

    // Legacy per-tensor callers may provide [H], but the kernel consumes element 0.
    bool scaleShapeOk = (this->quantMode_ == QuantMode::PER_TENSOR) ?
                            (scaleShape->GetStorageShape().GetShapeSize() == 1 || (*scaleShape) == (*gammaShape)) :
                            CheckOptionalShape(elewiseDimNum, gammaShape, scaleShape);
    OP_CHECK_IF(
        (!scaleShapeOk),
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(context_->GetNodeName(), "scales and gamma",
                                               (Ops::Base::ToString(scaleShape->GetStorageShape()) + " and " +
                                                Ops::Base::ToString(gammaShape->GetStorageShape()))
                                                   .c_str(),
                                               "scales must be scalar or equal gamma for per_tensor, and gamma-shaped "
                                               "for other modes"),
        return false);
    // zero_points follows the same compatibility rule as scales.
    bool offsetShapeOk = !this->offsetExist_ ||
                         ((this->quantMode_ == QuantMode::PER_TENSOR) ?
                              (offsetShape->GetStorageShape().GetShapeSize() == 1 || (*offsetShape) == (*gammaShape)) :
                              CheckOptionalShape(elewiseDimNum, gammaShape, offsetShape));
    OP_CHECK_IF((!offsetShapeOk),
                OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                    context_->GetNodeName(), "zero_points and gamma",
                    (Ops::Base::ToString(offsetShape->GetStorageShape()) + " and " +
                     Ops::Base::ToString(gammaShape->GetStorageShape()))
                        .c_str(),
                    "zero_points must be scalar or equal gamma for per_tensor, and gamma-shaped for other modes"),
                return false);

    auto scaleDesc = context_->GetInputDesc(SCALE_IDX);
    OP_CHECK_IF(scaleDesc == nullptr, OP_LOGE(context_->GetNodeName(), "scales desc is nullptr"), return false);
    this->dataTypeScale_ = scaleDesc->GetDataType();
    this->dtSizeScale_ = GetSizeByDataType(this->dataTypeScale_);
    return true;
}

bool QuantizeAddLayerNormRegbaseTiling::CheckOptionalInputDtype(ge::DataType x1Dtype)
{
    auto checkDt = [x1Dtype](ge::DataType dt, const char* name, const gert::TilingContext* ctx) -> bool {
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
        OP_LOGE_FOR_INVALID_DTYPE(context_->GetNodeName(), "x1", Ops::Base::ToString(x1Dtype).c_str(),
                                  "FLOAT16, BF16 or FLOAT");
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

static ge::graphStatus Tiling4QuantizeAddLayerNorm(gert::TilingContext* context)
{
    OP_CHECK_IF(context == nullptr,
                ReportTilingStageError("QuantizeAddLayerNorm",
                                       "the TilingContext is null; check the tiling framework invocation"),
                return ge::GRAPH_FAILED);
    OP_LOGW(context, "Regbase tiling starts.");
    QuantizeAddLayerNormRegbaseTiling regTiling(context);
    OP_CHECK_IF(!regTiling.DoTiling(), OP_LOGE_WITHOUT_REPORT(context, "Regbase tiling failed."),
                return ge::GRAPH_FAILED);
    OP_LOGW(context, "Regbase tiling succeeded.");
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(QuantizeAddLayerNorm).Tiling(Tiling4QuantizeAddLayerNorm);

} // namespace optiling

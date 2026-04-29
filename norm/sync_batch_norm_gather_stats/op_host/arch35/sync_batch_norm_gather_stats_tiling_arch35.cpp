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
 * \file sync_batch_norm_gather_stats_tiling_arch35.cc
 * \brief
 */
#include "sync_batch_norm_gather_stats_tiling_arch35.h"
#include "log/log.h"
#include "util/math_util.h"
#include "util/platform_util.h"
#include "platform/platform_info.h"
#include "register/op_impl_registry.h"
#include "register/tilingdata_base.h"
#include "op_host/tiling_templates_registry.h"

using namespace AscendC;
namespace optiling {
    static constexpr int16_t INPUT_IDX_TOTAL_SUM = 0;
    static constexpr int16_t INPUT_IDX_TOTAL_SQUARE_SUM = 1;
    static constexpr int16_t INPUT_IDX_SAMPLE_COUNT = 2;
    static constexpr int16_t INPUT_IDX_MEAN = 3;
    static constexpr int16_t INPUT_IDX_VAR = 4;
    static constexpr int16_t INDEX_MOMENTUM = 0;
    static constexpr int16_t INDEX_EPS = 1;
    constexpr float DEFAULT_MOMENTUM = 0.1;
    constexpr float DEFAULT_EPS = 1e-5;
    static constexpr int16_t TOTAL_SUMS_SHAPE_SIZE = 2;
    static constexpr int16_t TOTAL_SQUARE_SUMS_SHAPE_SIZE = 2;
    static constexpr int16_t SAMPLE_COUNT_SHAPE_SIZE = 1;
    static constexpr int16_t MEAN_SHAPE_SIZE = 1;
    static constexpr int16_t VAR_SHAPE_SIZE = 1;
    static constexpr int16_t UBSIZE_RSV = 96;

    static constexpr int16_t CACL_UBSIZE_SAMPLE_COUNT_TYPE32 = 3;
    static constexpr int16_t CACL_UBSIZE_SAMPLE_COUNT_TYPE16 = 6;
    static constexpr int16_t INPUT_SIZE_TYPE32 = 14;
    static constexpr int16_t INPUT_SIZE_TYPE16 = 16;
    static constexpr int16_t FLOAT_DTYPE_SIZE = 4;
    static constexpr int16_t BIG_INPUT_SIZE_TYPE32 = 4;
    static constexpr int16_t BIG_INPUT_SIZE_TYPE16 = 8;
    static constexpr int16_t TILING_KEY_N_FULL_LOAD = 10001;
    static constexpr int16_t TILING_KEY_N_NOT_FULL_LOAD = 20001;
    static constexpr int16_t HALF_CORE_NUM = 2;
    static constexpr int16_t UB_MIN_SIZE = 32;
    static constexpr int64_t PER_CORE_MIN_UB_BYTE = 8 * 1024;
    static constexpr uint64_t DEFAULT_WORKSPACE_SIZE = 16 * 1024 * 1024;
    static const uint64_t ALIGN_DIM_LEN_LINE = 32;

    static constexpr int16_t BLOCK_SIZE = 32;
    static constexpr int16_t ALIGN_FACTOR_BASE_BYTE_LEN = 128;
    static constexpr int16_t TILE_NUM_BASE_LEN_B32 = 512;
    static constexpr int16_t CONST_ZERO = 0;
    static constexpr int16_t CONST_ONE = 1;
    static constexpr int16_t CONST_TWO = 2;
    static constexpr int16_t CONST_THREE = 3;
    static constexpr int16_t CONST_FOUR = 4;
    static constexpr int16_t CONST_SIXTY_THREE = 63;
    static constexpr int16_t CONST_SIXTY_FOUR = 64;

bool SyncBatchNormGatherStatsTiling::TotalSumShapeCheck()
{
    auto totalSumDesc = context_->GetInputDesc(INPUT_IDX_TOTAL_SUM);
    OP_CHECK_NULL_WITH_CONTEXT(context_, totalSumDesc);
    totalSumDType_ = totalSumDesc->GetDataType();
    OP_CHECK_IF((totalSumDType_ != ge::DT_BF16 && totalSumDType_ != ge::DT_FLOAT16 && totalSumDType_ != ge::DT_FLOAT),
        OP_LOGE_FOR_INVALID_DTYPE(
            opName, "total_sum", ge::TypeUtils::DataTypeToSerialString(totalSumDType_).c_str(),
            "float16, bfloat16 and float"),
        return false);

    auto totalSumStorageShape = context_->GetInputShape(INPUT_IDX_TOTAL_SUM);
    OP_CHECK_NULL_WITH_CONTEXT(context_, totalSumStorageShape);
    totalSumShape_ = totalSumStorageShape->GetStorageShape();
    auto totalSumDims = totalSumShape_.GetDimNum();
    OP_CHECK_IF((static_cast<uint16_t>(totalSumDims) != TOTAL_SUMS_SHAPE_SIZE),
        OP_LOGE_FOR_INVALID_SHAPEDIM(
            opName, "total_sum", std::to_string(totalSumDims).c_str(), std::to_string(TOTAL_SUMS_SHAPE_SIZE).c_str()),
        return false);

    totalSumDim0_ = totalSumShape_.GetDim(0);
    totalSumDim1_ = totalSumShape_.GetDim(1);

    OP_CHECK_IF((totalSumDim0_ <= 0 || totalSumDim1_ <= 0),
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
            opName, "total_sum", Ops::Base::ToString(totalSumShape_).c_str(), "Input total_sum cannot be an empty tensor"),
        return false);

    return true;
}

bool SyncBatchNormGatherStatsTiling::TotalSquareSumShapeCheck()
{
    auto totalSquareSumDesc = context_->GetInputDesc(INPUT_IDX_TOTAL_SQUARE_SUM);
    OP_CHECK_NULL_WITH_CONTEXT(context_, totalSquareSumDesc);
    totalSquareSumDType_ = totalSquareSumDesc->GetDataType();
    OP_CHECK_IF((totalSquareSumDType_ != ge::DT_BF16 && totalSquareSumDType_ != ge::DT_FLOAT16 && totalSquareSumDType_ != ge::DT_FLOAT),
        OP_LOGE_FOR_INVALID_DTYPE(opName, "total_square_sum", ge::TypeUtils::DataTypeToSerialString(totalSquareSumDType_).c_str(),
            "float16, bfloat16 and float"),
        return false);

    auto totalSquareSumStorageShape = context_->GetInputShape(INPUT_IDX_TOTAL_SQUARE_SUM);
    OP_CHECK_NULL_WITH_CONTEXT(context_, totalSquareSumStorageShape);
    totalSquareSumShape_ = totalSquareSumStorageShape->GetStorageShape();
    auto totalSquareSumDims = totalSquareSumShape_.GetDimNum();
    OP_CHECK_IF((static_cast<uint16_t>(totalSquareSumDims) != TOTAL_SQUARE_SUMS_SHAPE_SIZE),
        OP_LOGE_FOR_INVALID_SHAPEDIM(opName, "total_square_sum", std::to_string(totalSquareSumDims).c_str(),
            std::to_string(TOTAL_SQUARE_SUMS_SHAPE_SIZE).c_str()),
        return false);

    totalSquareSumDim0_ = totalSquareSumShape_.GetDim(0);
    totalSquareSumDim1_ = totalSquareSumShape_.GetDim(1);

    OP_CHECK_IF((totalSquareSumDim0_ <= 0 || totalSquareSumDim1_ <= 0),
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(opName, "total_square_sum", Ops::Base::ToString(totalSquareSumShape_).c_str(),
            "Input total_square_sum cannot be an empty tensor"),
        return false);

    return true;
}

bool SyncBatchNormGatherStatsTiling::SampleCountShapeCheck()
{
    auto sampleCountDesc = context_->GetInputDesc(INPUT_IDX_SAMPLE_COUNT);
    OP_CHECK_NULL_WITH_CONTEXT(context_, sampleCountDesc);
    sampleCountDType_ = sampleCountDesc->GetDataType();
    OP_CHECK_IF((sampleCountDType_ != ge::DT_INT32),
        OP_LOGE_FOR_INVALID_DTYPE(
            opName, "sample_count", ge::TypeUtils::DataTypeToSerialString(sampleCountDType_).c_str(), "int32"),
        return false);

    auto sampleCountStorageShape = context_->GetInputShape(INPUT_IDX_SAMPLE_COUNT);
    OP_CHECK_NULL_WITH_CONTEXT(context_, sampleCountStorageShape);
    sampleCountShape_ = sampleCountStorageShape->GetStorageShape();
    auto sampleCountDims = sampleCountShape_.GetDimNum();
    OP_CHECK_IF((static_cast<uint16_t>(sampleCountDims) != SAMPLE_COUNT_SHAPE_SIZE),
        OP_LOGE_FOR_INVALID_SHAPEDIM(
            opName, "sample_count", std::to_string(sampleCountDims).c_str(),
            std::to_string(SAMPLE_COUNT_SHAPE_SIZE).c_str()),
        return false);

    sampleCountDim0_ = sampleCountShape_.GetDim(0);

    OP_CHECK_IF((sampleCountDim0_ <= 0),
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
            opName, "sample_count", Ops::Base::ToString(sampleCountShape_).c_str(),
            "Input sample_count cannot be an empty tensor"),
        return false);

    return true;
}

bool SyncBatchNormGatherStatsTiling::MeanShapeCheck()
{
    auto meanDesc = context_->GetInputDesc(INPUT_IDX_MEAN);
    OP_CHECK_NULL_WITH_CONTEXT(context_, meanDesc);
    meanDType_ = meanDesc->GetDataType();
    OP_CHECK_IF((meanDType_ != ge::DT_BF16 && meanDType_ != ge::DT_FLOAT16 && meanDType_ != ge::DT_FLOAT),
        OP_LOGE_FOR_INVALID_DTYPE(
            opName, "mean", ge::TypeUtils::DataTypeToSerialString(meanDType_).c_str(), "float16, bfloat16 and float"),
        return false);

    auto meanStorageShape = context_->GetInputShape(INPUT_IDX_MEAN);
    OP_CHECK_NULL_WITH_CONTEXT(context_, meanStorageShape);
    meanShape_ = meanStorageShape->GetStorageShape();
    auto meanDims = meanShape_.GetDimNum();
    OP_CHECK_IF((static_cast<uint16_t>(meanDims) != MEAN_SHAPE_SIZE),
        OP_LOGE_FOR_INVALID_SHAPEDIM(
            opName, "mean", std::to_string(meanDims).c_str(), std::to_string(MEAN_SHAPE_SIZE).c_str()),
        return false);

    meanDim0_ = meanShape_.GetDim(0);

    OP_CHECK_IF((meanDim0_ <= 0),
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
            opName, "mean", Ops::Base::ToString(meanShape_).c_str(), "Input mean cannot be an empty tensor"),
        return false);

    return true;
}

bool SyncBatchNormGatherStatsTiling::VarShapeCheck()
{
    auto varDesc = context_->GetInputDesc(INPUT_IDX_VAR);
    OP_CHECK_NULL_WITH_CONTEXT(context_, varDesc);
    varDType_ = varDesc->GetDataType();
    OP_CHECK_IF((varDType_ != ge::DT_BF16 && varDType_ != ge::DT_FLOAT16 && varDType_ != ge::DT_FLOAT),
        OP_LOGE_FOR_INVALID_DTYPE(
            opName, "variance", ge::TypeUtils::DataTypeToSerialString(varDType_).c_str(),
            "float16, bfloat16 and float"),
        return false);

    auto varStorageShape = context_->GetInputShape(INPUT_IDX_VAR);
    OP_CHECK_NULL_WITH_CONTEXT(context_, varStorageShape);
    varShape_ = varStorageShape->GetStorageShape();
    auto varDims = varShape_.GetDimNum();
    OP_CHECK_IF((static_cast<uint16_t>(varDims) != VAR_SHAPE_SIZE),
        OP_LOGE_FOR_INVALID_SHAPEDIM(
            opName, "variance", std::to_string(varDims).c_str(), std::to_string(VAR_SHAPE_SIZE).c_str()),
        return false);

    varDim0_ = varShape_.GetDim(0);

    OP_CHECK_IF((varDim0_ <= 0),
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
            opName, "variance", Ops::Base::ToString(varShape_).c_str(), "Input variance cannot be an empty tensor"),
        return false);

    return true;
}

bool SyncBatchNormGatherStatsTiling::BatchMeanCheck()
{
    auto batchMeanDesc = context_->GetOutputDesc(INPUT_IDX_TOTAL_SUM);
    OP_CHECK_NULL_WITH_CONTEXT(context_, batchMeanDesc);
    batchMeanDType_ = batchMeanDesc->GetDataType();
    OP_CHECK_IF((batchMeanDType_ != ge::DT_BF16 && batchMeanDType_ != ge::DT_FLOAT16 && batchMeanDType_ != ge::DT_FLOAT),
        OP_LOGE_FOR_INVALID_DTYPE(
            opName, "batch_mean", ge::TypeUtils::DataTypeToSerialString(batchMeanDType_).c_str(),
            "float16, bfloat16 and float"),
        return false);

    auto batchMeanStorageShape = context_->GetOutputShape(INPUT_IDX_TOTAL_SUM);
    OP_CHECK_NULL_WITH_CONTEXT(context_, batchMeanStorageShape);
    batchMeanShape_ = batchMeanStorageShape->GetStorageShape();
    auto batchMeanDims = batchMeanShape_.GetDimNum();
    OP_CHECK_IF((static_cast<uint16_t>(batchMeanDims) != MEAN_SHAPE_SIZE),
        OP_LOGE_FOR_INVALID_SHAPEDIM(
            opName, "batch_mean", std::to_string(batchMeanDims).c_str(), std::to_string(MEAN_SHAPE_SIZE).c_str()),
        return false);

    batchMeanDim0_ = batchMeanShape_.GetDim(0);

    OP_CHECK_IF((batchMeanDim0_ <= 0),
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(opName, "batch_mean",
            Ops::Base::ToString(batchMeanShape_).c_str(), "Output batch_mean cannot be an empty tensor"),
        return false);

    return true;
}

bool SyncBatchNormGatherStatsTiling::BatchInvstdCheck()
{
    auto batchInvStdDesc = context_->GetOutputDesc(INPUT_IDX_TOTAL_SQUARE_SUM);
    OP_CHECK_NULL_WITH_CONTEXT(context_, batchInvStdDesc);
    batchInvStdDType_ = batchInvStdDesc->GetDataType();
    OP_CHECK_IF((batchInvStdDType_ != ge::DT_BF16 && batchInvStdDType_ != ge::DT_FLOAT16 && batchInvStdDType_ != ge::DT_FLOAT),
        OP_LOGE_FOR_INVALID_DTYPE(
            opName, "batch_invstd", ge::TypeUtils::DataTypeToSerialString(batchInvStdDType_).c_str(),
            "float16, bfloat16 and float"),
        return false);

    auto batchInvStdStorageShape = context_->GetOutputShape(INPUT_IDX_TOTAL_SQUARE_SUM);
    OP_CHECK_NULL_WITH_CONTEXT(context_, batchInvStdStorageShape);
    batchInvStdShape_ = batchInvStdStorageShape->GetStorageShape();
    auto batchInvStdDims = batchInvStdShape_.GetDimNum();
    OP_CHECK_IF((static_cast<uint16_t>(batchInvStdDims) != MEAN_SHAPE_SIZE),
        OP_LOGE_FOR_INVALID_SHAPEDIM(
            opName, "batch_invstd", std::to_string(batchInvStdDims).c_str(), std::to_string(MEAN_SHAPE_SIZE).c_str()),
        return false);

    batchInvStdDim0_ = batchInvStdShape_.GetDim(0);

    OP_CHECK_IF((batchInvStdDim0_ <= 0),
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
            opName, "batch_invstd", Ops::Base::ToString(batchInvStdShape_).c_str(),
            "Output batch_invstd cannot be an empty tensor"),
        return false);

    return true;
}

bool SyncBatchNormGatherStatsTiling::RunningMeanCheck()
{
    auto runningMeanDesc = context_->GetOutputDesc(INPUT_IDX_SAMPLE_COUNT);
    OP_CHECK_NULL_WITH_CONTEXT(context_, runningMeanDesc);
    runningMeanDType_ = runningMeanDesc->GetDataType();
    OP_CHECK_IF((runningMeanDType_ != ge::DT_BF16 && runningMeanDType_ != ge::DT_FLOAT16 && runningMeanDType_ != ge::DT_FLOAT),
        OP_LOGE_FOR_INVALID_DTYPE(
            opName, "mean", ge::TypeUtils::DataTypeToSerialString(runningMeanDType_).c_str(),
            "float16, bfloat16 and float"),
        return false);

    auto runningMeanStorageShape = context_->GetOutputShape(INPUT_IDX_SAMPLE_COUNT);
    OP_CHECK_NULL_WITH_CONTEXT(context_, runningMeanStorageShape);
    runningMeanShape_ = runningMeanStorageShape->GetStorageShape();
    auto runningMeanDims = runningMeanShape_.GetDimNum();
    OP_CHECK_IF((static_cast<uint16_t>(runningMeanDims) != MEAN_SHAPE_SIZE),
        OP_LOGE_FOR_INVALID_SHAPEDIM(
            opName, "mean", std::to_string(runningMeanDims).c_str(), std::to_string(MEAN_SHAPE_SIZE).c_str()),
        return false);

    runningMeanDim0_ = runningMeanShape_.GetDim(0);

    OP_CHECK_IF((runningMeanDim0_ <= 0),
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
            opName, "mean", Ops::Base::ToString(runningMeanShape_).c_str(), "Output mean cannot be an empty tensor"),
        return false);

    return true;
}

bool SyncBatchNormGatherStatsTiling::RunningVarCheck()
{
    auto runningVarDesc = context_->GetOutputDesc(INPUT_IDX_MEAN);
    OP_CHECK_NULL_WITH_CONTEXT(context_, runningVarDesc);
    runningVarDType_ = runningVarDesc->GetDataType();
    OP_CHECK_IF((runningVarDType_ != ge::DT_BF16 && runningVarDType_ != ge::DT_FLOAT16 && runningVarDType_ != ge::DT_FLOAT),
        OP_LOGE_FOR_INVALID_DTYPE(
            opName, "variance", ge::TypeUtils::DataTypeToSerialString(runningVarDType_).c_str(),
            "float16, bfloat16 and float"),
        return false);

    auto runningVarStorageShape = context_->GetOutputShape(INPUT_IDX_MEAN);
    OP_CHECK_NULL_WITH_CONTEXT(context_, runningVarStorageShape);
    runningVarShape_ = runningVarStorageShape->GetStorageShape();
    auto runningVarDims = runningVarShape_.GetDimNum();
    OP_CHECK_IF((static_cast<uint16_t>(runningVarDims) != MEAN_SHAPE_SIZE),
        OP_LOGE_FOR_INVALID_SHAPEDIM(
            opName, "variance", std::to_string(runningVarDims).c_str(), std::to_string(MEAN_SHAPE_SIZE).c_str()),
        return false);

    runningVarDim0_ = runningVarShape_.GetDim(0);

    OP_CHECK_IF((runningVarDim0_ <= 0),
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
            opName, "variance", Ops::Base::ToString(runningVarShape_).c_str(), "Output variance cannot be an empty tensor"),
        return false);

    return true;
}

// Validate input tensor shapes
bool SyncBatchNormGatherStatsTiling::ValidateInputShapes()
{
    if (!TotalSumShapeCheck()) {
        return false;
    }
    if (!TotalSquareSumShapeCheck()) {
        return false;
    }
    if (!SampleCountShapeCheck()) {
        return false;
    }
    if (!MeanShapeCheck()) {
        return false;
    }
    if (!VarShapeCheck()) {
        return false;
    }
    return true;
}

// Validate output tensor shapes
bool SyncBatchNormGatherStatsTiling::ValidateOutputShapes()
{
    if (!BatchMeanCheck()) {
        return false;
    }
    if (!BatchInvstdCheck()) {
        return false;
    }
    if (!RunningMeanCheck()) {
        return false;
    }
    if (!RunningVarCheck()) {
        return false;
    }
    return true;
}

// Get attributes
void SyncBatchNormGatherStatsTiling::GetAttrs()
{
    auto attrs = context_->GetAttrs();
    const float* momentum = attrs->GetFloat(INDEX_MOMENTUM);
    momentum_ = (momentum == nullptr) ? DEFAULT_MOMENTUM : *momentum;

    const float* eps = attrs->GetFloat(INDEX_EPS);
    eps_ = (eps == nullptr) ? DEFAULT_EPS : *eps;
}

// Validate input dtype consistency
bool SyncBatchNormGatherStatsTiling::ValidateInputDtypes()
{
    if ((totalSumDType_ != totalSquareSumDType_) ||
        (totalSquareSumDType_ != meanDType_) ||
        (meanDType_ != varDType_)) {
        std::string dtypeMsg = ge::TypeUtils::DataTypeToSerialString(totalSumDType_) + ", " +
                               ge::TypeUtils::DataTypeToSerialString(totalSquareSumDType_) + ", " +
                               ge::TypeUtils::DataTypeToSerialString(meanDType_) + " and " +
                               ge::TypeUtils::DataTypeToSerialString(varDType_);
        std::string reasonMsg = "The dtypes of input total_sum, total_square_sum, mean and variance must be the same";
        OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(
            opName, "total_sum, total_square_sum, mean and variance", dtypeMsg.c_str(), reasonMsg.c_str());
        return false;
    }
    return true;
}

// Validate output dtype consistency
bool SyncBatchNormGatherStatsTiling::ValidateOutputDtypes()
{
    if ((batchMeanDType_ != batchInvStdDType_) ||
        (batchMeanDType_ != runningMeanDType_) ||
        (batchMeanDType_ != runningVarDType_) ||
        (batchMeanDType_ != meanDType_)) {
        std::string dtypeMsg = ge::TypeUtils::DataTypeToSerialString(batchMeanDType_) + ", " +
                               ge::TypeUtils::DataTypeToSerialString(batchInvStdDType_) + ", " +
                               ge::TypeUtils::DataTypeToSerialString(runningMeanDType_) + " and " +
                               ge::TypeUtils::DataTypeToSerialString(runningVarDType_);
        std::string reasonMsg = "The dtypes of output batch_mean, batch_invstd, mean and variance must be the same";
        OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(
            opName, "batch_mean, batch_invstd, mean and variance", dtypeMsg.c_str(), reasonMsg.c_str());
        return false;
    }
    return true;
}

// Validate input dimension consistency
bool SyncBatchNormGatherStatsTiling::ValidateInputDimensions()
{
    if ((totalSumDim0_ != totalSquareSumDim0_) ||
        (totalSumDim0_ != sampleCountDim0_) ||
        (totalSumDim1_ != totalSquareSumDim1_) ||
        (totalSumDim1_ != meanDim0_) ||
        (totalSumDim1_ != varDim0_)) {
        std::string shapeMsg = Ops::Base::ToString(totalSumShape_) + ", " +
                               Ops::Base::ToString(totalSquareSumShape_) +
                               ", " + Ops::Base::ToString(sampleCountShape_) + ", " +
                               Ops::Base::ToString(meanShape_) +
                               " and " + Ops::Base::ToString(varShape_);
        std::string reasonMsg =
            "The first dimension of total_sum, total_square_sum and sample_count must be the same, "
            "and the second dimension of total_sum, total_square_sum, mean and variance must be the same";
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
            opName, "total_sum, total_square_sum, sample_count, mean and variance", shapeMsg.c_str(),
            reasonMsg.c_str());
        return false;
    }
    return true;
}

// Validate output dimension consistency
bool SyncBatchNormGatherStatsTiling::ValidateOutputDimensions()
{
    if ((batchMeanDim0_ != batchInvStdDim0_) ||
        (batchMeanDim0_ != runningMeanDim0_) ||
        (batchMeanDim0_ != runningVarDim0_) ||
        (batchMeanDim0_ != meanDim0_)) {
        std::string shapeMsg = Ops::Base::ToString(batchMeanShape_) + ", " +
                               Ops::Base::ToString(batchInvStdShape_) +
                               ", " + Ops::Base::ToString(runningMeanShape_) + " and " +
                               Ops::Base::ToString(runningVarShape_);
        std::string reasonMsg = "The first dimension of batch_mean, batch_invstd, mean and variance must be the same";
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
            opName, "batch_mean, batch_invstd, mean and variance", shapeMsg.c_str(), reasonMsg.c_str());
        return false;
    }
    return true;
}

ge::graphStatus SyncBatchNormGatherStatsTiling::GetShapeAttrsInfo()
{
    // 1. Validate input shapes
    if (!ValidateInputShapes()) {
        return ge::GRAPH_FAILED;
    }

    // 2. Validate output shapes
    if (!ValidateOutputShapes()) {
        return ge::GRAPH_FAILED;
    }

    // 3. Get attributes
    GetAttrs();

    // 4. Validate input dtype consistency
    if (!ValidateInputDtypes()) {
        return ge::GRAPH_FAILED;
    }

    // 5. Validate output dtype consistency
    if (!ValidateOutputDtypes()) {
        return ge::GRAPH_FAILED;
    }

    // 6. Validate input dimension consistency
    if (!ValidateInputDimensions()) {
        return ge::GRAPH_FAILED;
    }

    // 7. Validate output dimension consistency
    if (!ValidateOutputDimensions()) {
        return ge::GRAPH_FAILED;
    }

    // 8. Set output length
    nLen = totalSumDim0_;
    cLen = totalSumDim1_;

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SyncBatchNormGatherStatsTiling::GetPlatformInfo()
{
    auto compileInfo = reinterpret_cast<const SyncBatchNormGatherStatsCompileInfo *>(context_->GetCompileInfo());
    OP_CHECK_NULL_WITH_CONTEXT(context_, compileInfo);
    coreNum_ = compileInfo->coreNum;
    ubSize_ = compileInfo->ubSize;
    blockSize_ = compileInfo->blockSize;

    return ge::GRAPH_SUCCESS;
}

bool SyncBatchNormGatherStatsTiling::IsCapable()
{
    return true;
}

int64_t SyncBatchNormGatherStatsTiling::FindNearestPower2(const int64_t value)
{
    if (value <= CONST_ONE) {
        return CONST_ZERO;
    } else if (value <= CONST_TWO) {
        return CONST_ONE;
    } else if (value <= CONST_FOUR) {
        return CONST_TWO;
    } else {
        const int64_t num = value - CONST_ONE;
        const int64_t pow = CONST_SIXTY_THREE - __builtin_clzl(num);
        return (CONST_ONE << pow);
    }
}

int64_t SyncBatchNormGatherStatsTiling::GetCacheID(const int64_t idx)
{
    return __builtin_popcountll(idx ^ (idx + CONST_ONE)) - CONST_ONE;
}

ge::graphStatus SyncBatchNormGatherStatsTiling::DoNNotFullLoadTiling()
{
    OP_LOGD(opName, "DoNNotFullLoadTiling");
    const uint32_t dTypeSize = ge::GetSizeByDataType(meanDType_);
    int64_t cBlockFactorBase = ALIGN_FACTOR_BASE_BYTE_LEN / dTypeSize;
    int64_t cBlockFactorAlignBase = ALIGN_FACTOR_BASE_BYTE_LEN / dTypeSize;
    int64_t cBlockFactor = Ops::Base::CeilDiv(cLen, coreNum_);
    cBlockFactor = Ops::Base::CeilDiv(cBlockFactor, cBlockFactorAlignBase) * cBlockFactorAlignBase;
    cBlockFactor = std::max(cBlockFactorBase, cBlockFactor);
    blockNum = Ops::Base::CeilDiv(cLen, cBlockFactor);
    int64_t cTileBlockFactor = cLen - (blockNum - 1) * cBlockFactor;
    int64_t cTileNumBase;
    if (dTypeSize == CONST_FOUR) {
        cTileNumBase = TILE_NUM_BASE_LEN_B32 / dTypeSize;
    } else {
        cTileNumBase = ALIGN_FACTOR_BASE_BYTE_LEN / dTypeSize;
    }
    if (cTileNumBase > cBlockFactor) {
        cTileNumBase = cBlockFactor;
    }

    std::vector<int64_t> nTileNumList = {64, 128, 256, 512};

    for (const int64_t& nTileNumPossible : nTileNumList) {
        int64_t nTileNum = nTileNumPossible;
        int64_t nLoop = nLen / nTileNum;
        int64_t nTotalLoop = Ops::Base::CeilDiv(nLen, nTileNum);
        int64_t nTail = nLen - nLoop * nTileNum;
        int64_t basicBlockLoop = FindNearestPower2(nTotalLoop);
        int64_t mainFoldCount = nLoop - basicBlockLoop;
        int64_t cacheBufferCount = CONST_ONE;
        int32_t resultCacheID = CONST_ZERO;
        if (basicBlockLoop != 0) {
            cacheBufferCount = CONST_SIXTY_FOUR - __builtin_clzl(basicBlockLoop);
            resultCacheID = GetCacheID(basicBlockLoop - 1);
        }

        int64_t cTileNum = 0;
        if (dTypeSize == CONST_FOUR) {
            cTileNum = (ubSize_ - blockSize_ * CONST_THREE - CONST_TWO * nTileNum * CONST_FOUR - 
                nTileNum * CONST_FOUR * CONST_TWO - (cacheBufferCount+1) * blockSize_) / 
                (dTypeSize * ((CONST_FOUR * nTileNum) + (cacheBufferCount+CONST_TWO) + CONST_FOUR * CONST_TWO));
        } else if (dTypeSize == CONST_TWO) {
            cTileNum = (ubSize_ - blockSize_ * CONST_THREE - CONST_TWO * nTileNum * CONST_FOUR - 
                nTileNum * CONST_FOUR * CONST_TWO - (cacheBufferCount+1) * blockSize_) / 
                (CONST_TWO * CONST_THREE * nTileNum * dTypeSize + nTileNum * CONST_FOUR + 
                (cacheBufferCount+CONST_TWO) * CONST_FOUR + CONST_FOUR * CONST_TWO * dTypeSize);
        }
        cTileNum = cTileNum / cTileNumBase * cTileNumBase;

        if ((cTileNum <= cBlockFactor && nTileNum != nTileNumList[0]) || cTileNum <=0) {
            break;
        }

        int64_t cLoopMain = Ops::Base::CeilDiv(cBlockFactor, cTileNum);
        int64_t cTailMain = cBlockFactor - (cLoopMain - 1) * cTileNum;
        int64_t cLoopTail = Ops::Base::CeilDiv(cTileBlockFactor, cTileNum);
        int64_t cTailTail = cTileBlockFactor - (cLoopTail - 1) * cTileNum;

        cTileNum_ = cTileNum;
        nTileNum_ = nTileNum;
        basicBlockLoop_ = basicBlockLoop;
        mainFoldCount_ = mainFoldCount;
        nTail_ = nTail;
        cacheBufferCount_ = cacheBufferCount;
        resultCacheID_ = resultCacheID;
        cLoopMain_ = cLoopMain;
        cTailMain_ = cTailMain;
        cLoopTail_ = cLoopTail;
        cTailTail_ = cTailTail;
    }

    OP_CHECK_IF((cTileNum_ <= 0),
        OP_LOGE(opName, "The axis of C can not be Tiled"),
        return ge::GRAPH_FAILED);
    
    nNotFullLoadTilingData.set_blockDim(blockNum);
    nNotFullLoadTilingData.set_cLen(cLen);
    nNotFullLoadTilingData.set_cFactor(cTileNum_);
    nNotFullLoadTilingData.set_cLoopMainBlock(cLoopMain_);
    nNotFullLoadTilingData.set_cTileMainBlock(cTailMain_);
    nNotFullLoadTilingData.set_cLoopTailBlock(cLoopTail_);
    nNotFullLoadTilingData.set_cTailTailBlock(cTailTail_);
    nNotFullLoadTilingData.set_nFactor(nTileNum_);
    nNotFullLoadTilingData.set_nLoop(basicBlockLoop_);
    nNotFullLoadTilingData.set_nMainFoldCount(mainFoldCount_);
    nNotFullLoadTilingData.set_nTail(nTail_);
    nNotFullLoadTilingData.set_cacheBufferCount(cacheBufferCount_);
    nNotFullLoadTilingData.set_resultCacheId(resultCacheID_);
    nNotFullLoadTilingData.set_momentum(momentum_);
    nNotFullLoadTilingData.set_eps(eps_);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SyncBatchNormGatherStatsTiling::DoOpTiling()
{
    OP_LOGI("SyncBatchNormGatherStatsTiling::DoOpTiling start");

    const uint32_t dTypeSize = ge::GetSizeByDataType(meanDType_);

    OP_CHECK_IF((dTypeSize == 0),
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
            context_->GetNodeName(), "mean", ge::TypeUtils::DataTypeToSerialString(meanDType_).c_str(),
            "The dtype size of input parameter mean must be not equal to 0"),
        return ge::GRAPH_FAILED);

    if (dTypeSize == FLOAT_DTYPE_SIZE) {
        cTileNum_ = ((ubSize_ - UBSIZE_RSV) / dTypeSize - CACL_UBSIZE_SAMPLE_COUNT_TYPE32 * nLen) /
            (INPUT_SIZE_TYPE32 + BIG_INPUT_SIZE_TYPE32 * nLen);
    } else {
        cTileNum_ = ((ubSize_ - UBSIZE_RSV) / dTypeSize - CACL_UBSIZE_SAMPLE_COUNT_TYPE16 * nLen) /
            (INPUT_SIZE_TYPE16 + BIG_INPUT_SIZE_TYPE16 * nLen);
    }
    cTileNum_ = cTileNum_ / (ALIGN_DIM_LEN_LINE / dTypeSize) * (ALIGN_DIM_LEN_LINE / dTypeSize);

    if (cTileNum_ <= 0 || cTileNum_ * dTypeSize < ALIGN_FACTOR_BASE_BYTE_LEN) {
        cTileNum_ = 0;
        return DoNNotFullLoadTiling();
    }
    
    ubOuter = (cLen + cTileNum_ - 1) / cTileNum_;
    ubTail = cLen % cTileNum_;
    ubTail = (ubTail == 0) ? cTileNum_ : ubTail;
    blockFormer = (ubOuter + coreNum_ - 1) / coreNum_;
    blockTail = ubOuter % blockFormer;
    blockTail = (blockTail == 0) ? blockFormer : blockTail;
    blockNum = (ubOuter + blockFormer - 1) / blockFormer;
    int64_t dimPerCore = 0;
    int64_t alignDimPerCore = 0;
    int64_t lowestcTileNum = 0;

    if (static_cast<int64_t>(blockNum) < (coreNum_ / HALF_CORE_NUM)) {
        // 1/2 的coreNum进行分配
        dimPerCore = (cLen + (coreNum_ / HALF_CORE_NUM) - 1) / (coreNum_ / HALF_CORE_NUM);
        // 向上对齐到32Byte，理论上不会超过原来计算的cTileNum_
        alignDimPerCore = ((((dimPerCore * dTypeSize) + ALIGN_DIM_LEN_LINE - 1) / ALIGN_DIM_LEN_LINE) *
                ALIGN_DIM_LEN_LINE) / dTypeSize;
        
        // 如果分配到核数一半后，小于32B(开DB后)，则直接按照 32B 计算。进入该分支，说明前面已经判断过开db后ub大小大于32B
        lowestcTileNum = UB_MIN_SIZE / dTypeSize;
        if (alignDimPerCore < cTileNum_) {
            if (alignDimPerCore < lowestcTileNum && lowestcTileNum < cTileNum_) {
                cTileNum_ = lowestcTileNum;
            } else {
                cTileNum_ = alignDimPerCore;
            }
            // 重新计算分核参数
            ubOuter = (cLen + cTileNum_ - 1) / cTileNum_;
            ubTail = cLen % cTileNum_;
            ubTail = (ubTail == 0) ? cTileNum_ : ubTail;
            blockFormer = (ubOuter + coreNum_ - 1) / coreNum_;
            blockTail = ubOuter % blockFormer;
            blockTail = (blockTail == 0) ? blockFormer : blockTail;
            blockNum = (ubOuter + blockFormer - 1) / blockFormer;
        }
    }

    /* tilingData信息赋值 */
    SetTilingDataInfo();

    return ge::GRAPH_SUCCESS;
}

void SyncBatchNormGatherStatsTiling::SetTilingDataInfo()
{
    tilingData.set_blockDim(blockNum);
    tilingData.set_blockFormer(blockFormer);
    tilingData.set_blockTail(blockTail);
    tilingData.set_nLen(nLen);
    tilingData.set_cLen(cLen);
    tilingData.set_ubFormer(cTileNum_);
    tilingData.set_ubTail(ubTail);
    tilingData.set_momentum(momentum_);
    tilingData.set_eps(eps_);
}

ge::graphStatus SyncBatchNormGatherStatsTiling::DoLibApiTiling()
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SyncBatchNormGatherStatsTiling::GetWorkspaceSize()
{
    workspaceSize_ = DEFAULT_WORKSPACE_SIZE;
    return ge::GRAPH_SUCCESS;
}

void SyncBatchNormGatherStatsTiling::PrintTilingData()
{
    if (nTileNum_ == 0) {
        OP_LOGD(opName, "totalSumDim0_:%ld.\n", totalSumDim0_);
        OP_LOGD(opName, "totalSumDim1_:%ld.\n", totalSumDim1_);
        OP_LOGD(opName, "totalSquareSumDim0_:%ld.\n", totalSquareSumDim0_);
        OP_LOGD(opName, "totalSquareSumDim1_:%ld.\n", totalSquareSumDim1_);
        OP_LOGD(opName, "sampleCountDim0_:%ld.\n", sampleCountDim0_);
        OP_LOGD(opName, "meanDim0_:%ld.\n", meanDim0_);
        OP_LOGD(opName, "varDim0_:%ld.\n", varDim0_);
        OP_LOGD(opName, "momentum_:%f.\n", momentum_);
        OP_LOGD(opName, "eps_:%f.\n", eps_);
        OP_LOGD(opName, "nLen:%ld.\n", nLen);
        OP_LOGD(opName, "cLen:%ld.\n", cLen);
        OP_LOGD(opName, "coreNum_:%ld.\n", coreNum_);
        OP_LOGD(opName, "ubSize_:%ld.\n", ubSize_);
        OP_LOGD(opName, "blockFormer:%ld.\n", blockFormer);
        OP_LOGD(opName, "ubOuter:%ld.\n", ubOuter);
        OP_LOGD(opName, "ubTail:%ld.\n", ubTail);
        OP_LOGD(opName, "blockNum:%ld.\n", blockNum);
        OP_LOGD(opName, "blockTail:%ld.\n", blockTail);
        OP_LOGD(opName, "workspaceSize_:%ld.\n", workspaceSize_);
    }
}

ge::graphStatus SyncBatchNormGatherStatsTiling::PostTiling()
{
    auto workspaces = context_->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, workspaces);
    workspaces[0] = workspaceSize_;
    context_->SetTilingKey(GetTilingKey());
    context_->SetBlockDim(blockNum);
    if (nTileNum_ > 0) {
        nNotFullLoadTilingData.SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
        context_->GetRawTilingData()->SetDataSize(nNotFullLoadTilingData.GetDataSize());
    } else {
        tilingData.SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
        context_->GetRawTilingData()->SetDataSize(tilingData.GetDataSize());
    }

    PrintTilingData();
    return ge::GRAPH_SUCCESS;
}

uint64_t SyncBatchNormGatherStatsTiling::GetTilingKey() const
{
    if (nTileNum_ > 0) {
        return TILING_KEY_N_NOT_FULL_LOAD;
    } else {
        return TILING_KEY_N_FULL_LOAD;
    }
}

ge::graphStatus TilingSyncBatchNormGatherStats(gert::TilingContext *context)
{
    auto compileInfo = reinterpret_cast<const SyncBatchNormGatherStatsCompileInfo*>(context->GetCompileInfo());
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);

    SyncBatchNormGatherStatsTiling tilingObj(context);
    return tilingObj.DoTiling();
}

ge::graphStatus SyncBatchNormGatherStatsParse(gert::TilingParseContext *context)
{
    auto compileInfoPtr = context->GetCompiledInfo<SyncBatchNormGatherStatsCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfoPtr);

    OP_LOGD(context, "Enter SyncBatchNormGatherStatsTilingPrepareAscendC");
    auto platformInfo = context->GetPlatformInfo();
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    compileInfoPtr->coreNum = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF((compileInfoPtr->coreNum <= 0),
        OP_LOGE(context->GetNodeName(), "Get core num failed, coreNum <= 0"),
        return ge::GRAPH_FAILED);
    
    uint64_t ubSizePlatForm;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatForm);    
    compileInfoPtr->ubSize = ubSizePlatForm;
    OP_CHECK_IF((compileInfoPtr->ubSize <= 0),
        OP_LOGE(context->GetNodeName(), "ubSize is less than or equal 0. please check"),
        return ge::GRAPH_FAILED);

    compileInfoPtr->blockSize = Ops::Base::GetUbBlockSize(context);
    OP_CHECK_IF((compileInfoPtr->blockSize <= 0),
        OP_LOGE(context->GetNodeName(), "ubBlockSize is less than or equal 0. please check"),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

// register tiling interface of SyncBatchNormGatherStats op.
IMPL_OP_OPTILING(SyncBatchNormGatherStats).Tiling(TilingSyncBatchNormGatherStats).TilingParse<SyncBatchNormGatherStatsCompileInfo>(SyncBatchNormGatherStatsParse);
} // namespace optiling
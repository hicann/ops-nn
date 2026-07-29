/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "op_common/log/log.h"
#include "op_common/op_host/util/platform_util.h"
#include "op_host/tiling_util.h"
#include "op_host/tiling_templates_registry.h"
#include "tiling/math/log_tiling.h"
#include "graph/utils/type_utils.h"
#include "register/op_def_registry.h"
#include "../op_kernel/gaussian_nll_loss_tiling_data.h"
#include "../op_kernel/gaussian_nll_loss_tiling_key.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>

namespace optiling {
namespace {
constexpr uint64_t kTensorQueueCount = 4; // input, target, var, and loss
constexpr uint64_t kDoubleBufferDepth = 2;
constexpr uint64_t kFloatCalculationBufferCount = 2; // result and reusable temporary
constexpr uint64_t kHalfUpcastBufferCount = 3;       // input, target, and var
constexpr uint32_t kReductionWorkspaceFloatsPerCore = 8;
constexpr size_t kSystemWorkspaceBytes = 16 * 1024 * 1024;
constexpr float kHalfLogTwoPi = 0.91893853320467274178f;

enum class ReductionMode : uint32_t {
    NONE = GAUSSIAN_NLL_LOSS_REDUCTION_NONE,
    SUM = GAUSSIAN_NLL_LOSS_REDUCTION_SUM,
    MEAN = GAUSSIAN_NLL_LOSS_REDUCTION_MEAN,
};

enum class TargetBroadcastMode : uint32_t {
    NONE = 0,
    ONE_AXIS = 1,
};

enum class VarBroadcastMode : uint32_t {
    NONE = 0,
    TRAILING = 1,
    SCALAR = 2,
};

struct BroadcastInfo {
    TargetBroadcastMode targetMode = TargetBroadcastMode::NONE;
    VarBroadcastMode varMode = VarBroadcastMode::NONE;
    uint64_t targetAxisSpan = 1;
    uint64_t targetInnerSize = 1;
    uint64_t targetElementCount = 1;
    uint64_t varInnerSize = 1;
    uint64_t varElementCount = 1;
};

uint64_t CeilDiv(uint64_t value, uint64_t divisor) { return (value + divisor - 1) / divisor; }

bool FitsUint32(uint64_t value) { return value <= std::numeric_limits<uint32_t>::max(); }

bool SameDimensions(const gert::Shape& lhs, const gert::Shape& rhs)
{
    if (lhs.GetDimNum() != rhs.GetDimNum()) {
        return false;
    }
    for (size_t i = 0; i < lhs.GetDimNum(); ++i) {
        if (lhs.GetDim(i) != rhs.GetDim(i)) {
            return false;
        }
    }
    return true;
}

ge::graphStatus ClassifyTargetBroadcast(gert::TilingContext* context, const gert::Shape& input,
                                        const gert::Shape& target, BroadcastInfo& info)
{
    const int64_t targetSize = target.GetShapeSize();
    OP_CHECK_IF(targetSize < 0, OP_LOGE(context, "unknown target storage shape is not supported"),
                return ge::GRAPH_FAILED);
    info.targetElementCount = static_cast<uint64_t>(targetSize);
    if (SameDimensions(input, target)) {
        return ge::GRAPH_SUCCESS;
    }
    OP_CHECK_IF(input.GetDimNum() != target.GetDimNum(), OP_LOGE(context, "target must have the same rank as input"),
                return ge::GRAPH_FAILED);
    size_t broadcastAxis = input.GetDimNum();
    for (size_t i = 0; i < input.GetDimNum(); ++i) {
        if (input.GetDim(i) == target.GetDim(i)) {
            continue;
        }
        OP_CHECK_IF(target.GetDim(i) != 1 || broadcastAxis != input.GetDimNum(),
                    OP_LOGE(context, "target may broadcast exactly one dimension of size 1"), return ge::GRAPH_FAILED);
        broadcastAxis = i;
    }
    OP_CHECK_IF(broadcastAxis == input.GetDimNum(), OP_LOGE(context, "invalid target broadcast classification"),
                return ge::GRAPH_FAILED);
    uint64_t innerSize = 1;
    for (size_t i = broadcastAxis + 1; i < input.GetDimNum(); ++i) {
        innerSize *= static_cast<uint64_t>(input.GetDim(i));
    }
    info.targetMode = TargetBroadcastMode::ONE_AXIS;
    info.targetInnerSize = innerSize;
    info.targetAxisSpan = static_cast<uint64_t>(input.GetDim(broadcastAxis)) * innerSize;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ClassifyVarBroadcast(gert::TilingContext* context, const gert::Shape& input, const gert::Shape& var,
                                     BroadcastInfo& info)
{
    const int64_t varSize = var.GetShapeSize();
    OP_CHECK_IF(varSize < 0, OP_LOGE(context, "unknown var storage shape is not supported"), return ge::GRAPH_FAILED);
    info.varElementCount = static_cast<uint64_t>(varSize);
    if (SameDimensions(input, var)) {
        return ge::GRAPH_SUCCESS;
    }
    if (var.IsScalar()) {
        info.varMode = VarBroadcastMode::SCALAR;
        return ge::GRAPH_SUCCESS;
    }
    const size_t inputRank = input.GetDimNum();
    const size_t varRank = var.GetDimNum();
    OP_CHECK_IF(inputRank == 0, OP_LOGE(context, "non-scalar var is invalid for scalar input"),
                return ge::GRAPH_FAILED);
    bool trailingBroadcast = false;
    if (varRank == inputRank && var.GetDim(varRank - 1) == 1) {
        trailingBroadcast = true;
        for (size_t i = 0; i + 1 < inputRank; ++i) {
            trailingBroadcast = trailingBroadcast && input.GetDim(i) == var.GetDim(i);
        }
    } else if (varRank + 1 == inputRank) {
        trailingBroadcast = true;
        for (size_t i = 0; i < varRank; ++i) {
            trailingBroadcast = trailingBroadcast && input.GetDim(i) == var.GetDim(i);
        }
    }
    OP_CHECK_IF(
        !trailingBroadcast,
        OP_LOGE(context, "var must match input, have trailing dimension 1, omit the last dimension, or be scalar"),
        return ge::GRAPH_FAILED);
    info.varMode = VarBroadcastMode::TRAILING;
    info.varInnerSize = static_cast<uint64_t>(input.GetDim(inputRank - 1));
    return ge::GRAPH_SUCCESS;
}

uint64_t GetTilingKey(ReductionMode reduction)
{
    if (reduction == ReductionMode::SUM) {
        return GET_TPL_TILING_KEY(GAUSSIAN_NLL_LOSS_REDUCTION_SUM);
    }
    if (reduction == ReductionMode::MEAN) {
        return GET_TPL_TILING_KEY(GAUSSIAN_NLL_LOSS_REDUCTION_MEAN);
    }
    return GET_TPL_TILING_KEY(GAUSSIAN_NLL_LOSS_REDUCTION_NONE);
}
} // namespace

static ge::graphStatus GaussianNllLossTiling(gert::TilingContext* context)
{
    OP_CHECK_IF(context == nullptr, OP_LOGE("GaussianNllLoss", "context is null"), return ge::GRAPH_FAILED);
    const auto* inputShape = context->GetInputShape(0);
    const auto* targetShape = context->GetInputShape(1);
    const auto* varShape = context->GetInputShape(2);
    const auto* lossShape = context->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, targetShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, varShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, lossShape);
    const gert::Shape& input = inputShape->GetStorageShape();
    const int64_t signedTotalElements = input.GetShapeSize();
    OP_CHECK_IF(signedTotalElements < 0, OP_LOGE(context, "unknown input storage shape is not supported"),
                return ge::GRAPH_FAILED);
    const uint64_t totalElements = static_cast<uint64_t>(signedTotalElements);

    BroadcastInfo broadcast;
    OP_CHECK_IF(ClassifyTargetBroadcast(context, input, targetShape->GetStorageShape(), broadcast) != ge::GRAPH_SUCCESS,
                OP_LOGE(context, "invalid target shape"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(ClassifyVarBroadcast(context, input, varShape->GetStorageShape(), broadcast) != ge::GRAPH_SUCCESS,
                OP_LOGE(context, "invalid var shape"), return ge::GRAPH_FAILED);

    const auto* inputDesc = context->GetInputDesc(0);
    const auto* targetDesc = context->GetInputDesc(1);
    const auto* varDesc = context->GetInputDesc(2);
    const auto* lossDesc = context->GetOutputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputDesc);
    OP_CHECK_NULL_WITH_CONTEXT(context, targetDesc);
    OP_CHECK_NULL_WITH_CONTEXT(context, varDesc);
    OP_CHECK_NULL_WITH_CONTEXT(context, lossDesc);
    const ge::DataType inputDtype = inputDesc->GetDataType();
    OP_CHECK_IF(inputDtype != ge::DT_FLOAT && inputDtype != ge::DT_FLOAT16 && inputDtype != ge::DT_BF16,
                OP_LOGE(context, "unsupported dtype"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(targetDesc->GetDataType() != inputDtype || varDesc->GetDataType() != inputDtype ||
                    lossDesc->GetDataType() != inputDtype,
                OP_LOGE(context, "input, target, var, and loss dtypes must match"), return ge::GRAPH_FAILED);

    const auto* attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const bool* fullAttr = attrs->GetAttrPointer<bool>(0);
    const float* epsAttr = attrs->GetAttrPointer<float>(1);
    const char* reductionAttr = attrs->GetAttrPointer<char>(2);
    OP_CHECK_NULL_WITH_CONTEXT(context, fullAttr);
    OP_CHECK_NULL_WITH_CONTEXT(context, epsAttr);
    OP_CHECK_NULL_WITH_CONTEXT(context, reductionAttr);
    OP_CHECK_IF(!std::isfinite(*epsAttr) || *epsAttr <= 0.0f,
                OP_LOGE(context, "eps must be finite and greater than zero"), return ge::GRAPH_FAILED);

    ReductionMode reduction;
    if (std::strcmp(reductionAttr, "none") == 0) {
        reduction = ReductionMode::NONE;
        OP_CHECK_IF(!SameDimensions(lossShape->GetStorageShape(), input),
                    OP_LOGE(context, "none loss shape must match input"), return ge::GRAPH_FAILED);
    } else if (std::strcmp(reductionAttr, "sum") == 0) {
        reduction = ReductionMode::SUM;
    } else if (std::strcmp(reductionAttr, "mean") == 0) {
        reduction = ReductionMode::MEAN;
    } else {
        OP_LOGE(context, "reduction must be none, sum, or mean");
        return ge::GRAPH_FAILED;
    }
    if (reduction != ReductionMode::NONE) {
        OP_CHECK_IF(lossShape->GetStorageShape().GetShapeSize() != 1,
                    OP_LOGE(context, "reduced loss must contain one element"), return ge::GRAPH_FAILED);
    }

    uint32_t elementSizeBytes = 0;
    OP_CHECK_IF(!ge::TypeUtils::GetDataTypeLength(inputDtype, elementSizeBytes) || elementSizeBytes == 0,
                OP_LOGE(context, "failed to get dtype length"), return ge::GRAPH_FAILED);
    const int64_t ubBlockSize = Ops::Base::GetUbBlockSize(context);
    OP_CHECK_IF(ubBlockSize <= 0 || static_cast<uint64_t>(ubBlockSize) % elementSizeBytes != 0,
                OP_LOGE(context, "invalid UB block size"), return ge::GRAPH_FAILED);
    const uint64_t elementsPerUbBlock = static_cast<uint64_t>(ubBlockSize) / elementSizeBytes;

    fe::PlatFormInfos* platformInfo = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    platform_ascendc::PlatformAscendC platform(platformInfo);
    uint64_t unifiedBufferBytes = 0;
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, unifiedBufferBytes);
    const uint64_t availableVectorCoreCount = platform.GetCoreNumAiv();
    OP_CHECK_IF(unifiedBufferBytes == 0 || availableVectorCoreCount == 0,
                OP_LOGE(context, "invalid platform information"), return ge::GRAPH_FAILED);

    uint32_t logMaxLiveNodeCount = 0;
    uint32_t logExtraBufferBytes = 0;
    AscendC::GetLogTmpBufferFactorSize(sizeof(float), logMaxLiveNodeCount, logExtraBufferBytes);

    const uint64_t queuedTensorBytes = kTensorQueueCount * kDoubleBufferDepth * elementSizeBytes;
    const uint64_t floatBufferCount = kFloatCalculationBufferCount +
                                      (inputDtype == ge::DT_FLOAT ? 0 : kHalfUpcastBufferCount) + logMaxLiveNodeCount;
    const uint64_t bytesPerLogicalElement = queuedTensorBytes + floatBufferCount * sizeof(float);
    const bool needsReduction = reduction != ReductionMode::NONE;
    const uint64_t partialSumBufferBytes = kReductionWorkspaceFloatsPerCore * sizeof(float);
    const uint64_t allCorePartialBufferBytes = availableVectorCoreCount * kReductionWorkspaceFloatsPerCore *
                                               sizeof(float);
    const uint64_t reductionScratchBytes = needsReduction ? partialSumBufferBytes + allCorePartialBufferBytes : 0;
    const uint64_t fixedScratchBytes = reductionScratchBytes + logExtraBufferBytes;
    OP_CHECK_IF(unifiedBufferBytes <= fixedScratchBytes, OP_LOGE(context, "UB is too small"), return ge::GRAPH_FAILED);
    const uint64_t unalignedTileElements = (unifiedBufferBytes - fixedScratchBytes) / bytesPerLogicalElement;
    const uint64_t tileElements = (unalignedTileElements / elementsPerUbBlock) * elementsPerUbBlock;
    OP_CHECK_IF(tileElements == 0 || !FitsUint32(tileElements), OP_LOGE(context, "invalid tile size"),
                return ge::GRAPH_FAILED);

    const uint64_t usedCoreCount = totalElements == 0 ? 1 : std::min(totalElements, availableVectorCoreCount);
    const uint64_t smallCoreElements = totalElements / usedCoreCount;
    const uint64_t extraElementCoreCount = totalElements % usedCoreCount;
    const uint64_t bigCoreElements = smallCoreElements + (extraElementCoreCount > 0 ? 1 : 0);
    const uint64_t smallCoreTileCount = smallCoreElements == 0 ? 0 : CeilDiv(smallCoreElements, tileElements);
    const uint64_t bigCoreTileCount = bigCoreElements == 0 ? 0 : CeilDiv(bigCoreElements, tileElements);
    const uint64_t smallCoreLastTileElements = smallCoreElements == 0 ?
                                                   0 :
                                                   smallCoreElements - (smallCoreTileCount - 1) * tileElements;
    const uint64_t bigCoreLastTileElements = bigCoreElements == 0 ?
                                                 0 :
                                                 bigCoreElements - (bigCoreTileCount - 1) * tileElements;
    OP_CHECK_IF(!FitsUint32(smallCoreElements) || !FitsUint32(bigCoreElements) || !FitsUint32(smallCoreTileCount) ||
                    !FitsUint32(bigCoreTileCount) || !FitsUint32(smallCoreLastTileElements) ||
                    !FitsUint32(bigCoreLastTileElements) || !FitsUint32(extraElementCoreCount) ||
                    !FitsUint32(usedCoreCount),
                OP_LOGE(context, "tiling value exceeds uint32_t"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(extraElementCoreCount * bigCoreElements + (usedCoreCount - extraElementCoreCount) * smallCoreElements !=
                    totalElements,
                OP_LOGE(context, "logical element conservation failed"), return ge::GRAPH_FAILED);

    auto* data = context->GetTilingData<GaussianNllLossTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, data);
    data->smallCoreDataNum = static_cast<uint32_t>(smallCoreElements);
    data->bigCoreDataNum = static_cast<uint32_t>(bigCoreElements);
    data->finalBigTileNum = static_cast<uint32_t>(bigCoreTileCount);
    data->finalSmallTileNum = static_cast<uint32_t>(smallCoreTileCount);
    data->tileDataNum = static_cast<uint32_t>(tileElements);
    data->smallTailDataNum = static_cast<uint32_t>(smallCoreLastTileElements);
    data->bigTailDataNum = static_cast<uint32_t>(bigCoreLastTileElements);
    data->tailBlockNum = static_cast<uint32_t>(extraElementCoreCount);
    data->blockNum = static_cast<uint32_t>(usedCoreCount);
    data->workspaceFloatsPerCore = kReductionWorkspaceFloatsPerCore;
    data->targetBroadcastMode = static_cast<uint32_t>(broadcast.targetMode);
    data->varBroadcastMode = static_cast<uint32_t>(broadcast.varMode);
    data->targetAxisSpan = broadcast.targetAxisSpan;
    data->targetInnerSize = broadcast.targetInnerSize;
    data->targetElementCount = broadcast.targetElementCount;
    data->varInnerSize = broadcast.varInnerSize;
    data->varElementCount = broadcast.varElementCount;
    data->eps = *epsAttr;
    data->fullConstant = *fullAttr ? kHalfLogTwoPi : 0.0f;
    data->meanScale = totalElements == 0 ? std::numeric_limits<float>::quiet_NaN() :
                                           1.0f / static_cast<float>(totalElements);

    context->SetTilingKey(GetTilingKey(reduction));
    context->SetBlockDim(static_cast<uint32_t>(usedCoreCount));
    size_t* workspace = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, workspace);
    workspace[0] = needsReduction && usedCoreCount > 1 ?
                       kSystemWorkspaceBytes + usedCoreCount * kReductionWorkspaceFloatsPerCore * sizeof(float) :
                       0;
    if (needsReduction && usedCoreCount > 1) {
        context->SetScheduleMode(1);
    }
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(GaussianNllLoss).Tiling(GaussianNllLossTiling);
} // namespace optiling

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
#include "graph/utils/type_utils.h"
#include "register/op_def_registry.h"
#include "../op_kernel/hinge_embedding_loss_tiling_data.h"
#include "../op_kernel/hinge_embedding_loss_tiling_key.h"
#include <cmath>
#include <cstring>
#include <limits>

namespace optiling {
namespace {
constexpr uint64_t kTensorQueueCount = 3; // input, target, and loss
constexpr uint64_t kDoubleBufferDepth = 2;
constexpr uint64_t kResultFloatBufferCount = 1;
constexpr uint64_t kUpcastFloatBufferCount = 2; // input and target for FLOAT16/BFLOAT16
constexpr uint32_t kReductionWorkspaceFloatsPerCore = 8;
constexpr size_t kSystemWorkspaceBytes = 16 * 1024 * 1024;

enum class ReductionMode : uint32_t {
    NONE = HINGE_EMBEDDING_LOSS_REDUCTION_NONE,
    SUM = HINGE_EMBEDDING_LOSS_REDUCTION_SUM,
    MEAN = HINGE_EMBEDDING_LOSS_REDUCTION_MEAN,
};

uint64_t CeilDiv(uint64_t value, uint64_t divisor) { return (value + divisor - 1) / divisor; }

bool FitsUint32(uint64_t value) { return value <= std::numeric_limits<uint32_t>::max(); }

uint64_t GetTilingKey(ReductionMode reduction)
{
    if (reduction == ReductionMode::SUM) {
        return GET_TPL_TILING_KEY(HINGE_EMBEDDING_LOSS_REDUCTION_SUM);
    }
    if (reduction == ReductionMode::MEAN) {
        return GET_TPL_TILING_KEY(HINGE_EMBEDDING_LOSS_REDUCTION_MEAN);
    }
    return GET_TPL_TILING_KEY(HINGE_EMBEDDING_LOSS_REDUCTION_NONE);
}
} // namespace

static ge::graphStatus HingeEmbeddingLossTiling(gert::TilingContext* context)
{
    OP_CHECK_IF(context == nullptr, OP_LOGE("HingeEmbeddingLoss", "context is null"), return ge::GRAPH_FAILED);
    const auto* inputShape = context->GetInputShape(0);
    const auto* targetShape = context->GetInputShape(1);
    const auto* outputShape = context->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, targetShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, outputShape);
    const gert::Shape& input = inputShape->GetStorageShape();
    const gert::Shape& target = targetShape->GetStorageShape();
    OP_CHECK_IF(input != target, OP_LOGE(context, "input and target shapes must match"), return ge::GRAPH_FAILED);
    const int64_t signedTotalElements = input.GetShapeSize();
    OP_CHECK_IF(signedTotalElements < 0, OP_LOGE(context, "unknown storage shape is not supported"),
                return ge::GRAPH_FAILED);
    const uint64_t totalElements = static_cast<uint64_t>(signedTotalElements);
    const auto* inputDesc = context->GetInputDesc(0);
    const auto* targetDesc = context->GetInputDesc(1);
    const auto* outputDesc = context->GetOutputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputDesc);
    OP_CHECK_NULL_WITH_CONTEXT(context, targetDesc);
    OP_CHECK_NULL_WITH_CONTEXT(context, outputDesc);
    const ge::DataType inputDtype = inputDesc->GetDataType();
    OP_CHECK_IF(inputDtype != ge::DT_FLOAT && inputDtype != ge::DT_FLOAT16 && inputDtype != ge::DT_BF16,
                OP_LOGE(context, "unsupported dtype"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(targetDesc->GetDataType() != inputDtype || outputDesc->GetDataType() != inputDtype,
                OP_LOGE(context, "input, target, and loss dtypes must match"), return ge::GRAPH_FAILED);

    const auto* attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const float* marginAttr = attrs->GetAttrPointer<float>(0);
    const char* reductionAttr = attrs->GetAttrPointer<char>(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, reductionAttr);
    const float margin = marginAttr == nullptr ? 1.0f : *marginAttr;
    OP_CHECK_IF(!std::isfinite(margin), OP_LOGE(context, "margin must be finite"), return ge::GRAPH_FAILED);
    ReductionMode reduction;
    if (std::strcmp(reductionAttr, "none") == 0) {
        reduction = ReductionMode::NONE;
        OP_CHECK_IF(outputShape->GetStorageShape() != input, OP_LOGE(context, "none output shape must match input"),
                    return ge::GRAPH_FAILED);
    } else if (std::strcmp(reductionAttr, "sum") == 0) {
        reduction = ReductionMode::SUM;
    } else if (std::strcmp(reductionAttr, "mean") == 0) {
        reduction = ReductionMode::MEAN;
    } else {
        OP_LOGE(context, "reduction must be none, sum, or mean");
        return ge::GRAPH_FAILED;
    }
    if (reduction != ReductionMode::NONE) {
        OP_CHECK_IF(outputShape->GetStorageShape().GetShapeSize() != 1,
                    OP_LOGE(context, "reduced output must contain one element"), return ge::GRAPH_FAILED);
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

    // Three tensor queues use double buffering, so each logical element occupies
    // six queue slots. Half types also reserve two FLOAT buffers for upcasting.
    // Reduction modes need one local partial and a core-0 download area.
    const uint64_t queuedTensorCopies = kTensorQueueCount * kDoubleBufferDepth;
    const uint64_t floatCalculationBuffers = kResultFloatBufferCount +
                                             (inputDtype == ge::DT_FLOAT ? 0 : kUpcastFloatBufferCount);
    const uint64_t bytesPerLogicalElement = queuedTensorCopies * elementSizeBytes +
                                            floatCalculationBuffers * sizeof(float);
    const bool needsReduction = reduction != ReductionMode::NONE;
    const uint64_t partialSumBufferBytes = kReductionWorkspaceFloatsPerCore * sizeof(float);
    const uint64_t allCorePartialDownloadBytes = availableVectorCoreCount * kReductionWorkspaceFloatsPerCore *
                                                 sizeof(float);
    const uint64_t reductionScratchBytes = needsReduction ? partialSumBufferBytes + allCorePartialDownloadBytes : 0;
    OP_CHECK_IF(unifiedBufferBytes <= reductionScratchBytes, OP_LOGE(context, "UB is too small"),
                return ge::GRAPH_FAILED);
    const uint64_t unalignedTileElements = (unifiedBufferBytes - reductionScratchBytes) / bytesPerLogicalElement;
    const uint64_t tileElements = (unalignedTileElements / elementsPerUbBlock) * elementsPerUbBlock;
    OP_CHECK_IF(tileElements == 0 || !FitsUint32(tileElements), OP_LOGE(context, "invalid tile size"),
                return ge::GRAPH_FAILED);
    const uint64_t usedCoreCount = totalElements == 0 ?
                                       1 :
                                       (totalElements < availableVectorCoreCount ? totalElements :
                                                                                   availableVectorCoreCount);
    const uint64_t smallCoreElements = totalElements / usedCoreCount;
    // The first extraElementCoreCount cores receive one more logical element.
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
                    !FitsUint32(bigCoreLastTileElements) || !FitsUint32(extraElementCoreCount),
                OP_LOGE(context, "tiling value exceeds uint32_t"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(extraElementCoreCount * bigCoreElements + (usedCoreCount - extraElementCoreCount) * smallCoreElements !=
                    totalElements,
                OP_LOGE(context, "logical element conservation failed"), return ge::GRAPH_FAILED);

    auto* data = context->GetTilingData<HingeEmbeddingLossTilingData>();
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
    data->margin = margin;
    data->meanScale = totalElements == 0 ? std::numeric_limits<float>::quiet_NaN() :
                                           1.0f / static_cast<float>(totalElements);
    context->SetTilingKey(GetTilingKey(reduction));
    context->SetBlockDim(static_cast<uint32_t>(usedCoreCount));
    size_t* workspace = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, workspace);
    workspace[0] = needsReduction && usedCoreCount > 1 ?
                       kSystemWorkspaceBytes + usedCoreCount * data->workspaceFloatsPerCore * sizeof(float) :
                       0;
    if (needsReduction && usedCoreCount > 1) {
        context->SetScheduleMode(1);
    }
    return ge::GRAPH_SUCCESS;
}
IMPL_OP_OPTILING(HingeEmbeddingLoss).Tiling(HingeEmbeddingLossTiling);
} // namespace optiling

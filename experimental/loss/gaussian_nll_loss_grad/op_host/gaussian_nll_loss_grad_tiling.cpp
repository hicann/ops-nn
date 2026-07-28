/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <algorithm>
#include <cstring>
#include <limits>

#include "op_common/log/log.h"
#include "op_common/op_host/util/platform_util.h"
#include "register/op_def_registry.h"
#include "../op_kernel/gaussian_nll_loss_grad_tiling_data.h"

namespace optiling {
namespace {
constexpr uint32_t REDUCTION_NONE = 0;
constexpr uint32_t REDUCTION_SUM = 1;
constexpr uint32_t REDUCTION_MEAN = 2;
constexpr uint32_t BROADCAST_NONE = 0;
constexpr uint32_t BROADCAST_TARGET_AXIS = 1;
constexpr uint32_t VAR_SAME = 0;
constexpr uint32_t VAR_LAST_DIM_ONE = 1;
constexpr uint32_t VAR_MISSING_LAST_DIM = 2;
constexpr uint32_t VAR_SCALAR = 3;
constexpr uint64_t MAX_TILE_ELEMS = 4096;
constexpr uint64_t ALIGN_ELEMS = 8;
constexpr uint64_t RAW_BUFFER_COUNT = 5;
constexpr uint64_t FLOAT_BUFFER_COUNT = 6;
constexpr uint64_t RAW_BUFFER_EXTRA_BYTES = 32;
constexpr uint64_t UB_BYTES_PER_LOGICAL_ELEMENT = (RAW_BUFFER_COUNT + FLOAT_BUFFER_COUNT) * sizeof(float);
constexpr uint64_t UB_FIXED_BYTES = RAW_BUFFER_COUNT * RAW_BUFFER_EXTRA_BYTES;

struct BroadcastInfo {
    uint32_t targetMode = BROADCAST_NONE;
    uint64_t targetAxisSize = 1;
    uint64_t targetInnerStride = 1;
    uint32_t varMode = VAR_SAME;
    uint64_t varReduceSize = 1;
};

static bool FitsU32(uint64_t value) { return value <= std::numeric_limits<uint32_t>::max(); }

static uint64_t CeilDiv(uint64_t value, uint64_t divisor) { return (value + divisor - 1) / divisor; }

static bool SameShape(const gert::Shape& lhs, const gert::Shape& rhs)
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

static bool HasKnownNonNegativeDims(const gert::Shape& shape)
{
    for (size_t i = 0; i < shape.GetDimNum(); ++i) {
        if (shape.GetDim(i) < 0) {
            return false;
        }
    }
    return true;
}

static bool ClassifyTarget(const gert::Shape& input, const gert::Shape& target, BroadcastInfo& info)
{
    if (SameShape(input, target)) {
        return true;
    }
    if (input.GetDimNum() != target.GetDimNum()) {
        return false;
    }
    size_t broadcastAxis = input.GetDimNum();
    for (size_t i = 0; i < input.GetDimNum(); ++i) {
        if (input.GetDim(i) == target.GetDim(i)) {
            continue;
        }
        if (broadcastAxis != input.GetDimNum() || target.GetDim(i) != 1) {
            return false;
        }
        broadcastAxis = i;
    }
    if (broadcastAxis == input.GetDimNum()) {
        return false;
    }
    uint64_t innerStride = 1;
    for (size_t i = broadcastAxis + 1; i < input.GetDimNum(); ++i) {
        innerStride *= static_cast<uint64_t>(input.GetDim(i));
    }
    info.targetMode = BROADCAST_TARGET_AXIS;
    info.targetAxisSize = static_cast<uint64_t>(input.GetDim(broadcastAxis));
    info.targetInnerStride = innerStride;
    return true;
}

static bool PrefixMatches(const gert::Shape& input, const gert::Shape& var, size_t prefixDims)
{
    if (var.GetDimNum() != prefixDims) {
        return false;
    }
    for (size_t i = 0; i < prefixDims; ++i) {
        if (input.GetDim(i) != var.GetDim(i)) {
            return false;
        }
    }
    return true;
}

static bool ClassifyVar(const gert::Shape& input, const gert::Shape& var, BroadcastInfo& info)
{
    if (SameShape(input, var)) {
        return true;
    }
    if (var.GetShapeSize() == 1) {
        info.varMode = VAR_SCALAR;
        info.varReduceSize = static_cast<uint64_t>(input.GetShapeSize());
        return true;
    }
    if (input.GetDimNum() == 0) {
        return false;
    }
    const size_t last = input.GetDimNum() - 1;
    if (var.GetDimNum() == input.GetDimNum() && var.GetDim(last) == 1) {
        bool prefixMatches = true;
        for (size_t i = 0; i < last; ++i) {
            if (input.GetDim(i) != var.GetDim(i)) {
                prefixMatches = false;
                break;
            }
        }
        if (prefixMatches) {
            info.varMode = VAR_LAST_DIM_ONE;
            info.varReduceSize = static_cast<uint64_t>(input.GetDim(last));
            return true;
        }
    }
    if (PrefixMatches(input, var, last)) {
        info.varMode = VAR_MISSING_LAST_DIM;
        info.varReduceSize = static_cast<uint64_t>(input.GetDim(last));
        return true;
    }
    return false;
}

static bool IsScalarShape(const gert::Shape& shape) { return shape.GetShapeSize() == 1; }

static ge::graphStatus ValidateDtypes(gert::TilingContext* context)
{
    const auto* firstDesc = context->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, firstDesc);
    const ge::DataType dtype = firstDesc->GetDataType();
    OP_CHECK_IF(dtype != ge::DT_FLOAT && dtype != ge::DT_FLOAT16 && dtype != ge::DT_BF16,
                OP_LOGE(context, "unsupported dtype %d", static_cast<int32_t>(dtype)), return ge::GRAPH_FAILED);
    for (size_t i = 1; i < 4; ++i) {
        const auto* desc = context->GetInputDesc(i);
        OP_CHECK_NULL_WITH_CONTEXT(context, desc);
        OP_CHECK_IF(desc->GetDataType() != dtype, OP_LOGE(context, "all inputs must have the same dtype"),
                    return ge::GRAPH_FAILED);
    }
    for (size_t i = 0; i < 2; ++i) {
        const auto* desc = context->GetOutputDesc(i);
        OP_CHECK_NULL_WITH_CONTEXT(context, desc);
        OP_CHECK_IF(desc->GetDataType() != dtype, OP_LOGE(context, "outputs must match gradOutput dtype"),
                    return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}
} // namespace

static ge::graphStatus GaussianNllLossGradTiling(gert::TilingContext* context)
{
    OP_CHECK_IF(context == nullptr, OP_LOGE("GaussianNllLossGrad", "context is null"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(ValidateDtypes(context) != ge::GRAPH_SUCCESS, OP_LOGE(context, "dtype validation failed"),
                return ge::GRAPH_FAILED);

    const auto* gradOutputShapeInfo = context->GetInputShape(0);
    const auto* inputShapeInfo = context->GetInputShape(1);
    const auto* targetShapeInfo = context->GetInputShape(2);
    const auto* varShapeInfo = context->GetInputShape(3);
    const auto* gradInputShapeInfo = context->GetOutputShape(0);
    const auto* gradVarShapeInfo = context->GetOutputShape(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, gradOutputShapeInfo);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputShapeInfo);
    OP_CHECK_NULL_WITH_CONTEXT(context, targetShapeInfo);
    OP_CHECK_NULL_WITH_CONTEXT(context, varShapeInfo);
    OP_CHECK_NULL_WITH_CONTEXT(context, gradInputShapeInfo);
    OP_CHECK_NULL_WITH_CONTEXT(context, gradVarShapeInfo);

    const auto& gradOutputShape = gradOutputShapeInfo->GetStorageShape();
    const auto& inputShape = inputShapeInfo->GetStorageShape();
    const auto& targetShape = targetShapeInfo->GetStorageShape();
    const auto& varShape = varShapeInfo->GetStorageShape();
    const auto& gradInputShape = gradInputShapeInfo->GetStorageShape();
    const auto& gradVarShape = gradVarShapeInfo->GetStorageShape();
    OP_CHECK_IF(!HasKnownNonNegativeDims(gradOutputShape) || !HasKnownNonNegativeDims(inputShape) ||
                    !HasKnownNonNegativeDims(targetShape) || !HasKnownNonNegativeDims(varShape),
                OP_LOGE(context, "tiling requires known non-negative dimensions"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(!SameShape(inputShape, gradInputShape) || !SameShape(varShape, gradVarShape),
                OP_LOGE(context, "output shapes must be input shape and var shape"), return ge::GRAPH_FAILED);

    BroadcastInfo broadcast;
    OP_CHECK_IF(!ClassifyTarget(inputShape, targetShape, broadcast),
                OP_LOGE(context, "target must match input or broadcast in exactly one size-1 dimension"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(!ClassifyVar(inputShape, varShape, broadcast),
                OP_LOGE(context, "var shape is not a supported broadcast form"), return ge::GRAPH_FAILED);

    const auto* attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const bool* full = attrs->GetBool(0);
    const float* eps = attrs->GetAttrPointer<float>(1);
    const char* reductionText = attrs->GetStr(2);
    OP_CHECK_NULL_WITH_CONTEXT(context, full);
    OP_CHECK_NULL_WITH_CONTEXT(context, eps);
    OP_CHECK_NULL_WITH_CONTEXT(context, reductionText);
    (void)full;
    OP_CHECK_IF(!(*eps > 0.0f), OP_LOGE(context, "eps must be greater than zero"), return ge::GRAPH_FAILED);

    uint32_t reduction = REDUCTION_MEAN;
    if (std::strcmp(reductionText, "none") == 0) {
        reduction = REDUCTION_NONE;
        OP_CHECK_IF(!SameShape(gradOutputShape, inputShape),
                    OP_LOGE(context, "none reduction requires gradOutput to match input"), return ge::GRAPH_FAILED);
    } else if (std::strcmp(reductionText, "sum") == 0) {
        reduction = REDUCTION_SUM;
        OP_CHECK_IF(!IsScalarShape(gradOutputShape), OP_LOGE(context, "sum reduction requires scalar gradOutput"),
                    return ge::GRAPH_FAILED);
    } else if (std::strcmp(reductionText, "mean") == 0) {
        OP_CHECK_IF(!IsScalarShape(gradOutputShape), OP_LOGE(context, "mean reduction requires scalar gradOutput"),
                    return ge::GRAPH_FAILED);
    } else {
        OP_LOGE(context, "invalid reduction %s", reductionText);
        return ge::GRAPH_FAILED;
    }

    const int64_t totalSigned = inputShape.GetShapeSize();
    const int64_t targetSigned = targetShape.GetShapeSize();
    const int64_t varSigned = varShape.GetShapeSize();
    OP_CHECK_IF(totalSigned < 0 || targetSigned < 0 || varSigned < 0,
                OP_LOGE(context, "shape size must be non-negative"), return ge::GRAPH_FAILED);
    const uint64_t total = static_cast<uint64_t>(totalSigned);
    const uint64_t targetDataNum = static_cast<uint64_t>(targetSigned);
    const uint64_t varDataNum = static_cast<uint64_t>(varSigned);
    OP_CHECK_IF(!FitsU32(total) || !FitsU32(targetDataNum) || !FitsU32(varDataNum) ||
                    !FitsU32(broadcast.targetAxisSize) || !FitsU32(broadcast.targetInnerStride) ||
                    !FitsU32(broadcast.varReduceSize),
                OP_LOGE(context, "logical shape exceeds uint32 tiling range"), return ge::GRAPH_FAILED);

    fe::PlatFormInfos* platformInfo = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    platform_ascendc::PlatformAscendC platform(platformInfo);
    uint64_t ubBytes = 0;
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubBytes);
    const uint64_t availableCores = platform.GetCoreNumAiv();
    const uint64_t minimumUbBytes = UB_FIXED_BYTES + UB_BYTES_PER_LOGICAL_ELEMENT * ALIGN_ELEMS;
    OP_CHECK_IF(availableCores == 0 || ubBytes < minimumUbBytes,
                OP_LOGE(context, "insufficient platform core or UB information"), return ge::GRAPH_FAILED);
    uint64_t tile = (ubBytes - UB_FIXED_BYTES) / UB_BYTES_PER_LOGICAL_ELEMENT;
    tile = std::min(tile / ALIGN_ELEMS * ALIGN_ELEMS, MAX_TILE_ELEMS);
    OP_CHECK_IF(tile == 0 || !FitsU32(tile), OP_LOGE(context, "invalid tile size"), return ge::GRAPH_FAILED);

    const uint64_t usedCores = total == 0 ? 1 : std::min(total, availableCores);
    const uint64_t small = total / usedCores;
    const uint64_t tailCores = total % usedCores;
    const uint64_t big = small + (tailCores > 0 ? 1 : 0);
    const uint64_t smallTiles = small == 0 ? 0 : CeilDiv(small, tile);
    const uint64_t bigTiles = big == 0 ? 0 : CeilDiv(big, tile);
    const uint64_t smallTail = small == 0 ? 0 : small - (smallTiles - 1) * tile;
    const uint64_t bigTail = big == 0 ? 0 : big - (bigTiles - 1) * tile;
    OP_CHECK_IF(tailCores * big + (usedCores - tailCores) * small != total,
                OP_LOGE(context, "logical element conservation failed"), return ge::GRAPH_FAILED);

    auto* data = context->GetTilingData<GaussianNllLossGradTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, data);
    data->smallCoreDataNum = static_cast<uint32_t>(small);
    data->bigCoreDataNum = static_cast<uint32_t>(big);
    data->finalBigTileNum = static_cast<uint32_t>(bigTiles);
    data->finalSmallTileNum = static_cast<uint32_t>(smallTiles);
    data->tileDataNum = static_cast<uint32_t>(tile);
    data->smallTailDataNum = static_cast<uint32_t>(smallTail);
    data->bigTailDataNum = static_cast<uint32_t>(bigTail);
    data->tailBlockNum = static_cast<uint32_t>(tailCores);
    data->totalDataNum = static_cast<uint32_t>(total);
    data->targetDataNum = static_cast<uint32_t>(targetDataNum);
    data->varDataNum = static_cast<uint32_t>(varDataNum);
    data->targetBroadcastAxisSize = static_cast<uint32_t>(broadcast.targetAxisSize);
    data->targetInnerStride = static_cast<uint32_t>(broadcast.targetInnerStride);
    data->targetBroadcastMode = broadcast.targetMode;
    data->varBroadcastMode = broadcast.varMode;
    data->varReduceSize = static_cast<uint32_t>(broadcast.varReduceSize);
    data->reduction = reduction;
    data->eps = *eps;
    data->meanScale = total == 0 ? 0.0f : 1.0f / static_cast<float>(total);
    context->SetBlockDim(static_cast<uint32_t>(usedCores));
    context->SetTilingKey(0);
    size_t* workspace = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, workspace);
    workspace[0] = 0;
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(GaussianNllLossGrad).Tiling(GaussianNllLossGradTiling);
} // namespace optiling

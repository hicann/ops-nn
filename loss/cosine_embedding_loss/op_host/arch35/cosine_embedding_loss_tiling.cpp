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
 * \file cosine_embedding_loss_tiling.cpp
 * \brief CosineEmbeddingLoss RegBase (arch35 / ascend950) tiling.
 *
 * Broadcasts x1/x2 to a common ND shape, reduces fixed axis 1, then broadcasts target
 * with the reduced x shape. The kernel uses the generated row-major strides to support
 * non-contiguous axis-1 reduction and broadcasted inputs.
 */
#include "cosine_embedding_loss_tiling.h"

#include <algorithm>
#include <cstring>
#include <limits>
#include <securec.h>
#include <string>
#include <vector>

#include "error_util.h"
#include "graph/utils/type_utils.h"
#include "log/log.h"
#include "tiling/platform/platform_ascendc.h"
#include "tiling/tiling_api.h"
#include "util/math_util.h"
#include "util/shape_util.h"
#include "register/op_impl_registry.h"
#include "../cosine_embedding_loss_common.h"
#include "../../op_kernel/arch35/cosine_embedding_loss_tiling_data.h"

using namespace ge;

namespace optiling {
namespace {
constexpr size_t INPUT_X1 = 0;
constexpr size_t INPUT_X2 = 1;
constexpr size_t INPUT_TARGET = 2;
constexpr size_t ATTR_MARGIN = 0;
constexpr size_t ATTR_REDUCTION = 1;
constexpr int64_t D_ALIGN = 16;
constexpr int64_t VECTOR_LENGTH_FP32 = 64;
constexpr int64_t OUTPUT_BLOCK_ELEMENTS_FP32 = 8;
constexpr uint64_t UB_RESERVED_BYTES = 4UL * 1024UL;
constexpr size_t SYS_WORKSPACE_BYTES = 16UL * 1024UL * 1024UL;
namespace cel = ops::cosine_embedding_loss;

bool IsSupportedDtype(ge::DataType dt) { return dt == ge::DT_FLOAT || dt == ge::DT_FLOAT16 || dt == ge::DT_INT32; }

bool CheckedMul(int64_t lhs, int64_t rhs, int64_t& out)
{
    if (lhs != 0 && rhs > std::numeric_limits<int64_t>::max() / lhs) {
        return false;
    }
    out = lhs * rhs;
    return true;
}

bool CheckedAlignUp(int64_t value, int64_t alignment, int64_t& aligned)
{
    if (value <= 0 || alignment <= 0) {
        return false;
    }
    const int64_t remainder = value % alignment;
    const int64_t increment = remainder == 0 ? 0 : alignment - remainder;
    if (value > std::numeric_limits<int64_t>::max() - increment) {
        return false;
    }
    aligned = value + increment;
    return true;
}

std::string DimsToString(const std::vector<int64_t>& dims)
{
    gert::Shape shape;
    shape.SetDimNum(dims.size());
    for (size_t i = 0; i < dims.size(); ++i) {
        shape.SetDim(i, dims[i]);
    }
    return Ops::Base::ToString(shape);
}

bool MakeRowMajorStrides(const std::vector<int64_t>& dims, std::vector<int64_t>& strides)
{
    strides.assign(dims.size(), 0);
    int64_t running = 1;
    for (int64_t i = static_cast<int64_t>(dims.size()) - 1; i >= 0; --i) {
        strides[static_cast<size_t>(i)] = running;
        if (!CheckedMul(running, dims[static_cast<size_t>(i)], running)) {
            return false;
        }
    }
    return true;
}

bool MakeBroadcastFullStrides(const std::vector<int64_t>& inputShape, const std::vector<int64_t>& broadcastShape,
                              std::vector<int64_t>& fullStrides)
{
    std::vector<int64_t> inputStrides;
    if (!MakeRowMajorStrides(inputShape, inputStrides)) {
        return false;
    }
    const size_t rankOffset = broadcastShape.size() - inputShape.size();
    fullStrides.assign(broadcastShape.size(), 0);
    for (size_t axis = 0; axis < broadcastShape.size(); ++axis) {
        if (axis < rankOffset) {
            continue;
        }
        const size_t inAxis = axis - rankOffset;
        if (inputShape[inAxis] != 1 || broadcastShape[axis] == 1) {
            fullStrides[axis] = inputStrides[inAxis];
        }
    }
    return true;
}

void FillOutputStridesFromFull(const std::vector<int64_t>& reducedShape, const std::vector<int64_t>& outputShape,
                               const std::vector<int64_t>& fullStrides, int64_t outStrides[])
{
    const int64_t offset = static_cast<int64_t>(outputShape.size()) - static_cast<int64_t>(reducedShape.size());
    for (size_t reducedAxis = 0; reducedAxis < reducedShape.size(); ++reducedAxis) {
        const int64_t outAxis = offset + static_cast<int64_t>(reducedAxis);
        const size_t fullAxis = reducedAxis == 0 ? 0 : reducedAxis + 1;
        if (outAxis >= 0 && (reducedShape[reducedAxis] != 1 || outputShape[static_cast<size_t>(outAxis)] == 1)) {
            outStrides[static_cast<size_t>(outAxis)] = fullStrides[fullAxis];
        }
    }
}

bool FillTargetOutputStrides(const std::vector<int64_t>& targetShape, const std::vector<int64_t>& outputShape,
                             int64_t outStrides[])
{
    std::vector<int64_t> targetStrides;
    if (!MakeRowMajorStrides(targetShape, targetStrides)) {
        return false;
    }
    const int64_t offset = static_cast<int64_t>(outputShape.size()) - static_cast<int64_t>(targetShape.size());
    for (size_t targetAxis = 0; targetAxis < targetShape.size(); ++targetAxis) {
        const int64_t outAxis = offset + static_cast<int64_t>(targetAxis);
        if (outAxis >= 0 && (targetShape[targetAxis] != 1 || outputShape[static_cast<size_t>(outAxis)] == 1)) {
            outStrides[static_cast<size_t>(outAxis)] = targetStrides[targetAxis];
        }
    }
    return true;
}

int64_t DtypeBytes(ge::DataType dtype)
{
    return dtype == ge::DT_FLOAT16 ? static_cast<int64_t>(sizeof(uint16_t)) : static_cast<int64_t>(sizeof(uint32_t));
}

ge::graphStatus GetPlatform(gert::TilingContext* context, int64_t& coreNum, uint64_t& ubSize)
{
    auto platformInfo = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    auto platform = platform_ascendc::PlatformAscendC(platformInfo);
    coreNum = static_cast<int64_t>(platform.GetCoreNumAiv());
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    OP_CHECK_IF(coreNum <= 0 || ubSize == 0,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "platform", "invalid",
                                                      "AIV core num and UB size should be positive"),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CheckSupportedDtype(gert::TilingContext* context, const char* name, ge::DataType dtype)
{
    OP_CHECK_IF(!IsSupportedDtype(dtype),
                OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(context->GetNodeName(), name,
                                                      ge::TypeUtils::DataTypeToSerialString(dtype).c_str(),
                                                      "dtype should be int32, float16 or float32"),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}
} // namespace

static ge::graphStatus TilingForCosineEmbeddingLoss(gert::TilingContext* context)
{
    OP_CHECK_NULL_WITH_CONTEXT(context, context);
    int64_t coreNum = 0;
    uint64_t ubSize = 0;
    OP_CHECK_IF(
        GetPlatform(context, coreNum, ubSize) != ge::GRAPH_SUCCESS,
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "parameter", "invalid", "get platform failed"),
        return ge::GRAPH_FAILED);

    auto x1Desc = context->GetInputDesc(INPUT_X1);
    OP_CHECK_NULL_WITH_CONTEXT(context, x1Desc);
    auto x2Desc = context->GetInputDesc(INPUT_X2);
    OP_CHECK_NULL_WITH_CONTEXT(context, x2Desc);
    auto targetDesc = context->GetInputDesc(INPUT_TARGET);
    OP_CHECK_NULL_WITH_CONTEXT(context, targetDesc);
    auto x1Shape = context->GetInputShape(INPUT_X1);
    OP_CHECK_NULL_WITH_CONTEXT(context, x1Shape);
    auto x2Shape = context->GetInputShape(INPUT_X2);
    OP_CHECK_NULL_WITH_CONTEXT(context, x2Shape);
    auto targetShape = context->GetInputShape(INPUT_TARGET);
    OP_CHECK_NULL_WITH_CONTEXT(context, targetShape);
    const gert::Shape& shp = x1Shape->GetStorageShape();
    const gert::Shape& x2StorageShape = x2Shape->GetStorageShape();
    const gert::Shape& targetStorageShape = targetShape->GetStorageShape();
    const ge::DataType x1Dt = x1Desc->GetDataType();
    const ge::DataType x2Dt = x2Desc->GetDataType();
    const ge::DataType targetDt = targetDesc->GetDataType();
    if (CheckSupportedDtype(context, "x1", x1Dt) != ge::GRAPH_SUCCESS ||
        CheckSupportedDtype(context, "x2", x2Dt) != ge::GRAPH_SUCCESS ||
        CheckSupportedDtype(context, "target", targetDt) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (x1Dt != x2Dt) {
        const std::string dtypeMsg = ge::TypeUtils::DataTypeToSerialString(x1Dt) + " and " +
                                     ge::TypeUtils::DataTypeToSerialString(x2Dt);
        OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(context->GetNodeName(), "x1 and x2", dtypeMsg.c_str(),
                                               "x1 and x2 should have the same dtype");
        return ge::GRAPH_FAILED;
    }

    cel::Dims x1Dims;
    cel::Dims x2Dims;
    cel::Dims targetDims;
    OP_CHECK_IF(
        !cel::RuntimeShapeToDims(shp, x1Dims) || !cel::RuntimeShapeToDims(x2StorageShape, x2Dims) ||
            !cel::RuntimeShapeToDims(targetStorageShape, targetDims),
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context->GetNodeName(), "input", "invalid shape",
                                              "rank should be in [1, 8] and all runtime dimensions should be > 0"),
        return ge::GRAPH_FAILED);

    cel::Dims xBroadcastDims;
    OP_CHECK_IF(!cel::BroadcastShapes(x1Dims, x2Dims, xBroadcastDims),
                OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                    context->GetNodeName(), "x1 and x2",
                    (Ops::Base::ToString(shp) + " and " + Ops::Base::ToString(x2StorageShape)).c_str(),
                    "x1 and x2 should be broadcastable"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(xBroadcastDims.size() < 2,
                OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(context->GetNodeName(), "x1 and x2",
                                                         std::to_string(xBroadcastDims.size()).c_str(),
                                                         "broadcast rank should be at least 2 for axis=1 reduction"),
                return ge::GRAPH_FAILED);

    cel::Dims xReducedDims;
    OP_CHECK_IF(
        !cel::RemoveAxis(xBroadcastDims, cel::kFeatureAxis, xReducedDims),
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "axis", "1", "failed to remove the feature axis"),
        return ge::GRAPH_FAILED);
    cel::Dims outputDims;
    OP_CHECK_IF(!cel::BroadcastShapes(xReducedDims, targetDims, outputDims),
                OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                    context->GetNodeName(), "target",
                    (Ops::Base::ToString(targetStorageShape) + " to " + DimsToString(xReducedDims)).c_str(),
                    "target should be broadcastable with x1/x2 shape after reducing axis 1"),
                return ge::GRAPH_FAILED);

    int64_t n = 0;
    OP_CHECK_IF(!cel::ElementCount(outputDims, n) || n <= 0,
                OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(context->GetNodeName(), "y", "invalid shape size",
                                                          "output element count should be positive and fit int64"),
                return ge::GRAPH_FAILED);
    const int64_t d = xBroadcastDims[1];

    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const float* marginPtr = attrs->GetAttrPointer<float>(ATTR_MARGIN);
    const char* reductionStr = attrs->GetAttrPointer<char>(ATTR_REDUCTION);
    OP_CHECK_NULL_WITH_CONTEXT(context, reductionStr);
    const float margin = marginPtr != nullptr ? *marginPtr : 0.0f;
    uint32_t reduction = cel::kReductionMean;
    if (!cel::ParseReduction(reductionStr, reduction)) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "reduction", reductionStr,
                                              "reduction should be none, sum or mean");
        return ge::GRAPH_FAILED;
    }

    int64_t dAlign = 0;
    OP_CHECK_IF(!CheckedAlignUp(d, D_ALIGN, dAlign),
                OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(context->GetNodeName(), "x1 and x2", "invalid shape size",
                                                          "aligned feature dimension should fit int64"),
                return ge::GRAPH_FAILED);

    // All modes split output rows across cores. sum/mean merge one fp32 partial per core
    // through 32B-aligned workspace slots after the per-core row computation.
    const int64_t availableCoreNum = std::min(coreNum, COSINE_EMBEDDING_LOSS_MAX_CORE_NUM);
    int64_t rowsPerCore = Ops::Base::CeilDiv(n, availableCoreNum);
    int64_t usedCoreNum = Ops::Base::CeilDiv(n, rowsPerCore);
    int64_t tailRows = n - rowsPerCore * (usedCoreNum - 1);

    auto tiling = context->GetTilingData<CosineEmbeddingLossTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
    OP_CHECK_IF(
        memset_s(tiling, sizeof(*tiling), 0, sizeof(*tiling)) != EOK,
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "parameter", "invalid", "memset tiling failed"),
        return ge::GRAPH_FAILED);
    tiling->n = n;
    tiling->d = d;
    tiling->dAlign = dAlign;
    tiling->rowsPerCore = rowsPerCore;
    tiling->tailRows = tailRows;
    tiling->usedCoreNum = usedCoreNum;
    tiling->ubTileRows = 1;
    tiling->featureTile = 0;
    tiling->reduceTmpBytes = 0;
    tiling->reduction = reduction;
    tiling->outputRank = static_cast<uint32_t>(outputDims.size());
    tiling->xBroadcastRank = static_cast<uint32_t>(xBroadcastDims.size());
    tiling->margin = margin;
    tiling->meanCoef = (reduction == cel::kReductionMean && n > 0) ? (1.0f / static_cast<float>(n)) : 1.0f;
    tiling->eps = 1e-12f; // A2/A3 legacy: added inside sqrt(sum(x^2)+eps)
    for (size_t i = 0; i < outputDims.size(); ++i) {
        tiling->outputShape[i] = outputDims[i];
    }

    std::vector<int64_t> x1FullStrides;
    std::vector<int64_t> x2FullStrides;
    OP_CHECK_IF(!MakeBroadcastFullStrides(x1Dims, xBroadcastDims, x1FullStrides) ||
                    !MakeBroadcastFullStrides(x2Dims, xBroadcastDims, x2FullStrides) ||
                    !FillTargetOutputStrides(targetDims, outputDims, tiling->targetOutStrides),
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "stride", "invalid",
                                                      "failed to build row-major broadcast strides"),
                return ge::GRAPH_FAILED);
    FillOutputStridesFromFull(xReducedDims, outputDims, x1FullStrides, tiling->x1OutStrides);
    FillOutputStridesFromFull(xReducedDims, outputDims, x2FullStrides, tiling->x2OutStrides);
    tiling->x1ReduceStride = x1FullStrides[1];
    tiling->x2ReduceStride = x2FullStrides[1];

    const bool isContiguous2D = xBroadcastDims.size() == 2 && outputDims.size() == 1 && tiling->x1ReduceStride == 1 &&
                                tiling->x2ReduceStride == 1 &&
                                (tiling->targetOutStrides[0] == 0 || tiling->targetOutStrides[0] == 1);
    if (isContiguous2D && ubSize > UB_RESERVED_BYTES) {
        const int64_t availableBytes = static_cast<int64_t>(ubSize - UB_RESERVED_BYTES);
        const int64_t rawFeatureTile = availableBytes / (2 * DtypeBytes(x1Dt));
        const int64_t maxFeatureTile = Ops::Base::FloorAlign(rawFeatureTile, VECTOR_LENGTH_FP32);
        if (maxFeatureTile >= VECTOR_LENGTH_FP32) {
            const int64_t featureTile = d < maxFeatureTile ? Ops::Base::CeilAlign(d, VECTOR_LENGTH_FP32) :
                                                             maxFeatureTile;
            tiling->fastPath = COSINE_EMBEDDING_LOSS_CONTIG_2D_PATH;
            tiling->featureTile = featureTile;
            tiling->reduceTmpBytes = 32;
        }
    }

    if (tiling->fastPath == COSINE_EMBEDDING_LOSS_CONTIG_2D_PATH && reduction == cel::kReductionNone &&
        usedCoreNum > 1) {
        // Each non-tail core owns complete 32B output blocks, so 4B MTE writes from different cores never overlap.
        rowsPerCore = Ops::Base::CeilAlign(rowsPerCore, OUTPUT_BLOCK_ELEMENTS_FP32);
        usedCoreNum = Ops::Base::CeilDiv(n, rowsPerCore);
        tailRows = n - rowsPerCore * (usedCoreNum - 1);
        tiling->rowsPerCore = rowsPerCore;
        tiling->usedCoreNum = usedCoreNum;
        tiling->tailRows = tailRows;
    }

    auto workspaces = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, workspaces);
    workspaces[0] = SYS_WORKSPACE_BYTES +
                    static_cast<size_t>(usedCoreNum) * COSINE_EMBEDDING_LOSS_WS_CORE_STRIDE * sizeof(float);
    context->SetBlockDim(usedCoreNum);
    if (reduction != cel::kReductionNone) {
        // The cross-core reduction calls SyncAll, so every launched AIV must be co-resident.
        context->SetScheduleMode(1);
    }
    context->SetTilingKey(0);
    OP_LOGD(context->GetNodeName(),
            "CEL tiling: outputNum=%ld reduceDim=%ld xRank=%u outRank=%u cores=%ld rows/core=%ld tail=%ld red=%u "
            "fast=%u featureTile=%ld",
            n, d, tiling->xBroadcastRank, tiling->outputRank, usedCoreNum, rowsPerCore, tailRows, reduction,
            tiling->fastPath, tiling->featureTile);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingPrepareForCosineEmbeddingLoss(gert::TilingParseContext* context)
{
    (void)context;
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(CosineEmbeddingLoss)
    .Tiling(TilingForCosineEmbeddingLoss)
    .TilingParse<CosineEmbeddingLossCompileInfo>(TilingPrepareForCosineEmbeddingLoss);

} // namespace optiling

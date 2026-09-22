/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "swiglu_backward_group_quant_with_dual_axis_tiling.h"
#include "register/op_def_registry.h"

#include <algorithm>
#include <cstring>
#include <string>
#include <graph/utils/type_utils.h>
#include "error_util.h"
#include "log/log.h"
#include "op_common/op_host/util/platform_util.h"
#include "util/math_util.h"
#include "../../op_kernel/arch35/swiglu_backward_group_quant_with_dual_axis_tiling_key.h"

using namespace ge;
using namespace SwigluBackwardGroupQuantWithDualAxisOp;

namespace optiling {
namespace {
constexpr int64_t INPUT_GRAD_Y = 0;
constexpr int64_t INPUT_X = 1;
constexpr int64_t INPUT_WEIGHT = 2;
constexpr int64_t INPUT_Y_ORIGIN = 3;
constexpr int64_t INPUT_GROUP_INDEX = 4;
constexpr int64_t OUTPUT_Y1 = 0;
constexpr int64_t OUTPUT_SCALE1 = 1;
constexpr int64_t OUTPUT_Y2 = 2;
constexpr int64_t OUTPUT_SCALE2 = 3;
constexpr int64_t OUTPUT_GRAD_WEIGHT = 4;
constexpr int64_t ATTR_CLAMP_LIMIT = 0;
constexpr int64_t ATTR_ALPHA = 1;
constexpr int64_t ATTR_BIAS = 2;
constexpr int64_t ATTR_QUANT_MODE = 3;
constexpr int64_t ATTR_DST_TYPE = 4;
constexpr int64_t QUANT_DUAL_AXIS_MX = 1;
constexpr int64_t FP8_E4M3FN = 36;
constexpr int64_t FP8_E5M2 = 35;
constexpr int64_t MX_BLOCK_SIZE = 32;
constexpr int64_t SCALE_PAIR = 2;
constexpr int64_t TILE_M = 64;
constexpr int64_t TILE_N = 128;
constexpr int64_t DB = 2;
constexpr int64_t GRAD_WEIGHT_MAX_TILE_H = 2048;
constexpr int64_t GRAD_WEIGHT_MAX_TILE_TOKENS = TILE_M;
constexpr int64_t VF_LEN_FP32 = 64;
constexpr int64_t UB_BLOCK_SIZE = 32;

bool IsXType(ge::DataType dtype) { return dtype == ge::DT_FLOAT16 || dtype == ge::DT_BF16; }

bool IsWeightType(ge::DataType dtype) { return IsXType(dtype) || dtype == ge::DT_FLOAT; }
} // namespace

ge::graphStatus SwigluBackwardGroupQuantWithDualAxisMxTiling::GetPlatformInfo()
{
    auto platformInfo = context_->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context_, platformInfo);
    auto platform = platform_ascendc::PlatformAscendC(platformInfo);
    param_.totalCoreNum = platform.GetCoreNumAiv();
    uint64_t ubSize = 0;
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    param_.ubSize = static_cast<int64_t>(ubSize);
    OP_CHECK_IF(param_.totalCoreNum <= 0 || param_.ubSize <= 0,
                OP_LOGE(context_->GetNodeName(),
                        "invalid platform info: actual totalCoreNum=%ld, ubSize=%ld, expected both values > 0",
                        param_.totalCoreNum, param_.ubSize),
                return ge::GRAPH_FAILED);
    auto workspace = context_->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, workspace);
    workspace[0] = platform.GetLibApiWorkSpaceSize();
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SwigluBackwardGroupQuantWithDualAxisMxTiling::ParseAttrs()
{
    auto attrs = context_->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context_, attrs);
    auto clampLimit = attrs->GetAttrPointer<float>(ATTR_CLAMP_LIMIT);
    auto alpha = attrs->GetAttrPointer<float>(ATTR_ALPHA);
    auto bias = attrs->GetAttrPointer<float>(ATTR_BIAS);
    auto quantMode = attrs->GetAttrPointer<int64_t>(ATTR_QUANT_MODE);
    auto dstType = attrs->GetAttrPointer<int64_t>(ATTR_DST_TYPE);
    OP_CHECK_NULL_WITH_CONTEXT(context_, clampLimit);
    OP_CHECK_NULL_WITH_CONTEXT(context_, alpha);
    OP_CHECK_NULL_WITH_CONTEXT(context_, bias);
    OP_CHECK_NULL_WITH_CONTEXT(context_, quantMode);
    OP_CHECK_NULL_WITH_CONTEXT(context_, dstType);

    param_.clampLimit = *clampLimit;
    param_.hasClampLimit = param_.clampLimit > 0.0f;
    param_.alpha = *alpha;
    param_.bias = *bias;
    param_.quantMode = *quantMode;
    param_.dstType = *dstType;
    OP_CHECK_IF(param_.clampLimit != -1.0f && param_.clampLimit <= 0.0f,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->GetNodeName(), "clamp_limit",
                                                      std::to_string(param_.clampLimit).c_str(),
                                                      "clamp_limit must be -1.0 or greater than 0.0"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        param_.alpha <= 0.0f,
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->GetNodeName(), "alpha", std::to_string(param_.alpha).c_str(),
                                              "alpha must be greater than 0.0"),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        param_.quantMode != QUANT_DUAL_AXIS_MX,
        OP_LOGE_FOR_INVALID_VALUE(context_->GetNodeName(), "quant_mode", std::to_string(param_.quantMode).c_str(),
                                  std::to_string(QUANT_DUAL_AXIS_MX).c_str()),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(param_.dstType != FP8_E5M2 && param_.dstType != FP8_E4M3FN,
                OP_LOGE_FOR_INVALID_VALUE(context_->GetNodeName(), "dst_type", std::to_string(param_.dstType).c_str(),
                                          "35 (FLOAT8_E5M2) or 36 (FLOAT8_E4M3FN)"),
                return ge::GRAPH_FAILED);

    auto x = context_->GetInputShape(INPUT_X);
    OP_CHECK_NULL_WITH_CONTEXT(context_, x);
    const auto& shape = x->GetStorageShape();
    const int64_t rank = shape.GetDimNum();
    OP_CHECK_IF(rank != 2,
                OP_LOGE_FOR_INVALID_SHAPEDIM(context_->GetNodeName(), "x", std::to_string(rank).c_str(), "2D"),
                return ge::GRAPH_FAILED);
    param_.totalRows = 1;
    param_.dimBatch = 1;
    for (int64_t i = 0; i < rank - 1; ++i) {
        const int64_t dim = shape.GetDim(i);
        const std::string dimName = "x.shape[" + std::to_string(i) + "]";
        OP_CHECK_IF(
            dim <= 0,
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->GetNodeName(), dimName.c_str(), std::to_string(dim).c_str(),
                                                  "the dimension value must be greater than 0"),
            return ge::GRAPH_FAILED);
        param_.totalRows *= dim;
        if (i < rank - 2) {
            param_.dimBatch *= shape.GetDim(i);
        }
    }
    param_.dimM = shape.GetDim(rank - 2);
    const int64_t width = shape.GetDim(rank - 1);
    OP_CHECK_IF(
        width <= 0 || width % 64 != 0,
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->GetNodeName(), "x.shape[-1]", std::to_string(width).c_str(),
                                              "the last dimension must be a positive multiple of 64"),
        return ge::GRAPH_FAILED);
    param_.dimN = width / SCALE_PAIR;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SwigluBackwardGroupQuantWithDualAxisMxTiling::CheckDtypes()
{
    auto gradY = context_->GetInputDesc(INPUT_GRAD_Y);
    auto x = context_->GetInputDesc(INPUT_X);
    OP_CHECK_NULL_WITH_CONTEXT(context_, gradY);
    OP_CHECK_NULL_WITH_CONTEXT(context_, x);
    const auto xType = x->GetDataType();
    const std::string xGradYTypeMsg = ge::TypeUtils::DataTypeToSerialString(xType) + ", " +
                                      ge::TypeUtils::DataTypeToSerialString(gradY->GetDataType());
    OP_CHECK_IF(!IsXType(xType) || gradY->GetDataType() != xType,
                OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(
                    context_->GetNodeName(), "x and grad_y", xGradYTypeMsg.c_str(),
                    "x and grad_y must both be FLOAT16 or BFLOAT16 and must have the same dtype"),
                return ge::GRAPH_FAILED);

    auto weight = context_->GetOptionalInputDesc(INPUT_WEIGHT);
    auto yOrigin = context_->GetOptionalInputDesc(INPUT_Y_ORIGIN);
    param_.hasWeight = weight != nullptr;
    const std::string optionalInputMsg = std::string(weight == nullptr ? "weight=None" : "weight=provided") + ", " +
                                         (yOrigin == nullptr ? "y_origin=None" : "y_origin=provided");
    OP_CHECK_IF(
        (weight == nullptr) != (yOrigin == nullptr),
        OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(context_->GetNodeName(), "weight and y_origin", optionalInputMsg.c_str(),
                                               "weight and y_origin must be provided together"),
        return ge::GRAPH_FAILED);
    if (weight != nullptr) {
        OP_CHECK_IF(!IsWeightType(weight->GetDataType()),
                    OP_LOGE_FOR_INVALID_DTYPE(context_->GetNodeName(), "weight",
                                              ge::TypeUtils::DataTypeToSerialString(weight->GetDataType()).c_str(),
                                              "FLOAT16, BFLOAT16 or FLOAT32"),
                    return ge::GRAPH_FAILED);
        const std::string yOriginXTypeMsg = ge::TypeUtils::DataTypeToSerialString(yOrigin->GetDataType()) + ", " +
                                            ge::TypeUtils::DataTypeToSerialString(xType);
        OP_CHECK_IF(
            yOrigin->GetDataType() != xType,
            OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(context_->GetNodeName(), "y_origin and x", yOriginXTypeMsg.c_str(),
                                                   "y_origin and x must have the same dtype"),
            return ge::GRAPH_FAILED);
        auto gradWeight = context_->GetOutputDesc(OUTPUT_GRAD_WEIGHT);
        OP_CHECK_NULL_WITH_CONTEXT(context_, gradWeight);
        const std::string gradWeightWeightTypeMsg = ge::TypeUtils::DataTypeToSerialString(gradWeight->GetDataType()) +
                                                    ", " + ge::TypeUtils::DataTypeToSerialString(weight->GetDataType());
        OP_CHECK_IF(gradWeight->GetDataType() != weight->GetDataType(),
                    OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(context_->GetNodeName(), "grad_weight and weight",
                                                           gradWeightWeightTypeMsg.c_str(),
                                                           "grad_weight and weight must have the same dtype"),
                    return ge::GRAPH_FAILED);
        if (weight->GetDataType() == ge::DT_FLOAT16) {
            param_.weightDtype = TPL_WEIGHT_FP16;
        } else if (weight->GetDataType() == ge::DT_BF16) {
            param_.weightDtype = TPL_WEIGHT_BF16;
        } else {
            param_.weightDtype = TPL_WEIGHT_FP32;
        }
    }

    auto groupIndex = context_->GetOptionalInputDesc(INPUT_GROUP_INDEX);
    param_.hasGroupIndex = groupIndex != nullptr;
    const std::string groupWeightMsg = std::string(groupIndex == nullptr ? "group_index=None" :
                                                                           "group_index=provided") +
                                       ", " + (weight == nullptr ? "weight=None" : "weight=provided");
    OP_CHECK_IF(groupIndex == nullptr && weight != nullptr,
                OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(context_->GetNodeName(), "group_index and weight",
                                                       groupWeightMsg.c_str(),
                                                       "group_index must be provided when weight is provided"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        groupIndex != nullptr && groupIndex->GetDataType() != ge::DT_INT64,
        OP_LOGE_FOR_INVALID_DTYPE(context_->GetNodeName(), "group_index",
                                  ge::TypeUtils::DataTypeToSerialString(groupIndex->GetDataType()).c_str(), "INT64"),
        return ge::GRAPH_FAILED);

    auto y1 = context_->GetOutputDesc(OUTPUT_Y1);
    auto scale1 = context_->GetOutputDesc(OUTPUT_SCALE1);
    auto y2 = context_->GetOutputDesc(OUTPUT_Y2);
    auto scale2 = context_->GetOutputDesc(OUTPUT_SCALE2);
    OP_CHECK_NULL_WITH_CONTEXT(context_, y1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, scale1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, y2);
    OP_CHECK_NULL_WITH_CONTEXT(context_, scale2);
    const auto yType = param_.dstType == FP8_E5M2 ? ge::DT_FLOAT8_E5M2 : ge::DT_FLOAT8_E4M3FN;
    const std::string y1Y2TypeMsg = ge::TypeUtils::DataTypeToSerialString(y1->GetDataType()) + ", " +
                                    ge::TypeUtils::DataTypeToSerialString(y2->GetDataType());
    OP_CHECK_IF(y1->GetDataType() != yType || y2->GetDataType() != yType,
                OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(context_->GetNodeName(), "y1 and y2", y1Y2TypeMsg.c_str(),
                                                       "y1 and y2 must both match dst_type"),
                return ge::GRAPH_FAILED);
    const std::string scaleTypeMsg = ge::TypeUtils::DataTypeToSerialString(scale1->GetDataType()) + ", " +
                                     ge::TypeUtils::DataTypeToSerialString(scale2->GetDataType());
    OP_CHECK_IF(
        scale1->GetDataType() != ge::DT_FLOAT8_E8M0 || scale2->GetDataType() != ge::DT_FLOAT8_E8M0,
        OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(context_->GetNodeName(), "scale1 and scale2", scaleTypeMsg.c_str(),
                                               "scale1 and scale2 must both be FLOAT8_E8M0"),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SwigluBackwardGroupQuantWithDualAxisMxTiling::CheckShapes()
{
    auto x = context_->GetInputShape(INPUT_X);
    auto gradY = context_->GetInputShape(INPUT_GRAD_Y);
    OP_CHECK_NULL_WITH_CONTEXT(context_, x);
    OP_CHECK_NULL_WITH_CONTEXT(context_, gradY);
    const auto& xs = x->GetStorageShape();
    const auto& gs = gradY->GetStorageShape();
    const int64_t rank = xs.GetDimNum();
    const std::string gradYXRankMsg = std::to_string(gs.GetDimNum()) + ", " + std::to_string(rank);
    OP_CHECK_IF(
        gs.GetDimNum() != rank,
        OP_LOGE_FOR_INVALID_SHAPEDIMS_WITH_REASON(context_->GetNodeName(), "grad_y and x", gradYXRankMsg.c_str(),
                                                  "grad_y and x must have the same rank"),
        return ge::GRAPH_FAILED);
    for (int64_t i = 0; i < rank - 1; ++i) {
        const std::string gradYXShapeMsg = Ops::Base::ToString(gs) + ", " + Ops::Base::ToString(xs);
        const std::string reason = "grad_y.shape[" + std::to_string(i) + "] must be equal to x.shape[" +
                                   std::to_string(i) + "]";
        OP_CHECK_IF(gs.GetDim(i) != xs.GetDim(i),
                    OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(context_->GetNodeName(), "grad_y and x",
                                                           gradYXShapeMsg.c_str(), reason.c_str()),
                    return ge::GRAPH_FAILED);
    }
    const std::string gradYShapeMsg = Ops::Base::ToString(gs);
    const std::string lastDimReason = "grad_y.shape[-1] must equal " + std::to_string(param_.dimN);
    OP_CHECK_IF(gs.GetDim(rank - 1) != param_.dimN,
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context_->GetNodeName(), "grad_y", gradYShapeMsg.c_str(),
                                                      lastDimReason.c_str()),
                return ge::GRAPH_FAILED);

    auto groupIndex = context_->GetOptionalInputShape(INPUT_GROUP_INDEX);
    param_.numGroups = 1;
    if (groupIndex != nullptr) {
        const auto& shape = groupIndex->GetStorageShape();
        OP_CHECK_IF(shape.GetDimNum() != 1 || shape.GetDim(0) <= 0,
                    OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context_->GetNodeName(), "group_index",
                                                          Ops::Base::ToString(shape).c_str(),
                                                          "group_index must be a non-empty 1D tensor"),
                    return ge::GRAPH_FAILED);
        param_.numGroups = shape.GetDim(0);
    }

    auto weight = context_->GetOptionalInputShape(INPUT_WEIGHT);
    auto yOrigin = context_->GetOptionalInputShape(INPUT_Y_ORIGIN);
    if (weight != nullptr) {
        auto gradWeight = context_->GetOutputShape(OUTPUT_GRAD_WEIGHT);
        OP_CHECK_NULL_WITH_CONTEXT(context_, yOrigin);
        OP_CHECK_NULL_WITH_CONTEXT(context_, gradWeight);
        const int64_t weightSize = weight->GetStorageShape().GetShapeSize();
        OP_CHECK_IF(weightSize != param_.totalRows,
                    OP_LOGE_FOR_INVALID_SHAPESIZE(context_->GetNodeName(), "weight", std::to_string(weightSize).c_str(),
                                                  std::to_string(param_.totalRows).c_str()),
                    return ge::GRAPH_FAILED);
        const std::string yOriginShapeMsg = Ops::Base::ToString(yOrigin->GetStorageShape());
        const std::string yOriginShapeReason = "y_origin shape must be equal to grad_y shape " +
                                               Ops::Base::ToString(gs);
        OP_CHECK_IF(yOrigin->GetStorageShape() != gs,
                    OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context_->GetNodeName(), "y_origin", yOriginShapeMsg.c_str(),
                                                          yOriginShapeReason.c_str()),
                    return ge::GRAPH_FAILED);
        const std::string gradWeightShapeMsg = Ops::Base::ToString(gradWeight->GetStorageShape());
        const std::string gradWeightShapeReason = "grad_weight shape must be equal to weight shape " +
                                                  Ops::Base::ToString(weight->GetStorageShape());
        OP_CHECK_IF(gradWeight->GetStorageShape() != weight->GetStorageShape(),
                    OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context_->GetNodeName(), "grad_weight",
                                                          gradWeightShapeMsg.c_str(), gradWeightShapeReason.c_str()),
                    return ge::GRAPH_FAILED);
    }

    auto y1 = context_->GetOutputShape(OUTPUT_Y1);
    auto scale1 = context_->GetOutputShape(OUTPUT_SCALE1);
    auto y2 = context_->GetOutputShape(OUTPUT_Y2);
    auto scale2 = context_->GetOutputShape(OUTPUT_SCALE2);
    OP_CHECK_NULL_WITH_CONTEXT(context_, y1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, scale1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, y2);
    OP_CHECK_NULL_WITH_CONTEXT(context_, scale2);
    const std::string y1Y2ShapeMsg = Ops::Base::ToString(y1->GetStorageShape()) + ", " +
                                     Ops::Base::ToString(y2->GetStorageShape());
    OP_CHECK_IF(y1->GetStorageShape() != xs || y2->GetStorageShape() != xs,
                OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                    context_->GetNodeName(), "y1 and y2", y1Y2ShapeMsg.c_str(),
                    "y1 and y2 shapes must both be equal to x shape " + Ops::Base::ToString(xs)),
                return ge::GRAPH_FAILED);

    const auto& s1 = scale1->GetStorageShape();
    OP_CHECK_IF(s1.GetDimNum() != rank + 1,
                OP_LOGE_FOR_INVALID_SHAPEDIM(context_->GetNodeName(), "scale1", std::to_string(s1.GetDimNum()).c_str(),
                                             std::to_string(rank + 1).c_str()),
                return ge::GRAPH_FAILED);
    for (int64_t i = 0; i < rank - 1; ++i) {
        const std::string scale1ShapeMsg = Ops::Base::ToString(s1);
        const std::string reason = "scale1.shape[" + std::to_string(i) + "] must be equal to x.shape[" +
                                   std::to_string(i) + "]";
        OP_CHECK_IF(s1.GetDim(i) != xs.GetDim(i),
                    OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context_->GetNodeName(), "scale1", scale1ShapeMsg.c_str(),
                                                          reason.c_str()),
                    return ge::GRAPH_FAILED);
    }
    const int64_t width = param_.dimN * SCALE_PAIR;
    const int64_t expectedScale1Blocks = Ops::Base::CeilDiv(width, MX_BLOCK_SIZE * SCALE_PAIR);
    const std::string scale1TailReason = "scale1 tail shape must be [" + std::to_string(expectedScale1Blocks) + ", " +
                                         std::to_string(SCALE_PAIR) + "]";
    OP_CHECK_IF(s1.GetDim(rank - 1) != expectedScale1Blocks || s1.GetDim(rank) != SCALE_PAIR,
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context_->GetNodeName(), "scale1",
                                                      Ops::Base::ToString(s1).c_str(), scale1TailReason.c_str()),
                return ge::GRAPH_FAILED);

    const auto& s2 = scale2->GetStorageShape();
    if (param_.hasGroupIndex != 0) {
        const int64_t scale2Rows = param_.totalRows / (MX_BLOCK_SIZE * SCALE_PAIR) + param_.numGroups;
        const std::string expectedScale2Shape = "[" + std::to_string(scale2Rows) + ", " + std::to_string(width) + ", " +
                                                std::to_string(SCALE_PAIR) + "]";
        OP_CHECK_IF(
            s2.GetDimNum() != 3 || s2.GetDim(0) != scale2Rows || s2.GetDim(1) != width || s2.GetDim(2) != SCALE_PAIR,
            OP_LOGE_FOR_INVALID_SHAPE(context_->GetNodeName(), "scale2", Ops::Base::ToString(s2).c_str(),
                                      expectedScale2Shape.c_str()),
            return ge::GRAPH_FAILED);
    } else {
        OP_CHECK_IF(
            s2.GetDimNum() != rank + 1,
            OP_LOGE_FOR_INVALID_SHAPEDIM(context_->GetNodeName(), "scale2", std::to_string(s2.GetDimNum()).c_str(),
                                         std::to_string(rank + 1).c_str()),
            return ge::GRAPH_FAILED);
        for (int64_t i = 0; i < rank - 2; ++i) {
            const std::string scale2ShapeMsg = Ops::Base::ToString(s2);
            const std::string reason = "scale2.shape[" + std::to_string(i) + "] must be equal to x.shape[" +
                                       std::to_string(i) + "]";
            OP_CHECK_IF(s2.GetDim(i) != xs.GetDim(i),
                        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context_->GetNodeName(), "scale2", scale2ShapeMsg.c_str(),
                                                              reason.c_str()),
                        return ge::GRAPH_FAILED);
        }
        const int64_t expectedScale2Rows = Ops::Base::CeilDiv(param_.dimM, MX_BLOCK_SIZE * SCALE_PAIR);
        const std::string scale2TailReason = "scale2 tail shape must be [" + std::to_string(expectedScale2Rows) + ", " +
                                             std::to_string(width) + ", " + std::to_string(SCALE_PAIR) + "]";
        OP_CHECK_IF(
            s2.GetDim(rank - 2) != expectedScale2Rows || s2.GetDim(rank - 1) != width || s2.GetDim(rank) != SCALE_PAIR,
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context_->GetNodeName(), "scale2", Ops::Base::ToString(s2).c_str(),
                                                  scale2TailReason.c_str()),
            return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SwigluBackwardGroupQuantWithDualAxisMxTiling::ComputeTiling()
{
    const int64_t nTiles = Ops::Base::CeilDiv(param_.dimN, TILE_N);
    const int64_t rowBlocks = param_.hasGroupIndex != 0 ? Ops::Base::CeilDiv(param_.totalRows, TILE_M) :
                                                          param_.dimBatch * Ops::Base::CeilDiv(param_.dimM, TILE_M);
    const int64_t workBlocks = rowBlocks * nTiles;
    param_.usedCoreNum = std::min(param_.totalCoreNum, std::max<int64_t>(1, workBlocks));
    if (param_.hasGroupIndex != 0) {
        param_.usedCoreNum = param_.totalCoreNum;
    }
    param_.mode = TPL_MODE_ROTATE;

    const int64_t elementBytes = 2;
    const int64_t halfTile = TILE_M * TILE_N;
    const int64_t inputQueueBytes = halfTile * elementBytes * (DB * 2 + DB);
    const int64_t weightQueueBytes = param_.hasWeight == 0 ? 0 : TILE_M * sizeof(float) * DB;
    const int64_t quantScratchBytes = halfTile * SCALE_PAIR * elementBytes;
    const int64_t quantOutputQueueBytes = halfTile * SCALE_PAIR * DB * 2;
    const int64_t scale1Bytes = TILE_M * MX_BLOCK_SIZE;
    const int64_t scale1QueueBytes = scale1Bytes * SCALE_PAIR * DB;
    const int64_t scale2Bytes = TILE_N * SCALE_PAIR * (TILE_M / 64 * 3) + 128;
    const int64_t scale2QueueBytes = scale2Bytes * DB;
    const int64_t scale2ReciprocalBytes = TILE_N * SCALE_PAIR * (TILE_M / 64 * SCALE_PAIR) * elementBytes;
    const int64_t fixedUbBytes = inputQueueBytes + weightQueueBytes + quantOutputQueueBytes + scale1QueueBytes +
                                 scale2QueueBytes + scale1Bytes + scale2ReciprocalBytes;
    int64_t gradXScratchBytes = quantScratchBytes;
    param_.gradWeightTileTokens = 1;
    if (param_.hasWeight != 0) {
        param_.gradWeightTileH = std::min(param_.dimN, halfTile);
        param_.gradWeightTileTokens = std::min(GRAD_WEIGHT_MAX_TILE_TOKENS, halfTile / param_.gradWeightTileH);
        const int64_t gradWeightScratchBytes = param_.gradWeightTileH * sizeof(float) + VF_LEN_FP32 * sizeof(float) +
                                               UB_BLOCK_SIZE;
        gradXScratchBytes = std::max(quantScratchBytes, gradWeightScratchBytes);
    }
    const int64_t needUb = fixedUbBytes + gradXScratchBytes;
    OP_CHECK_IF(needUb > param_.ubSize,
                OP_LOGE(context_->GetNodeName(),
                        "MX base tile exceeds UB: actual need=%ld bytes, available=%ld bytes, expected need <= "
                        "available",
                        needUb, param_.ubSize),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

void SwigluBackwardGroupQuantWithDualAxisMxTiling::SetTilingKey()
{
    const uint64_t tilingKey = GET_TPL_TILING_KEY(
        static_cast<uint32_t>(param_.mode), TPL_MX_MODE, static_cast<uint32_t>(param_.hasGroupIndex),
        static_cast<uint32_t>(param_.hasWeight), static_cast<uint32_t>(param_.weightDtype),
        static_cast<uint32_t>(param_.hasClampLimit));
    context_->SetTilingKey(tilingKey);
    context_->SetBlockDim(param_.usedCoreNum);
}

ge::graphStatus SwigluBackwardGroupQuantWithDualAxisMxTiling::SaveTiling()
{
    SwigluBackwardGroupQuantWithDualAxisMxTilingData tilingData{};
    tilingData.usedCoreNum = param_.usedCoreNum;
    tilingData.totalRows = param_.totalRows;
    tilingData.dimBatch = param_.dimBatch;
    tilingData.dimM = param_.dimM;
    tilingData.dimN = param_.dimN;
    tilingData.numGroups = param_.numGroups;
    tilingData.quantMode = param_.quantMode;
    tilingData.tileM = TILE_M;
    tilingData.tileN = TILE_N;
    tilingData.nTiles = Ops::Base::CeilDiv(param_.dimN, TILE_N);
    tilingData.gradWeightTileH = param_.gradWeightTileH;
    tilingData.gradWeightTileTokens = param_.gradWeightTileTokens;
    tilingData.alpha = param_.alpha;
    tilingData.clampLimit = param_.clampLimit;
    tilingData.bias = param_.bias;
    auto raw = context_->GetRawTilingData();
    OP_CHECK_NULL_WITH_CONTEXT(context_, raw);
    OP_CHECK_IF(sizeof(tilingData) > raw->GetCapacity(),
                OP_LOGE(context_->GetNodeName(),
                        "tiling data exceeds capacity: actual size=%zu bytes, expected capacity >= %zu bytes",
                        sizeof(tilingData), raw->GetCapacity()),
                return ge::GRAPH_FAILED);
    auto ret = memcpy_s(raw->GetData(), raw->GetCapacity(), &tilingData, sizeof(tilingData));
    OP_CHECK_IF(ret != EOK,
                OP_LOGE(context_->GetNodeName(), "failed to save tiling data: actual memcpy_s return=%d, expected=%d",
                        ret, EOK),
                return ge::GRAPH_FAILED);
    raw->SetDataSize(sizeof(tilingData));
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SwigluBackwardGroupQuantWithDualAxisMxTiling::DoTiling()
{
    if (GetPlatformInfo() != ge::GRAPH_SUCCESS || ParseAttrs() != ge::GRAPH_SUCCESS ||
        CheckDtypes() != ge::GRAPH_SUCCESS || CheckShapes() != ge::GRAPH_SUCCESS ||
        ComputeTiling() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    SetTilingKey();
    return SaveTiling();
}

ge::graphStatus TilingForSwigluBackwardGroupQuantWithDualAxisMx(gert::TilingContext* context)
{
    OP_CHECK_IF(
        context == nullptr,
        OP_LOGE("SwigluBackwardGroupQuantWithDualAxisMx", "invalid tiling context: actual=nullptr, expected=non-null"),
        return ge::GRAPH_FAILED);
    SwigluBackwardGroupQuantWithDualAxisMxTiling tiling(context);
    return tiling.DoTiling();
}

static ge::graphStatus Tiling4SwigluBackwardGroupQuantWithDualAxis(gert::TilingContext* context)
{
    return TilingForSwigluBackwardGroupQuantWithDualAxisMx(context);
}

static ge::graphStatus TilingPrepare4SwigluBackwardGroupQuantWithDualAxis(gert::TilingParseContext* context)
{
    (void)context;
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(SwigluBackwardGroupQuantWithDualAxis)
    .Tiling(Tiling4SwigluBackwardGroupQuantWithDualAxis)
    .TilingParse<CoreCompileInfo>(TilingPrepare4SwigluBackwardGroupQuantWithDualAxis);

} // namespace optiling

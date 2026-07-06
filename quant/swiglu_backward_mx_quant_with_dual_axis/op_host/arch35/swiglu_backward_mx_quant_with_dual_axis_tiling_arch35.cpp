/* *
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file swiglu_backward_mx_quant_with_dual_axis_tiling_arch35.cpp
 * \brief Tiling implementation for SwigluBackwardMxQuantWithDualAxis
 */

#include "swiglu_backward_mx_quant_with_dual_axis_tiling_arch35.h"
#include "op_common/op_host/util/platform_util.h"
#include "error_util.h"
#include "log/log.h"
#include "util/math_util.h"
#include "quant/swiglu_backward_mx_quant_with_dual_axis/op_kernel/arch35/swiglu_backward_mx_quant_with_dual_axis_tiling_key.h"

using namespace std;
using namespace ge;
using namespace AscendC;
using namespace SwigluBackwardMxQuantWithDualAxisOp;

namespace optiling {
// Attribute index constants (must match _def.cpp Attr declaration order)
constexpr int64_t INDEX_ATTR_DST_DTYPE = 3;
constexpr int64_t INDEX_ATTR_SCALE_ALG = 2;
constexpr int64_t INDEX_ATTR_ROUND_MODE = 1;
constexpr int64_t INDEX_ATTR_ACTIVATE_LEFT = 0;

// Input/Output index constants
constexpr int64_t INDEX_INPUT_X = 0;
constexpr int64_t INDEX_INPUT_Y_GRAD = 1;
constexpr int64_t INDEX_INPUT_GROUP_INDEX = 2;
constexpr int64_t INDEX_OUTPUT_Y1 = 0;
constexpr int64_t INDEX_OUTPUT_MX_SCALE1 = 1;
constexpr int64_t INDEX_OUTPUT_Y2 = 2;
constexpr int64_t INDEX_OUTPUT_MX_SCALE2 = 3;

// Numeric constants
constexpr int64_t DIGIT_TWO = 2;
constexpr int64_t DIGIT_THREE = 3;
constexpr int64_t DB = 2; // 开db
constexpr int64_t DIGIT_FOUR = 4;
constexpr int64_t DIGIT_256 = 256;
constexpr int64_t BLOCK_SIZE = 32;
constexpr int64_t SPLIT_BLOCK_H = 64;

const std::set<ge::DataType> INPUT_SUPPORT_DTYPE_SET = {ge::DT_FLOAT16, ge::DT_BF16};
const std::set<ge::DataType> Y_SUPPORT_DTYPE_SET = {ge::DT_FLOAT8_E4M3FN};
const std::set<ge::DataType> Y_SUPPORT_DTYPE_FP4_SET = {ge::DT_FLOAT4_E2M1, ge::DT_FLOAT4_E1M2};
const std::set<ge::DataType> Y_SUPPORT_DTYPE_FP8_SET = {ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E5M2};
const std::set<ge::DataType> OUTPUT_SCALE_DTYPE_SET = {ge::DT_FLOAT8_E8M0};
const std::set<ge::DataType> GROUP_INDEX_DTYPE_SET = {ge::DT_INT64};

RoundModeList SwigluBackwardMxQuantWithDualAxisTiling::GetRoundMode(const std::string& roundMode)
{
    if (roundMode == "rint") {
        return RoundModeList::MODE_RINT;
    } else if (roundMode == "round") {
        return RoundModeList::MODE_ROUND;
    } else if (roundMode == "floor") {
        return RoundModeList::MODE_FLOOR;
    }
    return RoundModeList::MODE_UNDEFINED;
}

ge::graphStatus SwigluBackwardMxQuantWithDualAxisTiling::GetAttr()
{
    auto* attrs = context_->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context_, attrs);
    // activate_left: Bool attribute at position0
    auto* attrActivateLeft = attrs->GetAttrPointer<bool>(INDEX_ATTR_ACTIVATE_LEFT);
    OP_CHECK_NULL_WITH_CONTEXT(context_, attrActivateLeft);
    tilingParams_.activateLeft = *attrActivateLeft ? 1 : 0;
    // round_mode: String attribute at position 1
    auto* attrRoundMode = attrs->GetAttrPointer<char>(INDEX_ATTR_ROUND_MODE);
    OP_CHECK_NULL_WITH_CONTEXT(context_, attrRoundMode);
    std::string roundModeStr = attrRoundMode;
    RoundModeList roundMode = GetRoundMode(roundModeStr);
    OP_CHECK_IF(
        roundMode == RoundModeList::MODE_UNDEFINED,
        OP_LOGE_WITH_INVALID_ATTR(context_->GetNodeName(), "round_mode", roundModeStr.c_str(), "[rint, floor, round]"),
        return ge::GRAPH_FAILED);

    // FP8 round_mode validation: FP8 only supports rint on Ascend950
    OP_CHECK_IF((Y_SUPPORT_DTYPE_FP8_SET.count(tilingParams_.yDtype) != 0 && roundMode != RoundModeList::MODE_RINT),
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                    context_->GetNodeName(), "round_mode", roundModeStr.c_str(),
                    "When output y's data type is FLOAT8_E4M3FN or FLOAT8_E5M2, round_mode only support rint"),
                return ge::GRAPH_FAILED);

    tilingParams_.roundMode = static_cast<int64_t>(roundMode);
    // scale_alg: Int attribute at position 2
    auto* attrScaleAlg = attrs->GetAttrPointer<int64_t>(INDEX_ATTR_SCALE_ALG);
    OP_CHECK_NULL_WITH_CONTEXT(context_, attrScaleAlg);
    tilingParams_.scaleAlg = static_cast<int64_t>(*attrScaleAlg);
    OP_CHECK_IF(
        tilingParams_.scaleAlg != 1,
        OP_LOGE_WITH_INVALID_ATTR(context_->GetNodeName(), "scale_alg", std::to_string(tilingParams_.scaleAlg).c_str(),
                                  "{1}, only cuBLAS mode is supported"),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF(tilingParams_.activateLeft != 1,
                OP_LOGE_WITH_INVALID_ATTR(context_->GetNodeName(), "activate_left",
                                          std::to_string(tilingParams_.activateLeft).c_str(),
                                          "{1}, activate_left only support 1"),
                return ge::GRAPH_FAILED);

    // dst_dtype: Int attribute at position 3
    auto* attrDstType = attrs->GetAttrPointer<int64_t>(INDEX_ATTR_DST_DTYPE);
    OP_CHECK_NULL_WITH_CONTEXT(context_, attrDstType);
    tilingParams_.dstType = static_cast<int64_t>(*attrDstType);
    std::string reasonMsg = "y1_out dtype is " + std::to_string(static_cast<int32_t>(tilingParams_.yDtype)) +
                            ", dst_dtype is " + std::to_string(tilingParams_.dstType) +
                            ", these two must be identical, dst_dtype represents the output dtype.";
    OP_CHECK_IF((static_cast<int64_t>(tilingParams_.yDtype) != tilingParams_.dstType),
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->GetNodeName(), "dst_dtype",
                                                      std::to_string(tilingParams_.dstType).c_str(), reasonMsg.c_str()),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SwigluBackwardMxQuantWithDualAxisTiling::CheckDtype()
{
    auto inputXPtr = context_->GetInputDesc(INDEX_INPUT_X);
    OP_CHECK_NULL_WITH_CONTEXT(context_, inputXPtr);
    auto xDtype = inputXPtr->GetDataType();
    OP_CHECK_IF(INPUT_SUPPORT_DTYPE_SET.count(xDtype) == 0,
                OP_LOGE_FOR_INVALID_DTYPE(context_->GetNodeName(), "x",
                                          ge::TypeUtils::DataTypeToSerialString(xDtype).c_str(), "float16, bfloat16"),
                return ge::GRAPH_FAILED);

    auto inputYGradPtr = context_->GetInputDesc(INDEX_INPUT_Y_GRAD);
    OP_CHECK_NULL_WITH_CONTEXT(context_, inputYGradPtr);
    auto yGradDtype = inputYGradPtr->GetDataType();
    OP_CHECK_IF(INPUT_SUPPORT_DTYPE_SET.count(yGradDtype) == 0 || yGradDtype != xDtype,
                OP_LOGE_FOR_INVALID_DTYPE(context_->GetNodeName(), "y_grad",
                                          ge::TypeUtils::DataTypeToSerialString(yGradDtype).c_str(),
                                          "y_grad dtype must be float16 or bfloat16, and same as x"),
                return ge::GRAPH_FAILED);

    auto groupIndexDtype = ge::DT_INT64;
    auto groupIndexPtr = context_->GetOptionalInputDesc(INDEX_INPUT_GROUP_INDEX);
    if (groupIndexPtr != nullptr) {
        groupIndexDtype = groupIndexPtr->GetDataType();
        tilingParams_.isGroupIdx = 1;
        OP_CHECK_IF(GROUP_INDEX_DTYPE_SET.count(groupIndexDtype) == 0,
                    OP_LOGE_FOR_INVALID_DTYPE(context_->GetNodeName(), "group_index",
                                              ge::TypeUtils::DataTypeToSerialString(groupIndexDtype).c_str(), "int64"),
                    return ge::GRAPH_FAILED);
    }

    auto outputY1Ptr = context_->GetOutputDesc(INDEX_OUTPUT_Y1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, outputY1Ptr);
    auto y1Dtype = outputY1Ptr->GetDataType();
    auto outputY2Ptr = context_->GetOutputDesc(INDEX_OUTPUT_Y2);
    OP_CHECK_NULL_WITH_CONTEXT(context_, outputY2Ptr);
    auto y2Dtype = outputY2Ptr->GetDataType();
    std::string yDtypeMsg = ge::TypeUtils::DataTypeToSerialString(y1Dtype) + ", " +
                            ge::TypeUtils::DataTypeToSerialString(y2Dtype);
    OP_CHECK_IF(Y_SUPPORT_DTYPE_SET.count(y1Dtype) == 0 || y2Dtype != y1Dtype,
                OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(
                    context_->GetNodeName(), "y1_out and y2_out", yDtypeMsg.c_str(),
                    "y1_out dtype must be FLOAT8_E4M3FN, and y2_out dtype must be same as y1_out dtype"),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(CheckOutputScaleDtype() != ge::GRAPH_SUCCESS,
                OP_LOGE(context_->GetNodeName(), "CheckOutputScaleDtype failed"), return ge::GRAPH_FAILED);
    tilingParams_.yDtype = y1Dtype;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SwigluBackwardMxQuantWithDualAxisTiling::CheckOutputScaleDtype()
{
    auto outputScale1Ptr = context_->GetOutputDesc(INDEX_OUTPUT_MX_SCALE1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, outputScale1Ptr);
    auto scale1Dtype = outputScale1Ptr->GetDataType();
    auto outputScale2Ptr = context_->GetOutputDesc(INDEX_OUTPUT_MX_SCALE2);
    OP_CHECK_NULL_WITH_CONTEXT(context_, outputScale2Ptr);
    auto scale2Dtype = outputScale2Ptr->GetDataType();
    std::string scaleDtypeMsg = ge::TypeUtils::DataTypeToSerialString(scale1Dtype) + ", " +
                                ge::TypeUtils::DataTypeToSerialString(scale2Dtype);
    OP_CHECK_IF(OUTPUT_SCALE_DTYPE_SET.count(scale1Dtype) == 0 || OUTPUT_SCALE_DTYPE_SET.count(scale2Dtype) == 0,
                OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(context_->GetNodeName(), "mxscale1_out and mxscale2_out",
                                                       scaleDtypeMsg.c_str(),
                                                       "mxscale1_out and mxscale2_out dtype must be FLOAT8_E8M0"),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SwigluBackwardMxQuantWithDualAxisTiling::CheckScaleShape(const gert::Shape& mxScale1Shape,
                                                                         const gert::Shape& mxScale2Shape,
                                                                         const gert::Shape& xShape)
{
    int64_t xDimNum = xShape.GetDimNum();
    int64_t expectScaleDimNum = xDimNum + 1;
    std::string dimMsg = std::to_string(mxScale1Shape.GetDimNum()) + " and" + std::to_string(mxScale2Shape.GetDimNum());
    OP_CHECK_IF(mxScale1Shape.GetDimNum() != expectScaleDimNum || mxScale2Shape.GetDimNum() != expectScaleDimNum,
                OP_LOGE_FOR_INVALID_SHAPEDIMS_WITH_REASON(
                    context_->GetNodeName(), "mxscale1_out and mxscale2_out", dimMsg.c_str(),
                    ("mxscale1_out and mxscale2_out shape dims must be " + std::to_string(expectScaleDimNum)).c_str()),
                return ge::GRAPH_FAILED);

    int64_t scaleLastIdx = expectScaleDimNum - 1;
    std::string dimMsgScale = Ops::Base::ToString(mxScale1Shape) + " and " + Ops::Base::ToString(mxScale2Shape);
    OP_CHECK_IF((mxScale1Shape.GetDim(scaleLastIdx) != DIGIT_TWO) || (mxScale2Shape.GetDim(scaleLastIdx) != DIGIT_TWO),
                OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(context_->GetNodeName(), "mxscale1_out and mxscale2_out",
                                                       dimMsgScale.c_str(),
                                                       "mxscale1_out and mxscale2_out shape last dim must be 2"),
                return ge::GRAPH_FAILED);

    int64_t dimGradX = tilingParams_.dimN * DIGIT_TWO;
    int64_t batchDimCount = xDimNum - 2;
    OP_CHECK_IF(CheckMxScale1Shape(mxScale1Shape, xShape, dimGradX, batchDimCount) != ge::GRAPH_SUCCESS,
                OP_LOGE(context_->GetNodeName(), "CheckMxScale1Shape failed"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        CheckMxScale2Shape(mxScale2Shape, xShape, dimGradX, batchDimCount, expectScaleDimNum) != ge::GRAPH_SUCCESS,
        OP_LOGE(context_->GetNodeName(), "CheckMxScale2Shape failed"), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SwigluBackwardMxQuantWithDualAxisTiling::CheckMxScale1Shape(const gert::Shape& mxScale1Shape,
                                                                            const gert::Shape& xShape, int64_t dimGradX,
                                                                            int64_t batchDimCount)
{
    if (batchDimCount > 0) {
        for (int64_t i = 0; i < batchDimCount; i++) {
            std::string reasonMsgBatch = "x shape is " + Ops::Base::ToString(xShape) + ", the " + std::to_string(i) +
                                         "th dimension of mxscale1_out must match x";
            OP_CHECK_IF(mxScale1Shape.GetDim(i) != xShape.GetDim(i),
                        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context_->GetNodeName(), "mxscale1_out",
                                                              Ops::Base::ToString(mxScale1Shape).c_str(),
                                                              reasonMsgBatch.c_str()),
                        return ge::GRAPH_FAILED);
        }
    }
    int64_t mxScale1MDimIdx = batchDimCount;
    std::string reasonMsgM = "x shape is " + Ops::Base::ToString(xShape) +
                             ", dimM = " + std::to_string(tilingParams_.dimM) +
                             ", the m-dimension of mxscale1_out shape must equal dimM";
    OP_CHECK_IF(mxScale1Shape.GetDim(mxScale1MDimIdx) != tilingParams_.dimM,
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context_->GetNodeName(), "mxscale1_out",
                                                      Ops::Base::ToString(mxScale1Shape).c_str(), reasonMsgM.c_str()),
                return ge::GRAPH_FAILED);
    int64_t mxScaleShapeDim1 = Ops::Base::CeilDiv(dimGradX, SPLIT_BLOCK_H);
    int64_t mxScale1CeilDimIdx = batchDimCount + 1;
    std::string reasonMsgDim1 = "x shape is " + Ops::Base::ToString(xShape) +
                                ", grad_x column count = " + std::to_string(dimGradX) +
                                ", the ceil-dimension of mxscale1_out shape must be equal to Ceil(grad_x_col / 64)";
    OP_CHECK_IF(
        mxScale1Shape.GetDim(mxScale1CeilDimIdx) != mxScaleShapeDim1,
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context_->GetNodeName(), "mxscale1_out",
                                              Ops::Base::ToString(mxScale1Shape).c_str(), reasonMsgDim1.c_str()),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SwigluBackwardMxQuantWithDualAxisTiling::CheckMxScale2Shape(const gert::Shape& mxScale2Shape,
                                                                            const gert::Shape& xShape, int64_t dimGradX,
                                                                            int64_t batchDimCount,
                                                                            int64_t expectScaleDimNum)
{
    if (batchDimCount > 0) {
        for (int64_t i = 0; i < batchDimCount; i++) {
            std::string reasonMsgBatch = "x shape is " + Ops::Base::ToString(xShape) + ", the " + std::to_string(i) +
                                         "th dimension of mxscale2_out must match x";
            OP_CHECK_IF(mxScale2Shape.GetDim(i) != xShape.GetDim(i),
                        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context_->GetNodeName(), "mxscale2_out",
                                                              Ops::Base::ToString(mxScale2Shape).c_str(),
                                                              reasonMsgBatch.c_str()),
                        return ge::GRAPH_FAILED);
        }
    }
    int64_t mxScale2NDimIdx = expectScaleDimNum - 2;
    std::string reasonScale2MsgDim1 = "x shape is " + Ops::Base::ToString(xShape) +
                                      ", grad_x column count = " + std::to_string(dimGradX) + ", mxscale2_out shape[" +
                                      std::to_string(mxScale2NDimIdx) + "] must be equal to grad_x column count";
    OP_CHECK_IF(
        mxScale2Shape.GetDim(mxScale2NDimIdx) != dimGradX,
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context_->GetNodeName(), "mxscale2_out",
                                              Ops::Base::ToString(mxScale2Shape).c_str(), reasonScale2MsgDim1.c_str()),
        return ge::GRAPH_FAILED);
    int64_t mxScale2CeilDimIdx = batchDimCount;
    int64_t mxScaleShapeDim2 = tilingParams_.isGroupIdx == 1 ?
                                   tilingParams_.dimM / SPLIT_BLOCK_H + tilingParams_.numGroups :
                                   Ops::Base::CeilDiv(tilingParams_.dimM, SPLIT_BLOCK_H);
    std::string reasonScale2MsgDim0 = "x shape is " + Ops::Base::ToString(xShape) + " isGroupIdx is " +
                                      std::to_string(tilingParams_.isGroupIdx) + " numGroups is " +
                                      std::to_string(tilingParams_.numGroups) + " dimM is " +
                                      std::to_string(tilingParams_.dimM) + ", mxscale2_out shape ceil-dim must be " +
                                      (tilingParams_.isGroupIdx == 1 ? "FloorDiv(dimM, 64) + numGroups" :
                                                                       "CeilDiv(dimM, 64)");
    OP_CHECK_IF(
        mxScale2Shape.GetDim(mxScale2CeilDimIdx) != mxScaleShapeDim2,
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context_->GetNodeName(), "mxscale2_out",
                                              Ops::Base::ToString(mxScale2Shape).c_str(), reasonScale2MsgDim0.c_str()),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SwigluBackwardMxQuantWithDualAxisTiling::CheckYShape(const gert::Shape& xShape,
                                                                     const gert::Shape& y1Shape,
                                                                     const gert::Shape& y2Shape)
{
    int64_t xDimNum = xShape.GetDimNum();
    int64_t y1DimNum = y1Shape.GetDimNum();

    // x and y1 must have same number of dimensions
    OP_CHECK_IF(xDimNum != y1DimNum,
                OP_LOGE_FOR_INVALID_SHAPEDIM(context_->GetNodeName(), "x and y1_out",
                                             (std::to_string(xDimNum) + " vs " + std::to_string(y1DimNum)).c_str(),
                                             "same dim num"),
                return ge::GRAPH_FAILED);

    // Last dim must be even (for SwiGLU split)
    int64_t lastDim = xShape.GetDim(xDimNum - 1);
    OP_CHECK_IF(lastDim % SPLIT_BLOCK_H != 0,
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context_->GetNodeName(), "x", Ops::Base::ToString(xShape).c_str(),
                                                      "input x's last dim (2N) must be a multiple of 64"),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF((Y_SUPPORT_DTYPE_FP4_SET.count(tilingParams_.yDtype) != 0 && lastDim % DIGIT_FOUR != 0),
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                    context_->GetNodeName(), "x", Ops::Base::ToString(xShape).c_str(),
                    "When output y's data type is FLOAT4_E2M1 or FLOAT4_E1M2, lastDim must be divisible by 4"),
                return ge::GRAPH_FAILED);

    // Compute batch = product of all dimensions except last two
    int64_t batch = 1;
    for (int64_t i = 0; i < xDimNum - 2; i++) {
        batch *= xShape.GetDim(i);
    }
    tilingParams_.dimBatch = batch;
    tilingParams_.dimM = xShape.GetDim(xDimNum - 2);
    tilingParams_.dimN = lastDim / DIGIT_TWO; // N = x last dim / 2
    int64_t dimGradX = lastDim;               // grad_x column count = 2N

    // Check all batch dimensions match between x and y1_out
    for (int64_t i = 0; i < xDimNum - 2; i++) {
        std::string reasonBatch = "x shape is " + Ops::Base::ToString(xShape) + ", y1_out shape is " +
                                  Ops::Base::ToString(y1Shape) + ", the " + std::to_string(i) +
                                  "th dimension must match";
        OP_CHECK_IF(xShape.GetDim(i) != y1Shape.GetDim(i),
                    OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context_->GetNodeName(), "y1_out",
                                                          Ops::Base::ToString(y1Shape).c_str(), reasonBatch.c_str()),
                    return ge::GRAPH_FAILED);
    }

    // Check m dimension: y1_out's second-to-last dim must equal x's second-to-last dim (dimM)
    std::string reasonMsgM = "x shape is " + Ops::Base::ToString(xShape) + ", y1_out shape is " +
                             Ops::Base::ToString(y1Shape) + ", the m-dimension (second-to-last) of y1_out must match x";
    OP_CHECK_IF(y1Shape.GetDim(y1DimNum - 2) != tilingParams_.dimM,
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context_->GetNodeName(), "y1_out",
                                                      Ops::Base::ToString(y1Shape).c_str(), reasonMsgM.c_str()),
                return ge::GRAPH_FAILED);

    // Check last dimension: y1_out's last dim must equal grad_x column count (2N)
    std::string reasonMsg = "x shape is " + Ops::Base::ToString(xShape) +
                            ", grad_x column count = x last dim = " + std::to_string(dimGradX) +
                            ". The last dimension of y1_out's shape must equal grad_x column count.";
    OP_CHECK_IF(y1Shape.GetDim(y1DimNum - 1) != dimGradX,
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context_->GetNodeName(), "y1_out",
                                                      Ops::Base::ToString(y1Shape).c_str(), reasonMsg.c_str()),
                return ge::GRAPH_FAILED);
    std::string dimMsgScale = Ops::Base::ToString(y1Shape) + " and " + Ops::Base::ToString(y2Shape);
    OP_CHECK_IF(
        y1Shape != y2Shape,
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(context_->GetNodeName(), "y1_out and y2_out", dimMsgScale.c_str(),
                                               "y1_out shape must be same as y2_out shape"),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(CheckGroupIndexShape() != ge::GRAPH_SUCCESS,
                OP_LOGE(context_->GetNodeName(), "CheckGroupIndexShape failed"), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SwigluBackwardMxQuantWithDualAxisTiling::CheckGroupIndexShape()
{
    auto groupIndexShapePtr = context_->GetOptionalInputShape(INDEX_INPUT_GROUP_INDEX);
    tilingParams_.numGroups = 1;
    if (groupIndexShapePtr != nullptr) {
        auto groupIndexShape = groupIndexShapePtr->GetStorageShape();
        OP_CHECK_IF(groupIndexShape.GetDimNum() != 1,
                    OP_LOGE_FOR_INVALID_SHAPEDIM(context_->GetNodeName(), "group_index",
                                                 std::to_string(groupIndexShape.GetDimNum()).c_str(), "1D"),
                    return ge::GRAPH_FAILED);
        tilingParams_.numGroups = groupIndexShape.GetDim(0);
    }
    OP_CHECK_IF(tilingParams_.numGroups <= 0,
                OP_LOGE_FOR_INVALID_SHAPESIZE(context_->GetNodeName(), "group_index",
                                              std::to_string(tilingParams_.numGroups).c_str(), " > 0"),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SwigluBackwardMxQuantWithDualAxisTiling::CheckShape()
{
    auto xShapePtr = context_->GetInputShape(INDEX_INPUT_X);
    OP_CHECK_NULL_WITH_CONTEXT(context_, xShapePtr);
    auto y1ShapePtr = context_->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, y1ShapePtr);
    auto xShape = xShapePtr->GetStorageShape();
    auto y1Shape = y1ShapePtr->GetStorageShape();
    auto y2ShapePtr = context_->GetOutputShape(INDEX_OUTPUT_Y2);
    OP_CHECK_NULL_WITH_CONTEXT(context_, y2ShapePtr);
    auto y2Shape = y2ShapePtr->GetStorageShape();
    int64_t xSize = xShape.GetShapeSize();
    int64_t y1Size = y1Shape.GetShapeSize();
    std::string sizeMsg = std::to_string(xSize) + " and " + std::to_string(y1Size);
    OP_CHECK_IF(xSize <= 0 || y1Size <= 0,
                OP_LOGE_FOR_INVALID_SHAPESIZES_WITH_REASON(context_->GetNodeName(), "x and y1_out", sizeMsg.c_str(),
                                                           "x and y1_out shape size must > 0"),
                return ge::GRAPH_FAILED);
    auto mxScale1ShapePtr = context_->GetOutputShape(INDEX_OUTPUT_MX_SCALE1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, mxScale1ShapePtr);
    auto mxScale2ShapePtr = context_->GetOutputShape(INDEX_OUTPUT_MX_SCALE2);
    OP_CHECK_NULL_WITH_CONTEXT(context_, mxScale2ShapePtr);
    auto mxScale1Shape = mxScale1ShapePtr->GetStorageShape();
    auto mxScale2Shape = mxScale2ShapePtr->GetStorageShape();
    int64_t mxScale1Size = mxScale1Shape.GetShapeSize();
    int64_t mxScale2Size = mxScale2Shape.GetShapeSize();
    std::string sizeMxscaleMsg = std::to_string(mxScale1Size) + " and " + std::to_string(mxScale2Size);
    OP_CHECK_IF(mxScale1Size <= 0 || mxScale2Size <= 0,
                OP_LOGE_FOR_INVALID_SHAPESIZES_WITH_REASON(context_->GetNodeName(), "mxscale1_out and mxscale2_out",
                                                           sizeMxscaleMsg.c_str(),
                                                           "mxscale1_out and mxscale2_out shape size must > 0"),
                return ge::GRAPH_FAILED);

    // Input x must be 2D to 6D: [batch..., m, 2n]
    int64_t xDimNum = xShape.GetDimNum();
    int64_t y1DimNum = y1Shape.GetDimNum();
    std::string shapeDimMsg = "x shape dim is " + std::to_string(xDimNum) + " y1_out shape dim is " +
                              std::to_string(y1DimNum);
    OP_CHECK_IF(xDimNum < DIGIT_TWO || xDimNum > 6 || y1DimNum != xDimNum,
                OP_LOGE_FOR_INVALID_SHAPEDIM(context_->GetNodeName(), "x and y1_out", shapeDimMsg.c_str(),
                                             "2D to 6D, same dim"),
                return ge::GRAPH_FAILED);

    // x and y1_out must have matching shape
    std::string shapeMsg = "x shape is " + Ops::Base::ToString(xShape) + " y1_out shape is " +
                           Ops::Base::ToString(y1Shape);
    OP_CHECK_IF(xShape != y1Shape,
                OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(context_->GetNodeName(), "x and y1_out", shapeMsg.c_str(),
                                                       "x shape and y1_out shape must be identical."),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(CheckYShape(xShape, y1Shape, y2Shape) != ge::GRAPH_SUCCESS,
                OP_LOGE(context_->GetNodeName(), "CheckYShape failed"), return ge::GRAPH_FAILED);

    OP_CHECK_IF(CheckYGradShape(xShape, xDimNum) != ge::GRAPH_SUCCESS,
                OP_LOGE(context_->GetNodeName(), "CheckYGradShape failed"), return ge::GRAPH_FAILED);

    OP_CHECK_IF(CheckScaleShape(mxScale1Shape, mxScale2Shape, xShape) != ge::GRAPH_SUCCESS,
                OP_LOGE(context_->GetNodeName(), "CheckScaleShape failed"), return ge::GRAPH_FAILED);
    tilingParams_.blockW = DIGIT_256 / DIGIT_TWO; // 128: A/B/yGrad half-width per tile
    tilingParams_.splitBlockH = SPLIT_BLOCK_H;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SwigluBackwardMxQuantWithDualAxisTiling::GetPlatformInfo()
{
    OP_LOGI(context_->GetNodeName(), "Enter GetPlatformInfo.");

    auto platformInfo = context_->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context_, platformInfo);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    tilingParams_.totalCoreNum = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF((tilingParams_.totalCoreNum <= 0), OP_LOGE(context_->GetNodeName(), "Failed to get core num."),
                return ge::GRAPH_FAILED);

    uint64_t ubSize;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    tilingParams_.ubSize = static_cast<int64_t>(ubSize);
    OP_CHECK_IF((tilingParams_.ubSize <= 0), OP_LOGE(context_->GetNodeName(), "Failed to get ub size."),
                return ge::GRAPH_FAILED);

    tilingParams_.workspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SwigluBackwardMxQuantWithDualAxisTiling::ComputeTilingParams()
{
    OP_LOGI(context_->GetNodeName(), "Enter ComputeTilingParams.");
    // Core distribution:
    // nSplitNum = CeilDiv(N, 128)
    tilingParams_.dimNBlockNum = Ops::Base::CeilDiv(tilingParams_.dimN, tilingParams_.blockW);

    int64_t nSplitNum = tilingParams_.dimNBlockNum;
    tilingParams_.usedCoreNum = tilingParams_.totalCoreNum;
    if (tilingParams_.isGroupIdx == 0) {
        int64_t countM = Ops::Base::CeilDiv(tilingParams_.dimM, tilingParams_.splitBlockH);
        // In batch mode, total blocks = batch * countM * nSplitNum
        int64_t allCount = countM * nSplitNum * tilingParams_.dimBatch;
        tilingParams_.usedCoreNum = allCount < tilingParams_.totalCoreNum ? allCount : tilingParams_.totalCoreNum;
    }
    tilingParams_.dimNTail = tilingParams_.dimN % tilingParams_.blockW == 0 ? tilingParams_.blockW :
                                                                              tilingParams_.dimN % tilingParams_.blockW;

    int64_t inHalfSize = tilingParams_.blockW * tilingParams_.splitBlockH;
    int64_t gradXTileSize = inHalfSize * DIGIT_TWO;
    int64_t xUb = inHalfSize * tilingParams_.dtypeSize * DIGIT_TWO * DB;
    int64_t yGradUb = inHalfSize * tilingParams_.dtypeSize * DB;
    int64_t gradXUb = gradXTileSize * tilingParams_.dtypeSize;
    int64_t y1Ub = gradXTileSize * DB;
    int64_t y2Ub = y1Ub;
    int64_t scale1Ub = tilingParams_.splitBlockH * BLOCK_SIZE * DB * DIGIT_TWO;
    int64_t scale2Ub = (tilingParams_.blockW * DIGIT_TWO) *
                       ((tilingParams_.splitBlockH / SPLIT_BLOCK_H) * DIGIT_THREE) * DB;
    int64_t tmpScale1Ub = tilingParams_.splitBlockH * BLOCK_SIZE;
    int64_t tmpScale2Ub = gradXTileSize / BLOCK_SIZE * DIGIT_TWO * tilingParams_.dtypeSize;
    int64_t allNeedUb = xUb + yGradUb + gradXUb + y1Ub + y2Ub + scale1Ub + scale2Ub + tmpScale1Ub + tmpScale2Ub;
    OP_CHECK_IF(
        (allNeedUb > tilingParams_.ubSize),
        OP_LOGE(context_->GetNodeName(), "The basic split block (64, 128) cannot fit, allNeedUb is %ld, ubSize is %ld",
                allNeedUb, tilingParams_.ubSize),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

void SwigluBackwardMxQuantWithDualAxisTiling::SetTilingKeyAndCore()
{
    if (tilingParams_.roundMode == static_cast<int64_t>(RoundModeList::MODE_ROUND)) {
        roundMode_ = TPL_ROUND;
    } else if (tilingParams_.roundMode == static_cast<int64_t>(RoundModeList::MODE_FLOOR)) {
        roundMode_ = TPL_FLOOR;
    } else if (tilingParams_.roundMode == static_cast<int64_t>(RoundModeList::MODE_RINT)) {
        roundMode_ = TPL_RINT;
    }
    scaleAlg_ = tilingParams_.scaleAlg;
    // mode determined by SplitCore: ROTATE(0) if dimNBlockNum < allCore, BLOCK(1) otherwise
    mode_ = (tilingParams_.dimNBlockNum < tilingParams_.totalCoreNum) ? TPL_MODE_ROTATE : TPL_MODE_BLOCK;
    if (tilingParams_.isGroupIdx == 0) {
        mode_ = TPL_MODE_BLOCK;
    }
    isGroupIdx_ = (tilingParams_.isGroupIdx == 0) ? TPL_NO_GROUP_INDEX : TPL_GROUP_INDEX;
    int64_t tilingKey = GET_TPL_TILING_KEY(mode_, roundMode_, scaleAlg_, isGroupIdx_);
    OP_LOGI(context_->GetNodeName(), "mode=%ld, roundMode=%ld, scaleAlg=%ld, isGroupIdx_=%ld, tilingKey=%ld", mode_,
            roundMode_, scaleAlg_, isGroupIdx_, tilingKey);
    context_->SetTilingKey(tilingKey);
    OP_LOGI(context_->GetNodeName(), "Tiling usedCoreNum is %ld.", tilingParams_.usedCoreNum);
    context_->SetBlockDim(tilingParams_.usedCoreNum);
}

ge::graphStatus SwigluBackwardMxQuantWithDualAxisTiling::SetTilingData()
{
    // Get tiling data pointer
    tilingData_ = context_->GetTilingData<SwigluBackwardMxQuantWithDualAxisTilingData>();
    OP_CHECK_IF(tilingData_ == nullptr, OP_LOGE(context_->GetNodeName(), "get tilingdata ptr failed"),
                return ge::GRAPH_FAILED);

    // Clear tiling data
    OP_CHECK_IF((memset_s(tilingData_, sizeof(SwigluBackwardMxQuantWithDualAxisTilingData), 0,
                          sizeof(SwigluBackwardMxQuantWithDualAxisTilingData)) != EOK),
                OP_LOGE(context_->GetNodeName(), "memset tilingData failed"), return ge::GRAPH_FAILED);
    tilingData_->usedCoreNum = tilingParams_.usedCoreNum;
    tilingData_->dstType = tilingParams_.dstType;
    tilingData_->activateLeft = tilingParams_.activateLeft;
    tilingData_->dimM = tilingParams_.dimM;
    tilingData_->dimN = tilingParams_.dimN;
    tilingData_->numGroups = tilingParams_.numGroups;
    tilingData_->blockW = tilingParams_.blockW;
    tilingData_->splitBlockH = tilingParams_.splitBlockH;
    tilingData_->dimNBlockNum = tilingParams_.dimNBlockNum;
    tilingData_->dimNTail = tilingParams_.dimNTail;
    tilingData_->yGradRowStride = tilingParams_.dimN;
    tilingData_->dimGradX = tilingParams_.dimN * DIGIT_TWO;
    tilingData_->dimBatch = tilingParams_.dimBatch;
    return ge::GRAPH_SUCCESS;
}

void SwigluBackwardMxQuantWithDualAxisTiling::PrintTilingData()
{
    OP_LOGI(context_->GetNodeName(),
            "TilingData usedCoreNum: %ld, dstType: %ld, "
            "activateLeft: %ld, blockW: %ld, splitBlockH: %ld, "
            "dimM: %ld, dimN: %ld, numGroups: %ld, dimBatch: %ld, "
            "dimNBlockNum: %ld, dimNTail: %ld",
            tilingData_->usedCoreNum, tilingData_->dstType, tilingData_->activateLeft, tilingData_->blockW,
            tilingData_->splitBlockH, tilingData_->dimM, tilingData_->dimN, tilingData_->numGroups,
            tilingData_->dimBatch, tilingData_->dimNBlockNum, tilingData_->dimNTail);
}

ge::graphStatus SwigluBackwardMxQuantWithDualAxisTiling::DoTiling()
{
    OP_LOGI(context_->GetNodeName(), "Enter SwigluBackwardMxQuantWithDualAxisTiling DoTiling.");
    tilingParams_.isGroupIdx = 0;
    OP_CHECK_IF(GetPlatformInfo() != ge::GRAPH_SUCCESS, OP_LOGE(context_->GetNodeName(), "GetPlatformInfo failed."),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(CheckDtype() != ge::GRAPH_SUCCESS, OP_LOGE(context_->GetNodeName(), "CheckDtype failed."),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(GetAttr() != ge::GRAPH_SUCCESS, OP_LOGE(context_->GetNodeName(), "GetAttr failed."),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(CheckShape() != ge::GRAPH_SUCCESS, OP_LOGE(context_->GetNodeName(), "CheckShape failed."),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(ComputeTilingParams() != ge::GRAPH_SUCCESS,
                OP_LOGE(context_->GetNodeName(), "ComputeTilingParams failed."), return ge::GRAPH_FAILED);

    SetTilingKeyAndCore();
    OP_CHECK_IF(SetTilingData() != ge::GRAPH_SUCCESS, OP_LOGE(context_->GetNodeName(), "SetTilingData failed."),
                return ge::GRAPH_FAILED);
    PrintTilingData();
    size_t* workspaces = context_->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, workspaces);
    workspaces[0] = tilingParams_.workspaceSize;
    OP_LOGI(context_->GetNodeName(), "DoTiling done.");
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SwigluBackwardMxQuantWithDualAxisTiling::CheckYGradShape(const gert::Shape& xShape, int64_t xDimNum)
{
    auto yGradShapePtr = context_->GetInputShape(INDEX_INPUT_Y_GRAD);
    OP_CHECK_NULL_WITH_CONTEXT(context_, yGradShapePtr);
    auto yGradShape = yGradShapePtr->GetStorageShape();
    int64_t yGradDimNum = yGradShape.GetDimNum();
    OP_CHECK_IF(yGradDimNum != xDimNum,
                OP_LOGE_FOR_INVALID_SHAPEDIM(context_->GetNodeName(), "y_grad", std::to_string(yGradDimNum).c_str(),
                                             (std::to_string(xDimNum) + "D").c_str()),
                return ge::GRAPH_FAILED);

    std::string yGradShapeMsg = "x shape is " + Ops::Base::ToString(xShape) + " y_grad shape is " +
                                Ops::Base::ToString(yGradShape);
    for (int64_t i = 0; i < xDimNum - 2; i++) {
        OP_CHECK_IF(yGradShape.GetDim(i) != xShape.GetDim(i),
                    OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                        context_->GetNodeName(), "y_grad", yGradShapeMsg.c_str(),
                        ("The " + std::to_string(i) + "th dimension of y_grad must match x").c_str()),
                    return ge::GRAPH_FAILED);
    }
    OP_CHECK_IF(yGradShape.GetDim(yGradDimNum - 2) != xShape.GetDim(xDimNum - 2),
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context_->GetNodeName(), "y_grad", yGradShapeMsg.c_str(),
                                                      "The m-dimension (second-to-last) of y_grad must match x"),
                return ge::GRAPH_FAILED);

    std::string yGradDim1Msg = "x shape is " + Ops::Base::ToString(xShape) + " y_grad shape is " +
                               Ops::Base::ToString(yGradShape) +
                               ", dimN = x last dim / 2 = " + std::to_string(tilingParams_.dimN);
    OP_CHECK_IF(yGradShape.GetDim(yGradDimNum - 1) != tilingParams_.dimN,
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                    context_->GetNodeName(), "y_grad", yGradDim1Msg.c_str(),
                    "The last dimension of y_grad shape must be equal to half of x shape's last dim (dimN)."),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingForSwigluBackwardMxQuantWithDualAxis(gert::TilingContext* context)
{
    OP_LOGI("SwigluBackwardMxQuantWithDualAxisTiling", "Enter TilingForSwigluBackwardMxQuantWithDualAxis");

    OP_CHECK_IF(context == nullptr, OP_LOGE("SwigluBackwardMxQuantWithDualAxisTiling", "Tiling context is null."),
                return ge::GRAPH_FAILED);

    SwigluBackwardMxQuantWithDualAxisTiling tiling(context);
    return tiling.DoTiling();
}

ge::graphStatus TilingPrepareForSwigluBackwardMxQuantWithDualAxis(gert::TilingParseContext* context)
{
    OP_CHECK_IF(context == nullptr, OP_LOGE("SwigluBackwardMxQuantWithDualAxisTiling", "TilingParse context is null."),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(SwigluBackwardMxQuantWithDualAxis)
    .Tiling(TilingForSwigluBackwardMxQuantWithDualAxis)
    .TilingParse<SwigluBackwardMxQuantWithDualAxisCompileInfo>(TilingPrepareForSwigluBackwardMxQuantWithDualAxis);
} // namespace optiling

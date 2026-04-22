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
 * \file swiglu_mx_quant_with_dual_axis_tiling_arch35.cpp
 * \brief Tiling implementation for SwigluMxQuantWithDualAxis
 */

#include "swiglu_mx_quant_with_dual_axis_tiling_arch35.h"
#include "op_common/op_host/util/platform_util.h"
#include "error_util.h"
#include "util/math_util.h"
#include "quant/swiglu_mx_quant_with_dual_axis/op_kernel/arch35/swiglu_mx_quant_with_dual_axis_tiling_key.h"

using namespace std;
using namespace ge;
using namespace AscendC;
using namespace SwigluMxQuantWithDualAxisOp;

namespace optiling {
// Attribute index constants (must match _def.cpp Attr declaration order)
constexpr int64_t INDEX_ATTR_DST_TYPE = 3;
constexpr int64_t INDEX_ATTR_SCALE_ALG = 2;
constexpr int64_t INDEX_ATTR_ROUND_MODE = 1;
constexpr int64_t INDEX_ATTR_ACTIVATE_LEFT = 0;

// Input/Output index constants
constexpr int64_t INDEX_INPUT_X = 0;
constexpr int64_t INDEX_INPUT_GROUP_INDEX = 1;
constexpr int64_t INDEX_OUTPUT_Y1 = 0;
constexpr int64_t INDEX_OUTPUT_MX_SCALE1 = 1;
constexpr int64_t INDEX_OUTPUT_Y2 = 2;
constexpr int64_t INDEX_OUTPUT_MX_SCALE2 = 3;

// Numeric constants
constexpr int64_t DIGIT_TWO = 2;
constexpr int64_t DIGIT_HTREE = 3;
constexpr int64_t DB = 2; // 开db
constexpr int64_t DIGIT_FOUR = 4;
constexpr int64_t DIGIT_256 = 256;
constexpr int64_t BLOCK_SIZE = 32;
constexpr int64_t SPLIT_BLOCK_H = 64;

const std::set<ge::DataType> INPUT_SUPPORT_DTYPE_SET = { ge::DT_FLOAT16, ge::DT_BF16 };
const std::set<ge::DataType> Y_SUPPORT_DTYPE_SET = { ge::DT_FLOAT4_E2M1, ge::DT_FLOAT4_E1M2, ge::DT_FLOAT8_E4M3FN,
    ge::DT_FLOAT8_E5M2 };
const std::set<ge::DataType> Y_SUPPORT_DTYPE_FP4_SET = { ge::DT_FLOAT4_E2M1, ge::DT_FLOAT4_E1M2 };
const std::set<ge::DataType> Y_SUPPORT_DTYPE_FP8_SET = { ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E5M2 };
const std::set<ge::DataType> OUTPUT_SCALE_DTYPE_SET = { ge::DT_FLOAT8_E8M0 };
const std::set<ge::DataType> GROUP_INDEX_DTYPE_SET = { ge::DT_INT64 };

RoundModeList SwigluMxQuantWithDualAxisTiling::GetRoundMode(const std::string &roundMode)
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

ge::graphStatus SwigluMxQuantWithDualAxisTiling::GetAttr()
{
    auto *attrs = context_->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context_, attrs);
    // activate_left: Bool attribute at position0
    auto *attrActivateLeft = attrs->GetAttrPointer<bool>(INDEX_ATTR_ACTIVATE_LEFT);
    OP_CHECK_NULL_WITH_CONTEXT(context_, attrActivateLeft);
    tilingParams_.activateLeft = *attrActivateLeft ? 1 : 0;
    // round_mode: String attribute at position 1
    auto *attrRoundMode = attrs->GetAttrPointer<char>(INDEX_ATTR_ROUND_MODE);
    OP_CHECK_NULL_WITH_CONTEXT(context_, attrRoundMode);
    std::string roundModeStr = attrRoundMode;
    RoundModeList roundMode = GetRoundMode(roundModeStr);
    OP_CHECK_IF(roundMode == RoundModeList::MODE_UNDEFINED,
        OP_LOGE_WITH_INVALID_ATTR(context_->GetNodeName(), "round_mode", roundModeStr.c_str(), "[rint, floor, round]"),
        return ge::GRAPH_FAILED);

    // FP8 round_mode validation: FP8 only supports rint on Ascend950
    OP_CHECK_IF((Y_SUPPORT_DTYPE_FP8_SET.count(tilingParams_.yDtype) != 0 && roundMode != RoundModeList::MODE_RINT),
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->GetNodeName(), "round_mode", roundModeStr.c_str(),
        "When output y's data type is FLOAT8_E4M3FN or FLOAT8_E5M2, round_mode only support rint"),
        return ge::GRAPH_FAILED);

    tilingParams_.roundMode = static_cast<int64_t>(roundMode);
    // scale_alg: Int attribute at position 2
    auto *attrScaleAlg = attrs->GetAttrPointer<int64_t>(INDEX_ATTR_SCALE_ALG);
    OP_CHECK_NULL_WITH_CONTEXT(context_, attrScaleAlg);
    tilingParams_.scaleAlg = static_cast<int64_t>(*attrScaleAlg);
    OP_CHECK_IF(tilingParams_.scaleAlg != 0 && tilingParams_.scaleAlg != 1,
        OP_LOGE_WITH_INVALID_ATTR(context_->GetNodeName(), "scale_alg", std::to_string(tilingParams_.scaleAlg).c_str(),
        "[0, 1]"),
        return ge::GRAPH_FAILED);
    // FP4 output only supports scaleAlg=0 (OCP); scaleAlg=1 (CLUB) requires FP8 output
    OP_CHECK_IF((Y_SUPPORT_DTYPE_FP4_SET.count(tilingParams_.yDtype) != 0 && tilingParams_.scaleAlg != 0),
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->GetNodeName(), "scale_alg",
        std::to_string(tilingParams_.scaleAlg).c_str(),
        "When output y's data type is FLOAT4_E2M1 or FLOAT4_E1M2, scaleAlg must be 0"),
        return ge::GRAPH_FAILED);

    // dst_type: Int attribute at position 3
    auto *attrDstType = attrs->GetAttrPointer<int64_t>(INDEX_ATTR_DST_TYPE);
    OP_CHECK_NULL_WITH_CONTEXT(context_, attrDstType);
    tilingParams_.dstType = static_cast<int64_t>(*attrDstType);
    std::string reasonMsg = "y1 dtype is " + std::to_string(static_cast<int32_t>(tilingParams_.yDtype)) +
        ", dst_type is " + std::to_string(tilingParams_.dstType) +
        ", hese two must be identical, dst_type represents the output dtype.";
    OP_CHECK_IF((static_cast<int64_t>(tilingParams_.yDtype) != tilingParams_.dstType),
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->GetNodeName(), "dst_type",
        std::to_string(tilingParams_.dstType).c_str(), reasonMsg.c_str()),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SwigluMxQuantWithDualAxisTiling::CheckDtype()
{
    auto inputXPtr = context_->GetInputDesc(INDEX_INPUT_X);
    OP_CHECK_NULL_WITH_CONTEXT(context_, inputXPtr);
    auto xDtype = inputXPtr->GetDataType();
    OP_CHECK_IF(INPUT_SUPPORT_DTYPE_SET.count(xDtype) == 0,
        OP_LOGE_FOR_INVALID_DTYPE(context_->GetNodeName(), "x", ge::TypeUtils::DataTypeToSerialString(xDtype).c_str(),
        "float16, bfloat16"),
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
    std::string yDtypeMsg =
        ge::TypeUtils::DataTypeToSerialString(y1Dtype) + ", " + ge::TypeUtils::DataTypeToSerialString(y2Dtype);
    OP_CHECK_IF(Y_SUPPORT_DTYPE_SET.count(y1Dtype) == 0 || y2Dtype != y1Dtype,
        OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(context_->GetNodeName(), "y1 and y2",
        yDtypeMsg.c_str(),
        "y1 dtype must in [FLOAT4_E2M1, FLOAT4_E1M2, FLOAT8_E4M3FN, FLOAT8_E5M2], and y2 dtype must be same as y1 dtype"),
        return ge::GRAPH_FAILED);
    

    auto outputScale1Ptr = context_->GetOutputDesc(INDEX_OUTPUT_MX_SCALE1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, outputScale1Ptr);
    auto scale1Dtype = outputScale1Ptr->GetDataType();
    auto outputScale2Ptr = context_->GetOutputDesc(INDEX_OUTPUT_MX_SCALE2);
    OP_CHECK_NULL_WITH_CONTEXT(context_, outputScale2Ptr);
    auto scale2Dtype = outputScale2Ptr->GetDataType();
    std::string scaleDtypeMsg =
        ge::TypeUtils::DataTypeToSerialString(scale1Dtype) + ", " + ge::TypeUtils::DataTypeToSerialString(scale2Dtype);
    OP_CHECK_IF(OUTPUT_SCALE_DTYPE_SET.count(scale1Dtype) == 0 || OUTPUT_SCALE_DTYPE_SET.count(scale2Dtype) == 0,
        OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(context_->GetNodeName(), "mx_scale1 and mx_scale2",
        scaleDtypeMsg.c_str(), "mx_scale1 and mx_scal2 dtype must be FLOAT8_E8M0"),
        return ge::GRAPH_FAILED);
    tilingParams_.yDtype = y1Dtype;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SwigluMxQuantWithDualAxisTiling::CheckScaleShape(const gert::Shape &mxScale1Shape,
    const gert::Shape &mxScale2Shape, const gert::Shape &xShape)
{
    // 开始校验mxScale1和mxScale2的shape是否正确
    std::string dimMsg = std::to_string(mxScale1Shape.GetDimNum()) + " and " + std::to_string(mxScale2Shape.GetDimNum());
    OP_CHECK_IF(mxScale1Shape.GetDimNum() != DIGIT_HTREE || mxScale2Shape.GetDimNum() != DIGIT_HTREE,
        OP_LOGE_FOR_INVALID_SHAPEDIMS_WITH_REASON(context_->GetNodeName(), "mx_scale1 and mx_scale2", dimMsg.c_str(),
        "mx_scale1 and mx_scale2 shape dims must be three"),
        return ge::GRAPH_FAILED);
    std::string dimMsgScale =
        Ops::Base::ToString(mxScale1Shape) + " and " + Ops::Base::ToString(mxScale2Shape);
    OP_CHECK_IF((mxScale1Shape.GetDim(DIGIT_TWO) != DIGIT_TWO) || (mxScale2Shape.GetDim(DIGIT_TWO) != DIGIT_TWO),
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(context_->GetNodeName(), "mx_scale1 and mx_scale2", dimMsgScale.c_str(),
        "mx_scale1 and mx_scale2 shape last dim must be 2"),
        return ge::GRAPH_FAILED);
    std::string reasonMsg = "x shape is " + Ops::Base::ToString(xShape) +
        ", the 0th dimension of mx_scale1 shape must be the same as the 0th dimension of x shape.";
    OP_CHECK_IF(mxScale1Shape.GetDim(0) != tilingParams_.dimM,
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context_->GetNodeName(), "mx_scale1",
        Ops::Base::ToString(mxScale1Shape).c_str(), reasonMsg.c_str()),
        return ge::GRAPH_FAILED);
    int64_t mxScaleShapeDim1 = Ops::Base::CeilDiv(tilingParams_.dimN, SPLIT_BLOCK_H);
    std::string reasonMsgDim1 = "x shape is " + Ops::Base::ToString(xShape) +
        ", the value of the 1st dimension of mx_scale1 shape must be equal to Ceil(x_shape[1] / 2 / 64)";
    OP_CHECK_IF(mxScale1Shape.GetDim(1) != mxScaleShapeDim1,
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context_->GetNodeName(), "mx_scale1",
        Ops::Base::ToString(mxScale1Shape).c_str(), reasonMsgDim1.c_str()),
        return ge::GRAPH_FAILED);

    std::string reasonScale2MsgDim1 =
        "x shape is " + Ops::Base::ToString(xShape) + ", mx_scale2 shape[1] must be equal to x_shape[1] / 2";
    OP_CHECK_IF(mxScale2Shape.GetDim(1) != tilingParams_.dimN,
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context_->GetNodeName(), "mx_cale2",
        Ops::Base::ToString(mxScale2Shape).c_str(), reasonScale2MsgDim1.c_str()),
        return ge::GRAPH_FAILED);
    int64_t mxScaleShapeDim2 = tilingParams_.isGroupIdx == 1 ?
        tilingParams_.dimM / SPLIT_BLOCK_H + tilingParams_.numGroups :
        Ops::Base::CeilDiv(tilingParams_.dimM, SPLIT_BLOCK_H);
    std::string reasonScale2MsgDim0 = "x shape is " + Ops::Base::ToString(xShape) + " isGroupIdx is " +
        std::to_string(tilingParams_.isGroupIdx) + " numGroups is " +
        std::to_string(tilingParams_.numGroups) +
        " M represents the 0th dimension of x shape, numGroups represents the 0th dimension of group_index shape, "
        "isGroupIdx indicates whether input group_index exists, when group_index exists, dim0 = CeilDiv(M, 64) + "
        "numGroups, when not exists, dim0 = FloorDiv(M, 64)";
    OP_CHECK_IF(mxScale2Shape.GetDim(0) != mxScaleShapeDim2,
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context_->GetNodeName(), "mx_scale2",
        Ops::Base::ToString(mxScale2Shape).c_str(), reasonScale2MsgDim0.c_str()),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SwigluMxQuantWithDualAxisTiling::CheckYShape(const gert::Shape &xShape, const gert::Shape &y1Shape,
    const gert::Shape &y2Shape)
{
    // Last dim must be even (for SwiGLU split)
    int64_t lastDim = xShape.GetDim(1);
    OP_CHECK_IF(lastDim % DIGIT_TWO != 0,
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context_->GetNodeName(), "x", Ops::Base::ToString(xShape).c_str(),
        "input x's last dim must be a multiple of two."),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF((Y_SUPPORT_DTYPE_FP4_SET.count(tilingParams_.yDtype) != 0 && lastDim % DIGIT_FOUR != 0),
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context_->GetNodeName(), "x", Ops::Base::ToString(xShape).c_str(),
        "When output y's data type is FLOAT4_E2M1 or FLOAT4_E1M2, lastDim must be divisible by 4"),
        return ge::GRAPH_FAILED);
    tilingParams_.dimM = xShape.GetDim(0);
    tilingParams_.dimN = lastDim / DIGIT_TWO; // SwiGLU halves the last dim
    std::string reasonMsg = "x shape is " + Ops::Base::ToString(xShape) +
        ".The first dimension of y1's shape must be equal to half of the first dimension of x's shape.";
    OP_CHECK_IF(
        y1Shape.GetDim(1) != tilingParams_.dimN,
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context_->GetNodeName(), "y1", Ops::Base::ToString(y1Shape).c_str(),
        reasonMsg.c_str()),
        return ge::GRAPH_FAILED);
    std::string dimMsgScale = Ops::Base::ToString(y1Shape) + " and " + Ops::Base::ToString(y2Shape);
    OP_CHECK_IF(y1Shape != y2Shape,
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(context_->GetNodeName(), "y1 and y2", dimMsgScale.c_str(),
        "y1 shape must be same as y2 shape"),
        return ge::GRAPH_FAILED);


    // Check group_index is 1D
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

ge::graphStatus SwigluMxQuantWithDualAxisTiling::CheckShape()
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
        OP_LOGE_FOR_INVALID_SHAPESIZES_WITH_REASON(context_->GetNodeName(), "x and y1", sizeMsg.c_str(),
        "x and y1 shape size must > 0"),
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
        OP_LOGE_FOR_INVALID_SHAPESIZES_WITH_REASON(context_->GetNodeName(), "mx_scale1 and mx_scale2",
        sizeMxscaleMsg.c_str(), "mx_scale1 and mx_scale2 shape size must > 0"),
        return ge::GRAPH_FAILED);
    // Input x must be 2D [M, 2N]
    std::string shapeDimMsg = "x shape dim is " + std::to_string(xShape.GetDimNum()) + " y1 shape dim is " +
        std::to_string(y1Shape.GetDimNum());
    OP_CHECK_IF(xShape.GetDimNum() != DIGIT_TWO || y1Shape.GetDimNum() != DIGIT_TWO,
        OP_LOGE_FOR_INVALID_SHAPEDIM(context_->GetNodeName(), "x and y1", shapeDimMsg.c_str(), "2D"), return ge::GRAPH_FAILED);
    std::string shapeMsg =
        "x shape is " + Ops::Base::ToString(xShape) + " y1 shape is " + Ops::Base::ToString(y1Shape);
    OP_CHECK_IF(xShape.GetDim(0) != y1Shape.GetDim(0),
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context_->GetNodeName(), "x and y1", shapeMsg.c_str(),
        "The 0th dimension of x shape and y1 shape must be the same."),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(CheckYShape(xShape, y1Shape, y2Shape) != ge::GRAPH_SUCCESS,
        OP_LOGE(context_->GetNodeName(), "CheckYShape failed"), return ge::GRAPH_FAILED);

    OP_CHECK_IF(CheckScaleShape(mxScale1Shape, mxScale2Shape, xShape) != ge::GRAPH_SUCCESS,
        OP_LOGE(context_->GetNodeName(), "CheckScaleShape failed"), return ge::GRAPH_FAILED);
    tilingParams_.blockW = DIGIT_256;
    tilingParams_.splitBlockH = SPLIT_BLOCK_H;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SwigluMxQuantWithDualAxisTiling::GetPlatformInfo()
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

ge::graphStatus SwigluMxQuantWithDualAxisTiling::ComputeTilingParams()
{
    OP_LOGI(context_->GetNodeName(), "Enter ComputeTilingParams.");
    // Core distribution:
    // nSplitNum = CeilDiv(N, 256)
    tilingParams_.dimNBlockNum = Ops::Base::CeilDiv(tilingParams_.dimN, tilingParams_.blockW);

    int64_t nSplitNum = tilingParams_.dimNBlockNum;
    tilingParams_.usedCoreNum = tilingParams_.totalCoreNum;
    if (tilingParams_.isGroupIdx == 0) {
        int64_t countM = Ops::Base::CeilDiv(tilingParams_.dimM, tilingParams_.splitBlockH);
        int64_t allCount = countM * nSplitNum;
        tilingParams_.usedCoreNum = allCount < tilingParams_.totalCoreNum ? allCount : tilingParams_.totalCoreNum;
    }
    tilingParams_.dimNTail = tilingParams_.dimN % tilingParams_.blockW == 0 ? tilingParams_.blockW :
                                                                              tilingParams_.dimN % tilingParams_.blockW;
    // 检查下64， 256基本块是否可以放的下，如果放不下就报错，此处是做ub越界的保护
    int64_t inHalfSize = tilingParams_.blockW * tilingParams_.splitBlockH;
    int64_t swigluUb = inHalfSize * tilingParams_.dtypeSize;
    int64_t xUb = swigluUb * DIGIT_TWO * DB;
    int64_t y1Ub = inHalfSize * DB;
    int64_t y2Ub = y1Ub;
    int64_t scale1Ub = tilingParams_.splitBlockH * BLOCK_SIZE * DB;
    int64_t scale2Ub = tilingParams_.blockW * DIGIT_TWO * DB;
    int64_t tmpScale1Ub = tilingParams_.splitBlockH * BLOCK_SIZE;
    int64_t tmpScale2Ub = tilingParams_.blockW * DIGIT_TWO * tilingParams_.dtypeSize;
    int64_t allNeedUb = xUb + swigluUb + y1Ub + y2Ub + scale1Ub + scale2Ub + tmpScale1Ub + tmpScale2Ub;
    OP_CHECK_IF((allNeedUb > tilingParams_.ubSize),
        OP_LOGE(context_->GetNodeName(), "The basic split block (64, 256) cannot fit, allNeedUb is %ld, ubSize is %ld",
        allNeedUb, tilingParams_.ubSize),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

void SwigluMxQuantWithDualAxisTiling::SetTilingKeyAndCore()
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

ge::graphStatus SwigluMxQuantWithDualAxisTiling::SetTilingData()
{
    // Get tiling data pointer
    tilingData_ = context_->GetTilingData<SwigluMxQuantWithDualAxisTilingData>();
    OP_CHECK_IF(tilingData_ == nullptr, OP_LOGE(context_->GetNodeName(), "get tilingdata ptr failed"),
        return ge::GRAPH_FAILED);

    // Clear tiling data
    OP_CHECK_IF((memset_s(tilingData_, sizeof(SwigluMxQuantWithDualAxisTilingData), 0,
        sizeof(SwigluMxQuantWithDualAxisTilingData)) != EOK),
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
    return ge::GRAPH_SUCCESS;
}

void SwigluMxQuantWithDualAxisTiling::PrintTilingData()
{
    OP_LOGI(context_->GetNodeName(),
        "TilingData usedCoreNum: %ld, dstType: %ld, "
        "activateLeft: %ld, blockW: %ld, splitBlockH: %ld, "
        "dimM: %ld, dimN: %ld, numGroups: %ld, "
        "dimNBlockNum: %ld, dimNTail: %ld",
        tilingData_->usedCoreNum, tilingData_->dstType, tilingData_->activateLeft, tilingData_->blockW,
        tilingData_->splitBlockH, tilingData_->dimM, tilingData_->dimN, tilingData_->numGroups,
        tilingData_->dimNBlockNum, tilingData_->dimNTail);
}

ge::graphStatus SwigluMxQuantWithDualAxisTiling::DoTiling()
{
    OP_LOGI(context_->GetNodeName(), "Enter SwigluMxQuantWithDualAxisTiling DoTiling.");
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
    size_t *workspaces = context_->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, workspaces);
    workspaces[0] = tilingParams_.workspaceSize;
    OP_LOGI(context_->GetNodeName(), "DoTiling done.");
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingForSwigluMxQuantWithDualAxis(gert::TilingContext *context)
{
    OP_LOGI("SwigluMxQuantWithDualAxisTiling", "Enter TilingForSwigluMxQuantWithDualAxis");

    OP_CHECK_IF(context == nullptr, OP_LOGE("SwigluMxQuantWithDualAxisTiling", "Tiling context is null."),
        return ge::GRAPH_FAILED);

    SwigluMxQuantWithDualAxisTiling tiling(context);
    return tiling.DoTiling();
}

ge::graphStatus TilingPrepareForSwigluMxQuantWithDualAxis(gert::TilingParseContext *context)
{
    OP_CHECK_IF(context == nullptr, OP_LOGE("SwigluMxQuantWithDualAxisTiling", "TilingParse context is null."),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(SwigluMxQuantWithDualAxis)
    .Tiling(TilingForSwigluMxQuantWithDualAxis)
    .TilingParse<SwigluMxQuantWithDualAxisCompileInfo>(TilingPrepareForSwigluMxQuantWithDualAxis);
} // namespace optiling

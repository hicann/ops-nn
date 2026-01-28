/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file dynamic_mx_quant_tiling.cpp
 * \brief
 */
#include "dynamic_mx_quant_tiling_arch35.h"
#include <cmath>
#include "platform/platform_info.h"

using namespace std;
using namespace ge;
using namespace AscendC;

namespace optiling {
constexpr int64_t INDEX_ATTR_AXIS = 0;
constexpr int64_t INDEX_ATTR_ROUND_MODE = 1;
constexpr int64_t INDEX_ATTR_DST_DTYPE = 2;
constexpr int64_t INDEX_ATTR_BLOCK_SIZE = 3;
constexpr int64_t INDEX_ATTR_SCALE_ALG = 4;
constexpr int64_t BYTES_OF_INPUT_TYPE = 2;
constexpr int64_t MAX_BYTES_OF_OUTPUT_TYPE = 1;
constexpr int64_t DIGIT_TEN_THOUSAND = 10000;
constexpr int64_t DIGIT_THOUSAND = 1000;
constexpr int64_t DIGIT_HUNDRED = 100;
constexpr int64_t DIGIT_TEN = 10;
constexpr int64_t N_ALIGN32 = 32;
constexpr int64_t N_ALIGN64 = 64;
constexpr int64_t N_ALIGN128 = 128;
constexpr int64_t BLOCK_PER_GROUP = 2;
constexpr int64_t UINT16_BYTES_SIZE = 2;
constexpr int64_t UINT8_BYTES_SIZE = 1;
constexpr int64_t DIGIT_TWO = 2;
constexpr int64_t DIGIT_FOUR = 4;
constexpr int64_t TILING_KEY_INPUT = 1000;
constexpr int64_t TILING_KEY_OUTPUT = 100;
constexpr int64_t TILING_KEY_TMPLATE = 10;
constexpr int64_t TILING_KEY_AXIS_SCALE = 1;
constexpr int64_t N_BUFFER = 2;
constexpr int64_t EXIST_NODE_NUM = 3;
constexpr int64_t ATTR_BLOCK_SIZE = 32;
constexpr int64_t AXIS_NUM_AFTER_MERGE = 3;
constexpr int64_t NEW_SHAPE_INDEX_TWO = 2;
constexpr int64_t WORKSPACE_SIZE = 32;
constexpr int64_t WORKSPACE_ALIGN_SIZE = 512;
constexpr int64_t OPTIMISE_MAX_BLOCK_SIZE = 320;
constexpr int64_t DYNAMIC_MX_QUANT_MAX_BLOCK_SIZE = 1024;
constexpr int64_t RESERVED_UB_SIZE = 2 * 1024; // 预留空间
const std::set<ge::DataType> INPUT_SUPPORT_DTYPE_SET = {ge::DT_FLOAT16, ge::DT_BF16};
const std::set<ge::DataType> INPUT_FP16_DTYPE_SET = {ge::DT_FLOAT16};
// FLOATX_EYMZ: Total bit of one float is X, in which first 1 bit for sign, following Y bit for exponent and Z bit for
// mantissa.
const std::set<ge::DataType> Y_SUPPORT_DTYPE_SET = {
    ge::DT_FLOAT4_E2M1, ge::DT_FLOAT4_E1M2, ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E5M2};
const std::set<ge::DataType> Y_SUPPORT_DTYPE_FP4_SET = {ge::DT_FLOAT4_E2M1, ge::DT_FLOAT4_E1M2};
const std::set<ge::DataType> Y_SUPPORT_DTYPE_FP8_SET = {ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E5M2};
const std::set<ge::DataType> OUTPUT_SUPPORT_DTYPE_SET = {ge::DT_FLOAT8_E8M0};
constexpr int64_t DIM1_BLOCK_COUNT = 8;
constexpr int64_t BLOCK_SIZE = 32;
constexpr int64_t TAIL_TILING_KEY_DIGIT = 4;
constexpr int64_t SINGLE_LOOP_MIN_COLS = 128;
constexpr size_t MAX_DIM_NUM = 7;

template <typename T>
static inline uint64_t GetRemainder(uint64_t num, T div)
{
    return div == 0 ? div : num % div;
}

template <typename T>
std::string Shape2String(const T& shape)
{
    std::ostringstream oss;
    oss << "[";
    if (shape.GetDimNum() > 0) {
        for (size_t i = 0; i < shape.GetDimNum() - 1; ++i) {
            oss << shape.GetDim(i) << ", ";
        }
        oss << shape.GetDim(shape.GetDimNum() - 1);
    }
    oss << "]";
    return oss.str();
}

inline static ge::graphStatus DynamicMxQuantSetTilingData(
    gert::TilingContext* context, DynamicMxQuantTilingData& tilingData)
{
    if (tilingData.GetDataSize() > context->GetRawTilingData()->GetCapacity()) {
        return ge::GRAPH_FAILED;
    }
    tilingData.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tilingData.GetDataSize());
    return ge::GRAPH_SUCCESS;
}

inline static void PrintTilingData(const gert::TilingContext* context, DynamicMxQuantTilingData& tilingData)
{
    OP_LOGI(
        context->GetNodeName(),
        "tilingData is totalCoreNum:%ld, usedCoreNum:%ld,  ubFactor:%ld, \
tailUbFactor:%ld, blockFactor:%ld, tailBlockFactor:%ld, uo:%ld, ubDim:%ld, dstType:%ld, blockSize:%ld, scaleAlg:%ld, \
blockSizeNumInAxis:%ld, tailBlockSize:%ld, isPad:%ld, postAxisSize:%ld, tilingKey:%ld, ubFactorDim0TailAxis:%ld, \
ubFactorDim1TailAxis:%ld, blockFactorDim0TailAxis:%ld, blockFactorDim1TailAxis:%ld, tailBlockFactorDim0TailAxis:%ld, \
tailBlockCountDim1TailAxis:%ld, tailCoreStartIdxDim0TailAxis:%ld, cutNumberForDim1TailAxis:%ld, \
tailUbFactorDim0TailAxis:%ld, isSplitDim1TailAxis:%ld, tailBlockFactorDim1ForSplitDim1:%ld, \
tailLoopBlockTailCoreDim1ForSplitDim1:%ld, dim1EachCoreForSplitDim1:%ld, inputShapeDim1TailAxis:%ld",
        tilingData.get_totalCoreNum(), tilingData.get_usedCoreNum(), tilingData.get_ubFactor(),
        tilingData.get_tailUbFactor(), tilingData.get_blockFactor(), tilingData.get_tailBlockFactor(),
        tilingData.get_uo(), tilingData.get_ubDim(), tilingData.get_dstType(), tilingData.get_blockSize(),
        tilingData.get_scaleAlg(), tilingData.get_blockSizeNumInAxis(), tilingData.get_tailBlockSize(),
        tilingData.get_isPad(), tilingData.get_postAxisSize(), tilingData.get_tilingKey(),
        tilingData.get_ubFactorDim0TailAxis(), tilingData.get_ubFactorDim1TailAxis(),
        tilingData.get_blockFactorDim0TailAxis(), tilingData.get_blockFactorDim1TailAxis(),
        tilingData.get_tailBlockFactorDim0TailAxis(), tilingData.get_tailBlockCountDim1TailAxis(),
        tilingData.get_tailCoreStartIdxDim0TailAxis(), tilingData.get_cutNumberForDim1TailAxis(),
        tilingData.get_tailUbFactorDim0TailAxis(), tilingData.get_isSplitDim1TailAxis(),
        tilingData.get_tailBlockFactorDim1ForSplitDim1(), tilingData.get_tailLoopBlockTailCoreDim1ForSplitDim1(),
        tilingData.get_dim1EachCoreForSplitDim1(), tilingData.get_inputShapeDim1TailAxis());
}

static RoundModeList GetRoundMode(const std::string& roundMode)
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

static ge::graphStatus GetAttr(const gert::TilingContext* context, DynamicMxQuantTilingParam& tilingParam)
{
    auto* attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    auto* attrAxis = attrs->GetAttrPointer<int64_t>(INDEX_ATTR_AXIS);
    OP_CHECK_NULL_WITH_CONTEXT(context, attrAxis);
    tilingParam.axis = static_cast<int64_t>(*attrAxis);
    OP_LOGD(context->GetNodeName(), "The attr axis is %ld", tilingParam.axis);

    auto outputYPtr = context->GetOutputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, outputYPtr);
    auto yDtype = outputYPtr->GetDataType();
    auto* attrRoundMode = attrs->GetAttrPointer<char>(INDEX_ATTR_ROUND_MODE);
    OP_CHECK_NULL_WITH_CONTEXT(context, attrRoundMode);
    std::string roundModeStr = attrRoundMode;
    RoundModeList roundMode = GetRoundMode(roundModeStr);
    OP_CHECK_IF(
        (roundMode == RoundModeList::MODE_UNDEFINED),
        OP_LOGE(
            context->GetNodeName(), "invalid round_mode:%s; round_mode should be one of {rint, floor, round}",
            attrRoundMode),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        (Y_SUPPORT_DTYPE_FP8_SET.count(yDtype) != 0 && roundMode != RoundModeList::MODE_RINT),
        OP_LOGE(
            context->GetNodeName(),
            "When output y's data type is FLOAT8_E4M3FN/FLOAT8_E5M2, round_mode only support rint, please check."),
        return ge::GRAPH_FAILED);
    tilingParam.roundMode = static_cast<int64_t>(roundMode);

    auto* attrDstType = attrs->GetAttrPointer<int64_t>(INDEX_ATTR_DST_DTYPE);
    OP_CHECK_NULL_WITH_CONTEXT(context, attrDstType);
    tilingParam.dstType = static_cast<int64_t>(*attrDstType);
    int checkDstType = static_cast<int>(*attrDstType);
    OP_CHECK_IF(
        (yDtype == ge::DT_FLOAT4_E2M1 && checkDstType != 40) || (yDtype == ge::DT_FLOAT4_E1M2 && checkDstType != 41) ||
            (yDtype == ge::DT_FLOAT8_E4M3FN && checkDstType != 36) ||
            (yDtype == ge::DT_FLOAT8_E5M2 && checkDstType != 35),
        OP_LOGE(
            context->GetNodeName(),
            "y's data type[%s] and dst_type[%d] is not corresponded, y's data type: "
            "FLOAT4_E2M1/FLOAT4_E1M2/FLOAT8_E4M3FN/FLOAT8_E5M2 correspond to dst_type: 40/41/36/35, please check.",
            Ops::Base::ToString(static_cast<ge::DataType>(yDtype)).c_str(), checkDstType),
        return ge::GRAPH_FAILED);

    auto* attrBlockSize = attrs->GetAttrPointer<int64_t>(INDEX_ATTR_BLOCK_SIZE);
    OP_CHECK_NULL_WITH_CONTEXT(context, attrBlockSize);
    tilingParam.blockSize = static_cast<int64_t>(*attrBlockSize);
    OP_CHECK_IF(
        tilingParam.blockSize <= 0 || tilingParam.blockSize > DYNAMIC_MX_QUANT_MAX_BLOCK_SIZE,
        OP_LOGE(
            context->GetNodeName(),
            "The blocksize[%ld] should should be larger than 0 and not be larger than 1024, please check.",
            tilingParam.blockSize),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        tilingParam.blockSize % ATTR_BLOCK_SIZE != 0,
        OP_LOGE(
            context->GetNodeName(), "The blocksize[%ld] should be a multiple of 32, please check.",
            tilingParam.blockSize),
        return ge::GRAPH_FAILED);

    auto* attrScaleAlg = attrs->GetAttrPointer<int64_t>(INDEX_ATTR_SCALE_ALG);
    OP_CHECK_NULL_WITH_CONTEXT(context, attrScaleAlg);
    tilingParam.scaleAlg = static_cast<int64_t>(*attrScaleAlg);
    OP_CHECK_IF(
        tilingParam.scaleAlg < 0 || tilingParam.scaleAlg > 1,
        OP_LOGE(context->GetNodeName(), "The scaleAlg[%ld] should be 0 or 1, please check.", tilingParam.scaleAlg),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF(
        tilingParam.scaleAlg == 1 && Y_SUPPORT_DTYPE_FP4_SET.count(yDtype) != 0,
        OP_LOGE(context->GetNodeName(), "When y's data type is FLOAT4_E2M1/FLOAT4_E1M2, scaleAlg must be set to 0."),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckDtype(const gert::TilingContext* context)
{
    auto inputXPtr = context->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputXPtr);
    auto xDtype = inputXPtr->GetDataType();
    OP_CHECK_IF(
        INPUT_SUPPORT_DTYPE_SET.count(xDtype) == 0,
        OP_LOGE(
            context->GetNodeName(),
            "Input x's data type[%s] only support FLOAT16 and BFLOAT16 currently, please check.",
            Ops::Base::ToString(static_cast<ge::DataType>(xDtype)).c_str()),
        return ge::GRAPH_FAILED);

    auto outputYPtr = context->GetOutputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, outputYPtr);
    auto yDtype = outputYPtr->GetDataType();
    OP_CHECK_IF(
        Y_SUPPORT_DTYPE_SET.count(yDtype) == 0,
        OP_LOGE(
            context->GetNodeName(),
            "Output y's data type[%s] only support FLOAT4_E2M1/FLOAT4_E1M2/FLOAT8_E4M3FN/FLOAT8_E5M2 currently, please "
            "check.",
            Ops::Base::ToString(static_cast<ge::DataType>(yDtype)).c_str()),
        return ge::GRAPH_FAILED);

    auto outputMxScalePtr = context->GetOutputDesc(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, outputMxScalePtr);
    auto scaleDtype = outputMxScalePtr->GetDataType();
    OP_CHECK_IF(
        OUTPUT_SUPPORT_DTYPE_SET.count(scaleDtype) == 0,
        OP_LOGE(
            context->GetNodeName(), "Input mxscale's data type[%s] only support FLOAT8_E8M0 currently, please check.",
            Ops::Base::ToString(static_cast<ge::DataType>(scaleDtype)).c_str()),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckShape(const gert::TilingContext* context, const DynamicMxQuantTilingParam& tilingParam)
{
    auto xShapePtr = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, xShapePtr);
    auto xShape = xShapePtr->GetStorageShape();

    OP_CHECK_IF(
        xShape.GetDimNum() < 1 || xShape.GetDimNum() > MAX_DIM_NUM,
        OP_LOGE(context->GetNodeName(), "Input x rank[%zu] should be in [1, 7].", xShape.GetDimNum()),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF(
        tilingParam.axis >= static_cast<int64_t>(xShape.GetDimNum()) ||
            tilingParam.axis < static_cast<int64_t>(-1 * xShape.GetDimNum()),
        OP_LOGE(
            context->GetNodeName(), "The attr axis is invalid, axis should be in [%ld, %ld], please check.",
            static_cast<int64_t>(-1 * xShape.GetDimNum()), static_cast<int64_t>(xShape.GetDimNum()) - 1),
        return ge::GRAPH_FAILED);

    auto outputYPtr = context->GetOutputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, outputYPtr);
    auto yDtype = outputYPtr->GetDataType();
    if (Y_SUPPORT_DTYPE_FP4_SET.count(yDtype) != 0) {
        OP_CHECK_IF(
            GetRemainder(xShape.GetDim(xShape.GetDimNum() - 1), DIGIT_TWO) != 0,
            OP_LOGE(
                context->GetNodeName(),
                "When output y's data type is FLOAT4_E2M1/FLOAT4_E1M2, the last axis should be even, please check."),
            return ge::GRAPH_FAILED);
    }

    auto yShapePtr = context->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, yShapePtr);
    auto yShape = yShapePtr->GetStorageShape();

    OP_CHECK_IF(
        xShape != yShape,
        OP_LOGE(context->GetNodeName(), "The shape of output y must be same with shape of input x, please check."),
        return ge::GRAPH_FAILED);

    auto axis = tilingParam.axis >= 0 ? tilingParam.axis : tilingParam.axis + xShape.GetDimNum();
    xShape.SetDim(axis, Ops::Base::CeilDiv(xShape.GetDim(axis), tilingParam.blockSize));

    auto mxScaleShapePtr = context->GetOutputShape(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, mxScaleShapePtr);
    auto mxScaleShape = mxScaleShapePtr->GetStorageShape();

    OP_CHECK_IF(
        mxScaleShape.GetDimNum() != xShape.GetDimNum() + 1,
        OP_LOGE(
            context->GetNodeName(), "Output mxscale rank[%zu] should be equal x rank[%zu] + 1.",
            mxScaleShape.GetDimNum(), xShape.GetDimNum()),
        return ge::GRAPH_FAILED);

    auto newScaleShape = xShape;
    newScaleShape.SetDim(axis, (xShape.GetDim(axis) + DIGIT_TWO - 1) / DIGIT_TWO);
    newScaleShape.AppendDim(DIGIT_TWO);

    OP_CHECK_IF(
        newScaleShape != mxScaleShape,
        OP_LOGE(
            context->GetNodeName(), "The shape of output mxscale %s is incorrect, correct shape is %s, please check.",
            Shape2String(mxScaleShape).c_str(), Shape2String(newScaleShape).c_str()),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

inline static void SetTilingKeyForTail(DataType inputType, DataType outputType, DynamicMxQuantTilingParam& tilingParam)
{
    int64_t hundredDigit = inputType == DT_FLOAT16 ? 1 : DIGIT_TWO;
    int64_t tenDigit = 0;
    if (outputType == DT_FLOAT4_E1M2) {
        tenDigit = 1;
    } else if (outputType == DT_FLOAT8_E4M3FN) {
        tenDigit = 2;
    } else if (outputType == DT_FLOAT8_E5M2) {
        tenDigit = 3;
    }
    int64_t digit = 0;
    if (tilingParam.isTailAxis && tilingParam.blockSize == ATTR_BLOCK_SIZE && inputType == DT_FLOAT16 && 
        Y_SUPPORT_DTYPE_FP4_SET.count(outputType) != 0 &&
        (tilingParam.roundMode == static_cast<int64_t>(RoundModeList::MODE_RINT) ||
         tilingParam.roundMode == static_cast<int64_t>(RoundModeList::MODE_ROUND))) {
        digit = 5;
    } else {
        digit = TAIL_TILING_KEY_DIGIT;
    }
    int64_t isOdd = tilingParam.blockSizeNumInAxis % DIGIT_TWO;
    int64_t axisScaleKey = isOdd == 0 && tilingParam.isTailAxis ? 0 : 1;
    tilingParam.tilingKey =
        hundredDigit * TILING_KEY_INPUT + tenDigit * TILING_KEY_OUTPUT + digit * TILING_KEY_TMPLATE + axisScaleKey;
    return;
}

inline static void CalcTilingKey(DataType inputType, DataType outputType, DynamicMxQuantTilingParam& tilingParam)
{
    // 百位数为1、2，分别表示输入类型是float16、bfloat16;
    int64_t hundredDigit = inputType == DT_FLOAT16 ? 1 : DIGIT_TWO;
    // 十位数为0、1、2、3，分别表示输出类型是float4_e2m1、float4_e1m2、float8_e4m3fn、float8_e5m2
    // 前面已做过Dtype校验
    int64_t tenDigit = 0;
    if (outputType == DT_FLOAT4_E1M2) {
        tenDigit = 1;
    } else if (outputType == DT_FLOAT8_E4M3FN) {
        tenDigit = 2;
    } else if (outputType == DT_FLOAT8_E5M2) {
        tenDigit = 3;
    }
    // 个位数为0、1、2，分别表示reduce尾轴、非尾轴且尾轴大于VL/2、非尾轴且尾轴小于等于VL/2
    int64_t digit = 0;  
    int64_t dtypeSize = inputType == DT_FLOAT16 ? DIGIT_FOUR : DIGIT_TWO; // float16需要转化为float32计算
    int64_t vRegSize = static_cast<int64_t>(tilingParam.vfLen) / dtypeSize / DIGIT_TWO;

    if (tilingParam.isTailAxis && tilingParam.blockSize == ATTR_BLOCK_SIZE && inputType == DT_FLOAT16 &&
 	         Y_SUPPORT_DTYPE_FP4_SET.count(outputType) != 0 &&
 	         (tilingParam.roundMode == static_cast<int64_t>(RoundModeList::MODE_RINT) ||
 	          tilingParam.roundMode == static_cast<int64_t>(RoundModeList::MODE_ROUND))) {
 	         digit = 5;
    } else {
        digit =
            tilingParam.isTailAxis ?
                0 :
                ((tilingParam.postAxisSize > vRegSize || tilingParam.blockSize > OPTIMISE_MAX_BLOCK_SIZE) ? 1 :
                                                                                                            DIGIT_TWO);
    }
 	
    int64_t isOdd = tilingParam.blockSizeNumInAxis % DIGIT_TWO;
    int64_t axisScaleKey = isOdd == 0 && tilingParam.isTailAxis ? 0 : 1;
    tilingParam.tilingKey =
        hundredDigit * TILING_KEY_INPUT + tenDigit * TILING_KEY_OUTPUT + digit * TILING_KEY_TMPLATE + axisScaleKey;
}

static void CalcAxisSize(DynamicMxQuantTilingParam& tilingParam, const gert::Shape& xShape)
{
    int64_t dimSize = xShape.GetDim(tilingParam.axis);
    if (dimSize < tilingParam.blockSize) {
        tilingParam.blockSize = dimSize;
    }
    for (int64_t i = 0; i < tilingParam.axis; i++) {
        tilingParam.preAxisSize *= xShape.GetDim(i);
    }
    tilingParam.blockSizeNumInAxis = Ops::Base::CeilDiv(dimSize, tilingParam.blockSize);
    tilingParam.preAxisSize *= tilingParam.blockSizeNumInAxis;
    tilingParam.tailBlockSize = GetRemainder(dimSize, tilingParam.blockSize);
    tilingParam.isPad = tilingParam.tailBlockSize != 0;
    if (tilingParam.tailBlockSize == 0) {
        tilingParam.tailBlockSize = tilingParam.blockSize;
    }
    tilingParam.axisSize = tilingParam.blockSize;
    tilingParam.quantAxisSize = tilingParam.blockSize;
    for (size_t i = tilingParam.axis + 1; i < xShape.GetDimNum(); i++) {
        tilingParam.postAxisSize *= xShape.GetDim(i);
    }
}
inline static int64_t FuseDimForTailAxis(const gert::Shape& inputShape)
{
    int64_t fusedDim0Value = 1;
    for (size_t i = 0; i < inputShape.GetDimNum() - 1; i++) {
        fusedDim0Value *= inputShape.GetDim(i);
    }
    return fusedDim0Value;
}
static void SetTilingDataForTailAxis(DynamicMxQuantTilingData& tilingData, const DynamicMxQuantTilingParam& tilingParam)
{
    tilingData.set_totalCoreNum(tilingParam.totalCoreNum);
    tilingData.set_usedCoreNum(tilingParam.usedCoreNum);
    tilingData.set_tilingKey(tilingParam.tilingKey);
    tilingData.set_roundMode(tilingParam.roundMode);
    tilingData.set_dstType(tilingParam.dstType);
    tilingData.set_blockSize(tilingParam.blockSize);
    tilingData.set_scaleAlg(tilingParam.scaleAlg);
    tilingData.set_ubFactorDim0TailAxis(tilingParam.ubFactorDim0TailAxis);
    tilingData.set_ubFactorDim1TailAxis(tilingParam.ubFactorDim1TailAxis);
    tilingData.set_blockFactorDim0TailAxis(tilingParam.blockFactorDim0TailAxis);
    tilingData.set_blockFactorDim1TailAxis(tilingParam.blockFactorDim1TailAxis);
    tilingData.set_tailBlockFactorDim0TailAxis(tilingParam.tailBlockFactorDim0TailAxis);
    tilingData.set_tailBlockCountDim1TailAxis(tilingParam.tailBlockCountDim1TailAxis);
    tilingData.set_tailCoreStartIdxDim0TailAxis(tilingParam.tailCoreStartIdxDim0TailAxis);
    tilingData.set_cutNumberForDim1TailAxis(tilingParam.cutNumberForDim1TailAxis);
    tilingData.set_isPad(tilingParam.isPad);
    tilingData.set_tailBlockSize(tilingParam.tailBlockSize);
    tilingData.set_tailUbFactorDim0TailAxis(tilingParam.tailUbFactorDim0TailAxis);
    tilingData.set_isSplitDim1TailAxis(tilingParam.isSplitDim1TailAxis);
    tilingData.set_tailBlockFactorDim1ForSplitDim1(tilingParam.tailBlockFactorDim1ForSplitDim1);
    tilingData.set_tailLoopBlockTailCoreDim1ForSplitDim1(tilingParam.tailLoopBlockTailCoreDim1ForSplitDim1);
    tilingData.set_dim1EachCoreForSplitDim1(tilingParam.dim1EachCoreForSplitDim1);
    tilingData.set_inputShapeDim1TailAxis(tilingParam.inputShapeDim1TailAxis);
    int64_t preAxisSize = tilingParam.preAxisSize / tilingParam.blockSizeNumInAxis;
    tilingData.set_mxScaleSize(tilingParam.mxScaleSize);
    tilingData.set_preAxisSize(preAxisSize);
    tilingData.set_postAxisSize(tilingParam.postAxisSize);
    tilingData.set_blockSizeNumInAxis(tilingParam.blockSizeNumInAxis);
    tilingData.set_isTailAxis(static_cast<int64_t>(tilingParam.isTailAxis));
    return;
}
static void CalcBaseTilingParam(
    const gert::Shape& inputShape, const DataType& inputDataType, const DataType& output1DataType,
    DynamicMxQuantTilingParam& tilingParam)
{
    int64_t fusedDim0 = FuseDimForTailAxis(inputShape);
    int64_t dim1Size = inputShape.GetDim(inputShape.GetDimNum() - 1);
    tilingParam.inputShapeDim1TailAxis = dim1Size;
    int64_t realDim1TailAxis = Ops::Base::CeilDiv(dim1Size, tilingParam.blockSize);
    int64_t ubFactorDim1TailAxis = DIM1_BLOCK_COUNT;
    int64_t doubleBufferMulByte = 1;
    if (Y_SUPPORT_DTYPE_FP4_SET.count(output1DataType) != 0) {
        doubleBufferMulByte = 1;
    } else if (Y_SUPPORT_DTYPE_FP8_SET.count(output1DataType) != 0) {
        doubleBufferMulByte = 2;
    }

    tilingParam.isPad = false;
    tilingParam.tailBlockSize = 0;
    if (dim1Size % tilingParam.blockSize != 0) {
        tilingParam.isPad = true;
        tilingParam.tailBlockSize = dim1Size - dim1Size / tilingParam.blockSize * tilingParam.blockSize;
    }
    int64_t maxUbAvailable = tilingParam.ubSize / N_BUFFER;

    int64_t inputSize =
        (DIM1_BLOCK_COUNT * tilingParam.blockSize * ge::GetSizeByDataType(inputDataType) + BLOCK_SIZE - 1) /
        BLOCK_SIZE * BLOCK_SIZE;
    int64_t ouputdataAndBuffer =
        (DIM1_BLOCK_COUNT * tilingParam.blockSize + BLOCK_SIZE - 1) / BLOCK_SIZE * BLOCK_SIZE * doubleBufferMulByte;
    int64_t outputScale = (DIM1_BLOCK_COUNT + BLOCK_SIZE - 1) / BLOCK_SIZE * BLOCK_SIZE;
    int64_t buffer = (DIM1_BLOCK_COUNT * ge::GetSizeByDataType(inputDataType) * DIGIT_TWO + BLOCK_SIZE - 1) /
                     BLOCK_SIZE * BLOCK_SIZE;
    // 输入的2个字节输出4位
    int64_t ubFactorDim0TailAxis = maxUbAvailable / (inputSize + ouputdataAndBuffer + outputScale + buffer);
    ubFactorDim0TailAxis = ubFactorDim0TailAxis / DIGIT_FOUR * DIGIT_FOUR;

    if (ubFactorDim0TailAxis > fusedDim0) {
        ubFactorDim0TailAxis = fusedDim0;
    }
    // 先按照dim0分核
    int64_t usedCoreNum = min(Ops::Base::CeilDiv(fusedDim0, ubFactorDim0TailAxis), tilingParam.totalCoreNum);
    int64_t blockFactorDim0TailAxis =
        Ops::Base::CeilDiv(Ops::Base::CeilDiv(fusedDim0, ubFactorDim0TailAxis), usedCoreNum);

    usedCoreNum = Ops::Base::CeilDiv(fusedDim0, ubFactorDim0TailAxis * blockFactorDim0TailAxis);

    int64_t tailBlockFactorDim0TailAxis = Ops::Base::CeilDiv(
        (fusedDim0 - (usedCoreNum - 1) * blockFactorDim0TailAxis * ubFactorDim0TailAxis), ubFactorDim0TailAxis);

    // 尾核最后一轮循环处理的dim0个数
    int64_t tailUbFactorDim0TailAxis = fusedDim0 - (usedCoreNum - 1) * blockFactorDim0TailAxis * ubFactorDim0TailAxis -
                                       (tailBlockFactorDim0TailAxis - 1) * ubFactorDim0TailAxis;
    if (realDim1TailAxis < DIM1_BLOCK_COUNT) {
        ubFactorDim1TailAxis = realDim1TailAxis;
    }
    SetTilingKeyForTail(inputDataType, output1DataType, tilingParam);
    tilingParam.ubFactorDim0TailAxis = ubFactorDim0TailAxis;
    tilingParam.ubFactorDim1TailAxis = ubFactorDim1TailAxis;
    tilingParam.tailUbFactorDim0TailAxis = tailUbFactorDim0TailAxis;
    tilingParam.usedCoreNum = usedCoreNum;
    tilingParam.blockFactorDim0TailAxis = blockFactorDim0TailAxis;
    tilingParam.tailBlockFactorDim0TailAxis = tailBlockFactorDim0TailAxis;
    return;
}
static void SplitDim0(DynamicMxQuantTilingParam& tilingParam)
{
    int64_t realDim1TailAxis = Ops::Base::CeilDiv(tilingParam.inputShapeDim1TailAxis, tilingParam.blockSize);
    int64_t ubFactorDim1TailAxis = tilingParam.ubFactorDim1TailAxis;
    int64_t blockFactorDim1TailAxis = Ops::Base::CeilDiv(realDim1TailAxis, ubFactorDim1TailAxis);

    // 包含最后非对齐的块
    tilingParam.blockFactorDim1TailAxis = blockFactorDim1TailAxis;
    tilingParam.tailBlockCountDim1TailAxis = realDim1TailAxis - (blockFactorDim1TailAxis - 1) * ubFactorDim1TailAxis;
    tilingParam.tailCoreStartIdxDim0TailAxis = tilingParam.usedCoreNum - 1;
    return;
}

static void SplitDim1(DynamicMxQuantTilingParam& tilingParam)
{
    // 从dim1借轴进行开核
    int64_t usedCoreNum = tilingParam.usedCoreNum;
    int64_t ubFactorDim1TailAxis = tilingParam.ubFactorDim1TailAxis;
    int64_t coreNumRatio = tilingParam.totalCoreNum / usedCoreNum;
    int64_t realDim1TailAxis = Ops::Base::CeilDiv(tilingParam.inputShapeDim1TailAxis, tilingParam.blockSize);
    int64_t dim1ForEachCore = Ops::Base::CeilDiv(realDim1TailAxis, coreNumRatio);
    dim1ForEachCore = max(dim1ForEachCore, DIM1_BLOCK_COUNT);
    int64_t dim1SplitCount = Ops::Base::CeilDiv(realDim1TailAxis, dim1ForEachCore);
    int64_t dim1ForTailCore = realDim1TailAxis - (dim1SplitCount - 1) * dim1ForEachCore;
    int64_t rowUsedCoreNum = usedCoreNum;
    usedCoreNum = usedCoreNum * dim1SplitCount;
    int64_t tailDim0StartCoreIdx = (rowUsedCoreNum - 1) * dim1SplitCount;

    bool isSplitDim1 = true;

    int64_t blockFactorDim1ForSplitDim1 = Ops::Base::CeilDiv(dim1ForEachCore, ubFactorDim1TailAxis);
    int64_t tailLoopBlockDim1ForSplitDim1 = dim1ForEachCore - (blockFactorDim1ForSplitDim1 - 1) * ubFactorDim1TailAxis;
    int64_t tailBlockFactorDim1ForSplitDim1 = Ops::Base::CeilDiv(dim1ForTailCore, ubFactorDim1TailAxis);

    int64_t tailLoopBlockTailCoreDim1ForSplitDim1 =
        dim1ForTailCore - (tailBlockFactorDim1ForSplitDim1 - 1) * ubFactorDim1TailAxis;
    tilingParam.blockFactorDim0TailAxis = 1;
    tilingParam.tailBlockFactorDim0TailAxis = 1;
    tilingParam.blockFactorDim1TailAxis = blockFactorDim1ForSplitDim1;
    tilingParam.tailBlockCountDim1TailAxis = tailLoopBlockDim1ForSplitDim1;
    tilingParam.tailBlockFactorDim1ForSplitDim1 = tailBlockFactorDim1ForSplitDim1;
    tilingParam.tailLoopBlockTailCoreDim1ForSplitDim1 = tailLoopBlockTailCoreDim1ForSplitDim1;
    tilingParam.tailCoreStartIdxDim0TailAxis = tailDim0StartCoreIdx;
    tilingParam.dim1EachCoreForSplitDim1 = dim1ForEachCore;
    tilingParam.usedCoreNum = usedCoreNum;
    tilingParam.isSplitDim1TailAxis = isSplitDim1;
    tilingParam.cutNumberForDim1TailAxis = dim1SplitCount;
    return;
}
static ge::graphStatus DoTilingForTailAxis(
    const gert::Shape& inputShape, const DataType& inputDataType, const DataType& output1DataType,
    DynamicMxQuantTilingParam& tilingParam)
{
    CalcBaseTilingParam(inputShape, inputDataType, output1DataType, tilingParam);
    // dim0方向分核
    // 增加判断列大小是否小于DIM1_BLOCK_COUNT
    if (tilingParam.usedCoreNum > tilingParam.totalCoreNum / DIGIT_TWO ||
        tilingParam.inputShapeDim1TailAxis <= DIM1_BLOCK_COUNT * tilingParam.blockSize) {
        SplitDim0(tilingParam);
    } else {
        SplitDim1(tilingParam);
    }
    return ge::GRAPH_SUCCESS;
}

static void CalScaleSize(const gert::Shape& inputShape, DynamicMxQuantTilingParam& tilingParam)
{
    int64_t scaleShapeSize = 1;
    int64_t xDimNum = inputShape.GetDimNum();
    int64_t dimSize = 0;
    for (int64_t i = 0; i < xDimNum; i++) {
        dimSize = inputShape.GetDim(i);
        if (i == tilingParam.axis) {
            dimSize = Ops::Base::CeilDiv(inputShape.GetDim(i), tilingParam.blockSize);
            dimSize = (dimSize + DIGIT_TWO - 1) / DIGIT_TWO * DIGIT_TWO;
        }
        scaleShapeSize = scaleShapeSize * dimSize;
    }
    tilingParam.mxScaleSize = scaleShapeSize;
}

static bool IsMinUbFactor(const DynamicMxQuantTilingParam& tilingParam, int64_t& multiples)
{
    bool isMinUbFactor = false;
    if (tilingParam.ubDim == 0) {
        if (tilingParam.postAxisSize <= SINGLE_LOOP_MIN_COLS) {
            isMinUbFactor = tilingParam.ubFactor <= 1;
            multiples = isMinUbFactor ? 1 : tilingParam.ubFactor;
        } else {
            multiples = tilingParam.ubFactor * Ops::Base::CeilDiv(tilingParam.postAxisSize, SINGLE_LOOP_MIN_COLS);
        }
    } else {
        isMinUbFactor = tilingParam.ubFactor <= SINGLE_LOOP_MIN_COLS;
        multiples = isMinUbFactor ? 1 : Ops::Base::CeilDiv(tilingParam.ubFactor, SINGLE_LOOP_MIN_COLS);
    }
    return isMinUbFactor;
}

static void SpliteUB(DynamicMxQuantTilingParam& tilingParam, int64_t maxUbAvailable)
{
    if (!tilingParam.isTailAxis) {
        if (tilingParam.postAxisSize > maxUbAvailable) {
            tilingParam.ubFactor = maxUbAvailable == 1 ? 1 : maxUbAvailable / DIGIT_TWO * DIGIT_TWO;
            tilingParam.ubDim = NEW_SHAPE_INDEX_TWO;
            maxUbAvailable = 0;
            tilingParam.uo = Ops::Base::CeilDiv(tilingParam.postAxisSize, tilingParam.ubFactor);
            tilingParam.tailUbFactor = GetRemainder(tilingParam.postAxisSize, tilingParam.ubFactor);
        } else {
            tilingParam.ubFactor = tilingParam.postAxisSize;
            maxUbAvailable /= tilingParam.ubFactor;
            tilingParam.ubDim = 0;
            tilingParam.uo = 1;
            tilingParam.tailUbFactor = 0;
        }
    }
    if (tilingParam.isTailAxis || maxUbAvailable != 0) {
        if (tilingParam.preAxisSize >= maxUbAvailable) {
            tilingParam.ubFactor = maxUbAvailable;
            if (maxUbAvailable > tilingParam.blockSizeNumInAxis) {
                tilingParam.ubFactor = maxUbAvailable / tilingParam.blockSizeNumInAxis * tilingParam.blockSizeNumInAxis;
            }
            tilingParam.uo = Ops::Base::CeilDiv(tilingParam.preAxisSize, tilingParam.ubFactor);
            tilingParam.tailUbFactor = GetRemainder(tilingParam.preAxisSize, tilingParam.ubFactor);
        } else {
            tilingParam.ubFactor = tilingParam.preAxisSize;
        }
    }
}

static ge::graphStatus DoTiling(const gert::TilingContext* context, DynamicMxQuantTilingParam& tilingParam)
{
    auto xShapePtr = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, xShapePtr);
    auto xShape = xShapePtr->GetStorageShape();
    tilingParam.axis = tilingParam.axis >= 0 ? tilingParam.axis : tilingParam.axis + xShape.GetDimNum();
    tilingParam.isTailAxis = tilingParam.axis == static_cast<int64_t>(xShape.GetDimNum() - 1);
    CalcAxisSize(tilingParam, xShape);

    // 获取输入/输出数据类型
    auto inputXPtr = context->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputXPtr);
    auto inDtype = inputXPtr->GetDataType();
    auto outputYPtr = context->GetOutputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, outputYPtr);
    auto outDtype = outputYPtr->GetDataType();
    CalcTilingKey(inDtype, outDtype, tilingParam);

    CalScaleSize(xShape, tilingParam);

    if (tilingParam.isTailAxis && tilingParam.blockSize == ATTR_BLOCK_SIZE) {
        tilingParam.tailAxisOptimize = true;
        return DoTilingForTailAxis(xShape, inDtype, outDtype, tilingParam);
    }

    int64_t maxUbAvailable = (tilingParam.ubSize - RESERVED_UB_SIZE) / N_BUFFER / BYTES_OF_INPUT_TYPE / EXIST_NODE_NUM /
                             tilingParam.blockSize;
    // 计算ubFactor
    SpliteUB(tilingParam, maxUbAvailable);
    int64_t spliteCoreData =
        (tilingParam.ubDim == NEW_SHAPE_INDEX_TWO) ? tilingParam.uo * tilingParam.preAxisSize : tilingParam.uo;
    int64_t multiples = 1;
    if (spliteCoreData <= tilingParam.totalCoreNum / DIGIT_TWO && !IsMinUbFactor(tilingParam, multiples)) {
        if (tilingParam.totalCoreNum % spliteCoreData == 0) {
            maxUbAvailable /= min(multiples, tilingParam.totalCoreNum / spliteCoreData);
        } else {
            maxUbAvailable = (multiples <= (tilingParam.totalCoreNum / spliteCoreData)) ?
                                 maxUbAvailable / multiples :
                                 maxUbAvailable * spliteCoreData / tilingParam.totalCoreNum;
        }
        SpliteUB(tilingParam, maxUbAvailable);
        spliteCoreData =
            (tilingParam.ubDim == NEW_SHAPE_INDEX_TWO) ? tilingParam.uo * tilingParam.preAxisSize : tilingParam.uo;
    }
    int64_t coreData = Ops::Base::CeilDiv(spliteCoreData, tilingParam.totalCoreNum);
    tilingParam.usedCoreNum = Ops::Base::CeilDiv(spliteCoreData, coreData);
    tilingParam.blockFactor = Ops::Base::CeilDiv(spliteCoreData, tilingParam.usedCoreNum);
    tilingParam.tailBlockFactor = spliteCoreData - (tilingParam.usedCoreNum - 1) * tilingParam.blockFactor;
    if (tilingParam.tailUbFactor == 0) {
        tilingParam.tailUbFactor = tilingParam.ubFactor;
    }
    return ge::GRAPH_SUCCESS;
}

static bool checkShapeForNotLastQuantAxis(
    const uint64_t nAlignNum, const uint64_t postAxisSize, const gert::TilingContext* context,
    DynamicMxQuantTilingParam tilingParam)
{
    uint64_t xUbSize = nAlignNum * tilingParam.blockSize * BLOCK_PER_GROUP * BYTES_OF_INPUT_TYPE;
    uint64_t yUbSize = nAlignNum * tilingParam.blockSize * BLOCK_PER_GROUP * MAX_BYTES_OF_OUTPUT_TYPE;
    uint64_t scaleUbSize = BLOCK_PER_GROUP * nAlignNum * UINT8_BYTES_SIZE; // scale is fp8
    int64_t groupPerUb = 0;
    bool isOptimize = true;

    if (postAxisSize > N_ALIGN128) {
        uint64_t tmpBufferUbSize = BLOCK_PER_GROUP * nAlignNum * UINT16_BYTES_SIZE;     // for 1 / scale
        uint64_t tmpScaleBufferUbSize = BLOCK_PER_GROUP * nAlignNum * UINT8_BYTES_SIZE; // for scale interleave
        groupPerUb = (tilingParam.ubSize - RESERVED_UB_SIZE) / N_BUFFER /
                     (xUbSize + yUbSize + scaleUbSize + tmpBufferUbSize + tmpScaleBufferUbSize);
        if (groupPerUb == 0) {
            OP_LOGD(context->GetNodeName(), "large tail : Shape too large to fit the optimal template.");
            return false;
        }
    } else {
        if (tilingParam.blockSize != 32 || postAxisSize < 64) {
            OP_LOGD(
                context->GetNodeName(),
                "small tail optimization template blockSize[%ld] should be 32, postAxisSize[%ld] should be no less "
                "than 64, please check",
                tilingParam.blockSize, postAxisSize);
            return false;
        }
        groupPerUb = (tilingParam.ubSize - RESERVED_UB_SIZE) / N_BUFFER / (xUbSize + yUbSize + scaleUbSize);
        if (groupPerUb == 0) {
            OP_LOGD(context->GetNodeName(), "small tail : Shape too large to fit the optimal template.");
            return false;
        }
    }

    return isOptimize;
}
// capable to be optimized
static bool IsOptForNotLastQuantAxis(const gert::TilingContext* context, DynamicMxQuantTilingParam& tilingParam)
{
    OP_LOGD(context->GetNodeName(), "Start to check input possible to be optimized.");

    auto xShapePtr = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, xShapePtr);
    auto xShape = xShapePtr->GetStorageShape();
    auto outputYPtr = context->GetOutputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, outputYPtr);
    auto outDtype = outputYPtr->GetDataType();

    auto* attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    auto* attrBlockSize = attrs->GetAttrPointer<int64_t>(INDEX_ATTR_BLOCK_SIZE);
    OP_CHECK_NULL_WITH_CONTEXT(context, attrBlockSize);

    int64_t negativeAxis = tilingParam.axis < 0 ? tilingParam.axis : tilingParam.axis - xShape.GetDimNum();
    OP_LOGD(context->GetNodeName(), "The quant axis is the %ld th axis. ", negativeAxis);
    if (negativeAxis >= -1) {
        OP_LOGD(context->GetNodeName(), "The quant axis is not the last axis, it can not be optimized. ");
        return false;
    }

    // 调整axis取值，使其始终为非负整数
    uint64_t axis = tilingParam.axis >= 0 ? tilingParam.axis : tilingParam.axis + xShape.GetDimNum();
    uint64_t postAxisSize = 1;
    uint64_t preAxisSize = 1;

    for (size_t i = axis + 1; i < xShape.GetDimNum(); i++) {
        postAxisSize *= xShape.GetDim(i);
    }

    for (size_t i = 0; i < axis; i++) {
        preAxisSize *= xShape.GetDim(i);
    }

    // 获取量化轴输入的shape大小
    int64_t dimSize = xShape.GetDim(axis);
    if (dimSize < tilingParam.blockSize) {
        tilingParam.blockSize = dimSize;
    }

    // 尾轴对齐后的大小
    auto nAlignNum = N_ALIGN128;
    if (postAxisSize > N_ALIGN128) {
        nAlignNum = N_ALIGN128;
    } else {
        if (postAxisSize <= N_ALIGN32) {
            nAlignNum = N_ALIGN32;
        } else if (postAxisSize <= N_ALIGN64) {
            nAlignNum = N_ALIGN64;
        } else {
            nAlignNum = N_ALIGN128;
        }
        if (nAlignNum == N_ALIGN32 && (Y_SUPPORT_DTYPE_FP4_SET.count(outDtype) != 0)) { // fp4需64对齐
            nAlignNum = N_ALIGN64;
        }
    }

    if (!checkShapeForNotLastQuantAxis(nAlignNum, postAxisSize, context, tilingParam)) {
        return false;
    }
    return true;
}

inline static void SetTilingData(DynamicMxQuantTilingData& tilingData, const DynamicMxQuantTilingParam& tilingParam)
{
    tilingData.set_totalCoreNum(tilingParam.totalCoreNum);
    tilingData.set_usedCoreNum(tilingParam.usedCoreNum);
    tilingData.set_uo(tilingParam.uo);
    tilingData.set_ubDim(tilingParam.ubDim);
    tilingData.set_ubFactor(tilingParam.ubFactor);
    tilingData.set_tailUbFactor(tilingParam.tailUbFactor);
    tilingData.set_blockFactor(tilingParam.blockFactor);
    tilingData.set_tailBlockFactor(tilingParam.tailBlockFactor);
    tilingData.set_tilingKey(tilingParam.tilingKey);
    tilingData.set_roundMode(tilingParam.roundMode);
    tilingData.set_dstType(tilingParam.dstType);
    tilingData.set_blockSize(tilingParam.blockSize);
    tilingData.set_scaleAlg(tilingParam.scaleAlg);
    tilingData.set_tailBlockSize(tilingParam.tailBlockSize);
    tilingData.set_blockSizeNumInAxis(tilingParam.blockSizeNumInAxis);
    int64_t isPad = tilingParam.isPad ? 1 : 0;
    tilingData.set_isPad(isPad);
    int64_t preAxisSize = tilingParam.preAxisSize / tilingParam.blockSizeNumInAxis;
    tilingData.set_preAxisSize(preAxisSize);
    tilingData.set_postAxisSize(tilingParam.postAxisSize);
    tilingData.set_mxScaleSize(tilingParam.mxScaleSize);
    tilingData.set_isTailAxis(static_cast<int64_t>(tilingParam.isTailAxis));
}

ge::graphStatus Tiling4DynamicMxQuant(gert::TilingContext* context)
{
    OP_LOGD(context->GetNodeName(), "Tiling4DynamicMxQuant running begin.");

    DynamicMxQuantTilingParam tilingParam;

    OP_CHECK_IF(
        CheckDtype(context) != ge::GRAPH_SUCCESS, OP_LOGE(context->GetNodeName(), "The data type check failed."),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF(
        GetAttr(context, tilingParam) != ge::GRAPH_SUCCESS, OP_LOGE(context->GetNodeName(), "The attr get failed."),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF(
        CheckShape(context, tilingParam) != ge::GRAPH_SUCCESS,
        OP_LOGE(context->GetNodeName(), "The shape check failed."), return ge::GRAPH_FAILED);

    auto platformInfo = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    tilingParam.totalCoreNum = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF(
        (tilingParam.totalCoreNum <= 0), OP_LOGE(context->GetNodeName(), "Failed to core num."),
        return ge::GRAPH_FAILED);
    uint64_t ubSize;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    tilingParam.ubSize = static_cast<int64_t>(ubSize);

    OP_CHECK_IF(
        (tilingParam.ubSize <= 0), OP_LOGE(context->GetNodeName(), "Failed to get ub size."), return ge::GRAPH_FAILED);
    tilingParam.vfLen = Ops::Base::GetVRegSize(context);
    tilingParam.workspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();

    bool isOptimizable = IsOptForNotLastQuantAxis(context, tilingParam);
    // 进入性能优化模板判断
    if (isOptimizable) {
        DynamicMxQuantOptimzieTiling regbaseTiling(context);
        return regbaseTiling.DoTiling();
    } else {
        OP_CHECK_IF(
            DoTiling(context, tilingParam) != ge::GRAPH_SUCCESS, OP_LOGE(context->GetNodeName(), "Dotiling failed."),
            return ge::GRAPH_FAILED);
    }

    DynamicMxQuantTilingData tilingData;
    if (tilingParam.tailAxisOptimize) {
        SetTilingDataForTailAxis(tilingData, tilingParam);
    } else {
        SetTilingData(tilingData, tilingParam);
    }
    OP_CHECK_IF(
        DynamicMxQuantSetTilingData(context, tilingData) != ge::GRAPH_SUCCESS,
        OP_LOGE(context->GetNodeName(), "DynamicMxQuantSetTilingData set tiling data fail."), return ge::GRAPH_FAILED);
    context->SetBlockDim(tilingData.get_usedCoreNum());
    context->SetTilingKey(tilingData.get_tilingKey());
    size_t* workspaces = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, workspaces);
    workspaces[0] = tilingParam.workspaceSize + Ops::Base::CeilAlign(tilingParam.mxScaleSize, WORKSPACE_ALIGN_SIZE);

    PrintTilingData(context, tilingData);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingPrepare4DynamicMxQuant(gert::TilingParseContext* context)
{
    OP_LOGD(context->GetNodeName(), "TilingPrepare4DynamicMxQuant entering.");
    return ge::GRAPH_SUCCESS;
}

// register tiling interface of the DynamicMxQuant op.
IMPL_OP_OPTILING(DynamicMxQuant)
    .Tiling(Tiling4DynamicMxQuant)
    .TilingParse<DynamicMxQuantCompileInfo>(TilingPrepare4DynamicMxQuant);
} // namespace optiling

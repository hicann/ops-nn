/**
 * Copyright (c) 2026-2026 Huawei Technologies Co., Ltd.
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
using namespace DynamicMxQuant;

namespace optiling {
constexpr int64_t INDEX_ATTR_AXIS = 0;
constexpr int64_t INDEX_ATTR_ROUND_MODE = 1;
constexpr int64_t INDEX_ATTR_DST_DTYPE = 2;
constexpr int64_t INDEX_ATTR_BLOCK_SIZE = 3;
constexpr int64_t INDEX_ATTR_SCALE_ALG = 4;
constexpr int64_t INDEX_ATTR_DST_DTYPE_MAX = 5;
constexpr int64_t INDEX_ATTR_MAX_LOW_BOUND = 6;
constexpr int64_t NUM_ZERO = 0;
constexpr int64_t NUM_ONE = 1;
constexpr int64_t NUM_TWO = 2;
constexpr float DST_TYPE_DEFAULT = 0.0;
constexpr float NUM_SIX_FLOAT = 6.0;
constexpr float NUM_SEVEN_FLOAT = 7.0;
constexpr int64_t MODE_ZERO = 0;
constexpr int64_t MODE_ONE = 1;
constexpr int64_t MODE_TWO = 2;
constexpr int64_t MODE_THREE = 3;
constexpr int64_t BYTES_OF_INPUT_TYPE = 2;
constexpr int64_t BYTES_OF_INPUT_TYPE_FP32 = 4;
constexpr int64_t BYTES_OF_BF16_SCRATCH = 2; // fp32 输入需要的 bf16 中转 buffer，每元素 2 字节
constexpr int64_t MAX_BYTES_OF_OUTPUT_TYPE = 1;
constexpr int64_t DIGIT_TEN_THOUSAND = 10000;
constexpr int64_t DIGIT_HUNDRED = 100;
constexpr int64_t DIGIT_TEN = 10;
constexpr int64_t N_ALIGN32 = 32;

constexpr int64_t BLOCK_PER_GROUP = 2;
constexpr int64_t UINT16_BYTES_SIZE = 2;
constexpr int64_t UINT8_BYTES_SIZE = 1;
constexpr int64_t DIGIT_TWO = 2;
constexpr int64_t DIGIT_FOUR = 4;
constexpr int64_t TILING_KEY_TAIL_AXIS = 100;
constexpr int64_t TILING_KEY_OTHER_AXIS_LARGE = 110;
constexpr int64_t TILING_KEY_OTHER_AXIS_SMALL = 120;
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
const std::set<ge::DataType> INPUT_SUPPORT_DTYPE_SET = {ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT};
const std::set<ge::DataType> INPUT_FP16_DTYPE_SET = {ge::DT_FLOAT16};
const std::set<ge::DataType> INPUT_FP32_DTYPE_SET = {ge::DT_FLOAT};
// FLOATX_EYMZ: Total bit of one float is X, in which first 1 bit for sign, following Y bit for exponent and Z bit for
// mantissa.
const std::set<ge::DataType> Y_SUPPORT_DTYPE_SET = {ge::DT_FLOAT4_E2M1, ge::DT_FLOAT4_E1M2, ge::DT_FLOAT8_E4M3FN,
                                                    ge::DT_FLOAT8_E5M2};
const std::set<ge::DataType> Y_SUPPORT_DTYPE_FP4_SET = {ge::DT_FLOAT4_E2M1, ge::DT_FLOAT4_E1M2};
const std::set<ge::DataType> Y_SUPPORT_DTYPE_FP8_SET = {ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E5M2};
const std::set<ge::DataType> OUTPUT_SUPPORT_DTYPE_SET = {ge::DT_FLOAT8_E8M0};
constexpr int64_t DIM1_BLOCK_COUNT = 8;
constexpr int64_t BLOCK_SIZE = 32;
constexpr int64_t TAIL_TILING_KEY_DIGIT = 4;
constexpr int64_t SINGLE_LOOP_MIN_COLS = 128;
constexpr float FP4E2M1_MAX = 6.0;
constexpr size_t MAX_DIM_NUM = 7;

// 主模板 tilingKey 只编码 kernel 需要选择的算法路径，不再携带输入/输出 dtype。
// dtype 组合已经在 binary 编译时通过 DTYPE_X/DTYPE_Y 固化，host 只负责区分 axis/tail 形态和是否需要 scale post：
// 100/101: 尾轴主路径，101 需要先写 workspace，再由 post 逻辑回写 mxScale。
// 110/111: 非尾轴大尾路径，111 需要 scale post。
// 120/121: 非尾轴小尾优化路径，121 需要 scale post。

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

inline static ge::graphStatus DynamicMxQuantSetTilingData(gert::TilingContext* context,
                                                          DynamicMxQuantTilingData& tilingData)
{
    uint64_t tilingDataSize = sizeof(tilingData);
    OP_CHECK_NULL_WITH_CONTEXT(context, context->GetRawTilingData());
    auto rawTilingData = context->GetRawTilingData();
    errno_t ret = memcpy_s(rawTilingData->GetData(), rawTilingData->GetCapacity(), reinterpret_cast<void*>(&tilingData),
                           tilingDataSize);
    if (ret != EOK) {
        OP_LOGE(context->GetNodeName(), "memcpy_s failed, ret = %d", ret);
        return ge::GRAPH_FAILED;
    }
    context->GetRawTilingData()->SetDataSize(tilingDataSize);
    return ge::GRAPH_SUCCESS;
}

inline static void PrintTilingData(const gert::TilingContext* context, DynamicMxQuantTilingData& tilingData)
{
    OP_LOGI(context->GetNodeName(), "tilingData is totalCoreNum:%ld, usedCoreNum:%ld,  ubFactor:%ld, \
        tailUbFactor:%ld, blockFactor:%ld, tailBlockFactor:%ld, uo:%ld, ubDim:%ld, dstType:%ld, blockSize:%ld, scaleAlg:%ld, \
        blockSizeNumInAxis:%ld, tailBlockSize:%ld, isPad:%ld, postAxisSize:%ld, tilingKey:%ld, calcMode: %ld, \
        subNumForScale: %d, subNumForFP16Scale: %d,dstTypeMax:%f, invDstTypeMax:%f, maxLowBound:%f.",
            tilingData.totalCoreNum, tilingData.usedCoreNum, tilingData.ubFactor, tilingData.tailUbFactor,
            tilingData.blockFactor, tilingData.tailBlockFactor, tilingData.uo, tilingData.ubDim, tilingData.dstType,
            tilingData.blockSize, tilingData.scaleAlg, tilingData.blockSizeNumInAxis, tilingData.tailBlockSize,
            tilingData.isPad, tilingData.postAxisSize, tilingData.tilingKey, tilingData.calcMode,
            tilingData.subNumForScale, tilingData.subNumForFP16Scale, tilingData.dstTypeMax, tilingData.invDstTypeMax,
            tilingData.maxLowBound);
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
    OP_CHECK_IF((roundMode == RoundModeList::MODE_UNDEFINED),
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "round_mode", roundModeStr,
                                                      "The value of round_mode must be [rint, round, floor]"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF((Y_SUPPORT_DTYPE_FP8_SET.count(yDtype) != 0 && roundMode != RoundModeList::MODE_RINT),
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                    context->GetNodeName(), "round_mode", roundModeStr,
                    "If the dtype of output y is FLOAT8_E4M3FN/FLOAT8_E5M2, parameter round_mode must be rint"),
                return ge::GRAPH_FAILED);
    tilingParam.roundMode = static_cast<int64_t>(roundMode);

    auto* attrDstType = attrs->GetAttrPointer<int64_t>(INDEX_ATTR_DST_DTYPE);
    OP_CHECK_NULL_WITH_CONTEXT(context, attrDstType);
    tilingParam.dstType = static_cast<int64_t>(*attrDstType);
    int64_t checkDstType = static_cast<int64_t>(*attrDstType);
    OP_CHECK_IF((yDtype == ge::DT_FLOAT4_E2M1 && checkDstType != ge::DT_FLOAT4_E2M1) ||
                    (yDtype == ge::DT_FLOAT4_E1M2 && checkDstType != ge::DT_FLOAT4_E1M2) ||
                    (yDtype == ge::DT_FLOAT8_E4M3FN && checkDstType != ge::DT_FLOAT8_E4M3FN) ||
                    (yDtype == ge::DT_FLOAT8_E5M2 && checkDstType != ge::DT_FLOAT8_E5M2),
                OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(
                    context->GetNodeName(), "y, dst_type",
                    ge::TypeUtils::DataTypeToSerialString(yDtype) + ", " + std::to_string(checkDstType),
                    "The dtypes of y and dst_type must be the same"),
                return ge::GRAPH_FAILED);

    auto* attrBlockSize = attrs->GetAttrPointer<int64_t>(INDEX_ATTR_BLOCK_SIZE);
    OP_CHECK_NULL_WITH_CONTEXT(context, attrBlockSize);
    tilingParam.blockSize = static_cast<int64_t>(*attrBlockSize);
    OP_CHECK_IF(tilingParam.blockSize <= 0 || tilingParam.blockSize > DYNAMIC_MX_QUANT_MAX_BLOCK_SIZE,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "block_size",
                                                      std::to_string(tilingParam.blockSize),
                                                      "The value of block_size must be within the range (0, 1024]"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(tilingParam.blockSize % ATTR_BLOCK_SIZE != 0,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "block_size",
                                                      std::to_string(tilingParam.blockSize),
                                                      "The value of block_size must be a multiple of 32"),
                return ge::GRAPH_FAILED);

    auto inputXPtr = context->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputXPtr);
    auto xDtype = inputXPtr->GetDataType();
    OP_CHECK_IF(xDtype == ge::DT_FLOAT && tilingParam.blockSize != ATTR_BLOCK_SIZE,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                    context->GetNodeName(), "block_size", std::to_string(tilingParam.blockSize),
                    "If the dtype of input x is FLOAT, parameter block_size must be 32"),
                return ge::GRAPH_FAILED);

    auto* attrScaleAlg = attrs->GetAttrPointer<int64_t>(INDEX_ATTR_SCALE_ALG);
    OP_CHECK_NULL_WITH_CONTEXT(context, attrScaleAlg);
    tilingParam.scaleAlg = static_cast<int64_t>(*attrScaleAlg);
    OP_CHECK_IF(
        tilingParam.scaleAlg < 0 || tilingParam.scaleAlg > 2,
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "scale_alg", std::to_string(tilingParam.scaleAlg),
                                              "The value of scale_alg must be [0, 1, 2]"),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF(tilingParam.scaleAlg == 1 && yDtype == ge::DT_FLOAT4_E2M1,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                    context->GetNodeName(), "scale_alg", std::to_string(tilingParam.scaleAlg),
                    "If the dtype of output y is FLOAT4_E2M1, parameter scale_alg must be 0 or 2"),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(
        (tilingParam.scaleAlg == 1 || tilingParam.scaleAlg == 2) && yDtype == ge::DT_FLOAT4_E1M2,
        OP_LOGE_FOR_INVALID_VALUE(context->GetNodeName(), "scale_alg", std::to_string(tilingParam.scaleAlg), "0"),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF(tilingParam.scaleAlg == 2 && Y_SUPPORT_DTYPE_FP8_SET.count(yDtype) != 0,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                    context->GetNodeName(), "scale_alg", std::to_string(tilingParam.scaleAlg),
                    "If the dtype of output y is FLOAT8_E4M3FN/FLOAT8_E5M2, parameter scale_alg must be 0 or 1"),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(
        tilingParam.scaleAlg == 2 && tilingParam.blockSize != ATTR_BLOCK_SIZE,
        OP_LOGE_FOR_INVALID_VALUE(context->GetNodeName(), "block_size", std::to_string(tilingParam.blockSize), "32"),
        return ge::GRAPH_FAILED);

    auto* attrDstTypeMax = attrs->GetAttrPointer<float>(INDEX_ATTR_DST_DTYPE_MAX);
    OP_CHECK_NULL_WITH_CONTEXT(context, attrDstTypeMax);
    tilingParam.dstTypeMax = static_cast<float>(*attrDstTypeMax);

    if (!Ops::Base::IsFloatEqual(tilingParam.dstTypeMax, 0.0f)) {
        tilingParam.invDstTypeMax = 1.0 / tilingParam.dstTypeMax;
    } else {
        // 当dstTypeMax=0时，默认使用目标数据类型最大值，当前为FP4E2M1最大值6
        tilingParam.invDstTypeMax = 1.0 / FP4E2M1_MAX;
    }

    OP_CHECK_IF(tilingParam.dstTypeMax < 0 || (tilingParam.dstTypeMax > 0 && tilingParam.dstTypeMax < 6) ||
                    tilingParam.dstTypeMax > 12,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                    context->GetNodeName(), "dst_type_max", std::to_string(tilingParam.dstTypeMax),
                    "The value of dst_type_max must be within the range [6, 12] or equal to 0"),
                return ge::GRAPH_FAILED);

    auto* attrMaxLowBound = attrs->GetAttrPointer<float>(INDEX_ATTR_MAX_LOW_BOUND);
    if (attrMaxLowBound != nullptr) {
        tilingParam.maxLowBound = static_cast<float>(*attrMaxLowBound);
    }

    OP_CHECK_IF(tilingParam.scaleAlg != 1 && !Ops::Base::IsFloatEqual(tilingParam.maxLowBound, 0.0f),
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "max_low_bound",
                                                      std::to_string(tilingParam.maxLowBound),
                                                      "max_low_bound must be 0 when scale_alg != 1"),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(tilingParam.maxLowBound < 0,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "max_low_bound",
                                                      std::to_string(tilingParam.maxLowBound),
                                                      "max_low_bound must be non-negative"),
                return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckDtype(const gert::TilingContext* context)
{
    auto inputXPtr = context->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputXPtr);
    auto xDtype = inputXPtr->GetDataType();
    OP_CHECK_IF(INPUT_SUPPORT_DTYPE_SET.count(xDtype) == 0,
                OP_LOGE_FOR_INVALID_DTYPE(context->GetNodeName(), "x", ge::TypeUtils::DataTypeToSerialString(xDtype),
                                          "[DT_FLOAT16, DT_BF16, DT_FLOAT]"),
                return ge::GRAPH_FAILED);

    auto outputYPtr = context->GetOutputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, outputYPtr);
    auto yDtype = outputYPtr->GetDataType();
    OP_CHECK_IF(Y_SUPPORT_DTYPE_SET.count(yDtype) == 0,
                OP_LOGE_FOR_INVALID_DTYPE(context->GetNodeName(), "y", ge::TypeUtils::DataTypeToSerialString(yDtype),
                                          "[DT_FLOAT4_E2M1, DT_FLOAT4_E1M2, DT_FLOAT8_E4M3FN, DT_FLOAT8_E5M2]"),
                return ge::GRAPH_FAILED);

    auto outputMxScalePtr = context->GetOutputDesc(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, outputMxScalePtr);
    auto scaleDtype = outputMxScalePtr->GetDataType();
    OP_CHECK_IF(OUTPUT_SUPPORT_DTYPE_SET.count(scaleDtype) == 0,
                OP_LOGE_FOR_INVALID_DTYPE(context->GetNodeName(), "mxscale",
                                          ge::TypeUtils::DataTypeToSerialString(scaleDtype), "[DT_FLOAT8_E8M0]"),
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
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(context->GetNodeName(), "x", std::to_string(xShape.GetDimNum()),
                                                 "The shape dim of x must be within the range [1, 7]"),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF(tilingParam.axis >= static_cast<int64_t>(xShape.GetDimNum()) ||
                    tilingParam.axis < static_cast<int64_t>(-1 * xShape.GetDimNum()),
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "axis", std::to_string(tilingParam.axis),
                                                      "The value of axis must be within the range [-" +
                                                          std::to_string(xShape.GetDimNum()) + ", " +
                                                          std::to_string(xShape.GetDimNum() - 1) + "]"),
                return ge::GRAPH_FAILED);

    auto outputYPtr = context->GetOutputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, outputYPtr);
    auto yDtype = outputYPtr->GetDataType();
    if (Y_SUPPORT_DTYPE_FP4_SET.count(yDtype) != 0) {
        OP_CHECK_IF(GetRemainder(xShape.GetDim(xShape.GetDimNum() - 1), DIGIT_TWO) != 0,
                    OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                        context->GetNodeName(), "x", Ops::Base::ToString(xShape),
                        "When the yDtype is FLOAT4_E2M1 or FLOAT4_E1M2, the tail axis of x must be an even number"),
                    return ge::GRAPH_FAILED);
    }

    auto yShapePtr = context->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, yShapePtr);
    auto yShape = yShapePtr->GetStorageShape();

    OP_CHECK_IF(xShape != yShape,
                OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(context->GetNodeName(), "x, y",
                                                       Ops::Base::ToString(xShape) + ", " + Ops::Base::ToString(yShape),
                                                       "The shapes of x and y must be the same"),
                return ge::GRAPH_FAILED);

    auto axis = tilingParam.axis >= 0 ? tilingParam.axis : tilingParam.axis + xShape.GetDimNum();
    xShape.SetDim(axis, Ops::Base::CeilDiv(xShape.GetDim(axis), tilingParam.blockSize));

    auto mxScaleShapePtr = context->GetOutputShape(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, mxScaleShapePtr);
    auto mxScaleShape = mxScaleShapePtr->GetStorageShape();

    OP_CHECK_IF(mxScaleShape.GetDimNum() != xShape.GetDimNum() + 1,
                OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(context->GetNodeName(), "mxscale",
                                                         std::to_string(mxScaleShape.GetDimNum()),
                                                         "The shape dim of mxscale must be the shape dim of x plus 1"),
                return ge::GRAPH_FAILED);

    auto newScaleShape = xShape;
    newScaleShape.SetDim(axis, (xShape.GetDim(axis) + DIGIT_TWO - 1) / DIGIT_TWO);
    newScaleShape.AppendDim(DIGIT_TWO);

    OP_CHECK_IF(newScaleShape != mxScaleShape,
                OP_LOGE_FOR_INVALID_SHAPE(context->GetNodeName(), "mxscale", Shape2String(mxScaleShape),
                                          Shape2String(newScaleShape)),
                return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

inline static void CalcTilingKey(DataType inputType, DynamicMxQuantTilingParam& tilingParam)
{
    // dtype 不再直接参与 tilingKey 编码，这里只保留输入 dtype 对寄存器容量估算的影响，
    // 用于判断非尾轴场景走 large-tail 还是 small-tail 优化路径。
    // float16 / float32 都按 4 字节估算（fp16 升 fp32 计算，fp32 原生 4 字节）
    int64_t dtypeSize = (inputType == DT_FLOAT16 || inputType == DT_FLOAT) ? DIGIT_FOUR : DIGIT_TWO;
    int64_t vRegSize = static_cast<int64_t>(tilingParam.vfLen) / dtypeSize / DIGIT_TWO;
    uint64_t optiMode = TPL_TAIL_AXIS_QUANT_NORMAL;
    if (!tilingParam.isTailAxis) {
        optiMode = (tilingParam.postAxisSize > vRegSize || tilingParam.blockSize > OPTIMISE_MAX_BLOCK_SIZE) ?
                       TPL_NOT_TAIL_AXIS_QUANT_NORMAL :
                       TPL_NOT_TAIL_AXIS_QUANT_OPTI;
    }
    int64_t isOdd = tilingParam.blockSizeNumInAxis % DIGIT_TWO;
    // 尾轴且 block 数为偶数时可以直接写回 mxScale，其余情况都需要走 scale post。
    uint64_t isOddScale = (isOdd == 0 && tilingParam.isTailAxis) ? TPL_EVEN_SCALE : TPL_ODD_SCALE;
    // 这里RoundMode和ScaleAlg不重要，模板从TilingData中获取。
    int64_t tilingKey = GET_TPL_TILING_KEY(optiMode, TPL_NOT_CARE_SCALE_ALG, TPL_NOT_CARE_ROUND_MODE, isOddScale);
    tilingParam.tilingKey = tilingKey;
}

static ge::graphStatus SetCalcMode(const gert::TilingContext* context, DynamicMxQuantTilingParam& tilingParam)
{
    if (tilingParam.scaleAlg == NUM_ZERO) {
        tilingParam.calcMode = MODE_ZERO;
    } else if (tilingParam.scaleAlg == NUM_ONE) {
        tilingParam.calcMode = MODE_ONE;
    } else if (tilingParam.scaleAlg == NUM_TWO) {
        if (Ops::Base::IsFloatEqual(tilingParam.dstTypeMax, DST_TYPE_DEFAULT) ||
            Ops::Base::IsFloatEqual(tilingParam.dstTypeMax, NUM_SIX_FLOAT) ||
            Ops::Base::IsFloatEqual(tilingParam.dstTypeMax, NUM_SEVEN_FLOAT)) {
            tilingParam.calcMode = MODE_TWO;
        } else {
            tilingParam.calcMode = MODE_THREE;
        }
    } else {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "scale_alg", std::to_string(tilingParam.scaleAlg),
                                              "The value of scale_alg must be [0, 1, 2]");
        return ge::GRAPH_FAILED;
    }

    if (tilingParam.calcMode == MODE_TWO) {
        if (Ops::Base::IsFloatEqual(tilingParam.dstTypeMax, DST_TYPE_DEFAULT) ||
            Ops::Base::IsFloatEqual(tilingParam.dstTypeMax, NUM_SIX_FLOAT)) {
            tilingParam.subNumForScale = 0x000000c1;
            tilingParam.subNumForFP16Scale = 0x00c00001;
        } else {
            tilingParam.subNumForScale = 0x000000e1;
            tilingParam.subNumForFP16Scale = 0x00e00001;
        }
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus BaseCalc(const gert::TilingContext* context, DynamicMxQuantTilingParam& tilingParam)
{
    auto xShapePtr = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, xShapePtr);
    auto xShape = xShapePtr->GetStorageShape();
    // 将量化轴索引转换成非负数
    tilingParam.axis = tilingParam.axis >= 0 ? tilingParam.axis : tilingParam.axis + xShape.GetDimNum();
    // 量化轴是否是尾轴
    tilingParam.isTailAxis = tilingParam.axis == static_cast<int64_t>(xShape.GetDimNum() - 1);
    int64_t dimSize = xShape.GetDim(tilingParam.axis);
    if (tilingParam.blockSize == ATTR_BLOCK_SIZE) {
        tilingParam.isBlockSize32 = true;
    }
    if (dimSize < tilingParam.blockSize) {
        tilingParam.blockSize = dimSize;
    }
    if (tilingParam.blockSize ==
        ATTR_BLOCK_SIZE) { // 需要再次判断，存在量化轴为尾轴，尾轴大小为32，但是blockSize>32时，也应该走尾轴、blockSize=32的模板
        tilingParam.isBlockSize32 = true;
    }
    // 合轴
    for (int64_t i = 0; i < tilingParam.axis; i++) {
        tilingParam.preAxisSize *= xShape.GetDim(i);
    }
    tilingParam.quantAxisSize = dimSize;
    OP_CHECK_IF(
        tilingParam.isFp32Input && dimSize < ATTR_BLOCK_SIZE,
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "x", std::to_string(dimSize),
                                              "FP32 input with quantization axis size less than 32 is not supported"),
        return ge::GRAPH_FAILED);
    for (size_t i = tilingParam.axis + 1; i < xShape.GetDimNum(); i++) {
        tilingParam.postAxisSize *= xShape.GetDim(i);
    }
    tilingParam.blockSizeNumInAxis = Ops::Base::CeilDiv(dimSize, tilingParam.blockSize);
    tilingParam.tailBlockSize = GetRemainder(dimSize, tilingParam.blockSize);
    tilingParam.isPad = tilingParam.tailBlockSize != 0;
    if (tilingParam.tailBlockSize == 0) {
        tilingParam.tailBlockSize = tilingParam.blockSize;
    }
    tilingParam.nAlignNum = static_cast<int64_t>(tilingParam.vfLen) / tilingParam.inputDtypeSize;
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

    // 获取输入/输出数据类型
    auto inputXPtr = context->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputXPtr);
    auto inDtype = inputXPtr->GetDataType();
    tilingParam.isFp32Input = (inDtype == DT_FLOAT);
    tilingParam.inputDtypeSize = tilingParam.isFp32Input ? DIGIT_FOUR : DIGIT_TWO;
    CalcTilingKey(inDtype, tilingParam);
    OP_CHECK_IF(SetCalcMode(context, tilingParam) != ge::GRAPH_SUCCESS,
                OP_LOGE(context->GetNodeName(), "Set calculation mode failed."), return ge::GRAPH_FAILED);

    tilingParam.preAxisSize *= tilingParam.blockSizeNumInAxis;

    CalScaleSize(xShape, tilingParam);

    // fp32 输入：UB 需要同时存放 fp32 数据（4B）和 bf16 中转 buffer（2B），
    // 等效 6 字节/元素；其它 dtype（fp16/bf16）保持 2 字节/元素。
    int64_t bytesPerElem = (inDtype == DT_FLOAT) ? (BYTES_OF_INPUT_TYPE_FP32 + BYTES_OF_BF16_SCRATCH) :
                                                   BYTES_OF_INPUT_TYPE;
    int64_t maxUbAvailable = (tilingParam.ubSize - RESERVED_UB_SIZE) / N_BUFFER / bytesPerElem / EXIST_NODE_NUM /
                             tilingParam.blockSize;
    // 计算ubFactor
    SpliteUB(tilingParam, maxUbAvailable);
    int64_t multiples = 1;
    int64_t spliteCoreData = (tilingParam.ubDim == NEW_SHAPE_INDEX_TWO) ? tilingParam.uo * tilingParam.preAxisSize :
                                                                          tilingParam.uo;

    OP_CHECK_IF(spliteCoreData == 0, OP_LOGE(context->GetNodeName(), "The divisor cannot be zero, please check"),
                return ge::GRAPH_FAILED);

    if (spliteCoreData <= tilingParam.totalCoreNum / DIGIT_TWO && !IsMinUbFactor(tilingParam, multiples)) {
        if (tilingParam.totalCoreNum % spliteCoreData == 0) {
            maxUbAvailable /= min(multiples, tilingParam.totalCoreNum / spliteCoreData);
        } else {
            maxUbAvailable = (multiples <= (tilingParam.totalCoreNum / spliteCoreData)) ?
                                 maxUbAvailable / multiples :
                                 maxUbAvailable * spliteCoreData / tilingParam.totalCoreNum;
        }
        SpliteUB(tilingParam, maxUbAvailable);
        spliteCoreData = (tilingParam.ubDim == NEW_SHAPE_INDEX_TWO) ? tilingParam.uo * tilingParam.preAxisSize :
                                                                      tilingParam.uo;
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

static int64_t GetNotAxisGroupPerUb(const DynamicMxQuantTilingParam& tilingParam)
{
    int64_t groupPerUb = 0;
    // fp32 输入需要在 UB 中额外保留一份 bf16 中转 buffer（每元素 2 字节）。
    int64_t bytesInputAndScratch = tilingParam.inputDtypeSize + (tilingParam.isFp32Input ? BYTES_OF_BF16_SCRATCH : 0);
    uint64_t xUbSize = tilingParam.nAlignNum * tilingParam.blockSize * BLOCK_PER_GROUP * bytesInputAndScratch;
    uint64_t yUbSize = tilingParam.nAlignNum * tilingParam.blockSize * BLOCK_PER_GROUP * MAX_BYTES_OF_OUTPUT_TYPE;
    uint64_t scaleUbSize = BLOCK_PER_GROUP * tilingParam.nAlignNum * UINT8_BYTES_SIZE; // scale is fp8

    if (tilingParam.postAxisSize > tilingParam.nAlignNum) {
        uint64_t tmpBufferUbSize = BLOCK_PER_GROUP * tilingParam.nAlignNum * UINT16_BYTES_SIZE; // for 1 / scale
        uint64_t tmpScaleBufferUbSize = BLOCK_PER_GROUP * tilingParam.nAlignNum *
                                        UINT8_BYTES_SIZE; // for scale interleave
        groupPerUb = (tilingParam.ubSize - RESERVED_UB_SIZE) / N_BUFFER /
                     (xUbSize + yUbSize + scaleUbSize + tmpBufferUbSize + tmpScaleBufferUbSize);
    } else {
        groupPerUb = (tilingParam.ubSize - RESERVED_UB_SIZE) / N_BUFFER / (xUbSize + yUbSize + scaleUbSize);
    }
    return groupPerUb;
}
// capable to be optimized
static bool IsOptForNotLastQuantAxis(const gert::TilingContext* context, const DynamicMxQuantTilingParam& tilingParam)
{
    OP_LOGD(context->GetNodeName(), "Start to check input possible to be optimized.");
    if (tilingParam.isTailAxis) {
        OP_LOGD(context->GetNodeName(), "Tail axis quantization is not supported in this optimization branch.");
        return false;
    }
    if (tilingParam.postAxisSize <= tilingParam.nAlignNum) { // 小尾轴
        // 小尾轴时，not_tail_axis_opti会多个轴一起处理，性能更好
        if (tilingParam.blockSize != N_ALIGN32) {
            OP_LOGD(context->GetNodeName(),
                    "small tail optimization template blockSize[%ld] not supported in this optimization branch, only "
                    "support 32",
                    tilingParam.blockSize);
            return false;
        }
    }

    int64_t groupPerUb = GetNotAxisGroupPerUb(tilingParam);
    if (groupPerUb == 0) {
        OP_LOGD(context->GetNodeName(), "blockSize too large to fit the optimal template.");
        return false;
    }

    return true;
}

static bool IsTailAxisAndBlockSize32(DynamicMxQuantTilingParam& tilingParam)
{
    if (tilingParam.isTailAxis && tilingParam.isBlockSize32) {
        tilingParam.blockSize = ATTR_BLOCK_SIZE;
        return true;
    }
    return false;
}

inline static void SetTilingData(DynamicMxQuantTilingData& tilingData, const DynamicMxQuantTilingParam& tilingParam)
{
    tilingData.totalCoreNum = tilingParam.totalCoreNum;
    tilingData.usedCoreNum = tilingParam.usedCoreNum;
    tilingData.uo = tilingParam.uo;
    tilingData.ubDim = tilingParam.ubDim;
    tilingData.ubFactor = tilingParam.ubFactor;
    tilingData.tailUbFactor = tilingParam.tailUbFactor;
    tilingData.blockFactor = tilingParam.blockFactor;
    tilingData.tailBlockFactor = tilingParam.tailBlockFactor;
    tilingData.tilingKey = tilingParam.tilingKey;
    tilingData.roundMode = tilingParam.roundMode;
    tilingData.dstType = tilingParam.dstType;
    tilingData.blockSize = tilingParam.blockSize;
    tilingData.scaleAlg = tilingParam.scaleAlg;
    tilingData.tailBlockSize = tilingParam.tailBlockSize;
    tilingData.blockSizeNumInAxis = tilingParam.blockSizeNumInAxis;
    int64_t isPad = tilingParam.isPad ? 1 : 0;
    tilingData.isPad = isPad;
    int64_t preAxisSize = tilingParam.preAxisSize / tilingParam.blockSizeNumInAxis;
    tilingData.preAxisSize = preAxisSize;
    tilingData.postAxisSize = tilingParam.postAxisSize;
    tilingData.mxScaleSize = tilingParam.mxScaleSize;
    tilingData.isTailAxis = static_cast<int64_t>(tilingParam.isTailAxis);
    tilingData.calcMode = tilingParam.calcMode;
    tilingData.subNumForScale = tilingParam.subNumForScale;
    tilingData.subNumForFP16Scale = tilingParam.subNumForFP16Scale;
    tilingData.dstTypeMax = tilingParam.dstTypeMax;
    tilingData.invDstTypeMax = tilingParam.invDstTypeMax;
    tilingData.maxLowBound = tilingParam.maxLowBound;
}

ge::graphStatus Tiling4DynamicMxQuant(gert::TilingContext* context)
{
    OP_LOGD(context->GetNodeName(), "Tiling4DynamicMxQuant running begin.");

    DynamicMxQuantTilingParam tilingParam;

    OP_CHECK_IF(CheckDtype(context) != ge::GRAPH_SUCCESS,
                OP_LOGE(context->GetNodeName(), "The data type check failed."), return ge::GRAPH_FAILED);

    OP_CHECK_IF(GetAttr(context, tilingParam) != ge::GRAPH_SUCCESS,
                OP_LOGE(context->GetNodeName(), "The attr get failed."), return ge::GRAPH_FAILED);

    OP_CHECK_IF(CheckShape(context, tilingParam) != ge::GRAPH_SUCCESS,
                OP_LOGE(context->GetNodeName(), "The shape check failed."), return ge::GRAPH_FAILED);

    auto platformInfo = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    tilingParam.totalCoreNum = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF((tilingParam.totalCoreNum <= 0), OP_LOGE(context->GetNodeName(), "Failed to core num."),
                return ge::GRAPH_FAILED);
    uint64_t ubSize;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    tilingParam.ubSize = static_cast<int64_t>(ubSize);

    OP_CHECK_IF((tilingParam.ubSize <= 0), OP_LOGE(context->GetNodeName(), "Failed to get ub size."),
                return ge::GRAPH_FAILED);
    tilingParam.vfLen = Ops::Base::GetVRegSize(context);
    tilingParam.workspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();

    auto inputXPtrEarly = context->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputXPtrEarly);
    auto inDtypeEarly = inputXPtrEarly->GetDataType();
    tilingParam.isFp32Input = (inDtypeEarly == DT_FLOAT);
    tilingParam.inputDtypeSize = tilingParam.isFp32Input ? DIGIT_FOUR : DIGIT_TWO;

    OP_CHECK_IF(BaseCalc(context, tilingParam) != ge::GRAPH_SUCCESS,
                OP_LOGE(context->GetNodeName(), "The base calculation failed."), return ge::GRAPH_FAILED);

    // 尾轴+block32 场景优先判断，避免被 IsOptForNotLastQuantAxis 误拦截
    // （IsOptForNotLastQuantAxis 不检查 isTailAxis，会导致尾轴 case 路由到 optimize 路径，
    //  产生 isTailAxisQuant=TAIL_AXIS + optiMode=SMALL_TAIL_AXIS_OPTI 的无效 tilingKey）
    bool IsTailAnd32 = IsTailAxisAndBlockSize32(tilingParam);
    if (IsTailAnd32) {
        DynamicMxQuantTailAxisTiling tailAxisTiling(context, tilingParam);
        return tailAxisTiling.DoTiling();
    }

    bool isOptimizable = IsOptForNotLastQuantAxis(context, tilingParam);
    // 进入性能优化模板判断
    if (isOptimizable) {
        int64_t x = GetNotAxisGroupPerUb(tilingParam);
        tilingParam.groupPerUb = x;
        DynamicMxQuantOptimzieTiling regbaseTiling(context, tilingParam);
        return regbaseTiling.DoTiling();
    }

    OP_CHECK_IF(DoTiling(context, tilingParam) != ge::GRAPH_SUCCESS,
                OP_LOGE(context->GetNodeName(), "Dotiling failed."), return ge::GRAPH_FAILED);

    DynamicMxQuantTilingData tilingData;
    SetTilingData(tilingData, tilingParam);

    OP_CHECK_IF(DynamicMxQuantSetTilingData(context, tilingData) != ge::GRAPH_SUCCESS,
                OP_LOGE(context->GetNodeName(), "DynamicMxQuantSetTilingData set tiling data fail."),
                return ge::GRAPH_FAILED);

    if (tilingParam.blockSizeNumInAxis % DIGIT_TWO) {
        context->SetScheduleMode(1); // 设置为batch mode模式，所有核同时启动
    }

    context->SetBlockDim(tilingData.usedCoreNum);
    context->SetTilingKey(tilingData.tilingKey);
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

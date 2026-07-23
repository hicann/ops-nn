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
 * \file grouped_dynamic_mx_quant_tiling_arch35.cpp
 * \brief
 */

#include "grouped_dynamic_mx_quant_tiling_arch35.h"
#include "quant/grouped_dynamic_mx_quant/op_kernel/arch35/grouped_dynamic_mx_quant_tilingdata.h"
#include "quant/grouped_dynamic_mx_quant/op_kernel/arch35/grouped_dynamic_mx_quant_struct.h"
#include <cmath>
#include "op_common/op_host/util/platform_util.h"
#include "platform/platform_info.h"
#include "tiling/platform/platform_ascendc.h"
#include "platform/platform_infos_def.h"
#include "error_util.h"
#include "graph/utils/type_utils.h"
#include "util/math_util.h"
#include "register/op_impl_registry.h"
#include "register/tilingdata_base.h"

using namespace std;
using namespace ge;
using namespace Ops::Base;
using namespace GroupedDynamicMxQuantOp;

namespace optiling {
constexpr int64_t INDEX_ATTR_ROUND_MODE = 0;
constexpr int64_t INDEX_ATTR_DST_DTYPE = 1;
constexpr int64_t INDEX_ATTR_BLOCK_SIZE = 2;
constexpr int64_t INDEX_ATTR_SCALE_ALG = 3;
constexpr int64_t INDEX_ATTR_DST_TYPE_MAX = 4;
constexpr int64_t BYTES_OF_INPUT_TYPE = 2;
constexpr int64_t DIGIT_ZERO = 0;
constexpr float DIGIT_ZERO_FLOAT = 0.0;
constexpr int64_t DIGIT_ONE = 1;
constexpr int64_t DIGIT_TWO = 2;
constexpr int64_t DIGIT_TEN = 10;
constexpr int64_t N_BUFFER = 2;
constexpr int64_t EXIST_NODE_NUM = 3;
constexpr int64_t ATTR_BLOCK_SIZE = 32;
constexpr int64_t SCALE_DIM_NUM = 3;
constexpr size_t WORKSPACE_SIZE = 32;

constexpr float FP4E2M1_MAX = 6.0;
constexpr float FP4E1M2_MAX = 1.75;
constexpr float FP8_E4M3FN_MAX = 448;
constexpr float FP8_E5M2_MAX = 57344;

const std::set<ge::DataType> INPUT_SUPPORT_DTYPE_SET = {ge::DT_FLOAT16, ge::DT_BF16};
const std::set<ge::DataType> GROUPIDX_SUPPORT_DTYPE_SET = {ge::DT_INT32};
const std::set<ge::DataType> Y_SUPPORT_DTYPE_FP4_SET = {ge::DT_FLOAT4_E2M1, ge::DT_FLOAT4_E1M2};
const std::set<ge::DataType> Y_SUPPORT_DTYPE_FP8_SET = {ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E5M2};
const std::set<ge::DataType> Y_SUPPORT_DTYPE_SET = {ge::DT_FLOAT4_E2M1, ge::DT_FLOAT4_E1M2, ge::DT_FLOAT8_E4M3FN,
                                                    ge::DT_FLOAT8_E5M2};
const std::set<ge::DataType> OUTPUT_SUPPORT_DTYPE_SET = {ge::DT_FLOAT8_E8M0};

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

static ge::graphStatus GetAttr(const gert::TilingContext* context, GroupedDynamicMxQuantTilingParam& tilingParam)
{
    OP_LOGD(context, "GetAttr begin.");
    auto* attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);

    auto* attrRoundMode = attrs->GetAttrPointer<char>(INDEX_ATTR_ROUND_MODE);
    OP_CHECK_NULL_WITH_CONTEXT(context, attrRoundMode);
    std::string roundModeStr = attrRoundMode;
    RoundModeList roundMode = GetRoundMode(roundModeStr);

    OP_CHECK_IF((roundMode == RoundModeList::MODE_UNDEFINED),
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "round_mode", roundModeStr,
                                                      "The value of round_mode must be [rint, round, floor]"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF((Y_SUPPORT_DTYPE_FP8_SET.count(tilingParam.outDtype) != 0 && roundMode != RoundModeList::MODE_RINT),
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                    context->GetNodeName(), "round_mode", roundModeStr,
                    "If the dtype of output y is FLOAT8_E4M3FN/FLOAT8_E5M2, parameter round_mode must be rint"),
                return ge::GRAPH_FAILED);

    tilingParam.roundMode = static_cast<int64_t>(roundMode);

    auto* attrDstType = attrs->GetAttrPointer<int64_t>(INDEX_ATTR_DST_DTYPE);
    OP_CHECK_NULL_WITH_CONTEXT(context, attrDstType);
    int checkDstType = static_cast<int>(*attrDstType);

    OP_CHECK_IF((tilingParam.outDtype == ge::DT_FLOAT8_E4M3FN && checkDstType != 36) ||
                    (tilingParam.outDtype == ge::DT_FLOAT8_E5M2 && checkDstType != 35) ||
                    (tilingParam.outDtype == ge::DT_FLOAT4_E2M1 && checkDstType != 40) ||
                    (tilingParam.outDtype == ge::DT_FLOAT4_E1M2 && checkDstType != 41),
                OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(
                    context->GetNodeName(), "y, dst_type",
                    ge::TypeUtils::DataTypeToSerialString(tilingParam.outDtype) + ", " + std::to_string(checkDstType),
                    "The dtypes of y and dst_type must be the same"),
                return ge::GRAPH_FAILED);

    auto* attrBlockSize = attrs->GetAttrPointer<int64_t>(INDEX_ATTR_BLOCK_SIZE);
    OP_CHECK_NULL_WITH_CONTEXT(context, attrBlockSize);
    tilingParam.blockSize = static_cast<int64_t>(*attrBlockSize);

    OP_CHECK_IF(
        tilingParam.blockSize != ATTR_BLOCK_SIZE,
        OP_LOGE_FOR_INVALID_VALUE(context->GetNodeName(), "block_size", std::to_string(tilingParam.blockSize), "32"),
        return ge::GRAPH_FAILED);

    auto* attrScaleAlg = attrs->GetAttrPointer<int64_t>(INDEX_ATTR_SCALE_ALG);
    OP_CHECK_NULL_WITH_CONTEXT(context, attrScaleAlg);
    tilingParam.scaleAlg = static_cast<int64_t>(*attrScaleAlg);

    OP_CHECK_IF(
        tilingParam.scaleAlg < 0 || tilingParam.scaleAlg > 2,
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "scale_alg", std::to_string(tilingParam.scaleAlg),
                                              "The value of scale_alg must be [0, 1, 2]"),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF(tilingParam.scaleAlg == 1 && Y_SUPPORT_DTYPE_FP4_SET.count(tilingParam.outDtype) != 0,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                    context->GetNodeName(), "scale_alg", std::to_string(tilingParam.scaleAlg),
                    "If the dtype of output y is FLOAT4_E2M1/FLOAT_E1M2, parameter scale_alg must be 0 or 2"),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(tilingParam.scaleAlg == 2 && Y_SUPPORT_DTYPE_FP8_SET.count(tilingParam.outDtype) != 0,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                    context->GetNodeName(), "scale_alg", std::to_string(tilingParam.scaleAlg),
                    "If the dtype of output y is FLOAT8_E4M3FN/FLOAT8_E5M2, parameter scale_alg must be 0 or 1"),
                return ge::GRAPH_FAILED);

    auto* attrDstTypeMax = attrs->GetAttrPointer<float>(INDEX_ATTR_DST_TYPE_MAX);
    OP_CHECK_NULL_WITH_CONTEXT(context, attrDstTypeMax);
    tilingParam.dstTypeMax = static_cast<float>(*attrDstTypeMax);

    OP_CHECK_IF((tilingParam.dstTypeMax < 0 || (tilingParam.dstTypeMax > 0 && tilingParam.dstTypeMax < 6) ||
                 tilingParam.dstTypeMax > 12) &&
                    tilingParam.outDtype == ge::DT_FLOAT4_E2M1,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "dst_type_max",
                                                      std::to_string(tilingParam.dstTypeMax),
                                                      "If the dtype of output y is FLOAT4_E2M1, parameter dst_type_max "
                                                      "must be within the range [6, 12] or equal to 0"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF((tilingParam.dstTypeMax < 0 || (tilingParam.dstTypeMax > 0 && tilingParam.dstTypeMax < 1.75) ||
                 tilingParam.dstTypeMax > 3.5) &&
                    tilingParam.outDtype == ge::DT_FLOAT4_E1M2,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "dst_type_max",
                                                      std::to_string(tilingParam.dstTypeMax),
                                                      "If the dtype of output y is FLOAT4_E1M2, parameter dst_type_max "
                                                      "must be within the range [1.75, 3.5] or equal to 0"),
                return ge::GRAPH_FAILED);

    // 当dstTypeMax=0时，默认使用目标数据类型最大值，FP4E2M1最大值6、FP4E1M2最大值1.75、FP8_E4M3FN最大值448、FP8_E5M2最大值57344
    if (!Ops::Base::IsFloatEqual(tilingParam.dstTypeMax, 0.0f)) {
        tilingParam.invDstTypeMax = 1.0 / tilingParam.dstTypeMax;
    } else {
        if (tilingParam.outDtype == ge::DT_FLOAT4_E2M1) {
            tilingParam.invDstTypeMax = 1.0 / FP4E2M1_MAX;
        } else if (tilingParam.outDtype == ge::DT_FLOAT4_E1M2) {
            tilingParam.invDstTypeMax = 1.0 / FP4E1M2_MAX;
        } else if (tilingParam.outDtype == ge::DT_FLOAT8_E4M3FN) {
            tilingParam.invDstTypeMax = 1.0 / FP8_E4M3FN_MAX;
        } else if (tilingParam.outDtype == ge::DT_FLOAT8_E5M2) {
            tilingParam.invDstTypeMax = 1.0 / FP8_E5M2_MAX;
        }
    }

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckDtype(const gert::TilingContext* context, GroupedDynamicMxQuantTilingParam& tilingParam)
{
    OP_LOGD(context, "CheckDtype begin.");
    auto inputXPtr = context->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputXPtr);
    tilingParam.inDtype = inputXPtr->GetDataType();
    OP_CHECK_IF(INPUT_SUPPORT_DTYPE_SET.count(tilingParam.inDtype) == 0,
                OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(context->GetNodeName(), "x",
                                                      ge::TypeUtils::DataTypeToSerialString(tilingParam.inDtype),
                                                      "The dtype of x must be DT_FLOAT16 or DT_BF16"),
                return ge::GRAPH_FAILED);

    auto groupIndexPtr = context->GetInputDesc(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, groupIndexPtr);
    auto groupIndexDtype = groupIndexPtr->GetDataType();
    OP_CHECK_IF(GROUPIDX_SUPPORT_DTYPE_SET.count(groupIndexDtype) == 0,
                OP_LOGE_FOR_INVALID_DTYPE(context->GetNodeName(), "group_index",
                                          ge::TypeUtils::DataTypeToSerialString(groupIndexDtype), "DT_INT32"),
                return ge::GRAPH_FAILED);

    auto outputYPtr = context->GetOutputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, outputYPtr);
    tilingParam.outDtype = outputYPtr->GetDataType();
    OP_CHECK_IF(Y_SUPPORT_DTYPE_SET.count(tilingParam.outDtype) == 0,
                OP_LOGE_FOR_INVALID_DTYPE(context->GetNodeName(), "y",
                                          ge::TypeUtils::DataTypeToSerialString(tilingParam.outDtype),
                                          "[DT_FLOAT4_E2M1, DT_FLOAT4_E1M2, DT_FLOAT8_E4M3FN, DT_FLOAT8_E5M2]"),
                return ge::GRAPH_FAILED);

    auto outputMxScalePtr = context->GetOutputDesc(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, outputMxScalePtr);
    auto scaleDtype = outputMxScalePtr->GetDataType();
    OP_CHECK_IF(OUTPUT_SUPPORT_DTYPE_SET.count(scaleDtype) == 0,
                OP_LOGE_FOR_INVALID_DTYPE(context->GetNodeName(), "mxscale",
                                          ge::TypeUtils::DataTypeToSerialString(scaleDtype), "DT_FLOAT8_E8M0"),
                return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckShape(const gert::TilingContext* context, GroupedDynamicMxQuantTilingParam& tilingParam)
{
    OP_LOGD(context, "CheckShape begin.");
    auto xShapePtr = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, xShapePtr);
    auto xShape = xShapePtr->GetStorageShape();

    auto groupIndexShapePtr = context->GetInputShape(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, groupIndexShapePtr);
    auto groupIndexShape = groupIndexShapePtr->GetStorageShape();

    auto yShapePtr = context->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, yShapePtr);
    auto yShape = yShapePtr->GetStorageShape();

    auto mxScaleShapePtr = context->GetOutputShape(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, mxScaleShapePtr);
    auto mxScaleShape = mxScaleShapePtr->GetStorageShape();

    OP_CHECK_IF(xShape != yShape,
                OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(context->GetNodeName(), "x, y",
                                                       Ops::Base::ToString(xShape) + ", " + Ops::Base::ToString(yShape),
                                                       "The shapes of x and y must be the same"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(xShape.GetDimNum() != 2,
                OP_LOGE_FOR_INVALID_SHAPEDIM(context->GetNodeName(), "x", std::to_string(xShape.GetDimNum()), "2"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(groupIndexShape.GetDimNum() != 1,
                OP_LOGE_FOR_INVALID_SHAPEDIM(context->GetNodeName(), "group_index",
                                             std::to_string(groupIndexShape.GetDimNum()), "1"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        mxScaleShape.GetDimNum() != SCALE_DIM_NUM,
        OP_LOGE_FOR_INVALID_SHAPEDIM(context->GetNodeName(), "mxscale", std::to_string(mxScaleShape.GetDimNum()), "3"),
        return ge::GRAPH_FAILED);

    tilingParam.groupNum = groupIndexShape.GetDim(0);
    tilingParam.colSize = xShape.GetDim(0);
    tilingParam.rowSize = xShape.GetDim(1);

    OP_CHECK_IF(tilingParam.rowSize % DIGIT_TWO != 0 && Y_SUPPORT_DTYPE_FP4_SET.count(tilingParam.outDtype) != 0,
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                    context->GetNodeName(), "x", Ops::Base::ToString(xShape),
                    "When the yDtype is FLOAT4_E2M1 or FLOAT4_E1M2, the tail axis of x must be an even number"),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(tilingParam.groupNum == 0,
                OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(context->GetNodeName(), "group_index", "0",
                                                          "group_index does not support empty tensor"),
                return ge::GRAPH_FAILED);

    xShape.SetDim(0, tilingParam.colSize / (tilingParam.blockSize * DIGIT_TWO) + tilingParam.groupNum);
    xShape.SetDim(1, tilingParam.rowSize * DIGIT_TWO);
    OP_CHECK_IF(
        mxScaleShape[0] != xShape[0] || mxScaleShape[1] != tilingParam.rowSize ||
            mxScaleShape[SCALE_DIM_NUM - 1] != DIGIT_TWO,
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
            context->GetNodeName(), "x, mxscale",
            Ops::Base::ToString(xShape) + ", " + Ops::Base::ToString(mxScaleShape),
            "The shape of mxscale must be [x.shape[0] / (2 * block_size) + group_index.shape[0], x.shape[1], 2]"),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus GetPlatInfo(const gert::TilingContext* context, GroupedDynamicMxQuantTilingParam& tilingParam)
{
    OP_LOGD(context, "GetPlatInfo begin.");
    auto platformInfo = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);

    tilingParam.totalCoreNum = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF((tilingParam.totalCoreNum <= 0),
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "totalCoreNum",
                                                      std::to_string(tilingParam.totalCoreNum),
                                                      "The value of totalCoreNum must be greater than 0"),
                return ge::GRAPH_FAILED);

    uint64_t ubSize;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);

    tilingParam.ubSize = static_cast<int64_t>(ubSize);
    OP_CHECK_IF(
        (tilingParam.ubSize <= 0),
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "ubSize", std::to_string(tilingParam.ubSize),
                                              "The value of ubsize must be greater than 0"),
        return ge::GRAPH_FAILED);

    tilingParam.vfLen = Ops::Base::GetVRegSize(context);
    OP_CHECK_IF(
        (tilingParam.vfLen <= 0),
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "vfLen", std::to_string(tilingParam.vfLen),
                                              "The value of vfLen must be greater than 0"),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus DoTiling(const gert::TilingContext* context, GroupedDynamicMxQuantTilingParam& tilingParam)
{
    OP_LOGD(context, "DoTiling begin.");

    tilingParam.blockRowSize = static_cast<int64_t>(tilingParam.vfLen / BYTES_OF_INPUT_TYPE);
    tilingParam.blockRowCount = Ops::Base::CeilDiv(tilingParam.rowSize, tilingParam.blockRowSize);
    tilingParam.blockRowTailSize = tilingParam.rowSize - (tilingParam.blockRowCount - 1) * tilingParam.blockRowSize;

    tilingParam.blockColSize = tilingParam.blockSize * DIGIT_TWO;
    tilingParam.usedCoreNum = tilingParam.totalCoreNum;

    return ge::GRAPH_SUCCESS;
}

inline static ge::graphStatus SetTilingKeyParam(gert::TilingContext* context,
                                                GroupedDynamicMxQuantTilingParam& tilingParam)
{
    uint64_t scale_alg = static_cast<uint64_t>(tilingParam.scaleAlg);

    uint64_t dst_type_max = TPL_DST_TYPE_MAX_0;
    if ((Ops::Base::IsFloatEqual(tilingParam.dstTypeMax, 0.0f) && tilingParam.outDtype == ge::DT_FLOAT4_E2M1) ||
        Ops::Base::IsFloatEqual(tilingParam.dstTypeMax, 6.0f)) {
        dst_type_max = TPL_DST_TYPE_MAX_1;
    } else if (Ops::Base::IsFloatEqual(tilingParam.dstTypeMax, 7.0f)) {
        dst_type_max = TPL_DST_TYPE_MAX_2;
    } else if (Ops::Base::IsFloatEqual(tilingParam.dstTypeMax, 1.875f)) {
        dst_type_max = TPL_DST_TYPE_MAX_3;
    }

    uint64_t dst_type = TPL_DST_TYPE_0;
    if (tilingParam.outDtype == ge::DT_FLOAT4_E2M1) {
        dst_type = TPL_DST_TYPE_1;
    } else if (tilingParam.outDtype == ge::DT_FLOAT4_E1M2) {
        dst_type = TPL_DST_TYPE_2;
    }
    uint64_t round_mode = static_cast<uint64_t>(tilingParam.roundMode);
    tilingParam.tilingKey = GET_TPL_TILING_KEY(scale_alg, dst_type_max, dst_type, round_mode);
    context->SetTilingKey(tilingParam.tilingKey);

    return ge::GRAPH_SUCCESS;
}

inline static ge::graphStatus SetTilingData(const GroupedDynamicMxQuantTilingParam& tilingParam,
                                            GroupedDynamicMxQuantTilingData* tilingData)
{
    tilingData->totalCoreNum = tilingParam.totalCoreNum;
    tilingData->usedCoreNum = tilingParam.usedCoreNum;
    tilingData->rowSize = tilingParam.rowSize;
    tilingData->colSize = tilingParam.colSize;
    tilingData->blockRowSize = tilingParam.blockRowSize;
    tilingData->blockColSize = tilingParam.blockColSize;
    tilingData->blockRowTailSize = tilingParam.blockRowTailSize;
    tilingData->blockRowCount = tilingParam.blockRowCount;
    tilingData->groupNum = tilingParam.groupNum;
    tilingData->invDstTypeMax = tilingParam.invDstTypeMax;
    tilingData->tilingKey = tilingParam.tilingKey;

    return ge::GRAPH_SUCCESS;
}

inline static void PrintTilingData(const gert::TilingContext* context, GroupedDynamicMxQuantTilingData* tilingData)
{
    OP_LOGI(context, "tilingData is totalCoreNum:%ld, usedCoreNum:%ld, rowSize:%ld, colSize:%ld, blockRowSize:%ld, \
        blockColSize:%ld, blockRowTailSize:%ld, blockRowCount:%ld, groupNum:%ld, invDstTypeMax:%f, tilingKey:%ld",
            tilingData->totalCoreNum, tilingData->usedCoreNum, tilingData->rowSize, tilingData->colSize,
            tilingData->blockRowSize, tilingData->blockColSize, tilingData->blockRowTailSize, tilingData->blockRowCount,
            tilingData->groupNum, tilingData->invDstTypeMax, tilingData->tilingKey);
}

ge::graphStatus Tiling4GroupedDynamicMxQuant(gert::TilingContext* context)
{
    OP_LOGD(context, "Tiling4GroupedDynamicMxQuant running begin.");

    GroupedDynamicMxQuantTilingParam tilingParam;

    OP_CHECK_IF(CheckDtype(context, tilingParam) != ge::GRAPH_SUCCESS, OP_LOGE(context, "The data type check failed."),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(GetAttr(context, tilingParam) != ge::GRAPH_SUCCESS, OP_LOGE(context, "The attr get failed."),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(CheckShape(context, tilingParam) != ge::GRAPH_SUCCESS, OP_LOGE(context, "The shape check failed."),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(GetPlatInfo(context, tilingParam) != ge::GRAPH_SUCCESS, OP_LOGE(context, "GetPlatInfo failed."),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(DoTiling(context, tilingParam) != ge::GRAPH_SUCCESS, OP_LOGE(context, "DoTiling failed."),
                return ge::GRAPH_FAILED);

    auto* tilingData = context->GetTilingData<GroupedDynamicMxQuantTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tilingData);

    SetTilingKeyParam(context, tilingParam);

    OP_CHECK_IF(SetTilingData(tilingParam, tilingData) != ge::GRAPH_SUCCESS,
                OP_LOGE(context, "SetTilingData set tiling data fail."), return ge::GRAPH_FAILED);

    context->SetBlockDim(tilingData->usedCoreNum);
    size_t* workspaces = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, workspaces);
    workspaces[0] = WORKSPACE_SIZE;
    PrintTilingData(context, tilingData);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingPrepare4GroupedDynamicMxQuant(gert::TilingParseContext* context)
{
    OP_LOGD(context, "TilingPrepare4GroupedDynamicMxQuant entering.");
    return ge::GRAPH_SUCCESS;
}

// register tiling interface of the GroupedDynamicMxQuant op.
IMPL_OP_OPTILING(GroupedDynamicMxQuant)
    .Tiling(Tiling4GroupedDynamicMxQuant)
    .TilingParse<GroupedDynamicMxQuantCompileInfo>(TilingPrepare4GroupedDynamicMxQuant);
} // namespace optiling

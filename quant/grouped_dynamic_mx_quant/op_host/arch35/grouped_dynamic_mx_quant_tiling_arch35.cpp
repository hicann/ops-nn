/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
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
#include <cmath>
#include "op_host/util/platform_util.h"
#include "graph/utils/type_utils.h"
#include "util/math_util.h"
#include "register/op_impl_registry.h"

using namespace std;
using namespace ge;
using namespace Ops::Base;

namespace optiling {
constexpr int64_t INDEX_ATTR_ROUND_MODE = 0;
constexpr int64_t INDEX_ATTR_DST_DTYPE = 1;
constexpr int64_t INDEX_ATTR_BLOCK_SIZE = 2;
constexpr int64_t BYTES_OF_INPUT_TYPE = 2;
constexpr int64_t DIGIT_TWO = 2;
constexpr int64_t DIGIT_TEN = 10;
constexpr int64_t N_BUFFER = 2;
constexpr int64_t EXIST_NODE_NUM = 3;
constexpr int64_t ATTR_BLOCK_SIZE = 32;
constexpr int64_t SCALE_DIM_NUM = 3;
constexpr size_t WORKSPACE_SIZE = 32;
const std::set<ge::DataType> INPUT_SUPPORT_DTYPE_SET = { ge::DT_FLOAT16, ge::DT_BF16 };
const std::set<ge::DataType> GROUPIDX_SUPPORT_DTYPE_SET = { ge::DT_INT32 };
const std::set<ge::DataType> Y_SUPPORT_DTYPE_SET = { ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E5M2 };
const std::set<ge::DataType> OUTPUT_SUPPORT_DTYPE_SET = { ge::DT_FLOAT8_E8M0 };

static ge::graphStatus GetAttr(const gert::TilingContext *context, GroupedDynamicMxQuantTilingParam &tilingParam)
{
    OP_LOGD(context->GetNodeName(), "GetAttr begin.");
    auto *attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    
    auto *attrRoundMode = attrs->GetAttrPointer<char>(INDEX_ATTR_ROUND_MODE);
    OP_CHECK_NULL_WITH_CONTEXT(context, attrRoundMode);
    std::string roundModeStr = attrRoundMode;
    OP_CHECK_IF((roundModeStr != "rint"),
        OP_LOGE(context->GetNodeName(), 
        "invalid round_mode:%s; round_mode only supports rint", roundModeStr.c_str()),
        return ge::GRAPH_FAILED);

    auto *attrDstType = attrs->GetAttrPointer<int64_t>(INDEX_ATTR_DST_DTYPE);
    OP_CHECK_NULL_WITH_CONTEXT(context, attrDstType);
    int checkDstType = static_cast<int>(*attrDstType);
    OP_CHECK_IF((tilingParam.outDtype == ge::DT_FLOAT8_E4M3FN && checkDstType != 36) ||
        (tilingParam.outDtype == ge::DT_FLOAT8_E5M2 && checkDstType != 35),
        OP_LOGE(context->GetNodeName(),
        "y's data type and dst_type is not corresponded, y's data type: FLOAT8_E4M3FN/FLOAT8_E5M2 correspond to dst_type: 36/35."),
        return ge::GRAPH_FAILED);

    auto *attrBlockSize = attrs->GetAttrPointer<int64_t>(INDEX_ATTR_BLOCK_SIZE);
    OP_CHECK_NULL_WITH_CONTEXT(context, attrBlockSize);
    tilingParam.blockSize = static_cast<int64_t>(*attrBlockSize);
    OP_CHECK_IF(tilingParam.blockSize != ATTR_BLOCK_SIZE,
        OP_LOGE(context->GetNodeName(),
        "The blocksize only supports 32."),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckDtype(const gert::TilingContext *context, GroupedDynamicMxQuantTilingParam &tilingParam)
{
    OP_LOGD(context->GetNodeName(), "CheckDtype begin.");
    auto inputXPtr = context->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputXPtr);
    tilingParam.inDtype = inputXPtr->GetDataType();
    OP_CHECK_IF(INPUT_SUPPORT_DTYPE_SET.count(tilingParam.inDtype) == 0,
        OP_LOGE(context->GetNodeName(), 
            "Input x's data type is [%s], current only supports FLOAT16/BFLOAT16.",
            ge::TypeUtils::DataTypeToSerialString(tilingParam.inDtype).c_str()),
        return ge::GRAPH_FAILED);

    auto groupIndexPtr = context->GetInputDesc(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, groupIndexPtr);
    auto groupIndexDtype = groupIndexPtr->GetDataType();
    OP_CHECK_IF(GROUPIDX_SUPPORT_DTYPE_SET.count(groupIndexDtype) == 0,
        OP_LOGE(context->GetNodeName(),
            "group_index's data type is [%s], current only supports Int32.",
            ge::TypeUtils::DataTypeToSerialString(groupIndexDtype).c_str()),
        return ge::GRAPH_FAILED);

    auto outputYPtr = context->GetOutputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, outputYPtr);
    tilingParam.outDtype = outputYPtr->GetDataType();
    OP_CHECK_IF(Y_SUPPORT_DTYPE_SET.count(tilingParam.outDtype) == 0,
        OP_LOGE(context->GetNodeName(),
            "Output y's data type is [%s], current only supports FLOAT8_E4M3FN/FLOAT8_E5M2.",
            ge::TypeUtils::DataTypeToSerialString(tilingParam.outDtype).c_str()),
        return ge::GRAPH_FAILED);

    auto outputMxScalePtr = context->GetOutputDesc(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, outputMxScalePtr);
    auto scaleDtype = outputMxScalePtr->GetDataType();
    OP_CHECK_IF(OUTPUT_SUPPORT_DTYPE_SET.count(scaleDtype) == 0,
        OP_LOGE(context->GetNodeName(),
            "Input mxscale's data type is [%s], current only supports FLOAT8_E8M0.",
            ge::TypeUtils::DataTypeToSerialString(scaleDtype).c_str()),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckShape(const gert::TilingContext *context, GroupedDynamicMxQuantTilingParam &tilingParam)
{
    OP_LOGD(context->GetNodeName(), "CheckShape begin.");
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
        OP_LOGE(context->GetNodeName(),
        "The shape of output y must be same with shape of input x."),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(xShape.GetDimNum() != 2,
        OP_LOGE(context->GetNodeName(),
        "The shape of input x must be 2-D."),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(groupIndexShape.GetDimNum() != 1,
        OP_LOGE(context->GetNodeName(),
        "The shape of input group_index must be 1-D."),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(mxScaleShape.GetDimNum() != SCALE_DIM_NUM,
        OP_LOGE(context->GetNodeName(),
        "The shape of output mxscale must be 3-D."),
        return ge::GRAPH_FAILED);

    tilingParam.groupSize = groupIndexShape.GetDim(0);
    tilingParam.preAxisSize = xShape.GetDim(0);
    tilingParam.postAxisSize = xShape.GetDim(1);
    OP_CHECK_IF(tilingParam.groupSize == 0,
        OP_LOGE(context->GetNodeName(),
        "group_index does not support empty tensor."),
        return ge::GRAPH_FAILED);

    xShape.SetDim(0, tilingParam.preAxisSize/(tilingParam.blockSize *DIGIT_TWO)+tilingParam.groupSize);
    xShape.SetDim(1, tilingParam.postAxisSize * DIGIT_TWO);
    OP_CHECK_IF(
        mxScaleShape[0] != xShape[0] || mxScaleShape[1] != tilingParam.postAxisSize ||
        mxScaleShape[SCALE_DIM_NUM - 1] != DIGIT_TWO,
        OP_LOGE(
            context->GetNodeName(),
            "The shape of output mxscale is incorrect, it should be [x.shape[0] / (2 * "
            "blocksize) + group_index.shape[0], x.shape[1], 2]."),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus GetPlatInfo(const gert::TilingContext *context, GroupedDynamicMxQuantTilingParam &tilingParam)
{
    OP_LOGD(context->GetNodeName(), "GetPlatInfo begin.");
    auto platformInfo = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    tilingParam.totalCoreNum = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF((tilingParam.totalCoreNum <= 0),
        OP_LOGE(context->GetNodeName(), "Failed to get core num."), return ge::GRAPH_FAILED);
    uint64_t ubSize;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    tilingParam.ubSize = static_cast<int64_t>(ubSize);
    OP_CHECK_IF((tilingParam.ubSize <= 0),
        OP_LOGE(context->GetNodeName(), "Failed to get ub size."), return ge::GRAPH_FAILED);
    tilingParam.vfLen = Ops::Base::GetVRegSize(context);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus DoTiling(const gert::TilingContext *context, GroupedDynamicMxQuantTilingParam &tilingParam)
{
    OP_LOGD(context->GetNodeName(), "DoTiling begin.");
    // 计算tilingkey
    // 十位数为1、2，分别表示输入类型是float16、bfloat16;
    int64_t hundredDigit = tilingParam.inDtype == DT_FLOAT16 ? 1 : DIGIT_TWO;
    // 个位数为1、2，分别表示输出类型是float8_e4m3fn、float8_e5m2
    int64_t tenDigit = tilingParam.outDtype == DT_FLOAT8_E4M3FN? 1: 2;
    tilingParam.tilingKey = hundredDigit * DIGIT_TEN + tenDigit;

    // 计算ubFactor
    const int64_t cacheline = static_cast<int64_t>(tilingParam.vfLen / BYTES_OF_INPUT_TYPE);
    int64_t maxUbAvailable = tilingParam.ubSize / N_BUFFER / EXIST_NODE_NUM;
    // 按照2倍blocksize对齐，保证e8m0_2可以ub内交织计算
    tilingParam.maxUbCol = static_cast<int64_t>(maxUbAvailable / static_cast<int64_t>(tilingParam.vfLen) / (tilingParam.blockSize*DIGIT_TWO) * (tilingParam.blockSize*DIGIT_TWO)); 
    tilingParam.ubFactor = cacheline;
    tilingParam.uo = CeilDiv(tilingParam.postAxisSize, tilingParam.ubFactor);
    tilingParam.tailUbFactor = tilingParam.postAxisSize - (tilingParam.uo - 1) * tilingParam.ubFactor;

    int64_t spliteCoreData = tilingParam.uo * tilingParam.groupSize;
    int64_t coreData = CeilDiv(spliteCoreData, tilingParam.totalCoreNum);
    tilingParam.usedCoreNum = CeilDiv(spliteCoreData, coreData);
    tilingParam.blockFactor = CeilDiv(spliteCoreData, tilingParam.usedCoreNum);
    tilingParam.tailBlockFactor = spliteCoreData - (tilingParam.usedCoreNum - 1) * tilingParam.blockFactor;

    return ge::GRAPH_SUCCESS;
}

inline static ge::graphStatus SetTilingData(gert::TilingContext *context,
    const GroupedDynamicMxQuantTilingParam &tilingParam, GroupedDynamicMxQuantTilingData &tilingData)
{
    OP_LOGD(context->GetNodeName(), "SetTilingData begin.");
    tilingData.set_totalCoreNum(tilingParam.totalCoreNum);
    tilingData.set_usedCoreNum(tilingParam.usedCoreNum);
    tilingData.set_blockFactor(tilingParam.blockFactor);
    tilingData.set_tailBlockFactor(tilingParam.tailBlockFactor);
    tilingData.set_uo(tilingParam.uo);
    tilingData.set_maxUbCol(tilingParam.maxUbCol);
    tilingData.set_ubFactor(tilingParam.ubFactor);
    tilingData.set_tailUbFactor(tilingParam.tailUbFactor);
    tilingData.set_blockSize(tilingParam.blockSize);
    tilingData.set_preAxisSize(tilingParam.preAxisSize);
    tilingData.set_postAxisSize(tilingParam.postAxisSize);

    OP_CHECK_IF(tilingData.GetDataSize() > context->GetRawTilingData()->GetCapacity(),
            OP_LOGE(context->GetNodeName(), "tiling datasize: %zu is bigger than %zu",
                                        tilingData.GetDataSize(), context->GetRawTilingData()->GetCapacity()),
            return ge::GRAPH_FAILED);
    tilingData.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tilingData.GetDataSize());
    context->SetBlockDim(tilingData.get_usedCoreNum());
    context->SetTilingKey(tilingParam.tilingKey);
    size_t *workspaces = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, workspaces);
    workspaces[0] = WORKSPACE_SIZE;
    return ge::GRAPH_SUCCESS;
}

inline static void PrintTilingData(const gert::TilingContext *context, GroupedDynamicMxQuantTilingData &tilingData)
{
    OP_LOGI(context->GetNodeName(), "tilingData is totalCoreNum:%ld, usedCoreNum:%ld,  ubFactor:%ld, \
        tailUbFactor:%ld, blockFactor:%ld, tailBlockFactor:%ld, uo:%ld, maxUbCol:%ld, blockSize:%ld, preAxisSize:%ld, postAxisSize:%ld",
        tilingData.get_totalCoreNum(), tilingData.get_usedCoreNum(), tilingData.get_ubFactor(),
        tilingData.get_tailUbFactor(), tilingData.get_blockFactor(), tilingData.get_tailBlockFactor(),
        tilingData.get_uo(), tilingData.get_maxUbCol(), tilingData.get_blockSize(),
        tilingData.get_preAxisSize(), tilingData.get_postAxisSize());
}

ge::graphStatus Tiling4GroupedDynamicMxQuant(gert::TilingContext *context)
{
    OP_LOGD(context->GetNodeName(), "Tiling4GroupedDynamicMxQuant running begin.");

    GroupedDynamicMxQuantTilingParam tilingParam;

    OP_CHECK_IF(CheckDtype(context, tilingParam) != ge::GRAPH_SUCCESS,
        OP_LOGE(context->GetNodeName(), "The data type check failed."), return ge::GRAPH_FAILED);

    OP_CHECK_IF(GetAttr(context, tilingParam) != ge::GRAPH_SUCCESS,
        OP_LOGE(context->GetNodeName(), "The attr get failed."), return ge::GRAPH_FAILED);

    OP_CHECK_IF(CheckShape(context, tilingParam) != ge::GRAPH_SUCCESS,
        OP_LOGE(context->GetNodeName(), "The shape check failed."), return ge::GRAPH_FAILED);

    OP_CHECK_IF(GetPlatInfo(context, tilingParam) != ge::GRAPH_SUCCESS,
        OP_LOGE(context->GetNodeName(), "GetPlatInfo failed."), return ge::GRAPH_FAILED);

    OP_CHECK_IF(DoTiling(context, tilingParam) != ge::GRAPH_SUCCESS,
        OP_LOGE(context->GetNodeName(), "DoTiling failed."), return ge::GRAPH_FAILED);

    GroupedDynamicMxQuantTilingData tilingData;
    OP_CHECK_IF(SetTilingData(context, tilingParam, tilingData) != ge::GRAPH_SUCCESS,
        OP_LOGE(context->GetNodeName(), "SetContext fail."),
        return ge::GRAPH_FAILED);

    PrintTilingData(context, tilingData);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingPrepare4GroupedDynamicMxQuant(gert::TilingParseContext *context)
{
    OP_LOGD(context->GetNodeName(), "TilingPrepare4GroupedDynamicMxQuant entering.");
    return ge::GRAPH_SUCCESS;
}

// register tiling interface of the GroupedDynamicMxQuant op.
IMPL_OP_OPTILING(GroupedDynamicMxQuant)
    .Tiling(Tiling4GroupedDynamicMxQuant)
    .TilingParse<GroupedDynamicMxQuantCompileInfo>(TilingPrepare4GroupedDynamicMxQuant);
}
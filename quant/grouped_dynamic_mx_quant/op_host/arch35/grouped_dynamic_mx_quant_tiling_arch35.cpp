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
constexpr int64_t DIGIT_ZERO= 0;
constexpr float DIGIT_ZERO_FLOAT= 0.0;
constexpr int64_t DIGIT_ONE = 1;
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

inline static ge::graphStatus GroupedDynamicMxQuantSetTilingData(gert::TilingContext* context, GroupedDynamicMxQuantTilingData& tilingData)
{
    uint64_t tilingDataSize = sizeof(tilingData);
    OP_CHECK_NULL_WITH_CONTEXT(context, context->GetRawTilingData());
    auto rawTilingData = context->GetRawTilingData();
    errno_t ret = memcpy_s(rawTilingData->GetData(), rawTilingData->GetCapacity(),
        reinterpret_cast<void *>(&tilingData), tilingDataSize);
    if (ret != EOK) {
        OP_LOGE(context->GetNodeName(), "memcpy_s failed, ret = %d", ret);
        return ge::GRAPH_FAILED;
    }
    context->GetRawTilingData()->SetDataSize(tilingDataSize);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus GetAttr(const gert::TilingContext *context, GroupedDynamicMxQuantTilingParam &tilingParam)
{
    OP_LOGD(context, "GetAttr begin.");
    auto *attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    
    auto *attrRoundMode = attrs->GetAttrPointer<char>(INDEX_ATTR_ROUND_MODE);
    OP_CHECK_NULL_WITH_CONTEXT(context, attrRoundMode);
    std::string roundModeStr = attrRoundMode;
    OP_CHECK_IF((roundModeStr != "rint"),
        OP_LOGE(context, "round_mode only supports rint currently, please check."),
        return ge::GRAPH_FAILED);

    auto *attrDstType = attrs->GetAttrPointer<int64_t>(INDEX_ATTR_DST_DTYPE);
    OP_CHECK_NULL_WITH_CONTEXT(context, attrDstType);
    int checkDstType = static_cast<int>(*attrDstType);
    OP_CHECK_IF((tilingParam.outDtype == ge::DT_FLOAT8_E4M3FN && checkDstType != 36) ||
        (tilingParam.outDtype == ge::DT_FLOAT8_E5M2 && checkDstType != 35),
        OP_LOGE(context,
        "y's data type and dst_type is not corresponded, y's data type: FLOAT8_E4M3FN/FLOAT8_E5M2 correspond to dst_type: 36/35."),
        return ge::GRAPH_FAILED);

    auto *attrBlockSize = attrs->GetAttrPointer<int64_t>(INDEX_ATTR_BLOCK_SIZE);
    OP_CHECK_NULL_WITH_CONTEXT(context, attrBlockSize);
    tilingParam.blockSize = static_cast<int64_t>(*attrBlockSize);
    OP_CHECK_IF(tilingParam.blockSize != ATTR_BLOCK_SIZE,
        OP_LOGE(context,
        "The blocksize only supports 32."),
        return ge::GRAPH_FAILED);
    
    auto *attrScaleAlg = attrs->GetAttrPointer<int64_t>(INDEX_ATTR_SCALE_ALG);
    OP_CHECK_NULL_WITH_CONTEXT(context, attrScaleAlg);
    tilingParam.scaleAlg = static_cast<int64_t>(*attrScaleAlg);
    OP_CHECK_IF(tilingParam.scaleAlg != DIGIT_ZERO && tilingParam.scaleAlg != DIGIT_ONE,
        OP_LOGE(context, "The scale_alg only supports 0 or 1."),
        return ge::GRAPH_FAILED);
    
    auto *attrDstTypeMax = attrs->GetAttrPointer<float>(INDEX_ATTR_DST_TYPE_MAX);
    OP_CHECK_NULL_WITH_CONTEXT(context, attrDstTypeMax);
    tilingParam.dstTypeMax = static_cast<float>(*attrDstTypeMax);
    OP_CHECK_IF(tilingParam.dstTypeMax != DIGIT_ZERO_FLOAT,
        OP_LOGE(context, "The dst_type_max only supports 0.0."),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckDtype(const gert::TilingContext *context, GroupedDynamicMxQuantTilingParam &tilingParam)
{
    OP_LOGD(context, "CheckDtype begin.");
    auto inputXPtr = context->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputXPtr);
    tilingParam.inDtype = inputXPtr->GetDataType();
    OP_CHECK_IF(INPUT_SUPPORT_DTYPE_SET.count(tilingParam.inDtype) == 0,
        OP_LOGE(context, "Input x dtype only support FLOAT16/BFLOAT16 currently, please check."),
        return ge::GRAPH_FAILED);

    auto groupIndexPtr = context->GetInputDesc(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, groupIndexPtr);
    auto groupIndexDtype = groupIndexPtr->GetDataType();
    OP_CHECK_IF(GROUPIDX_SUPPORT_DTYPE_SET.count(groupIndexDtype) == 0,
        OP_LOGE(context->GetNodeName(), "Input group_index only supports Int32 currently, please check."),
        return ge::GRAPH_FAILED);

    auto outputYPtr = context->GetOutputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, outputYPtr);
    tilingParam.outDtype = outputYPtr->GetDataType();
    OP_CHECK_IF(Y_SUPPORT_DTYPE_SET.count(tilingParam.outDtype) == 0,
        OP_LOGE(context, "Output y only supports FLOAT8_E4M3FN/FLOAT8_E5M2 currently, please check."),
        return ge::GRAPH_FAILED);

    auto outputMxScalePtr = context->GetOutputDesc(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, outputMxScalePtr);
    auto scaleDtype = outputMxScalePtr->GetDataType();
    OP_CHECK_IF(OUTPUT_SUPPORT_DTYPE_SET.count(scaleDtype) == 0,
        OP_LOGE(context, "Output mxscale only supports FLOAT8_E8M0 currently, please check."),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckShape(const gert::TilingContext *context, GroupedDynamicMxQuantTilingParam &tilingParam)
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
        OP_LOGE(context, "The shape of output y must be same with shape of input x."),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(xShape.GetDimNum() != 2,
        OP_LOGE(context, "The shape of input x must be 2-D."),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(groupIndexShape.GetDimNum() != 1,
        OP_LOGE(context, "The shape of input group_index must be 1-D."),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(mxScaleShape.GetDimNum() != SCALE_DIM_NUM,
        OP_LOGE(context, "The shape of output mxscale must be 3-D."),
        return ge::GRAPH_FAILED);

    tilingParam.groupSize = groupIndexShape.GetDim(0);
    tilingParam.preAxisSize = xShape.GetDim(0);
    tilingParam.postAxisSize = xShape.GetDim(1);
    OP_CHECK_IF(tilingParam.groupSize == 0,
        OP_LOGE(context, "group_index does not support empty tensor."),
        return ge::GRAPH_FAILED);

    xShape.SetDim(0, tilingParam.preAxisSize / (tilingParam.blockSize * DIGIT_TWO) + tilingParam.groupSize);
    xShape.SetDim(1, tilingParam.postAxisSize * DIGIT_TWO);
    OP_CHECK_IF(
        mxScaleShape[0] != xShape[0] || mxScaleShape[1] != tilingParam.postAxisSize ||
        mxScaleShape[SCALE_DIM_NUM - 1] != DIGIT_TWO,
        OP_LOGE(
            context,
            "The shape of output mxscale is incorrect, it should be [x.shape[0] / (2 * "
            "blocksize) + group_index.shape[0], x.shape[1], 2]."),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus GetPlatInfo(const gert::TilingContext *context, GroupedDynamicMxQuantTilingParam &tilingParam)
{
    OP_LOGD(context, "GetPlatInfo begin.");
    auto platformInfo = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);

    tilingParam.totalCoreNum = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF((tilingParam.totalCoreNum <= 0),
        OP_LOGE(context, "Failed to get core num."), return ge::GRAPH_FAILED);

    uint64_t ubSize;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    tilingParam.ubSize = static_cast<int64_t>(ubSize);
    OP_CHECK_IF((tilingParam.ubSize <= 0),
        OP_LOGE(context, "Failed to get ub size."), return ge::GRAPH_FAILED);

    tilingParam.vfLen = Ops::Base::GetVRegSize(context);
    OP_CHECK_IF((tilingParam.ubSize <= 0),
        OP_LOGE(context, "Failed to get ub size."), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus DoTiling(const gert::TilingContext *context, GroupedDynamicMxQuantTilingParam &tilingParam)
{
    OP_LOGD(context, "DoTiling begin.");
    // 计算tilingkey
    // 十位数为1、2，分别表示输入类型是float16、bfloat16;
    int64_t hundredDigit = tilingParam.inDtype == DT_FLOAT16 ? DIGIT_ONE : DIGIT_TWO;
    // 个位数为1、2，分别表示输出类型是float8_e4m3fn、float8_e5m2
    int64_t tenDigit = tilingParam.outDtype == DT_FLOAT8_E4M3FN ? DIGIT_ONE : DIGIT_TWO;
    tilingParam.tilingKey = hundredDigit * DIGIT_TEN + tenDigit;

    // 计算ubFactor
    const int64_t cacheline = static_cast<int64_t>(tilingParam.vfLen / BYTES_OF_INPUT_TYPE);
    int64_t maxUbAvailable = tilingParam.ubSize / N_BUFFER / EXIST_NODE_NUM;
    // 按照2倍blocksize对齐，保证e8m0_2可以ub内交织计算
    tilingParam.maxUbCol = static_cast<int64_t>(maxUbAvailable / static_cast<int64_t>(tilingParam.vfLen) /
        (tilingParam.blockSize * DIGIT_TWO) * (tilingParam.blockSize * DIGIT_TWO)); 
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

inline static ge::graphStatus SetTilingKeyParam(gert::TilingContext *context,
    const GroupedDynamicMxQuantTilingParam &tilingParam, GroupedDynamicMxQuantTilingData &tilingData)
{
    uint64_t mode = 0;
    int64_t tilingKey = GET_TPL_TILING_KEY(mode);
    OP_LOGD(context->GetNodeName(), "mode is %ld", mode);
    context->SetTilingKey(tilingKey);
    
    return ge::GRAPH_SUCCESS;
}

inline static ge::graphStatus SetTilingData(gert::TilingContext *context,
    const GroupedDynamicMxQuantTilingParam &tilingParam, GroupedDynamicMxQuantTilingData &tilingData)
{
    tilingData.totalCoreNum = tilingParam.totalCoreNum;
    tilingData.usedCoreNum = tilingParam.usedCoreNum;
    tilingData.blockFactor = tilingParam.blockFactor;
    tilingData.tailBlockFactor = tilingParam.tailBlockFactor;
    tilingData.uo = tilingParam.uo;
    tilingData.maxUbCol = tilingParam.maxUbCol;
    tilingData.ubFactor = tilingParam.ubFactor;
    tilingData.tailUbFactor = tilingParam.tailUbFactor;
    tilingData.blockSize = tilingParam.blockSize;
    tilingData.scaleAlg = tilingParam.scaleAlg;
    tilingData.preAxisSize = tilingParam.preAxisSize;
    tilingData.postAxisSize = tilingParam.postAxisSize;
    tilingData.dstTypeMax = tilingParam.dstTypeMax;

    return ge::GRAPH_SUCCESS;
}

inline static void PrintTilingData(const gert::TilingContext *context, GroupedDynamicMxQuantTilingData &tilingData)
{
    OP_LOGI(context, "tilingData is totalCoreNum:%ld, usedCoreNum:%ld, blockFactor:%ld, tailUbFactor:%ld, uo:%ld, \
        maxUbCol:%ld, ubFactor:%ld, tailBlockFactor:%ld, blockSize:%ld, scaleAlg:%ld, preAxisSize:%ld, postAxisSize:%ld, \
        dstTypeMax:%f",
        tilingData.totalCoreNum, tilingData.usedCoreNum, tilingData.blockFactor, tilingData.tailUbFactor, tilingData.uo,
        tilingData.maxUbCol, tilingData.ubFactor, tilingData.tailBlockFactor, tilingData.blockSize, tilingData.scaleAlg,
        tilingData.preAxisSize, tilingData.postAxisSize, tilingData.dstTypeMax);
}

ge::graphStatus Tiling4GroupedDynamicMxQuant(gert::TilingContext *context)
{
    OP_LOGD(context, "Tiling4GroupedDynamicMxQuant running begin.");

    GroupedDynamicMxQuantTilingParam tilingParam;

    OP_CHECK_IF(CheckDtype(context, tilingParam) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "The data type check failed."), return ge::GRAPH_FAILED);

    OP_CHECK_IF(GetAttr(context, tilingParam) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "The attr get failed."), return ge::GRAPH_FAILED);

    OP_CHECK_IF(CheckShape(context, tilingParam) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "The shape check failed."), return ge::GRAPH_FAILED);

    OP_CHECK_IF(GetPlatInfo(context, tilingParam) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "GetPlatInfo failed."), return ge::GRAPH_FAILED);

    GroupedDynamicMxQuantTilingData tilingData = {};
    OP_CHECK_IF(DoTiling(context, tilingParam) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "DoTiling failed."), return ge::GRAPH_FAILED);
    
    SetTilingKeyParam(context, tilingParam, tilingData);
    SetTilingData(context, tilingParam, tilingData);
    OP_CHECK_IF(GroupedDynamicMxQuantSetTilingData(context, tilingData) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "GroupedDynamicMxQuantSetTilingData set tiling data fail."),
        return ge::GRAPH_FAILED);
    
    context->SetBlockDim(tilingData.usedCoreNum);
    size_t *workspaces = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, workspaces);
    workspaces[0] = WORKSPACE_SIZE;
    PrintTilingData(context, tilingData);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingPrepare4GroupedDynamicMxQuant(gert::TilingParseContext *context)
{
    OP_LOGD(context, "TilingPrepare4GroupedDynamicMxQuant entering.");
    return ge::GRAPH_SUCCESS;
}

// register tiling interface of the GroupedDynamicMxQuant op.
IMPL_OP_OPTILING(GroupedDynamicMxQuant)
    .Tiling(Tiling4GroupedDynamicMxQuant)
    .TilingParse<GroupedDynamicMxQuantCompileInfo>(TilingPrepare4GroupedDynamicMxQuant);
}
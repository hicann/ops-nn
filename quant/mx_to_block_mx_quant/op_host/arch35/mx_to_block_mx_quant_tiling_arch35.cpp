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
 * \file mx_to_block_mx_quant_tiling_arch35.cpp
 * \brief
 */

#include "mx_to_block_mx_quant_tiling_arch35.h"
#include "quant/mx_to_block_mx_quant/op_kernel/arch35/mx_to_block_mx_quant_tilingdata.h"
#include "quant/mx_to_block_mx_quant/op_kernel/arch35/mx_to_block_mx_quant_struct.h"
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
using namespace MxToBlockMxQuantOp;

namespace optiling {

constexpr int64_t INDEX_ATTR_DST_TYPE = 0;
constexpr int64_t DIGIT_ONE = 1;
constexpr int64_t DIGIT_TWO = 2;
constexpr int64_t BLOCK_SIZE = 32;
constexpr int64_t MAX_DIM_NUM = 3;
constexpr int64_t MIN_DIM_NUM = 2;
constexpr int64_t SPLIT_M = 64;
constexpr int64_t SPLIT_N = 512;
constexpr int64_t DST_TYPE_FP8_E5M2 = 35;
constexpr int64_t DST_TYPE_FP8_E4M3FN = 36;
constexpr int64_t DIM_OFFSET_3 = 3;

const std::set<ge::DataType> INPUT_SUPPORT_DTYPE_SET = {ge::DT_FLOAT4_E2M1, ge::DT_FLOAT4_E1M2};
const std::set<ge::DataType> MXSCALE_SUPPORT_DTYPE_SET = {ge::DT_FLOAT8_E8M0};
const std::set<ge::DataType> Y_SUPPORT_DTYPE_SET = {ge::DT_FLOAT8_E5M2, ge::DT_FLOAT8_E4M3FN};
const std::set<ge::DataType> SCALE_SUPPORT_DTYPE_SET = {ge::DT_FLOAT8_E8M0};
const std::set<int64_t> DST_TYPE_SUPPORT_SET = {DST_TYPE_FP8_E5M2, DST_TYPE_FP8_E4M3FN};

inline static ge::graphStatus MxToBlockMxQuantSetTilingData(gert::TilingContext* context,
                                                            MxToBlockMxQuantTilingData& tilingData)
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

template <typename T>
static std::string Shape2String(const T& shape)
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

static ge::graphStatus CheckDtype(const gert::TilingContext* context)
{
    OP_LOGD(context->GetNodeName(), "CheckDtype begin.");
    auto inputXPtr = context->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputXPtr);
    OP_CHECK_IF(INPUT_SUPPORT_DTYPE_SET.count(inputXPtr->GetDataType()) == 0,
                OP_LOGE(context->GetNodeName(), "Input x dtype not supported."), return ge::GRAPH_FAILED);

    auto inputMxscalePtr = context->GetInputDesc(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputMxscalePtr);
    OP_CHECK_IF(MXSCALE_SUPPORT_DTYPE_SET.count(inputMxscalePtr->GetDataType()) == 0,
                OP_LOGE(context->GetNodeName(), "Input mxscale dtype not supported."), return ge::GRAPH_FAILED);

    auto outputYPtr = context->GetOutputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, outputYPtr);
    OP_CHECK_IF(Y_SUPPORT_DTYPE_SET.count(outputYPtr->GetDataType()) == 0,
                OP_LOGE(context->GetNodeName(), "Output y dtype not supported."), return ge::GRAPH_FAILED);

    auto outputScale1Ptr = context->GetOutputDesc(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, outputScale1Ptr);
    OP_CHECK_IF(SCALE_SUPPORT_DTYPE_SET.count(outputScale1Ptr->GetDataType()) == 0,
                OP_LOGE(context->GetNodeName(), "Output scale1 dtype not supported."), return ge::GRAPH_FAILED);

    auto outputScale2Ptr = context->GetOutputDesc(2);
    OP_CHECK_NULL_WITH_CONTEXT(context, outputScale2Ptr);
    OP_CHECK_IF(SCALE_SUPPORT_DTYPE_SET.count(outputScale2Ptr->GetDataType()) == 0,
                OP_LOGE(context->GetNodeName(), "Output scale2 dtype not supported."), return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus GetAttr(const gert::TilingContext* context, MxToBlockMxQuantTilingParam& tilingParam)
{
    OP_LOGD(context->GetNodeName(), "GetAttr begin.");
    auto* attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);

    auto* attrDstType = attrs->GetAttrPointer<int64_t>(INDEX_ATTR_DST_TYPE);
    OP_CHECK_NULL_WITH_CONTEXT(context, attrDstType);
    tilingParam.dstType = static_cast<int64_t>(*attrDstType);

    auto outputYPtr = context->GetOutputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, outputYPtr);
    auto yDtype = outputYPtr->GetDataType();

    OP_CHECK_IF(DST_TYPE_SUPPORT_SET.count(tilingParam.dstType) == 0,
                OP_LOGE(context->GetNodeName(), "dst_type[%ld] only supports 35 (FLOAT8_E5M2) or 36 (FLOAT8_E4M3FN).",
                        tilingParam.dstType),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF((yDtype == ge::DT_FLOAT8_E5M2 && tilingParam.dstType != 35) ||
                    (yDtype == ge::DT_FLOAT8_E4M3FN && tilingParam.dstType != 36),
                OP_LOGE(context->GetNodeName(),
                        "y's data type [%s] and dst_type [%ld] do not correspond. "
                        "FLOAT8_E5M2 / FLOAT8_E4M3FN map to dst_type 35 / 36 respectively, please check.",
                        Ops::Base::ToString(static_cast<ge::DataType>(yDtype)).c_str(), tilingParam.dstType),
                return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckShape(const gert::TilingContext* context)
{
    OP_LOGD(context->GetNodeName(), "CheckShape begin.");
    auto xShapePtr = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, xShapePtr);
    auto xShape = xShapePtr->GetStorageShape();

    OP_CHECK_IF(xShape.GetDimNum() < MIN_DIM_NUM || xShape.GetDimNum() > MAX_DIM_NUM,
                OP_LOGE(context->GetNodeName(), "Input x rank[%zu] should be in [2, 3].", xShape.GetDimNum()),
                return ge::GRAPH_FAILED);

    if (xShape.GetDim(xShape.GetDimNum() - 1) % DIGIT_TWO != 0) {
        OP_LOGE(context->GetNodeName(), "The last dimension must be even.");
        return ge::GRAPH_FAILED;
    }

    // 放宽 -2 轴约束：原要求 secondLastDim % 64 == 0，现允许任意正整数。
    // 不对齐场景（M % 64 != 0）由 rowMode=1 (NOT_ALIGNED) 兜底切分模板处理。
    int64_t secondLastDim = xShape.GetDim(xShape.GetDimNum() - DIGIT_TWO);
    OP_CHECK_IF(secondLastDim <= 0, OP_LOGE(context->GetNodeName(), "The -2 dim[%ld] must be positive.", secondLastDim),
                return ge::GRAPH_FAILED);

    auto mxScaleShapePtr = context->GetInputShape(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, mxScaleShapePtr);
    auto mxScaleShape = mxScaleShapePtr->GetStorageShape();

    auto expectedMxScaleShape = xShape;
    int64_t mxScaleAxisDim = (CeilDiv(xShape.GetDim(xShape.GetDimNum() - 1), BLOCK_SIZE) + DIGIT_ONE) / DIGIT_TWO;
    expectedMxScaleShape.SetDim(xShape.GetDimNum() - 1, mxScaleAxisDim);
    expectedMxScaleShape.AppendDim(DIGIT_TWO);
    OP_CHECK_IF(expectedMxScaleShape != mxScaleShape,
                OP_LOGE(context->GetNodeName(), "Input mxscale shape %s incorrect, expected %s.",
                        Shape2String(mxScaleShape).c_str(), Shape2String(expectedMxScaleShape).c_str()),
                return ge::GRAPH_FAILED);

    auto yShapePtr = context->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, yShapePtr);
    OP_CHECK_IF(yShapePtr->GetStorageShape() != xShape,
                OP_LOGE(context->GetNodeName(), "Output y shape must be same with input x shape."),
                return ge::GRAPH_FAILED);

    auto scale1ShapePtr = context->GetOutputShape(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, scale1ShapePtr);
    auto scale1Shape = scale1ShapePtr->GetStorageShape();
    OP_CHECK_IF(scale1Shape.GetDimNum() != xShape.GetDimNum() + 1,
                OP_LOGE(context->GetNodeName(), "Output scale1 rank incorrect."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(scale1Shape != mxScaleShape,
                OP_LOGE(context->GetNodeName(), "Output scale1 shape %s incorrect, expected %s.",
                        Shape2String(scale1Shape).c_str(), Shape2String(mxScaleShape).c_str()),
                return ge::GRAPH_FAILED);

    auto scale2ShapePtr = context->GetOutputShape(2);
    OP_CHECK_NULL_WITH_CONTEXT(context, scale2ShapePtr);
    auto scale2Shape = scale2ShapePtr->GetStorageShape();
    OP_CHECK_IF(scale2Shape.GetDimNum() != xShape.GetDimNum() + 1,
                OP_LOGE(context->GetNodeName(), "Output scale2 rank incorrect."), return ge::GRAPH_FAILED);
    int64_t expectedScale2Dim3 = ((CeilDiv(xShape.GetDim(xShape.GetDimNum() - DIGIT_TWO), BLOCK_SIZE) + DIGIT_ONE) /
                                  DIGIT_TWO) *
                                 DIGIT_TWO / DIGIT_TWO;
    int64_t expectedScale2Dim2 = xShape.GetDim(xShape.GetDimNum() - DIGIT_ONE);
    OP_CHECK_IF(xShape.GetDimNum() == 3 && scale2Shape.GetDim(0) != xShape.GetDim(0),
                OP_LOGE(context->GetNodeName(), "Output scale2 shape %s incorrect.", Shape2String(scale2Shape).c_str()),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(scale2Shape.GetDim(scale2Shape.GetDimNum() - DIM_OFFSET_3) != expectedScale2Dim3 ||
                    scale2Shape.GetDim(scale2Shape.GetDimNum() - DIGIT_TWO) != expectedScale2Dim2 ||
                    scale2Shape.GetDim(scale2Shape.GetDimNum() - 1) != DIGIT_TWO,
                OP_LOGE(context->GetNodeName(), "Output scale2 shape %s incorrect.", Shape2String(scale2Shape).c_str()),
                return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus GetPlatInfo(const gert::TilingContext* context, MxToBlockMxQuantTilingParam& tilingParam)
{
    OP_LOGD(context->GetNodeName(), "GetPlatInfo begin.");
    auto platformInfo = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);

    tilingParam.totalCoreNum = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF(tilingParam.totalCoreNum <= 0, OP_LOGE(context->GetNodeName(), "Failed to get core num."),
                return ge::GRAPH_FAILED);

    uint64_t ubSize;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    tilingParam.ubSize = static_cast<int64_t>(ubSize);
    OP_CHECK_IF(tilingParam.ubSize <= 0, OP_LOGE(context->GetNodeName(), "Failed to get ub size."),
                return ge::GRAPH_FAILED);

    tilingParam.vfLen = Ops::Base::GetVRegSize(context);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus DoTiling(const gert::TilingContext* context, MxToBlockMxQuantTilingParam& tilingParam)
{
    OP_LOGD(context->GetNodeName(), "DoTiling begin.");

    auto xShapePtr = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, xShapePtr);
    auto xShape = xShapePtr->GetStorageShape();
    auto mxscaleShapePtr = context->GetInputShape(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, mxscaleShapePtr);
    auto mxscaleShape = mxscaleShapePtr->GetStorageShape();

    // 取单 batch 的 M/B/K：M 为行数，B 为 batch 数，K 为列数。
    int64_t M = xShape.GetDim(xShape.GetDimNum() - DIGIT_TWO);
    int64_t B = (xShape.GetDimNum() == DIM_OFFSET_3) ? xShape.GetDim(xShape.GetDimNum() - DIM_OFFSET_3) : 1;
    int64_t K = xShape.GetDim(xShape.GetDimNum() - DIGIT_ONE);

    // 每个 batch 独立按 64×512 切基本块，不跨 batch 合并行。
    int64_t rowBlockNumPerBatch = CeilDiv(M, SPLIT_M);
    int64_t colBlockNumPerBatch = CeilDiv(K, SPLIT_N);
    int64_t rowTailLenPerBatch = M - (rowBlockNumPerBatch - DIGIT_ONE) * SPLIT_M;
    int64_t colTailLenPerBatch = K - (colBlockNumPerBatch - DIGIT_ONE) * SPLIT_N;

    int64_t totalBlockNum = B * rowBlockNumPerBatch * colBlockNumPerBatch;

    // former/tail 分核：前 headCoreNum 个头核各处理 headCoreBlockNum 块，
    // 后 tailCoreNum 个尾核各处理 tailCoreBlockNum 块，各核块数之和等于 totalBlockNum。
    int64_t usedCoreNum = std::min(tilingParam.totalCoreNum, totalBlockNum);
    usedCoreNum = (usedCoreNum == 0) ? 1 : usedCoreNum;
    int64_t headCoreBlockNum = CeilDiv(totalBlockNum, usedCoreNum);
    int64_t tailCoreBlockNum = headCoreBlockNum - 1;
    int64_t tailCoreNum = headCoreBlockNum * usedCoreNum - totalBlockNum;
    int64_t headCoreNum = usedCoreNum - tailCoreNum;

    tilingParam.batchNum = B;
    tilingParam.rowNum = M;
    tilingParam.colNum = K;
    tilingParam.usedCoreNum = usedCoreNum;
    tilingParam.rowBlockNumPerBatch = rowBlockNumPerBatch;
    tilingParam.colBlockNumPerBatch = colBlockNumPerBatch;
    tilingParam.rowTailLenPerBatch = rowTailLenPerBatch;
    tilingParam.colTailLenPerBatch = colTailLenPerBatch;
    tilingParam.totalBlockNum = totalBlockNum;
    tilingParam.headCoreBlockNum = headCoreBlockNum;
    tilingParam.tailCoreBlockNum = tailCoreBlockNum;
    tilingParam.headCoreNum = headCoreNum;
    tilingParam.tailCoreNum = tailCoreNum;

    tilingParam.colScaleNum = mxscaleShape.GetDim(mxscaleShape.GetDimNum() - 1) *
                              mxscaleShape.GetDim(mxscaleShape.GetDimNum() - DIGIT_TWO);

    return ge::GRAPH_SUCCESS;
}

inline static ge::graphStatus SetTilingKeyParam(gert::TilingContext* context,
                                                const MxToBlockMxQuantTilingParam& tilingParam)
{
    uint64_t mode = static_cast<uint64_t>(tilingParam.rowMode);
    int64_t tilingKey = GET_TPL_TILING_KEY(mode);
    OP_LOGD(context->GetNodeName(), "mode is %lu", mode);
    context->SetTilingKey(tilingKey);
    return ge::GRAPH_SUCCESS;
}

inline static ge::graphStatus SetTilingData(const MxToBlockMxQuantTilingParam& tilingParam,
                                            MxToBlockMxQuantTilingData& tilingData)
{
    tilingData.ubSize = tilingParam.ubSize;
    tilingData.dstType = tilingParam.dstType;
    tilingData.totalCoreNum = tilingParam.totalCoreNum;
    tilingData.usedCoreNum = tilingParam.usedCoreNum;
    tilingData.batchNum = tilingParam.batchNum;
    tilingData.rowNum = tilingParam.rowNum;
    tilingData.colNum = tilingParam.colNum;
    tilingData.colScaleNum = tilingParam.colScaleNum;
    tilingData.rowMode = tilingParam.rowMode;
    tilingData.rowBlockNumPerBatch = tilingParam.rowBlockNumPerBatch;
    tilingData.colBlockNumPerBatch = tilingParam.colBlockNumPerBatch;
    tilingData.rowTailLenPerBatch = tilingParam.rowTailLenPerBatch;
    tilingData.colTailLenPerBatch = tilingParam.colTailLenPerBatch;
    tilingData.totalBlockNum = tilingParam.totalBlockNum;
    tilingData.headCoreBlockNum = tilingParam.headCoreBlockNum;
    tilingData.tailCoreBlockNum = tilingParam.tailCoreBlockNum;
    tilingData.headCoreNum = tilingParam.headCoreNum;
    tilingData.tailCoreNum = tilingParam.tailCoreNum;
    return ge::GRAPH_SUCCESS;
}

inline static void PrintTilingData(const gert::TilingContext* context, MxToBlockMxQuantTilingData& tilingData)
{
    OP_LOGI(context->GetNodeName(),
            "tilingData is ubSize:%ld, dstType:%ld, totalCoreNum:%ld, usedCoreNum:%ld, "
            "batchNum:%ld, rowNum:%ld, colNum:%ld, colScaleNum:%ld, rowMode:%ld, "
            "rowBlockNumPerBatch:%ld, colBlockNumPerBatch:%ld, rowTailLenPerBatch:%ld, colTailLenPerBatch:%ld, "
            "totalBlockNum:%ld, headCoreBlockNum:%ld, tailCoreBlockNum:%ld, "
            "headCoreNum:%ld, tailCoreNum:%ld.",
            tilingData.ubSize, tilingData.dstType, tilingData.totalCoreNum, tilingData.usedCoreNum, tilingData.batchNum,
            tilingData.rowNum, tilingData.colNum, tilingData.colScaleNum, tilingData.rowMode,
            tilingData.rowBlockNumPerBatch, tilingData.colBlockNumPerBatch, tilingData.rowTailLenPerBatch,
            tilingData.colTailLenPerBatch, tilingData.totalBlockNum, tilingData.headCoreBlockNum,
            tilingData.tailCoreBlockNum, tilingData.headCoreNum, tilingData.tailCoreNum);
}

ge::graphStatus Tiling4MxToBlockMxQuant(gert::TilingContext* context)
{
    OP_LOGD(context->GetNodeName(), "Tiling4MxToBlockMxQuant running begin.");

    MxToBlockMxQuantTilingParam tilingParam;

    OP_CHECK_IF(CheckDtype(context) != ge::GRAPH_SUCCESS,
                OP_LOGE(context->GetNodeName(), "The data type check failed."), return ge::GRAPH_FAILED);

    OP_CHECK_IF(GetAttr(context, tilingParam) != ge::GRAPH_SUCCESS,
                OP_LOGE(context->GetNodeName(), "The attr get failed."), return ge::GRAPH_FAILED);

    OP_CHECK_IF(CheckShape(context) != ge::GRAPH_SUCCESS, OP_LOGE(context->GetNodeName(), "The shape check failed."),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(GetPlatInfo(context, tilingParam) != ge::GRAPH_SUCCESS,
                OP_LOGE(context->GetNodeName(), "GetPlatInfo failed."), return ge::GRAPH_FAILED);

    MxToBlockMxQuantTilingData tilingData = {};

    // 按 M 是否 64 倍数选择模板：rowMode=0 走对齐模板，rowMode=1 走非对齐模板。
    // 两套模板共用同一套切分与搬运，仅 Compute 路径不同。
    auto xShapePtrForMode = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, xShapePtrForMode);
    auto xShapeForMode = xShapePtrForMode->GetStorageShape();
    int64_t MForMode = xShapeForMode.GetDim(xShapeForMode.GetDimNum() - DIGIT_TWO);
    tilingParam.rowMode = (MForMode % (BLOCK_SIZE * DIGIT_TWO) == 0) ? TPL_ROW_ALIGNED : TPL_ROW_NOT_ALIGNED;

    OP_CHECK_IF(DoTiling(context, tilingParam) != ge::GRAPH_SUCCESS,
                OP_LOGE(context->GetNodeName(), "DoTiling failed."), return ge::GRAPH_FAILED);

    SetTilingKeyParam(context, tilingParam);
    SetTilingData(tilingParam, tilingData);
    OP_CHECK_IF(MxToBlockMxQuantSetTilingData(context, tilingData) != ge::GRAPH_SUCCESS,
                OP_LOGE(context->GetNodeName(), "MxToBlockMxQuantSetTilingData set tiling data fail."),
                return ge::GRAPH_FAILED);

    context->SetBlockDim(tilingData.usedCoreNum);
    size_t* workspaces = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, workspaces);
    workspaces[0] = 0;
    PrintTilingData(context, tilingData);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingPrepare4MxToBlockMxQuant(gert::TilingParseContext* context)
{
    OP_LOGD(context->GetNodeName(), "TilingPrepare4MxToBlockMxQuant entering.");
    return ge::GRAPH_SUCCESS;
}

// Register tiling interface of the MxToBlockMxQuant op.
IMPL_OP_OPTILING(MxToBlockMxQuant)
    .Tiling(Tiling4MxToBlockMxQuant)
    .TilingParse<MxToBlockMxQuantCompileInfo>(TilingPrepare4MxToBlockMxQuant);

} // namespace optiling

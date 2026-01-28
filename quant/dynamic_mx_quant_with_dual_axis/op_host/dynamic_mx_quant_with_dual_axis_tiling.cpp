/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file dynamic_mx_quant_with_dual_axis_tiling.cpp
 * \brief
 */

#include "dynamic_mx_quant_with_dual_axis_tiling.h"
#include "op_common/op_host/util/platform_util.h"
#include "error_util.h"
#include "util/math_util.h"
#include "quant/dynamic_mx_quant_with_dual_axis/op_kernel/arch35/dynamic_mx_quant_with_dual_axis_struct.h"

using namespace std;
using namespace ge;
using namespace AscendC;
using namespace DynamicMxQuantWithDualAxisOp;

namespace optiling {
constexpr int64_t INDEX_ATTR_ROUND_MODE = 0;
constexpr int64_t INDEX_ATTR_DST_DTYPE = 1;
constexpr int64_t INDEX_ATTR_SCALE_ALG = 2;
constexpr int64_t N_BUFFER = 2;
constexpr int64_t DIGIT_TWO = 2;
constexpr int64_t DIGIT_64 = 64;
constexpr int64_t DIGIT_128 = 128;
constexpr int64_t DIGIT_256 = 256;
constexpr int64_t DIGIT_101 = 101;
constexpr int64_t DIGIT_102 = 102;
constexpr int64_t BLOCK_SIZE = 32;
constexpr int64_t BLOCK_PER_GROUP = 2;
constexpr int64_t UINT16_BYTES_SIZE = 2;
constexpr int64_t UINT8_BYTES_SIZE = 1;
constexpr int64_t BYTES_OF_INPUT_TYPE = 2;
constexpr int64_t MAX_BYTES_OF_OUTPUT_TYPE = 1;

constexpr int64_t RESERVED_UB_SIZE = 1024;

const std::set<ge::DataType> INPUT_SUPPORT_DTYPE_SET = {ge::DT_FLOAT16, ge::DT_BF16};
const std::set<ge::DataType> INPUT_FP16_DTYPE_SET = {ge::DT_FLOAT16};
const std::set<ge::DataType> Y_SUPPORT_DTYPE_SET = {
    ge::DT_FLOAT4_E2M1, ge::DT_FLOAT4_E1M2, ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E5M2};
const std::set<ge::DataType> Y_SUPPORT_DTYPE_FP4_SET = {ge::DT_FLOAT4_E2M1, ge::DT_FLOAT4_E1M2};
const std::set<ge::DataType> Y_SUPPORT_DTYPE_FP8_SET = {ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E5M2};
const std::set<ge::DataType> OUTPUT_SUPPORT_DTYPE_SET = {ge::DT_FLOAT8_E8M0};

RoundModeList DynamicMxQuantWithDualAxisTiling::GetRoundMode(const std::string& roundMode)
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

// roundMode, dstType, scaleAlg
ge::graphStatus DynamicMxQuantWithDualAxisTiling::GetAttr()
{
    auto* attrs = context_->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context_, attrs);

    auto outputYPtr = context_->GetOutputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, outputYPtr);
    auto yDtype = outputYPtr->GetDataType();
    auto* attrRoundMode = attrs->GetAttrPointer<char>(INDEX_ATTR_ROUND_MODE);
    OP_CHECK_NULL_WITH_CONTEXT(context_, attrRoundMode);
    std::string roundModeStr = attrRoundMode;
    RoundModeList roundMode = GetRoundMode(roundModeStr);
    OP_CHECK_IF(
        (roundMode == RoundModeList::MODE_UNDEFINED),
        OP_LOGE(
            context_->GetNodeName(), "invalid round_mode:%s; round_mode should be one of {rint, floor, round}",
            attrRoundMode),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        (Y_SUPPORT_DTYPE_FP8_SET.count(yDtype) != 0 && roundMode != RoundModeList::MODE_RINT),
        OP_LOGE(
            context_->GetNodeName(),
            "When output y's data type is FLOAT8_E4M3FN/FLOAT8_E5M2, round_mode:[%s] only support rint, please check.",
            attrRoundMode),
        return ge::GRAPH_FAILED);
    tilingParams.roundMode = static_cast<int64_t>(roundMode);

    auto* attrDstType = attrs->GetAttrPointer<int64_t>(INDEX_ATTR_DST_DTYPE);
    OP_CHECK_NULL_WITH_CONTEXT(context_, attrDstType);
    tilingParams.dstType = static_cast<int64_t>(*attrDstType);
    int checkDstType = static_cast<int>(*attrDstType);
    OP_CHECK_IF(
        (yDtype == ge::DT_FLOAT4_E2M1 && checkDstType != 40) || (yDtype == ge::DT_FLOAT4_E1M2 && checkDstType != 41) ||
            (yDtype == ge::DT_FLOAT8_E4M3FN && checkDstType != 36) ||
            (yDtype == ge::DT_FLOAT8_E5M2 && checkDstType != 35),
        OP_LOGE(
            context_->GetNodeName(),
            "y's data type [%s] and dst_type [%ld] is not corresponded, y's data type: "
            "FLOAT4_E2M1/FLOAT4_E1M2/FLOAT8_E4M3FN/FLOAT8_E5M2 correspond to dst_type: 40/41/36/35, please check.",
            Ops::Base::ToString(yDtype).c_str(), tilingParams.dstType),
        return ge::GRAPH_FAILED);

    auto* attrScaleAlg = attrs->GetAttrPointer<int64_t>(INDEX_ATTR_SCALE_ALG);
    OP_CHECK_NULL_WITH_CONTEXT(context_, attrScaleAlg);
    tilingParams.scaleAlg = static_cast<int64_t>(*attrScaleAlg);
    OP_CHECK_IF(
        tilingParams.scaleAlg != 0,
        OP_LOGE(context_->GetNodeName(), "The scaleAlg[%ld] should be 0, please check.", tilingParams.scaleAlg),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus DynamicMxQuantWithDualAxisTiling::CheckDtype() const
{
    auto inputXPtr = context_->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, inputXPtr);
    auto xDtype = inputXPtr->GetDataType();
    OP_CHECK_IF(
        INPUT_SUPPORT_DTYPE_SET.count(xDtype) == 0,
        OP_LOGE(
            context_->GetNodeName(),
            "Input x's data type only support FLOAT16 and BFLOAT16 currently, but x is [%s], please check.",
            Ops::Base::ToString(xDtype).c_str()),
        return ge::GRAPH_FAILED);

    auto outputYPtr = context_->GetOutputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, outputYPtr);
    auto yDtype = outputYPtr->GetDataType();
    OP_CHECK_IF(
        Y_SUPPORT_DTYPE_SET.count(yDtype) == 0,
        OP_LOGE(
            context_->GetNodeName(),
            "Output y's data type only support FLOAT4_E2M1/FLOAT4_E1M2/FLOAT8_E4M3FN/FLOAT8_E5M2 currently, but y is "
            "[%s], please "
            "check.",
            Ops::Base::ToString(yDtype).c_str()),
        return ge::GRAPH_FAILED);

    auto outputMxScalePtr = context_->GetOutputDesc(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, outputMxScalePtr);
    auto scaleDtype = outputMxScalePtr->GetDataType();
    OP_CHECK_IF(
        OUTPUT_SUPPORT_DTYPE_SET.count(scaleDtype) == 0,
        OP_LOGE(
            context_->GetNodeName(),
            "Input mxscale's data type only support FLOAT8_E8M0 currently, but scale is [%s], please check.",
            Ops::Base::ToString(scaleDtype).c_str()),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus DynamicMxQuantWithDualAxisTiling::CheckShape() const
{
    auto xShapePtr = context_->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, xShapePtr);
    auto xShape = xShapePtr->GetStorageShape();

    OP_CHECK_IF(
        xShape.GetDimNum() < 2,
        OP_LOGE(
            context_->GetNodeName(),
            "The shape is invalid, axis num [%ld] should be large than or equal to 2, please check.",
            static_cast<int64_t>(xShape.GetDimNum())),
        return ge::GRAPH_FAILED);

    auto outputYPtr = context_->GetOutputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, outputYPtr);
    auto yDtype = outputYPtr->GetDataType();

    OP_CHECK_IF(
        xShape.GetDim(xShape.GetDimNum() - 1) % DIGIT_TWO != 0 && Y_SUPPORT_DTYPE_FP4_SET.count(yDtype) != 0,
        OP_LOGE(
            context_->GetNodeName(),
            "When output y's data type is FLOAT4_E2M1/FLOAT4_E1M2, the last axis should be even, please check."),
        return ge::GRAPH_FAILED);

    auto y1ShapePtr = context_->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, y1ShapePtr);
    auto y1Shape = y1ShapePtr->GetStorageShape();

    auto y2ShapePtr = context_->GetOutputShape(2);
    OP_CHECK_NULL_WITH_CONTEXT(context_, y2ShapePtr);
    auto y2Shape = y2ShapePtr->GetStorageShape();

    OP_CHECK_IF(
        xShape != y1Shape || xShape != y2Shape || y2Shape != y1Shape,
        OP_LOGE(
            context_->GetNodeName(),
            "The shape of output y1:%s, y2:%s must be same with shape of input x:%s, please check.",
            Shape2String(y1Shape).c_str(), Shape2String(y2Shape).c_str(), Shape2String(xShape).c_str()),
        return ge::GRAPH_FAILED);

    auto mxScale1ShapePtr = context_->GetOutputShape(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, mxScale1ShapePtr);
    auto mxScale1Shape = mxScale1ShapePtr->GetStorageShape();

    auto mxScale2ShapePtr = context_->GetOutputShape(3);
    OP_CHECK_NULL_WITH_CONTEXT(context_, mxScale2ShapePtr);
    auto mxScale2Shape = mxScale2ShapePtr->GetStorageShape();

    auto newScale1Shape = xShape;
    newScale1Shape.SetDim(xShape.GetDimNum() - 1, Ops::Base::CeilDiv(xShape.GetDim(xShape.GetDimNum() - 1), DIGIT_64));
    newScale1Shape.AppendDim(DIGIT_TWO);

    auto newScale2Shape = xShape;
    newScale2Shape.SetDim(
        xShape.GetDimNum() - 2, Ops::Base::CeilDiv(xShape.GetDim(xShape.GetDimNum() - DIGIT_TWO), DIGIT_64));
    newScale2Shape.AppendDim(DIGIT_TWO);

    OP_CHECK_IF(
        newScale1Shape != mxScale1Shape,
        OP_LOGE(
            context_->GetNodeName(), "The shape of output mxscale1 %s is incorrect, correct shape is %s, please check.",
            Shape2String(mxScale1Shape).c_str(), Shape2String(newScale1Shape).c_str()),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF(
        newScale2Shape != mxScale2Shape,
        OP_LOGE(
            context_->GetNodeName(), "The shape of output mxscale2 %s is incorrect, correct shape is %s, please check.",
            Shape2String(mxScale2Shape).c_str(), Shape2String(newScale2Shape).c_str()),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

// totalCoreNum，ubSize，workspaceSize
ge::graphStatus DynamicMxQuantWithDualAxisTiling::GetPlatformInfo()
{
    OP_LOGD(context_->GetNodeName(), "Enter DynamicMxQuantWithDualAxisTiling GetPlatformInfo.");

    auto platformInfo = context_->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context_, platformInfo);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    tilingParams.totalCoreNum = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF(
        (tilingParams.totalCoreNum <= 0), OP_LOGE(context_->GetNodeName(), "Failed to core num."),
        return ge::GRAPH_FAILED);
    uint64_t ubSize;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    tilingParams.ubSize = static_cast<int64_t>(ubSize);
    OP_CHECK_IF(
        (tilingParams.ubSize <= 0), OP_LOGE(context_->GetNodeName(), "Failed to get ub size."),
        return ge::GRAPH_FAILED);
    tilingParams.workspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();

    return ge::GRAPH_SUCCESS;
}

// 合轴
ge::graphStatus DynamicMxQuantWithDualAxisTiling::MergeAxis()
{
    auto xShapePtr = context_->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, xShapePtr);
    auto xShape = xShapePtr->GetStorageShape();

    for (size_t i = 0; i < xShape.GetDimNum() - DIGIT_TWO; i++) {
        tilingParams.dim0 *= xShape.GetDim(i);
    }
    tilingParams.dimNeg2 = xShape.GetDim(xShape.GetDimNum() - DIGIT_TWO);
    tilingParams.dimNeg1 = xShape.GetDim(xShape.GetDimNum() - 1);

    return ge::GRAPH_SUCCESS;
}

// 切块
void DynamicMxQuantWithDualAxisTiling::SplitCore(int64_t blockW, int64_t blockSize)
{
    uint64_t xUbSize = blockSize * BLOCK_PER_GROUP * blockW * BYTES_OF_INPUT_TYPE;
    uint64_t y1UbSize = blockSize * BLOCK_PER_GROUP * blockW * MAX_BYTES_OF_OUTPUT_TYPE;
    uint64_t y2UbSize = blockSize * BLOCK_PER_GROUP * blockW * MAX_BYTES_OF_OUTPUT_TYPE;
    uint64_t scale2Tmp = BLOCK_PER_GROUP * blockW * BYTES_OF_INPUT_TYPE;
    uint64_t scale1Tmp = blockSize * BLOCK_PER_GROUP *
                         Ops::Base::CeilAlign(Ops::Base::CeilDiv(blockW, blockSize), blockSize) * BYTES_OF_INPUT_TYPE;
    uint64_t scale2UbSize = BLOCK_PER_GROUP * blockW * UINT8_BYTES_SIZE;
    uint64_t scale1UbSize = blockSize * BLOCK_PER_GROUP *
                            Ops::Base::CeilAlign(Ops::Base::CeilDiv(blockW, blockSize), blockSize) * UINT8_BYTES_SIZE;

    tilingParams.groupPerUb = (tilingParams.ubSize - RESERVED_UB_SIZE) / N_BUFFER /
                              (xUbSize + y1UbSize + y2UbSize + scale2Tmp + scale1Tmp + scale2UbSize + scale1UbSize);

    tilingParams.splitBlockH = BLOCK_PER_GROUP * blockSize * tilingParams.groupPerUb;
    tilingParams.dimNeg2SplitBlockNum = Ops::Base::CeilDiv(tilingParams.dimNeg2, tilingParams.splitBlockH);
    tilingParams.dimNeg1BlockNum = Ops::Base::CeilDiv(tilingParams.dimNeg1, blockW);
    tilingParams.totalTaskNum = tilingParams.dim0 * tilingParams.dimNeg2SplitBlockNum * tilingParams.dimNeg1BlockNum;
}

ge::graphStatus DynamicMxQuantWithDualAxisTiling::SetTilingParams()
{
    OP_LOGD(context_->GetNodeName(), "Enter DynamicMxQuantWithDualAxisTiling SetTilingParams.");

    MergeAxis();

    tilingParams.blockSize = BLOCK_SIZE;
    tilingParams.blockW = DIGIT_256;
    SplitCore(tilingParams.blockW, tilingParams.blockSize);
    tilingParams.dimNeg2Tail = tilingParams.dimNeg2 % tilingParams.splitBlockH == 0 ?
                                   tilingParams.splitBlockH :
                                   tilingParams.dimNeg2 % tilingParams.splitBlockH;
    tilingParams.dimNeg1Tail = tilingParams.dimNeg1 % tilingParams.blockW == 0 ?
                                   tilingParams.blockW :
                                   tilingParams.dimNeg1 % tilingParams.blockW;

    tilingParams.dimNeg1IsPad = tilingParams.dimNeg1 % tilingParams.blockSize == 0 ? 0 : 1;
    tilingParams.dimNeg1IsOdd =
        Ops::Base::CeilDiv(tilingParams.dimNeg1, tilingParams.blockSize) % DIGIT_TWO == 0 ? 0 : 1; // 尾轴scale奇数列补0
    tilingParams.dimNeg2IsOdd = Ops::Base::CeilDiv(tilingParams.dimNeg2, tilingParams.blockSize) % DIGIT_TWO == 0 ?
                                    0 : 1; // 非尾轴scale奇数行补0

    tilingParams.blockCountPerBatch = tilingParams.dimNeg2SplitBlockNum * tilingParams.dimNeg1BlockNum;
    tilingParams.scale1ColCountPerBatch =
        Ops::Base::CeilDiv(tilingParams.dimNeg1, DIGIT_TWO * tilingParams.blockSize) * DIGIT_TWO;
    tilingParams.scale2RowCountPerBatch =
        Ops::Base::CeilDiv(tilingParams.dimNeg2, DIGIT_TWO * tilingParams.blockSize) * DIGIT_TWO;

    tilingParams.usedCoreNum = std::min(tilingParams.totalCoreNum, tilingParams.totalTaskNum);
    tilingParams.blockPerHeadCore = Ops::Base::CeilDiv(tilingParams.totalTaskNum, tilingParams.totalCoreNum);
    tilingParams.blockPerTailCore = tilingParams.totalTaskNum / tilingParams.totalCoreNum;
    tilingParams.headCoreNum = tilingParams.totalTaskNum % tilingParams.totalCoreNum;

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus DynamicMxQuantWithDualAxisTiling::DoTiling()
{
    OP_LOGD(context_->GetNodeName(), "Enter DynamicMxQuantWithDualAxisTiling DoTiling.");

    OP_CHECK_IF(
        GetPlatformInfo() != ge::GRAPH_SUCCESS, OP_LOGE(context_->GetNodeName(), "The platforminfo get failed"),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF(
        CheckDtype() != ge::GRAPH_SUCCESS, OP_LOGE(context_->GetNodeName(), "The data type check failed."),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF(
        GetAttr() != ge::GRAPH_SUCCESS, OP_LOGE(context_->GetNodeName(), "The attr get failed."),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF(
        CheckShape() != ge::GRAPH_SUCCESS, OP_LOGE(context_->GetNodeName(), "The shape check failed."),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF(
        SetTilingParams() != ge::GRAPH_SUCCESS,
        OP_LOGE(context_->GetNodeName(), "DynamicMxQuantWithDualAxisTiling SetTilingParams failed"),
        return ge::GRAPH_FAILED);

    SetTilingKey();
    SetTilingData();
    PrintTilingData();

    OP_CHECK_NULL_WITH_CONTEXT(context_, context_->GetRawTilingData());
    auto rawTilingData = context_->GetRawTilingData();
    if (tilingData.GetDataSize() > rawTilingData->GetCapacity()) {
        OP_LOGD(context_->GetNodeName(), "Tiling DataSize Greater than capacity, please check.");
        return ge::GRAPH_FAILED;
    }
    tilingData.SaveToBuffer(rawTilingData->GetData(), rawTilingData->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(tilingData.GetDataSize());

    OP_LOGD(context_->GetNodeName(), "Tiling usedCoreNum is %lu.", tilingParams.usedCoreNum);
    context_->SetBlockDim(tilingParams.usedCoreNum);

    size_t* workspaces = context_->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, workspaces);
    workspaces[0] = tilingParams.workspaceSize;

    return ge::GRAPH_SUCCESS;
}

void DynamicMxQuantWithDualAxisTiling::SetTilingKey()
{
    if (tilingParams.roundMode == static_cast<int64_t>(RoundModeList::MODE_ROUND)) {
        roundMode_ = TPL_ROUND;
    } else if (tilingParams.roundMode == static_cast<int64_t>(RoundModeList::MODE_FLOOR)) {
        roundMode_ = TPL_FLOOR;
    } else if (tilingParams.roundMode == static_cast<int64_t>(RoundModeList::MODE_RINT)) {
        roundMode_ = TPL_RINT;
    }
    scaleAlg_ = tilingParams.scaleAlg;
    mode_ = 0;

    int64_t tilingKey = GET_TPL_TILING_KEY(mode_, roundMode_, scaleAlg_);
    OP_LOGD(context_->GetNodeName(), "mode is %ld,roundMode is %ld,scaleAlg is %ld", mode_, roundMode_, scaleAlg_);
    context_->SetTilingKey(tilingKey);
}

void DynamicMxQuantWithDualAxisTiling::SetTilingData()
{
    tilingData.set_totalCoreNum(tilingParams.totalCoreNum);
    tilingData.set_usedCoreNum(tilingParams.usedCoreNum);
    tilingData.set_roundMode(tilingParams.roundMode);
    tilingData.set_dstType(tilingParams.dstType);
    tilingData.set_scaleAlg(tilingParams.scaleAlg);
    tilingData.set_blockSize(tilingParams.blockSize);
    tilingData.set_dim0(tilingParams.dim0);
    tilingData.set_dimNeg2(tilingParams.dimNeg2);
    tilingData.set_dimNeg1(tilingParams.dimNeg1);
    tilingData.set_blockW(tilingParams.blockW);
    tilingData.set_tilingKey(tilingParams.tilingKey);
    tilingData.set_splitBlockH(tilingParams.splitBlockH);
    tilingData.set_dimNeg2Tail(tilingParams.dimNeg2Tail);
    tilingData.set_dimNeg1Tail(tilingParams.dimNeg1Tail);
    tilingData.set_dimNeg2SplitBlockNum(tilingParams.dimNeg2SplitBlockNum);
    tilingData.set_dimNeg1BlockNum(tilingParams.dimNeg1BlockNum);
    tilingData.set_blockPerHeadCore(tilingParams.blockPerHeadCore);
    tilingData.set_blockPerTailCore(tilingParams.blockPerTailCore);
    tilingData.set_headCoreNum(tilingParams.headCoreNum);
    tilingData.set_dimNeg2IsOdd(tilingParams.dimNeg2IsOdd);
    tilingData.set_dimNeg1IsOdd(tilingParams.dimNeg1IsOdd);
    tilingData.set_dimNeg1IsPad(tilingParams.dimNeg1IsPad);
    tilingData.set_blockCountPerBatch(tilingParams.blockCountPerBatch);
    tilingData.set_scale1ColCountPerBatch(tilingParams.scale1ColCountPerBatch);
    tilingData.set_scale2RowCountPerBatch(tilingParams.scale2RowCountPerBatch);
}

void DynamicMxQuantWithDualAxisTiling::PrintTilingData()
{
    OP_LOGD(
        context_->GetNodeName(),
        "TilingData totalCoreNum: %ld, usedCoreNum: %ld, roundMode: %ld, dstType: %ld, "
        "scaleAlg: %ld, dim0: %ld, dimNeg2: %ld, dimNeg1: %ld, blockSize: %ld, blockW: %ld, "
        "tilingKey: %ld, splitBlockH: %ld, dimNeg2Tail: %ld, "
        "dimNeg1Tail: %ld, dimNeg2SplitBlockNum: %ld, dimNeg1BlockNum: %ld, "
        "blockPerHeadCore: %ld, blockPerTailCore: %ld, headCoreNum: %ld, "
        "dimNeg2IsOdd: %ld, dimNeg1IsOdd: %ld, dimNeg1IsPad: %ld, blockCountPerBatch: %ld, "
        "scale1ColCountPerBatch: %ld, scale2RowCountPerBatch: %ld ",
        tilingData.get_totalCoreNum(), tilingData.get_usedCoreNum(), tilingData.get_roundMode(),
        tilingData.get_dstType(), tilingData.get_scaleAlg(), tilingData.get_dim0(), tilingData.get_dimNeg2(),
        tilingData.get_dimNeg1(), tilingData.get_blockSize(), tilingData.get_blockW(), tilingData.get_tilingKey(),
        tilingData.get_splitBlockH(), tilingData.get_dimNeg2Tail(), tilingData.get_dimNeg1Tail(),
        tilingData.get_dimNeg2SplitBlockNum(), tilingData.get_dimNeg1BlockNum(), tilingData.get_blockPerHeadCore(),
        tilingData.get_blockPerTailCore(), tilingData.get_headCoreNum(), tilingData.get_dimNeg2IsOdd(),
        tilingData.get_dimNeg1IsOdd(), tilingData.get_dimNeg1IsPad(), tilingData.get_blockCountPerBatch(),
        tilingData.get_scale1ColCountPerBatch(), tilingData.get_scale2RowCountPerBatch());
}

static ge::graphStatus TilingForDynamicMxQuantWithDualAxis(gert::TilingContext* context)
{
    OP_LOGD("DynamicMxQuantWithDualAxisTiling", "Enter TilingForDynamicMxQuantWithDualAxisTiling");

    OP_CHECK_IF(
        context == nullptr, OP_LOGE("DynamicMxQuantWithDualAxisTiling", "Tiling context is null."),
        return ge::GRAPH_FAILED);

    DynamicMxQuantWithDualAxisTiling mxQunatTiling(context);
    return mxQunatTiling.DoTiling();
}

static ge::graphStatus TilingPrepareForDynamicMxQuantWithDualAxis(gert::TilingParseContext* context)
{
    OP_LOGD("DynamicMxQuantWithDualAxisTiling", "Enter TilingPrepareForDynamicMxQuantWithDualAxisTiling");

    OP_CHECK_IF(
        context == nullptr, OP_LOGE("DynamicMxQuantWithDualAxisTiling", "TilingParse context is null."),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(DynamicMxQuantWithDualAxis)
    .Tiling(TilingForDynamicMxQuantWithDualAxis)
    .TilingParse<DynamicMxQuantWithDualAxisCompileInfo>(TilingPrepareForDynamicMxQuantWithDualAxis);

} // namespace optiling
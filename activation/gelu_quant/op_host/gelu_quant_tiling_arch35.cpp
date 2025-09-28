/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file gelu_quant_regbase_tiling.cpp
 * \brief
 */

#include "gelu_quant_tiling_base.h"
#include "log/log.h"
#include "register/tilingdata_base.h"
#include "register/op_impl_registry.h"
#include "tiling_base/tiling_base.h"
#include "tiling_base/tiling_templates_registry.h"
#include "gelu_quant_tiling_arch35.h"

namespace optiling {
namespace geluquantregbase {
constexpr int64_t QUANT_REGBASE_COEXISTING_QUANTITY = 11;
constexpr int64_t DYNAMIC_QUANT_COEXISTING_QUANTITY_DB = 13;
// 获取baseInfoOp的vectorCoreNum、ubSize
ge::graphStatus GeluQuantRegbaseTiling::GetPlatformInfo()
{
    OP_LOGD(nodeName_, "GetPlatformInfo start running.");
    auto compileInfo = context_->GetCompileInfo<GeluQuantCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context_, compileInfo);
    baseInfoOp.vectorCoreNum = compileInfo->vectorCoreNum;
    OP_CHECK_IF(
        (baseInfoOp.vectorCoreNum <= 0),
        OP_LOGE(nodeName_, "GeluQuantRegbaseTiling get num of vector core is less than or equal to 0."),
        return ge::GRAPH_FAILED);

    baseInfoOp.ubSize = compileInfo->ubSize;
    OP_CHECK_IF(
        (baseInfoOp.ubSize <= 0), OP_LOGE(nodeName_, "GeluQuantRegbaseTiling get ub size is less than or equal to 0."),
        return ge::GRAPH_FAILED);

    baseInfoOp.ubSize -= RESERVED_UB_SIZE; // 可用UB空间

    return ge::GRAPH_SUCCESS;
}

RoundMode GeluQuantRegbaseTiling::GetRoundMode(std::string& roundMode)
{
    if (baseInfoOp.dstType == ge::DT_FLOAT8_E5M2 || baseInfoOp.dstType == ge::DT_FLOAT8_E4M3FN) {
        if (roundMode == "rint") {
            return geluquantregbase::RoundMode::MODE_RINT;
        }
        errorMsg_ = "round_mode only supports 'rint' for float8_e5m2/float8_e4m3fn.";
        return RoundMode::MODE_UNDEFINED;
    } else if (baseInfoOp.dstType == ge::DT_HIFLOAT8) {
        if (roundMode == "round") {
            return RoundMode::MODE_ROUND;
        } else if (roundMode == "hybrid") {
            return RoundMode::MODE_HYBRID;
        }
        errorMsg_ = "round_mode only supports 'round' and 'hybrid' for hifloat8.";
        return RoundMode::MODE_UNDEFINED;
    } else {
        if (roundMode == "rint") {
            return RoundMode::MODE_RINT;
        }
        errorMsg_ = "round_mode only supports 'rint' for int8";
        return RoundMode::MODE_UNDEFINED;
    }
}

// 获取baseInfoOp的approximate、quantMode。
ge::graphStatus GeluQuantRegbaseTiling::ProcessAttrsInfo()
{
    OP_LOGD(nodeName_, "ProcessAttrsInfo start running.");
    auto attrs = context_->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context_, attrs);

    const char* approximate = attrs->GetAttrPointer<char>(APPROXIMATE_ATTR_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, approximate);
    if (strcmp(approximate, "none") == 0) {
        baseInfoOp.approximate = APPROXIMATE_NONE;
    } else if (strcmp(approximate, "tanh") == 0) {
        baseInfoOp.approximate = APPROXIMATE_TANH;
    } else {
        OP_LOGE(nodeName_, "the attr of approximate should be none or tanh.");
        return ge::GRAPH_FAILED;
    }

    const char* quantMode = attrs->GetAttrPointer<char>(QUANT_MODE_ATTR_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, quantMode);
    if (strcmp(quantMode, "static") == 0) {
        baseInfoOp.quantMode = STATIC_QUANT_MODE;
    } else if (strcmp(quantMode, "dynamic") == 0) {
        baseInfoOp.quantMode = DYNAMIC_QUANT_MODE;
    } else {
        OP_LOGE(nodeName_, "the attr of quant mode should be static or dynamic.");
        return ge::GRAPH_FAILED;
    }

    const int32_t* dstTypePtr = attrs->GetAttrPointer<int32_t>(DST_TYPE_ATTR_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, dstTypePtr);
    baseInfoOp.dstType = *dstTypePtr;
    if (baseInfoOp.dstType != ge::DT_INT8 && baseInfoOp.dstType != ge::DT_FLOAT8_E5M2 &&
        baseInfoOp.dstType != ge::DT_FLOAT8_E4M3FN && baseInfoOp.dstType != ge::DT_HIFLOAT8) {
        OP_LOGE(
            nodeName_, "dst type:%s is invalid",
            Ops::Base::ToString(static_cast<ge::DataType>(baseInfoOp.dstType)).c_str());
        return ge::GRAPH_FAILED;
    }

    const char* roundModePtr = attrs->GetAttrPointer<char>(ROUND_MODE_ATTR_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, roundModePtr);
    std::string roundModeStr = roundModePtr;
    baseInfoOp.roundMode = GetRoundMode(roundModeStr);
    OP_CHECK_IF(
        (baseInfoOp.roundMode == RoundMode::MODE_UNDEFINED),
        OP_LOGE(nodeName_, "invalid round mode:%s, %s", roundModeStr.c_str(), errorMsg_.c_str()),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

// 获取baseInfoOp的xInputDtype、xDimNum、endAxisLen、endAxisLenAligned、fusedFrontAxis、fusedAllAxis、elementNumAlign
ge::graphStatus GeluQuantRegbaseTiling::ProcessRequiredInfo()
{
    OP_LOGD(nodeName_, "ProcessRequiredInfo start running.");
    // x
    auto xInputDesc = context_->GetInputDesc(X_INPUT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, xInputDesc);
    baseInfoOp.xInputDtype = xInputDesc->GetDataType();
    OP_CHECK_IF(
        (baseInfoOp.xInputDtype != ge::DT_FLOAT && baseInfoOp.xInputDtype != ge::DT_FLOAT16 &&
         baseInfoOp.xInputDtype != ge::DT_BF16),
        OP_LOGE(nodeName_, "the dtype of input x should be one of FP32/FP16/BF16 ."), return ge::GRAPH_FAILED);

    auto xInputShapePtr = context_->GetInputShape(X_INPUT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, xInputShapePtr);
    auto xInputShape = xInputShapePtr->GetStorageShape();
    baseInfoOp.xDimNum = xInputShape.GetDimNum();
    OP_CHECK_IF(
        (baseInfoOp.xDimNum > INPUT_MAX_DIMENSIONS),
        OP_LOGE(nodeName_, "the input of x should be no more than 8 dimensions."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        (baseInfoOp.xDimNum < INPUT_MIN_DIMENSIONS && baseInfoOp.quantMode == DYNAMIC_QUANT_MODE),
        OP_LOGE(nodeName_, "the input of x should be at least 2 dimensions when quant_mode is dynamic."),
        return ge::GRAPH_FAILED);

    baseInfoOp.endAxisLen = xInputShape.GetDim(baseInfoOp.xDimNum - 1);
    baseInfoOp.endAxisLenAligned = AlignToCeil(baseInfoOp.endAxisLen, FP32_BLOCK_NUM);
    for (int64_t i = 0; i < baseInfoOp.xDimNum - 1; i++) {
        baseInfoOp.fusedFrontAxis *= xInputShape.GetDim(i);
    }
    baseInfoOp.fusedAllAxis = baseInfoOp.fusedFrontAxis * baseInfoOp.endAxisLen;
    baseInfoOp.elementNumAlign = AlignToCeil(baseInfoOp.fusedAllAxis, FP32_BLOCK_NUM);

    // y
    auto yOutputShapePtr = context_->GetOutputShape(Y_OUTPUT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, yOutputShapePtr);
    auto yOutputShape = yOutputShapePtr->GetStorageShape();
    OP_CHECK_IF(
        (xInputShape != yOutputShape), OP_LOGE(nodeName_, "the shape of y should be same as x."),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

// 获取baseInfoOp的inputOffsetType、offsetInputDtype，这个参数dynamic不用，static可选
ge::graphStatus GeluQuantRegbaseTiling::ProcessOptionalOffsetInfo()
{
    OP_LOGD(nodeName_, "ProcessOptionalOffsetInfo start running.");
    // dynamic不需要offset这个参数
    if (baseInfoOp.quantMode == DYNAMIC_QUANT_MODE) {
        return ge::GRAPH_SUCCESS;
    }
    // 静态不输入offset，将inputOffsetType设置为EMPTY_TENSOR
    auto offsetInputShapePtr = context_->GetOptionalInputShape(OFFSET_INPUT_INDEX);
    if (offsetInputShapePtr == nullptr) {
        baseInfoOp.inputOffsetType = EMPTY_TENSOR;
        return ge::GRAPH_SUCCESS;
    }
    // 校验offset和scale数据类型一致
    auto offsetInputDesc = context_->GetOptionalInputDesc(OFFSET_INPUT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, offsetInputDesc);
    baseInfoOp.offsetInputDtype = offsetInputDesc->GetDataType();
    OP_CHECK_IF(
        (baseInfoOp.scaleInputDtype != baseInfoOp.offsetInputDtype),
        OP_LOGE(nodeName_, "the dtype of input_scale should be same with input_offset."), return ge::GRAPH_FAILED);
    // 校验offset和scale的shape一致，如果shape是[1]，则将inputOffsetType设置为SCALAR_TENSOR
    if (offsetInputShapePtr->GetStorageShape().GetShapeSize() == 1) {
        baseInfoOp.inputOffsetType = SCALAR_TENSOR;
        OP_CHECK_IF(
            (baseInfoOp.inputScaleType != SCALAR_TENSOR),
            OP_LOGE(nodeName_, "the shape of input_scale should be same with input_offset."), return ge::GRAPH_FAILED);
        // 校验shape是一维，并且校验size和输入x的最后一维保持一致
    } else {
        baseInfoOp.inputOffsetType = NORMAL_TENSOR;
        OP_CHECK_IF(
            (baseInfoOp.inputScaleType != NORMAL_TENSOR),
            OP_LOGE(nodeName_, "the shape of input_scale should be same with input_offset."), return ge::GRAPH_FAILED);
        auto offsetInputDimNum = offsetInputShapePtr->GetStorageShape().GetDimNum();
        OP_CHECK_IF(
            (offsetInputDimNum != 1),
            OP_LOGE(nodeName_, "the shape dim of input_offset should be 1 or 0,but got %zu .", offsetInputDimNum),
            return ge::GRAPH_FAILED);
        auto offsetInputDim0 = offsetInputShapePtr->GetStorageShape().GetDim(0);
        OP_CHECK_IF(
            (offsetInputDim0 != baseInfoOp.endAxisLen),
            OP_LOGE(
                nodeName_, "the shape of input_offset should be [%ld] ,but got [%ld].", baseInfoOp.endAxisLen,
                offsetInputDim0),
            return ge::GRAPH_FAILED);
    }

    return ge::GRAPH_SUCCESS;
}
// 获取baseInfoOp的inputScaleType、scaleInputDtype，静态该输入必选校验，校验输入精度高于x，其他校验与offset一致
ge::graphStatus GeluQuantRegbaseTiling::ProcessOptionalScaleInfo()
{
    OP_LOGD(nodeName_, "ProcessOptionalScaleInfo start running.");

    auto scaleInputShapePtr = context_->GetOptionalInputShape(SCALE_INPUT_INDEX);
    if (scaleInputShapePtr == nullptr) {
        baseInfoOp.inputScaleType = EMPTY_TENSOR;
        OP_CHECK_IF(
            (baseInfoOp.quantMode == STATIC_QUANT_MODE),
            OP_LOGE(nodeName_, "the input_scale should be required when quant_mode is static."),
            return ge::GRAPH_FAILED);
        return ge::GRAPH_SUCCESS;
    }

    auto scaleInputDesc = context_->GetOptionalInputDesc(SCALE_INPUT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, scaleInputDesc);
    baseInfoOp.scaleInputDtype = scaleInputDesc->GetDataType();
    OP_CHECK_IF(
        (baseInfoOp.xInputDtype == ge::DT_FLOAT && baseInfoOp.scaleInputDtype != ge::DT_FLOAT),
        OP_LOGE(nodeName_, "the dtype of x should be float when the dtype of scale is float."),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        (baseInfoOp.xInputDtype == ge::DT_FLOAT16 && baseInfoOp.scaleInputDtype == ge::DT_BF16),
        OP_LOGE(nodeName_, "the dtype of x should not be half when the dtype of scale is bfloat16."),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        (baseInfoOp.xInputDtype == ge::DT_BF16 && baseInfoOp.scaleInputDtype == ge::DT_FLOAT16),
        OP_LOGE(nodeName_, "the dtype of x should not be bfloat16 when the dtype of scale is half."),
        return ge::GRAPH_FAILED);

    if (scaleInputShapePtr->GetStorageShape().GetShapeSize() == 1) {
        baseInfoOp.inputScaleType = SCALAR_TENSOR;
    } else {
        baseInfoOp.inputScaleType = NORMAL_TENSOR;
        auto scaleInputDimNum = scaleInputShapePtr->GetStorageShape().GetDimNum();
        OP_CHECK_IF(
            (scaleInputDimNum != 1),
            OP_LOGE(nodeName_, "the shape dim of input_scale should be 1 or 0,but got %zu .", scaleInputDimNum),
            return ge::GRAPH_FAILED);
        auto scaleInputDim0 = scaleInputShapePtr->GetStorageShape().GetDim(0);
        OP_CHECK_IF(
            (scaleInputDim0 != baseInfoOp.endAxisLen),
            OP_LOGE(
                nodeName_, "the shape of input_scale should be [%ld] ,but got [%ld].", baseInfoOp.endAxisLen,
                scaleInputDim0),
            return ge::GRAPH_FAILED);
    }

    return ge::GRAPH_SUCCESS;
}
// 获取输入信息并校验
ge::graphStatus GeluQuantRegbaseTiling::GetInputInfo()
{
    OP_LOGD(nodeName_, "GetInputInfo start running.");
    ge::graphStatus ret = ge::GRAPH_SUCCESS;
    ret = ProcessAttrsInfo();
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }

    ret = ProcessRequiredInfo();
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }

    ret = ProcessOptionalScaleInfo();
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }

    ret = ProcessOptionalOffsetInfo();
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }

    OP_LOGD(nodeName_, "GetInputInfo run completed.");
    return ge::GRAPH_SUCCESS;
}
// perTensor模板下，设置usedCoreNum、normalCoreProcessNum、tailCoreProcessNum
ge::graphStatus GeluQuantRegbaseTiling::DoStaticQuantPerTensorTiling()
{
    OP_LOGD(nodeName_, "DoStaticQuantPerTensorTiling start running.");
    // 这边的节点数为计算DB后的节点数
    splitCoreOp.coexistentNodeNum = QUANT_REGBASE_COEXISTING_QUANTITY;
    // 每个UB能容下的计算次数（通过计算节点得出）向下32对齐
    splitCoreOp.coexistentNodeElementNum = AlignToFloor(
        (SafeDivide(
            static_cast<int64_t>(baseInfoOp.ubSize),
            static_cast<int64_t>(sizeof(float)) * splitCoreOp.coexistentNodeNum)),
        FP32_BLOCK_NUM);
    // 如果整个输入的size小于等于128，那么单核处理
    if (baseInfoOp.fusedAllAxis <= SINGLE_CORE_PROCESS_MIN_NUM) {
        splitCoreOp.usedCoreNum = 1;
        splitCoreOp.normalCoreProcessNum = baseInfoOp.fusedAllAxis;
        splitCoreOp.tailCoreProcessNum = baseInfoOp.fusedAllAxis;
        // 设置normalCoreProcessNum（至少为128）、usedCoreNum、tailCoreProcessNum
    } else {
        splitCoreOp.normalCoreProcessNum = CeilDivide(baseInfoOp.fusedAllAxis, baseInfoOp.vectorCoreNum);
        splitCoreOp.normalCoreProcessNum = splitCoreOp.normalCoreProcessNum < SINGLE_CORE_PROCESS_MIN_NUM ?
                                               SINGLE_CORE_PROCESS_MIN_NUM :
                                               splitCoreOp.normalCoreProcessNum;
        splitCoreOp.usedCoreNum = CeilDivide(baseInfoOp.fusedAllAxis, splitCoreOp.normalCoreProcessNum);
        splitCoreOp.tailCoreProcessNum =
            baseInfoOp.fusedAllAxis - splitCoreOp.normalCoreProcessNum * (splitCoreOp.usedCoreNum - 1);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GeluQuantRegbaseTiling::DoStaticQuantFullKernelSmallEndAxis()
{
    OP_LOGD(nodeName_, "DoStaticQuantFullKernelSmallEndAxis start running.");
    // mulRowsInUb，每个UB中处理尾轴row的数量
    int64_t mulRowsInUb = splitCoreOp.coexistentNodeElementNum / baseInfoOp.endAxisLenAligned;
    while (mulRowsInUb >= TWO_END_AXIS) {
        // 总共有多少个row/一个UB能处理的row = ubNum，理解为处理完所有数据需要使用多少次UB
        int64_t ubNum = CeilDivide(baseInfoOp.fusedFrontAxis, mulRowsInUb);
        if (ubNum >= baseInfoOp.vectorCoreNum) {
            break;
        } else {
            mulRowsInUb--;
        }
    }
    // 如果每个UB中只能处理一个row，那么回到STATIC_FUNCTION_TEMPLATE
    if (mulRowsInUb == 1) {
        splitCoreOp.templateMode = STATIC_FUNCTION_TEMPLATE;
        return ge::GRAPH_SUCCESS;
    }

    splitCoreOp.rowInner = mulRowsInUb;
    splitCoreOp.rowOuter = CeilDivide(baseInfoOp.fusedFrontAxis, mulRowsInUb);
    int64_t rowTailTmp = mulRowsInUb == 0 ? baseInfoOp.fusedFrontAxis : baseInfoOp.fusedFrontAxis % mulRowsInUb;
    splitCoreOp.rowTail = rowTailTmp == 0 ? splitCoreOp.rowInner : rowTailTmp;

    splitCoreOp.colInner = baseInfoOp.endAxisLen;
    splitCoreOp.colOuter = 1;
    splitCoreOp.colTail = baseInfoOp.endAxisLen;

    splitCoreOp.normalCoreProcessNum = CeilDivide(splitCoreOp.rowOuter, baseInfoOp.vectorCoreNum);
    splitCoreOp.usedCoreNum = CeilDivide(splitCoreOp.rowOuter, splitCoreOp.normalCoreProcessNum);
    splitCoreOp.tailCoreProcessNum =
        splitCoreOp.rowOuter - splitCoreOp.normalCoreProcessNum * (splitCoreOp.usedCoreNum - 1);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GeluQuantRegbaseTiling::DoStaticQuantNotFullKernelSplitEndAxis()
{
    OP_LOGD(nodeName_, "DoStaticQuantNotFullKernelSplitEndAxis start running.");
    splitCoreOp.rowInner = 1;
    splitCoreOp.rowOuter = baseInfoOp.fusedFrontAxis;
    splitCoreOp.rowTail = 1;

    int64_t colSplitNum = CeilDivide(baseInfoOp.vectorCoreNum, baseInfoOp.fusedFrontAxis);
    int64_t colInnerTmp = CeilDivide(baseInfoOp.endAxisLen, colSplitNum);
    colInnerTmp = colInnerTmp < SINGLE_CORE_PROCESS_MIN_NUM ? SINGLE_CORE_PROCESS_MIN_NUM :
                                                              AlignToFloor(colInnerTmp, SINGLE_CORE_PROCESS_MIN_NUM);
    if (colInnerTmp > splitCoreOp.coexistentNodeElementNum) {
        colInnerTmp = splitCoreOp.coexistentNodeElementNum;
    }

    splitCoreOp.colInner = colInnerTmp;
    splitCoreOp.colOuter = CeilDivide(baseInfoOp.endAxisLen, colInnerTmp);
    int64_t colTailTmp = colInnerTmp == 0 ? baseInfoOp.endAxisLen : baseInfoOp.endAxisLen % colInnerTmp;
    splitCoreOp.colTail = colTailTmp == 0 ? splitCoreOp.colInner : colTailTmp;

    splitCoreOp.normalCoreProcessNum =
        CeilDivide(splitCoreOp.rowOuter * splitCoreOp.colOuter, baseInfoOp.vectorCoreNum);
    splitCoreOp.usedCoreNum = CeilDivide(splitCoreOp.rowOuter * splitCoreOp.colOuter, splitCoreOp.normalCoreProcessNum);
    splitCoreOp.tailCoreProcessNum =
        splitCoreOp.rowOuter * splitCoreOp.colOuter - splitCoreOp.normalCoreProcessNum * (splitCoreOp.usedCoreNum - 1);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GeluQuantRegbaseTiling::DoStaticQuantTiling()
{
    OP_LOGD(nodeName_, "DoStaticQuantTiling start running.");
    // 如果scale的输入shape为[1]，走per_tensor模板
    if (baseInfoOp.inputScaleType == SCALAR_TENSOR) {
        splitCoreOp.templateMode = STATIC_PER_TENSOR_TEMPLATE;
        return DoStaticQuantPerTensorTiling();
    }

    splitCoreOp.coexistentNodeNum = QUANT_REGBASE_COEXISTING_QUANTITY;
    splitCoreOp.coexistentNodeElementNum = AlignToFloor(
        (SafeDivide(
            static_cast<int64_t>(baseInfoOp.ubSize),
            static_cast<int64_t>(sizeof(float)) * splitCoreOp.coexistentNodeNum)),
        FP32_BLOCK_NUM);

    // 对除最后一轴的其他轴切分
    splitCoreOp.normalCoreProcessNum = CeilDivide(baseInfoOp.fusedFrontAxis, baseInfoOp.vectorCoreNum);
    splitCoreOp.usedCoreNum = CeilDivide(baseInfoOp.fusedFrontAxis, splitCoreOp.normalCoreProcessNum);
    splitCoreOp.tailCoreProcessNum =
        baseInfoOp.fusedFrontAxis - splitCoreOp.normalCoreProcessNum * (splitCoreOp.usedCoreNum - 1);
    // mulRowsInUb，UB中处理row的数量
    int64_t mulRowsInUb = splitCoreOp.coexistentNodeElementNum / baseInfoOp.endAxisLenAligned;
    // 满核且UB只能放下一个row，走STATIC_FUNCTION_TEMPLATE
    if (baseInfoOp.fusedFrontAxis >= baseInfoOp.vectorCoreNum && mulRowsInUb < TWO_END_AXIS) {
        splitCoreOp.templateMode = STATIC_FUNCTION_TEMPLATE; // 满核模式 正常尾轴或者大尾轴
        return ge::GRAPH_SUCCESS;
    }

    splitCoreOp.templateMode = STATIC_PERFORMANCE_TEMPLATE;
    if (baseInfoOp.fusedFrontAxis >= baseInfoOp.vectorCoreNum) { // 满核模式 小尾轴
        return DoStaticQuantFullKernelSmallEndAxis();
        // 前轴不足64，非满核模式
    } else {
        return DoStaticQuantNotFullKernelSplitEndAxis();
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GeluQuantRegbaseTiling::DoDynamicQuantTiling()
{
    OP_LOGD(nodeName_, "DoDynamicQuantTiling start running.");
    splitCoreOp.normalCoreProcessNum = CeilDivide(baseInfoOp.fusedFrontAxis, baseInfoOp.vectorCoreNum);
    splitCoreOp.usedCoreNum = CeilDivide(baseInfoOp.fusedFrontAxis, splitCoreOp.normalCoreProcessNum);
    splitCoreOp.tailCoreProcessNum =
        baseInfoOp.fusedFrontAxis - splitCoreOp.normalCoreProcessNum * (splitCoreOp.usedCoreNum - 1);

    splitCoreOp.coexistentNodeNum = DYNAMIC_QUANT_COEXISTING_QUANTITY_DB;
    splitCoreOp.coexistentNodeElementNum = AlignToFloor(
        (SafeDivide(
            static_cast<int64_t>(baseInfoOp.ubSize),
            static_cast<int64_t>(sizeof(float)) * splitCoreOp.coexistentNodeNum)),
        FP32_BLOCK_NUM);

    int64_t mulRowsInUb = splitCoreOp.coexistentNodeElementNum / baseInfoOp.endAxisLenAligned;

    if (mulRowsInUb == 0) {
        splitCoreOp.templateMode = DYNAMIC_WORKSPACE_TEMPLATE;
    } else {
        splitCoreOp.templateMode = DYNAMIC_NORMAL_TEMPLATE;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GeluQuantRegbaseTiling::DoTiling()
{
    OP_LOGD(nodeName_, "DoTiling start running.");
    if (baseInfoOp.quantMode == STATIC_QUANT_MODE) {
        DoStaticQuantTiling();
    } else {
        DoDynamicQuantTiling();
    }

    OP_LOGD(nodeName_, "DoTiling run completed.");
    return ge::GRAPH_SUCCESS;
}

void GeluQuantRegbaseTiling::SaveToTilingData()
{
    tilingData.set_usedCoreNum(splitCoreOp.usedCoreNum);
    tilingData.set_normalCoreProcessNum(splitCoreOp.normalCoreProcessNum);
    tilingData.set_tailCoreProcessNum(splitCoreOp.tailCoreProcessNum);
    tilingData.set_coexistentNodeNum(splitCoreOp.coexistentNodeNum);
    tilingData.set_coexistentNodeElementNum(splitCoreOp.coexistentNodeElementNum);
    tilingData.set_rowInner(splitCoreOp.rowInner);
    tilingData.set_rowOuter(splitCoreOp.rowOuter);
    tilingData.set_rowTail(splitCoreOp.rowTail);
    tilingData.set_colInner(splitCoreOp.colInner);
    tilingData.set_colOuter(splitCoreOp.colOuter);
    tilingData.set_colTail(splitCoreOp.colTail);
    tilingData.set_tilingKey(splitCoreOp.tilingKey);

    tilingData.set_endAxisLen(baseInfoOp.endAxisLen);
    tilingData.set_endAxisLenAligned(baseInfoOp.endAxisLenAligned);
    tilingData.set_quantMode(baseInfoOp.quantMode);
    tilingData.set_approximate(baseInfoOp.approximate);
    tilingData.set_inputScaleType(baseInfoOp.inputScaleType);
    tilingData.set_inputOffsetType(baseInfoOp.inputOffsetType);
    tilingData.set_dstType(baseInfoOp.dstType);
    tilingData.set_roundMode(static_cast<uint32_t>(baseInfoOp.roundMode));
}

ge::graphStatus GeluQuantRegbaseTiling::PostTiling()
{
    OP_LOGD(nodeName_, "PostTiling start running.");
    size_t* userWorkspaceSize = context_->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, userWorkspaceSize);
    size_t workspaceSize = WORKSPACE_BUFFER;
    if (splitCoreOp.templateMode == DYNAMIC_WORKSPACE_TEMPLATE) {
        workspaceSize += baseInfoOp.endAxisLen * sizeof(float) * splitCoreOp.usedCoreNum;
    }

    userWorkspaceSize[0] = workspaceSize;

    splitCoreOp.tilingKey = GetTilingKey();
    SaveToTilingData();

    context_->SetBlockDim(splitCoreOp.usedCoreNum);
    if (tilingData.GetDataSize() > context_->GetRawTilingData()->GetCapacity()) {
        return ge::GRAPH_FAILED;
    }

    tilingData.SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(tilingData.GetDataSize());
    context_->SetTilingKey(splitCoreOp.tilingKey);
    OP_LOGD(nodeName_, "PostTiling run completed");
    return ge::GRAPH_SUCCESS;
}

uint64_t GeluQuantRegbaseTiling::GetTilingKey() const
{
    OP_LOGD(nodeName_, "GetTilingKey start running.");
    InputDataType inputDataType = InputDataType::FLOAT_FLOAT;
    if (baseInfoOp.scaleInputDtype == ge::DT_FLOAT16) {
        inputDataType = InputDataType::HALF_HALF;
    } else if (baseInfoOp.scaleInputDtype == ge::DT_BF16) {
        inputDataType = InputDataType::BF16_BF16;
    } else if (baseInfoOp.xInputDtype == ge::DT_FLOAT) {
        inputDataType = InputDataType::FLOAT_FLOAT;
    } else if (baseInfoOp.xInputDtype == ge::DT_FLOAT16) {
        inputDataType = InputDataType::HALF_FLOAT;
    } else {
        inputDataType = InputDataType::BF16_FLOAT;
    }
    uint64_t tilingKey = 1000UL + 10UL * splitCoreOp.templateMode + static_cast<uint64_t>(inputDataType);
    OP_LOGD(nodeName_, "GetTilingKey [%lu].", tilingKey);
    return tilingKey;
}

void GeluQuantRegbaseTiling::DumpTilingInfo() const
{
    OP_LOGD(nodeName_, "DumpTilingInfo start running");

    std::ostringstream info;
    info << "GeluQuantRegbaseTiling input info: " << std::endl;
    info << "baseInfoOp.vectorCoreNum: " << baseInfoOp.vectorCoreNum << std::endl;
    info << "baseInfoOp.ubSize: " << baseInfoOp.ubSize << std::endl;
    info << "baseInfoOp.xDimNum: " << baseInfoOp.xDimNum << std::endl;
    info << "baseInfoOp.endAxisLen: " << baseInfoOp.endAxisLen << std::endl;
    info << "baseInfoOp.endAxisLenAligned: " << baseInfoOp.endAxisLenAligned << std::endl;
    info << "baseInfoOp.fusedFrontAxis: " << baseInfoOp.fusedFrontAxis << std::endl;
    info << "baseInfoOp.fusedAllAxis: " << baseInfoOp.fusedAllAxis << std::endl;
    info << "baseInfoOp.elementNumAlign: " << baseInfoOp.elementNumAlign << std::endl;
    info << "dtype map: 0 [float]  1 [float16]  27 [bf16]  " << std::endl;
    info << "baseInfoOp.xInputDtype: " << baseInfoOp.xInputDtype << std::endl;
    info << "baseInfoOp.scaleInputDtype: " << baseInfoOp.scaleInputDtype << std::endl;
    info << "baseInfoOp.offsetInputDtype: " << baseInfoOp.offsetInputDtype << std::endl;
    info << "baseInfoOp.quantMode: " << baseInfoOp.quantMode << " [0:static  1:dynamic] " << std::endl;
    info << "baseInfoOp.approximate: " << baseInfoOp.approximate << " [0:none  1:tanh] " << std::endl;
    info << "input type map: 0 [empty]  1 [scalar]  2 [normal]  " << std::endl;
    info << "baseInfoOp.inputScaleType: " << baseInfoOp.inputScaleType << std::endl;
    info << "baseInfoOp.inputOffsetType: " << baseInfoOp.inputOffsetType << std::endl;
    OP_LOGI(nodeName_, "%s", info.str().c_str());
    info.str("");

    info << "GeluQuantRegbaseTiling split info: " << std::endl;
    info << "splitCoreOp.usedCoreNum: " << splitCoreOp.usedCoreNum << std::endl;
    info << "splitCoreOp.normalCoreProcessNum: " << splitCoreOp.normalCoreProcessNum << std::endl;
    info << "splitCoreOp.tailCoreProcessNum: " << splitCoreOp.tailCoreProcessNum << std::endl;
    info << "splitCoreOp.coexistentNodeNum: " << splitCoreOp.coexistentNodeNum << std::endl;
    info << "splitCoreOp.coexistentNodeElementNum: " << splitCoreOp.coexistentNodeElementNum << std::endl;
    info << "templateMode: 0 [static_per_tensor]  1 [static_function]  2 [static_performance]  3 [dynamic_normal]  4 "
            "[dynamic_workspace] "
         << std::endl;
    info << "splitCoreOp.templateMode: " << splitCoreOp.templateMode << std::endl;
    info << "splitCoreOp.rowInner: " << splitCoreOp.rowInner << std::endl;
    info << "splitCoreOp.rowOuter: " << splitCoreOp.rowOuter << std::endl;
    info << "splitCoreOp.rowTail: " << splitCoreOp.rowTail << std::endl;
    info << "splitCoreOp.colInner: " << splitCoreOp.colInner << std::endl;
    info << "splitCoreOp.colOuter: " << splitCoreOp.colOuter << std::endl;
    info << "splitCoreOp.colTail: " << splitCoreOp.colTail << std::endl;
    info << "splitCoreOp.tilingKey: " << splitCoreOp.tilingKey << std::endl;

    OP_LOGI(nodeName_, "%s", info.str().c_str());
}

ge::graphStatus GeluQuantRegbaseTiling::RunGeluQuantRegbaseTiling()
{
    OP_LOGD(nodeName_, "RunGeluQuantRegbaseTiling start running");
    ge::graphStatus ret = ge::GRAPH_SUCCESS;
    ret = GetInputInfo();
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }

    ret = GetPlatformInfo();
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }

    ret = DoTiling();
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }

    ret = PostTiling();
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }

    DumpTilingInfo();
    return ge::GRAPH_SUCCESS;
}
} // namespace geluquantregbase
} // namespace optiling

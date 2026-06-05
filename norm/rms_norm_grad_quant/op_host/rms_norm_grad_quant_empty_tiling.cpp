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
 * \file rms_norm_grad_quant_empty_tiling.cpp
 * \brief RmsNormGradQuant empty tiling file
 */

#include "op_common/log/log.h"
#include "op_common/op_host/util/platform_util.h"
#include "util/math_util.h"
#include "rms_norm_grad_quant_tiling.h"

namespace optiling {
using namespace Ops::Base;
static constexpr int64_t INPUT_NUM = 4;
static constexpr int64_t INPUT_INDEX_0 = 0;
static constexpr int64_t INPUT_INDEX_1 = 1;
static constexpr int64_t INPUT_INDEX_2 = 2;
static constexpr int64_t INPUT_INDEX_3 = 3;
static constexpr int64_t INPUT_INDEX_4 = 4;
static constexpr int64_t INPUT_INDEX_5 = 5;
static constexpr int64_t ATTR_INDEX_0 = 0;
static constexpr int64_t ATTR_INDEX_1 = 1;
static constexpr int64_t STATIC_QUANT_MODE = 0;
static constexpr int64_t TWO = 2;
static constexpr int64_t BUFFER_NUM = 2;
static constexpr int64_t FLOATBYTESIZE = 4;
static constexpr int64_t MAX_CORE_COLS = 8192;
static constexpr uint32_t EMPTY_TENSOR_KEY = 8000;
static const gert::Shape g_vec_1_shape = {1};

const std::string OP_NAME = "RmsNormGradQuant";
static const std::vector<std::string> inputNames = {"dy", "x", "rstd", "gamma"};

ge::graphStatus RmsNormGradQuantEmptyTiling::GetPlatformInfo()
{
    auto platformInfo = context_->GetPlatformInfo(); // 判断条件修改
    if (platformInfo != nullptr) {
        auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
        aivCoreNum_ = ascendcPlatform.GetCoreNumAiv();
        uint64_t ubSizePlatform;
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatform);
        ubSize_ = ubSizePlatform;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus RmsNormGradQuantEmptyTiling::CheckShapeAllPositive(gert::Shape& shape)
{
    for (size_t i = 0; i < shape.GetDimNum(); i++) {
        OP_CHECK_IF(
            shape.GetDim(i) < 0,
            OP_LOGE(
                context_->GetNodeName(), "Dim %lu of input should be positive, but actual %ld.", i, shape.GetDim(i)),
            return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus RmsNormGradQuantEmptyTiling::CheckShapesEqual(gert::Shape& shape0, gert::Shape& shape1)
{
    OP_CHECK_IF(
        shape0.GetDimNum() != shape1.GetDimNum(),
        OP_LOGE(
            context_->GetNodeName(), "DimNum of shapes are not equal: %lu vs %lu", shape0.GetDimNum(),
            shape1.GetDimNum()),
        return ge::GRAPH_FAILED);

    for (size_t i = 0; i < shape0.GetDimNum(); i++) {
        OP_CHECK_IF(
            shape0.GetDim(i) != shape1.GetDim(i),
            OP_LOGE(
                context_->GetNodeName(), "Dim %lu of shapes are not equal: %ld vs %ld", i, shape0.GetDim(i),
                shape1.GetDim(i)),
            return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

const gert::Shape& RmsNormGradQuantEmptyTiling::EnsureNotScalar(const gert::Shape& inShape)
{
    if (inShape.IsScalar()) {
        return g_vec_1_shape;
    }
    return inShape;
}

void RmsNormGradQuantEmptyTiling::CalcRowsAndCols(gert::Shape& gammaShape)
{
    rows_ = 0;
    cols_ = 1;
    for (size_t i = 0; i < gammaShape.GetDimNum(); i++) {
        cols_ *= gammaShape.GetDim(i);
    }
}

ge::graphStatus RmsNormGradQuantEmptyTiling::CheckInputsShape()
{
    // check dy
    auto inputShape = context_->GetInputShape(INPUT_INDEX_0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, inputShape);
    auto storageShape0 = EnsureNotScalar(inputShape->GetStorageShape());
    if (CheckShapeAllPositive(storageShape0) != ge::GRAPH_SUCCESS) {
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context_->GetNodeName(), "dy", ToString(storageShape0).c_str(),
            "The shape of input dy can not be an invalid tensor with a negative dim");
        return ge::GRAPH_FAILED;
    }

    // check x
    inputShape = context_->GetInputShape(INPUT_INDEX_1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, inputShape);
    auto storageShape1 = EnsureNotScalar(inputShape->GetStorageShape());
    if (CheckShapeAllPositive(storageShape1) != ge::GRAPH_SUCCESS) {
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context_->GetNodeName(), "x", ToString(storageShape1).c_str(),
            "The shape of input x can not be an invalid tensor with a negative dim");
        return ge::GRAPH_FAILED;
    }

    // check shapes of input0 and input1 are equal
    if (CheckShapesEqual(storageShape0, storageShape1) != ge::GRAPH_SUCCESS) {
        std::string shapeMsg = ToString(storageShape0) + " and " + ToString(storageShape1);
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(context_->GetNodeName(), "dy and x", shapeMsg.c_str(),
            "The shapes of input dy and input x should be the same");
        return ge::GRAPH_FAILED;
    }

    // check rstd
    inputShape = context_->GetInputShape(INPUT_INDEX_2);
    OP_CHECK_NULL_WITH_CONTEXT(context_, inputShape);
    auto storageShape2 = EnsureNotScalar(inputShape->GetStorageShape());
    if (CheckShapeAllPositive(storageShape2) != ge::GRAPH_SUCCESS) {
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context_->GetNodeName(), "rstd", ToString(storageShape2).c_str(),
            "The shape of input rstd can not be an invalid tensor with a negative dim");
        return ge::GRAPH_FAILED;
    }

    // check gamma
    inputShape = context_->GetInputShape(INPUT_INDEX_3);
    OP_CHECK_NULL_WITH_CONTEXT(context_, inputShape);
    auto storageShape3 = EnsureNotScalar(inputShape->GetStorageShape());
    if (CheckShapeAllPositive(storageShape3) != ge::GRAPH_SUCCESS) {
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context_->GetNodeName(), "gamma", ToString(storageShape3).c_str(),
            "The shape of input gamma can not be an invalid tensor with a negative dim");
        return ge::GRAPH_FAILED;
    }

    CalcRowsAndCols(storageShape3);
    OP_CHECK_IF(
        cols_ == 0,
        OP_LOGE(context_->GetNodeName(), "The shape of input gamma should not be zero."),
        return ge::GRAPH_FAILED);

    // check scalesX (required input)
    inputShape = context_->GetInputShape(INPUT_INDEX_4);
    OP_CHECK_NULL_WITH_CONTEXT(context_, inputShape);
    {
        auto storageShape4 = inputShape->GetStorageShape();
        if (storageShape4.GetDimNum() != 1 || storageShape4.GetDim(0) != 1) {
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                context_->GetNodeName(), "scalesX", ToString(storageShape4).c_str(),
                "The shape of input scalesX should be same with [1]");
            return ge::GRAPH_FAILED;
        }
    }

    // check offsetX
    inputShape = context_->GetInputShape(INPUT_INDEX_5);
    if (inputShape != nullptr) {
        hasOffsetX_ = rms_norm_grad_quant::ComputeModeOffsetX::WITH_OFFSET_X;
        auto storageShape5 = inputShape->GetStorageShape();
        if (storageShape5.GetDimNum() != 1 || storageShape5.GetDim(0) != 1) {
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                context_->GetNodeName(), "offsetX", ToString(storageShape5).c_str(),
                "The shape of input offsetX should be same with [1]");
            return ge::GRAPH_FAILED;
        }
    } else {
        hasOffsetX_ = rms_norm_grad_quant::ComputeModeOffsetX::WITHOUT_OFFSET_X;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus RmsNormGradQuantEmptyTiling::CheckInputsDtype()
{
    for (int i = 0; i < INPUT_NUM; i++) {
        auto inputDesc = context_->GetInputDesc(i);
        OP_CHECK_NULL_WITH_CONTEXT(context_, inputDesc);
        // check dtype
        auto dtype = inputDesc->GetDataType();
        if (dtype != ge::DataType::DT_FLOAT16 && dtype != ge::DataType::DT_BF16 && dtype != ge::DataType::DT_FLOAT) {
            OP_LOGE_FOR_INVALID_DTYPE(context_->GetNodeName(), inputNames[i].c_str(),
                ToString(dtype).c_str(), "FLOAT, FLOAT16 or BF16");
            return ge::GRAPH_FAILED;
        }
    }

    // check scalesX (required input)
    auto scaleXDesc = context_->GetInputDesc(INPUT_INDEX_4);
    OP_CHECK_NULL_WITH_CONTEXT(context_, scaleXDesc);
    {
        auto scaleXDtype = scaleXDesc->GetDataType();
        if (scaleXDtype != ge::DataType::DT_FLOAT && scaleXDtype != ge::DataType::DT_BF16 &&
            scaleXDtype != ge::DataType::DT_FLOAT16) {
            std::string dtypeMsg = ToString(scaleXDtype);
            OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
                context_->GetNodeName(), "scalesX", dtypeMsg.c_str(),
                "The dtype of input scalesX should be FLOAT, FLOAT16 or BF16");
            return ge::GRAPH_FAILED;
        }
    }

    // check offsetX
    auto offsetXDesc = context_->GetInputDesc(INPUT_INDEX_5);
    if (offsetXDesc != nullptr) {
        auto offsetXDtype = offsetXDesc->GetDataType();
        if (offsetXDtype != ge::DataType::DT_INT32) {
            std::string dtypeMsg = ToString(offsetXDtype);
            OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
                context_->GetNodeName(), "offsetX", dtypeMsg.c_str(),
                "The dtype of input offsetX should be INT32");
            return ge::GRAPH_FAILED;
        }
    }

    // check dx dtype (scales_x is required, always quant mode)
    auto dxDesc = context_->GetOutputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, dxDesc);
    auto dxDtype = dxDesc->GetDataType();
    OP_CHECK_IF(
        (dxDtype != ge::DataType::DT_HIFLOAT8 && dxDtype != ge::DataType::DT_INT8),
        OP_LOGE_FOR_INVALID_DTYPE(context_->GetNodeName(), "dx", ToString(dxDtype).c_str(), "HIFLOAT8 or INT8"),
        return ge::GRAPH_FAILED);

    // check dgamma dtype
    auto dgammaDesc = context_->GetOutputDesc(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, dgammaDesc);
    auto dgammaDtype = dgammaDesc->GetDataType();
    OP_CHECK_IF(
        (dgammaDtype != ge::DataType::DT_FLOAT),
        OP_LOGE_FOR_INVALID_DTYPE(context_->GetNodeName(), "dgamma", ToString(dgammaDtype).c_str(), "FLOAT"),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus RmsNormGradQuantEmptyTiling::GetShapeAttrsInfo()
{
    if (context_ == nullptr) {
        OP_LOGE(OP_NAME, "Tiling context is null");
        return ge::GRAPH_FAILED;
    }

    // check inputs shape
    if (CheckInputsShape() != ge::GRAPH_SUCCESS) {
        OP_LOGE(context_->GetNodeName(), "Inputs shape invalid.");
        return ge::GRAPH_FAILED;
    }
    // check inputs dtype
    if (CheckInputsDtype() != ge::GRAPH_SUCCESS) {
        OP_LOGE(context_->GetNodeName(), "Inputs dtype invalid.");
        return ge::GRAPH_FAILED;
    }

    // set Attrs
    auto attrs = context_->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context_, attrs);
    // check quant
    const char* quantMode = attrs->GetAttrPointer<char>(ATTR_INDEX_0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, quantMode);
    if (strcmp(quantMode, "static") == 0) {
        quantMode_ = STATIC_QUANT_MODE;
    } else {
        OP_CHECK_IF(
            (true), OP_LOGE(context_->GetNodeName(), "the attr of quant mode should be static."),
            return ge::GRAPH_FAILED);
    }
    // set divMode
    const bool* divModePtr = attrs->GetBool(ATTR_INDEX_1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, divModePtr);
    divMode_ = *divModePtr ? rms_norm_grad_quant::ComputeModeDivMode::DIV_MODE
                           : rms_norm_grad_quant::ComputeModeDivMode::NOT_DIV_MODE;
    return ge::GRAPH_SUCCESS;
}

bool RmsNormGradQuantEmptyTiling::IsCapable()
{
    const gert::StorageShape* dy_shape = context_->GetInputShape(0);
    const gert::StorageShape* gamma = context_->GetInputShape(3);
    auto gamma_dim_num = gamma->GetStorageShape().GetDimNum();
    auto dyDimNum = dy_shape->GetStorageShape().GetDimNum();
    if (dyDimNum < gamma_dim_num) {
        return false;
    }
    int64_t row_val = 1;
    for (uint32_t i = 0; i < dyDimNum - gamma_dim_num; i++) {
        row_val *= dy_shape->GetStorageShape().GetDim(i);
    }
    if (row_val == 0) {
        return true;// 高优先级
    }
    return false;
}

uint64_t RmsNormGradQuantEmptyTiling::GetTilingKey() const
{
    rms_norm_grad_quant::RmsNormGradQuantTilingKey tilingKey;
    tilingKey.SetComputeModeDgamma(computeModeDgamma_);
    return tilingKey.GetTilingKey();
}

ge::graphStatus RmsNormGradQuantEmptyTiling::PostTiling()
{
    context_->SetBlockDim(aivCoreNum_);
    auto rawTilingData = context_->GetRawTilingData();
    OP_CHECK_IF(
        sizeof(tilingData_) > rawTilingData->GetCapacity(),
        OP_LOGE(
            context_->GetNodeName(), "actual tiling data size %zu > context tiling data size %zu", sizeof(tilingData_),
            rawTilingData->GetCapacity()),
        return ge::GRAPH_FAILED);
    auto capSize = rawTilingData->GetCapacity();
    void* ptrData = rawTilingData->GetData();
    OP_CHECK_NULL_WITH_CONTEXT(context_, ptrData);
    void* ptrStruct = static_cast<void*>(&tilingData_);
    OP_CHECK_NULL_WITH_CONTEXT(context_, ptrStruct);
    OP_CHECK_IF(
        memcpy_s(ptrData, capSize, ptrStruct, sizeof(tilingData_)) != 0,
        OP_LOGE(context_->GetNodeName(), "Set tiling data is failed!"), return ge::GRAPH_FAILED);
    rawTilingData->SetDataSize(sizeof(tilingData_));

    size_t* currentWorkspace = context_->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, currentWorkspace);
    currentWorkspace[0] = workspaceSize_;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus RmsNormGradQuantEmptyTiling::DoLibApiTiling()
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus RmsNormGradQuantEmptyTiling::GetWorkspaceSize()
{
    fe::PlatFormInfos* platformInfoPtr = context_->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context_, platformInfoPtr);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    workspaceSize_ = ascendcPlatform.GetLibApiWorkSpaceSize();
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus RmsNormGradQuantEmptyTiling::CalcUsedCoreNumGamma()
{
    if (cols_ <= MAX_CORE_COLS) {
        // 单核计算单colset
        usedCoreNumDG_ = 1;
        colsPerCoreDG_ = cols_;
        colsLastCoreDG_ = cols_;
        coreUbBlockCount_ = 0;
        tailUbCols_ = colsPerCoreDG_;
        colsPerUBDG_ = colsPerCoreDG_;
        lastCoreBlockCount_ = 0;
        lastCoreTailUbCols_ = colsLastCoreDG_;
    } else {
        // 多核计算
        ge::graphStatus result = ge::GRAPH_SUCCESS;
        usedCoreNumDG_ = aivCoreNum_;
        colsPerCoreDG_ = Ops::Base::CeilDiv(cols_, usedCoreNumDG_);
        colsPerUBDG_ = colsPerCoreDG_;
        colsLastCoreDG_ = cols_ - colsPerCoreDG_ * (usedCoreNumDG_ - 1);
        result = CalcTilingDataDgamma();
        return result;
    }
    return ge::GRAPH_SUCCESS;
}

int32_t RmsNormGradQuantEmptyTiling::NearestLowerPowerOfTwo(int32_t temp)
{
    int64_t power = 0;
    int32_t powerofTwoValueDG = 0;
    while (power <= temp) {
        powerofTwoValueDG += 1;
        power = std::pow(TWO, powerofTwoValueDG);
    }
    return powerofTwoValueDG - 1;
}

ge::graphStatus RmsNormGradQuantEmptyTiling::CalcTilingDataDgamma()
{
    if (static_cast<uint64_t>(ubSize_) >= BUFFER_NUM * (colsPerCoreDG_ * FLOATBYTESIZE)) {
        // full-load
        coreUbBlockCount_ = 0;
        tailUbCols_ = colsPerCoreDG_;
        lastCoreBlockCount_ = 0;
        lastCoreTailUbCols_ = colsLastCoreDG_;
    } else {
        //UB最多存放多少列
        int maxRowsNumDG_ = ubSize_ / (BUFFER_NUM * FLOATBYTESIZE);
        OP_CHECK_IF(
            maxRowsNumDG_ <= 0, OP_LOGE(context_->GetNodeName(), "The maxRowsNumDG_ size is neg: %d.", maxRowsNumDG_),
            return ge::GRAPH_FAILED);
        colsPerUBDG_ = std::pow(TWO, NearestLowerPowerOfTwo(maxRowsNumDG_));
        OP_CHECK_IF(
            colsPerUBDG_ <= 0, OP_LOGE(context_->GetNodeName(), "The colsPerUBDG_ size is invalid: %d.", colsPerUBDG_),
            return ge::GRAPH_FAILED);
        coreUbBlockCount_ = (colsPerCoreDG_ + colsPerUBDG_ -1) / colsPerUBDG_ - 1;
        tailUbCols_ = colsPerCoreDG_ - colsPerUBDG_ * coreUbBlockCount_;
        if (colsPerUBDG_ > colsLastCoreDG_) {
            lastCoreBlockCount_ = 0;
            lastCoreTailUbCols_ = colsLastCoreDG_;
        } else {
            lastCoreBlockCount_ = (colsLastCoreDG_ + colsPerUBDG_ -1) / colsPerUBDG_ - 1;
            lastCoreTailUbCols_ = colsLastCoreDG_ - colsPerUBDG_ * lastCoreBlockCount_;
        }
    }
    return ge::GRAPH_SUCCESS;
}

void RmsNormGradQuantEmptyTiling::LogTilingResult()
{
    OP_LOGI(OP_NAME, "rows: %ld, cols_: %ld", rows_, cols_);
}

ge::graphStatus RmsNormGradQuantEmptyTiling::DoOpTiling()
{
    tilingKey_ = EMPTY_TENSOR_KEY;
    computeModeDx_ = rms_norm_grad_quant::ComputeModeDx::FULL_LOAD;
    computeModeDgamma_ = rms_norm_grad_quant::ComputeModeDgamma::EMPTY;
    // split cores
    ge::graphStatus result = CalcUsedCoreNumGamma();
    if (result != ge::GRAPH_SUCCESS) {
        return result;
    }

    tilingData_.colsPerUBDG = colsPerUBDG_;
    tilingData_.tailUbCols = tailUbCols_;
    tilingData_.coreUbBlockCount = coreUbBlockCount_;
    tilingData_.lastCoreBlockCount = lastCoreBlockCount_;
    tilingData_.lastCoreTailUbCols = lastCoreTailUbCols_;
    tilingData_.ubSize = ubSize_;
    tilingData_.cols = cols_;
    tilingData_.usedCoreNumDG = usedCoreNumDG_;
    tilingData_.colsPerCoreDG = colsPerCoreDG_;
    tilingData_.colsLastCoreDG = colsLastCoreDG_;
    LogTilingResult();
    return ge::GRAPH_SUCCESS;
}

REGISTER_OPS_TILING_TEMPLATE(RmsNormGradQuant, RmsNormGradQuantEmptyTiling, 100);
} // namespace optiling
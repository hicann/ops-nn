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
 * \file rms_norm_grad_quant_regbase_tiling.cpp
 * \brief RmsNormGradQuant regbase tiling file
 */

#include "op_common/log/log.h"
#include "op_common/op_host/util/platform_util.h"
#include "util/math_util.h"
#include "rms_norm_grad_quant_tiling.h"

namespace optiling {
using namespace Ops::Base;
using namespace rms_norm_grad_quant;
constexpr int64_t MIN_DATANUM_PER_CORE = 1024;
constexpr int64_t UB_RESERVED_SIZE = 256;
constexpr int64_t INPUT_NUM = 4;
constexpr int64_t OUTPUT_NUM = 2;
constexpr int64_t BUFFER_NUM = 2;
constexpr int64_t ATTR_INDEX_0 = 0;
constexpr int64_t ATTR_INDEX_1 = 1;
constexpr int64_t INPUT_INDEX_0 = 0;
constexpr int64_t INPUT_INDEX_1 = 1;
constexpr int64_t INPUT_INDEX_2 = 2;
constexpr int64_t INPUT_INDEX_3 = 3;
constexpr int64_t INPUT_INDEX_4 = 4;
constexpr int64_t INPUT_INDEX_5 = 5;
constexpr int64_t TWO = 2;
constexpr int64_t THREE = 3;
constexpr int64_t FLOAT_BYTE_SIZE = 4;
constexpr int64_t HALF_BYTE_SIZE = 2;
constexpr int64_t INT8_BYTE_SIZE = 1;
constexpr int64_t DOUBLE_VL = 2;
constexpr int64_t DGAMMA_VL_BUFFER_COUNT = 4; // dy + x + dgamma + dgamma_tmp
constexpr int64_t DX_UB_FACTOR = 6144;
constexpr uint32_t REG_BASE_KEY = 7000;
constexpr uint32_t DX_SPLIT_KEY = 10;
constexpr uint32_t DG_SPLIT_KEY = 1;
constexpr uint32_t STATIC_QUANT_MODE = 0;
constexpr uint32_t LOG_2 = 2;
constexpr int64_t RETAINED_SIZE_1K = 1024;
constexpr int64_t MIN_DIMS_DY_X_DX = 2;
constexpr int64_t MAX_DIMS_DY_X_DX = 8;

const std::string OP_NAME = "RmsNormGradQuant";

ge::graphStatus RmsNormGradQuantRegbaseTiling::GetPlatformInfo()
{
    auto platformInfo = context_->GetPlatformInfo(); // 判断条件修改
    if (platformInfo != nullptr) {
        auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
        uint64_t ubSizePlatform;
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatform);
        ubSize_ = ubSizePlatform;
        aivCoreNum_ = ascendcPlatform.GetCoreNumAiv();
        blockSize_ = Ops::Base::GetUbBlockSize(context_);
        vecRegSize_ = Ops::Base::GetVRegSize(context_);
    }
    vlFp32_ = vecRegSize_ / sizeof(float);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus RmsNormGradQuantRegbaseTiling::CheckShapeAllPositive(const gert::Shape& shape)
{
    for (size_t i = 0; i < shape.GetDimNum(); i++) {
        OP_CHECK_IF(shape.GetDim(i) <= 0,
                    OP_LOGE(context_->GetNodeName(), "Dim %lu of input should be positive, but actual %ld.", i,
                            shape.GetDim(i)),
                    return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus RmsNormGradQuantRegbaseTiling::CheckShapeBeSameWithOne(const gert::Shape& shape)
{
    OP_CHECK_IF(shape.GetDimNum() != 1,
                OP_LOGE(context_->GetNodeName(), "DimNum of shapes are not equal: %zu vs 1", shape.GetDimNum()),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(shape.GetDim(0) != 1,
                OP_LOGE(context_->GetNodeName(), "Dim[0] of shapes are not equal: %ld vs 1", shape.GetDim(0)),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus RmsNormGradQuantRegbaseTiling::CheckShapesEqual(const gert::Shape& shape0, const gert::Shape& shape1)
{
    OP_CHECK_IF(shape0.GetDimNum() != shape1.GetDimNum(),
                OP_LOGE(context_->GetNodeName(), "DimNum of shapes are not equal: %zu vs %zu", shape0.GetDimNum(),
                        shape1.GetDimNum()),
                return ge::GRAPH_FAILED);

    for (size_t i = 0; i < shape0.GetDimNum(); i++) {
        OP_CHECK_IF(shape0.GetDim(i) != shape1.GetDim(i),
                    OP_LOGE(context_->GetNodeName(), "Dim %lu of shapes are not equal: %ld vs %ld", i, shape0.GetDim(i),
                            shape1.GetDim(i)),
                    return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus RmsNormGradQuantRegbaseTiling::CheckDyAndXShapes(gert::Shape& dyShape)
{
    // check dy
    auto inputShape = context_->GetInputShape(INPUT_INDEX_0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, inputShape);
    dyShape = inputShape->GetStorageShape();
    if (CheckShapeAllPositive(dyShape) != ge::GRAPH_SUCCESS) {
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
            context_->GetNodeName(), "dy", ToString(dyShape).c_str(),
            "The shape of input dy can not be an empty tensor or an invalid tensor with a negative dim");
        return ge::GRAPH_FAILED;
    }

    auto dimNum = static_cast<int64_t>(dyShape.GetDimNum());
    OP_CHECK_IF(dimNum < MIN_DIMS_DY_X_DX || dimNum > MAX_DIMS_DY_X_DX,
                OP_LOGE(context_->GetNodeName(), "DimNum of %s should be in range [%ld, %ld], but actual %ld.", "dy",
                        MIN_DIMS_DY_X_DX, MAX_DIMS_DY_X_DX, dimNum),
                return ge::GRAPH_FAILED);

    // check x
    inputShape = context_->GetInputShape(INPUT_INDEX_1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, inputShape);
    auto storageShape1 = inputShape->GetStorageShape();
    if (CheckShapeAllPositive(storageShape1) != ge::GRAPH_SUCCESS) {
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
            context_->GetNodeName(), "x", ToString(storageShape1).c_str(),
            "The shape of input x can not be an empty tensor or an invalid tensor with a negative dim");
        return ge::GRAPH_FAILED;
    }

    // check shapes of input0 and input1 are equal
    if (CheckShapesEqual(dyShape, storageShape1) != ge::GRAPH_SUCCESS) {
        std::string shapeMsg = ToString(dyShape) + " and " + ToString(storageShape1);
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(context_->GetNodeName(), "dy and x", shapeMsg.c_str(),
                                               "The shapes of input dy and input x should be the same");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus RmsNormGradQuantRegbaseTiling::CheckRstdGammaAndScalesShapes(gert::Shape& rstdShape,
                                                                             gert::Shape& gammaShape)
{
    // check rstd
    auto inputShape = context_->GetInputShape(INPUT_INDEX_2);
    OP_CHECK_NULL_WITH_CONTEXT(context_, inputShape);
    rstdShape = inputShape->GetStorageShape();
    if (CheckShapeAllPositive(rstdShape) != ge::GRAPH_SUCCESS) {
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
            context_->GetNodeName(), "rstd", ToString(rstdShape).c_str(),
            "The shape of input rstd can not be an empty tensor or an invalid tensor with a negative dim");
        return ge::GRAPH_FAILED;
    }

    // check gamma
    inputShape = context_->GetInputShape(INPUT_INDEX_3);
    OP_CHECK_NULL_WITH_CONTEXT(context_, inputShape);
    gammaShape = inputShape->GetStorageShape();
    if (CheckShapeAllPositive(gammaShape) != ge::GRAPH_SUCCESS) {
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
            context_->GetNodeName(), "gamma", ToString(gammaShape).c_str(),
            "The shape of input gamma can not be an empty tensor or an invalid tensor with a negative dim");
        return ge::GRAPH_FAILED;
    }

    // check scalesX (required input)
    inputShape = context_->GetInputShape(INPUT_INDEX_4);
    OP_CHECK_NULL_WITH_CONTEXT(context_, inputShape);
    {
        auto storageShape4 = inputShape->GetStorageShape();
        if (CheckShapeBeSameWithOne(storageShape4) != ge::GRAPH_SUCCESS) {
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context_->GetNodeName(), "scalesX", ToString(storageShape4).c_str(),
                                                  "The shape of input scalesX should be same with [1]");
            return ge::GRAPH_FAILED;
        }
    }

    // check zeroPintX
    inputShape = context_->GetInputShape(INPUT_INDEX_5);
    if (inputShape != nullptr) {
        hasOffsetX_ = ComputeModeOffsetX::WITH_OFFSET_X;
        auto storageShape5 = inputShape->GetStorageShape();
        if (CheckShapeBeSameWithOne(storageShape5) != ge::GRAPH_SUCCESS) {
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context_->GetNodeName(), "zeroPintX", ToString(storageShape5).c_str(),
                                                  "The shape of input zeroPintX should be same with [1]");
            return ge::GRAPH_FAILED;
        }
    } else {
        hasOffsetX_ = ComputeModeOffsetX::WITHOUT_OFFSET_X;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus RmsNormGradQuantRegbaseTiling::CheckRelationShapes(const gert::Shape& dyShape,
                                                                   const gert::Shape& rstdShape,
                                                                   const gert::Shape& gammaShape)
{
    rows_ = 1;
    cols_ = 1;
    for (size_t i = 0; i < dyShape.GetDimNum() - gammaShape.GetDimNum(); i++) {
        rows_ *= dyShape.GetDim(i);
    }
    for (size_t i = 0; i < gammaShape.GetDimNum(); i++) {
        cols_ *= gammaShape.GetDim(i);
    }
    OP_CHECK_IF(dyShape.GetDimNum() != rstdShape.GetDimNum() + gammaShape.GetDimNum(),
                OP_LOGE(context_->GetNodeName(),
                        "The dimNum of x (%zu) should be equal to the sum of dimNum of rstd (%zu) and gamma (%zu).",
                        dyShape.GetDimNum(), rstdShape.GetDimNum(), gammaShape.GetDimNum()),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(rstdShape.GetDimNum() > dyShape.GetDimNum(),
                OP_LOGE(context_->GetNodeName(), "DimNum of %s (%zu) should not exceed dimNum of %s (%zu).", "rstd",
                        rstdShape.GetDimNum(), "x", dyShape.GetDimNum()),
                return ge::GRAPH_FAILED);
    for (size_t i = 0; i < rstdShape.GetDimNum(); i++) {
        OP_CHECK_IF(rstdShape.GetDim(i) != dyShape.GetDim(i),
                    OP_LOGE(context_->GetNodeName(), "Dim %lu of %s (%ld) should be equal to dim %lu of %s (%ld).", i,
                            "rstd", rstdShape.GetDim(i), i, "x", dyShape.GetDim(i)),
                    return ge::GRAPH_FAILED);
    }
    OP_CHECK_IF(gammaShape.GetDimNum() > dyShape.GetDimNum(),
                OP_LOGE(context_->GetNodeName(), "DimNum of %s (%zu) should not exceed dimNum of %s (%zu).", "gamma",
                        gammaShape.GetDimNum(), "x", dyShape.GetDimNum()),
                return ge::GRAPH_FAILED);
    size_t offset = dyShape.GetDimNum() - gammaShape.GetDimNum();
    for (size_t i = 0; i < gammaShape.GetDimNum(); i++) {
        OP_CHECK_IF(gammaShape.GetDim(i) != dyShape.GetDim(offset + i),
                    OP_LOGE(context_->GetNodeName(), "Dim %lu of %s (%ld) should be equal to dim %zu of %s (%ld).", i,
                            "gamma", gammaShape.GetDim(i), offset + i, "x", dyShape.GetDim(offset + i)),
                    return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus RmsNormGradQuantRegbaseTiling::CheckOutputShapes(const gert::Shape& dyShape,
                                                                 const gert::Shape& gammaShape)
{
    // check output dxOut shape equal to dy shape
    auto outputShape = context_->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, outputShape);
    auto storageShapeDxOut = outputShape->GetStorageShape();
    if (CheckShapesEqual(storageShapeDxOut, dyShape) != ge::GRAPH_SUCCESS) {
        std::string shapeMsg = ToString(storageShapeDxOut) + " and " + ToString(dyShape);
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(context_->GetNodeName(), "dxOut and dy", shapeMsg.c_str(),
                                               "The shape of output dxOut should be the same as the shape of input dy");
        return ge::GRAPH_FAILED;
    }

    // check output dgammaOut shape equal to gamma shape
    outputShape = context_->GetOutputShape(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, outputShape);
    auto storageShapeDgammaOut = outputShape->GetStorageShape();
    if (CheckShapesEqual(storageShapeDgammaOut, gammaShape) != ge::GRAPH_SUCCESS) {
        std::string shapeMsg = ToString(storageShapeDgammaOut) + " and " + ToString(gammaShape);
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
            context_->GetNodeName(), "dgammaOut and gamma", shapeMsg.c_str(),
            "The shape of output dgammaOut should be the same as the shape of input gamma");
        return ge::GRAPH_FAILED;
    }

    OP_CHECK_IF(cols_ == 0, OP_LOGE(context_->GetNodeName(), "The shape of input gamma should not be zero."),
                return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus RmsNormGradQuantRegbaseTiling::CheckInputsShape()
{
    gert::Shape dyShape;
    gert::Shape rstdShape;
    gert::Shape gammaShape;
    if (CheckDyAndXShapes(dyShape) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (CheckRstdGammaAndScalesShapes(rstdShape, gammaShape) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (CheckRelationShapes(dyShape, rstdShape, gammaShape) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return CheckOutputShapes(dyShape, gammaShape);
}

ge::graphStatus RmsNormGradQuantRegbaseTiling::CheckDyAndXDtype()
{
    auto dyDesc = context_->GetInputDesc(INPUT_INDEX_0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, dyDesc);
    dyDtype_ = dyDesc->GetDataType();
    OP_CHECK_IF(
        (dyDtype_ != ge::DataType::DT_FLOAT16 && dyDtype_ != ge::DataType::DT_BF16 &&
         dyDtype_ != ge::DataType::DT_FLOAT),
        OP_LOGE_FOR_INVALID_DTYPE(context_->GetNodeName(), "dy", ToString(dyDtype_).c_str(), "FLOAT, FLOAT16 or BF16"),
        return ge::GRAPH_FAILED);

    auto xDesc = context_->GetInputDesc(INPUT_INDEX_1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, xDesc);
    auto xDtype = xDesc->GetDataType();
    if (dyDtype_ != xDtype) {
        std::string dtypeMsg = ToString(xDtype) + " and " + ToString(dyDtype_);
        OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(context_->GetNodeName(), "x and dy", dtypeMsg.c_str(),
                                               "The dtypes of input x and input dy should be the same");
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus RmsNormGradQuantRegbaseTiling::CheckRstdGammaAndScaleXDtype()
{
    auto rstdDesc = context_->GetInputDesc(INPUT_INDEX_2);
    OP_CHECK_NULL_WITH_CONTEXT(context_, rstdDesc);
    auto rstdDtype = rstdDesc->GetDataType();
    OP_CHECK_IF((rstdDtype != ge::DataType::DT_FLOAT),
                OP_LOGE_FOR_INVALID_DTYPE(context_->GetNodeName(), "rstd", ToString(rstdDtype).c_str(), "FLOAT"),
                return ge::GRAPH_FAILED);

    auto gammaDesc = context_->GetInputDesc(INPUT_INDEX_3);
    OP_CHECK_NULL_WITH_CONTEXT(context_, gammaDesc);
    gammaDtype_ = gammaDesc->GetDataType();
    if (gammaDtype_ != ge::DataType::DT_FLOAT && gammaDtype_ != dyDtype_) {
        std::string dtypeMsg = ToString(gammaDtype_);
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
            context_->GetNodeName(), "gamma", dtypeMsg.c_str(),
            "The dtype of input gamma should be FLOAT or the same as the dtype of input dy");
        return ge::GRAPH_FAILED;
    }

    // check scaleX (required input)
    auto scaleXDesc = context_->GetInputDesc(INPUT_INDEX_4);
    OP_CHECK_NULL_WITH_CONTEXT(context_, scaleXDesc);
    {
        auto scaleXDtype = scaleXDesc->GetDataType();
        if (scaleXDtype != ge::DataType::DT_FLOAT && scaleXDtype != dyDtype_) {
            std::string dtypeMsg = ToString(scaleXDtype);
            OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
                context_->GetNodeName(), "scaleX", dtypeMsg.c_str(),
                "The dtype of input scaleX should be FLOAT or the same as the dtype of input dy");
            return ge::GRAPH_FAILED;
        }
    }

    // check offsetX
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus RmsNormGradQuantRegbaseTiling::CheckOffsetAndOutputDtype()
{
    auto regbaseOffsetXDesc = context_->GetInputDesc(INPUT_INDEX_5);
    if (regbaseOffsetXDesc != nullptr) {
        auto regbaseOffsetXDtype = regbaseOffsetXDesc->GetDataType();
        if (regbaseOffsetXDtype != ge::DataType::DT_INT32) {
            std::string regbaseOffsetDtypeMsg = ToString(regbaseOffsetXDtype);
            OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(context_->GetNodeName(), "offsetX", regbaseOffsetDtypeMsg.c_str(),
                                                  "The dtype of input offsetX should be INT32");
            return ge::GRAPH_FAILED;
        }
    }

    auto dxDesc = context_->GetOutputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, dxDesc);
    auto dxDtype = dxDesc->GetDataType();
    // scales_x is required, always quant mode
    OP_CHECK_IF((dxDtype != ge::DataType::DT_HIFLOAT8 && dxDtype != ge::DataType::DT_INT8),
                OP_LOGE_FOR_INVALID_DTYPE(context_->GetNodeName(), "dx", ToString(dxDtype).c_str(), "HIFLOAT8 or INT8"),
                return ge::GRAPH_FAILED);

    auto dgammaDesc = context_->GetOutputDesc(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, dgammaDesc);
    auto dgammaDtype = dgammaDesc->GetDataType();
    OP_CHECK_IF((dgammaDtype != ge::DataType::DT_FLOAT),
                OP_LOGE_FOR_INVALID_DTYPE(context_->GetNodeName(), "dgamma", ToString(dgammaDtype).c_str(), "FLOAT"),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus RmsNormGradQuantRegbaseTiling::CheckInputsDtype()
{
    if (CheckDyAndXDtype() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (CheckRstdGammaAndScaleXDtype() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return CheckOffsetAndOutputDtype();
}

ge::graphStatus RmsNormGradQuantRegbaseTiling::GetShapeAttrsInfo()
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
    auto regbaseAttrs = context_->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context_, regbaseAttrs);
    // check quant
    const char* regbaseQuantMode = regbaseAttrs->GetAttrPointer<char>(ATTR_INDEX_0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, regbaseQuantMode);
    if (strcmp(regbaseQuantMode, "static") == 0) {
        quantMode_ = STATIC_QUANT_MODE;
    } else {
        OP_CHECK_IF((true), OP_LOGE(context_->GetNodeName(), "the attr of quant mode should be static."),
                    return ge::GRAPH_FAILED);
    }
    // set divMode
    const bool* regbaseDivModePtr = regbaseAttrs->GetBool(ATTR_INDEX_1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, regbaseDivModePtr);
    divMode_ = *regbaseDivModePtr ? ComputeModeDivMode::DIV_MODE : ComputeModeDivMode::NOT_DIV_MODE;
    return ge::GRAPH_SUCCESS;
}

bool RmsNormGradQuantRegbaseTiling::IsCapable() { return true; }

uint64_t RmsNormGradQuantRegbaseTiling::GetTilingKey() const
{
    RmsNormGradQuantTilingKey tilingKey;

    tilingKey.SetComputeModeDx(computeModeDx_);
    tilingKey.SetComputeModeDgamma(computeModeDgamma_);
    tilingKey.SetComputeModeOffsetX(hasOffsetX_);
    tilingKey.SetComputeModeDivMode(divMode_);

    return tilingKey.GetTilingKey();
}

ge::graphStatus RmsNormGradQuantRegbaseTiling::PostTiling()
{
    OP_LOGD(context_->GetNodeName(), "Tiling usedCoreNum is %lu.", aivCoreNum_);
    context_->SetBlockDim(aivCoreNum_);

    auto rawTilingData = context_->GetRawTilingData();
    OP_CHECK_IF(sizeof(tilingData_) > rawTilingData->GetCapacity(),
                OP_LOGE(context_->GetNodeName(), "actual tiling data size %zu > context tiling data size %zu",
                        sizeof(tilingData_), rawTilingData->GetCapacity()),
                return ge::GRAPH_FAILED);
    auto capSize = rawTilingData->GetCapacity();
    void* ptrData = rawTilingData->GetData();
    OP_CHECK_NULL_WITH_CONTEXT(context_, ptrData);
    void* ptrStruct = static_cast<void*>(&tilingData_);
    OP_CHECK_NULL_WITH_CONTEXT(context_, ptrStruct);
    OP_CHECK_IF(memcpy_s(ptrData, capSize, ptrStruct, sizeof(tilingData_)) != 0,
                OP_LOGE(context_->GetNodeName(), "Set tiling data is failed!"), return ge::GRAPH_FAILED);
    rawTilingData->SetDataSize(sizeof(tilingData_));

    size_t* currentWorkspace = context_->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, currentWorkspace);
    currentWorkspace[0] = workspaceSize_;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus RmsNormGradQuantRegbaseTiling::DoLibApiTiling() { return ge::GRAPH_SUCCESS; }

ge::graphStatus RmsNormGradQuantRegbaseTiling::GetWorkspaceSize()
{
    fe::PlatFormInfos* platformInfoPtr = context_->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context_, platformInfoPtr);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    workspaceSize_ = ascendcPlatform.GetLibApiWorkSpaceSize();
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus RmsNormGradQuantRegbaseTiling::CalcTilingDataDx()
{
    blockFactorDx_ = Ops::Base::CeilDiv(rows_, static_cast<int64_t>(aivCoreNum_));
    usedCoreNumDx_ = Ops::Base::CeilDiv(rows_, blockFactorDx_);
    if (cols_ > DX_UB_FACTOR) {
        // tilingKey_ += DX_SPLIT_KEY;
        computeModeDx_ = ComputeModeDx::SPLIT_D;
        bodyPart_ = 1; // only for splitD
        int32_t powerofTwoValueDx = 0;
        while (bodyPart_ < cols_ / TWO) {
            powerofTwoValueDx += 1;
            bodyPart_ = std::pow(TWO, powerofTwoValueDx);
        }
    } else {
        bodyPart_ = 1;
    }

    return ge::GRAPH_SUCCESS;
}

void RmsNormGradQuantRegbaseTiling::CalcUsedCoreNumGamma()
{
    cols_sets_ = (cols_ + vlFp32_ - 1) / vlFp32_;
    if (cols_sets_ <= aivCoreNum_) {
        // 单核计算单colset
        isMultiColset_ = false;
        usedCoreNumDG_ = cols_sets_;
        colsPerCoreDG_ = vlFp32_;
        colsPerTailCoreDG_ = cols_ - (cols_sets_ - 1) * vlFp32_;
        tailCoreNumDG_ = 1;
        colsLastCoreDG_ = colsPerTailCoreDG_;
    } else {
        // 单核计算多个colset
        isMultiColset_ = true;
        usedCoreNumDG_ = aivCoreNum_;
        colsPerCoreDG_ = (cols_sets_ / usedCoreNumDG_) * vlFp32_;
        if (colsPerCoreDG_ * usedCoreNumDG_ > cols_) {
            colsPerCoreDG_ = colsPerCoreDG_ - vlFp32_;
        }
        colsPerTailCoreDG_ = colsPerCoreDG_ + vlFp32_;
        tailCoreNumDG_ = cols_sets_ - (colsPerCoreDG_ / vlFp32_ * usedCoreNumDG_);
        colsLastCoreDG_ = cols_ - (cols_sets_ - 1) * vlFp32_;
    }
}

int64_t RmsNormGradQuantRegbaseTiling::GetSizeOfBlockAlign(int64_t nonAlignSize)
{
    return (nonAlignSize + blockSize_ - 1) / blockSize_ * blockSize_;
}

int32_t RmsNormGradQuantRegbaseTiling::NearestLowerPowerOfTwo(int32_t temp)
{
    int64_t power = 0;
    int32_t powerofTwoValueDG = 0;
    while (power <= temp) {
        powerofTwoValueDG += 1;
        power = std::pow(TWO, powerofTwoValueDG);
    }
    return powerofTwoValueDG - 1;
}

ge::graphStatus RmsNormGradQuantRegbaseTiling::CalcTilingDataDgamma()
{
    // dy, x, gamma
    int64_t xUbSizeForOneColSet = GetSizeOfBlockAlign(rows_ * FLOAT_BYTE_SIZE) * vlFp32_;
    int64_t dyUbSizeForOneColSet = GetSizeOfBlockAlign(rows_ * FLOAT_BYTE_SIZE) * vlFp32_;
    int64_t rstdUbSize = GetSizeOfBlockAlign(rows_ * FLOAT_BYTE_SIZE);
    int64_t inputUbSizeForOneColSet = xUbSizeForOneColSet + dyUbSizeForOneColSet + rstdUbSize;

    // dgamma
    int64_t outputUbSizeForOneColSet = GetSizeOfBlockAlign((rows_ + 1) * FLOAT_BYTE_SIZE) * vlFp32_;
    colsPerUBDG_ = vlFp32_;
    if (static_cast<int64_t>(ubSize_) >= TWO * (inputUbSizeForOneColSet + outputUbSizeForOneColSet)) {
        // full-load
        isFullLoadDG_ = true;
        rowsPerUBDG_ = rows_;
    } else {
        // UB二分累加
        isFullLoadDG_ = false;
        computeModeDgamma_ = ComputeModeDgamma::WITH_LARGE_ROWS;
        // 反算一次UB能放下多少数据rows
        // 一行数据的buffer为(dy + x + dgamma + dgamma1) *64 + (binaryAddCache + rstd)
        maxRowsNumDG_ = ubSize_ / BUFFER_NUM / ((vlFp32_ * DGAMMA_VL_BUFFER_COUNT + TWO) * FLOAT_BYTE_SIZE);
        OP_CHECK_IF(maxRowsNumDG_ <= 0,
                    OP_LOGE(context_->GetNodeName(), "The maxRowsNumDG_ size is neg: %d.", maxRowsNumDG_),
                    return ge::GRAPH_FAILED);
        rowsPerUBDG_ = std::pow(TWO, NearestLowerPowerOfTwo(maxRowsNumDG_));
        OP_CHECK_IF(rowsPerUBDG_ <= 0,
                    OP_LOGE(context_->GetNodeName(), "The rowsPerUBDG_ size is invalid: %d.", rowsPerUBDG_),
                    return ge::GRAPH_FAILED);
        totalBlockCountDG_ = (rows_ + rowsPerUBDG_ - 1) / rowsPerUBDG_;
        mainBlockCountDG_ = totalBlockCountDG_;
        tailBlockCountwithPadDG_ = 0;
        if (rows_ % rowsPerUBDG_) {
            mainBlockCountDG_ = totalBlockCountDG_ - 1;
            tailBlockCountwithPadDG_ = 1;
        }
        binaryAddKDG_ = NearestLowerPowerOfTwo(mainBlockCountDG_);
        powerOfTwoBlockCountDG_ = std::pow(TWO, binaryAddKDG_);
        tailBlockCountWithoutPadDG_ = mainBlockCountDG_ - powerOfTwoBlockCountDG_;
    }
    int32_t n = NearestLowerPowerOfTwo(rows_);
    rowsTailDG_ = rows_ - std::pow(TWO, n);
    return ge::GRAPH_SUCCESS;
}

void RmsNormGradQuantRegbaseTiling::LogTilingResult() { OP_LOGI(OP_NAME, "rows: %ld, cols_: %ld", rows_, cols_); }

ge::graphStatus RmsNormGradQuantRegbaseTiling::DoOpTiling()
{
    ge::graphStatus result = ge::GRAPH_SUCCESS;
    computeModeDx_ = ComputeModeDx::FULL_LOAD;
    computeModeDgamma_ = ComputeModeDgamma::FULL_LOAD;

    // split cores
    CalcUsedCoreNumGamma();

    // choose template and calc ub buffer size
    result = CalcTilingDataDgamma();
    OP_CHECK_IF(result != ge::GRAPH_SUCCESS, OP_LOGE(OP_NAME, "Regbase template do op tiling dgamma failed."),
                return result);

    result = CalcTilingDataDx();
    OP_CHECK_IF(result != ge::GRAPH_SUCCESS, OP_LOGE(OP_NAME, "Regbase template do op tiling dx failed."),
                return result);

    tilingData_.dxTilingData.cols = cols_;
    tilingData_.dxTilingData.rows = rows_;
    tilingData_.dxTilingData.blockFactorDx = blockFactorDx_;
    tilingData_.dxTilingData.bodyPart = bodyPart_;
    tilingData_.dxTilingData.usedCoreNumDx = usedCoreNumDx_;

    tilingData_.blockSize = blockSize_;
    tilingData_.usedCoreNumDG = usedCoreNumDG_;
    tilingData_.colsPerCoreDG = colsPerCoreDG_;
    tilingData_.rowsPerUBDG = rowsPerUBDG_;
    tilingData_.vlFp32 = vlFp32_;
    tilingData_.colsPerTailCoreDG = colsPerTailCoreDG_;
    tilingData_.isFullLoad = isFullLoadDG_;
    tilingData_.isMultiColset = isMultiColset_;
    tilingData_.colsLastCoreDG = colsLastCoreDG_;
    tilingData_.tailCoreNumDG = tailCoreNumDG_;
    tilingData_.powerofTwoValueDG = powerofTwoValueDG_;
    tilingData_.rowsTailDG = rowsTailDG_;

    tilingData_.totalBlockCountDG = totalBlockCountDG_;
    tilingData_.mainBlockCountDG = mainBlockCountDG_;
    tilingData_.tailBlockCountwithPadDG = tailBlockCountwithPadDG_;
    tilingData_.powerOfTwoBlockCountDG = powerOfTwoBlockCountDG_;
    tilingData_.tailBlockCountWithoutPadDG = tailBlockCountWithoutPadDG_;
    tilingData_.binaryAddKDG = binaryAddKDG_;

    LogTilingResult();
    return result;
}

REGISTER_OPS_TILING_TEMPLATE(RmsNormGradQuant, RmsNormGradQuantRegbaseTiling, 1000);
} // namespace optiling

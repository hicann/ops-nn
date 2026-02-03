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
 * \file rms_norm_quant_v2_tiling_arch35_base.cpp
 * \brief
 */

#include "rms_norm_quant_v2_tiling.h"
#include "norm/norm_common/op_host/norm_tiling_check_common.h"

namespace optiling {
using namespace NormCheck;

constexpr uint32_t MAX_DIM_CNT = 8;
constexpr uint32_t ONE_DIM = 1;
constexpr uint32_t B32_BLOCK_NUM = 8;
constexpr uint32_t BLOCK_SIZE = 32;
constexpr uint32_t FLOAT_NUM = 64;
constexpr uint32_t ALIGN_FACTOR_512 = 512;
constexpr uint32_t ONCE_VECTOR_SIZE = 256;

constexpr uint32_t LEVEL_BUFFER_CNT = 3;
constexpr uint32_t MULTI_FACTOR_2 = 2;
constexpr uint32_t ZERO_POINTS1_BIN_OFFSET = 2;
constexpr uint32_t SCALES2_BIN_OFFSET = 1;

constexpr float DEFAULT_DPSILON = 1e-6;
constexpr bool DEFAULT_DIVMODE = true;

const gert::Shape g_vec_1_shape = {1};
/**
 * Ensure that the returned shape is non-scalar.
 * When the dim num of shape is 0, this shape is considered to express a scalar.
 * This function returns the original shape when it receives a non-scalar shape,
 * and returns the vector shape that returns a {1} when it receives a scalar shape
 * @param in_shape input shape
 * @return non-scalar shape
 */
inline const gert::Shape &EnsureNotScalar(const gert::Shape &in_shape) {
    if (in_shape.IsScalar()) {
        return g_vec_1_shape;
    }
    return in_shape;
}

ge::graphStatus RmsNormQuantV2RegbaseTilingBase::CheckDtypeVaild(
    ge::DataType& srcDtype, std::vector<ge::DataType>& supportDtypeList, string srcName)
{
    for (const auto& supportedDtype : supportDtypeList) {
        if (supportedDtype == srcDtype) {
            return ge::GRAPH_SUCCESS;
        }
    }
    OP_LOGE(
        context_->GetNodeName(), "Dtype check invalid, %s dtype is %s, not in supportDtypeList.", srcName.c_str(),
        Ops::Base::ToString(srcDtype).c_str());
    return ge::GRAPH_FAILED;
}

bool RmsNormQuantV2RegbaseTilingBase::CheckShapeNull()
{
    OP_LOGD(context_->GetNodeName(), "Enter RmsNormQuantV2RegbaseTiling CheckShapeNull.");
    const gert::StorageShape* xShape = context_->GetInputShape(X_INDEX);
    const gert::StorageShape* gammaShape = context_->GetInputShape(GAMMA_INDEX);
    const gert::StorageShape* scales1Shape = context_->GetInputShape(SCALES1_INDEX);
    const gert::StorageShape* y1Shape = context_->GetOutputShape(Y1_INDEX);
    const gert::StorageShape* y2Shape = context_->GetOutputShape(Y2_INDEX);

    OP_CHECK_IF(
        (nullptr == xShape) || (nullptr == gammaShape) || (nullptr == scales1Shape) || 
         (nullptr == y1Shape) || (nullptr == y2Shape), 
         , return false);
    return true;
}

bool RmsNormQuantV2RegbaseTilingBase::CheckOptionalInput()
{
    OP_LOGD(context_->GetNodeName(), "Enter RmsNormQuantV2RegbaseTiling CheckOptionalInput.");
    const gert::StorageShape* scales2Shape = context_->GetOptionalInputShape(SCALES2_INDEX);
    const gert::StorageShape* zeroPoints1Shape = context_->GetOptionalInputShape(ZERO_POINTS1_INDEX);
    const gert::StorageShape* zeroPoints2Shape = context_->GetOptionalInputShape(ZERO_POINTS2_INDEX);
    const gert::StorageShape* betaShape = context_->GetOptionalInputShape(BETA_INDEX);

    if (scales2Shape != nullptr) {
        tilingParams.hasScales2 = true;
    }
    if (zeroPoints1Shape != nullptr) {
        tilingParams.hasZeroPoints1 = true;
    }
    if (scales2Shape != nullptr && zeroPoints2Shape != nullptr) { // 没有scales2就不需要zeropoints2
        tilingParams.hasZeroPoints2 = true;
    }
    if (betaShape != nullptr) {
        tilingParams.hasBeta = true;
    }
    tilingParams.hasY2 = tilingParams.hasScales2; // 没有scales2就没有y2
    if (scales2Shape == nullptr && zeroPoints2Shape != nullptr) {
        OP_LOGE(context_->GetNodeName(), "Scales2 is required when zero_points2 is present.");
        return false;
    }
    return true;
}

bool RmsNormQuantV2RegbaseTilingBase::CheckInputShapeDim()
{
    OP_LOGD(context_->GetNodeName(), "Enter RmsNormQuantV2RegbaseTiling CheckInputShapeDim.");
    const gert::StorageShape* xStoregeShape = context_->GetInputShape(X_INDEX);
    auto xShape = EnsureNotScalar(xStoregeShape->GetStorageShape());
    const gert::StorageShape* gammaStoregeShape = context_->GetInputShape(GAMMA_INDEX);
    auto gammaShape = EnsureNotScalar(gammaStoregeShape->GetStorageShape());
    const gert::StorageShape* scales1StorageShape = context_->GetInputShape(SCALES1_INDEX);
    auto scales1Shape = EnsureNotScalar(scales1StorageShape->GetStorageShape());
    const gert::StorageShape* scales2StorageShape = context_->GetOptionalInputShape(SCALES2_INDEX);
    const gert::StorageShape* zp1StorageShape = context_->GetOptionalInputShape(ZERO_POINTS1_INDEX);
    const gert::StorageShape* zp2StorageShape = context_->GetOptionalInputShape(ZERO_POINTS2_INDEX);
    const gert::StorageShape* betaStorageShape = context_->GetOptionalInputShape(BETA_INDEX);

    size_t xDimNum = xShape.GetDimNum();
    size_t gammaDimNum = gammaShape.GetDimNum();
    size_t scales1DimNum = scales1Shape.GetDimNum();
    size_t scales2DimNum = 0;
    if (tilingParams.hasScales2) {
        auto scales2Shape = EnsureNotScalar(scales2StorageShape->GetStorageShape());
        scales2DimNum = scales2Shape.GetDimNum();
    }
    size_t zp1DimNum = 0;
    if (tilingParams.hasZeroPoints1) {
        auto zp1Shape = EnsureNotScalar(zp1StorageShape->GetStorageShape());
        zp1DimNum = zp1Shape.GetDimNum();
    }
    size_t zp2DimNum = 0;
    if (tilingParams.hasZeroPoints2) {
        auto zp2Shape = EnsureNotScalar(zp2StorageShape->GetStorageShape());
        zp2DimNum = zp2Shape.GetDimNum();
    }
    size_t betaDimNum = 0;
    if (tilingParams.hasBeta) {
        auto betaShape = EnsureNotScalar(betaStorageShape->GetStorageShape());
        betaDimNum = betaShape.GetDimNum();
    }
    OP_CHECK_IF(
        (xDimNum > MAX_DIM_CNT) || (gammaDimNum > MAX_DIM_CNT) || (scales1DimNum > MAX_DIM_CNT) ||
            (tilingParams.hasScales2 && scales2DimNum > MAX_DIM_CNT) ||
            (tilingParams.hasZeroPoints1 && zp1DimNum > MAX_DIM_CNT) ||
            (tilingParams.hasZeroPoints2 && zp2DimNum > MAX_DIM_CNT) ||
            (tilingParams.hasBeta && betaDimNum > MAX_DIM_CNT),
        OP_LOGE(context_->GetNodeName(), "All input dim should not bigger than %u.", MAX_DIM_CNT), return false);
    return true;
}

bool RmsNormQuantV2RegbaseTilingBase::CheckInputShapeValue()
{
    OP_LOGD(context_->GetNodeName(), "Enter RmsNormQuantV2RegbaseTiling CheckInputShapeValue.");
    const gert::StorageShape* xShape = context_->GetInputShape(X_INDEX);
    const gert::StorageShape* gammaShape = context_->GetInputShape(GAMMA_INDEX);
    const gert::StorageShape* scales1Shape = context_->GetInputShape(SCALES1_INDEX);
    const gert::StorageShape* scales2Shape = context_->GetOptionalInputShape(SCALES2_INDEX);
    const gert::StorageShape* zeroPoints1Shape = context_->GetOptionalInputShape(ZERO_POINTS1_INDEX);
    const gert::StorageShape* zeroPoints2Shape = context_->GetOptionalInputShape(ZERO_POINTS2_INDEX);
    const gert::StorageShape* betaShape = context_->GetOptionalInputShape(BETA_INDEX);
    const gert::StorageShape* y1Shape = context_->GetOutputShape(Y1_INDEX);
    const gert::StorageShape* y2Shape = context_->GetOutputShape(Y2_INDEX);

    // Check x&y1&y2's shape should be equal
    if (!CheckShapeSame(xShape, y1Shape, nodeName, "x", "y1")) {
        return false;
    };
    if (tilingParams.hasY2 && !CheckShapeSame(xShape, y2Shape, nodeName, "x", "y2")) {
        return false;
    };

    // Check scales1&scales2's shape should be equal
    if (tilingParams.hasScales2 &&
        !CheckShapeSame(scales1Shape, scales2Shape, nodeName, "scales1", "scales2")) {
        return false;
    };

    // Check gamma&beta's shape should be equal
    if (tilingParams.hasBeta && !CheckShapeSame(gammaShape, betaShape, nodeName, "gamma", "beta")) {
        return false;
    };

    // Check scales&zeropoints's shape should be equal
    if (tilingParams.hasZeroPoints1 &&
        !CheckShapeSame(scales1Shape, zeroPoints1Shape, nodeName, "scales1", "zeroPoints1")) {
        return false;
    };
    if (tilingParams.hasScales2 && tilingParams.hasZeroPoints2 &&
        !CheckShapeSame(scales2Shape, zeroPoints2Shape, nodeName, "scales2", "zeroPoints2")) {
        return false;
    };

    // Check gamma should be last few dim of x
    if (!CheckShapeBC(xShape, gammaShape, nodeName, "x", "gamma")) {
        return false;
    };

    // Check scales1 should be last few dim of x or be 1
    if ( !CheckAllDimsAreOne(scales1Shape) &&!CheckShapeBC(xShape, scales1Shape, nodeName, "x", "scales1")) {
        OP_LOGE(context_->GetNodeName(), "Scales1 should be last few dim of x or all dims of scales1 should be 1.");
        return false;
    }

    return true;
}

bool RmsNormQuantV2RegbaseTilingBase::CheckShapeSame(const gert::StorageShape* src1Shape, const gert::StorageShape* src2Shape,
    string inNodeName, string inSrc1Name, string inSrc2Name)
{
    size_t src1DimNum = EnsureNotScalar(src1Shape->GetStorageShape()).GetDimNum();
    size_t src2DimNum = EnsureNotScalar(src2Shape->GetStorageShape()).GetDimNum();

    OP_TILING_CHECK(
        (src1DimNum != src2DimNum),
        OP_LOGE(
            inNodeName.c_str(), "Dim num check invalid, %s is %lu %s is %lu, not equal.", inSrc1Name.c_str(), src1DimNum,
            inSrc2Name.c_str(), src2DimNum),
        return false);
    for (size_t i = 0; i < src1DimNum; i++) {
        uint64_t src1DimValue = EnsureNotScalar(src1Shape->GetStorageShape()).GetDim(i);
        uint64_t src2DimValue = EnsureNotScalar(src2Shape->GetStorageShape()).GetDim(i);

        OP_TILING_CHECK(
            (src1DimValue != src2DimValue),
            OP_LOGE(
                inNodeName.c_str(), "Dim value check invalid, %s[%lu] is %lu, %s[%lu] is %lu, not equal.",
                inSrc1Name.c_str(), i, src1DimValue, inSrc2Name.c_str(), i, src2DimValue),
            return false);
    }
    return true;
}

bool RmsNormQuantV2RegbaseTilingBase::CheckShapeBC(const gert::StorageShape* srcBcShape, const gert::StorageShape* srcShape,
    string inNodeName, string inSrcBcName, string inSrcName)
{
    size_t srcBcDimNum = EnsureNotScalar(srcBcShape->GetStorageShape()).GetDimNum();
    size_t srcDimNum = EnsureNotScalar(srcShape->GetStorageShape()).GetDimNum();
    bool isBcHeader =  true;
    OP_TILING_CHECK(
        (srcBcDimNum < srcDimNum),
        OP_LOGE(
            inNodeName.c_str(), "Dim num check invalid, %s is %lu %s is %lu, not bigger.", inSrcBcName.c_str(), srcBcDimNum,
            inSrcName.c_str(), srcDimNum),
        return false);
    for (size_t i = 0; i < srcDimNum; i++) {
        uint64_t srcBcDimValue;
        if (isBcHeader) {
            srcBcDimValue = EnsureNotScalar(srcBcShape->GetStorageShape()).GetDim(srcBcDimNum - srcDimNum + i);
        } else {
            srcBcDimValue = EnsureNotScalar(srcBcShape->GetStorageShape()).GetDim(i);
        }
        uint64_t srcDimValue = EnsureNotScalar(srcShape->GetStorageShape()).GetDim(i);

        OP_TILING_CHECK(
            (srcBcDimValue != srcDimValue),
            OP_LOGE(
                inNodeName.c_str(), "Dim value check invalid, %s[%lu] is %lu, %s[%lu] is %lu, not equal.",
                inSrcBcName.c_str(), i, srcBcDimValue, inSrcName.c_str(), i, srcDimValue),
            return false);
    }
    return true;
}

bool RmsNormQuantV2RegbaseTilingBase::CheckAllDimsAreOne(const gert::StorageShape* storegeShape)
{
    auto shape = EnsureNotScalar(storegeShape->GetStorageShape());
    size_t dimNum = shape.GetDimNum();
    for (size_t i = 0; i < dimNum; i++) {
        uint64_t dimValue = shape.GetDim(i);
        if(dimValue != 1)
            return false;
    }
    return true;
}

bool RmsNormQuantV2RegbaseTilingBase::CheckOutputDtype()
{
    OP_LOGD(context_->GetNodeName(), "Enter RmsNormQuantV2RegbaseTiling CheckOutputDtype.");
    std::vector<ge::DataType> supportedYDtypes = {
        ge::DataType::DT_INT8, ge::DataType::DT_INT4, ge::DataType::DT_HIFLOAT8, ge::DataType::DT_FLOAT8_E4M3FN,
        ge::DataType::DT_FLOAT8_E5M2};
    auto y1DataType = context_->GetOutputDesc(Y1_INDEX)->GetDataType();
    auto y2DataType = context_->GetOutputDesc(Y2_INDEX)->GetDataType();
    if ((ge::GRAPH_SUCCESS != CheckDtypeVaild(y1DataType, supportedYDtypes, "RmsNormQuantV2")) ||
        (ge::GRAPH_SUCCESS != CheckDtypeVaild(y2DataType, supportedYDtypes, "RmsNormQuantV2")) ||
        (y1DataType != y2DataType)) {
        OP_LOGE(
            context_->GetNodeName(),
            "Output dtype should be int8 fp8e4m3 fp8e5m2 hifp8 and y1DataType y2DataType need same.");
        return false;
    }
    // if yDtype=int4 , last dim of x should be even
    if (y1DataType == ge::DataType::DT_INT4) {
        const gert::StorageShape* xStoregeShape = context_->GetInputShape(X_INDEX);
        auto xShape = EnsureNotScalar(xStoregeShape->GetStorageShape());
        size_t xDimNum = xShape.GetDimNum();
        int64_t xlastDimValue = xShape.GetDim(xDimNum - 1);
        if (xlastDimValue % 2 != 0) {
            OP_LOGE(context_->GetNodeName(), " Output dtype is int4, last dimension of x must be even.");
            return false;
        }
    }
    return true;
}

bool RmsNormQuantV2RegbaseTilingBase::CheckInputDtype()
{
    OP_LOGD(context_->GetNodeName(), "Enter RmsNormQuantV2RegbaseTiling CheckInputDtype.");
    std::vector<ge::DataType> supportedXGammaDtypes = {
        ge::DataType::DT_FLOAT, ge::DataType::DT_FLOAT16, ge::DataType::DT_BF16};
    std::vector<ge::DataType> supportedScalesDtypes = {
        ge::DataType::DT_FLOAT, ge::DataType::DT_FLOAT16, ge::DataType::DT_BF16};
    std::vector<ge::DataType> supportedZeroPointsDtypes = {
        ge::DataType::DT_INT32, ge::DataType::DT_INT8, ge::DataType::DT_FLOAT, ge::DataType::DT_FLOAT16,
        ge::DataType::DT_BF16};
    std::vector<ge::DataType> supportedYDtypes = {
        ge::DataType::DT_INT8, ge::DataType::DT_INT4, ge::DataType::DT_FLOAT8_E5M2, ge::DataType::DT_FLOAT8_E4M3FN,
        ge::DataType::DT_HIFLOAT8};

    const uint32_t totalCheckCnt = 7;
    string checkNameList[totalCheckCnt] = {"x", "gamma", "scales1", "scales2", "zeroPoints1", "zeroPoints2", "beta"};
    uint32_t idxList[totalCheckCnt] = {
        X_INDEX, GAMMA_INDEX, SCALES1_INDEX, SCALES2_INDEX, ZERO_POINTS1_INDEX, ZERO_POINTS2_INDEX, BETA_INDEX};
    bool isOptionalList[totalCheckCnt] = {false, false, false, true, true, true, true};
    bool needCheckList[totalCheckCnt] = {
        true,
        true,
        true,
        tilingParams.hasScales2,
        tilingParams.hasZeroPoints1,
        tilingParams.hasZeroPoints2,
        tilingParams.hasBeta};
    std::vector<ge::DataType>* supportedList[totalCheckCnt] = {
        &supportedXGammaDtypes,     &supportedXGammaDtypes,     &supportedScalesDtypes, &supportedScalesDtypes,
        &supportedZeroPointsDtypes, &supportedZeroPointsDtypes, &supportedXGammaDtypes};

    for (uint32_t checkIdx = 0; checkIdx < totalCheckCnt; checkIdx++) {
        if (!needCheckList[checkIdx]) {
            continue;
        }

        ge::DataType srcDtype;
        if (isOptionalList[checkIdx]) {
            srcDtype = context_->GetOptionalInputTensor(idxList[checkIdx])->GetDataType();
        } else {
            srcDtype = context_->GetInputTensor(idxList[checkIdx])->GetDataType();
        }
        if (ge::GRAPH_SUCCESS != CheckDtypeVaild(srcDtype, *(supportedList[checkIdx]), checkNameList[checkIdx])) {
            return false;
        };
    }

    ge::DataType xDtype = context_->GetInputTensor(X_INDEX)->GetDataType();
    ge::DataType gammaDtype = context_->GetInputTensor(GAMMA_INDEX)->GetDataType();
    ge::DataType scales1Dtype = context_->GetInputTensor(SCALES1_INDEX)->GetDataType();
    ge::DataType scales2Dtype = ge::DT_BOOL;     // Init one not support dtype
    ge::DataType zeroPoints1Dtype = ge::DT_BOOL; // Init one not support dtype
    ge::DataType zeroPoints2Dtype = ge::DT_BOOL; // Init one not support dtype
    ge::DataType zeroPointsDtype = ge::DT_BOOL; // Init one not support dtype
    bool hasZeroPoints = false;
    ge::DataType betaDtype = ge::DT_BOOL;        // Init one not support dtype
    if (tilingParams.hasScales2) {
        scales2Dtype = context_->GetOptionalInputTensor(SCALES2_INDEX)->GetDataType();
    }
    if (tilingParams.hasZeroPoints1) {
        zeroPoints1Dtype = context_->GetOptionalInputTensor(ZERO_POINTS1_INDEX)->GetDataType();
    }
    if (tilingParams.hasScales2 && tilingParams.hasZeroPoints2) {  //没有scales2就不用有zeropoints2
        zeroPoints2Dtype = context_->GetOptionalInputTensor(ZERO_POINTS2_INDEX)->GetDataType();
    }
    if(tilingParams.hasZeroPoints1){
        zeroPointsDtype = zeroPoints1Dtype;
        hasZeroPoints = true;
    }else if(tilingParams.hasScales2 && tilingParams.hasZeroPoints2){
        zeroPointsDtype = zeroPoints2Dtype;
        hasZeroPoints = true;
    }
    if (tilingParams.hasBeta) {
        betaDtype = context_->GetOptionalInputTensor(BETA_INDEX)->GetDataType();
    }
    if (xDtype != gammaDtype) {
        OP_LOGE(context_->GetNodeName(), "Input x/gamma dtype should be equal.");
        return false;
    }
    if ((tilingParams.hasBeta) && (xDtype != betaDtype)) {
        OP_LOGE(context_->GetNodeName(), "Input x/beta dtype should be equal.");
        return false;
    }
    if (tilingParams.hasScales2 && (scales1Dtype != scales2Dtype)) {
        OP_LOGE(context_->GetNodeName(), "Input scales1/scales2 dtype should be equal.");
        return false;
    }
    if (tilingParams.hasZeroPoints1 && tilingParams.hasZeroPoints2 && (zeroPoints1Dtype != zeroPointsDtype)) {
        OP_LOGE(context_->GetNodeName(), "Input zeropoints1/zeropoints2 dtype should be equal.");
        return false;
    }
    //check support dtypes 
    if(xDtype == ge::DataType::DT_FLOAT){
        if(scales1Dtype != ge::DataType::DT_FLOAT){
            OP_LOGE(context_->GetNodeName(), "Input x dtype is fp32, scales1 dtype should be fp32.");
            return false;
        }
        if(hasZeroPoints && zeroPointsDtype != ge::DataType::DT_FLOAT){
            OP_LOGE(context_->GetNodeName(), "Input x dtype is fp32, zeropoints dtype should be fp32.");
            return false;
        }
    }else if(xDtype == ge::DataType::DT_FLOAT16){
        if(scales1Dtype == ge::DataType::DT_FLOAT){
            if(hasZeroPoints && zeroPointsDtype != ge::DataType::DT_FLOAT && zeroPointsDtype != ge::DataType::DT_INT32){
                OP_LOGE(context_->GetNodeName(), "Input x dtype is fp16 and scales1 dtype is fp32, zeropoints dtype should be fp32 or int32.");
                return false;
            }
        }else if(scales1Dtype == ge::DataType::DT_FLOAT16){
            if(hasZeroPoints && zeroPointsDtype != ge::DataType::DT_FLOAT16 && zeroPointsDtype != ge::DataType::DT_INT8){
                OP_LOGE(context_->GetNodeName(), "Input x dtype is fp16 and scales1 dtype is bf16, zeropoints dtype should be bf16 or int8.");
                return false;
            }
        }else {
            OP_LOGE(context_->GetNodeName(), "Input x dtype is fp16, scales1 dtype should be fp32 or fp16.");
            return false;
        }
    }else if(xDtype == ge::DataType::DT_BF16){
        if(scales1Dtype == ge::DataType::DT_FLOAT){
            if(hasZeroPoints && zeroPointsDtype != ge::DataType::DT_FLOAT && zeroPointsDtype != ge::DataType::DT_INT32){
                OP_LOGE(context_->GetNodeName(), "Input x dtype is bf16 and scales1 dtype is fp32, zeropoints dtype should be fp32 or int32.");
                return false;
            }
        }else if(scales1Dtype == ge::DataType::DT_BF16){
            if(hasZeroPoints && zeroPointsDtype != ge::DataType::DT_BF16 && zeroPointsDtype != ge::DataType::DT_INT8){
                OP_LOGE(context_->GetNodeName(), "Input x dtype is bf16 and scales1 dtype is bf16, zeropoints dtype should be bf16 or int8.");
                return false;
            }
        }else {
            OP_LOGE(context_->GetNodeName(), "Input x dtype is bf16, scales1 dtype should be fp32 or bf16.");
            return false;
        }
    }

    return true;
}

ge::graphStatus RmsNormQuantV2RegbaseTilingBase::SetInputParams()
{
    OP_LOGD(context_->GetNodeName(), "Enter RmsNormQuantV2RegbaseTiling SetInputParams.");
    // Set input dim
    const gert::StorageShape* xStoregeShape = context_->GetInputShape(X_INDEX);
    auto xShape = EnsureNotScalar(xStoregeShape->GetStorageShape());
    const gert::StorageShape* gammaStoregeShape = context_->GetInputShape(GAMMA_INDEX);
    auto gammaShape = EnsureNotScalar(gammaStoregeShape->GetStorageShape());
    const gert::StorageShape* scales1StorageShape = context_->GetInputShape(SCALES1_INDEX);
    auto scales1Shape = EnsureNotScalar(scales1StorageShape->GetStorageShape());
    size_t xDimNum = xShape.GetDimNum();
    size_t gammaDimNum = gammaShape.GetDimNum();
    size_t scales1DimNum = scales1Shape.GetDimNum();
    tilingParams.a = 1;
    tilingParams.r = 1;
    tilingParams.q = 1;
    
    for (size_t i = 0; i < xDimNum - gammaDimNum; i++) {
        tilingParams.a *= xShape.GetDim(i);
    }
    if (0 == tilingParams.a) tilingParams.a=1; //x轴全为r轴，a轴为1
    for (size_t i = 0; i < gammaDimNum; i++) {
        tilingParams.r *= gammaShape.GetDim(i);
    }
    for (size_t i = 0; i < scales1DimNum; i++) {
        tilingParams.q *= scales1Shape.GetDim(i);
    }
    if (0 == tilingParams.r) {
        OP_LOGE(context_->GetNodeName(), "Can not div zero.");
        return ge::GRAPH_FAILED;
    }

    // Set input dtype
    auto xDataType = context_->GetInputTensor(X_INDEX)->GetDataType();
    auto scaleDataType = context_->GetInputTensor(SCALES1_INDEX)->GetDataType();
    tilingParams.dstDtype = context_->GetOutputDesc(Y1_INDEX)->GetDataType();
    tilingParams.xDtypeSize = GetSizeByDataType(xDataType);
    tilingParams.scaleDtypeSize = GetSizeByDataType(scaleDataType);
    tilingParams.xDtypeAlignNum = BLOCK_SIZE / tilingParams.xDtypeSize; // 一个block能容纳x类型的数据个数
    tilingParams.scaleDtypeAlignNum = BLOCK_SIZE / tilingParams.scaleDtypeSize; // 一个block能容纳的scales类型的数据个数
    if(tilingParams.hasZeroPoints1){
        auto zeroPointDtype = context_->GetOptionalInputTensor(ZERO_POINTS1_INDEX)->GetDataType();
        tilingParams.zeroPointDtypeSize = GetSizeByDataType(zeroPointDtype);
        tilingParams.zeroPointDtypeAlignNum = BLOCK_SIZE / tilingParams.zeroPointDtypeSize; // 一个block能容纳的zeropoints类型的数据个数
    }
    else if(tilingParams.hasZeroPoints2){
        auto zeroPointDtype = context_->GetOptionalInputTensor(ZERO_POINTS2_INDEX)->GetDataType();
        tilingParams.zeroPointDtypeSize = GetSizeByDataType(zeroPointDtype);
        tilingParams.zeroPointDtypeAlignNum = BLOCK_SIZE / tilingParams.zeroPointDtypeSize; // 一个block能容纳的zeropoints类型的数据个数
    }

    tilingParams.vecLength = Ops::Base::GetVRegSize(context_) / sizeof(float);
    tilingParams.ubBlockSize = Ops::Base::GetUbBlockSize(context_);

    // Set input attr
    auto attrs = context_->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context_, attrs);
    const float* epsilon = attrs->GetFloat(EPS_ATTR_INDEX);
    tilingParams.epsilon = epsilon == nullptr ? DEFAULT_DPSILON : *epsilon;
    tilingParams.avgFactor = 1.0f / static_cast<float>(tilingParams.r);
    tilingParams.optionMask =
        ((static_cast<uint64_t>(tilingParams.hasScales2 ? 1 : 0) << 0) |
         (static_cast<uint64_t>(tilingParams.hasZeroPoints1 ? 1 : 0) << 1) |
         (static_cast<uint64_t>(tilingParams.hasZeroPoints2 ? 1 : 0) << 2) |
         (static_cast<uint64_t>(tilingParams.hasBeta ? 1 : 0) << 3) |
         (static_cast<uint64_t>(tilingParams.q == 1 ? 1 : 0) << 4));
    const bool* divModePtr = attrs->GetBool(DIV_MODE_ATTR_INDEX); // 添加类型判断
    tilingParams.divMode = (divModePtr == nullptr) ? DEFAULT_DIVMODE : *divModePtr;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus RmsNormQuantV2RegbaseTilingBase::GetShapeAttrsInfo()
{
    OP_LOGD(context_->GetNodeName(), "Enter RmsNormQuantV2RegbaseTiling GetShapeAttrsInfo.");
    OP_CHECK_IF(
        !CheckShapeNull(), OP_LOGE(context_->GetNodeName(), "The not optional input is null."),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        !CheckOptionalInput(), OP_LOGE(context_->GetNodeName(), "Scales2 is required when zero_points2 is present."),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        !CheckInputShapeDim(), OP_LOGE(context_->GetNodeName(), "The input shape dim is invalid."),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        !CheckInputShapeValue(), OP_LOGE(context_->GetNodeName(), "The input shape relationship is invalid."),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        !CheckInputDtype(), OP_LOGE(context_->GetNodeName(), "The input dtype is invalid."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        !CheckOutputDtype(), OP_LOGE(context_->GetNodeName(), "The output dtype is invalid."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        ge::GRAPH_SUCCESS != SetInputParams(), OP_LOGE(context_->GetNodeName(), "Set input shape failed."),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus RmsNormQuantV2RegbaseTilingBase::GetPlatformInfo()
{
    OP_LOGD(context_->GetNodeName(), "Enter RmsNormQuantV2RegbaseTiling GetPlatformInfo.");
    if (context_->GetCompileInfo() == nullptr) {
        OP_LOGD(context_->GetNodeName(), "GetPlatformInfo return nullptr, need re get later.");
        tilingParams.needGetCompileInfo = true;
    } else {
        auto compileInfo = reinterpret_cast<const RmsNormQuantV2CompileInfo*>(context_->GetCompileInfo());
        tilingParams.totalCoreNum = compileInfo->totalCoreNum;
        tilingParams.maxUbSize = compileInfo->maxUbSize;
        tilingParams.needGetCompileInfo = false;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus RmsNormQuantV2RegbaseTilingBase::DoLibApiTiling()
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus RmsNormQuantV2RegbaseTilingBase::GetWorkspaceSize()
{
    tilingParams.workspaceSize = 0;
    return ge::GRAPH_SUCCESS;
}

} // namespace optiling

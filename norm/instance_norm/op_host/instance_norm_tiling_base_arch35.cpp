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
 * \file instance_norm_tiling_base_arch35.cpp
 * \brief
 */
#include <vector>
#include <algorithm>
#include "instance_norm_tiling.h"

using namespace ge;

namespace {
constexpr int64_t NCHW_DIM_NUM = 4;
constexpr int64_t NCDHW_DIM_NUM = 5;
constexpr int64_t NHWC_DIM_NUM = 4;
constexpr int64_t NDHWC_DIM_NUM = 5;
constexpr int64_t ND_MIN_DIM_NUM = 2;
constexpr int64_t ND_MAX_DIM_NUM = 8;

constexpr int64_t DIM_0 = 0;
constexpr int64_t DIM_1 = 1;
constexpr int64_t DIM_2 = 2;
constexpr int64_t DIM_3 = 3;
constexpr int64_t DIM_4 = 4;

const std::vector<ge::DataType> DTYPE_LIST = {ge::DataType::DT_FLOAT16, ge::DataType::DT_BF16, ge::DataType::DT_FLOAT};
} // namespace

namespace optiling {
ge::graphStatus InstanceNormRegbaseTilingBase::GetPlatformInfo()
{
    auto platformInfo = context_->GetPlatformInfo();
    auto compileInfoPtr = reinterpret_cast<const InstanceNormCompileInfo*>(context_->GetCompileInfo());
    OP_CHECK_IF(
        compileInfoPtr == nullptr, OP_LOGE(context_->GetNodeName(), "compile info is null"), return ge::GRAPH_FAILED);
    vlfp32 = compileInfoPtr->vectorLength / sizeof(float);
    ubBlockSize = compileInfoPtr->ubBlockSize;
    vectorLength = compileInfoPtr->vectorLength;

    if (platformInfo != nullptr) {
        auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
        aicoreParams_.blockDim = ascendcPlatform.GetCoreNumAiv();
        uint64_t ubSizePlatForm;
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatForm);
        aicoreParams_.ubSize = ubSizePlatForm;
    } else {
        aicoreParams_.blockDim = compileInfoPtr->coreNum;
        aicoreParams_.ubSize = compileInfoPtr->ubSize;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus InstanceNormRegbaseTilingBase::GetShapeAttrsInfo()
{
    if (context_ == nullptr) {
        OP_LOGE("InstanceNorm", "TilingContext is nullptr.");
        return ge::GRAPH_FAILED;
    }

    // 获取attr
    auto attrs = context_->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context_, attrs);
    const float* epsilonPtr = attrs->GetFloat(ATTR_EPSILON_IDX);
    epsilon = (epsilonPtr == nullptr) ? DEFAULT_EPSILON : *epsilonPtr;
    // 获取输入shape
    auto xShape = context_->GetInputShape(INPUT_X_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, xShape);
    xStorageShape = xShape->GetStorageShape();
    if (CheckShapeAllNotNegative(xStorageShape) != ge::GRAPH_SUCCESS) {
        OP_LOGE(context_->GetNodeName(), "Not supported shape info.");
        return ge::GRAPH_FAILED;
    }
    auto xDesc = context_->GetInputDesc(INPUT_X_INDEX);
    auto gammaDesc = context_->GetInputDesc(INPUT_GAMMA_INDEX);
    auto meanDesc = context_->GetOutputDesc(OUTPUT_MEAN_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, xDesc);
    OP_CHECK_NULL_WITH_CONTEXT(context_, gammaDesc);
    OP_CHECK_NULL_WITH_CONTEXT(context_, meanDesc);
    dataType = xDesc->GetDataType();
    gammaDataType = gammaDesc->GetDataType();
    meanDataType = meanDesc->GetDataType();
    format = xDesc->GetFormat().GetStorageFormat();
    if (format == FORMAT_NCHW) {
        OP_CHECK_IF(
            xStorageShape.GetDimNum() != NCHW_DIM_NUM,
            OP_LOGE(context_->GetNodeName(), "Dims should be 4 with NCHW format."), return ge::GRAPH_FAILED);
        a1 = xStorageShape.GetDim(DIM_0);
        a0 = xStorageShape.GetDim(DIM_1);
        r = xStorageShape.GetDim(DIM_2) * xStorageShape.GetDim(DIM_3);
    } else if (format == FORMAT_NCDHW) {
        OP_CHECK_IF(
            xStorageShape.GetDimNum() != NCDHW_DIM_NUM,
            OP_LOGE(context_->GetNodeName(), "Dims should be 5 with NCDHW format."), return ge::GRAPH_FAILED);
        a1 = xStorageShape.GetDim(DIM_0);
        a0 = xStorageShape.GetDim(DIM_1);
        r = xStorageShape.GetDim(DIM_2) * xStorageShape.GetDim(DIM_3) * xStorageShape.GetDim(DIM_4);
    } else if (format == FORMAT_NHWC) {
        OP_CHECK_IF(
            xStorageShape.GetDimNum() != NHWC_DIM_NUM,
            OP_LOGE(context_->GetNodeName(), "Dims should be 4 with NHWC format."), return ge::GRAPH_FAILED);
        a1 = xStorageShape.GetDim(DIM_0);
        r = xStorageShape.GetDim(DIM_1) * xStorageShape.GetDim(DIM_2);
        a0 = xStorageShape.GetDim(DIM_3);
    } else if (format == FORMAT_NDHWC) {
        OP_CHECK_IF(
            xStorageShape.GetDimNum() != NDHWC_DIM_NUM,
            OP_LOGE(context_->GetNodeName(), "Dims should be 5 with NDHWC format."), return ge::GRAPH_FAILED);
        a1 = xStorageShape.GetDim(DIM_0);
        r = xStorageShape.GetDim(DIM_1) * xStorageShape.GetDim(DIM_2) * xStorageShape.GetDim(DIM_3);
        a0 = xStorageShape.GetDim(DIM_4);
    } else if (format == FORMAT_ND) {
        OP_CHECK_IF(
            xStorageShape.GetDimNum() < ND_MIN_DIM_NUM || xStorageShape.GetDimNum() > ND_MAX_DIM_NUM,
            OP_LOGE(context_->GetNodeName(), "Dims should be 2~8 with ND format."), return ge::GRAPH_FAILED);
        a1 = xStorageShape.GetDim(DIM_0);
        a0 = xStorageShape.GetDim(DIM_1);
        r = xStorageShape.GetShapeSize() / a1 / a0;
    } else {
        OP_LOGE(context_->GetNodeName(), "Not supported format.");
        return ge::GRAPH_FAILED;
    }
    OP_CHECK_IF(a1 <= 0 || a0 <= 0, 
        OP_LOGE(context_->GetNodeName(),"Input dims N (=%ld) and C (=%ld) must be positive (>0).", a1, a0),
        return ge::GRAPH_FAILED);

    if (CheckDtypeValid() != ge::GRAPH_SUCCESS) {
        OP_LOGE(context_->GetNodeName(), "Not supported datatype info.");
        return ge::GRAPH_FAILED;
    }
    if (CheckShapeValid() != ge::GRAPH_SUCCESS) {
        OP_LOGE(context_->GetNodeName(), "Not supported shape info.");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus InstanceNormRegbaseTilingBase::CheckDtypeValid()
{
    // Step1.校验x数据类型
    OP_CHECK_IF(
        std::find(DTYPE_LIST.begin(), DTYPE_LIST.end(), dataType) == DTYPE_LIST.end(),
        OP_LOGE(context_->GetNodeName(), "Unsupported dtype %s for input 0.",
            ge::TypeUtils::DataTypeToSerialString(dataType).c_str()), return ge::GRAPH_FAILED);
    
    // Step2.校验gamma/betta数据类型
    OP_CHECK_IF(
        // 支持gamma和x的混合数据类型
        (gammaDataType != dataType) && (gammaDataType != ge::DT_FLOAT),
        OP_LOGE(context_->GetNodeName(), "Dtype of input gamma expect %s, but actual %s.",
            ge::TypeUtils::DataTypeToSerialString(dataType).c_str(),
            ge::TypeUtils::DataTypeToSerialString(gammaDataType).c_str()), return ge::GRAPH_FAILED);
    
    auto bettaDesc = context_->GetInputDesc(INPUT_BETA_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, bettaDesc);
    ge::DataType bettaDataType = bettaDesc->GetDataType();
    OP_CHECK_IF(
        (bettaDataType != gammaDataType),
        OP_LOGE(context_->GetNodeName(), "Dtype of input betta expect %s, but actual %s.",
            ge::TypeUtils::DataTypeToSerialString(gammaDataType).c_str(),
            ge::TypeUtils::DataTypeToSerialString(bettaDataType).c_str()), return ge::GRAPH_FAILED);
    
    // Step3.校验输出y数据类型
    auto yDesc = context_->GetOutputDesc(OUTPUT_Y_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, yDesc);
    ge::DataType yDataType = yDesc->GetDataType();
    OP_CHECK_IF(
        (yDataType != dataType),
        OP_LOGE(context_->GetNodeName(), "Dtype of output y expect %s, but actual %s.",
            ge::TypeUtils::DataTypeToSerialString(dataType).c_str(),
            ge::TypeUtils::DataTypeToSerialString(yDataType).c_str()), return ge::GRAPH_FAILED);
    
    // Step4.校验输出mean/varience数据类型
    OP_CHECK_IF(
        // 支持mean和x的混合数据类型
        (meanDataType != dataType) && (meanDataType != ge::DT_FLOAT),
        OP_LOGE(context_->GetNodeName(), "Dtype of output mean expect %s, but actual %s.",
            ge::TypeUtils::DataTypeToSerialString(dataType).c_str(),
            ge::TypeUtils::DataTypeToSerialString(meanDataType).c_str()), return ge::GRAPH_FAILED);
    
    auto varDesc = context_->GetOutputDesc(OUTPUT_VARIANCE_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, varDesc);
    ge::DataType varDataType = varDesc->GetDataType();
    OP_CHECK_IF(
        (varDataType != meanDataType),
        OP_LOGE(context_->GetNodeName(), "Dtype of output variance expect %s, but actual %s.",
            ge::TypeUtils::DataTypeToSerialString(meanDataType).c_str(),
            ge::TypeUtils::DataTypeToSerialString(varDataType).c_str()), return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus InstanceNormRegbaseTilingBase::CheckShapeValid()
{
    // Step1.校验y的shape与x是否一致
    if (CheckXYShapeValid() != ge::GRAPH_SUCCESS) {
        OP_LOGE(context_->GetNodeName(), "Not supported X Y shape info.");
        return ge::GRAPH_FAILED;
    }
    
    // Step2.校验gamma/betta的shape
    if (CheckGammaBettaShapeValid() != ge::GRAPH_SUCCESS) {
        OP_LOGE(context_->GetNodeName(), "Not supported gamma betta shape info.");
        return ge::GRAPH_FAILED;
    }
    
    // Step3.校验输出mean/varience的shape
    if (CheckMeanVarianceShapeValid() != ge::GRAPH_SUCCESS) {
        OP_LOGE(context_->GetNodeName(), "Not supported mean varience shape info.");
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus InstanceNormRegbaseTilingBase::CheckXYShapeValid()
{
    int64_t xShapeSize = xStorageShape.GetDimNum();

    auto yShape = context_->GetOutputShape(OUTPUT_Y_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, yShape);
    auto yStorageShape = yShape->GetStorageShape();
    int64_t yShapeSize = yStorageShape.GetDimNum();

    auto yDesc = context_->GetOutputDesc(OUTPUT_Y_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, yDesc);
    ge::Format yFormat = yDesc->GetFormat().GetStorageFormat();

    OP_CHECK_IF(
        (xShapeSize != yShapeSize),
        OP_LOGE(
            context_->GetNodeName(), "Input X dim size [%ld] is not equal to Output Y dim size [%ld]",
            xShapeSize, yShapeSize), return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        (format != yFormat),
        OP_LOGE(
            context_->GetNodeName(), "Input X format [%s] does not match Output Y format [%s]",
            ge::TypeUtils::FormatToAscendString(format).GetString(), ge::TypeUtils::FormatToAscendString(yFormat).GetString()),
        return ge::GRAPH_FAILED);

    for (int64_t i = 0; i < xShapeSize; i++) {
        OP_CHECK_IF((xStorageShape.GetDim(i) != yStorageShape.GetDim(i)),
            OP_LOGE(
                context_->GetNodeName(),
                "Input X dim [%ld] is [%ld] and Output Y dim [%ld] is [%ld] should be same", i,
                xStorageShape.GetDim(i), i, yStorageShape.GetDim(i)),
            return ge::GRAPH_FAILED);
    }
    OP_LOGI(context_->GetNodeName(), "CheckXYShapeValid success.");
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus InstanceNormRegbaseTilingBase::CheckGammaBettaShapeValid()
{
    // 依次对gamma、betta的format和shape进行校验
    for (int64_t i = INPUT_GAMMA_INDEX; i <= INPUT_BETA_INDEX; i++) {
        auto gammaBettaShape = context_->GetInputShape(i);
        OP_CHECK_NULL_WITH_CONTEXT(context_, gammaBettaShape);
        auto gammaBettaStorageShape = gammaBettaShape->GetStorageShape();
        int64_t gammaBettaShapeSize = gammaBettaStorageShape.GetDimNum();
        
        OP_CHECK_IF(
            (gammaBettaShapeSize != 1),
            OP_LOGE(context_->GetNodeName(), "Input [%ld] dim size [%ld] should be 1",
                    i, gammaBettaShapeSize), return ge::GRAPH_FAILED);

        OP_CHECK_IF(
            (gammaBettaStorageShape.GetDim(DIM_0) != a0),
            OP_LOGE(
                context_->GetNodeName(), "Input [%ld] dim 0 is [%ld], whitch should be %ld (input channel dim)",
                i, gammaBettaStorageShape.GetDim(DIM_0), a0), return ge::GRAPH_FAILED);
    }

    OP_LOGI(context_->GetNodeName(), "CheckGammaBettaShapeValid success.");
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus InstanceNormRegbaseTilingBase::CheckMeanVarianceShapeValid()
{
    int64_t xShapeSize = xStorageShape.GetDimNum();
    // 依次对mean、variance的format和shape进行校验
    for (int64_t i = OUTPUT_MEAN_INDEX; i <= OUTPUT_VARIANCE_INDEX; i++) {
        auto meanVarianceShape = context_->GetOutputShape(i);
        OP_CHECK_NULL_WITH_CONTEXT(context_, meanVarianceShape);
        auto meanVarianceStorageShape = meanVarianceShape->GetStorageShape();
        int64_t meanVarianceShapeSize = meanVarianceStorageShape.GetDimNum();

        OP_CHECK_IF(
            (meanVarianceShapeSize != xShapeSize),
            OP_LOGE(
                context_->GetNodeName(), "Output [%ld] dim size [%ld] is not equal to Input X dim size [%ld]",
                i, meanVarianceShapeSize, xShapeSize), return ge::GRAPH_FAILED);
        // 针对不同format，对各个维度进行校验
        for (int64_t j = 0; j < meanVarianceShapeSize; j++) {
            if (j == 0) {
                // 第一个维度，需要与X的第一维相等（N轴）
                OP_CHECK_IF(
                    (meanVarianceStorageShape.GetDim(j) != xStorageShape.GetDim(j)),
                    OP_LOGE(
                        context_->GetNodeName(), "Output [%ld] dim [%ld] is [%ld], whitch should be %ld",
                        i, j, meanVarianceStorageShape.GetDim(j), xStorageShape.GetDim(j)), return ge::GRAPH_FAILED);
            } else if (j == 1 && (format == ge::FORMAT_NCHW || format == ge::FORMAT_NCDHW || format == ge::FORMAT_ND)) {
                // 第二个维度，对于X的NCHW/NCDHW/ND格式下，需要与X的第二维相等（C轴）
                OP_CHECK_IF(
                    (meanVarianceStorageShape.GetDim(j) != xStorageShape.GetDim(j)),
                    OP_LOGE(
                        context_->GetNodeName(), "Output [%ld] dim [%ld] is [%ld], whitch should be %ld",
                        i, j, meanVarianceStorageShape.GetDim(j), xStorageShape.GetDim(j)), return ge::GRAPH_FAILED);
            } else if (j == (meanVarianceShapeSize - 1) && (format == ge::FORMAT_NHWC || format == ge::FORMAT_NDHWC)) {
                // 最后一个维度，对于X的NHWC/NDHWC格式下，需要与X的最后一维相等（C轴）
                OP_CHECK_IF(
                    (meanVarianceStorageShape.GetDim(j) != xStorageShape.GetDim(j)),
                    OP_LOGE(
                        context_->GetNodeName(), "Output [%ld] dim [%ld] is [%ld], whitch should be %ld",
                        i, j, meanVarianceStorageShape.GetDim(j), xStorageShape.GetDim(j)), return ge::GRAPH_FAILED);
            } else {
                // 其他维度需要是1
                OP_CHECK_IF(
                    (meanVarianceStorageShape.GetDim(j) != 1),
                    OP_LOGE(
                        context_->GetNodeName(), "Output [%ld] dim [%ld] is [%ld], whitch should be 1",
                        i, j, meanVarianceStorageShape.GetDim(j)), return ge::GRAPH_FAILED);
            }
        }
    }
    OP_LOGI(context_->GetNodeName(), "CheckMeanVarianceShapeValid success.");
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus InstanceNormRegbaseTilingBase::CheckShapeAllNotNegative(gert::Shape& shape)
{
    for (size_t i = 0; i < shape.GetDimNum(); i++) {
        OP_CHECK_IF(
            shape.GetDim(i) < 0,
            OP_LOGE(
                context_->GetNodeName(), "Dim %lu of input expect be not negative, but actual %ld.", i, shape.GetDim(i)),
            return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus InstanceNormRegbaseTilingBase::DoLibApiTiling()
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus InstanceNormRegbaseTilingBase::GetWorkspaceSize()
{
    // 计算workspace大小
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context_->GetPlatformInfo());
    workspaceSize_ = ascendcPlatform.GetLibApiWorkSpaceSize();
    return ge::GRAPH_SUCCESS;
}
} // namespace optiling
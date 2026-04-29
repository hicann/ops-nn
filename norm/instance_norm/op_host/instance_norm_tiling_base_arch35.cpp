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
using namespace Ops::Base;

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
    int64_t xDimNum = xStorageShape.GetDimNum();
    if (format == FORMAT_NCHW) {
        OP_CHECK_IF(
            xDimNum != NCHW_DIM_NUM,
            OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(context_->GetNodeName(), "x", 
                std::to_string(xDimNum).c_str(), "The dim num of input x should be 4 in NCHW format"),
            return ge::GRAPH_FAILED);
        a1 = xStorageShape.GetDim(DIM_0);
        a0 = xStorageShape.GetDim(DIM_1);
        r = xStorageShape.GetDim(DIM_2) * xStorageShape.GetDim(DIM_3);
    } else if (format == FORMAT_NCDHW) {
        OP_CHECK_IF(
            xDimNum != NCDHW_DIM_NUM,
            OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(context_->GetNodeName(), "x",
                std::to_string(xDimNum).c_str(), "The dim num of input x should be 5 in NCDHW format"),
            return ge::GRAPH_FAILED);
        a1 = xStorageShape.GetDim(DIM_0);
        a0 = xStorageShape.GetDim(DIM_1);
        r = xStorageShape.GetDim(DIM_2) * xStorageShape.GetDim(DIM_3) * xStorageShape.GetDim(DIM_4);
    } else if (format == FORMAT_NHWC) {
        OP_CHECK_IF(
            xDimNum != NHWC_DIM_NUM,
            OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(context_->GetNodeName(), "x",
                std::to_string(xDimNum).c_str(), "The dim num of input x should be 4 in NHWC format"),
            return ge::GRAPH_FAILED);
        a1 = xStorageShape.GetDim(DIM_0);
        r = xStorageShape.GetDim(DIM_1) * xStorageShape.GetDim(DIM_2);
        a0 = xStorageShape.GetDim(DIM_3);
    } else if (format == FORMAT_NDHWC) {
        OP_CHECK_IF(
            xDimNum != NDHWC_DIM_NUM,
            OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(context_->GetNodeName(), "x",
                std::to_string(xDimNum).c_str(), "The dim num of input x should be 5 in NDHWC format"),
            return ge::GRAPH_FAILED);
        a1 = xStorageShape.GetDim(DIM_0);
        r = xStorageShape.GetDim(DIM_1) * xStorageShape.GetDim(DIM_2) * xStorageShape.GetDim(DIM_3);
        a0 = xStorageShape.GetDim(DIM_4);
    } else if (format == FORMAT_ND) {
        OP_CHECK_IF(
            xDimNum < ND_MIN_DIM_NUM || xDimNum > ND_MAX_DIM_NUM,
            OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(context_->GetNodeName(), "x",
                std::to_string(xDimNum).c_str(),
                "The dim num of input x should be in the range of [2, 8] in ND format"),
            return ge::GRAPH_FAILED);
        a1 = xStorageShape.GetDim(DIM_0);
        a0 = xStorageShape.GetDim(DIM_1);
        r = xStorageShape.GetShapeSize() / a1 / a0;
    } else {
        OP_LOGE_FOR_INVALID_FORMAT(context_->GetNodeName(), "x", ToString(format).c_str(),
            "NCHW, NCDHW, NHWC, NDHWC or ND");
        return ge::GRAPH_FAILED;
    }
    if (a1 <= 0 || a0 <= 0) {
        std::string reasonMsg = "The N axis and C axis of input x must be positive, "
            "where N refers to the 0th dim, "
            "C refers to the 1st dim in NCHW, NCDHW or ND, the last in NHWC and NDHWC";
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context_->GetNodeName(), "x", ToString(xStorageShape).c_str(),
            reasonMsg.c_str());
        return ge::GRAPH_FAILED;
    }

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
        OP_LOGE_FOR_INVALID_DTYPE(context_->GetNodeName(), "x", ToString(dataType).c_str(), "FLOAT, FLOAT16 or BF16"),
        return ge::GRAPH_FAILED);
    
    // Step2.校验gamma/betta数据类型
    if ((gammaDataType != dataType) && (gammaDataType != ge::DT_FLOAT)) {
        std::string dtypeMsg = ToString(gammaDataType);
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(context_->GetNodeName(), "gamma", dtypeMsg.c_str(),
            "The dtype of input gamma should be FLOAT or the same as the dtype of input x");
        return ge::GRAPH_FAILED;
    }

    auto bettaDesc = context_->GetInputDesc(INPUT_BETA_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, bettaDesc);
    ge::DataType bettaDataType = bettaDesc->GetDataType();
    if (bettaDataType != gammaDataType) {
        std::string dtypeMsg = ToString(bettaDataType) + " and " + ToString(gammaDataType);
        OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(context_->GetNodeName(), "beta and gamma", dtypeMsg.c_str(),
            "The dtypes of input beta and input gamma should be the same");
        return ge::GRAPH_FAILED;
    }

    // Step3.校验输出y数据类型
    auto yDesc = context_->GetOutputDesc(OUTPUT_Y_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, yDesc);
    ge::DataType yDataType = yDesc->GetDataType();
    if (yDataType != dataType) {
        std::string dtypeMsg = ToString(yDataType) + " and " + ToString(dataType);
        OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(context_->GetNodeName(), "y and x", dtypeMsg.c_str(),
            "The dtypes of output y and input x should be the same");
        return ge::GRAPH_FAILED;
    }

    // Step4.校验输出mean/varience数据类型
    if ((meanDataType != dataType) && (meanDataType != ge::DT_FLOAT)) {
        std::string dtypeMsg = ToString(meanDataType);
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(context_->GetNodeName(), "mean", dtypeMsg.c_str(),
            "The dtype of output mean should be FLOAT or the same as the dtype of input x");
        return ge::GRAPH_FAILED;
    }

    auto varDesc = context_->GetOutputDesc(OUTPUT_VARIANCE_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, varDesc);
    ge::DataType varDataType = varDesc->GetDataType();
    if (varDataType != meanDataType) {
        std::string dtypeMsg = ToString(varDataType) + " and " + ToString(meanDataType);
        OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(context_->GetNodeName(), "variance and mean", dtypeMsg.c_str(),
            "The dtypes of output variance and output mean should be the same");
        return ge::GRAPH_FAILED;
    }

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
    int64_t xDimNum = xStorageShape.GetDimNum();

    auto yShape = context_->GetOutputShape(OUTPUT_Y_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, yShape);
    auto yStorageShape = yShape->GetStorageShape();
    int64_t yDimNum = yStorageShape.GetDimNum();

    auto yDesc = context_->GetOutputDesc(OUTPUT_Y_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, yDesc);
    ge::Format yFormat = yDesc->GetFormat().GetStorageFormat();

    if (xDimNum != yDimNum) {
        std::string dimsMsg = std::to_string(xDimNum) + " and " + std::to_string(yDimNum);
        OP_LOGE_FOR_INVALID_SHAPEDIMS_WITH_REASON(context_->GetNodeName(), "x and y", dimsMsg.c_str(),
            "The dim nums of input x and output y  should be the same");
        return ge::GRAPH_FAILED;
    }
    if (format != yFormat) {
        std::string formatMsg = ToString(format) + " and " + ToString(yFormat);
        OP_LOGE_FOR_INVALID_FORMATS_WITH_REASON(context_->GetNodeName(), "x and y", formatMsg.c_str(),
            "The formats of input x and output y should be the same");
        return ge::GRAPH_FAILED;
    }


    for (int64_t i = 0; i < xDimNum; i++) {
        if (xStorageShape.GetDim(i) != yStorageShape.GetDim(i)) {
            std::string shapeMsg = ToString(xStorageShape) + " and " + ToString(yStorageShape);
            OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(context_->GetNodeName(), "x and y", shapeMsg.c_str(),
                "The shapes of input x and output y should be the same");
            return ge::GRAPH_FAILED;    
        }
    }
    OP_LOGI(context_->GetNodeName(), "CheckXYShapeValid success.");
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus InstanceNormRegbaseTilingBase::CheckGammaBettaShapeValid()
{
    // 依次对gamma、betta的format和shape进行校验
    const std::vector<std::pair<int, std::string>> gammaBetta = {
        {INPUT_GAMMA_INDEX, "gamma"}, {INPUT_BETA_INDEX, "beta"}
    };
    for (const auto& [inputIdx, inputName] : gammaBetta) {
        auto gammaBettaShape = context_->GetInputShape(inputIdx);
        OP_CHECK_NULL_WITH_CONTEXT(context_, gammaBettaShape);
        auto gammaBettaStorageShape = gammaBettaShape->GetStorageShape();
        int64_t gammaBettaDimNum = gammaBettaStorageShape.GetDimNum();
        
        if (gammaBettaDimNum != 1) {
            OP_LOGE_FOR_INVALID_SHAPEDIM(context_->GetNodeName(), inputName.c_str(),
                std::to_string(gammaBettaDimNum).c_str(), "1D");
            return ge::GRAPH_FAILED;
        }

        if (gammaBettaStorageShape.GetDim(DIM_0) != a0) {
            std::string paramMsg = inputName + " and x";
            std::string shapeMsg = ToString(gammaBettaStorageShape) + " and " + ToString(xStorageShape);
            std::string reasonMsg = "The 0th axis of input " + inputName +
                " should be equal to the C axis of input x, "
                "where C refers to the 1st dim in NCHW, NCDHW or ND, the last in NHWC and NDHWC";
            OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(context_->GetNodeName(), paramMsg.c_str(),
                shapeMsg.c_str(), reasonMsg.c_str());
            return ge::GRAPH_FAILED;
        }
    }

    OP_LOGI(context_->GetNodeName(), "CheckGammaBettaShapeValid success.");
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus InstanceNormRegbaseTilingBase::CheckMeanVarianceShapeValid()
{
    // 依次对mean、variance的format和shape进行校验
    const std::vector<std::pair<int, std::string>> meanVariance = {
        {OUTPUT_MEAN_INDEX, "mean"}, {OUTPUT_VARIANCE_INDEX, "variance"}
    };
    int64_t xDimNum = xStorageShape.GetDimNum();
    for (const auto& [outputIdx, outputName] : meanVariance) {
        auto meanVarianceShape = context_->GetOutputShape(outputIdx);
        OP_CHECK_NULL_WITH_CONTEXT(context_, meanVarianceShape);
        auto meanVarianceStorageShape = meanVarianceShape->GetStorageShape();
        int64_t meanVarianceDimNum = meanVarianceStorageShape.GetDimNum();

        if (meanVarianceDimNum != xDimNum) {
            std::string paramMsg = outputName + " and x";
            std::string dimNumMsg = std::to_string(meanVarianceDimNum) + " and " + std::to_string(xDimNum);
            std::string reasonMsg = "The dim nums of output " + outputName + " and input x should be the same";
            OP_LOGE_FOR_INVALID_SHAPEDIMS_WITH_REASON(context_->GetNodeName(), paramMsg.c_str(),
                dimNumMsg.c_str(), reasonMsg.c_str());
            return ge::GRAPH_FAILED;
        }

        // 针对不同format，对各个维度进行校验
        std::string paramMsg = outputName + " and x";
        std::string shapeMsg = ToString(meanVarianceStorageShape) + " and " + ToString(xStorageShape);
        for (int64_t j = 0; j < meanVarianceDimNum; j++) {
            if (j == 0) {
                // 第一个维度，需要与X的第一维相等（N轴）
                if (meanVarianceStorageShape.GetDim(j) != xStorageShape.GetDim(j)) {
                    std::string reasonMsg = "The N axis of output " + outputName +
                        " should be equal to the N axis of input x, where N refers to the 0th dim";
                    OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(context_->GetNodeName(), paramMsg.c_str(),
                        shapeMsg.c_str(), reasonMsg.c_str());
                    return ge::GRAPH_FAILED;
                }
            } else if (j == 1 && (format == ge::FORMAT_NCHW || format == ge::FORMAT_NCDHW || format == ge::FORMAT_ND)) {
                // 第二个维度，对于X的NCHW/NCDHW/ND格式下，需要与X的第二维相等（C轴）
                if (meanVarianceStorageShape.GetDim(j) != xStorageShape.GetDim(j)) {
                    std::string reasonMsg =" The C axis of output " + outputName +
                        " should be equal to the C axis of input x, "
                        "where C refers to the 1st dim when x's format is NCHW, NCDHW or ND";
                    OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(context_->GetNodeName(), paramMsg.c_str(),
                        shapeMsg.c_str(), reasonMsg.c_str());
                    return ge::GRAPH_FAILED;
                }
            } else if (j == (meanVarianceDimNum - 1) && (format == ge::FORMAT_NHWC || format == ge::FORMAT_NDHWC)) {
                // 最后一个维度，对于X的NHWC/NDHWC格式下，需要与X的最后一维相等（C轴）
                if (meanVarianceStorageShape.GetDim(j) != xStorageShape.GetDim(j)) {
                    std::string reasonMsg =" The C axis of output " + outputName +
                        " should be equal to the C axis of input x, "
                        "where C refers to the last dim when x's format is NHWC or NDHWC";
                    OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(context_->GetNodeName(), paramMsg.c_str(),
                        shapeMsg.c_str(), reasonMsg.c_str());
                    return ge::GRAPH_FAILED;
                }
            } else {
                // 其他维度需要是1
                if (meanVarianceStorageShape.GetDim(j) != 1) {
                    std::string reasonMsg = "All axes of output " + outputName +
                        " except the N axis and C axis should be 1, "
                        "where N refers to the 0th dim, "
                        "C refers to the 1st dim when x's format is NCHW, NCDHW or ND, "
                        "or the last dim when x's format is NHWC or NDHWC";
                    OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context_->GetNodeName(), outputName.c_str(),
                        ToString(meanVarianceStorageShape).c_str(), reasonMsg.c_str());
                    return ge::GRAPH_FAILED;
                }
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
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context_->GetNodeName(), "x", ToString(shape).c_str(),
                "The shape of input x can not be an invalid tensor with a negative dim"),
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
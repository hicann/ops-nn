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
 * \file group_norm_grad_tiling_base.cpp
 * \brief
 */

#include "group_norm_grad_tiling.h"
#include <unordered_map>

namespace optiling {
static constexpr uint16_t INPUT_IDX_DY = 0;
static constexpr uint16_t INPUT_IDX_MEAN = 1;
static constexpr uint16_t INPUT_IDX_RSTD = 2;
static constexpr uint16_t INPUT_IDX_X = 3;
static constexpr uint16_t INPUT_IDX_GAMMA = 4;
static constexpr uint16_t OUTPUT_IDX_DX = 0;
static constexpr uint16_t OUTPUT_IDX_DGAMMA = 1;
static constexpr uint16_t OUTPUT_IDX_DBETA = 2;
static constexpr uint16_t DIM0 = 0;
static constexpr uint16_t DIM1 = 1;
static constexpr uint16_t MIN_X_DIM = 2;
static constexpr uint32_t UPPER_CARRYING_LIMIT = 8000;
static constexpr uint32_t FLOAT_DTYPE_BYTES = 4;

static const std::unordered_map<ge::DataType, uint32_t> DATA_TYPE_TO_INT{
    {ge::DataType::DT_FLOAT, 1}, {ge::DataType::DT_FLOAT16, 2}, {ge::DataType::DT_BF16, 3}};

ge::graphStatus GroupNormGradTilingBase::GetPlatformInfo()
{
    OP_LOGD(context_->GetNodeName(), "Tiling initializing.");
    // Get compileInfo
    auto compileInfo = context_->GetCompileInfo<GroupNormGradCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context_, compileInfo);
    commonParams.ubSizePlatForm = compileInfo->ubSizePlatForm;
    OP_TILING_CHECK((commonParams.ubSizePlatForm <= 0),
                    OP_LOGE(context_->GetNodeName(), "ubSize should be greater than zero."), return ge::GRAPH_FAILED);
    commonParams.coreNum = compileInfo->totalCoreNum;
    OP_TILING_CHECK((commonParams.coreNum <= 0),
                    OP_LOGE(context_->GetNodeName(), "core num should be greater than zero."), return ge::GRAPH_FAILED);
    commonParams.vectorLen = compileInfo->vectorLen / FLOAT_DTYPE_BYTES;
    OP_TILING_CHECK((commonParams.vectorLen <= 0),
                    OP_LOGE(context_->GetNodeName(), "vectorLen should be greater than zero."),
                    return ge::GRAPH_FAILED);
    commonParams.blockSize = compileInfo->blockSize;
    OP_TILING_CHECK((commonParams.blockSize <= 0),
                    OP_LOGE(context_->GetNodeName(), "blockSize should be greater than zero."),
                    return ge::GRAPH_FAILED);
    commonParams.sysWorkspaceSize = compileInfo->sysWorkspaceSize;
    commonParams.isRegBase = compileInfo->isRegBase;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GroupNormGradTilingBase::GetShapeAttrsInfo()
{
    auto dyDesc = context_->GetInputDesc(INPUT_IDX_DY);
    OP_CHECK_NULL_WITH_CONTEXT(context_, dyDesc);
    commonParams.tTypeStr = dyDesc->GetDataType();
    auto dyShape = context_->GetInputShape(INPUT_IDX_DY)->GetStorageShape();
    OP_TILING_CHECK(InputCheck(dyShape) == ge::GRAPH_FAILED, OP_LOGE(context_->GetNodeName(), "Input check failed."),
                    return ge::GRAPH_FAILED);

    auto attrs = context_->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context_, attrs);
    const int64_t* gValue = attrs->GetAttrPointer<int64_t>(0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, gValue);
    OP_TILING_CHECK((*gValue <= 0),
                    OP_LOGE_FOR_INVALID_VALUE(context_->GetNodeName(), "num_groups", std::to_string(*gValue).c_str(),
                                              "bigger than 0"),
                    return ge::GRAPH_FAILED);
    commonParams.G = static_cast<int64_t>(*gValue);
    commonParams.N = dyShape.GetDim(DIM0);
    commonParams.C = dyShape.GetDim(DIM1);
    commonParams.CPerG = commonParams.C / commonParams.G;
    OP_TILING_CHECK(ParamsCheck() == ge::GRAPH_FAILED, OP_LOGE(context_->GetNodeName(), "Params check failed."),
                    return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GroupNormGradTilingBase::InputCheck(gert::Shape& dyShape)
{
    auto xDesc = context_->GetInputDesc(INPUT_IDX_X);
    auto dxDesc = context_->GetOutputDesc(OUTPUT_IDX_DX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, xDesc);
    OP_CHECK_NULL_WITH_CONTEXT(context_, dxDesc);
    auto xShape = context_->GetInputShape(INPUT_IDX_X)->GetStorageShape();
    auto dxShape = context_->GetOutputShape(OUTPUT_IDX_DX)->GetStorageShape();
    ge::DataType xDtypeStr = xDesc->GetDataType();
    ge::DataType dxDtypeStr = dxDesc->GetDataType();
    OP_TILING_CHECK(
        (commonParams.tTypeStr != dxDtypeStr || dxDtypeStr != xDtypeStr),
        OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(context_->GetNodeName(), "dy, dx and x",
                                               (ge::TypeUtils::DataTypeToSerialString(commonParams.tTypeStr) + ", " +
                                                ge::TypeUtils::DataTypeToSerialString(dxDtypeStr) + " and " +
                                                ge::TypeUtils::DataTypeToSerialString(xDtypeStr))
                                                   .c_str(),
                                               "The dtypes of dy, dx and x must be same"),
        return ge::GRAPH_FAILED);
    auto iter = DATA_TYPE_TO_INT.find(commonParams.tTypeStr);
    if (iter == DATA_TYPE_TO_INT.end()) {
        OP_LOGE_FOR_INVALID_DTYPE(context_->GetNodeName(), "x",
                                  ge::TypeUtils::DataTypeToSerialString(commonParams.tTypeStr).c_str(),
                                  "float32, float16 or bfloat16");
        return ge::GRAPH_FAILED;
    }
    if (dyShape != dxShape || dxShape != xShape) {
        std::string incorrectShapes = Ops::Base::ToString(xShape) + ", " + Ops::Base::ToString(dyShape) + " and " +
                                      Ops::Base::ToString(dxShape);
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(context_->GetNodeName(), "x, dy and dx", incorrectShapes.c_str(),
                                               "The shapes of x, dy and dx must be the same");
        return ge::GRAPH_FAILED;
    }
    for (uint32_t dimIdx = 0; dimIdx < dyShape.GetDimNum(); dimIdx++) {
        auto currShape = dyShape.GetDim(dimIdx);
        if (currShape < 1) {
            std::string errMsg = Ops::Base::ToString(dyShape);
            std::string reasonMsg = "All axes of dy must be greater or equal to 1. Currently, dim[" +
                                    std::to_string(currShape) + "] of dy does not meet the condition";
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context_->GetNodeName(), "dy", errMsg.c_str(), reasonMsg.c_str());
            return ge::GRAPH_FAILED;
        }
    }
    auto dimNum = dyShape.GetDimNum();
    if (dimNum < MIN_X_DIM) {
        std::string incorrectDims = std::to_string(dimNum) + ", " + std::to_string(dimNum) + " and " +
                                    std::to_string(dimNum);
        OP_LOGE_FOR_INVALID_SHAPEDIMS_WITH_REASON(context_->GetNodeName(), "x, dy and dx", incorrectDims.c_str(),
                                                  "The shape dims of x, dy and dx must be at least 2");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GroupNormGradTilingBase::ParamsCheck()
{
    // C / G must not be zero, and C must be an integer multiple of G
    // while also not exceeding the operator's current carrying capacity.
    auto dyShape = context_->GetInputShape(INPUT_IDX_DY)->GetStorageShape();
    if (commonParams.CPerG == 0 || commonParams.C % commonParams.G != 0 || commonParams.CPerG > UPPER_CARRYING_LIMIT) {
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
            context_->GetNodeName(), "dy", Ops::Base::ToString(dyShape).c_str(),
            ("The C axis(first axis) of parameter dy must can be completely divided into num_groups(" +
             std::to_string(commonParams.G) +
             ") uniform groups, "
             "and each group must not be empty or exceed the operator's current carrying capacity( " +
             std::to_string(UPPER_CARRYING_LIMIT) + ")")
                .c_str());
        return ge::GRAPH_FAILED;
    }
    auto gammaDesc = context_->GetInputDesc(INPUT_IDX_GAMMA);
    auto meanDesc = context_->GetInputDesc(INPUT_IDX_MEAN);
    auto rstdDesc = context_->GetInputDesc(INPUT_IDX_RSTD);
    auto dGammaDesc = context_->GetOutputDesc(OUTPUT_IDX_DGAMMA);
    auto dBetaDesc = context_->GetOutputDesc(OUTPUT_IDX_DBETA);
    OP_CHECK_NULL_WITH_CONTEXT(context_, gammaDesc);
    OP_CHECK_NULL_WITH_CONTEXT(context_, meanDesc);
    OP_CHECK_NULL_WITH_CONTEXT(context_, rstdDesc);
    auto gammaShape = context_->GetInputShape(INPUT_IDX_GAMMA)->GetStorageShape();
    auto meanShape = context_->GetInputShape(INPUT_IDX_MEAN)->GetStorageShape();
    auto rstdShape = context_->GetInputShape(INPUT_IDX_RSTD)->GetStorageShape();
    commonParams.uTypeStr = gammaDesc->GetDataType();
    ge::DataType meanDtypeStr = meanDesc->GetDataType();
    ge::DataType rstdDtypeStr = rstdDesc->GetDataType();
    ge::DataType dGammaDtypeStr = dGammaDesc->GetDataType();
    ge::DataType dBetaDtypeStr = dBetaDesc->GetDataType();
    int64_t gammaSize = 1;
    for (uint32_t dimIdx = 0; dimIdx < gammaShape.GetDimNum(); dimIdx++) {
        gammaSize *= gammaShape.GetDim(dimIdx);
    }
    OP_TILING_CHECK(gammaShape.GetDimNum() != 1 || gammaSize != commonParams.C,
                    OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(
                        context_->GetNodeName(), "gammaSize", Ops::Base::ToString(gammaShape).c_str(),
                        ("The shape size of gamma must be the same as C(dim[1] of dy), where C is " +
                         std::to_string(commonParams.C))
                            .c_str()),
                    return ge::GRAPH_FAILED);
    int64_t meanSize = 1;
    for (uint32_t dimIdx = 0; dimIdx < meanShape.GetDimNum(); dimIdx++) {
        meanSize *= meanShape.GetDim(dimIdx);
    }
    int64_t rstdSize = 1;
    for (uint32_t dimIdx = 0; dimIdx < rstdShape.GetDimNum(); dimIdx++) {
        rstdSize *= rstdShape.GetDim(dimIdx);
    }
    auto iter = DATA_TYPE_TO_INT.find(commonParams.uTypeStr);
    if (iter == DATA_TYPE_TO_INT.end()) {
        OP_LOGE_FOR_INVALID_DTYPE(context_->GetNodeName(), "gamma",
                                  ge::TypeUtils::DataTypeToSerialString(commonParams.uTypeStr).c_str(),
                                  "float32, float16 or bfloat16");
        return ge::GRAPH_FAILED;
    }
    if (meanDtypeStr != rstdDtypeStr || rstdDtypeStr != dGammaDtypeStr || dGammaDtypeStr != dBetaDtypeStr ||
        dBetaDtypeStr != commonParams.uTypeStr) {
        std::string dtypesMsg = ge::TypeUtils::DataTypeToSerialString(rstdDtypeStr) + " , " +
                                ge::TypeUtils::DataTypeToSerialString(meanDtypeStr) + " , " +
                                ge::TypeUtils::DataTypeToSerialString(commonParams.uTypeStr) + " , " +
                                ge::TypeUtils::DataTypeToSerialString(dBetaDtypeStr) + " and " +
                                ge::TypeUtils::DataTypeToSerialString(dGammaDtypeStr);
        OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(context_->GetNodeName(), "rstd, mean, gamma, dbeta and dgamma",
                                               dtypesMsg.c_str(),
                                               "The dtypes of rstd, mean, gamma, dbeta and dgamma must be the same");
        return ge::GRAPH_FAILED;
    }
    OP_TILING_CHECK((commonParams.tTypeStr == ge::DT_FLOAT &&
                     (commonParams.uTypeStr == ge::DT_FLOAT16 || commonParams.uTypeStr == ge::DT_BF16)),
                    OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(
                        context_->GetNodeName(), "mean and rstd",
                        (ge::TypeUtils::DataTypeToSerialString(commonParams.uTypeStr) + " and " +
                         ge::TypeUtils::DataTypeToSerialString(commonParams.uTypeStr))
                            .c_str(),
                        "The dtype of mean and rstd must be float when the dtype of x/dy/dx is float"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(meanSize != rstdSize || meanSize != commonParams.N * commonParams.G,
                    OP_LOGE_FOR_INVALID_SHAPESIZES_WITH_REASON(
                        context_->GetNodeName(), "mean and rstd",
                        (std::to_string(meanSize) + " and " + std::to_string(rstdSize)).c_str(),
                        ("The shape sizes of mean and rstd must be N * G, where N is dim[0] of dy, G is num_groups, "
                         "got N = " +
                         std::to_string(commonParams.N) + ", G = " + std::to_string(commonParams.G))
                            .c_str()),
                    return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

bool GroupNormGradTilingBase::IsCapable() { return true; }

ge::graphStatus GroupNormGradTilingBase::DoOpTiling() { return ge::GRAPH_SUCCESS; }

ge::graphStatus GroupNormGradTilingBase::DoLibApiTiling() { return ge::GRAPH_SUCCESS; }

uint64_t GroupNormGradTilingBase::GetTilingKey() const { return 0; }

ge::graphStatus GroupNormGradTilingBase::GetWorkspaceSize() { return ge::GRAPH_SUCCESS; }

ge::graphStatus GroupNormGradTilingBase::PostTiling() { return ge::GRAPH_SUCCESS; }
} // namespace optiling

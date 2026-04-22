/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file layer_norm_grad_v3_tiling_base.cc
 * \brief
 */

#include "layer_norm_grad_v3_tiling.h"

using namespace Ops::Base;

namespace optiling {
static const size_t INPUT_IDX_ZERO = 0;
static const size_t INPUT_IDX_ONE = 1;
static const size_t INPUT_IDX_TWO = 2;
static const size_t INPUT_IDX_THREE = 3;
static const size_t INPUT_IDX_FOUR = 4;
static const size_t OUTPUT_IDX_ZERO = 0;
static const size_t OUTPUT_IDX_ONE = 1;
static const size_t OUTPUT_IDX_TWO = 2;
static const size_t BASE_WSP_SIZE = 0;
static const size_t GRAD_OUT_NUM = 3;

static const std::vector<std::string> inputIdx = {"dy", "x", "rstd", "mean", "gamma"};
static const std::vector<std::string> outputIdx = {"pd_x", "pd_gamma", "pd_beta"};

bool CheckShapeSame(
    const gert::TilingContext* context_, const size_t leftIndex, const size_t rightIndex, const bool isLeftInput,
    const bool isRightInput)
{
    const gert::StorageShape* leftShape = nullptr;
    const gert::StorageShape* rightShape = nullptr;
    std::string leftName;
    std::string rightName;
    if (isLeftInput) {
        leftShape = context_->GetInputShape(leftIndex);
        leftName = "input " + inputIdx[leftIndex];
        OP_CHECK_NULL_WITH_CONTEXT(context_, leftShape);
    } else {
        leftShape = context_->GetOutputShape(leftIndex);
        leftName = "output " + outputIdx[leftIndex];
        OP_CHECK_NULL_WITH_CONTEXT(context_, leftShape);
    }
    if (isRightInput) {
        rightShape = context_->GetInputShape(rightIndex);
        rightName = "input " + inputIdx[rightIndex];
        OP_CHECK_NULL_WITH_CONTEXT(context_, rightShape);
    } else {
        rightShape = context_->GetOutputShape(rightIndex);
        rightName = "output " + outputIdx[rightIndex];
        OP_CHECK_NULL_WITH_CONTEXT(context_, rightShape);
    }

    // get storage shape
    gert::Shape leftShapeVal = leftShape->GetStorageShape();
    gert::Shape rightShapeVal = rightShape->GetStorageShape();

    // check the leftIndex shape and rightIndex shape are the same
    if (leftShapeVal != rightShapeVal) {
        std::string paramMsg = rightName + " and " + leftName;
        std::string shapeMsg = ToString(rightShapeVal) + " and " + ToString(leftShapeVal);
        std::string reasonMsg = "The shapes of " + rightName + " and " + leftName + " should be the same";
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(context_->GetNodeName(), paramMsg.c_str(),
            shapeMsg.c_str(), reasonMsg.c_str());
        return false;
    }
    return true;
}

ge::graphStatus InputDtypeCheck(
    gert::TilingContext* context_, ge::DataType dyDtype, ge::DataType xDtype, 
    ge::DataType rstdDtype, ge::DataType meanDtype, ge::DataType gammaDtype)
{
    // input check
    OP_CHECK_IF(
        (dyDtype != ge::DataType::DT_FLOAT) && (dyDtype != ge::DataType::DT_FLOAT16) && (dyDtype != ge::DataType::DT_BF16),
        OP_LOGE_FOR_INVALID_DTYPE(context_->GetNodeName(), "dy", ToString(dyDtype).c_str(), "FLOAT, FLOAT16 or BF16"),
        return ge::GRAPH_FAILED);

    if (xDtype != dyDtype) {
        std::string dtypeMsg = ToString(xDtype) + " and " + ToString(dyDtype);
        OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(context_->GetNodeName(), "x and dy", dtypeMsg.c_str(),
            "The dtypes of input x and input dy should be the same");
        return ge::GRAPH_FAILED;
    }
    OP_CHECK_IF(
        rstdDtype != ge::DataType::DT_FLOAT,
        OP_LOGE_FOR_INVALID_DTYPE(context_->GetNodeName(), "rstd", ToString(rstdDtype).c_str(), "FLOAT"),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        meanDtype != ge::DataType::DT_FLOAT,
        OP_LOGE_FOR_INVALID_DTYPE(context_->GetNodeName(), "mean", ToString(meanDtype).c_str(), "FLOAT"),
        return ge::GRAPH_FAILED);
    if ((gammaDtype != dyDtype) && (gammaDtype != ge::DataType::DT_FLOAT)) {
        std::string dtypeMsg = ToString(gammaDtype) + " and " + ToString(dyDtype);
        OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(
            context_->GetNodeName(), "gamma and dy", dtypeMsg.c_str(),
            "The dtype of input gamma should be FLOAT or the same as the dtype of input dy");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus LayerNormGradV3TilingBase::GetShapeAttrsInfo()
{
    // check dy and x and pdx shape must be the same
    CheckShapeSame(context_, INPUT_IDX_ZERO, INPUT_IDX_ONE, true, true);
    CheckShapeSame(context_, INPUT_IDX_ZERO, OUTPUT_IDX_ZERO, true, false);
    // check rstd and mean shape must be the same
    CheckShapeSame(context_, INPUT_IDX_TWO, INPUT_IDX_THREE, true, true);
    // check gamma and pdbeta and pdgamma shape must be the same
    CheckShapeSame(context_, INPUT_IDX_FOUR, OUTPUT_IDX_ONE, true, false);
    CheckShapeSame(context_, INPUT_IDX_FOUR, OUTPUT_IDX_TWO, true, false);

    // get input
    auto dyDesc = context_->GetInputDesc(INPUT_IDX_ZERO);
    OP_CHECK_NULL_WITH_CONTEXT(context_, dyDesc);
    commonParams.dyDtype = dyDesc->GetDataType();

    auto xDesc = context_->GetInputDesc(INPUT_IDX_ONE);
    OP_CHECK_NULL_WITH_CONTEXT(context_, xDesc);
    commonParams.xDtype = xDesc->GetDataType();

    auto rstdDesc = context_->GetInputDesc(INPUT_IDX_TWO);
    OP_CHECK_NULL_WITH_CONTEXT(context_, rstdDesc);
    commonParams.rstdDtype = rstdDesc->GetDataType();

    auto meanDesc = context_->GetInputDesc(INPUT_IDX_THREE);
    OP_CHECK_NULL_WITH_CONTEXT(context_, meanDesc);
    commonParams.meanDtype = meanDesc->GetDataType();

    auto gammaDesc = context_->GetInputDesc(INPUT_IDX_FOUR);
    OP_CHECK_NULL_WITH_CONTEXT(context_, gammaDesc);
    commonParams.gammaDtype = gammaDesc->GetDataType();

    // get shape
    auto dy = context_->GetInputShape(INPUT_IDX_ZERO);
    OP_CHECK_NULL_WITH_CONTEXT(context_, dy);
    auto dyShape = dy->GetStorageShape();
    int64_t dyDimNum = dyShape.GetDimNum();

    auto gamma = context_->GetInputShape(INPUT_IDX_FOUR);
    OP_CHECK_NULL_WITH_CONTEXT(context_, gamma);
    auto gammaShape = gamma->GetStorageShape();
    int64_t gammaDimNum = gammaShape.GetDimNum();

    if (dyDimNum < gammaDimNum) {
        std::string dimsMsg = std::to_string(dyDimNum) + " and " + std::to_string(gammaDimNum);
        OP_LOGE_FOR_INVALID_SHAPEDIMS_WITH_REASON(
            context_->GetNodeName(), "dy and gamma", dimsMsg.c_str(),
            "The dimNum of input dy should be greater than or equal to the dimNum of input gamma");
        return ge::GRAPH_FAILED;
    }
    // fuse dims
    int64_t row = 1;
    int64_t col = 1;
    for (int64_t i = 0; i < dyDimNum; i++) {
        OP_CHECK_IF(
            (dyShape.GetDim(i) <= 0),
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                context_->GetNodeName(), "dy", ToString(dyShape).c_str(),
                "The shape of input dy can not be an empty tensor or an invalid tensor with a negative dim"),
            return ge::GRAPH_FAILED);
        if (i < dyDimNum - gammaDimNum) {
            row *= dyShape.GetDim(i);
        } else {
            if (dyShape.GetDim(i) != gammaShape.GetDim(i - dyDimNum + gammaDimNum)) {
                std::string shapeMsg = ToString(dyShape) + " and " + ToString(gammaShape);
                std::string reasonMsg = "The shape of input gamma should be the same as the suffix shape of input dy";
                OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                    context_->GetNodeName(), "dy and gamma", shapeMsg.c_str(), reasonMsg.c_str());
                return ge::GRAPH_FAILED;
            }
            col *= dyShape.GetDim(i);
        }
    }

    // get attr
    auto attrs = context_->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context_, attrs);
    auto outputMask = attrs->GetAttrPointer<gert::ContinuousVector>(0);
    if (outputMask == nullptr) {
        commonParams.pdxIsRequire = true;
        commonParams.pdgammaIsRequire = true;
        commonParams.pdbetaIsRequire = true;
    } else {
        OP_CHECK_IF(
            (outputMask->GetSize() != GRAD_OUT_NUM),
            OP_LOGE_WITH_INVALID_ATTR_SIZE(
                context_->GetNodeName(), "output_mask",
                std::to_string(outputMask->GetSize()).c_str(), std::to_string(GRAD_OUT_NUM).c_str()),
            return ge::GRAPH_FAILED);
        auto outputMaskData = static_cast<const bool*>(outputMask->GetData());
        commonParams.pdxIsRequire = static_cast<bool>(outputMaskData[OUTPUT_IDX_ZERO]);
        commonParams.pdgammaIsRequire = static_cast<bool>(outputMaskData[OUTPUT_IDX_ONE]);
        commonParams.pdbetaIsRequire = static_cast<bool>(outputMaskData[OUTPUT_IDX_TWO]);
    }

    // check input dtype
    OP_CHECK_IF(
        InputDtypeCheck(context_, commonParams.dyDtype, commonParams.xDtype, commonParams.rstdDtype,
        commonParams.meanDtype,commonParams.gammaDtype) == ge::GRAPH_FAILED,
        OP_LOGE(context_->GetNodeName(), "input dtype check failed."), 
        return ge::GRAPH_FAILED);

    // check output dtype
    if (commonParams.pdxIsRequire) {
        auto dxDesc = context_->GetOutputDesc(OUTPUT_IDX_ZERO);
        OP_CHECK_NULL_WITH_CONTEXT(context_, dxDesc);
        commonParams.dxDtype = dxDesc->GetDataType();
        if (commonParams.dxDtype != commonParams.dyDtype) {
            std::string dtypeMsg = ToString(commonParams.dxDtype) + " and " + ToString(commonParams.dyDtype);
            OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(
                context_->GetNodeName(), "pd_x and dy", dtypeMsg.c_str(),
                "The dtypes of output pd_x and input dy should be the same");
            return ge::GRAPH_FAILED;
        }
    }
    if (commonParams.pdgammaIsRequire) {
        auto dgammaDesc = context_->GetOutputDesc(OUTPUT_IDX_ONE);
        OP_CHECK_NULL_WITH_CONTEXT(context_, dgammaDesc);
        commonParams.dgammaDtype = dgammaDesc->GetDataType();
        if ((commonParams.dgammaDtype != commonParams.gammaDtype) && (commonParams.dgammaDtype != ge::DataType::DT_FLOAT)) {
            std::string dtypeMsg = ToString(commonParams.dgammaDtype) + " and " + ToString(commonParams.gammaDtype);
            OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(
                context_->GetNodeName(), "pd_gamma and gamma", dtypeMsg.c_str(),
                "The dtype of output pd_gamma should be FLOAT or the same as the dtype of input gamma");
            return ge::GRAPH_FAILED;
        }
    }
    if (commonParams.pdbetaIsRequire) {
        auto dbetaDesc = context_->GetOutputDesc(OUTPUT_IDX_TWO);
        OP_CHECK_NULL_WITH_CONTEXT(context_, dbetaDesc);
        commonParams.dbetaDtype = dbetaDesc->GetDataType();
        if ((commonParams.dbetaDtype != commonParams.gammaDtype) && (commonParams.dbetaDtype != ge::DataType::DT_FLOAT)) {
            std::string dtypeMsg = ToString(commonParams.dbetaDtype) + " and " + ToString(commonParams.gammaDtype);
            OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(
                context_->GetNodeName(), "pd_beta and gamma", dtypeMsg.c_str(),
                "The dtype of output pd_beta should be FLOAT or the same as the dtype of input gamma");
            return ge::GRAPH_FAILED;
        }
    }
    if (commonParams.pdgammaIsRequire && commonParams.pdbetaIsRequire) {
        if (commonParams.dgammaDtype != commonParams.dbetaDtype) {
            std::string dtypeMsg = ToString(commonParams.dgammaDtype) + " and " + ToString(commonParams.dbetaDtype);
            OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(
                context_->GetNodeName(), "pd_gamma and pd_beta", dtypeMsg.c_str(),
                "The dtypes of output pd_gamma and output pd_beta should be the same");
            return ge::GRAPH_FAILED;
        }
    }

    commonParams.colSize = col;
    commonParams.rowSize = row;
    commonParams.colAlign =
        (commonParams.colSize + B16_BLOCK_ALIGN_NUM - 1) / B16_BLOCK_ALIGN_NUM * B16_BLOCK_ALIGN_NUM;
    commonParams.isDeterministicKey = 1;
    context_->SetScheduleMode(SCHEDULE_MODE);
    if (commonParams.dyDtype == ge::DataType::DT_FLOAT) {
        commonParams.colAlign =
            (commonParams.colSize + B32_BLOCK_ALIGN_NUM - 1) / B32_BLOCK_ALIGN_NUM * B32_BLOCK_ALIGN_NUM;
        commonParams.dtypeKey = LNGDtypeKey::FLOAT_FLOAT;
    } else if (commonParams.dyDtype == ge::DataType::DT_FLOAT16) {
        if (commonParams.gammaDtype == ge::DataType::DT_FLOAT16) {
            commonParams.dtypeKey = LNGDtypeKey::FLOAT16_FLOAT16;
        } else if (commonParams.gammaDtype == ge::DataType::DT_FLOAT) {
            commonParams.dtypeKey = LNGDtypeKey::FLOAT16_FLOAT;
        }
    } else if (commonParams.dyDtype == ge::DataType::DT_BF16) {
        if (commonParams.gammaDtype == ge::DataType::DT_BF16) {
            commonParams.dtypeKey = LNGDtypeKey::BFLOAT16_BFLOAT16;
        } else if (commonParams.gammaDtype == ge::DataType::DT_FLOAT) {
            commonParams.dtypeKey = LNGDtypeKey::BFLOAT16_FLOAT;
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus LayerNormGradV3TilingBase::GetPlatformInfo()
{
    auto compileInfo = context_->GetCompileInfo<LayerNormGradV3CompileInfo>();
    commonParams.coreNum = compileInfo->coreNum;
    commonParams.ubSizePlatForm = compileInfo->ubSizePlatForm;
    commonParams.isRegBase = compileInfo->isRegBase;
    commonParams.blockSize = compileInfo->blockSize;
    commonParams.vlFp32 = compileInfo->vlFp32;
    return ge::GRAPH_SUCCESS;
}

bool LayerNormGradV3TilingBase::IsCapable()
{
    return true;
}

ge::graphStatus LayerNormGradV3TilingBase::DoOpTiling()
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus LayerNormGradV3TilingBase::DoLibApiTiling()
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus LayerNormGradV3TilingBase::GetWorkspaceSize()
{
    size_t* workspaces = context_->GetWorkspaceSizes(1);
    workspaces[0] = BASE_WSP_SIZE;

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus LayerNormGradV3TilingBase::PostTiling()
{
    return ge::GRAPH_SUCCESS;
}

uint64_t LayerNormGradV3TilingBase::GetTilingKey() const
{
    return 0;
}

constexpr static int64_t CONST_ZERO = 0;
constexpr static int64_t CONST_ONE = 1;
constexpr static int64_t CONST_TWO = 2;
constexpr static int64_t CONST_FOUR = 4;
constexpr static int64_t CONST_SIXTY_THREE = 63;

int64_t LayerNormGradV3TilingBase::FindNearestPower2(const int64_t value)
{
    if (value <= CONST_ONE) {
        return CONST_ZERO;
    } else if (value <= CONST_TWO) {
        return CONST_ONE;
    } else if (value <= CONST_FOUR) {
        return CONST_TWO;
    } else {
        const int64_t num = value - CONST_ONE;
        const int64_t pow = CONST_SIXTY_THREE - __builtin_clzl(num);
        return (CONST_ONE << pow);
    }
}

int64_t LayerNormGradV3TilingBase::GetCacheID(const int64_t idx)
{
    return __builtin_popcountll(idx ^ (idx + CONST_ONE)) - CONST_ONE;
}

} // namespace optiling

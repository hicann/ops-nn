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
 * \file layer_norm_grad_tiling_base.cc
 * \brief
 */

#include "layer_norm_grad_tiling.h"

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

bool LayerNormGradTilingBase::CheckShapeSame(
    const gert::TilingContext* context_, const size_t leftIndex, const size_t rightIndex, const bool isLeftInput,
    const bool isRightInput)
{
    const gert::StorageShape* leftShape = nullptr;
    const gert::StorageShape* rightShape = nullptr;
    if (isLeftInput) {
        leftShape = context_->GetInputShape(leftIndex);
        OP_CHECK_NULL_WITH_CONTEXT(context_, leftShape);
    } else {
        leftShape = context_->GetOutputShape(leftIndex);
        OP_CHECK_NULL_WITH_CONTEXT(context_, leftShape);
    }
    if (isRightInput) {
        rightShape = context_->GetInputShape(rightIndex);
        OP_CHECK_NULL_WITH_CONTEXT(context_, rightShape);
    } else {
        rightShape = context_->GetOutputShape(rightIndex);
        OP_CHECK_NULL_WITH_CONTEXT(context_, rightShape);
    }

    // get storage shape
    gert::Shape leftShapeVal = leftShape->GetStorageShape();
    gert::Shape rightShapeVal = rightShape->GetStorageShape();

    // check the leftIndex shape and rightIndex shape are the same
    OP_CHECK_IF(
        (leftShapeVal != rightShapeVal),
        OP_LOGE(
            context_->GetNodeName(),
            "The shape of leftIndex(%zu) is not equal to"
            "the shape of rightIndex(%zu), please check.",
            leftIndex, rightIndex),
        return false);
    return true;
}

ge::graphStatus LayerNormGradTilingBase::InputDtypeCheck(
    gert::TilingContext* context_, ge::DataType dyDtype, ge::DataType xDtype, 
    ge::DataType varianceDtype, ge::DataType meanDtype, ge::DataType gammaDtype)
{
    // input check
    OP_CHECK_IF(
        (dyDtype != ge::DataType::DT_FLOAT) && (dyDtype != ge::DataType::DT_FLOAT16) && (dyDtype != ge::DataType::DT_BF16),
        OP_LOGE(context_->GetNodeName(), "dy dtype must be in float32, float16, bfloat16."),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        xDtype != dyDtype, OP_LOGE(context_->GetNodeName(), "x dtype must be the same as dy."),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        (varianceDtype != ge::DataType::DT_FLOAT) || (meanDtype != ge::DataType::DT_FLOAT), OP_LOGE(context_->GetNodeName(), "variance and mean dtype must be the float32."),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        (gammaDtype != dyDtype) && (gammaDtype != ge::DataType::DT_FLOAT), 
        OP_LOGE(context_->GetNodeName(), "when gamma dtype is not same as dy dtype, gamma dtype must be float32."),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus LayerNormGradTilingBase::GetShapeAttrsInfo()
{
    // check dy and x and pdx shape must be the same
    CheckShapeSame(context_, INPUT_IDX_ZERO, INPUT_IDX_ONE, true, true);
    CheckShapeSame(context_, INPUT_IDX_ZERO, OUTPUT_IDX_ZERO, true, false);
    // check variance and mean shape must be the same
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
    
    auto varianceDesc = context_->GetInputDesc(INPUT_IDX_TWO);
    OP_CHECK_NULL_WITH_CONTEXT(context_, varianceDesc);
    commonParams.varianceDtype = varianceDesc->GetDataType();
    
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
    
    OP_CHECK_IF(
        (dyDimNum < gammaDimNum),
        OP_LOGE(
            context_->GetNodeName(),
            "TilingForLayerNormGrad: dy dim num(=%ld) is less than"
            "gamma dim num(=%ld), please check.",
            static_cast<long>(dyDimNum), static_cast<long>(gammaDimNum)),
        return ge::GRAPH_FAILED);
    // fuse dims
    int64_t row = 1;
    int64_t col = 1;
    for (int64_t i = 0; i < dyDimNum; i++) {
        OP_CHECK_IF(
            (dyShape.GetDim(i) <= 0),
            OP_LOGE(context_->GetNodeName(), "Tiling4LayerNormGrad: dy shape contains zero or negative num."),
            return ge::GRAPH_FAILED);
        if (i < dyDimNum - gammaDimNum) {
            row *= dyShape.GetDim(i);
        } else {
            OP_CHECK_IF(
                (dyShape.GetDim(i) != gammaShape.GetDim(i - dyDimNum + gammaDimNum)),
                OP_LOGE(
                    context_->GetNodeName(),
                    "TilingForLayerNormGrad: dy dim value(=%ld) is not equal to"
                    "gamma dim value(=%ld), please check.",
                    static_cast<long>(dyShape.GetDim(i)),
                    static_cast<long>(gammaShape.GetDim(i - dyDimNum + gammaDimNum))),
                return ge::GRAPH_FAILED);
            col *= dyShape.GetDim(i);
        }
    }
    
    // check input dtype
    OP_CHECK_IF(
        InputDtypeCheck(context_, commonParams.dyDtype, commonParams.xDtype, commonParams.varianceDtype,
        commonParams.meanDtype,commonParams.gammaDtype) == ge::GRAPH_FAILED,
        OP_LOGE(context_->GetNodeName(), "input dtype check failed."), 
        return ge::GRAPH_FAILED);

    // check output dtype
    if (commonParams.pdxIsRequire) {
        auto dxDesc = context_->GetOutputDesc(OUTPUT_IDX_ZERO);
        OP_CHECK_NULL_WITH_CONTEXT(context_, dxDesc);
        commonParams.dxDtype = dxDesc->GetDataType();
        OP_CHECK_IF(
            (commonParams.dxDtype != commonParams.dyDtype), OP_LOGE(context_->GetNodeName(),
            "when dx is Require, dx dtype must be the same as dy."),
            return ge::GRAPH_FAILED);
    }
    if (commonParams.pdgammaIsRequire) {
        auto dgammaDesc = context_->GetOutputDesc(OUTPUT_IDX_ONE);
        OP_CHECK_NULL_WITH_CONTEXT(context_, dgammaDesc);
        commonParams.dgammaDtype = dgammaDesc->GetDataType();
        OP_CHECK_IF(
            (commonParams.dgammaDtype != commonParams.gammaDtype) && (commonParams.dgammaDtype != ge::DataType::DT_FLOAT),OP_LOGE(context_->GetNodeName(), 
            "when dgamma is Require, dgamma dtype must be the same as gamma or be float32."),
            return ge::GRAPH_FAILED);
    }
    if (commonParams.pdbetaIsRequire) {
        auto dbetaDesc = context_->GetOutputDesc(OUTPUT_IDX_TWO);
        OP_CHECK_NULL_WITH_CONTEXT(context_, dbetaDesc);
        commonParams.dbetaDtype = dbetaDesc->GetDataType();
        OP_CHECK_IF(
            (commonParams.dbetaDtype != commonParams.gammaDtype) && (commonParams.dbetaDtype != ge::DataType::DT_FLOAT), OP_LOGE(context_->GetNodeName(),
            "when dbeta is Require, dbeta dtype must be the same as gamma or be float32."),
            return ge::GRAPH_FAILED);
    }
    if (commonParams.pdgammaIsRequire && commonParams.pdbetaIsRequire) {
        OP_CHECK_IF(
            (commonParams.dgammaDtype != commonParams.dbetaDtype), OP_LOGE(context_->GetNodeName(),
            "when dgamma and pdbeta are Require, dbeta dtype must be the same as dgamma."),
            return ge::GRAPH_FAILED);
    }

    commonParams.colSize = col;
    commonParams.rowSize = row;
    commonParams.colAlign =
        (commonParams.colSize + B16_BLOCK_ALIGN_NUM - 1) / B16_BLOCK_ALIGN_NUM * B16_BLOCK_ALIGN_NUM;
    commonParams.isDeterministicKey = 1;
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

    // set epsilon and listmask
    commonParams.epsilon = (commonParams.xDtype == ge::DataType::DT_FLOAT) ? 1e-12 : 1e-5;
    commonParams.pdxIsRequire = true;
    commonParams.pdgammaIsRequire = true;
    commonParams.pdbetaIsRequire = true;

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus LayerNormGradTilingBase::GetPlatformInfo()
{
    auto compileInfo = context_->GetCompileInfo<LayerNormGradCompileInfo>();
    commonParams.coreNum = compileInfo->coreNum;
    commonParams.ubSizePlatForm = compileInfo->ubSizePlatForm;
    commonParams.isRegBase = compileInfo->isRegBase;
    commonParams.blockSize = compileInfo->blockSize;
    commonParams.vlFp32 = compileInfo->vlFp32;
    return ge::GRAPH_SUCCESS;
}

bool LayerNormGradTilingBase::IsCapable()
{
    return true;
}

ge::graphStatus LayerNormGradTilingBase::DoOpTiling()
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus LayerNormGradTilingBase::DoLibApiTiling()
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus LayerNormGradTilingBase::GetWorkspaceSize()
{
    size_t* workspaces = context_->GetWorkspaceSizes(1);
    workspaces[0] = BASE_WSP_SIZE;

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus LayerNormGradTilingBase::PostTiling()
{
    return ge::GRAPH_SUCCESS;
}

uint64_t LayerNormGradTilingBase::GetTilingKey() const
{
    return 0;
}

constexpr static int64_t CONST_ZERO = 0;
constexpr static int64_t CONST_ONE = 1;
constexpr static int64_t CONST_TWO = 2;
constexpr static int64_t CONST_FOUR = 4;
constexpr static int64_t CONST_SIXTY_THREE = 63;

int64_t LayerNormGradTilingBase::FindNearestPower2(const int64_t value)
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

int64_t LayerNormGradTilingBase::GetCacheID(const int64_t idx)
{
    return __builtin_popcountll(idx ^ (idx + CONST_ONE)) - CONST_ONE;
}

} // namespace optiling

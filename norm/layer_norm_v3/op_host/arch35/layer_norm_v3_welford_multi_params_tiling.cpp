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
 * \file layer_norm_v3_welford_multi_params_tiling.cpp
 * \brief
 */

#include "layer_norm_v3_tiling.h"
#include "platform/platform_info.h"
#include "op_host/tiling_templates_registry.h"

namespace optiling {
constexpr static int64_t WELFORD_MP_CONSTANT_TWO = 2;
constexpr static int64_t WELFORD_MP_TILELENGTH_STEP_SIZE = 64;
constexpr static int64_t WELFORD_MP_DOUBLE_BUFFER = 2;
constexpr static int64_t WELFORD_MP_AGGREGATION_COUNT = 256;
constexpr static uint32_t WELFORD_MP_DEFAULT_WORKSPACE = 16 * 1024 * 1024;
constexpr static int64_t WELFORD_MP_B32_SIZE = 4;
constexpr static int64_t WELFORD_MP_B16_SIZE = 2;

bool LayerNormV3WelfordMultiParamsTiling::IsCapable()
{
    if (!commonParams.isRegBase) {
        return false;
    }
    return true;
}

ge::graphStatus LayerNormV3WelfordMultiParamsTiling::DoOpTiling()
{
    td_.set_N(commonParams.rowSize);
    td_.set_M(commonParams.colSize);
    td_.set_rAlign(commonParams.rowAlign);
    td_.set_nullptrBeta(commonParams.betaNullPtr);
    td_.set_nullptrGamma(commonParams.gammaNullPtr);
    td_.set_epsilon(commonParams.eps);
    td_.set_paramsToNormSize(commonParams.paramsToNormSize);

    int64_t blockFactor = (td_.get_M() + commonParams.coreNum - 1) / commonParams.coreNum;
    int64_t mainBlockCount = td_.get_M() / blockFactor;
    int64_t numBlocks = (td_.get_M() + blockFactor - 1) / blockFactor;
    td_.set_mainBlockCount(mainBlockCount);
    td_.set_mainBlockFactor(blockFactor);
    td_.set_numBlocks(numBlocks);
    td_.set_tailBlockFactor(td_.get_M() - mainBlockCount * blockFactor);

    return ge::GRAPH_SUCCESS;
}

bool LayerNormV3WelfordMultiParamsTiling::IsValidTileLength(int64_t tileLength)
{
    int64_t xSize = 0;
    int64_t ySize = 0;
    int64_t rstdSize = 0;
    int64_t meanSize = 0;
    int64_t welfordTempSize = 0;
    int64_t betaSize = 0;
    int64_t gammaSize = 0;
    int64_t apiTempSize = 0;

    if (commonParams.tensorDtype == ge::DT_FLOAT16 || commonParams.tensorDtype == ge::DT_BF16) {
        xSize = WELFORD_MP_DOUBLE_BUFFER * tileLength * WELFORD_MP_B16_SIZE;
        ySize = WELFORD_MP_DOUBLE_BUFFER * tileLength * WELFORD_MP_B16_SIZE;
    } else {
        xSize = WELFORD_MP_DOUBLE_BUFFER * tileLength * WELFORD_MP_B32_SIZE;
        ySize = WELFORD_MP_DOUBLE_BUFFER * tileLength * WELFORD_MP_B32_SIZE;
    }

    meanSize = WELFORD_MP_DOUBLE_BUFFER * WELFORD_MP_AGGREGATION_COUNT * WELFORD_MP_B32_SIZE;
    rstdSize = WELFORD_MP_DOUBLE_BUFFER * WELFORD_MP_AGGREGATION_COUNT * WELFORD_MP_B32_SIZE;
    welfordTempSize += WELFORD_MP_DOUBLE_BUFFER * tileLength * WELFORD_MP_B32_SIZE;
    welfordTempSize += WELFORD_MP_AGGREGATION_COUNT * WELFORD_MP_B32_SIZE;
    td_.set_welfordTempSize(welfordTempSize);

    int64_t GammaBetaTypeSize = WELFORD_MP_B16_SIZE;
    if (commonParams.paramDtype == ge::DT_FLOAT16 || commonParams.paramDtype == ge::DT_BF16) {
        gammaSize = WELFORD_MP_DOUBLE_BUFFER * tileLength * WELFORD_MP_B16_SIZE;
        betaSize = WELFORD_MP_DOUBLE_BUFFER * tileLength * WELFORD_MP_B16_SIZE;
    } else {
        gammaSize = WELFORD_MP_DOUBLE_BUFFER * tileLength * WELFORD_MP_B32_SIZE;
        betaSize = WELFORD_MP_DOUBLE_BUFFER * tileLength * WELFORD_MP_B32_SIZE;
        GammaBetaTypeSize = WELFORD_MP_B32_SIZE;
    }
    if (commonParams.gammaNullPtr) {
        gammaSize = 0;
    }
    if (commonParams.betaNullPtr) {
        betaSize = 0;
    }

    int64_t normalizeApiTempSize = 0;
    int64_t welfordFinalizeApiTempSize = 0;
    int64_t welfordUpdateApiTempSize = 0;
    uint32_t maxValue{0}, minValue{0};
    ge::Shape tensorShape({1, tileLength});
    AscendC::GetWelfordUpdateMaxMinTmpSize(
        tensorShape, WELFORD_MP_B32_SIZE, GammaBetaTypeSize, false, true, maxValue, minValue);
    welfordUpdateApiTempSize = minValue;
    AscendC::GetWelfordFinalizeMaxMinTmpSize(tensorShape, WELFORD_MP_B32_SIZE, false, maxValue, minValue);
    welfordFinalizeApiTempSize = minValue;
    AscendC::GetNormalizeMaxMinTmpSize(
        tensorShape, WELFORD_MP_B32_SIZE, GammaBetaTypeSize, true, true, false, maxValue, minValue);
    normalizeApiTempSize = minValue;
    apiTempSize = normalizeApiTempSize + welfordUpdateApiTempSize + welfordFinalizeApiTempSize;
    td_.set_apiTempBufferSize(apiTempSize);

    int64_t totalSize =
        (xSize + ySize) + (meanSize + rstdSize) + (gammaSize + betaSize) + welfordTempSize + apiTempSize;
    return (totalSize <= static_cast<int64_t>(commonParams.ubSizePlatForm));
}

ge::graphStatus LayerNormV3WelfordMultiParamsTiling::DoLibApiTiling()
{
    int64_t tileLength = 0;
    while (IsValidTileLength(tileLength + WELFORD_MP_CONSTANT_TWO * WELFORD_MP_TILELENGTH_STEP_SIZE)) {
        tileLength += WELFORD_MP_TILELENGTH_STEP_SIZE;
    }
    OP_CHECK_IF(
        (tileLength == 0), OP_LOGE(context_->GetNodeName(), "tileLength must greater than 0"), return ge::GRAPH_FAILED);

    int64_t welfordUpdateTimes = td_.get_N() / tileLength;
    int64_t welfordUpdateTail = td_.get_N() - welfordUpdateTimes * tileLength;
    td_.set_tileLength(tileLength);
    td_.set_welfordUpdateTail(welfordUpdateTail);
    td_.set_welfordUpdateTimes(welfordUpdateTimes);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus LayerNormV3WelfordMultiParamsTiling::PostTiling()
{
    int64_t numBlocks = td_.get_numBlocks();
    context_->SetBlockDim(numBlocks);
    td_.SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(td_.GetDataSize());
    size_t* currentWorkspace = context_->GetWorkspaceSizes(1);
    currentWorkspace[0] = WELFORD_MP_DEFAULT_WORKSPACE;

    return ge::GRAPH_SUCCESS;
}

uint64_t LayerNormV3WelfordMultiParamsTiling::GetTilingKey() const
{
    uint64_t templateKey = static_cast<uint64_t>(LNTemplateKey::WELFORD_MULTI_PARAMS);
    return templateKey * LN_TEMPLATE_KEY_WEIGHT + static_cast<uint64_t>(commonParams.dtypeKey);
}

REGISTER_OPS_TILING_TEMPLATE(LayerNormV3, LayerNormV3WelfordMultiParamsTiling, 7000);
REGISTER_OPS_TILING_TEMPLATE(LayerNorm, LayerNormV3WelfordMultiParamsTiling, 7000);
} // namespace optiling

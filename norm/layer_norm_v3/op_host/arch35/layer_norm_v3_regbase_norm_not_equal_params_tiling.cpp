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
 * \file layer_norm_v3_regbase_norm_not_equal_params_tiling.cpp
 * \brief
 */

#include "layer_norm_v3_tiling.h"
#include "op_host/tiling_templates_registry.h"

namespace optiling {
static constexpr int64_t LNV3_DOUBLE_BUFFER = 2;
static constexpr uint32_t LNV3_MINIMAL_WORKSPACE = 32;
static constexpr int64_t LNV3_NUM_TWO = 2;
static constexpr int64_t LNV3_BLOCK_SIZE = 32;
static constexpr int64_t LNV3_B32_ALIGN_NUM = LNV3_BLOCK_SIZE / sizeof(float);

int64_t LayerNormV3RegBaseNormNotEqualParamsTiling::GetUBCanUseSize()
{
    int64_t binaryAddTmpSize = commonParams.vlFp32 * sizeof(float) * LNV3_NUM_TWO * LNV3_NUM_TWO;
    return commonParams.ubSizePlatForm - binaryAddTmpSize;
}

int64_t LayerNormV3RegBaseNormNotEqualParamsTiling::GetRowWeight(bool isFullB)
{
    int64_t xElemSize = sizeof(float);
    if (commonParams.tensorDtype == ge::DT_FLOAT16 || commonParams.tensorDtype == ge::DT_BF16) {
        xElemSize = xElemSize / LNV3_NUM_TWO;
    }
    int64_t betaElemSize = sizeof(float);
    if (commonParams.paramDtype == ge::DT_FLOAT16 || commonParams.paramDtype == ge::DT_BF16) {
        betaElemSize = betaElemSize / LNV3_NUM_TWO;
    }

    if (!isFullB) {
        return LNV3_NUM_TWO * LNV3_DOUBLE_BUFFER * (betaElemSize + xElemSize) + sizeof(float);
    }
    return LNV3_NUM_TWO * betaElemSize + LNV3_DOUBLE_BUFFER * LNV3_NUM_TWO * xElemSize + sizeof(float);
}

bool LayerNormV3RegBaseNormNotEqualParamsTiling::CanFitInBuffer(int64_t curAxisNum, bool isFullB)
{
    int64_t curAxisNumAlign = (curAxisNum + LNV3_B32_ALIGN_NUM - 1) / LNV3_B32_ALIGN_NUM * LNV3_B32_ALIGN_NUM;
    int64_t ubCanUseSize = GetUBCanUseSize();
    int64_t rowWeight = GetRowWeight(isFullB);

    return curAxisNum * static_cast<int64_t>(commonParams.rowAlign) * rowWeight <=
           ubCanUseSize - (LNV3_NUM_TWO * LNV3_DOUBLE_BUFFER + 1) * curAxisNumAlign * sizeof(float);
}

bool LayerNormV3RegBaseNormNotEqualParamsTiling::IsCapable()
{
    if (!commonParams.isRegBase) {
        return false;
    }

    if (static_cast<int64_t>(commonParams.rowAlign) >
        LNV3_NUM_TWO * commonParams.vlFp32 * commonParams.vlFp32 * LNV3_NUM_TWO) {
        return false;
    }

    if (!CanFitInBuffer(1, true)) {
        return false;
    }

    return true;
}

uint64_t LayerNormV3RegBaseNormNotEqualParamsTiling::GetTilingKey() const
{
    return static_cast<uint64_t>(LayerNormV3TilingKey::LAYER_NORM_REGBASE_NORM_NOT_EQUAL_PARAMS);
}

static inline int64_t CeilDiv(int64_t a, int64_t b) { return b == 0 ? a : (a + b - 1) / b; }

static inline int64_t AlignB32(int64_t val) { return CeilDiv(val, LNV3_B32_ALIGN_NUM) * LNV3_B32_ALIGN_NUM; }

void LayerNormV3RegBaseNormNotEqualParamsTiling::SetBasicTilingParams()
{
    td_.set_nullptrGamma(static_cast<int8_t>(commonParams.gammaNullPtr));
    td_.set_nullptrBeta(static_cast<int8_t>(commonParams.betaNullPtr));
    td_.set_epsilon(commonParams.eps);
    td_.set_a(commonParams.colSize);
    td_.set_r(commonParams.rowSize);
    td_.set_rAlign(commonParams.rowAlign);
    td_.set_gammaBetaRAlign(commonParams.gammaBetaRAlign);
    td_.set_b(commonParams.paramsToNormSize);
    td_.set_r1(commonParams.normToParamsSize);
}

bool LayerNormV3RegBaseNormNotEqualParamsTiling::UpdateTiling()
{
    if (!CanFitInBuffer(1, false)) {
        return false; // 终止当前模板，走非全载模板
    }

    ubFactor = GetUBCanUseSize() / (commonParams.rowAlign * GetRowWeight(false));
    while (!CanFitInBuffer(ubFactor, false)) {
        ubFactor--;
    }

    int64_t a = commonParams.colSize;
    ubFactorAlignB32 = AlignB32(ubFactor);
    formerBlockUbLoops = CeilDiv(blockFactor, ubFactor);
    tailBlockUbLoops = CeilDiv(a - blockFactor * (blockNum_ - 1), ubFactor);

    rAxisCount = ubFactor;
    isFullB = false;
    return true;
}

bool LayerNormV3RegBaseNormNotEqualParamsTiling::ComputeTiling()
{
    int64_t r1 = commonParams.normToParamsSize;
    int64_t b = commonParams.paramsToNormSize;
    int64_t a = commonParams.colSize;

    blockFactor = CeilDiv(a, commonParams.coreNum);
    blockNum_ = CeilDiv(a, blockFactor);
    ubFactorAlignB32 = AlignB32(ubFactor);

    formerBlockUbLoops = CeilDiv(blockFactor, ubFactor);
    tailBlockUbLoops = CeilDiv(a - blockFactor * (blockNum_ - 1), ubFactor);

    if (r1 != 1) {
        // begin_norm_axis < begin_params_axis
        rAxisCount = 1;
    } else if (b != 1 && ubFactor >= b) {
        // begin_norm_axis > begin_params_axis 且 b 能全载
        rAxisCount = b;
        isFullB = true;
    } else if (b != 1 && ubFactor < b) {
        // begin_norm_axis > begin_params_axis 且 b 不能全载，则重新计算 tiling
        return UpdateTiling();
    }
    return true;
}

void LayerNormV3RegBaseNormNotEqualParamsTiling::WriteTiling()
{
    td_.set_blockFactor(blockFactor);
    td_.set_ubFactor(ubFactor);
    td_.set_ubFactorAlignB32(ubFactorAlignB32);
    td_.set_formerBlockUbLoops(formerBlockUbLoops);
    td_.set_tailBlockUbLoops(tailBlockUbLoops);
    td_.set_rAxisCount(rAxisCount);
    td_.set_isFullB(isFullB);

    int64_t powerOfTwoForR = 1;
    int64_t r = commonParams.rowSize;
    while (powerOfTwoForR < r) {
        powerOfTwoForR *= LNV3_NUM_TWO;
    }
    td_.set_powerOfTwoForR(powerOfTwoForR);
}

ge::graphStatus LayerNormV3RegBaseNormNotEqualParamsTiling::DoOpTiling()
{
    SetBasicTilingParams();

    ubFactor = GetUBCanUseSize() / (commonParams.rowAlign * GetRowWeight(true));
    while (!CanFitInBuffer(ubFactor, true)) {
        ubFactor--;
    }

    bool status = ComputeTiling();
    OP_CHECK_IF(!status, OP_LOGI(context_->GetNodeName(), "LayerNormV3NormNotEqualParamsTiling ubFactor == 0"),
                return ge::GRAPH_PARAM_INVALID);

    WriteTiling();
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus LayerNormV3RegBaseNormNotEqualParamsTiling::DoLibApiTiling() { return ge::GRAPH_SUCCESS; }

ge::graphStatus LayerNormV3RegBaseNormNotEqualParamsTiling::PostTiling()
{
    context_->SetBlockDim(blockNum_);
    td_.SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(td_.GetDataSize());
    size_t* currentWorkspace = context_->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, currentWorkspace);
    currentWorkspace[0] = LNV3_MINIMAL_WORKSPACE;

    return ge::GRAPH_SUCCESS;
}

REGISTER_OPS_TILING_TEMPLATE(LayerNormV3, LayerNormV3RegBaseNormNotEqualParamsTiling, 5000);
REGISTER_OPS_TILING_TEMPLATE(LayerNorm, LayerNormV3RegBaseNormNotEqualParamsTiling, 5000);
} // namespace optiling

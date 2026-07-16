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
 * \file layer_norm_v3_welford_multi_reduce_tiling.cpp
 * \brief
 */

#include "layer_norm_v3_tiling.h"
#include "platform/platform_info.h"
#include "op_host/tiling_templates_registry.h"

namespace optiling {
constexpr static int64_t CONSTANT_TWO = 2;
constexpr static int64_t TILELENGTH_STEP_SIZE = 64;
constexpr static int64_t DOUBLE_BUFFER = 2;
constexpr static int64_t AGGREGATION_COUNT = 256;
constexpr static uint32_t DEFAULT_WORKSPACE = 16 * 1024 * 1024;
constexpr static int64_t B32_SIZE = 4;
constexpr static int64_t B16_SIZE = 2;
constexpr static int64_t B32_ALIGN_NUM = 32 / sizeof(float);
constexpr static int64_t BLOCK_BYTES = 32;

static inline int64_t CeilDiv(int64_t a, int64_t b) { return b == 0 ? a : (a + b - 1) / b; }

bool LayerNormV3WelfordMultiReduceTiling::IsCapable()
{
    if (!commonParams.isRegBase) {
        return false;
    }
    if (commonParams.normToParamsSize <= 1 || commonParams.paramsToNormSize != 1) {
        return false;
    }
    return true;
}

uint64_t LayerNormV3WelfordMultiReduceTiling::GetTilingKey() const
{
    uint64_t templateKey = static_cast<uint64_t>(LNTemplateKey::WELFORD_MULTI_REDUCE);
    return templateKey * LN_TEMPLATE_KEY_WEIGHT + static_cast<uint64_t>(commonParams.dtypeKey);
}

bool LayerNormV3WelfordMultiReduceTiling::IsValidTileLength(int64_t tileLength, int64_t gammaFixedR0)
{
    // gammaFixedR0 == 0: cutR0 mode, gamma uses double buffer with tileLength
    // gammaFixedR0 > 0: cutR1 mode, gamma uses TBuf resident with fixed size gammaFixedR0
    int64_t xSize = 0;
    int64_t ySize = 0;

    if (commonParams.tensorDtype == ge::DT_FLOAT16 || commonParams.tensorDtype == ge::DT_BF16) {
        xSize = DOUBLE_BUFFER * tileLength * B16_SIZE;
        ySize = DOUBLE_BUFFER * tileLength * B16_SIZE;
    } else {
        xSize = DOUBLE_BUFFER * tileLength * B32_SIZE;
        ySize = DOUBLE_BUFFER * tileLength * B32_SIZE;
    }

    int64_t meanSize = DOUBLE_BUFFER * AGGREGATION_COUNT * B32_SIZE;
    int64_t rstdSize = DOUBLE_BUFFER * AGGREGATION_COUNT * B32_SIZE;
    int64_t welfordTempSize = DOUBLE_BUFFER * tileLength * B32_SIZE + AGGREGATION_COUNT * B32_SIZE;

    int64_t gammaBetaTypeSize = B16_SIZE;
    int64_t gammaSize = 0;
    int64_t betaSize = 0;
    if (gammaFixedR0 == 0) {
        // cutR0: gamma/beta double buffer with tileLength
        if (commonParams.paramDtype == ge::DT_FLOAT16 || commonParams.paramDtype == ge::DT_BF16) {
            gammaSize = DOUBLE_BUFFER * tileLength * B16_SIZE;
            betaSize = DOUBLE_BUFFER * tileLength * B16_SIZE;
        } else {
            gammaSize = DOUBLE_BUFFER * tileLength * B32_SIZE;
            betaSize = DOUBLE_BUFFER * tileLength * B32_SIZE;
            gammaBetaTypeSize = B32_SIZE;
        }
    } else {
        // cutR1: gamma/beta TBuf resident with fixed r0
        // Packed gamma/beta buffers replace resident buffers when r0 is small
        int64_t vlFp32 = commonParams.vlFp32;
        bool needPacked = (gammaFixedR0 < vlFp32 / CONSTANT_TWO) && (!commonParams.gammaNullPtr) &&
                          (!commonParams.betaNullPtr);
        if (commonParams.paramDtype == ge::DT_FLOAT16 || commonParams.paramDtype == ge::DT_BF16) {
            gammaSize = needPacked ? (vlFp32 * B16_SIZE) : (gammaFixedR0 * B16_SIZE);
            betaSize = gammaSize;
        } else {
            gammaSize = needPacked ? (vlFp32 * B32_SIZE) : (gammaFixedR0 * B32_SIZE);
            betaSize = gammaSize;
            gammaBetaTypeSize = B32_SIZE;
        }
    }
    if (commonParams.gammaNullPtr) {
        gammaSize = 0;
    }
    if (commonParams.betaNullPtr) {
        betaSize = 0;
    }

    uint32_t maxValue{0}, minValue{0};
    ge::Shape tensorShape({1, tileLength});
    AscendC::GetWelfordUpdateMaxMinTmpSize(tensorShape, B32_SIZE, gammaBetaTypeSize, false, true, maxValue, minValue);
    int64_t apiTempSize = static_cast<int64_t>(minValue);
    AscendC::GetWelfordFinalizeMaxMinTmpSize(tensorShape, B32_SIZE, false, maxValue, minValue);
    apiTempSize += static_cast<int64_t>(minValue);
    AscendC::GetNormalizeMaxMinTmpSize(tensorShape, B32_SIZE, gammaBetaTypeSize, true, true, false, maxValue, minValue);
    apiTempSize += static_cast<int64_t>(minValue);

    td_.set_welfordTempSize(welfordTempSize);
    td_.set_apiTempBufferSize(apiTempSize);

    int64_t totalSize = (xSize + ySize) + (meanSize + rstdSize) + (gammaSize + betaSize) + welfordTempSize +
                        apiTempSize;
    return (totalSize <= static_cast<int64_t>(commonParams.ubSizePlatForm));
}

ge::graphStatus LayerNormV3WelfordMultiReduceTiling::DoOpTiling()
{
    int64_t A = static_cast<int64_t>(commonParams.colSize);
    int64_t r1 = static_cast<int64_t>(commonParams.normToParamsSize);
    int64_t r0 = static_cast<int64_t>(commonParams.rowSize) / r1;
    int64_t n = r1 * r0;

    td_.set_r1(r1);
    td_.set_r0(r0);
    td_.set_n(n);
    td_.set_nullptrGamma(commonParams.gammaNullPtr);
    td_.set_nullptrBeta(commonParams.betaNullPtr);
    td_.set_epsilon(commonParams.eps);

    // r0Align: r0 aligned to 32B boundary in elements
    int64_t tensorTypeSize = (commonParams.tensorDtype == ge::DT_FLOAT16 || commonParams.tensorDtype == ge::DT_BF16) ?
                                 B16_SIZE :
                                 B32_SIZE;
    int64_t r0AlignedBytes = CeilDiv(r0 * tensorTypeSize, BLOCK_BYTES) * BLOCK_BYTES;
    int64_t r0Align = r0AlignedBytes / tensorTypeSize;
    td_.set_r0Align(r0Align);

    int64_t blockFactor = CeilDiv(A, static_cast<int64_t>(commonParams.coreNum));
    int64_t numBlocks = CeilDiv(A, blockFactor);
    int64_t mainBlockCount = A / blockFactor;
    int64_t tailBlockFactor = A - mainBlockCount * blockFactor;

    td_.set_numBlocks(numBlocks);
    td_.set_mainBlockCount(mainBlockCount);
    td_.set_mainBlockFactor(blockFactor);
    td_.set_tailBlockFactor(tailBlockFactor);
    blockNum_ = numBlocks;

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus LayerNormV3WelfordMultiReduceTiling::DoLibApiTiling()
{
    int64_t r1 = td_.get_r1();
    int64_t r0 = td_.get_r0();
    int64_t n = td_.get_n();

    // Step 1: compute conservative tileLength assuming cutR0 (gamma double buffer)
    int64_t tileLengthCutR0 = 0;
    while (IsValidTileLength(tileLengthCutR0 + CONSTANT_TWO * TILELENGTH_STEP_SIZE, 0)) {
        tileLengthCutR0 += TILELENGTH_STEP_SIZE;
    }
    OP_CHECK_IF((tileLengthCutR0 == 0), OP_LOGE(context_->GetNodeName(), "tileLengthCutR0 must greater than 0"),
                return ge::GRAPH_FAILED);
    // Step 2: decide cutR1 or cutR0
    int64_t r0Align = td_.get_r0Align();
    int64_t vlFp32 = commonParams.vlFp32;

    if (r0 > tileLengthCutR0) {
        // cutR0: r0 doesn't fit, gamma segmented load with double buffer
        td_.set_tileLength(tileLengthCutR0);
        td_.set_cutR1OrR0(0);
        td_.set_r0Factor(tileLengthCutR0);
        td_.set_loopR0outer(CeilDiv(r0, tileLengthCutR0));
        td_.set_r1Factor(1);
        td_.set_loopR1outer(r1);
        td_.set_r1ComputeFactor(1);
        td_.set_gammaPackedSize(0);
    } else {
        // cutR1: r0 fits, gamma TBuf resident — recompute tileLength with fixed gamma
        int64_t tileLength = 0;
        while (IsValidTileLength(tileLength + CONSTANT_TWO * TILELENGTH_STEP_SIZE, r0)) {
            tileLength += TILELENGTH_STEP_SIZE;
        }
        OP_CHECK_IF((tileLength == 0), OP_LOGE(context_->GetNodeName(), "tileLength must greater than 0"),
                    return ge::GRAPH_FAILED);
        td_.set_tileLength(tileLength);
        OP_CHECK_IF((r0Align == 0), OP_LOGE(context_->GetNodeName(), "r0Align must greater than 0"),
                    return ge::GRAPH_FAILED);
        int64_t r1Factor = tileLength / r0Align;
        if (r1Factor < 1) {
            r1Factor = 1;
        }
        if (r1Factor > r1) {
            r1Factor = r1;
        }
        td_.set_cutR1OrR0(1);
        td_.set_r0Factor(r0);
        td_.set_loopR0outer(1);
        td_.set_r1Factor(r1Factor);
        td_.set_loopR1outer(CeilDiv(r1, r1Factor));

        // Only enable when r0Align < vlFp32/2 (matches kernel-side condition)
        int64_t r1ComputeFactor = 1;
        if (r0Align < vlFp32 / CONSTANT_TWO && !commonParams.gammaNullPtr && !commonParams.betaNullPtr) {
            r1ComputeFactor = vlFp32 / r0Align;
            if (r1ComputeFactor > r1Factor) {
                r1ComputeFactor = r1Factor;
            }
        }
        td_.set_r1ComputeFactor(r1ComputeFactor);

        // gammaPackedSize: UB space for replicated gamma/beta (r1ComputeFactor * r0Align elements)
        int64_t gammaPackedSize = 0;
        if (r1ComputeFactor > 1) {
            int64_t paramTypeSize = (commonParams.paramDtype == ge::DT_FLOAT16 ||
                                     commonParams.paramDtype == ge::DT_BF16) ?
                                        B16_SIZE :
                                        B32_SIZE;
            gammaPackedSize = r1ComputeFactor * r0Align * paramTypeSize;
        }
        td_.set_gammaPackedSize(gammaPackedSize);
    }

    int64_t tileLength = td_.get_tileLength();
    OP_CHECK_IF((tileLength == 0), OP_LOGE(context_->GetNodeName(), "tileLength must greater than 0"),
                return ge::GRAPH_FAILED);
    int64_t welfordUpdateTimes = n / tileLength;
    int64_t welfordUpdateTail = n - welfordUpdateTimes * tileLength;
    td_.set_welfordUpdateTimes(welfordUpdateTimes);
    td_.set_welfordUpdateTail(welfordUpdateTail);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus LayerNormV3WelfordMultiReduceTiling::PostTiling()
{
    context_->SetBlockDim(blockNum_);
    td_.SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(td_.GetDataSize());
    size_t* currentWorkspace = context_->GetWorkspaceSizes(1);
    currentWorkspace[0] = DEFAULT_WORKSPACE;

    return ge::GRAPH_SUCCESS;
}

REGISTER_OPS_TILING_TEMPLATE(LayerNormV3, LayerNormV3WelfordMultiReduceTiling, 6000);
REGISTER_OPS_TILING_TEMPLATE(LayerNorm, LayerNormV3WelfordMultiReduceTiling, 6000);
} // namespace optiling

/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file layer_norm_v4_regbase_two_pass_tiling.cpp
 * \brief
 */

#include "layer_norm_v4_tiling.h"

namespace optiling {
constexpr int64_t BINARY_ADD_COEF = 2;
constexpr int64_t DOUBLE_BUFFER = 2;
constexpr int64_t SMALL_BUFFER_NUM = 2;
constexpr int64_t LARGE_BUFFER_NUM = 2;
constexpr int64_t FP32_BYTE = 4;
constexpr int64_t FP16_BYTE = 2;
constexpr uint32_t MINIMAL_WORKSPACE = 32;

bool LayerNormV4RegBaseTwoPassTiling::CanFitInBuffer(
    int64_t curA, int64_t largeBufferMemPerA, int64_t baseMemSize, int64_t& tmpBufferUse, int64_t xElemSize)
{
    uint32_t minValue;
    uint32_t maxValue;
    int64_t r = commonParams.rowSize;
    ge::Shape inputShape({curA, r});
    AscendC::GetLayerNormMaxMinTmpSize(inputShape, xElemSize, true, true, false, maxValue, minValue);
    tmpBufferUse = maxValue;
    int64_t ubCanUseSize = commonParams.ubSizePlatForm;
    int64_t bufferUse =
        curA * largeBufferMemPerA +
        Ops::Base::CeilAlign(static_cast<int64_t>(curA * FP32_BYTE), static_cast<int64_t>(commonParams.blockSize)) *
            SMALL_BUFFER_NUM * DOUBLE_BUFFER +
        baseMemSize;
    return (bufferUse + tmpBufferUse) <= ubCanUseSize;
}

bool LayerNormV4RegBaseTwoPassTiling::IsCapable()
{
    if (!commonParams.isRegBase) {
        return false;
    }

    int64_t xElemSize = FP32_BYTE;
    if (commonParams.tensorDtype == ge::DT_FLOAT16 || commonParams.tensorDtype == ge::DT_BF16) {
        xElemSize = FP16_BYTE;
    }
    int64_t betaElemSize = FP32_BYTE;
    if (commonParams.paramDtype == ge::DT_FLOAT16 || commonParams.paramDtype == ge::DT_BF16) {
        betaElemSize = FP16_BYTE;
    }

    int64_t a = commonParams.colSize;
    int64_t r = commonParams.rowSize;
    int64_t rAlign = commonParams.rowAlign;
    binaryAddQuotient = commonParams.vlFp32;
    while (binaryAddQuotient < r) {
        binaryAddQuotient *= BINARY_ADD_COEF;
    }
    binaryAddQuotient /= BINARY_ADD_COEF;

    int64_t baseMemSize = (commonParams.gammaNullPtr == 1 ? 0 : rAlign) * betaElemSize +
                          (commonParams.betaNullPtr == 1 ? 0 : rAlign) * betaElemSize;
    int64_t largeBufferMemPerA = rAlign * xElemSize * LARGE_BUFFER_NUM * DOUBLE_BUFFER;
    int64_t lastACanUse = -1;
    int64_t lastTmpBufferUse = 0;
    int64_t tmpBufferUse = 0;
    int64_t begin = 1;
    int64_t end = a;
    while (begin <= end) {
        int64_t mid = (begin + end) / BINARY_ADD_COEF;
        if (CanFitInBuffer(mid, largeBufferMemPerA, baseMemSize, tmpBufferUse, xElemSize)) {
            lastACanUse = mid;
            lastTmpBufferUse = tmpBufferUse;
            begin = mid + 1;
        } else {
            end = mid - 1;
        }
    }
    if (lastACanUse == -1) {
        return false;
    }

    td_.set_aFactor(lastACanUse);
    td_.set_tmpBufferSize(lastTmpBufferUse);
    td_.set_a(a);
    td_.set_r(r);
    td_.set_rAlign(rAlign);
    td_.set_binaryAddQuotient(binaryAddQuotient);
    return true;
}

uint64_t LayerNormV4RegBaseTwoPassTiling::GetTilingKey() const
{
    uint64_t tilingKey = 0;
    if (commonParams.tensorDtype == ge::DT_FLOAT && commonParams.paramDtype == ge::DT_FLOAT) {
        tilingKey = static_cast<uint64_t>(LayerNormV4TilingKey::LAYER_NORM_REGBASE_TWO_PASS_FLOAT32_FLOAT32);
    }
    if (commonParams.tensorDtype == ge::DT_FLOAT16 && commonParams.paramDtype == ge::DT_FLOAT) {
        tilingKey = static_cast<uint64_t>(LayerNormV4TilingKey::LAYER_NORM_REGBASE_TWO_PASS_FLOAT16_FLOAT32);
    }
    if (commonParams.tensorDtype == ge::DT_FLOAT16 && commonParams.paramDtype == ge::DT_FLOAT16) {
        tilingKey = static_cast<uint64_t>(LayerNormV4TilingKey::LAYER_NORM_REGBASE_TWO_PASS_FLOAT16_FLOAT16);
    }
    if (commonParams.tensorDtype == ge::DT_BF16 && commonParams.paramDtype == ge::DT_FLOAT) {
        tilingKey = static_cast<uint64_t>(LayerNormV4TilingKey::LAYER_NORM_REGBASE_TWO_PASS_BFLOAT16_FLOAT32);
    }
    if (commonParams.tensorDtype == ge::DT_BF16 && commonParams.paramDtype == ge::DT_BF16) {
        tilingKey = static_cast<uint64_t>(LayerNormV4TilingKey::LAYER_NORM_REGBASE_TWO_PASS_BFLOAT16_BFLOAT16);
    }
    return tilingKey;
}

ge::graphStatus LayerNormV4RegBaseTwoPassTiling::DoOpTiling()
{
    // optional input
    int64_t isGammaNullptr = commonParams.gammaNullPtr;
    int64_t isBetaNullptr = commonParams.betaNullPtr;
    td_.set_nullptrGamma(isGammaNullptr);
    td_.set_nullptrBeta(isBetaNullptr);

    // dim
    int64_t r = commonParams.rowSize;
    int64_t powerOfTwoForR = 1;
    while (powerOfTwoForR < r) {
        powerOfTwoForR *= BINARY_ADD_COEF;
    }
    td_.set_powerOfTwoForR(powerOfTwoForR);

    // attr
    td_.set_epsilon(commonParams.eps);

    // core num
    int64_t a = commonParams.colSize;
    int64_t blockFactor = (a + commonParams.coreNum - 1) / commonParams.coreNum;
    blockNum = (a + blockFactor - 1) / blockFactor;
    td_.set_aBlockFactor(blockFactor);
    td_.set_blockNum(blockNum);

    // binary add k
    int64_t vcaddNum = binaryAddQuotient / commonParams.vlFp32; // 2的幂次方的数据要做二分
    if (vcaddNum <= commonParams.vlFp32) {
        td_.set_binaryAddK(0);
        td_.set_binaryAddLastNum(vcaddNum);
    } else {
        int64_t binaryAddNum = vcaddNum / commonParams.vlFp32; // VL_FP32为一块，要累加的块，当前肯定是2的幂次方
        int64_t binaryAddK = 0;
        int64_t curBinaryAddNum = 1;
        while (curBinaryAddNum < binaryAddNum) {
            binaryAddK++;
            curBinaryAddNum *= BINARY_ADD_COEF;
        }
        td_.set_binaryAddK(binaryAddK);
        td_.set_binaryAddLastNum(commonParams.vlFp32);
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus LayerNormV4RegBaseTwoPassTiling::DoLibApiTiling()
{
    int64_t xElemSize = FP32_BYTE;
    if (commonParams.tensorDtype == ge::DT_FLOAT16 || commonParams.tensorDtype == ge::DT_BF16) {
        xElemSize = FP16_BYTE;
    }
    int64_t aFactor = td_.get_aFactor();
    int64_t r = commonParams.rowSize;
    uint32_t maxValue;
    uint32_t minValue;

    ge::Shape inputShape({aFactor, r});
    AscendC::GetLayerNormMaxMinTmpSize(inputShape, xElemSize, true, true, false, maxValue, minValue);
    AscendC::GetLayerNormNDTilingInfo(inputShape, maxValue, xElemSize, true, true, td_.layerNormTiling);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus LayerNormV4RegBaseTwoPassTiling::PostTiling()
{
    context_->SetBlockDim(blockNum);
    td_.SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(td_.GetDataSize());
    size_t* currentWorkspace = context_->GetWorkspaceSizes(1);
    currentWorkspace[0] = MINIMAL_WORKSPACE;

    return ge::GRAPH_SUCCESS;
}

REGISTER_OPS_TILING_TEMPLATE(LayerNormV3, LayerNormV4RegBaseTwoPassTiling, 200);
REGISTER_OPS_TILING_TEMPLATE(LayerNormV4, LayerNormV4RegBaseTwoPassTiling, 200);
} // namespace optiling

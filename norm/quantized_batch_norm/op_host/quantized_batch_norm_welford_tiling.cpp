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
 * \file quantized_batch_norm_welford_tiling.cc
 * \brief
 */

#include "quantized_batch_norm_tiling.h"

namespace {
static constexpr uint64_t QBN_WELFORD_R0_SPLIT_NOT_ALIGN_TILING_KEY = 1000;
static constexpr uint64_t QBN_WELFORD_R0_SPLIT_ALIGN_TILING_KEY = 1001;
static constexpr uint64_t QBN_WELFORD_R1_SPLIT_NOT_ALIGN_TILING_KEY = 1002;
static constexpr uint64_t QBN_WELFORD_R1_SPLIT_ALIGN_TILING_KEY = 1003;
static constexpr uint64_t R0_ALIGN_TILING_KEY_BIAS = 10;
static constexpr uint32_t TWO_POWER_ONE = 2;
static constexpr uint32_t TWO_POWER_TWO = 4;
static constexpr uint32_t TWO_POWER_THREE = 8;
static constexpr uint32_t TWO_POWER_FOUR = 16;
static constexpr uint64_t FLOAT_SIZE = 4;
static constexpr uint64_t INT8_ELE = 1;
static constexpr uint64_t INT32_ELE = 4;
static constexpr int64_t A_UB_SIZE_LIMIT = 512;
static constexpr int64_t I8_BLOCK_ALIGN_NUM = 32;
static constexpr int64_t F32_BLOCK_ALIGN_NUM = 8;
static constexpr int64_t I32_BLOCK_ALIGN_NUM = 8;
static constexpr int64_t A_UB_NUM = 5;
static constexpr int64_t INT8_R0_UB_NUM = (1 * 2 + 2 + 2 + 4); // 反量化 + pingpong
static constexpr int64_t INT32_R0_UB_NUM = (1 * 2 + 2);
static constexpr int64_t TWO_NUM = 2;
static constexpr int64_t TMP_BUFFER = 1 * 1024;
static constexpr int64_t INT8_COMPARE_FLOAT = 4;
static constexpr int64_t INT32_COMPARE_FLOAT = 1;
}; // namespace

namespace optiling {
bool QuantizedBatchNormWelfordTiling::IsCapable()
{
    return true;
}

uint64_t QuantizedBatchNormWelfordTiling::GetTilingKey() const
{
    return welfordTilingkey;
}

void QuantizedBatchNormWelfordTiling::DoUbTiling(int64_t& aUbFactor, int64_t& r0UbFactor)
{
    // cast需要32字节对齐
    int64_t elementNum = 0;
    if (commonParams.xDtype == ge::DT_INT32) {
        elementNum = Ops::Base::FloorDiv(commonParams.ubSizePlatForm - TWO_NUM * TMP_BUFFER, INT32_ELE);
        aUbFactor = (td_.get_blockFactor() > A_UB_SIZE_LIMIT) ?
                        A_UB_SIZE_LIMIT :
                        Ops::Base::CeilAlign(td_.get_blockFactor(), F32_BLOCK_ALIGN_NUM);
        r0UbFactor = Ops::Base::FloorAlign(
            Ops::Base::FloorDiv(elementNum - aUbFactor * INT32_COMPARE_FLOAT * A_UB_NUM, INT32_R0_UB_NUM),
            I32_BLOCK_ALIGN_NUM);
    } else {
        elementNum = Ops::Base::FloorDiv(commonParams.ubSizePlatForm - TWO_NUM * TMP_BUFFER, INT8_ELE);
        aUbFactor = (td_.get_blockFactor() > A_UB_SIZE_LIMIT) ?
                        A_UB_SIZE_LIMIT :
                        Ops::Base::CeilAlign(td_.get_blockFactor(), F32_BLOCK_ALIGN_NUM);
        r0UbFactor = Ops::Base::FloorAlign(
            Ops::Base::FloorDiv(elementNum - aUbFactor * INT8_COMPARE_FLOAT * A_UB_NUM, INT8_R0_UB_NUM),
            I8_BLOCK_ALIGN_NUM);
    }
}

ge::graphStatus QuantizedBatchNormWelfordTiling::DoOpTiling()
{
    int64_t blockFactor = Ops::Base::CeilDiv(commonParams.patternA, static_cast<int64_t>(commonParams.coreNum));
    usedCoreNum = Ops::Base::CeilDiv(commonParams.patternA, blockFactor);
    td_.set_blockFactor(blockFactor);
    td_.set_tailCoreBlockFactor(commonParams.patternA - (usedCoreNum - 1) * blockFactor);
    int64_t aUbFactor = 1;
    int64_t r0UbFactor = 1;
    DoUbTiling(aUbFactor, r0UbFactor);
    td_.set_aUbFactor(aUbFactor);
    td_.set_r0UbFactor(r0UbFactor);
    td_.set_aUbLoop(Ops::Base::CeilDiv(blockFactor, aUbFactor));
    td_.set_aUbTail(blockFactor - (td_.get_aUbLoop() - 1) * aUbFactor);
    td_.set_tailCoreAUbLoop(Ops::Base::CeilDiv(td_.get_tailCoreBlockFactor(), aUbFactor));
    td_.set_tailCoreAUbTail(td_.get_tailCoreBlockFactor() - (td_.get_tailCoreAUbLoop() - 1) * aUbFactor);
    td_.set_r0UbLoop(Ops::Base::CeilDiv(commonParams.patternR0, r0UbFactor));
    td_.set_r0UbTail(commonParams.patternR0 - (td_.get_r0UbLoop() - 1) * r0UbFactor);
    td_.set_procNR0(1);
    td_.set_nR0Loop(commonParams.patternR1);
    td_.set_lastLoopNR0(1);
    if ((td_.get_r0UbLoop() == 1) || (td_.get_r0UbFactor() == td_.get_r0UbTail())) {
        welfordTilingkey = QBN_WELFORD_R0_SPLIT_ALIGN_TILING_KEY;
    } else {
        welfordTilingkey = QBN_WELFORD_R0_SPLIT_NOT_ALIGN_TILING_KEY;
    }
    if ((commonParams.patternR0Align <= (r0UbFactor / TWO_NUM)) && commonParams.patternR1 > 1) {
        int64_t procNR0 = Ops::Base::FloorDiv(r0UbFactor, commonParams.patternR0Align);
        int64_t nR0Loop = Ops::Base::CeilDiv(commonParams.patternR1, procNR0);
        int64_t lastLoopNR0 = commonParams.patternR1 - (nR0Loop - 1) * procNR0;
        if (procNR0 > commonParams.patternR1) {
            procNR0 = commonParams.patternR1;
        }
        td_.set_procNR0(procNR0);
        td_.set_nR0Loop(nR0Loop);
        td_.set_lastLoopNR0(lastLoopNR0);
        uint64_t r0AlignTilingKeyBias =
            (commonParams.patternR0 == commonParams.patternR0Align) ? R0_ALIGN_TILING_KEY_BIAS : 0;
        if ((nR0Loop == 1) || (lastLoopNR0 == procNR0)) {
            welfordTilingkey = QBN_WELFORD_R1_SPLIT_ALIGN_TILING_KEY + r0AlignTilingKeyBias;
        } else {
            welfordTilingkey = QBN_WELFORD_R1_SPLIT_NOT_ALIGN_TILING_KEY + r0AlignTilingKeyBias;
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QuantizedBatchNormWelfordTiling::PostTiling()
{
    td_.set_patternR1(commonParams.patternR1);
    td_.set_patternR0(commonParams.patternR0);
    td_.set_patternA(commonParams.patternA);
    td_.set_patternR0Align(commonParams.patternR0Align);
    td_.set_epsilon(commonParams.epsilon);
    context_->SetBlockDim(usedCoreNum);
    auto rawTilingData = context_->GetRawTilingData();
    OP_CHECK_IF(
        td_.GetDataSize() > rawTilingData->GetCapacity(),
        OP_LOGE(
            commonParams.nodeName, "actual tiling data size %zu > context tiling data size %zu", td_.GetDataSize(),
            rawTilingData->GetCapacity()),
        return ge::GRAPH_FAILED);
    td_.SaveToBuffer(rawTilingData->GetData(), rawTilingData->GetCapacity());
    rawTilingData->SetDataSize(td_.GetDataSize());

    return ge::GRAPH_SUCCESS;
}

REGISTER_TILING_TEMPLATE("QuantizedBatchNorm", QuantizedBatchNormWelfordTiling, 2000);
} // namespace optiling
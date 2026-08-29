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
 * \file add_rms_norm_dynamic_mx_quant_tiling_r_full_load_arch35.cpp
 * \brief
 */
#include "add_rms_norm_dynamic_mx_quant_tiling.h"
#include "norm/norm_common/op_host/norm_tiling_check_common.h"

using namespace optiling::add_rms_norm_dynamic_mx_quant;

namespace optiling {
using namespace NormCheck;

uint64_t AddRmsNormDynamicMxQuantRFullLoadTiling::CalUBTotalSize()
{
    uint64_t R_Align = numColAlign_;
    // mxscale buffer per row: CeilAlign(CeilDiv(R, 32), 32) * FP8_SIZE
    uint64_t mxscaleBufPerRow = mxScaleSize_ * xDtypeSize_;

    // binAdd buffer per row
    uint64_t vlfp32 = vecLengthFP32_;
    uint64_t ubfp32 = ubBlockSize_ / FP32_SIZE;
    uint64_t binAddBufPerRow = Ops::Base::CeilAlign(Ops::Base::CeilDiv(binAddQuotient_, vlfp32), ubfp32) * FP32_SIZE;

    // Max_tmp and Half_tmp per row: CeilDiv(R, 32) * B16_SIZE, 32-byte aligned
    uint64_t maxTmpPerRow = Ops::Base::CeilAlign(blockNumInColAxis_ * xDtypeSize_, ubBlockSize_);
    uint64_t halfTmpPerRow = maxTmpPerRow;

    // InQue (per row)
    uint64_t x1Buf = DOUBLE_BUFFER * Ops::Base::CeilAlign(R_Align * xDtypeSize_, ubBlockSize_);
    uint64_t x2Buf = DOUBLE_BUFFER * Ops::Base::CeilAlign(R_Align * xDtypeSize_, ubBlockSize_);
    uint64_t x3Buf = hasX3_ ? DOUBLE_BUFFER * Ops::Base::CeilAlign(R_Align * xDtypeSize_, ubBlockSize_) : 0;

    // OutQue: xOut per row; quantY/rstd are rowFactor-dependent fixed costs (handled in CalFixedCost).
    // The original code used fixed UB_RESERVE constants (1024+1536) to cover the gap between per-row
    // estimates and actual rowFactor-scaled buffer sizes. CalFixedCost computes the exact cost instead,
    // avoiding UB overflow when numCol is not aligned to MX_STEP_PROCESS_NUM (256) and rowFactor=1.
    uint64_t xOutBuf = DOUBLE_BUFFER * Ops::Base::CeilAlign(R_Align * xDtypeSize_, ubBlockSize_);
    uint64_t mxscaleBuf = DOUBLE_BUFFER * Ops::Base::CeilAlign(mxscaleBufPerRow, ubBlockSize_);

    // TmpBuffer (per row)
    uint64_t xTmpBuf = Ops::Base::CeilAlign(R_Align * FP32_SIZE, ubBlockSize_);
    uint64_t binAddBuf = Ops::Base::CeilAlign(binAddBufPerRow, ubBlockSize_);

    uint64_t maxTmpBuf = maxTmpPerRow;
    uint64_t halfTmpBuf = halfTmpPerRow;

    // Per-row total only (excludes rstd/xReduce/quantY which are rowFactor-dependent fixed costs)
    return x1Buf + x2Buf + x3Buf + xOutBuf + mxscaleBuf + xTmpBuf + binAddBuf + maxTmpBuf + halfTmpBuf;
}

uint64_t AddRmsNormDynamicMxQuantRFullLoadTiling::CalFixedCost(uint64_t rowFactor)
{
    // rstd + xReduce: 3 * CeilAlign(rowFactor, VL_F32) * sizeof(float)
    // (2 double-buffered rstd queues + 1 xReduce buffer, each CeilAlign(rowFactor, VL_F32) * 4 bytes)
    uint64_t ceilRf = Ops::Base::CeilAlign(rowFactor, vecLengthFP32_);
    uint64_t rstdXReduceCost = (DOUBLE_BUFFER + 1) * ceilRf * FP32_SIZE;

    // quantY: matches kernel's quantYBufSize formula exactly
    // kernel: CeilAlign(CeilDiv(numColAlign * rowFactor [/2 for FP4], MX_STEP_PROCESS_NUM), 4) * MX_STEP_PROCESS_NUM
    uint64_t mxStepProcessNum = vecLengthFP32_ * FP32_SIZE; // 256 bytes
    uint64_t quantYElements;
    if (Y_SUPPORT_DTYPE_FP8_SET.count(yDtype_) != 0) {
        quantYElements = numColAlign_ * rowFactor;
    } else {
        quantYElements = numColAlign_ * rowFactor / NUM_TWO;
    }
    uint64_t quantYBufSize = Ops::Base::CeilAlign(Ops::Base::CeilDiv(quantYElements, mxStepProcessNum),
                                                  static_cast<uint64_t>(NUM_FOUR)) *
                             mxStepProcessNum;
    uint64_t quantYCost = DOUBLE_BUFFER * quantYBufSize;

    return rstdXReduceCost + quantYCost;
}

ge::graphStatus AddRmsNormDynamicMxQuantRFullLoadTiling::SetTilingParams()
{
    OP_LOGD(context_->GetNodeName(), "Enter SetTilingParams.");

    // Binary add quotient: power of 2 <= numColAlign
    if (numColAlign_ == 0) {
        binAddQuotient_ = 1;
    } else {
        binAddQuotient_ = 1UL << (ULONG_BIT_LEN - 1 - __builtin_clzl(numColAlign_));
        if (binAddQuotient_ == numColAlign_) {
            binAddQuotient_ /= NUM_TWO;
        }
    }

    uint64_t binaryAddElemtMaxLen = vecLengthFP32_ * vecLengthFP32_ * NUM_TWO * NUM_TWO;

    uint64_t gammaBuf = Ops::Base::CeilAlign(numCol_, ubBlockSize_ / gammaDtypeSize_) * gammaDtypeSize_;
    uint64_t betaBuf = Ops::Base::CeilAlign(betaFlag_ * numCol_, ubBlockSize_ / gammaDtypeSize_) * gammaDtypeSize_;
    uint64_t availableUb = maxUbSize_ - gammaBuf - betaBuf;

    uint64_t perRowCost = CalUBTotalSize();

    uint64_t rowFactor = 0;

    if (availableUb > 0 && numColAlign_ <= binaryAddElemtMaxLen) {
        // Two-pass: first estimate rowFactor ignoring fixed costs, then compute actual fixed costs and re-estimate
        uint64_t estRowFactor = availableUb / perRowCost;
        if (estRowFactor > 0) {
            uint64_t fixedCost = CalFixedCost(estRowFactor);
            if (availableUb > fixedCost) {
                rowFactor = (availableUb - fixedCost) / perRowCost;
            }
        }
    }

    if (rowFactor < 1) {
        OP_LOGI(context_->GetNodeName(), "Cannot fit even 1 row in UB for R-full-load. R=%lu.", numCol_);
        return ge::GRAPH_PARAM_INVALID; // R轴不能全载，继续调下个模板
    }

    rowFactor_ = std::min(rowFactor, blockFactor_);

    OP_LOGI(context_->GetNodeName(), "R-full-load: rowFactor=%lu, numCol=%lu, numColAlign=%lu.", rowFactor_, numCol_,
            numColAlign_);
    return ge::GRAPH_SUCCESS;
}

bool AddRmsNormDynamicMxQuantRFullLoadTiling::IsCapable()
{
    if (Y_SUPPORT_DTYPE_SET.count(yDtype_) == 0) {
        return false;
    }
    uint64_t fullLoadRMax = DOUBLE_BUFFER * vecLengthFP32_ * vecLengthFP32_ * NUM_TWO;
    if (numCol_ > fullLoadRMax) {
        OP_LOGD(context_->GetNodeName(),
                "FullLoad IsCapable false: numCol=%ld >= fullLoadRMax=%ld, "
                "binary add rounds increase, recommend SplitR mode.",
                numCol_, fullLoadRMax);
        return false;
    }
    return true;
}

ge::graphStatus AddRmsNormDynamicMxQuantRFullLoadTiling::DoOpTiling()
{
    OP_LOGD(context_->GetNodeName(), "Enter DoOpTiling.");

    dstStrideUbBlocks_ = (numColAlign_ - numCol_) * xDtypeSize_ / ubBlockSize_;
    // Multi-core split on A axis
    mPerCore_ = Ops::Base::CeilDiv(numRow_, totalCoreNum_);
    usedCoreNum_ = Ops::Base::CeilDiv(numRow_, mPerCore_);
    mLastCore_ = numRow_ - (usedCoreNum_ - 1) * mPerCore_;
    blockFactor_ = mPerCore_;

    // R-full-load tiling
    ge::graphStatus res = SetTilingParams();
    OP_CHECK_IF(ge::GRAPH_SUCCESS != res, , return res);

    SetTilingData();
    PrintTilingData();
    return ge::GRAPH_SUCCESS;
}

void AddRmsNormDynamicMxQuantRFullLoadTiling::SetTilingData()
{
    // AddRmsNorm fields
    tilingData.numRow = numRow_;
    tilingData.numCol = numCol_;
    tilingData.numColAlign = numColAlign_;
    tilingData.blockFactor = blockFactor_;
    tilingData.rowFactor = rowFactor_;
    tilingData.binAddQuotient = binAddQuotient_;
    tilingData.epsilon = epsilon_;
    tilingData.avgFactor = avgFactor_;
    // DynamicMxQuant fields
    tilingData.roundMode = roundMode_;
    tilingData.mxBlockSize = mxBlockSize_;
    tilingData.scaleAlg = scaleAlg_;
    tilingData.blockNumInColAxis = blockNumInColAxis_;
    tilingData.dstStrideUbBlocks = dstStrideUbBlocks_;
    tilingData.mxScaleSize = mxScaleSize_;
    // Flags
    tilingData.betaFlag = betaFlag_;
    tilingData.rstdFlag = rstdFlag_;
}

void AddRmsNormDynamicMxQuantRFullLoadTiling::PrintTilingData()
{
    OP_LOGI(context_->GetNodeName(),
            "TilingData numRow: %lu, numCol: %lu, numColAlign: %lu, "
            "blockFactor: %lu, rowFactor: %lu, binAddQuotient: %lu, "
            "epsilon: %f, avgFactor: %f.",
            tilingData.numRow, tilingData.numCol, tilingData.numColAlign, tilingData.blockFactor, tilingData.rowFactor,
            tilingData.binAddQuotient, tilingData.epsilon, tilingData.avgFactor);
    OP_LOGI(context_->GetNodeName(),
            "TilingData roundMode: %ld, mxBlockSize: %ld, scaleAlg: %ld, "
            "blockNumInColAxis: %ld, dstStrideUbBlocks: %ld, mxScaleSize: %ld, betaFlag: %u, rstdFlag: %u.",
            tilingData.roundMode, tilingData.mxBlockSize, tilingData.scaleAlg, tilingData.blockNumInColAxis,
            tilingData.dstStrideUbBlocks, tilingData.mxScaleSize, tilingData.betaFlag, tilingData.rstdFlag);
}

ge::graphStatus AddRmsNormDynamicMxQuantRFullLoadTiling::DoLibApiTiling() { return ge::GRAPH_SUCCESS; }

ge::graphStatus AddRmsNormDynamicMxQuantRFullLoadTiling::PostTiling()
{
    OP_LOGD(context_->GetNodeName(), "Tiling usedCoreNum is %lu.", usedCoreNum_);
    return PostTilingImpl(static_cast<void*>(&tilingData), sizeof(tilingData));
}

uint64_t AddRmsNormDynamicMxQuantRFullLoadTiling::GetTilingKey() const
{
    return GetTilingKeyCommon(ComputeMode::FULL_LOAD);
}

REGISTER_OPS_TILING_TEMPLATE(AddRmsNormDynamicMxQuant, AddRmsNormDynamicMxQuantRFullLoadTiling,
                             ARND_R_FULL_LOAD_PRIORITY);
} // namespace optiling

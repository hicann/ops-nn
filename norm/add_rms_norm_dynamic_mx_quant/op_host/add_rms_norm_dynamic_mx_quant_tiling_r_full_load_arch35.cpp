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

namespace optiling {
using namespace NormCheck;

/**
 * @brief Compute total UB usage for given rowFactor (A rows per UB iteration)
 * Buffer allocation follows the design document strictly:
 *   InQue: x1_Q, x2_Q, gamma_Q, beta_Q
 *   OutQue: x_Q, rstd_Q, mxscale_Q, y_Q
 *   TmpBuffer: x_tmp, binAdd_tmp, Max_tmp, Half_tmp
 */
uint64_t AddRmsNormDynamicMxQuantRFullLoadTiling::CalUBTotalSize()
{
    uint64_t R_Align = numColAlign_;

    // mxscale buffer per row: CeilAlign(CeilDiv(R, 32), 32) * FP8_SIZE
    uint64_t mxscaleBufPerRow = mxScaleSize_ * xDtypeSize_;

    // binAdd buffer per row
    uint64_t vlfp32 = vecLengthFP32_;
    uint64_t ubfp32 = ubBlockSize_ / FP32_SIZE;
    uint64_t binAddBufPerRow = Ops::Base::CeilAlign(
        Ops::Base::CeilDiv(binAddQuotient_, vlfp32), ubfp32) * FP32_SIZE;

    // Max_tmp and Half_tmp per row: CeilDiv(R, 32) * B16_SIZE, 32-byte aligned
    uint64_t maxTmpPerRow = Ops::Base::CeilAlign(blockNumInColAxis_ * xDtypeSize_, ubBlockSize_);
    uint64_t halfTmpPerRow = maxTmpPerRow;

    // InQue
    uint64_t x1Buf = DOUBLE_BUFFER * Ops::Base::CeilAlign(R_Align * xDtypeSize_, ubBlockSize_);
    uint64_t x2Buf = DOUBLE_BUFFER * Ops::Base::CeilAlign(R_Align * xDtypeSize_, ubBlockSize_);

    // OutQue
    uint64_t yBuf = DOUBLE_BUFFER * Ops::Base::CeilAlign(R_Align, ubBlockSize_);
    uint64_t xOutBuf = DOUBLE_BUFFER * Ops::Base::CeilAlign(R_Align * xDtypeSize_, ubBlockSize_);
    uint64_t mxscaleBuf = DOUBLE_BUFFER * Ops::Base::CeilAlign(mxscaleBufPerRow, ubBlockSize_);
    uint64_t rstdBuf = 0;
    if (rstdFlag_) {
        rstdBuf = DOUBLE_BUFFER * FP32_SIZE;
    }

    // TmpBuffer
    uint64_t xTmpBuf = Ops::Base::CeilAlign(R_Align * FP32_SIZE, ubBlockSize_); // dui
    uint64_t binAddBuf = Ops::Base::CeilAlign(binAddBufPerRow, ubBlockSize_);  // dui
    uint64_t xReduceBuff = FP32_SIZE;  // dui

    uint64_t maxTmpBuf = maxTmpPerRow;
    uint64_t halfTmpBuf = halfTmpPerRow;

    uint64_t total = x1Buf + x2Buf +
                     yBuf + xOutBuf + mxscaleBuf + rstdBuf +
                     xTmpBuf + binAddBuf + xReduceBuff + maxTmpBuf + halfTmpBuf;

    return total;
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

    // Pre-reserve UB for rstd alignment padding
    uint64_t binaryAddElemtMaxLen = vecLengthFP32_ * vecLengthFP32_ * NUM_TWO * NUM_TWO;

    uint64_t gammaBuf = Ops::Base::CeilAlign(numCol_, ubBlockSize_ / gammaDtypeSize_) * gammaDtypeSize_;
    uint64_t betaBuf = Ops::Base::CeilAlign(betaFlag_ * numCol_, ubBlockSize_ / gammaDtypeSize_) * gammaDtypeSize_;
    uint64_t availableUb = maxUbSize_ - UB_RESERVE_FOR_RSTD_ALIGN - UB_RESERVE_FOR_OUTPUT_Y_ALIGN - gammaBuf - betaBuf;
    uint64_t rowFactor = 0;

    // Try R-full-load: find max rowFactor A that fits in UB
    if (availableUb > 0 && numColAlign_ <= binaryAddElemtMaxLen) {
        rowFactor = availableUb / CalUBTotalSize();
    }

    if (rowFactor < 1) {
        OP_LOGE(context_->GetNodeName(), "Cannot fit even 1 row in UB for R-full-load. R=%lu.", numCol_);
        return ge::GRAPH_PARAM_INVALID; // R轴不能全载，继续调下个模板
    }

    rowFactor_ = std::min(rowFactor, blockFactor_);

    OP_LOGI(context_->GetNodeName(), "R-full-load: rowFactor=%lu, numCol=%lu, numColAlign=%lu.",
            rowFactor_, numCol_, numColAlign_);
    return ge::GRAPH_SUCCESS;
}

bool AddRmsNormDynamicMxQuantRFullLoadTiling::IsCapable()
{
    if (yDtype_ != ge::DT_FLOAT8_E4M3FN && yDtype_ != ge::DT_FLOAT8_E5M2) {
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
            tilingData.numRow, tilingData.numCol, tilingData.numColAlign,
            tilingData.blockFactor, tilingData.rowFactor, tilingData.binAddQuotient,
            tilingData.epsilon, tilingData.avgFactor);
    OP_LOGI(context_->GetNodeName(),
            "TilingData roundMode: %ld, mxBlockSize: %ld, scaleAlg: %ld, "
            "blockNumInColAxis: %ld, dstStrideUbBlocks: %ld, mxScaleSize: %ld, betaFlag: %u, rstdFlag: %u.",
            tilingData.roundMode, tilingData.mxBlockSize,
            tilingData.scaleAlg, tilingData.blockNumInColAxis,
            tilingData.dstStrideUbBlocks, tilingData.mxScaleSize,
            tilingData.betaFlag, tilingData.rstdFlag);
}

ge::graphStatus AddRmsNormDynamicMxQuantRFullLoadTiling::DoLibApiTiling()
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AddRmsNormDynamicMxQuantRFullLoadTiling::PostTiling()
{
    OP_LOGD(context_->GetNodeName(), "Tiling usedCoreNum is %lu.", usedCoreNum_);
    context_->SetBlockDim(usedCoreNum_);

    auto rawTilingData = context_->GetRawTilingData();
    OP_CHECK_IF(
        sizeof(tilingData) > rawTilingData->GetCapacity(),
        OP_LOGE(
            context_->GetNodeName(), "actual tiling data size %zu > context tiling data size %zu", sizeof(tilingData),
            rawTilingData->GetCapacity()),
        return ge::GRAPH_FAILED);
    auto capSize = rawTilingData->GetCapacity();
    void* ptrData = rawTilingData->GetData();
    OP_CHECK_NULL_WITH_CONTEXT(context_, ptrData);
    void* ptrStruct = static_cast<void*>(&tilingData);
    OP_CHECK_NULL_WITH_CONTEXT(context_, ptrStruct);
    OP_CHECK_IF(
        memcpy_s(ptrData, capSize, ptrStruct, sizeof(tilingData)) != 0,
        OP_LOGE(context_->GetNodeName(), "Set tiling data is failed!"), return ge::GRAPH_FAILED);
    rawTilingData->SetDataSize(sizeof(tilingData));

    size_t* currentWorkspace = context_->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, currentWorkspace);
    currentWorkspace[0] = workspaceSize_;
    return ge::GRAPH_SUCCESS;
}

uint64_t AddRmsNormDynamicMxQuantRFullLoadTiling::GetTilingKey() const
{
    // Tiling key
    OP_LOGD(context_->GetNodeName(), "TilingKey is %lu.", TILING_KEY_FP8_R_FULL_LOAD);
    return TILING_KEY_FP8_R_FULL_LOAD;
}

REGISTER_OPS_TILING_TEMPLATE(AddRmsNormDynamicMxQuant, AddRmsNormDynamicMxQuantRFullLoadTiling, ARND_R_FULL_LOAD_PRIORITY);
} // namespace optiling
/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!\file add_rms_norm_hi_f4_quant_tiling_r_full_load.cpp
 * \brief
 */
#include "../add_rms_norm_hi_f4_quant_tiling.h"
#include "norm/norm_common/op_host/norm_tiling_check_common.h"

namespace optiling {
using namespace NormCheck;

constexpr uint64_t TILING_KEY_HIF4_R_FULL_LOAD = 101;

uint64_t AddRmsNormHiF4QuantRFullLoadTiling::CalUBTotalSize() { return CalUBTotalSizeWithRowFactor(1); }

uint64_t AddRmsNormHiF4QuantRFullLoadTiling::CalUBTotalSizeWithRowFactor(uint64_t rowFactor)
{
    uint64_t R_Align = numColAlign_;

    uint64_t vlfp32 = vecLengthFP32_;
    uint64_t ubfp32 = ubBlockSize_ / FP32_SIZE;
    uint64_t binAddBufPerRow = Ops::Base::CeilAlign(Ops::Base::CeilDiv(binAddQuotient_, vlfp32), ubfp32) * FP32_SIZE;

    uint64_t x1Buf = DOUBLE_BUFFER * Ops::Base::CeilAlign(R_Align * xDtypeSize_, ubBlockSize_) * rowFactor;
    uint64_t x2Buf = DOUBLE_BUFFER * Ops::Base::CeilAlign(R_Align * xDtypeSize_, ubBlockSize_) * rowFactor;

    uint64_t xOutBuf = DOUBLE_BUFFER * Ops::Base::CeilAlign(R_Align * xDtypeSize_, ubBlockSize_) * rowFactor;
    uint64_t rstdBuf = DOUBLE_BUFFER * Ops::Base::CeilAlign(rowFactor, vlfp32) * FP32_SIZE;

    uint64_t xTmpBuf = Ops::Base::CeilAlign(R_Align * FP32_SIZE, ubBlockSize_) * rowFactor;
    uint64_t binAddBuf = Ops::Base::CeilAlign(binAddBufPerRow, ubBlockSize_) * rowFactor;
    uint64_t xReduceBuff = Ops::Base::CeilAlign(rowFactor, vlfp32) * FP32_SIZE;

    uint64_t normTotal = x1Buf + x2Buf + xOutBuf + rstdBuf + xTmpBuf + binAddBuf + xReduceBuff;

    int64_t totalGroups = static_cast<int64_t>(rowFactor * blockNumInColAxis_);
    int64_t alignTotalGroups = Ops::Base::CeilAlign(totalGroups, DQ_HIFP4_BLOCK_BYTE);

    int64_t bufE8Size = Ops::Base::CeilAlign(totalGroups, DQ_HIFP4_BLOCK_BYTE) * sizeof(int16_t);
    int64_t buf1E8Size = Ops::Base::CeilAlign(totalGroups, DQ_HIFP4_BLOCK_BYTE) * sizeof(int32_t);
    int64_t buf1E16Size = Ops::Base::CeilAlign(totalGroups, DQ_HIFP4_BLOCK_BYTE) * sizeof(int16_t);
    int64_t maxPer4Size = Ops::Base::CeilAlign(totalGroups, DQ_HIFP4_MAXPER4_ALIGN) * 16 * sizeof(int16_t);
    int64_t maxPer8Size = Ops::Base::CeilAlign(totalGroups, DQ_HIFP4_MAXPER8_ALIGN) * 8 * sizeof(int16_t);
    int64_t tmpScaleSize = Ops::Base::CeilAlign(totalGroups, DQ_HIFP4_SCALETMP_ALIGN) * 32 * sizeof(int16_t);

    int64_t quantBufTotal = bufE8Size + buf1E8Size + buf1E16Size + maxPer4Size + maxPer8Size + tmpScaleSize;

    uint64_t quantYBufSize = Ops::Base::CeilAlign(Ops::Base::CeilDiv(R_Align * rowFactor / NUM_TWO,
                                                                     static_cast<uint64_t>(MX_STEP_PROCESS_NUM)),
                                                  static_cast<uint64_t>(HIFP4_DIGIT_FOUR)) *
                             static_cast<uint64_t>(MX_STEP_PROCESS_NUM);
    int64_t alignTo256UbFormer = Ops::Base::CeilAlign(alignTotalGroups, DQ_HIFP4_SCALE_ALIGN);
    uint64_t quantOutBuf = static_cast<uint64_t>(DOUBLE_BUFFER) * quantYBufSize +
                           static_cast<uint64_t>(DOUBLE_BUFFER) * alignTo256UbFormer * sizeof(float);

    return normTotal + static_cast<uint64_t>(quantBufTotal) + quantOutBuf;
}

bool AddRmsNormHiF4QuantRFullLoadTiling::IsCapable()
{
    if (numCol_ > FULL_LOAD_R_MAX) {
        OP_LOGI(context_->GetNodeName(), "R-full-load not capable: numCol %lu > FULL_LOAD_R_MAX %lu.", numCol_,
                FULL_LOAD_R_MAX);
        return false;
    }

    if (numColAlign_ == 0) {
        return false;
    }
    uint64_t binAddQuot = 1UL << (ULONG_BIT_LEN - 1 - __builtin_clzl(numColAlign_));
    if (binAddQuot == numColAlign_) {
        binAddQuot /= NUM_TWO;
    }
    binAddQuotient_ = binAddQuot;

    uint64_t gammaBuf = Ops::Base::CeilAlign(numCol_, ubBlockSize_ / gammaDtypeSize_) * gammaDtypeSize_;
    uint64_t betaBuf = Ops::Base::CeilAlign(betaFlag_ * numCol_, ubBlockSize_ / gammaDtypeSize_) * gammaDtypeSize_;
    uint64_t availableUb = maxUbSize_ - UB_RESERVE_FOR_RSTD_ALIGN - gammaBuf - betaBuf;

    uint64_t totalPerRow = CalUBTotalSizeWithRowFactor(1);
    if (availableUb < totalPerRow) {
        OP_LOGI(context_->GetNodeName(),
                "R-full-load not capable: availableUb %lu < totalPerRow %lu (1 row cannot fit).", availableUb,
                totalPerRow);
        return false;
    }

    uint64_t rowFactor = availableUb / totalPerRow;
    if (rowFactor < 1) {
        OP_LOGI(context_->GetNodeName(), "R-full-load not capable: rowFactor %lu < 1.", rowFactor);
        return false;
    }

    return true;
}

ge::graphStatus AddRmsNormHiF4QuantRFullLoadTiling::DoOpTiling()
{
    OP_LOGD(context_->GetNodeName(), "Enter DoOpTiling.");

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
    uint64_t availableUb = maxUbSize_ - UB_RESERVE_FOR_RSTD_ALIGN - gammaBuf - betaBuf;

    mPerCore_ = Ops::Base::CeilDiv(numRow_, totalCoreNum_);
    usedCoreNum_ = Ops::Base::CeilDiv(numRow_, mPerCore_);
    mLastCore_ = numRow_ - (usedCoreNum_ - 1) * mPerCore_;
    blockFactor_ = mPerCore_;

    uint64_t rowFactor = 0;

    if (availableUb > 0 && numColAlign_ <= binaryAddElemtMaxLen) {
        uint64_t upperBound = availableUb / CalUBTotalSizeWithRowFactor(1);
        if (upperBound > blockFactor_) {
            upperBound = blockFactor_;
        }
        if (upperBound < 1) {
            upperBound = 1;
        }
        for (uint64_t rf = upperBound; rf >= 1; rf--) {
            if (CalUBTotalSizeWithRowFactor(rf) <= availableUb) {
                rowFactor = rf;
                break;
            }
        }
    }

    if (rowFactor < 1) {
        rowFactor = 1;
    }

    rowFactor_ = std::min(rowFactor, blockFactor_);

    OP_LOGI(context_->GetNodeName(), "R-full-load: rowFactor=%lu, numCol=%lu, numColAlign=%lu.", rowFactor_, numCol_,
            numColAlign_);

    SetTilingData();
    PrintTilingData();
    return ge::GRAPH_SUCCESS;
}

void AddRmsNormHiF4QuantRFullLoadTiling::SetTilingData()
{
    tilingData.numRow = numRow_;
    tilingData.numCol = numCol_;
    tilingData.numColAlign = numColAlign_;
    tilingData.blockFactor = blockFactor_;
    tilingData.rowFactor = rowFactor_;
    tilingData.binAddQuotient = binAddQuotient_;
    tilingData.epsilon = epsilon_;
    tilingData.avgFactor = avgFactor_;
    tilingData.mxBlockSize = mxBlockSize_;
    tilingData.blockNumInColAxis = blockNumInColAxis_;
    tilingData.mxScaleSize = mxScaleSize_;
    tilingData.betaFlag = betaFlag_;
    tilingData.rstdFlag = rstdFlag_;
    tilingData.scaleAlg = scaleAlg_;

    int64_t totalGroupsPerUbRow = rowFactor_ * blockNumInColAxis_;
    int64_t ubFormer = totalGroupsPerUbRow;

    int64_t vlByteSize = static_cast<int64_t>(Ops::Base::GetVRegSize(context_));
    int64_t ubReserveAlignBuf = HIFP4_MID_UB_RES_NUM * HIFP4_UB_BLOCK_SIZE * DOUBLE_BUFFER +
                                DOUBLE_BUFFER * vlByteSize * HIFP4_DIGIT_FOUR + HIFP4_RESERVED_UB_SIZE * 4;
    int64_t ubCap = (static_cast<int64_t>(maxUbSize_) - ubReserveAlignBuf) / DOUBLE_BUFFER /
                    (HIFP4_ONE_GROUP_MID_UB_RES_BYTES + HIFP4_ONE_GROUP_INPUT_BYTES + HIFP4_ONE_GROUP_OUTPUT_BYTES +
                     HIFP4_ONE_GROUP_OUTPUT_SCALE_BYTES);
    ubFormer = ubCap / HIFP4_OUTPUT_REG_ONELOOP_GROUP_NUM * HIFP4_OUTPUT_REG_ONELOOP_GROUP_NUM;
    if (ubFormer <= 0)
        ubFormer = HIFP4_OUTPUT_REG_ONELOOP_GROUP_NUM;
    ubFormer = std::min(ubFormer, totalGroupsPerUbRow);

    tilingData.innerLoopEle = Ops::Base::CeilDiv(ubFormer, HIFP4_OUTPUT_REG_ONELOOP_GROUP_NUM);
    tilingData.max8LoopEle = Ops::Base::CeilDiv(ubFormer, HIFP4_MAXPER8_REG_ONELOOP);
    tilingData.scaleLoopEle = Ops::Base::CeilDiv(ubFormer, HIFP4_OUTPUT_SCALE_REG_ONELOOP_GROUP_NUM);
    tilingData.outLoopEle = Ops::Base::CeilDiv(ubFormer, HIFP4_OUTPUT_REG_ONELOOP);
}

void AddRmsNormHiF4QuantRFullLoadTiling::PrintTilingData()
{
    OP_LOGI(context_->GetNodeName(),
            "TilingData numRow: %lu, numCol: %lu, numColAlign: %lu, "
            "blockFactor: %lu, rowFactor: %lu, binAddQuotient: %lu, "
            "epsilon: %f, avgFactor: %f.",
            tilingData.numRow, tilingData.numCol, tilingData.numColAlign, tilingData.blockFactor, tilingData.rowFactor,
            tilingData.binAddQuotient, tilingData.epsilon, tilingData.avgFactor);
    OP_LOGI(context_->GetNodeName(),
            "TilingData mxBlockSize: %ld, blockNumInColAxis: %ld, "
            "mxScaleSize: %ld, betaFlag: %u, rstdFlag: %u, "
            "innerLoopEle: %ld, max8LoopEle: %ld, "
            "scaleLoopEle: %ld, outLoopEle: %ld.",
            tilingData.mxBlockSize, tilingData.blockNumInColAxis, tilingData.mxScaleSize, tilingData.betaFlag,
            tilingData.rstdFlag, tilingData.innerLoopEle, tilingData.max8LoopEle, tilingData.scaleLoopEle,
            tilingData.outLoopEle);
}

ge::graphStatus AddRmsNormHiF4QuantRFullLoadTiling::DoLibApiTiling() { return ge::GRAPH_SUCCESS; }

ge::graphStatus AddRmsNormHiF4QuantRFullLoadTiling::PostTiling()
{
    OP_LOGD(context_->GetNodeName(), "Tiling usedCoreNum is %lu.", usedCoreNum_);
    context_->SetBlockDim(usedCoreNum_);

    auto rawTilingData = context_->GetRawTilingData();
    OP_CHECK_IF(sizeof(tilingData) > rawTilingData->GetCapacity(),
                OP_LOGE(context_->GetNodeName(), "actual tiling data size %zu > context tiling data size %zu",
                        sizeof(tilingData), rawTilingData->GetCapacity()),
                return ge::GRAPH_FAILED);
    auto capSize = rawTilingData->GetCapacity();
    void* ptrData = rawTilingData->GetData();
    OP_CHECK_NULL_WITH_CONTEXT(context_, ptrData);
    void* ptrStruct = static_cast<void*>(&tilingData);
    OP_CHECK_NULL_WITH_CONTEXT(context_, ptrStruct);
    OP_CHECK_IF(memcpy_s(ptrData, capSize, ptrStruct, sizeof(tilingData)) != 0,
                OP_LOGE(context_->GetNodeName(), "Set tiling data is failed!"), return ge::GRAPH_FAILED);
    rawTilingData->SetDataSize(sizeof(tilingData));

    size_t* currentWorkspace = context_->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, currentWorkspace);
    currentWorkspace[0] = workspaceSize_;
    return ge::GRAPH_SUCCESS;
}

uint64_t AddRmsNormHiF4QuantRFullLoadTiling::GetTilingKey() const
{
    uint64_t tilingKey = TILING_KEY_HIF4_R_FULL_LOAD;
    OP_LOGD(context_->GetNodeName(), "TilingKey is %lu.", TILING_KEY_HIF4_R_FULL_LOAD);
    return tilingKey;
}

REGISTER_OPS_TILING_TEMPLATE(AddRmsNormHiF4Quant, AddRmsNormHiF4QuantRFullLoadTiling, ARND_R_FULL_LOAD_PRIORITY);
} // namespace optiling

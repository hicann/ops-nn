/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file rms_norm_tiling_arch35.cpp
 * \brief RmsNorm Op Tiling
 */
#include <iostream>
#include "rms_norm_tiling.h"

namespace optiling {
ge::graphStatus TilingArch354RmsNorm(
    gert::TilingContext* context, uint64_t numRow, uint64_t numCol, uint32_t numCore, uint64_t ubSize,
    ge::DataType xDataType, ge::DataType gammaDataType, float epsilon, RMSNormTilingData& rmsNormTilingData, RMSNormArch35TilingData& rmsNormArch35TilingData)
{
    uint64_t blockFactor = 1ULL;
    uint64_t tileNum = CeilDiv(numRow, static_cast<uint64_t>(numCore));
    blockFactor *= tileNum;
    uint32_t useCoreNum = CeilDiv(numRow, blockFactor);
    context->SetBlockDim(useCoreNum);

    auto dtypeByteIterator_x = dTypeByteMap.find(xDataType);
    OP_TILING_CHECK(
        dtypeByteIterator_x == dTypeByteMap.end(), OPS_REPORT_VECTOR_INNER_ERR(context, "Fail to get dtype factor."),
        return ge::GRAPH_FAILED);
    uint32_t curElementByteX = dtypeByteIterator_x->second;
    auto dtypeByteIterator_gamma = dTypeByteMap.find(gammaDataType);
    OP_TILING_CHECK(
        dtypeByteIterator_gamma == dTypeByteMap.end(),
        OPS_REPORT_VECTOR_INNER_ERR(context, "Fail to get dtype factor."), return ge::GRAPH_FAILED);
    uint32_t curElementByteGamma = dtypeByteIterator_gamma->second;

    auto curElementByte = std::max(curElementByteX, curElementByteGamma);
    OP_TILING_CHECK(
        curElementByte == 0, OPS_REPORT_VECTOR_INNER_ERR(context, "curElementByte is zero."), return ge::GRAPH_FAILED);
    uint64_t blockSize = Ops::Base::GetUbBlockSize(context);
    uint64_t vectorLength = Ops::Base::GetVRegSize(context) / FLOAT_BYTE_SIZE;
    uint64_t numColAlignFullLoad =
        CeilDiv(numCol * curElementByteX, static_cast<uint64_t>(blockSize)) * blockSize / curElementByteX;

    uint64_t powerFactor = std::floor(std::log(numColAlignFullLoad - 1) / std::log(2));
    powerFactor = std::pow(LOG_2, powerFactor); // 公式保持一致add_rms_norm
    uint64_t firstVcaddLength = CeilDiv(CeilDiv(powerFactor, vectorLength), blockSize) * blockSize;
    int64_t ubFactor;

    uint32_t tilingKey;
    // RETAINED_SIZE_1K : outQue_rstd and x_reduce_tmp align to blockSize
    ubFactor =
        (ubSize - RETAINED_SIZE_1K - numColAlignFullLoad * curElementByteGamma) /
        (numColAlignFullLoad * curElementByteX * MULTI_FACTOR_2 * DOUBLE_BUFFER_NUM +
         FLOAT_BYTE_SIZE * (DOUBLE_BUFFER_NUM + X_REDUCE_TMP_NUM) + firstVcaddLength);
    if (ubFactor >= 1 && numColAlignFullLoad <= FULL_LOAD_R_MAX) {
        float avgFactor = (numCol == 0ULL) ? 0.0f : 1.0f / static_cast<float>(numCol);
        uint64_t colFlodFactor = powerFactor; // 二分累加折叠点
        uint64_t lastBlockFactor = numRow - (useCoreNum - 1) * blockFactor;
        ubFactor = std::min(ubFactor, static_cast<int64_t>(blockFactor));
        tilingKey = RMSNORM_REGBASE_NORMAL;
        rmsNormArch35TilingData.set_num_row(numRow);
        rmsNormArch35TilingData.set_num_col(numCol);
        rmsNormArch35TilingData.set_num_col_align(numColAlignFullLoad);
        rmsNormArch35TilingData.set_block_factor(blockFactor);
        rmsNormArch35TilingData.set_col_flod_factor(colFlodFactor);
        rmsNormArch35TilingData.set_ub_factor(ubFactor);
        rmsNormArch35TilingData.set_epsilon(epsilon);
        rmsNormArch35TilingData.set_avg_factor(avgFactor);
        rmsNormArch35TilingData.set_last_block_factor(lastBlockFactor);
        OP_LOGD(
            context,
            "TilingData[%u] numCore: %u, ubSize: %lu, numRow: %lu, numCol: %lu, numColAlignFullLoad: %lu "
            "blockFactor: %lu, colFlodFactor: %lu, ubFactor: %u, epsilon: %f, avgFactor: %f, lastBlockFactor: %ld. ",
            tilingKey, useCoreNum, ubSize, numRow, numCol, numColAlignFullLoad, blockFactor, colFlodFactor, ubFactor, epsilon,
            avgFactor, lastBlockFactor);
        context->SetTilingKey(tilingKey);
        rmsNormArch35TilingData.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
        context->GetRawTilingData()->SetDataSize(rmsNormArch35TilingData.GetDataSize());
    } else {
        uint64_t rowFactor = FLOAT_PER_REAPEAT;
        uint64_t numColAlign = CeilDiv(numCol * curElementByte, static_cast<uint64_t>(ALING_FACTOR_512)) *
                               ALING_FACTOR_512 / curElementByte;
        ComputeTotalBufSizeParam computeTotalBufSizeParam{
            .bufferNum = DOUBLE_BUFFER_NUM,
            .dtype = xDataType,
            .dtypeSizeX = curElementByteX,
            .dtypeSizeGamma = curElementByteGamma,
            .length = static_cast<uint32_t>(numColAlign),
            .split = false};
        uint32_t total = ComputeTotalBufSize(computeTotalBufSizeParam);
        uint32_t multiNNum = static_cast<uint32_t>(ubSize) / static_cast<uint32_t>(total);
        uint64_t colBufferLength{0};
        uint64_t ubLoop{0};
        uint32_t isNddma = 1;
        ubFactor = 1UL;
        computeTotalBufSizeParam.length = static_cast<uint32_t>(ubFactor) * MULTI_FACTOR_2;
        computeTotalBufSizeParam.split = true;
        while (ComputeTotalBufSize(computeTotalBufSizeParam) < ubSize) {
            ubFactor *= MULTI_FACTOR_2;
            computeTotalBufSizeParam.length = static_cast<uint32_t>(ubFactor) * MULTI_FACTOR_2;
        }
        ubLoop = 1UL;
        while (ubLoop * MULTI_FACTOR_2 * ubFactor <= numCol) {
            ubLoop *= MULTI_FACTOR_2;
        }
        colBufferLength = ubFactor;
        tilingKey = RMSNORM_REGBASE_SPLIT;
        if (numCol >= NDDMA_BETTER_STAGE) {
            isNddma = 0U;
        }
        rmsNormTilingData.set_num_row(numRow);
        rmsNormTilingData.set_num_col(numCol);
        rmsNormTilingData.set_num_col_align(numColAlign);
        rmsNormTilingData.set_block_factor(blockFactor);
        rmsNormTilingData.set_last_block_factor(0);
        rmsNormTilingData.set_row_factor(rowFactor);
        rmsNormTilingData.set_ub_factor(ubFactor);
        rmsNormTilingData.set_epsilon(epsilon);
        rmsNormTilingData.set_ub_loop(ubLoop);
        rmsNormTilingData.set_col_buffer_length(colBufferLength);
        rmsNormTilingData.set_multi_n_num(multiNNum);
        rmsNormTilingData.set_is_nddma(isNddma);

        OP_LOGD(
            context,
            "TilingData[%u] numCore: %u, ubSize: %lu, numRow: %lu, numCol: %lu, numColAlign: %lu, col_buffer_length: %u, "
            "blockFactor: %lu, rowFactor: %u, ubFactor: %u, epsilon: %f, ubLoop: %u, multi_n_num: %u, isNddma: %u.",
            tilingKey, numCore, ubSize, rmsNormTilingData.get_num_row(), rmsNormTilingData.get_num_col(), rmsNormTilingData.get_num_col_align(),
            rmsNormTilingData.get_col_buffer_length(), rmsNormTilingData.get_block_factor(), rmsNormTilingData.get_row_factor(), rmsNormTilingData.get_ub_factor(),
            rmsNormTilingData.get_epsilon(), rmsNormTilingData.get_ub_loop(), rmsNormTilingData.get_multi_n_num(), rmsNormTilingData.get_is_nddma());

        context->SetTilingKey(tilingKey);
        rmsNormTilingData.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
        context->GetRawTilingData()->SetDataSize(rmsNormTilingData.GetDataSize());
    }
    return ge::GRAPH_SUCCESS;
}
} // namespace optiling
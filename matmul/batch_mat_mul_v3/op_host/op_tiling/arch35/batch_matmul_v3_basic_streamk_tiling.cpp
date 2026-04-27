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
 * \file batch_matmul_v3_basic_streamk_tiling.cpp
 * \brief
 */
#include "batch_matmul_v3_basic_streamk_tiling.h"
#include "batch_matmul_v3_tiling_strategy.h"
#include "matmul/mat_mul_v3/op_host/op_tiling/arch35/matmul_tiling_registry.h"
#include "matmul/common/op_host/math_util.h"
#include "batch_matmul_v3_tiling_key.h"

using Ops::NN::MathUtil;
namespace optiling {
namespace batch_matmul_v3_advanced {
using namespace strategy;
MM_REGISTER_TILING_TEMPLATE(BatchMatMulV3, BatchMatMulV3BasicStreamKTiling, DAV_3510, BATCH_STREAM_K);

MatMulV3L0C2Out BatchMatMulV3BasicStreamKTiling::GetL0C2OutFlag() const
{
    if (args_.nValue > BASIC_BLOCK_SIZE_64 && args_.nValue % BASIC_BLOCK_SIZE_16 != 0 && args_.mValue > NUM_TWO &&
        args_.mValue * args_.nValue >= BASIC_BLOCK_SIZE_256) {
        return MatMulV3L0C2Out::ND_FIXPIPE_1_2;
    }
    return MatMulV3L0C2Out::ON_THE_FLY;
}

bool BatchMatMulV3BasicStreamKTiling::CheckStreamKSKTiling() const
{
    constexpr uint64_t STREAM_K_MAX_K_THRESHOLD = 2000000UL;
    // 如果dtype是fp32且k轴大于200万 则走基础模板来保证fp32的精度
    if (args_.aDtypeSize == DATA_SIZE_FP32 && !args_.isHf32 &&
        static_cast<uint64_t>(args_.kValue) > STREAM_K_MAX_K_THRESHOLD) {
        OP_LOGD(args_.opName, "Due to the requirement of binary accumulation, current fp32 does not support StreamK");
        return false;
    }
    constexpr uint64_t STREAM_K_MIN_K_THRESHOLD = 8192UL;
    if (ops::CeilAlign(static_cast<uint64_t>(args_.kValue), BASIC_BLOCK_SIZE_256) <
        ops::FloorDiv(std::max(STREAM_K_MIN_K_THRESHOLD, compileInfo_.aicNum * BASIC_BLOCK_K_256_BYTE),
                      args_.aDtypeSize)) {
        OP_LOGD(args_.opName, "BatchMatMulV3 tiling unenable state is DoStreamK value[%lu]", args_.kValue);
        return false;
    }

    uint64_t alignValue = BASIC_BLOCK_SIZE_256;
    if (args_.aDtypeSize == DATA_SIZE_FP32 && !args_.isHf32) {
        alignValue = BLOCK_BYTE_SIZE; // 如果是Fp32 基本块判断要用32
    }
    // 判断mn是否需要已经能切32份及以上
    uint64_t mCnt = MathUtil::CeilDivision(args_.mValue, alignValue);
    uint64_t nCnt = MathUtil::CeilDivision(args_.nValue, alignValue);
    if (batchInfo_->batchC * mCnt * nCnt > compileInfo_.aicNum / NUM_TWO) {
        OP_LOGD(args_.opName, "BatchMatMulV3 tiling unenable state is DoStreamK batch[%lu] mCnt[%lu], nCnt[%lu]",
                batchInfo_->batchC, mCnt, nCnt);
        return false;
    }
    OP_LOGI(args_.opName, "BatchMatMulV3 tiling enable state is DoBasicApiSplitK.");
    return true;
}

bool BatchMatMulV3BasicStreamKTiling::IsCapable()
{
    bool isNotEqualBatch = batchInfo_->batchA0 != batchInfo_->batchB0 || batchInfo_->batchA1 != batchInfo_->batchB1 ||
                           batchInfo_->batchA2 != batchInfo_->batchB2 || batchInfo_->batchA3 != batchInfo_->batchB3;
    if (isNotEqualBatch) {
        return false;
    }
    if (args_.aFormat != ge::FORMAT_ND) {
        OP_LOGD(args_.opName, "ND is the only supported format for tensor_a in basic api");
        return false;
    }
    if (batchInfo_->batchBias > 1UL) {
        return false;
    }
    if (MatMulV3TilingHelper::IsSelfNonContiguous(context_)) {
        OP_LOGD(args_.opName, "NonContiguous self does not support StreamK");
        return false;
    }
    if (compileInfo_.aivNum != (compileInfo_.aicNum * NUM_TWO)) {
        OP_LOGD(args_.opName, "streamk only support aivNum == aicNum * 2");
        return false;
    }
    return CheckStreamKSKTiling();
}

ge::graphStatus BatchMatMulV3BasicStreamKTiling::DoOpTiling()
{
    MatMulV3TilingHelper::ResetBase(compileInfo_, args_, runInfo_);

    mCnt_ = MathUtil::CeilDivision(args_.mValue, runInfo_.baseM);
    nCnt_ = MathUtil::CeilDivision(args_.nValue, runInfo_.baseN);
    uint64_t blocksPerBatch = ops::FloorDiv(compileInfo_.aicNum, batchInfo_->batchC);
    if (mCnt_ > blocksPerBatch / NUM_THREE && mCnt_ < blocksPerBatch / NUM_TWO) {
        mCnt_ = blocksPerBatch / NUM_TWO;
    }
    if (nCnt_ > blocksPerBatch / NUM_THREE && nCnt_ < blocksPerBatch / NUM_TWO) {
        nCnt_ = blocksPerBatch / NUM_TWO;
    }
    uint64_t mnCnt = mCnt_ * nCnt_;
    runInfo_.baseM = ops::CeilAlign(MathUtil::CeilDivision(args_.mValue, mCnt_), BASIC_BLOCK_SIZE_16);
    runInfo_.baseN = ops::CeilAlign(MathUtil::CeilDivision(args_.nValue, nCnt_), BASIC_BLOCK_SIZE_16);
    runInfo_.tailInfo.kCnt = ops::FloorDiv(blocksPerBatch, mnCnt);
    runInfo_.singleCoreK = MathUtil::CeilDivision(args_.kValue, runInfo_.tailInfo.kCnt);
    l0C2Out_ = GetL0C2OutFlag();
    uint64_t baseKAlignValue =
        !args_.isATrans || args_.isBTrans ? BASIC_BLOCK_SIZE_128 / args_.aDtypeSize : BASIC_BLOCK_SIZE_16;
    uint64_t kValueMax = ops::FloorAlign(
        L0A_SIZE_2 / DB_SIZE / args_.aDtypeSize / std::max(runInfo_.baseM, runInfo_.baseN), baseKAlignValue);
    runInfo_.baseK = std::min(runInfo_.singleCoreK, kValueMax);
    MatMulV3TilingHelper::CalL1Tiling(compileInfo_, args_, runInfo_);
    // depthb1 is less than deptha1
    if (runInfo_.baseM == runInfo_.baseN && runInfo_.depthB1 == runInfo_.depthA1 * NUM_TWO) {
        runInfo_.depthA1 = runInfo_.depthA1 * NUM_TWO;
        runInfo_.depthB1 = runInfo_.depthB1 / NUM_TWO;
        runInfo_.stepKb = runInfo_.depthB1 / DB_SIZE;
        runInfo_.stepKa = runInfo_.depthA1 / DB_SIZE;
    }
    return ge::GRAPH_SUCCESS;
}

std::vector<size_t> BatchMatMulV3BasicStreamKTiling::GetWorkspaceSize() const
{
    size_t workspaceSize =
        compileInfo_.aicNum * BASIC_BLOCK_SIZE_256 * BASIC_BLOCK_SIZE_256 * DATA_SIZE_FP32 + RPC_WORKSIZE * MB_SIZE;
    OP_LOGI(args_.opName, "BatchMatMulV3 tiling workspace size is %lu", workspaceSize);
    return {workspaceSize};
}

uint64_t BatchMatMulV3BasicStreamKTiling::GetTilingKey() const
{
    return BatchMatMulV3TilingKey()
        .SetTrans(args_.isATrans, args_.isBTrans)
        .SetBatchModel(MatMulV3BatchModel::BATCH_MODEL)
        .SetApiLevel(MatMulV3ApiLevel::BASIC_LEVEL)
        .SetModel(MatMulV3Model::STREAM_K)
        .SetFullLoad(MatMulV3FullLoad::NONE_FULL_LOAD)
        .SetL0C2Out(l0C2Out_)
        .GetTilingKey();
}

ge::graphStatus BatchMatMulV3BasicStreamKTiling::GetTilingData(TilingResult& tiling) const
{
    return GetTilingDataImpl<BatchMatMulV3BasicTilingData>(tiling);
}
} // namespace batch_matmul_v3_advanced
} // namespace optiling

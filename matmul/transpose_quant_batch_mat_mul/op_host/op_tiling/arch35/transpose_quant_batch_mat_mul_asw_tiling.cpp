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
 * \file transpose_quant_batch_mat_mul_v3_tiling.cc
 * \brief
 */
#include "transpose_quant_batch_mat_mul_asw_tiling.h"
#include "transpose_quant_batch_mat_mul_common.h"
#include "matmul/mat_mul_v3/op_host/op_tiling/arch35/matmul_tiling_registry.h"
#include "transpose_quant_batch_mat_mul_tiling_strategy.h"
#include "tiling_base/tiling_key.h"
#include "../../../op_kernel/arch35/transpose_quant_batch_mat_mul_tiling_key.h"

using Ops::NN::MathUtil;
namespace optiling {
namespace transpose_quant_batch_mat_mul_advanced {
using namespace strategy;
MM_REGISTER_TILING_TEMPLATE(TransposeQuantBatchMatMul, TransposeQuantBatchMatMulAswTiling, DAV_3510, BASE);

ge::graphStatus TransposeQuantBatchMatMulAswTiling::DoOpTiling()
{
    MatMulV3TilingHelper::ResetBase(compileInfo_, args_, runInfo_);

    runInfo_.baseM = std::min(args_.mValue, static_cast<uint64_t>(BASIC_BLOCK_SIZE_256));
    runInfo_.baseM = ops::CeilAlign(runInfo_.baseM, CUBE_BLOCK);
    runInfo_.baseN = std::min(args_.nValue, static_cast<uint64_t>(BASIC_BLOCK_SIZE_256));
    runInfo_.baseN = ops::CeilAlign(runInfo_.baseN, L1_ALIGN_SIZE);
    uint64_t baseKDefaultSize = BASIC_BLOCK_SIZE_128;
    runInfo_.baseK = ops::CeilAlign(std::min(baseKDefaultSize, args_.kValue), CUBE_REDUCE_BLOCK);
    AdjustBasicBlock();
    runInfo_.singleCoreM = runInfo_.baseM;
    runInfo_.singleCoreN = runInfo_.baseN;

    MatMulV3TilingHelper::CalL1Tiling(compileInfo_, args_, runInfo_);
    return ge::GRAPH_SUCCESS;
}

void TransposeQuantBatchMatMulAswTiling::AdjustBasicBlock()
{
    uint64_t baseMAlignNum = args_.isATrans ? L2_ALIGN_SIZE : CUBE_BLOCK;
    uint64_t baseNAlignNum = args_.isBTrans ? CUBE_BLOCK : L2_ALIGN_SIZE;
    uint64_t baseKAlignNum = (args_.isATrans && ! args_.isBTrans) ?
                                 BASIC_BLOCK_SIZE_32:
                                 L2_ALIGN_SIZE;
    uint64_t mMaxtile = MathUtil::CeilDivision(args_.mValue, baseMAlignNum);
    uint64_t nMaxtile = MathUtil::CeilDivision(args_.nValue, baseNAlignNum);
    uint64_t tempBaseM = runInfo_.baseM;
    uint64_t tempBaseN = runInfo_.baseN;
    uint64_t coreNumMN = compileInfo_.aicNum;
    if (mMaxtile * nMaxtile >= coreNumMN || (!args_.isATrans && args_.isBTrans)) {
        uint64_t mCore = MathUtil::CeilDivision(args_.mValue, runInfo_.baseM);
        uint64_t nCore = MathUtil::CeilDivision(args_.nValue, runInfo_.baseN);
        if (mMaxtile < nMaxtile || (mMaxtile == nMaxtile && baseNAlignNum == CUBE_BLOCK)) {
            tempBaseM = ops::CeilAlign(MathUtil::CeilDivision(args_.mValue, mCore), baseMAlignNum);
            mCore = MathUtil::CeilDivision(args_.mValue, tempBaseM);
            nCore = ops::FloorDiv(coreNumMN, mCore);
            tempBaseN = ops::CeilAlign(MathUtil::CeilDivision(args_.nValue, nCore), baseNAlignNum);
        } else {
            tempBaseN = ops::CeilAlign(MathUtil::CeilDivision(args_.nValue, nCore), baseNAlignNum);
            nCore = MathUtil::CeilDivision(args_.nValue, tempBaseN);
            mCore = ops::FloorDiv(coreNumMN, nCore);
            tempBaseM = ops::CeilAlign(MathUtil::CeilDivision(args_.mValue, mCore), baseMAlignNum);
        }

        while (tempBaseN >= tempBaseM * BASEM_BASEN_RATIO && nCore < coreNumMN / NUM_HALF &&
               tempBaseN != baseNAlignNum) {
            nCore = nCore * DOUBLE_CORE_NUM;
            mCore = ops::FloorDiv(coreNumMN, nCore);
            tempBaseM = ops::CeilAlign(MathUtil::CeilDivision(args_.mValue, mCore), baseMAlignNum);
            tempBaseN = ops::CeilAlign(MathUtil::CeilDivision(args_.nValue, nCore), baseNAlignNum);
            mCore = MathUtil::CeilDivision(args_.mValue, static_cast<uint64_t>(tempBaseM));
            nCore = MathUtil::CeilDivision(args_.nValue, static_cast<uint64_t>(tempBaseN));
        }
        while (tempBaseM >= tempBaseN * BASEM_BASEN_RATIO && mCore < coreNumMN / NUM_HALF &&
               tempBaseM != baseMAlignNum) {
            mCore = mCore * DOUBLE_CORE_NUM;
            nCore = ops::FloorDiv(coreNumMN, mCore);
            tempBaseM = ops::CeilAlign(MathUtil::CeilDivision(args_.mValue, mCore), baseMAlignNum);
            tempBaseN = ops::CeilAlign(MathUtil::CeilDivision(args_.nValue, nCore), baseNAlignNum);
            mCore = MathUtil::CeilDivision(args_.mValue, static_cast<uint64_t>(tempBaseM));
            nCore = MathUtil::CeilDivision(args_.nValue, static_cast<uint64_t>(tempBaseN));
        }
        uint64_t kValueAlign = ops::CeilAlign(static_cast<uint64_t>(args_.kValue), baseKAlignNum);
        uint64_t kValueMax = compileInfo_.l0ASize / DB_SIZE / std::max(tempBaseM, tempBaseN);
        if (kValueMax >= baseKAlignNum) {
            runInfo_.baseM = tempBaseM;
            runInfo_.baseN = tempBaseN;
            kValueMax = ops::FloorAlign(kValueMax, baseKAlignNum);
            runInfo_.baseK = std::min(kValueAlign, kValueMax);
            runInfo_.baseK = runInfo_.baseK > BASEK_LIMIT ? runInfo_.baseK / NUM_HALF : runInfo_.baseK;
        }
    }
}

uint64_t TransposeQuantBatchMatMulAswTiling::GetTilingKey() const
{
    return GET_TPL_TILING_KEY(static_cast<uint64_t>(permX1_), static_cast<uint64_t>(permX2_),
                              static_cast<uint64_t>(batchSplitMode_));
}

uint64_t TransposeQuantBatchMatMulAswTiling::GetNumBlocks() const
{
    return compileInfo_.aicNum;
}

ge::graphStatus TransposeQuantBatchMatMulAswTiling::GetTilingData(TilingResult& tiling) const
{
    return GetTilingDataImpl<BatchMatMulV3TilingData>(tiling);
}

ge::graphStatus TransposeQuantBatchMatMulAswTiling::GetTilingDataProcess(BatchMatMulV3TilingData& tilingData) const
{
    tilingData.batchSplitFactor = batchSplitFactor_;
    return MatMulV3BaseTiling::GetTilingDataProcess(tilingData);
}

std::vector<size_t> TransposeQuantBatchMatMulAswTiling::GetWorkspaceSize() const
{
    std::vector<size_t> workspaceSize{0};
    return workspaceSize;
};
} // namespace transpose_quant_batch_mat_mul_advanced

} // namespace optiling
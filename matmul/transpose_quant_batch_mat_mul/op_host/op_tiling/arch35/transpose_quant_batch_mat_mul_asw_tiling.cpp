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
#include "op_host/tiling_key.h"
#include "../../../op_kernel/arch35/transpose_quant_batch_mat_mul_tiling_key.h"
#include "matmul/common/op_host/math_util.h"

using Ops::NN::MathUtil;
namespace optiling {
namespace transpose_quant_batch_mat_mul_advanced {
using namespace strategy;
MM_REGISTER_TILING_TEMPLATE(TransposeQuantBatchMatMul, TransposeQuantBatchMatMulAswTiling, DAV_3510, BASE);

ge::graphStatus TransposeQuantBatchMatMulAswTiling::DoOpTiling()
{
    MatMulV3TilingHelper::ResetBase(compileInfo_, args_, runInfo_);
    if (IsMicroScaling(context_->GetOptionalInputDesc(SCALE_X1_IDX), context_->GetOptionalInputDesc(SCALE_X2_IDX))) {
        precisionMode_ = TQBMMPrecisionMode::PRECISION_MODE_MXFP8;
        CalL1Tiling();
    } else {
        precisionMode_ = TQBMMPrecisionMode::PRECISION_MODE_FP8;
        MatMulV3TilingHelper::CalL1Tiling(compileInfo_, args_, runInfo_);
    }
    GetTransposeBatchMatMulInfo();
    return ge::GRAPH_SUCCESS;
}

bool TransposeQuantBatchMatMulAswTiling::IsCapable()
{
    if (!IsMicroScaling(context_->GetOptionalInputDesc(SCALE_X1_IDX), context_->GetOptionalInputDesc(SCALE_X2_IDX)) &&
        compileInfo_.aivNum != compileInfo_.aicNum * NUM_TWO) {
        OP_LOGE(args_.opName,
                "TransposeQuantBatchMatMul is only supported for aivNum == aicNum *2.aivNum:%llu,aicNum:%llu",
                compileInfo_.aivNum, compileInfo_.aicNum);
        return false;
    }
    return true;
}


void TransposeQuantBatchMatMulAswTiling::GetTransposeBatchMatMulInfo()
{
    auto attrs = context_->GetAttrs();
    const gert::ContinuousVector* aPermList = attrs->GetAttrPointer<gert::ContinuousVector>(PERM_X1_IDX);
    const gert::ContinuousVector* bPermList = attrs->GetAttrPointer<gert::ContinuousVector>(PERM_X2_IDX);
    if (aPermList != nullptr && aPermList->GetSize() == ALLOW_DIM) {
        const int64_t* aPerm = static_cast<const int64_t*>(aPermList->GetData());
        if ((aPerm[BATCH_IDX] == 1L) && (aPerm[M_IDX] == 0L) && (aPerm[KA_IDX] == 2L)) {
            permX1_ = TQBMMPermX1::PERM_X1_1_0_2;
        } else if ((aPerm[BATCH_IDX] == 0L) && (aPerm[M_IDX] == 1L) && (aPerm[KA_IDX] == 2L)) {
            permX1_ = TQBMMPermX1::PERM_X1_0_1_2;
        }
    }
    if (bPermList != nullptr && bPermList->GetSize() == ALLOW_DIM) {
        const int64_t* bPerm = static_cast<const int64_t*>(bPermList->GetData());
        if ((bPerm[BATCH_IDX] == 0L) && (bPerm[M_IDX] == 1L) && (bPerm[KA_IDX] == 2L)) {
            permX2_ = TQBMMPermX2::PERM_X2_0_1_2;
        } else if ((bPerm[BATCH_IDX] == 0L) && (bPerm[M_IDX] == 2L) && (bPerm[KA_IDX] == 1L)) {
            permX2_ = TQBMMPermX2::PERM_X2_0_2_1;
        }
    }
    if (attrs->GetAttrNum() >= ATTR_NUM) {
        batchSplitFactor_ = std::max(*(attrs->GetAttrPointer<int32_t>(ATTR_NUM - 1)), 1);
        batchSplitMode_ = batchSplitFactor_ > 1 ? TQBMMBatchSplit::BATCH_SPLIT_TRUE : TQBMMBatchSplit::BATCH_SPLIT_FALSE;
    }
}

void TransposeQuantBatchMatMulAswTiling::CalL1Tiling()
{
    uint64_t leftL1Size = compileInfo_.l1Size;
    if (args_.hasBias) {
        leftL1Size -= runInfo_.baseN * ge::GetSizeByDataType(args_.biasType);
    }
    uint64_t baseASize = runInfo_.baseM * runInfo_.baseK * args_.aDtypeSize;
    uint64_t baseBSize = runInfo_.baseN * runInfo_.baseK * args_.bDtypeSize;
    uint64_t baseScaleASize =
        ops::CeilAlign(ops::CeilDiv(static_cast<uint64_t>(runInfo_.baseK), MX_GROUP_SIZE), MXFP_MULTI_BASE_SIZE) *
        runInfo_.baseM * ge::GetSizeByDataType(context_->GetOptionalInputDesc(SCALE_X1_IDX)->GetDataType());
    uint64_t baseScaleBSize =
        ops::CeilAlign(ops::CeilDiv(static_cast<uint64_t>(runInfo_.baseK), MX_GROUP_SIZE), MXFP_MULTI_BASE_SIZE) *
        runInfo_.baseN * ge::GetSizeByDataType(context_->GetOptionalInputDesc(SCALE_X1_IDX)->GetDataType());
    uint64_t baseL1Size = baseASize + baseBSize + baseScaleASize + baseScaleBSize;
    uint64_t depthInit = GetDepthA1B1(leftL1Size, baseL1Size, 1UL);
    uint64_t leftL1SizeByDepthInit = leftL1Size - depthInit * (baseL1Size);
    uint64_t depthASec = GetDepthA1B1(leftL1SizeByDepthInit, (baseASize + baseScaleASize) * depthInit, depthInit);
    uint64_t depthBSec = GetDepthA1B1(leftL1SizeByDepthInit, (baseBSize + baseScaleBSize) * depthInit, depthInit);
    runInfo_.depthA1 = std::max(depthASec, depthBSec);
    runInfo_.depthB1 = runInfo_.depthA1;
    if (runInfo_.depthA1 * baseL1Size > leftL1Size) {
        runInfo_.depthA1 = depthASec >= depthBSec ? depthASec : depthInit;
        runInfo_.depthB1 = depthASec < depthBSec ? depthBSec : depthInit;
    }
    CalStepKs();
    CalScaleFactors(baseASize, baseBSize, baseScaleASize, baseScaleBSize);
    runInfo_.singleCoreM = runInfo_.baseM;
    runInfo_.singleCoreN = runInfo_.baseN;
    return;
}

void TransposeQuantBatchMatMulAswTiling::CalScaleFactors(uint64_t baseASize, uint64_t baseBSize,
                                                         uint64_t baseScaleASize, uint64_t baseScaleBSize)
{
    uint64_t biasDtypeSize = ge::GetSizeByDataType(args_.biasType);
    uint64_t baseBiasSize = args_.hasBias ? runInfo_.baseN * biasDtypeSize : 0;

    // 计算scaleFactorA, scaleFactorB
    // 来自K轴的约束
    uint32_t scaleFactorAMax =
        std::min(static_cast<uint32_t>(ops::FloorDiv(MTE2_MIN_LOAD_SIZE_V100, baseScaleASize)), SCALER_FACTOR_MAX);
    uint32_t scaleFactorBMax =
        std::min(static_cast<uint32_t>(ops::FloorDiv(MTE2_MIN_LOAD_SIZE_V100, baseScaleBSize)), SCALER_FACTOR_MAX);
    uint32_t scaleFactorA = static_cast<uint32_t>(args_.kValue) / (runInfo_.stepKa * runInfo_.baseK);
    uint32_t scaleFactorB = static_cast<uint32_t>(args_.kValue) / (runInfo_.stepKb * runInfo_.baseK);
    scaleFactorA_ = std::max(SCALER_FACTOR_MIN, scaleFactorA);
    scaleFactorB_ = std::max(SCALER_FACTOR_MIN, scaleFactorB);
    scaleFactorA_ = std::min(scaleFactorAMax, scaleFactorA);
    scaleFactorB_ = std::min(scaleFactorBMax, scaleFactorB);

    // 来自L1 size 的约束
    uint64_t leftL1sie =
        compileInfo_.l1Size - (runInfo_.depthA1 * baseASize + runInfo_.depthB1 * baseBSize + baseBiasSize);
    uint32_t scaleInit =
        static_cast<uint32_t>(leftL1sie / (runInfo_.depthA1 * baseScaleASize + runInfo_.depthB1 * baseScaleBSize));
    if (scaleFactorA_ <= scaleInit && scaleFactorB_ > scaleInit) {
        leftL1sie -= (static_cast<uint64_t>(scaleFactorA_) * runInfo_.depthA1 * baseScaleASize);
        scaleFactorB_ = std::min(static_cast<uint32_t>(leftL1sie / (runInfo_.depthB1 * baseScaleBSize)), scaleFactorB_);
    } else if (scaleFactorB_ <= scaleInit && scaleFactorA_ > scaleInit) {
        leftL1sie -= (static_cast<uint64_t>(scaleFactorB_) * runInfo_.depthB1 * baseScaleBSize);
        scaleFactorA_ = std::min(static_cast<uint32_t>(leftL1sie / (runInfo_.depthA1 * baseScaleASize)), scaleFactorA_);
    } else if (scaleFactorA_ > scaleInit && scaleFactorB_ > scaleInit) {
        leftL1sie -= (static_cast<uint64_t>(scaleInit) * runInfo_.depthB1 * baseScaleBSize +
                      static_cast<uint64_t>(scaleInit) * runInfo_.depthA1 * baseScaleASize);
        uint32_t scaleASec =
            std::min(static_cast<uint32_t>(leftL1sie / (runInfo_.depthA1 * baseScaleASize)), scaleFactorA_ - scaleInit);
        uint32_t scaleBSec =
            std::min(static_cast<uint32_t>(leftL1sie / (runInfo_.depthB1 * baseScaleBSize)), scaleFactorB_ - scaleInit);
        scaleFactorA_ = scaleASec >= scaleBSec ? (scaleASec + scaleInit) : scaleInit;
        scaleFactorB_ = scaleASec < scaleBSec ? (scaleBSec + scaleInit) : scaleInit;
    }
}

void TransposeQuantBatchMatMulAswTiling::CalStepKs()
{
    runInfo_.stepKa = runInfo_.depthA1 / DB_SIZE;
    runInfo_.stepKb = runInfo_.depthB1 / DB_SIZE;

    if (static_cast<uint64_t>(runInfo_.stepKa * runInfo_.baseK) > args_.kValue) {
        runInfo_.stepKa = ops::CeilDiv(args_.kValue, static_cast<uint64_t>(runInfo_.baseK));
    }

    if (static_cast<uint64_t>(runInfo_.stepKb * runInfo_.baseK) > args_.kValue) {
        runInfo_.stepKb = ops::CeilDiv(args_.kValue, static_cast<uint64_t>(runInfo_.baseK));
    }

    if (runInfo_.stepKa > runInfo_.stepKb) {
        runInfo_.stepKa = runInfo_.stepKa / runInfo_.stepKb * runInfo_.stepKb;
    }
    if (runInfo_.stepKb > runInfo_.stepKa) {
        runInfo_.stepKb = runInfo_.stepKb / runInfo_.stepKa * runInfo_.stepKa;
    }
    runInfo_.stepKa =
        std::min(runInfo_.stepKa, static_cast<uint64_t>(4)); // 限制stepKa最大为4, 防止issue queue阻塞
    runInfo_.stepKb =
        std::min(runInfo_.stepKb, static_cast<uint64_t>(4)); // 限制stepKb最大为4, 防止issue queue阻塞
    runInfo_.depthA1 = runInfo_.stepKa * DB_SIZE;
    runInfo_.depthB1 = runInfo_.stepKb * DB_SIZE;
}

uint64_t TransposeQuantBatchMatMulAswTiling::GetDepthA1B1(uint64_t leftSize, uint64_t perDepthSize, uint64_t depthInit)
{
    if (depthInit > 1UL && perDepthSize > DB_SIZE * MTE2_MIN_LOAD_SIZE_V100) {
        return depthInit;
    }
    uint64_t depthScale = ops::FloorDiv(leftSize, perDepthSize);
    if (depthInit > 1UL) {
        uint64_t baseKSize = static_cast<uint64_t>(runInfo_.baseK) * args_.aDtypeSize;
        while ((depthScale * baseKSize) % BASIC_BLOCK_SIZE_512 != 0UL &&
               (depthScale * baseKSize) > BASIC_BLOCK_SIZE_512) {
            depthScale -= 1UL;
        }
        if ((depthScale * baseKSize) % BASIC_BLOCK_SIZE_512 != 0UL &&
            (depthScale * baseKSize) >= BASIC_BLOCK_SIZE_256) {
            depthScale = BASIC_BLOCK_SIZE_256 / baseKSize;
        }
        depthScale = std::max(depthScale, static_cast<uint64_t>(1));
    } else {
        constexpr uint64_t index = 2; // 2: depth的值是2的幂
        depthScale = 1UL;
        while (depthScale * (perDepthSize) < leftSize) {
            depthScale *= index;
        }
        depthScale = depthScale == 1UL ? depthScale : depthScale / index;
    }
    return depthInit * depthScale;
}

uint64_t TransposeQuantBatchMatMulAswTiling::GetTilingKey() const
{
    return GET_TPL_TILING_KEY(static_cast<uint64_t>(permX1_), static_cast<uint64_t>(permX2_),
                              static_cast<uint64_t>(batchSplitMode_), static_cast<uint64_t>(precisionMode_));
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
    ge::graphStatus ret = MatMulV3BaseTiling::GetTilingDataProcess(tilingData);
    auto x1Scale = context_->GetOptionalInputDesc(SCALE_X1_IDX);
    if (x1Scale != nullptr && x1Scale->GetDataType() == ge::DT_FLOAT8_E8M0) {
        tilingData.matMulTilingData.tCubeTiling.mxTypePara =
            (SCALER_FACTOR_MIN << SCALER_FACTOR_N_BIT) + (SCALER_FACTOR_MIN << SCALER_FACTOR_M_BIT);
        if (scaleFactorA_ >= SCALER_FACTOR_MIN && scaleFactorA_ <= SCALER_FACTOR_MAX &&
            scaleFactorB_ >= SCALER_FACTOR_MIN && scaleFactorB_ <= SCALER_FACTOR_MAX) {
            tilingData.matMulTilingData.tCubeTiling.mxTypePara +=
                (scaleFactorB_ << SCALER_FACTOR_B_BIT) + scaleFactorA_;
        } else {
            tilingData.matMulTilingData.tCubeTiling.mxTypePara +=
                (SCALER_FACTOR_DEFAULT << SCALER_FACTOR_B_BIT) + SCALER_FACTOR_DEFAULT;
        }
    }
    tilingData.batchSplitFactor = batchSplitFactor_;
    return ret;
}

std::vector<size_t> TransposeQuantBatchMatMulAswTiling::GetWorkspaceSize() const
{
    std::vector<size_t> workspaceSize{0};
    return workspaceSize;
};
} // namespace transpose_quant_batch_mat_mul_advanced

} // namespace optiling
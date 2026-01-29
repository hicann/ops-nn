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
 * \file weight_quant_batch_matmul_v2_iterbatch_tiling.cpp
 * \brief
 */

#include "weight_quant_batch_matmul_v2_iterbatch_tiling.h"

#include "matmul/common/op_host/math_util.h"
#include "matmul/weight_quant_batch_matmul_v2/op_kernel/arch35/weight_quant_batch_matmul_v2_arch35_tiling_data.h"
#include "../../../op_kernel/arch35/weight_quant_batch_matmul_v2_arch35_tiling_key.h"
#include "../weight_quant_batch_matmul_v2_tiling_key.h"
using namespace platform_ascendc;

namespace {
constexpr uint64_t CUBE_BLOCK = 16;
constexpr uint64_t L1_ALIGN_SIZE = 32;
constexpr uint64_t CUBE_REDUCE_BLOCK = 32;
constexpr uint32_t BASIC_BLOCK_SIZE_128 = 128;
constexpr uint32_t DB_SIZE = 2;
constexpr uint32_t DATA_SIZE_L0C = 4;
constexpr int32_t ITERBATCH_PRIORITY = 9;
}  // namespace

namespace optiling {
namespace weight_quant_batch_matmul_v2 {
ge::graphStatus WeightQuantBatchMatmulV2IterbatchTiling::DoOpTiling()
{
    OP_LOGD(opName_, "DoOpTiling of iterate batch tiling strategy.");
    OP_TILING_CHECK(InstantiateTilingData() == ge::GRAPH_FAILED,
                    VECTOR_INNER_ERR_REPORT_TILIING(opName_, "unable to get pointer of tiling data"),
                    return ge::GRAPH_FAILED);

    CalL1Tiling();
    SetBatchParams();
    return ge::GRAPH_SUCCESS;
}

void WeightQuantBatchMatmulV2IterbatchTiling::CalL1Tiling()
{
    uint64_t calcBatch = 0;
    if (matmulInfoPtr_->batchX3 != matmulInfoPtr_->batchWeight3) {
        calcBatch = matmulInfoPtr_->batchY3;
    } else if (matmulInfoPtr_->batchX2 != matmulInfoPtr_->batchWeight2) {
        calcBatch = matmulInfoPtr_->batchY2 * matmulInfoPtr_->batchY3;
    } else if (matmulInfoPtr_->batchX1 != matmulInfoPtr_->batchWeight1) {
        calcBatch = matmulInfoPtr_->batchY1 * matmulInfoPtr_->batchY2 * matmulInfoPtr_->batchY3;
    } else {
        calcBatch = matmulInfoPtr_->batchY;
    }
    basicTiling_.usedCoreNum = std::min(static_cast<uint64_t>(compileInfoPtr_->aicNum), calcBatch);
    basicTiling_.singleCoreM = matmulInfoPtr_->mSize;
    basicTiling_.singleCoreN = matmulInfoPtr_->nSize;
    basicTiling_.singleCoreK = matmulInfoPtr_->kSize;
    
    basicTiling_.baseM = std::min(matmulInfoPtr_->mSize, static_cast<uint64_t>(BASIC_BLOCK_SIZE_128));
    basicTiling_.baseM =
        !matmulInfoPtr_->transA
            ? ops::CeilAlign(static_cast<uint64_t>(basicTiling_.baseM), CUBE_BLOCK)
            : ops::CeilAlign(static_cast<uint64_t>(basicTiling_.baseM),
                             GetShapeWithDataType(L1_ALIGN_SIZE, matmulInfoPtr_->aDtype));
    basicTiling_.baseN = std::min(matmulInfoPtr_->nSize, static_cast<uint64_t>(BASIC_BLOCK_SIZE_128));
    basicTiling_.baseN =
        matmulInfoPtr_->transB
            ? ops::CeilAlign(static_cast<uint64_t>(basicTiling_.baseN), CUBE_BLOCK)
            : ops::CeilAlign(static_cast<uint64_t>(basicTiling_.baseN),
                             GetShapeWithDataType(L1_ALIGN_SIZE, matmulInfoPtr_->bDtype));

    uint64_t minBaseK =
        std::min(std::min(GetShapeWithDataType(static_cast<uint64_t>(BASIC_BLOCK_SIZE_128), matmulInfoPtr_->aDtype),
                          GetShapeWithDataType(static_cast<uint64_t>(BASIC_BLOCK_SIZE_128), matmulInfoPtr_->bDtype)),
                 matmulInfoPtr_->kSize);
    uint64_t maxAlignSize =
        std::max(static_cast<uint64_t>(GetShapeWithDataType(CUBE_REDUCE_BLOCK, matmulInfoPtr_->aDtype)),
                 static_cast<uint64_t>(GetShapeWithDataType(CUBE_REDUCE_BLOCK, matmulInfoPtr_->bDtype)));
    basicTiling_.baseK = ops::CeilAlign(minBaseK, maxAlignSize);

    basicTiling_.stepM = 1;
    basicTiling_.stepN = 1;

    basicTiling_.iterateOrder = 0;
    basicTiling_.dbL0c =
        ((basicTiling_.baseM * basicTiling_.baseN * DATA_SIZE_L0C * DB_SIZE <= compileInfoPtr_->l0cSize) &&
         CheckAntiQuantScale(basicTiling_.baseN, DB_SIZE))
            ? DB_SIZE
            : 1;
    CalL1TilingDepth(leftL1Size_);
}

uint32_t WeightQuantBatchMatmulV2IterbatchTiling::CalcIterBatch()
{
    uint64_t biasDtypeSize = ge::GetSizeByDataType(matmulInfoPtr_->biasDtype);
    uint64_t scaleDtypeSize = ge::GetSizeByDataType(matmulInfoPtr_->antiQuantScaleDtype);
    uint64_t totalL1Size = compileInfoPtr_->l1Size;
    uint64_t singleCoreBiasSize = matmulInfoPtr_->hasBias ? basicTiling_.baseN * biasDtypeSize : 0;
    uint64_t singleCoreScaleSize =
        matmulInfoPtr_->antiQuantType == QuantType::PER_CHANNEL ? basicTiling_.baseN * scaleDtypeSize : 0;
    leftL1Size_ = totalL1Size - singleCoreBiasSize - singleCoreScaleSize;
    // get align m,k,n value
    uint64_t baseMAlignNum =
        matmulInfoPtr_->transA ? GetShapeWithDataType(L1_ALIGN_SIZE, matmulInfoPtr_->aDtype) : CUBE_BLOCK;
    uint64_t baseNAlignNum =
        matmulInfoPtr_->transB ? CUBE_BLOCK : GetShapeWithDataType(L1_ALIGN_SIZE, matmulInfoPtr_->bDtype);
    uint64_t baseKAlignNum = (matmulInfoPtr_->transA && !matmulInfoPtr_->transB)
                                ? CUBE_BLOCK
                                : GetShapeWithDataType(L1_ALIGN_SIZE, matmulInfoPtr_->aDtype);
    uint64_t alignMValue = ops::CeilAlign(static_cast<uint64_t>(matmulInfoPtr_->mSize), baseMAlignNum);
    uint64_t alignNValue = ops::CeilAlign(static_cast<uint64_t>(matmulInfoPtr_->nSize), baseNAlignNum);
    uint64_t alignKValue = ops::CeilAlign(static_cast<uint64_t>(matmulInfoPtr_->kSize), baseKAlignNum);

    uint32_t iterBatch = ops::FloorDiv(
        leftL1Size_, GetSizeWithDataType(alignMValue * alignKValue, matmulInfoPtr_->aDtype) +
                         GetSizeWithDataType(alignKValue * alignNValue, matmulInfoPtr_->bDtype));
    return iterBatch;
}

void WeightQuantBatchMatmulV2IterbatchTiling::GetBroadCastInfo(uint64_t& broadcastNum, uint64_t& innerBatchNum,
                                                               bool& isBroadcastA, bool& isBroadcastB)
{
    if (matmulInfoPtr_->batchX3 != matmulInfoPtr_->batchWeight3) {
        broadcastNum += 1UL;
        isBroadcastA = (matmulInfoPtr_->batchX3 < matmulInfoPtr_->batchWeight3 || isBroadcastA);
        isBroadcastB = (matmulInfoPtr_->batchX3 > matmulInfoPtr_->batchWeight3 || isBroadcastB);
    }
    if (matmulInfoPtr_->batchX2 != matmulInfoPtr_->batchWeight2) {
        broadcastNum += 1UL;
        innerBatchNum *= matmulInfoPtr_->batchY3;
        isBroadcastA = (matmulInfoPtr_->batchX2 < matmulInfoPtr_->batchWeight2 || isBroadcastA);
        isBroadcastB = (matmulInfoPtr_->batchX2 > matmulInfoPtr_->batchWeight2 || isBroadcastB);
    }
    if (matmulInfoPtr_->batchX1 != matmulInfoPtr_->batchWeight1) {
        broadcastNum += 1UL;
        innerBatchNum *= matmulInfoPtr_->batchY2;
        isBroadcastA = (matmulInfoPtr_->batchX1 < matmulInfoPtr_->batchWeight1 || isBroadcastA);
        isBroadcastB = (matmulInfoPtr_->batchX1 > matmulInfoPtr_->batchWeight1 || isBroadcastB);
    }
    if (matmulInfoPtr_->batchX0 != matmulInfoPtr_->batchWeight0) {
        broadcastNum += 1UL;
        innerBatchNum *= matmulInfoPtr_->batchY1;
        isBroadcastA = (matmulInfoPtr_->batchX0 < matmulInfoPtr_->batchWeight0 || isBroadcastA);
        isBroadcastB = (matmulInfoPtr_->batchX0 > matmulInfoPtr_->batchWeight0 || isBroadcastB);
    }
}

uint32_t WeightQuantBatchMatmulV2IterbatchTiling::GetGcd(uint32_t numA, uint32_t numB) const
{
    if (numA < numB) {
        std::swap(numA, numB);
    }
    if (numB == 0) {
        return 0;
    }
    if (numA % numB == 0) {
        return numB;
    } else {
        return (GetGcd(numB, numA % numB));
    }
}

bool WeightQuantBatchMatmulV2IterbatchTiling::IsCapable()
{
    OP_TILING_CHECK(
        !matmulInfoPtr_->transA && matmulInfoPtr_->batchWeight == 1UL,
        OP_LOGI(opName_, "When transA = False and batchB = 1, batchA can be co-axial with M"),
        return false);

    OP_TILING_CHECK(
        matmulInfoPtr_->batchX == 1UL && matmulInfoPtr_->batchWeight == 1UL,
        OP_LOGI(opName_, "the iter batch template doesn't support tensor X and Weight batch size is 1."
                "batchX: %lu, batchWeight: %lu", matmulInfoPtr_->batchX, matmulInfoPtr_->batchWeight),
        return false);

    uint64_t broadcastNum = 0UL;
    uint64_t innerBatchNum = 1UL;
    bool isBroadcastA = false;
    bool isBroadcastB = false;
    GetBroadCastInfo(broadcastNum, innerBatchNum, isBroadcastA, isBroadcastB);

    OP_TILING_CHECK(
        isBroadcastA && isBroadcastB,
        OP_LOGI(opName_, "The multi-batch optimization currently only supports one matrix being broadcasted"),
        return false);

    OP_TILING_CHECK(
        broadcastNum != 0UL && broadcastNum != 1UL,
        OP_LOGI(opName_, "the multi-batch optimization currently only supports one batch axis can be broadcasted."
                "The number of axis need to broadcast is %lu", broadcastNum),
        return false);
    uint32_t iterBatch = CalcIterBatch();
    OP_TILING_CHECK(
        iterBatch <= 1UL,
        OP_LOGI(opName_, "the iter batch should be large than 1 but %lu", iterBatch),
        return false);

    uint64_t perCoreBatch = ops::CeilDiv(matmulInfoPtr_->batchY, static_cast<uint64_t>(compileInfoPtr_->aicNum));
    iterBatch = std::max(std::min(static_cast<uint64_t>(iterBatch), perCoreBatch), 1UL);
    if (broadcastNum == 1UL) {
        // broadcast场景下，为了保证不出现计算的batch包含broadcast维度的多个batch，batchNum需要满足如下条件：
        // 1. 不能大于broadcast的内轴
        // 2. 需要能整除broadcast的内轴
        iterBatch = GetGcd(static_cast<uint32_t>(innerBatchNum), iterBatch);
    }

    basicTiling_.iterBatch = iterBatch;
    OP_LOGD(opName_, "entering iter batch template. iterBatch = %u", basicTiling_.iterBatch);
    return true;
}

uint64_t WeightQuantBatchMatmulV2IterbatchTiling::GetTilingKey() const
{
    constexpr uint64_t socVersionType = WQBMMV2_SOC_SUPPORT_MMAD_S8S4;
    constexpr uint64_t subSocVersionType = WQBMMV2_DEFAULT;
    constexpr uint64_t antiquantScenario = WQBMMV2_DEFAULT;
    constexpr uint64_t algorithm = WQBMMV2_ALGO_FIXPIPE_ANTIQUANT;
    constexpr uint64_t subAlgorithm = static_cast<uint64_t>(OptimizationAlgorithmSubCategory::ITERATE_BATCH);
    constexpr uint64_t templateCustom = static_cast<uint64_t>(Mte2Configuration::MTE2_INNER_SIZE_512_BUF_NUM_2);
    constexpr uint64_t apiConstexpr = 0UL;
    bool transA = matmulInfoPtr_->transA;
    bool transB = matmulInfoPtr_->transB;
    uint64_t antiquantType = static_cast<uint64_t>(matmulInfoPtr_->antiQuantType);
    uint64_t quantType = static_cast<uint64_t>(matmulInfoPtr_->quantType);
    bool hasAntiquantOffset = matmulInfoPtr_->hasAntiQuantOffset;
    bool hasBias = matmulInfoPtr_->hasBias;
    bool isBiasFp32 = matmulInfoPtr_->biasDtype == ge::DT_FLOAT && matmulInfoPtr_->hasBias;
    constexpr bool isWeightNz = false;
    uint64_t tilingKey = GET_TPL_TILING_KEY(
        socVersionType, subSocVersionType, antiquantScenario, algorithm, subAlgorithm, templateCustom, apiConstexpr,
        transA, transB, antiquantType, quantType, hasAntiquantOffset, hasBias, isBiasFp32, isWeightNz);
    return tilingKey;
}
REGISTER_TILING_TEMPLATE("WeightQuantBatchMatmulV2", WeightQuantBatchMatmulV2IterbatchTiling, ITERBATCH_PRIORITY);
}  // namespace weight_quant_batch_matmul_v2
}  // namespace optiling
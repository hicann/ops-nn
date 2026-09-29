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

#include "matmul/common/op_host/math_util_nn.h"
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
} // namespace

namespace optiling {
namespace weight_quant_batch_matmul_v2 {
ge::graphStatus WeightQuantBatchMatmulV2IterbatchTiling::DoOpTiling()
{
    OP_LOGD(opName_, "DoOpTiling of iterate batch tiling strategy.");
    OP_TILING_CHECK(InstantiateIterbatchTilingData() == ge::GRAPH_FAILED,
                    OP_LOGE(opName_, "unable to get pointer of tiling data"), return ge::GRAPH_FAILED);

    CalL1Tiling();
    iterbatchTilingData_->shiftValue = shiftValue_;
    iterbatchTilingData_->l2CacheDisable = SetDisableL2cache(basicTiling_.baseM, basicTiling_.baseK, basicTiling_.baseK,
                                                             basicTiling_.baseN);
    SetIterbatchBatchParams();
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus WeightQuantBatchMatmulV2IterbatchTiling::InstantiateIterbatchTilingData()
{
    if (iterbatchTilingData_ == nullptr) {
        try {
            // make_unique不会返回空指针，只会返回异常，无需在后面加空指针校验
            iterbatchTilingData_ = std::make_unique<wqbmmv2_tiling::WeightQuantBatchMatmulV2ASWTilingDataParams>();
        } catch (std::bad_alloc&) {
            OP_LOGE(opName_, "tiling data memory allocation failed");
            return ge::GRAPH_FAILED;
        }
    }
    OP_TILING_CHECK(context_->GetRawTilingData()->GetCapacity() < iterbatchTilingDataSize_,
                    OP_LOGE(opName_, "tiling data capacity %zu < actual tiling data size %zu",
                            context_->GetRawTilingData()->GetCapacity(), iterbatchTilingDataSize_),
                    return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

void WeightQuantBatchMatmulV2IterbatchTiling::SetIterbatchBatchParams()
{
    iterbatchTilingData_->params.batchA1 = matmulInfoPtr_->batchX0;
    iterbatchTilingData_->params.batchA2 = matmulInfoPtr_->batchX1;
    iterbatchTilingData_->params.batchA3 = matmulInfoPtr_->batchX2;
    iterbatchTilingData_->params.batchA4 = matmulInfoPtr_->batchX3;
    iterbatchTilingData_->params.batchA = matmulInfoPtr_->batchX;
    iterbatchTilingData_->params.batchB1 = matmulInfoPtr_->batchWeight0;
    iterbatchTilingData_->params.batchB2 = matmulInfoPtr_->batchWeight1;
    iterbatchTilingData_->params.batchB3 = matmulInfoPtr_->batchWeight2;
    iterbatchTilingData_->params.batchB4 = matmulInfoPtr_->batchWeight3;
    iterbatchTilingData_->params.batchB = matmulInfoPtr_->batchWeight;
    iterbatchTilingData_->params.batchC1 = matmulInfoPtr_->batchY0;
    iterbatchTilingData_->params.batchC2 = matmulInfoPtr_->batchY1;
    iterbatchTilingData_->params.batchC3 = matmulInfoPtr_->batchY2;
    iterbatchTilingData_->params.batchC4 = matmulInfoPtr_->batchY3;
    iterbatchTilingData_->params.batchC = matmulInfoPtr_->batchY;
    iterbatchTilingData_->params.biasWithBatch = static_cast<uint64_t>(matmulInfoPtr_->biasWithBatch);
}

ge::graphStatus WeightQuantBatchMatmulV2IterbatchTiling::DoLibApiTiling()
{
    iterbatchTilingData_->matmulTiling.M = matmulInfoPtr_->mSize;
    iterbatchTilingData_->matmulTiling.N = matmulInfoPtr_->nSize;
    iterbatchTilingData_->matmulTiling.Ka = matmulInfoPtr_->kSize;
    iterbatchTilingData_->matmulTiling.Kb = matmulInfoPtr_->kSize;
    iterbatchTilingData_->matmulTiling.usedCoreNum = basicTiling_.usedCoreNum;
    iterbatchTilingData_->matmulTiling.singleCoreM = basicTiling_.singleCoreM;
    iterbatchTilingData_->matmulTiling.singleCoreN = basicTiling_.singleCoreN;
    iterbatchTilingData_->matmulTiling.singleCoreK = basicTiling_.singleCoreK;
    iterbatchTilingData_->matmulTiling.baseM = basicTiling_.baseM;
    iterbatchTilingData_->matmulTiling.baseN = basicTiling_.baseN;
    iterbatchTilingData_->matmulTiling.baseK = basicTiling_.baseK;
    iterbatchTilingData_->matmulTiling.depthA1 = basicTiling_.depthA1;
    iterbatchTilingData_->matmulTiling.depthB1 = basicTiling_.depthB1;
    iterbatchTilingData_->matmulTiling.stepM = basicTiling_.stepM;
    iterbatchTilingData_->matmulTiling.stepN = basicTiling_.stepN;
    iterbatchTilingData_->matmulTiling.stepKa = basicTiling_.stepKa;
    iterbatchTilingData_->matmulTiling.stepKb = basicTiling_.stepKb;
    iterbatchTilingData_->matmulTiling.isBias = matmulInfoPtr_->hasBias;
    iterbatchTilingData_->matmulTiling.iterateOrder = basicTiling_.iterateOrder;
    iterbatchTilingData_->matmulTiling.dbL0A = 2; // db switch, 1: off, 2: on
    iterbatchTilingData_->matmulTiling.dbL0B = 2; // db switch, 1: off, 2: on
    iterbatchTilingData_->matmulTiling.dbL0C = basicTiling_.dbL0c;
    if (basicTiling_.iterBatch > 0U) {
        // additional tiling for bmm
        iterbatchTilingData_->matmulTiling.BatchNum = basicTiling_.iterBatch;
        iterbatchTilingData_->matmulTiling.ALayoutInfoB = basicTiling_.iterBatch;
        iterbatchTilingData_->matmulTiling.ALayoutInfoS = basicTiling_.singleCoreM;
        iterbatchTilingData_->matmulTiling.ALayoutInfoN = 1;
        iterbatchTilingData_->matmulTiling.ALayoutInfoG = 1;
        iterbatchTilingData_->matmulTiling.ALayoutInfoD = basicTiling_.singleCoreK;
        iterbatchTilingData_->matmulTiling.BLayoutInfoB = basicTiling_.iterBatch;
        iterbatchTilingData_->matmulTiling.BLayoutInfoS = basicTiling_.singleCoreN;
        iterbatchTilingData_->matmulTiling.BLayoutInfoN = 1;
        iterbatchTilingData_->matmulTiling.BLayoutInfoG = 1;
        iterbatchTilingData_->matmulTiling.BLayoutInfoD = basicTiling_.singleCoreK;
        iterbatchTilingData_->matmulTiling.CLayoutInfoB = basicTiling_.iterBatch;
        iterbatchTilingData_->matmulTiling.CLayoutInfoS1 = basicTiling_.singleCoreM;
        iterbatchTilingData_->matmulTiling.CLayoutInfoN = 1;
        iterbatchTilingData_->matmulTiling.CLayoutInfoG = 1;
        iterbatchTilingData_->matmulTiling.CLayoutInfoS2 = basicTiling_.singleCoreN;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus WeightQuantBatchMatmulV2IterbatchTiling::PostTiling()
{
    OP_LOGD(opName_, "final tiling data size: %zu", iterbatchTilingDataSize_);
    OP_TILING_CHECK(iterbatchTilingDataSize_ % sizeof(uint64_t) != 0,
                    OP_LOGE(opName_, "tiling data size[%zu] is not aligned to 8", iterbatchTilingDataSize_),
                    return ge::GRAPH_FAILED);
    errno_t ret = memcpy_s(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity(),
                           static_cast<const void*>(iterbatchTilingData_.get()), iterbatchTilingDataSize_);
    if (ret != EOK) {
        OP_LOGE(context_->GetNodeName(), "memcpy_s failed, ret=%d", ret);
        return ge::GRAPH_FAILED;
    }
    context_->SetBlockDim(basicTiling_.usedCoreNum);
    context_->GetRawTilingData()->SetDataSize(iterbatchTilingDataSize_);
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
    basicTiling_.baseM = !matmulInfoPtr_->transA ?
                             ops::CeilAlign(static_cast<uint64_t>(basicTiling_.baseM), CUBE_BLOCK) :
                             ops::CeilAlign(static_cast<uint64_t>(basicTiling_.baseM),
                                            GetShapeWithDataType(L1_ALIGN_SIZE, matmulInfoPtr_->aDtype));
    basicTiling_.baseN = std::min(matmulInfoPtr_->nSize, static_cast<uint64_t>(BASIC_BLOCK_SIZE_128));
    basicTiling_.baseN = matmulInfoPtr_->transB ?
                             ops::CeilAlign(static_cast<uint64_t>(basicTiling_.baseN), CUBE_BLOCK) :
                             ops::CeilAlign(static_cast<uint64_t>(basicTiling_.baseN),
                                            GetShapeWithDataType(L1_ALIGN_SIZE, matmulInfoPtr_->bDtype));

    uint64_t minBaseK = std::min(
        std::min(GetShapeWithDataType(static_cast<uint64_t>(BASIC_BLOCK_SIZE_128), matmulInfoPtr_->aDtype),
                 GetShapeWithDataType(static_cast<uint64_t>(BASIC_BLOCK_SIZE_128), matmulInfoPtr_->bDtype)),
        matmulInfoPtr_->kSize);
    uint64_t maxAlignSize = std::max(
        static_cast<uint64_t>(GetShapeWithDataType(CUBE_REDUCE_BLOCK, matmulInfoPtr_->aDtype)),
        static_cast<uint64_t>(GetShapeWithDataType(CUBE_REDUCE_BLOCK, matmulInfoPtr_->bDtype)));
    basicTiling_.baseK = ops::CeilAlign(minBaseK, maxAlignSize);

    basicTiling_.stepM = 1;
    basicTiling_.stepN = 1;

    basicTiling_.iterateOrder = 0;
    basicTiling_.dbL0c = ((basicTiling_.baseM * basicTiling_.baseN * DATA_SIZE_L0C * DB_SIZE <=
                           compileInfoPtr_->l0cSize) &&
                          CheckAntiQuantScale(basicTiling_.baseN, DB_SIZE)) ?
                             DB_SIZE :
                             1;
    CalL1TilingDepth(leftL1Size_);
}

uint32_t WeightQuantBatchMatmulV2IterbatchTiling::CalcIterBatch()
{
    uint64_t biasDtypeSize = ge::GetSizeByDataType(matmulInfoPtr_->biasDtype);
    uint64_t scaleDtypeSize = ge::GetSizeByDataType(matmulInfoPtr_->antiQuantScaleDtype);
    uint64_t totalL1Size = compileInfoPtr_->l1Size;
    uint64_t singleCoreBiasSize = matmulInfoPtr_->hasBias ? basicTiling_.baseN * biasDtypeSize : 0;
    uint64_t singleCoreScaleSize = matmulInfoPtr_->antiQuantType == QuantType::PER_CHANNEL ?
                                       basicTiling_.baseN * scaleDtypeSize :
                                       0;
    leftL1Size_ = totalL1Size - singleCoreBiasSize - singleCoreScaleSize;
    // get align m,k,n value
    uint64_t baseMAlignNum = matmulInfoPtr_->transA ? GetShapeWithDataType(L1_ALIGN_SIZE, matmulInfoPtr_->aDtype) :
                                                      CUBE_BLOCK;
    uint64_t baseNAlignNum = matmulInfoPtr_->transB ? CUBE_BLOCK :
                                                      GetShapeWithDataType(L1_ALIGN_SIZE, matmulInfoPtr_->bDtype);
    uint64_t baseKAlignNum = (matmulInfoPtr_->transA && !matmulInfoPtr_->transB) ?
                                 CUBE_BLOCK :
                                 GetShapeWithDataType(L1_ALIGN_SIZE, matmulInfoPtr_->aDtype);
    uint64_t alignMValue = ops::CeilAlign(static_cast<uint64_t>(matmulInfoPtr_->mSize), baseMAlignNum);
    uint64_t alignNValue = ops::CeilAlign(static_cast<uint64_t>(matmulInfoPtr_->nSize), baseNAlignNum);
    uint64_t alignKValue = ops::CeilAlign(static_cast<uint64_t>(matmulInfoPtr_->kSize), baseKAlignNum);

    uint32_t iterBatch = ops::FloorDiv(leftL1Size_,
                                       GetSizeWithDataType(alignMValue * alignKValue, matmulInfoPtr_->aDtype) +
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
    OP_TILING_CHECK(!matmulInfoPtr_->transA && matmulInfoPtr_->batchWeight == 1UL,
                    OP_LOGI(opName_, "When transA = False and batchB = 1, batchA can be co-axial with M"),
                    return false);

    OP_TILING_CHECK(matmulInfoPtr_->batchX == 1UL && matmulInfoPtr_->batchWeight == 1UL,
                    OP_LOGI(opName_,
                            "the iter batch template doesn't support tensor X and Weight batch size is 1."
                            "batchX: %lu, batchWeight: %lu",
                            matmulInfoPtr_->batchX, matmulInfoPtr_->batchWeight),
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

    OP_TILING_CHECK(broadcastNum != 0UL && broadcastNum != 1UL,
                    OP_LOGI(opName_,
                            "the multi-batch optimization currently only supports one batch axis being broadcasted."
                            "The number of axis need to broadcast is %lu",
                            broadcastNum),
                    return false);
    uint32_t iterBatch = CalcIterBatch();
    OP_TILING_CHECK(iterBatch <= 1UL, OP_LOGI(opName_, "the iter batch should be greater than 1 but %u", iterBatch),
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
    // 开启NBatchOut需满足如下条件：
    // 1. baseM >= singleCoreM && baseN >= singleCoreN
    // 2. N/16向上取整应为偶数
    bool enableBatchOut = basicTiling_.baseM >= basicTiling_.singleCoreM &&
                          basicTiling_.baseN >= basicTiling_.singleCoreN &&
                          static_cast<uint64_t>(ops::CeilDiv(matmulInfoPtr_->nSize, CUBE_BLOCK)) % 2 == 0;
    uint64_t subAlgorithm = enableBatchOut ?
                                static_cast<uint64_t>(OptimizationAlgorithmSubCategory::ITERATE_BATCH) :
                                static_cast<uint64_t>(OptimizationAlgorithmSubCategory::ITERATE_BATCH_NO_BATCH_OUT);
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
    uint64_t tilingKey = GET_TPL_TILING_KEY(socVersionType, subSocVersionType, antiquantScenario, algorithm,
                                            subAlgorithm, templateCustom, apiConstexpr, transA, transB, antiquantType,
                                            quantType, hasAntiquantOffset, hasBias, isBiasFp32, isWeightNz);
    return tilingKey;
}
REGISTER_TILING_TEMPLATE("WeightQuantBatchMatmulV2", WeightQuantBatchMatmulV2IterbatchTiling, ITERBATCH_PRIORITY);
} // namespace weight_quant_batch_matmul_v2
} // namespace optiling

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
 * \file weight_quant_batch_matmul_v2_adaptive_sliding_window_for_mix_tiling.cpp
 * \brief
 */

#include "weight_quant_batch_matmul_v2_adaptive_sliding_window_mix_tiling.h"
#include "common/inc/op_host/op_const_def_nn.h"
#include "matmul/common/op_host/math_util_nn.h"
#include "tiling/platform/platform_ascendc.h"
#include "matmul/weight_quant_batch_matmul_v2/op_kernel/arch35/weight_quant_batch_matmul_v2_arch35_tiling_data.h"
#include "../../../op_kernel/arch35/weight_quant_batch_matmul_v2_arch35_tiling_key.h"

using namespace platform_ascendc;

namespace optiling {
namespace weight_quant_batch_matmul_v2 {

constexpr uint64_t CUBE_BLOCK = 16;
constexpr uint64_t L1_ALIGN_SIZE = 32;
constexpr uint64_t CUBE_REDUCE_BLOCK = 32;
constexpr uint64_t CUBE_BOUND_BASE_K = 64UL;
constexpr uint64_t UB_QUADRUPLE_LOWER_THRESHOLD = 4096UL;
constexpr uint64_t MEGABYTE = 1024 * 1024;
constexpr uint32_t BASIC_BLOCK_SIZE_128 = 128;
constexpr uint32_t BASIC_BLOCK_SIZE_256 = 256;
constexpr uint64_t BASIC_BLOCK_SIZE_512 = 512;
constexpr uint32_t NUM_HALF = 2;
constexpr uint32_t DB_SIZE = 2;
constexpr uint32_t SLIDING_WINOW_SHORTER_EDGE = 4;
constexpr uint32_t SLIDING_WINOW_LONGER_EDGE = 16;
constexpr uint32_t DATA_SIZE_L0C = 4;
constexpr int32_t ASW_MIX_PRIORITY = 12;
constexpr uint32_t MAX_PERF_STEPKB = 4;
// baseK典型值为128,baseN最小取80可以使BL1单次载入量大于等于40K,保证MTE2带宽
constexpr uint64_t MIN_BASIC_BLOCK_N = 80UL;
constexpr uint64_t HALF_L0AB_SIZE = 32768UL;
constexpr uint64_t BASE_N_ALIGN_FOR_PERGROUP = 64UL;
constexpr uint64_t BASE_K_ALIGN_FOR_MX = 64UL;
constexpr uint64_t MAX_BASE_MN_SIZE = 32768UL;
constexpr uint64_t BYTE_BITS = 8UL;
constexpr uint64_t L1_SIZE = 524288UL;
constexpr uint64_t MIN_RESPLIT_N_SIZE = 4096UL;
constexpr uint64_t DOUBLE_BUFFER_NUM = 2UL;
constexpr uint64_t QUADRUPLE_BUFFER_NUM = 4UL;
constexpr uint64_t PER_GROUP_WEIGHT_MIN_TILE_SIZE = 16 * 1024UL;
constexpr uint64_t MX_GROUP_SIZE = 32UL;
constexpr uint64_t HIF4_GROUP_SIZE = 64UL;
constexpr uint64_t B4_C0_SIZE = 64UL;

ge::graphStatus WeightQuantBatchMatmulV2TilingASWMix::DoOpTiling()
{
    OP_LOGD(opName_, "DoOpTiling of adaptive sliding window tiling strategy.");
    OP_TILING_CHECK(InstantiateTilingData() == ge::GRAPH_FAILED,
                    VECTOR_INNER_ERR_REPORT_TILIING(opName_, "unable to get pointer of tiling data"),
                    return ge::GRAPH_FAILED);

    if (!AnalyseSlidingWinInfo()) {
        OP_LOGE(opName_, "DoOpTiling fail");
        return ge::GRAPH_FAILED;
    }
    CalL1Tiling();
    SetTilingData();
    return ge::GRAPH_SUCCESS;
}

bool WeightQuantBatchMatmulV2TilingASWMix::IsCapable()
{
    if (matmulInfoPtr_->antiQuantType == QuantType::PER_GROUP) {
        OP_TILING_CHECK(
            (matmulInfoPtr_->bDtype == ge::DT_INT4 || matmulInfoPtr_->bDtype == ge::DT_FLOAT4_E2M1 ||
             matmulInfoPtr_->bDtype == ge::DT_INT8) &&
                !matmulInfoPtr_->transB,
            OP_LOGI(
                opName_,
                "When weight's dtype is INT4/FLOAT4_E2M1/INT8 , Optimized per-group template only support weight is "
                "transposed."),
            return false);
    }
    if (matmulInfoPtr_->antiQuantType == QuantType::PER_CHANNEL ||
        matmulInfoPtr_->antiQuantType == QuantType::PER_TENSOR) {
        OP_TILING_CHECK(matmulInfoPtr_->antiQuantScaleDtype != ge::DT_UINT64,
                        OP_LOGI(opName_,
                                "Optimized per-channel/per-tensor template only support antiQuantScaleDtype is uint64, "
                                "current dtype is %s",
                                ge::TypeUtils::DataTypeToAscendString(matmulInfoPtr_->antiQuantScaleDtype).GetString()),
                        return false);
    }
    return true;
}

ge::graphStatus WeightQuantBatchMatmulV2TilingASWMix::InstantiateTilingData()
{
    if (tilingData_ == nullptr) {
        try {
            tilingData_ = std::make_unique<wqbmmv2_tiling::WeightQuantBatchMatmulV2ASWTilingData>();
        } catch (std::bad_alloc&) {
            OP_LOGE(opName_, "tiling data memory allocation failed");
            return ge::GRAPH_FAILED;
        }
    }
    OP_TILING_CHECK(tilingData_ == nullptr,
                    VECTOR_INNER_ERR_REPORT_TILIING(opName_, "failed to instantiate tilingData"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(context_->GetRawTilingData()->GetCapacity() < tilingDataSize_,
                    VECTOR_INNER_ERR_REPORT_TILIING(opName_, "tiling data capacity %zu < actual tiling data size %zu",
                                                    context_->GetRawTilingData()->GetCapacity(), tilingDataSize_),
                    return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

bool WeightQuantBatchMatmulV2TilingASWMix::AnalyseSlidingWinInfo()
{
    if (!CalcBasicBlock()) {
        OP_LOGE(opName_, "inappropriate basicBlock");
        return false;
    }
    adaptiveWin_.mBlockCnt = ops::CeilDiv(matmulInfoPtr_->mSize, adaptiveWin_.baseM);
    adaptiveWin_.nBlockCnt = ops::CeilDiv(matmulInfoPtr_->nSize, adaptiveWin_.baseN);
    adaptiveWin_.totalBlockCnt = adaptiveWin_.mBlockCnt * adaptiveWin_.nBlockCnt;
    adaptiveWin_.mTail = matmulInfoPtr_->mSize - (adaptiveWin_.mBlockCnt - 1) * adaptiveWin_.baseM;
    adaptiveWin_.nTail = matmulInfoPtr_->nSize - (adaptiveWin_.nBlockCnt - 1) * adaptiveWin_.baseN;
    adaptiveWin_.totalWinCnt = ops::CeilDiv(adaptiveWin_.totalBlockCnt, static_cast<uint64_t>(compileInfoPtr_->aicNum));
    if (adaptiveWin_.mBlockCnt <= adaptiveWin_.nBlockCnt) {
        adaptiveWin_.singleWinM = SLIDING_WINOW_SHORTER_EDGE;
        adaptiveWin_.singleWinN = SLIDING_WINOW_LONGER_EDGE;
    } else {
        adaptiveWin_.singleWinM = SLIDING_WINOW_LONGER_EDGE;
        adaptiveWin_.singleWinN = SLIDING_WINOW_SHORTER_EDGE;
    }
    if (adaptiveWin_.mBlockCnt % adaptiveWin_.singleWinM == 0) {
        adaptiveWin_.mainRow = adaptiveWin_.mBlockCnt / adaptiveWin_.singleWinM;
    } else {
        adaptiveWin_.mainRow = adaptiveWin_.mBlockCnt < adaptiveWin_.singleWinM ?
                                   0 :
                                   adaptiveWin_.mBlockCnt / adaptiveWin_.singleWinM - 1;
    }
    adaptiveWin_.tailWinBlockCnt = (adaptiveWin_.totalBlockCnt) % compileInfoPtr_->aicNum;
    if (adaptiveWin_.useTailWinLogic) {
        CalcTailBasicBlock();
    }
    return true;
}

bool WeightQuantBatchMatmulV2TilingASWMix::CheckL1QuadrupleBufferSize(uint64_t baseM, uint64_t baseN,
                                                                      uint64_t baseK) const
{
    uint64_t kL1Size = baseK * MAX_PERF_STEPKB * QUADRUPLE_BUFFER_NUM;
    uint64_t l1LoadSize = (baseM * kL1Size * basicBlockParam_.aDtypeBits +
                           baseN * kL1Size * basicBlockParam_.bDtypeBits) /
                          BYTE_BITS;
    // A16Fp4-PerGroup需要借用MMAD_MX实现，L1上多一个全1的Mx的Scale
    if (matmulInfoPtr_->antiQuantType == QuantType::PER_GROUP && matmulInfoPtr_->bDtype == ge::DT_FLOAT4_E2M1) {
        l1LoadSize += baseN * (baseK / MX_GROUP_SIZE) * MAX_PERF_STEPKB; // Mx的scale是FP8_E8M0
    }
    if (l1LoadSize > compileInfoPtr_->l1Size) {
        return false;
    }

    // 2 buff时K方向L0每次载入baseK，4 buff时需要载入stepK * baseK，L0A/B上开Double Buffer
    uint64_t kL0Size = baseK * MAX_PERF_STEPKB * DOUBLE_BUFFER_NUM;
    if (baseM * kL0Size * basicBlockParam_.aDtypeBits / BYTE_BITS > compileInfoPtr_->l0aSize) {
        return false;
    }

    if (baseN * kL0Size * basicBlockParam_.bDtypeBits / BYTE_BITS > compileInfoPtr_->l0bSize) {
        return false;
    }

    // 4：L0C sizeoOf(float32)
    if (4 * QUADRUPLE_BUFFER_NUM * baseM * baseN > compileInfoPtr_->l0cSize) {
        return false;
    }

    return true;
}

bool WeightQuantBatchMatmulV2TilingASWMix::OpenL1QuadrupleBufferCond() const
{
    uint64_t actualNSize = std::min(basicBlockParam_.nSize, adaptiveWin_.baseN);
    uint64_t bL1Size = actualNSize * adaptiveWin_.baseK * MAX_PERF_STEPKB * basicBlockParam_.bDtypeBits / BYTE_BITS;
    if (bL1Size <= PER_GROUP_WEIGHT_MIN_TILE_SIZE &&
        CheckL1QuadrupleBufferSize(adaptiveWin_.baseM, adaptiveWin_.baseN, adaptiveWin_.baseK)) {
        return true;
    }

    return false;
}

bool WeightQuantBatchMatmulV2TilingASWMix::OpenUBQuadrupleBufferCond(uint64_t baseM, uint64_t baseN,
                                                                     uint64_t baseK) const
{
    return baseM * baseN >= UB_QUADRUPLE_LOWER_THRESHOLD && baseK <= CUBE_BOUND_BASE_K &&
           // 因 scale 存放于 2 个 ub bank 中，baseN 限制 256
           baseN <= BASIC_BLOCK_SIZE_256;
}

bool WeightQuantBatchMatmulV2TilingASWMix::CalculateBasicBlockForPergroup()
{
    optionalResults_.clear();
    InitBasicBlockParam();
    CalculateBasicBlockTiling();
    if (!optionalResults_.empty()) {
        stable_sort(optionalResults_.begin(), optionalResults_.end());
        basicBlockParam_ = optionalResults_.front();
        adaptiveWin_.baseM = basicBlockParam_.baseM;
        adaptiveWin_.baseN = basicBlockParam_.baseN;
        adaptiveWin_.baseK = basicBlockParam_.baseK;
        return true;
    }
    return false;
}

void WeightQuantBatchMatmulV2TilingASWMix::GetBasicBlockSolution()
{
    basicBlockParam_.baseK = std::min(static_cast<uint64_t>(BASIC_BLOCK_SIZE_128), matmulInfoPtr_->groupSize);
    for (uint64_t baseM = 16; baseM <= basicBlockParam_.singleM; baseM += CUBE_BLOCK) {
        uint64_t baseNMax = std::min(BASIC_BLOCK_SIZE_512,
                                     ops::CeilAlign(MAX_BASE_MN_SIZE / baseM, BASE_N_ALIGN_FOR_PERGROUP));
        baseNMax = std::min(baseNMax, basicBlockParam_.singleN);
        for (uint64_t baseN = baseNMax; baseN >= BASE_N_ALIGN_FOR_PERGROUP; baseN -= BASE_N_ALIGN_FOR_PERGROUP) {
            if (baseN * baseM > MAX_BASE_MN_SIZE || baseM > BASIC_BLOCK_SIZE_128) {
                continue;
            }
            if (!CheckBufferSize(baseM, baseN, basicBlockParam_.baseK)) {
                continue;
            }
            basicBlockParam_.baseM = baseM;
            basicBlockParam_.baseN = baseN;
            GetMte2Cost();
            optionalResults_.push_back(basicBlockParam_);
        }
    }
}

void WeightQuantBatchMatmulV2TilingASWMix::GetMte2Cost()
{
    bool ubQuadruple = OpenUBQuadrupleBufferCond(basicBlockParam_.baseM, basicBlockParam_.baseN,
                                                 basicBlockParam_.baseK);
    double ubQuadrupleRatio = !ubQuadruple ? 1 : matmulInfoPtr_->mSize > 128 ? 0.75 : 0.9; // 经验值，鼓励开启4buffer
    double l2HbmBWRatio = 1.8; // 1.8:L2带宽和HBM带宽比值
    double mte2Cost = basicBlockParam_.mSize *
                          (basicBlockParam_.nDim * ops::CeilDiv(basicBlockParam_.singleN, basicBlockParam_.baseN) - 1) *
                          basicBlockParam_.aDtypeBits +
                      basicBlockParam_.mSize * basicBlockParam_.aDtypeBits * l2HbmBWRatio +
                      basicBlockParam_.nSize *
                          (basicBlockParam_.mDim * ops::CeilDiv(basicBlockParam_.singleM, basicBlockParam_.baseM) - 1) *
                          basicBlockParam_.bDtypeBits +
                      basicBlockParam_.nSize * basicBlockParam_.bDtypeBits * l2HbmBWRatio;
    // 降低总核数较少时被选中的概率
    basicBlockParam_.mte2Cost = mte2Cost / BYTE_BITS / BWUtilization(basicBlockParam_.mDim, basicBlockParam_.nDim) *
                                ubQuadrupleRatio;
}

bool WeightQuantBatchMatmulV2TilingASWMix::CheckBufferSize(uint64_t baseM, uint64_t baseN, uint64_t baseK)
{
    bool ubQuadruple = OpenUBQuadrupleBufferCond(baseM, baseN, baseK);
    // Weight非fp4时，L1上只载入A和B矩阵，在固定setpK=4时，不超L0空间就可以保证不超L1空间
    if (matmulInfoPtr_->bDtype == ge::DT_FLOAT4_E2M1) {
        if (baseM * baseK * basicBlockParam_.aDtypeBits * MAX_PERF_STEPKB * DB_SIZE / BYTE_BITS +
                baseN * baseK * basicBlockParam_.bDtypeBits * MAX_PERF_STEPKB * DB_SIZE / BYTE_BITS +
                baseN * baseK / MX_GROUP_SIZE >
            L1_SIZE) {
            return false;
        }
    }
    if (baseM * baseK * basicBlockParam_.aDtypeBits / BYTE_BITS > HALF_L0AB_SIZE) {
        return false;
    }
    if (baseN * baseK * basicBlockParam_.bDtypeBits / BYTE_BITS > HALF_L0AB_SIZE) {
        return false;
    }
    if (ubQuadruple) {
        // 16：4 个 bank 的空间 * sizeof(float32)；2: 开启 4 buffer 需要 2 倍的空间；第二项为outBuf，2：sizeof(yType)
        return 16 * baseM * baseN * 2 + 2 * baseM * baseN <= compileInfoPtr_->ubSize;
    } else {
        // 第一个16: 4个bank的空间 * sizeof(float32)；第二个16：db(2) * 4(stepK) *
        // 2(sizeof(scaletype))；最后的2：sizeof(yType)
        uint64_t baseSize = 16 * baseM * baseN + 16 * baseN + 2 * baseM * baseN;
        uint64_t extraSize = matmulInfoPtr_->hasBias ? basicBlockParam_.biasDtypeBits / BYTE_BITS * baseN : 0;
        if (baseSize + extraSize > compileInfoPtr_->ubSize) {
            return false;
        }
    }
    return true;
}

void WeightQuantBatchMatmulV2TilingASWMix::CalculateBasicBlockTiling()
{
    uint64_t mDimMax = std::min(ops::CeilDiv(basicBlockParam_.mSize, CUBE_BLOCK),
                                static_cast<uint64_t>(compileInfoPtr_->aicNum));
    for (uint64_t mDim = mDimMax; mDim >= 1; mDim--) {
        uint64_t nDimMax = std::min(ops::CeilDiv(basicBlockParam_.nSize, CUBE_BLOCK), compileInfoPtr_->aicNum / mDim);
        for (uint64_t nDim = nDimMax; nDim >= 1; nDim--) {
            if ((!optionalResults_.empty()) &&
                // 0.8：当已有可选解时剪枝分核情况小于0.8倍总核数的解
                mDim * nDim < 0.8 * compileInfoPtr_->aicNum) {
                continue;
            }
            basicBlockParam_.mDim = mDim;
            basicBlockParam_.nDim = nDim;
            basicBlockParam_.singleM = ops::CeilAlign(ops::CeilDiv(basicBlockParam_.mSize, mDim), CUBE_BLOCK);
            basicBlockParam_.singleN = ops::CeilAlign(ops::CeilDiv(basicBlockParam_.nSize, nDim),
                                                      BASE_N_ALIGN_FOR_PERGROUP);
            uint64_t mDimFixed = ops::CeilDiv(basicBlockParam_.mSize, basicBlockParam_.singleM);
            uint64_t nDimFixed = ops::CeilDiv(basicBlockParam_.nSize, basicBlockParam_.singleN);
            if (mDimFixed != mDim || nDimFixed != nDim) {
                continue;
            }
            GetBasicBlockSolution();
        }
    }
}

void WeightQuantBatchMatmulV2TilingASWMix::InitBasicBlockParam()
{
    basicBlockParam_.mSize = matmulInfoPtr_->mSize;
    basicBlockParam_.nSize = matmulInfoPtr_->nSize;
    basicBlockParam_.kSize = matmulInfoPtr_->kSize;
    basicBlockParam_.singleM = 0;
    basicBlockParam_.singleN = 0;
    basicBlockParam_.mDim = 0;
    basicBlockParam_.nDim = 0;
    basicBlockParam_.baseM = 0;
    basicBlockParam_.baseN = 0;
    basicBlockParam_.baseK = 0;
    basicBlockParam_.aDtypeBits = GetDtypeBits(matmulInfoPtr_->aDtype);
    basicBlockParam_.bDtypeBits = GetDtypeBits(matmulInfoPtr_->bDtype);
    basicBlockParam_.biasDtypeBits = GetDtypeBits(matmulInfoPtr_->biasDtype);
}

void WeightQuantBatchMatmulV2TilingASWMix::CalculateBaseNForPergroup()
{
    uint64_t baseNAlign = ops::CeilAlign(matmulInfoPtr_->nSize, BASE_N_ALIGN_FOR_PERGROUP);
    uint64_t baseNMax = std::min(baseNAlign, BASIC_BLOCK_SIZE_512);
    // baseN需要与64对齐
    for (uint64_t baseN = baseNMax; baseN > 0; baseN -= 64) {
        if (!CheckBufferSize(adaptiveWin_.baseM, baseN, adaptiveWin_.baseK)) {
            continue;
        }

        adaptiveWin_.baseN = baseN;
        return;
    }
}

void WeightQuantBatchMatmulV2TilingASWMix::CalculateBaseN()
{
    uint64_t mBlockCnt = ops::CeilDiv(matmulInfoPtr_->mSize, adaptiveWin_.baseM);
    uint64_t baseNAlignSize = (!matmulInfoPtr_->transB) && (matmulInfoPtr_->bFormat == ge::FORMAT_FRACTAL_NZ) ?
                                  GetShapeWithDataType(L1_ALIGN_SIZE, matmulInfoPtr_->bDtype) :
                                  CUBE_BLOCK;
    uint64_t baseNMax = // L1空间限制 & L0C空间限制
        adaptiveWin_.baseM == BASIC_BLOCK_SIZE_128 ? BASIC_BLOCK_SIZE_256 - baseNAlignSize : BASIC_BLOCK_SIZE_256;
    uint64_t edgeLengthM = matmulInfoPtr_->mSize > matmulInfoPtr_->nSize ? SLIDING_WINOW_LONGER_EDGE :
                                                                           SLIDING_WINOW_SHORTER_EDGE;
    adaptiveWin_.baseN = std::max(
        MIN_BASIC_BLOCK_N,
        ops::CeilDiv(matmulInfoPtr_->nSize,
                     static_cast<uint64_t>(compileInfoPtr_->aicNum / std::min(mBlockCnt, edgeLengthM))));
    adaptiveWin_.baseN = std::min(matmulInfoPtr_->nSize, adaptiveWin_.baseN);
    if (adaptiveWin_.baseN > baseNMax) {
        if (adaptiveWin_.baseN % baseNMax < baseNMax / 2) { // 取模小于baseN的1/2说明最后会有一个很不友好的尾块
            adaptiveWin_.baseN = adaptiveWin_.baseN / (ops::CeilDiv(adaptiveWin_.baseN, baseNMax));
        } else {
            adaptiveWin_.baseN = baseNMax;
        }
    }

    adaptiveWin_.baseN = ops::CeilAlign(adaptiveWin_.baseN, baseNAlignSize);

    uint64_t nBlockCnt = ops::CeilDiv(matmulInfoPtr_->nSize, adaptiveWin_.baseN);

    uint64_t tailBlockCount = (mBlockCnt * nBlockCnt) % compileInfoPtr_->aicNum;
    // 小于等于aicNum/2时可以走原尾块切分; 大于aicNum/2时，重新切分基本块，寻找更加负载均衡的解.
    if (tailBlockCount > compileInfoPtr_->aicNum / 2) {
        uint64_t tmpTailCount = tailBlockCount;
        uint64_t tmpBaseN = adaptiveWin_.baseN;
        // 为避免基本块过小，只遍历到原大小的1/2
        while (tmpTailCount >= tailBlockCount && tmpBaseN >= adaptiveWin_.baseN / 2 &&
               tmpTailCount > compileInfoPtr_->aicNum / 2) { // 尾块大aicNum/2是尾块基本块切分的准入条件
            tmpBaseN = tmpBaseN - baseNAlignSize;
            nBlockCnt = ops::CeilDiv(matmulInfoPtr_->nSize, tmpBaseN);
            tmpTailCount = (mBlockCnt * nBlockCnt) % compileInfoPtr_->aicNum;
        }
        adaptiveWin_.baseN = tmpBaseN + baseNAlignSize; // 循环中多减了一次
    }
}

bool WeightQuantBatchMatmulV2TilingASWMix::CalcBasicBlock()
{
    adaptiveWin_.baseM = std::min(matmulInfoPtr_->mSize, static_cast<uint64_t>(BASIC_BLOCK_SIZE_128));
    uint64_t mBlockCnt = ops::CeilDiv(matmulInfoPtr_->mSize, adaptiveWin_.baseM);
    adaptiveWin_.baseM = ops::CeilDiv(matmulInfoPtr_->mSize, mBlockCnt);
    adaptiveWin_.baseM = !matmulInfoPtr_->transA ?
                             ops::CeilAlign(adaptiveWin_.baseM, CUBE_BLOCK) :
                             ops::CeilAlign(adaptiveWin_.baseM,
                                            GetShapeWithDataType(L1_ALIGN_SIZE, matmulInfoPtr_->aDtype));
    bool isPergroup = matmulInfoPtr_->antiQuantType == QuantType::PER_GROUP;
    bool isA16MxFp4 = (matmulInfoPtr_->aDtype == ge::DT_FLOAT16 || matmulInfoPtr_->aDtype == ge::DT_BF16) &&
                      matmulInfoPtr_->bDtype == ge::DT_FLOAT4_E2M1 && matmulInfoPtr_->antiQuantType == QuantType::MX;
    if (isPergroup) {
        if (!CalculateBasicBlockForPergroup()) {
            adaptiveWin_.useTailWinLogic = false;
            adaptiveWin_.baseK = std::min(static_cast<uint64_t>(BASIC_BLOCK_SIZE_128), matmulInfoPtr_->groupSize);
            CalculateBaseNForPergroup();
        }
    } else {
        CalculateBaseN();
        ReSplitBasicBlock();
        // A16MxFP4场景下针对使用核数小于50核的做重切分
        if (isA16MxFp4) {
            ReSplitBasicBlockForCoreNum();
        }

        adaptiveWin_.baseK = CalcBasicBlockBaseK(adaptiveWin_.baseM, adaptiveWin_.baseN);
    }
    adaptiveWin_.enablePreAlloc = !ClosePreAllocCond();
    return true;
}

void WeightQuantBatchMatmulV2TilingASWMix::ReSplitBasicBlockForCoreNum()
{
    uint64_t mBlockCnt = ops::CeilDiv(matmulInfoPtr_->mSize, adaptiveWin_.baseM);
    uint64_t nBlockCnt = ops::CeilDiv(matmulInfoPtr_->nSize, adaptiveWin_.baseN);

    constexpr uint64_t baseMMin = 16;
    constexpr uint64_t baseNMin = 32;

    // 使用的核数小于一半时（aicNum / 2），做重切分
    while (nBlockCnt * mBlockCnt < compileInfoPtr_->aicNum / 2 &&
           (adaptiveWin_.baseM > baseMMin || adaptiveWin_.baseN > baseNMin)) {
        if (adaptiveWin_.baseM > baseMMin) {
            // 使用的核数小于一半时， baseM除以2
            adaptiveWin_.baseM = ops::CeilAlign(adaptiveWin_.baseM / 2, static_cast<uint64_t>(baseMMin));
            mBlockCnt = ops::CeilDiv(matmulInfoPtr_->mSize, adaptiveWin_.baseM);
            continue;
        } else if (adaptiveWin_.baseN > baseNMin) {
            // 使用的核数小于一半时， baseN除以2
            adaptiveWin_.baseN = ops::CeilAlign(adaptiveWin_.baseN / 2, static_cast<uint64_t>(baseNMin));
            nBlockCnt = ops::CeilDiv(matmulInfoPtr_->nSize, adaptiveWin_.baseN);
            continue;
        }
        break;
    }
}

double WeightQuantBatchMatmulV2TilingASWMix::BWUtilization(const uint64_t mDim, const uint64_t nDim) const
{
    // 当前简单认为带宽利用率和实际使用的核数成正比
    return (mDim * nDim) / static_cast<double>(compileInfoPtr_->aicNum);
}

uint64_t WeightQuantBatchMatmulV2TilingASWMix::CalcBasicBlockBaseK(uint64_t baseM, uint64_t baseN) const
{
    uint64_t aByteSize = ge::GetSizeByDataType(matmulInfoPtr_->aDtype);
    uint64_t scaleByteSize = ge::GetSizeByDataType(matmulInfoPtr_->antiQuantScaleDtype);
    uint64_t baseNAlign = ops::CeilAlign(
        baseN, matmulInfoPtr_->transB ? CUBE_BLOCK : GetShapeWithDataType(L1_ALIGN_SIZE, matmulInfoPtr_->bDtype));
    // baseK受L0B空间大小的限制，2：4 bit 位宽为 0.5，除以 0.5 即乘 2
    uint64_t baseK = Is4BitDtype(matmulInfoPtr_->bDtype) ?
                         (HALF_L0AB_SIZE / baseNAlign * 2) :
                         (HALF_L0AB_SIZE / baseNAlign / ge::GetSizeByDataType(matmulInfoPtr_->bDtype));

    baseK = std::min(baseK, HALF_L0AB_SIZE / ge::GetSizeByDataType(matmulInfoPtr_->aDtype) / baseM);

    baseK = ops::FloorAlign(baseK, GetShapeWithDataType(L1_ALIGN_SIZE, matmulInfoPtr_->bDtype));
    // 当AL1数据量较大(大于40K)时，优先保证512B对齐，否则优先保证数据量; 40K/4(stepK)/2(bytes)=5120
    if (baseM * baseK > 5120) {
        // baseK 512/sizeof(b16)/stepK=64对齐时，可以保证AL1内轴512B对齐
        baseK = std::max(64UL, ops::FloorAlign(baseK, 64UL));
    }

    // baseN方向L1大小：baseN * stepK * DB * sizeof(bDtype)，2：4 bit 位宽为 0.5，乘 0.5 即除以 2
    uint64_t baseNL1Size = Is4BitDtype(matmulInfoPtr_->bDtype) ? (baseNAlign * MAX_PERF_STEPKB * DB_SIZE / 2) :
                                                                 (baseNAlign * MAX_PERF_STEPKB * DB_SIZE);
    // baseK受L1空间大小的限制
    uint64_t maxBaseKL1Limited = 0;
    if (matmulInfoPtr_->antiQuantType == QuantType::PER_CHANNEL) {
        // (L1_SIZE - scale空间_PerChannel(db*sizeof(uint64)*baseN)) / (baseML1Size + baseNL1Size)
        maxBaseKL1Limited = ops::FloorAlign(
            (L1_SIZE - scaleByteSize * DB_SIZE * baseN) / (baseM * DB_SIZE * MAX_PERF_STEPKB * aByteSize + baseNL1Size),
            64UL);
    } else {
        // L1_SIZE / (baseML1Size + baseNL1Size)
        maxBaseKL1Limited = ops::FloorAlign(L1_SIZE / (baseM * DB_SIZE * MAX_PERF_STEPKB * aByteSize + baseNL1Size),
                                            64UL);
    }
    if (matmulInfoPtr_->antiQuantType == QuantType::MX) {
        // baseM * DB * stepK * sizeof(aDtype) + baseN * db * stepK * sizeof(bDtype)，2：4 bit 位宽为 0.5，乘 0.5 即除以
        // 2
        uint64_t loadL1Size = baseM * DB_SIZE * MAX_PERF_STEPKB * aByteSize + baseN * DB_SIZE * MAX_PERF_STEPKB / 2;
        uint64_t l1LeftSize = L1_SIZE;
        if (matmulInfoPtr_->hasBias) {
            l1LeftSize -= baseN * ge::GetSizeByDataType(matmulInfoPtr_->biasDtype) * DB_SIZE;
        }
        maxBaseKL1Limited = ops::FloorAlign(l1LeftSize / loadL1Size, BASE_K_ALIGN_FOR_MX);
        baseK = std::min(baseK, maxBaseKL1Limited);
        uint64_t bScaleL1Size = 0;
        if (matmulInfoPtr_->bDtype == ge::DT_FLOAT4_E2M1) {
            // 考虑扣除FP8_E8M0 scale:baseN * DB * stepK * baseK / group_size
            bScaleL1Size = (baseN * DB_SIZE * MAX_PERF_STEPKB * baseK / MX_GROUP_SIZE);
        } else if (matmulInfoPtr_->bDtype == Ops::NN::DT_HIFLOAT4) {
            // 考虑扣除FP32 scale:baseN * DB * stepK * baseK * sizeof(fp32) / group_size
            bScaleL1Size = (baseN * DB_SIZE * MAX_PERF_STEPKB * baseK * scaleByteSize / HIF4_GROUP_SIZE);
        }
        maxBaseKL1Limited = ops::FloorAlign((l1LeftSize - bScaleL1Size) / loadL1Size, BASE_K_ALIGN_FOR_MX);
    }

    baseK = std::min(baseK, maxBaseKL1Limited);
    return baseK;
}

bool WeightQuantBatchMatmulV2TilingASWMix::CheckBasicBlockL1BufferSize(uint64_t baseM, uint64_t baseN,
                                                                       uint64_t baseK) const
{
    uint64_t aByteSize = ge::GetSizeByDataType(matmulInfoPtr_->aDtype);
    uint64_t bByteSize = ge::GetSizeByDataType(matmulInfoPtr_->bDtype);
    uint64_t scaleByteSize = ge::GetSizeByDataType(matmulInfoPtr_->antiQuantScaleDtype);
    uint64_t baseNAlign = ops::CeilAlign(
        baseN, matmulInfoPtr_->transB ? CUBE_BLOCK : GetShapeWithDataType(L1_ALIGN_SIZE, matmulInfoPtr_->bDtype));
    uint64_t l1Size = baseM * baseK * DB_SIZE * MAX_PERF_STEPKB * aByteSize +
                      baseNAlign * baseK * DB_SIZE * MAX_PERF_STEPKB * bByteSize +
                      (matmulInfoPtr_->antiQuantType == QuantType::PER_CHANNEL ? scaleByteSize * DB_SIZE * baseN : 0);
    // A16Fp4 perchannel 在L1上还需要占用一个全1的矩阵用于MX的计算
    if (matmulInfoPtr_->bDtype == ge::DT_FLOAT4_E2M1 && matmulInfoPtr_->antiQuantType == QuantType::PER_CHANNEL) {
        l1Size = baseM * baseK * DB_SIZE * MAX_PERF_STEPKB * aByteSize + scaleByteSize * DB_SIZE * baseN +
                 baseN * baseK * DB_SIZE * MAX_PERF_STEPKB / 2 +
                 MAX_PERF_STEPKB * baseN * baseK / MX_GROUP_SIZE; // 2：4 bit 位宽为 0.5，乘 0.5 即除以 2
    } else if (matmulInfoPtr_->bDtype == ge::DT_FLOAT4_E2M1 && matmulInfoPtr_->antiQuantType == QuantType::MX) {
        // 2：4 bit 位宽为 0.5，乘 0.5 即除以 2；sizeof(float8_e8m0) = 1
        l1Size = baseM * baseK * DB_SIZE * MAX_PERF_STEPKB * aByteSize + baseN * baseK * DB_SIZE * MAX_PERF_STEPKB / 2 +
                 MAX_PERF_STEPKB * baseN * baseK * DB_SIZE / MX_GROUP_SIZE;
        if (matmulInfoPtr_->hasBias) {
            l1Size += baseN * ge::GetSizeByDataType(matmulInfoPtr_->biasDtype) * DB_SIZE;
        }
    } else if (matmulInfoPtr_->bDtype == Ops::NN::DT_HIFLOAT4) {
        // 2：4 bit 位宽为 0.5，乘 0.5 即除以 2
        l1Size = baseM * baseK * DB_SIZE * MAX_PERF_STEPKB * aByteSize + baseN * baseK * DB_SIZE * MAX_PERF_STEPKB / 2 +
                 MAX_PERF_STEPKB * baseN * baseK * DB_SIZE * scaleByteSize / HIF4_GROUP_SIZE;
        l1Size += matmulInfoPtr_->hasBias ? baseN * DB_SIZE * aByteSize : 0;
    } else if (Is4BitDtype(matmulInfoPtr_->bDtype)) {
        l1Size = baseM * baseK * DB_SIZE * MAX_PERF_STEPKB * aByteSize + baseNAlign * baseK * MAX_PERF_STEPKB +
                 (matmulInfoPtr_->antiQuantType == QuantType::PER_CHANNEL ? scaleByteSize * DB_SIZE * baseN : 0);
    }

    if (l1Size >= L1_SIZE) {
        return false;
    }

    return true;
}

void WeightQuantBatchMatmulV2TilingASWMix::ReSplitBasicBlock()
{
    uint64_t mBlockCnt = ops::CeilDiv(matmulInfoPtr_->mSize, adaptiveWin_.baseM);
    uint64_t nBlockCnt = ops::CeilDiv(matmulInfoPtr_->nSize, adaptiveWin_.baseN);
    // 主要针对N较小的场景，N>4096时，原切分过程可以切出合理的baseN>=80的基本块，N<=4096时, 混合考虑M方向重新切分
    bool noNeedReSplit = (matmulInfoPtr_->antiQuantType != QuantType::MX &&
                          (nBlockCnt * mBlockCnt >= compileInfoPtr_->aicNum ||
                           matmulInfoPtr_->nSize > MIN_RESPLIT_N_SIZE));
    if (noNeedReSplit) {
        return;
    }

    double minMte2Cost = std::numeric_limits<double>::max();
    uint64_t baseM = 0;
    uint64_t baseN = 0;
    uint64_t baseNAlignSize = (!matmulInfoPtr_->transB) && (matmulInfoPtr_->bFormat == ge::FORMAT_FRACTAL_NZ) ?
                                  GetShapeWithDataType(L1_ALIGN_SIZE, matmulInfoPtr_->bDtype) :
                                  CUBE_BLOCK;
    // 根据MTE2耗时，选择切分的基本块
    for (uint64_t nBlockNum = 1; nBlockNum <= compileInfoPtr_->aicNum; nBlockNum++) {
        baseN = ops::CeilAlign(matmulInfoPtr_->nSize / nBlockNum, baseNAlignSize);
        if (baseN > BASIC_BLOCK_SIZE_256 || baseN < MIN_BASIC_BLOCK_N) {
            continue;
        }
        uint64_t nDim = ops::CeilDiv(matmulInfoPtr_->nSize, baseN);
        uint64_t mDim = compileInfoPtr_->aicNum / nDim;
        baseM = ops::CeilAlign(ops::CeilDiv(matmulInfoPtr_->mSize, mDim), CUBE_BLOCK);
        mDim = ops::CeilDiv(matmulInfoPtr_->mSize, baseM);
        if (baseM == BASIC_BLOCK_SIZE_128 && baseN > BASIC_BLOCK_SIZE_256 - CUBE_BLOCK) {
            continue;
        }
        if (baseM > BASIC_BLOCK_SIZE_128) {
            break;
        }
        uint64_t aByteSize = ge::GetSizeByDataType(matmulInfoPtr_->aDtype);
        uint64_t bByteSize = ge::GetSizeByDataType(matmulInfoPtr_->bDtype);
        double l2HbmBWRatio = 1.8; // 1.8:L2带宽和HBM带宽比值
        double mte2Cost = matmulInfoPtr_->mSize * (nDim - 1) * aByteSize +
                          matmulInfoPtr_->mSize * aByteSize * l2HbmBWRatio;
        if (Is4BitInput(matmulInfoPtr_->bDtype)) {
            // 除以2：uint4的数据类型的字节数是0.5，乘以0.5这里做除以2处理
            mte2Cost += matmulInfoPtr_->nSize * (mDim - 1) / 2 + matmulInfoPtr_->nSize * l2HbmBWRatio / 2;
        } else {
            mte2Cost += matmulInfoPtr_->nSize * (mDim - 1) * bByteSize +
                        matmulInfoPtr_->nSize * bByteSize * l2HbmBWRatio;
        }
        // 降低总核数较少时被选中的概率
        mte2Cost = mte2Cost / BWUtilization(mDim, nDim);

        // 计算baseK
        uint64_t baseK = CalcBasicBlockBaseK(baseM, baseN);
        if (!CheckBasicBlockL1BufferSize(baseM, baseN, baseK)) {
            continue;
        }
        if (matmulInfoPtr_->antiQuantType != QuantType::MX && matmulInfoPtr_->hasBias) {
            uint64_t ubSize = baseM * baseN * (sizeof(float) + ge::GetSizeByDataType(matmulInfoPtr_->cDtype)) +
                              baseN * ge::GetSizeByDataType(matmulInfoPtr_->biasDtype);
            // 256: VF读数据单位是256B，当单核计算量非256B对齐时，会向后多读一块数据，此处空间预留
            if (ubSize * DB_SIZE + 256 > compileInfoPtr_->ubSize) {
                continue;
            }
        }
        if (mte2Cost < minMte2Cost) {
            minMte2Cost = mte2Cost;
            adaptiveWin_.baseM = baseM;
            adaptiveWin_.baseN = baseN;
            adaptiveWin_.baseK = baseK;
        }
    }
}

constexpr bool WeightQuantBatchMatmulV2TilingASWMix::Is4BitInput(ge::DataType dtype) const
{
    return dtype == ge::DT_FLOAT4_E2M1 || dtype == ge::DT_FLOAT4_E1M2 || dtype == ge::DT_INT4 ||
           dtype == Ops::NN::DT_HIFLOAT4;
}
uint64_t WeightQuantBatchMatmulV2TilingASWMix::GetShapeWithDataType(uint64_t shapeSize, ge::DataType dtype) const
{
    if (Is4BitInput(dtype)) {
        return shapeSize + shapeSize;
    } else {
        return shapeSize / static_cast<uint64_t>(ge::GetSizeByDataType(dtype));
    }
}

uint64_t WeightQuantBatchMatmulV2TilingASWMix::GetSizeWithDataType(uint64_t shapeSize, ge::DataType dtype) const
{
    // shapeSize应该是偶数
    if (Is4BitInput(dtype)) {
        // 2: 判断是否是偶数
        OP_TILING_CHECK(
            shapeSize % 2 != 0,
            CUBE_INNER_ERR_REPORT(
                opName_, "To get size of matrix/array, the number of elements must be even when dtype is FLOAT4/INT4"),
            return 0);
        // 1/2: 这几种数据类型的dsize=1/2
        return shapeSize / 2;
    } else {
        return shapeSize * static_cast<uint64_t>(ge::GetSizeByDataType(dtype));
    }
}

void WeightQuantBatchMatmulV2TilingASWMix::CalcTailBasicBlock()
{
    bool isPerGroup = matmulInfoPtr_->antiQuantType == QuantType::PER_GROUP;
    if (isPerGroup) {
        return;
    }
    if (adaptiveWin_.tailWinBlockCnt == 0UL) {
        return;
    }

    uint64_t mTile = 1UL;
    uint64_t nTile = 1UL;
    uint64_t preSplit = 1UL;
    uint64_t secSplit = 1UL;
    auto& preSplitValid = adaptiveWin_.mTail >= adaptiveWin_.nTail ? mTile : nTile;
    auto& secSplitValid = adaptiveWin_.mTail >= adaptiveWin_.nTail ? nTile : mTile;
    while (CalUsedCoreNum(preSplit + 1, secSplit) <= compileInfoPtr_->aicNum) {
        preSplit += 1UL;
        if (!IsInValidWeighNzTailSplit(preSplit, true)) {
            preSplitValid = preSplit;
        }

        if (CalUsedCoreNum(preSplit, secSplit + 1) <= compileInfoPtr_->aicNum) {
            secSplit += 1UL;
            if (!IsInValidWeighNzTailSplit(secSplit, false)) {
                secSplitValid = secSplit;
            }
        }
    }
    adaptiveWin_.mTailTile = mTile;
    adaptiveWin_.nTailTile = nTile;
    if (mTile == 1 && nTile == 1 && matmulInfoPtr_->mSize <= BASIC_BLOCK_SIZE_128 &&
        adaptiveWin_.totalBlockCnt > compileInfoPtr_->aicNum) {
        // 此时m方向只有1块，N方向单独重新计算尾块，走到这里，tailWinBlockCnt一定大于aicNum/2
        adaptiveWin_.mainTileNum = ops::FloorAlign(adaptiveWin_.totalBlockCnt,
                                                   static_cast<uint64_t>(compileInfoPtr_->aicNum));
        uint64_t baseNAlignSize = (!matmulInfoPtr_->transB) && (matmulInfoPtr_->bFormat == ge::FORMAT_FRACTAL_NZ) ?
                                      GetShapeWithDataType(L1_ALIGN_SIZE, matmulInfoPtr_->bDtype) :
                                      CUBE_BLOCK;
        uint64_t lastRoundNSize = matmulInfoPtr_->nSize - adaptiveWin_.mainTileNum * adaptiveWin_.baseN;
        adaptiveWin_.lastRoundBaseN = ops::CeilAlign(
            ops::CeilDiv(lastRoundNSize, static_cast<uint64_t>(compileInfoPtr_->aicNum)), baseNAlignSize);
        uint64_t lastRoundBlockNum = ops::CeilDiv(lastRoundNSize, adaptiveWin_.lastRoundBaseN);
        adaptiveWin_.lastRoundBaseNTail = lastRoundNSize - (lastRoundBlockNum - 1) * adaptiveWin_.lastRoundBaseN;
        adaptiveWin_.evadeDuplicateBlock = true;
    }
}

bool WeightQuantBatchMatmulV2TilingASWMix::IsInValidWeighNzTailSplit(uint64_t splitCnt, bool isPreSplit) const
{
    if (splitCnt == 0) {
        return false;
    }
    if ((isPreSplit && adaptiveWin_.mTail >= adaptiveWin_.nTail) ||
        (!isPreSplit && adaptiveWin_.mTail < adaptiveWin_.nTail)) {
        // 当前正在切 M 方向，baseM 最小切到 16
        if (ops::CeilDiv(adaptiveWin_.baseM, splitCnt) < CUBE_BLOCK) {
            return true;
        }
        return false;
    }
    uint64_t tailNSplit = ops::CeilDiv(adaptiveWin_.nTail, splitCnt);
    uint64_t baseNSplit = ops::CeilDiv(adaptiveWin_.baseN, splitCnt);

    if (matmulInfoPtr_->bDtype == ge::DT_INT4 && matmulInfoPtr_->bFormat == ge::FORMAT_ND &&
        (matmulInfoPtr_->antiQuantType == QuantType::PER_CHANNEL ||
         matmulInfoPtr_->antiQuantType == QuantType::PER_TENSOR)) {
        // A16S4 weight-ND PerChannel/PerTensor 场景支持 N/K 非 C0 对齐，切分结果仅要求 2 对齐，baseN 最小切到 16
        if (tailNSplit % 2 == 0 && baseNSplit % 2 == 0 && baseNSplit >= CUBE_BLOCK) {
            return false;
        } else {
            return true;
        }
    }
    // 对齐场景，要求 N 方向切分后仍保持对齐
    uint64_t baseNAlignSize = matmulInfoPtr_->transB ? CUBE_BLOCK :
                                                       GetShapeWithDataType(L1_ALIGN_SIZE, matmulInfoPtr_->bDtype);
    if (tailNSplit % baseNAlignSize != 0 || baseNSplit % baseNAlignSize != 0) {
        return true;
    }
    // 如果是perchannel并且含有bias，需要通过fixpipe搬运到ub，重切分之后的内轴需要32B对齐
    if ((matmulInfoPtr_->antiQuantType == QuantType::PER_CHANNEL ||
         matmulInfoPtr_->antiQuantType == QuantType::PER_TENSOR) &&
        matmulInfoPtr_->hasBias) {
        if (adaptiveWin_.baseN % splitCnt != 0 || adaptiveWin_.nTail % splitCnt != 0) {
            return true;
        }
        // 16:最后cast成b16类型写回UB需要32B对齐，则重切分之后的n需要对32 / 2 = 16对齐
        if ((adaptiveWin_.baseN / splitCnt) % 16 != 0 || (adaptiveWin_.nTail / splitCnt) % 16 != 0) {
            return true;
        }
    }

    return false;
}

uint32_t WeightQuantBatchMatmulV2TilingASWMix::CalUsedCoreNum()
{
    if (adaptiveWin_.totalWinCnt > 1 || adaptiveWin_.tailWinBlockCnt == 0) {
        return compileInfoPtr_->aicNum;
    }

    return adaptiveWin_.tailWinBlockCnt * adaptiveWin_.mTailTile * adaptiveWin_.nTailTile;
}

uint32_t WeightQuantBatchMatmulV2TilingASWMix::CalUsedCoreNum(uint32_t mTile, uint32_t nTile)
{
    return mTile * nTile * adaptiveWin_.tailWinBlockCnt;
}

void WeightQuantBatchMatmulV2TilingASWMix::CalL1Tiling()
{
    basicTiling_.usedCoreNum = CalUsedCoreNum();
    OP_LOGD(opName_, "coreNum: %u", basicTiling_.usedCoreNum);
    basicTiling_.baseM = adaptiveWin_.baseM;
    basicTiling_.baseN = adaptiveWin_.baseN;

    bool isA16MxFp4 = (matmulInfoPtr_->aDtype == ge::DT_FLOAT16 || matmulInfoPtr_->aDtype == ge::DT_BF16) &&
                      matmulInfoPtr_->bDtype == ge::DT_FLOAT4_E2M1 && matmulInfoPtr_->antiQuantType == QuantType::MX;
    if (isA16MxFp4) {
        basicTiling_.baseK = std::min(adaptiveWin_.baseK, ops::CeilAlign<uint64_t>(matmulInfoPtr_->kSize, B4_C0_SIZE));
    } else {
        basicTiling_.baseK = std::min(adaptiveWin_.baseK, matmulInfoPtr_->kSize);
    }

    basicTiling_.stepM = 1;
    basicTiling_.stepN = 1;
    basicTiling_.singleCoreM = std::min(matmulInfoPtr_->mSize, static_cast<uint64_t>(basicTiling_.baseM));
    basicTiling_.singleCoreN = std::min(matmulInfoPtr_->nSize, static_cast<uint64_t>(basicTiling_.baseN));
    basicTiling_.singleCoreK = matmulInfoPtr_->kSize;

    uint64_t biasDtypeSize = ge::GetSizeByDataType(matmulInfoPtr_->biasDtype);
    uint64_t scaleDtypeSize = ge::GetSizeByDataType(matmulInfoPtr_->antiQuantScaleDtype);
    uint64_t totalL1Size = compileInfoPtr_->l1Size;

    basicTiling_.iterateOrder = 0;
    basicTiling_.dbL0c = ((basicTiling_.baseM * basicTiling_.baseN * DATA_SIZE_L0C * DB_SIZE <=
                           compileInfoPtr_->l0cSize) &&
                          CheckAntiQuantScale(basicTiling_.baseN, DB_SIZE)) ?
                             DB_SIZE :
                             1;
    uint64_t singleCoreBiasSize = (matmulInfoPtr_->hasBias && matmulInfoPtr_->antiQuantType == QuantType::MX) ?
                                      basicTiling_.baseN * biasDtypeSize :
                                      0;
    uint64_t singleCoreScaleSize = (matmulInfoPtr_->antiQuantType == QuantType::PER_CHANNEL ||
                                    matmulInfoPtr_->antiQuantType == QuantType::MX ||
                                    (matmulInfoPtr_->antiQuantType == QuantType::PER_GROUP &&
                                     matmulInfoPtr_->bDtype == ge::DT_FLOAT4_E2M1)) ?
                                       basicTiling_.baseN * scaleDtypeSize :
                                       0;
    uint64_t leftL1Size = totalL1Size - singleCoreBiasSize - singleCoreScaleSize;
    CalL1TilingDepthANotfullload(leftL1Size);
}

void WeightQuantBatchMatmulV2TilingASWMix::CalL1TilingDepthANotfullload(uint64_t leftL1Size)
{
    basicTiling_.depthA1 = GetShapeWithDataType(leftL1Size / NUM_HALF / basicTiling_.baseM / basicTiling_.baseK,
                                                matmulInfoPtr_->aDtype);
    basicTiling_.depthB1 = GetShapeWithDataType(leftL1Size / NUM_HALF / basicTiling_.baseN / basicTiling_.baseK,
                                                matmulInfoPtr_->bDtype);
    CalStepKs();
}

void WeightQuantBatchMatmulV2TilingASWMix::CalStepKs()
{
    basicTiling_.stepKa = MAX_PERF_STEPKB;
    basicTiling_.stepKb = MAX_PERF_STEPKB;
    basicTiling_.depthA1 = basicTiling_.stepKa * DB_SIZE;
    basicTiling_.depthB1 = basicTiling_.stepKb * DB_SIZE;
}

bool WeightQuantBatchMatmulV2TilingASWMix::CheckAntiQuantScale(uint64_t baseN, uint64_t dbL0c) const
{
    uint64_t maxScaleBaseN = dbL0c == 1 ? BASIC_BLOCK_SIZE_512 : BASIC_BLOCK_SIZE_256;
    bool isScaleInvalid = baseN > maxScaleBaseN;
    return !isScaleInvalid;
}

void WeightQuantBatchMatmulV2TilingASWMix::SetTilingData()
{
    tilingData_->mTailTile = adaptiveWin_.mTailTile;
    tilingData_->nTailTile = adaptiveWin_.nTailTile;
    tilingData_->mainTileNum = adaptiveWin_.mainTileNum;
    tilingData_->lastRoundBaseN = adaptiveWin_.lastRoundBaseN;
    tilingData_->lastRoundBaseNTail = adaptiveWin_.lastRoundBaseNTail;
    tilingData_->evadeDuplicateBlock = static_cast<uint16_t>(adaptiveWin_.evadeDuplicateBlock);
    tilingData_->groupSize = matmulInfoPtr_->groupSize;
    tilingData_->enablePreAlloc = adaptiveWin_.enablePreAlloc;
}

ge::graphStatus WeightQuantBatchMatmulV2TilingASWMix::DoLibApiTiling()
{
    tilingData_->matmulTiling.M = matmulInfoPtr_->mSize;
    tilingData_->matmulTiling.N = matmulInfoPtr_->nSize;
    tilingData_->matmulTiling.Ka = matmulInfoPtr_->kSize;
    tilingData_->matmulTiling.Kb = matmulInfoPtr_->kSize;
    tilingData_->matmulTiling.usedCoreNum = basicTiling_.usedCoreNum;
    tilingData_->matmulTiling.singleCoreM = basicTiling_.singleCoreM;
    tilingData_->matmulTiling.singleCoreN = basicTiling_.singleCoreN;
    tilingData_->matmulTiling.singleCoreK = basicTiling_.singleCoreK;
    tilingData_->matmulTiling.baseM = basicTiling_.baseM;
    tilingData_->matmulTiling.baseN = basicTiling_.baseN;
    tilingData_->matmulTiling.baseK = basicTiling_.baseK;
    tilingData_->matmulTiling.depthA1 = basicTiling_.depthA1;
    tilingData_->matmulTiling.depthB1 = basicTiling_.depthB1;
    tilingData_->matmulTiling.stepM = basicTiling_.stepM;
    tilingData_->matmulTiling.stepN = basicTiling_.stepN;
    tilingData_->matmulTiling.stepKa = basicTiling_.stepKa;
    tilingData_->matmulTiling.stepKb = basicTiling_.stepKb;
    tilingData_->matmulTiling.isBias = static_cast<int32_t>(matmulInfoPtr_->hasBias);
    tilingData_->matmulTiling.iterateOrder = basicTiling_.iterateOrder;
    tilingData_->matmulTiling.dbL0A = 2; // db switch, 1: off, 2: on
    tilingData_->matmulTiling.dbL0B = 2; // db switch, 1: off, 2: on
    tilingData_->matmulTiling.dbL0C = basicTiling_.dbL0c;
    if (matmulInfoPtr_->antiQuantType == QuantType::MX) {
        // _mxTypePara的高8位是拷贝scaleB的倍数，257是 1 << 8 + 1 表示scale就拷贝一倍的k
        tilingData_->matmulTiling.mxTypePara = 257;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus WeightQuantBatchMatmulV2TilingASWMix::GetWorkspaceSize()
{
    size_t* workspaces = context_->GetWorkspaceSizes(1);
    OP_TILING_CHECK(workspaces == nullptr, VECTOR_INNER_ERR_REPORT_TILIING(opName_, "failed to get workspace size"),
                    return ge::GRAPH_FAILED);
    // 申请的 workspace 大小为 16 MB
    workspaces[0] = 16 * MEGABYTE;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus WeightQuantBatchMatmulV2TilingASWMix::PostTiling()
{
    OP_LOGD(opName_, "final tiling data size: %zu", tilingDataSize_);
    OP_TILING_CHECK(tilingDataSize_ % sizeof(uint64_t) != 0,
                    CUBE_INNER_ERR_REPORT(opName_, "tiling data size[%zu] is not aligned to 8", tilingDataSize_),
                    return ge::GRAPH_FAILED);
    errno_t ret = memcpy_s(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity(),
                           static_cast<const void*>(tilingData_.get()), tilingDataSize_);
    if (ret != EOK) {
        OP_LOGE(context_->GetNodeName(), "memcpy_s failed, ret=%d", ret);
        return ge::GRAPH_FAILED;
    }
    context_->SetBlockDim(basicTiling_.usedCoreNum);
    context_->GetRawTilingData()->SetDataSize(tilingDataSize_);
    return ge::GRAPH_SUCCESS;
}

bool WeightQuantBatchMatmulV2TilingASWMix::ClosePreAllocCond() const
{
    int64_t baseK = std::min(adaptiveWin_.baseK, matmulInfoPtr_->kSize);
    int64_t actualMSize = std::min(matmulInfoPtr_->mSize, adaptiveWin_.baseM);
    int64_t actualNSize = std::min(matmulInfoPtr_->nSize, adaptiveWin_.baseN);
    int64_t aL1Size = actualMSize * baseK * MAX_PERF_STEPKB * ge::GetSizeByDataType(matmulInfoPtr_->aDtype);
    int64_t bL1Size = actualNSize * baseK * MAX_PERF_STEPKB * GetDtypeBits(matmulInfoPtr_->bDtype) / BYTE_BITS;
    // AL1 + BL1载入数据量较大(128 * 1024 B,经验值)时，开启prealloc会导致后2轮的mte2抢占前面的ostd,
    // 导致前面的带宽跟不上cube计算 当K较小时，单轮计算量较小， 如果不开启prealloc,
    // 下一轮MN的计算的MTE2无法提前发出，会导致mte2断流
    bool condition = aL1Size + bL1Size >= 128 * 1024 && matmulInfoPtr_->kSize > 1024;
    // 尾轮切分可能会破坏 aL1Size + bL1Size 较大的条件，当只有一轮计算并且进行了重切分时，数据量会变小，需要开启
    condition = condition && !(adaptiveWin_.totalWinCnt == 1 && adaptiveWin_.mTailTile * adaptiveWin_.nTailTile > 1);
    return condition;
}

uint8_t WeightQuantBatchMatmulV2TilingASWMix::GetPerGroupCustomConfig() const
{
    if (OpenL1QuadrupleBufferCond()) {
        return static_cast<uint16_t>(WQBMMV2_TEMPLATE_L1_QUARTER_BUFFER);
    } else if (OpenUBQuadrupleBufferCond(adaptiveWin_.baseM, adaptiveWin_.baseN, adaptiveWin_.baseK)) {
        return static_cast<uint16_t>(WQBMMV2_TEMPLATE_UB_QUARTER_BUFFER);
    }
    return static_cast<uint16_t>(WQBMMV2_TEMPLATE_DOUBLE_BUFFER);
}

uint64_t WeightQuantBatchMatmulV2TilingASWMix::GetTilingKey() const
{
    uint64_t socVersionType = WQBMMV2_SOC_ASCEND960DT;
    uint64_t subSocVersionType = WQBMMV2_DEFAULT;
    uint64_t antiquantScenario = WQBMMV2_DEFAULT;

    uint64_t algorithm = WQBMMV2_ALGO_VECTOR_ANTIQUANT;

    if ((matmulInfoPtr_->aDtype == ge::DT_FLOAT16 || matmulInfoPtr_->aDtype == ge::DT_BF16) &&
        (matmulInfoPtr_->bDtype == ge::DT_INT4 || matmulInfoPtr_->bDtype == ge::DT_FLOAT4_E2M1 ||
         matmulInfoPtr_->bDtype == ge::DT_INT8 || matmulInfoPtr_->bDtype == ge::DT_FLOAT8_E4M3FN) &&
        matmulInfoPtr_->antiQuantType == QuantType::PER_GROUP) {
        algorithm = WQBMMV2_ALGO_CUBE_VECTOR_ANTIQUANT;
    } else if ((matmulInfoPtr_->aDtype == ge::DT_FLOAT16 || matmulInfoPtr_->aDtype == ge::DT_BF16) &&
               matmulInfoPtr_->bDtype == ge::DT_FLOAT4_E2M1 &&
               (matmulInfoPtr_->antiQuantType == QuantType::PER_GROUP ||
                matmulInfoPtr_->antiQuantType == QuantType::MX)) {
        algorithm = WQBMMV2_ALGO_CUBE_ANTIQUANT;
    } else if ((matmulInfoPtr_->aDtype == ge::DT_FLOAT16 || matmulInfoPtr_->aDtype == ge::DT_BF16) &&
               matmulInfoPtr_->bDtype == Ops::NN::DT_HIFLOAT4) {
        algorithm = WQBMMV2_ALGO_CUBE_ANTIQUANT;
    } else {
        algorithm = WQBMMV2_ALGO_FIXPIPE_ANTIQUANT;
    }

    uint64_t subAlgorithm = WQBMMV2_SUB_ALGO_ASW;
    uint64_t templateCustom = 0UL;
    if (matmulInfoPtr_->antiQuantType == QuantType::PER_GROUP) {
        templateCustom = GetPerGroupCustomConfig();
    }

    uint64_t apiConstexpr = 0UL;

    bool transA = matmulInfoPtr_->transA;
    bool transB = matmulInfoPtr_->transB;
    uint64_t antiquantType = static_cast<uint64_t>(matmulInfoPtr_->antiQuantType);
    uint64_t quantType = WQBMMV2_QUANT_TYPE_NONE;
    bool hasAntiquantOffset = matmulInfoPtr_->hasAntiQuantOffset;
    bool hasBias = matmulInfoPtr_->hasBias;
    bool isBiasFp32 = matmulInfoPtr_->biasDtype == ge::DT_FLOAT;
    bool isWeightNz = matmulInfoPtr_->bFormat == ge::FORMAT_FRACTAL_NZ;
    uint64_t tilingKey = GET_TPL_TILING_KEY(socVersionType, subSocVersionType, antiquantScenario, algorithm,
                                            subAlgorithm, templateCustom, apiConstexpr, transA, transB, antiquantType,
                                            quantType, hasAntiquantOffset, hasBias, isBiasFp32, isWeightNz);
    return tilingKey;
}

REGISTER_TILING_TEMPLATE("WeightQuantBatchMatmulV2", WeightQuantBatchMatmulV2TilingASWMix, ASW_MIX_PRIORITY);
} // namespace weight_quant_batch_matmul_v2
} // namespace optiling

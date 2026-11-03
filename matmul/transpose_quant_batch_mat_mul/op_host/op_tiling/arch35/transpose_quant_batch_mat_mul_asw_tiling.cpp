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
#include "../../../../../common/inc/op_host/op_const_def_nn.h"
#include "matmul/mat_mul_v3/op_host/op_tiling/arch35/matmul_tiling_registry.h"
#include "transpose_quant_batch_mat_mul_tiling_strategy.h"
#include "op_host/tiling_key.h"
#include "../../../op_kernel/arch35/transpose_quant_batch_mat_mul_tiling_key.h"
#include "matmul/common/op_host/math_util_nn.h"
#include "matmul/quant_batch_matmul_v3/op_host/op_tiling/arch35/base_block_calculator.h"
#include "matmul/quant_batch_matmul_v3/op_host/op_tiling/arch35/l1_tiling_data_calculator.h"
#include "matmul/quant_batch_matmul_v3/op_host/op_tiling/arch35/quant_batch_matmul_v3_tiling_util.h"
#include "matmul/quant_batch_matmul_v3/op_host/op_tiling/quant_batch_matmul_v3_compile_info.h"
#include "matmul/quant_batch_matmul_v3/op_host/op_tiling/quant_batch_matmul_v3_tiling_base.h"
#include "tiling/platform/platform_ascendc.h"

using Ops::NN::MathUtil;
namespace optiling {
namespace transpose_quant_batch_mat_mul_advanced {
using namespace strategy;
MM_REGISTER_TILING_TEMPLATE(TransposeQuantBatchMatMul, TransposeQuantBatchMatMulAswTiling, DAV_3510, BASE);
MM_REGISTER_TILING_TEMPLATE(TransposeQuantBatchMatMul, TransposeQuantBatchMatMulAswTiling, DAV_9201, BASE);
MM_REGISTER_TILING_TEMPLATE(TransposeQuantBatchMatMul, TransposeQuantBatchMatMulAswTiling, DAV_9202, BASE);

ge::graphStatus TransposeQuantBatchMatMulAswTiling::DoOpTiling()
{
    MatMulV3TilingHelper::ResetBase(compileInfo_, args_, runInfo_);
    apiLevel_ = TQBMMApiLevel::TENSOR_LEVEL;
    isMXFP4_ = IsMXFP4(context_->GetInputDesc(X1_IDX), context_->GetInputDesc(X2_IDX),
                       context_->GetOptionalInputDesc(SCALE_X1_IDX), context_->GetOptionalInputDesc(SCALE_X2_IDX));
    isHIFP4_ = IsHIFP4(context_->GetInputDesc(X1_IDX), context_->GetInputDesc(X2_IDX),
                       context_->GetOptionalInputDesc(SCALE_X1_IDX), context_->GetOptionalInputDesc(SCALE_X2_IDX));
    if (isHIFP4_) {
        runInfo_.baseK = BASIC_BLOCK_K_128_BYTE * NUM_TWO;
        precisionMode_ = TQBMMPrecisionMode::PRECISION_MODE_HIFP4;
        CalL1Tiling();
    } else if (isMXFP4_) {
        runInfo_.baseK = BASIC_BLOCK_K_128_BYTE;
        precisionMode_ = TQBMMPrecisionMode::PRECISION_MODE_MXFP4;
        CalL1Tiling();
    } else if (IsMxFp8(context_->GetInputDesc(X1_IDX), context_->GetInputDesc(X2_IDX),
                       context_->GetOptionalInputDesc(SCALE_X1_IDX), context_->GetOptionalInputDesc(SCALE_X2_IDX))) {
        precisionMode_ = TQBMMPrecisionMode::PRECISION_MODE_MXFP8;
        if (compileInfo_.npuArch == NpuArch::DAV_9201) {
            QuantBatchMatmulInfo qbmmParams;
            BuildQbmmParams(qbmmParams);
            QuantBatchMatmulV3CompileInfo qbmmCompileInfo = {};
            if (!BuildQbmmCompileInfo(qbmmCompileInfo)) {
                CalL1Tiling();
            } else {
                ApplyQbmmLoadBalanceBaseBlock(qbmmParams, qbmmCompileInfo);
                ApplyQbmmMxTailWindow(qbmmCompileInfo);
                ApplyQbmmMxL1Tiling(qbmmParams, qbmmCompileInfo);
            }
        } else {
            CalL1Tiling();
        }
    } else if (IsHIFP8(context_->GetInputDesc(X1_IDX), context_->GetInputDesc(X2_IDX))) {
        precisionMode_ = TQBMMPrecisionMode::PRECISION_MODE_HIFP8;
        if (context_->GetOptionalInputShape(BIAS_IDX) != nullptr &&
            context_->GetOptionalInputDesc(BIAS_IDX)->GetDataType() != ge::DT_FLOAT) {
            OP_LOGE(args_.opName, "HIFP8 only supports float32 bias, but got dtype %d",
                    static_cast<int32_t>(context_->GetOptionalInputDesc(BIAS_IDX)->GetDataType()));
            return ge::GRAPH_FAILED;
        }
        CalL1Tiling();
    } else {
        precisionMode_ = TQBMMPrecisionMode::PRECISION_MODE_FP8;
        // 960 系列（9201/9202）为 fixpipe 1v1（AIC:AIV=1:1），dequant epilogue 需在单个 AIV 的
        // UB 内同时容纳整块 fp32 L0C 与 scale/bias/输出；先按 UB 预算收敛 baseM，再做 L1 切分。
        CapBaseMForDequantUb();
        MatMulV3TilingHelper::CalL1Tiling(compileInfo_, args_, runInfo_);
    }
    GetTransposeBatchMatMulInfo();
    return ge::GRAPH_SUCCESS;
}

bool TransposeQuantBatchMatMulAswTiling::IsCapable()
{
    uint64_t aivToAicRatio = (compileInfo_.npuArch == NpuArch::DAV_3510) ? NUM_TWO : NUM_ONE;
    if (!IsMicroScaling(context_->GetOptionalInputDesc(SCALE_X1_IDX), context_->GetOptionalInputDesc(SCALE_X2_IDX)) &&
        compileInfo_.aivNum != compileInfo_.aicNum * aivToAicRatio) {
        OP_LOGE(args_.opName,
                "TransposeQuantBatchMatMul is only supported for aivNum == aicNum *%lu.aivNum:%llu,aicNum:%llu",
                aivToAicRatio, compileInfo_.aivNum, compileInfo_.aicNum);
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
        batchSplitMode_ = batchSplitFactor_ > 1 ? TQBMMBatchSplit::BATCH_SPLIT_TRUE :
                                                  TQBMMBatchSplit::BATCH_SPLIT_FALSE;
    }
}

// 960 系列（9201/9202）AIC:AIV 核数比为 1:1（fixpipe 1v1，非 3510 的双 AIV 半 M 切分）：
// AIC 会把整块 baseM x baseN 的 fp32 L0C 累加器一次 fixpipe 进单个 AIV 的 UB，
// BlockEpilogueDequant::SetupUbLayout 按该整块预留 UB 空间。若 baseM*baseN*4 超出 UB
// （如 256x256x4B=256KB > 248KB），UB 布局必然越界。此处按 epilogue 布局逐字节计算 UB 占用，
// 以 16 为粒度下调 baseM，保证布局可放下；过小则保持不收敛。
void TransposeQuantBatchMatMulAswTiling::CapBaseMForDequantUb()
{
    if (compileInfo_.npuArch != NpuArch::DAV_9201 && compileInfo_.npuArch != NpuArch::DAV_9202) {
        return;
    }
    // 与 BlockEpilogueDequant::SetupUbLayout（CV=1）一致的预算公式：
    //   L0C  : baseM * align(baseN, 32/sizeof(L0CType)) * sizeof(L0CType)   (FP8 -> fp32, 4B)
    //   scale: align(baseN*4, 32) + align(baseM*4, 32)                     (x2Scale/x1Scale 均 fp32)
    //   bias : align(baseN*biasSize, 32)                                   (可选)
    //   out  : 2 * align(ceil(baseM/fp32OutTimes) * align(baseN, 32/ySize) * ySize, 32)
    const uint64_t nAligned = ops::CeilAlign(runInfo_.baseN, BASIC_BLOCK_SIZE_32 / DATA_SIZE_FP32);
    const uint64_t x2ScaleBytes = ops::CeilAlign(runInfo_.baseN * DATA_SIZE_FP32, BASIC_BLOCK_SIZE_32);
    // bias 运行期 dtype 可能为 bf16/fp16/fp32，统一按 4B 保守估计，最多只会让 baseM 略小。
    const uint64_t biasBytes = args_.hasBias ? ops::CeilAlign(runInfo_.baseN * DATA_SIZE_FP32, BASIC_BLOCK_SIZE_32) :
                                               0UL;
    uint64_t outDtypeSize = ge::GetSizeByDataType(args_.cType);
    const bool isFp32Out = outDtypeSize == DATA_SIZE_FP32;
    if (!isFp32Out) {
        outDtypeSize = 2UL; // FP8 MIX 的 Y 仅支持 fp16/bf16
    }
    const uint64_t fp32OutTimes = isFp32Out ? 4UL : 2UL;
    while (runInfo_.baseM > BASIC_BLOCK_SIZE_16) {
        const uint64_t curBaseM = runInfo_.baseM;
        const uint64_t l0cBytes = curBaseM * nAligned * DATA_SIZE_FP32;
        const uint64_t x1ScaleBytes = ops::CeilAlign(curBaseM * DATA_SIZE_FP32, BASIC_BLOCK_SIZE_32);
        const uint64_t outOnceBytes = ops::CeilAlign(
            ops::CeilDiv(curBaseM, fp32OutTimes) * ops::CeilAlign(runInfo_.baseN, BASIC_BLOCK_SIZE_32 / outDtypeSize) *
                outDtypeSize,
            BASIC_BLOCK_SIZE_32);
        if (l0cBytes + x2ScaleBytes + x1ScaleBytes + biasBytes + 2UL * outOnceBytes <= compileInfo_.ubSize) {
            break;
        }
        runInfo_.baseM -= BASIC_BLOCK_SIZE_16;
    }
    runInfo_.singleCoreM = runInfo_.baseM;
}

void TransposeQuantBatchMatMulAswTiling::BuildQbmmParams(QuantBatchMatmulInfo& qbmmParams) const
{
    qbmmParams.opName = args_.opName;
    qbmmParams.mSize = args_.mValue;
    qbmmParams.nSize = args_.nValue;
    qbmmParams.kSize = args_.kValue;
    qbmmParams.transA = args_.isATrans;
    qbmmParams.transB = args_.isBTrans;
    qbmmParams.aDtype = args_.aType;
    qbmmParams.bDtype = args_.bType;
    qbmmParams.cDtype = args_.cType;
    qbmmParams.scaleDtype = context_->GetOptionalInputDesc(SCALE_X2_IDX)->GetDataType();
    qbmmParams.perTokenScaleDtype = context_->GetOptionalInputDesc(SCALE_X1_IDX)->GetDataType();
    qbmmParams.isMxPerGroup = true;
    qbmmParams.batchA = batchInfo_->batchA;
    qbmmParams.batchB = batchInfo_->batchB;
    qbmmParams.batchC = batchInfo_->batchC;
}

bool TransposeQuantBatchMatMulAswTiling::BuildQbmmCompileInfo(QuantBatchMatmulV3CompileInfo& qbmmCompileInfo) const
{
    fe::PlatFormInfos* platformInfo = context_->GetPlatformInfo();
    if (platformInfo == nullptr) {
        OP_LOGE(args_.opName, "platformInfo is null, skip qbmm tiling.");
        return false;
    }
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    qbmmCompileInfo.ubSize = 0UL;
    qbmmCompileInfo.l1Size = 0UL;
    qbmmCompileInfo.l2Size = 0UL;
    qbmmCompileInfo.l0cSize = 0UL;
    qbmmCompileInfo.l0aSize = 0UL;
    qbmmCompileInfo.l0bSize = 0UL;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, qbmmCompileInfo.ubSize);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L1, qbmmCompileInfo.l1Size);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L2, qbmmCompileInfo.l2Size);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_C, qbmmCompileInfo.l0cSize);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_A, qbmmCompileInfo.l0aSize);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_B, qbmmCompileInfo.l0bSize);
    qbmmCompileInfo.aicNum = ascendcPlatform.GetCoreNumAic();
    qbmmCompileInfo.aivNum = ascendcPlatform.GetCoreNumAiv();
    qbmmCompileInfo.npuArch = compileInfo_.npuArch;
    qbmmCompileInfo.socVersion = compileInfo_.socVersion;
    return true;
}

void TransposeQuantBatchMatMulAswTiling::ApplyQbmmLoadBalanceBaseBlock(
    const QuantBatchMatmulInfo& qbmmParams, const QuantBatchMatmulV3CompileInfo& qbmmCompileInfo)
{
    BaseBlockCalculator calculator(qbmmParams, qbmmCompileInfo, batchInfo_->batchC, context_->GetPlatformInfo());
    if (!calculator.Compute(BaseBlockMode::LOAD_BALANCE)) {
        OP_LOGD(args_.opName, "qbmm load balance base block search failed, keep default base block.");
        return;
    }
    const BaseBlockRes& baseBlock = calculator.GetOutput();
    runInfo_.baseM = baseBlock.baseM;
    runInfo_.baseN = baseBlock.baseN;
    runInfo_.baseK = baseBlock.baseK;
    runInfo_.singleCoreM = baseBlock.baseM;
    runInfo_.singleCoreN = baseBlock.baseN;
}

void TransposeQuantBatchMatMulAswTiling::ApplyQbmmMxTailWindow(const QuantBatchMatmulV3CompileInfo& qbmmCompileInfo)
{
    const uint64_t aicNum = qbmmCompileInfo.aicNum;
    const uint64_t baseM = runInfo_.baseM;
    const uint64_t baseN = runInfo_.baseN;
    const uint64_t mBlockCnt = ops::CeilDiv(args_.mValue, baseM);
    const uint64_t nBlockCnt = ops::CeilDiv(args_.nValue, baseN);
    const uint64_t totalBlockCnt = batchInfo_->batchC * mBlockCnt * nBlockCnt;
    const uint64_t mTail = args_.mValue - (mBlockCnt - 1UL) * baseM;
    const uint64_t nTail = args_.nValue - (nBlockCnt - 1UL) * baseN;
    const uint64_t totalWinCnt = ops::CeilDiv(totalBlockCnt, aicNum);
    const uint64_t tailWinBlockCnt = totalBlockCnt % aicNum;
    const uint64_t windowLen = optiling::GetAswWindowLen(compileInfo_.npuArch, aicNum);
    constexpr uint64_t balanceFixpKThreshold = 1024UL;
    constexpr uint64_t minNBlockCntFixpBalanced = 8UL;

    uint64_t mBaseTailSplitCnt = 1UL;
    uint64_t nBaseTailSplitCnt = 1UL;
    uint64_t mTailMain = 0UL;
    uint64_t nTailMain = 0UL;

    const uint64_t mBaseTail = args_.mValue % baseM;
    const uint64_t nBaseTail = args_.nValue % baseN;
    const bool isInnerAxisAlign = args_.kValue * args_.aDtypeSize % qmmv3_tiling_const::MTE2_ADDRESS_ALIGN_SIZE == 0UL;
    const bool balanceAfterFixp = args_.kValue < balanceFixpKThreshold ||
                                  (args_.kValue == balanceFixpKThreshold && nBlockCnt >= minNBlockCntFixpBalanced);
    if (mBlockCnt != 1UL && nBlockCnt != 1UL) {
        if (mBaseTail > 0UL && (isInnerAxisAlign || args_.mValue >= qmmv3_tiling_const::LOAD_BALANCE_THRESHOLD)) {
            const uint64_t baseTailCntMax = std::min((baseM - mBaseTail) / BASIC_BLOCK_SIZE_16, mBlockCnt);
            const uint64_t windowSize = std::min(windowLen, mBlockCnt);
            const uint64_t mainWindowNum = mBlockCnt / windowSize - 1UL;
            const uint64_t tailWindowSize = mBlockCnt - mainWindowNum * windowSize;
            uint64_t perfRes = (mainWindowNum + 1UL) * baseM;
            uint64_t mergeWindowNum = 1UL;
            for (uint64_t mergeLen = tailWindowSize - 1UL; mergeLen < baseTailCntMax;
                 mergeLen += windowSize, ++mergeWindowNum) {
                const uint64_t newTailMain = MathUtil::Align(
                    MathUtil::CeilDivision(mergeLen * baseM + mBaseTail, mergeLen + 1UL), BASIC_BLOCK_SIZE_16);
                const uint64_t curPerf = (mainWindowNum + 1UL - mergeWindowNum) * baseM + mergeWindowNum * newTailMain;
                if (curPerf <= perfRes) {
                    perfRes = curPerf;
                    mTailMain = newTailMain;
                    mBaseTailSplitCnt = mergeLen + 1UL;
                }
            }
        }
        if (nBaseTail > 0UL && args_.isBTrans && !balanceAfterFixp &&
            (isInnerAxisAlign || args_.nValue >= qmmv3_tiling_const::LOAD_BALANCE_THRESHOLD)) {
            const uint64_t baseTailCntMax = std::min((baseN - nBaseTail) / BASIC_BLOCK_SIZE_16, nBlockCnt);
            const uint64_t totalWindows = MathUtil::CeilDivision(nBlockCnt * mBlockCnt, aicNum);
            uint64_t mainWindows = MathUtil::CeilDivision((nBlockCnt - 1UL) * mBlockCnt + mBlockCnt % aicNum, aicNum);
            if (nBlockCnt * mBlockCnt <= aicNum ||
                (mBlockCnt % aicNum == 0UL && (nBlockCnt % windowLen == 0UL || windowLen % nBlockCnt == 0UL))) {
                mainWindows = totalWindows;
            }
            const uint64_t tailWindows = totalWindows - mainWindows;
            uint64_t perfRes = mainWindows * baseN + tailWindows * nBaseTail;
            for (uint64_t mergeLen = 1UL; mergeLen < baseTailCntMax; ++mergeLen) {
                const uint64_t newTailMain = MathUtil::Align(
                    MathUtil::CeilDivision(mergeLen * baseN + nBaseTail, mergeLen + 1UL), BASIC_BLOCK_SIZE_16);
                const uint64_t newMainRound = (mergeLen < nBlockCnt - 1UL) ?
                                                  MathUtil::CeilDivision((nBlockCnt - 1UL - mergeLen) * mBlockCnt +
                                                                             (mergeLen + 1UL) * mBlockCnt % aicNum,
                                                                         aicNum) :
                                                  0UL;
                const uint64_t newTailRound = (mergeLen > 0UL) ?
                                                  std::min(MathUtil::CeilDivision(
                                                               mergeLen * mBlockCnt + mBlockCnt % aicNum, aicNum),
                                                           totalWindows - newMainRound) :
                                                  0UL;
                const uint64_t newTailLast = mergeLen * (baseN - newTailMain) + nBaseTail;
                const uint64_t curPerf = newMainRound * baseN + newTailRound * newTailMain +
                                         (totalWindows - newMainRound - newTailRound) * newTailLast;
                if (curPerf < perfRes) {
                    perfRes = curPerf;
                    nTailMain = newTailMain;
                    nBaseTailSplitCnt = mergeLen + 1UL;
                }
            }
        }
    }

    uint32_t mTailTile = 1UL;
    uint32_t nTailTile = 1UL;
    if (tailWinBlockCnt != 0UL) {
        const bool isPreSplitM = mTail >= nTail;
        const uint64_t tileMax = aicNum / tailWinBlockCnt;
        const uint64_t tailBaseM = mBaseTailSplitCnt != 1UL ? mTailMain : baseM;
        const uint64_t mTailSplitSize = std::min(args_.mValue, tailBaseM);
        const uint64_t nTailSplitSize = std::min(args_.nValue, baseN);
        const auto getSplitAlignNum = [&args = args_](bool isMSplit) -> uint64_t {
            return isMSplit ? qmmv3_tiling_const::CUBE_BLOCK :
                              (args.isBTrans ? qmmv3_tiling_const::CUBE_BLOCK :
                                               qmmv3_tiling_const::L1_ALIGN_SIZE / args.bDtypeSize);
        };
        const uint64_t mTileMax = std::min(tileMax, MathUtil::CeilDivision(mTailSplitSize, getSplitAlignNum(true)));
        const uint64_t nTileMax = std::min(tileMax, MathUtil::CeilDivision(nTailSplitSize, getSplitAlignNum(false)));
        const uint64_t preSplitMax = isPreSplitM ? mTileMax : nTileMax;
        const uint64_t secSplitMax = isPreSplitM ? nTileMax : mTileMax;
        const uint64_t preSplitSize = isPreSplitM ? mTailSplitSize : nTailSplitSize;
        const uint64_t secSplitSize = isPreSplitM ? nTailSplitSize : mTailSplitSize;

        uint64_t preSplit = 1UL;
        uint64_t secSplit = 1UL;
        uint64_t preSplitValid = 1UL;
        uint64_t secSplitValid = 1UL;
        uint64_t preSplitAlignValid = 1UL;
        uint64_t secSplitAlignValid = 1UL;
        const auto canIncreaseTailSplit = [&](bool isPreSplit) -> bool {
            const uint64_t nextPreSplit = isPreSplit ? preSplit + 1UL : preSplit;
            const uint64_t nextSecSplit = isPreSplit ? secSplit : secSplit + 1UL;
            const uint64_t mTile = isPreSplitM ? nextPreSplit : nextSecSplit;
            const uint64_t nTile = isPreSplitM ? nextSecSplit : nextPreSplit;
            return (isPreSplit ? preSplit : secSplit) < (isPreSplit ? preSplitMax : secSplitMax) &&
                   mTile * nTile * tailWinBlockCnt <= aicNum;
        };
        const auto getTailSplitState = [&](bool isPreSplit, uint64_t split) -> uint64_t {
            const bool isMSplit = isPreSplit == isPreSplitM;
            const bool needWeightNzCheck = args_.bFormat == ge::FORMAT_FRACTAL_NZ && !isMSplit;
            if (needWeightNzCheck) {
                if (!((isPreSplit && mTail >= nTail) || (!isPreSplit && mTail < nTail))) {
                    const uint64_t tailN = baseN / split;
                    if (tailN % (qmmv3_tiling_const::L1_ALIGN_SIZE / args_.bDtypeSize) != 0UL) {
                        return 0UL;
                    }
                }
            }
            const bool isInnerAxis = !isMSplit && !args_.isBTrans;
            if (!isInnerAxis || totalWinCnt != 1UL) {
                return 1UL;
            }
            const uint64_t splitAlignNum = isMSplit ? BASIC_BLOCK_SIZE_32 / args_.aDtypeSize :
                                                      BASIC_BLOCK_SIZE_32 / args_.bDtypeSize;
            return MathUtil::CeilDivision(isMSplit ? mTailSplitSize : nTailSplitSize, split) % splitAlignNum == 0UL ?
                       2UL :
                       1UL;
        };
        while (canIncreaseTailSplit(true) || canIncreaseTailSplit(false)) {
            if (canIncreaseTailSplit(true)) {
                ++preSplit;
                const uint64_t splitState = getTailSplitState(true, preSplit);
                if (splitState > 0UL) {
                    preSplitValid = preSplit;
                }
                if (splitState > 1UL) {
                    preSplitAlignValid = preSplit;
                }
            }
            if (canIncreaseTailSplit(false)) {
                ++secSplit;
                const uint64_t splitState = getTailSplitState(false, secSplit);
                if (splitState > 0UL) {
                    secSplitValid = secSplit;
                }
                if (splitState > 1UL) {
                    secSplitAlignValid = secSplit;
                }
            }
        }
        const uint64_t preTile = preSplitAlignValid != 1UL ? preSplitAlignValid : preSplitValid;
        const uint64_t secTile = secSplitAlignValid != 1UL ? secSplitAlignValid : secSplitValid;
        if (isPreSplitM) {
            mTailTile = static_cast<uint32_t>(preTile);
            nTailTile = static_cast<uint32_t>(secTile);
        } else {
            mTailTile = static_cast<uint32_t>(secTile);
            nTailTile = static_cast<uint32_t>(preTile);
        }
    }

    runInfo_.mBaseTailSplitCnt = mBaseTailSplitCnt;
    runInfo_.nBaseTailSplitCnt = nBaseTailSplitCnt;
    runInfo_.tailInfo.mTailMain = mTailMain;
    runInfo_.tailInfo.nTailMain = nTailMain;
    mTailTile_ = mTailTile;
    nTailTile_ = nTailTile;
    runInfo_.usedCoreNum = (totalWinCnt > 1UL || tailWinBlockCnt == 0UL) ? aicNum :
                                                                           tailWinBlockCnt * mTailTile * nTailTile;
    isMxTailWindowed_ = true;
}

void TransposeQuantBatchMatMulAswTiling::ApplyQbmmMxL1Tiling(const QuantBatchMatmulInfo& qbmmParams,
                                                             const QuantBatchMatmulV3CompileInfo& qbmmCompileInfo)
{
    constexpr uint64_t maxScaleBaseNWithDb = 256UL;
    const bool isUbQuant = args_.cType == ge::DT_BF16;
    const bool isScaleInvalid = !isUbQuant && runInfo_.baseN > maxScaleBaseNWithDb;
    const bool l0cEnough = runInfo_.baseM * runInfo_.baseN * qmmv3_tiling_const::DATA_SIZE_L0C *
                               qmmv3_tiling_const::DOUBLE_BUFFER_NUM <=
                           qbmmCompileInfo.l0cSize;
    runInfo_.dbL0C = (l0cEnough && !isScaleInvalid) ? qmmv3_tiling_const::DOUBLE_BUFFER_NUM : 1UL;

    L1TilingDataCalculator l1Calculator(qbmmParams, qbmmCompileInfo, runInfo_.baseM, runInfo_.baseN, runInfo_.baseK);
    if (!l1Calculator.Compute(L1TilingMode::DEFAULT)) {
        OP_LOGD(args_.opName, "qbmm L1 tiling failed, fallback to legacy CalL1Tiling path.");
        CalL1Tiling();
        return;
    }
    const L1TilingData& l1TilingData = l1Calculator.GetOutput();
    runInfo_.stepKa = l1TilingData.stepKa_;
    runInfo_.stepKb = l1TilingData.stepKb_;
    runInfo_.depthA1 = l1TilingData.depthKa_;
    runInfo_.depthB1 = l1TilingData.depthKb_;
    scaleFactorA_ = static_cast<uint32_t>(l1TilingData.scaleFactorA_);
    scaleFactorB_ = static_cast<uint32_t>(l1TilingData.scaleFactorB_);
    runInfo_.singleCoreM = runInfo_.baseM;
    runInfo_.singleCoreN = runInfo_.baseN;

    const uint64_t kL1 = std::min(l1TilingData.stepKa_, l1TilingData.stepKb_) * runInfo_.baseK;
    scaleKL1_ = static_cast<uint32_t>(
        std::min(l1TilingData.scaleFactorA_ * l1TilingData.stepKa_, l1TilingData.scaleFactorB_ * l1TilingData.stepKb_) *
        runInfo_.baseK);

    const uint64_t scaleGroupSize = qmmv3_tiling_const::MX_GROUP_SIZE;
    auto getSizeWithDtype = [](uint64_t shape, ge::DataType dtype) -> uint64_t {
        return shape * static_cast<uint64_t>(ge::GetSizeByDataType(dtype));
    };
    uint64_t usedL1Size = getSizeWithDtype(runInfo_.baseN * kL1, qbmmParams.bDtype) *
                          qmmv3_tiling_const::L1_FOUR_BUFFER;
    usedL1Size += getSizeWithDtype(runInfo_.baseN * ops::CeilDiv(static_cast<uint64_t>(scaleKL1_), scaleGroupSize),
                                   qbmmParams.scaleDtype) *
                  qmmv3_tiling_const::DOUBLE_BUFFER_NUM;
    usedL1Size += getSizeWithDtype(runInfo_.baseM * kL1, qbmmParams.aDtype) * qmmv3_tiling_const::L1_FOUR_BUFFER;
    usedL1Size += getSizeWithDtype(runInfo_.baseM * ops::CeilDiv(static_cast<uint64_t>(scaleKL1_), scaleGroupSize),
                                   qbmmParams.perTokenScaleDtype) *
                  qmmv3_tiling_const::DOUBLE_BUFFER_NUM;
    runInfo_.l1BufferNum = usedL1Size < qbmmCompileInfo.l1Size ? qmmv3_tiling_const::L1_FOUR_BUFFER :
                                                                 qmmv3_tiling_const::L1_TWO_BUFFER;
    isMxL1Tiled_ = true;
}

void TransposeQuantBatchMatMulAswTiling::CalL1Tiling()
{
    uint64_t leftL1Size = compileInfo_.l1Size;
    if (args_.hasBias) {
        leftL1Size -= runInfo_.baseN * ge::GetSizeByDataType(args_.biasType);
    }
    const bool isHalfByteData = isMXFP4_ || isHIFP4_;
    uint64_t baseASize = isHalfByteData ? ((runInfo_.baseM * runInfo_.baseK + 1) >> 1) :
                                          runInfo_.baseM * runInfo_.baseK * args_.aDtypeSize;
    uint64_t baseBSize = isHalfByteData ? ((runInfo_.baseN * runInfo_.baseK + 1) >> 1) :
                                          runInfo_.baseN * runInfo_.baseK * args_.bDtypeSize;
    bool hasMXScale = IsMicroScaling(context_->GetOptionalInputDesc(SCALE_X1_IDX),
                                     context_->GetOptionalInputDesc(SCALE_X2_IDX)) ||
                      isHIFP4_;
    uint64_t baseScaleK = isHIFP4_ ? ops::CeilDiv(static_cast<uint64_t>(runInfo_.baseK), HIF4_GROUP_SIZE) :
                                     ops::CeilAlign(ops::CeilDiv(static_cast<uint64_t>(runInfo_.baseK), MX_GROUP_SIZE),
                                                    MXFP_MULTI_BASE_SIZE);
    uint64_t baseScaleASize = !hasMXScale ? 0 :
                                            baseScaleK * runInfo_.baseM *
                                                ge::GetSizeByDataType(
                                                    context_->GetOptionalInputDesc(SCALE_X1_IDX)->GetDataType());
    uint64_t baseScaleBSize = !hasMXScale ? 0 :
                                            baseScaleK * runInfo_.baseN *
                                                ge::GetSizeByDataType(
                                                    context_->GetOptionalInputDesc(SCALE_X2_IDX)->GetDataType());
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
    if (hasMXScale) {
        CalScaleFactors(baseASize, baseBSize, baseScaleASize, baseScaleBSize);
    }
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
    uint32_t scaleFactorAMax = std::min(static_cast<uint32_t>(ops::FloorDiv(MTE2_MIN_LOAD_SIZE_V100, baseScaleASize)),
                                        SCALER_FACTOR_MAX);
    uint32_t scaleFactorBMax = std::min(static_cast<uint32_t>(ops::FloorDiv(MTE2_MIN_LOAD_SIZE_V100, baseScaleBSize)),
                                        SCALER_FACTOR_MAX);
    uint32_t scaleFactorA = static_cast<uint32_t>(args_.kValue) / (runInfo_.stepKa * runInfo_.baseK);
    uint32_t scaleFactorB = static_cast<uint32_t>(args_.kValue) / (runInfo_.stepKb * runInfo_.baseK);
    scaleFactorA_ = std::max(SCALER_FACTOR_MIN, scaleFactorA);
    scaleFactorB_ = std::max(SCALER_FACTOR_MIN, scaleFactorB);
    scaleFactorA_ = std::min(scaleFactorAMax, scaleFactorA);
    scaleFactorB_ = std::min(scaleFactorBMax, scaleFactorB);

    // 来自L1 size 的约束
    uint64_t leftL1sie = compileInfo_.l1Size -
                         (runInfo_.depthA1 * baseASize + runInfo_.depthB1 * baseBSize + baseBiasSize);
    uint32_t scaleInit = static_cast<uint32_t>(leftL1sie /
                                               (runInfo_.depthA1 * baseScaleASize + runInfo_.depthB1 * baseScaleBSize));
    if (scaleFactorA_ <= scaleInit && scaleFactorB_ > scaleInit) {
        leftL1sie -= (static_cast<uint64_t>(scaleFactorA_) * runInfo_.depthA1 * baseScaleASize);
        scaleFactorB_ = std::min(static_cast<uint32_t>(leftL1sie / (runInfo_.depthB1 * baseScaleBSize)), scaleFactorB_);
    } else if (scaleFactorB_ <= scaleInit && scaleFactorA_ > scaleInit) {
        leftL1sie -= (static_cast<uint64_t>(scaleFactorB_) * runInfo_.depthB1 * baseScaleBSize);
        scaleFactorA_ = std::min(static_cast<uint32_t>(leftL1sie / (runInfo_.depthA1 * baseScaleASize)), scaleFactorA_);
    } else if (scaleFactorA_ > scaleInit && scaleFactorB_ > scaleInit) {
        leftL1sie -= (static_cast<uint64_t>(scaleInit) * runInfo_.depthB1 * baseScaleBSize +
                      static_cast<uint64_t>(scaleInit) * runInfo_.depthA1 * baseScaleASize);
        uint32_t scaleASec = std::min(static_cast<uint32_t>(leftL1sie / (runInfo_.depthA1 * baseScaleASize)),
                                      scaleFactorA_ - scaleInit);
        uint32_t scaleBSec = std::min(static_cast<uint32_t>(leftL1sie / (runInfo_.depthB1 * baseScaleBSize)),
                                      scaleFactorB_ - scaleInit);
        scaleFactorA_ = scaleASec >= scaleBSec ? (scaleASec + scaleInit) : scaleInit;
        scaleFactorB_ = scaleASec < scaleBSec ? (scaleBSec + scaleInit) : scaleInit;
    }
}

void TransposeQuantBatchMatMulAswTiling::CalStepKs()
{
    runInfo_.stepKa = std::max(runInfo_.depthA1 / DB_SIZE, 1UL);
    runInfo_.stepKb = std::max(runInfo_.depthB1 / DB_SIZE, 1UL);

    if (static_cast<uint64_t>(runInfo_.stepKa * runInfo_.baseK) > args_.kValue) {
        runInfo_.stepKa = ops::CeilDiv(args_.kValue, static_cast<uint64_t>(runInfo_.baseK));
    }

    if (static_cast<uint64_t>(runInfo_.stepKb * runInfo_.baseK) > args_.kValue) {
        runInfo_.stepKb = ops::CeilDiv(args_.kValue, static_cast<uint64_t>(runInfo_.baseK));
    }

    if (runInfo_.stepKa > runInfo_.stepKb) {
        runInfo_.stepKa = ops::FloorAlign(runInfo_.stepKa, runInfo_.stepKb);
    }
    if (runInfo_.stepKb > runInfo_.stepKa) {
        runInfo_.stepKb = ops::FloorAlign(runInfo_.stepKb, runInfo_.stepKa);
    }
    runInfo_.stepKa = std::min(runInfo_.stepKa, static_cast<uint64_t>(4)); // 限制stepKa最大为4, 防止issue queue阻塞
    runInfo_.stepKb = std::min(runInfo_.stepKb, static_cast<uint64_t>(4)); // 限制stepKb最大为4, 防止issue queue阻塞
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
        uint64_t baseKSize = (isMXFP4_ || isHIFP4_) ? ((static_cast<uint64_t>(runInfo_.baseK) + 1) >> 1) :
                                                      static_cast<uint64_t>(runInfo_.baseK) * args_.aDtypeSize;
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
                              static_cast<uint64_t>(batchSplitMode_), static_cast<uint64_t>(precisionMode_),
                              static_cast<uint64_t>(apiLevel_));
}

ge::graphStatus TransposeQuantBatchMatMulAswTiling::GetTilingData(TilingResult& tiling) const
{
    return GetTilingDataImpl<BatchMatMulV3TilingData>(tiling);
}

ge::graphStatus TransposeQuantBatchMatMulAswTiling::GetTilingDataProcess(BatchMatMulV3TilingData& tilingData) const
{
    ge::graphStatus ret = MatMulV3BaseTiling::GetTilingDataProcess(tilingData);
    auto x1Scale = context_->GetOptionalInputDesc(SCALE_X1_IDX);
    bool isMxScale = x1Scale != nullptr && (x1Scale->GetDataType() == ge::DT_FLOAT8_E8M0 ||
                                            x1Scale->GetDataType() == Ops::NN::DT_HIFLOAT4_SCALE);
    if (isMxScale) {
        tilingData.matMulTilingData.tCubeTiling.mxTypePara = (SCALER_FACTOR_MIN << SCALER_FACTOR_N_BIT) +
                                                             (SCALER_FACTOR_MIN << SCALER_FACTOR_M_BIT);
        if (x1Scale->GetDataType() == ge::DT_FLOAT8_E8M0 && scaleFactorA_ >= SCALER_FACTOR_MIN &&
            scaleFactorA_ <= SCALER_FACTOR_MAX && scaleFactorB_ >= SCALER_FACTOR_MIN &&
            scaleFactorB_ <= SCALER_FACTOR_MAX) {
            tilingData.matMulTilingData.tCubeTiling.mxTypePara += (scaleFactorB_ << SCALER_FACTOR_B_BIT) +
                                                                  scaleFactorA_;
        } else {
            tilingData.matMulTilingData.tCubeTiling.mxTypePara += (SCALER_FACTOR_DEFAULT << SCALER_FACTOR_B_BIT) +
                                                                  SCALER_FACTOR_DEFAULT;
        }
    }
    if (isMxScale) {
        // 950/960 共用 Blaze MX kernel，两架构都消费 scaleKL1，不可按 npuArch 隔离写入；
        // 未走 qbmm L1 tiling 时兜底值必须等于 kL1（scaleFactor=1），直接引用基类写入的
        // tilingData.kL1，避免重复推导公式与基类 kL1 的 min+钳4 定义产生偏差。
        tilingData.scaleKL1 = isMxL1Tiled_ ? scaleKL1_ : tilingData.kL1;
    }
    if (isMxTailWindowed_) {
        tilingData.matMulTilingData.mTailCnt = mTailTile_;
        tilingData.matMulTilingData.nTailCnt = nTailTile_;
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

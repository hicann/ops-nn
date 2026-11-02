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
 * \file tiling_cost_evaluator.cpp
 * \brief
 */
#include "tiling_cost_evaluator.h"
#include "common/inc/op_host/op_const_def_nn.h"
#include "matmul/common/op_host/math_util_nn.h"
#include "quant_batch_matmul_v3_tiling_util.h"
#include <algorithm>
#include <cerrno>
#include <cstdlib>

namespace optiling {

using Ops::NN::MathUtil;

static constexpr uint64_t DATA_SIZE_FP32 = 4UL;
static constexpr double SECONDS_TO_US = 1e6;

// Tiling cost evaluation constants
static constexpr uint64_t FIXP_ALIGN_SIZE = 128UL;
static constexpr double FIXP_ALIGN_PENALTY_FACTOR = 0.6;      // baseN/N非128B对齐，强惩罚
static constexpr double FIXP_TAIL_ALIGN_PENALTY_FACTOR = 0.8; // 仅尾块n非128B对齐，弱惩罚
static constexpr double VECTOR_HEAD_COST_US = 0.6;
static constexpr int64_t VEC_CAL_BYTES_PER_CYCLE = 256;
static constexpr double MXFP4_PER_GROUP_SCALE_RATIO = 1.0 / 16;
static constexpr double MXFP8_PER_GROUP_SCALE_RATIO = 1.0 / 32;
static constexpr double HIF4_PER_GROUP_SCALE_RATIO = 1.0 / 8;
static constexpr uint64_t SCALE_GROUP_BYTES_RATIO_MX = 32UL;
static constexpr uint64_t SCALE_GROUP_BYTES_RATIO_HIF4 = 16UL;

// MTE2 burst对齐粒度，非对齐时产生拆包导致带宽折损
static constexpr uint64_t MTE2_BURST_ALIGN_SIZE = 128UL;
static constexpr double MTE2_UNALIGNED_PENALTY = 0.85; // 拆包惩罚因子

// MTE1 LoadData 拆分相关常量（参考 block_mmad_mx.h）
static constexpr uint64_t BLOCK_CUBE = 16UL;
static constexpr uint64_t B4_MIN_STEP = 4UL;    // FP4类型最小搬运步长，单位为BLOCK_CUBE
static constexpr uint64_t B4_SPLIT_BASE = 64UL; // B4_MIN_STEP * BLOCK_CUBE，baseM/baseN为64倍数时不拆

TilingCostEvaluator::TilingCostEvaluator(const SocSpecifyParams& socParmas) : socParams_(socParmas) {}

double TilingCostEvaluator::GetRealL2Bw(const CaseInfo& caseInfo, const QuantBatchMatmulInfo& inputInfo,
                                        const QuantBatchMatmulV3CompileInfo& compileInfo)
{
    // 单核峰值带宽由总带宽按满核数换算，核数不足时按比例缩减
    double perCoreBw = socParams_.l2Bw / static_cast<double>(compileInfo.aicNum);
    double realBw = std::min(socParams_.l2Bw, perCoreBw * static_cast<double>(caseInfo.usedCoreNum));

    // 拆包惩罚：K为内轴时kL1非128B对齐，K为外轴时baseM/baseN非128B对齐
    bool misalignedKInner = false;
    bool misalignedKouterA = false;
    bool misalignedKouterB = false;
    if (!inputInfo.transA || inputInfo.transB) {
        misalignedKInner = GetSizeWithDataType(caseInfo.kL1, inputInfo.aDtype) % MTE2_BURST_ALIGN_SIZE != 0 ||
                           GetSizeWithDataType(caseInfo.kSize, inputInfo.aDtype) % MTE2_BURST_ALIGN_SIZE != 0;
    } else if (inputInfo.transA) {
        misalignedKouterA = GetSizeWithDataType(caseInfo.baseM, inputInfo.aDtype) % MTE2_BURST_ALIGN_SIZE != 0 ||
                            GetSizeWithDataType(caseInfo.mSize, inputInfo.aDtype) % MTE2_BURST_ALIGN_SIZE != 0;
    } else if (!inputInfo.transB) {
        misalignedKouterB = GetSizeWithDataType(caseInfo.baseN, inputInfo.bDtype) % MTE2_BURST_ALIGN_SIZE != 0 ||
                            GetSizeWithDataType(caseInfo.nSize, inputInfo.bDtype) % MTE2_BURST_ALIGN_SIZE != 0;
    }
    if (misalignedKInner) {
        realBw *= MTE2_UNALIGNED_PENALTY;
    }

    if (misalignedKouterA || misalignedKouterB) {
        realBw *= MTE2_UNALIGNED_PENALTY;
    }
    return realBw;
}

double TilingCostEvaluator::GetRealHbmBw(const CaseInfo& caseInfo, const QuantBatchMatmulInfo& inputInfo,
                                         const QuantBatchMatmulV3CompileInfo& compileInfo)
{
    // 单核峰值带宽由总带宽按满核数换算，核数不足时按比例缩减
    double perCoreBw = socParams_.hbmBw / static_cast<double>(compileInfo.aicNum);
    double realBw = std::min(socParams_.hbmBw, perCoreBw * static_cast<double>(caseInfo.usedCoreNum));

    // 拆包惩罚：K为内轴时kL1非128B对齐，K为外轴时baseM/baseN非128B对齐
    bool misalignedKInner = false;
    bool misalignedKouterA = false;
    bool misalignedKouterB = false;
    if (!inputInfo.transA || inputInfo.transB) {
        misalignedKInner = GetSizeWithDataType(caseInfo.kL1, inputInfo.aDtype) % MTE2_BURST_ALIGN_SIZE != 0 ||
                           GetSizeWithDataType(caseInfo.kSize, inputInfo.aDtype) % MTE2_BURST_ALIGN_SIZE != 0;
    } else if (inputInfo.transA) {
        misalignedKouterA = GetSizeWithDataType(caseInfo.baseM, inputInfo.aDtype) % MTE2_BURST_ALIGN_SIZE != 0 ||
                            GetSizeWithDataType(caseInfo.mSize, inputInfo.aDtype) % MTE2_BURST_ALIGN_SIZE != 0;
    } else if (!inputInfo.transB) {
        misalignedKouterB = GetSizeWithDataType(caseInfo.baseN, inputInfo.bDtype) % MTE2_BURST_ALIGN_SIZE != 0 ||
                            GetSizeWithDataType(caseInfo.nSize, inputInfo.bDtype) % MTE2_BURST_ALIGN_SIZE != 0;
    }
    if (misalignedKInner) {
        realBw *= MTE2_UNALIGNED_PENALTY;
    }

    if (misalignedKouterA || misalignedKouterB) {
        realBw *= MTE2_UNALIGNED_PENALTY;
    }
    return realBw;
}

void TilingCostEvaluator::EsitmateMmadTime(const CaseInfo& caseInfo, const QuantBatchMatmulInfo& inputInfo,
                                           AicTimeCost& res)
{
    uint64_t mCnt = MathUtil::CeilDivision(caseInfo.mSize, caseInfo.baseM);
    uint64_t nCnt = MathUtil::CeilDivision(caseInfo.nSize, caseInfo.baseN);
    uint64_t totalTiles = mCnt * nCnt;
    uint64_t roundPerCore = MathUtil::CeilDivision(totalTiles, caseInfo.usedCoreNum);

    uint64_t baseMFracM = MathUtil::CeilDivision(caseInfo.baseM, socParams_.mmadFracM);
    uint64_t baseNFracN = MathUtil::CeilDivision(caseInfo.baseN, socParams_.mmadFracN);
    uint64_t baseKFracK = MathUtil::CeilDivision(GetSizeWithDataType(caseInfo.baseK, inputInfo.aDtype),
                                                 socParams_.mmadFracK);
    double freqMhz = static_cast<double>(socParams_.freq * GHZ_TO_MHZ);
    double oneBlockMmadtime = static_cast<double>(baseMFracM * baseNFracN * baseKFracK) / freqMhz;

    uint64_t fullKIter = caseInfo.kSize / caseInfo.baseK;
    uint64_t kTail = caseInfo.kSize % caseInfo.baseK;
    uint64_t kTailFracK = 0UL;
    double oneRoundMmadTime = oneBlockMmadtime * static_cast<double>(fullKIter);
    if (kTail != 0UL) {
        kTailFracK = MathUtil::CeilDivision(GetSizeWithDataType(kTail, inputInfo.aDtype), socParams_.mmadFracK);
        double tailBlockMmadtime = static_cast<double>(baseMFracM * baseNFracN * kTailFracK) / freqMhz;
        oneRoundMmadTime += tailBlockMmadtime;
    }

    bool mHasTail = (caseInfo.mSize % caseInfo.baseM != 0);
    bool nHasTail = (caseInfo.nSize % caseInfo.baseN != 0);
    uint64_t mTail = mHasTail ? (caseInfo.mSize % caseInfo.baseM) : caseInfo.baseM;
    uint64_t nTail = nHasTail ? (caseInfo.nSize % caseInfo.baseN) : caseInfo.baseN;

    // 尾轮 tile 范围：单轮时全部 tile 都在尾轮，多轮时仅最后一轮
    uint64_t fullRoundTiles = (roundPerCore > 1UL) ? (roundPerCore - 1UL) * caseInfo.usedCoreNum : 0UL;
    double maxLastRoundTime = 0;
    for (uint64_t t = fullRoundTiles; t < totalTiles; t++) {
        uint64_t mi = t / nCnt;
        uint64_t ni = t % nCnt;
        uint64_t mDim = (mi == mCnt - 1UL && mHasTail) ? mTail : caseInfo.baseM;
        uint64_t nDim = (ni == nCnt - 1UL && nHasTail) ? nTail : caseInfo.baseN;
        uint64_t tileMFracM = MathUtil::CeilDivision(mDim, socParams_.mmadFracM);
        uint64_t tileNFracN = MathUtil::CeilDivision(nDim, socParams_.mmadFracN);
        double tileTime = static_cast<double>(tileMFracM * tileNFracN * baseKFracK) / freqMhz *
                          static_cast<double>(fullKIter);
        if (kTail != 0UL) {
            tileTime += static_cast<double>(tileMFracM * tileNFracN * kTailFracK) / freqMhz;
        }
        if (tileTime > maxLastRoundTime) {
            maxLastRoundTime = tileTime;
        }
    }
    res.mmadLastRoundMaxtime = maxLastRoundTime;

    if (roundPerCore <= 1UL) {
        res.mmadTotalTime = maxLastRoundTime;
    } else {
        res.mmadTotalTime = (roundPerCore - 1UL) * oneRoundMmadTime + maxLastRoundTime;
    }
}

void TilingCostEvaluator::EstimateMte1Time(const CaseInfo& caseInfo, const QuantBatchMatmulInfo& inputInfo,
                                           AicTimeCost& res)
{
    uint64_t mCnt = MathUtil::CeilDivision(caseInfo.mSize, caseInfo.baseM);
    uint64_t nCnt = MathUtil::CeilDivision(caseInfo.nSize, caseInfo.baseN);
    uint64_t totalTiles = mCnt * nCnt;
    uint64_t roundPerCore = MathUtil::CeilDivision(totalTiles, caseInfo.usedCoreNum);

    uint64_t fullKIter = caseInfo.kSize / caseInfo.baseK;
    uint64_t kTail = caseInfo.kSize % caseInfo.baseK;

    // FP4/HIF4 且 K 为外轴时才使用 l0ABwB4KoutterAxis，其余情况均用 l0ABw
    bool isFp4Type = (inputInfo.aDtype == ge::DT_FLOAT4_E2M1 || inputInfo.aDtype == ge::DT_FLOAT4_E1M2 ||
                      inputInfo.aDtype == Ops::NN::DT_HIFLOAT4);
    double aBw = (isFp4Type && inputInfo.transA) ? socParams_.l0BwB4KoutterAxis : socParams_.l0Bw;
    double bBw = (isFp4Type && !inputInfo.transB) ? socParams_.l0BwB4KoutterAxis : socParams_.l0Bw;
    double freqMhz = static_cast<double>(socParams_.freq * GHZ_TO_MHZ);

    // 计算给定 (mDim, nDim, kDim) 单个 tile 的 MTE1 耗时
    auto calcTileMte1Time = [&](uint64_t mDim, uint64_t nDim, uint64_t kDim) -> double {
        double aBytes = GetSizeWithDataType(mDim * kDim, inputInfo.aDtype);
        double bBytes = GetSizeWithDataType(nDim * kDim, inputInfo.aDtype);
        double scaleBytes = 0;
        if (inputInfo.isMxPerGroup) {
            scaleBytes = static_cast<double>((mDim + nDim) * kDim) / SCALE_GROUP_BYTES_RATIO_MX;
        } else if (inputInfo.isHif4PerGroup) {
            scaleBytes = static_cast<double>((mDim + nDim) * kDim) / SCALE_GROUP_BYTES_RATIO_HIF4;
        }
        double time = scaleBytes / socParams_.l0mxBw + aBytes / aBw + bBytes / bBw;
        return time / freqMhz;
    };

    double oneFullBlockMte1time = calcTileMte1Time(caseInfo.baseM, caseInfo.baseN, caseInfo.baseK);
    double oneRoundMte1Time = oneFullBlockMte1time * static_cast<double>(fullKIter);
    if (kTail != 0UL) {
        oneRoundMte1Time += calcTileMte1Time(caseInfo.baseM, caseInfo.baseN, kTail);
    }

    bool mHasTail = (caseInfo.mSize % caseInfo.baseM != 0);
    bool nHasTail = (caseInfo.nSize % caseInfo.baseN != 0);
    uint64_t mTail = mHasTail ? (caseInfo.mSize % caseInfo.baseM) : caseInfo.baseM;
    uint64_t nTail = nHasTail ? (caseInfo.nSize % caseInfo.baseN) : caseInfo.baseN;

    // 尾轮 tile 范围：单轮时全部 tile 都在尾轮，多轮时仅最后一轮
    uint64_t fullRoundTiles = (roundPerCore > 1UL) ? (roundPerCore - 1UL) * caseInfo.usedCoreNum : 0UL;
    double maxLastRoundTime = 0;
    for (uint64_t t = fullRoundTiles; t < totalTiles; t++) {
        uint64_t mi = t / nCnt;
        uint64_t ni = t % nCnt;
        uint64_t mDim = (mi == mCnt - 1UL && mHasTail) ? mTail : caseInfo.baseM;
        uint64_t nDim = (ni == nCnt - 1UL && nHasTail) ? nTail : caseInfo.baseN;
        double tileTime = calcTileMte1Time(mDim, nDim, caseInfo.baseK) * static_cast<double>(fullKIter);
        if (kTail != 0UL) {
            tileTime += calcTileMte1Time(mDim, nDim, kTail);
        }
        if (tileTime > maxLastRoundTime) {
            maxLastRoundTime = tileTime;
        }
    }
    res.mte1LastRoundMaxtime = maxLastRoundTime;

    if (roundPerCore <= 1UL) {
        res.mte1TotalTime = maxLastRoundTime;
    } else {
        res.mte1TotalTime = (roundPerCore - 1UL) * oneRoundMte1Time + maxLastRoundTime;
    }
}

void TilingCostEvaluator::EstimateMte2Time(const CaseInfo& caseInfo, const QuantBatchMatmulInfo& inputInfo,
                                           const QuantBatchMatmulV3CompileInfo& compileInfo, AicTimeCost& res)
{
    // 后继会考虑全载的影响
    uint64_t mCnt = MathUtil::CeilDivision(caseInfo.mSize, caseInfo.baseM);
    uint64_t nCnt = MathUtil::CeilDivision(caseInfo.nSize, caseInfo.baseN);
    uint64_t totalTiles = mCnt * nCnt;
    uint64_t roundPerCore = MathUtil::CeilDivision(totalTiles, caseInfo.usedCoreNum);

    // L2命中：A/B在tile间复用的数据量；HBM命中：A/B首次从HBM加载的独占数据量
    double l2HitBytes = 0;
    double hbmHitBytes = 0;
    if (mCnt > 0 && nCnt > 0) {
        uint64_t lhsM = mCnt * static_cast<int64_t>(caseInfo.baseM);
        uint64_t lhsN = nCnt * static_cast<int64_t>(caseInfo.baseN);
        l2HitBytes = GetSizeWithDataType(
            (((lhsM / caseInfo.baseM - 1) * lhsN + (lhsN / caseInfo.baseN - 1) * lhsM) * inputInfo.kSize),
            inputInfo.aDtype);
        hbmHitBytes = GetSizeWithDataType(((lhsM + lhsN) * inputInfo.kSize), inputInfo.aDtype);
    }

    // per-group scale 额外访存量
    if (inputInfo.isMxPerGroup) {
        if (inputInfo.aDtype == ge::DT_FLOAT4_E2M1) {
            l2HitBytes *= (1 + MXFP4_PER_GROUP_SCALE_RATIO);
            hbmHitBytes *= (1 + MXFP4_PER_GROUP_SCALE_RATIO);
        } else {
            l2HitBytes *= (1 + MXFP8_PER_GROUP_SCALE_RATIO);
            hbmHitBytes *= (1 + MXFP8_PER_GROUP_SCALE_RATIO);
        }
    } else if (inputInfo.isHif4PerGroup) {
        hbmHitBytes *= (1 + HIF4_PER_GROUP_SCALE_RATIO);
        l2HitBytes *= (1 + HIF4_PER_GROUP_SCALE_RATIO);
    }

    // 实际带宽受核数和拆包影响，单位 TB/s -> B/us
    double realL2Bw = GetRealL2Bw(caseInfo, inputInfo, compileInfo) * SECONDS_TO_US;
    double realHbmBw = GetRealHbmBw(caseInfo, inputInfo, compileInfo) * SECONDS_TO_US;

    res.mte2TotalTime = l2HitBytes / realL2Bw + hbmHitBytes / realHbmBw;

    // 尾轮MTE2耗时：单轮时等于总耗时，多轮时按尾轮tile数占比估算
    if (roundPerCore <= 1UL) {
        res.mte2LastRoundMaxtime = res.mte2TotalTime;
    } else {
        uint64_t fullRoundTiles = (roundPerCore - 1UL) * caseInfo.usedCoreNum;
        uint64_t lastRoundTiles = totalTiles - fullRoundTiles;
        res.mte2LastRoundMaxtime = res.mte2TotalTime *
                                   (static_cast<double>(lastRoundTiles) / static_cast<double>(totalTiles));
    }
}

void TilingCostEvaluator::EstimateFixpTime(const CaseInfo& caseInfo, const QuantBatchMatmulInfo& inputInfo,
                                           const QuantBatchMatmulV3CompileInfo& compileInfo, AicTimeCost& res)
{
    uint64_t mCnt = MathUtil::CeilDivision(caseInfo.mSize, caseInfo.baseM);
    uint64_t nCnt = MathUtil::CeilDivision(caseInfo.nSize, caseInfo.baseN);
    uint64_t totalTiles = mCnt * nCnt;
    uint64_t roundPerCore = MathUtil::CeilDivision(totalTiles, caseInfo.usedCoreNum);
    uint64_t outTypeSize = caseInfo.outTypeSize;

    // 理想单核每cycle搬运字节，由l2Bw按满核数换算
    double l2BwInBPerCycle = socParams_.l2Bw * SECONDS_TO_US /
                             (static_cast<double>(socParams_.freq) * compileInfo.aicNum);

    // baseN或N非128B对齐时对fixp带宽影响最大，强惩罚影响所有轮的主块
    bool strongPenalty = (caseInfo.baseN * outTypeSize % FIXP_ALIGN_SIZE != 0) ||
                         (caseInfo.nSize * outTypeSize % FIXP_ALIGN_SIZE != 0);

    // 主块fixp耗时
    double mainBw = l2BwInBPerCycle;
    if (strongPenalty) {
        mainBw *= FIXP_ALIGN_PENALTY_FACTOR;
    }
    double mainFixpTime = caseInfo.baseM * caseInfo.baseN * outTypeSize /
                          (mainBw * static_cast<double>(socParams_.freq));

    // 尾块fixp耗时：仅当N方向有尾块时计算
    bool nHasTail = (caseInfo.nSize % caseInfo.baseN != 0);
    uint64_t nTail = caseInfo.nSize % caseInfo.baseN;
    double tailFixpTime = 0;
    if (nHasTail) {
        double tailBw = l2BwInBPerCycle;
        bool tailUnaligned = (nTail * outTypeSize % FIXP_ALIGN_SIZE != 0);
        if (strongPenalty) {
            // baseN或N非对齐，强惩罚
            tailBw *= FIXP_ALIGN_PENALTY_FACTOR;
        } else if (tailUnaligned) {
            // 主块对齐但尾块n非128B对齐，影响相对较小，弱惩罚
            tailBw *= FIXP_TAIL_ALIGN_PENALTY_FACTOR;
        }
        tailFixpTime = caseInfo.baseM * nTail * outTypeSize / (tailBw * static_cast<double>(socParams_.freq));
    }

    // 每轮取主块和尾块耗时最大值
    double oneRoundFixpTime = std::max(mainFixpTime, tailFixpTime);

    res.fixpipeTotalTime = oneRoundFixpTime * static_cast<double>(roundPerCore);

    if (roundPerCore <= 1UL) {
        res.fixpipeLastRoundMaxtime = res.fixpipeTotalTime;
    } else {
        res.fixpipeLastRoundMaxtime = oneRoundFixpTime;
    }
}

void TilingCostEvaluator::EvaluateAicTime(const CaseInfo& caseInfo, const QuantBatchMatmulInfo& inputInfo,
                                          const QuantBatchMatmulV3CompileInfo& compileInfo, AicTimeCost& res)
{
    // 1. 分段耗时估算
    EsitmateMmadTime(caseInfo, inputInfo, res);
    EstimateMte1Time(caseInfo, inputInfo, res);
    EstimateMte2Time(caseInfo, inputInfo, compileInfo, res);
    EstimateFixpTime(caseInfo, inputInfo, compileInfo, res);

    // 2. 根据分段结果计算流水线总耗时
    uint64_t mCnt = MathUtil::CeilDivision(caseInfo.mSize, caseInfo.baseM);
    uint64_t nCnt = MathUtil::CeilDivision(caseInfo.nSize, caseInfo.baseN);
    uint64_t roundPerCore = MathUtil::CeilDivision(mCnt * nCnt, caseInfo.usedCoreNum);
    double kIter = static_cast<double>(caseInfo.kSize) / static_cast<double>(caseInfo.baseK);
    double kL1d = static_cast<double>(caseInfo.kL1);

    // 从分段结果推导单块/单轮耗时用于流水线分析
    double oneBlockMmadtime = res.mmadTotalTime / (kIter * static_cast<double>(roundPerCore));
    double oneBlockMte1time = res.mte1TotalTime / (kIter * static_cast<double>(roundPerCore));
    double oneRoundFixpTime = res.fixpipeTotalTime / static_cast<double>(roundPerCore);

    // 3. 流水线重叠分析
    bool isMmadlongerMte1 = oneBlockMmadtime > oneBlockMte1time;
    bool isMmadLongerMte2 = res.mmadTotalTime > res.mte2TotalTime;
    bool isMte2longerMte1 = res.mte2TotalTime > oneBlockMte1time * kIter;

    double totalTime = 0;
    double mte2TimeForFirstKl1 = (res.mte2TotalTime / static_cast<double>(roundPerCore)) *
                                 (kL1d / static_cast<double>(caseInfo.kSize));
    if (isMmadLongerMte2 && isMmadlongerMte1) {
        // mmad为瓶颈：mte2首次加载不重叠，后续被mmad掩盖
        totalTime = mte2TimeForFirstKl1 + res.mmadTotalTime;
    } else if (!isMmadLongerMte2 && isMte2longerMte1) {
        // mte2为瓶颈：mmad/mte1在每个kL1块内被mte2掩盖
        totalTime = res.mte2TotalTime +
                    (isMmadlongerMte1 ? oneBlockMmadtime : oneBlockMte1time) * (kL1d / caseInfo.baseK);
    } else if (!isMmadlongerMte1 && !isMte2longerMte1) {
        // mte1为瓶颈：mte2首次加载 + mte1全量 + mmad末尾块
        totalTime = mte2TimeForFirstKl1 + res.mte1TotalTime + oneBlockMmadtime;
    }

    // 4. fixp是否成为瓶颈
    if (oneRoundFixpTime > res.mmadTotalTime / static_cast<double>(roundPerCore)) {
        // fixp单轮耗时超过mmad单轮，fixp成为瓶颈
        totalTime = mte2TimeForFirstKl1 + oneBlockMmadtime * (kL1d / caseInfo.baseK) + res.fixpipeTotalTime;
    } else {
        // fixp可被mmad掩盖，仅追加单轮fixp耗时
        totalTime += oneRoundFixpTime;
    }

    res.totalTime = totalTime;
}

void TilingCostEvaluator::EstimateStreamkAivTime(const CaseInfo& caseInfo, const QuantBatchMatmulInfo& inputInfo,
                                                 const QuantBatchMatmulV3CompileInfo& compileInfo, uint64_t kCnt,
                                                 StreamKTimeCost& res)
{
    uint64_t mCnt = MathUtil::CeilDivision(caseInfo.mSize, caseInfo.baseM);
    uint64_t nCnt = MathUtil::CeilDivision(caseInfo.nSize, caseInfo.baseN);
    uint64_t mnCnt = mCnt * nCnt;
    uint64_t aicNum = static_cast<uint64_t>(compileInfo.aicNum);
    uint64_t aivNum = static_cast<uint64_t>(compileInfo.aivNum);

    // streamK 尾部 tile 数：未能占满一整轮核的 tile，需要跨 K 方向 reduce
    uint64_t tailMnCnt = mnCnt % aicNum;
    if (tailMnCnt == 0UL) {
        tailMnCnt = mnCnt;
    }

    bool mHasTail = (caseInfo.mSize % caseInfo.baseM != 0);
    bool nHasTail = (caseInfo.nSize % caseInfo.baseN != 0);
    uint64_t mTail = mHasTail ? (caseInfo.mSize % caseInfo.baseM) : caseInfo.baseM;
    uint64_t nTail = nHasTail ? (caseInfo.nSize % caseInfo.baseN) : caseInfo.baseN;

    // 参考 block_epilogue_streamk.h 的 UpdateAivParams 逻辑：
    //   taskRation = GetTaskRation() = aivNum / aicNum
    //   curML1InAiv 为该 tile 的实际 m 维度 (尾块为 mTail)
    //   curAlignedNInAiv = CeilAlign(curNL1InAiv, ONE_BLK_SIZE) (元素数, FP32 时 32 元素)
    //   mBurstBase = CeilAlign(CeilDiv(curML1InAiv, kCnt * taskRation),
    //                          CeilDiv(UB2GM_SRCGAP_UNIT=32, curAlignedNInAiv))
    //   mBurstCnt  = CeilDiv(curML1InAiv, mBurstBase)  <- 真正干活的 AIV 任务数
    //   kCntIndex < mBurstCnt 的 AIV 才搬运，其余 mBurst=0 直接返回
    uint64_t taskRation = (aicNum == 0UL) ? 1UL : aivNum / aicNum;
    constexpr uint64_t ONE_BLK_ELEMS_FP32 = 32UL; // AscendC::ONE_BLK_SIZE / sizeof(fp32)
    constexpr uint64_t UB2GM_SRCGAP_UNIT = 32UL;

    uint64_t totalWs2UBBytes = 0UL;
    uint64_t totalParticipatingAiv = 0UL;
    uint64_t maxSKblockPerCore = 0;

    // 这里可以统计每组的
    for (uint64_t i = 0UL; i < tailMnCnt; i++) {
        uint64_t t = mnCnt - 1UL - i;
        uint64_t mi = t / nCnt;
        uint64_t ni = t % nCnt;
        uint64_t mDim = (mi == mCnt - 1UL && mHasTail) ? mTail : caseInfo.baseM;
        uint64_t nDim = (ni == nCnt - 1UL && nHasTail) ? nTail : caseInfo.baseN;

        uint64_t curAlignedN = ops::CeilAlign(nDim, ONE_BLK_ELEMS_FP32);
        uint64_t mAlignUnit = (curAlignedN == 0UL) ? 1UL : MathUtil::CeilDivision(UB2GM_SRCGAP_UNIT, curAlignedN);
        uint64_t perTaskM = MathUtil::CeilDivision(mDim, kCnt * taskRation);
        uint64_t mBurstBase = ops::CeilAlign(perTaskM, mAlignUnit);
        if (mBurstBase == 0UL) {
            mBurstBase = 1UL;
        }
        uint64_t mBurstCnt = MathUtil::CeilDivision(mDim, mBurstBase);
        // 每个 tile 真正干活的 AIV 任务数 = min(mBurstCnt, kCnt * taskRation)
        uint64_t activeAiv = std::min(mBurstCnt, kCnt * taskRation);
        totalParticipatingAiv += activeAiv;

        // 每个 tile 将 kCnt 个 mDim*nDim 的 workspace partial sum 搬入 UB
        uint64_t curBlock = mDim * nDim * DATA_SIZE_FP32;
        totalWs2UBBytes += kCnt * curBlock;

        if (curBlock > maxSKblockPerCore) {
            maxSKblockPerCore = curBlock;
        }
    }

    // 总参与搬运核数不超过 aivNum
    uint64_t participatingAiv = std::min(totalParticipatingAiv, aivNum);
    if (participatingAiv == 0UL) {
        participatingAiv = 1UL;
    }

    // 实际 L2 带宽：按参与核数扩展，上限为总 L2 带宽
    double perAivBw = (aivNum == 0UL) ? socParams_.l2Bw : socParams_.l2Bw / static_cast<double>(aivNum);
    double realBw = std::min(socParams_.l2Bw, perAivBw * static_cast<double>(participatingAiv));

    // baseN 非对齐惩罚：workspace 为 FP32, baseN*FP32 需 128B 对齐，否则 MTE2 拆包导致带宽折损
    if (caseInfo.baseN * DATA_SIZE_FP32 % MTE2_BURST_ALIGN_SIZE != 0UL) {
        realBw *= MTE2_UNALIGNED_PENALTY;
    }
    res.workspace2UBCost = static_cast<double>(totalWs2UBBytes) / (realBw * SECONDS_TO_US);

    // 4. vecCalCost
    res.vecCalCost = static_cast<double>(maxSKblockPerCore) /
                     (VEC_CAL_BYTES_PER_CYCLE * static_cast<double>(socParams_.freq * GHZ_TO_MHZ) *
                      (compileInfo.aivNum / compileInfo.aicNum));

    // 5. ub2GMCost
    uint64_t cDtypeBytes = GetSizeWithDataType(1UL, inputInfo.cDtype);
    uint64_t outTileBytes = totalWs2UBBytes * cDtypeBytes / static_cast<uint64_t>(DATA_SIZE_FP32);
    if (caseInfo.baseN * cDtypeBytes % MTE2_BURST_ALIGN_SIZE != 0UL) {
        realBw *= MTE2_UNALIGNED_PENALTY;
    }
    res.ub2GMCost = static_cast<double>(outTileBytes) / (realBw * SECONDS_TO_US);

    res.aivTotalTime = res.vectorHeadCost + res.workspace2UBCost + res.vecCalCost + res.ub2GMCost;
}

StreamKTimeCost TilingCostEvaluator::EvaluateStreamkCost(const CaseInfo& caseInfo, uint64_t skSingleCoreK,
                                                         const QuantBatchMatmulInfo& inputInfo,
                                                         const QuantBatchMatmulV3CompileInfo& compileInfo)
{
    StreamKTimeCost res;

    uint64_t mCnt = MathUtil::CeilDivision(caseInfo.mSize, caseInfo.baseM);
    uint64_t nCnt = MathUtil::CeilDivision(caseInfo.nSize, caseInfo.baseN);
    uint64_t mnCnt = mCnt * nCnt;
    uint64_t aicNum = static_cast<uint64_t>(compileInfo.aicNum);
    uint64_t kCnt = MathUtil::CeilDivision(caseInfo.kSize, skSingleCoreK);
    uint64_t cDtypeBytes = GetSizeWithDataType(1UL, inputInfo.cDtype);

    // 1. aicCost: DP+SK or pure SK
    if (mnCnt > aicNum) {
        // DP+SK
        int64_t tailMNTileNum = static_cast<int64_t>(mnCnt % aicNum);
        int64_t totalMNTileNumInDP = static_cast<int64_t>(mnCnt) - tailMNTileNum;

        int64_t dpMTileNum = static_cast<int64_t>(mCnt);
        int64_t dpNTileNum = static_cast<int64_t>(nCnt);
        if (tailMNTileNum > 0) {
            dpNTileNum = totalMNTileNumInDP / static_cast<int64_t>(mCnt);
        }

        uint64_t dpMCoveredSize = static_cast<uint64_t>(dpMTileNum) * caseInfo.baseM;
        if (dpMCoveredSize > caseInfo.mSize) {
            dpMCoveredSize = caseInfo.mSize;
        }
        uint64_t dpNCoveredSize = static_cast<uint64_t>(dpNTileNum) * caseInfo.baseN;
        if (dpNCoveredSize > caseInfo.nSize) {
            dpNCoveredSize = caseInfo.nSize;
        }

        CaseInfo dpCaseInfo = caseInfo;
        dpCaseInfo.mSize = dpMCoveredSize;
        dpCaseInfo.nSize = dpNCoveredSize;
        dpCaseInfo.usedCoreNum = aicNum;
        dpCaseInfo.outTypeSize = cDtypeBytes;
        EvaluateAicTime(dpCaseInfo, inputInfo, compileInfo, res.dpAicCost);

        int64_t skUsedCoreNum = std::min(tailMNTileNum * static_cast<int64_t>(kCnt), static_cast<int64_t>(aicNum));
        int64_t skNTileNum = MathUtil::CeilDivision(tailMNTileNum, static_cast<int64_t>(mCnt));
        uint64_t skNCoveredSize = static_cast<uint64_t>(skNTileNum) * caseInfo.baseN;
        if (skNCoveredSize > caseInfo.nSize) {
            skNCoveredSize = caseInfo.nSize;
        }

        CaseInfo skCaseInfo = caseInfo;
        skCaseInfo.nSize = skNCoveredSize;
        skCaseInfo.kSize = skSingleCoreK;
        skCaseInfo.usedCoreNum = static_cast<uint64_t>(skUsedCoreNum);
        skCaseInfo.outTypeSize = DATA_SIZE_FP32;
        EvaluateAicTime(skCaseInfo, inputInfo, compileInfo, res.skAicCost);

        res.aicCost = res.dpAicCost.totalTime + res.skAicCost.totalTime;
    } else {
        // pure SK
        uint64_t skUsedCoreNum = std::min(mnCnt * kCnt, aicNum);
        CaseInfo skCaseInfo = caseInfo;
        skCaseInfo.kSize = skSingleCoreK;
        skCaseInfo.usedCoreNum = skUsedCoreNum;
        skCaseInfo.outTypeSize = DATA_SIZE_FP32;
        EvaluateAicTime(skCaseInfo, inputInfo, compileInfo, res.skAicCost);

        res.dpAicCost = {};
        res.aicCost = res.skAicCost.totalTime;
    }

    // tailMnCnt: tiles that produce workspace partial sums (need cross-K reduce)
    uint64_t tailMnCnt = mnCnt % aicNum;
    if (tailMnCnt == 0UL) {
        tailMnCnt = mnCnt;
    }

    // 2. vectorHeadCost
    res.vectorHeadCost = VECTOR_HEAD_COST_US;
    EstimateStreamkAivTime(caseInfo, inputInfo, compileInfo, kCnt, res);

    // 如果有DP时，SK的aiv部分耗时会被DP 掩盖
    if (res.aivTotalTime <= res.dpAicCost.totalTime) {
        res.totalTime = res.aicCost;
    } else {
        res.totalTime = res.aicCost + res.aivTotalTime;
    }

    return res;
}

AicTimeCost TilingCostEvaluator::Evaluate(const DequantBmm::QuantBatchMatmulV3BasicAPITilingData& tilingData,
                                          const QuantBatchMatmulInfo& inputInfo,
                                          const QuantBatchMatmulV3CompileInfo& compileInfo)
{
    auto& mt = tilingData.matmulTiling;
    uint64_t mCnt = MathUtil::CeilDivision(mt.m, mt.baseM);
    uint64_t nCnt = MathUtil::CeilDivision(mt.n, mt.baseN);
    uint64_t usedCoreNum = std::min(mCnt * nCnt, static_cast<uint64_t>(compileInfo.aicNum));
    uint64_t cDtypeBytes = GetSizeWithDataType(1UL, inputInfo.cDtype);

    CaseInfo caseInfo{mt.m, mt.n, mt.k, mt.baseM, mt.baseN, mt.baseK, usedCoreNum, mt.scaleKL1, cDtypeBytes};
    AicTimeCost res;
    EvaluateAicTime(caseInfo, inputInfo, compileInfo, res);

    return res;
}

} // namespace optiling

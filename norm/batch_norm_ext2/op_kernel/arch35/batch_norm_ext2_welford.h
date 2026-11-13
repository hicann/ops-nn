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
 * \file batch_norm_ext2_welford.h
 * \brief
 */
#ifndef NORM_BATCH_NORM_EXT2_WELFORD_H
#define NORM_BATCH_NORM_EXT2_WELFORD_H

#include "batch_norm_ext2_base.h"
#include "../../norm_common/reduce_common_regbase.h"
namespace BatchNormExt2Ops {
using namespace AscendC;

using AscendC::MicroAPI::LoadDist;
using AscendC::MicroAPI::MaskMergeMode;
using AscendC::MicroAPI::MaskReg;
using AscendC::MicroAPI::RegTensor;
using AscendC::MicroAPI::StoreDist;
using AscendC::Reg::LoadAlign;
using AscendC::Reg::Reduce;
using AscendC::Reg::StoreAlign;

// ===================== 末级部分和的补偿(TwoSum)归约（Welford 专用） =====================
// Welford finalize 的归约分两段：先用二分树把每个通道的 parallelN 个部分和两两折半，
// 最后再把第一个 VL 内的 ≤VL_FP32 个部分和横向求和成标量 mean / M2。后一段是整条归约链的
// 末级，它对 ≤64 个数做一次朴素树形求和，每层一次 fp32 舍入都会原样留在结果上：mean 值域
// 本身很小、随后又要在 (x-mean) 里做近零对消，最敏感；M2 则被 rsqrt 折算进 rstd。把这段改成
// 补偿求和，可把累加误差从 O(log2(N)·ε·Σ|x|) 压到 O(ε·Σ|x|)。该收尾每个通道只跑一次、
// 不在元素热循环里，代价可忽略。
//
// 共享头 norm/norm_common/reduce_common_regbase.h 的 DichotomyAdd 被多个算子复用，不能改；
// 故在本文件内内联一份等价实现：树形折半部分逐位照抄（口径不变），只把最后的 ReduceSum
// 换成补偿归约。函数名带 Welford 前缀，与 full_reduce.h 的同名机制区分，对其它算子零影响。
constexpr static uint16_t WELFORD_VL_LOG2 = 6; // log2(VL_FP32)，VL_FP32 = 64 时即 6 层树
constexpr static float WELFORD_MAX_FINITE_FP32 = 3.4028234663852886e38F;

// (hi, lo) += value：hi 是舍入后的和，lo 累积每次加法被舍掉的低位残差，使 hi + lo 精确等于
// 真实累加和。Knuth TwoSum 求残差，不要求 |hi| >= |value|，与 Neumaier 的分支补偿逐位等价
// （与 full_reduce.h 的同一处理，只是这里每个 lane 是一个独立通道，残差不必跨 lane 合并）。
__simd_callee__ inline void WelfordAccumulateLaneWithError(RegTensor<float>& hi, RegTensor<float>& lo,
                                                           RegTensor<float>& value, MaskReg& mask)
{
    RegTensor<float> total;
    RegTensor<float> back;
    RegTensor<float> error;
    Add(total, hi, value, mask);
    Sub(back, total, hi, mask);
    Sub(error, total, back, mask);
    Sub(error, hi, error, mask);
    Sub(back, value, back, mask);
    Add(error, error, back, mask);
    Add(lo, lo, error, mask);
    AscendC::Reg::Move(hi, total, mask);
}

// 低位残差非有限即置 0：hi 溢出成 ±inf 时 lo 会退化成 inf - inf = NaN，再加回 hi 会把本该是
// inf 的和污染成 NaN。补偿只在有限域有意义，非有限值按 IEEE 自然传播。
__simd_callee__ inline void WelfordClampLaneResidual(RegTensor<float>& lo, MaskReg& mask)
{
    RegTensor<float> zeroReg;
    MaskReg finiteMask;
    Duplicate(zeroReg, 0.0f, mask);
    AscendC::Reg::Compares<float, AscendC::CMPMODE::LE>(finiteMask, lo, WELFORD_MAX_FINITE_FP32, mask);
    AscendC::Reg::Select(lo, lo, zeroReg, finiteMask);
    AscendC::Reg::Compares<float, AscendC::CMPMODE::GE>(finiteMask, lo, -WELFORD_MAX_FINITE_FP32, mask);
    AscendC::Reg::Select(lo, lo, zeroReg, finiteMask);
}

// 把 srcAddr 起前 validNum 个 fp32 部分和补偿归约成 1 个，结果落在 dst 的 lane 0（与原
// ReduceSum 的落点一致，后续 Muls/StoreAlign/DUPLICATE 的口径不变）。validNum ≤ VL_FP32；
// 不是 2 的幂也安全——有效 lane 之外先清零，补 0 对和是精确的（残差也是 0）。
// 实现是成对 TwoSum 树 + 低位残差逐级回灌：每一级的加法残差都留在 lo 里继续参与后续各级，
// 不再丢；树形（而非串行链）让链长仍是 log2(VL_FP32) = 6，不额外拉长依赖。
__simd_callee__ inline void WelfordCompensatedLaneReduce(RegTensor<float>& dst, __local_mem__ float* srcAddr,
                                                         uint32_t validNum)
{
    RegTensor<float> sums;
    RegTensor<float> low;
    RegTensor<float> rhsSums;
    RegTensor<float> rhsLow;
    RegTensor<uint32_t> evenIdx;
    RegTensor<uint32_t> oddIdx;
    MaskReg pregMain = CreateMask<float, MaskPattern::ALL>();
    MaskReg pregValid = UpdateMask<float>(validNum);
    MaskReg pregOne = CreateMask<float, MaskPattern::VL1>();

    AscendC::Reg::DataCopy(sums, srcAddr);
    // 有效 lane 之外是 buffer 上一阶段留下的脏值，必须先清零（x + 0 = x，残差也是 0）。
    ShiftLefts((RegTensor<uint32_t>&)sums, (RegTensor<uint32_t>&)sums, static_cast<int16_t>(0), pregValid);
    Duplicate(low, 0.0f, pregMain);

    // 奇偶 gather 索引：第 i 个 lane 取原向量的第 2i / 2i+1 个，做相邻成对归约。
    AscendC::Reg::Arange((RegTensor<int32_t>&)evenIdx, static_cast<int32_t>(0));
    ShiftLefts(evenIdx, evenIdx, static_cast<int16_t>(1), pregMain);
    Duplicate(oddIdx, static_cast<uint32_t>(VL_FP32 - 1), pregMain);
    AscendC::Reg::And(evenIdx, evenIdx, oddIdx, pregMain);
    Adds(oddIdx, evenIdx, static_cast<uint32_t>(1), pregMain);

#pragma unroll
    for (uint16_t level = 0; level < WELFORD_VL_LOG2; ++level) {
        uint32_t activeNum = VL_FP32 >> (level + 1);
        MaskReg pregActive = UpdateMask<float>(activeNum);
        AscendC::Reg::Gather(rhsSums, sums, oddIdx);
        AscendC::Reg::Gather(rhsLow, low, oddIdx);
        AscendC::Reg::Gather(sums, sums, evenIdx);
        AscendC::Reg::Gather(low, low, evenIdx);
        Add(low, low, rhsLow, pregActive);
        WelfordAccumulateLaneWithError(sums, low, rhsSums, pregActive);
    }
    WelfordClampLaneResidual(low, pregMain);
    // 结果 = s + c，最终只舍入这一次。
    Add(dst, sums, low, pregOne);
}

// 与 NormCommon::DichotomyAdd 逐位等价的二分归约，只把末级的横向 ReduceSum 换成补偿版本。
// 参数含义与原实现完全一致（outerLoop 次折半、innerLoop 个 VL 块、末级 lastNum 个 lane）。
// 读缓冲改用 DataCopy（与原实现一致，不强依赖 VL 对齐），故调用点无需任何额外 UB。
__simd_callee__ inline void WelfordDichotomyAddCompensated(RegTensor<float>& dstReg, __local_mem__ float* src,
                                                           uint16_t outerLoop, uint16_t innerLoop, uint32_t lastNum)
{
    RegTensor<float> tmpReg1;
    RegTensor<float> tmpReg2;
    RegTensor<float> tmpReg3;
    LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
    MaskReg pregMain = CreateMask<float, MaskPattern::ALL>();
    for (uint16_t k = 0; k < outerLoop; k++) {
        innerLoop = innerLoop / NormCommon::DICHOTOMY_ADD_COEFF;
        for (uint16_t i = 0; i < innerLoop; i++) {
            AscendC::Reg::DataCopy(tmpReg1, src + i * VL_FP32);
            AscendC::Reg::DataCopy(tmpReg2, src + (i + innerLoop) * VL_FP32);
            Add(tmpReg3, tmpReg1, tmpReg2, pregMain);
            AscendC::Reg::DataCopy(src + i * VL_FP32, tmpReg3, pregMain);
        }
        LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
    }
    WelfordCompensatedLaneReduce(dstReg, src, lastNum);
}

template <typename T1, typename T2>
class BatchNormExt2Welford {
    static constexpr int32_t FOUR_UNROLL = 4;
    static constexpr int32_t INDEX_0 = 0;
    static constexpr int32_t INDEX_1 = 1;
    static constexpr int32_t INDEX_2 = 2;
    static constexpr int32_t INDEX_3 = 3;

public:
    __aicore__ inline BatchNormExt2Welford(){};
    __aicore__ inline uint32_t RoundUp32(uint32_t x, int32_t size)
    {
        if (size == 0) {
            return x;
        }

        int64_t blockElemNum = BLOCK_SIZE / size;
        return (x + blockElemNum - 1) / blockElemNum * blockElemNum;
    }

    __aicore__ inline BatchNormExt2Welford(const BatchNormExt2WelfordRegbaseTilingData* tilingData, TPipe* pipeIn)
    {
        pipe = pipeIn;
        elemNum = tilingData->elemNum; // all R axis number in one common axis.
        loopR1outer = tilingData->loopR1outer;
        loopR0outer = tilingData->loopR0outer;
        r1Factor = tilingData->r1Factor;
        r0Factor = tilingData->r0Factor;
        aBlockFactor = tilingData->aBlockFactor;
        aGatherLimit = tilingData->aGatherLimit;
        a0 = tilingData->a0;
        r0 = tilingData->r0;
        r1 = tilingData->r1;
        eps = tilingData->epsilon;
        momentum = tilingData->momentum;
        realCoreNum = tilingData->realCoreNum;
        aBlockFactor = tilingData->aBlockFactor;
        numLastCore = tilingData->numLastCore;
        parallelN = tilingData->parallelN;
        ubSize = tilingData->ubSize;
        binaryAddK = tilingData->binaryAddK;
        binaryAddLastNum = tilingData->binaryAddLastNum;
        binaryAddQuotient = tilingData->binaryAddQuotient;
        cutR1OrR0 = tilingData->cutR1OrR0;
        useRunningMeanVar = tilingData->useRunningMeanVar > 0 ? true : false;
    }

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR gamma, GM_ADDR beta, GM_ADDR mean, GM_ADDR var, GM_ADDR y,
                                GM_ADDR mean_out, GM_ADDR var_out, GM_ADDR batch_mean, GM_ADDR batch_var)
    {
        auto blockIdx = GetBlockIdx();

        oneSubMomentum = (float)1.0 - momentum;
        // 方差输出改为「真除法」所需的两个标量：n 与 n-1。
        // n 与 n-1 均小于 2^24，在 fp32 中精确可表示，因此真除法只引入 ≤0.5 ULP 的随机舍入。
        // 原实现把「除以 n（或 n-1）」折叠成预乘倒数常数（reduceScale1 = cofFactor0/n、
        // momentumForVar = momentum*n/(n-1)），这些常数本身无法精确表示，其舍入误差会给
        // 所有通道叠加同向的系统性常数偏差（实测约 -0.3 ULP），是方差输出精度对比失败的主因。
        nValue = static_cast<float>(elemNum);
        nMinusOneValue = static_cast<float>(elemNum - 1);
        isLastCore = (blockIdx == realCoreNum - 1) && (numLastCore != 0);
        processNumPerCore = isLastCore ? numLastCore : aBlockFactor;

        float cofFactor0 = static_cast<float>(FindCofFactor(static_cast<uint32_t>(elemNum)));
        float cofFactor1 = static_cast<float>(FindCofFactor(static_cast<uint32_t>(parallelN)));
        // 真除法后再乘回该 2 的幂（精确），把「先按 1/cofFactor0 缩放累加」还原成除以 n / n-1。
        cofFactor0Value = cofFactor0;
        reduceScale0 = float(1.0) / cofFactor0;
        scale0 = float(1.0) / cofFactor1;
        // mean 与 var 同一口径：不再乘倒数常数 scale1 = cofFactor1/parallelN（该常数的舍入是
        // 所有通道同向的系统性偏差），改为在 finalize 里真除法 ÷parallelN 后再乘回 cofFactor1
        // （2 的幂，精确），等价于一次正确舍入的除法。parallelN 与 cofFactor1 在此落成标量。
        cofFactor1Value = cofFactor1;
        parallelNValue = static_cast<float>(parallelN);

        int64_t aGmOffset = aBlockFactor * blockIdx;
        int64_t arGmOffset = aGmOffset * r0;
        xGm.SetGlobalBuffer((__gm__ T1*)x + arGmOffset);
        betaGm.SetGlobalBuffer((__gm__ T2*)beta + aGmOffset);
        gammaGm.SetGlobalBuffer((__gm__ T2*)gamma + aGmOffset);
        if (this->useRunningMeanVar) {
            meanGm.SetGlobalBuffer((__gm__ T2*)mean + aGmOffset);
            varGm.SetGlobalBuffer((__gm__ T2*)var + aGmOffset);
        }

        yGm.SetGlobalBuffer((__gm__ T1*)y + arGmOffset);
        batchMeanGm.SetGlobalBuffer((__gm__ float*)batch_mean + aGmOffset);
        // batch_var 即第 5 个输出 output_reserve_space_2：训练模式下保存**有偏方差** E[(x-mean)^2]
        batchVarGm.SetGlobalBuffer((__gm__ float*)batch_var + aGmOffset);
        outMeanGm.SetGlobalBuffer((__gm__ T2*)mean_out + aGmOffset);
        outVarGm.SetGlobalBuffer((__gm__ T2*)var_out + aGmOffset);

        int64_t aGatherLimitBetaGamaAlign = RoundUp32(aGatherLimit, sizeof(T2));
        pipe->InitBuffer(betaQueue, DOUBLE_BUFFER, aGatherLimitBetaGamaAlign * sizeof(T2));
        pipe->InitBuffer(gammaQueue, DOUBLE_BUFFER, aGatherLimitBetaGamaAlign * sizeof(T2));

        int64_t aGatherLimitAlign = RoundUp32(aGatherLimit, sizeof(float));
        if (this->useRunningMeanVar) {
            pipe->InitBuffer(meanQueue, DOUBLE_BUFFER, aGatherLimitAlign * sizeof(float));
            pipe->InitBuffer(varQueue, DOUBLE_BUFFER, aGatherLimitAlign * sizeof(float));
        }
        pipe->InitBuffer(batchMeanQueue, DOUBLE_BUFFER, aGatherLimitAlign * sizeof(float));
        pipe->InitBuffer(batchVarQueue, DOUBLE_BUFFER, aGatherLimitAlign * sizeof(float));
        pipe->InitBuffer(outMeanQueue, DOUBLE_BUFFER, aGatherLimitAlign * sizeof(float));
        pipe->InitBuffer(outVarQueue, DOUBLE_BUFFER, aGatherLimitAlign * sizeof(float));

        pipe->InitBuffer(xQueue, DOUBLE_BUFFER, ubSize * sizeof(T1));
        pipe->InitBuffer(yQueue, DOUBLE_BUFFER, ubSize * sizeof(T1));
        pipe->InitBuffer(tMeanBuff, ubSize * sizeof(float));
        pipe->InitBuffer(tVarBuff, ubSize * sizeof(float));
        // rstd 暂存(只参与归一化，不进 GM)；原 batchVarBuff 改作此用，UB 预算不变。
        pipe->InitBuffer(rstdBuff, aGatherLimitAlign * sizeof(float));
        // 无偏方差 M2/(n-1) 暂存(只给 running var 用，不进 GM)。
        // 不额外改 tiling 预算：最坏情况(T2=float32)下 beta/gamma/mean/var/batchMean/batchVar/outMean/outVar
        // 共 8 个双缓冲(16) + rstd(1) + 本缓冲(1) = 18，恰好等于 SMALL_BUFFER_NUM(9)*DOUBLE_BUFFER(2)。
        pipe->InitBuffer(unbiasedVarBuff, aGatherLimitAlign * sizeof(float));
        int64_t vcaddNum = RoundUp32(binaryAddQuotient / VL_FP32, sizeof(float));
        // NormCommon::DichotomyAdd 对该缓冲整 VL 无掩码读取；binaryAddQuotient 只有一个 VL 时
        // 这里只有 32 B，会越读一个 VL，故分配至少抬到一个 VL（VECTOR_LENGTH 字节）。
        int64_t dichotomyAddBufSize = vcaddNum * sizeof(float);
        if (dichotomyAddBufSize < static_cast<int64_t>(VECTOR_LENGTH)) {
            dichotomyAddBufSize = static_cast<int64_t>(VECTOR_LENGTH);
        }
        pipe->InitBuffer(dichotomyAddBuf, dichotomyAddBufSize);
    }

    __aicore__ inline uint32_t FindCofFactor(uint32_t n)
    {
        // 找到比n大的最邻近的二次幂数, n = 15，结果为16
        if ((n & (n - 1)) != 0) {
            uint32_t temp = n - 1;
            temp |= temp >> 1;
            temp |= temp >> 2;
            temp |= temp >> 4;
            temp |= temp >> 8;
            temp |= temp >> 16;
            return (temp + 1);
        } else {
            return n;
        }
    }

    __aicore__ inline void Process()
    {
        if (blockIdx >= this->realCoreNum) {
            return;
        }

        int32_t aCycle = ops::CeilDiv(processNumPerCore, aGatherLimit);
        int32_t aGatherTail = processNumPerCore % aGatherLimit;
        int32_t aProcessLength = aGatherLimit;

        for (int32_t i = 0; i < aCycle; i++) {
            if (aGatherTail != 0 and i == aCycle - 1) {
                aProcessLength = aGatherTail;
            }
            ProcessAParallel(i, aProcessLength);
        }
    }

    __aicore__ inline void ProcessCutR0(__ubuf__ float* tmpMeanLocal, __ubuf__ float* tmpVarLocal,
                                        __ubuf__ float* meanLocal, __ubuf__ float* rstdLocal, __ubuf__ float* varLocal,
                                        __ubuf__ float* unbiasedVarLocal, __ubuf__ float* dichotomyAddLocal,
                                        int32_t aParallelIndex, int32_t aParallelLength)
    {
        int64_t r0Tail = r0 % r0Factor;
        bool r0HasTail = r0Tail != 0;
        for (int64_t i = 0; i < aParallelLength; i++) {
            int64_t count = 0;
            int64_t countTail = 0;
            int64_t nBurst = 1;
            int64_t burstLen = r0Factor;
            for (uint64_t j = 0; j < loopR1outer; j++) {
                for (uint64_t k = 0; k < loopR0outer; k++) {
                    int64_t index = j * r1Factor * a0 * r0 + (aParallelIndex * aGatherLimit + i) * r0 + k * r0Factor;
                    // after ub spit, there may comes tail block at the end cycle.
                    if (r0HasTail && k == loopR0outer - 1) {
                        continue;
                    }

                    bool isFirstStep = count == 0;
                    count = count + 1;
                    ProcessSplitWelfordUpdate(tmpMeanLocal, tmpVarLocal, index, nBurst, burstLen, count, isFirstStep);
                }
            }

            // process r0 tail block
            int64_t index_last_k = r0 / r0Factor;
            burstLen = r0Tail;
            if (r0HasTail) {
                for (uint64_t j = 0; j < loopR1outer; j++) {
                    int64_t index = j * r1Factor * a0 * r0 + (aParallelIndex * aGatherLimit + i) * r0 +
                                    index_last_k * r0Factor;
                    // after ub spit, there may comes tail block at the end cycle.
                    bool isFirstStep = count == 0;
                    count = count + 1;
                    countTail = countTail + 1;
                    ProcessSplitWelfordUpdate(tmpMeanLocal, tmpVarLocal, index, nBurst, burstLen, count, isFirstStep);
                }
            }
            if (countTail == 0) {
                VFWelfordParallelFinalizeAlign(meanLocal, rstdLocal, varLocal, unbiasedVarLocal, tmpMeanLocal,
                                               tmpVarLocal, dichotomyAddLocal, parallelN, binaryAddQuotient, binaryAddK,
                                               binaryAddLastNum, i, reduceScale0, scale0, count, eps);
            } else {
                countTail = count - countTail;
                VFWelfordParallelFinalizeNonAlign(meanLocal, rstdLocal, varLocal, unbiasedVarLocal, tmpMeanLocal,
                                                  tmpVarLocal, dichotomyAddLocal, parallelN, binaryAddQuotient,
                                                  binaryAddK, binaryAddLastNum, i, r0Tail, reduceScale0, countTail,
                                                  count, eps);
            }
        }
    }

    __aicore__ inline void ProcessCutR1(__ubuf__ float* tmpMeanLocal, __ubuf__ float* tmpVarLocal,
                                        __ubuf__ float* meanLocal, __ubuf__ float* rstdLocal, __ubuf__ float* varLocal,
                                        __ubuf__ float* unbiasedVarLocal, __ubuf__ float* dichotomyAddLocal,
                                        int32_t aParallelIndex, int32_t aParallelLength)
    {
        int64_t r1Tail = r1 % r1Factor;
        bool r1HasTail = r1Tail != 0;
        for (int64_t i = 0; i < aParallelLength; i++) {
            int64_t count = 0;
            int64_t countTail = 0;
            uint64_t nBurst = r1Factor;
            int64_t burstLen = r0Factor;
            for (uint64_t j = 0; j < loopR1outer; j++) {
                int64_t index = j * r1Factor * a0 * r0 + (aParallelIndex * aGatherLimit + i) * r0;
                if (r1HasTail && j == loopR1outer - 1) {
                    nBurst = r1Tail;
                    countTail = 1;
                }

                bool isFirstStep = count == 0;
                count = count + 1;
                ProcessSplitWelfordUpdate(tmpMeanLocal, tmpVarLocal, index, nBurst, burstLen, count, isFirstStep);
            }
            if (countTail == 0) {
                VFWelfordParallelFinalizeAlign(meanLocal, rstdLocal, varLocal, unbiasedVarLocal, tmpMeanLocal,
                                               tmpVarLocal, dichotomyAddLocal, parallelN, binaryAddQuotient, binaryAddK,
                                               binaryAddLastNum, i, reduceScale0, scale0, count, eps);
            } else {
                countTail = count - countTail;
                uint32_t tailNum = r1Tail * r0Factor;
                VFWelfordParallelFinalizeNonAlign(meanLocal, rstdLocal, varLocal, unbiasedVarLocal, tmpMeanLocal,
                                                  tmpVarLocal, dichotomyAddLocal, parallelN, binaryAddQuotient,
                                                  binaryAddK, binaryAddLastNum, i, tailNum, reduceScale0, countTail,
                                                  count, eps);
            }
        }
    }

    __aicore__ inline void ProcessAParallel(int32_t aParallelIndex, int32_t aParallelLength)
    {
        // process core
        LocalTensor<float> tMeanTensor = tMeanBuff.Get<float>();
        LocalTensor<float> tVarTensor = tVarBuff.Get<float>();
        LocalTensor<float> rstdTensor = rstdBuff.Get<float>();
        LocalTensor<float> unbiasedVarTensor = unbiasedVarBuff.Get<float>();
        LocalTensor<float> dichotomyAddTensor = dichotomyAddBuf.Get<float>();
        LocalTensor<float> batchMeanTensor = batchMeanQueue.AllocTensor<float>();
        // batchVarQueue 的 tensor 承载第 5 个输出 output_reserve_space_2：**有偏方差** M2/n。
        LocalTensor<float> batchVarTensor = batchVarQueue.AllocTensor<float>();

        __ubuf__ float* batchMeanLocal = (__ubuf__ float*)batchMeanTensor.GetPhyAddr();
        __ubuf__ float* batchVarLocal = (__ubuf__ float*)batchVarTensor.GetPhyAddr();
        __ubuf__ float* unbiasedVarLocal = (__ubuf__ float*)unbiasedVarTensor.GetPhyAddr();
        __ubuf__ float* rstdLocal = (__ubuf__ float*)rstdTensor.GetPhyAddr();
        __ubuf__ float* tmpMeanLocal = (__ubuf__ float*)tMeanTensor.GetPhyAddr();
        __ubuf__ float* tmpVarLocal = (__ubuf__ float*)tVarTensor.GetPhyAddr();
        __ubuf__ float* dichotomyAddLocal = (__ubuf__ float*)dichotomyAddTensor.GetPhyAddr();

        // copy small tensor to ub
        int32_t index_a = blockIdx * this->aBlockFactor + aParallelIndex * aGatherLimit;
        int32_t copyLen = aParallelLength;
        CopySmallTensor2UB(index_a, copyLen);

        // process A loop
        // 最终化时：varLocal <- 有偏方差(直接进 output_reserve_space_2)，
        //           unbiasedVarLocal <- 无偏方差(只供 running var 计算)，
        //           rstdLocal <- rstd(仅归一化用)
        if (cutR1OrR0 == 0) {
            ProcessCutR0(tmpMeanLocal, tmpVarLocal, batchMeanLocal, rstdLocal, batchVarLocal, unbiasedVarLocal,
                         dichotomyAddLocal, aParallelIndex, aParallelLength);
        } else if (cutR1OrR0 == 1) {
            ProcessCutR1(tmpMeanLocal, tmpVarLocal, batchMeanLocal, rstdLocal, batchVarLocal, unbiasedVarLocal,
                         dichotomyAddLocal, aParallelIndex, aParallelLength);
        }

        LocalTensor<T2> meanTensor;
        LocalTensor<T2> varTensor;
        __ubuf__ T2* meanLocal = nullptr;
        __ubuf__ T2* varLocal = nullptr;

        if (this->useRunningMeanVar) {
            meanTensor = meanQueue.DeQue<T2>();
            varTensor = varQueue.DeQue<T2>();
            meanLocal = (__ubuf__ T2*)meanTensor.GetPhyAddr();
            varLocal = (__ubuf__ T2*)varTensor.GetPhyAddr();
        }

        LocalTensor<T2> outMeanTensor = outMeanQueue.AllocTensor<T2>();
        LocalTensor<T2> outVarTensor = outVarQueue.AllocTensor<T2>();

        __ubuf__ T2* outMeanLocal = (__ubuf__ T2*)outMeanTensor.GetPhyAddr();
        __ubuf__ T2* outVarLocal = (__ubuf__ T2*)outVarTensor.GetPhyAddr();

        uint16_t loopCount2 = ops::CeilDiv(aParallelLength, static_cast<int32_t>(VL_FP32));

        __ubuf__ float* batchMeanLocal2 = (__ubuf__ float*)batchMeanTensor.GetPhyAddr();
        // running var 直接用**无偏方差** M2/(n-1)（finalize 里真除法算出），
        // 不再对 batch_var 再乘一次 Bessel 常数 n/(n-1)，避免二次常数舍入。
        CalculateRunningMeanAndVar(batchMeanLocal2, unbiasedVarLocal, meanLocal, varLocal, outMeanLocal, outVarLocal,
                                   loopCount2, aParallelLength);

        if (this->useRunningMeanVar) {
            meanQueue.FreeTensor(meanTensor);
            varQueue.FreeTensor(varTensor);
        }

        batchMeanQueue.EnQue(batchMeanTensor);
        batchVarQueue.EnQue(batchVarTensor);
        outMeanQueue.EnQue(outMeanTensor);
        outVarQueue.EnQue(outVarTensor);

        CopyOutMeanAndVar2GM(index_a, aParallelLength);
        ProcessNormalize(index_a, aParallelIndex, aParallelLength, rstdLocal);
    }

private:
    __aicore__ inline void ProcessSplitWelfordUpdate(__ubuf__ float* tmpMeanLocal, __ubuf__ float* tmpVarLocal,
                                                     int64_t index, int64_t nBurst, int64_t burstLen, int64_t count,
                                                     bool isFirstStep)
    {
        // copy large data into ub
        CopyInX(index, nBurst, burstLen);

        LocalTensor<T1> xInUb = xQueue.DeQue<T1>();
        __ubuf__ T1* xLocal = (__ubuf__ T1*)xInUb.GetPhyAddr();
        // process welford after copy ubSize data into ub.
        float scale = (float)1.0 / static_cast<float>(count);
        uint64_t processNum = nBurst * burstLen;
        uint16_t updateLoopCount = ops::CeilDiv(processNum, static_cast<uint64_t>(VL_FP32));
        if (isFirstStep) {
            // 第一次更新时，需要将tmp mean和tmp var清0
            VFWelfordParallelUpdateWithInit(xLocal, tmpMeanLocal, tmpVarLocal, processNum, updateLoopCount, scale);
        } else {
            VFWelfordParallelUpdate(xLocal, tmpMeanLocal, tmpVarLocal, processNum, updateLoopCount, scale);
        }
        xQueue.FreeTensor(xInUb);
    }

    __aicore__ inline void CopyInX(const int64_t& index, const int64_t& nBurst, const int64_t& burstLen)
    {
        LocalTensor<T1> xInUb = xQueue.AllocTensor<T1>();
        DataCopyPadExtParams<T1> padParams;
        padParams.isPad = false;

        DataCopyExtParams dataCopyParams;
        dataCopyParams.blockCount = nBurst;
        dataCopyParams.blockLen = burstLen * sizeof(T1);
        dataCopyParams.srcStride = 0;
        dataCopyParams.dstStride = 0;
        if (nBurst != 1) {
            dataCopyParams.srcStride = (a0 - 1) * r0 * sizeof(T1);
        }
        DataCopyPad<T1, PaddingMode::Compact>(xInUb, xGm[index], dataCopyParams, padParams);

        xQueue.EnQue(xInUb);
    }

    __aicore__ inline void CopySmallTensor2UB(const int64_t& index, const int64_t& copyLen)
    {
        DataCopyPadExtParams<T2> padParams;
        padParams.isPad = true;
        padParams.paddingValue = static_cast<T2>(0.0);
        // isPad配置True，rightPadding配置0，表示自动Pad到32B对齐
        padParams.rightPadding = 0;

        DataCopyExtParams dataCopyParams;
        dataCopyParams.blockCount = 1;
        dataCopyParams.blockLen = copyLen * sizeof(T2);
        dataCopyParams.srcStride = 0;
        dataCopyParams.dstStride = 0;

        LocalTensor<T2> gammaInUb = gammaQueue.AllocTensor<T2>();
        LocalTensor<T2> betaInUb = betaQueue.AllocTensor<T2>();
        DataCopyPad(gammaInUb, gammaGm[index], dataCopyParams, padParams);
        DataCopyPad(betaInUb, betaGm[index], dataCopyParams, padParams);
        gammaQueue.EnQue(gammaInUb);
        betaQueue.EnQue(betaInUb);

        if (this->useRunningMeanVar) {
            LocalTensor<T2> meanInUb = meanQueue.AllocTensor<T2>();
            LocalTensor<T2> varInUb = varQueue.AllocTensor<T2>();
            DataCopyPad(meanInUb, meanGm[index], dataCopyParams, padParams);
            DataCopyPad(varInUb, varGm[index], dataCopyParams, padParams);
            meanQueue.EnQue(meanInUb);
            varQueue.EnQue(varInUb);
        }
    }

    __aicore__ inline void CopyY2Gm(const int64_t& index, const int64_t& nBurst, const int64_t& burstLen)
    {
        LocalTensor<T1> yOutUb = yQueue.DeQue<T1>();

        DataCopyExtParams dataCopyParams;
        dataCopyParams.blockCount = nBurst;
        dataCopyParams.blockLen = burstLen * sizeof(T1);
        dataCopyParams.srcStride = 0;
        dataCopyParams.dstStride = 0;
        if (nBurst != 1) {
            dataCopyParams.dstStride = (a0 - 1) * r0 * sizeof(T1);
        }
        DataCopyPad<T1, PaddingMode::Compact>(yGm[index], yOutUb, dataCopyParams);
        yQueue.FreeTensor(yOutUb);
    }

    __aicore__ inline void CopyOutMeanAndVar2GM(int64_t index, int64_t copyLen)
    {
        LocalTensor<T2> outMeanTensor = outMeanQueue.DeQue<T2>();
        LocalTensor<T2> outVarTensor = outVarQueue.DeQue<T2>();

        DataCopyExtParams dataCopyParams;
        dataCopyParams.blockCount = 1;
        dataCopyParams.blockLen = copyLen * sizeof(T2);
        dataCopyParams.srcStride = 0;
        dataCopyParams.dstStride = 0;

        DataCopyPad(outMeanGm[index], outMeanTensor, dataCopyParams);
        DataCopyPad(outVarGm[index], outVarTensor, dataCopyParams);

        outMeanQueue.FreeTensor(outMeanTensor);
        outVarQueue.FreeTensor(outVarTensor);
    }

    // 拷出 batch_mean(output_reserve_space_1) 与**有偏方差**(output_reserve_space_2)。
    __aicore__ inline void CopyBatchMeanAndVar2GM(int64_t index, int64_t copyLen, LocalTensor<float> batchMeanTensor,
                                                  LocalTensor<float> batchVarTensor)
    {
        DataCopyExtParams dataCopyParams;
        dataCopyParams.blockCount = 1;
        dataCopyParams.blockLen = copyLen * sizeof(float);
        dataCopyParams.srcStride = 0;
        dataCopyParams.dstStride = 0;

        DataCopyPad(batchMeanGm[index], batchMeanTensor, dataCopyParams);
        DataCopyPad(batchVarGm[index], batchVarTensor, dataCopyParams);

        batchMeanQueue.FreeTensor(batchMeanTensor);
        batchVarQueue.FreeTensor(batchVarTensor);
    }

    // unbiasedVarLocal 是 finalize 用真除法算出的无偏方差 M2/(n-1)：
    // running_var = (1-m)*running_var + m * M2/(n-1)。
    // 原实现是 ((M2/n) * fl(n/(n-1))) * m —— 对 batch_var 再乘一次 Bessel 常数，
    // 该常数舍入误差会叠加成系统性偏差，故改为直接用真除法得到的无偏方差。
    __aicore__ inline void CalculateRunningMeanAndVar(__ubuf__ float* batchMeanLocal, __ubuf__ float* unbiasedVarLocal,
                                                      __ubuf__ T2* meanLocal, __ubuf__ T2* varLocal,
                                                      __ubuf__ T2* outMeanLocal, __ubuf__ T2* outVarLocal,
                                                      uint16_t loopCount, uint32_t calLen)
    {
        bool vfUseRunningMeanVar = this->useRunningMeanVar;
        float vfMomentum = this->momentum;
        float vfOneSubMomentum = this->oneSubMomentum;

        __VEC_SCOPE__
        {
            RegTensor<float> batchMean;
            RegTensor<float> batchVar;
            RegTensor<float> mean;
            RegTensor<float> var;
            uint32_t sreg0 = calLen;
            MaskReg pregLoop;
            for (uint16_t i = 0; i < loopCount; i++) {
                pregLoop = AscendC::MicroAPI::UpdateMask<float>(sreg0);

                // mean
                LoadAlign(batchMean, ((__ubuf__ float*)batchMeanLocal + i * VL_FP32));
                Muls(batchMean, batchMean, vfMomentum, pregLoop);

                if (vfUseRunningMeanVar) {
                    LoadTensorForDtypeT<T2>(mean, meanLocal, pregLoop, i * VL_FP32);
                    Muls(mean, mean, vfOneSubMomentum, pregLoop);
                    Add(batchMean, mean, batchMean, pregLoop);
                }
                StoreTensorForDtypeT<T2>(outMeanLocal, batchMean, pregLoop, i * VL_FP32);

                // var（无偏方差 M2/(n-1) 已在 finalize 用真除法算好，这里只需乘 momentum）
                LoadAlign(batchVar, (__ubuf__ float*)unbiasedVarLocal + i * VL_FP32);
                Muls(batchVar, batchVar, vfMomentum, pregLoop);
                if (vfUseRunningMeanVar) {
                    LoadTensorForDtypeT<T2>(var, varLocal, pregLoop, i * VL_FP32);
                    Muls(var, var, vfOneSubMomentum, pregLoop);
                    Add(batchVar, var, batchVar, pregLoop);
                }
                StoreTensorForDtypeT<T2>(outVarLocal, batchVar, pregLoop, i * VL_FP32);
            }
        }
    }

    __aicore__ inline void VFWelfordParallelUpdateWithInit(__ubuf__ T1* x1Local, __ubuf__ float* tmpMeanLocal,
                                                           __ubuf__ float* tmpVarLocal, uint64_t calLen,
                                                           uint16_t loopCount, float scale)
    {
        __VEC_SCOPE__
        {
            RegTensor<float> x1;
            RegTensor<float> tmpMean;
            RegTensor<float> tmpVar;
            RegTensor<float> delta1;
            RegTensor<float> delta2;
            RegTensor<float> delta3;
            RegTensor<float> delat4;
            RegTensor<float> zero;
            RegTensor<float> absMean;
            RegTensor<float> sumX1Mean;
            MaskReg infMeanMask;
            MaskReg nanSumMask;
            MaskReg notNanSumMask;
            MaskReg zeroDeltaMask;
            vector_bool pregMain = pset_b8(PAT_ALL);
            vector_bool pregLoop;
            Duplicate(zero, 0.0f, pregMain);
            uint32_t sreg0 = calLen;
            for (uint16_t i = 0; i < loopCount; i++) {
                pregLoop = plt_b32(sreg0, POST_UPDATE);
                LoadTensorForDtypeT(x1, x1Local, pregLoop, i * VL_FP32);
                Duplicate(tmpMean, 0.0, pregLoop);
                // delata1 = x1 - mean
                Sub(delta1, x1, tmpMean, pregLoop);
                // mean 为 ±inf 且 x1 + mean 不是 NaN 时，delta1 置 0，使 mean 保持 ±inf 不变；
                // 异号 inf（x1 = ∓inf、mean = ±inf）时 x1 + mean 未定义（NaN），必须保留 delta1 = ±inf，
                // 让 mean = mean + delta1 * scale 得到 NaN，与 golden 一致；
                // 同号 inf 与有限值 x1 的 x1 + mean = ±inf，仍走清零分支（有限值 - (±inf) = ∓inf 会污染均值）
                AscendC::Reg::Abs(absMean, tmpMean, pregLoop);
                Compares(infMeanMask, absMean, POS_INF, pregLoop);
                AscendC::Reg::Add(sumX1Mean, x1, tmpMean, pregLoop);
                AscendC::Reg::Compare<float, AscendC::CMPMODE::NE>(nanSumMask, sumX1Mean, sumX1Mean, pregLoop);
                AscendC::Reg::Not(notNanSumMask, nanSumMask, pregLoop);
                AscendC::Reg::And(zeroDeltaMask, infMeanMask, notNanSumMask, pregLoop);
                Select(delta1, zero, delta1, zeroDeltaMask);
                // delta2 = delta1 * scale
                Muls(delta2, delta1, scale, pregLoop);
                // mean = mean + delta2
                Add(tmpMean, tmpMean, delta2, pregLoop);
                StoreAlign(tmpMeanLocal + i * VL_FP32, tmpMean, pregLoop);

                Duplicate(tmpVar, 0.0, pregLoop);
                // delta3 = x1 - mean
                Sub(delta3, x1, tmpMean, pregLoop);
                // delta4 = delta1 * delta3
                Mul(delat4, delta1, delta3, pregLoop);
                // var = var + delta4
                Add(tmpVar, tmpVar, delat4, pregLoop);
                StoreAlign(tmpVarLocal + i * VL_FP32, tmpVar, pregLoop);
            }
        }
    }

    __aicore__ inline void VFWelfordParallelUpdate(__ubuf__ T1* x1Local, __ubuf__ float* tmpMeanLocal,
                                                   __ubuf__ float* tmpVarLocal, uint64_t calLen, uint16_t loopCount,
                                                   float scale)
    {
        __VEC_SCOPE__
        {
            RegTensor<float> x1;
            RegTensor<float> tmpMean;
            RegTensor<float> tmpVar;
            RegTensor<float> delta1;
            RegTensor<float> delta2;
            RegTensor<float> delta3;
            RegTensor<float> delat4;
            RegTensor<float> zero;
            RegTensor<float> absMean;
            RegTensor<float> sumX1Mean;
            MaskReg infMeanMask;
            MaskReg nanSumMask;
            MaskReg notNanSumMask;
            MaskReg zeroDeltaMask;
            MaskReg pregLoop;
            MaskReg pregAll = AscendC::MicroAPI::CreateMask<float, AscendC::MicroAPI::MaskPattern::ALL>();
            Duplicate(zero, 0.0f, pregAll);
            uint32_t sreg0 = calLen;
            for (uint16_t i = 0; i < loopCount; i++) {
                pregLoop = AscendC::MicroAPI::UpdateMask<float>(sreg0);
                LoadTensorForDtypeT(x1, x1Local, pregLoop, i * VL_FP32);

                LoadAlign(tmpMean, tmpMeanLocal + i * VL_FP32);
                // delata1 = x1 - mean
                Sub(delta1, x1, tmpMean, pregLoop);
                // mean 为 ±inf 且 x1 + mean 不是 NaN 时，delta1 置 0，使 mean 保持 ±inf 不变；
                // 异号 inf（x1 = ∓inf、mean = ±inf）时 x1 + mean 未定义（NaN），必须保留 delta1 = ±inf，
                // 让 mean = mean + delta1 * scale 得到 NaN，与 golden 一致；
                // 同号 inf 与有限值 x1 的 x1 + mean = ±inf，仍走清零分支（有限值 - (±inf) = ∓inf 会污染均值）
                AscendC::Reg::Abs(absMean, tmpMean, pregLoop);
                Compares(infMeanMask, absMean, POS_INF, pregLoop);
                AscendC::Reg::Add(sumX1Mean, x1, tmpMean, pregLoop);
                AscendC::Reg::Compare<float, AscendC::CMPMODE::NE>(nanSumMask, sumX1Mean, sumX1Mean, pregLoop);
                AscendC::Reg::Not(notNanSumMask, nanSumMask, pregLoop);
                AscendC::Reg::And(zeroDeltaMask, infMeanMask, notNanSumMask, pregLoop);
                Select(delta1, zero, delta1, zeroDeltaMask);
                // delta2 = delta1 * scale
                Muls(delta2, delta1, scale, pregLoop);
                // mean = mean + delta2
                Add(tmpMean, tmpMean, delta2, pregLoop);
                StoreAlign(tmpMeanLocal + i * VL_FP32, tmpMean, pregLoop);

                LoadAlign(tmpVar, tmpVarLocal + i * VL_FP32);
                // delta3 = x1 - mean
                Sub(delta3, x1, tmpMean, pregLoop);
                // delta4 = delta1 * delta3
                Mul(delat4, delta1, delta3, pregLoop);
                // var = var + delta4
                Add(tmpVar, tmpVar, delat4, pregLoop);
                StoreAlign(tmpVarLocal + i * VL_FP32, tmpVar, pregLoop);
            }
        }
    }

    // 末级部分均值若全部相同且有限，加权归约本身仍可能在 fp32 中溢出成 ±inf/NaN
    // （典型：x 恒为 ±FLT_MAX，fp32 求和必然上溢，而 fp64 真值的均值就是那个有限常量）。
    // 这里用 min/max 找回各分道部分均值的公共有限值：命中时把最终均值还原成它，
    // 从而 var = Σ M2 + count*(mean_i - mean)^2 里的修正项精确为 0。
    // 另外记录「各分道 M2 是否全为 0」：公共均值 + M2 全 0 可证明数据恒定 ⇒ 方差精确为 0，
    // 用 zeroVarMask 把方差（含无偏方差与 rstd）钉到 0。
    // 掩码不命中时 Select 选回原值，正常输入逐位不变；全部逻辑只在 finalize 执行一次。
    __aicore__ inline void RestoreUniformFiniteMeanAndZeroVar(RegTensor<float>& mean, __ubuf__ float* partialMeans,
                                                              __ubuf__ float* partialVar, uint32_t count,
                                                              MaskReg& scalarMask, MaskReg& fullMask,
                                                              MaskReg& zeroVarMask, RegTensor<float>& values,
                                                              RegTensor<float>& minMean, RegTensor<float>& maxMean,
                                                              RegTensor<float>& maxAbsVar, RegTensor<float>& chunkMin,
                                                              RegTensor<float>& chunkMax)
    {
        MaskReg uniformMask;
        MaskReg tempMask;
        Duplicate(minMean, POS_INF, scalarMask);
        Duplicate(maxMean, -POS_INF, scalarMask);
        Duplicate(maxAbsVar, float(0.0), scalarMask);
        uint16_t fullLoops = static_cast<uint16_t>(count / VL_FP32);
        for (uint16_t i = 0; i < fullLoops; i++) {
            LoadAlign(values, partialMeans + i * VL_FP32);
            Reduce<ReduceType::MIN>(chunkMin, values, fullMask);
            Reduce<ReduceType::MAX>(chunkMax, values, fullMask);
            AscendC::Reg::Min(minMean, minMean, chunkMin, scalarMask);
            AscendC::Reg::Max(maxMean, maxMean, chunkMax, scalarMask);
            LoadAlign(values, partialVar + i * VL_FP32);
            AscendC::Reg::Abs(values, values, fullMask);
            Reduce<ReduceType::MAX>(chunkMax, values, fullMask);
            AscendC::Reg::Max(maxAbsVar, maxAbsVar, chunkMax, scalarMask);
        }
        // 部分均值缓冲只按块对齐，尾块逐元素广播读取，避免读出缓冲区末尾。
        uint16_t tail = static_cast<uint16_t>(count - fullLoops * VL_FP32);
        for (uint16_t i = 0; i < tail; i++) {
            LoadAlign<float, LoadDist::DIST_BRC_B32>(values, partialMeans + fullLoops * VL_FP32 + i);
            AscendC::Reg::Min(minMean, minMean, values, scalarMask);
            AscendC::Reg::Max(maxMean, maxMean, values, scalarMask);
            LoadAlign<float, LoadDist::DIST_BRC_B32>(values, partialVar + fullLoops * VL_FP32 + i);
            AscendC::Reg::Abs(values, values, scalarMask);
            AscendC::Reg::Max(maxAbsVar, maxAbsVar, values, scalarMask);
        }
        // uniform = (minMean == maxMean) && (|minMean| < POS_INF) && (mean == mean)。
        // POS_INF 字面量在编译期上溢为 +inf，故 `|minMean| < POS_INF` 的语义正是「有限」。
        AscendC::Reg::Compare<float, AscendC::CMPMODE::EQ>(uniformMask, minMean, maxMean, scalarMask);
        AscendC::Reg::Abs(chunkMin, minMean, scalarMask);
        AscendC::Reg::CompareScalar<float, AscendC::CMPMODE::LT>(tempMask, chunkMin, POS_INF, scalarMask);
        AscendC::MicroAPI::And(uniformMask, uniformMask, tempMask, scalarMask);
        AscendC::Reg::Compare<float, AscendC::CMPMODE::EQ>(tempMask, mean, mean, scalarMask);
        AscendC::MicroAPI::And(uniformMask, uniformMask, tempMask, scalarMask);
        AscendC::Reg::Select(mean, minMean, mean, uniformMask);
        // allM2Zero = (max |M2_i| == 0)；zeroVarMask = uniform && allM2Zero。
        AscendC::Reg::CompareScalar<float, AscendC::CMPMODE::EQ>(tempMask, maxAbsVar, float(0.0), scalarMask);
        AscendC::MicroAPI::And(zeroVarMask, uniformMask, tempMask, scalarMask);
    }

    /*
      Welford Finalize对齐场景计算公式如下:
      finalize_mean = sum_fun(mean) / parallel_N
      finalize_delta = mean - finalize_mean
      finalize_delta_square = finalize_delta * finalize_delta
      M2_fixed = M2 + float(count) * finalize_delta_square
      finalize_std = sum_fun(M2_fixed) / float(parallel_N * count)
    */
    __aicore__ inline void VFWelfordParallelFinalizeAlign(__ubuf__ float* meanLocal, __ubuf__ float* rstdLocal,
                                                          __ubuf__ float* varLocal, __ubuf__ float* unbiasedVarLocal,
                                                          __ubuf__ float* tmpMeanLocal, __ubuf__ float* tmpVarLocal,
                                                          __ubuf__ float* dichotomyAddLocal, uint32_t reduceCount,
                                                          uint32_t dichotomyAddPower, uint32_t dichotomyAddK,
                                                          uint32_t dichotomyAddLastNum, uint32_t offset,
                                                          float reduceScale0, float scale0, float cnt, float eps)
    {
        uint32_t dichotomyAddReminder = reduceCount - dichotomyAddPower;
        uint16_t dichotomyAddReminderLoopCount = ops::CeilDiv(dichotomyAddReminder, static_cast<uint32_t>(VL_FP32));
        uint16_t dichotomyAddPowerLoopCount = dichotomyAddPower / VL_FP32;
        uint16_t innerLoopCountOrigin = dichotomyAddPowerLoopCount / VL_FP32;
        float vfN = this->nValue;
        float vfNMinusOne = this->nMinusOneValue;
        float vfCofFactor = this->cofFactor0Value;
        float vfParallelN = this->parallelNValue;
        float vfCofFactor1 = this->cofFactor1Value;
        __VEC_SCOPE__
        {
            RegTensor<float> dichotomyAddMeanL;
            RegTensor<float> dichotomyAddMeanR;
            RegTensor<float> dichotomyAddVarL;
            RegTensor<float> dichotomyAddVarR;
            RegTensor<float> sumMean;
            RegTensor<float> mean;
            RegTensor<float> sumVar;
            RegTensor<float> var;
            RegTensor<float> varUnbiased;
            RegTensor<float> nReg;
            RegTensor<float> nMinusOneReg;
            RegTensor<float> deltaL;
            RegTensor<float> deltaR;
            RegTensor<float> one;
            RegTensor<float> rstd;
            MaskReg pregLoop;
            MaskReg pregMain = AscendC::MicroAPI::CreateMask<float, AscendC::MicroAPI::MaskPattern::ALL>();
            MaskReg pregMerge = AscendC::MicroAPI::CreateMask<float, AscendC::MicroAPI::MaskPattern::VL1>();
            MaskReg zeroVarMask;
            uint32_t sreg0 = dichotomyAddReminder;
            // 计算mean
            // PART1: 整尾块合并
            for (uint16_t i = 0; i < dichotomyAddReminderLoopCount; i++) {
                pregLoop = AscendC::MicroAPI::UpdateMask<float>(sreg0);
                LoadAlign(dichotomyAddMeanL, tmpMeanLocal + i * VL_FP32);
                LoadAlign(dichotomyAddMeanR, tmpMeanLocal + i * VL_FP32 + dichotomyAddPower);
                Muls(dichotomyAddMeanL, dichotomyAddMeanL, scale0, pregMain);
                Muls(dichotomyAddMeanR, dichotomyAddMeanR, scale0, pregLoop);
                Add(sumMean, dichotomyAddMeanL, dichotomyAddMeanR, pregMain);
                Reduce<ReduceType::SUM>(mean, sumMean, pregMain);
                StoreAlign<float, StoreDist::DIST_FIRST_ELEMENT_B32>(dichotomyAddLocal + i, mean, pregMerge);
            }

            // PART2: 整块剩余部分vcadd回刷UB
            for (uint16_t i = 0; i < static_cast<uint16_t>(dichotomyAddPowerLoopCount - dichotomyAddReminderLoopCount);
                 i++) {
                LoadAlign(dichotomyAddMeanL, tmpMeanLocal + (i + dichotomyAddReminderLoopCount) * VL_FP32);
                Muls(dichotomyAddMeanL, dichotomyAddMeanL, scale0, pregMain);
                Reduce<ReduceType::SUM>(mean, dichotomyAddMeanL, pregMain);
                StoreAlign<float, StoreDist::DIST_FIRST_ELEMENT_B32>(
                    dichotomyAddLocal + dichotomyAddReminderLoopCount + i, mean, pregMerge);
            }

            // 末级归约换成补偿求和（见文件头 WelfordCompensatedLaneReduce 说明），
            // mean 缩放改「真除法 + 乘回 2 的幂」：累加时每项已乘 scale0 = 1/cofFactor1（2 的
            // 幂，精确），故此时 mean = Σ/cofFactor1。真除法 ÷parallelN（< 2^24 可精确表示）后
            // 乘回 cofFactor1（精确），等价于一次正确舍入的除法；不再乘倒数常数
            // scale1 = cofFactor1/parallelN（其舍入是所有通道同向的常量偏置）。
            WelfordDichotomyAddCompensated(mean, dichotomyAddLocal, dichotomyAddK, innerLoopCountOrigin,
                                           dichotomyAddLastNum);
            Duplicate(nReg, vfParallelN, pregMain);
            Div(mean, mean, nReg, pregMerge);
            Muls(mean, mean, vfCofFactor1, pregMerge);
            RestoreUniformFiniteMeanAndZeroVar(mean, tmpMeanLocal, tmpVarLocal, reduceCount, pregMerge, pregMain,
                                               zeroVarMask, dichotomyAddMeanL, dichotomyAddMeanR, dichotomyAddVarL,
                                               dichotomyAddVarR, sumMean, sumVar);
            StoreAlign<float, StoreDist::DIST_FIRST_ELEMENT_B32>(meanLocal + offset, mean, pregMerge);

            Duplicate(one, float(1.0), pregMain);
            Duplicate<float, AscendC::MicroAPI::HighLowPart::LOWEST, MaskMergeMode::ZEROING>(mean, mean, pregMain);
            sreg0 = dichotomyAddReminder;
            // PART1: 整尾块合并
            for (uint16_t i = 0; i < dichotomyAddReminderLoopCount; i++) {
                pregLoop = AscendC::MicroAPI::UpdateMask<float>(sreg0);
                LoadAlign(dichotomyAddMeanL, tmpMeanLocal + i * VL_FP32);
                Sub(deltaL, dichotomyAddMeanL, mean, pregMain);
                Mul(deltaL, deltaL, deltaL, pregMain);
                Muls(deltaL, deltaL, cnt, pregMain);
                LoadAlign(dichotomyAddVarL, tmpVarLocal + i * VL_FP32);
                Add(dichotomyAddVarL, dichotomyAddVarL, deltaL, pregMain);
                Muls(dichotomyAddVarL, dichotomyAddVarL, reduceScale0, pregMain);

                LoadAlign(dichotomyAddMeanR, tmpMeanLocal + i * VL_FP32 + dichotomyAddPower);
                Sub(deltaR, dichotomyAddMeanR, mean, pregLoop);
                Mul(deltaR, deltaR, deltaR, pregLoop);
                Muls(deltaR, deltaR, cnt, pregLoop);
                LoadAlign(dichotomyAddVarR, tmpVarLocal + i * VL_FP32 + dichotomyAddPower);
                Add(dichotomyAddVarR, dichotomyAddVarR, deltaR, pregLoop);
                Muls(dichotomyAddVarR, dichotomyAddVarR, reduceScale0, pregLoop);

                Add(sumVar, dichotomyAddVarL, dichotomyAddVarR, pregMain);
                Reduce<ReduceType::SUM>(var, sumVar, pregMain);
                StoreAlign<float, StoreDist::DIST_FIRST_ELEMENT_B32>(dichotomyAddLocal + i, var, pregMerge);
            }

            // PART2: 整块剩余部分vcadd回刷UB
            for (uint16_t i = 0; i < static_cast<uint16_t>(dichotomyAddPowerLoopCount - dichotomyAddReminderLoopCount);
                 i++) {
                LoadAlign(dichotomyAddMeanL, tmpMeanLocal + (i + dichotomyAddReminderLoopCount) * VL_FP32);
                Sub(deltaL, dichotomyAddMeanL, mean, pregMain);
                Mul(deltaL, deltaL, deltaL, pregMain);
                Muls(deltaL, deltaL, cnt, pregMain);
                LoadAlign(dichotomyAddVarL, tmpVarLocal + (i + dichotomyAddReminderLoopCount) * VL_FP32);
                Add(dichotomyAddVarL, dichotomyAddVarL, deltaL, pregMain);
                Muls(dichotomyAddVarL, dichotomyAddVarL, reduceScale0, pregMain);
                Reduce<ReduceType::SUM>(var, dichotomyAddVarL, pregMain);
                StoreAlign<float, StoreDist::DIST_FIRST_ELEMENT_B32>(
                    dichotomyAddLocal + dichotomyAddReminderLoopCount + i, var, pregMerge);
            }

            // 末级归约换成补偿求和（M2 的末级横向求和与 mean 同口径，误差经 rsqrt 折算进 rstd）。
            WelfordDichotomyAddCompensated(var, dichotomyAddLocal, dichotomyAddK, innerLoopCountOrigin,
                                           dichotomyAddLastNum);
            // 累加时每项已乘 reduceScale0 = 1/cofFactor0（cofFactor0 是 2 的幂，该缩放精确），
            // 故此时 var = M2/cofFactor0。有偏方差 M2/n 与无偏方差 M2/(n-1) 都用**真除法**：
            // n 与 n-1 均 < 2^24、在 fp32 中精确可表示，真除法只引入 ≤0.5 ULP 的随机舍入；
            // 而乘倒数会把常数自身的舍入误差叠加成所有通道同向的系统性偏差（实测约 -0.3 ULP）。
            // 先除再乘回 cofFactor0（2 的幂，精确、且中间值更小不会溢出），结果等价于除以 n / n-1。
            // varLocal 写**有偏方差** M2/n（第 5 个输出 output_reserve_space_2 的契约值），
            // unbiasedVarLocal 写**无偏方差** M2/(n-1)（只供 running var 使用），rstd 单独写 rstdLocal。
            Duplicate(nReg, vfN, pregMain);
            Duplicate(nMinusOneReg, vfNMinusOne, pregMain);
            Div(varUnbiased, var, nMinusOneReg, pregMerge);
            Div(var, var, nReg, pregMerge);
            Muls(varUnbiased, varUnbiased, vfCofFactor, pregMerge);
            Muls(var, var, vfCofFactor, pregMerge);
            // uniform && allM2Zero 时方差精确为 0；掩码不命中时 Select 选回原值。
            Duplicate(sumMean, float(0.0), pregMerge);
            AscendC::Reg::Select(var, sumMean, var, zeroVarMask);
            AscendC::Reg::Select(varUnbiased, sumMean, varUnbiased, zeroVarMask);
            StoreAlign<float, StoreDist::DIST_FIRST_ELEMENT_B32>(varLocal + offset, var, pregMerge);
            StoreAlign<float, StoreDist::DIST_FIRST_ELEMENT_B32>(unbiasedVarLocal + offset, varUnbiased, pregMerge);
            ComputeRstdNewtonRaphsonRegExt2<false>(var, rstd, pregMerge, eps);
            StoreAlign<float, StoreDist::DIST_FIRST_ELEMENT_B32>(rstdLocal + offset, rstd, pregMerge);
        }
    }
    /*
      // Welford Finalize非对齐场景计算公式如下:
      counts = np.ones(len(mean), dtype=np.float32)*count
      for i in range(tail_size):
          counts[i] += 1
      finalize_mean = np.sum(mean*counts) / np.sum(counts)
      finalize_delta = mean - finalize_mean
      finalize_delta_square = finalize_delta * finalize_delta
      M2_fixed = M2 + counts * finalize_delta_square
      finalize_std = np.sum(M2_fixed) / np.sum(counts)

      // Welford Finalize非对齐场景下，二分累加存在如下几种场景:
      首先,非对齐场景下存在两种尾块类型
      1. welford自身的整块和尾块，需要注意的是，welford自身的整块也可能非对齐，整块+尾块=并行度
      2. 二分累加的整块和尾块
      3.
      3.1 welford整块小于二分累加整块，这种场景下，可以进一步细分为两种场景
      3.1.1 welford整块小于二分累加尾块向上对齐，那么welford整块处理逻辑就需要体现在二分累加整尾块折叠逻辑中
      3.1.2 welford整块大于等于二分累加尾块向上对齐，那么welford整块处理逻辑就需要体现剩余整块回刷UB逻辑中
      3.2 welford整块大于等于二分累加整块，那么welford整块处理逻辑就需要体现在二分累加整尾块折叠逻辑中
    */

    // welford整块大于等于二分累加整块
    __aicore__ inline void VFWelfordParallelFinalizeNonAlignSituation1(
        __ubuf__ float* meanLocal, __ubuf__ float* rstdLocal, __ubuf__ float* varLocal,
        __ubuf__ float* unbiasedVarLocal, __ubuf__ float* tmpMeanLocal, __ubuf__ float* tmpVarLocal,
        __ubuf__ float* dichotomyAddLocal, uint32_t reduceCount, uint32_t dichotomyAddPower, uint32_t dichotomyAddK,
        uint32_t dichotomyAddLastNum, uint32_t offset, uint32_t tailSize, float reduceScale0, float cnt, float tailCnt,
        float eps)
    {
        // cnt  是不带尾块的累加次数
        // tailCnt 是带尾块的累加次数， tailCnt >= cnt
        float vfN = this->nValue;
        float vfNMinusOne = this->nMinusOneValue;
        float vfCofFactor = this->cofFactor0Value;
        float coeff = tailCnt / cnt;
        float tailCountScale = tailCnt * reduceScale0;
        float countScale = cnt * reduceScale0;

        uint32_t dichotomyAddReminder = reduceCount - dichotomyAddPower;
        uint16_t dichotomyAddReminderRealLoopCount = ops::CeilDiv(dichotomyAddReminder, static_cast<uint32_t>(VL_FP32));
        uint16_t dichotomyAddPowerLoopCount = dichotomyAddPower / VL_FP32;
        uint16_t innerLoopCountOrigin = dichotomyAddPowerLoopCount / VL_FP32;

        uint32_t welfordDiff = tailSize - dichotomyAddPower;
        uint16_t welfordDiffLoopCount = welfordDiff / VL_FP32;
        uint32_t welfordDiffReminder = welfordDiff - welfordDiffLoopCount * VL_FP32;
        uint32_t welfordDiffReminderAlign = welfordDiffReminder == 0 ? 0 : VL_FP32;
        uint16_t welfordReminderLoopCount = welfordDiffReminderAlign / VL_FP32;

        uint32_t dichotomyAddReminderAfterSplit = dichotomyAddReminder -
                                                  (welfordDiffLoopCount + welfordReminderLoopCount) * VL_FP32;
        uint16_t dichotomyAddReminderLoopCount = ops::CeilDiv(dichotomyAddReminderAfterSplit,
                                                              static_cast<uint32_t>(VL_FP32));
        __VEC_SCOPE__
        {
            RegTensor<float> dichotomyAddMeanL;
            RegTensor<float> dichotomyAddMeanR;
            RegTensor<float> dichotomyAddVarL;
            RegTensor<float> dichotomyAddVarR;
            RegTensor<float> sumMean;
            RegTensor<float> mean;
            RegTensor<float> sumVar;
            RegTensor<float> var;
            RegTensor<float> deltaL;
            RegTensor<float> deltaR;
            RegTensor<float> one;
            RegTensor<float> rstd;
            RegTensor<float> tmp;
            RegTensor<float> varUnbiased;
            RegTensor<float> nReg;
            RegTensor<float> nMinusOneReg;

            MaskReg pregLoop;
            MaskReg pregLoop1;
            MaskReg pregMain = AscendC::MicroAPI::CreateMask<float, AscendC::MicroAPI::MaskPattern::ALL>();
            MaskReg pregMerge = AscendC::MicroAPI::CreateMask<float, AscendC::MicroAPI::MaskPattern::VL1>();
            MaskReg zeroVarMask;
            uint32_t sreg0;

            // 整块使用tailCountScale,尾块使用tailCountScale
            for (uint16_t i = 0; i < welfordDiffLoopCount; i++) {
                LoadAlign(dichotomyAddMeanL, tmpMeanLocal + i * VL_FP32);
                LoadAlign(dichotomyAddMeanR, tmpMeanLocal + i * VL_FP32 + dichotomyAddPower);
                Muls(dichotomyAddMeanL, dichotomyAddMeanL, tailCountScale, pregMain);
                Muls(dichotomyAddMeanR, dichotomyAddMeanR, tailCountScale, pregMain);
                Add(sumMean, dichotomyAddMeanL, dichotomyAddMeanR, pregMain);
                Reduce<ReduceType::SUM>(mean, sumMean, pregMain);
                StoreAlign<float, StoreDist::DIST_FIRST_ELEMENT_B32>(dichotomyAddLocal + i, mean, pregMerge);
            }

            // 处理welford第一次非对齐点, 整块使用tailCountScale,尾块部分使用tailCountScale, 部分使用countScale
            sreg0 = dichotomyAddReminder - welfordDiffLoopCount * VL_FP32;
            uint32_t sreg1 = welfordDiffReminder;
            for (uint16_t i = 0; i < welfordReminderLoopCount; i++) {
                pregLoop = AscendC::MicroAPI::UpdateMask<float>(sreg0);
                pregLoop1 = AscendC::MicroAPI::UpdateMask<float>(sreg1);
                LoadAlign(dichotomyAddMeanL, tmpMeanLocal + (i + welfordDiffLoopCount) * VL_FP32);
                LoadAlign(dichotomyAddMeanR, tmpMeanLocal + (i + welfordDiffLoopCount) * VL_FP32 + dichotomyAddPower);
                Muls(dichotomyAddMeanL, dichotomyAddMeanL, tailCountScale, pregMain);
                Muls(dichotomyAddMeanR, dichotomyAddMeanR, countScale, pregLoop);
                Muls(tmp, dichotomyAddMeanR, coeff, pregLoop1);
                Copy<float, MaskMergeMode::MERGING>(dichotomyAddMeanR, tmp, pregLoop1);
                Add(sumMean, dichotomyAddMeanL, dichotomyAddMeanR, pregMain);
                Reduce<ReduceType::SUM>(mean, sumMean, pregMain);
                StoreAlign<float, StoreDist::DIST_FIRST_ELEMENT_B32>(dichotomyAddLocal + i + welfordDiffLoopCount, mean,
                                                                     pregMerge);
            }

            // 整块使用tailCountScale,尾块使用countScale
            for (uint16_t i = 0; i < dichotomyAddReminderLoopCount; i++) {
                pregLoop = AscendC::MicroAPI::UpdateMask<float>(sreg0);
                LoadAlign(dichotomyAddMeanL,
                          tmpMeanLocal + (i + welfordDiffLoopCount) * VL_FP32 + welfordDiffReminderAlign);
                LoadAlign(dichotomyAddMeanR, tmpMeanLocal + (i + welfordDiffLoopCount) * VL_FP32 +
                                                 welfordDiffReminderAlign + dichotomyAddPower);
                Muls(dichotomyAddMeanL, dichotomyAddMeanL, tailCountScale, pregMain);
                Muls(dichotomyAddMeanR, dichotomyAddMeanR, countScale, pregLoop);
                Add(sumMean, dichotomyAddMeanL, dichotomyAddMeanR, pregMain);
                Reduce<ReduceType::SUM>(mean, sumMean, pregMain);
                StoreAlign<float, StoreDist::DIST_FIRST_ELEMENT_B32>(
                    dichotomyAddLocal + i + welfordDiffLoopCount + welfordReminderLoopCount, mean, pregMerge);
            }
            // PART2: 整块剩余部分vcadd回刷UB,使用tailCountScale
            for (uint16_t i = 0;
                 i < static_cast<uint16_t>(dichotomyAddPowerLoopCount - dichotomyAddReminderRealLoopCount); i++) {
                LoadAlign(dichotomyAddMeanL, tmpMeanLocal + (i + dichotomyAddReminderRealLoopCount) * VL_FP32);
                Muls(dichotomyAddMeanL, dichotomyAddMeanL, tailCountScale, pregMain);
                Reduce<ReduceType::SUM>(mean, dichotomyAddMeanL, pregMain);
                StoreAlign<float, StoreDist::DIST_FIRST_ELEMENT_B32>(
                    dichotomyAddLocal + dichotomyAddReminderRealLoopCount + i, mean, pregMerge);
            }
            // 末级归约换成补偿求和 + mean 缩放改「真除法」：NonAlign 流式累加时每项已乘
            // reduceScale0 = 1/cofFactor0（cofFactor0 = FindCofFactor(elemNum)，2 的幂，该缩放精确），
            // 故此时 mean = (Σx)/cofFactor0。真除法 ÷elemNum（< 2^24 可精确表示）后乘回
            // cofFactor0（2 的幂，精确），等价于一次正确舍入的除法；不再乘倒数常数。
            // 口径与同函数下方 var 收尾（÷n 再 ×cofFactor0）一致。
            WelfordDichotomyAddCompensated(mean, dichotomyAddLocal, dichotomyAddK, innerLoopCountOrigin,
                                           dichotomyAddLastNum);
            Duplicate(nReg, vfN, pregMain);
            Div(mean, mean, nReg, pregMerge);
            Muls(mean, mean, vfCofFactor, pregMerge);
            RestoreUniformFiniteMeanAndZeroVar(mean, tmpMeanLocal, tmpVarLocal, reduceCount, pregMerge, pregMain,
                                               zeroVarMask, dichotomyAddMeanL, dichotomyAddMeanR, dichotomyAddVarL,
                                               dichotomyAddVarR, sumMean, sumVar);
            StoreAlign<float, StoreDist::DIST_FIRST_ELEMENT_B32>(meanLocal + offset, mean, pregMerge);

            // 计算rstd
            Duplicate(one, float(1.0), pregMain);
            Duplicate<float, AscendC::MicroAPI::HighLowPart::LOWEST, MaskMergeMode::ZEROING>(mean, mean, pregMain);
            for (uint16_t i = 0; i < welfordDiffLoopCount; i++) {
                LoadAlign(dichotomyAddMeanL, tmpMeanLocal + i * VL_FP32);
                Sub(deltaL, dichotomyAddMeanL, mean, pregMain);
                Mul(deltaL, deltaL, deltaL, pregMain);
                Muls(deltaL, deltaL, tailCnt, pregMain);
                LoadAlign(dichotomyAddMeanR, tmpMeanLocal + i * VL_FP32 + dichotomyAddPower);
                Sub(deltaR, dichotomyAddMeanR, mean, pregMain);
                Mul(deltaR, deltaR, deltaR, pregMain);
                Muls(deltaR, deltaR, tailCnt, pregMain);

                LoadAlign(dichotomyAddVarL, tmpVarLocal + i * VL_FP32);
                Add(dichotomyAddVarL, dichotomyAddVarL, deltaL, pregMain);
                Muls(dichotomyAddVarL, dichotomyAddVarL, reduceScale0, pregMain);
                LoadAlign(dichotomyAddVarR, tmpVarLocal + i * VL_FP32 + dichotomyAddPower);
                Add(dichotomyAddVarR, dichotomyAddVarR, deltaR, pregMain);
                Muls(dichotomyAddVarR, dichotomyAddVarR, reduceScale0, pregMain);

                Add(sumVar, dichotomyAddVarL, dichotomyAddVarR, pregMain);
                Reduce<ReduceType::SUM>(var, sumVar, pregMain);
                StoreAlign<float, StoreDist::DIST_FIRST_ELEMENT_B32>(dichotomyAddLocal + i, var, pregMerge);
            }
            sreg0 = dichotomyAddReminder - welfordDiffLoopCount * VL_FP32;
            sreg1 = welfordDiffReminder;
            for (uint16_t i = 0; i < welfordReminderLoopCount; i++) {
                pregLoop = AscendC::MicroAPI::UpdateMask<float>(sreg0);
                pregLoop1 = AscendC::MicroAPI::UpdateMask<float>(sreg1);
                LoadAlign(dichotomyAddMeanL, tmpMeanLocal + (i + welfordDiffLoopCount) * VL_FP32);
                Sub(deltaL, dichotomyAddMeanL, mean, pregMain);
                Mul(deltaL, deltaL, deltaL, pregMain);
                Muls(deltaL, deltaL, tailCnt, pregMain);
                LoadAlign(dichotomyAddMeanR, tmpMeanLocal + (i + welfordDiffLoopCount) * VL_FP32 + dichotomyAddPower);
                Sub(deltaR, dichotomyAddMeanR, mean, pregLoop);
                Mul(deltaR, deltaR, deltaR, pregLoop);
                Muls(deltaR, deltaR, cnt, pregLoop);
                Muls(tmp, deltaR, coeff, pregLoop1);
                Copy<float, MaskMergeMode::MERGING>(deltaR, tmp, pregLoop1);

                LoadAlign(dichotomyAddVarL, tmpVarLocal + (i + welfordDiffLoopCount) * VL_FP32);
                Add(dichotomyAddVarL, dichotomyAddVarL, deltaL, pregMain);
                Muls(dichotomyAddVarL, dichotomyAddVarL, reduceScale0, pregMain);
                LoadAlign(dichotomyAddVarR, tmpVarLocal + (i + welfordDiffLoopCount) * VL_FP32 + dichotomyAddPower);
                Add(dichotomyAddVarR, dichotomyAddVarR, deltaR, pregLoop);
                Muls(dichotomyAddVarR, dichotomyAddVarR, reduceScale0, pregLoop);

                Add(sumVar, dichotomyAddVarL, dichotomyAddVarR, pregMain);
                Reduce<ReduceType::SUM>(var, sumVar, pregMain);
                StoreAlign<float, StoreDist::DIST_FIRST_ELEMENT_B32>(dichotomyAddLocal + i + welfordDiffLoopCount, var,
                                                                     pregMerge);
            }

            for (uint16_t i = 0; i < dichotomyAddReminderLoopCount; i++) {
                pregLoop = AscendC::MicroAPI::UpdateMask<float>(sreg0);
                LoadAlign(dichotomyAddMeanL,
                          tmpMeanLocal + (i + welfordDiffLoopCount) * VL_FP32 + welfordDiffReminderAlign);
                Sub(deltaL, dichotomyAddMeanL, mean, pregMain);
                Mul(deltaL, deltaL, deltaL, pregMain);
                Muls(deltaL, deltaL, tailCnt, pregMain);
                LoadAlign(dichotomyAddMeanR, tmpMeanLocal + (i + welfordDiffLoopCount) * VL_FP32 +
                                                 welfordDiffReminderAlign + dichotomyAddPower);
                Sub(deltaR, dichotomyAddMeanR, mean, pregLoop);
                Mul(deltaR, deltaR, deltaR, pregLoop);
                Muls(deltaR, deltaR, cnt, pregLoop);

                LoadAlign(dichotomyAddVarL,
                          tmpVarLocal + (i + welfordDiffLoopCount) * VL_FP32 + welfordDiffReminderAlign);
                Add(dichotomyAddVarL, dichotomyAddVarL, deltaL, pregMain);
                Muls(dichotomyAddVarL, dichotomyAddVarL, reduceScale0, pregMain);
                LoadAlign(dichotomyAddVarR, tmpVarLocal + (i + welfordDiffLoopCount) * VL_FP32 +
                                                welfordDiffReminderAlign + dichotomyAddPower);
                Add(dichotomyAddVarR, dichotomyAddVarR, deltaR, pregLoop);
                Muls(dichotomyAddVarR, dichotomyAddVarR, reduceScale0, pregLoop);
                Add(sumVar, dichotomyAddVarL, dichotomyAddVarR, pregMain);
                Reduce<ReduceType::SUM>(var, sumVar, pregMain);
                StoreAlign<float, StoreDist::DIST_FIRST_ELEMENT_B32>(
                    dichotomyAddLocal + i + welfordDiffLoopCount + welfordReminderLoopCount, var, pregMerge);
            }
            for (uint16_t i = 0;
                 i < static_cast<uint16_t>(dichotomyAddPowerLoopCount - dichotomyAddReminderRealLoopCount); i++) {
                LoadAlign(dichotomyAddMeanL, tmpMeanLocal + (i + dichotomyAddReminderRealLoopCount) * VL_FP32);
                Sub(deltaL, dichotomyAddMeanL, mean, pregMain);
                Mul(deltaL, deltaL, deltaL, pregMain);
                Muls(deltaL, deltaL, tailCnt, pregMain);
                LoadAlign(dichotomyAddVarL, tmpVarLocal + (i + dichotomyAddReminderRealLoopCount) * VL_FP32);
                Add(dichotomyAddVarL, dichotomyAddVarL, deltaL, pregMain);
                Muls(dichotomyAddVarL, dichotomyAddVarL, reduceScale0, pregMain);
                Reduce<ReduceType::SUM>(var, dichotomyAddVarL, pregMain);
                StoreAlign<float, StoreDist::DIST_FIRST_ELEMENT_B32>(
                    dichotomyAddLocal + dichotomyAddReminderRealLoopCount + i, var, pregMerge);
            }
            // 末级归约换成补偿求和（M2 的末级横向求和与 mean 同口径，误差经 rsqrt 折算进 rstd）。
            WelfordDichotomyAddCompensated(var, dichotomyAddLocal, dichotomyAddK, innerLoopCountOrigin,
                                           dichotomyAddLastNum);
            // 累加时每项已乘 reduceScale0 = 1/cofFactor0（cofFactor0 是 2 的幂，该缩放精确），
            // 故此时 var = M2/cofFactor0。有偏方差 M2/n 与无偏方差 M2/(n-1) 都用**真除法**：
            // n 与 n-1 均 < 2^24、在 fp32 中精确可表示，真除法只引入 ≤0.5 ULP 的随机舍入；
            // 而乘倒数会把常数自身的舍入误差叠加成所有通道同向的系统性偏差（实测约 -0.3 ULP）。
            // 先除再乘回 cofFactor0（2 的幂，精确、且中间值更小不会溢出），结果等价于除以 n / n-1。
            // varLocal 写**有偏方差** M2/n（第 5 个输出 output_reserve_space_2 的契约值），
            // unbiasedVarLocal 写**无偏方差** M2/(n-1)（只供 running var 使用），rstd 单独写 rstdLocal。
            Duplicate(nReg, vfN, pregMain);
            Duplicate(nMinusOneReg, vfNMinusOne, pregMain);
            Div(varUnbiased, var, nMinusOneReg, pregMerge);
            Div(var, var, nReg, pregMerge);
            Muls(varUnbiased, varUnbiased, vfCofFactor, pregMerge);
            Muls(var, var, vfCofFactor, pregMerge);
            // uniform && allM2Zero 时方差精确为 0；掩码不命中时 Select 选回原值。
            Duplicate(sumMean, float(0.0), pregMerge);
            AscendC::Reg::Select(var, sumMean, var, zeroVarMask);
            AscendC::Reg::Select(varUnbiased, sumMean, varUnbiased, zeroVarMask);
            StoreAlign<float, StoreDist::DIST_FIRST_ELEMENT_B32>(varLocal + offset, var, pregMerge);
            StoreAlign<float, StoreDist::DIST_FIRST_ELEMENT_B32>(unbiasedVarLocal + offset, varUnbiased, pregMerge);
            ComputeRstdNewtonRaphsonRegExt2<false>(var, rstd, pregMerge, eps);
            StoreAlign<float, StoreDist::DIST_FIRST_ELEMENT_B32>(rstdLocal + offset, rstd, pregMerge);
        }
    }

    // welford整块小于二分累加整块，并且小于等于二分累加尾块向上对齐
    __aicore__ inline void VFWelfordParallelFinalizeNonAlignSituation2(
        __ubuf__ float* meanLocal, __ubuf__ float* rstdLocal, __ubuf__ float* varLocal,
        __ubuf__ float* unbiasedVarLocal, __ubuf__ float* tmpMeanLocal, __ubuf__ float* tmpVarLocal,
        __ubuf__ float* dichotomyAddLocal, uint32_t reduceCount, uint32_t dichotomyAddPower, uint32_t dichotomyAddK,
        uint32_t dichotomyAddLastNum, uint32_t offset, uint32_t tailSize, float reduceScale0, float cnt, float tailCnt,
        float eps)
    {
        float vfN = this->nValue;
        float vfNMinusOne = this->nMinusOneValue;
        float vfCofFactor = this->cofFactor0Value;
        float coeff = tailCnt / cnt;
        float tailCountScale = tailCnt * reduceScale0;
        float countScale = cnt * reduceScale0;

        uint32_t dichotomyAddReminder = reduceCount - dichotomyAddPower;
        uint16_t welfordDiffLoopCount = tailSize / VL_FP32;
        uint32_t welfordDiffReminder = tailSize - welfordDiffLoopCount * VL_FP32;
        uint32_t welfordDiffReminderAlign = welfordDiffReminder == 0 ? 0 : VL_FP32;
        uint16_t welfordReminderLoopCount = welfordDiffReminderAlign / VL_FP32;

        uint16_t dichotomyAddReminderRealLoopCount = ops::CeilDiv(dichotomyAddReminder, static_cast<uint32_t>(VL_FP32));
        uint16_t dichotomyAddPowerLoopCount = dichotomyAddPower / VL_FP32;
        uint32_t tmpReduceCount = dichotomyAddPower / VL_FP32;
        uint16_t innerLoopCountOrigin = tmpReduceCount / VL_FP32;

        uint32_t dichotomyAddReminderAfterSplit = dichotomyAddReminder - welfordDiffLoopCount * VL_FP32 -
                                                  welfordDiffReminderAlign;
        uint16_t dichotomyAddReminderLoopCount = ops::CeilDiv(dichotomyAddReminderAfterSplit,
                                                              static_cast<uint32_t>(VL_FP32));
        __VEC_SCOPE__
        {
            RegTensor<float> dichotomyAddMeanL;
            RegTensor<float> dichotomyAddMeanR;
            RegTensor<float> dichotomyAddVarL;
            RegTensor<float> dichotomyAddVarR;
            RegTensor<float> sumMean;
            RegTensor<float> mean;
            RegTensor<float> sumVar;
            RegTensor<float> var;
            RegTensor<float> deltaL;
            RegTensor<float> deltaR;
            RegTensor<float> one;
            RegTensor<float> rstd;
            RegTensor<float> tmp;
            RegTensor<float> varUnbiased;
            RegTensor<float> nReg;
            RegTensor<float> nMinusOneReg;

            MaskReg pregLoop;
            MaskReg pregLoop1;
            MaskReg pregMain = AscendC::MicroAPI::CreateMask<float, AscendC::MicroAPI::MaskPattern::ALL>();
            MaskReg pregMerge = AscendC::MicroAPI::CreateMask<float, AscendC::MicroAPI::MaskPattern::VL1>();
            MaskReg zeroVarMask;
            uint32_t sreg0;
            uint32_t sreg1;

            // 整块使用tailCountScale,尾块使用countScale
            for (uint16_t i = 0; i < welfordDiffLoopCount; i++) {
                LoadAlign(dichotomyAddMeanL, tmpMeanLocal + i * VL_FP32);
                LoadAlign(dichotomyAddMeanR, tmpMeanLocal + i * VL_FP32 + dichotomyAddPower);
                Muls(dichotomyAddMeanL, dichotomyAddMeanL, tailCountScale, pregMain);
                Muls(dichotomyAddMeanR, dichotomyAddMeanR, countScale, pregMain);
                Add(sumMean, dichotomyAddMeanL, dichotomyAddMeanR, pregMain);
                Reduce<ReduceType::SUM>(mean, sumMean, pregMain);
                StoreAlign<float, StoreDist::DIST_FIRST_ELEMENT_B32>(dichotomyAddLocal + i, mean, pregMerge);
            }

            // 处理welford第一次非对齐点, 尾块使用countScale,整块部分使用tailCountScale, 部分使用countScale
            sreg0 = dichotomyAddReminder - welfordDiffLoopCount * VL_FP32;
            sreg1 = welfordDiffReminder;
            for (uint16_t i = 0; i < welfordReminderLoopCount; i++) {
                pregLoop = AscendC::MicroAPI::UpdateMask<float>(sreg0);
                pregLoop1 = AscendC::MicroAPI::UpdateMask<float>(sreg1);
                LoadAlign(dichotomyAddMeanL, tmpMeanLocal + (i + welfordDiffLoopCount) * VL_FP32);
                LoadAlign(dichotomyAddMeanR, tmpMeanLocal + (i + welfordDiffLoopCount) * VL_FP32 + dichotomyAddPower);
                Muls(dichotomyAddMeanL, dichotomyAddMeanL, countScale, pregMain);
                Muls(dichotomyAddMeanR, dichotomyAddMeanR, countScale, pregLoop);
                Muls(tmp, dichotomyAddMeanL, coeff, pregLoop1);
                Copy<float, MaskMergeMode::MERGING>(dichotomyAddMeanL, tmp, pregLoop1);
                Add(sumMean, dichotomyAddMeanL, dichotomyAddMeanR, pregMain);
                Reduce<ReduceType::SUM>(mean, sumMean, pregMain);
                StoreAlign<float, StoreDist::DIST_FIRST_ELEMENT_B32>(dichotomyAddLocal + i + welfordDiffLoopCount, mean,
                                                                     pregMerge);
            }

            // 整块使用countScale,尾块使用countScale
            for (uint16_t i = 0; i < dichotomyAddReminderLoopCount; i++) {
                pregLoop = AscendC::MicroAPI::UpdateMask<float>(sreg0);
                LoadAlign(dichotomyAddMeanL,
                          tmpMeanLocal + (i + welfordDiffLoopCount) * VL_FP32 + welfordDiffReminderAlign);
                LoadAlign(dichotomyAddMeanR, tmpMeanLocal + (i + welfordDiffLoopCount) * VL_FP32 +
                                                 welfordDiffReminderAlign + dichotomyAddPower);
                Muls(dichotomyAddMeanL, dichotomyAddMeanL, countScale, pregMain);
                Muls(dichotomyAddMeanR, dichotomyAddMeanR, countScale, pregLoop);
                Add(sumMean, dichotomyAddMeanL, dichotomyAddMeanR, pregMain);
                Reduce<ReduceType::SUM>(mean, sumMean, pregMain);
                StoreAlign<float, StoreDist::DIST_FIRST_ELEMENT_B32>(
                    dichotomyAddLocal + i + welfordDiffLoopCount + welfordReminderLoopCount, mean, pregMerge);
            }
            // PART2: 整块剩余部分vcadd回刷UB,使用countScale
            for (uint16_t i = 0;
                 i < static_cast<uint16_t>(dichotomyAddPowerLoopCount - dichotomyAddReminderRealLoopCount); i++) {
                LoadAlign(dichotomyAddMeanL, tmpMeanLocal + (i + dichotomyAddReminderRealLoopCount) * VL_FP32);
                Muls(dichotomyAddMeanL, dichotomyAddMeanL, countScale, pregMain);
                Reduce<ReduceType::SUM>(mean, dichotomyAddMeanL, pregMain);
                StoreAlign<float, StoreDist::DIST_FIRST_ELEMENT_B32>(
                    dichotomyAddLocal + dichotomyAddReminderRealLoopCount + i, mean, pregMerge);
            }
            // 末级归约换成补偿求和 + mean 缩放改「真除法」：NonAlign 流式累加时每项已乘
            // reduceScale0 = 1/cofFactor0（cofFactor0 = FindCofFactor(elemNum)，2 的幂，该缩放精确），
            // 故此时 mean = (Σx)/cofFactor0。真除法 ÷elemNum（< 2^24 可精确表示）后乘回
            // cofFactor0（2 的幂，精确），等价于一次正确舍入的除法；不再乘倒数常数。
            // 口径与同函数下方 var 收尾（÷n 再 ×cofFactor0）一致。
            WelfordDichotomyAddCompensated(mean, dichotomyAddLocal, dichotomyAddK, innerLoopCountOrigin,
                                           dichotomyAddLastNum);
            Duplicate(nReg, vfN, pregMain);
            Div(mean, mean, nReg, pregMerge);
            Muls(mean, mean, vfCofFactor, pregMerge);
            RestoreUniformFiniteMeanAndZeroVar(mean, tmpMeanLocal, tmpVarLocal, reduceCount, pregMerge, pregMain,
                                               zeroVarMask, dichotomyAddMeanL, dichotomyAddMeanR, dichotomyAddVarL,
                                               dichotomyAddVarR, sumMean, sumVar);
            StoreAlign<float, StoreDist::DIST_FIRST_ELEMENT_B32>(meanLocal + offset, mean, pregMerge);

            // 计算rstd
            Duplicate(one, float(1.0), pregMain);
            Duplicate<float, AscendC::MicroAPI::HighLowPart::LOWEST, MaskMergeMode::ZEROING>(mean, mean, pregMain);
            for (uint16_t i = 0; i < welfordDiffLoopCount; i++) {
                LoadAlign(dichotomyAddMeanL, tmpMeanLocal + i * VL_FP32);
                Sub(deltaL, dichotomyAddMeanL, mean, pregMain);
                Mul(deltaL, deltaL, deltaL, pregMain);
                Muls(deltaL, deltaL, tailCnt, pregMain);
                LoadAlign(dichotomyAddMeanR, tmpMeanLocal + i * VL_FP32 + dichotomyAddPower);
                Sub(deltaR, dichotomyAddMeanR, mean, pregMain);
                Mul(deltaR, deltaR, deltaR, pregMain);
                Muls(deltaR, deltaR, cnt, pregMain);

                LoadAlign(dichotomyAddVarL, tmpVarLocal + i * VL_FP32);
                Add(dichotomyAddVarL, dichotomyAddVarL, deltaL, pregMain);
                Muls(dichotomyAddVarL, dichotomyAddVarL, reduceScale0, pregMain);
                LoadAlign(dichotomyAddVarR, tmpVarLocal + i * VL_FP32 + dichotomyAddPower);
                Add(dichotomyAddVarR, dichotomyAddVarR, deltaR, pregMain);
                Muls(dichotomyAddVarR, dichotomyAddVarR, reduceScale0, pregMain);

                Add(sumVar, dichotomyAddVarL, dichotomyAddVarR, pregMain);
                Reduce<ReduceType::SUM>(var, sumVar, pregMain);
                StoreAlign<float, StoreDist::DIST_FIRST_ELEMENT_B32>(dichotomyAddLocal + i, var, pregMerge);
            }
            sreg0 = dichotomyAddReminder - welfordDiffLoopCount * VL_FP32;
            sreg1 = welfordDiffReminder;
            for (uint16_t i = 0; i < welfordReminderLoopCount; i++) {
                pregLoop = AscendC::MicroAPI::UpdateMask<float>(sreg0);
                pregLoop1 = AscendC::MicroAPI::UpdateMask<float>(sreg1);
                LoadAlign(dichotomyAddMeanL, tmpMeanLocal + (i + welfordDiffLoopCount) * VL_FP32);
                Sub(deltaL, dichotomyAddMeanL, mean, pregMain);
                Mul(deltaL, deltaL, deltaL, pregMain);
                Muls(deltaL, deltaL, cnt, pregMain);
                LoadAlign(dichotomyAddMeanR, tmpMeanLocal + (i + welfordDiffLoopCount) * VL_FP32 + dichotomyAddPower);
                Sub(deltaR, dichotomyAddMeanR, mean, pregLoop);
                Mul(deltaR, deltaR, deltaR, pregLoop);
                Muls(deltaR, deltaR, cnt, pregLoop);
                Muls(tmp, deltaL, coeff, pregLoop1);
                Copy<float, MaskMergeMode::MERGING>(deltaL, tmp, pregLoop1);

                LoadAlign(dichotomyAddVarL, tmpVarLocal + (i + welfordDiffLoopCount) * VL_FP32);
                Add(dichotomyAddVarL, dichotomyAddVarL, deltaL, pregMain);
                Muls(dichotomyAddVarL, dichotomyAddVarL, reduceScale0, pregMain);
                LoadAlign(dichotomyAddVarR, tmpVarLocal + (i + welfordDiffLoopCount) * VL_FP32 + dichotomyAddPower);
                Add(dichotomyAddVarR, dichotomyAddVarR, deltaR, pregLoop);
                Muls(dichotomyAddVarR, dichotomyAddVarR, reduceScale0, pregLoop);

                Add(sumVar, dichotomyAddVarL, dichotomyAddVarR, pregMain);
                Reduce<ReduceType::SUM>(var, sumVar, pregMain);
                StoreAlign<float, StoreDist::DIST_FIRST_ELEMENT_B32>(dichotomyAddLocal + i + welfordDiffLoopCount, var,
                                                                     pregMerge);
            }

            for (uint16_t i = 0; i < dichotomyAddReminderLoopCount; i++) {
                pregLoop = AscendC::MicroAPI::UpdateMask<float>(sreg0);
                LoadAlign(dichotomyAddMeanL,
                          tmpMeanLocal + (i + welfordDiffLoopCount) * VL_FP32 + welfordDiffReminderAlign);
                Sub(deltaL, dichotomyAddMeanL, mean, pregMain);
                Mul(deltaL, deltaL, deltaL, pregMain);
                Muls(deltaL, deltaL, cnt, pregMain);
                LoadAlign(dichotomyAddMeanR, tmpMeanLocal + (i + welfordDiffLoopCount) * VL_FP32 +
                                                 welfordDiffReminderAlign + dichotomyAddPower);
                Sub(deltaR, dichotomyAddMeanR, mean, pregLoop);
                Mul(deltaR, deltaR, deltaR, pregLoop);
                Muls(deltaR, deltaR, cnt, pregLoop);

                LoadAlign(dichotomyAddVarL,
                          tmpVarLocal + (i + welfordDiffLoopCount) * VL_FP32 + welfordDiffReminderAlign);
                Add(dichotomyAddVarL, dichotomyAddVarL, deltaL, pregMain);
                Muls(dichotomyAddVarL, dichotomyAddVarL, reduceScale0, pregMain);
                LoadAlign(dichotomyAddVarR, tmpVarLocal + (i + welfordDiffLoopCount) * VL_FP32 +
                                                welfordDiffReminderAlign + dichotomyAddPower);
                Add(dichotomyAddVarR, dichotomyAddVarR, deltaR, pregLoop);
                Muls(dichotomyAddVarR, dichotomyAddVarR, reduceScale0, pregLoop);
                Add(sumVar, dichotomyAddVarL, dichotomyAddVarR, pregMain);
                Reduce<ReduceType::SUM>(var, sumVar, pregMain);
                StoreAlign<float, StoreDist::DIST_FIRST_ELEMENT_B32>(
                    dichotomyAddLocal + i + welfordDiffLoopCount + welfordReminderLoopCount, var, pregMerge);
            }

            for (uint16_t i = 0;
                 i < static_cast<uint16_t>(dichotomyAddPowerLoopCount - dichotomyAddReminderRealLoopCount); i++) {
                LoadAlign(dichotomyAddMeanL, tmpMeanLocal + (i + dichotomyAddReminderRealLoopCount) * VL_FP32);
                Sub(deltaL, dichotomyAddMeanL, mean, pregMain);
                Mul(deltaL, deltaL, deltaL, pregMain);
                Muls(deltaL, deltaL, cnt, pregMain);
                LoadAlign(dichotomyAddVarL, tmpVarLocal + (i + dichotomyAddReminderRealLoopCount) * VL_FP32);
                Add(dichotomyAddVarL, dichotomyAddVarL, deltaL, pregMain);
                Muls(dichotomyAddVarL, dichotomyAddVarL, reduceScale0, pregMain);
                Reduce<ReduceType::SUM>(var, dichotomyAddVarL, pregMain);
                StoreAlign<float, StoreDist::DIST_FIRST_ELEMENT_B32>(
                    dichotomyAddLocal + dichotomyAddReminderRealLoopCount + i, var, pregMerge);
            }
            // 末级归约换成补偿求和（M2 的末级横向求和与 mean 同口径，误差经 rsqrt 折算进 rstd）。
            WelfordDichotomyAddCompensated(var, dichotomyAddLocal, dichotomyAddK, innerLoopCountOrigin,
                                           dichotomyAddLastNum);
            // 累加时每项已乘 reduceScale0 = 1/cofFactor0（cofFactor0 是 2 的幂，该缩放精确），
            // 故此时 var = M2/cofFactor0。有偏方差 M2/n 与无偏方差 M2/(n-1) 都用**真除法**：
            // n 与 n-1 均 < 2^24、在 fp32 中精确可表示，真除法只引入 ≤0.5 ULP 的随机舍入；
            // 而乘倒数会把常数自身的舍入误差叠加成所有通道同向的系统性偏差（实测约 -0.3 ULP）。
            // 先除再乘回 cofFactor0（2 的幂，精确、且中间值更小不会溢出），结果等价于除以 n / n-1。
            // varLocal 写**有偏方差** M2/n（第 5 个输出 output_reserve_space_2 的契约值），
            // unbiasedVarLocal 写**无偏方差** M2/(n-1)（只供 running var 使用），rstd 单独写 rstdLocal。
            Duplicate(nReg, vfN, pregMain);
            Duplicate(nMinusOneReg, vfNMinusOne, pregMain);
            Div(varUnbiased, var, nMinusOneReg, pregMerge);
            Div(var, var, nReg, pregMerge);
            Muls(varUnbiased, varUnbiased, vfCofFactor, pregMerge);
            Muls(var, var, vfCofFactor, pregMerge);
            // uniform && allM2Zero 时方差精确为 0；掩码不命中时 Select 选回原值。
            Duplicate(sumMean, float(0.0), pregMerge);
            AscendC::Reg::Select(var, sumMean, var, zeroVarMask);
            AscendC::Reg::Select(varUnbiased, sumMean, varUnbiased, zeroVarMask);
            StoreAlign<float, StoreDist::DIST_FIRST_ELEMENT_B32>(varLocal + offset, var, pregMerge);
            StoreAlign<float, StoreDist::DIST_FIRST_ELEMENT_B32>(unbiasedVarLocal + offset, varUnbiased, pregMerge);
            ComputeRstdNewtonRaphsonRegExt2<false>(var, rstd, pregMerge, eps);
            StoreAlign<float, StoreDist::DIST_FIRST_ELEMENT_B32>(rstdLocal + offset, rstd, pregMerge);
        }
    }

    // 场景3：welford整块小于二分累加整块，并且大于二分累加尾块向上对齐
    __aicore__ inline void VFWelfordParallelFinalizeNonAlignSituation3(
        __ubuf__ float* meanLocal, __ubuf__ float* rstdLocal, __ubuf__ float* varLocal,
        __ubuf__ float* unbiasedVarLocal, __ubuf__ float* tmpMeanLocal, __ubuf__ float* tmpVarLocal,
        __ubuf__ float* dichotomyAddLocal, uint32_t reduceCount, uint32_t dichotomyAddPower, uint32_t dichotomyAddK,
        uint32_t dichotomyAddLastNum, uint32_t offset, uint32_t tailSize, float reduceScale0, float cnt, float tailCnt,
        float eps)
    {
        float vfN = this->nValue;
        float vfNMinusOne = this->nMinusOneValue;
        float vfCofFactor = this->cofFactor0Value;
        float coeff = tailCnt / cnt;
        float tailCountScale = tailCnt * reduceScale0;
        float countScale = cnt * reduceScale0;

        // 二分累加
        uint32_t dichotomyAddReminder = reduceCount - dichotomyAddPower;
        uint16_t dichotomyAddReminderLoopCount = ops::CeilDiv(dichotomyAddReminder, static_cast<uint32_t>(VL_FP32));
        uint16_t dichotomyAddPowerLoopCount = dichotomyAddPower / VL_FP32;
        uint32_t tmpReduceCount = dichotomyAddPower / VL_FP32;
        uint16_t innerLoopCountOrigin = tmpReduceCount / VL_FP32;
        uint32_t dichotomyAddReminderRoundUp = dichotomyAddReminderLoopCount * VL_FP32;

        uint32_t welfordDiff = tailSize - dichotomyAddReminderRoundUp;
        uint16_t welfordDiffLoopCount = welfordDiff / VL_FP32;
        uint32_t welfordDiffReminder = welfordDiff - welfordDiffLoopCount * VL_FP32;
        uint32_t welfordDiffReminderAlign = welfordDiffReminder == 0 ? 0 : VL_FP32;
        uint16_t welfordReminderLoopCount = welfordDiffReminderAlign / VL_FP32;
        uint16_t dichotomyAddPowerRemainLoopCount = dichotomyAddPowerLoopCount - dichotomyAddReminderLoopCount -
                                                    welfordDiffLoopCount - welfordReminderLoopCount;
        uint32_t dichotomyAddPowerOffset = dichotomyAddReminderRoundUp + welfordDiffLoopCount * VL_FP32 +
                                           welfordDiffReminderAlign;

        __VEC_SCOPE__
        {
            RegTensor<float> dichotomyAddMeanL;
            RegTensor<float> dichotomyAddMeanR;
            RegTensor<float> dichotomyAddVarL;
            RegTensor<float> dichotomyAddVarR;
            RegTensor<float> sumMean;
            RegTensor<float> mean;
            RegTensor<float> sumVar;
            RegTensor<float> var;
            RegTensor<float> deltaL;
            RegTensor<float> deltaR;
            RegTensor<float> one;
            RegTensor<float> rstd;
            RegTensor<float> tmp;
            RegTensor<float> varUnbiased;
            RegTensor<float> nReg;
            RegTensor<float> nMinusOneReg;

            MaskReg pregLoop;
            MaskReg pregMain = AscendC::MicroAPI::CreateMask<float, AscendC::MicroAPI::MaskPattern::ALL>();
            MaskReg pregMerge = AscendC::MicroAPI::CreateMask<float, AscendC::MicroAPI::MaskPattern::VL1>();
            MaskReg zeroVarMask;
            uint32_t sreg0 = dichotomyAddReminder;
            // 整块使用tailCountScale, 尾块使用CountScale
            for (uint16_t i = 0; i < dichotomyAddReminderLoopCount; i++) {
                pregLoop = AscendC::MicroAPI::UpdateMask<float>(sreg0);
                LoadAlign(dichotomyAddMeanL, tmpMeanLocal + i * VL_FP32);
                LoadAlign(dichotomyAddMeanR, tmpMeanLocal + i * VL_FP32 + dichotomyAddPower);
                Muls(dichotomyAddMeanL, dichotomyAddMeanL, tailCountScale, pregMain);
                Muls(dichotomyAddMeanR, dichotomyAddMeanR, countScale, pregLoop);
                Add(sumMean, dichotomyAddMeanL, dichotomyAddMeanR, pregMain);
                Reduce<ReduceType::SUM>(mean, sumMean, pregMain);
                StoreAlign<float, StoreDist::DIST_FIRST_ELEMENT_B32>(dichotomyAddLocal + i, mean, pregMerge);
            }

            // 剩余整块需要拆分成多部分
            // 整块剩余部分回刷UB，整块使用tailCountScale
            for (uint16_t i = 0; i < welfordDiffLoopCount; i++) {
                LoadAlign(dichotomyAddMeanL, tmpMeanLocal + i * VL_FP32 + dichotomyAddReminderRoundUp);
                Muls(dichotomyAddMeanL, dichotomyAddMeanL, tailCountScale, pregMain);
                Reduce<ReduceType::SUM>(mean, dichotomyAddMeanL, pregMain);
                StoreAlign<float, StoreDist::DIST_FIRST_ELEMENT_B32>(
                    dichotomyAddLocal + dichotomyAddReminderLoopCount + i, mean, pregMerge);
            }

            sreg0 = welfordDiffReminder;
            for (uint16_t i = 0; i < welfordReminderLoopCount; i++) {
                pregLoop = AscendC::MicroAPI::UpdateMask<float>(sreg0);
                LoadAlign(dichotomyAddMeanL,
                          tmpMeanLocal + (i + welfordDiffLoopCount) * VL_FP32 + dichotomyAddReminderRoundUp);
                Muls(dichotomyAddMeanL, dichotomyAddMeanL, countScale, pregMain);
                Muls(tmp, dichotomyAddMeanL, coeff, pregLoop);
                Copy<float, MaskMergeMode::MERGING>(dichotomyAddMeanL, tmp, pregLoop);
                Reduce<ReduceType::SUM>(mean, dichotomyAddMeanL, pregMain);
                StoreAlign<float, StoreDist::DIST_FIRST_ELEMENT_B32>(
                    dichotomyAddLocal + dichotomyAddReminderLoopCount + welfordDiffLoopCount + i, mean, pregMerge);
            }

            for (uint16_t i = 0; i < dichotomyAddPowerRemainLoopCount; i++) {
                LoadAlign(dichotomyAddMeanL, tmpMeanLocal + i * VL_FP32 + dichotomyAddPowerOffset);
                Muls(dichotomyAddMeanL, dichotomyAddMeanL, countScale, pregMain);
                Reduce<ReduceType::SUM>(mean, dichotomyAddMeanL, pregMain);
                StoreAlign<float, StoreDist::DIST_FIRST_ELEMENT_B32>(dichotomyAddLocal + dichotomyAddReminderLoopCount +
                                                                         welfordDiffLoopCount +
                                                                         welfordReminderLoopCount + i,
                                                                     mean, pregMerge);
            }

            // 末级归约换成补偿求和 + mean 缩放改「真除法」：NonAlign 流式累加时每项已乘
            // reduceScale0 = 1/cofFactor0（cofFactor0 = FindCofFactor(elemNum)，2 的幂，该缩放精确），
            // 故此时 mean = (Σx)/cofFactor0。真除法 ÷elemNum（< 2^24 可精确表示）后乘回
            // cofFactor0（2 的幂，精确），等价于一次正确舍入的除法；不再乘倒数常数。
            // 口径与同函数下方 var 收尾（÷n 再 ×cofFactor0）一致。
            WelfordDichotomyAddCompensated(mean, dichotomyAddLocal, dichotomyAddK, innerLoopCountOrigin,
                                           dichotomyAddLastNum);
            Duplicate(nReg, vfN, pregMain);
            Div(mean, mean, nReg, pregMerge);
            Muls(mean, mean, vfCofFactor, pregMerge);
            RestoreUniformFiniteMeanAndZeroVar(mean, tmpMeanLocal, tmpVarLocal, reduceCount, pregMerge, pregMain,
                                               zeroVarMask, dichotomyAddMeanL, dichotomyAddMeanR, dichotomyAddVarL,
                                               dichotomyAddVarR, sumMean, sumVar);
            StoreAlign<float, StoreDist::DIST_FIRST_ELEMENT_B32>(meanLocal + offset, mean, pregMerge);

            // 计算rstd
            Duplicate(one, float(1.0), pregMain);
            Duplicate<float, AscendC::MicroAPI::HighLowPart::LOWEST, MaskMergeMode::ZEROING>(mean, mean, pregMain);
            // 整块使用tailCountScale, 尾块使用CountScale
            sreg0 = dichotomyAddReminder;
            for (uint16_t i = 0; i < dichotomyAddReminderLoopCount; i++) {
                pregLoop = AscendC::MicroAPI::UpdateMask<float>(sreg0);
                LoadAlign(dichotomyAddMeanL, tmpMeanLocal + i * VL_FP32);
                Sub(deltaL, dichotomyAddMeanL, mean, pregMain);
                Mul(deltaL, deltaL, deltaL, pregMain);
                Muls(deltaL, deltaL, tailCnt, pregMain);
                LoadAlign(dichotomyAddVarL, tmpVarLocal + i * VL_FP32);
                Add(dichotomyAddVarL, dichotomyAddVarL, deltaL, pregMain);
                Muls(dichotomyAddVarL, dichotomyAddVarL, reduceScale0, pregMain);

                LoadAlign(dichotomyAddMeanR, tmpMeanLocal + i * VL_FP32 + dichotomyAddPower);
                Sub(deltaR, dichotomyAddMeanR, mean, pregLoop);
                Mul(deltaR, deltaR, deltaR, pregLoop);
                Muls(deltaR, deltaR, cnt, pregLoop);
                LoadAlign(dichotomyAddVarR, tmpVarLocal + i * VL_FP32 + dichotomyAddPower);
                Add(dichotomyAddVarR, dichotomyAddVarR, deltaR, pregLoop);
                Muls(dichotomyAddVarR, dichotomyAddVarR, reduceScale0, pregLoop);

                Add(sumVar, dichotomyAddVarL, dichotomyAddVarR, pregMain);
                Reduce<ReduceType::SUM>(var, sumVar, pregMain);
                StoreAlign<float, StoreDist::DIST_FIRST_ELEMENT_B32>(dichotomyAddLocal + i, var, pregMerge);
            }

            // 整块剩余部分回刷UB，整块使用tailCountScale
            for (uint16_t i = 0; i < welfordDiffLoopCount; i++) {
                LoadAlign(dichotomyAddMeanL, tmpMeanLocal + i * VL_FP32 + dichotomyAddReminderRoundUp);
                Sub(deltaL, dichotomyAddMeanL, mean, pregMain);
                Mul(deltaL, deltaL, deltaL, pregMain);
                Muls(deltaL, deltaL, tailCnt, pregMain);
                LoadAlign(dichotomyAddVarL, tmpVarLocal + i * VL_FP32 + dichotomyAddReminderRoundUp);
                Add(dichotomyAddVarL, dichotomyAddVarL, deltaL, pregMain);
                Muls(dichotomyAddVarL, dichotomyAddVarL, reduceScale0, pregMain);
                Reduce<ReduceType::SUM>(var, dichotomyAddVarL, pregMain);
                StoreAlign<float, StoreDist::DIST_FIRST_ELEMENT_B32>(
                    dichotomyAddLocal + dichotomyAddReminderLoopCount + i, var, pregMerge);
            }

            sreg0 = welfordDiffReminder;
            for (uint16_t i = 0; i < welfordReminderLoopCount; i++) {
                pregLoop = AscendC::MicroAPI::UpdateMask<float>(sreg0);
                LoadAlign(dichotomyAddMeanL,
                          tmpMeanLocal + (i + welfordDiffLoopCount) * VL_FP32 + dichotomyAddReminderRoundUp);
                Sub(deltaL, dichotomyAddMeanL, mean, pregMain);
                Mul(deltaL, deltaL, deltaL, pregMain);
                Muls(deltaL, deltaL, cnt, pregMain);
                Muls(tmp, deltaL, coeff, pregLoop);
                Copy<float, MaskMergeMode::MERGING>(deltaL, tmp, pregLoop);
                LoadAlign(dichotomyAddVarL,
                          tmpVarLocal + (i + welfordDiffLoopCount) * VL_FP32 + dichotomyAddReminderRoundUp);
                Add(dichotomyAddVarL, dichotomyAddVarL, deltaL, pregMain);
                Muls(dichotomyAddVarL, dichotomyAddVarL, reduceScale0, pregMain);
                Reduce<ReduceType::SUM>(var, dichotomyAddVarL, pregMain);
                StoreAlign<float, StoreDist::DIST_FIRST_ELEMENT_B32>(
                    dichotomyAddLocal + dichotomyAddReminderLoopCount + welfordDiffLoopCount + i, var, pregMerge);
            }

            for (uint16_t i = 0; i < dichotomyAddPowerRemainLoopCount; i++) {
                LoadAlign(dichotomyAddMeanL, tmpMeanLocal + i * VL_FP32 + dichotomyAddPowerOffset);
                Sub(deltaL, dichotomyAddMeanL, mean, pregMain);
                Mul(deltaL, deltaL, deltaL, pregMain);
                Muls(deltaL, deltaL, cnt, pregMain);
                LoadAlign(dichotomyAddVarL, tmpVarLocal + i * VL_FP32 + dichotomyAddPowerOffset);
                Add(dichotomyAddVarL, dichotomyAddVarL, deltaL, pregMain);
                Muls(dichotomyAddVarL, dichotomyAddVarL, reduceScale0, pregMain);
                Reduce<ReduceType::SUM>(var, dichotomyAddVarL, pregMain);
                StoreAlign<float, StoreDist::DIST_FIRST_ELEMENT_B32>(dichotomyAddLocal + dichotomyAddReminderLoopCount +
                                                                         welfordDiffLoopCount +
                                                                         welfordReminderLoopCount + i,
                                                                     var, pregMerge);
            }

            // 末级归约换成补偿求和（M2 的末级横向求和与 mean 同口径，误差经 rsqrt 折算进 rstd）。
            WelfordDichotomyAddCompensated(var, dichotomyAddLocal, dichotomyAddK, innerLoopCountOrigin,
                                           dichotomyAddLastNum);
            // 累加时每项已乘 reduceScale0 = 1/cofFactor0（cofFactor0 是 2 的幂，该缩放精确），
            // 故此时 var = M2/cofFactor0。有偏方差 M2/n 与无偏方差 M2/(n-1) 都用**真除法**：
            // n 与 n-1 均 < 2^24、在 fp32 中精确可表示，真除法只引入 ≤0.5 ULP 的随机舍入；
            // 而乘倒数会把常数自身的舍入误差叠加成所有通道同向的系统性偏差（实测约 -0.3 ULP）。
            // 先除再乘回 cofFactor0（2 的幂，精确、且中间值更小不会溢出），结果等价于除以 n / n-1。
            // varLocal 写**有偏方差** M2/n（第 5 个输出 output_reserve_space_2 的契约值），
            // unbiasedVarLocal 写**无偏方差** M2/(n-1)（只供 running var 使用），rstd 单独写 rstdLocal。
            Duplicate(nReg, vfN, pregMain);
            Duplicate(nMinusOneReg, vfNMinusOne, pregMain);
            Div(varUnbiased, var, nMinusOneReg, pregMerge);
            Div(var, var, nReg, pregMerge);
            Muls(varUnbiased, varUnbiased, vfCofFactor, pregMerge);
            Muls(var, var, vfCofFactor, pregMerge);
            // uniform && allM2Zero 时方差精确为 0；掩码不命中时 Select 选回原值。
            Duplicate(sumMean, float(0.0), pregMerge);
            AscendC::Reg::Select(var, sumMean, var, zeroVarMask);
            AscendC::Reg::Select(varUnbiased, sumMean, varUnbiased, zeroVarMask);
            StoreAlign<float, StoreDist::DIST_FIRST_ELEMENT_B32>(varLocal + offset, var, pregMerge);
            StoreAlign<float, StoreDist::DIST_FIRST_ELEMENT_B32>(unbiasedVarLocal + offset, varUnbiased, pregMerge);
            ComputeRstdNewtonRaphsonRegExt2<false>(var, rstd, pregMerge, eps);
            StoreAlign<float, StoreDist::DIST_FIRST_ELEMENT_B32>(rstdLocal + offset, rstd, pregMerge);
        }
    }

    __aicore__ inline void VFWelfordParallelFinalizeNonAlign(__ubuf__ float* meanLocal, __ubuf__ float* rstdLocal,
                                                             __ubuf__ float* varLocal, __ubuf__ float* unbiasedVarLocal,
                                                             __ubuf__ float* tmpMeanLocal, __ubuf__ float* tmpVarLocal,
                                                             __ubuf__ float* dichotomyAddLocal, uint32_t reduceCount,
                                                             uint32_t dichotomyAddPower, uint32_t dichotomyAddK,
                                                             uint32_t dichotomyAddLastNum, uint32_t offset,
                                                             uint32_t tailSize, float reduceScale0, float cnt,
                                                             float tailCnt, float eps)
    {
        // 非对齐Welford finalize阶段由于自身存在整尾块，二分折叠存在整尾块，会出现多种不同的场景，每个场景都有独立的VF
        uint32_t dichotomyAddReminder = reduceCount - dichotomyAddPower;
        uint32_t dichotomyAddReminderRoundUp = ops::CeilDiv(dichotomyAddReminder, static_cast<uint32_t>(VL_FP32)) *
                                               VL_FP32;
        if (tailSize >= dichotomyAddPower) {
            VFWelfordParallelFinalizeNonAlignSituation1(meanLocal, rstdLocal, varLocal, unbiasedVarLocal, tmpMeanLocal,
                                                        tmpVarLocal, dichotomyAddLocal, reduceCount, dichotomyAddPower,
                                                        dichotomyAddK, dichotomyAddLastNum, offset, tailSize,
                                                        reduceScale0, cnt, tailCnt, eps);
            return;
        }
        if (tailSize <= dichotomyAddReminderRoundUp) {
            VFWelfordParallelFinalizeNonAlignSituation2(meanLocal, rstdLocal, varLocal, unbiasedVarLocal, tmpMeanLocal,
                                                        tmpVarLocal, dichotomyAddLocal, reduceCount, dichotomyAddPower,
                                                        dichotomyAddK, dichotomyAddLastNum, offset, tailSize,
                                                        reduceScale0, cnt, tailCnt, eps);
            return;
        }
        VFWelfordParallelFinalizeNonAlignSituation3(
            meanLocal, rstdLocal, varLocal, unbiasedVarLocal, tmpMeanLocal, tmpVarLocal, dichotomyAddLocal, reduceCount,
            dichotomyAddPower, dichotomyAddK, dichotomyAddLastNum, offset, tailSize, reduceScale0, cnt, tailCnt, eps);
    }

    __aicore__ inline void ProcessNormalize(int32_t index_a, int32_t aParallelIndex, int32_t aParallelLength,
                                            __ubuf__ float* rstdLocal)
    {
        LocalTensor<T2> betaTensor = betaQueue.DeQue<T2>();
        LocalTensor<T2> gammaTensor = gammaQueue.DeQue<T2>();
        LocalTensor<float> batchMeanTensor = batchMeanQueue.DeQue<float>();
        // 该 tensor 承载**有偏方差**(输出)，归一化用的 rstd 来自 rstdBuff 暂存。
        LocalTensor<float> batchVarTensor = batchVarQueue.DeQue<float>();
        __ubuf__ T2* gammaLocal = (__ubuf__ T2*)gammaTensor.GetPhyAddr();
        __ubuf__ T2* betaLocal = (__ubuf__ T2*)betaTensor.GetPhyAddr();
        __ubuf__ float* meanLocal = (__ubuf__ float*)batchMeanTensor.GetPhyAddr();

        int64_t actualProcessSize = 0;
        int64_t r0FactorTail = r0 % r0Factor;
        int64_t r1FactorTail = r1 % r1Factor;
        bool r0HasTail = r0FactorTail != 0;
        bool r1HasTail = r1FactorTail != 0;
        for (int64_t i = 0; i < aParallelLength; i++) {
            for (uint64_t j = 0; j < loopR1outer; j++) {
                int64_t nBurst = r1Factor;
                if (r1HasTail && j == loopR1outer - 1) {
                    nBurst = r1FactorTail;
                }
                for (uint64_t k = 0; k < loopR0outer; k++) {
                    int64_t index = j * r1Factor * a0 * r0 + (aParallelIndex * aGatherLimit + i) * r0 + k * r0Factor;
                    // after ub spit, there may comes tail block at the end cycle.
                    int64_t burstLen = r0Factor;
                    if (r0HasTail && k == loopR0outer - 1) {
                        burstLen = r0FactorTail;
                    }

                    actualProcessSize = nBurst * burstLen;
                    CopyInX(index, nBurst, burstLen);
                    LocalTensor<T1> xInUb = xQueue.DeQue<T1>();
                    LocalTensor<T1> yOutUb = yQueue.AllocTensor<T1>();

                    __ubuf__ T1* xLocal = (__ubuf__ T1*)xInUb.GetPhyAddr();
                    __ubuf__ T1* yOutLocal = (__ubuf__ T1*)yOutUb.GetPhyAddr();

                    int32_t loopCount = ops::CeilDiv(actualProcessSize, static_cast<int64_t>(VL_FP32));
                    int32_t reduceCount = actualProcessSize;
                    VFNormalize(xLocal, gammaLocal, betaLocal, meanLocal, rstdLocal, yOutLocal, i, loopCount,
                                reduceCount);
                    yQueue.EnQue(yOutUb);
                    CopyY2Gm(index, nBurst, burstLen);
                    xQueue.FreeTensor(xInUb);
                }
            }
        }

        CopyBatchMeanAndVar2GM(index_a, aParallelLength, batchMeanTensor, batchVarTensor);
        betaQueue.FreeTensor(betaTensor);
        gammaQueue.FreeTensor(gammaTensor);
    }

    __aicore__ inline void VFNormalize(__ubuf__ T1* xLocal, __ubuf__ T2* gammaLocal, __ubuf__ T2* betaLocal,
                                       __ubuf__ float* meanLocal, __ubuf__ float* rstdLocal, __ubuf__ T1* yLocal,
                                       int64_t i, uint16_t loopCount, int32_t reduceCount)
    {
        __VEC_SCOPE__
        {
            RegTensor<float> x;
            RegTensor<float> gamma;
            RegTensor<float> beta;
            RegTensor<float> mean;
            RegTensor<float> rstd;
            RegTensor<float> y;

            MaskReg pregLoop;
            LoadAlign<float, LoadDist::DIST_BRC_B32>(rstd, rstdLocal + i);
            LoadAlign<float, LoadDist::DIST_BRC_B32>(mean, meanLocal + i);
            MaskReg pregMain = AscendC::MicroAPI::CreateMask<float, AscendC::MicroAPI::MaskPattern::ALL>();
            LoadTwoTensorForDtypeTBrc<T2>(gamma, beta, gammaLocal, betaLocal, pregMain, pregMain, i, i);
            uint32_t sreg0 = reduceCount;
            for (uint16_t j = 0; j < loopCount; j++) {
                pregLoop = AscendC::MicroAPI::UpdateMask<float>(sreg0);
                // norm
                LoadTensorForDtypeT<T1>(x, xLocal, pregLoop, j * VL_FP32);
                Sub(x, x, mean, pregLoop);
                Mul(x, x, rstd, pregLoop);
                // 与 torch.compile 的 FMA 融合对齐，减少一次舍入：y = beta + x * gamma
                AscendC::Reg::Move(y, beta);
                Mula(y, x, gamma, pregLoop);

                if constexpr (IsSameType<T1, half>::value) {
                    RegTensor<half> yFp16;
                    Cast<half, float, castTraitB322B16>(yFp16, y, pregLoop);
                    StoreAlign<half, StoreDist::DIST_PACK_B32>(((__ubuf__ half*)yLocal) + j * VL_FP32, yFp16, pregLoop);
                } else if constexpr (IsSameType<T1, bfloat16_t>::value) {
                    RegTensor<bfloat16_t> xBf16;
                    Cast<bfloat16_t, float, castTraitB322B16>(xBf16, y, pregLoop);
                    StoreAlign<bfloat16_t, StoreDist::DIST_PACK_B32>(((__ubuf__ bfloat16_t*)yLocal) + j * VL_FP32,
                                                                     xBf16, pregLoop);
                } else {
                    StoreAlign(((__ubuf__ float*)yLocal) + j * VL_FP32, y, pregLoop);
                }
            }
        }
    }

    /* global memory address */
    GlobalTensor<T1> xGm;
    GlobalTensor<T2> betaGm;
    GlobalTensor<T2> gammaGm;
    GlobalTensor<T2> meanGm;
    GlobalTensor<T2> varGm;

    GlobalTensor<T1> yGm;
    GlobalTensor<float> batchMeanGm;
    GlobalTensor<float> batchVarGm;
    GlobalTensor<T2> outMeanGm;
    GlobalTensor<T2> outVarGm;

    int64_t realCoreNum;
    int64_t numPerCore;
    int64_t numLastCore;
    int64_t elemNum; // all reduce num in one A for cycle.

    int64_t a0;
    int64_t r0;
    int64_t r1;
    float eps;
    float momentum;
    float oneSubMomentum;
    // 真除法标量：n = elemNum，均 < 2^24 可被 fp32 精确表示。
    float nValue = 0;
    float nMinusOneValue = 0;
    float cofFactor0Value = 0; // FindCofFactor(elemNum)，2 的幂
    // mean 的真除法标量：除数 parallelN 与累加时预缩放的 2 的幂 cofFactor1 = FindCofFactor(parallelN)。
    float parallelNValue = 0;
    float cofFactor1Value = 0;

    int64_t aBlockFactor;
    int64_t aGatherLimit;
    int64_t processNumPerCore; // A loop size after block spilt.
    int64_t loopR1outer;       // outside R loop size
    int64_t loopR0outer;       // inside R loop size
    int64_t r1Factor;          // outside R loop size
    int64_t r0Factor;          // inside R loop size
    int64_t ubSize;
    uint32_t binaryAddK;
    uint32_t binaryAddLastNum;
    uint32_t binaryAddQuotient;
    int64_t cutR1OrR0;

    int64_t parallelN; // welford每个分组的元素

    bool isLastCore = false; // judge if process last core currently.
    int32_t blockIdx = 0;

    float reduceScale0 = 0;
    float scale0 = 0;

    bool useRunningMeanVar = true;

    /* ascendc variable */
    TPipe* pipe;
    TQue<QuePosition::VECIN, 1> xQueue;
    TQue<QuePosition::VECIN, 1> betaQueue;
    TQue<QuePosition::VECIN, 1> gammaQueue;
    TQue<QuePosition::VECIN, 1> meanQueue;
    TQue<QuePosition::VECIN, 1> varQueue;

    TQue<QuePosition::VECOUT, 1> yQueue;
    TQue<QuePosition::VECOUT, 1> batchMeanQueue;
    TQue<QuePosition::VECOUT, 1> batchVarQueue;
    TQue<QuePosition::VECOUT, 1> outMeanQueue;
    TQue<QuePosition::VECOUT, 1> outVarQueue;

    TBuf<TPosition::VECCALC> tMeanBuff;
    TBuf<TPosition::VECCALC> tVarBuff;
    TBuf<TPosition::VECCALC> rstdBuff;
    TBuf<TPosition::VECCALC> unbiasedVarBuff;
    TBuf<TPosition::VECCALC> dichotomyAddBuf;
};
} // namespace BatchNormExt2Ops

#endif // NORM_BATCH_NORM_EXT2_WELFORD_H

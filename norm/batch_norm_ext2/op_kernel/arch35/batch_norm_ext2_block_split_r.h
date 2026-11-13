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
 * \file batch_norm_ext2_block_split_r.h
 * \brief
 */
#ifndef NORM_BATCH_NORM_EXT2_BLOCK_SPLIT_R_H
#define NORM_BATCH_NORM_EXT2_BLOCK_SPLIT_R_H

#include "batch_norm_ext2_base.h"
#include "../../norm_common/reduce_common_regbase.h"

namespace BatchNormExt2Ops {
using namespace AscendC;
using AscendC::MicroAPI::CreateMask;
using AscendC::MicroAPI::LoadDist;
using AscendC::MicroAPI::LocalMemBar;
using AscendC::MicroAPI::MaskPattern;
using AscendC::MicroAPI::MaskReg;
using AscendC::MicroAPI::MemType;
using AscendC::MicroAPI::RegTensor;
using AscendC::MicroAPI::StoreDist;
using AscendC::MicroAPI::UpdateMask;
using AscendC::Reg::LoadAlign;
using AscendC::Reg::StoreAlign;

// ===================== 末级归约的补偿(TwoSum)求和（BLOCK_SPLIT_R 专用） =====================
// LastFinalize 的末级把 usedCoreNum 个 per-core 部分和（count 加权的 mean / M2）用 BinaryAddVF
// 折半成一列结果，每个 lane 是一个独立通道。这一段是整条归约链的末级：朴素树形加法每层一次
// fp32 舍入都会原样留在 batch_mean 与 batch_var（第 5 个输出 output_reserve_space_2）上，
// 而 batch_mean 随后还要参与 (x-mean) 的近零对消、batch_var 要过 rsqrt，对这段累加误差最敏感。
// 改成补偿求和可把误差从 O(log2(N)·ε·Σ|x|) 压到 O(ε·Σ|x|)：每次加法的低位残差留在 low 里，
// 全部加完后一次性加回，末级只额外舍入一次。该收尾每个通道只跑一次、不在元素热循环里。
// 只在 LastFinalize（末级）启用；CalculateMeanVar 里每核内部的归约保持原样，不改既有口径。
// 与 full_reduce.h 的 CompensatedLastLevelReduce 同一机理，但那里归约的是同一通道的多个
// lane（需要跨 lane 合并残差），这里每个 lane 就是独立通道，残差可逐 lane 直接累加。
constexpr static float BATCH_SPLIT_MAX_FINITE_FP32 = 3.4028234663852886e38F;

// (hi, lo) += value：hi 是舍入后的和，lo 累积每次加法被舍掉的低位残差，使 hi + lo 精确等于
// 真实累加和。Knuth TwoSum 求残差，不要求 |hi| >= |value|，与 Neumaier 的分支补偿逐位等价。
__simd_callee__ inline void BatchSplitAccumulateWithError(RegTensor<float>& hi, RegTensor<float>& lo,
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
__simd_callee__ inline void BatchSplitClampResidual(RegTensor<float>& lo, MaskReg& mask)
{
    RegTensor<float> zeroReg;
    MaskReg finiteMask;
    Duplicate(zeroReg, 0.0f, mask);
    AscendC::Reg::Compares<float, AscendC::CMPMODE::LE>(finiteMask, lo, BATCH_SPLIT_MAX_FINITE_FP32, mask);
    AscendC::Reg::Select(lo, lo, zeroReg, finiteMask);
    AscendC::Reg::Compares<float, AscendC::CMPMODE::GE>(finiteMask, lo, -BATCH_SPLIT_MAX_FINITE_FP32, mask);
    AscendC::Reg::Select(lo, lo, zeroReg, finiteMask);
}

template <typename T1, typename T2>
class BatchNormExt2BlockSplitR {
public:
    __aicore__ inline uint32_t CEIL_DIV(uint32_t x, uint32_t y) { return (x + y - 1) / y; }

    __aicore__ inline uint32_t CEIL_ALIGN(uint32_t x, uint32_t y) { return (x + y - 1) / y * y; }

    __aicore__ inline BatchNormExt2BlockSplitR(const BatchNormExt2BlockSplitRTilingData* tilingDataIn, TPipe* pipeIn)
    {
        this->pipe = pipeIn;
        tilingData = tilingDataIn;
        // 无偏方差不再乘 Bessel 常数 n/(n-1)：该常数无法在 fp32 中精确表示，其舍入误差会
        // 变成所有通道同向的系统性常数偏差（实测约 -0.3 ULP）。改为在 LastFinalize 里用
        // 真除法直接算 M2/(n-1)（n 与 n-1 均 < 2^24，fp32 精确可表示）。
    }

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR gamma, GM_ADDR beta, GM_ADDR mean, GM_ADDR var, GM_ADDR y,
                                GM_ADDR mean_out, GM_ADDR var_out, GM_ADDR batch_mean, GM_ADDR batch_var,
                                GM_ADDR workspace)
    {
        blockIdx = GetBlockIdx();
        usedCoreNum = GetBlockNum();
        int64_t rBlockOffset = 0;
        if (blockIdx < tilingData->formerCoreNums) {
            this->rLoop = tilingData->formerCoreBlockFactor;
            rBlockOffset = blockIdx * this->rLoop * tilingData->rUbFactor;
        } else {
            this->rLoop = tilingData->tailCoreBlockFactor;
            rBlockOffset = (tilingData->formerCoreBlockFactor * tilingData->formerCoreNums +
                            tilingData->tailCoreBlockFactor * (blockIdx - tilingData->formerCoreNums)) *
                           tilingData->rUbFactor;
        }
        uint32_t nowCoreRConut = (blockIdx == (usedCoreNum - 1)) ? tilingData->rUbFactor * rLoop + tilingData->tailR :
                                                                   tilingData->rUbFactor * rLoop;
        uint32_t nowCoreRConutPowOfTow = FindCofFactor(nowCoreRConut);
        uint32_t rPowOfTow = FindCofFactor(tilingData->patternR);
        this->nFactor = static_cast<float>(1) / static_cast<float>(nowCoreRConutPowOfTow);
        this->lastNFactor = static_cast<float>(1) / static_cast<float>(rPowOfTow);
        // mean / var 缩放走「真除法 + 乘回 2 的幂」，所需标量：
        //   nCount / lastNCount       —— 除数 n（本核 r 元素数 / patternR），< 2^24，fp32 精确表示
        //   nPowOfTwo / lastNPowOfTwo —— 累加时预缩放的 2 的幂，除法后乘回（精确）
        //   lastNMinusOne             —— patternR - 1（无偏方差除数）
        // 原来的倒数常数 2^k/n 与 2^m/patternR 已不再使用（其舍入是所有通道同向的常量偏置）。
        this->nCount = static_cast<float>(nowCoreRConut);
        this->nPowOfTwo = static_cast<float>(nowCoreRConutPowOfTow);
        this->lastNCount = static_cast<float>(tilingData->patternR);
        this->lastNPowOfTwo = static_cast<float>(rPowOfTow);
        this->lastNMinusOne = static_cast<float>(tilingData->patternR - 1);

        xGm.SetGlobalBuffer((__gm__ T1*)x + rBlockOffset * tilingData->patternA);
        betaGm.SetGlobalBuffer((__gm__ T2*)beta);
        gammaGm.SetGlobalBuffer((__gm__ T2*)gamma);
        useRunningMeanVar = tilingData->useRunningMeanVar > 0 ? true : false;
        if (useRunningMeanVar) {
            runningMeanGm.SetGlobalBuffer((__gm__ T2*)mean);
            runningVarGm.SetGlobalBuffer((__gm__ T2*)var);
        }
        yGm.SetGlobalBuffer((__gm__ T1*)y + rBlockOffset * tilingData->patternA);
        batchMeanGm.SetGlobalBuffer((__gm__ float*)batch_mean);
        // batch_var 即第 5 个输出 output_reserve_space_2：训练模式下保存**有偏方差** E[(x-mean)^2]
        batchVarGm.SetGlobalBuffer((__gm__ float*)batch_var);
        runningMeanOutGm.SetGlobalBuffer((__gm__ T2*)mean_out);
        runningVarOutGm.SetGlobalBuffer((__gm__ T2*)var_out);
        meanWsp.SetGlobalBuffer((__gm__ float*)workspace + blockIdx * tilingData->patternAAlign);
        varWsp.SetGlobalBuffer((__gm__ float*)workspace + (usedCoreNum + blockIdx) * tilingData->patternAAlign);
        workspaceGm.SetGlobalBuffer((__gm__ float*)workspace);
        pipe->InitBuffer(xQueue, DOUBLE_BUFFER, tilingData->rUbFactor * tilingData->aUbFactor * sizeof(T1));
        pipe->InitBuffer(yQueue, DOUBLE_BUFFER, tilingData->rUbFactor * tilingData->aUbFactor * sizeof(T1));
        pipe->InitBuffer(gammaQueue, 1, tilingData->aUbFactor * sizeof(T2));
        pipe->InitBuffer(betaQueue, 1, tilingData->aUbFactor * sizeof(T2));
        pipe->InitBuffer(batchMeanQueue, 1, tilingData->aUbFactor * sizeof(float));
        pipe->InitBuffer(batchVarQueue, 1, tilingData->aUbFactor * sizeof(float));
        if (useRunningMeanVar) {
            pipe->InitBuffer(runningMeanInQueue, 1, tilingData->aUbFactor * sizeof(T2));
            pipe->InitBuffer(runningVarInQueue, 1, tilingData->aUbFactor * sizeof(T2));
        }
        pipe->InitBuffer(runningMeanOutQueue, 1, tilingData->aUbFactor * sizeof(T2));
        pipe->InitBuffer(runningVarOutQueue, 1, tilingData->aUbFactor * sizeof(T2));
        pipe->InitBuffer(tmpTbuf1, tilingData->tBufUbFactor * tilingData->aUbFactor * sizeof(float));
        pipe->InitBuffer(tmpTbuf2, tilingData->tBufUbFactor * tilingData->aUbFactor * sizeof(float));
        pipe->InitBuffer(tmpTbuf3, tilingData->tBufUbFactor * tilingData->aUbFactor * sizeof(float));
        int64_t rUbFactorAlign = CEIL_ALIGN(tilingData->rUbFactor, FP32_BLOCK_ALIGN_SIZE);
        int64_t usedCoreNumAlign = CEIL_ALIGN(usedCoreNum, FP32_BLOCK_ALIGN_SIZE);
        pipe->InitBuffer(countTbuf1, rUbFactorAlign * sizeof(float));
        pipe->InitBuffer(countTbuf2, usedCoreNumAlign * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        LocalTensor<float> meanTensor = tmpTbuf1.Get<float>();
        LocalTensor<float> m2Tensor = tmpTbuf2.Get<float>();
        LocalTensor<float> tmpTensor = tmpTbuf3.Get<float>();
        LocalTensor<float> countTensor1 = countTbuf1.Get<float>();
        LocalTensor<float> countTensor2 = countTbuf2.Get<float>();
        int64_t aLoopOffset = 0;
        for (int64_t aUbLoopIdx = 0; aUbLoopIdx < tilingData->aUbLoop; aUbLoopIdx++) {
            currentA = tilingData->aUbFactor;
            currentAAlign = tilingData->aUbFactor;
            if (aUbLoopIdx == (tilingData->aUbLoop - 1)) {
                currentA = tilingData->aUbTail;
                currentAAlign = CEIL_ALIGN(tilingData->aUbTail, T_BLOCK_ALIGN_SIZE);
            }
            aLoopOffset = aUbLoopIdx * tilingData->aUbFactor;
            uint32_t calcLen = tilingData->rUbFactor * currentAAlign;
            currentR = tilingData->rUbFactor;
            MeanM2TensorInit(meanTensor, m2Tensor, calcLen);
            int64_t xGmOffset = 0;
            int64_t count = 0;
            for (int64_t rUbLoopIdx = 0; rUbLoopIdx < this->rLoop; rUbLoopIdx++) {
                xGmOffset = rUbLoopIdx * tilingData->rUbFactor * tilingData->patternA + aLoopOffset;
                WelfordParallelUpdate(count, meanTensor, m2Tensor, xGmOffset, calcLen);
            }
            if ((tilingData->tailR != 0) && (blockIdx == (usedCoreNum - 1))) {
                calcLen = tilingData->tailR * currentAAlign;
                currentR = tilingData->tailR;
                xGmOffset = this->rLoop * tilingData->rUbFactor * tilingData->patternA + aLoopOffset;
                WelfordParallelUpdate(count, meanTensor, m2Tensor, xGmOffset, calcLen);
            }
            CaculateCountBuf(countTensor1, countTensor2);
            LocalTensor<float> localMeanTensor = batchMeanQueue.AllocTensor<float>();
            LocalTensor<float> localVarTensor = batchVarQueue.AllocTensor<float>();
            ProcessWelfordFinalize(meanTensor, m2Tensor, countTensor1, localMeanTensor, localVarTensor, tmpTensor);
            batchMeanQueue.EnQue(localMeanTensor);
            batchVarQueue.EnQue(localVarTensor);
            localMeanTensor = batchMeanQueue.template DeQue<float>();
            localVarTensor = batchVarQueue.template DeQue<float>();
            DataCopy(meanWsp[aLoopOffset], localMeanTensor, currentAAlign);
            DataCopy(varWsp[aLoopOffset], localVarTensor, currentAAlign);
            batchMeanQueue.FreeTensor(localMeanTensor);
            batchVarQueue.FreeTensor(localVarTensor);
        }
        SyncAll();
        LocalTensor<float> allMeanTensor = tmpTbuf1.Get<float>();
        LocalTensor<float> alllVarTensor = tmpTbuf2.Get<float>();
        event_t eIdMte2ToVec = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::MTE2_V>());
        event_t eIdVecToMte2 = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::V_MTE2>());
        for (int64_t aUbLoopIdx = 0; aUbLoopIdx < tilingData->aUbLoop; aUbLoopIdx++) {
            currentA = tilingData->aUbFactor;
            currentAAlign = tilingData->aUbFactor;
            if (aUbLoopIdx == (tilingData->aUbLoop - 1)) {
                currentA = tilingData->aUbTail;
                currentAAlign = CEIL_ALIGN(tilingData->aUbTail, T_BLOCK_ALIGN_SIZE);
            }
            aLoopOffset = aUbLoopIdx * tilingData->aUbFactor;
            // 反向同步，第2次的搬入需要第1次内存free
            if (aUbLoopIdx > 0) {
                WaitFlag<HardEvent::V_MTE2>(eIdVecToMte2);
            }
            CopyInAllMeanVar(allMeanTensor, alllVarTensor, aLoopOffset);
            SetFlag<HardEvent::MTE2_V>(eIdMte2ToVec);
            WaitFlag<HardEvent::MTE2_V>(eIdMte2ToVec);
            LocalTensor<float> batchMeanTensor = batchMeanQueue.AllocTensor<float>();
            LocalTensor<float> batchVarTensor = batchVarQueue.AllocTensor<float>();
            LastFinalize(batchMeanTensor, batchVarTensor, allMeanTensor, alllVarTensor, countTensor2, tmpTensor);
            // 反向同步，最后一次循环前，都需要set
            if (aUbLoopIdx < tilingData->aUbLoop - 1) {
                SetFlag<HardEvent::V_MTE2>(eIdVecToMte2);
            }
            LocalTensor<T2> gammaTensor = gammaQueue.AllocTensor<T2>();
            LocalTensor<T2> betaTensor = betaQueue.AllocTensor<T2>();
            CopyInGammaBeta(gammaTensor, betaTensor, aLoopOffset);
            gammaQueue.EnQue(gammaTensor);
            betaQueue.EnQue(betaTensor);
            if (blockIdx == 0) {
                LocalTensor<T2> runningMeanInTensor;
                LocalTensor<T2> runningVarInTensor;
                if (useRunningMeanVar) {
                    runningMeanInTensor = runningMeanInQueue.AllocTensor<T2>();
                    runningVarInTensor = runningVarInQueue.AllocTensor<T2>();
                    CopyInRunningMeanVar(runningMeanInTensor, runningVarInTensor, aLoopOffset);
                    runningMeanInQueue.EnQue(runningMeanInTensor);
                    runningVarInQueue.EnQue(runningVarInTensor);
                    runningMeanInTensor = runningMeanInQueue.template DeQue<T2>();
                    runningVarInTensor = runningVarInQueue.template DeQue<T2>();
                }
                LocalTensor<T2> runningMeanOutTensor = runningMeanOutQueue.AllocTensor<T2>();
                LocalTensor<T2> runningVarOutTensor = runningVarOutQueue.AllocTensor<T2>();
                // alllVarTensor(tmpTbuf2) 在 LastFinalize 里被复用为「无偏方差」输出。
                CalculateRunningMeanVar(runningMeanInTensor, runningVarInTensor, runningMeanOutTensor,
                                        runningVarOutTensor, batchMeanTensor, alllVarTensor);
                if (useRunningMeanVar) {
                    runningMeanInQueue.FreeTensor(runningMeanInTensor);
                    runningVarInQueue.FreeTensor(runningVarInTensor);
                }
                runningMeanOutQueue.EnQue(runningMeanOutTensor);
                runningVarOutQueue.EnQue(runningVarOutTensor);
                runningMeanOutTensor = runningMeanOutQueue.template DeQue<T2>();
                runningVarOutTensor = runningVarOutQueue.template DeQue<T2>();
                CopyOutRunningMeanVar(runningMeanOutTensor, runningVarOutTensor, aLoopOffset);
                runningMeanOutQueue.FreeTensor(runningMeanOutTensor);
                runningVarOutQueue.FreeTensor(runningVarOutTensor);
            }
            gammaTensor = gammaQueue.DeQue<T2>();
            betaTensor = betaQueue.DeQue<T2>();
            // 需要等runningMeanVar计算完成后，才能计算成Rstd。
            // batchVarTensor 里是第 5 个输出要保存的**有偏方差**，不能被 rstd 覆盖；
            // 这里把 rstd 写到 tmpTensor(tmpTbuf3) —— LastFinalize 用完后的暂存区，不再另占 UB。
            ComputeRstdNewtonRaphsonExt2<false>(batchVarTensor, tmpTensor, static_cast<uint32_t>(currentA),
                                                tilingData->epsilon, 1.0f, VL_F32);
            int64_t yGmOffset = 0;
            currentR = tilingData->rUbFactor;
            for (int64_t rUbLoopIdx = 0; rUbLoopIdx < this->rLoop; rUbLoopIdx++) {
                yGmOffset = rUbLoopIdx * tilingData->rUbFactor * tilingData->patternA + aLoopOffset;
                NormalizeX(batchMeanTensor, tmpTensor, gammaTensor, betaTensor, yGmOffset);
            }
            if ((tilingData->tailR != 0) && (blockIdx == (usedCoreNum - 1))) {
                currentR = tilingData->tailR;
                yGmOffset = this->rLoop * tilingData->rUbFactor * tilingData->patternA + aLoopOffset;
                NormalizeX(batchMeanTensor, tmpTensor, gammaTensor, betaTensor, yGmOffset);
            }
            gammaQueue.FreeTensor(gammaTensor);
            betaQueue.FreeTensor(betaTensor);
            if (blockIdx == 0) {
                batchMeanQueue.EnQue(batchMeanTensor);
                batchVarQueue.EnQue(batchVarTensor);
                batchMeanTensor = batchMeanQueue.template DeQue<float>();
                batchVarTensor = batchVarQueue.template DeQue<float>();
                CopyOutBatchMeanVar(batchMeanTensor, batchVarTensor, aLoopOffset);
            }
            batchMeanQueue.FreeTensor(batchMeanTensor);
            batchVarQueue.FreeTensor(batchVarTensor);
        }
    }

private:
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

    __aicore__ inline void CaculateCountBuf(LocalTensor<float>& tCountTensor1, LocalTensor<float>& tCountTensor2)
    {
        __ubuf__ float* tmpCountLocal1 = (__ubuf__ float*)tCountTensor1.GetPhyAddr();
        __ubuf__ float* tmpCountLocal2 = (__ubuf__ float*)tCountTensor2.GetPhyAddr();
        float baseAddCount = static_cast<float>(this->rLoop);
        float tailAddCount = static_cast<float>(this->rLoop + 1);
        uint32_t baseNum = tilingData->rUbFactor;
        uint32_t tailNum = (blockIdx == (usedCoreNum - 1)) ? tilingData->tailR : 0;
        uint16_t baseLoopCount = CEIL_DIV(baseNum, VL_F32);
        uint16_t tailLoopCount = CEIL_DIV(tailNum, VL_F32);
        float lastCoreAddCount = static_cast<float>(tilingData->tailR +
                                                    tilingData->tailCoreBlockFactor * tilingData->rUbFactor);
        if (tilingData->tailCoreNums == 0) {
            lastCoreAddCount = static_cast<float>(tilingData->tailR +
                                                  tilingData->formerCoreBlockFactor * tilingData->rUbFactor);
        }
        float tailCoreAddCount = static_cast<float>(tilingData->tailCoreBlockFactor * tilingData->rUbFactor);
        float formerCoreAddCount = static_cast<float>(tilingData->formerCoreBlockFactor * tilingData->rUbFactor);
        uint32_t firstNum = usedCoreNum;
        uint32_t secondNum = usedCoreNum - 1;
        uint32_t thirdNum = usedCoreNum - tilingData->tailCoreNums;
        if (tilingData->tailCoreNums == 0) {
            secondNum = 0;
            thirdNum = usedCoreNum - 1;
        }
        uint16_t fisrstLoopCount = CEIL_DIV(firstNum, VL_F32);
        uint16_t secondLoopCount = CEIL_DIV(secondNum, VL_F32);
        uint16_t thirdLoopCount = CEIL_DIV(thirdNum, VL_F32);
        __VEC_SCOPE__
        {
            RegTensor<float> tmpCount;
            MaskReg pregMain = CreateMask<float, MaskPattern::ALL>();
            MaskReg pregLoop;
            uint32_t sreg1 = baseNum;
            Duplicate(tmpCount, baseAddCount, pregMain);
            for (uint16_t i = 0; i < baseLoopCount; i++) {
                pregLoop = AscendC::MicroAPI::UpdateMask<float>(sreg1);
                StoreAlign(((__ubuf__ float*)tmpCountLocal1 + i * VL_F32), tmpCount, pregLoop);
            }
            uint32_t sreg2 = tailNum;
            Duplicate(tmpCount, tailAddCount, pregMain);
            for (uint16_t i = 0; i < tailLoopCount; i++) {
                pregLoop = AscendC::MicroAPI::UpdateMask<float>(sreg2);
                StoreAlign(((__ubuf__ float*)tmpCountLocal1 + i * VL_F32), tmpCount, pregLoop);
            }
            uint32_t sreg3 = firstNum;
            Duplicate(tmpCount, lastCoreAddCount, pregMain);
            for (uint16_t i = 0; i < fisrstLoopCount; i++) {
                pregLoop = AscendC::MicroAPI::UpdateMask<float>(sreg3);
                StoreAlign(((__ubuf__ float*)tmpCountLocal2 + i * VL_F32), tmpCount, pregLoop);
            }
            uint32_t sreg4 = secondNum;
            Duplicate(tmpCount, tailCoreAddCount, pregMain);
            for (uint16_t i = 0; i < secondLoopCount; i++) {
                pregLoop = AscendC::MicroAPI::UpdateMask<float>(sreg4);
                StoreAlign(((__ubuf__ float*)tmpCountLocal2 + i * VL_F32), tmpCount, pregLoop);
            }
            uint32_t sreg5 = thirdNum;
            Duplicate(tmpCount, formerCoreAddCount, pregMain);
            for (uint16_t i = 0; i < thirdLoopCount; i++) {
                pregLoop = AscendC::MicroAPI::UpdateMask<float>(sreg5);
                StoreAlign(((__ubuf__ float*)tmpCountLocal2 + i * VL_F32), tmpCount, pregLoop);
            }
        }
    }

    __aicore__ inline void MeanM2TensorInit(LocalTensor<float>& meanTensor, LocalTensor<float>& m2Tensor, uint32_t len)
    {
        __ubuf__ float* meanTensorAddr = (__ubuf__ float*)meanTensor.GetPhyAddr();
        __ubuf__ float* m2TensorAddr = (__ubuf__ float*)m2Tensor.GetPhyAddr();
        uint16_t loopCount = CEIL_DIV(len, VL_F32);
        __VEC_SCOPE__
        {
            RegTensor<float> tmpMean;
            RegTensor<float> tmpM2;
            MaskReg mask0 = AscendC::MicroAPI::CreateMask<float, AscendC::MicroAPI::MaskPattern::ALL>();
            Duplicate(tmpMean, 0.0, mask0);
            Duplicate(tmpM2, 0.0, mask0);
            MaskReg mask1;
            uint32_t sreg0 = len;
            for (uint16_t i = 0; i < loopCount; i++) {
                mask1 = AscendC::MicroAPI::UpdateMask<float>(sreg0);
                StoreAlign(meanTensorAddr + i * VL_F32, tmpMean, mask1);
                StoreAlign(m2TensorAddr + i * VL_F32, tmpM2, mask1);
            }
        }
    }

    __aicore__ inline void CopyInX(LocalTensor<T1>& xInUb, int64_t offset)
    {
        DataCopyPadExtParams<T1> dataCopyPadExtParams;
        dataCopyPadExtParams.isPad = (currentA != currentAAlign);
        dataCopyPadExtParams.leftPadding = 0;
        // isPad配置True，rightPadding配置0，表示自动Pad到32B对齐
        dataCopyPadExtParams.rightPadding = 0;
        dataCopyPadExtParams.paddingValue = 0;
        DataCopyExtParams copyInParams;
        copyInParams.blockCount = currentR;
        copyInParams.blockLen = currentA * sizeof(T1);
        copyInParams.srcStride = (tilingData->patternA - currentA) * sizeof(T1);
        copyInParams.dstStride = 0;
        DataCopyPad(xInUb, xGm[offset], copyInParams, dataCopyPadExtParams);
    }

    __aicore__ inline void CopyInAllMeanVar(LocalTensor<float>& allMeanTensor, LocalTensor<float>& alllVarTensor,
                                            int64_t aLoopOffset)
    {
        DataCopyPadExtParams<float> meanVarPadParams;
        meanVarPadParams.isPad = false;
        meanVarPadParams.leftPadding = 0;
        meanVarPadParams.rightPadding = 0;
        meanVarPadParams.paddingValue = 0;
        DataCopyExtParams copyInMeanVarParams;
        copyInMeanVarParams.blockCount = usedCoreNum;
        copyInMeanVarParams.dstStride = 0;
        copyInMeanVarParams.blockLen = currentAAlign * sizeof(float);
        copyInMeanVarParams.srcStride = (tilingData->patternAAlign - currentAAlign) * sizeof(float);
        DataCopyPad(allMeanTensor, workspaceGm[aLoopOffset], copyInMeanVarParams, meanVarPadParams);
        DataCopyPad(alllVarTensor, workspaceGm[usedCoreNum * tilingData->patternAAlign + aLoopOffset],
                    copyInMeanVarParams, meanVarPadParams);
    }

    __aicore__ inline void WelfordParallelUpdate(int64_t& count, LocalTensor<float>& meanTensor,
                                                 LocalTensor<float>& m2Tensor, int64_t xGmOffset, uint32_t len)
    {
        // copy in x
        LocalTensor<T1> xTensor = xQueue.AllocTensor<T1>();
        CopyInX(xTensor, xGmOffset);
        xQueue.EnQue(xTensor);
        xTensor = xQueue.DeQue<T1>();
        // ---------
        count += 1;
        float scale = (float)1.0 / static_cast<float>(count);
        __ubuf__ float* meanTensorAddr = (__ubuf__ float*)meanTensor.GetPhyAddr();
        __ubuf__ float* m2TensorAddr = (__ubuf__ float*)m2Tensor.GetPhyAddr();
        __ubuf__ T1* xTensorAddr = (__ubuf__ T1*)xTensor.GetPhyAddr();
        uint16_t loopCount = CEIL_DIV(len, VL_F32);
        __VEC_SCOPE__
        {
            RegTensor<float> x1;
            RegTensor<float> tmpMean;
            RegTensor<float> tmpM2;
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
            MaskReg mask0;
            MaskReg pregAll = AscendC::MicroAPI::CreateMask<float, AscendC::MicroAPI::MaskPattern::ALL>();
            Duplicate(zero, 0.0f, pregAll);
            uint32_t sreg0 = len;
            for (uint16_t i = 0; i < loopCount; i++) {
                mask0 = AscendC::MicroAPI::UpdateMask<float>(sreg0);
                LoadTensorForDtypeT<T1>(x1, xTensorAddr, mask0, i * VL_F32);
                LoadAlign(tmpMean, meanTensorAddr + i * VL_F32);
                LoadAlign(tmpM2, m2TensorAddr + i * VL_F32);
                // delata1 = x1 - mean
                Sub(delta1, x1, tmpMean, mask0);
                // mean 为 ±inf 且 x1 + mean 不是 NaN 时，delta1 置 0，使 mean 保持 ±inf 不变；
                // 异号 inf（x1 = ∓inf、mean = ±inf）时 x1 + mean 未定义（NaN），必须保留 delta1 = ±inf，
                // 让 mean = mean + delta1 * scale 得到 NaN，与 golden 一致；
                // 同号 inf 与有限值 x1 的 x1 + mean = ±inf，仍走清零分支（有限值 - (±inf) = ∓inf 会污染均值）
                AscendC::Reg::Abs(absMean, tmpMean, mask0);
                Compares(infMeanMask, absMean, POS_INF, mask0);
                AscendC::Reg::Add(sumX1Mean, x1, tmpMean, mask0);
                AscendC::Reg::Compare<float, AscendC::CMPMODE::NE>(nanSumMask, sumX1Mean, sumX1Mean, mask0);
                AscendC::Reg::Not(notNanSumMask, nanSumMask, mask0);
                AscendC::Reg::And(zeroDeltaMask, infMeanMask, notNanSumMask, mask0);
                Select(delta1, zero, delta1, zeroDeltaMask);
                // delta2 = delta1 * scale
                Muls(delta2, delta1, scale, mask0);
                // mean = mean + delta2
                Add(tmpMean, tmpMean, delta2, mask0);
                StoreAlign(meanTensorAddr + i * VL_F32, tmpMean, mask0);
                // delta3 = x1 - mean
                Sub(delta3, x1, tmpMean, mask0);
                // delta4 = delta1 * delta3
                Mul(delat4, delta1, delta3, mask0);
                // M2 = M2 + delta4
                Add(tmpM2, tmpM2, delat4, mask0);
                StoreAlign(m2TensorAddr + i * VL_F32, tmpM2, mask0);
            }
        }
        xQueue.FreeTensor(xTensor);
    }

    __aicore__ inline void ProcessWelfordFinalize(LocalTensor<float>& meanTensor, LocalTensor<float>& m2Tensor,
                                                  LocalTensor<float>& countTensor, LocalTensor<float>& finalMeanTensor,
                                                  LocalTensor<float>& finalVarTensor, LocalTensor<float>& tmpTensor)

    {
        __ubuf__ float* tmpMeanLocal = (__ubuf__ float*)meanTensor.GetPhyAddr();
        __ubuf__ float* tmpVarLocal = (__ubuf__ float*)m2Tensor.GetPhyAddr();
        __ubuf__ float* tmpCountLocal = (__ubuf__ float*)countTensor.GetPhyAddr();
        __ubuf__ float* batchMeanInUbAddr = (__ubuf__ float*)finalMeanTensor.GetPhyAddr();
        __ubuf__ float* batchVarInUbAddr = (__ubuf__ float*)finalVarTensor.GetPhyAddr();
        __ubuf__ float* tmpUbAddr = (__ubuf__ float*)tmpTensor.GetPhyAddr();
        WelfordFinalizeMeanVF(tmpMeanLocal, tmpVarLocal, tmpCountLocal, batchMeanInUbAddr, batchVarInUbAddr, tmpUbAddr);
        WelfordFinalizeVarVF(tmpMeanLocal, tmpVarLocal, tmpCountLocal, batchMeanInUbAddr, batchVarInUbAddr, tmpUbAddr);
    }

    __aicore__ inline void WelfordFinalizeMeanVF(__ubuf__ float* tmpMeanLocal, __ubuf__ float* tmpVarLocal,
                                                 __ubuf__ float* tmpCountLocal, __ubuf__ float* batchMeanInUbAddr,
                                                 __ubuf__ float* batchVarInUbAddr, __ubuf__ float* binaryAddTmpAddr)
    {
        uint16_t rLoopCount = tilingData->rUbFactor;
        uint16_t aLoopCount = CEIL_DIV(currentA, VL_F32);
        uint32_t rLoopStride = currentAAlign;

        uint16_t remainderLoopCount = (tilingData->rUbFactor - tilingData->binaryAddQuotient) / SCALE_COEF_FOUR;
        uint16_t quotientLoopCount = (tilingData->binaryAddQuotient / SCALE_COEF_FOUR) - remainderLoopCount;
        uint32_t baseLineOffset = SCALE_COEF_FOUR * rLoopStride;
        uint32_t remainderOffset = tilingData->binaryAddQuotient * rLoopStride;
        uint32_t remainderCountOffset = tilingData->binaryAddQuotient;

        uint16_t binaryAddKLoop = tilingData->binaryAddK;
        uint16_t binaryAddInnerLoop = tilingData->binaryAddQuotient / SCALE_COEF_FOUR;
        uint16_t binaryAddLastLoop = tilingData->binaryAddLast;

        float numScale = this->nFactor;
        // mean 缩放改「真除法」：累加时每行已乘 numScale = 1/2^k（2 的幂，精确），故此时 x1 = Σ(count_i·mean_i)/2^k。
        // 先真除 n（= nowCoreRConut，< 2^24 可精确表示）再乘回 2^k（精确），整体等价于一次正确舍入的除法。
        // 原写法 ×(2^k/n) 的倒数常数舍入是所有通道同向的常量偏置（实测 δmean 均值 +6.63e-8），故不再使用。
        // 复用方差路径已在 Init 里建好的 nCount / nPowOfTwo 标量，不新增 buffer。
        float meanDivN = this->nCount;
        float meanPowOfTwo = this->nPowOfTwo;

        uint32_t twoRLoopSize = ROW_TWO_OFFSET * rLoopStride;
        uint32_t threeRLoopSize = ROW_THREE_OFFSET * rLoopStride;
        __VEC_SCOPE__
        {
            RegTensor<float> x1;
            RegTensor<float> x2;
            RegTensor<float> x3;
            RegTensor<float> x4;

            RegTensor<float> nextRow;
            RegTensor<float> rem;
            RegTensor<float> remNextRow;

            RegTensor<float> rowCount;
            RegTensor<float> nextRowCount;
            RegTensor<float> remCount;
            RegTensor<float> nextRemCount;
            RegTensor<float> nReg; // 除数 n（本核参与累加的 r 元素数）

            MaskReg pregLoop;
            MaskReg pregAll = AscendC::MicroAPI::CreateMask<float, AscendC::MicroAPI::MaskPattern::ALL>();
            Duplicate(nReg, meanDivN, pregAll);
            uint32_t sreg0 = currentA;
            for (uint16_t aIndex = 0; aIndex < aLoopCount; aIndex++) {
                uint32_t aLoopOffset = aIndex * VL_F32;
                pregLoop = AscendC::MicroAPI::UpdateMask<float>(sreg0);
                // 尾块部分四行，和前面四行相加，最终是一行
                for (uint16_t i = 0; i < remainderLoopCount; i++) {
                    uint32_t quotOffset = i * baseLineOffset + aLoopOffset;
                    uint32_t remOffset = i * baseLineOffset + remainderOffset + aLoopOffset;
                    uint32_t quotCountOffset = i * SCALE_COEF_FOUR;
                    uint32_t remCountOffset = i * SCALE_COEF_FOUR + remainderCountOffset;
                    // 前两行
                    TwoRowAddForMeanWithTail(x1, tmpMeanLocal, tmpCountLocal, pregLoop, quotOffset, remOffset,
                                             quotOffset + rLoopStride, remOffset + rLoopStride, quotCountOffset,
                                             remCountOffset, quotCountOffset + 1, remCountOffset + 1, rem, nextRow,
                                             remNextRow, rowCount, nextRowCount, remCount, nextRemCount, numScale);
                    // 后两行
                    TwoRowAddForMeanWithTail(x2, tmpMeanLocal, tmpCountLocal, pregLoop, quotOffset + twoRLoopSize,
                                             remOffset + twoRLoopSize, quotOffset + threeRLoopSize,
                                             remOffset + threeRLoopSize, quotCountOffset + ROW_TWO_OFFSET,
                                             remCountOffset + ROW_TWO_OFFSET, quotCountOffset + ROW_THREE_OFFSET,
                                             remCountOffset + ROW_THREE_OFFSET, rem, nextRow, remNextRow, rowCount,
                                             nextRowCount, remCount, nextRemCount, numScale);
                    Add(x1, x1, x2, pregLoop);
                    StoreAlign(((__ubuf__ float*)binaryAddTmpAddr + i * rLoopStride + aLoopOffset), x1, pregLoop);
                }
                // 剩余的前半部分，一次for循环，处理4行，4行加成1行
                for (uint16_t i = 0; i < quotientLoopCount; i++) {
                    uint32_t baseOffset = (remainderLoopCount + i) * baseLineOffset + aLoopOffset;
                    uint32_t baseCountOffset = (remainderLoopCount + i) * SCALE_COEF_FOUR;
                    TwoRowAddForMean(x1, tmpMeanLocal, tmpCountLocal, pregLoop, baseOffset, baseOffset + rLoopStride,
                                     baseCountOffset, baseCountOffset + 1, rem, rowCount, nextRowCount, numScale);
                    TwoRowAddForMean(x2, tmpMeanLocal, tmpCountLocal, pregLoop, baseOffset + twoRLoopSize,
                                     baseOffset + threeRLoopSize, baseCountOffset + ROW_TWO_OFFSET,
                                     baseCountOffset + ROW_THREE_OFFSET, rem, rowCount, nextRowCount, numScale);
                    Add(x1, x1, x2, pregLoop);
                    StoreAlign(
                        ((__ubuf__ float*)binaryAddTmpAddr + (remainderLoopCount + i) * rLoopStride + aLoopOffset), x1,
                        pregLoop);
                }
                LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
                // 最后对2的幂次行 二分累加
                BinaryAddVF(binaryAddTmpAddr, rLoopStride, binaryAddKLoop, binaryAddInnerLoop, binaryAddLastLoop,
                            pregLoop, aLoopOffset, x1, x2, x3, x4);
                LoadAlign(x1, ((__ubuf__ float*)binaryAddTmpAddr + aLoopOffset));
                // 此时 x1 = Σ(count_i·mean_i)/2^k。真除法 ÷n 后乘回 2^k（精确）= 一次正确舍入的均值。
                Div(x1, x1, nReg, pregLoop);
                Muls(x1, x1, meanPowOfTwo, pregLoop);
                StoreAlign(((__ubuf__ float*)batchMeanInUbAddr + aLoopOffset), x1, pregLoop);
            }
        }
    }

    __aicore__ inline void WelfordFinalizeVarVF(__ubuf__ float* tmpMeanLocal, __ubuf__ float* tmpVarLocal,
                                                __ubuf__ float* tmpCountLocal, __ubuf__ float* batchMeanInUbAddr,
                                                __ubuf__ float* batchVarInUbAddr, __ubuf__ float* binaryAddTmpAddr)
    {
        uint16_t rLoopCount = tilingData->rUbFactor;
        uint16_t aLoopCount = CEIL_DIV(currentA, VL_F32);
        uint32_t rLoopStride = currentAAlign;

        uint16_t remainderLoopCount = (tilingData->rUbFactor - tilingData->binaryAddQuotient) / SCALE_COEF_FOUR;
        uint16_t quotientLoopCount = (tilingData->binaryAddQuotient / SCALE_COEF_FOUR) - remainderLoopCount;
        uint32_t baseLineOffset = SCALE_COEF_FOUR * rLoopStride;
        uint32_t remainderOffset = tilingData->binaryAddQuotient * rLoopStride;
        uint32_t remainderCountOffset = tilingData->binaryAddQuotient;

        uint16_t binaryAddKLoop = tilingData->binaryAddK;
        uint16_t binaryAddInnerLoop = tilingData->binaryAddQuotient / SCALE_COEF_FOUR;
        uint16_t binaryAddLastLoop = tilingData->binaryAddLast;

        float numScale = this->nFactor;
        // 方差：先用真除法除以 n，再乘回 2 的幂 nPowOfTwo（精确），
        // 等价于原来的 ×(2^k/n) 但消除了倒数常数的系统性舍入偏差。
        float varTrueDivN = this->nCount;
        float varPowOfTwo = this->nPowOfTwo;

        uint32_t twoRLoopSize = ROW_TWO_OFFSET * rLoopStride;
        uint32_t threeRLoopSize = ROW_THREE_OFFSET * rLoopStride;
        __VEC_SCOPE__
        {
            RegTensor<float> saveMean;

            RegTensor<float> x1;
            RegTensor<float> x2;
            RegTensor<float> x3;
            RegTensor<float> x4;

            RegTensor<float> nextRow;
            RegTensor<float> rem;
            RegTensor<float> remNextRow;

            RegTensor<float> rowCount;
            RegTensor<float> nextRowCount;
            RegTensor<float> remCount;
            RegTensor<float> nextRemCount;

            RegTensor<float> rowM2;
            RegTensor<float> nextRowM2;
            RegTensor<float> remM2;
            RegTensor<float> nextRemM2;
            RegTensor<float> nReg;

            MaskReg pregLoop;
            MaskReg pregAll = AscendC::MicroAPI::CreateMask<float, AscendC::MicroAPI::MaskPattern::ALL>();
            Duplicate(nReg, varTrueDivN, pregAll);
            uint32_t sreg0 = currentA;
            for (uint16_t aIndex = 0; aIndex < aLoopCount; aIndex++) {
                uint32_t aLoopOffset = aIndex * VL_F32;
                pregLoop = AscendC::MicroAPI::UpdateMask<float>(sreg0);
                LoadAlign(saveMean, ((__ubuf__ float*)batchMeanInUbAddr + aLoopOffset));
                for (uint16_t i = 0; i < remainderLoopCount; i++) {
                    uint32_t quotOffset = i * baseLineOffset + aLoopOffset;
                    uint32_t remOffset = i * baseLineOffset + remainderOffset + aLoopOffset;
                    uint32_t quotCountOffset = i * SCALE_COEF_FOUR;
                    uint32_t remCountOffset = i * SCALE_COEF_FOUR + remainderCountOffset;
                    TwoRowAddForVarWithTail(x1, tmpMeanLocal, tmpVarLocal, tmpCountLocal, pregLoop, quotOffset,
                                            remOffset, quotOffset + rLoopStride, remOffset + rLoopStride,
                                            quotCountOffset, remCountOffset, quotCountOffset + 1, remCountOffset + 1,
                                            saveMean, rem, nextRow, remNextRow, rowCount, nextRowCount, remCount,
                                            nextRemCount, rowM2, nextRowM2, remM2, nextRemM2, numScale);
                    TwoRowAddForVarWithTail(x2, tmpMeanLocal, tmpVarLocal, tmpCountLocal, pregLoop,
                                            quotOffset + twoRLoopSize, remOffset + twoRLoopSize,
                                            quotOffset + threeRLoopSize, remOffset + threeRLoopSize,
                                            quotCountOffset + ROW_TWO_OFFSET, remCountOffset + ROW_TWO_OFFSET,
                                            quotCountOffset + ROW_THREE_OFFSET, remCountOffset + ROW_THREE_OFFSET,
                                            saveMean, rem, nextRow, remNextRow, rowCount, nextRowCount, remCount,
                                            nextRemCount, rowM2, nextRowM2, remM2, nextRemM2, numScale);
                    Add(x1, x1, x2, pregLoop);
                    StoreAlign(((__ubuf__ float*)binaryAddTmpAddr + i * rLoopStride + aLoopOffset), x1, pregLoop);
                }
                // 剩余的前半部分，一次for循环，处理4行
                for (uint16_t i = 0; i < quotientLoopCount; i++) {
                    uint32_t baseOffset = (remainderLoopCount + i) * baseLineOffset + aLoopOffset;
                    uint32_t baseCountOffset = (remainderLoopCount + i) * SCALE_COEF_FOUR;
                    TwoRowAddForVar(x1, tmpMeanLocal, tmpVarLocal, tmpCountLocal, pregLoop, baseOffset,
                                    baseOffset + rLoopStride, baseCountOffset, baseCountOffset + 1, saveMean, rem,
                                    rowCount, nextRowCount, rowM2, remM2, numScale);
                    TwoRowAddForVar(x2, tmpMeanLocal, tmpVarLocal, tmpCountLocal, pregLoop, baseOffset + twoRLoopSize,
                                    baseOffset + threeRLoopSize, baseCountOffset + ROW_TWO_OFFSET,
                                    baseCountOffset + ROW_THREE_OFFSET, saveMean, rem, rowCount, nextRowCount, rowM2,
                                    remM2, numScale);
                    Add(x1, x1, x2, pregLoop);
                    StoreAlign(
                        ((__ubuf__ float*)binaryAddTmpAddr + (remainderLoopCount + i) * rLoopStride + aLoopOffset), x1,
                        pregLoop);
                }
                LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
                BinaryAddVF(binaryAddTmpAddr, rLoopStride, binaryAddKLoop, binaryAddInnerLoop, binaryAddLastLoop,
                            pregLoop, aLoopOffset, x1, x2, x3, x4);
                LoadAlign(x1, ((__ubuf__ float*)binaryAddTmpAddr + aLoopOffset));
                // 此时 x1 = M2_sum/2^k（累加时逐项乘 numScale=1/2^k，精确）。
                // 真除法 ÷n 后再乘回 2^k（精确），得到 M2_sum/n；不再乘倒数常数 (2^k/n)。
                Div(x1, x1, nReg, pregLoop);
                Muls(x1, x1, varPowOfTwo, pregLoop);
                StoreAlign(((__ubuf__ float*)batchVarInUbAddr + aLoopOffset), x1, pregLoop);
            }
        }
    }

    __aicore__ inline void BinaryAddVF(__ubuf__ float* binaryAddTmpAddr, uint32_t rLoopStride, uint16_t binaryAddKLoop,
                                       uint16_t binaryAddInnerLoop, uint16_t binaryAddLastLoop, MaskReg& pregLoop,
                                       uint32_t offset, RegTensor<float>& x1, RegTensor<float>& x2,
                                       RegTensor<float>& x3, RegTensor<float>& x4)
    {
        uint16_t curBinaryAddInnerLoop = binaryAddInnerLoop;
        for (uint16_t i = 0; i < binaryAddKLoop; i++) {
            curBinaryAddInnerLoop = curBinaryAddInnerLoop / ROW_FOUR_OFFSET;
            for (uint16_t j = 0; j < curBinaryAddInnerLoop; j++) {
                LoadAlign(x1, ((__ubuf__ float*)binaryAddTmpAddr + (j * ROW_FOUR_OFFSET) * rLoopStride + offset));
                LoadAlign(x2, ((__ubuf__ float*)binaryAddTmpAddr + (j * ROW_FOUR_OFFSET + 1) * rLoopStride + offset));
                Add(x1, x1, x2, pregLoop);
                LoadAlign(x3, ((__ubuf__ float*)binaryAddTmpAddr +
                               (j * ROW_FOUR_OFFSET + ROW_TWO_OFFSET) * rLoopStride + offset));
                LoadAlign(x4, ((__ubuf__ float*)binaryAddTmpAddr +
                               (j * ROW_FOUR_OFFSET + ROW_THREE_OFFSET) * rLoopStride + offset));
                Add(x3, x3, x4, pregLoop);
                Add(x1, x1, x3, pregLoop);
                StoreAlign(((__ubuf__ float*)binaryAddTmpAddr + j * rLoopStride + offset), x1, pregLoop);
            }
            LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
        }
        for (uint16_t i = 0; i < binaryAddLastLoop; i++) {
            LoadAlign(x1, ((__ubuf__ float*)binaryAddTmpAddr + offset));
            LoadAlign(x2, ((__ubuf__ float*)binaryAddTmpAddr + rLoopStride + offset));
            Add(x1, x1, x2, pregLoop);
            StoreAlign(((__ubuf__ float*)binaryAddTmpAddr + offset), x1, pregLoop);
            LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
        }
    }

    // 与 BinaryAddVF 结构逐位等价的末级归约，只把每一次加法换成补偿加法（Knuth TwoSum），
    // 末级再把低位残差 low 一次性加回、只额外舍入这一次。结构（4 路折半 + 串行尾加）、
    // 读写位置与 BinaryAddVF 完全一致，故调用点的 resMean/resVar 读取口径不变。
    // low 由调用点在 __VEC_SCOPE__ 里声明并传入（与 x1~x4 同一做法），函数内部不额外申请寄存器。
    __aicore__ inline void CompensatedBinaryAddVF(__ubuf__ float* binaryAddTmpAddr, uint32_t rLoopStride,
                                                  uint16_t binaryAddKLoop, uint16_t binaryAddInnerLoop,
                                                  uint16_t binaryAddLastLoop, MaskReg& pregLoop, uint32_t offset,
                                                  RegTensor<float>& x1, RegTensor<float>& x2, RegTensor<float>& x3,
                                                  RegTensor<float>& x4, RegTensor<float>& low)
    {
        uint16_t curBinaryAddInnerLoop = binaryAddInnerLoop;
        // 有效 lane 之外是脏值，先清零（x + 0 = x，残差也是 0，不引入偏差）。
        Duplicate(low, 0.0f, pregLoop);
        for (uint16_t i = 0; i < binaryAddKLoop; i++) {
            curBinaryAddInnerLoop = curBinaryAddInnerLoop / ROW_FOUR_OFFSET;
            for (uint16_t j = 0; j < curBinaryAddInnerLoop; j++) {
                LoadAlign(x1, ((__ubuf__ float*)binaryAddTmpAddr + (j * ROW_FOUR_OFFSET) * rLoopStride + offset));
                LoadAlign(x2, ((__ubuf__ float*)binaryAddTmpAddr + (j * ROW_FOUR_OFFSET + 1) * rLoopStride + offset));
                BatchSplitAccumulateWithError(x1, low, x2, pregLoop);
                LoadAlign(x3, ((__ubuf__ float*)binaryAddTmpAddr +
                               (j * ROW_FOUR_OFFSET + ROW_TWO_OFFSET) * rLoopStride + offset));
                LoadAlign(x4, ((__ubuf__ float*)binaryAddTmpAddr +
                               (j * ROW_FOUR_OFFSET + ROW_THREE_OFFSET) * rLoopStride + offset));
                BatchSplitAccumulateWithError(x3, low, x4, pregLoop);
                // 与 BinaryAddVF 一致：x1 = (row0+row1) + (row2+row3)，两级加法的残差都进 low。
                BatchSplitAccumulateWithError(x1, low, x3, pregLoop);
                StoreAlign(((__ubuf__ float*)binaryAddTmpAddr + j * rLoopStride + offset), x1, pregLoop);
            }
            LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
        }
        for (uint16_t i = 0; i < binaryAddLastLoop; i++) {
            LoadAlign(x1, ((__ubuf__ float*)binaryAddTmpAddr + offset));
            LoadAlign(x2, ((__ubuf__ float*)binaryAddTmpAddr + rLoopStride + offset));
            BatchSplitAccumulateWithError(x1, low, x2, pregLoop);
            StoreAlign(((__ubuf__ float*)binaryAddTmpAddr + offset), x1, pregLoop);
            LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
        }
        // 折半结束后最终结果落在 row 0（不论上面的循环是否执行过），显式读回再补上残差，
        // 保证与 BinaryAddVF「结果在 row 0」的语义一致。
        LoadAlign(x1, ((__ubuf__ float*)binaryAddTmpAddr + offset));
        BatchSplitClampResidual(low, pregLoop);
        Add(x1, x1, low, pregLoop);
        StoreAlign(((__ubuf__ float*)binaryAddTmpAddr + offset), x1, pregLoop);
    }

    __aicore__ inline void TwoRowAddForMeanWithTail(RegTensor<float>& dst, __ubuf__ float* input,
                                                    __ubuf__ float* tCount, MaskReg& preg, uint32_t offset1,
                                                    uint32_t offset2, uint32_t offset3, uint32_t offset4,
                                                    uint32_t offset5, uint32_t offset6, uint32_t offset7,
                                                    uint32_t offset8, RegTensor<float>& rem, RegTensor<float>& nextRow,
                                                    RegTensor<float>& remNextRow, RegTensor<float>& dstCount,
                                                    RegTensor<float>& remCount, RegTensor<float>& nextRowCount,
                                                    RegTensor<float>& remNextRowCount, float n)
    {
        LoadAlign(dst, ((__ubuf__ float*)(input) + (offset1)));
        LoadAlign(rem, ((__ubuf__ float*)(input) + (offset2)));
        LoadAlign<float, LoadDist::DIST_BRC_B32>(dstCount, ((__ubuf__ float*)(tCount) + (offset5)));
        LoadAlign<float, LoadDist::DIST_BRC_B32>(remCount, ((__ubuf__ float*)(tCount) + (offset6)));
        Mul(dst, dst, dstCount, preg);
        Mul(rem, rem, remCount, preg);
        Muls(dst, dst, n, preg);
        Muls(rem, rem, n, preg);
        Add(dst, dst, rem, preg);
        LoadAlign(nextRow, ((__ubuf__ float*)(input) + (offset3)));
        LoadAlign(remNextRow, ((__ubuf__ float*)(input) + (offset4)));
        LoadAlign<float, LoadDist::DIST_BRC_B32>(nextRowCount, ((__ubuf__ float*)(tCount) + (offset7)));
        LoadAlign<float, LoadDist::DIST_BRC_B32>(remNextRowCount, ((__ubuf__ float*)(tCount) + (offset8)));
        Mul(nextRow, nextRow, nextRowCount, preg);
        Mul(remNextRow, remNextRow, remNextRowCount, preg);
        Muls(nextRow, nextRow, n, preg);
        Muls(remNextRow, remNextRow, n, preg);
        Add(nextRow, nextRow, remNextRow, preg);
        Add(dst, dst, nextRow, preg);
    }

    __aicore__ inline void TwoRowAddForMean(RegTensor<float>& dst, __ubuf__ float* input, __ubuf__ float* tCount,
                                            MaskReg& preg, uint32_t offset1, uint32_t offset2, uint32_t offset5,
                                            uint32_t offset6, RegTensor<float>& rem, RegTensor<float>& dstCount,
                                            RegTensor<float>& remCount, float n)
    {
        LoadAlign(dst, ((__ubuf__ float*)(input) + (offset1)));
        LoadAlign(rem, ((__ubuf__ float*)(input) + (offset2)));
        LoadAlign<float, LoadDist::DIST_BRC_B32>(dstCount, ((__ubuf__ float*)(tCount) + (offset5)));
        LoadAlign<float, LoadDist::DIST_BRC_B32>(remCount, ((__ubuf__ float*)(tCount) + (offset6)));
        Mul(dst, dst, dstCount, preg);
        Mul(rem, rem, remCount, preg);
        Muls(dst, dst, n, preg);
        Muls(rem, rem, n, preg);
        Add(dst, dst, rem, preg);
    }

    __aicore__ inline void TwoRowAddForVarWithTail(
        RegTensor<float>& dst, __ubuf__ float* tmpMean, __ubuf__ float* tmpM2, __ubuf__ float* tCount, MaskReg& preg,
        uint32_t offset1, uint32_t offset2, uint32_t offset3, uint32_t offset4, uint32_t offset5, uint32_t offset6,
        uint32_t offset7, uint32_t offset8, RegTensor<float>& mean, RegTensor<float>& rem, RegTensor<float>& nextRow,
        RegTensor<float>& remNextRow, RegTensor<float>& dstCount, RegTensor<float>& remCount,
        RegTensor<float>& nextRowCount, RegTensor<float>& remNextRowCount, RegTensor<float>& dstM2,
        RegTensor<float>& remM2, RegTensor<float>& nextRowM2, RegTensor<float>& remNextRowM2, float n)
    {
        LoadAlign(dst, ((__ubuf__ float*)(tmpMean) + (offset1)));
        LoadAlign(rem, ((__ubuf__ float*)(tmpMean) + (offset2)));
        LoadAlign<float, LoadDist::DIST_BRC_B32>(dstCount, ((__ubuf__ float*)(tCount) + (offset5)));
        LoadAlign<float, LoadDist::DIST_BRC_B32>(remCount, ((__ubuf__ float*)(tCount) + (offset6)));
        Sub(dst, dst, mean, preg);
        Mul(dst, dst, dst, preg);
        Sub(rem, rem, mean, preg);
        Mul(rem, rem, rem, preg);
        Mul(dst, dst, dstCount, preg);
        Mul(rem, rem, remCount, preg);
        LoadAlign(dstM2, ((__ubuf__ float*)(tmpM2) + (offset1)));
        LoadAlign(remM2, ((__ubuf__ float*)(tmpM2) + (offset2)));
        Add(dst, dstM2, dst, preg);
        Muls(dst, dst, n, preg);
        Add(rem, remM2, rem, preg);
        Muls(rem, rem, n, preg);
        Add(dst, dst, rem, preg);

        LoadAlign(nextRow, ((__ubuf__ float*)(tmpMean) + (offset3)));
        LoadAlign(remNextRow, ((__ubuf__ float*)(tmpMean) + (offset4)));
        LoadAlign<float, LoadDist::DIST_BRC_B32>(nextRowCount, ((__ubuf__ float*)(tCount) + (offset7)));
        LoadAlign<float, LoadDist::DIST_BRC_B32>(remNextRowCount, ((__ubuf__ float*)(tCount) + (offset8)));
        Sub(nextRow, nextRow, mean, preg);
        Mul(nextRow, nextRow, nextRow, preg);
        Sub(remNextRow, remNextRow, mean, preg);
        Mul(remNextRow, remNextRow, remNextRow, preg);
        Mul(nextRow, nextRow, nextRowCount, preg);
        Mul(remNextRow, remNextRow, remNextRowCount, preg);
        LoadAlign(nextRowM2, ((__ubuf__ float*)(tmpM2) + (offset3)));
        LoadAlign(remNextRowM2, ((__ubuf__ float*)(tmpM2) + (offset4)));
        Add(nextRow, nextRowM2, nextRow, preg);
        Muls(nextRow, nextRow, n, preg);
        Add(remNextRow, remNextRowM2, remNextRow, preg);
        Muls(remNextRow, remNextRow, n, preg);
        Add(nextRow, nextRow, remNextRow, preg);

        Add(dst, dst, nextRow, preg);
    }

    __aicore__ inline void TwoRowAddForVar(RegTensor<float>& dst, __ubuf__ float* tmpMean, __ubuf__ float* tmpM2,
                                           __ubuf__ float* tCount, MaskReg& preg, uint32_t offset1, uint32_t offset2,
                                           uint32_t offset5, uint32_t offset6, RegTensor<float>& mean,
                                           RegTensor<float>& rem, RegTensor<float>& dstCount,
                                           RegTensor<float>& remCount, RegTensor<float>& dstM2, RegTensor<float>& remM2,
                                           float n)
    {
        LoadAlign(dst, ((__ubuf__ float*)(tmpMean) + (offset1)));
        LoadAlign(rem, ((__ubuf__ float*)(tmpMean) + (offset2)));
        LoadAlign<float, LoadDist::DIST_BRC_B32>(dstCount, ((__ubuf__ float*)(tCount) + (offset5)));
        LoadAlign<float, LoadDist::DIST_BRC_B32>(remCount, ((__ubuf__ float*)(tCount) + (offset6)));
        Sub(dst, dst, mean, preg);
        Mul(dst, dst, dst, preg);
        Sub(rem, rem, mean, preg);
        Mul(rem, rem, rem, preg);
        Mul(dst, dst, dstCount, preg);
        Mul(rem, rem, remCount, preg);
        LoadAlign(dstM2, ((__ubuf__ float*)(tmpM2) + (offset1)));
        LoadAlign(remM2, ((__ubuf__ float*)(tmpM2) + (offset2)));
        Add(dst, dstM2, dst, preg);
        Muls(dst, dst, n, preg);
        Add(rem, remM2, rem, preg);
        Muls(rem, rem, n, preg);
        Add(dst, dst, rem, preg);
    }

    __aicore__ inline void LastFinalize(LocalTensor<float>& batchMeanTensor, LocalTensor<float>& batchVarTensor,
                                        LocalTensor<float>& meanTensor, LocalTensor<float>& varTensor,
                                        LocalTensor<float>& countTensor, LocalTensor<float>& tmpTensor)
    {
        __ubuf__ float* tmpMeanLocal = (__ubuf__ float*)meanTensor.GetPhyAddr();
        __ubuf__ float* tmpVarLocal = (__ubuf__ float*)varTensor.GetPhyAddr();
        __ubuf__ float* tmpCountLocal = (__ubuf__ float*)countTensor.GetPhyAddr();
        __ubuf__ float* batchMeanTensorAddr = (__ubuf__ float*)batchMeanTensor.GetPhyAddr();
        __ubuf__ float* batchVarTensorAddr = (__ubuf__ float*)batchVarTensor.GetPhyAddr();
        __ubuf__ float* tmpUbAddr = (__ubuf__ float*)tmpTensor.GetPhyAddr();
        uint16_t aLoopCount = CEIL_DIV(currentA, VL_F32);
        uint32_t rLoopStride = currentAAlign;
        uint16_t remainderLoopCount = (usedCoreNum - tilingData->lastBinaryAddQuotient);
        uint16_t quotientLoopCount = tilingData->lastBinaryAddQuotient - remainderLoopCount;
        uint32_t baseLineOffset = rLoopStride;
        uint32_t remainderOffset = tilingData->lastBinaryAddQuotient * rLoopStride;
        uint32_t remainderCountOffset = tilingData->lastBinaryAddQuotient;
        uint16_t binaryAddKLoop = tilingData->lastBinaryAddK;
        uint16_t binaryAddInnerLoop = tilingData->lastBinaryAddQuotient;
        uint16_t binaryAddLastLoop = tilingData->lastBinaryAddLast;
        float numScale = this->lastNFactor;
        // 最后一级缩放统一走「真除法 + 乘回 2 的幂」，mean 与 var 共用同一组标量：
        //   累加时每行已乘 numScale = 1/2^m（2 的幂，精确），故 BinaryAddVF 的结果是 Σ/2^m。
        //   ÷patternR（mean 与有偏方差共用）、÷(patternR-1)（无偏方差）后再乘回 2^m（精确），
        //   等价于一次正确舍入的除法；不再乘倒数常数 2^m/patternR 与 Bessel 常数 n/(n-1)。
        float divN = this->lastNCount;
        float divNMinusOne = this->lastNMinusOne;
        float powOfTwo = this->lastNPowOfTwo;
        __VEC_SCOPE__
        {
            RegTensor<float> quot;
            RegTensor<float> rem;
            RegTensor<float> quotCount;
            RegTensor<float> remCount;
            RegTensor<float> oriQuotMean;
            RegTensor<float> oriRemMean;
            RegTensor<float> resMean;
            RegTensor<float> resVar;
            RegTensor<float> resVarUnbiased;
            RegTensor<float> nReg;
            RegTensor<float> nMinusOneReg;
            RegTensor<float> addLow; // 末级补偿归约的低位残差（每通道独立，逐次归约内复用）

            MaskReg pregLoop;
            MaskReg pregAll = AscendC::MicroAPI::CreateMask<float, AscendC::MicroAPI::MaskPattern::ALL>();
            Duplicate(nReg, divN, pregAll);
            Duplicate(nMinusOneReg, divNMinusOne, pregAll);
            uint32_t sreg0 = currentA;
            for (uint16_t aIndex = 0; aIndex < aLoopCount; aIndex++) {
                uint32_t aLoopOffset = aIndex * VL_F32;
                pregLoop = AscendC::MicroAPI::UpdateMask<float>(sreg0);
                // 尾块部分按行加至前面
                for (uint16_t i = 0; i < remainderLoopCount; i++) {
                    uint32_t quotOffset = i * baseLineOffset + aLoopOffset;
                    uint32_t remOffset = i * baseLineOffset + remainderOffset + aLoopOffset;
                    uint32_t quotCountOffset = i;
                    uint32_t remCountOffset = i + remainderCountOffset;
                    LoadAlign(quot, ((__ubuf__ float*)(tmpMeanLocal) + (quotOffset)));
                    LoadAlign(rem, ((__ubuf__ float*)(tmpMeanLocal) + (remOffset)));
                    LoadAlign<float, LoadDist::DIST_BRC_B32>(quotCount,
                                                             ((__ubuf__ float*)(tmpCountLocal) + quotCountOffset));
                    LoadAlign<float, LoadDist::DIST_BRC_B32>(remCount,
                                                             ((__ubuf__ float*)(tmpCountLocal) + remCountOffset));
                    Mul(quot, quot, quotCount, pregLoop);
                    Mul(rem, rem, remCount, pregLoop);
                    Muls(quot, quot, numScale, pregLoop);
                    Muls(rem, rem, numScale, pregLoop);
                    Add(quot, quot, rem, pregLoop);
                    StoreAlign(((__ubuf__ float*)tmpUbAddr + i * rLoopStride + aLoopOffset), quot, pregLoop);
                }
                // 整块部分除已经叠加了尾块的，需要乘count和scale
                for (uint16_t i = 0; i < quotientLoopCount; i++) {
                    uint32_t baseOffset = (remainderLoopCount + i) * baseLineOffset + aLoopOffset;
                    uint32_t baseCountOffset = remainderLoopCount + i;
                    LoadAlign(quot, ((__ubuf__ float*)(tmpMeanLocal) + (baseOffset)));
                    LoadAlign<float, LoadDist::DIST_BRC_B32>(quotCount,
                                                             ((__ubuf__ float*)(tmpCountLocal) + baseCountOffset));
                    Mul(quot, quot, quotCount, pregLoop);
                    Muls(quot, quot, numScale, pregLoop);
                    StoreAlign(((__ubuf__ float*)tmpUbAddr + (remainderLoopCount + i) * rLoopStride + aLoopOffset),
                               quot, pregLoop);
                }
                LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
                // 最后对2的幂次行 二分累加
                // 末级换成补偿归约：mean 的错误会原样传进 (x-mean) 的近零对消，最敏感。
                CompensatedBinaryAddVF(tmpUbAddr, rLoopStride, binaryAddKLoop, binaryAddInnerLoop, binaryAddLastLoop,
                                       pregLoop, aLoopOffset, quot, rem, quotCount, remCount, addLow);
                LoadAlign(resMean, ((__ubuf__ float*)tmpUbAddr + aLoopOffset));
                // 此时 resMean = Σ(count_i·mean_i)/2^m。真除法 ÷patternR 后乘回 2^m（精确），
                // 得到一次正确舍入的全局均值；不再乘倒数常数 (2^m/patternR)。
                Div(resMean, resMean, nReg, pregLoop);
                Muls(resMean, resMean, powOfTwo, pregLoop);
                StoreAlign(((__ubuf__ float*)batchMeanTensorAddr + aLoopOffset), resMean, pregLoop);
                for (uint16_t i = 0; i < remainderLoopCount; i++) {
                    uint32_t quotOffset = i * baseLineOffset + aLoopOffset;
                    uint32_t remOffset = i * baseLineOffset + remainderOffset + aLoopOffset;
                    uint32_t quotCountOffset = i;
                    uint32_t remCountOffset = i + remainderCountOffset;
                    LoadAlign(quot, ((__ubuf__ float*)(tmpVarLocal) + (quotOffset)));
                    LoadAlign(rem, ((__ubuf__ float*)(tmpVarLocal) + (remOffset)));
                    LoadAlign(oriQuotMean, ((__ubuf__ float*)(tmpMeanLocal) + (quotOffset)));
                    LoadAlign(oriRemMean, ((__ubuf__ float*)(tmpMeanLocal) + (remOffset)));
                    LoadAlign<float, LoadDist::DIST_BRC_B32>(quotCount,
                                                             ((__ubuf__ float*)(tmpCountLocal) + quotCountOffset));
                    LoadAlign<float, LoadDist::DIST_BRC_B32>(remCount,
                                                             ((__ubuf__ float*)(tmpCountLocal) + remCountOffset));
                    Sub(oriQuotMean, oriQuotMean, resMean, pregLoop);
                    Sub(oriRemMean, oriRemMean, resMean, pregLoop);
                    Mul(oriQuotMean, oriQuotMean, oriQuotMean, pregLoop);
                    Mul(oriRemMean, oriRemMean, oriRemMean, pregLoop);
                    Mul(oriQuotMean, oriQuotMean, quotCount, pregLoop);
                    Mul(oriRemMean, oriRemMean, remCount, pregLoop);
                    Mul(quot, quot, quotCount, pregLoop);
                    Mul(rem, rem, remCount, pregLoop);
                    Add(quot, quot, oriQuotMean, pregLoop);
                    Add(rem, rem, oriRemMean, pregLoop);
                    Muls(quot, quot, numScale, pregLoop);
                    Muls(rem, rem, numScale, pregLoop);
                    Add(quot, quot, rem, pregLoop);
                    StoreAlign(((__ubuf__ float*)tmpUbAddr + i * rLoopStride + aLoopOffset), quot, pregLoop);
                }
                for (uint16_t i = 0; i < quotientLoopCount; i++) {
                    uint32_t baseOffset = (remainderLoopCount + i) * baseLineOffset + aLoopOffset;
                    uint32_t baseCountOffset = remainderLoopCount + i;
                    LoadAlign(quot, ((__ubuf__ float*)(tmpVarLocal) + (baseOffset)));
                    LoadAlign(oriQuotMean, ((__ubuf__ float*)(tmpMeanLocal) + (baseOffset)));
                    LoadAlign<float, LoadDist::DIST_BRC_B32>(quotCount,
                                                             ((__ubuf__ float*)(tmpCountLocal) + baseCountOffset));
                    Sub(oriQuotMean, oriQuotMean, resMean, pregLoop);
                    Mul(oriQuotMean, oriQuotMean, oriQuotMean, pregLoop);
                    Mul(oriQuotMean, oriQuotMean, quotCount, pregLoop);
                    Mul(quot, quot, quotCount, pregLoop);
                    Add(quot, quot, oriQuotMean, pregLoop);
                    Muls(quot, quot, numScale, pregLoop);
                    StoreAlign(((__ubuf__ float*)tmpUbAddr + (remainderLoopCount + i) * rLoopStride + aLoopOffset),
                               quot, pregLoop);
                }
                LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
                // 最后对2的幂次行 二分累加
                // 末级换成补偿归约：M2 的误差会被 ÷n 与 rsqrt 折算到第 5 个输出。
                CompensatedBinaryAddVF(tmpUbAddr, rLoopStride, binaryAddKLoop, binaryAddInnerLoop, binaryAddLastLoop,
                                       pregLoop, aLoopOffset, quot, rem, quotCount, remCount, addLow);
                LoadAlign(resVar, ((__ubuf__ float*)tmpUbAddr + aLoopOffset));
                // 累加时已乘 numScale = 1/2^m = 1/2^ceil(log2(patternR))（精确），故此时 resVar = M2_sum/2^m。
                // 真除法直接得到**有偏方差** M2/n（第 5 个输出 output_reserve_space_2 的契约值）
                // 与**无偏方差** M2/(n-1)（供 running var 使用，替代原来乘 Bessel 常数）。
                Div(resVarUnbiased, resVar, nMinusOneReg, pregLoop);
                Div(resVar, resVar, nReg, pregLoop);
                Muls(resVarUnbiased, resVarUnbiased, powOfTwo, pregLoop);
                Muls(resVar, resVar, powOfTwo, pregLoop);
                StoreAlign(((__ubuf__ float*)batchVarTensorAddr + aLoopOffset), resVar, pregLoop);
                // 复用 tmpVarLocal(tmpTbuf2) 存放无偏方差：LastFinalize 对该列的读取已在本次
                // aIndex 内完成，后续 aIndex 只访问自己那一列，故不会互相覆盖。
                StoreAlign(((__ubuf__ float*)tmpVarLocal + aLoopOffset), resVarUnbiased, pregLoop);
            }
        }
    }

    __aicore__ inline void CopyInGammaBeta(LocalTensor<T2>& gammaInUb, LocalTensor<T2>& betaInUb, int64_t gmOffset)
    {
        DataCopyPadExtParams<T2> dataCopyPadExtParamsT;
        dataCopyPadExtParamsT.isPad = false;
        dataCopyPadExtParamsT.leftPadding = 0;
        dataCopyPadExtParamsT.rightPadding = 0;
        dataCopyPadExtParamsT.paddingValue = 0;
        DataCopyExtParams copyInParamsT;
        copyInParamsT.blockCount = 1;
        copyInParamsT.blockLen = currentA * sizeof(T2);
        copyInParamsT.srcStride = 0;
        copyInParamsT.dstStride = 0;
        DataCopyPad(betaInUb, betaGm[gmOffset], copyInParamsT, dataCopyPadExtParamsT);
        DataCopyPad(gammaInUb, gammaGm[gmOffset], copyInParamsT, dataCopyPadExtParamsT);
    }

    __aicore__ inline void CopyInRunningMeanVar(LocalTensor<T2>& runningMeanInUb, LocalTensor<T2>& runningVarInUb,
                                                int64_t gmOffset)
    {
        DataCopyPadExtParams<T2> dataCopyPadExtParams;
        dataCopyPadExtParams.isPad = false;
        dataCopyPadExtParams.leftPadding = 0;
        dataCopyPadExtParams.rightPadding = 0;
        dataCopyPadExtParams.paddingValue = 0;
        DataCopyExtParams copyInParams;
        copyInParams.blockCount = 1;
        copyInParams.blockLen = currentA * sizeof(T2);
        copyInParams.srcStride = 0;
        copyInParams.dstStride = 0;
        DataCopyPad(runningMeanInUb, runningMeanGm[gmOffset], copyInParams, dataCopyPadExtParams);
        DataCopyPad(runningVarInUb, runningVarGm[gmOffset], copyInParams, dataCopyPadExtParams);
    }

    // unbiasedVarTensor 里是 LastFinalize 用真除法算出的无偏方差 M2/(n-1)，
    // running_var = (1-m)*running_var + m * M2/(n-1)。
    // 原实现把 batch_var(=M2/n) 再乘 Bessel 常数 fl(n/(n-1))，该常数舍入会带来系统性偏差。
    __aicore__ inline void CalculateRunningMeanVar(LocalTensor<T2>& runningMeanInUb, LocalTensor<T2>& runningVarInUb,
                                                   LocalTensor<T2>& runningMeanOutUb, LocalTensor<T2>& runningVarOutUb,
                                                   LocalTensor<float>& batchMeanTensor,
                                                   LocalTensor<float>& unbiasedVarTensor)
    {
        __ubuf__ T2* runningMeanInUbAddr = nullptr;
        __ubuf__ T2* runningVarInUbAddr = nullptr;
        if (useRunningMeanVar) {
            runningMeanInUbAddr = (__ubuf__ T2*)runningMeanInUb.GetPhyAddr();
            runningVarInUbAddr = (__ubuf__ T2*)runningVarInUb.GetPhyAddr();
        }
        __ubuf__ T2* runningMeanOutUbAddr = (__ubuf__ T2*)runningMeanOutUb.GetPhyAddr();
        __ubuf__ T2* runningVarOutUbAddr = (__ubuf__ T2*)runningVarOutUb.GetPhyAddr();
        __ubuf__ float* batchMeanTensorAddr = (__ubuf__ float*)batchMeanTensor.GetPhyAddr();
        __ubuf__ float* unbiasedVarTensorAddr = (__ubuf__ float*)unbiasedVarTensor.GetPhyAddr();
        uint16_t aLoop = CEIL_DIV(currentA, VL_F32);

        float m = tilingData->momentum;
        float oneSubM = tilingData->momentumReverse;
        bool vfUseRunningMeanVar = useRunningMeanVar;
        __VEC_SCOPE__
        {
            RegTensor<float> mean;
            RegTensor<float> var;
            RegTensor<float> runningMean;
            RegTensor<float> saveMean;
            RegTensor<float> runningVar;
            RegTensor<float> saveVar;
            MaskReg pregLoop;
            uint32_t sreg2 = currentA;
            for (uint16_t k = 0; k < aLoop; k++) {
                pregLoop = UpdateMask<float>(sreg2);
                // running var（无偏方差已由真除法得到，这里只乘 momentum，不再乘 Bessel 常数）
                LoadAlign(var, ((__ubuf__ float*)unbiasedVarTensorAddr + k * VL_F32));
                Muls(saveVar, var, m, pregLoop);
                if (vfUseRunningMeanVar) {
                    LoadTensorForDtypeT<T2>(runningVar, runningVarInUbAddr, pregLoop, k * VL_F32);
                    Muls(runningVar, runningVar, oneSubM, pregLoop);
                    Add(saveVar, saveVar, runningVar, pregLoop);
                }

                StoreTensorForDtypeT<T2>(runningVarOutUbAddr, saveVar, pregLoop, k * VL_FP32);

                // running mean
                LoadAlign(mean, ((__ubuf__ float*)batchMeanTensorAddr + k * VL_F32));
                Muls(saveMean, mean, m, pregLoop);
                if (vfUseRunningMeanVar) {
                    LoadTensorForDtypeT<T2>(runningMean, runningMeanInUbAddr, pregLoop, k * VL_F32);
                    Muls(runningMean, runningMean, oneSubM, pregLoop);
                    Add(saveMean, saveMean, runningMean, pregLoop);
                }

                StoreTensorForDtypeT<T2>(runningMeanOutUbAddr, saveMean, pregLoop, k * VL_F32);
            }
        }
    }

    __aicore__ inline void CopyOutRunningMeanVar(LocalTensor<T2>& runningMeanOutUb, LocalTensor<T2>& runningVarOutUb,
                                                 int64_t gmOffset)
    {
        DataCopyExtParams copyInParams;
        copyInParams.blockCount = 1;
        copyInParams.blockLen = currentA * sizeof(T2);
        copyInParams.srcStride = 0;
        copyInParams.dstStride = 0;
        DataCopyPad(runningMeanOutGm[gmOffset], runningMeanOutUb, copyInParams);
        DataCopyPad(runningVarOutGm[gmOffset], runningVarOutUb, copyInParams);
    }

    // 拷出 batch_mean(output_reserve_space_1) 与**有偏方差**(output_reserve_space_2)。
    __aicore__ inline void CopyOutBatchMeanVar(LocalTensor<float>& batchMeanInUb, LocalTensor<float>& batchVarInUb,
                                               int64_t gmOffset)
    {
        DataCopyExtParams copyInParams;
        copyInParams.blockCount = 1;
        copyInParams.blockLen = currentA * sizeof(float);
        copyInParams.srcStride = 0;
        copyInParams.dstStride = 0;
        DataCopyPad(batchMeanGm[gmOffset], batchMeanInUb, copyInParams);
        DataCopyPad(batchVarGm[gmOffset], batchVarInUb, copyInParams);
    }

    __aicore__ inline void NormalizeX(LocalTensor<float>& batchMeanTensor, LocalTensor<float>& rstdTensor,
                                      LocalTensor<T2>& gammaTensor, LocalTensor<T2>& betaTensor, int64_t yGmOffset)
    {
        LocalTensor<T1> xTensor = xQueue.AllocTensor<T1>();
        CopyInX(xTensor, yGmOffset);
        xQueue.EnQue(xTensor);
        xTensor = xQueue.DeQue<T1>();
        LocalTensor<T1> yTensor = yQueue.AllocTensor<T1>();
        CalcY(batchMeanTensor, rstdTensor, gammaTensor, betaTensor, xTensor, yTensor);
        xQueue.FreeTensor(xTensor);
        yQueue.EnQue(yTensor);
        yTensor = yQueue.template DeQue<T1>();
        CopyOutY(yTensor, yGmOffset);
        yQueue.FreeTensor(yTensor);
    }

    __aicore__ inline void CalcY(LocalTensor<float>& batchMeanTensor, LocalTensor<float>& rstdTensor,
                                 LocalTensor<T2>& gammaTensor, LocalTensor<T2>& betaTensor, LocalTensor<T1>& xTensor,
                                 LocalTensor<T1>& yTensor)
    {
        __ubuf__ float* batchMeanTensorAddr = (__ubuf__ float*)batchMeanTensor.GetPhyAddr();
        __ubuf__ float* rstdTensorAddr = (__ubuf__ float*)rstdTensor.GetPhyAddr();
        __ubuf__ T1* xTensorAddr = (__ubuf__ T1*)xTensor.GetPhyAddr();
        __ubuf__ T1* yTensorAddr = (__ubuf__ T1*)yTensor.GetPhyAddr();
        __ubuf__ T2* gammaTensorAddr = (__ubuf__ T2*)gammaTensor.GetPhyAddr();
        __ubuf__ T2* betaTensorAddr = (__ubuf__ T2*)betaTensor.GetPhyAddr();
        uint16_t numLoop = CEIL_DIV(currentA, VL_F32);
        uint16_t lineLoop = currentR;
        __VEC_SCOPE__
        {
            RegTensor<float> x1;
            RegTensor<float> mean;
            RegTensor<float> rstd;
            RegTensor<float> gamma;
            RegTensor<float> beta;
            RegTensor<float> y;
            MaskReg mask0;
            uint32_t sreg0 = currentA;
            for (uint16_t i = 0; i < numLoop; i++) {
                mask0 = AscendC::MicroAPI::UpdateMask<float>(sreg0);
                LoadAlign(mean, batchMeanTensorAddr + i * VL_F32);
                LoadAlign(rstd, rstdTensorAddr + i * VL_F32);
                LoadTensorForDtypeT<T2>(gamma, gammaTensorAddr, mask0, i * VL_F32);
                LoadTensorForDtypeT<T2>(beta, betaTensorAddr, mask0, i * VL_F32);
                for (uint16_t j = 0; j < lineLoop; j++) {
                    LoadTensorForDtypeT<T1>(x1, xTensorAddr, mask0, i * VL_F32 + j * currentAAlign);
                    Sub(x1, x1, mean, mask0);
                    Mul(x1, x1, rstd, mask0);
                    // 与 torch.compile 的 FMA 融合对齐，减少一次舍入：y = beta + x * gamma
                    AscendC::Reg::Move(y, beta);
                    Mula(y, x1, gamma, mask0);
                    if constexpr (IsSameType<T1, half>::value) {
                        RegTensor<half> yFp16;
                        Cast<half, float, castTraitB322B16>(yFp16, y, mask0);
                        StoreAlign<half, StoreDist::DIST_PACK_B32>(yTensorAddr + i * VL_F32 + j * currentAAlign, yFp16,
                                                                   mask0);
                    } else if constexpr (IsSameType<T1, bfloat16_t>::value) {
                        RegTensor<bfloat16_t> xBf16;
                        Cast<bfloat16_t, float, castTraitB322B16>(xBf16, y, mask0);
                        StoreAlign<bfloat16_t, StoreDist::DIST_PACK_B32>(yTensorAddr + i * VL_F32 + j * currentAAlign,
                                                                         xBf16, mask0);
                    } else {
                        StoreAlign(yTensorAddr + i * VL_F32 + j * currentAAlign, y, mask0);
                    }
                }
            }
        }
    }

    __aicore__ inline void CopyOutY(LocalTensor<T1>& yOutUb, int64_t offset)
    {
        DataCopyExtParams copyInParams;
        copyInParams.blockCount = currentR;
        copyInParams.blockLen = currentA * sizeof(T1);
        copyInParams.srcStride = 0;
        copyInParams.dstStride = (tilingData->patternA - currentA) * sizeof(T1);
        DataCopyPad(yGm[offset], yOutUb, copyInParams);
    }

    /* global memory address */
    GlobalTensor<T1> xGm;
    GlobalTensor<T2> betaGm;
    GlobalTensor<T2> gammaGm;
    GlobalTensor<T2> runningMeanGm;
    GlobalTensor<T2> runningVarGm;

    GlobalTensor<T1> yGm;
    GlobalTensor<float> batchMeanGm;
    GlobalTensor<float> batchVarGm;
    GlobalTensor<T2> runningMeanOutGm;
    GlobalTensor<T2> runningVarOutGm;
    GlobalTensor<float> meanWsp;
    GlobalTensor<float> varWsp;
    GlobalTensor<float> workspaceGm;

    const BatchNormExt2BlockSplitRTilingData* tilingData;
    TPipe* pipe;

    /* variable */
    int64_t rLoop = 0;
    int64_t currentA = 0;
    int64_t currentAAlign = 0;
    int64_t currentR = 0;

    float nFactor = 0;
    float lastNFactor = 0;

    // mean / var 的真除法标量（n 与 n-1 均 < 2^24，fp32 精确可表示）
    float nCount = 0;        // 本核 r 元素数 n
    float nPowOfTwo = 0;     // 累加预缩放用的 2 的幂（除法后乘回，精确）
    float lastNCount = 0;    // patternR
    float lastNMinusOne = 0; // patternR - 1
    float lastNPowOfTwo = 0; // 最后一级累加预缩放用的 2 的幂

    uint32_t usedCoreNum = 0;
    uint32_t blockIdx = 0;
    bool useRunningMeanVar = true;

    static constexpr uint32_t VL_F32 = VECTOR_REG_WIDTH / sizeof(float);
    static constexpr int64_t BLOCK_SIZE = 32;
    static constexpr int64_t T_BLOCK_ALIGN_SIZE = BLOCK_SIZE / sizeof(T1);
    static constexpr int64_t FP32_BLOCK_ALIGN_SIZE = BLOCK_SIZE / sizeof(float);
    static constexpr int64_t DOUBLE_BUFFER = 2;
    static constexpr int64_t SCALE_COEF_FOUR = 4;
    static constexpr uint32_t ROW_TWO_OFFSET = 2;
    static constexpr uint32_t ROW_THREE_OFFSET = 3;
    static constexpr uint32_t ROW_FOUR_OFFSET = 4;

    constexpr static AscendC::MicroAPI::CastTrait castTraitB162B32 = {
        AscendC::MicroAPI::RegLayout::ZERO,
        AscendC::MicroAPI::SatMode::UNKNOWN,
        AscendC::MicroAPI::MaskMergeMode::ZEROING,
        AscendC::RoundMode::UNKNOWN,
    };

    constexpr static AscendC::MicroAPI::CastTrait castTraitB322B16 = {
        AscendC::MicroAPI::RegLayout::ZERO,
        AscendC::MicroAPI::SatMode::NO_SAT,
        AscendC::MicroAPI::MaskMergeMode::ZEROING,
        AscendC::RoundMode::CAST_RINT,
    };

    /* ascendc variable */
    TQue<QuePosition::VECIN, 1> xQueue;
    TQue<QuePosition::VECIN, 1> gammaQueue;
    TQue<QuePosition::VECIN, 1> betaQueue;
    TQue<QuePosition::VECIN, 1> runningMeanInQueue;
    TQue<QuePosition::VECIN, 1> runningVarInQueue;

    TQue<QuePosition::VECOUT, 1> yQueue;
    TQue<QuePosition::VECOUT, 1> batchMeanQueue;
    TQue<QuePosition::VECOUT, 1> batchVarQueue;
    TQue<QuePosition::VECOUT, 1> runningMeanOutQueue;
    TQue<QuePosition::VECOUT, 1> runningVarOutQueue;

    TBuf<TPosition::VECCALC> tmpTbuf1;
    TBuf<TPosition::VECCALC> tmpTbuf2;
    TBuf<TPosition::VECCALC> tmpTbuf3;
    TBuf<TPosition::VECCALC> countTbuf1;
    TBuf<TPosition::VECCALC> countTbuf2;
};
} // namespace BatchNormExt2Ops

#endif // NORM_BATCH_NORM_EXT2_BLOCK_SPLIT_R_H

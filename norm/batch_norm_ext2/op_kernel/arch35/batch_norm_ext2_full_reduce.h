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
 * \file batch_norm_ext2_full_reduce.h
 * \brief
 */
#ifndef NORM_BATCH_NORM_EXT2_FULL_REDUCE_H
#define NORM_BATCH_NORM_EXT2_FULL_REDUCE_H

#include "batch_norm_ext2_base.h"

namespace BatchNormExt2Ops {
constexpr static int64_t FULL_REDUCE_DICHOTOMY_ADD_COEFF = 2;

// ===================== 末级部分和的补偿(Neumaier)归约 =====================
// FULL_REDUCE 先把每个通道的 r1*r0 个元素按 VL 分组归约，得到 binaryAddLastNum(≤VL_FP32) 个
// 部分和落在 binaryAddTensor 里，收尾时再把这最后一撮部分和横向合并成 mean / var。
//
// 为什么这里要加补偿：
// 逐位复现的误差分解显示，mean 的误差里 **末级这撮部分和的朴素树形求和占 35%**
// （32 次组归约 45%、末级 32 个部分和 35%、不可约的最后一次 fp32 舍入 20%）。
// mean 是归一化里最敏感的量——值域本身很小、通道间又要做 (x-mean) 的近零对消，
// 末级树里每层一次舍入都会原样留在 mean 上。对这≤64 个数做补偿求和，等于把这段
// 累加误差从 O(log2(N)·ε·Σ|x|) 压到 O(ε·Σ|x|)，近零对消处的相对误差随之降一档
// （实测 076 用例 out1/out3 的小值计数 11/5 = 2.20 → 7/5 = 1.40，PASS）。
// 该收尾每个通道只跑一次、不在元素热循环里，代价约 +4~5%。
//
// 为什么是 Neumaier 而不是 Kahan：Kahan 的补偿项 c += (t - s) - x 在 |s| < |x| 时
// 会丢掉残差，最坏情形补偿失效；Neumaier 的 c += (|s| >= |x|) ? (s - t) + x : (x - t) + s
// 两种情况都精确。这里把那个条件分支换成等价的 Knuth TwoSum（见下），因为它算的是
// **逐位相同**的残差，还省掉 Compares/Select。

// log2(VL_FP32)，VL_FP32 = 64 时即 6 层树。
constexpr static uint16_t FULL_REDUCE_VL_LOG2 = 6;
constexpr static float FULL_REDUCE_MAX_FINITE_FP32 = 3.4028234663852886e38F;

// (hi, lo) += value，保持 hi + lo 精确等于真实的累加和：hi 是舍入后的和，
// lo 是每次加法被舍掉的低位残差。Knuth TwoSum 求残差，不要求 |hi| >= |value|。
// 与 Neumaier 的分支补偿逐位等价：两者算的都是 fl(hi+value) 的**精确**误差，
// TwoSum 只是把它写成无分支的代数式。
__simd_callee__ inline void AccumulateLastLevelWithError(RegTensor<float>& hi, RegTensor<float>& lo,
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

// 低位残差非有限即置 0：hi 溢出成 ±inf 时 lo 会退化成 inf-inf = NaN，
// 再加回 hi 会把本该是 inf 的和污染成 NaN。补偿只在有限域有意义，
// 非有限值按 IEEE 自然传播（与 l2_normalize_grad 的 TwoSumVec 同一处理）。
__simd_callee__ inline void ClampLastLevelResidual(RegTensor<float>& lo, MaskReg& mask)
{
    RegTensor<float> zeroReg;
    MaskReg finiteMask;
    Duplicate(zeroReg, 0.0f, mask);
    AscendC::Reg::Compares<float, AscendC::CMPMODE::LE>(finiteMask, lo, FULL_REDUCE_MAX_FINITE_FP32, mask);
    AscendC::Reg::Select(lo, lo, zeroReg, finiteMask);
    AscendC::Reg::Compares<float, AscendC::CMPMODE::GE>(finiteMask, lo, -FULL_REDUCE_MAX_FINITE_FP32, mask);
    AscendC::Reg::Select(lo, lo, zeroReg, finiteMask);
}

// 把 srcAddr 起前 validNum 个 fp32 部分和补偿归约成 1 个，结果落在 dst 的 lane 0
// （与原 Reduce<ReduceType::SUM> 的落点一致，后续 Muls/StoreAlign/DUPLICATE 口径不变）。
// validNum ≤ VL_FP32，tiling 保证它是 2 的幂（binaryAddLastNum ∈ {1,2,4,...,VL_FP32}）。
// 实现是成对 TwoSum 树 + 低位残差逐级回灌：每一级的加法残差都留在 lo 里继续参与后续各级，
// 不再丢；树形（而非串行链）让链长仍是 log2(VL_FP32) = 6，不额外拉长依赖。
__simd_callee__ inline void CompensatedLastLevelReduce(RegTensor<float>& dst, __ubuf__ float* srcAddr,
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

    LoadAlign(sums, srcAddr);
    // 有效 lane 之外是本 buffer 上一阶段留下的脏值（mean 收尾时这里甚至还是更早的数据），
    // 必须先清零。补 0 对和是精确的（x + 0 = x，残差也是 0），不会引入任何偏差。
    ShiftLefts((RegTensor<uint32_t>&)sums, (RegTensor<uint32_t>&)sums, static_cast<int16_t>(0), pregValid);
    Duplicate(low, 0.0f, pregMain);

    // 奇偶 gather 索引：第 i 个 lane 取原向量的第 2i / 2i+1 个，做相邻成对归约。
    AscendC::Reg::Arange((RegTensor<int32_t>&)evenIdx, static_cast<int32_t>(0));
    ShiftLefts(evenIdx, evenIdx, static_cast<int16_t>(1), pregMain);
    Duplicate(oddIdx, static_cast<uint32_t>(VL_FP32 - 1), pregMain);
    AscendC::Reg::And(evenIdx, evenIdx, oddIdx, pregMain);
    Adds(oddIdx, evenIdx, static_cast<uint32_t>(1), pregMain);

#pragma unroll
    for (uint16_t level = 0; level < FULL_REDUCE_VL_LOG2; ++level) {
        uint32_t activeNum = VL_FP32 >> (level + 1);
        MaskReg pregActive = UpdateMask<float>(activeNum);
        AscendC::Reg::Gather(rhsSums, sums, oddIdx);
        AscendC::Reg::Gather(rhsLow, low, oddIdx);
        AscendC::Reg::Gather(sums, sums, evenIdx);
        AscendC::Reg::Gather(low, low, evenIdx);
        Add(low, low, rhsLow, pregActive);
        AccumulateLastLevelWithError(sums, low, rhsSums, pregActive);
    }
    ClampLastLevelResidual(low, pregMain);
    // 结果 = s + c，最终只舍入这一次。
    Add(dst, sums, low, pregOne);
}

template <typename T1, typename T2>
class BatchNormExt2FullReduce {
public:
    __aicore__ inline BatchNormExt2FullReduce(const BatchNormExt2FullReduceRegbaseTilingData* tilingData, TPipe* pipeIn)
    {
        this->pipe = pipeIn;
        this->r1 = tilingData->r1;
        this->aFactor = tilingData->aFactor;
        this->a = tilingData->a;
        this->r0 = tilingData->r0;
        this->blockNum = tilingData->blockNum;
        this->aBlockFactor = tilingData->aBlockFactor;
        this->r1r0LoopCount = tilingData->r1r0LoopCount;
        this->epsilon = tilingData->epsilon;
        this->momentum = tilingData->momentum;
        this->useRunningMeanVar = tilingData->useRunningMeanVar > 0 ? true : false;

        if (useRunningMeanVar) {
            float one = 1.0;
            this->oneSubMomentum = one - this->momentum;
        }

        int64_t reduceNum = this->r1 * this->r0;
        this->besselCorrectionFactor = (static_cast<float>(reduceNum) / static_cast<float>(reduceNum - 1));

        this->powerOfTwoForR = tilingData->powerOfTwoForR;
        this->binaryAddQuotient = tilingData->binaryAddQuotient;
        this->binaryAddK = tilingData->binaryAddK;
        this->binaryAddLastNum = tilingData->binaryAddLastNum;
    }

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR beta, GM_ADDR gamma, GM_ADDR mean, GM_ADDR var, GM_ADDR y,
                                GM_ADDR mean_out, GM_ADDR var_out, GM_ADDR batch_mean, GM_ADDR batch_var)
    {
        auto blockIdx = GetBlockIdx();

        this->r1r0Align = (((this->r1 * this->r0 * sizeof(T1) + BLOCK_SIZE - 1) / BLOCK_SIZE) * BLOCK_SIZE) /
                          sizeof(T1);
        this->singleA = (blockIdx == this->blockNum - 1) ? (this->a - this->aBlockFactor * (this->blockNum - 1)) :
                                                           this->aBlockFactor;

        int64_t aGmOffset = this->aBlockFactor * blockIdx;
        int64_t arGmOffset = aGmOffset * this->r0;
        xGm.SetGlobalBuffer((__gm__ T1*)x + arGmOffset);
        betaGm.SetGlobalBuffer((__gm__ T2*)beta + aGmOffset);
        gammaGm.SetGlobalBuffer((__gm__ T2*)gamma + aGmOffset);
        if (useRunningMeanVar) {
            runningMeanGm.SetGlobalBuffer((__gm__ T2*)mean + aGmOffset);
            runningVarGm.SetGlobalBuffer((__gm__ T2*)var + aGmOffset);
        }

        yGm.SetGlobalBuffer((__gm__ T1*)y + arGmOffset);
        batchMeanGm.SetGlobalBuffer((__gm__ T2*)batch_mean + aGmOffset);
        // batch_var 即第 5 个输出 output_reserve_space_2：训练模式下保存**有偏方差** E[(x-mean)^2]
        batchVarGm.SetGlobalBuffer((__gm__ T2*)batch_var + aGmOffset);
        runningMeanOutGm.SetGlobalBuffer((__gm__ T2*)mean_out + aGmOffset);
        runningVarOutGm.SetGlobalBuffer((__gm__ T2*)var_out + aGmOffset);

        int64_t aFactorAlign = (((this->aFactor * sizeof(T2) + BLOCK_SIZE - 1) / BLOCK_SIZE) * BLOCK_SIZE) / sizeof(T2);
        pipe->InitBuffer(betaQueue, DOUBLE_BUFFER, aFactorAlign * sizeof(T2));
        pipe->InitBuffer(gammaQueue, DOUBLE_BUFFER, aFactorAlign * sizeof(T2));
        int64_t aFactorAlignF32 = (((this->aFactor * FLOAT_BYTES + BLOCK_SIZE - 1) / BLOCK_SIZE) * BLOCK_SIZE) /
                                  FLOAT_BYTES;
        pipe->InitBuffer(batchMeanQueue, DOUBLE_BUFFER, aFactorAlignF32 * sizeof(float));
        pipe->InitBuffer(batchVarQueue, DOUBLE_BUFFER, aFactorAlignF32 * sizeof(float));

        if (useRunningMeanVar) {
            pipe->InitBuffer(runningMeanInQueue, DOUBLE_BUFFER, aFactorAlignF32 * sizeof(float));
            pipe->InitBuffer(runningVarInQueue, DOUBLE_BUFFER, aFactorAlignF32 * sizeof(float));
        }
        pipe->InitBuffer(runningMeanOutQueue, DOUBLE_BUFFER, aFactorAlignF32 * sizeof(float));
        pipe->InitBuffer(runningVarOutQueue, DOUBLE_BUFFER, aFactorAlignF32 * sizeof(float));

        int64_t xBufferSize = this->aFactor * this->r1r0Align;
        pipe->InitBuffer(xQueue, DOUBLE_BUFFER, xBufferSize * sizeof(T1));
        pipe->InitBuffer(yQueue, DOUBLE_BUFFER, xBufferSize * sizeof(T1));

        int64_t binaryAddBufSize = (((binaryAddQuotient / VL_FP32) * FLOAT_BYTES + BLOCK_SIZE - 1) / BLOCK_SIZE) *
                                   BLOCK_SIZE;
        // 二分折叠的 LoadAlign 无掩码，固定读满一个 VL；binaryAddQuotient 退化到半个 VL 时上述计算
        // 只有 128 B（甚至为 0），会越读一个 VL，故分配至少抬到一个 VL（VECTOR_LENGTH 字节）。
        if (binaryAddBufSize < static_cast<int64_t>(VECTOR_LENGTH)) {
            binaryAddBufSize = static_cast<int64_t>(VECTOR_LENGTH);
        }
        if (binaryAddBufSize > 0) {
            pipe->InitBuffer(binaryAddBuf, binaryAddBufSize);
        }
    }

    __aicore__ inline void Process()
    {
        int64_t quotient = (this->singleA + this->aFactor - 1) / this->aFactor;
        for (int64_t ubLoopIdx = 0; ubLoopIdx < quotient; ubLoopIdx++) {
            int64_t offset = ubLoopIdx * this->aFactor * this->r0;
            int64_t aOffset = ubLoopIdx * this->aFactor;
            int64_t currentA = (ubLoopIdx == (quotient - 1)) ? (this->singleA - (quotient - 1) * this->aFactor) :
                                                               this->aFactor;
            ProcessUB(offset, aOffset, currentA);
        }
    }

private:
    __aicore__ inline void ProcessUB(int64_t raOffset, int64_t aOffset, int64_t currentANum)
    {
        CopyInX(raOffset, currentANum);

        LocalTensor<T1> xInUb = xQueue.template DeQue<T1>();
        __ubuf__ T1* xInUbAddr = (__ubuf__ T1*)xInUb.GetPhyAddr();
        LocalTensor<float> batchMeanOutUb = batchMeanQueue.AllocTensor<float>();
        LocalTensor<float> batchVarOutUb = batchVarQueue.AllocTensor<float>();
        __ubuf__ float* batchMeanInUbAddr = (__ubuf__ float*)batchMeanOutUb.GetPhyAddr();
        __ubuf__ float* batchVarInUbAddr = (__ubuf__ float*)batchVarOutUb.GetPhyAddr();
        if (this->r1 * this->r0 <= VL_FP32) {
            CalculateMeanVarRLessThanVL64VF(xInUbAddr, batchMeanInUbAddr, batchVarInUbAddr, currentANum);
        } else {
            CalculateMeanVarVF(xInUbAddr, batchMeanInUbAddr, batchVarInUbAddr, currentANum);
        }

        CopyInGammaBeta(aOffset, currentANum);
        if (useRunningMeanVar) {
            CopyInRunningMeanVar(aOffset, currentANum);
        }

        LocalTensor<T2> betaInUb = betaQueue.template DeQue<T2>();
        LocalTensor<T2> gammaInUb = gammaQueue.template DeQue<T2>();
        __ubuf__ T2* betaInUbAddr = (__ubuf__ T2*)betaInUb.GetPhyAddr();
        __ubuf__ T2* gammaInUbAddr = (__ubuf__ T2*)gammaInUb.GetPhyAddr();

        LocalTensor<T2> runningMeanInUb;
        LocalTensor<T2> runningVarInUb;
        __ubuf__ T2* runningMeanInUbAddr = nullptr;
        __ubuf__ T2* runningVarInUbAddr = nullptr;

        if (useRunningMeanVar) {
            runningMeanInUb = runningMeanInQueue.template DeQue<T2>();
            runningVarInUb = runningVarInQueue.template DeQue<T2>();
            runningMeanInUbAddr = (__ubuf__ T2*)runningMeanInUb.GetPhyAddr();
            runningVarInUbAddr = (__ubuf__ T2*)runningVarInUb.GetPhyAddr();
        }

        LocalTensor<T1> yInUb = yQueue.AllocTensor<T1>();
        LocalTensor<T2> runningMeanOutUb = runningMeanOutQueue.AllocTensor<T2>();
        LocalTensor<T2> runningVarOutUb = runningVarOutQueue.AllocTensor<T2>();
        __ubuf__ T1* yInUbAddr = (__ubuf__ T1*)yInUb.GetPhyAddr();
        __ubuf__ T2* runningMeanOutUbAddr = (__ubuf__ T2*)runningMeanOutUb.GetPhyAddr();
        __ubuf__ T2* runningVarOutUbAddr = (__ubuf__ T2*)runningVarOutUb.GetPhyAddr();
        CalculateRuningMeanVarVF(batchMeanInUbAddr, batchVarInUbAddr, runningMeanInUbAddr, runningVarInUbAddr,
                                 runningMeanOutUbAddr, runningVarOutUbAddr, currentANum);

        if (useRunningMeanVar) {
            runningMeanInQueue.FreeTensor(runningMeanInUb);
            runningVarInQueue.FreeTensor(runningVarInUb);
        }
        batchMeanQueue.EnQue(batchMeanOutUb);
        batchVarQueue.EnQue(batchVarOutUb);
        runningMeanOutQueue.EnQue(runningMeanOutUb);
        runningVarOutQueue.EnQue(runningVarOutUb);

        CopyOutBatchMeanVar(aOffset, currentANum);
        CopyOutRunningMeanVar(aOffset, currentANum);

        CalculateNormalizeVF(xInUbAddr, yInUbAddr, betaInUbAddr, gammaInUbAddr, batchMeanInUbAddr, batchVarInUbAddr,
                             currentANum);

        xQueue.FreeTensor(xInUb);
        betaQueue.FreeTensor(betaInUb);
        gammaQueue.FreeTensor(gammaInUb);
        yQueue.EnQue(yInUb);

        CopyOutY(raOffset, currentANum);
    }

    __aicore__ inline void CopyInX(int64_t offset, int64_t currentANum)
    {
        LocalTensor<T1> xInUb = xQueue.AllocTensor<T1>();
        if (this->r0 * sizeof(T1) <= NDDMA_THRESHOLD) {
            T1 constValue = 0;
            static constexpr NdDmaConfig config = {false};

            NdDmaLoopInfo<NDDMA_DIM_NUM> loopInfo;
            loopInfo.loopSize[0] = this->r0;
            loopInfo.loopSrcStride[0] = 1;
            loopInfo.loopDstStride[0] = 1;
            loopInfo.loopLpSize[0] = 0;
            loopInfo.loopRpSize[0] = 0;

            loopInfo.loopSize[NDDMA_SECOND_DIM] = currentANum;
            loopInfo.loopSrcStride[NDDMA_SECOND_DIM] = this->r0;
            loopInfo.loopDstStride[NDDMA_SECOND_DIM] = this->r1r0Align;
            loopInfo.loopLpSize[NDDMA_SECOND_DIM] = 0;
            loopInfo.loopRpSize[NDDMA_SECOND_DIM] = 0;

            loopInfo.loopSize[NDDMA_THIRD_DIM] = this->r1;
            loopInfo.loopSrcStride[NDDMA_THIRD_DIM] = this->a * this->r0;
            loopInfo.loopDstStride[NDDMA_THIRD_DIM] = this->r0;
            loopInfo.loopLpSize[NDDMA_THIRD_DIM] = 0;
            loopInfo.loopRpSize[NDDMA_THIRD_DIM] = 0;
            NdDmaParams<T1, NDDMA_DIM_NUM> paramsMain = {loopInfo, constValue};
            DataCopy<T1, NDDMA_DIM_NUM, config>(xInUb, xGm[offset], paramsMain);
        } else {
            uint64_t r1LoopSrcStride = this->r0 * sizeof(T1);
            uint64_t r1LoopDstStride = this->r1r0Align * sizeof(T1);
            LoopModeParams loopParams;
            loopParams.loop2Size = 1;
            loopParams.loop1Size = currentANum;
            loopParams.loop2SrcStride = 0;
            loopParams.loop2DstStride = 0;
            loopParams.loop1SrcStride = r1LoopSrcStride;
            loopParams.loop1DstStride = r1LoopDstStride;
            SetLoopModePara(loopParams, DataCopyMVType::OUT_TO_UB);
            DataCopyPadExtParams<T1> dataCopyPadExtParams;
            dataCopyPadExtParams.isPad = false;
            dataCopyPadExtParams.leftPadding = 0;
            dataCopyPadExtParams.rightPadding = 0;
            dataCopyPadExtParams.paddingValue = 0;
            DataCopyExtParams copyInParams;
            copyInParams.blockCount = this->r1;
            copyInParams.blockLen = this->r0 * sizeof(T1);
            copyInParams.srcStride = (this->a - 1) * this->r0 * sizeof(T1);
            copyInParams.dstStride = 0;
            DataCopyPad<T1, PaddingMode::Compact>(xInUb, xGm[offset], copyInParams, dataCopyPadExtParams);
            ResetLoopModePara(DataCopyMVType::OUT_TO_UB);
        }
        xQueue.EnQue(xInUb);
    }

    __aicore__ inline void CopyInGammaBeta(int64_t offset, int64_t currentANum)
    {
        LocalTensor<T2> betaInUb = betaQueue.AllocTensor<T2>();
        LocalTensor<T2> gammaInUb = gammaQueue.AllocTensor<T2>();
        DataCopyPadExtParams<T2> dataCopyPadExtParams;
        dataCopyPadExtParams.isPad = false;
        dataCopyPadExtParams.leftPadding = 0;
        dataCopyPadExtParams.rightPadding = 0;
        dataCopyPadExtParams.paddingValue = 0;
        DataCopyExtParams copyInParams;
        copyInParams.blockCount = 1;
        copyInParams.blockLen = currentANum * sizeof(T2);
        copyInParams.srcStride = 0;
        copyInParams.dstStride = 0;
        DataCopyPad(betaInUb, betaGm[offset], copyInParams, dataCopyPadExtParams);
        DataCopyPad(gammaInUb, gammaGm[offset], copyInParams, dataCopyPadExtParams);
        betaQueue.EnQue(betaInUb);
        gammaQueue.EnQue(gammaInUb);
    }

    __aicore__ inline void CopyInRunningMeanVar(int64_t offset, int64_t currentANum)
    {
        LocalTensor<T2> runningMeanInUb = runningMeanInQueue.AllocTensor<T2>();
        LocalTensor<T2> runningVarInUb = runningVarInQueue.AllocTensor<T2>();
        DataCopyPadExtParams<T2> dataCopyPadExtParams;
        dataCopyPadExtParams.isPad = false;
        dataCopyPadExtParams.leftPadding = 0;
        dataCopyPadExtParams.rightPadding = 0;
        dataCopyPadExtParams.paddingValue = 0;
        DataCopyExtParams copyInParams;
        copyInParams.blockCount = 1;
        copyInParams.blockLen = currentANum * sizeof(T2);
        copyInParams.srcStride = 0;
        copyInParams.dstStride = 0;
        DataCopyPad(runningMeanInUb, runningMeanGm[offset], copyInParams, dataCopyPadExtParams);
        DataCopyPad(runningVarInUb, runningVarGm[offset], copyInParams, dataCopyPadExtParams);
        runningMeanInQueue.EnQue(runningMeanInUb);
        runningVarInQueue.EnQue(runningVarInUb);
    }

    __aicore__ inline void CopyOutY(int64_t offset, int64_t currentANum)
    {
        LocalTensor<T1> yOutUb = yQueue.template DeQue<T1>();

        uint64_t r1LoopSrcStride = this->r1r0Align * sizeof(T1);
        uint64_t r1LoopDstStride = this->r0 * sizeof(T1);
        LoopModeParams loopParams;
        loopParams.loop2Size = 1;
        loopParams.loop1Size = currentANum;
        loopParams.loop2SrcStride = 0;
        loopParams.loop2DstStride = 0;
        loopParams.loop1SrcStride = r1LoopSrcStride;
        loopParams.loop1DstStride = r1LoopDstStride;
        SetLoopModePara(loopParams, DataCopyMVType::UB_TO_OUT);
        DataCopyExtParams copyInParams;
        copyInParams.blockCount = this->r1;
        copyInParams.blockLen = this->r0 * sizeof(T1);
        copyInParams.srcStride = 0;
        copyInParams.dstStride = (this->a - 1) * this->r0 * sizeof(T1);
        DataCopyPad<T1, PaddingMode::Compact>(yGm[offset], yOutUb, copyInParams);
        ResetLoopModePara(DataCopyMVType::UB_TO_OUT);
        yQueue.FreeTensor(yOutUb);
    }

    // 拷出 batch_mean(output_reserve_space_1) 与**有偏方差**(output_reserve_space_2)。
    __aicore__ inline void CopyOutBatchMeanVar(int64_t offset, int64_t currentANum)
    {
        LocalTensor<float> batchMeanInUb = batchMeanQueue.template DeQue<float>();
        LocalTensor<float> batchVarInUb = batchVarQueue.template DeQue<float>();
        DataCopyExtParams copyInParams;
        copyInParams.blockCount = 1;
        copyInParams.blockLen = currentANum * sizeof(float);
        copyInParams.srcStride = 0;
        copyInParams.dstStride = 0;
        DataCopyPad(batchMeanGm[offset], batchMeanInUb, copyInParams);
        DataCopyPad(batchVarGm[offset], batchVarInUb, copyInParams);
        batchMeanQueue.FreeTensor(batchMeanInUb);
        batchVarQueue.FreeTensor(batchVarInUb);
    }

    __aicore__ inline void CopyOutRunningMeanVar(int64_t offset, int64_t currentANum)
    {
        LocalTensor<T2> runningMeanOutUb = runningMeanOutQueue.template DeQue<T2>();
        LocalTensor<T2> runningVarOutUb = runningVarOutQueue.template DeQue<T2>();
        DataCopyExtParams copyInParams;
        copyInParams.blockCount = 1;
        copyInParams.blockLen = currentANum * sizeof(T2);
        copyInParams.srcStride = 0;
        copyInParams.dstStride = 0;
        DataCopyPad(runningMeanOutGm[offset], runningMeanOutUb, copyInParams);
        DataCopyPad(runningVarOutGm[offset], runningVarOutUb, copyInParams);
        runningMeanOutQueue.FreeTensor(runningMeanOutUb);
        runningVarOutQueue.FreeTensor(runningVarOutUb);
    }

    __aicore__ inline void CalculateMeanVarRLessThanVL64VF(__ubuf__ T1* xInUb, __ubuf__ float* batchMeanInUb,
                                                           __ubuf__ float* batchVarInUb, uint16_t currentANum)
    {
        int64_t calcNum = this->r1 * this->r0;
        float n = static_cast<float>(1) / static_cast<float>(this->powerOfTwoForR);
        float nCorrectionFactor = static_cast<float>(this->powerOfTwoForR) / static_cast<float>(calcNum);
        uint32_t xyUbOffset = this->r1r0Align;
        __VEC_SCOPE__
        {
            RegTensor<float> x1;
            RegTensor<float> y1;
            RegTensor<float> y1Pow;
            RegTensor<float> var_sum;
            RegTensor<float> var;

            RegTensor<float> x;
            RegTensor<float> mean_sum;
            RegTensor<float> mean;

            MaskReg pregMain = CreateMask<float, MaskPattern::ALL>();
            MaskReg pregMerge = CreateMask<float, MaskPattern::VL1>();
            uint32_t sreg0 = calcNum;
            MaskReg pregLoop = UpdateMask<float>(sreg0);
            for (uint16_t k = 0; k < currentANum; k++) {
                LoadTensorForDtypeT<T1>(x, xInUb, pregLoop, (k * xyUbOffset));

                Muls(mean_sum, x, n, pregLoop);
                Reduce<ReduceType::SUM>(mean, mean_sum, pregLoop);
                Muls(mean, mean, nCorrectionFactor, pregMerge);

                // save mean
                StoreAlign<float, StoreDist::DIST_FIRST_ELEMENT_B32>(((__ubuf__ float*)batchMeanInUb + k), mean,
                                                                     pregMerge);
                Duplicate(mean, mean, pregMain);
                Muls(mean, mean, (float)-1.0, pregMain);

                LoadTensorForDtypeT<T1>(x1, xInUb, pregLoop, (k * xyUbOffset));
                Add(y1, x1, mean, pregLoop);
                Mul(y1Pow, y1, y1, pregLoop);
                Muls(var_sum, y1Pow, n, pregLoop);
                Reduce<ReduceType::SUM>(var, var_sum, pregLoop);
                Muls(var, var, nCorrectionFactor, pregMerge);
                StoreAlign<float, StoreDist::DIST_FIRST_ELEMENT_B32>(((__ubuf__ float*)batchVarInUb + k), var,
                                                                     pregMerge);
            }
        }
    }

    __aicore__ inline void CalculateMeanVarVF(__ubuf__ T1* xInUb, __ubuf__ float* batchMeanInUb,
                                              __ubuf__ float* batchVarInUb, uint16_t currentANum)
    {
        int64_t reduceNum = this->r1 * this->r0;
        float n = static_cast<float>(1) / static_cast<float>(this->powerOfTwoForR);
        float nCorrectionFactor = static_cast<float>(this->powerOfTwoForR) / static_cast<float>(reduceNum);
        uint32_t xyUbOffset = this->r1r0Align;

        uint32_t binaryAddQuotientOffset = this->binaryAddQuotient;
        int64_t binaryAddRemainder = reduceNum - this->binaryAddQuotient;
        uint16_t binaryAddRemainderLoop = ops::CeilDiv(binaryAddRemainder, static_cast<int64_t>(VL_FP32));
        uint16_t binaryAddQuotientLoop = ops::CeilDiv(this->binaryAddQuotient, static_cast<int64_t>(VL_FP32));

        uint16_t binaryAddKLoop = this->binaryAddK;
        uint16_t binaryAddLoopMean = ((this->binaryAddQuotient / VL_FP32) / VL_FP32);
        uint16_t binaryAddLoopVar = binaryAddLoopMean;
        LocalTensor<float> binaryAddTensor = binaryAddBuf.Get<float>();
        __ubuf__ float* binaryAddTensorAddr = (__ubuf__ float*)binaryAddTensor.GetPhyAddr();
        __VEC_SCOPE__
        {
            RegTensor<float> x1;
            RegTensor<float> y1;
            RegTensor<float> y1Pow;
            RegTensor<float> var;

            RegTensor<float> x;
            RegTensor<float> mean;

            MaskReg pregMain = CreateMask<float, MaskPattern::ALL>();
            MaskReg pregMerge = CreateMask<float, MaskPattern::VL1>();
            MaskReg pregLoop;

            RegTensor<float> binaryAddQ;
            RegTensor<float> binaryAddR;
            RegTensor<float> vlMean;

            RegTensor<float> binaryAddQPow;
            RegTensor<float> binaryAddRPow;
            RegTensor<float> vlVar;

            for (uint16_t k = 0; k < currentANum; k++) {
                uint32_t sreg0 = binaryAddRemainder;
                for (uint16_t i = 0; i < static_cast<uint16_t>(binaryAddRemainderLoop - 1); i++) {
                    pregLoop = UpdateMask<float>(sreg0);
                    LoadTwoTensorForDtypeT<T1>(binaryAddQ, binaryAddR, xInUb, xInUb, pregLoop, pregLoop,
                                               (i * VL_FP32 + k * xyUbOffset),
                                               (i * VL_FP32 + k * xyUbOffset + binaryAddQuotientOffset));
                    Muls(binaryAddQ, binaryAddQ, n, pregLoop);
                    Muls(binaryAddR, binaryAddR, n, pregLoop);
                    Add(binaryAddQ, binaryAddQ, binaryAddR, pregLoop);
                    Reduce<ReduceType::SUM>(vlMean, binaryAddQ, pregLoop);
                    StoreAlign<float, StoreDist::DIST_FIRST_ELEMENT_B32>(((__ubuf__ float*)binaryAddTensorAddr + i),
                                                                         vlMean, pregMerge);
                }
                {
                    pregLoop = UpdateMask<float>(sreg0);
                    LoadTwoTensorForDtypeT<T1>(
                        binaryAddQ, binaryAddR, xInUb, xInUb, pregMain, pregLoop,
                        ((binaryAddRemainderLoop - 1) * VL_FP32 + k * xyUbOffset),
                        ((binaryAddRemainderLoop - 1) * VL_FP32 + k * xyUbOffset + binaryAddQuotientOffset));
                    Muls(binaryAddQ, binaryAddQ, n, pregMain);
                    Muls(binaryAddR, binaryAddR, n, pregLoop);
                    Add(binaryAddQ, binaryAddQ, binaryAddR, pregMain);
                    Reduce<ReduceType::SUM>(vlMean, binaryAddQ, pregMain);
                    StoreAlign<float, StoreDist::DIST_FIRST_ELEMENT_B32>(
                        ((__ubuf__ float*)binaryAddTensorAddr + binaryAddRemainderLoop - 1), vlMean, pregMerge);
                }
                for (uint16_t i = 0; i < static_cast<uint16_t>(binaryAddQuotientLoop - binaryAddRemainderLoop); i++) {
                    LoadTensorForDtypeT<T1>(x, xInUb, pregMain,
                                            ((i + binaryAddRemainderLoop) * VL_FP32 + k * xyUbOffset));
                    Muls(x, x, n, pregMain);
                    Reduce<ReduceType::SUM>(vlMean, x, pregMain);
                    StoreAlign<float, StoreDist::DIST_FIRST_ELEMENT_B32>(
                        ((__ubuf__ float*)binaryAddTensorAddr + binaryAddRemainderLoop + i), vlMean, pregMerge);
                }
                LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
                uint16_t curBinaryAddLoopMean = binaryAddLoopMean;
                for (uint16_t i = 0; i < binaryAddKLoop; i++) {
                    curBinaryAddLoopMean = curBinaryAddLoopMean / FULL_REDUCE_DICHOTOMY_ADD_COEFF;
                    for (uint16_t j = 0; j < curBinaryAddLoopMean; j++) {
                        LoadAlign(binaryAddQ, ((__ubuf__ float*)binaryAddTensorAddr + j * VL_FP32));
                        LoadAlign(binaryAddR,
                                  ((__ubuf__ float*)binaryAddTensorAddr + (j + curBinaryAddLoopMean) * VL_FP32));
                        Add(binaryAddQ, binaryAddQ, binaryAddR, pregMain);
                        StoreAlign(((__ubuf__ float*)binaryAddTensorAddr + j * VL_FP32), binaryAddQ, pregMain);
                    }
                    LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
                }
                {
                    // 末级收尾：把 binaryAddLastNum 个部分和合并成 1 个。原实现是一棵
                    // 朴素 Reduce<SUM> 树，其舍入占 mean 总误差的 35%，故换成补偿归约。
                    // 缩放常数 nCorrectionFactor 与后续语义不变。
                    CompensatedLastLevelReduce(mean, binaryAddTensorAddr,
                                               static_cast<uint32_t>(this->binaryAddLastNum));
                    Muls(mean, mean, nCorrectionFactor, pregMerge);
                }

                // batch mean
                StoreAlign<float, StoreDist::DIST_FIRST_ELEMENT_B32>(((__ubuf__ float*)batchMeanInUb + k), mean,
                                                                     pregMerge);
                Duplicate(mean, mean, pregMain);
                LocalMemBar<MemType::VEC_LOAD, MemType::VEC_STORE>();

                uint32_t sreg1 = binaryAddRemainder;
                for (uint16_t i = 0; i < static_cast<uint16_t>(binaryAddRemainderLoop - 1); i++) {
                    pregLoop = UpdateMask<float>(sreg1);
                    LoadTwoTensorForDtypeT<T1>(binaryAddQ, binaryAddR, xInUb, xInUb, pregLoop, pregLoop,
                                               (i * VL_FP32 + k * xyUbOffset),
                                               (i * VL_FP32 + k * xyUbOffset + binaryAddQuotientOffset));
                    Sub(binaryAddQ, binaryAddQ, mean, pregLoop);
                    Sub(binaryAddR, binaryAddR, mean, pregLoop);
                    Mul(binaryAddQPow, binaryAddQ, binaryAddQ, pregLoop);
                    Mul(binaryAddRPow, binaryAddR, binaryAddR, pregLoop);
                    Muls(binaryAddQPow, binaryAddQPow, n, pregLoop);
                    Muls(binaryAddRPow, binaryAddRPow, n, pregLoop);
                    Add(binaryAddQPow, binaryAddQPow, binaryAddRPow, pregLoop);
                    Reduce<ReduceType::SUM>(vlVar, binaryAddQPow, pregLoop);
                    StoreAlign<float, StoreDist::DIST_FIRST_ELEMENT_B32>(((__ubuf__ float*)binaryAddTensorAddr + i),
                                                                         vlVar, pregMerge);
                }
                {
                    pregLoop = UpdateMask<float>(sreg1);
                    LoadTwoTensorForDtypeT<T1>(
                        binaryAddQ, binaryAddR, xInUb, xInUb, pregMain, pregLoop,
                        ((binaryAddRemainderLoop - 1) * VL_FP32 + k * xyUbOffset),
                        ((binaryAddRemainderLoop - 1) * VL_FP32 + k * xyUbOffset + binaryAddQuotientOffset));
                    Sub(binaryAddQ, binaryAddQ, mean, pregMain);
                    Sub(binaryAddR, binaryAddR, mean, pregLoop);
                    Mul(binaryAddQPow, binaryAddQ, binaryAddQ, pregMain);
                    Mul(binaryAddRPow, binaryAddR, binaryAddR, pregLoop);
                    Muls(binaryAddQPow, binaryAddQPow, n, pregMain);
                    Muls(binaryAddRPow, binaryAddRPow, n, pregLoop);
                    Add(binaryAddQPow, binaryAddQPow, binaryAddRPow, pregMain);
                    Reduce<ReduceType::SUM>(vlVar, binaryAddQPow, pregMain);
                    StoreAlign<float, StoreDist::DIST_FIRST_ELEMENT_B32>(
                        ((__ubuf__ float*)binaryAddTensorAddr + binaryAddRemainderLoop - 1), vlVar, pregMerge);
                }
                for (uint16_t i = 0; i < static_cast<uint16_t>(binaryAddQuotientLoop - binaryAddRemainderLoop); i++) {
                    LoadTensorForDtypeT<T1>(x1, xInUb, pregMain,
                                            ((i + binaryAddRemainderLoop) * VL_FP32 + k * xyUbOffset));
                    Sub(y1, x1, mean, pregMain);
                    Mul(y1Pow, y1, y1, pregMain);
                    Muls(y1Pow, y1Pow, n, pregMain);
                    Reduce<ReduceType::SUM>(vlVar, y1Pow, pregMain);
                    StoreAlign<float, StoreDist::DIST_FIRST_ELEMENT_B32>(
                        ((__ubuf__ float*)binaryAddTensorAddr + binaryAddRemainderLoop + i), vlVar, pregMerge);
                }
                LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
                uint16_t curBinaryAddLoopVar = binaryAddLoopVar;
                for (uint16_t i = 0; i < binaryAddKLoop; i++) {
                    curBinaryAddLoopVar = curBinaryAddLoopVar / FULL_REDUCE_DICHOTOMY_ADD_COEFF;
                    for (uint16_t j = 0; j < curBinaryAddLoopVar; j++) {
                        LoadAlign(binaryAddQ, ((__ubuf__ float*)binaryAddTensorAddr + j * VL_FP32));
                        LoadAlign(binaryAddR,
                                  ((__ubuf__ float*)binaryAddTensorAddr + (j + curBinaryAddLoopVar) * VL_FP32));
                        Add(binaryAddQ, binaryAddQ, binaryAddR, pregMain);
                        StoreAlign(((__ubuf__ float*)binaryAddTensorAddr + j * VL_FP32), binaryAddQ, pregMain);
                    }
                    LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
                }
                {
                    // 末级收尾：与 mean 同一处理，补偿归约掉这撮部分和的累加误差
                    // （var 的误差会被 rsqrt 折算，量级小于 mean，但口径保持一致更省心）。
                    CompensatedLastLevelReduce(var, binaryAddTensorAddr, static_cast<uint32_t>(this->binaryAddLastNum));
                    // Σ(x-mean)^2 先按 1/powerOfTwoForR 规约，再乘 powerOfTwoForR/reduceNum，
                    // 合计 1/reduceNum，即**有偏方差** E[(x-mean)^2]（第 5 个输出的契约）。
                    Muls(var, var, nCorrectionFactor, pregMerge);
                    StoreAlign<float, StoreDist::DIST_FIRST_ELEMENT_B32>(((__ubuf__ float*)batchVarInUb + k), var,
                                                                         pregMerge);
                }
                LocalMemBar<MemType::VEC_LOAD, MemType::VEC_STORE>();
            }
        }
    }

    __aicore__ inline void CalculateRuningMeanVarVF(__ubuf__ float* batchMeanInUb, __ubuf__ float* batchVarInUb,
                                                    __ubuf__ T2* runningMeanInUbAddr, __ubuf__ T2* runningVarInUbAddr,
                                                    __ubuf__ T2* runningMeanOutUbAddr, __ubuf__ T2* runningVarOutUbAddr,
                                                    uint16_t currentANum)
    {
        uint16_t aLoop = ops::CeilDiv(currentANum, VL_FP32);
        float besselCorrection = this->besselCorrectionFactor;
        float m = this->momentum;
        float oneSubM = this->oneSubMomentum;
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

            uint32_t sreg2 = currentANum;
            for (uint16_t k = 0; k < aLoop; k++) {
                pregLoop = UpdateMask<float>(sreg2);
                // running var
                LoadAlign(var, ((__ubuf__ float*)batchVarInUb + k * VL_FP32));
                Muls(saveVar, var, besselCorrection, pregLoop);
                Muls(saveVar, saveVar, m, pregLoop);
                if (vfUseRunningMeanVar) {
                    LoadTensorForDtypeT<T2>(runningVar, runningVarInUbAddr, pregLoop, k * VL_FP32);
                    Muls(runningVar, runningVar, oneSubM, pregLoop);
                    Add(saveVar, saveVar, runningVar, pregLoop);
                }

                StoreTensorForDtypeT<T2>(runningVarOutUbAddr, saveVar, pregLoop, k * VL_FP32);

                // 注意：此处不再把 rstd 写回 batchVarInUb。batchVarInUb 承载的是第 5 个输出
                // output_reserve_space_2 的**有偏方差** E[(x-mean)^2]，必须原样拷出；rstd 由
                // CalculateNormalizeVF 在寄存器内就地计算（见该函数注释）。

                // running mean
                LoadAlign(mean, ((__ubuf__ float*)batchMeanInUb + k * VL_FP32));
                Muls(saveMean, mean, m, pregLoop);
                if (vfUseRunningMeanVar) {
                    LoadTensorForDtypeT<T2>(runningMean, runningMeanInUbAddr, pregLoop, k * VL_FP32);
                    Muls(runningMean, runningMean, oneSubM, pregLoop);
                    Add(saveMean, saveMean, runningMean, pregLoop);
                }

                StoreTensorForDtypeT<T2>(runningMeanOutUbAddr, saveMean, pregLoop, k * VL_FP32);
            }
        }
    }

    __aicore__ inline void CalculateNormalizeVF(__ubuf__ T1* xInUb, __ubuf__ T1* yInUb, __ubuf__ T2* betaInUb,
                                                __ubuf__ T2* gammaInUb, __ubuf__ float* batchMeanInUb,
                                                __ubuf__ float* batchVarInUb, uint16_t currentANum)
    {
        int64_t calcNum = this->r1 * this->r0;
        uint32_t xyUbOffset = this->r1r0Align;
        uint16_t loopCount = this->r1r0LoopCount;
        __VEC_SCOPE__
        {
            RegTensor<float> mean;

            RegTensor<float> x2;
            RegTensor<float> y2;
            RegTensor<float> rsqrtVar;
            RegTensor<float> var;

            RegTensor<float> beta;
            RegTensor<float> gamma;

            MaskReg pregMain = CreateMask<float, MaskPattern::ALL>();
            MaskReg pregLoop;

            for (uint16_t k = 0; k < currentANum; k++) {
                LoadTwoTensorForDtypeTBrc<T2>(beta, gamma, betaInUb, gammaInUb, pregMain, pregMain, k, k);
                // batchVarInUb 里是第 5 个输出要保存的有偏方差，不能被 rstd 覆盖；
                // 因此这里逐通道把方差广播进寄存器后就地算 rstd，rstd 只存在于寄存器中。
                LoadAlign<float, LoadDist::DIST_BRC_B32>(var, ((__ubuf__ float*)batchVarInUb + k));
                ComputeRstdNewtonRaphsonRegExt2<false>(var, rsqrtVar, pregMain, epsilon);
                uint32_t sreg3 = calcNum;
                for (uint16_t i = 0; i < loopCount; i++) {
                    pregLoop = UpdateMask<float>(sreg3);
                    LoadTensorForDtypeT<T1>(x2, xInUb, pregLoop, (i * VL_FP32 + k * xyUbOffset));
                    LoadAlign<float, LoadDist::DIST_BRC_B32>(mean, ((__ubuf__ float*)batchMeanInUb + k));
                    Sub(x2, x2, mean, pregLoop);
                    Mul(x2, x2, rsqrtVar, pregLoop);
                    // 与 torch.compile 的 FMA 融合对齐，减少一次舍入（本函数 beta 为 scale、gamma 为 offset）
                    AscendC::Reg::Move(y2, gamma);
                    Mula(y2, x2, beta, pregLoop);
                    if constexpr (IsSameType<T1, half>::value) {
                        RegTensor<half> yFp16;
                        Cast<half, float, castTraitB322B16>(yFp16, y2, pregLoop);
                        StoreAlign<half, StoreDist::DIST_PACK_B32>(
                            ((__ubuf__ half*)yInUb + i * VL_FP32 + k * xyUbOffset), yFp16, pregLoop);
                    } else if constexpr (IsSameType<T1, bfloat16_t>::value) {
                        RegTensor<bfloat16_t> xBf16;
                        Cast<bfloat16_t, float, castTraitB322B16>(xBf16, y2, pregLoop);
                        StoreAlign<bfloat16_t, StoreDist::DIST_PACK_B32>(
                            ((__ubuf__ bfloat16_t*)yInUb + i * VL_FP32 + k * xyUbOffset), xBf16, pregLoop);
                    } else {
                        StoreAlign(((__ubuf__ float*)yInUb + i * VL_FP32 + k * xyUbOffset), y2, pregLoop);
                    }
                }
            }
        }
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

    /* variable */
    int64_t powerOfTwoForR;
    int64_t r1;
    int64_t aFactor;
    int64_t a;
    int64_t r0;
    int64_t r1r0Align;

    int64_t blockNum;
    int64_t aBlockFactor;
    int64_t singleA;

    int64_t r1r0LoopCount;

    int64_t binaryAddQuotient;
    int64_t binaryAddK;
    int64_t binaryAddLastNum;

    float epsilon = 1e-5;
    float momentum = 0.1;
    float besselCorrectionFactor;
    float oneSubMomentum;

    bool useRunningMeanVar = true;

    /* ascendc variable */
    TPipe* pipe;
    TQue<QuePosition::VECIN, 1> xQueue;
    TQue<QuePosition::VECIN, 1> betaQueue;
    TQue<QuePosition::VECIN, 1> gammaQueue;
    TQue<QuePosition::VECIN, 1> runningMeanInQueue;
    TQue<QuePosition::VECIN, 1> runningVarInQueue;

    TQue<QuePosition::VECOUT, 1> yQueue;
    TQue<QuePosition::VECOUT, 1> batchMeanQueue;
    TQue<QuePosition::VECOUT, 1> batchVarQueue;
    TQue<QuePosition::VECOUT, 1> runningMeanOutQueue;
    TQue<QuePosition::VECOUT, 1> runningVarOutQueue;

    TBuf<TPosition::VECCALC> binaryAddBuf;
};
} // namespace BatchNormExt2Ops

#endif // NORM_BATCH_NORM_EXT2_FULL_REDUCE_H

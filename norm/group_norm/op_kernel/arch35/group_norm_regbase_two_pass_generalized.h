/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file group_norm_regbase_two_pass_generalized.h
 * \brief
 */

#ifndef GROUP_NORM_REGBASE_TWO_PASS_GENERALIZED_H_
#define GROUP_NORM_REGBASE_TWO_PASS_GENERALIZED_H_

#include "group_norm_regbase_base.h"

namespace GroupNorm {
using namespace AscendC;
template <typename T1, typename T2, int32_t BUFFER_NUM = 2>
class GroupNormTwoPassGeneralized {
public:
    __aicore__ inline GroupNormTwoPassGeneralized(){};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR gamma, GM_ADDR beta, GM_ADDR y, GM_ADDR mean, GM_ADDR variance,
                                const GroupNormTilingData* tilingData)
    {
        // 绑定GM张量并按tiling规划UB。
        tiling = tilingData;
        blockIdx = GetBlockIdx();
        blockNum = GetBlockNum();
        xGm.SetGlobalBuffer((__gm__ T1*)x);
        if (gamma != nullptr) {
            hasGamma = true;
            gammaGm.SetGlobalBuffer((__gm__ T2*)gamma);
        }
        if (beta != nullptr) {
            hasBeta = true;
            betaGm.SetGlobalBuffer((__gm__ T2*)beta);
        }
        yGm.SetGlobalBuffer((__gm__ T1*)y);
        meanGm.SetGlobalBuffer((__gm__ T1*)mean);
        varianceGm.SetGlobalBuffer((__gm__ T1*)variance);
        ParseTilingData();
        InitInnerBuffer();
    }

    __aicore__ inline void Process()
    {
        // 分批完成大通道TwoPass计算和统计量写回。
        uint32_t numPerCoreLoop = CeilDiv(numPerCore, innerNumPerCore);
        uint32_t numPerCoreTail = numPerCore % innerNumPerCore == 0 ? innerNumPerCore : numPerCore % innerNumPerCore;
        uint32_t numPerCoreOneLoop = innerNumPerCore;
        event_t eventIdMte3ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));
        // 跨 CalNormalize 调用的 MTE2 同步：迭代末 V_MTE2 置位、迭代首等待，
        // 确保上一迭代尾批 VEC 读 x/γ/β 完成后，下一迭代的 MTE2 拷贝才发射（N>1 多调用精度依赖）。
        event_t eventIdVToMte2Cross = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE2));
        for (uint64_t i = 0; i < numPerCoreLoop; i++) {
            if (i > 0) {
                WaitFlag<HardEvent::MTE3_V>(eventIdMte3ToV);
                WaitFlag<HardEvent::V_MTE2>(eventIdVToMte2Cross);
            }
            if (i == numPerCoreLoop - 1) {
                numPerCoreOneLoop = numPerCoreTail;
            }
            CalNormalize(i * innerNumPerCore, numPerCoreOneLoop);
            SetFlag<HardEvent::V_MTE2>(eventIdVToMte2Cross);
            ProcessMeanAndVariance<T1>(meanTensor, meanOutTensor, meanGm, varianceOutTensor, varianceGm,
                                       blockIdx * tiling->numPerCore + i * innerNumPerCore, numPerCoreOneLoop);
            if (i < numPerCoreLoop - 1) {
                SetFlag<HardEvent::MTE3_V>(eventIdMte3ToV);
            }
        }
    }

private:
    __aicore__ inline void CalNormalize(uint64_t offset, uint32_t numPerCoreTmp)
    {
        // 按x-批（batchFactor组）批量拷贝x与γ/β跨度，同步降到批级；批内统计与仿射为纯VF。
        int64_t numPerCoreExtent = CeilDiv(numPerCoreTmp, batchFactor);
        uint32_t numPerCoreTail = numPerCoreTmp % batchFactor == 0 ? batchFactor : numPerCoreTmp % batchFactor;
        uint32_t numPerCoreProcess = batchFactor;
        uint32_t xUbStride = maxBatch * elemNumAlign;
        uint64_t xGmBaseOffset = blockIdx * tiling->numPerCore * elemNum + offset * elemNum;
        auto eventIDMte2ToVPing = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::MTE2_V>());
        auto eventIDMte2ToVPong = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::MTE2_V>());
        auto eventIDVToMte3Ping = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::V_MTE3>());
        auto eventIDVToMte3Pong = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::V_MTE3>());
        auto eventIDVToMte2Span = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::V_MTE2>());
        auto eventIDVToMte2XPing = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::V_MTE2>());
        auto eventIDVToMte2XPong = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::V_MTE2>());
        auto eventIDMte3ToVPing = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::MTE3_V>());
        auto eventIDMte3ToVPong = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::MTE3_V>());
        __local_mem__ float* dichotomyAddLocal = (__local_mem__ float*)dichotomyAddTensor.GetPhyAddr();
        for (int64_t i = 0; i < numPerCoreExtent; i++) {
            if (i == numPerCoreExtent - 1) {
                numPerCoreProcess = numPerCoreTail;
            }
            bool isPing = (i % BUFFER_NUM) == 0;
            if (i > 1) {
                WaitFlag<HardEvent::V_MTE2>(isPing ? eventIDVToMte2XPing : eventIDVToMte2XPong);
            }
            if (i > 0) {
                WaitFlag<HardEvent::V_MTE2>(eventIDVToMte2Span);
            }
            int64_t xGmOffset = xGmBaseOffset + i * batchFactor * elemNum;
            uint32_t xUbOffset = isPing * xUbStride;
            uint32_t stageBase = (xUbOffset + xUbStride) % (BUFFER_NUM * xUbStride);
            // 退化批（elemNum==1）路径判定：γ/β 跨度越过 shapeC 回卷点时需拆段；
            // 批槽位基址须 32B 对齐（向量存储约束），不满足时回退基线逐组路径。
            bool spanSingle = true;
            bool slotsAligned = (batchFactor * i) % (BLOCK_SIZE / sizeof(T1)) == 0;
            bool degenerateBatch = false;
            if (degeneratePath) {
                uint64_t gbStart = static_cast<uint64_t>(blockIdx * tiling->numPerCore) + offset +
                                   static_cast<uint64_t>(i) * batchFactor;
                uint64_t chanStart = gbStart % static_cast<uint64_t>(numGroups); // shapeD==1
                spanSingle = chanStart + numPerCoreProcess <= static_cast<uint64_t>(shapeC);
                degenerateBatch = slotsAligned && (spanSingle || BUFFER_NUM >= 2);
            }
            if (degenerateBatch) {
                CopyX2UB<T1>(xGm[xGmOffset], xTensor[xUbOffset], 1, numPerCoreProcess);
            } else {
                CopyX2UB<T1>(xGm[xGmOffset], xTensor[xUbOffset], numPerCoreProcess, elemNum);
            }
            CopyGammaBetaSpan2UB(offset + i * batchFactor, numPerCoreProcess);
            if (degenerateBatch && !spanSingle) {
                CopyGammaBetaStagedSpan2UB(numPerCoreProcess, stageBase);
            }
            SetFlag<HardEvent::MTE2_V>(isPing ? eventIDMte2ToVPing : eventIDMte2ToVPong);
            WaitFlag<HardEvent::MTE2_V>(isPing ? eventIDMte2ToVPing : eventIDMte2ToVPong);
            __local_mem__ T1* xLocal = (__local_mem__ T1*)xTensor[xUbOffset].GetPhyAddr();
            __local_mem__ float* meanLocal = (__local_mem__ float*)meanTensor[batchFactor * i].GetPhyAddr();
            __local_mem__ float* rstdLocal = (__local_mem__ float*)rstdTensor[batchFactor * i].GetPhyAddr();
            __local_mem__ T1* varianceOutLocal = (__local_mem__ T1*)varianceOutTensor[batchFactor * i].GetPhyAddr();
            if (i > 1) {
                WaitFlag<HardEvent::MTE3_V>(isPing ? eventIDMte3ToVPing : eventIDMte3ToVPong);
            }
            if (degenerateBatch) {
                __local_mem__ T1* yLocal = (__local_mem__ T1*)yTensor[xUbOffset].GetPhyAddr();
                DegenerateNormalizeBatch(xLocal, yLocal, meanLocal, varianceOutLocal, numPerCoreProcess, stageBase);
            } else if (useSingleChunk && !degeneratePath && slotsAligned) {
                // elemNum∈[2, VL_FP32]：统计拆分为逐组 mean/var + 批量化 Newton rstd，仿射沿用单块路径。
                CalMeanAndVarBatchedRstd(xLocal, meanLocal, rstdLocal, varianceOutLocal,
                                         static_cast<uint16_t>(numPerCoreProcess), elemNum, reduceScale, eps);
                NormalizeAndSwish(xUbOffset, offset + i * batchFactor, numPerCoreProcess, i);
            } else {
                CalMeanAndRstd<T1>(xLocal, meanLocal, rstdLocal, varianceOutLocal, dichotomyAddLocal, numPerCoreProcess,
                                   dichotomyAddPower, dichotomyAddK, dichotomyAddLastNum, elemNum, reduceScale, eps);
                NormalizeAndSwish(xUbOffset, offset + i * batchFactor, numPerCoreProcess, i);
            }
            if (i < numPerCoreExtent - 1) {
                SetFlag<HardEvent::V_MTE2>(eventIDVToMte2Span);
            }
            SetFlag<HardEvent::V_MTE3>(isPing ? eventIDVToMte3Ping : eventIDVToMte3Pong);
            WaitFlag<HardEvent::V_MTE3>(isPing ? eventIDVToMte3Ping : eventIDVToMte3Pong);
            if (i < numPerCoreExtent - BUFFER_NUM) {
                SetFlag<HardEvent::V_MTE2>(isPing ? eventIDVToMte2XPing : eventIDVToMte2XPong);
            }
            if (degenerateBatch) {
                CopyY2Gm<T1>(yGm[xGmOffset], yTensor[xUbOffset], 1, numPerCoreProcess);
            } else {
                CopyY2Gm<T1>(yGm[xGmOffset], yTensor[xUbOffset], numPerCoreProcess, elemNum);
            }
            if (i < numPerCoreExtent - BUFFER_NUM) {
                SetFlag<HardEvent::MTE3_V>(isPing ? eventIDMte3ToVPing : eventIDMte3ToVPong);
            }
        }
        GetTPipePtr()->ReleaseEventID<HardEvent::MTE2_V>(eventIDMte2ToVPing);
        GetTPipePtr()->ReleaseEventID<HardEvent::MTE2_V>(eventIDMte2ToVPong);
        GetTPipePtr()->ReleaseEventID<HardEvent::V_MTE3>(eventIDVToMte3Ping);
        GetTPipePtr()->ReleaseEventID<HardEvent::V_MTE3>(eventIDVToMte3Pong);
        GetTPipePtr()->ReleaseEventID<HardEvent::V_MTE2>(eventIDVToMte2Span);
        GetTPipePtr()->ReleaseEventID<HardEvent::V_MTE2>(eventIDVToMte2XPing);
        GetTPipePtr()->ReleaseEventID<HardEvent::V_MTE2>(eventIDVToMte2XPong);
        GetTPipePtr()->ReleaseEventID<HardEvent::MTE3_V>(eventIDMte3ToVPing);
        GetTPipePtr()->ReleaseEventID<HardEvent::MTE3_V>(eventIDMte3ToVPong);
    }

    __aicore__ inline void CopyGammaBetaSpan2UB(uint32_t numPerCoreoffset, int64_t numPerCoreProcess)
    {
        // 按批一次拷贝γ/β连续通道段；越过组-通道回卷点（shapeC）时拆两段，段2目的基址 32B 对齐。
        uint64_t groupStart = blockIdx * tiling->numPerCore + numPerCoreoffset;
        uint64_t chanStart = (groupStart % numGroups) * shapeD;
        uint64_t spanLen = numPerCoreProcess * shapeD;
        if (chanStart + spanLen <= shapeC) {
            CopyGammaAndBeta2UB<T2>(gammaGm[chanStart], betaGm[chanStart], gammaTensor, betaTensor, 1, spanLen);
            spanSeg1Len = spanLen;
            spanSeg2Offset = 0;
            return;
        }
        uint64_t seg1Len = shapeC - chanStart;
        uint64_t seg2Len = spanLen - seg1Len;
        CopyGammaAndBeta2UB<T2>(gammaGm[chanStart], betaGm[chanStart], gammaTensor, betaTensor, 1, seg1Len);
        int64_t seg2Offset = RoundUp<T2>(seg1Len);
        CopyGammaAndBeta2UB<T2>(gammaGm[0], betaGm[0], gammaTensor[seg2Offset], betaTensor[seg2Offset], 1, seg2Len);
        spanSeg1Len = seg1Len;
        spanSeg2Offset = seg2Offset;
    }

    __aicore__ inline void CopyGammaBetaStagedSpan2UB(int64_t numPerCoreProcess, uint32_t stageBase)
    {
        // 退化两段路径的段2 γ/β 暂存：拷入对侧 x ping/pong 缓冲，leftPadding 补偿相位使装载 32B 对齐；
        // 暂存区仅本批 VF 阶段读取，重叠头的填充值由随后执行的段1以正确 γ/β 重写。
        uint32_t alignElem = BLOCK_SIZE / sizeof(T2);
        uint32_t seg1Len = static_cast<uint32_t>(spanSeg1Len);
        uint32_t phi = seg1Len % alignElem;
        uint32_t seg2Len = static_cast<uint32_t>(numPerCoreProcess) - seg1Len;
        uint32_t copyLenAlign = RoundUp<T2>(seg2Len + phi);
        DataCopyPadExtParams<T2> padParams;
        padParams.isPad = true;
        padParams.paddingValue = static_cast<T2>(0.0);
        padParams.leftPadding = static_cast<uint8_t>(phi);
        padParams.rightPadding = copyLenAlign - seg2Len - phi;
        DataCopyExtParams dataCopyParams;
        dataCopyParams.blockCount = 1;
        dataCopyParams.blockLen = seg2Len * sizeof(T2);
        dataCopyParams.srcStride = 0;
        dataCopyParams.dstStride = 0;
        LocalTensor<T2> gammaStaged = xTensor[stageBase].template ReinterpretCast<T2>();
        LocalTensor<T2> betaStaged = xTensor[stageBase + copyLenAlign].template ReinterpretCast<T2>();
        DataCopyPad(gammaStaged, gammaGm[0], dataCopyParams, padParams);
        DataCopyPad(betaStaged, betaGm[0], dataCopyParams, padParams);
    }

    __aicore__ inline void DegenerateNormalizeSegment(__local_mem__ T1* xLocal, __local_mem__ T1* yLocal,
                                                      __local_mem__ float* meanLocal,
                                                      __local_mem__ T1* varianceOutLocal, __local_mem__ T2* gammaSrc,
                                                      __local_mem__ T2* betaSrc, uint32_t segStart, uint32_t segEnd)
    {
        // 段内按 64 组块全向量化：块内组间独立、无归约耦合，逐元素算子序列与基线逐组路径一致（bit-exact）；
        // x/mean/variance/y 用批内自然组索引，γ/β 用段相对偏移。
        __VEC_SCOPE__
        {
            RegTensor<float> x;
            RegTensor<float> d;
            RegTensor<float> mean;
            RegTensor<float> var;
            RegTensor<float> rstd;
            RegTensor<float> gamma;
            RegTensor<float> beta;
            MaskReg pregLoop;
            uint32_t segLen = segEnd - segStart;
            uint16_t blockCount = static_cast<uint16_t>(CeilDiv(segLen, VL_FP32));
            uint32_t sreg0 = segLen;
            for (uint16_t blk = 0; blk < blockCount; blk++) {
                uint32_t g0 = segStart + static_cast<uint32_t>(blk) * VL_FP32;
                uint32_t gammaOff = static_cast<uint32_t>(blk) * VL_FP32;
                pregLoop = UpdateMask<float>(sreg0);
                LoadInputData<T1>(x, xLocal, pregLoop, g0);
                LoadInputData<T2>(gamma, gammaSrc, pregLoop, gammaOff);
                LoadInputData<T2>(beta, betaSrc, pregLoop, gammaOff);
                Muls(mean, x, reduceScale, pregLoop);
                Sub(d, x, mean, pregLoop);
                Mul(var, d, d, pregLoop);
                Muls(var, var, reduceScale, pregLoop);
                DataCopy<float>(meanLocal + g0, mean, pregLoop);
                StoreOutputData<T1>(varianceOutLocal, var, pregLoop, g0);
                NormCommon::ComputeRstdNewtonRaphsonReg<false>(var, rstd, pregLoop, eps);
                Mul(d, d, rstd, pregLoop);
                Mul(d, d, gamma, pregLoop);
                Add(d, d, beta, pregLoop);
                StoreOutputData<T1>(yLocal, d, pregLoop, g0);
            }
        }
    }

    __aicore__ inline void DegenerateNormalizeBatch(__local_mem__ T1* xLocal, __local_mem__ T1* yLocal,
                                                    __local_mem__ float* meanLocal, __local_mem__ T1* varianceOutLocal,
                                                    uint32_t numPerCoreProcess, uint32_t stageBase)
    {
        // elemNum==1 批内统计+仿射全向量化。γ/β 跨度回卷时拆两段：段2（staged γ/β）先行、段1随后重写
        // 重叠区的 y（mean/variance 与 γ 无关），VEC 顺序执行保证终值正确。
        __local_mem__ T2* gammaLocal = (__local_mem__ T2*)gammaTensor.GetPhyAddr();
        __local_mem__ T2* betaLocal = (__local_mem__ T2*)betaTensor.GetPhyAddr();
        if (spanSeg1Len >= static_cast<int64_t>(numPerCoreProcess)) {
            DegenerateNormalizeSegment(xLocal, yLocal, meanLocal, varianceOutLocal, gammaLocal, betaLocal, 0,
                                       numPerCoreProcess);
            return;
        }
        uint32_t alignElem = BLOCK_SIZE / sizeof(T2);
        uint32_t seg1Len = static_cast<uint32_t>(spanSeg1Len);
        uint32_t segStart = seg1Len - seg1Len % alignElem;
        uint32_t stageStride = RoundUp<T2>(numPerCoreProcess - seg1Len + seg1Len % alignElem);
        LocalTensor<T2> gammaStaged = xTensor[stageBase].template ReinterpretCast<T2>();
        LocalTensor<T2> betaStaged = xTensor[stageBase + stageStride].template ReinterpretCast<T2>();
        DegenerateNormalizeSegment(xLocal, yLocal, meanLocal, varianceOutLocal,
                                   (__local_mem__ T2*)gammaStaged.GetPhyAddr(),
                                   (__local_mem__ T2*)betaStaged.GetPhyAddr(), segStart, numPerCoreProcess);
        DegenerateNormalizeSegment(xLocal, yLocal, meanLocal, varianceOutLocal, gammaLocal, betaLocal, 0, seg1Len);
    }

    __aicore__ inline void CalMeanAndVarBatchedRstd(__local_mem__ T1* xLocal, __local_mem__ float* meanLocal,
                                                    __local_mem__ float* rstdLocal, __local_mem__ T1* varianceOutLocal,
                                                    uint16_t numPerCoreProcess, uint64_t reduceCount, float scale,
                                                    float eps)
    {
        // elemNum∈[2, VL_FP32] 统计拆分：循环A 逐组计算 mean/var（×2 展开交错，var 暂存 rstdLocal 槽位）；
        // 第二个 __VEC_SCOPE__ 批量化 Newton rstd（scope 边界保证 store→load 保序，无需显式同步）。
        uint32_t elemNumAlign = RoundUp<T1>(reduceCount);
        __VEC_SCOPE__
        {
            RegTensor<float> x0;
            RegTensor<float> x1;
            RegTensor<float> xScale0;
            RegTensor<float> xScale1;
            RegTensor<float> mean0;
            RegTensor<float> mean1;
            RegTensor<float> var0;
            RegTensor<float> var1;
            RegTensor<float> rstd;
            RegTensor<float> one;
            MaskReg pregLoop;
            MaskReg pregMain = CreateMask<float, AscendC::Reg::MaskPattern::ALL>();
            MaskReg pregMerge = CreateMask<float, AscendC::Reg::MaskPattern::VL1>();
            Duplicate(one, float(1.0), pregMain);
            uint16_t pairCount = numPerCoreProcess / 2;
            for (uint16_t p = 0; p < pairCount; p++) {
                uint32_t sreg0 = reduceCount;
                pregLoop = UpdateMask<float>(sreg0);
                uint32_t g0 = static_cast<uint32_t>(p) * 2;
                LoadInputData<T1>(x0, xLocal, pregLoop, g0 * elemNumAlign);
                Muls(xScale0, x0, scale, pregLoop);
                ReduceSum(mean0, xScale0, pregLoop);
                DataCopy<float, AscendC::Reg::StoreDist::DIST_FIRST_ELEMENT_B32>(meanLocal + g0, mean0, pregMerge);
                Duplicate(mean0, mean0, pregMain);
                Sub(x0, x0, mean0, pregLoop);
                Mul(x0, x0, x0, pregLoop);
                Muls(xScale0, x0, scale, pregLoop);
                ReduceSum(var0, xScale0, pregLoop);
                StoreStatisticData<T1>(varianceOutLocal, var0, pregMerge, g0);
                DataCopy<float, AscendC::Reg::StoreDist::DIST_FIRST_ELEMENT_B32>(rstdLocal + g0, var0, pregMerge);
                LoadInputData<T1>(x1, xLocal, pregLoop, (g0 + 1) * elemNumAlign);
                Muls(xScale1, x1, scale, pregLoop);
                ReduceSum(mean1, xScale1, pregLoop);
                DataCopy<float, AscendC::Reg::StoreDist::DIST_FIRST_ELEMENT_B32>(meanLocal + g0 + 1, mean1, pregMerge);
                Duplicate(mean1, mean1, pregMain);
                Sub(x1, x1, mean1, pregLoop);
                Mul(x1, x1, x1, pregLoop);
                Muls(xScale1, x1, scale, pregLoop);
                ReduceSum(var1, xScale1, pregLoop);
                StoreStatisticData<T1>(varianceOutLocal, var1, pregMerge, g0 + 1);
                DataCopy<float, AscendC::Reg::StoreDist::DIST_FIRST_ELEMENT_B32>(rstdLocal + g0 + 1, var1, pregMerge);
            }
            if (numPerCoreProcess % 2 == 1) {
                uint32_t sreg0 = reduceCount;
                pregLoop = UpdateMask<float>(sreg0);
                uint32_t gTail = numPerCoreProcess - 1;
                LoadInputData<T1>(x0, xLocal, pregLoop, gTail * elemNumAlign);
                Muls(xScale0, x0, scale, pregLoop);
                ReduceSum(mean0, xScale0, pregLoop);
                DataCopy<float, AscendC::Reg::StoreDist::DIST_FIRST_ELEMENT_B32>(meanLocal + gTail, mean0, pregMerge);
                Duplicate(mean0, mean0, pregMain);
                Sub(x0, x0, mean0, pregLoop);
                Mul(x0, x0, x0, pregLoop);
                Muls(xScale0, x0, scale, pregLoop);
                ReduceSum(var0, xScale0, pregLoop);
                StoreStatisticData<T1>(varianceOutLocal, var0, pregMerge, gTail);
                DataCopy<float, AscendC::Reg::StoreDist::DIST_FIRST_ELEMENT_B32>(rstdLocal + gTail, var0, pregMerge);
            }
        }
        __VEC_SCOPE__
        {
            RegTensor<float> var2;
            RegTensor<float> rstd2;
            MaskReg pregLoop;
            MaskReg pregMain = CreateMask<float, AscendC::Reg::MaskPattern::ALL>();
            uint32_t fullBlocks = numPerCoreProcess / VL_FP32;
            uint32_t tailCount = numPerCoreProcess % VL_FP32;
            for (uint16_t blk = 0; blk < static_cast<uint16_t>(fullBlocks); blk++) {
                DataCopy(var2, rstdLocal + static_cast<uint32_t>(blk) * VL_FP32);
                NormCommon::ComputeRstdNewtonRaphsonReg<false>(var2, rstd2, pregMain, eps);
                DataCopy(rstdLocal + static_cast<uint32_t>(blk) * VL_FP32, rstd2, pregMain);
            }
            if (tailCount > 0) {
                uint32_t sreg0 = tailCount;
                pregLoop = UpdateMask<float>(sreg0);
                DataCopy(var2, rstdLocal + fullBlocks * VL_FP32);
                RegTensor<float> safeVar;
                Duplicate(safeVar, float(1.0), pregMain);
                Select(var2, var2, safeVar, pregLoop);
                NormCommon::ComputeRstdNewtonRaphsonReg<false>(var2, rstd2, pregLoop, eps);
                DataCopy(rstdLocal + fullBlocks * VL_FP32, rstd2, pregLoop);
            }
        }
    }

    __aicore__ inline void NormalizeGroupsSingleChunk(__local_mem__ T1* xLocal, __local_mem__ T1* yLocal,
                                                      __local_mem__ float* meanLocal, __local_mem__ float* rstdLocal,
                                                      __local_mem__ T2* gammaLocal, __local_mem__ T2* betaLocal,
                                                      int64_t numPerCoreProcess)
    {
        // elemNum <= VL：每组单个masked向量块；γ/β由逐行DIST_BRC广播 + MERGING级联拼出组内分段常量向量，
        // x/y装载与存储均为32B对齐基址（组基址为elemNumAlign的倍数），无UnAlign机制。
        __VEC_SCOPE__
        {
            RegTensor<float> x;
            RegTensor<float> y;
            RegTensor<float> gamma;
            RegTensor<float> beta;
            RegTensor<float> gammaTmp;
            RegTensor<float> betaTmp;
            RegTensor<float> mean;
            RegTensor<float> rstd;
            MaskReg pregLoop;
            MaskReg pregMain = CreateMask<float, AscendC::Reg::MaskPattern::ALL>();
            for (uint16_t i = 0; i < static_cast<uint16_t>(numPerCoreProcess); i++) {
                uint32_t sreg0 = elemNum;
                uint64_t gammaBase = i * shapeD;
                uint64_t gammaSeg1 = static_cast<uint64_t>(spanSeg1Len);
                uint64_t gammaWrap = static_cast<uint64_t>(spanSeg2Offset) - gammaSeg1;
                uint64_t gammaIdx = gammaBase + static_cast<uint64_t>(gammaBase >= gammaSeg1) * gammaWrap;
                DataCopy<float, AscendC::Reg::LoadDist::DIST_BRC_B32>(mean, meanLocal + i);
                DataCopy<float, AscendC::Reg::LoadDist::DIST_BRC_B32>(rstd, rstdLocal + i);
                LoadGammaAndBetaData<T2>(gamma, beta, gammaLocal, betaLocal, pregMain,
                                         static_cast<uint32_t>(gammaIdx + shapeD - 1));
                for (uint16_t r = 0; r < static_cast<uint16_t>(shapeD - 1); r++) {
                    int64_t d = shapeD - 2 - r;
                    uint32_t sregD = static_cast<uint32_t>(d + 1) * static_cast<uint32_t>(hwNum);
                    pregLoop = UpdateMask<float>(sregD);
                    LoadGammaAndBetaData<T2>(gammaTmp, betaTmp, gammaLocal, betaLocal, pregMain,
                                             static_cast<uint32_t>(gammaIdx + d));
                    Copy<float, AscendC::Reg::MaskMergeMode::MERGING>(gamma, gammaTmp, pregLoop);
                    Copy<float, AscendC::Reg::MaskMergeMode::MERGING>(beta, betaTmp, pregLoop);
                }
                pregLoop = UpdateMask<float>(sreg0);
                LoadInputData<T1>(x, xLocal, pregLoop, static_cast<uint32_t>(i) * elemNumAlign);
                VFInnerNormalize(x, mean, rstd, gamma, beta, y, pregLoop);
                StoreOutputData<T1>(yLocal, y, pregLoop, static_cast<uint32_t>(i) * elemNumAlign);
            }
        }
    }

    __aicore__ inline void NormalizeGroupsRowAligned(__local_mem__ T1* xLocal, __local_mem__ T1* yLocal,
                                                     __local_mem__ float* meanLocal, __local_mem__ float* rstdLocal,
                                                     __local_mem__ T2* gammaLocal, __local_mem__ T2* betaLocal,
                                                     int64_t numPerCoreProcess)
    {
        // elemNum > VL 且 hwNum*sizeof(T1)为32B倍数：行基址天然对齐，逐行masked块 + 每行γ/β广播。
        int64_t rowChunks = CeilDiv(hwNum, VL_FP32);
        __VEC_SCOPE__
        {
            RegTensor<float> x;
            RegTensor<float> y;
            RegTensor<float> gamma;
            RegTensor<float> beta;
            RegTensor<float> mean;
            RegTensor<float> rstd;
            MaskReg pregLoop;
            MaskReg pregMain = CreateMask<float, AscendC::Reg::MaskPattern::ALL>();
            for (uint16_t i = 0; i < static_cast<uint16_t>(numPerCoreProcess); i++) {
                uint64_t gammaBase = i * shapeD;
                uint64_t gammaSeg1 = static_cast<uint64_t>(spanSeg1Len);
                uint64_t gammaWrap = static_cast<uint64_t>(spanSeg2Offset) - gammaSeg1;
                uint64_t gammaIdx = gammaBase + static_cast<uint64_t>(gammaBase >= gammaSeg1) * gammaWrap;
                DataCopy<float, AscendC::Reg::LoadDist::DIST_BRC_B32>(mean, meanLocal + i);
                DataCopy<float, AscendC::Reg::LoadDist::DIST_BRC_B32>(rstd, rstdLocal + i);
                for (uint16_t d = 0; d < static_cast<uint16_t>(shapeD); d++) {
                    LoadGammaAndBetaData<T2>(gamma, beta, gammaLocal, betaLocal, pregMain,
                                             static_cast<uint32_t>(gammaIdx + d));
                    for (uint16_t c = 0; c < static_cast<uint16_t>(rowChunks); c++) {
                        uint32_t sreg0 = static_cast<uint32_t>(hwNum) - static_cast<uint32_t>(c) * VL_FP32;
                        pregLoop = UpdateMask<float>(sreg0);
                        uint32_t offset = static_cast<uint32_t>(i) * elemNumAlign +
                                          static_cast<uint32_t>(d) * static_cast<uint32_t>(hwNum) +
                                          static_cast<uint32_t>(c) * VL_FP32;
                        LoadInputData<T1>(x, xLocal, pregLoop, offset);
                        VFInnerNormalize(x, mean, rstd, gamma, beta, y, pregLoop);
                        StoreOutputData<T1>(yLocal, y, pregLoop, offset);
                    }
                }
            }
        }
    }

    __aicore__ inline void NormalizeGroupsChunkCascade(__local_mem__ T1* xLocal, __local_mem__ T1* yLocal,
                                                       __local_mem__ float* meanLocal, __local_mem__ float* rstdLocal,
                                                       __local_mem__ T2* gammaLocal, __local_mem__ T2* betaLocal,
                                                       int64_t numPerCoreProcess)
    {
        // 通用回退（elemNum > VL 且行未32B对齐）：按64元素块从packed x/y装载/存储（块基址必为32B对齐），
        // 每块的γ/β分段常量向量用倒序前缀MERGING级联拼装（前缀边界与行边界d*hwNum对齐），
        // 无UnAlign机制、无staging、无额外GM读。
        int64_t chunkCount = CeilDiv(elemNum, VL_FP32);
        __VEC_SCOPE__
        {
            RegTensor<float> x;
            RegTensor<float> y;
            RegTensor<float> gamma;
            RegTensor<float> beta;
            RegTensor<float> gammaTmp;
            RegTensor<float> betaTmp;
            RegTensor<float> mean;
            RegTensor<float> rstd;
            MaskReg pregLoop;
            MaskReg pregMain = CreateMask<float, AscendC::Reg::MaskPattern::ALL>();
            for (uint16_t i = 0; i < static_cast<uint16_t>(numPerCoreProcess); i++) {
                uint64_t gammaBase = i * shapeD;
                uint64_t gammaSeg1 = static_cast<uint64_t>(spanSeg1Len);
                uint64_t gammaWrap = static_cast<uint64_t>(spanSeg2Offset) - gammaSeg1;
                uint64_t gammaIdx = gammaBase + static_cast<uint64_t>(gammaBase >= gammaSeg1) * gammaWrap;
                DataCopy<float, AscendC::Reg::LoadDist::DIST_BRC_B32>(mean, meanLocal + i);
                DataCopy<float, AscendC::Reg::LoadDist::DIST_BRC_B32>(rstd, rstdLocal + i);
                for (uint16_t c = 0; c < static_cast<uint16_t>(chunkCount); c++) {
                    uint32_t chunkStart = static_cast<uint32_t>(c) * VL_FP32;
                    uint32_t rowFirst = chunkStart / static_cast<uint32_t>(hwNum);
                    uint32_t rowLast = (chunkStart + VL_FP32 - 1) / static_cast<uint32_t>(hwNum);
                    if (rowLast > static_cast<uint32_t>(shapeD - 1)) {
                        rowLast = static_cast<uint32_t>(shapeD - 1);
                    }
                    LoadGammaAndBetaData<T2>(gamma, beta, gammaLocal, betaLocal, pregMain,
                                             static_cast<uint32_t>(gammaIdx + rowLast));
                    for (uint16_t r = 0; r < static_cast<uint16_t>(rowLast - rowFirst); r++) {
                        uint32_t d = rowLast - 1 - r;
                        uint32_t sregD = (d + 1) * static_cast<uint32_t>(hwNum) - chunkStart;
                        pregLoop = UpdateMask<float>(sregD);
                        LoadGammaAndBetaData<T2>(gammaTmp, betaTmp, gammaLocal, betaLocal, pregMain,
                                                 static_cast<uint32_t>(gammaIdx + d));
                        Copy<float, AscendC::Reg::MaskMergeMode::MERGING>(gamma, gammaTmp, pregLoop);
                        Copy<float, AscendC::Reg::MaskMergeMode::MERGING>(beta, betaTmp, pregLoop);
                    }
                    uint32_t sreg0 = static_cast<uint32_t>(elemNum) - chunkStart;
                    pregLoop = UpdateMask<float>(sreg0);
                    uint32_t offset = static_cast<uint32_t>(i) * elemNumAlign + chunkStart;
                    LoadInputData<T1>(x, xLocal, pregLoop, offset);
                    VFInnerNormalize(x, mean, rstd, gamma, beta, y, pregLoop);
                    StoreOutputData<T1>(yLocal, y, pregLoop, offset);
                }
            }
        }
    }

    __aicore__ inline void NormalizeAndSwish(uint32_t xUbOffset, uint32_t numPerCoreoffset, int64_t numPerCoreProcess,
                                             uint32_t numPerCoreLoop)
    {
        // 批内组循环为纯VF执行：无逐组MTE2搬运、无逐组SetFlag/WaitFlag、无UnAlign机制。
        __local_mem__ T1* xLocal = (__local_mem__ T1*)xTensor[xUbOffset].GetPhyAddr();
        __local_mem__ T1* yLocal = (__local_mem__ T1*)yTensor[xUbOffset].GetPhyAddr();
        __local_mem__ float* meanLocal = (__local_mem__ float*)meanTensor[batchFactor * numPerCoreLoop].GetPhyAddr();
        __local_mem__ float* rstdLocal = (__local_mem__ float*)rstdTensor[batchFactor * numPerCoreLoop].GetPhyAddr();
        __local_mem__ T2* gammaLocal = (__local_mem__ T2*)gammaTensor.GetPhyAddr();
        __local_mem__ T2* betaLocal = (__local_mem__ T2*)betaTensor.GetPhyAddr();
        if (useSingleChunk) {
            NormalizeGroupsSingleChunk(xLocal, yLocal, meanLocal, rstdLocal, gammaLocal, betaLocal, numPerCoreProcess);
        } else if (rowsAligned) {
            NormalizeGroupsRowAligned(xLocal, yLocal, meanLocal, rstdLocal, gammaLocal, betaLocal, numPerCoreProcess);
        } else {
            NormalizeGroupsChunkCascade(xLocal, yLocal, meanLocal, rstdLocal, gammaLocal, betaLocal, numPerCoreProcess);
        }
    }

    __aicore__ inline void ParseTilingData()
    {
        // 解析当前核的切分参数和归约参数；批因子kernel侧由现有tiling字段推导（≤onceNumPerCore且≤numGroups）。
        if (blockIdx == blockNum - 1) {
            numPerCore = tiling->numLastCore;
        } else {
            numPerCore = tiling->numPerCore;
        }
        numGroups = tiling->numGroups;
        ubSize = tiling->ubSize;
        elemNum = tiling->elemNum;
        shapeC = tiling->shapeC;
        shapeD = tiling->shapeD;
        eps = tiling->epsilon;
        hwNum = tiling->hwNum;
        processSize = tiling->processSize;
        elemNumAlign = RoundUp<T1>(elemNum);
        hwNumAlign = RoundUp<T1>(hwNum);
        onceNumPerCore = processSize / elemNumAlign;
        batchFactor = onceNumPerCore < numGroups ? onceNumPerCore : numGroups;
        int64_t maxBatchTmp = batchFactor < numPerCore ? batchFactor : numPerCore;
        maxBatch = maxBatchTmp < innerNumPerCore ? maxBatchTmp : innerNumPerCore;
        useSingleChunk = elemNum <= VL_FP32;
        rowsAligned = (hwNum * sizeof(T1)) % BLOCK_SIZE == 0;
        degeneratePath = (shapeD == 1) && (hwNum == 1); // ⟺ elemNum == 1（现有 tiling 字段推导）
        spanSeg1Len = 0;
        spanSeg2Offset = 0;
        reduceScale = (float)1.0 / static_cast<float>(elemNum);
        dichotomyAddPower = tiling->dichotomyAddPower;
        dichotomyAddK = tiling->dichotomyAddK;
        dichotomyAddLastNum = tiling->dichotomyAddLastNum;
    }

    __aicore__ inline void InitInnerBuffer()
    {
        // 单块UB划分输入、输出、统计与γ/β跨度空间；批因子kernel侧推导，UB不足时按2收缩，
        // 收缩到1即不超过基线布局，不会溢出。
        pipe.InitBuffer(innerBuf, ubSize);
        LocalTensor<T1> ubTensor = innerBuf.template Get<T1>();
        int32_t realNumPerCore = numPerCore > innerNumPerCore ? innerNumPerCore : numPerCore;
        int32_t meanSize = RoundUp<T1>(realNumPerCore * (FLOAT_BYTE_SIZE / sizeof(T1)));
        int32_t rstdSize = RoundUp<T1>(realNumPerCore * (FLOAT_BYTE_SIZE / sizeof(T1)));
        int32_t dichotomySize = RoundUp<T1>((dichotomyAddPower / FP32_ONE_REPEAT) * (FLOAT_BYTE_SIZE / sizeof(T1)));
        int32_t meanOutSize = 0;
        int32_t varianceOutSize = RoundUp<T1>(realNumPerCore);
        if constexpr (IsSameType<T1, half>::value || IsSameType<T1, bfloat16_t>::value) {
            meanOutSize = RoundUp<T1>(realNumPerCore);
        }
        int32_t xSize = 0;
        int32_t ySize = 0;
        int32_t spanSize = 0;
        while (true) {
            xSize = maxBatch * elemNumAlign * BUFFER_NUM;
            ySize = maxBatch * elemNumAlign * BUFFER_NUM;
            int64_t spanSlack = batchFactor > 1 ? 2 * (BLOCK_SIZE / sizeof(T2)) : 0;
            spanSize = RoundUp<T1>((maxBatch * shapeD + spanSlack) * (sizeof(T2) / sizeof(T1)));
            int64_t totalBytes = (static_cast<int64_t>(xSize) + ySize + meanSize + rstdSize + dichotomySize +
                                  meanOutSize + varianceOutSize) *
                                 sizeof(T1);
            if (hasGamma) {
                totalBytes += static_cast<int64_t>(spanSize) * sizeof(T1);
            }
            if (hasBeta) {
                totalBytes += static_cast<int64_t>(spanSize) * sizeof(T1);
            }
            if (totalBytes <= ubSize || batchFactor <= 1) {
                break;
            }
            batchFactor /= 2;
            int64_t maxBatchTmp = batchFactor < numPerCore ? batchFactor : numPerCore;
            maxBatch = maxBatchTmp < innerNumPerCore ? maxBatchTmp : innerNumPerCore;
        }

        int32_t yOffset = xSize;
        int32_t meanOffset = yOffset + ySize;
        int32_t rstdOffset = meanOffset + meanSize;
        int32_t dichotomyAddOffset = rstdOffset + rstdSize;

        xTensor = ubTensor;
        yTensor = ubTensor[yOffset];
        meanTensor = ubTensor[meanOffset].template ReinterpretCast<float>();
        rstdTensor = ubTensor[rstdOffset].template ReinterpretCast<float>();
        dichotomyAddTensor = ubTensor[dichotomyAddOffset].template ReinterpretCast<float>();
        int32_t curOffset = dichotomyAddOffset + dichotomySize;

        if constexpr (IsSameType<T1, half>::value || IsSameType<T1, bfloat16_t>::value) {
            meanOutTensor = ubTensor[curOffset];
            curOffset += meanOutSize;
        }
        varianceOutTensor = ubTensor[curOffset];
        curOffset += varianceOutSize;

        if (hasGamma) {
            gammaTensor = ubTensor[curOffset].template ReinterpretCast<T2>();
            curOffset += spanSize;
        }
        if (hasBeta) {
            betaTensor = ubTensor[curOffset].template ReinterpretCast<T2>();
            curOffset += spanSize;
        }
    }

private:
    const GroupNormTilingData* tiling;
    TPipe pipe;

    GlobalTensor<T1> xGm;
    GlobalTensor<T2> gammaGm;
    GlobalTensor<T2> betaGm;

    GlobalTensor<T1> yGm;
    GlobalTensor<T1> meanGm;
    GlobalTensor<T1> varianceGm;

    TBuf<> innerBuf;

    int64_t blockIdx;
    int64_t blockNum;
    int64_t blockElemNum;
    bool hasGamma{false};
    bool hasBeta{false};
    int64_t numPerCore;
    int64_t elemNum;
    int64_t elemNumAlign;
    int64_t ubSize;
    int64_t shapeC;
    int64_t shapeD;
    int64_t ubElemNum;
    int64_t hwNum;
    int64_t hwNumAlign;
    int64_t processSize;
    int64_t numGroups;
    int64_t innerNumPerCore{MAX_ONCE_NUM_PER_CORE};
    int64_t onceNumPerCore;
    int64_t batchFactor;    // 每批组数（UB容量不足时收缩）
    int64_t maxBatch;       // 单次内层迭代最大批数
    int64_t spanSeg1Len;    // γ/β跨度回卷时段1长度
    int64_t spanSeg2Offset; // γ/β跨度回卷时段2目的偏移
    bool useSingleChunk{false};
    bool rowsAligned{false};
    bool degeneratePath{false};
    int64_t dichotomyAddPower;
    int64_t dichotomyAddK;
    int64_t dichotomyAddLastNum;
    float eps;
    float reduceScale;

    LocalTensor<T1> xTensor;
    LocalTensor<T2> gammaTensor;
    LocalTensor<T2> betaTensor;
    LocalTensor<float> meanTensor;
    LocalTensor<float> rstdTensor;
    LocalTensor<float> dichotomyAddTensor;
    LocalTensor<T1> meanOutTensor;
    LocalTensor<T1> varianceOutTensor;

    LocalTensor<T1> yTensor;
};

} // namespace GroupNorm

#endif

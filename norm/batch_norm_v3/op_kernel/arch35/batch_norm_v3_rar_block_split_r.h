/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file batch_norm_v3_rar_block_split_r.h
 * \brief
 */
#ifndef BATCH_NORM_V3_RAR_BLOCK_SPLIT_R_REGBASE_H
#define BATCH_NORM_V3_RAR_BLOCK_SPLIT_R_REGBASE_H

#include "kernel_tiling/kernel_tiling.h"
#include "kernel_operator.h"
#include "op_kernel/platform_util.h"
#include "../../norm_common/reduce_common_regbase.h"
#include "batch_norm_v3_regbase_common.h"

namespace BatchNormV3Ops {
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

template <typename T, typename T_GAMMA, typename T_RUNNING_MEAN>
class BatchNormV3RARBlockSplitR {
    static constexpr int32_t UB_SPLIT_AXIS_R1 = 0;
    static constexpr int32_t UB_SPLIT_AXIS_A = 1;
    static constexpr int32_t UB_SPLIT_AXIS_R0 = 2;

public:
    __aicore__ inline uint64_t CEIL_DIV(uint64_t x, uint64_t y) { return (y != 0) ? (x + y - 1) / y : 0; }

    __aicore__ inline uint64_t CEIL_ALIGN(uint64_t x, uint64_t y) { return CEIL_DIV(x, y) * y; }

    __aicore__ inline BatchNormV3RARBlockSplitR(const BatchNormV3RARBlockSplitRTilingData* tilingDataIn)
    {
        tilingData = tilingDataIn;
        uint64_t reduceNum = tilingData->patternR1 * tilingData->patternR0;
        this->unbiasedEstimationCoeff = reduceNum == 1 ?
                                            AscendC::NumericLimits<float>::QuietNaN() :
                                            static_cast<float>(reduceNum) / static_cast<float>(reduceNum - 1);
    }

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR gamma, GM_ADDR beta, GM_ADDR mean, GM_ADDR var, GM_ADDR y,
                                GM_ADDR mean_out, GM_ADDR var_out, GM_ADDR batch_mean, GM_ADDR batch_rstd,
                                GM_ADDR workspace)
    {
        blockIdx = GetBlockIdx();
        usedCoreNum = GetBlockNum();

        int64_t xyGmOffset = 0;

        uint64_t r0LoopIdx = 1;
        r1EndIdx = tilingData->patternR1;
        r0EndIdx = tilingData->patternR0;
        r1BlockInner = 1;
        r0BlockInner = 1;
        if (tilingData->blockSplitAxis == 0) {
            // block切分轴是R1
            if (blockIdx < tilingData->formerBlockOuter) {
                r1BlockInner = tilingData->blockInner;
                r1StartIdx = blockIdx * r1BlockInner;
                r1EndIdx = r1StartIdx + r1BlockInner;

                ubSplitAxis = tilingData->formerCoreUbSplitAxis;
                ubOuter = tilingData->formerCoreUbOuter;
                ubInner = tilingData->formerCoreUbInner;
                rFlodFactor = tilingData->formerCoreBinaryAddQuotient; // binaryAddQuotient: rUbFactor向下2的幂次
            } else {
                r1BlockInner = tilingData->blockInner - 1;
                r1StartIdx = tilingData->formerBlockOuter * tilingData->blockInner +
                             (blockIdx - tilingData->formerBlockOuter) * r1BlockInner;
                r1EndIdx = r1StartIdx + r1BlockInner;

                ubSplitAxis = tilingData->tailCoreUbSplitAxis;
                ubOuter = tilingData->tailCoreUbOuter;
                ubInner = tilingData->tailCoreUbInner;
                rFlodFactor = tilingData->tailCoreBinaryAddQuotient; // binaryAddQuotient: rUbFactor向下2的幂次
            }
            r0BlockInner = tilingData->patternR0;
            xyGmOffset = r1StartIdx * tilingData->patternA * tilingData->patternR0;
        } else {
            // block切分轴是R0
            r1BlockInner = 1;
            r1StartIdx = blockIdx % tilingData->patternR1;
            r1EndIdx = r1StartIdx + 1;
            r0LoopIdx = blockIdx / tilingData->patternR1;

            if (r0LoopIdx < tilingData->formerBlockOuter) {
                r0BlockInner = tilingData->blockInner;
                r0StartIdx = r0LoopIdx * r0BlockInner;
                r0EndIdx = r0StartIdx + r0BlockInner;

                ubSplitAxis = tilingData->formerCoreUbSplitAxis;
                ubOuter = tilingData->formerCoreUbOuter;
                ubInner = tilingData->formerCoreUbInner;
                rFlodFactor = tilingData->formerCoreBinaryAddQuotient; // binaryAddQuotient: rUbFactor向下2的幂次
            } else {
                r0BlockInner = tilingData->blockInner - 1;
                r0StartIdx = tilingData->formerBlockOuter * tilingData->blockInner +
                             (r0LoopIdx - tilingData->formerBlockOuter) * r0BlockInner;
                r0EndIdx = r0StartIdx + r0BlockInner;

                ubSplitAxis = tilingData->tailCoreUbSplitAxis;
                ubOuter = tilingData->tailCoreUbOuter;
                ubInner = tilingData->tailCoreUbInner;
                rFlodFactor = tilingData->tailCoreBinaryAddQuotient; // binaryAddQuotient: rUbFactor向下2的幂次
            }
            xyGmOffset = r1StartIdx * tilingData->patternA * tilingData->patternR0 + r0StartIdx;
        }

        // 计算 allcount
        uint32_t nowCoreRConut = r1BlockInner * r0BlockInner;
        uint32_t nowCoreRConutPowOfTwo = BatchNormV3FindCofFactor(nowCoreRConut);
        uint32_t rPowOfTwo = BatchNormV3FindCofFactor(tilingData->patternR1 * tilingData->patternR0);
        this->nFactor = static_cast<float>(1) / static_cast<float>(nowCoreRConutPowOfTwo);
        this->nCorrectionFactor = static_cast<float>(nowCoreRConutPowOfTwo) / static_cast<float>(nowCoreRConut);
        this->lastNFactor = static_cast<float>(1) / static_cast<float>(rPowOfTwo);
        this->lastNCorrectionFactor = static_cast<float>(rPowOfTwo) /
                                      static_cast<float>(tilingData->patternR1 * tilingData->patternR0);

        xGm.SetGlobalBuffer((__gm__ T*)x + xyGmOffset);
        betaGm.SetGlobalBuffer((__gm__ T_GAMMA*)beta);
        gammaGm.SetGlobalBuffer((__gm__ T_GAMMA*)gamma);
        runningMeanGm.SetGlobalBuffer((__gm__ T_RUNNING_MEAN*)mean);
        runningVarGm.SetGlobalBuffer((__gm__ T_RUNNING_MEAN*)var);
        yGm.SetGlobalBuffer((__gm__ T*)y + xyGmOffset);
        batchMeanGm.SetGlobalBuffer((__gm__ float*)batch_mean);
        batchRstdGm.SetGlobalBuffer((__gm__ float*)batch_rstd);
        runningMeanOutGm.SetGlobalBuffer((__gm__ T_RUNNING_MEAN*)mean_out);
        runningVarOutGm.SetGlobalBuffer((__gm__ T_RUNNING_MEAN*)var_out);
        meanWsp.SetGlobalBuffer((__gm__ float*)workspace + blockIdx * tilingData->patternAAlign);
        varWsp.SetGlobalBuffer((__gm__ float*)workspace + (usedCoreNum + blockIdx) * tilingData->patternAAlign);
        workspaceGm.SetGlobalBuffer((__gm__ float*)workspace);

        int64_t aTGammaAlign = CEIL_ALIGN(tilingData->patternA, BLOCK_SIZE / sizeof(T_GAMMA));
        int64_t aTRunningMeanAlign = CEIL_ALIGN(tilingData->patternA, BLOCK_SIZE / sizeof(T_RUNNING_MEAN));

        pipe.InitBuffer(xQueue, DOUBLE_BUFFER, tilingData->ubFactor * sizeof(T));
        pipe.InitBuffer(yQueue, DOUBLE_BUFFER, tilingData->ubFactor * sizeof(T));
        pipe.InitBuffer(gammaQueue, 1, aTGammaAlign * sizeof(T_GAMMA));
        pipe.InitBuffer(betaQueue, 1, aTGammaAlign * sizeof(T_GAMMA));
        pipe.InitBuffer(batchMeanQueue, 1, tilingData->patternAAlign * sizeof(float));
        pipe.InitBuffer(batchRstdQueue, 1, tilingData->patternAAlign * sizeof(float));
        pipe.InitBuffer(runningMeanInQueue, 1, aTRunningMeanAlign * sizeof(T_RUNNING_MEAN));
        pipe.InitBuffer(runningVarInQueue, 1, aTRunningMeanAlign * sizeof(T_RUNNING_MEAN));
        pipe.InitBuffer(runningMeanOutQueue, 1, aTRunningMeanAlign * sizeof(T_RUNNING_MEAN));
        pipe.InitBuffer(runningVarOutQueue, 1, aTRunningMeanAlign * sizeof(T_RUNNING_MEAN));
        pipe.InitBuffer(tmpTbuf1, tilingData->ubFactor * sizeof(float));
        pipe.InitBuffer(tmpTbuf2, tilingData->ubFactor * sizeof(float));

        int64_t usedCoreNumAlign = CEIL_ALIGN(usedCoreNum, FP32_BLOCK_ALIGN_SIZE);
        pipe.InitBuffer(countTbuf1, tilingData->ubFactor * sizeof(float));
        pipe.InitBuffer(countTbuf2, usedCoreNumAlign * sizeof(float));

        pipe.InitBuffer(tmpTbuf3, AscendC::Std::max(static_cast<uint64_t>(usedCoreNum * tilingData->patternAAlign),
                                                    CEIL_ALIGN(FIRST_VCADD_RESULT_MAX_NUM, FP32_BLOCK_ALIGN_SIZE)) *
                                      sizeof(float));
    }

    __aicore__ inline void Process()
    {
        LocalTensor<float> meanTensor = tmpTbuf1.Get<float>();
        LocalTensor<float> m2Tensor = tmpTbuf2.Get<float>();
        LocalTensor<float> tmpTensor = tmpTbuf3.Get<float>();
        LocalTensor<float> countTensor1 = countTbuf1.Get<float>();
        LocalTensor<float> countTensor2 = countTbuf2.Get<float>();
        CaculateCountBuf(countTensor1, countTensor2);
        int64_t xGmOffset = 0;
        uint64_t formerRAlignNum = 0;
        if (ubSplitAxis == UB_SPLIT_AXIS_R1) {
            // RAR
            formerRAlignNum = CEIL_ALIGN(r0BlockInner * ubInner, T_BLOCK_ALIGN_SIZE);
            uint64_t calcLen = tilingData->patternA * formerRAlignNum;
            uint16_t meanM2LoopCount = CEIL_DIV(calcLen, VL_F32);
            BatchNormV3MeanM2TensorInit(meanTensor, m2Tensor, calcLen, meanM2LoopCount, VL_F32);
            int64_t count = 0;
            for (uint64_t r1Idx = r1StartIdx; r1Idx < r1EndIdx; r1Idx += ubInner) {
                uint64_t processR1Num = 1;
                if (r1Idx + ubInner < r1EndIdx) {
                    processR1Num = ubInner;
                } else {
                    processR1Num = r1EndIdx - r1Idx;
                }

                xGmOffset = (r1Idx - r1StartIdx) * tilingData->patternR0 * tilingData->patternA;
                // copy in x
                LocalTensor<T> xTensor = xQueue.AllocTensor<T>();
                CopyInXRAR(xTensor, xGmOffset, tilingData->patternA, r0BlockInner, processR1Num);
                xQueue.EnQue(xTensor);
                xTensor = xQueue.DeQue<T>();

                WelfordParallelUpdate(xTensor, count, meanTensor, m2Tensor, tilingData->patternA, r0BlockInner,
                                      processR1Num, formerRAlignNum);
                xQueue.FreeTensor(xTensor);
            }
            currentAAlign = tilingData->patternAAlign;
            LocalTensor<float> localMeanTensor = batchMeanQueue.AllocTensor<float>();
            LocalTensor<float> localVarTensor = batchRstdQueue.AllocTensor<float>();
            ProcessWelfordRARFinalize(meanTensor, m2Tensor, countTensor1, localMeanTensor, localVarTensor, tmpTensor,
                                      tilingData->patternA, r0BlockInner * ubInner, formerRAlignNum);
            batchMeanQueue.EnQue(localMeanTensor);
            batchRstdQueue.EnQue(localVarTensor);
            localMeanTensor = batchMeanQueue.template DeQue<float>();
            localVarTensor = batchRstdQueue.template DeQue<float>();
            DataCopy(meanWsp[0], localMeanTensor, currentAAlign);
            DataCopy(varWsp[0], localVarTensor, currentAAlign);
            batchMeanQueue.FreeTensor(localMeanTensor);
            batchRstdQueue.FreeTensor(localVarTensor);
        } else if (ubSplitAxis == UB_SPLIT_AXIS_A) {
            // AR
            formerRAlignNum = CEIL_ALIGN(r0BlockInner, T_BLOCK_ALIGN_SIZE);
            uint64_t calcLen = ubInner * formerRAlignNum;
            for (uint64_t aIdx = 0; aIdx < ubOuter; aIdx += 1) {
                uint64_t aStartIdx = aIdx * ubInner;
                uint64_t processANum = 1;
                if (aIdx < ubOuter - 1) {
                    processANum = ubInner;
                } else {
                    processANum = tilingData->patternA - aStartIdx;
                }
                xGmOffset = 0;
                calcLen = processANum * formerRAlignNum;
                uint16_t meanM2LoopCount = CEIL_DIV(calcLen, VL_F32);
                BatchNormV3MeanM2TensorInit(meanTensor, m2Tensor, calcLen, meanM2LoopCount, VL_F32);
                int64_t count = 0;
                for (uint64_t r1Idx = r1StartIdx; r1Idx < r1EndIdx; r1Idx += 1) {
                    xGmOffset = (r1Idx - r1StartIdx) * tilingData->patternR0 * tilingData->patternA +
                                aStartIdx * tilingData->patternR0;
                    // copy in x
                    LocalTensor<T> xTensor = xQueue.AllocTensor<T>();
                    CopyInXAR(xTensor, xGmOffset, processANum, r0BlockInner);
                    xQueue.EnQue(xTensor);
                    xTensor = xQueue.DeQue<T>();

                    WelfordParallelUpdate(xTensor, count, meanTensor, m2Tensor, processANum, r0BlockInner, 1,
                                          formerRAlignNum);
                    xQueue.FreeTensor(xTensor);
                }
                LocalTensor<float> localMeanTensor = batchMeanQueue.AllocTensor<float>();
                LocalTensor<float> localVarTensor = batchRstdQueue.AllocTensor<float>();
                ProcessWelfordRARFinalize(meanTensor, m2Tensor, countTensor1, localMeanTensor, localVarTensor,
                                          tmpTensor, processANum, r0BlockInner, formerRAlignNum);
                batchMeanQueue.EnQue(localMeanTensor);
                batchRstdQueue.EnQue(localVarTensor);
                localMeanTensor = batchMeanQueue.template DeQue<float>();
                localVarTensor = batchRstdQueue.template DeQue<float>();
                AscendC::DataCopyExtParams copyOutParams;
                copyOutParams.blockCount = 1;
                copyOutParams.blockLen = processANum * sizeof(float);
                copyOutParams.srcStride = 0;
                copyOutParams.dstStride = 0;
                AscendC::DataCopyPad<float>(meanWsp[aStartIdx], localMeanTensor, copyOutParams);
                AscendC::DataCopyPad<float>(varWsp[aStartIdx], localVarTensor, copyOutParams);
                batchMeanQueue.FreeTensor(localMeanTensor);
                batchRstdQueue.FreeTensor(localVarTensor);
            }
        } else if (ubSplitAxis == UB_SPLIT_AXIS_R0) {
            // R
            formerRAlignNum = CEIL_ALIGN(ubInner, T_BLOCK_ALIGN_SIZE);
            for (uint64_t aIdx = 0; aIdx < tilingData->patternA; aIdx += 1) {
                uint64_t calcLen = formerRAlignNum;
                uint16_t meanM2LoopCount = CEIL_DIV(calcLen, VL_F32);
                BatchNormV3MeanM2TensorInit(meanTensor, m2Tensor, calcLen, meanM2LoopCount, VL_F32);
                int64_t count = 0;
                for (uint64_t r0Idx = r0StartIdx; r0Idx < r0EndIdx; r0Idx += ubInner) {
                    uint64_t processR0Num = 1;
                    if (r0Idx + ubInner < r0EndIdx) {
                        processR0Num = ubInner;
                    } else {
                        processR0Num = r0EndIdx - r0Idx;
                    }
                    xGmOffset = 0;
                    for (uint64_t r1Idx = r1StartIdx; r1Idx < r1EndIdx; r1Idx += 1) {
                        xGmOffset = (r1Idx - r1StartIdx) * tilingData->patternR0 * tilingData->patternA +
                                    aIdx * tilingData->patternR0 + r0Idx - r0StartIdx;
                        // copy in x
                        LocalTensor<T> xTensor = xQueue.AllocTensor<T>();
                        CopyInXAR(xTensor, xGmOffset, 1, processR0Num);
                        xQueue.EnQue(xTensor);
                        xTensor = xQueue.DeQue<T>();

                        WelfordParallelUpdate(xTensor, count, meanTensor, m2Tensor, 1, processR0Num, 1,
                                              formerRAlignNum);
                        xQueue.FreeTensor(xTensor);
                    }
                }
                LocalTensor<float> localMeanTensor = batchMeanQueue.AllocTensor<float>();
                LocalTensor<float> localVarTensor = batchRstdQueue.AllocTensor<float>();
                ProcessWelfordRARFinalize(meanTensor, m2Tensor, countTensor1, localMeanTensor, localVarTensor,
                                          tmpTensor, 1, ubInner, formerRAlignNum);
                batchMeanQueue.EnQue(localMeanTensor);
                batchRstdQueue.EnQue(localVarTensor);
                localMeanTensor = batchMeanQueue.template DeQue<float>();
                localVarTensor = batchRstdQueue.template DeQue<float>();
                AscendC::DataCopyExtParams copyOutParams;
                copyOutParams.blockCount = 1;
                copyOutParams.blockLen = 1 * sizeof(float);
                copyOutParams.srcStride = 0;
                copyOutParams.dstStride = 0;
                AscendC::DataCopyPad<float>(meanWsp[aIdx], localMeanTensor, copyOutParams);
                AscendC::DataCopyPad<float>(varWsp[aIdx], localVarTensor, copyOutParams);
                batchMeanQueue.FreeTensor(localMeanTensor);
                batchRstdQueue.FreeTensor(localVarTensor);
            }
        }

        SyncAll();
        LocalTensor<float> allMeanTensor = tmpTbuf1.Get<float>();
        LocalTensor<float> alllVarTensor = tmpTbuf2.Get<float>();
        event_t eIdMte2ToVec = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::MTE2_V>());

        currentA = tilingData->patternA;
        currentAAlign = tilingData->patternAAlign;

        CopyInAllMeanVarPad(allMeanTensor, alllVarTensor, workspaceGm, 0, usedCoreNum * tilingData->patternAAlign,
                            static_cast<uint32_t>(usedCoreNum), static_cast<uint32_t>(currentAAlign),
                            static_cast<uint32_t>(tilingData->patternAAlign));
        SetFlag<HardEvent::MTE2_V>(eIdMte2ToVec);
        WaitFlag<HardEvent::MTE2_V>(eIdMte2ToVec);
        LocalTensor<float> batchMeanTensor = batchMeanQueue.AllocTensor<float>();
        LocalTensor<float> batchRstdTensor = batchRstdQueue.AllocTensor<float>();
        LastFinalizeVF(batchMeanTensor, batchRstdTensor, allMeanTensor, alllVarTensor, countTensor2, tmpTensor,
                       static_cast<uint32_t>(currentAAlign), VL_F32, static_cast<uint16_t>(currentA),
                       static_cast<uint16_t>(usedCoreNum), static_cast<uint16_t>(tilingData->lastBinaryAddQuotient),
                       static_cast<uint16_t>(tilingData->lastBinaryAddK),
                       static_cast<uint16_t>(tilingData->lastBinaryAddLast), this->lastNFactor,
                       this->lastNCorrectionFactor);

        LocalTensor<T_GAMMA> gammaTensor = gammaQueue.AllocTensor<T_GAMMA>();
        LocalTensor<T_GAMMA> betaTensor = betaQueue.AllocTensor<T_GAMMA>();
        CopyInGammaBetaPad(gammaTensor, betaTensor, gammaGm, betaGm, 0, static_cast<uint32_t>(currentA));
        gammaQueue.EnQue(gammaTensor);
        betaQueue.EnQue(betaTensor);
        if (blockIdx == 0) {
            uint16_t aLoop = CEIL_DIV(currentA, VL_F32);
            UpdateRunningMeanVarCommon<T_RUNNING_MEAN>(
                batchMeanTensor, batchRstdTensor, runningMeanInQueue, runningVarInQueue, runningMeanOutQueue,
                runningVarOutQueue, runningMeanGm, runningVarGm, runningMeanOutGm, runningVarOutGm, 0,
                static_cast<uint32_t>(currentA), aLoop, VL_F32, this->unbiasedEstimationCoeff, tilingData->momentum,
                tilingData->momentumReverse);
        }
        gammaTensor = gammaQueue.DeQue<T_GAMMA>();
        betaTensor = betaQueue.DeQue<T_GAMMA>();
        // 需要等runningMeanVar计算完成后，才能计算成Rstd
        NormCommon::ComputeRstdNewtonRaphson<false>(batchRstdTensor, batchRstdTensor, static_cast<uint32_t>(currentA),
                                                    tilingData->epsilon, 1.0f, VL_F32);

        NormalizeX(batchMeanTensor, batchRstdTensor, gammaTensor, betaTensor);

        gammaQueue.FreeTensor(gammaTensor);
        betaQueue.FreeTensor(betaTensor);
        if (blockIdx == 0) {
            batchMeanQueue.EnQue(batchMeanTensor);
            batchRstdQueue.EnQue(batchRstdTensor);
            batchMeanTensor = batchMeanQueue.template DeQue<float>();
            batchRstdTensor = batchRstdQueue.template DeQue<float>();
            CopyOutBatchMeanRstdPad(batchMeanTensor, batchRstdTensor, batchMeanGm, batchRstdGm, 0,
                                    static_cast<uint32_t>(currentA));
        }
        batchMeanQueue.FreeTensor(batchMeanTensor);
        batchRstdQueue.FreeTensor(batchRstdTensor);
    }

private:
    __aicore__ inline void CaculateCountBuf(LocalTensor<float>& tCountTensor1, LocalTensor<float>& tCountTensor2)
    {
        __local_mem__ float* tmpCountLocal1 = (__local_mem__ float*)tCountTensor1.GetPhyAddr();
        __local_mem__ float* tmpCountLocal2 = (__local_mem__ float*)tCountTensor2.GetPhyAddr();
        float baseAddCount;
        float tailAddCount;
        uint32_t baseNum;
        uint32_t tailNum;
        if (ubSplitAxis == 0) {
            baseAddCount = (ubOuter == 1) ? 1 : (ubOuter - 1);
            tailAddCount = ubOuter;
            baseNum = ubInner * tilingData->patternR0;
            tailNum = (r1BlockInner % ubInner == 0) ? baseNum : (r1BlockInner % ubInner) * tilingData->patternR0;
        } else if (ubSplitAxis == 1) {
            // ub 切 A R0全载
            baseAddCount = r1BlockInner;
            tailAddCount = 0;
            baseNum = r0BlockInner;
            tailNum = 0;
        } else {
            baseAddCount = (ubOuter == 1) ? r1BlockInner : r1BlockInner * (ubOuter - 1);
            tailAddCount = r1BlockInner * ubOuter;
            baseNum = ubInner;
            tailNum = (r0BlockInner % ubInner == 0) ? baseNum : (r0BlockInner % ubInner);
        }
        uint16_t baseLoopCount = CEIL_DIV(baseNum, VL_F32);
        uint16_t tailLoopCount = CEIL_DIV(tailNum, VL_F32);

        int64_t tailcoreProcessNum = 0;
        int64_t formercoreProcessNum = 0;
        uint32_t firstNum = 0;
        uint32_t secondNum = 0;
        if (tilingData->blockSplitAxis == 0) {
            tailcoreProcessNum = (tilingData->blockInner - 1) * tilingData->patternR0;
            formercoreProcessNum = tilingData->blockInner * tilingData->patternR0;
            firstNum = usedCoreNum;
            secondNum = usedCoreNum - tilingData->tailBlockOuter;
        } else {
            tailcoreProcessNum = (tilingData->blockInner - 1);
            formercoreProcessNum = tilingData->blockInner;
            firstNum = usedCoreNum;
            secondNum = usedCoreNum - tilingData->tailBlockOuter * tilingData->patternR1;
        }

        float tailCoreAddCount = static_cast<float>(tailcoreProcessNum);
        float formerCoreAddCount = static_cast<float>(formercoreProcessNum);

        uint16_t firstLoopCount = CEIL_DIV(firstNum, VL_F32);
        uint16_t secondLoopCount = CEIL_DIV(secondNum, VL_F32);
        __VEC_SCOPE__
        {
            RegTensor<float> tmpCount;
            MaskReg pregMain = CreateMask<float, MaskPattern::ALL>();
            MaskReg pregLoop;
            uint32_t sreg1 = baseNum;
            Duplicate(tmpCount, baseAddCount, pregMain);
            for (uint16_t i = 0; i < baseLoopCount; i++) {
                pregLoop = AscendC::MicroAPI::UpdateMask<float>(sreg1);
                DataCopy(((__local_mem__ float*)tmpCountLocal1 + i * VL_F32), tmpCount, pregLoop);
            }
            uint32_t sreg2 = tailNum;
            Duplicate(tmpCount, tailAddCount, pregMain);
            for (uint16_t i = 0; i < tailLoopCount; i++) {
                pregLoop = AscendC::MicroAPI::UpdateMask<float>(sreg2);
                DataCopy(((__local_mem__ float*)tmpCountLocal1 + i * VL_F32), tmpCount, pregLoop);
            }
            uint32_t sreg3 = firstNum;
            Duplicate(tmpCount, tailCoreAddCount, pregMain);
            for (uint16_t i = 0; i < firstLoopCount; i++) {
                pregLoop = AscendC::MicroAPI::UpdateMask<float>(sreg3);
                DataCopy(((__local_mem__ float*)tmpCountLocal2 + i * VL_F32), tmpCount, pregLoop);
            }
            uint32_t sreg4 = secondNum;
            Duplicate(tmpCount, formerCoreAddCount, pregMain);
            for (uint16_t i = 0; i < secondLoopCount; i++) {
                pregLoop = AscendC::MicroAPI::UpdateMask<float>(sreg4);
                DataCopy(((__local_mem__ float*)tmpCountLocal2 + i * VL_F32), tmpCount, pregLoop);
            }
        }
    }

    __aicore__ inline void CopyInXAR(LocalTensor<T>& xInUb, int64_t offset, uint64_t processANum, uint64_t processR0Num)
    {
        AscendC::DataCopyExtParams copyInParams;
        copyInParams.blockCount = processANum;
        copyInParams.blockLen = processR0Num * sizeof(T);
        copyInParams.srcStride = (tilingData->patternR0 - processR0Num) * sizeof(T);
        copyInParams.dstStride = 0;
        AscendC::DataCopyPadExtParams<T> dataCopyPadExtParams;
        dataCopyPadExtParams.isPad = (processR0Num != CEIL_ALIGN(processR0Num, T_BLOCK_ALIGN_SIZE));
        dataCopyPadExtParams.leftPadding = 0;
        // isPad配置True，rightPadding配置0，表示自动Pad到32B对齐
        dataCopyPadExtParams.rightPadding = 0;
        dataCopyPadExtParams.paddingValue = 0;

        AscendC::DataCopyPad<T, PaddingMode::Normal>(xInUb, xGm[offset], copyInParams, dataCopyPadExtParams);
    }

    __aicore__ inline void CopyInXRAR(LocalTensor<T>& xInUb, int64_t offset, uint64_t processANum,
                                      uint64_t processR0Num, uint64_t processR1Num)
    {
        uint64_t rProcessNumAlign = CEIL_ALIGN(processR0Num * processR1Num, T_BLOCK_ALIGN_SIZE);

        AscendC::DataCopyExtParams copyInParams;
        copyInParams.blockCount = processR1Num;
        copyInParams.blockLen = processR0Num * sizeof(T);
        copyInParams.srcStride = (tilingData->patternR0 * tilingData->patternA - processR0Num) * sizeof(T);
        copyInParams.dstStride = 0;

        AscendC::DataCopyPadExtParams<T> dataCopyPadExtParams;
        dataCopyPadExtParams.isPad = (processR0Num * processR1Num != rProcessNumAlign);
        dataCopyPadExtParams.leftPadding = 0;
        // isPad配置True，rightPadding配置0，表示自动Pad到32B对齐
        dataCopyPadExtParams.rightPadding = 0;
        dataCopyPadExtParams.paddingValue = 0;

        uint32_t loop1Size = processANum;
        uint64_t loop1SrcStride = tilingData->patternR0 * sizeof(T);
        uint64_t loop1DstStride = rProcessNumAlign * sizeof(T);
        uint32_t loop2Size = 1;
        uint64_t loop2SrcStride = 0;
        uint64_t loop2DstStride = 0;

        AscendC::LoopModeParams LoopParams{loop1Size,      loop2Size,      loop1SrcStride,
                                           loop1DstStride, loop2SrcStride, loop2DstStride};

        AscendC::SetLoopModePara(LoopParams, DataCopyMVType::OUT_TO_UB);
        AscendC::DataCopyPad<T, PaddingMode::Compact>(xInUb, xGm[offset], copyInParams, dataCopyPadExtParams);
        AscendC::ResetLoopModePara(DataCopyMVType::OUT_TO_UB);
    }

    // xTensor是AR排布(A * (r1 * r0))，R轴向32B对齐，且R轴可能有尾块处理
    // 所以formerRAlignNum是头块R对齐的长度，目标是找到meanTensor以及m2Tensor的AR排布中一行的大小，将WelfordParallelUpdate的结果放入正确的a_idx所在的行
    __aicore__ inline void WelfordParallelUpdate(LocalTensor<T>& xTensor, int64_t& count,
                                                 LocalTensor<float>& meanTensor, LocalTensor<float>& m2Tensor,
                                                 uint64_t processANum, uint64_t processR0Num, uint64_t processR1Num,
                                                 uint64_t formerRAlignNum)
    {
        count += 1;
        float scale = (float)1.0 / static_cast<float>(count);
        __local_mem__ float* meanTensorAddr = (__local_mem__ float*)meanTensor.GetPhyAddr();
        __local_mem__ float* m2TensorAddr = (__local_mem__ float*)m2Tensor.GetPhyAddr();
        __local_mem__ T* xTensorAddr = (__local_mem__ T*)xTensor.GetPhyAddr();

        uint64_t processRAlignNum = CEIL_ALIGN(processR0Num * processR1Num, T_BLOCK_ALIGN_SIZE);
        uint16_t loopCount = CEIL_DIV(processR0Num * processR1Num, VL_F32);

        uint16_t processALoopCount = static_cast<uint16_t>(processANum);
        __VEC_SCOPE__
        {
            for (uint16_t a_idx = 0; a_idx < processALoopCount; a_idx++) {
                RegTensor<float> x1;
                RegTensor<float> tmpMean;
                RegTensor<float> tmpM2;
                RegTensor<float> delta1;
                RegTensor<float> delta2;
                RegTensor<float> delta3;
                RegTensor<float> delta4;
                MaskReg mask0;
                uint32_t sreg0 = processR0Num * processR1Num;
                for (uint16_t i = 0; i < loopCount; i++) {
                    mask0 = AscendC::MicroAPI::UpdateMask<float>(sreg0);
                    LoadOneTensorForDtypeT(xTensorAddr, x1, mask0, a_idx * processRAlignNum + i * VL_F32);
                    DataCopy(tmpMean, meanTensorAddr + a_idx * formerRAlignNum + i * VL_F32);
                    DataCopy(tmpM2, m2TensorAddr + a_idx * formerRAlignNum + i * VL_F32);
                    // delata1 = x1 - mean
                    Sub(delta1, x1, tmpMean, mask0);
                    // delta2 = delta1 * scale
                    Muls(delta2, delta1, scale, mask0);
                    // mean = mean + delta2
                    Add(tmpMean, tmpMean, delta2, mask0);
                    DataCopy(meanTensorAddr + a_idx * formerRAlignNum + i * VL_F32, tmpMean, mask0);
                    // delta3 = x1 - mean
                    Sub(delta3, x1, tmpMean, mask0);
                    // delta4 = delta1 * delta3
                    Mul(delta4, delta1, delta3, mask0);
                    // M2 = M2 + delta4
                    Add(tmpM2, tmpM2, delta4, mask0);
                    DataCopy(m2TensorAddr + a_idx * formerRAlignNum + i * VL_F32, tmpM2, mask0);
                }
            }
        }
    }

    // Finalize输入的AR：aUbFactor * rUbFactor，例如10000,4,10000在这里就是 1 * 8000；而10000,4,3000在这里就是2*3000
    // meanTensor & m2Tensor ：aUbFactor * rUbFactorAlign;  countTensor: 1*rUbFactorAlign
    // finalMeanTensor & finalVarTensor: aUbFactor与blocksize对齐后的大小
    // curAUbFactor: 循环A，aUbFactor; numR: rUbFactor; numRAlign: rUbFactorAlign
    // 还需注意下rFlodNum: rUbFactorAlign向下的2幂次
    __aicore__ inline void ProcessWelfordRARFinalize(LocalTensor<float>& meanTensor, LocalTensor<float>& m2Tensor,
                                                     LocalTensor<float>& countTensor,
                                                     LocalTensor<float>& finalMeanTensor,
                                                     LocalTensor<float>& finalVarTensor, LocalTensor<float>& tmpTensor,
                                                     uint64_t curAUbFactor, uint64_t numR, uint64_t numRAlign)
    {
        __local_mem__ float* tmpMeanLocal = (__local_mem__ float*)meanTensor.GetPhyAddr();
        __local_mem__ float* tmpVarLocal = (__local_mem__ float*)m2Tensor.GetPhyAddr();
        __local_mem__ float* tmpCountLocal = (__local_mem__ float*)countTensor.GetPhyAddr();
        __local_mem__ float* batchMeanInUbAddr = (__local_mem__ float*)finalMeanTensor.GetPhyAddr();
        __local_mem__ float* batchRstdInUbAddr = (__local_mem__ float*)finalVarTensor.GetPhyAddr();
        __local_mem__ float* tmpUbAddr = (__local_mem__ float*)tmpTensor.GetPhyAddr();
        // AR高性能二分累加  64 * 2<R轴大小
        if (numRAlign < VL_F32 * VL_F32 * NUM_TWO) {
            WelfordRARFinalizeMeanVF<NUM_ONE>(tmpMeanLocal, tmpVarLocal, tmpCountLocal, batchMeanInUbAddr,
                                              batchRstdInUbAddr, tmpUbAddr, curAUbFactor, numR, numRAlign);
            WelfordRARFinalizeVarVF<NUM_ONE>(tmpMeanLocal, tmpVarLocal, tmpCountLocal, batchMeanInUbAddr,
                                             batchRstdInUbAddr, tmpUbAddr, curAUbFactor, numR, numRAlign);
        } else {
            WelfordRARFinalizeMeanVF<NUM_TWO>(tmpMeanLocal, tmpVarLocal, tmpCountLocal, batchMeanInUbAddr,
                                              batchRstdInUbAddr, tmpUbAddr, curAUbFactor, numR, numRAlign);
            WelfordRARFinalizeVarVF<NUM_TWO>(tmpMeanLocal, tmpVarLocal, tmpCountLocal, batchMeanInUbAddr,
                                             batchRstdInUbAddr, tmpUbAddr, curAUbFactor, numR, numRAlign);
        }
    }

    template <int32_t LAST_LOOP_NUMS>
    __aicore__ inline void WelfordRARFinalizeMeanVF(__local_mem__ float* tmpMeanLocal, __local_mem__ float* tmpVarLocal,
                                                    __local_mem__ float* tmpCountLocal,
                                                    __local_mem__ float* batchMeanInUbAddr,
                                                    __local_mem__ float* batchRstdInUbAddr,
                                                    __local_mem__ float* tmpLocalUbAddr, uint64_t curAUbFactor,
                                                    uint64_t numR, uint64_t numRAlign)
    {
        uint32_t rNum = static_cast<uint32_t>(numR);
        uint32_t rNumAlign = static_cast<uint32_t>(numRAlign);
        uint16_t curAloops = static_cast<uint16_t>(curAUbFactor); // A
        uint32_t rFlodNum = static_cast<uint32_t>(rFlodFactor);   // numR对齐后 向下的2幂次

        // first flod  首次累加
        uint32_t firstFlodTial = static_cast<uint32_t>(rNum - rFlodFactor); // R - 向下的2幂次
        uint16_t firstFlodAddLoops = static_cast<uint16_t>((firstFlodTial + VL_F32 - 1) / VL_F32); // reg中尾块循环次数
        uint16_t firstFlodWithOutAddLoops = static_cast<uint16_t>((rFlodNum + VL_F32 - 1) / VL_F32) -
                                            firstFlodAddLoops; // 剩余循环

        // first vcadd
        uint32_t firstVcaddNum = static_cast<uint32_t>((rFlodFactor + VL_F32 - 1) / VL_F32); // 最终是二分对齐点个64
        uint32_t firstVcaddNumCeilAlign = static_cast<uint32_t>((firstVcaddNum + FP32_BLOCK_ALIGN_SIZE - 1) /
                                                                FP32_BLOCK_ALIGN_SIZE * FP32_BLOCK_ALIGN_SIZE);

        // n的作用  sum / allcount
        float scaleCorrection = this->nCorrectionFactor;
        float numScale = this->nFactor;

        __VEC_SCOPE__
        {
            RegTensor<float> xReg1; // 头
            RegTensor<float> xReg2; // 尾
            RegTensor<float> formerCount;
            RegTensor<float> tailCount;
            RegTensor<float> addReg;
            RegTensor<float> sumReg;
            RegTensor<float> xReg3; // 头尾加完后剩余块
            RegTensor<float> count3;
            RegTensor<float> sumReg3;

            MaskReg pregFull = CreateMask<float, MaskPattern::ALL>();
            MaskReg pregOne = CreateMask<float, MaskPattern::VL1>();
            MaskReg pregLoop;

            // 先循环A（aubfactor），再循环尾块
            // 取count是否是brc
            for (uint16_t i = 0; i < curAloops; i++) {
                uint32_t sregfirstFlodTial = firstFlodTial;
                for (uint16_t j = 0; j < firstFlodAddLoops; j++) {
                    pregLoop = UpdateMask<float>(sregfirstFlodTial);
                    DataCopy<float, LoadDist::DIST_NORM>(
                        xReg1, tmpMeanLocal + static_cast<uint32_t>(i * rNumAlign + j * VL_F32));
                    DataCopy<float, LoadDist::DIST_NORM>(formerCount,
                                                         tmpCountLocal + static_cast<uint32_t>(j * VL_F32));
                    Mul(xReg1, xReg1, formerCount, pregFull);
                    Muls(xReg1, xReg1, numScale, pregFull);
                    DataCopy<float, LoadDist::DIST_NORM>(
                        xReg2, tmpMeanLocal + rFlodNum + static_cast<uint32_t>(i * rNumAlign + j * VL_F32));
                    DataCopy<float, LoadDist::DIST_NORM>(tailCount,
                                                         tmpCountLocal + rFlodNum + static_cast<uint32_t>(j * VL_F32));
                    Mul(xReg2, xReg2, tailCount, pregLoop);
                    Muls(xReg2, xReg2, numScale, pregLoop);
                    Add(addReg, xReg1, xReg2, pregFull);
                    ReduceSum(sumReg, addReg, pregFull);
                    DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(
                        tmpLocalUbAddr + static_cast<uint32_t>(i * firstVcaddNumCeilAlign + j), sumReg, pregOne);
                }
                // 剩余块
                for (uint16_t j = 0; j < static_cast<uint16_t>(firstFlodWithOutAddLoops); j++) {
                    DataCopy<float, LoadDist::DIST_NORM>(xReg3, tmpMeanLocal + (firstFlodAddLoops * VL_F32) +
                                                                    static_cast<uint32_t>(i * rNumAlign + j * VL_F32));
                    DataCopy<float, LoadDist::DIST_NORM>(
                        count3, tmpCountLocal + (firstFlodAddLoops * VL_F32) + static_cast<uint32_t>(j * VL_F32));
                    Mul(xReg3, xReg3, count3, pregFull);
                    Muls(xReg3, xReg3, numScale, pregFull);
                    ReduceSum(sumReg3, xReg3, pregFull);
                    DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(
                        tmpLocalUbAddr + static_cast<uint32_t>(i * firstVcaddNumCeilAlign + firstFlodAddLoops + j),
                        sumReg3, pregOne);
                }
            }

            // if need a add to last repeat
            LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
            if constexpr (LAST_LOOP_NUMS == 1) {
                uint32_t sregSecondReduce = firstVcaddNum;
                MaskReg pregLast = UpdateMask<float>(sregSecondReduce);
                for (uint16_t i = 0; i < curAloops; i++) {
                    DataCopy(xReg1, tmpLocalUbAddr + static_cast<uint32_t>(i * firstVcaddNumCeilAlign));
                    ReduceSum(sumReg, xReg1, pregLast);
                    Muls(sumReg, sumReg, scaleCorrection, pregOne);
                    DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(batchMeanInUbAddr + i, sumReg, pregOne);
                }
            } else if constexpr (LAST_LOOP_NUMS == 2) {
                uint32_t sregSecondReduce = firstVcaddNum - VL_F32;
                MaskReg pregLast = UpdateMask<float>(sregSecondReduce);
                RegTensor<float> shiftLeft;
                for (uint16_t i = 0; i < curAloops; i++) {
                    DataCopy(xReg1, tmpLocalUbAddr + static_cast<uint32_t>(i * firstVcaddNumCeilAlign));
                    DataCopy(xReg2, tmpLocalUbAddr + static_cast<uint32_t>(i * firstVcaddNumCeilAlign + VL_F32));
                    ShiftLefts((RegTensor<uint32_t>&)shiftLeft, (RegTensor<uint32_t>&)xReg2, static_cast<int16_t>(0),
                               pregLast);
                    Add(addReg, xReg1, shiftLeft, pregFull);
                    ReduceSum(sumReg, addReg, pregFull);
                    Muls(sumReg, sumReg, scaleCorrection, pregOne);
                    DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(batchMeanInUbAddr + i, sumReg, pregOne);
                }
            }
        }
    }

    template <int32_t LAST_LOOP_NUMS>
    __aicore__ inline void WelfordRARFinalizeVarVF(__local_mem__ float* tmpMeanLocal, __local_mem__ float* tmpVarLocal,
                                                   __local_mem__ float* tmpCountLocal,
                                                   __local_mem__ float* batchMeanInUbAddr,
                                                   __local_mem__ float* batchRstdInUbAddr,
                                                   __local_mem__ float* tmpLocalUbAddr, uint64_t curAUbFactor,
                                                   uint64_t numR, uint64_t numRAlign)
    {
        uint32_t rNum = static_cast<uint32_t>(numR);
        uint32_t rNumAlign = static_cast<uint32_t>(numRAlign);
        uint32_t rFlodNum = static_cast<uint32_t>(rFlodFactor);   // numR对齐后 向下的2幂次
        uint16_t curAloops = static_cast<uint16_t>(curAUbFactor); // A

        // first flod  首次累加
        uint32_t firstFlodTial = static_cast<uint32_t>(rNum - rFlodFactor); // R - 向下的2幂次
        uint16_t firstFlodAddLoops = static_cast<uint16_t>((firstFlodTial + VL_F32 - 1) / VL_F32); // reg中尾块循环次数
        uint16_t firstFlodWithOutAddLoops = static_cast<uint16_t>((rFlodNum + VL_F32 - 1) / VL_F32) -
                                            firstFlodAddLoops; // 剩余循环

        // first vcadd
        uint32_t firstVcaddNum = static_cast<uint32_t>((rFlodFactor + VL_F32 - 1) / VL_F32); // 最终是二分对齐点个64
        uint32_t firstVcaddNumCeilAlign = static_cast<uint32_t>((firstVcaddNum + FP32_BLOCK_ALIGN_SIZE - 1) /
                                                                FP32_BLOCK_ALIGN_SIZE * FP32_BLOCK_ALIGN_SIZE);

        // n的作用  sum / allcount
        float numScale = this->nFactor;
        float scaleCorrection = this->nCorrectionFactor;

        // var计算时多出的welford计算：1、mean - sum_mean; 2、平方; 3、M2 + 平方
        __VEC_SCOPE__
        {
            RegTensor<float> xReg1; // 头
            RegTensor<float> xReg2; // 尾
            RegTensor<float> formerCount;
            RegTensor<float> tailCount;
            RegTensor<float> addReg;
            RegTensor<float> sumReg;
            RegTensor<float> xReg3; // 头尾加完后剩余块
            RegTensor<float> count3;
            RegTensor<float> sumReg3;

            RegTensor<float> saveMean; //保存的sum_mean
            RegTensor<float> rM2;

            MaskReg pregFull = CreateMask<float, MaskPattern::ALL>();
            MaskReg pregOne = CreateMask<float, MaskPattern::VL1>();
            MaskReg pregLoop;

            // 先循环A（aubfactor），再循环尾块
            for (uint16_t i = 0; i < curAloops; i++) {
                DataCopy<float, LoadDist::DIST_BRC_B32>(saveMean, ((__local_mem__ float*)batchMeanInUbAddr + i));
                uint32_t sregfirstFlodTial = firstFlodTial;
                for (uint16_t j = 0; j < firstFlodAddLoops; j++) {
                    pregLoop = UpdateMask<float>(sregfirstFlodTial);
                    DataCopy<float, LoadDist::DIST_NORM>(
                        xReg1, tmpMeanLocal + static_cast<uint32_t>(i * rNumAlign + j * VL_F32));
                    DataCopy<float, LoadDist::DIST_NORM>(formerCount,
                                                         tmpCountLocal + static_cast<uint32_t>(j * VL_F32));
                    Sub(xReg1, xReg1, saveMean, pregFull);
                    Mul(xReg1, xReg1, xReg1, pregFull);
                    Mul(xReg1, xReg1, formerCount, pregFull);
                    DataCopy<float, LoadDist::DIST_NORM>(
                        rM2, tmpVarLocal + static_cast<uint32_t>(i * rNumAlign + j * VL_F32));
                    Add(xReg1, rM2, xReg1, pregFull);
                    Muls(xReg1, xReg1, numScale, pregFull);

                    DataCopy<float, LoadDist::DIST_NORM>(
                        xReg2, tmpMeanLocal + rFlodNum + static_cast<uint32_t>(i * rNumAlign + j * VL_F32));
                    DataCopy<float, LoadDist::DIST_NORM>(tailCount,
                                                         tmpCountLocal + rFlodNum + static_cast<uint32_t>(j * VL_F32));
                    Sub(xReg2, xReg2, saveMean, pregLoop);
                    Mul(xReg2, xReg2, xReg2, pregLoop);
                    Mul(xReg2, xReg2, tailCount, pregLoop);
                    DataCopy<float, LoadDist::DIST_NORM>(
                        rM2, tmpVarLocal + rFlodNum + static_cast<uint32_t>(i * rNumAlign + j * VL_F32));
                    Add(xReg2, rM2, xReg2, pregLoop);
                    Muls(xReg2, xReg2, numScale, pregLoop);

                    Add(addReg, xReg1, xReg2, pregFull);
                    ReduceSum(sumReg, addReg, pregFull);
                    DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(
                        tmpLocalUbAddr + static_cast<uint32_t>(i * firstVcaddNumCeilAlign + j), sumReg, pregOne);
                }
                // 剩余块
                for (uint16_t j = 0; j < static_cast<uint16_t>(firstFlodWithOutAddLoops); j++) {
                    DataCopy<float, LoadDist::DIST_NORM>(
                        xReg3, tmpMeanLocal + (firstFlodAddLoops * VL_F32) + (i * rNumAlign + j * VL_F32));
                    DataCopy<float, LoadDist::DIST_NORM>(count3,
                                                         tmpCountLocal + (firstFlodAddLoops * VL_F32) + (j * VL_F32));
                    Sub(xReg3, xReg3, saveMean, pregFull);
                    Mul(xReg3, xReg3, xReg3, pregFull);
                    Mul(xReg3, xReg3, count3, pregFull);
                    DataCopy<float, LoadDist::DIST_NORM>(rM2, tmpVarLocal + (firstFlodAddLoops * VL_F32) +
                                                                  static_cast<uint32_t>(i * rNumAlign + j * VL_F32));
                    Add(xReg3, rM2, xReg3, pregFull);
                    Muls(xReg3, xReg3, numScale, pregFull);
                    ReduceSum(sumReg3, xReg3, pregFull);
                    DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(
                        tmpLocalUbAddr + static_cast<uint32_t>(i * firstVcaddNumCeilAlign + firstFlodAddLoops + j),
                        sumReg3, pregOne);
                }
            }

            // if need a add to last repeat
            LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
            if constexpr (LAST_LOOP_NUMS == 1) {
                uint32_t sregSecondReduce = firstVcaddNum;
                MaskReg pregLast = UpdateMask<float>(sregSecondReduce);
                for (uint16_t i = 0; i < curAloops; i++) {
                    DataCopy(xReg1, tmpLocalUbAddr + static_cast<uint32_t>(i * firstVcaddNumCeilAlign));
                    ReduceSum(sumReg, xReg1, pregLast);
                    Muls(sumReg, sumReg, scaleCorrection, pregOne);
                    DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(batchRstdInUbAddr + i, sumReg, pregOne);
                }
            } else if constexpr (LAST_LOOP_NUMS == 2) {
                uint32_t sregSecondReduce = firstVcaddNum - VL_F32;
                MaskReg pregLast = UpdateMask<float>(sregSecondReduce);
                RegTensor<float> shiftLeft;
                for (uint16_t i = 0; i < curAloops; i++) {
                    DataCopy(xReg1, tmpLocalUbAddr + static_cast<uint32_t>(i * firstVcaddNumCeilAlign));
                    DataCopy(xReg2, tmpLocalUbAddr + static_cast<uint32_t>(i * firstVcaddNumCeilAlign + VL_F32));
                    ShiftLefts((RegTensor<uint32_t>&)shiftLeft, (RegTensor<uint32_t>&)xReg2, static_cast<int16_t>(0),
                               pregLast);
                    Add(addReg, xReg1, shiftLeft, pregFull);
                    ReduceSum(sumReg, addReg, pregFull);
                    Muls(sumReg, sumReg, scaleCorrection, pregOne);
                    DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(batchRstdInUbAddr + i, sumReg, pregOne);
                }
            }
        }
    }

    // RAR计算Y,搬出Y
    __aicore__ inline void NormalizeX(LocalTensor<float>& batchMeanTensor, LocalTensor<float>& batchRstdTensor,
                                      LocalTensor<T_GAMMA>& gammaTensor, LocalTensor<T_GAMMA>& betaTensor)
    {
        int64_t xyGmOffset = 0;
        uint64_t formerRAlignNum = 0;
        if (ubSplitAxis == UB_SPLIT_AXIS_R1) {
            // RAR
            formerRAlignNum = CEIL_ALIGN(r0BlockInner * ubInner, T_BLOCK_ALIGN_SIZE);
            for (uint64_t r1Idx = r1StartIdx; r1Idx < r1EndIdx; r1Idx += ubInner) {
                uint64_t processR1Num = 1;
                if (r1Idx + ubInner < r1EndIdx) {
                    processR1Num = ubInner;
                } else {
                    processR1Num = r1EndIdx - r1Idx;
                }

                xyGmOffset = (r1Idx - r1StartIdx) * tilingData->patternR0 * tilingData->patternA;
                // copy in x
                LocalTensor<T> xTensor = xQueue.AllocTensor<T>();
                CopyInXRAR(xTensor, xyGmOffset, tilingData->patternA, r0BlockInner, processR1Num);
                xQueue.EnQue(xTensor);
                xTensor = xQueue.DeQue<T>();

                LocalTensor<T> yTensor = yQueue.AllocTensor<T>();
                formerRAlignNum = CEIL_ALIGN(r0BlockInner * processR1Num, T_BLOCK_ALIGN_SIZE);
                CalcYAR(batchMeanTensor, batchRstdTensor, gammaTensor, betaTensor, xTensor, yTensor, 0,
                        tilingData->patternA, r0BlockInner * processR1Num, formerRAlignNum);
                xQueue.FreeTensor(xTensor);
                yQueue.EnQue(yTensor);
                yTensor = yQueue.template DeQue<T>();
                CopyOutYRAR(yTensor, xyGmOffset, tilingData->patternA, r0BlockInner, processR1Num);
                yQueue.FreeTensor(yTensor);
            }
        } else if (ubSplitAxis == UB_SPLIT_AXIS_A) {
            // AR
            formerRAlignNum = CEIL_ALIGN(r0BlockInner, T_BLOCK_ALIGN_SIZE);
            for (uint64_t aIdx = 0; aIdx < ubOuter; aIdx += 1) {
                uint64_t aStartIdx = aIdx * ubInner;
                uint64_t processANum = 1;
                if (aIdx < ubOuter - 1) {
                    processANum = ubInner;
                } else {
                    processANum = tilingData->patternA - aStartIdx;
                }
                xyGmOffset = 0;

                for (uint64_t r1Idx = r1StartIdx; r1Idx < r1EndIdx; r1Idx += 1) {
                    xyGmOffset = (r1Idx - r1StartIdx) * tilingData->patternR0 * tilingData->patternA +
                                 aStartIdx * tilingData->patternR0;
                    // copy in x
                    LocalTensor<T> xTensor = xQueue.AllocTensor<T>();
                    CopyInXAR(xTensor, xyGmOffset, processANum, r0BlockInner);
                    xQueue.EnQue(xTensor);
                    xTensor = xQueue.DeQue<T>();

                    LocalTensor<T> yTensor = yQueue.AllocTensor<T>();
                    CalcYAR(batchMeanTensor, batchRstdTensor, gammaTensor, betaTensor, xTensor, yTensor, aStartIdx,
                            processANum, r0BlockInner, formerRAlignNum);
                    xQueue.FreeTensor(xTensor);
                    yQueue.EnQue(yTensor);
                    yTensor = yQueue.template DeQue<T>();
                    CopyOutYAR(yTensor, xyGmOffset, processANum, r0BlockInner);
                    yQueue.FreeTensor(yTensor);
                }
            }
        } else if (ubSplitAxis == UB_SPLIT_AXIS_R0) {
            // R
            formerRAlignNum = CEIL_ALIGN(ubInner, T_BLOCK_ALIGN_SIZE);
            for (uint64_t aIdx = 0; aIdx < tilingData->patternA; aIdx += 1) {
                for (uint64_t r0Idx = r0StartIdx; r0Idx < r0EndIdx; r0Idx += ubInner) {
                    uint64_t processR0Num = 1;
                    if (r0Idx + ubInner < r0EndIdx) {
                        processR0Num = ubInner;
                    } else {
                        processR0Num = r0EndIdx - r0Idx;
                    }
                    xyGmOffset = 0;
                    for (uint64_t r1Idx = r1StartIdx; r1Idx < r1EndIdx; r1Idx += 1) {
                        xyGmOffset = (r1Idx - r1StartIdx) * tilingData->patternR0 * tilingData->patternA +
                                     aIdx * tilingData->patternR0 + r0Idx - r0StartIdx;
                        // copy in x
                        LocalTensor<T> xTensor = xQueue.AllocTensor<T>();
                        CopyInXAR(xTensor, xyGmOffset, 1, processR0Num);
                        xQueue.EnQue(xTensor);
                        xTensor = xQueue.DeQue<T>();

                        LocalTensor<T> yTensor = yQueue.AllocTensor<T>();
                        formerRAlignNum = CEIL_ALIGN(processR0Num, T_BLOCK_ALIGN_SIZE);
                        CalcYAR(batchMeanTensor, batchRstdTensor, gammaTensor, betaTensor, xTensor, yTensor, aIdx, 1,
                                processR0Num, formerRAlignNum);
                        xQueue.FreeTensor(xTensor);
                        yQueue.EnQue(yTensor);
                        yTensor = yQueue.template DeQue<T>();
                        CopyOutYAR(yTensor, xyGmOffset, 1, processR0Num);
                        yQueue.FreeTensor(yTensor);
                    }
                }
            }
        }
    }

    template <typename T_SRC_GAMMA>
    __aicore__ inline void LoadOneNumberTensorForDtypeT(RegTensor<float>& dst, __local_mem__ T_SRC_GAMMA* input,
                                                        MaskReg& preg, uint32_t offset)
    {
        if constexpr (IsSameType<T_SRC_GAMMA, half>::value) {
            RegTensor<half> xFp16;
            DataCopy<half, LoadDist::DIST_BRC_B16>(xFp16, ((__local_mem__ half*)(input) + (offset)));
            Cast<float, half, NormCommon::castTraitB162B32>(dst, xFp16, preg);
        } else if constexpr (IsSameType<T_SRC_GAMMA, bfloat16_t>::value) {
            RegTensor<bfloat16_t> xBf16;
            DataCopy<bfloat16_t, LoadDist::DIST_BRC_B16>(xBf16, ((__local_mem__ bfloat16_t*)(input) + (offset)));
            Cast<float, bfloat16_t, NormCommon::castTraitB162B32>(dst, xBf16, preg);
        } else {
            DataCopy<float, LoadDist::DIST_BRC_B32>(dst, ((__local_mem__ float*)(input) + (offset)));
        }
    }

    __aicore__ inline void CalcYAR(LocalTensor<float>& batchMeanTensor, LocalTensor<float>& batchRstdTensor,
                                   LocalTensor<T_GAMMA>& gammaTensor, LocalTensor<T_GAMMA>& betaTensor,
                                   LocalTensor<T>& xTensor, LocalTensor<T>& yTensor, uint64_t aStartIdx,
                                   uint64_t processANum, uint64_t processRNum, uint64_t formerRAlignNum)
    {
        __local_mem__ float* batchMeanTensorAddr = (__local_mem__ float*)batchMeanTensor.GetPhyAddr();
        __local_mem__ float* batchRstdTensorAddr = (__local_mem__ float*)batchRstdTensor.GetPhyAddr();
        __local_mem__ T* xTensorAddr = (__local_mem__ T*)xTensor.GetPhyAddr();
        __local_mem__ T* yTensorAddr = (__local_mem__ T*)yTensor.GetPhyAddr();
        __local_mem__ T_GAMMA* gammaTensorAddr = (__local_mem__ T_GAMMA*)gammaTensor.GetPhyAddr();
        __local_mem__ T_GAMMA* betaTensorAddr = (__local_mem__ T_GAMMA*)betaTensor.GetPhyAddr();

        uint16_t numLoop = CEIL_DIV(processRNum, VL_F32);
        uint16_t processALoop = static_cast<uint16_t>(processANum);
        __VEC_SCOPE__
        {
            for (uint16_t j = 0; j < processALoop; j++) {
                RegTensor<float> x1;
                RegTensor<float> mean;
                RegTensor<float> rstd;
                RegTensor<float> gamma;
                RegTensor<float> beta;
                RegTensor<float> y;
                DataCopy<float, LoadDist::DIST_BRC_B32>(mean,
                                                        ((__local_mem__ float*)batchMeanTensorAddr + aStartIdx + j));
                DataCopy<float, LoadDist::DIST_BRC_B32>(rstd,
                                                        ((__local_mem__ float*)batchRstdTensorAddr + aStartIdx + j));

                MaskReg mask0;
                uint32_t sreg0 = processRNum;
                for (uint16_t i = 0; i < numLoop; i++) {
                    mask0 = AscendC::MicroAPI::UpdateMask<float>(sreg0);
                    LoadOneNumberTensorForDtypeT(gamma, gammaTensorAddr, mask0, aStartIdx + j);
                    LoadOneNumberTensorForDtypeT(beta, betaTensorAddr, mask0, aStartIdx + j);

                    LoadOneTensorForDtypeT(xTensorAddr, x1, mask0, i * VL_F32 + j * formerRAlignNum);
                    Sub(x1, x1, mean, mask0);
                    Mul(x1, x1, rstd, mask0);
                    Mul(x1, x1, gamma, mask0);
                    Add(y, x1, beta, mask0);
                    if constexpr (IsSameType<T, half>::value) {
                        RegTensor<half> yFp16;
                        Cast<half, float, NormCommon::castTraitB322B16>(yFp16, y, mask0);
                        DataCopy<half, StoreDist::DIST_PACK_B32>(yTensorAddr + i * VL_F32 + j * formerRAlignNum, yFp16,
                                                                 mask0);
                    } else if constexpr (IsSameType<T, bfloat16_t>::value) {
                        RegTensor<bfloat16_t> xBf16;
                        Cast<bfloat16_t, float, NormCommon::castTraitB322B16>(xBf16, y, mask0);
                        DataCopy<bfloat16_t, StoreDist::DIST_PACK_B32>(yTensorAddr + i * VL_F32 + j * formerRAlignNum,
                                                                       xBf16, mask0);
                    } else {
                        DataCopy(yTensorAddr + i * VL_F32 + j * formerRAlignNum, y, mask0);
                    }
                }
            }
        }
    }

    __aicore__ inline void CopyOutYAR(LocalTensor<T>& yOutUb, int64_t offset, uint64_t processANum,
                                      uint64_t processR0Num)
    {
        DataCopyExtParams copyInParams;
        copyInParams.blockCount = processANum;
        copyInParams.blockLen = processR0Num * sizeof(T);
        copyInParams.srcStride = 0;
        copyInParams.dstStride = (tilingData->patternR0 - processR0Num) * sizeof(T);
        DataCopyPad(yGm[offset], yOutUb, copyInParams);
    }

    __aicore__ inline void CopyOutYRAR(LocalTensor<T>& yOutUb, int64_t offset, uint64_t processANum,
                                       uint64_t processR0Num, uint64_t processR1Num)
    {
        uint64_t rProcessNumAlign = CEIL_ALIGN(processR0Num * processR1Num, T_BLOCK_ALIGN_SIZE);

        AscendC::DataCopyExtParams copyInParams;
        copyInParams.blockCount = processR1Num;
        copyInParams.blockLen = processR0Num * sizeof(T);
        copyInParams.srcStride = 0;
        copyInParams.dstStride = (tilingData->patternR0 * tilingData->patternA - processR0Num) * sizeof(T);

        uint32_t loop1Size = processANum;
        uint64_t loop1SrcStride = rProcessNumAlign * sizeof(T);
        uint64_t loop1DstStride = tilingData->patternR0 * sizeof(T);
        uint32_t loop2Size = 1;
        uint64_t loop2SrcStride = 0;
        uint64_t loop2DstStride = 0;

        AscendC::LoopModeParams LoopParams{loop1Size,      loop2Size,      loop1SrcStride,
                                           loop1DstStride, loop2SrcStride, loop2DstStride};

        AscendC::SetLoopModePara(LoopParams, DataCopyMVType::UB_TO_OUT);
        AscendC::DataCopyPad<T, PaddingMode::Compact>(yGm[offset], yOutUb, copyInParams);
        AscendC::ResetLoopModePara(DataCopyMVType::UB_TO_OUT);
    }

    /* global memory address */
    GlobalTensor<T> xGm;
    GlobalTensor<T_GAMMA> betaGm;
    GlobalTensor<T_GAMMA> gammaGm;
    GlobalTensor<T_RUNNING_MEAN> runningMeanGm;
    GlobalTensor<T_RUNNING_MEAN> runningVarGm;

    GlobalTensor<T> yGm;
    GlobalTensor<float> batchMeanGm;
    GlobalTensor<float> batchRstdGm;
    GlobalTensor<T_RUNNING_MEAN> runningMeanOutGm;
    GlobalTensor<T_RUNNING_MEAN> runningVarOutGm;
    GlobalTensor<float> meanWsp;
    GlobalTensor<float> varWsp;
    GlobalTensor<float> workspaceGm;

    const BatchNormV3RARBlockSplitRTilingData* tilingData;
    TPipe pipe;

    /* variable */
    int64_t currentR = 0;
    int64_t currentAAlign = 0;
    int64_t currentA = 0;
    int64_t rLoop = 0;
    float unbiasedEstimationCoeff = 0;

    float lastNCorrectionFactor = 0;
    float lastNFactor = 0;
    float nCorrectionFactor = 0;
    float nFactor = 0;

    uint32_t blockIdx = 0;
    uint32_t usedCoreNum = 0;

    uint64_t r1StartIdx = 0;
    uint64_t r1EndIdx = 0;
    uint64_t r0StartIdx = 0;
    uint64_t r0EndIdx = 0;
    uint64_t ubSplitAxis = 0;
    uint64_t ubOuter = 0;
    uint64_t ubInner = 0;

    uint64_t r1BlockInner = 1;
    uint64_t r0BlockInner = 1;

    uint64_t rFlodFactor;

    static constexpr uint32_t VL_F32 = VECTOR_REG_WIDTH / sizeof(float);
    static constexpr int64_t BLOCK_SIZE = Ops::Base::GetUbBlockSize();
    static constexpr int64_t T_BLOCK_ALIGN_SIZE = BLOCK_SIZE / sizeof(T);
    static constexpr int64_t FP32_BLOCK_ALIGN_SIZE = BLOCK_SIZE / sizeof(float);
    static constexpr int64_t DOUBLE_BUFFER = 2;
    static constexpr int64_t SCALE_COEF_FOUR = 4;
    static constexpr uint32_t ROW_TWO_OFFSET = 2;
    static constexpr uint32_t ROW_THREE_OFFSET = 3;
    static constexpr uint32_t ROW_FOUR_OFFSET = 4;
    static constexpr int32_t NUM_ONE = 1;
    static constexpr int32_t NUM_TWO = 2;
    static constexpr uint32_t FIRST_VCADD_RESULT_MAX_NUM = 128;
    /* ascendc variable */
    TQue<QuePosition::VECIN, 1> xQueue;
    TQue<QuePosition::VECIN, 1> gammaQueue;
    TQue<QuePosition::VECIN, 1> betaQueue;
    TQue<QuePosition::VECIN, 1> runningMeanInQueue;
    TQue<QuePosition::VECIN, 1> runningVarInQueue;

    TQue<QuePosition::VECOUT, 1> yQueue;
    TQue<QuePosition::VECOUT, 1> batchMeanQueue;
    TQue<QuePosition::VECOUT, 1> batchRstdQueue;
    TQue<QuePosition::VECOUT, 1> runningMeanOutQueue;
    TQue<QuePosition::VECOUT, 1> runningVarOutQueue;

    TBuf<TPosition::VECCALC> tmpTbuf1;
    TBuf<TPosition::VECCALC> tmpTbuf2;
    TBuf<TPosition::VECCALC> tmpTbuf3;
    TBuf<TPosition::VECCALC> countTbuf1;
    TBuf<TPosition::VECCALC> countTbuf2;
};
} // namespace BatchNormV3Ops
#endif

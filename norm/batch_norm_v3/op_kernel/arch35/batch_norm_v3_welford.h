/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file batch_norm_v3_welford.h
 * \brief
 */
#ifndef BATCH_NORM_V3_WELFORD_REGBASE_H
#define BATCH_NORM_V3_WELFORD_REGBASE_H

#include "kernel_tiling/kernel_tiling.h"
#include "kernel_operator.h"
namespace BatchNormV3Ops {
using namespace AscendC;

using AscendC::MicroAPI::LoadDist;
using AscendC::MicroAPI::MaskMergeMode;
using AscendC::MicroAPI::MaskReg;
using AscendC::MicroAPI::RegTensor;
using AscendC::MicroAPI::StoreDist;

template <typename T, typename T_RUNNING_MEAN>
class BatchNormV3Welford {
    static constexpr uint32_t BLOCK_SIZE = GetUbBlockSize();
    static constexpr uint16_t VECTOR_LENGTH = GetVRegSize();
    static constexpr uint32_t VL_FP32 = VECTOR_LENGTH / sizeof(float);
    static constexpr int32_t FLOAT_BYTE_SIZE = 4;
    static constexpr int32_t BUFFER_NUM = 2;
    static constexpr int32_t FOUR_UNROLL = 4;
    static constexpr int32_t INDEX_0 = 0;
    static constexpr int32_t INDEX_1 = 1;
    static constexpr int32_t INDEX_2 = 2;
    static constexpr int32_t INDEX_3 = 3;
    static constexpr int32_t INDEX_4 = 4;
    static constexpr int32_t INDEX_8 = 8;
    static constexpr int32_t INDEX_16 = 16;

    static constexpr int32_t DICHOTOMY_ADD_COEFF = 2;
    static constexpr float SCALAR1 = -0.5f;
    static constexpr float SCALAR2 = 1.5f;
    static constexpr float SCALAR3 = 0.5f;

    constexpr static AscendC::MicroAPI::CastTrait castTraitB162B32 = {
        AscendC::MicroAPI::RegLayout::ZERO, AscendC::MicroAPI::SatMode::UNKNOWN, MaskMergeMode::ZEROING,
        AscendC::RoundMode::UNKNOWN};

    constexpr static AscendC::MicroAPI::CastTrait castTraitB322B16 = {
        AscendC::MicroAPI::RegLayout::ZERO, AscendC::MicroAPI::SatMode::NO_SAT, MaskMergeMode::ZEROING,
        AscendC::RoundMode::CAST_RINT};

public:
    __aicore__ inline BatchNormV3Welford(){};
    __aicore__ inline uint32_t RoundUp32(uint32_t x, int32_t size)
    {
        if (size == 0) {
            return x;
        }

        int64_t blockElemNum = BLOCK_SIZE / size;
        return (x + blockElemNum - 1) / blockElemNum * blockElemNum;
    }

    __aicore__ inline BatchNormV3Welford(const BatchNormV3WelfordRegbaseTilingData* tilingData)
    {
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
    }

    __aicore__ inline void Init(
        GM_ADDR x, GM_ADDR gamma, GM_ADDR beta, GM_ADDR mean, GM_ADDR var, GM_ADDR y, GM_ADDR mean_out, GM_ADDR var_out,
        GM_ADDR batch_mean, GM_ADDR batch_rstd)
    {
        auto blockIdx = GetBlockIdx();

        oneSubMomentum = (float)1.0 - momentum;
        momentumForVar = momentum * elemNum / (elemNum - 1);
        isLastCore = (blockIdx == realCoreNum - 1) && (numLastCore != 0);
        processNumPerCore = isLastCore ? numLastCore : aBlockFactor;

        float cofFactor0 = static_cast<float>(FindCofFactor(static_cast<uint32_t>(elemNum)));
        float cofFactor1 = static_cast<float>(FindCofFactor(static_cast<uint32_t>(parallelN)));
        reduceScale0 = float(1.0) / cofFactor0;
        reduceScale1 = cofFactor0 / static_cast<float>(elemNum);
        scale0 = float(1.0) / cofFactor1;
        scale1 = cofFactor1 / static_cast<float>(parallelN);

        int64_t aGmOffset = aBlockFactor * blockIdx;
        int64_t arGmOffset = aGmOffset * r0;
        xGm.SetGlobalBuffer((__gm__ T*)x + arGmOffset);
        betaGm.SetGlobalBuffer((__gm__ T*)beta + aGmOffset);
        gammaGm.SetGlobalBuffer((__gm__ T*)gamma + aGmOffset);
        meanGm.SetGlobalBuffer((__gm__ T_RUNNING_MEAN*)mean + aGmOffset);
        varGm.SetGlobalBuffer((__gm__ T_RUNNING_MEAN*)var + aGmOffset);

        yGm.SetGlobalBuffer((__gm__ T*)y + arGmOffset);
        batchMeanGm.SetGlobalBuffer((__gm__ float*)batch_mean + aGmOffset);
        batchRstdGm.SetGlobalBuffer((__gm__ float*)batch_rstd + aGmOffset);
        outMeanGm.SetGlobalBuffer((__gm__ T_RUNNING_MEAN*)mean_out + aGmOffset);
        outVarGm.SetGlobalBuffer((__gm__ T_RUNNING_MEAN*)var_out + aGmOffset);

        int64_t aGatherLimitBetaGamaAlign = RoundUp32(aGatherLimit, sizeof(T));
        pipe.InitBuffer(betaQueue, BUFFER_NUM, aGatherLimitBetaGamaAlign * sizeof(T));
        pipe.InitBuffer(gammaQueue, BUFFER_NUM, aGatherLimitBetaGamaAlign * sizeof(T));

        int64_t aGatherLimitAlign = RoundUp32(aGatherLimit, sizeof(float));
        pipe.InitBuffer(meanQueue, BUFFER_NUM, aGatherLimitAlign * sizeof(float));
        pipe.InitBuffer(varQueue, BUFFER_NUM, aGatherLimitAlign * sizeof(float));
        pipe.InitBuffer(batchMeanQueue, BUFFER_NUM, aGatherLimitAlign * sizeof(float));
        pipe.InitBuffer(batchRstdQueue, BUFFER_NUM, aGatherLimitAlign * sizeof(float));
        pipe.InitBuffer(outMeanQueue, BUFFER_NUM, aGatherLimitAlign * sizeof(float));
        pipe.InitBuffer(outVarQueue, BUFFER_NUM, aGatherLimitAlign * sizeof(float));

        pipe.InitBuffer(xQueue, BUFFER_NUM, ubSize * sizeof(T));
        pipe.InitBuffer(yQueue, BUFFER_NUM, ubSize * sizeof(T));
        pipe.InitBuffer(tMeanBuff, ubSize * sizeof(float));
        pipe.InitBuffer(tVarBuff, ubSize * sizeof(float));
        pipe.InitBuffer(batchVarBuff, aGatherLimitAlign * sizeof(float));
        int64_t vcaddNum = RoundUp32(binaryAddQuotient / VL_FP32, sizeof(float));
        pipe.InitBuffer(dichotomyAddBuf, vcaddNum * sizeof(float));
    }

    __aicore__ inline uint32_t FindCofFactor(uint32_t n)
    {
        // 找到比n大的最邻近的二次幂数, n = 15，结果为16
        if ((n & (n - 1)) != 0) {
            uint32_t temp = n - 1;
            temp |= temp >> 1;
            temp |= temp >> INDEX_2;
            temp |= temp >> INDEX_4;
            temp |= temp >> INDEX_8;
            temp |= temp >> INDEX_16;
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

        int32_t aCycle = CeilDiv(processNumPerCore, aGatherLimit);
        int32_t aGatherTail = processNumPerCore % aGatherLimit;
        int32_t aProcessLength = aGatherLimit;

        for (int32_t i = 0; i < aCycle; i++) {
            if (aGatherTail != 0 and i == aCycle - 1) {
                aProcessLength = aGatherTail;
            }
            ProcessAParallel(i, aProcessLength);
        }
    }

    __aicore__ inline void ProcessCutR0(
        __local_mem__ float* tmpMeanLocal, __local_mem__ float* tmpVarLocal, __local_mem__ float* meanLocal,
        __local_mem__ float* rstdLocal, __local_mem__ float* varLocal, __local_mem__ float* dichotomyAddLocal,
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
                    int64_t index =
                        j * r1Factor * a0 * r0 + (aParallelIndex * aGatherLimit + i) * r0 + index_last_k * r0Factor;
                    // after ub spit, there may comes tail block at the end cycle.
                    bool isFirstStep = count == 0;
                    count = count + 1;
                    countTail = countTail + 1;
                    ProcessSplitWelfordUpdate(tmpMeanLocal, tmpVarLocal, index, nBurst, burstLen, count, isFirstStep);
                }
            }
            if (countTail == 0) {
                VFWelfordParallelFinalizeAlign(
                    meanLocal, rstdLocal, varLocal, tmpMeanLocal, tmpVarLocal, dichotomyAddLocal, parallelN,
                    binaryAddQuotient, binaryAddK, binaryAddLastNum, i, reduceScale0, reduceScale1, scale0, scale1,
                    count, eps);
            } else {
                countTail = count - countTail;
                VFWelfordParallelFinalizeNonAlign(
                    meanLocal, rstdLocal, varLocal, tmpMeanLocal, tmpVarLocal, dichotomyAddLocal, parallelN,
                    binaryAddQuotient, binaryAddK, binaryAddLastNum, i, r0Tail, reduceScale0, reduceScale1, countTail,
                    count, eps);
            }
        }
    }

    __aicore__ inline void ProcessCutR1(
        __local_mem__ float* tmpMeanLocal, __local_mem__ float* tmpVarLocal, __local_mem__ float* meanLocal,
        __local_mem__ float* rstdLocal, __local_mem__ float* varLocal, __local_mem__ float* dichotomyAddLocal,
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
                VFWelfordParallelFinalizeAlign(
                    meanLocal, rstdLocal, varLocal, tmpMeanLocal, tmpVarLocal, dichotomyAddLocal, parallelN,
                    binaryAddQuotient, binaryAddK, binaryAddLastNum, i, reduceScale0, reduceScale1, scale0, scale1,
                    count, eps);
            } else {
                countTail = count - countTail;
                uint32_t tailNum = r1Tail * r0Factor;
                VFWelfordParallelFinalizeNonAlign(
                    meanLocal, rstdLocal, varLocal, tmpMeanLocal, tmpVarLocal, dichotomyAddLocal, parallelN,
                    binaryAddQuotient, binaryAddK, binaryAddLastNum, i, tailNum, reduceScale0, reduceScale1, countTail,
                    count, eps);
            }
        }
    }

    __aicore__ inline void ProcessAParallel(int32_t aParallelIndex, int32_t aParallelLength)
    {
        // process core
        LocalTensor<float> tMeanTensor = tMeanBuff.Get<float>();
        LocalTensor<float> tVarTensor = tVarBuff.Get<float>();
        LocalTensor<float> batchVarTensor = batchVarBuff.Get<float>();
        LocalTensor<float> dichotomyAddTensor = dichotomyAddBuf.Get<float>();
        LocalTensor<float> batchMeanTensor = batchMeanQueue.AllocTensor<float>();
        LocalTensor<float> batchRstdTensor = batchRstdQueue.AllocTensor<float>();

        __local_mem__ float* batchMeanLocal = (__local_mem__ float*)batchMeanTensor.GetPhyAddr();
        __local_mem__ float* batchVarLocal = (__local_mem__ float*)batchVarTensor.GetPhyAddr();
        __local_mem__ float* batchRstdLocal = (__local_mem__ float*)batchRstdTensor.GetPhyAddr();
        __local_mem__ float* tmpMeanLocal = (__local_mem__ float*)tMeanTensor.GetPhyAddr();
        __local_mem__ float* tmpVarLocal = (__local_mem__ float*)tVarTensor.GetPhyAddr();
        __local_mem__ float* dichotomyAddLocal = (__local_mem__ float*)dichotomyAddTensor.GetPhyAddr();

        // copy small tensor to ub
        int32_t index_a = blockIdx * this->aBlockFactor + aParallelIndex * aGatherLimit;
        int32_t copyLen = aParallelLength;
        CopySmallTensor2UB(index_a, copyLen);

        // process A loop
        if (cutR1OrR0 == 0) {
            ProcessCutR0(
                tmpMeanLocal, tmpVarLocal, batchMeanLocal, batchRstdLocal, batchVarLocal, dichotomyAddLocal,
                aParallelIndex, aParallelLength);
        } else if (cutR1OrR0 == 1) {
            ProcessCutR1(
                tmpMeanLocal, tmpVarLocal, batchMeanLocal, batchRstdLocal, batchVarLocal, dichotomyAddLocal,
                aParallelIndex, aParallelLength);
        }

        LocalTensor<T_RUNNING_MEAN> meanTensor = meanQueue.DeQue<T_RUNNING_MEAN>();
        LocalTensor<T_RUNNING_MEAN> varTensor = varQueue.DeQue<T_RUNNING_MEAN>();
        LocalTensor<T_RUNNING_MEAN> outMeanTensor = outMeanQueue.AllocTensor<T_RUNNING_MEAN>();
        LocalTensor<T_RUNNING_MEAN> outVarTensor = outVarQueue.AllocTensor<T_RUNNING_MEAN>();
        __local_mem__ T_RUNNING_MEAN* meanLocal = (__local_mem__ T_RUNNING_MEAN*)meanTensor.GetPhyAddr();
        __local_mem__ T_RUNNING_MEAN* varLocal = (__local_mem__ T_RUNNING_MEAN*)varTensor.GetPhyAddr();
        __local_mem__ T_RUNNING_MEAN* outMeanLocal = (__local_mem__ T_RUNNING_MEAN*)outMeanTensor.GetPhyAddr();
        __local_mem__ T_RUNNING_MEAN* outVarLocal = (__local_mem__ T_RUNNING_MEAN*)outVarTensor.GetPhyAddr();

        uint16_t loopCount2 = CeilDiv(aParallelLength, VL_FP32);

        __local_mem__ float* batchMeanLocal2 = (__local_mem__ float*)batchMeanTensor.GetPhyAddr();
        __local_mem__ float* batchVarLocal2 = (__local_mem__ float*)batchVarTensor.GetPhyAddr();
        CalculateRunningMeanAndVar(
            batchMeanLocal2, batchVarLocal2, meanLocal, varLocal, outMeanLocal, outVarLocal, loopCount2,
            aParallelLength);

        batchMeanQueue.EnQue(batchMeanTensor);
        batchRstdQueue.EnQue(batchRstdTensor);
        outMeanQueue.EnQue(outMeanTensor);
        outVarQueue.EnQue(outVarTensor);

        CopyOutMeanAndVar2GM(index_a, aParallelLength);
        ProcessNormalize(index_a, aParallelIndex, aParallelLength);
    }

private:
    __aicore__ inline void LoadTwoTensorForDtypeTBrc(
        __local_mem__ T* src1, __local_mem__ T* src2, RegTensor<float>& dst1, RegTensor<float>& dst2, MaskReg& dst1Preg,
        MaskReg& dst2Preg, uint32_t src1Offset, uint32_t src2Offset)
    {
        if constexpr (IsSameType<T, half>::value) {
            RegTensor<half> xFp16Q;
            RegTensor<half> xFp16R;
            DataCopy<half, LoadDist::DIST_BRC_B16>(xFp16Q, ((__local_mem__ half*)(src1) + (src1Offset)));
            DataCopy<half, LoadDist::DIST_BRC_B16>(xFp16R, ((__local_mem__ half*)(src2) + (src2Offset)));
            Cast<float, half, castTraitB162B32>(dst1, xFp16Q, dst1Preg);
            Cast<float, half, castTraitB162B32>(dst2, xFp16R, dst2Preg);
        } else if constexpr (IsSameType<T, bfloat16_t>::value) {
            RegTensor<bfloat16_t> xFp16Q;
            RegTensor<bfloat16_t> xFp16R;
            DataCopy<bfloat16_t, LoadDist::DIST_BRC_B16>(xFp16Q, ((__local_mem__ bfloat16_t*)(src1) + (src1Offset)));
            DataCopy<bfloat16_t, LoadDist::DIST_BRC_B16>(xFp16R, ((__local_mem__ bfloat16_t*)(src2) + (src2Offset)));
            Cast<float, bfloat16_t, castTraitB162B32>(dst1, xFp16Q, dst1Preg);
            Cast<float, bfloat16_t, castTraitB162B32>(dst2, xFp16R, dst2Preg);
        } else {
            DataCopy<float, LoadDist::DIST_BRC_B32>(dst1, ((__local_mem__ float*)(src1) + (src1Offset)));
            DataCopy<float, LoadDist::DIST_BRC_B32>(dst2, ((__local_mem__ float*)(src2) + (src2Offset)));
        }
    }

    __aicore__ inline void LoadOneTensorForDtypeT(
        __local_mem__ T* input, RegTensor<float>& dst, MaskReg& preg, uint32_t offset)
    {
        if constexpr (IsSameType<T, half>::value) {
            RegTensor<half> xFp16;
            DataCopy<half, LoadDist::DIST_UNPACK_B16>(xFp16, ((__local_mem__ half*)(input) + (offset)));
            Cast<float, half, castTraitB162B32>(dst, xFp16, preg);
        } else if constexpr (IsSameType<T, bfloat16_t>::value) {
            RegTensor<bfloat16_t> xBf16;
            DataCopy<bfloat16_t, LoadDist::DIST_UNPACK_B16>(xBf16, ((__local_mem__ bfloat16_t*)(input) + (offset)));
            Cast<float, bfloat16_t, castTraitB162B32>(dst, xBf16, preg);
        } else {
            DataCopy(dst, ((__local_mem__ float*)(input) + (offset)));
        }
    }

    __aicore__ inline void DichotomyAdd(
        RegTensor<float>& dstReg, __local_mem__ float* src, uint16_t outerLoop, uint16_t innerLoop, uint32_t lastNum)
    {
        RegTensor<float> tmpReg1;
        RegTensor<float> tmpReg2;
        RegTensor<float> tmpReg3;
        AscendC::MicroAPI::LocalMemBar<AscendC::MicroAPI::MemType::VEC_STORE, AscendC::MicroAPI::MemType::VEC_LOAD>();
        MaskReg pregMain = AscendC::MicroAPI::CreateMask<float, AscendC::MicroAPI::MaskPattern::ALL>();
        for (uint16_t k = 0; k < outerLoop; k++) {
            innerLoop = innerLoop / DICHOTOMY_ADD_COEFF;
            for (uint16_t i = 0; i < innerLoop; i++) {
                DataCopy(tmpReg1, src + i * VL_FP32);
                DataCopy(tmpReg2, src + (i + innerLoop) * VL_FP32);
                Add(tmpReg3, tmpReg1, tmpReg2, pregMain);
                DataCopy(src + i * VL_FP32, tmpReg3, pregMain);
            }
            AscendC::MicroAPI::LocalMemBar<
                AscendC::MicroAPI::MemType::VEC_STORE, AscendC::MicroAPI::MemType::VEC_LOAD>();
        }
        uint32_t sreg0 = lastNum;
        MaskReg pregLoop = AscendC::MicroAPI::UpdateMask<float>(sreg0);
        DataCopy(tmpReg3, src);
        ReduceSum(dstReg, tmpReg3, pregLoop);
    }

    __aicore__ inline void CalRstdByHighPrecision(RegTensor<float>& var, RegTensor<float>& rstd, float epsilon)
    {
        RegTensor<float> r;
        RegTensor<float> y;
        RegTensor<float> s;
        RegTensor<float> t;
        RegTensor<float> e;
        RegTensor<float> one;
        RegTensor<float> scalar1;
        RegTensor<float> scalar2;
        RegTensor<float> t1;
        RegTensor<float> t2;
        RegTensor<float> t3;
        RegTensor<float> t4;
        MaskReg pregMerge = AscendC::MicroAPI::CreateMask<float, AscendC::MicroAPI::MaskPattern::VL1>();

        Duplicate(one, float(1.0), pregMerge);
        Duplicate(scalar1, SCALAR3, pregMerge);
        Duplicate(t1, SCALAR2, pregMerge);
        Duplicate(s, float(1.0), pregMerge);

        Adds(var, var, epsilon, pregMerge);
        Div(r, one, var, pregMerge);
        Sqrt(y, r, pregMerge);
        Muls(t, var, SCALAR1, pregMerge);
        Mul(t, t, y, pregMerge);                // -0.5 * x * y
        Mula(t1, t, y, pregMerge);              // 1.5 + (-0.5 * x * y) * y
        Mul(rstd, y, t1, pregMerge);            // y = y * (1.5 - 0.5 * x * y)
        Muls(t3, var, float(-1.0), pregMerge);  // -1 * x
        Mula(s, t3, r, pregMerge);              // 1 + (-1) * x * r
        Muls(t4, rstd, float(-1.0), pregMerge); // (-1) * y
        Mula(r, t4, rstd, pregMerge);           // r + (-1) * y * y
        Mula(s, var, r, pregMerge);             // s + x * t
        Mul(s, s, rstd, pregMerge);             // e * y
        Mula(rstd, s, scalar1, pregMerge);      // y + y * e * 0.5
    }

    __aicore__ inline void ProcessSplitWelfordUpdate(
        __local_mem__ float* tmpMeanLocal, __local_mem__ float* tmpVarLocal, int64_t index, int64_t nBurst,
        int64_t burstLen, int64_t count, bool isFirstStep)
    {
        // copy large data into ub
        CopyInX(index, nBurst, burstLen);

        LocalTensor<T> xInUb = xQueue.DeQue<T>();
        __local_mem__ T* xLocal = (__local_mem__ T*)xInUb.GetPhyAddr();
        // process welford after copy ubSize data into ub.
        float scale = (float)1.0 / static_cast<float>(count);
        uint64_t processNum = nBurst * burstLen;
        uint16_t updateLoopCount = CeilDiv(processNum, VL_FP32);
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
        LocalTensor<T> xInUb = xQueue.AllocTensor<T>();
        DataCopyPadExtParams<T> padParams;
        padParams.isPad = false;

        DataCopyExtParams dataCopyParams;
        dataCopyParams.blockCount = nBurst;
        dataCopyParams.blockLen = burstLen * sizeof(T);
        dataCopyParams.srcStride = 0;
        dataCopyParams.dstStride = 0;
        if (nBurst != 1) {
            dataCopyParams.srcStride = (a0 - 1) * r0 * sizeof(T);
        }
        DataCopyPad<T, PaddingMode::Compact>(xInUb, xGm[index], dataCopyParams, padParams);

        xQueue.EnQue(xInUb);
    }

    __aicore__ inline void CopySmallTensor2UB(const int64_t& index, const int64_t& copyLen)
    {
        LocalTensor<T> gammaInUb = gammaQueue.AllocTensor<T>();
        LocalTensor<T> betaInUb = betaQueue.AllocTensor<T>();
        LocalTensor<T_RUNNING_MEAN> meanInUb = meanQueue.AllocTensor<T_RUNNING_MEAN>();
        LocalTensor<T_RUNNING_MEAN> varInUb = varQueue.AllocTensor<T_RUNNING_MEAN>();

        DataCopyPadExtParams<T> padParamsBetaGamma;
        padParamsBetaGamma.isPad = true;
        padParamsBetaGamma.paddingValue = static_cast<T>(0.0);
        // isPad配置True，rightPadding配置0，表示自动Pad到32B对齐
        padParamsBetaGamma.rightPadding = 0;

        DataCopyExtParams dataCopyParamsBetaGamma;
        dataCopyParamsBetaGamma.blockCount = 1;
        dataCopyParamsBetaGamma.blockLen = copyLen * sizeof(T);
        dataCopyParamsBetaGamma.srcStride = 0;
        dataCopyParamsBetaGamma.dstStride = 0;

        DataCopyPad(gammaInUb, gammaGm[index], dataCopyParamsBetaGamma, padParamsBetaGamma);
        DataCopyPad(betaInUb, betaGm[index], dataCopyParamsBetaGamma, padParamsBetaGamma);

        DataCopyPadExtParams<T_RUNNING_MEAN> padParams;
        padParams.isPad = true;
        padParams.paddingValue = static_cast<T_RUNNING_MEAN>(0.0);
        // isPad配置True，rightPadding配置0，表示自动Pad到32B对齐
        padParams.rightPadding = 0;

        DataCopyExtParams dataCopyParams;
        dataCopyParams.blockCount = 1;
        dataCopyParams.blockLen = copyLen * sizeof(T_RUNNING_MEAN);
        dataCopyParams.srcStride = 0;
        dataCopyParams.dstStride = 0;

        DataCopyPad(meanInUb, meanGm[index], dataCopyParams, padParams);
        DataCopyPad(varInUb, varGm[index], dataCopyParams, padParams);

        gammaQueue.EnQue(gammaInUb);
        betaQueue.EnQue(betaInUb);
        meanQueue.EnQue(meanInUb);
        varQueue.EnQue(varInUb);
    }

    __aicore__ inline void CopyY2Gm(const int64_t& index, const int64_t& nBurst, const int64_t& burstLen)
    {
        LocalTensor<T> yOutUb = yQueue.DeQue<T>();

        DataCopyExtParams dataCopyParams;
        dataCopyParams.blockCount = nBurst;
        dataCopyParams.blockLen = burstLen * sizeof(T);
        dataCopyParams.srcStride = 0;
        dataCopyParams.dstStride = 0;
        if (nBurst != 1) {
            dataCopyParams.dstStride = (a0 - 1) * r0 * sizeof(T);
        }
        DataCopyPad<T, PaddingMode::Compact>(yGm[index], yOutUb, dataCopyParams);
        yQueue.FreeTensor(yOutUb);
    }

    __aicore__ inline void CopyOutMeanAndVar2GM(int64_t index, int64_t copyLen)
    {
        LocalTensor<T_RUNNING_MEAN> outMeanTensor = outMeanQueue.DeQue<T_RUNNING_MEAN>();
        LocalTensor<T_RUNNING_MEAN> outVarTensor = outVarQueue.DeQue<T_RUNNING_MEAN>();

        DataCopyExtParams dataCopyParams;
        dataCopyParams.blockCount = 1;
        dataCopyParams.blockLen = copyLen * sizeof(T_RUNNING_MEAN);
        dataCopyParams.srcStride = 0;
        dataCopyParams.dstStride = 0;

        DataCopyPad(outMeanGm[index], outMeanTensor, dataCopyParams);
        DataCopyPad(outVarGm[index], outVarTensor, dataCopyParams);

        outMeanQueue.FreeTensor(outMeanTensor);
        outVarQueue.FreeTensor(outVarTensor);
    }

    __aicore__ inline void CopyBatchMeanAndRstd2GM(
        int64_t index, int64_t copyLen, LocalTensor<float> batchMeanTensor, LocalTensor<float> batchRstdTensor)
    {
        DataCopyExtParams dataCopyParams;
        dataCopyParams.blockCount = 1;
        dataCopyParams.blockLen = copyLen * sizeof(float);
        dataCopyParams.srcStride = 0;
        dataCopyParams.dstStride = 0;

        DataCopyPad(batchMeanGm[index], batchMeanTensor, dataCopyParams);
        DataCopyPad(batchRstdGm[index], batchRstdTensor, dataCopyParams);

        batchMeanQueue.FreeTensor(batchMeanTensor);
        batchRstdQueue.FreeTensor(batchRstdTensor);
    }

    __aicore__ inline void CalculateRunningMeanAndVar(
        __local_mem__ float* batchMeanLocal, __local_mem__ float* batchVarLocal,
        __local_mem__ T_RUNNING_MEAN* meanLocal, __local_mem__ T_RUNNING_MEAN* varLocal,
        __local_mem__ T_RUNNING_MEAN* outMeanLocal, __local_mem__ T_RUNNING_MEAN* outVarLocal, uint16_t loopCount,
        uint32_t calLen)
    {
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
                if constexpr (!IsSameType<T_RUNNING_MEAN, float>::value) {
                    // 需要把T_RUNNING_MEAN的输入cast到float
                    AscendC::MicroAPI::RegTensor<T_RUNNING_MEAN> meanTmp;
                    AscendC::MicroAPI::DataCopy<T_RUNNING_MEAN, AscendC::MicroAPI::LoadDist::DIST_UNPACK_B16>(
                        meanTmp, ((__local_mem__ T_RUNNING_MEAN*)meanLocal + i * VL_FP32));
                    AscendC::MicroAPI::Cast<float, T_RUNNING_MEAN, castTraitB162B32>(mean, meanTmp, pregLoop);
                } else {
                    AscendC::MicroAPI::DataCopy(mean, ((__local_mem__ float*)meanLocal + i * VL_FP32));
                }
                DataCopy(batchMean, ((__local_mem__ float*)batchMeanLocal + i * VL_FP32));
                Muls(batchMean, batchMean, this->momentum, pregLoop);
                Muls(mean, mean, this->oneSubMomentum, pregLoop);
                Add(mean, mean, batchMean, pregLoop);
                if constexpr (!IsSameType<T_RUNNING_MEAN, float>::value) {
                    // 需要把float的结果cast回T_RUNNING_MEAN
                    AscendC::MicroAPI::RegTensor<T_RUNNING_MEAN> outMeanTmp;
                    AscendC::MicroAPI::Cast<T_RUNNING_MEAN, float, castTraitB322B16>(outMeanTmp, mean, pregLoop);
                    AscendC::MicroAPI::DataCopy<T_RUNNING_MEAN, AscendC::MicroAPI::StoreDist::DIST_PACK_B32>(
                        ((__local_mem__ T_RUNNING_MEAN*)outMeanLocal + i * VL_FP32), outMeanTmp, pregLoop);
                } else {
                    AscendC::MicroAPI::DataCopy(((__local_mem__ float*)outMeanLocal + i * VL_FP32), mean, pregLoop);
                }

                if constexpr (!IsSameType<T_RUNNING_MEAN, float>::value) {
                    // 需要把T_RUNNING_MEAN的输入cast到float
                    AscendC::MicroAPI::RegTensor<T_RUNNING_MEAN> varTmp;
                    AscendC::MicroAPI::DataCopy<T_RUNNING_MEAN, AscendC::MicroAPI::LoadDist::DIST_UNPACK_B16>(
                        varTmp, ((__local_mem__ T_RUNNING_MEAN*)varLocal + i * VL_FP32));
                    AscendC::MicroAPI::Cast<float, T_RUNNING_MEAN, castTraitB162B32>(var, varTmp, pregLoop);
                } else {
                    AscendC::MicroAPI::DataCopy(var, ((__local_mem__ float*)varLocal + i * VL_FP32));
                }
                DataCopy(batchVar, batchVarLocal + i * VL_FP32);
                Muls(batchVar, batchVar, this->momentumForVar, pregLoop);
                Muls(var, var, this->oneSubMomentum, pregLoop);
                Add(var, var, batchVar, pregLoop);
                if constexpr (!IsSameType<T_RUNNING_MEAN, float>::value) {
                    // 需要把float的结果cast回T_RUNNING_MEAN
                    AscendC::MicroAPI::RegTensor<T_RUNNING_MEAN> outVarTmp;
                    AscendC::MicroAPI::Cast<T_RUNNING_MEAN, float, castTraitB322B16>(outVarTmp, var, pregLoop);
                    AscendC::MicroAPI::DataCopy<T_RUNNING_MEAN, AscendC::MicroAPI::StoreDist::DIST_PACK_B32>(
                        ((__local_mem__ T_RUNNING_MEAN*)outVarLocal + i * VL_FP32), outVarTmp, pregLoop);
                } else {
                    AscendC::MicroAPI::DataCopy(((__local_mem__ float*)outVarLocal + i * VL_FP32), var, pregLoop);
                }
            }
        }
    }

    __aicore__ inline void VFWelfordParallelUpdateWithInit(
        __local_mem__ T* x1Local, __local_mem__ float* tmpMeanLocal, __local_mem__ float* tmpVarLocal, uint64_t calLen,
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
            vector_bool pregMain = pset_b8(PAT_ALL);
            vector_bool pregLoop;
            uint32_t sreg0 = calLen;
            for (uint16_t i = 0; i < loopCount; i++) {
                pregLoop = plt_b32(sreg0, POST_UPDATE);
                LoadOneTensorForDtypeT(x1Local, x1, pregLoop, i * VL_FP32);
                Duplicate(tmpMean, 0.0, pregLoop);
                // delata1 = x1 - mean
                Sub(delta1, x1, tmpMean, pregLoop);
                // delta2 = delta1 * scale
                Muls(delta2, delta1, scale, pregLoop);
                // mean = mean + delta2
                Add(tmpMean, tmpMean, delta2, pregLoop);
                DataCopy(tmpMeanLocal + i * VL_FP32, tmpMean, pregLoop);

                Duplicate(tmpVar, 0.0, pregLoop);
                // delta3 = x1 - mean
                Sub(delta3, x1, tmpMean, pregLoop);
                // delta4 = delta1 * delta3
                Mul(delat4, delta1, delta3, pregLoop);
                // var = var + delta4
                Add(tmpVar, tmpVar, delat4, pregLoop);
                DataCopy(tmpVarLocal + i * VL_FP32, tmpVar, pregLoop);
            }
        }
    }

    __aicore__ inline void VFWelfordParallelUpdate(
        __local_mem__ T* x1Local, __local_mem__ float* tmpMeanLocal, __local_mem__ float* tmpVarLocal, uint64_t calLen,
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
            MaskReg pregMain = AscendC::MicroAPI::CreateMask<float, AscendC::MicroAPI::MaskPattern::ALL>();
            MaskReg pregLoop;
            uint32_t sreg0 = calLen;
            for (uint16_t i = 0; i < loopCount; i++) {
                pregLoop = AscendC::MicroAPI::UpdateMask<float>(sreg0);
                LoadOneTensorForDtypeT(x1Local, x1, pregLoop, i * VL_FP32);

                DataCopy(tmpMean, tmpMeanLocal + i * VL_FP32);
                // delata1 = x1 - mean
                Sub(delta1, x1, tmpMean, pregLoop);
                // delta2 = delta1 * scale
                Muls(delta2, delta1, scale, pregLoop);
                // mean = mean + delta2
                Add(tmpMean, tmpMean, delta2, pregLoop);
                DataCopy(tmpMeanLocal + i * VL_FP32, tmpMean, pregLoop);

                DataCopy(tmpVar, tmpVarLocal + i * VL_FP32);
                // delta3 = x1 - mean
                Sub(delta3, x1, tmpMean, pregLoop);
                // delta4 = delta1 * delta3
                Mul(delat4, delta1, delta3, pregLoop);
                // var = var + delta4
                Add(tmpVar, tmpVar, delat4, pregLoop);
                DataCopy(tmpVarLocal + i * VL_FP32, tmpVar, pregLoop);
            }
        }
    }

    /*
      Welford Finalize对齐场景计算公式如下:
      finalize_mean = sum_fun(mean) / parallel_N
      finalize_delta = mean - finalize_mean
      finalize_delta_square = finalize_delta * finalize_delta
      M2_fixed = M2 + float(count) * finalize_delta_square
      finalize_std = sum_fun(M2_fixed) / float(parallel_N * count)
    */
    __aicore__ inline void VFWelfordParallelFinalizeAlign(
        __local_mem__ float* meanLocal, __local_mem__ float* rstdLocal, __local_mem__ float* varLocal,
        __local_mem__ float* tmpMeanLocal, __local_mem__ float* tmpVarLocal, __local_mem__ float* dichotomyAddLocal,
        uint32_t reduceCount, uint32_t dichotomyAddPower, uint32_t dichotomyAddK, uint32_t dichotomyAddLastNum,
        uint32_t offset, float reduceScale0, float reduceScale1, float scale0, float scale1, float cnt, float eps)
    {
        uint32_t dichotomyAddReminder = reduceCount - dichotomyAddPower;
        uint16_t dichotomyAddReminderLoopCount = CeilDiv(dichotomyAddReminder, VL_FP32);
        uint16_t dichotomyAddPowerLoopCount = dichotomyAddPower / VL_FP32;
        uint16_t innerLoopCountOrigin = dichotomyAddPowerLoopCount / VL_FP32;
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
            MaskReg pregLoop;
            MaskReg pregMain = AscendC::MicroAPI::CreateMask<float, AscendC::MicroAPI::MaskPattern::ALL>();
            MaskReg pregMerge = AscendC::MicroAPI::CreateMask<float, AscendC::MicroAPI::MaskPattern::VL1>();
            uint32_t sreg0 = dichotomyAddReminder;
            // 计算mean
            // PART1: 整尾块合并
            for (uint16_t i = 0; i < dichotomyAddReminderLoopCount; i++) {
                pregLoop = AscendC::MicroAPI::UpdateMask<float>(sreg0);
                DataCopy(dichotomyAddMeanL, tmpMeanLocal + i * VL_FP32);
                DataCopy(dichotomyAddMeanR, tmpMeanLocal + i * VL_FP32 + dichotomyAddPower);
                Muls(dichotomyAddMeanL, dichotomyAddMeanL, scale0, pregMain);
                Muls(dichotomyAddMeanR, dichotomyAddMeanR, scale0, pregLoop);
                Add(sumMean, dichotomyAddMeanL, dichotomyAddMeanR, pregMain);
                ReduceSum(mean, sumMean, pregMain);
                DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(dichotomyAddLocal + i, mean, pregMerge);
            }

            // PART2: 整块剩余部分vcadd回刷UB
            for (uint16_t i = 0; i < static_cast<uint16_t>(dichotomyAddPowerLoopCount - dichotomyAddReminderLoopCount);
                 i++) {
                DataCopy(dichotomyAddMeanL, tmpMeanLocal + (i + dichotomyAddReminderLoopCount) * VL_FP32);
                Muls(dichotomyAddMeanL, dichotomyAddMeanL, scale0, pregMain);
                ReduceSum(mean, dichotomyAddMeanL, pregMain);
                DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(
                    dichotomyAddLocal + dichotomyAddReminderLoopCount + i, mean, pregMerge);
            }

            DichotomyAdd(mean, dichotomyAddLocal, dichotomyAddK, innerLoopCountOrigin, dichotomyAddLastNum);
            Muls(mean, mean, scale1, pregMerge);
            DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(meanLocal + offset, mean, pregMerge);

            Duplicate(one, float(1.0), pregMain);
            Duplicate<float, AscendC::MicroAPI::HighLowPart::LOWEST, MaskMergeMode::ZEROING>(mean, mean, pregMain);
            sreg0 = dichotomyAddReminder;
            // PART1: 整尾块合并
            for (uint16_t i = 0; i < dichotomyAddReminderLoopCount; i++) {
                pregLoop = AscendC::MicroAPI::UpdateMask<float>(sreg0);
                DataCopy(dichotomyAddMeanL, tmpMeanLocal + i * VL_FP32);
                Sub(deltaL, dichotomyAddMeanL, mean, pregMain);
                Mul(deltaL, deltaL, deltaL, pregMain);
                Muls(deltaL, deltaL, cnt, pregMain);
                DataCopy(dichotomyAddVarL, tmpVarLocal + i * VL_FP32);
                Add(dichotomyAddVarL, dichotomyAddVarL, deltaL, pregMain);
                Muls(dichotomyAddVarL, dichotomyAddVarL, reduceScale0, pregMain);

                DataCopy(dichotomyAddMeanR, tmpMeanLocal + i * VL_FP32 + dichotomyAddPower);
                Sub(deltaR, dichotomyAddMeanR, mean, pregLoop);
                Mul(deltaR, deltaR, deltaR, pregLoop);
                Muls(deltaR, deltaR, cnt, pregLoop);
                DataCopy(dichotomyAddVarR, tmpVarLocal + i * VL_FP32 + dichotomyAddPower);
                Add(dichotomyAddVarR, dichotomyAddVarR, deltaR, pregLoop);
                Muls(dichotomyAddVarR, dichotomyAddVarR, reduceScale0, pregLoop);

                Add(sumVar, dichotomyAddVarL, dichotomyAddVarR, pregMain);
                ReduceSum(var, sumVar, pregMain);
                DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(dichotomyAddLocal + i, var, pregMerge);
            }

            // PART2: 整块剩余部分vcadd回刷UB
            for (uint16_t i = 0; i < static_cast<uint16_t>(dichotomyAddPowerLoopCount - dichotomyAddReminderLoopCount);
                 i++) {
                DataCopy(dichotomyAddMeanL, tmpMeanLocal + (i + dichotomyAddReminderLoopCount) * VL_FP32);
                Sub(deltaL, dichotomyAddMeanL, mean, pregMain);
                Mul(deltaL, deltaL, deltaL, pregMain);
                Muls(deltaL, deltaL, cnt, pregMain);
                DataCopy(dichotomyAddVarL, tmpVarLocal + (i + dichotomyAddReminderLoopCount) * VL_FP32);
                Add(dichotomyAddVarL, dichotomyAddVarL, deltaL, pregMain);
                Muls(dichotomyAddVarL, dichotomyAddVarL, reduceScale0, pregMain);
                ReduceSum(var, dichotomyAddVarL, pregMain);
                DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(
                    dichotomyAddLocal + dichotomyAddReminderLoopCount + i, var, pregMerge);
            }

            DichotomyAdd(var, dichotomyAddLocal, dichotomyAddK, innerLoopCountOrigin, dichotomyAddLastNum);
            Muls(var, var, reduceScale1, pregMerge);
            DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(varLocal + offset, var, pregMerge);
            CalRstdByHighPrecision(var, rstd, eps);
            DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(rstdLocal + offset, rstd, pregMerge);
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
        __local_mem__ float* meanLocal, __local_mem__ float* rstdLocal, __local_mem__ float* varLocal,
        __local_mem__ float* tmpMeanLocal, __local_mem__ float* tmpVarLocal, __local_mem__ float* dichotomyAddLocal,
        uint32_t reduceCount, uint32_t dichotomyAddPower, uint32_t dichotomyAddK, uint32_t dichotomyAddLastNum,
        uint32_t offset, uint32_t tailSize, float reduceScale0, float reduceScale1, float cnt, float tailCnt, float eps)
    {
        // cnt  是不带尾块的累加次数
        // tailCnt 是带尾块的累加次数， tailCnt >= cnt
        float coeff = tailCnt / cnt;
        float tailCountScale = tailCnt * reduceScale0;
        float countScale = cnt * reduceScale0;

        uint32_t dichotomyAddReminder = reduceCount - dichotomyAddPower;
        uint16_t dichotomyAddReminderRealLoopCount = CeilDiv(dichotomyAddReminder, VL_FP32);
        uint16_t dichotomyAddPowerLoopCount = dichotomyAddPower / VL_FP32;
        uint16_t innerLoopCountOrigin = dichotomyAddPowerLoopCount / VL_FP32;

        uint32_t welfordDiff = tailSize - dichotomyAddPower;
        uint16_t welfordDiffLoopCount = welfordDiff / VL_FP32;
        uint32_t welfordDiffReminder = welfordDiff - welfordDiffLoopCount * VL_FP32;
        uint32_t welfordDiffReminderAlign = welfordDiffReminder == 0 ? 0 : VL_FP32;
        uint16_t welfordReminderLoopCount = welfordDiffReminderAlign / VL_FP32;

        uint32_t dichotomyAddReminderAfterSplit =
            dichotomyAddReminder - (welfordDiffLoopCount + welfordReminderLoopCount) * VL_FP32;
        uint16_t dichotomyAddReminderLoopCount = CeilDiv(dichotomyAddReminderAfterSplit, VL_FP32);
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

            MaskReg pregLoop;
            MaskReg pregLoop1;
            MaskReg pregMain = AscendC::MicroAPI::CreateMask<float, AscendC::MicroAPI::MaskPattern::ALL>();
            MaskReg pregMerge = AscendC::MicroAPI::CreateMask<float, AscendC::MicroAPI::MaskPattern::VL1>();
            uint32_t sreg0;

            // 整块使用tailCountScale,尾块使用tailCountScale
            for (uint16_t i = 0; i < welfordDiffLoopCount; i++) {
                DataCopy(dichotomyAddMeanL, tmpMeanLocal + i * VL_FP32);
                DataCopy(dichotomyAddMeanR, tmpMeanLocal + i * VL_FP32 + dichotomyAddPower);
                Muls(dichotomyAddMeanL, dichotomyAddMeanL, tailCountScale, pregMain);
                Muls(dichotomyAddMeanR, dichotomyAddMeanR, tailCountScale, pregMain);
                Add(sumMean, dichotomyAddMeanL, dichotomyAddMeanR, pregMain);
                ReduceSum(mean, sumMean, pregMain);
                DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(dichotomyAddLocal + i, mean, pregMerge);
            }

            // 处理welford第一次非对齐点, 整块使用tailCountScale,尾块部分使用tailCountScale, 部分使用countScale
            sreg0 = dichotomyAddReminder - welfordDiffLoopCount * VL_FP32;
            uint32_t sreg1 = welfordDiffReminder;
            for (uint16_t i = 0; i < welfordReminderLoopCount; i++) {
                pregLoop = AscendC::MicroAPI::UpdateMask<float>(sreg0);
                pregLoop1 = AscendC::MicroAPI::UpdateMask<float>(sreg1);
                DataCopy(dichotomyAddMeanL, tmpMeanLocal + (i + welfordDiffLoopCount) * VL_FP32);
                DataCopy(dichotomyAddMeanR, tmpMeanLocal + (i + welfordDiffLoopCount) * VL_FP32 + dichotomyAddPower);
                Muls(dichotomyAddMeanL, dichotomyAddMeanL, tailCountScale, pregMain);
                Muls(dichotomyAddMeanR, dichotomyAddMeanR, countScale, pregLoop);
                Muls(tmp, dichotomyAddMeanR, coeff, pregLoop1);
                Copy<float, MaskMergeMode::MERGING>(dichotomyAddMeanR, tmp, pregLoop1);
                Add(sumMean, dichotomyAddMeanL, dichotomyAddMeanR, pregMain);
                ReduceSum(mean, sumMean, pregMain);
                DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(
                    dichotomyAddLocal + i + welfordDiffLoopCount, mean, pregMerge);
            }

            // 整块使用tailCountScale,尾块使用countScale
            for (uint16_t i = 0; i < dichotomyAddReminderLoopCount; i++) {
                pregLoop = AscendC::MicroAPI::UpdateMask<float>(sreg0);
                DataCopy(
                    dichotomyAddMeanL, tmpMeanLocal + (i + welfordDiffLoopCount) * VL_FP32 + welfordDiffReminderAlign);
                DataCopy(
                    dichotomyAddMeanR,
                    tmpMeanLocal + (i + welfordDiffLoopCount) * VL_FP32 + welfordDiffReminderAlign + dichotomyAddPower);
                Muls(dichotomyAddMeanL, dichotomyAddMeanL, tailCountScale, pregMain);
                Muls(dichotomyAddMeanR, dichotomyAddMeanR, countScale, pregLoop);
                Add(sumMean, dichotomyAddMeanL, dichotomyAddMeanR, pregMain);
                ReduceSum(mean, sumMean, pregMain);
                DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(
                    dichotomyAddLocal + i + welfordDiffLoopCount + welfordReminderLoopCount, mean, pregMerge);
            }
            // PART2: 整块剩余部分vcadd回刷UB,使用tailCountScale
            for (uint16_t i = 0;
                 i < static_cast<uint16_t>(dichotomyAddPowerLoopCount - dichotomyAddReminderRealLoopCount); i++) {
                DataCopy(dichotomyAddMeanL, tmpMeanLocal + (i + dichotomyAddReminderRealLoopCount) * VL_FP32);
                Muls(dichotomyAddMeanL, dichotomyAddMeanL, tailCountScale, pregMain);
                ReduceSum(mean, dichotomyAddMeanL, pregMain);
                DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(
                    dichotomyAddLocal + dichotomyAddReminderRealLoopCount + i, mean, pregMerge);
            }
            DichotomyAdd(mean, dichotomyAddLocal, dichotomyAddK, innerLoopCountOrigin, dichotomyAddLastNum);
            Muls(mean, mean, reduceScale1, pregMerge);
            DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(meanLocal + offset, mean, pregMerge);

            // 计算rstd
            Duplicate(one, float(1.0), pregMain);
            Duplicate<float, AscendC::MicroAPI::HighLowPart::LOWEST, MaskMergeMode::ZEROING>(mean, mean, pregMain);
            for (uint16_t i = 0; i < welfordDiffLoopCount; i++) {
                DataCopy(dichotomyAddMeanL, tmpMeanLocal + i * VL_FP32);
                Sub(deltaL, dichotomyAddMeanL, mean, pregMain);
                Mul(deltaL, deltaL, deltaL, pregMain);
                Muls(deltaL, deltaL, tailCnt, pregMain);
                DataCopy(dichotomyAddMeanR, tmpMeanLocal + i * VL_FP32 + dichotomyAddPower);
                Sub(deltaR, dichotomyAddMeanR, mean, pregMain);
                Mul(deltaR, deltaR, deltaR, pregMain);
                Muls(deltaR, deltaR, tailCnt, pregMain);

                DataCopy(dichotomyAddVarL, tmpVarLocal + i * VL_FP32);
                Add(dichotomyAddVarL, dichotomyAddVarL, deltaL, pregMain);
                Muls(dichotomyAddVarL, dichotomyAddVarL, reduceScale0, pregMain);
                DataCopy(dichotomyAddVarR, tmpVarLocal + i * VL_FP32 + dichotomyAddPower);
                Add(dichotomyAddVarR, dichotomyAddVarR, deltaR, pregMain);
                Muls(dichotomyAddVarR, dichotomyAddVarR, reduceScale0, pregMain);

                Add(sumVar, dichotomyAddVarL, dichotomyAddVarR, pregMain);
                ReduceSum(var, sumVar, pregMain);
                DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(dichotomyAddLocal + i, var, pregMerge);
            }
            sreg0 = dichotomyAddReminder - welfordDiffLoopCount * VL_FP32;
            sreg1 = welfordDiffReminder;
            for (uint16_t i = 0; i < welfordReminderLoopCount; i++) {
                pregLoop = AscendC::MicroAPI::UpdateMask<float>(sreg0);
                pregLoop1 = AscendC::MicroAPI::UpdateMask<float>(sreg1);
                DataCopy(dichotomyAddMeanL, tmpMeanLocal + (i + welfordDiffLoopCount) * VL_FP32);
                Sub(deltaL, dichotomyAddMeanL, mean, pregMain);
                Mul(deltaL, deltaL, deltaL, pregMain);
                Muls(deltaL, deltaL, tailCnt, pregMain);
                DataCopy(dichotomyAddMeanR, tmpMeanLocal + (i + welfordDiffLoopCount) * VL_FP32 + dichotomyAddPower);
                Sub(deltaR, dichotomyAddMeanR, mean, pregLoop);
                Mul(deltaR, deltaR, deltaR, pregLoop);
                Muls(deltaR, deltaR, cnt, pregLoop);
                Muls(tmp, deltaR, coeff, pregLoop1);
                Copy<float, MaskMergeMode::MERGING>(deltaR, tmp, pregLoop1);

                DataCopy(dichotomyAddVarL, tmpVarLocal + (i + welfordDiffLoopCount) * VL_FP32);
                Add(dichotomyAddVarL, dichotomyAddVarL, deltaL, pregMain);
                Muls(dichotomyAddVarL, dichotomyAddVarL, reduceScale0, pregMain);
                DataCopy(dichotomyAddVarR, tmpVarLocal + (i + welfordDiffLoopCount) * VL_FP32 + dichotomyAddPower);
                Add(dichotomyAddVarR, dichotomyAddVarR, deltaR, pregLoop);
                Muls(dichotomyAddVarR, dichotomyAddVarR, reduceScale0, pregLoop);

                Add(sumVar, dichotomyAddVarL, dichotomyAddVarR, pregMain);
                ReduceSum(var, sumVar, pregMain);
                DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(
                    dichotomyAddLocal + i + welfordDiffLoopCount, var, pregMerge);
            }

            for (uint16_t i = 0; i < dichotomyAddReminderLoopCount; i++) {
                pregLoop = AscendC::MicroAPI::UpdateMask<float>(sreg0);
                DataCopy(
                    dichotomyAddMeanL, tmpMeanLocal + (i + welfordDiffLoopCount) * VL_FP32 + welfordDiffReminderAlign);
                Sub(deltaL, dichotomyAddMeanL, mean, pregMain);
                Mul(deltaL, deltaL, deltaL, pregMain);
                Muls(deltaL, deltaL, tailCnt, pregMain);
                DataCopy(
                    dichotomyAddMeanR,
                    tmpMeanLocal + (i + welfordDiffLoopCount) * VL_FP32 + welfordDiffReminderAlign + dichotomyAddPower);
                Sub(deltaR, dichotomyAddMeanR, mean, pregLoop);
                Mul(deltaR, deltaR, deltaR, pregLoop);
                Muls(deltaR, deltaR, cnt, pregLoop);

                DataCopy(
                    dichotomyAddVarL, tmpVarLocal + (i + welfordDiffLoopCount) * VL_FP32 + welfordDiffReminderAlign);
                Add(dichotomyAddVarL, dichotomyAddVarL, deltaL, pregMain);
                Muls(dichotomyAddVarL, dichotomyAddVarL, reduceScale0, pregMain);
                DataCopy(
                    dichotomyAddVarR,
                    tmpVarLocal + (i + welfordDiffLoopCount) * VL_FP32 + welfordDiffReminderAlign + dichotomyAddPower);
                Add(dichotomyAddVarR, dichotomyAddVarR, deltaR, pregLoop);
                Muls(dichotomyAddVarR, dichotomyAddVarR, reduceScale0, pregLoop);
                Add(sumVar, dichotomyAddVarL, dichotomyAddVarR, pregMain);
                ReduceSum(var, sumVar, pregMain);
                DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(
                    dichotomyAddLocal + i + welfordDiffLoopCount + welfordReminderLoopCount, var, pregMerge);
            }
            for (uint16_t i = 0;
                 i < static_cast<uint16_t>(dichotomyAddPowerLoopCount - dichotomyAddReminderRealLoopCount); i++) {
                DataCopy(dichotomyAddMeanL, tmpMeanLocal + (i + dichotomyAddReminderRealLoopCount) * VL_FP32);
                Sub(deltaL, dichotomyAddMeanL, mean, pregMain);
                Mul(deltaL, deltaL, deltaL, pregMain);
                Muls(deltaL, deltaL, tailCnt, pregMain);
                DataCopy(dichotomyAddVarL, tmpVarLocal + (i + dichotomyAddReminderRealLoopCount) * VL_FP32);
                Add(dichotomyAddVarL, dichotomyAddVarL, deltaL, pregMain);
                Muls(dichotomyAddVarL, dichotomyAddVarL, reduceScale0, pregMain);
                ReduceSum(var, dichotomyAddVarL, pregMain);
                DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(
                    dichotomyAddLocal + dichotomyAddReminderRealLoopCount + i, var, pregMerge);
            }
            DichotomyAdd(var, dichotomyAddLocal, dichotomyAddK, innerLoopCountOrigin, dichotomyAddLastNum);
            Muls(var, var, reduceScale1, pregMerge);
            DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(varLocal + offset, var, pregMerge);
            CalRstdByHighPrecision(var, rstd, eps);
            DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(rstdLocal + offset, rstd, pregMerge);
        }
    }

    // welford整块小于二分累加整块，并且小于等于二分累加尾块向上对齐
    __aicore__ inline void VFWelfordParallelFinalizeNonAlignSituation2(
        __local_mem__ float* meanLocal, __local_mem__ float* rstdLocal, __local_mem__ float* varLocal,
        __local_mem__ float* tmpMeanLocal, __local_mem__ float* tmpVarLocal, __local_mem__ float* dichotomyAddLocal,
        uint32_t reduceCount, uint32_t dichotomyAddPower, uint32_t dichotomyAddK, uint32_t dichotomyAddLastNum,
        uint32_t offset, uint32_t tailSize, float reduceScale0, float reduceScale1, float cnt, float tailCnt, float eps)
    {
        float coeff = tailCnt / cnt;
        float tailCountScale = tailCnt * reduceScale0;
        float countScale = cnt * reduceScale0;

        uint32_t dichotomyAddReminder = reduceCount - dichotomyAddPower;
        uint16_t welfordDiffLoopCount = tailSize / VL_FP32;
        uint32_t welfordDiffReminder = tailSize - welfordDiffLoopCount * VL_FP32;
        uint32_t welfordDiffReminderAlign = welfordDiffReminder == 0 ? 0 : VL_FP32;
        uint16_t welfordReminderLoopCount = welfordDiffReminderAlign / VL_FP32;

        uint16_t dichotomyAddReminderRealLoopCount = CeilDiv(dichotomyAddReminder, VL_FP32);
        uint16_t dichotomyAddPowerLoopCount = dichotomyAddPower / VL_FP32;
        uint32_t tmpReduceCount = dichotomyAddPower / VL_FP32;
        uint16_t innerLoopCountOrigin = tmpReduceCount / VL_FP32;

        uint32_t dichotomyAddReminderAfterSplit =
            dichotomyAddReminder - welfordDiffLoopCount * VL_FP32 - welfordDiffReminderAlign;
        uint16_t dichotomyAddReminderLoopCount = CeilDiv(dichotomyAddReminderAfterSplit, VL_FP32);
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

            MaskReg pregLoop;
            MaskReg pregLoop1;
            MaskReg pregMain = AscendC::MicroAPI::CreateMask<float, AscendC::MicroAPI::MaskPattern::ALL>();
            MaskReg pregMerge = AscendC::MicroAPI::CreateMask<float, AscendC::MicroAPI::MaskPattern::VL1>();
            uint32_t sreg0;
            uint32_t sreg1;

            // 整块使用tailCountScale,尾块使用countScale
            for (uint16_t i = 0; i < welfordDiffLoopCount; i++) {
                DataCopy(dichotomyAddMeanL, tmpMeanLocal + i * VL_FP32);
                DataCopy(dichotomyAddMeanR, tmpMeanLocal + i * VL_FP32 + dichotomyAddPower);
                Muls(dichotomyAddMeanL, dichotomyAddMeanL, tailCountScale, pregMain);
                Muls(dichotomyAddMeanR, dichotomyAddMeanR, countScale, pregMain);
                Add(sumMean, dichotomyAddMeanL, dichotomyAddMeanR, pregMain);
                ReduceSum(mean, sumMean, pregMain);
                DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(dichotomyAddLocal + i, mean, pregMerge);
            }

            // 处理welford第一次非对齐点, 尾块使用countScale,整块部分使用tailCountScale, 部分使用countScale
            sreg0 = dichotomyAddReminder - welfordDiffLoopCount * VL_FP32;
            sreg1 = welfordDiffReminder;
            for (uint16_t i = 0; i < welfordReminderLoopCount; i++) {
                pregLoop = AscendC::MicroAPI::UpdateMask<float>(sreg0);
                pregLoop1 = AscendC::MicroAPI::UpdateMask<float>(sreg1);
                DataCopy(dichotomyAddMeanL, tmpMeanLocal + (i + welfordDiffLoopCount) * VL_FP32);
                DataCopy(dichotomyAddMeanR, tmpMeanLocal + (i + welfordDiffLoopCount) * VL_FP32 + dichotomyAddPower);
                Muls(dichotomyAddMeanL, dichotomyAddMeanL, countScale, pregMain);
                Muls(dichotomyAddMeanR, dichotomyAddMeanR, countScale, pregLoop);
                Muls(tmp, dichotomyAddMeanL, coeff, pregLoop1);
                Copy<float, MaskMergeMode::MERGING>(dichotomyAddMeanL, tmp, pregLoop1);
                Add(sumMean, dichotomyAddMeanL, dichotomyAddMeanR, pregMain);
                ReduceSum(mean, sumMean, pregMain);
                DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(
                    dichotomyAddLocal + i + welfordDiffLoopCount, mean, pregMerge);
            }

            // 整块使用countScale,尾块使用countScale
            for (uint16_t i = 0; i < dichotomyAddReminderLoopCount; i++) {
                pregLoop = AscendC::MicroAPI::UpdateMask<float>(sreg0);
                DataCopy(
                    dichotomyAddMeanL, tmpMeanLocal + (i + welfordDiffLoopCount) * VL_FP32 + welfordDiffReminderAlign);
                DataCopy(
                    dichotomyAddMeanR,
                    tmpMeanLocal + (i + welfordDiffLoopCount) * VL_FP32 + welfordDiffReminderAlign + dichotomyAddPower);
                Muls(dichotomyAddMeanL, dichotomyAddMeanL, countScale, pregMain);
                Muls(dichotomyAddMeanR, dichotomyAddMeanR, countScale, pregLoop);
                Add(sumMean, dichotomyAddMeanL, dichotomyAddMeanR, pregMain);
                ReduceSum(mean, sumMean, pregMain);
                DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(
                    dichotomyAddLocal + i + welfordDiffLoopCount + welfordReminderLoopCount, mean, pregMerge);
            }
            // PART2: 整块剩余部分vcadd回刷UB,使用countScale
            for (uint16_t i = 0;
                 i < static_cast<uint16_t>(dichotomyAddPowerLoopCount - dichotomyAddReminderRealLoopCount); i++) {
                DataCopy(dichotomyAddMeanL, tmpMeanLocal + (i + dichotomyAddReminderRealLoopCount) * VL_FP32);
                Muls(dichotomyAddMeanL, dichotomyAddMeanL, countScale, pregMain);
                ReduceSum(mean, dichotomyAddMeanL, pregMain);
                DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(
                    dichotomyAddLocal + dichotomyAddReminderRealLoopCount + i, mean, pregMerge);
            }
            DichotomyAdd(mean, dichotomyAddLocal, dichotomyAddK, innerLoopCountOrigin, dichotomyAddLastNum);
            Muls(mean, mean, reduceScale1, pregMerge);
            DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(meanLocal + offset, mean, pregMerge);

            // 计算rstd
            Duplicate(one, float(1.0), pregMain);
            Duplicate<float, AscendC::MicroAPI::HighLowPart::LOWEST, MaskMergeMode::ZEROING>(mean, mean, pregMain);
            for (uint16_t i = 0; i < welfordDiffLoopCount; i++) {
                DataCopy(dichotomyAddMeanL, tmpMeanLocal + i * VL_FP32);
                Sub(deltaL, dichotomyAddMeanL, mean, pregMain);
                Mul(deltaL, deltaL, deltaL, pregMain);
                Muls(deltaL, deltaL, tailCnt, pregMain);
                DataCopy(dichotomyAddMeanR, tmpMeanLocal + i * VL_FP32 + dichotomyAddPower);
                Sub(deltaR, dichotomyAddMeanR, mean, pregMain);
                Mul(deltaR, deltaR, deltaR, pregMain);
                Muls(deltaR, deltaR, cnt, pregMain);

                DataCopy(dichotomyAddVarL, tmpVarLocal + i * VL_FP32);
                Add(dichotomyAddVarL, dichotomyAddVarL, deltaL, pregMain);
                Muls(dichotomyAddVarL, dichotomyAddVarL, reduceScale0, pregMain);
                DataCopy(dichotomyAddVarR, tmpVarLocal + i * VL_FP32 + dichotomyAddPower);
                Add(dichotomyAddVarR, dichotomyAddVarR, deltaR, pregMain);
                Muls(dichotomyAddVarR, dichotomyAddVarR, reduceScale0, pregMain);

                Add(sumVar, dichotomyAddVarL, dichotomyAddVarR, pregMain);
                ReduceSum(var, sumVar, pregMain);
                DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(dichotomyAddLocal + i, var, pregMerge);
            }
            sreg0 = dichotomyAddReminder - welfordDiffLoopCount * VL_FP32;
            sreg1 = welfordDiffReminder;
            for (uint16_t i = 0; i < welfordReminderLoopCount; i++) {
                pregLoop = AscendC::MicroAPI::UpdateMask<float>(sreg0);
                pregLoop1 = AscendC::MicroAPI::UpdateMask<float>(sreg1);
                DataCopy(dichotomyAddMeanL, tmpMeanLocal + (i + welfordDiffLoopCount) * VL_FP32);
                Sub(deltaL, dichotomyAddMeanL, mean, pregMain);
                Mul(deltaL, deltaL, deltaL, pregMain);
                Muls(deltaL, deltaL, cnt, pregMain);
                DataCopy(dichotomyAddMeanR, tmpMeanLocal + (i + welfordDiffLoopCount) * VL_FP32 + dichotomyAddPower);
                Sub(deltaR, dichotomyAddMeanR, mean, pregLoop);
                Mul(deltaR, deltaR, deltaR, pregLoop);
                Muls(deltaR, deltaR, cnt, pregLoop);
                Muls(tmp, deltaL, coeff, pregLoop1);
                Copy<float, MaskMergeMode::MERGING>(deltaL, tmp, pregLoop1);

                DataCopy(dichotomyAddVarL, tmpVarLocal + (i + welfordDiffLoopCount) * VL_FP32);
                Add(dichotomyAddVarL, dichotomyAddVarL, deltaL, pregMain);
                Muls(dichotomyAddVarL, dichotomyAddVarL, reduceScale0, pregMain);
                DataCopy(dichotomyAddVarR, tmpVarLocal + (i + welfordDiffLoopCount) * VL_FP32 + dichotomyAddPower);
                Add(dichotomyAddVarR, dichotomyAddVarR, deltaR, pregLoop);
                Muls(dichotomyAddVarR, dichotomyAddVarR, reduceScale0, pregLoop);

                Add(sumVar, dichotomyAddVarL, dichotomyAddVarR, pregMain);
                ReduceSum(var, sumVar, pregMain);
                DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(
                    dichotomyAddLocal + i + welfordDiffLoopCount, var, pregMerge);
            }

            for (uint16_t i = 0; i < dichotomyAddReminderLoopCount; i++) {
                pregLoop = AscendC::MicroAPI::UpdateMask<float>(sreg0);
                DataCopy(
                    dichotomyAddMeanL, tmpMeanLocal + (i + welfordDiffLoopCount) * VL_FP32 + welfordDiffReminderAlign);
                Sub(deltaL, dichotomyAddMeanL, mean, pregMain);
                Mul(deltaL, deltaL, deltaL, pregMain);
                Muls(deltaL, deltaL, cnt, pregMain);
                DataCopy(
                    dichotomyAddMeanR,
                    tmpMeanLocal + (i + welfordDiffLoopCount) * VL_FP32 + welfordDiffReminderAlign + dichotomyAddPower);
                Sub(deltaR, dichotomyAddMeanR, mean, pregLoop);
                Mul(deltaR, deltaR, deltaR, pregLoop);
                Muls(deltaR, deltaR, cnt, pregLoop);

                DataCopy(
                    dichotomyAddVarL, tmpVarLocal + (i + welfordDiffLoopCount) * VL_FP32 + welfordDiffReminderAlign);
                Add(dichotomyAddVarL, dichotomyAddVarL, deltaL, pregMain);
                Muls(dichotomyAddVarL, dichotomyAddVarL, reduceScale0, pregMain);
                DataCopy(
                    dichotomyAddVarR,
                    tmpVarLocal + (i + welfordDiffLoopCount) * VL_FP32 + welfordDiffReminderAlign + dichotomyAddPower);
                Add(dichotomyAddVarR, dichotomyAddVarR, deltaR, pregLoop);
                Muls(dichotomyAddVarR, dichotomyAddVarR, reduceScale0, pregLoop);
                Add(sumVar, dichotomyAddVarL, dichotomyAddVarR, pregMain);
                ReduceSum(var, sumVar, pregMain);
                DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(
                    dichotomyAddLocal + i + welfordDiffLoopCount + welfordReminderLoopCount, var, pregMerge);
            }

            for (uint16_t i = 0;
                 i < static_cast<uint16_t>(dichotomyAddPowerLoopCount - dichotomyAddReminderRealLoopCount); i++) {
                DataCopy(dichotomyAddMeanL, tmpMeanLocal + (i + dichotomyAddReminderRealLoopCount) * VL_FP32);
                Sub(deltaL, dichotomyAddMeanL, mean, pregMain);
                Mul(deltaL, deltaL, deltaL, pregMain);
                Muls(deltaL, deltaL, cnt, pregMain);
                DataCopy(dichotomyAddVarL, tmpVarLocal + (i + dichotomyAddReminderRealLoopCount) * VL_FP32);
                Add(dichotomyAddVarL, dichotomyAddVarL, deltaL, pregMain);
                Muls(dichotomyAddVarL, dichotomyAddVarL, reduceScale0, pregMain);
                ReduceSum(var, dichotomyAddVarL, pregMain);
                DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(
                    dichotomyAddLocal + dichotomyAddReminderRealLoopCount + i, var, pregMerge);
            }
            DichotomyAdd(var, dichotomyAddLocal, dichotomyAddK, innerLoopCountOrigin, dichotomyAddLastNum);
            Muls(var, var, reduceScale1, pregMerge);
            DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(varLocal + offset, var, pregMerge);
            CalRstdByHighPrecision(var, rstd, eps);
            DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(rstdLocal + offset, rstd, pregMerge);
        }
    }

    // 场景3：welford整块小于二分累加整块，并且大于二分累加尾块向上对齐
    __aicore__ inline void VFWelfordParallelFinalizeNonAlignSituation3(
        __local_mem__ float* meanLocal, __local_mem__ float* rstdLocal, __local_mem__ float* varLocal,
        __local_mem__ float* tmpMeanLocal, __local_mem__ float* tmpVarLocal, __local_mem__ float* dichotomyAddLocal,
        uint32_t reduceCount, uint32_t dichotomyAddPower, uint32_t dichotomyAddK, uint32_t dichotomyAddLastNum,
        uint32_t offset, uint32_t tailSize, float reduceScale0, float reduceScale1, float cnt, float tailCnt, float eps)
    {
        float coeff = tailCnt / cnt;
        float tailCountScale = tailCnt * reduceScale0;
        float countScale = cnt * reduceScale0;

        // 二分累加
        uint32_t dichotomyAddReminder = reduceCount - dichotomyAddPower;
        uint16_t dichotomyAddReminderLoopCount = CeilDiv(dichotomyAddReminder, VL_FP32);
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
        uint32_t dichotomyAddPowerOffset =
            dichotomyAddReminderRoundUp + welfordDiffLoopCount * VL_FP32 + welfordDiffReminderAlign;

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

            MaskReg pregLoop;
            MaskReg pregMain = AscendC::MicroAPI::CreateMask<float, AscendC::MicroAPI::MaskPattern::ALL>();
            MaskReg pregMerge = AscendC::MicroAPI::CreateMask<float, AscendC::MicroAPI::MaskPattern::VL1>();
            uint32_t sreg0 = dichotomyAddReminder;
            // 整块使用tailCountScale, 尾块使用CountScale
            for (uint16_t i = 0; i < dichotomyAddReminderLoopCount; i++) {
                pregLoop = AscendC::MicroAPI::UpdateMask<float>(sreg0);
                DataCopy(dichotomyAddMeanL, tmpMeanLocal + i * VL_FP32);
                DataCopy(dichotomyAddMeanR, tmpMeanLocal + i * VL_FP32 + dichotomyAddPower);
                Muls(dichotomyAddMeanL, dichotomyAddMeanL, tailCountScale, pregMain);
                Muls(dichotomyAddMeanR, dichotomyAddMeanR, countScale, pregLoop);
                Add(sumMean, dichotomyAddMeanL, dichotomyAddMeanR, pregMain);
                ReduceSum(mean, sumMean, pregMain);
                DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(dichotomyAddLocal + i, mean, pregMerge);
            }

            // 剩余整块需要拆分成多部分
            // 整块剩余部分回刷UB，整块使用tailCountScale
            for (uint16_t i = 0; i < welfordDiffLoopCount; i++) {
                DataCopy(dichotomyAddMeanL, tmpMeanLocal + i * VL_FP32 + dichotomyAddReminderRoundUp);
                Muls(dichotomyAddMeanL, dichotomyAddMeanL, tailCountScale, pregMain);
                ReduceSum(mean, dichotomyAddMeanL, pregMain);
                DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(
                    dichotomyAddLocal + dichotomyAddReminderLoopCount + i, mean, pregMerge);
            }

            sreg0 = welfordDiffReminder;
            for (uint16_t i = 0; i < welfordReminderLoopCount; i++) {
                pregLoop = AscendC::MicroAPI::UpdateMask<float>(sreg0);
                DataCopy(
                    dichotomyAddMeanL,
                    tmpMeanLocal + (i + welfordDiffLoopCount) * VL_FP32 + dichotomyAddReminderRoundUp);
                Muls(dichotomyAddMeanL, dichotomyAddMeanL, countScale, pregMain);
                Muls(tmp, dichotomyAddMeanL, coeff, pregLoop);
                Copy<float, MaskMergeMode::MERGING>(dichotomyAddMeanL, tmp, pregLoop);
                ReduceSum(mean, dichotomyAddMeanL, pregMain);
                DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(
                    dichotomyAddLocal + dichotomyAddReminderLoopCount + welfordDiffLoopCount + i, mean, pregMerge);
            }

            for (uint16_t i = 0; i < dichotomyAddPowerRemainLoopCount; i++) {
                DataCopy(dichotomyAddMeanL, tmpMeanLocal + i * VL_FP32 + dichotomyAddPowerOffset);
                Muls(dichotomyAddMeanL, dichotomyAddMeanL, countScale, pregMain);
                ReduceSum(mean, dichotomyAddMeanL, pregMain);
                DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(
                    dichotomyAddLocal + dichotomyAddReminderLoopCount + welfordDiffLoopCount +
                        welfordReminderLoopCount + i,
                    mean, pregMerge);
            }

            DichotomyAdd(mean, dichotomyAddLocal, dichotomyAddK, innerLoopCountOrigin, dichotomyAddLastNum);
            Muls(mean, mean, reduceScale1, pregMerge);
            DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(meanLocal + offset, mean, pregMerge);

            // 计算rstd
            Duplicate(one, float(1.0), pregMain);
            Duplicate<float, AscendC::MicroAPI::HighLowPart::LOWEST, MaskMergeMode::ZEROING>(mean, mean, pregMain);
            // 整块使用tailCountScale, 尾块使用CountScale
            sreg0 = dichotomyAddReminder;
            for (uint16_t i = 0; i < dichotomyAddReminderLoopCount; i++) {
                pregLoop = AscendC::MicroAPI::UpdateMask<float>(sreg0);
                DataCopy(dichotomyAddMeanL, tmpMeanLocal + i * VL_FP32);
                Sub(deltaL, dichotomyAddMeanL, mean, pregMain);
                Mul(deltaL, deltaL, deltaL, pregMain);
                Muls(deltaL, deltaL, tailCnt, pregMain);
                DataCopy(dichotomyAddVarL, tmpVarLocal + i * VL_FP32);
                Add(dichotomyAddVarL, dichotomyAddVarL, deltaL, pregMain);
                Muls(dichotomyAddVarL, dichotomyAddVarL, reduceScale0, pregMain);

                DataCopy(dichotomyAddMeanR, tmpMeanLocal + i * VL_FP32 + dichotomyAddPower);
                Sub(deltaR, dichotomyAddMeanR, mean, pregLoop);
                Mul(deltaR, deltaR, deltaR, pregLoop);
                Muls(deltaR, deltaR, cnt, pregLoop);
                DataCopy(dichotomyAddVarR, tmpVarLocal + i * VL_FP32 + dichotomyAddPower);
                Add(dichotomyAddVarR, dichotomyAddVarR, deltaR, pregLoop);
                Muls(dichotomyAddVarR, dichotomyAddVarR, reduceScale0, pregLoop);

                Add(sumVar, dichotomyAddVarL, dichotomyAddVarR, pregMain);
                ReduceSum(var, sumVar, pregMain);
                DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(dichotomyAddLocal + i, var, pregMerge);
            }

            // 整块剩余部分回刷UB，整块使用tailCountScale
            for (uint16_t i = 0; i < welfordDiffLoopCount; i++) {
                DataCopy(dichotomyAddMeanL, tmpMeanLocal + i * VL_FP32 + dichotomyAddReminderRoundUp);
                Sub(deltaL, dichotomyAddMeanL, mean, pregMain);
                Mul(deltaL, deltaL, deltaL, pregMain);
                Muls(deltaL, deltaL, tailCnt, pregMain);
                DataCopy(dichotomyAddVarL, tmpVarLocal + i * VL_FP32 + dichotomyAddReminderRoundUp);
                Add(dichotomyAddVarL, dichotomyAddVarL, deltaL, pregMain);
                Muls(dichotomyAddVarL, dichotomyAddVarL, reduceScale0, pregMain);
                ReduceSum(var, dichotomyAddVarL, pregMain);
                DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(
                    dichotomyAddLocal + dichotomyAddReminderLoopCount + i, var, pregMerge);
            }

            sreg0 = welfordDiffReminder;
            for (uint16_t i = 0; i < welfordReminderLoopCount; i++) {
                pregLoop = AscendC::MicroAPI::UpdateMask<float>(sreg0);
                DataCopy(
                    dichotomyAddMeanL,
                    tmpMeanLocal + (i + welfordDiffLoopCount) * VL_FP32 + dichotomyAddReminderRoundUp);
                Sub(deltaL, dichotomyAddMeanL, mean, pregMain);
                Mul(deltaL, deltaL, deltaL, pregMain);
                Muls(deltaL, deltaL, cnt, pregMain);
                Muls(tmp, deltaL, coeff, pregLoop);
                Copy<float, MaskMergeMode::MERGING>(deltaL, tmp, pregLoop);
                DataCopy(
                    dichotomyAddVarL, tmpVarLocal + (i + welfordDiffLoopCount) * VL_FP32 + dichotomyAddReminderRoundUp);
                Add(dichotomyAddVarL, dichotomyAddVarL, deltaL, pregMain);
                Muls(dichotomyAddVarL, dichotomyAddVarL, reduceScale0, pregMain);
                ReduceSum(var, dichotomyAddVarL, pregMain);
                DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(
                    dichotomyAddLocal + dichotomyAddReminderLoopCount + welfordDiffLoopCount + i, var, pregMerge);
            }

            for (uint16_t i = 0; i < dichotomyAddPowerRemainLoopCount; i++) {
                DataCopy(dichotomyAddMeanL, tmpMeanLocal + i * VL_FP32 + dichotomyAddPowerOffset);
                Sub(deltaL, dichotomyAddMeanL, mean, pregMain);
                Mul(deltaL, deltaL, deltaL, pregMain);
                Muls(deltaL, deltaL, cnt, pregMain);
                DataCopy(dichotomyAddVarL, tmpVarLocal + i * VL_FP32 + dichotomyAddPowerOffset);
                Add(dichotomyAddVarL, dichotomyAddVarL, deltaL, pregMain);
                Muls(dichotomyAddVarL, dichotomyAddVarL, reduceScale0, pregMain);
                ReduceSum(var, dichotomyAddVarL, pregMain);
                DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(
                    dichotomyAddLocal + dichotomyAddReminderLoopCount + welfordDiffLoopCount +
                        welfordReminderLoopCount + i,
                    var, pregMerge);
            }

            DichotomyAdd(var, dichotomyAddLocal, dichotomyAddK, innerLoopCountOrigin, dichotomyAddLastNum);
            Muls(var, var, reduceScale1, pregMerge);
            DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(varLocal + offset, var, pregMerge);
            CalRstdByHighPrecision(var, rstd, eps);
            DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(rstdLocal + offset, rstd, pregMerge);
        }
    }

    __aicore__ inline void VFWelfordParallelFinalizeNonAlign(
        __local_mem__ float* meanLocal, __local_mem__ float* rstdLocal, __local_mem__ float* varLocal,
        __local_mem__ float* tmpMeanLocal, __local_mem__ float* tmpVarLocal, __local_mem__ float* dichotomyAddLocal,
        uint32_t reduceCount, uint32_t dichotomyAddPower, uint32_t dichotomyAddK, uint32_t dichotomyAddLastNum,
        uint32_t offset, uint32_t tailSize, float reduceScale0, float reduceScale1, float cnt, float tailCnt, float eps)
    {
        // 非对齐Welford finalize阶段由于自身存在整尾块，二分折叠存在整尾块，会出现多种不同的场景，每个场景都有独立的VF
        uint32_t dichotomyAddReminder = reduceCount - dichotomyAddPower;
        uint32_t dichotomyAddReminderRoundUp = CeilDiv(dichotomyAddReminder, VL_FP32) * VL_FP32;
        if (tailSize >= dichotomyAddPower) {
            VFWelfordParallelFinalizeNonAlignSituation1(
                meanLocal, rstdLocal, varLocal, tmpMeanLocal, tmpVarLocal, dichotomyAddLocal, reduceCount,
                dichotomyAddPower, dichotomyAddK, dichotomyAddLastNum, offset, tailSize, reduceScale0, reduceScale1,
                cnt, tailCnt, eps);
            return;
        }
        if (tailSize <= dichotomyAddReminderRoundUp) {
            VFWelfordParallelFinalizeNonAlignSituation2(
                meanLocal, rstdLocal, varLocal, tmpMeanLocal, tmpVarLocal, dichotomyAddLocal, reduceCount,
                dichotomyAddPower, dichotomyAddK, dichotomyAddLastNum, offset, tailSize, reduceScale0, reduceScale1,
                cnt, tailCnt, eps);
            return;
        }
        VFWelfordParallelFinalizeNonAlignSituation3(
            meanLocal, rstdLocal, varLocal, tmpMeanLocal, tmpVarLocal, dichotomyAddLocal, reduceCount,
            dichotomyAddPower, dichotomyAddK, dichotomyAddLastNum, offset, tailSize, reduceScale0, reduceScale1, cnt,
            tailCnt, eps);
    }

    __aicore__ inline void ProcessNormalize(int32_t index_a, int32_t aParallelIndex, int32_t aParallelLength)
    {
        LocalTensor<T> betaTensor = betaQueue.DeQue<T>();
        LocalTensor<T> gammaTensor = gammaQueue.DeQue<T>();
        LocalTensor<float> batchMeanTensor = batchMeanQueue.DeQue<float>();
        LocalTensor<float> batchRstdTensor = batchRstdQueue.DeQue<float>();
        __local_mem__ T* gammaLocal = (__local_mem__ T*)gammaTensor.GetPhyAddr();
        __local_mem__ T* betaLocal = (__local_mem__ T*)betaTensor.GetPhyAddr();
        __local_mem__ float* meanLocal = (__local_mem__ float*)batchMeanTensor.GetPhyAddr();
        __local_mem__ float* rstdLocal = (__local_mem__ float*)batchRstdTensor.GetPhyAddr();

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
                    LocalTensor<T> xInUb = xQueue.DeQue<T>();
                    LocalTensor<T> yOutUb = yQueue.AllocTensor<T>();

                    __local_mem__ T* xLocal = (__local_mem__ T*)xInUb.GetPhyAddr();
                    __local_mem__ T* yOutLocal = (__local_mem__ T*)yOutUb.GetPhyAddr();

                    int32_t loopCount = CeilDiv(actualProcessSize, VL_FP32);
                    int32_t reduceCount = actualProcessSize;
                    VFNormalize(
                        xLocal, gammaLocal, betaLocal, meanLocal, rstdLocal, yOutLocal, i, loopCount, reduceCount);
                    yQueue.EnQue(yOutUb);
                    CopyY2Gm(index, nBurst, burstLen);
                    xQueue.FreeTensor(xInUb);
                }
            }
        }

        CopyBatchMeanAndRstd2GM(index_a, aParallelLength, batchMeanTensor, batchRstdTensor);
        betaQueue.FreeTensor(betaTensor);
        gammaQueue.FreeTensor(gammaTensor);
    }

    __aicore__ inline void VFNormalize(
        __local_mem__ T* xLocal, __local_mem__ T* gammaLocal, __local_mem__ T* betaLocal,
        __local_mem__ float* meanLocal, __local_mem__ float* rstdLocal, __local_mem__ T* yLocal, int64_t i,
        uint16_t loopCount, int32_t reduceCount)
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
            DataCopy<float, LoadDist::DIST_BRC_B32>(rstd, rstdLocal + i);
            DataCopy<float, LoadDist::DIST_BRC_B32>(mean, meanLocal + i);
            MaskReg pregMain = AscendC::MicroAPI::CreateMask<float, AscendC::MicroAPI::MaskPattern::ALL>();
            LoadTwoTensorForDtypeTBrc(gammaLocal, betaLocal, gamma, beta, pregMain, pregMain, i, i);
            uint32_t sreg0 = reduceCount;
            for (uint16_t j = 0; j < loopCount; j++) {
                pregLoop = AscendC::MicroAPI::UpdateMask<float>(sreg0);
                // norm
                LoadOneTensorForDtypeT(xLocal, x, pregLoop, j * VL_FP32);
                Sub(x, x, mean, pregLoop);
                Mul(x, x, rstd, pregLoop);
                Mul(x, x, gamma, pregLoop);
                Add(y, x, beta, pregLoop);

                if constexpr (IsSameType<T, half>::value) {
                    RegTensor<half> yFp16;
                    Cast<half, float, castTraitB322B16>(yFp16, y, pregLoop);
                    DataCopy<half, StoreDist::DIST_PACK_B32>(
                        ((__local_mem__ half*)yLocal) + j * VL_FP32, yFp16, pregLoop);
                } else if constexpr (IsSameType<T, bfloat16_t>::value) {
                    RegTensor<bfloat16_t> xBf16;
                    Cast<bfloat16_t, float, castTraitB322B16>(xBf16, y, pregLoop);
                    DataCopy<bfloat16_t, StoreDist::DIST_PACK_B32>(
                        ((__local_mem__ bfloat16_t*)yLocal) + j * VL_FP32, xBf16, pregLoop);
                } else {
                    DataCopy(((__local_mem__ float*)yLocal) + j * VL_FP32, y, pregLoop);
                }
            }
        }
    }

    __aicore__ inline uint32_t CeilDiv(const uint32_t x, const uint32_t y)
    {
        if (y > 0) {
            return (x + y - 1) / y;
        }
        return 0;
    }

    /* global memory address */
    GlobalTensor<T> xGm;
    GlobalTensor<T> betaGm;
    GlobalTensor<T> gammaGm;
    GlobalTensor<T_RUNNING_MEAN> meanGm;
    GlobalTensor<T_RUNNING_MEAN> varGm;

    GlobalTensor<T> yGm;
    GlobalTensor<float> batchMeanGm;
    GlobalTensor<float> batchRstdGm;
    GlobalTensor<T_RUNNING_MEAN> outMeanGm;
    GlobalTensor<T_RUNNING_MEAN> outVarGm;

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
    float momentumForVar;

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
    float reduceScale1 = 0;
    float scale0 = 0;
    float scale1 = 0;

    /* ascendc variable */
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> xQueue;
    TQue<QuePosition::VECIN, BUFFER_NUM> betaQueue;
    TQue<QuePosition::VECIN, BUFFER_NUM> gammaQueue;
    TQue<QuePosition::VECIN, BUFFER_NUM> meanQueue;
    TQue<QuePosition::VECIN, BUFFER_NUM> varQueue;

    TQue<QuePosition::VECOUT, BUFFER_NUM> yQueue;
    TQue<QuePosition::VECOUT, BUFFER_NUM> batchMeanQueue;
    TQue<QuePosition::VECOUT, BUFFER_NUM> batchRstdQueue;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outMeanQueue;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outVarQueue;

    TBuf<TPosition::VECCALC> tMeanBuff;
    TBuf<TPosition::VECCALC> tVarBuff;
    TBuf<TPosition::VECCALC> batchVarBuff;
    TBuf<TPosition::VECCALC> dichotomyAddBuf;
};
} // namespace BatchNormV3Ops
#endif

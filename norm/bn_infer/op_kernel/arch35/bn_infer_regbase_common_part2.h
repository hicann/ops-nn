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
 * \file bn_infer_regbase_common_part2.h
 * \brief Split from bn_infer_regbase_common.h for code-check line budget.
 */
__aicore__ inline void LastFinalizeVF(LocalTensor<float>& batchMeanTensor, LocalTensor<float>& batchRstdTensor,
                                      LocalTensor<float>& meanTensor, LocalTensor<float>& varTensor,
                                      LocalTensor<float>& countTensor, LocalTensor<float>& tmpTensor,
                                      uint32_t currentAAlign, uint32_t vectorLen, uint16_t currentA,
                                      uint16_t usedCoreNum, uint16_t lastBinaryAddQuotient, uint16_t lastBinaryAddK,
                                      uint16_t lastBinaryAddLast, float lastNFactor, float lastNCorrectionFactor)
{
    __local_mem__ float* tmpMeanLocal = (__local_mem__ float*)meanTensor.GetPhyAddr();
    __local_mem__ float* tmpCountLocal = (__local_mem__ float*)countTensor.GetPhyAddr();
    __local_mem__ float* tmpVarLocal = (__local_mem__ float*)varTensor.GetPhyAddr();
    __local_mem__ float* batchMeanTensorAddr = (__local_mem__ float*)batchMeanTensor.GetPhyAddr();
    __local_mem__ float* tmpUbAddr = (__local_mem__ float*)tmpTensor.GetPhyAddr();
    __local_mem__ float* batchRstdTensorAddr = (__local_mem__ float*)batchRstdTensor.GetPhyAddr();
    uint32_t rLoopStride = currentAAlign;
    vectorLen = (vectorLen == 0) ? (NormCommon::NormCommonRegbase::GetVRegSize() / sizeof(float)) : vectorLen;
    uint16_t aLoopCount = ((currentA + vectorLen - 1) / vectorLen);
    uint16_t remainderLoopCount = usedCoreNum - lastBinaryAddQuotient;
    uint16_t quotientLoopCount = lastBinaryAddQuotient - remainderLoopCount;
    uint32_t baseLineOffset = rLoopStride;
    uint32_t remainderCountOffset = lastBinaryAddQuotient;
    uint32_t remainderOffset = lastBinaryAddQuotient * rLoopStride;
    uint16_t binaryAddKLoop = lastBinaryAddK;
    uint16_t binaryAddLastLoop = lastBinaryAddLast;
    uint16_t binaryAddInnerLoop = lastBinaryAddQuotient;
    float numScale = lastNFactor;
    float scaleCorrection = lastNCorrectionFactor;
    __VEC_SCOPE__
    {
        RegTensor<float> quot;
        RegTensor<float> rem;
        RegTensor<float> remCount;
        RegTensor<float> quotCount;
        RegTensor<float> oriQuotMean;
        RegTensor<float> resMean;
        RegTensor<float> oriRemMean;
        RegTensor<float> resVar;

        uint32_t sreg0 = currentA;
        MaskReg pregLoop;
        for (uint16_t aIndex = 0; aIndex < aLoopCount; aIndex++) {
            uint32_t aLoopOffset = aIndex * vectorLen;
            pregLoop = UpdateMask<float>(sreg0);
            for (uint16_t i = 0; i < remainderLoopCount; i++) {
                uint32_t quotOffset = i * baseLineOffset + aLoopOffset;
                uint32_t remOffset = i * baseLineOffset + remainderOffset + aLoopOffset;
                uint32_t quotCountOffset = i;
                uint32_t remCountOffset = i + remainderCountOffset;
                DataCopy(quot, ((__local_mem__ float*)(tmpMeanLocal) + (quotOffset)));
                DataCopy(rem, ((__local_mem__ float*)(tmpMeanLocal) + (remOffset)));
                DataCopy<float, LoadDist::DIST_BRC_B32>(quotCount,
                                                        ((__local_mem__ float*)(tmpCountLocal) + quotCountOffset));
                DataCopy<float, LoadDist::DIST_BRC_B32>(remCount,
                                                        ((__local_mem__ float*)(tmpCountLocal) + remCountOffset));
                Mul(quot, quot, quotCount, pregLoop);
                Mul(rem, rem, remCount, pregLoop);
                Muls(quot, quot, numScale, pregLoop);
                Muls(rem, rem, numScale, pregLoop);
                Add(quot, quot, rem, pregLoop);
                DataCopy(((__local_mem__ float*)tmpUbAddr + i * rLoopStride + aLoopOffset), quot, pregLoop);
            }
            for (uint16_t i = 0; i < quotientLoopCount; i++) {
                uint32_t baseOffset = (remainderLoopCount + i) * baseLineOffset + aLoopOffset;
                uint32_t baseCountOffset = remainderLoopCount + i;
                DataCopy(quot, ((__local_mem__ float*)(tmpMeanLocal) + (baseOffset)));
                DataCopy<float, LoadDist::DIST_BRC_B32>(quotCount,
                                                        ((__local_mem__ float*)(tmpCountLocal) + baseCountOffset));
                Mul(quot, quot, quotCount, pregLoop);
                Muls(quot, quot, numScale, pregLoop);
                DataCopy(((__local_mem__ float*)tmpUbAddr + (remainderLoopCount + i) * rLoopStride + aLoopOffset), quot,
                         pregLoop);
            }
            LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
            BinaryAddVF(tmpUbAddr, rLoopStride, binaryAddKLoop, binaryAddInnerLoop, binaryAddLastLoop, pregLoop,
                        aLoopOffset, quot, rem, quotCount, remCount);
            DataCopy(resMean, ((__local_mem__ float*)tmpUbAddr + aLoopOffset));
            Muls(resMean, resMean, scaleCorrection, pregLoop);
            DataCopy(((__local_mem__ float*)batchMeanTensorAddr + aLoopOffset), resMean, pregLoop);
            for (uint16_t i = 0; i < remainderLoopCount; i++) {
                uint32_t quotOffset = i * baseLineOffset + aLoopOffset;
                uint32_t remOffset = i * baseLineOffset + remainderOffset + aLoopOffset;
                uint32_t quotCountOffset = i;
                uint32_t remCountOffset = i + remainderCountOffset;
                DataCopy(quot, ((__local_mem__ float*)(tmpVarLocal) + (quotOffset)));
                DataCopy(rem, ((__local_mem__ float*)(tmpVarLocal) + (remOffset)));
                DataCopy(oriQuotMean, ((__local_mem__ float*)(tmpMeanLocal) + (quotOffset)));
                DataCopy(oriRemMean, ((__local_mem__ float*)(tmpMeanLocal) + (remOffset)));
                DataCopy<float, LoadDist::DIST_BRC_B32>(quotCount,
                                                        ((__local_mem__ float*)(tmpCountLocal) + quotCountOffset));
                DataCopy<float, LoadDist::DIST_BRC_B32>(remCount,
                                                        ((__local_mem__ float*)(tmpCountLocal) + remCountOffset));
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
                DataCopy(((__local_mem__ float*)tmpUbAddr + i * rLoopStride + aLoopOffset), quot, pregLoop);
            }
            for (uint16_t i = 0; i < quotientLoopCount; i++) {
                uint32_t baseOffset = (remainderLoopCount + i) * baseLineOffset + aLoopOffset;
                uint32_t baseCountOffset = remainderLoopCount + i;
                DataCopy(quot, ((__local_mem__ float*)(tmpVarLocal) + (baseOffset)));
                DataCopy(oriQuotMean, ((__local_mem__ float*)(tmpMeanLocal) + (baseOffset)));
                DataCopy<float, LoadDist::DIST_BRC_B32>(quotCount,
                                                        ((__local_mem__ float*)(tmpCountLocal) + baseCountOffset));
                Sub(oriQuotMean, oriQuotMean, resMean, pregLoop);
                Mul(oriQuotMean, oriQuotMean, oriQuotMean, pregLoop);
                Mul(oriQuotMean, oriQuotMean, quotCount, pregLoop);
                Mul(quot, quot, quotCount, pregLoop);
                Add(quot, quot, oriQuotMean, pregLoop);
                Muls(quot, quot, numScale, pregLoop);
                DataCopy(((__local_mem__ float*)tmpUbAddr + (remainderLoopCount + i) * rLoopStride + aLoopOffset), quot,
                         pregLoop);
            }
            LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
            BinaryAddVF(tmpUbAddr, rLoopStride, binaryAddKLoop, binaryAddInnerLoop, binaryAddLastLoop, pregLoop,
                        aLoopOffset, quot, rem, quotCount, remCount);
            DataCopy(resVar, ((__local_mem__ float*)tmpUbAddr + aLoopOffset));
            Muls(resVar, resVar, scaleCorrection, pregLoop);
            DataCopy(((__local_mem__ float*)batchRstdTensorAddr + aLoopOffset), resVar, pregLoop);
        }
    }
}

template <typename T_RUNNING_MEAN>
__aicore__ inline void CalculateRunningMeanVarWithRstdVF(
    __local_mem__ float* batchMeanInUb, __local_mem__ float* batchRstdInUb,
    __local_mem__ T_RUNNING_MEAN* runningMeanInUbAddr, __local_mem__ T_RUNNING_MEAN* runningVarInUbAddr,
    __local_mem__ T_RUNNING_MEAN* runningMeanOutUbAddr, __local_mem__ T_RUNNING_MEAN* runningVarOutUbAddr,
    uint16_t currentANum, uint16_t aLoop, uint32_t vectorLen, float besselCorrection, float momentum,
    float oneSubMomentum, float epsilon)
{
    __VEC_SCOPE__
    {
        RegTensor<float> mean;
        RegTensor<float> var;
        RegTensor<float> one;
        RegTensor<float> runningMean;
        RegTensor<float> saveMean;
        RegTensor<float> runningVar;
        RegTensor<float> saveVar;
        MaskReg pregMain = CreateMask<float, MaskPattern::ALL>();
        RegTensor<float> r;
        RegTensor<float> y;
        RegTensor<float> s;
        RegTensor<float> t;
        RegTensor<float> scalar1;
        RegTensor<float> scalarInf;
        RegTensor<float> scalarZero;
        RegTensor<float> t1;
        RegTensor<float> t3;
        RegTensor<float> t4;
        RegTensor<float> rstd;
        MaskReg cmpRegZero;
        MaskReg cmpRegInf;
        MaskReg pregLoop;

        Duplicate(one, 1.0, pregMain);
        uint32_t sreg2 = currentANum;
        for (uint16_t k = 0; k < aLoop; k++) {
            pregLoop = UpdateMask<float>(sreg2);
            Duplicate(scalar1, float(0.5), pregLoop);
            Duplicate(scalarInf, BN_INFER_POS_INF, pregLoop);
            Duplicate(scalarZero, float(0.0), pregLoop);
            Duplicate(t1, float(1.5), pregLoop);
            Duplicate(s, float(1.0), pregLoop);
            if constexpr (!IsSameType<T_RUNNING_MEAN, float>::value) {
                RegTensor<T_RUNNING_MEAN> runningVarTmp;
                DataCopy<T_RUNNING_MEAN, LoadDist::DIST_UNPACK_B16>(
                    runningVarTmp, ((__local_mem__ T_RUNNING_MEAN*)runningVarInUbAddr + k * vectorLen));
                Cast<float, T_RUNNING_MEAN, NormCommon::castTraitB162B32>(runningVar, runningVarTmp, pregLoop);
            } else {
                DataCopy(runningVar, ((__local_mem__ float*)runningVarInUbAddr + k * vectorLen));
            }
            DataCopy(var, ((__local_mem__ float*)batchRstdInUb + k * vectorLen));
            Muls(saveVar, var, besselCorrection, pregLoop);
            Muls(saveVar, saveVar, momentum, pregLoop);
            Muls(runningVar, runningVar, oneSubMomentum, pregLoop);
            Add(saveVar, saveVar, runningVar, pregLoop);
            if constexpr (!IsSameType<T_RUNNING_MEAN, float>::value) {
                RegTensor<T_RUNNING_MEAN> saveVarTmp;
                Cast<T_RUNNING_MEAN, float, NormCommon::castTraitB322B16>(saveVarTmp, saveVar, pregLoop);
                DataCopy<T_RUNNING_MEAN, StoreDist::DIST_PACK_B32>(
                    ((__local_mem__ T_RUNNING_MEAN*)runningVarOutUbAddr + k * vectorLen), saveVarTmp, pregLoop);
            } else {
                DataCopy(((__local_mem__ float*)runningVarOutUbAddr + k * vectorLen), saveVar, pregLoop);
            }

            Adds(var, var, epsilon, pregLoop);
            Div(r, one, var, pregLoop);
            Sqrt(y, r, pregLoop);
            Muls(t, var, float(-0.5), pregLoop);
            Mul(t, t, y, pregLoop);
            Mula(t1, t, y, pregLoop);
            Mul(rstd, y, t1, pregLoop);
            Muls(t3, var, float(-1.0), pregLoop);
            Mula(s, t3, r, pregLoop);
            Muls(t4, rstd, float(-1.0), pregLoop);
            Mula(r, t4, rstd, pregLoop);
            Mula(s, var, r, pregLoop);
            Mul(s, s, rstd, pregLoop);
            Mula(rstd, s, scalar1, pregLoop);
            CompareScalar<float, CMPMODE::EQ>(cmpRegZero, var, BN_INFER_POS_INF, pregLoop);
            Select(rstd, scalarZero, rstd, cmpRegZero);
            CompareScalar<float, CMPMODE::EQ>(cmpRegInf, var, float(0.0), pregLoop);
            Select(rstd, scalarInf, rstd, cmpRegInf);
            DataCopy(((__local_mem__ float*)batchRstdInUb + k * vectorLen), rstd, pregLoop);

            DataCopy(mean, ((__local_mem__ float*)batchMeanInUb + k * vectorLen));
            if constexpr (!IsSameType<T_RUNNING_MEAN, float>::value) {
                RegTensor<T_RUNNING_MEAN> runningMeanTmp;
                DataCopy<T_RUNNING_MEAN, LoadDist::DIST_UNPACK_B16>(
                    runningMeanTmp, ((__local_mem__ T_RUNNING_MEAN*)runningMeanInUbAddr + k * vectorLen));
                Cast<float, T_RUNNING_MEAN, NormCommon::castTraitB162B32>(runningMean, runningMeanTmp, pregLoop);
            } else {
                DataCopy(runningMean, ((__local_mem__ float*)runningMeanInUbAddr + k * vectorLen));
            }
            Muls(saveMean, mean, momentum, pregLoop);
            Muls(runningMean, runningMean, oneSubMomentum, pregLoop);
            Add(saveMean, saveMean, runningMean, pregLoop);
            if constexpr (!IsSameType<T_RUNNING_MEAN, float>::value) {
                RegTensor<T_RUNNING_MEAN> saveMeanTmp;
                Cast<T_RUNNING_MEAN, float, NormCommon::castTraitB322B16>(saveMeanTmp, saveMean, pregLoop);
                DataCopy<T_RUNNING_MEAN, StoreDist::DIST_PACK_B32>(
                    ((__local_mem__ T_RUNNING_MEAN*)runningMeanOutUbAddr + k * vectorLen), saveMeanTmp, pregLoop);
            } else {
                DataCopy(((__local_mem__ float*)runningMeanOutUbAddr + k * vectorLen), saveMean, pregLoop);
            }
        }
    }
}

template <typename T_RUNNING_MEAN>
__aicore__ inline void CalculateRunningMeanVarVF(__local_mem__ float* batchMeanInUb, __local_mem__ float* batchVarInUb,
                                                 __local_mem__ T_RUNNING_MEAN* runningMeanInUbAddr,
                                                 __local_mem__ T_RUNNING_MEAN* runningVarInUbAddr,
                                                 __local_mem__ T_RUNNING_MEAN* runningMeanOutUbAddr,
                                                 __local_mem__ T_RUNNING_MEAN* runningVarOutUbAddr,
                                                 uint16_t currentANum, uint16_t aLoop, uint32_t vectorLen,
                                                 float unbiasedEstimationCoeff, float momentum, float momentumReverse)
{
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
            if constexpr (!IsSameType<T_RUNNING_MEAN, float>::value) {
                RegTensor<T_RUNNING_MEAN> runningVarTmp;
                DataCopy<T_RUNNING_MEAN, LoadDist::DIST_UNPACK_B16>(
                    runningVarTmp, ((__local_mem__ T_RUNNING_MEAN*)runningVarInUbAddr + k * vectorLen));
                Cast<float, T_RUNNING_MEAN, NormCommon::castTraitB162B32>(runningVar, runningVarTmp, pregLoop);
            } else {
                DataCopy(runningVar, ((__local_mem__ float*)runningVarInUbAddr + k * vectorLen));
            }
            DataCopy(var, ((__local_mem__ float*)batchVarInUb + k * vectorLen));
            Muls(saveVar, var, unbiasedEstimationCoeff, pregLoop);
            Muls(saveVar, saveVar, momentum, pregLoop);
            Muls(runningVar, runningVar, momentumReverse, pregLoop);
            Add(saveVar, saveVar, runningVar, pregLoop);
            if constexpr (!IsSameType<T_RUNNING_MEAN, float>::value) {
                RegTensor<T_RUNNING_MEAN> saveVarTmp;
                Cast<T_RUNNING_MEAN, float, NormCommon::castTraitB322B16>(saveVarTmp, saveVar, pregLoop);
                DataCopy<T_RUNNING_MEAN, StoreDist::DIST_PACK_B32>(
                    ((__local_mem__ T_RUNNING_MEAN*)runningVarOutUbAddr + k * vectorLen), saveVarTmp, pregLoop);
            } else {
                DataCopy(((__local_mem__ float*)runningVarOutUbAddr + k * vectorLen), saveVar, pregLoop);
            }

            DataCopy(mean, ((__local_mem__ float*)batchMeanInUb + k * vectorLen));
            if constexpr (!IsSameType<T_RUNNING_MEAN, float>::value) {
                RegTensor<T_RUNNING_MEAN> runningMeanTmp;
                DataCopy<T_RUNNING_MEAN, LoadDist::DIST_UNPACK_B16>(
                    runningMeanTmp, ((__local_mem__ T_RUNNING_MEAN*)runningMeanInUbAddr + k * vectorLen));
                Cast<float, T_RUNNING_MEAN, NormCommon::castTraitB162B32>(runningMean, runningMeanTmp, pregLoop);
            } else {
                DataCopy(runningMean, ((__local_mem__ float*)runningMeanInUbAddr + k * vectorLen));
            }
            Muls(saveMean, mean, momentum, pregLoop);
            Muls(runningMean, runningMean, momentumReverse, pregLoop);
            Add(saveMean, saveMean, runningMean, pregLoop);
            if constexpr (!IsSameType<T_RUNNING_MEAN, float>::value) {
                RegTensor<T_RUNNING_MEAN> saveMeanTmp;
                Cast<T_RUNNING_MEAN, float, NormCommon::castTraitB322B16>(saveMeanTmp, saveMean, pregLoop);
                DataCopy<T_RUNNING_MEAN, StoreDist::DIST_PACK_B32>(
                    ((__local_mem__ T_RUNNING_MEAN*)runningMeanOutUbAddr + k * vectorLen), saveMeanTmp, pregLoop);
            } else {
                DataCopy(((__local_mem__ float*)runningMeanOutUbAddr + k * vectorLen), saveMean, pregLoop);
            }
        }
    }
}

template <typename T_GAMMA, typename T_RUNNING_MEAN>
__aicore__ inline void VFPrepareParamCache(__ubuf__ T_GAMMA* gammaLocal, __ubuf__ T_GAMMA* betaLocal,
                                           __ubuf__ T_RUNNING_MEAN* meanLocal, __ubuf__ T_RUNNING_MEAN* varLocal,
                                           __ubuf__ uint32_t* offsetLocal, __ubuf__ float* gammaFp32Local,
                                           __ubuf__ float* betaFp32Local, __ubuf__ float* meanFp32Local,
                                           __ubuf__ float* rstdFp32Local, uint32_t paramCacheElemLen, float epsilon)
{
    __VEC_SCOPE__
    {
        RegTensor<float> gamma;
        RegTensor<float> beta;
        RegTensor<float> mean;
        RegTensor<float> var;
        RegTensor<float> rstd;
        RegTensor<uint32_t> paramOffset;
        uint32_t maskLen = paramCacheElemLen;
        MaskReg pregMask = AscendC::MicroAPI::UpdateMask<float>(maskLen);

        AscendC::MicroAPI::LoadAlign<uint32_t, LoadDist::DIST_NORM>(paramOffset, offsetLocal);
        GatherParamForDtypeT(gammaLocal, gamma, paramOffset, pregMask, paramCacheElemLen);
        GatherParamForDtypeT(betaLocal, beta, paramOffset, pregMask, paramCacheElemLen);
        GatherRunningParamForDtypeT(varLocal, var, paramOffset, pregMask, paramCacheElemLen);
        NormCommon::ComputeRstdNewtonRaphsonReg(var, rstd, pregMask, epsilon);
        GatherRunningParamForDtypeT(meanLocal, mean, paramOffset, pregMask, paramCacheElemLen);

        StoreAlign(gammaFp32Local, gamma, pregMask);
        StoreAlign(betaFp32Local, beta, pregMask);
        StoreAlign(meanFp32Local, mean, pregMask);
        StoreAlign(rstdFp32Local, rstd, pregMask);
    }
}

template <typename T_GAMMA, typename T_RUNNING_MEAN, typename QueT, typename BufT>
__aicore__ inline void PrepareParamCache(QueT& betaQueue, QueT& gammaQueue, QueT& meanQueue, QueT& varQueue,
                                         BufT& offsetBuf, BufT& betaFp32Buf, BufT& gammaFp32Buf, BufT& meanFp32Buf,
                                         BufT& rstdFp32Buf, uint32_t paramCacheElemLen, float epsilon)
{
    LocalTensor<T_GAMMA> beta = betaQueue.template DeQue<T_GAMMA>();
    LocalTensor<T_GAMMA> gamma = gammaQueue.template DeQue<T_GAMMA>();
    LocalTensor<T_RUNNING_MEAN> mean = meanQueue.template DeQue<T_RUNNING_MEAN>();
    LocalTensor<T_RUNNING_MEAN> var = varQueue.template DeQue<T_RUNNING_MEAN>();
    LocalTensor<uint32_t> offset = offsetBuf.template Get<uint32_t>();
    LocalTensor<float> betaFp32 = betaFp32Buf.template Get<float>();
    LocalTensor<float> gammaFp32 = gammaFp32Buf.template Get<float>();
    LocalTensor<float> meanFp32 = meanFp32Buf.template Get<float>();
    LocalTensor<float> rstdFp32 = rstdFp32Buf.template Get<float>();

    __local_mem__ T_GAMMA* betaLocal = (__local_mem__ T_GAMMA*)beta.GetPhyAddr();
    __local_mem__ T_GAMMA* gammaLocal = (__local_mem__ T_GAMMA*)gamma.GetPhyAddr();
    __local_mem__ T_RUNNING_MEAN* meanLocal = (__local_mem__ T_RUNNING_MEAN*)mean.GetPhyAddr();
    __local_mem__ T_RUNNING_MEAN* varLocal = (__local_mem__ T_RUNNING_MEAN*)var.GetPhyAddr();
    __ubuf__ uint32_t* offsetLocal = (__ubuf__ uint32_t*)offset.GetPhyAddr();
    __local_mem__ float* betaFp32Local = (__local_mem__ float*)betaFp32.GetPhyAddr();
    __local_mem__ float* gammaFp32Local = (__local_mem__ float*)gammaFp32.GetPhyAddr();
    __local_mem__ float* meanFp32Local = (__local_mem__ float*)meanFp32.GetPhyAddr();
    __local_mem__ float* rstdFp32Local = (__local_mem__ float*)rstdFp32.GetPhyAddr();

    VFPrepareParamCache<T_GAMMA, T_RUNNING_MEAN>(gammaLocal, betaLocal, meanLocal, varLocal, offsetLocal,
                                                 gammaFp32Local, betaFp32Local, meanFp32Local, rstdFp32Local,
                                                 paramCacheElemLen, epsilon);

    betaQueue.template FreeTensor<T_GAMMA>(beta);
    gammaQueue.template FreeTensor<T_GAMMA>(gamma);
    meanQueue.template FreeTensor<T_RUNNING_MEAN>(mean);
    varQueue.template FreeTensor<T_RUNNING_MEAN>(var);
}

template <typename T>
__aicore__ inline void NormalizeUnalignWithParamCache(__ubuf__ T* xLocal, __ubuf__ float* gammaFp32Local,
                                                      __ubuf__ float* betaFp32Local, __ubuf__ float* meanFp32Local,
                                                      __ubuf__ float* rstdFp32Local, __ubuf__ T* yLocal,
                                                      uint32_t elemLen, uint32_t paramCacheElemLen)
{
    __VEC_SCOPE__
    {
        RegTensor<float> x;
        RegTensor<float> gamma;
        RegTensor<float> beta;
        RegTensor<float> mean;
        RegTensor<float> rstd;
        RegTensor<float> y;
        uint16_t loopNum = static_cast<uint16_t>((static_cast<uint64_t>(elemLen) + paramCacheElemLen - 1) /
                                                 paramCacheElemLen);
        __ubuf__ T* xLocalTmp = xLocal;
        __ubuf__ T* yLocalTmp = yLocal;
        AscendC::MicroAPI::UnalignRegForLoad uX;
        AscendC::MicroAPI::UnalignRegForStore uY;
        AscendC::MicroAPI::LoadUnAlignPre(uX, xLocalTmp);
        LoadAlign<float, LoadDist::DIST_NORM>(gamma, gammaFp32Local);
        LoadAlign<float, LoadDist::DIST_NORM>(beta, betaFp32Local);
        LoadAlign<float, LoadDist::DIST_NORM>(mean, meanFp32Local);
        LoadAlign<float, LoadDist::DIST_NORM>(rstd, rstdFp32Local);
        uint32_t elemOffset = 0;
        for (uint16_t i = 0; i < loopNum; i++) {
            uint32_t activeLen = elemLen - elemOffset > paramCacheElemLen ? paramCacheElemLen : elemLen - elemOffset;
            MaskReg pregMask = AscendC::MicroAPI::UpdateMask<float>(activeLen);
            NormCommon::LoadTensorUnAlignForDtypeT(xLocalTmp, x, uX, pregMask, activeLen);
            NormCommon::NormalizeWithScaleBiasReg(x, gamma, beta, mean, rstd, y, pregMask);
            NormCommon::StoreTensorUnAlignForDtypeT(yLocalTmp, y, uY, pregMask, activeLen);
            elemOffset += activeLen;
        }
        AscendC::MicroAPI::StoreUnAlignPost(yLocalTmp, uY, 0);
    }
}

template <typename T>
__aicore__ inline void NormalizeSmallAB1Unalign(__local_mem__ T* xLocal, __local_mem__ float* gammaFp32Local,
                                                __local_mem__ float* betaFp32Local, __local_mem__ float* meanFp32Local,
                                                __local_mem__ float* rstdFp32Local, __local_mem__ T* yLocal,
                                                uint32_t elemLen, uint32_t paramCacheElemLen)
{
    NormalizeUnalignWithParamCache(xLocal, gammaFp32Local, betaFp32Local, meanFp32Local, rstdFp32Local, yLocal, elemLen,
                                   paramCacheElemLen);
}

template <typename T>
__aicore__ inline void NormalizeSmallLastChannelUnalign(__local_mem__ T* xLocal, __local_mem__ float* gammaFp32Local,
                                                        __local_mem__ float* betaFp32Local,
                                                        __local_mem__ float* meanFp32Local,
                                                        __local_mem__ float* rstdFp32Local, __local_mem__ T* yLocal,
                                                        uint32_t elemLen, uint32_t paramCacheElemLen)
{
    NormalizeUnalignWithParamCache(xLocal, gammaFp32Local, betaFp32Local, meanFp32Local, rstdFp32Local, yLocal, elemLen,
                                   paramCacheElemLen);
}

/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* Internal implementation section of group_norm_regbase_base.h. Include only from group_norm_regbase_base.h. */

template <typename T1, typename T2>
__aicore__ inline void VFNormalizeUnAlign(__local_mem__ T1* xLocal, __local_mem__ T2* gammaLocal,
                                          __local_mem__ T2* betaLocal, __local_mem__ float* meanLocal,
                                          __local_mem__ float* rstdLocal, __local_mem__ T1* yLocal, uint32_t rowsCount,
                                          int32_t reduceCount)
{
    uint16_t VL = GetVLSize<T1>();
    uint16_t loopCount = reduceCount / VL;
    uint16_t tailNum = reduceCount - loopCount * VL;
    uint16_t tailLoop = CeilDiv(tailNum, VL);
    __VEC_SCOPE__
    {
        RegTensor<float> x;
        RegTensor<float> xOdd;
        RegTensor<float> xEven;
        RegTensor<float> gamma;
        RegTensor<float> beta;
        RegTensor<float> mean;
        RegTensor<float> rstd;
        RegTensor<float> y;
        RegTensor<float> yOdd;
        RegTensor<float> yEven;
        MaskReg pregLoop;
        MaskReg pregMain = CreateMask<T1, AscendC::Reg::MaskPattern::ALL>();

        UnalignReg uSrc;
        UnalignReg uDst;
        DataCopy<float, AscendC::Reg::LoadDist::DIST_BRC_B32>(rstd, rstdLocal);
        DataCopy<float, AscendC::Reg::LoadDist::DIST_BRC_B32>(mean, meanLocal);
        DataCopyUnAlignPre<T1>(uSrc, xLocal);
        for (uint16_t i = 0; i < static_cast<uint16_t>(rowsCount); i++) {
            LoadGammaAndBetaData<T2>(gamma, beta, gammaLocal, betaLocal, pregMain, i);
            if constexpr (IsSameType<T1, half>::value || IsSameType<T1, bfloat16_t>::value) {
                RegTensor<T1> xTmp;
                RegTensor<T1> yEvenTmp;
                RegTensor<T1> yOddTmp;
                RegTensor<T1> yTmp;
                for (uint16_t j = 0; j < loopCount; j++) {
                    DataCopyUnAlign(xTmp, uSrc, xLocal, VL);
                    Cast<float, T1, castTraitB162B32Even>(xEven, xTmp, pregMain);
                    Cast<float, T1, castTraitB162B32Odd>(xOdd, xTmp, pregMain);
                    VFInnerNormalize(xEven, mean, rstd, gamma, beta, yEven, pregMain);
                    VFInnerNormalize(xOdd, mean, rstd, gamma, beta, yOdd, pregMain);
                    Cast<T1, float, castTraitB322B16Even>(yEvenTmp, yEven, pregMain);
                    Cast<T1, float, castTraitB322B16Odd>(yOddTmp, yOdd, pregMain);
                    Or((RegTensor<int16_t>&)yTmp, (RegTensor<int16_t>&)yEvenTmp, (RegTensor<int16_t>&)yOddTmp,
                       pregMain);
                    DataCopyUnAlign(yLocal, yTmp, uDst, VL);
                }
                uint32_t sreg0 = tailNum;
                for (uint16_t k = 0; k < tailLoop; k++) {
                    pregLoop = UpdateMask<half>(sreg0);
                    DataCopyUnAlign(xTmp, uSrc, xLocal, tailNum);
                    Cast<float, T1, castTraitB162B32Even>(xEven, xTmp, pregLoop);
                    Cast<float, T1, castTraitB162B32Odd>(xOdd, xTmp, pregLoop);
                    VFInnerNormalize(xEven, mean, rstd, gamma, beta, yEven, pregLoop);
                    VFInnerNormalize(xOdd, mean, rstd, gamma, beta, yOdd, pregLoop);
                    Cast<T1, float, castTraitB322B16Even>(yEvenTmp, yEven, pregLoop);
                    Cast<T1, float, castTraitB322B16Odd>(yOddTmp, yOdd, pregLoop);
                    Or((RegTensor<int16_t>&)yTmp, (RegTensor<int16_t>&)yEvenTmp, (RegTensor<int16_t>&)yOddTmp,
                       pregLoop);
                    DataCopyUnAlign(yLocal, yTmp, uDst, tailNum);
                }
                DataCopyUnAlignPost(yLocal, uDst, 0);
            } else {
                for (uint16_t j = 0; j < loopCount; j++) {
                    DataCopyUnAlign(x, uSrc, xLocal, VL_FP32);
                    VFInnerNormalize(x, mean, rstd, gamma, beta, y, pregMain);
                    DataCopyUnAlign(yLocal, y, uDst, VL_FP32);
                }
                uint32_t sreg0 = tailNum;
                for (uint16_t k = 0; k < tailLoop; k++) {
                    pregLoop = UpdateMask<float>(sreg0);
                    DataCopyUnAlign(x, uSrc, xLocal, tailNum);
                    VFInnerNormalize(x, mean, rstd, gamma, beta, y, pregLoop);
                    DataCopyUnAlign(yLocal, y, uDst, tailNum);
                }
                DataCopyUnAlignPost(yLocal, uDst, 0);
            }
        }
    }
}

template <typename T1, typename T2>
__aicore__ inline void VFNormalizeAlign(__local_mem__ T1* xLocal, __local_mem__ T2* gammaLocal,
                                        __local_mem__ T2* betaLocal, __local_mem__ float* meanLocal,
                                        __local_mem__ float* rstdLocal, __local_mem__ T1* yLocal, uint16_t rowsCount,
                                        int32_t reduceCount)
{
    uint16_t loopCount = CeilDiv(reduceCount, VL_FP32);
    uint32_t reduceCountAlign = RoundUp<T1>(reduceCount);
    __VEC_SCOPE__
    {
        RegTensor<float> x;
        RegTensor<float> gamma;
        RegTensor<float> beta;
        RegTensor<float> mean;
        RegTensor<float> rstd;
        RegTensor<float> y;
        MaskReg pregLoop;
        MaskReg pregMain = CreateMask<float, AscendC::Reg::MaskPattern::ALL>();
        DataCopy<float, AscendC::Reg::LoadDist::DIST_BRC_B32>(rstd, rstdLocal);
        DataCopy<float, AscendC::Reg::LoadDist::DIST_BRC_B32>(mean, meanLocal);
        for (uint16_t i = 0; i < rowsCount; i++) {
            uint32_t sreg0 = reduceCount;
            LoadGammaAndBetaData<T2>(gamma, beta, gammaLocal, betaLocal, pregMain, i);
            for (uint16_t j = 0; j < loopCount; j++) {
                pregLoop = UpdateMask<float>(sreg0);
                LoadInputData<T1>(x, xLocal, pregLoop, i * reduceCountAlign + j * VL_FP32);
                VFInnerNormalize(x, mean, rstd, gamma, beta, y, pregLoop);
                StoreOutputData<T1>(yLocal, y, pregLoop, i * reduceCountAlign + j * VL_FP32);
            }
        }
    }
}

template <typename T>
__aicore__ inline void CopyGammaAndBeta2UB(const GlobalTensor<T>& gammaGm, const GlobalTensor<T>& betaGm,
                                           const LocalTensor<T>& gammaTensor, const LocalTensor<T>& betaTensor,
                                           const uint16_t blockCount, const uint32_t copyLen, bool hasGamma = true,
                                           bool hasBeta = true)
{
    int32_t copyLenAlign = RoundUp<T>(copyLen);
    DataCopyPadExtParams<T> padParams;
    padParams.isPad = true;
    padParams.paddingValue = static_cast<T>(0.0);
    padParams.rightPadding = 0;
    padParams.rightPadding = copyLenAlign - copyLen;

    DataCopyExtParams dataCopyParams;
    dataCopyParams.blockCount = blockCount;
    dataCopyParams.blockLen = copyLen * sizeof(T);
    dataCopyParams.srcStride = 0;
    dataCopyParams.dstStride = 0;

    if (hasGamma) {
        DataCopyPad(gammaTensor, gammaGm, dataCopyParams, padParams);
    }
    if (hasBeta) {
        DataCopyPad(betaTensor, betaGm, dataCopyParams, padParams);
    }
}

template <typename T>
__aicore__ inline void CopyX2UB(const GlobalTensor<T>& inputGm, const LocalTensor<T>& inputTensor,
                                const uint16_t blockCount, const uint32_t copyLen)
{
    int32_t copyLenAlign = RoundUp<T>(copyLen);
    DataCopyPadExtParams<T> padParams;
    padParams.isPad = true;
    padParams.paddingValue = static_cast<T>(0.0);
    padParams.rightPadding = copyLenAlign - copyLen;

    DataCopyExtParams dataCopyParams;
    dataCopyParams.blockCount = blockCount;
    dataCopyParams.blockLen = copyLen * sizeof(T);
    dataCopyParams.srcStride = 0;
    dataCopyParams.dstStride = 0;
    DataCopyPad(inputTensor, inputGm, dataCopyParams, padParams);
}

template <typename T>
__aicore__ inline void CopyMeanAndVariance2Gm(const GlobalTensor<T>& meanGm, const GlobalTensor<T>& varianceGm,
                                              const LocalTensor<T>& meanTensor, const LocalTensor<T>& varianceTensor,
                                              const uint16_t blockCount, const uint32_t copyLen)
{
    DataCopyExtParams dataCopyParams;
    dataCopyParams.blockCount = blockCount;
    dataCopyParams.blockLen = copyLen * sizeof(T);
    dataCopyParams.srcStride = 0;
    dataCopyParams.dstStride = 0;
    DataCopyPad(meanGm, meanTensor, dataCopyParams);
    DataCopyPad(varianceGm, varianceTensor, dataCopyParams);
}

template <typename T1>
__aicore__ inline void ProcessMeanAndVariance(LocalTensor<float>& meanTensor, LocalTensor<T1>& meanOutTensor,
                                              GlobalTensor<T1>& meanGm, LocalTensor<T1>& varianceOutTensor,
                                              GlobalTensor<T1>& varianceGm, uint64_t gmOffset, uint32_t curNumPerCore)
{
    uint16_t loopCount = CeilDiv(curNumPerCore, VL_FP32);
    __VEC_SCOPE__
    {
        if constexpr (!IsSameType<T1, float>::value) {
            __local_mem__ T1* meanOutLocal = (__local_mem__ T1*)meanOutTensor.GetPhyAddr();
            __local_mem__ float* meanLocal = (__local_mem__ float*)meanTensor.GetPhyAddr();
            uint32_t sreg0 = curNumPerCore;
            MaskReg pregLoop;
            for (uint16_t i = 0; i < loopCount; i++) {
                pregLoop = UpdateMask<float>(sreg0);
                RegTensor<float> mean;
                RegTensor<T1> meanOut;
                DataCopy(mean, meanLocal + i * VL_FP32);
                Cast<T1, float, castTraitB322B16Even>(meanOut, mean, pregLoop);
                DataCopy<T1, AscendC::Reg::StoreDist::DIST_PACK_B32>(meanOutLocal + i * VL_FP32, meanOut, pregLoop);
            }
        }
    }
    event_t eventIdVToMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
    SetFlag<HardEvent::V_MTE3>(eventIdVToMte3);
    WaitFlag<HardEvent::V_MTE3>(eventIdVToMte3);
    if constexpr (IsSameType<T1, float>::value) {
        CopyMeanAndVariance2Gm<float>(meanGm[gmOffset], varianceGm[gmOffset], meanTensor, varianceOutTensor, 1,
                                      curNumPerCore);
    } else {
        CopyMeanAndVariance2Gm<T1>(meanGm[gmOffset], varianceGm[gmOffset], meanOutTensor, varianceOutTensor, 1,
                                   curNumPerCore);
    }
}

template <typename T>
__aicore__ inline void CopyY2Gm(const GlobalTensor<T>& yGm, const LocalTensor<T>& yTensor, uint16_t blockCount,
                                const uint32_t copyLen)
{
    DataCopyExtParams dataCopyParams;
    dataCopyParams.blockCount = blockCount;
    dataCopyParams.blockLen = copyLen * sizeof(T);
    dataCopyParams.srcStride = 0;
    dataCopyParams.dstStride = 0;
    DataCopyPad(yGm, yTensor, dataCopyParams);
}

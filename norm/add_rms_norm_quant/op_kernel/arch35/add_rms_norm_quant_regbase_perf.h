/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file add_rms_norm_quant_regbase_perf.h
 * \brief
 */
#ifndef ADD_RMS_NORM_QUANT_REGBASE_PERF_H_
#define ADD_RMS_NORM_QUANT_REGBASE_PERF_H_

#include "kernel_utils.h"
#include "../inc/platform.h"
#include "add_rms_norm_quant_regbase_common.h"
#include "../../norm_common/reduce_common_regbase.h"

namespace AddRmsNormQuant {

template <typename T_X, typename T_Y, typename T_SCALES, typename T_ZEROPOINTS, uint64_t TILING_KEY>
class KernelAddRmsNormQuantRegbasePerf {
#define DIV_MODE ((TILING_KEY / 100) == 1)
#define INPUT_KEY ((TILING_KEY % 100) / 10)
#define HAS_ZEROPOINTS1 ((INPUT_KEY >> 2) % 2 == 1)
#define HAS_SCALE2 ((INPUT_KEY >> 1) % 2 == 1)
#define HAS_ZEROPOINTS2 (INPUT_KEY % 2 == 1)
#define HAS_Y2 (HAS_SCALE2 || HAS_ZEROPOINTS2)
public:
    __aicore__ inline KernelAddRmsNormQuantRegbasePerf(TPipe* pipe)
    {
        pipe_ = pipe;
    }

    __aicore__ inline void Init(
        GM_ADDR x1, GM_ADDR x2, GM_ADDR gamma, GM_ADDR scales1, GM_ADDR scales2, GM_ADDR zeroPoints1,
        GM_ADDR zeroPoints2, GM_ADDR y1, GM_ADDR y2, GM_ADDR x, const AddRmsNormQuantRegbaseTilingData* tilingData)
    {
        numM = tilingData->numM;
        numN = tilingData->numN;
        baseM = tilingData->baseM;
        baseN = tilingData->baseN;
        baseNDtypeAlign = tilingData->baseNDtypeAlign;
        powerSplit = tilingData->powerSplit;
        mPerCore = tilingData->mPerCore;
        mLastCore = tilingData->mLastCore;
        epsilon = tilingData->epsilon;
        avgFactor = tilingData->avgFactor;

        // dtype size
        xDtypeSize = blockSize / sizeof(T_X);
        scalesDtypeSize = blockSize / sizeof(T_SCALES);
        zeroPointsDtypeSize = blockSize / sizeof(T_ZEROPOINTS);
        yDtypeSize = blockSize / sizeof(T_Y);


        // dtype align
        xGammaAlign = CeilDiv(baseN, static_cast<int64_t>(xDtypeSize)) * static_cast<int64_t>(xDtypeSize);
        scalesAlign = CeilDiv(baseN, static_cast<int64_t>(scalesDtypeSize)) * static_cast<int64_t>(scalesDtypeSize);
        zeroPointsAlign = CeilDiv(baseN, static_cast<int64_t>(zeroPointsDtypeSize)) * static_cast<int64_t>(zeroPointsDtypeSize);
        yAlign = CeilDiv(baseN, static_cast<int64_t>(yDtypeSize)) * static_cast<int64_t>(yDtypeSize);
        rstdAlign = CeilDiv(baseM, static_cast<int64_t>(blockSizeB32)) * static_cast<int64_t>(blockSizeB32);
        
        blockNum = GetBlockNum();
        blockIdx = GetBlockIdx();

        CalBlockTail();
        InitBuffer(x1, x2, gamma, scales1, scales2, zeroPoints1, zeroPoints2, y1, y2, x);
    }

    __aicore__ inline void CalBlockTail()
    {
        mCore = blockIdx == (blockNum - 1) ? mLastCore : mPerCore;
        mLoops = CeilDiv(mCore, baseM);
        mTail = mCore - (mLoops - 1) * baseM;
    }

    __aicore__ inline void InitBuffer(
        GM_ADDR x1, GM_ADDR x2, GM_ADDR gamma, GM_ADDR scales1, GM_ADDR scales2, GM_ADDR zeroPoints1,
        GM_ADDR zeroPoints2, GM_ADDR y1, GM_ADDR y2, GM_ADDR x)
    {
        uint64_t gmOffset = blockIdx * mPerCore * numN;
        uint64_t gmLen = mCore * numN;
        x1Gm.SetGlobalBuffer((__gm__ T_X*)x1 + gmOffset, gmLen);
        x2Gm.SetGlobalBuffer((__gm__ T_X*)x2 + gmOffset, gmLen);
        y1Gm.SetGlobalBuffer((__gm__ T_Y*)y1 + gmOffset, gmLen);
        xGm.SetGlobalBuffer((__gm__ T_X*)x + gmOffset, gmLen);
        gammaGm.SetGlobalBuffer((__gm__ T_X*)gamma, numN);
        scales1Gm.SetGlobalBuffer((__gm__ T_SCALES*)scales1, numN);

        // gamma + scales1
        int64_t preloadDataSize = xGammaAlign * sizeof(T_X) + scalesAlign * sizeof(T_SCALES);
        if constexpr (HAS_SCALE2) {
            scales2Gm.SetGlobalBuffer((__gm__ T_SCALES*)scales2, numN);
            preloadDataSize = preloadDataSize + scalesAlign * sizeof(T_SCALES);
        }
        if constexpr (HAS_ZEROPOINTS1) {
            zeroPoints1Gm.SetGlobalBuffer((__gm__ T_ZEROPOINTS*)zeroPoints1, numN);
            preloadDataSize = preloadDataSize + zeroPointsAlign * sizeof(T_ZEROPOINTS);
        }
        if constexpr (HAS_ZEROPOINTS2) {
            zeroPoints2Gm.SetGlobalBuffer((__gm__ T_ZEROPOINTS*)zeroPoints2, numN);
            preloadDataSize = preloadDataSize + zeroPointsAlign * sizeof(T_ZEROPOINTS);
        }
        if constexpr (HAS_Y2) {
            y2Gm.SetGlobalBuffer((__gm__ T_Y*)y2 + gmOffset, gmLen);
        }

        pipe_->InitBuffer(inQueueX1, DOUBLE_BUFFER_NUM, baseM * xGammaAlign * sizeof(T_X));
        pipe_->InitBuffer(inQueueX2, DOUBLE_BUFFER_NUM, baseM * xGammaAlign * sizeof(T_X));
        // preload data
        pipe_->InitBuffer(inQueueOther, 1, preloadDataSize);
        int64_t yQueueSize = baseM * yAlign * sizeof(T_Y);
        pipe_->InitBuffer(outQueueY1, DOUBLE_BUFFER_NUM, yQueueSize);
        if (HAS_Y2) {
            pipe_->InitBuffer(outQueueY2, DOUBLE_BUFFER_NUM, yQueueSize);
        }
        pipe_->InitBuffer(xOutTmpBuf, baseM * xGammaAlign * sizeof(float));
        int64_t XQueueSize = baseM * xGammaAlign * sizeof(T_X);
        pipe_->InitBuffer(outQueueX, DOUBLE_BUFFER_NUM, XQueueSize);
        pipe_->InitBuffer(rstdBuf, rstdAlign * sizeof(float));
        // reduceTmpBuffer
        int64_t reduceTmpBufferSize = baseM * CeilDiv(CeilDiv(powerSplit, static_cast<int64_t>(vectorLenB32)), static_cast<int64_t>(blockSizeB32)) * blockSizeB32;
        pipe_->InitBuffer(reduceTmpBuf, reduceTmpBufferSize);
    }

    __aicore__ inline void Process()
    {        
        // copy other input (gamma scales zeropints)
        LocalTensor<uint8_t> otherLocal = inQueueOther.AllocTensor<uint8_t>();
        CopyInOthers(otherLocal);
        inQueueOther.EnQue(otherLocal);
        inQueueOther.DeQue<uint8_t>();

        for (uint64_t i = 0; i < mLoops; i++) {
            uint64_t realM = i == (mLoops - 1) ? mTail : baseM;
            uint64_t mOuterOffset = i * baseM;
            uint64_t gmOffset = mOuterOffset * baseN;

            // 1.x1 + x2
            CopyInXMutiMoveAlign(gmOffset, realM);
            LocalTensor<T_X> xLocal1 = inQueueX1.DeQue<T_X>();
            LocalTensor<T_X> xLocal2 = inQueueX2.DeQue<T_X>();
            LocalTensor<T_X> xLocal = outQueueX.AllocTensor<T_X>();
            LocalTensor<float> xOutTmpLocal = xOutTmpBuf.Get<float>();            
            CalculateXAdd(xLocal1, xLocal2, xLocal, xOutTmpLocal, realM);
            inQueueX1.FreeTensor(xLocal1);
            inQueueX2.FreeTensor(xLocal2);
            outQueueX.EnQue<T_X>(xLocal);
            CopyOutX(gmOffset, realM);

            // 2.SquareReduceSum、Rstd
            LocalTensor<float> rstdLocal = rstdBuf.Get<float>();
            LocalTensor<float> reduceTmpLocal = reduceTmpBuf.Get<float>();
            CalculateSquareReduceSum(xOutTmpLocal, reduceTmpLocal, realM);
            CalculateRstd(reduceTmpLocal, rstdLocal, realM);

            LocalTensor<T_Y> y1Local = outQueueY1.AllocTensor<T_Y>();
            LocalTensor<T_Y> y2Local;
            if constexpr (HAS_Y2) {
                y2Local = outQueueY2.AllocTensor<T_Y>();
            }

            // 3. Quant
            CalculateQuant(xOutTmpLocal,rstdLocal,gammaLocal,scales1Local,scales2Local,zeroPoints1Local,zeroPoints2Local,y1Local,y2Local,realM,baseN,xGammaAlign,scalesAlign,zeroPointsAlign,yAlign);

            outQueueY1.EnQue<T_Y>(y1Local);
            if constexpr (HAS_Y2) {
                outQueueY2.EnQue<T_Y>(y2Local);
            }
            CopyOutY(gmOffset, realM);

        }
        inQueueOther.FreeTensor(otherLocal);
    }

private:
    __aicore__ inline void CopyInOthers(LocalTensor<uint8_t> otherLocal)
    {
        uint32_t localOffset = 0;
        // LocalTensor<T_X> gammaLocal
        gammaLocal = otherLocal[localOffset].ReinterpretCast<T_X>();
        localOffset = localOffset + xGammaAlign * sizeof(T_X);
        DataCopyPadExtParams<T_X> dataCopyPadExtParamsGamma;
        dataCopyPadExtParamsGamma.isPad = false;
        dataCopyPadExtParamsGamma.leftPadding = 0;
        dataCopyPadExtParamsGamma.rightPadding = 0;
        dataCopyPadExtParamsGamma.paddingValue = 0;
        DataCopyExtParams copyInParamsGamma;
        copyInParamsGamma.blockCount = 1;
        copyInParamsGamma.blockLen = xGammaAlign * sizeof(T_X);
        copyInParamsGamma.srcStride = 0;
        copyInParamsGamma.dstStride = 0;
        DataCopyPad(gammaLocal, gammaGm, copyInParamsGamma, dataCopyPadExtParamsGamma);

        // LocalTensor<T_SCALES> scales1Local;
        scales1Local = otherLocal[localOffset].ReinterpretCast<T_SCALES>();
        localOffset = localOffset + scalesAlign * sizeof(T_SCALES);
        DataCopyPadExtParams<T_SCALES> dataCopyPadExtParamsScales;
        dataCopyPadExtParamsScales.isPad = false;
        dataCopyPadExtParamsScales.leftPadding = 0;
        dataCopyPadExtParamsScales.rightPadding = 0;
        dataCopyPadExtParamsScales.paddingValue = 0;
        DataCopyExtParams copyInParamsScales;
        copyInParamsScales.blockCount = 1;
        copyInParamsScales.blockLen = scalesAlign * sizeof(T_SCALES);
        copyInParamsScales.srcStride = 0;
        copyInParamsScales.dstStride = 0;
        DataCopyPad(scales1Local, scales1Gm, copyInParamsScales, dataCopyPadExtParamsScales);

        // zeroPoints
        DataCopyPadExtParams<T_ZEROPOINTS> dataCopyPadExtParamszeroPoints;
        dataCopyPadExtParamszeroPoints.isPad = false;
        dataCopyPadExtParamszeroPoints.leftPadding = 0;
        dataCopyPadExtParamszeroPoints.rightPadding = 0;
        dataCopyPadExtParamszeroPoints.paddingValue = 0;
        DataCopyExtParams copyInParamszeroPoints;
        copyInParamszeroPoints.blockCount = 1;
        copyInParamszeroPoints.blockLen = zeroPointsAlign * sizeof(T_ZEROPOINTS);
        copyInParamszeroPoints.srcStride = 0;
        copyInParamszeroPoints.dstStride = 0;
                
        if (HAS_SCALE2) {
            // LocalTensor<T_SCALES> scales2Local;
            scales2Local = otherLocal[localOffset].ReinterpretCast<T_SCALES>();
            localOffset = localOffset + scalesAlign * sizeof(T_SCALES);
            DataCopyPad(scales2Local, scales2Gm, copyInParamsScales, dataCopyPadExtParamsScales);
        }

        if (HAS_ZEROPOINTS1) {
            // LocalTensor<T_ZEROPOINTS> zeroPoints1Local;
            zeroPoints1Local = otherLocal[localOffset].ReinterpretCast<T_ZEROPOINTS>();
            localOffset = localOffset + zeroPointsAlign * sizeof(T_ZEROPOINTS);
            DataCopyPad(zeroPoints1Local, zeroPoints1Gm, copyInParamszeroPoints, dataCopyPadExtParamszeroPoints);
        }

        if (HAS_ZEROPOINTS2) {
            // LocalTensor<T_ZEROPOINTS> zeroPoints2Local;
            zeroPoints2Local = otherLocal[localOffset].ReinterpretCast<T_ZEROPOINTS>();
            localOffset = localOffset + zeroPointsAlign * sizeof(T_ZEROPOINTS);
            DataCopyPad(zeroPoints2Local, zeroPoints2Gm, copyInParamszeroPoints, dataCopyPadExtParamszeroPoints);
        }

    }
    __aicore__ inline void CopyInXMutiMoveAlign(uint64_t gmOffset, uint32_t realM)
    {
        LocalTensor<T_X> xLocal1 = inQueueX1.AllocTensor<T_X>();
        LocalTensor<T_X> xLocal2 = inQueueX2.AllocTensor<T_X>();
        DataCopyExtParams extParams{
            static_cast<uint16_t>(realM),                // blockCount
            static_cast<uint32_t>(baseN * sizeof(T_X)), // blockLen
            static_cast<uint32_t>(0),                    // srcStride
            static_cast<uint32_t>(0),                    // dstStride
            0                                            // rsv
        };
        DataCopyPadExtParams<T_X> padParams{
            false,                   // isPad
            static_cast<uint8_t>(0), // leftPadding
            static_cast<uint8_t>(0), // rightPadding
            static_cast<T_X>(0.0)    // paddingValue
        };
        DataCopyPad(xLocal1, x1Gm[gmOffset], extParams, padParams);
        DataCopyPad(xLocal2, x2Gm[gmOffset], extParams, padParams);
        inQueueX1.EnQue(xLocal1);
        inQueueX2.EnQue(xLocal2);
    }

template <typename T_IN>
__aicore__ inline void LoadTensorForDtypeTIn(
    __local_mem__ T_IN* src, RegTensor<float>& dst, MaskReg& preg, uint32_t offset)
{
    if constexpr (IsSameType<T_IN, float>::value) {
        DataCopy<float, LoadDist::DIST_NORM>(dst, src + offset);
    } else if constexpr (IsSameType<T_IN, int32_t>::value) {
        RegTensor<T_IN> xIn;
        DataCopy<int32_t, LoadDist::DIST_NORM>(xIn, src + offset);
        Cast<float, T_IN, castTraitInt322Fp32>(dst, xIn, preg);
    } else {
        RegTensor<T_IN> xIn;
        DataCopy<T_IN, LoadDist::DIST_UNPACK_B16>(xIn, src + offset);
        Cast<float, T_IN, castTraitF162F32>(dst, xIn, preg);
    }
}

template <typename T_OUT>
__aicore__ inline void StoreTensorForDtypeTOut(
    __local_mem__ T_OUT* dst, RegTensor<float>& xRegFp32, MaskReg& preg, uint32_t offset)
{
    if constexpr (IsSameType<T_OUT, float>::value) {
        DataCopy<T_OUT, StoreDist::DIST_NORM>(dst + offset, xRegFp32, preg);
    } else if constexpr (IsSameType<T_OUT, int8_t>::value) {
        RegTensor<T_OUT> xOut;
        RegTensor<half> xRegFp16;
        Cast<half, float, castTraitFp322Fp16>(xRegFp16, xRegFp32, preg);
        Cast<int8_t, half, castTraitFp162Int8>(xOut, xRegFp16, preg);
        DataCopy<T_OUT, StoreDist::DIST_PACK4_B32>(dst + offset, xOut, preg);
    } else if constexpr (IsSameType<T_OUT, fp8_e4m3fn_t>::value || IsSameType<T_OUT, fp8_e5m2_t>::value) {
        RegTensor<T_OUT> xOut;
        Cast<T_OUT, float, castTraitFp322Fp8>(xOut, xRegFp32, preg);
        DataCopy<T_OUT, StoreDist::DIST_PACK4_B32>(dst + offset, xOut, preg);
    } else if constexpr (IsSameType<T_OUT, hifloat8_t>::value) {
        RegTensor<T_OUT> xOut;
        Cast<T_OUT, float, castTraitFp322Hifp8>(xOut, xRegFp32, preg);
        DataCopy<T_OUT, StoreDist::DIST_PACK4_B32>(dst + offset, xOut, preg);
    }else {
        RegTensor<T_OUT> xOut;
        Cast<T_OUT, float, castTraitB322B16>(xOut, xRegFp32, preg);
        DataCopy<T_OUT, StoreDist::DIST_PACK_B32>(dst + offset, xOut, preg);        
    }
}

    __aicore__ inline void CopyOutX(uint64_t offset, uint32_t realM)
    {
        LocalTensor<T_X> xLocal = outQueueX.DeQue<T_X>();
        DataCopyExtParams copyParams{
            static_cast<uint16_t>(realM),                // blockCount
            static_cast<uint32_t>(baseN * sizeof(T_X)), // blockLen
            static_cast<uint32_t>(0),                    // srcStride
            static_cast<uint32_t>(0),                    // dstStride
            0                                            // rsv
        };
        DataCopyPad(xGm[offset], xLocal, copyParams);
        outQueueX.FreeTensor(xLocal);
    }

    __aicore__ inline void CopyOutY(uint64_t offset, uint32_t realM)
    {
        LocalTensor<T_Y> y1Local = outQueueY1.DeQue<T_Y>();
        DataCopyExtParams copyParams{
            static_cast<uint16_t>(realM),                // blockCount
            static_cast<uint32_t>(baseN * sizeof(T_Y)), // blockLen
            static_cast<uint32_t>(0),                    // srcStride
            static_cast<uint32_t>(0),                    // dstStride
            0                                            // rsv
        };
        DataCopyPad(y1Gm[offset], y1Local, copyParams);
        outQueueY1.FreeTensor(y1Local);
        if constexpr (HAS_Y2) {
            LocalTensor<T_Y> y2Local = outQueueY2.DeQue<T_Y>();
            DataCopyPad(y2Gm[offset], y2Local, copyParams);
            outQueueY2.FreeTensor(y2Local);
        }
    }

    __aicore__ inline void CalculateXAdd(
        LocalTensor<T_X>& xLocal1, LocalTensor<T_X>& xLocal2, LocalTensor<T_X>& xLocal, LocalTensor<float>& xOutTmpLocal, uint32_t realM)
    {
        __local_mem__ T_X* x1InUb = (__local_mem__ T_X*)xLocal1.GetPhyAddr();
        __local_mem__ T_X* x2InUb = (__local_mem__ T_X*)xLocal2.GetPhyAddr();
        __local_mem__ T_X* xOutInUb = (__local_mem__ T_X*)xLocal.GetPhyAddr();
        __local_mem__ float* xOutTmp = (__local_mem__ float*)xOutTmpLocal.GetPhyAddr();
        uint32_t sreg = realM * baseNDtypeAlign;
        uint16_t loopCount = (sreg + V_LENGTH - 1) / V_LENGTH;

        __VEC_SCOPE__
        {
            RegTensor<float> x1;
            RegTensor<float> x2;
            RegTensor<float> xSum;
            MaskReg pregLoop;
            for (uint16_t i = 0; i < loopCount; ++i) {
                uint32_t offset = i * V_LENGTH;
                pregLoop = UpdateMask<float>(sreg);
                LoadTensorForDtypeTIn<T_X>(x1InUb, x1, pregLoop, offset);
                LoadTensorForDtypeTIn<T_X>(x2InUb, x2, pregLoop, offset);
                Add(xSum, x1, x2, pregLoop);
                StoreTensorForDtypeTOut<T_X>(xOutInUb, xSum, pregLoop, offset);
                DataCopy<float, StoreDist::DIST_NORM_B32>(xOutTmp + offset, xSum, pregLoop);   
            }
        }
    }

    __aicore__ inline void CalculateSquareReduceSum(
        LocalTensor<float>& xOutTmpLocal, LocalTensor<float>& xReduceLocal, uint32_t realM)
    {
        LocalTensor<float> binaryAddBuffTmp = reduceTmpBuf.Get<float>();
        __local_mem__ float* xOutTmp = (__local_mem__ float*)xOutTmpLocal.GetPhyAddr();
        __local_mem__ float* xReduceUb = (__local_mem__ float*)xReduceLocal.GetPhyAddr();
        __local_mem__ float* tmpUb = (__local_mem__ float*)binaryAddBuffTmp.GetPhyAddr();

        if (baseN <= V_LENGTH) {
            CalculateSquareReduceSumLessThanVL(xOutTmp, xReduceUb, realM, baseN, baseNDtypeAlign);
        } else if (baseN <= V_LENGTH + V_LENGTH) {
            CalculateSquareReduceSumLessThanTwoVL(xOutTmp, xReduceUb, realM, baseN, baseNDtypeAlign);
        } else if (baseN <= V_LENGTH * V_LENGTH * NUM_TWO) {
            CalculateSquareReduceSumCommon<NUM_ONE>(xOutTmp, xReduceUb, tmpUb, realM, baseN, baseNDtypeAlign);
        } else {
            CalculateSquareReduceSumCommon<NUM_TWO>(xOutTmp, xReduceUb, tmpUb, realM, baseN, baseNDtypeAlign);
        }
    }
    __aicore__ inline void CalculateSquareReduceSumLessThanVL(
        __local_mem__ float* xOutTmp, __local_mem__ float* xReduceUb, uint16_t realM, uint64_t baseN, uint64_t baseNDtypeAlign)
    {
        __VEC_SCOPE__
        {
            RegTensor<float> x;
            RegTensor<float> vMean;
            RegTensor<float> rstdReg;
            RegTensor<float> onesReg;

            uint32_t sreg0 = baseN;
            MaskReg pregLoop = UpdateMask<float>(sreg0);
            MaskReg pregFull = CreateMask<float, MaskPattern::ALL>();
            MaskReg pregOne = CreateMask<float, MaskPattern::VL1>();
            Duplicate(onesReg, float(1.0), pregOne);

            for (uint16_t i = 0; i < realM; i++) {
                LoadTensorForDtypeTIn<float>(xOutTmp, x, pregLoop, i * baseNDtypeAlign);
                Mul(x, x, x, pregLoop);
                ReduceSum(vMean, x, pregLoop);
                DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(xReduceUb + i, vMean, pregOne);
            }
        }
    }

    __aicore__ inline void CalculateSquareReduceSumLessThanTwoVL(
        __local_mem__ float* xOutTmp, __local_mem__ float* xReduceUb, uint16_t realM, uint64_t baseN, uint64_t baseNDtypeAlign)
    {
        uint32_t tailLen = baseN - V_LENGTH;
        __VEC_SCOPE__
        {
            RegTensor<float> x;
            RegTensor<float> xFold;
            RegTensor<float> sumReg;
            RegTensor<float> vMean;
            RegTensor<float> rstdReg;
            RegTensor<float> onesReg;

            MaskReg pregFull = CreateMask<float, MaskPattern::ALL>();
            MaskReg pregOne = CreateMask<float, MaskPattern::VL1>();
            MaskReg pregTail = UpdateMask<float>(tailLen);
            Duplicate(onesReg, float(1.0), pregOne);

            for (uint16_t i = 0; i < realM; ++i) {
                LoadTensorForDtypeTIn<float>(xOutTmp, x, pregFull, i * baseNDtypeAlign);
                LoadTensorForDtypeTIn<float>(xOutTmp + V_LENGTH, xFold, pregTail, i * baseNDtypeAlign);
                Mul(x, x, x, pregFull);
                Mul(xFold, xFold, xFold, pregTail);
                ShiftLefts((RegTensor<uint32_t>&)xFold, (RegTensor<uint32_t>&)xFold, static_cast<int16_t>(0), pregTail);
                Add(sumReg, x, xFold, pregFull);
                ReduceSum(vMean, sumReg, pregFull);
                DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(xReduceUb + i, vMean, pregOne);
            }
        }
    }

    template <int32_t LAST_LOOP_NUMS>
    __aicore__ inline void CalculateSquareReduceSumCommon(
        __local_mem__ float* xOutTmp, __local_mem__ float* xReduceUb, __local_mem__ float* tmpUb, uint16_t realM, uint64_t baseN, uint64_t baseNDtypeAlign)
    {
        uint32_t binaryAddQuotient = powerSplit;
        uint16_t binaryAddQuotientLoop = (binaryAddQuotient + V_LENGTH - 1) / V_LENGTH;

        uint32_t lastBinaryAddNum = binaryAddQuotient / V_LENGTH;
        uint32_t lastBinaryAddNumAlign = (binaryAddQuotientLoop + B32_BLOCK_NUM - 1) / B32_BLOCK_NUM * B32_BLOCK_NUM;

        uint32_t binaryAddRemainder = baseN - binaryAddQuotient;
        uint16_t binaryAddRemainderCeilLoop = (binaryAddRemainder + V_LENGTH - 1) / V_LENGTH;
        uint16_t binaryAddRemainderFloorLoop = binaryAddRemainder / V_LENGTH;
        __VEC_SCOPE__
        {
            RegTensor<float> x;
            RegTensor<float> xFold;
            RegTensor<float> sumReg;
            RegTensor<float> vMean;
            RegTensor<float> rstdReg;
            RegTensor<float> onesReg;

            MaskReg pregFull = CreateMask<float, MaskPattern::ALL>();
            MaskReg pregOne = CreateMask<float, MaskPattern::VL1>();
            MaskReg pregLoop;
            Duplicate(onesReg, float(1.0), pregOne);

            for (uint16_t i = 0; i < realM; ++i) {
                uint32_t baseOffset = i * baseNDtypeAlign;
                for (uint16_t r = 0; r < binaryAddRemainderFloorLoop; ++r) {
                    uint32_t offset = r * V_LENGTH + baseOffset;
                    LoadTensorForDtypeTIn<float>(xOutTmp, x, pregFull, offset);
                    LoadTensorForDtypeTIn<float>(xOutTmp + binaryAddQuotient, xFold, pregFull, offset);
                    Mul(x, x, x, pregFull);
                    Mul(xFold, xFold, xFold, pregFull);
                    Add(sumReg, x, xFold, pregFull);
                    ReduceSum(vMean, sumReg, pregFull);
                    DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(
                        tmpUb + static_cast<uint32_t>(i * lastBinaryAddNumAlign + r), vMean, pregOne);
                }
                uint32_t sregRemainder = binaryAddRemainder - binaryAddRemainderFloorLoop * V_LENGTH;
                for (uint16_t r = 0;
                     r < static_cast<uint16_t>(binaryAddRemainderCeilLoop - binaryAddRemainderFloorLoop); r++) {
                    uint16_t offset = baseOffset;
                    pregLoop = UpdateMask<float>(sregRemainder);
                    LoadTensorForDtypeTIn<float>(xOutTmp + binaryAddRemainderFloorLoop * V_LENGTH, x, pregFull, offset);
                    LoadTensorForDtypeTIn<float>(
                        xOutTmp + binaryAddRemainderFloorLoop * V_LENGTH + binaryAddQuotient, xFold, pregLoop, offset);
                    Mul(x, x, x, pregFull);
                    Mul(xFold, xFold, xFold, pregLoop);
                    ShiftLefts(
                        (RegTensor<uint32_t>&)xFold, (RegTensor<uint32_t>&)xFold, static_cast<int16_t>(0), pregLoop);
                    Add(sumReg, x, xFold, pregFull);
                    ReduceSum(vMean, sumReg, pregFull);
                    DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(
                        tmpUb + static_cast<uint32_t>(i * lastBinaryAddNumAlign + binaryAddRemainderFloorLoop), vMean,
                        pregOne);
                }
                for (uint16_t r = 0; r < static_cast<uint16_t>(binaryAddQuotientLoop - binaryAddRemainderCeilLoop);
                     r++) {
                    uint32_t offset = r * V_LENGTH + baseOffset;
                    LoadTensorForDtypeTIn<float>(xOutTmp + binaryAddRemainderCeilLoop * V_LENGTH, x, pregFull, offset);
                    Mul(x, x, x, pregFull);
                    ReduceSum(vMean, x, pregFull);
                    DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(
                        tmpUb + static_cast<uint32_t>(i * lastBinaryAddNumAlign + binaryAddRemainderCeilLoop + r),
                        vMean, pregOne);
                }
            }
            LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
            if constexpr (LAST_LOOP_NUMS == 1) {
                MaskReg pregLast = UpdateMask<float>(lastBinaryAddNum);
                for (uint16_t i = 0; i < realM; ++i) {
                    DataCopy(x, tmpUb + static_cast<uint32_t>(i * lastBinaryAddNumAlign));
                    ReduceSum(vMean, x, pregLast);
                    DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(xReduceUb + i, vMean, pregOne);
                }
            } else if constexpr (LAST_LOOP_NUMS == 2) {
                lastBinaryAddNum -= V_LENGTH;
                MaskReg pregLast = UpdateMask<float>(lastBinaryAddNum);
                for (uint16_t i = 0; i < realM; ++i) {
                    DataCopy(x, tmpUb + static_cast<uint32_t>(i * lastBinaryAddNumAlign));
                    DataCopy(xFold, tmpUb + static_cast<uint32_t>(i * lastBinaryAddNumAlign + V_LENGTH));
                    ShiftLefts(
                        (RegTensor<uint32_t>&)xFold, (RegTensor<uint32_t>&)xFold, static_cast<int16_t>(0), pregLast);
                    Add(sumReg, x, xFold, pregFull);
                    ReduceSum(vMean, sumReg, pregFull);
                    DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(xReduceUb + i, vMean, pregOne);
                }
            }
        }
    }

    __aicore__ inline void CalculateRstd(
        LocalTensor<float>& xReduceLocal, LocalTensor<float>& rstdLocal, uint32_t realM)
    {
        static constexpr float POS_INF = 3.40282366920938E+38;
        static constexpr float SCALAR1 = -0.5;
        static constexpr float SCALAR2 = 1.5;
        static constexpr float SCALAR3 = 0.5;
        static constexpr float SCALAR0 = -99.99;

        __local_mem__ float* rstdInUb = (__local_mem__ float*)rstdLocal.GetPhyAddr();
        __local_mem__ float* xReduceUb = (__local_mem__ float*)xReduceLocal.GetPhyAddr();
        uint16_t loopRows = static_cast<uint16_t>((realM + V_LENGTH - 1) / V_LENGTH);
        __VEC_SCOPE__
        {
            RegTensor<float> var;
            RegTensor<float> rstd;
            RegTensor<float> r;
            RegTensor<float> y;
            RegTensor<float> s;
            RegTensor<float> t;
            RegTensor<float> one;
            RegTensor<float> scalar1;
            RegTensor<float> t1;
            RegTensor<float> t2;
            RegTensor<float> t3;
            RegTensor<float> t4;
            RegTensor<float> scalarInf;
            RegTensor<float> scalarZero;
            MaskReg cmpRegZero;
            MaskReg cmpRegInf;
            MaskReg pregFull = CreateMask<float, MaskPattern::ALL>();
            MaskReg pregLoop;

            uint32_t sreg = static_cast<uint32_t>(realM);
            for (uint16_t i = 0; i < loopRows; ++i) {
                pregLoop = UpdateMask<float>(sreg);
                Duplicate(scalarInf, POS_INF, pregLoop);
                Duplicate(scalarZero, float(0.0), pregLoop);
                Duplicate(one, float(1.0), pregLoop);
                Duplicate(scalar1, SCALAR3, pregLoop);
                Duplicate(t1, SCALAR2, pregLoop);
                Duplicate(s, float(1.0), pregLoop);
                // rstd
                DataCopy(var, xReduceUb + i * V_LENGTH);
                Muls(var, var, avgFactor, pregLoop);
                Adds(var, var, epsilon, pregLoop);
                Maxs(var, var, SCALAR0, pregLoop);
                Div(r, one, var, pregLoop);
                Sqrt(y, r, pregLoop);
                Muls(t, var, SCALAR1, pregLoop);
                Mul(t, t, y, pregLoop);                // -0.5 * x * y
                Mula(t1, t, y, pregLoop);              // 1.5 + (-0.5 * x * y) * y
                Mul(rstd, y, t1, pregLoop);            // y = y * (1.5 - 0.5 * x * y)
                Muls(t3, var, float(-1.0), pregLoop);  // -1 * x
                Mula(s, t3, r, pregLoop);              // 1 + (-1) * x * r
                Muls(t4, rstd, float(-1.0), pregLoop); // (-1) * y
                Mula(r, t4, rstd, pregLoop);           // r + (-1) * y * y
                Mula(s, var, r, pregLoop);             // s + x * t
                Mul(s, s, rstd, pregLoop);             // e * y
                Mula(rstd, s, scalar1, pregLoop);      // y + y * e * 0.5
                CompareScalar(cmpRegZero, var, POS_INF, pregLoop);
                Select(rstd, scalarZero, rstd, cmpRegZero);
                CompareScalar(cmpRegInf, var, float(0.0), pregLoop);
                Select(rstd, scalarInf, rstd, cmpRegInf);
                DataCopy(rstdInUb + i * V_LENGTH, rstd, pregLoop);
            }
        }
    }

    // template <bool HAS_ZEROPOINTS2, bool HAS_ZEROPOINTS1, bool HAS_SCALE2, bool DIV_MODE>
    __aicore__ inline void CalculateQuant(LocalTensor<float> xLocal, LocalTensor<float> rstdLocal, 
        LocalTensor<T_X> gammaLocal, 
        LocalTensor<T_SCALES> scales1Local, LocalTensor<T_SCALES> scales2Local, 
        LocalTensor<T_ZEROPOINTS> zeroPoints1Local, LocalTensor<T_ZEROPOINTS> zeroPoints2Local, 
        LocalTensor<T_Y> y1Local, LocalTensor<T_Y> y2Local, 
        int64_t realM, int64_t baseN, int64_t xGammaAlign, int64_t scalesAlign, 
        int64_t zeroPointsAlign, int64_t yAlign)
    {
        uint16_t loopsA = static_cast<uint16_t>(realM);
        uint16_t loopsR = static_cast<uint16_t>(CeilDiv(static_cast<uint32_t>(baseN), vectorLenB32));
        uint32_t sregR = static_cast<uint16_t>(baseN);
        uint32_t sregxGammaAlign = static_cast<uint16_t>(xGammaAlign);
        uint32_t sregyAlign = static_cast<uint16_t>(yAlign);
        __local_mem__ float* xAddr = (__ubuf__ float*)xLocal.GetPhyAddr();
        __local_mem__ float* rstdAddr = (__ubuf__ float*)rstdLocal.GetPhyAddr();
        __local_mem__ T_X* gammaAddr = (__ubuf__ T_X*)gammaLocal.GetPhyAddr();
        __local_mem__ T_SCALES* scales1Addr = (__ubuf__ T_SCALES*)scales1Local.GetPhyAddr();
        __local_mem__ T_SCALES* scales2Addr;
        __local_mem__ T_ZEROPOINTS* zeroPoints1Addr;
        __local_mem__ T_ZEROPOINTS* zeroPoints2Addr;
        __local_mem__ T_Y* y1Addr = (__ubuf__ T_Y*)y1Local.GetPhyAddr();
        __local_mem__ T_Y* y2Addr;

        if constexpr(HAS_SCALE2) {
            scales2Addr = (__ubuf__ T_SCALES*)scales2Local.GetPhyAddr();
        }
        if constexpr(HAS_ZEROPOINTS1) {
            zeroPoints1Addr = (__ubuf__ T_ZEROPOINTS*)zeroPoints1Local.GetPhyAddr();
        }
        if constexpr(HAS_ZEROPOINTS2) {
            zeroPoints2Addr = (__ubuf__ T_ZEROPOINTS*)zeroPoints2Local.GetPhyAddr();
        }       
        if constexpr((HAS_Y2)) {
            y2Addr = (__ubuf__ T_Y*)y2Local.GetPhyAddr();
        }
        // y = cast((x * rstd * gamma) */ scales + zeropints)
        __VEC_SCOPE__
        {
            RegTensor<float> xReg, rstdReg, gammaReg;
            RegTensor<float> scales1Reg, zeroPoints1Reg, scales2Reg, zeroPoints2Reg;
            RegTensor<float> mul1Reg, mul2Reg;
            RegTensor<float> scales1ResultReg, scales2ResultReg;
            MaskReg pregFull = CreateMask<float, MaskPattern::ALL>();
            MaskReg pregOne = CreateMask<float, MaskPattern::VL1>();
            // ld scales and zeropoints
            for (uint16_t i = 0; i < loopsA; i++) {
                // ld rstd
                uint32_t sregElewiseNum = baseN;
                DataCopy<float, LoadDist::DIST_BRC_B32>(rstdReg, rstdAddr + i);
                for (uint16_t j = 0; j < loopsR; j++) {
                    MaskReg pregCurLoop = UpdateMask<float>(sregElewiseNum);
                    LoadTensorForDtypeTIn(xAddr, xReg, pregCurLoop, (i * sregxGammaAlign + j * vectorLenB32));
                    Mul(mul1Reg, xReg, rstdReg, pregCurLoop);
                    LoadTensorForDtypeTIn(gammaAddr, gammaReg, pregCurLoop, j * vectorLenB32);
                    Mul(mul2Reg, gammaReg, mul1Reg, pregCurLoop);
                    LoadTensorForDtypeTIn(scales1Addr, scales1Reg, pregCurLoop, j * vectorLenB32);
                    if constexpr(DIV_MODE) {
                        Div(scales1ResultReg, mul2Reg, scales1Reg, pregCurLoop); 
                    } else {
                        Mul(scales1ResultReg, mul2Reg, scales1Reg, pregCurLoop); 
                    }

                    if constexpr(HAS_ZEROPOINTS1) {
                        LoadTensorForDtypeTIn(zeroPoints1Addr, zeroPoints1Reg, pregCurLoop, j * vectorLenB32);
                        Add(scales1ResultReg, scales1ResultReg, zeroPoints1Reg, pregCurLoop); 
                    }

                    StoreTensorForDtypeTOut(y1Addr, scales1ResultReg, pregCurLoop, (i * sregyAlign + j * vectorLenB32));

                    if constexpr(HAS_Y2) {
                        if constexpr(HAS_SCALE2) {
                            LoadTensorForDtypeTIn(scales2Addr, scales2Reg, pregCurLoop, j * vectorLenB32);
                            if constexpr(DIV_MODE) {
                                Div(scales2ResultReg, mul2Reg, scales2Reg, pregCurLoop); 
                            } else {
                                Mul(scales2ResultReg, mul2Reg, scales2Reg, pregCurLoop); 
                            }
                        }else {
                            Muls(scales2ResultReg, mul2Reg, float(1.0), pregCurLoop);
                        }
                        if constexpr(HAS_ZEROPOINTS2) {
                            LoadTensorForDtypeTIn(zeroPoints2Addr, zeroPoints2Reg, pregCurLoop, j * vectorLenB32);
                            Add(scales2ResultReg, scales2ResultReg, zeroPoints2Reg, pregCurLoop); 
                        }
                        StoreTensorForDtypeTOut(y2Addr, scales2ResultReg, pregCurLoop, (i * sregyAlign + j * vectorLenB32));
                    }
                }
            }
        }
        
    }


private:
    TPipe* pipe_ = nullptr;
    // GM Buffer
    GlobalTensor<T_X> x1Gm;
    GlobalTensor<T_X> x2Gm;
    GlobalTensor<T_X> gammaGm;
    GlobalTensor<T_X> xGm;
    GlobalTensor<T_SCALES> scales1Gm, scales2Gm;
    GlobalTensor<T_ZEROPOINTS> zeroPoints1Gm, zeroPoints2Gm;
    GlobalTensor<T_Y> y1Gm;
    GlobalTensor<T_Y> y2Gm;

    // UB Buffer
    TQue<QuePosition::VECIN, 1> inQueueX1;
    TQue<QuePosition::VECIN, 1> inQueueX2;
    // gamma scales0 scales1 zero_points0 zero_point1 all in this queue
    TQue<QuePosition::VECIN, 1> inQueueOther;
    TQue<QuePosition::VECOUT, 1> outQueueY1;
    TQue<QuePosition::VECOUT, 1> outQueueY2;
    TQue<QuePosition::VECOUT, 1> outQueueX;
    TBuf<TPosition::VECCALC> rstdBuf;
    TBuf<TPosition::VECCALC> reduceTmpBuf;
    TBuf<TPosition::VECCALC> xOutTmpBuf;

    LocalTensor<T_X> gammaLocal;
    LocalTensor<T_SCALES> scales1Local, scales2Local;
    LocalTensor<T_ZEROPOINTS> zeroPoints1Local, zeroPoints2Local;
    // Tiling data
    int64_t numN{0};
    int64_t numM{0};
    int64_t baseM{0};
    int64_t baseN{0};
    int64_t baseNDtypeAlign{0};
    int64_t powerSplit{0};
    int64_t mPerCore{0};
    int64_t mLastCore{0};
    float epsilon{0};
    float avgFactor{0};

    // Platform
    int64_t blockIdx{0};
    int64_t blockNum{0};
    uint32_t blockSize = platform::GetUbBlockSize();
    uint32_t vectorLen = platform::GetVRegSize();
    uint32_t blockSizeB32 = platform::GetUbBlockSize() / sizeof(float);
    uint32_t vectorLenB32 = platform::GetVRegSize() / sizeof(float);
    // dtypeSize
    uint32_t xDtypeSize{1};
    uint32_t scalesDtypeSize{1};
    uint32_t zeroPointsDtypeSize{1};
    uint32_t yDtypeSize{1}; 

    // align value
    int64_t xGammaAlign{32};
    int64_t scalesAlign{32};
    int64_t zeroPointsAlign{32};
    int64_t yAlign{32};
    int64_t rstdAlign{32};

    // calculate value
    int64_t mCore{0};
    int64_t mLoops{0};
    int64_t mTail{0};

    // const value
    static constexpr int32_t NUM_ONE = 1;
    static constexpr int32_t NUM_TWO = 2;
    static constexpr uint32_t DOUBLE_BUFFER_NUM = 2;
    static constexpr float RMS_POS_INF = 3.40282366920938E+38;
    static constexpr float RMS_ZERO = 0.0f;
};
} // namespace AddRmsNormQuant
#endif // _ADD_RMS_NORM_QUANT_REGBASE_H_

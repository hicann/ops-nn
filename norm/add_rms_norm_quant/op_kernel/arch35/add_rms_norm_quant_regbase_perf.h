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

#include "../inc/platform.h"
#include "add_rms_norm_quant_regbase_common.h"
#include "../../norm_common/reduce_common_regbase.h"

namespace AddRmsNormQuant {

template <typename T_X, typename T_Y, typename T_SCALES, typename T_ZEROPOINTS, uint64_t TILING_KEY>
class KernelAddRmsNormQuantRegbasePerf {
#define HAS_BETA (((TILING_KEY % 1000)/100) == 1)
#define HAS_RESOUT ((TILING_KEY / 1000) == 2)
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
        GM_ADDR zeroPoints2, GM_ADDR beta, GM_ADDR y1, GM_ADDR y2, GM_ADDR x, GM_ADDR res_out,
        const AddRmsNormQuantRegbaseTilingData* tilingData)
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
        divMode = tilingData->divMode != 0;

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
        oriOverflowMode = GetOverflowMode<T_Y>();

        CalBlockTail();
        InitBuffer(x1, x2, gamma, scales1, scales2, zeroPoints1, zeroPoints2, beta, y1, y2, x, res_out);
    }

    __aicore__ inline void CalBlockTail()
    {
        mCore = blockIdx == (blockNum - 1) ? mLastCore : mPerCore;
        mLoops = CeilDiv(mCore, baseM);
        mTail = mCore - (mLoops - 1) * baseM;
    }

    __aicore__ inline void InitBuffer(
        GM_ADDR x1, GM_ADDR x2, GM_ADDR gamma, GM_ADDR scales1, GM_ADDR scales2, GM_ADDR zeroPoints1,
        GM_ADDR zeroPoints2, GM_ADDR beta, GM_ADDR y1, GM_ADDR y2, GM_ADDR x, GM_ADDR res_out)
    {
        uint64_t gmOffset = blockIdx * mPerCore * numN;
        uint64_t gmLen = mCore * numN;
        x1Gm.SetGlobalBuffer((__gm__ T_X*)x1 + gmOffset, gmLen);
        x2Gm.SetGlobalBuffer((__gm__ T_X*)x2 + gmOffset, gmLen);
        y1Gm.SetGlobalBuffer((__gm__ T_Y*)y1 + gmOffset, gmLen);
        xGm.SetGlobalBuffer((__gm__ T_X*)x + gmOffset, gmLen);
        if constexpr (HAS_RESOUT) {
            resOutGm.SetGlobalBuffer((__gm__ T_X*)res_out + gmOffset, gmLen);
        }
        // gamma + scales1
        gammaGm.SetGlobalBuffer((__gm__ T_X*)gamma, numN);
        scales1Gm.SetGlobalBuffer((__gm__ T_SCALES*)scales1, numN);

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
        if constexpr (HAS_BETA) {
            betaGm.SetGlobalBuffer((__gm__ T_X*)beta, numN);
            preloadDataSize = preloadDataSize + xGammaAlign * sizeof(T_X);
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
        if constexpr (HAS_RESOUT) {
            pipe_->InitBuffer(outQueueResOut, DOUBLE_BUFFER_NUM, XQueueSize);
        }
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
            NormCommon::NormCommonRegbase::CalculateSquareReduceSum<float>(
                xOutTmpLocal, reduceTmpLocal, reduceTmpBuf, static_cast<uint16_t>(realM),
                static_cast<uint32_t>(baseNDtypeAlign), static_cast<uint32_t>(baseN),
                static_cast<uint32_t>(powerSplit), static_cast<uint32_t>(B32_BLOCK_NUM));
            NormCommon::ComputeRstdNewtonRaphson<true, true>(
                reduceTmpLocal, rstdLocal, realM, epsilon, avgFactor, V_LENGTH);

            LocalTensor<T_Y> y1Local = outQueueY1.AllocTensor<T_Y>();
            LocalTensor<T_Y> y2Local;
            if constexpr (HAS_Y2) {
                y2Local = outQueueY2.AllocTensor<T_Y>();
            }

            LocalTensor<T_X> resOutLocal;
            __local_mem__ T_X* resOutAddr = nullptr;
            if constexpr (HAS_RESOUT) {
                resOutLocal = outQueueResOut.AllocTensor<T_X>();
                resOutAddr = (__ubuf__ T_X*)resOutLocal.GetPhyAddr();
            }

            // 3. Quant
            SetOverflowMode<T_Y>(0);
            CalculateQuant(xOutTmpLocal,rstdLocal,gammaLocal,betaLocal,scales1Local,scales2Local,zeroPoints1Local,zeroPoints2Local,y1Local,y2Local,realM,baseN,xGammaAlign,scalesAlign,zeroPointsAlign,yAlign,resOutAddr);
            SetOverflowMode<T_Y>(oriOverflowMode);

            if constexpr (HAS_RESOUT) {
                outQueueResOut.EnQue<T_X>(resOutLocal);
            }
            outQueueY1.EnQue<T_Y>(y1Local);
            if constexpr (HAS_Y2) {
                outQueueY2.EnQue<T_Y>(y2Local);
            }
            CopyOutY(gmOffset, realM);
            if constexpr (HAS_RESOUT) {
                CopyOutResOut(gmOffset, realM);
            }
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

        if constexpr (HAS_SCALE2) {
            // LocalTensor<T_SCALES> scales2Local;
            scales2Local = otherLocal[localOffset].ReinterpretCast<T_SCALES>();
            localOffset = localOffset + scalesAlign * sizeof(T_SCALES);
            DataCopyPad(scales2Local, scales2Gm, copyInParamsScales, dataCopyPadExtParamsScales);
        }

        if constexpr (HAS_ZEROPOINTS1) {
            // LocalTensor<T_ZEROPOINTS> zeroPoints1Local;
            zeroPoints1Local = otherLocal[localOffset].ReinterpretCast<T_ZEROPOINTS>();
            localOffset = localOffset + zeroPointsAlign * sizeof(T_ZEROPOINTS);
            DataCopyPad(zeroPoints1Local, zeroPoints1Gm, copyInParamszeroPoints, dataCopyPadExtParamszeroPoints);
        }

        if constexpr (HAS_ZEROPOINTS2) {
            // LocalTensor<T_ZEROPOINTS> zeroPoints2Local;
            zeroPoints2Local = otherLocal[localOffset].ReinterpretCast<T_ZEROPOINTS>();
            localOffset = localOffset + zeroPointsAlign * sizeof(T_ZEROPOINTS);
            DataCopyPad(zeroPoints2Local, zeroPoints2Gm, copyInParamszeroPoints, dataCopyPadExtParamszeroPoints);
        }

        if constexpr (HAS_BETA) {
            // LocalTensor<T_X> betaLocal
            betaLocal = otherLocal[localOffset].ReinterpretCast<T_X>();
            localOffset = localOffset + xGammaAlign * sizeof(T_X);
            DataCopyPad(betaLocal, betaGm, copyInParamsGamma, dataCopyPadExtParamsGamma);
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
        RegTensor<int32_t> xRegInt32;
        Cast<int32_t, float, castTraitFp322Int32>(xRegInt32, xRegFp32, preg);
        Cast<float, int32_t, castTraitInt322Fp32>(xRegFp32, xRegInt32, preg);
        Cast<half, float, castTraitFp322Fp16>(xRegFp16, xRegFp32, preg);
        Cast<T_OUT, half, castTraitFp162Int8>(xOut, xRegFp16, preg);
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

    __aicore__ inline void CopyOutResOut(uint64_t offset, uint32_t realM)
    {
        LocalTensor<T_X> resOutLocal = outQueueResOut.DeQue<T_X>();
        DataCopyExtParams copyParams{
            static_cast<uint16_t>(realM),                // blockCount
            static_cast<uint32_t>(baseN * sizeof(T_X)), // blockLen
            static_cast<uint32_t>(0),                    // srcStride
            static_cast<uint32_t>(0),                    // dstStride
            0                                            // rsv
        };
        DataCopyPad(resOutGm[offset], resOutLocal, copyParams);
        outQueueResOut.FreeTensor(resOutLocal);
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

    // template <bool HAS_ZEROPOINTS2, bool HAS_ZEROPOINTS1, bool HAS_SCALE2, bool DIV_MODE>
    __aicore__ inline void CalculateQuant(LocalTensor<float> xLocal, LocalTensor<float> rstdLocal,
        LocalTensor<T_X> gammaLocal, LocalTensor<T_X> betaLocal,
        LocalTensor<T_SCALES> scales1Local, LocalTensor<T_SCALES> scales2Local,
        LocalTensor<T_ZEROPOINTS> zeroPoints1Local, LocalTensor<T_ZEROPOINTS> zeroPoints2Local,
        LocalTensor<T_Y> y1Local, LocalTensor<T_Y> y2Local,
        int64_t realM, int64_t baseN, int64_t xGammaAlign, int64_t scalesAlign,
        int64_t zeroPointsAlign, int64_t yAlign,
        __local_mem__ T_X* resOutAddr = nullptr)
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
        __local_mem__ T_X* betaAddr;
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
        if constexpr (HAS_BETA) {
            betaAddr = (__ubuf__ T_X*)betaLocal.GetPhyAddr();
        }
        if constexpr((HAS_Y2)) {
            y2Addr = (__ubuf__ T_Y*)y2Local.GetPhyAddr();
        }
        // y = cast((x * rstd * gamma) */ scales + zeropints)
        __VEC_SCOPE__
        {
            RegTensor<float> xReg, rstdReg, gammaReg, betaReg;
            RegTensor<float> scales1Reg, zeroPoints1Reg, scales2Reg, zeroPoints2Reg;
            RegTensor<float> mul1Reg, mul2Reg;
            RegTensor<float> scales1ResultReg, scales2ResultReg;
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
                    if constexpr (HAS_RESOUT) {
                        StoreTensorForDtypeTOut<T_X>(resOutAddr, mul2Reg, pregCurLoop, (i * sregxGammaAlign + j * vectorLenB32));
                    }
                    if constexpr (HAS_BETA) {
                        LoadTensorForDtypeTIn(betaAddr, betaReg, pregCurLoop, j * vectorLenB32);
                        Add(mul2Reg, mul2Reg, betaReg, pregCurLoop);
                    }
                    LoadTensorForDtypeTIn(scales1Addr, scales1Reg, pregCurLoop, j * vectorLenB32);
                    if (divMode) {
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
                            if (divMode) {
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
    GlobalTensor<T_X> betaGm;
    GlobalTensor<T_X> xGm;
    GlobalTensor<T_SCALES> scales1Gm, scales2Gm;
    GlobalTensor<T_ZEROPOINTS> zeroPoints1Gm, zeroPoints2Gm;
    GlobalTensor<T_Y> y1Gm;
    GlobalTensor<T_Y> y2Gm;
    GlobalTensor<T_X> resOutGm;

    // UB Buffer
    TQue<QuePosition::VECIN, 1> inQueueX1;
    TQue<QuePosition::VECIN, 1> inQueueX2;
    // gamma scales0 scales1 zero_points0 zero_point1 all in this queue
    TQue<QuePosition::VECIN, 1> inQueueOther;
    TQue<QuePosition::VECOUT, 1> outQueueY1;
    TQue<QuePosition::VECOUT, 1> outQueueY2;
    TQue<QuePosition::VECOUT, 1> outQueueX;
    TQue<QuePosition::VECOUT, 1> outQueueResOut;
    TBuf<TPosition::VECCALC> rstdBuf;
    TBuf<TPosition::VECCALC> reduceTmpBuf;
    TBuf<TPosition::VECCALC> xOutTmpBuf;

    LocalTensor<T_X> gammaLocal;
    LocalTensor<T_X> betaLocal;
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
    bool divMode{false};

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
    int64_t oriOverflowMode{0};

    // const value
    static constexpr int32_t NUM_ONE = 1;
    static constexpr int32_t NUM_TWO = 2;
    static constexpr uint32_t DOUBLE_BUFFER_NUM = 2;
    static constexpr float RMS_POS_INF = 3.40282366920938E+38;
    static constexpr float RMS_ZERO = 0.0f;
};
} // namespace AddRmsNormQuant
#endif // _ADD_RMS_NORM_QUANT_REGBASE_H_

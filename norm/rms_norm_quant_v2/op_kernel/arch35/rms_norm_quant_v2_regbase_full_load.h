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
 * \file rms_norm_quant_v2_regbase_full_load.h
 * \brief
 */
#ifndef RMS_NORM_QUANT_V2_REBASE_FULL_LOAD_H_
#define RMS_NORM_QUANT_V2_REBASE_FULL_LOAD_H_
#include "kernel_utils.h"
#include "../inc/platform.h"
#include "rms_norm_quant_v2_regbase_common.h"
#include "../../norm_common/reduce_common_regbase.h"

namespace RmsNormQuantV2 {

template <typename T_X, typename T_Y, typename T_SCALES, typename T_ZEROPOINTS>
class RmsNormQuantV2RegbaseFullLoad {
private:
    TPipe* pipe_ = nullptr;
    using yDtype = std::conditional_t<IsSameType<T_Y, int4b_t>::value, uint8_t, T_Y>;
    // GM Buffer
    GlobalTensor<T_X> xGm, gammaGm, betaGm;
    GlobalTensor<T_SCALES> scales1Gm, scales2Gm;
    GlobalTensor<T_ZEROPOINTS> zeroPoints1Gm, zeroPoints2Gm;
    GlobalTensor<yDtype> y1Gm, y2Gm;
    // UB Buffer
    TQue<QuePosition::VECIN, 1> inQueueX;
    // gamma beta scales0 scales1 zero_points0 zero_point1 all in this queue
    TQue<QuePosition::VECIN, 1> inQueueOhter;
    TQue<QuePosition::VECOUT, 1> outQueueY1, outQueueY2;
    TBuf<TPosition::VECCALC> rstdBuf;
    TBuf<TPosition::VECCALC> reduceTmpBuf;

    LocalTensor<T_X> gammaLocal;
    LocalTensor<T_SCALES> scales1Local, scales2Local;
    LocalTensor<T_ZEROPOINTS> zeroPoints1Local, zeroPoints2Local;
    LocalTensor<T_X> betaLocal;

    // Tiling data
    int64_t numA{0};
    int64_t numR{0};
    int64_t numQ{0};
    int64_t blockFactor{0};
    int64_t blockTail{0};
    int64_t ubFactor{0};
    int64_t binaryAdd{0};
    uint32_t optionMask{0};
    bool isScaleDiv{0};
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
    int64_t xGammaBetaAlign{32};
    int64_t scalesAlign{32};
    int64_t zeroPointsAlign{32};
    int64_t yAlign{32};
    int64_t rstdAlign{32};

    // calculate value
    int64_t curBlockFactor{0};
    int64_t curUbLoops{0};
    int64_t ubFactorTail{0};

    // option value
    bool hasZeroPoints1{false};
    bool hasScales2{false};
    bool hasZeroPoints2{false};
    bool hasBeta{false};
    bool hasY2{false};

    // option mask const value
    static constexpr uint32_t SCALES2_MASK = 0b0001;
    static constexpr uint32_t ZEROS_POINTS1_MASK = 0b0010;
    static constexpr uint32_t ZEROS_POINTS2_MASK = 0b0100;
    static constexpr uint32_t BETA_MASK = 0b1000;
    static constexpr uint32_t DOUBLE_BUFFER_NUM = 2;

    static constexpr float RMS_POS_INF = 3.40282366920938E+38;
    static constexpr float RMS_ZERO = 0.0f;

public:
    __aicore__ inline RmsNormQuantV2RegbaseFullLoad(TPipe* pipe)
    {
        pipe_ = pipe;
    }

    __aicore__ inline void Init(
        GM_ADDR x, GM_ADDR gamma, GM_ADDR scales1, GM_ADDR scales2, GM_ADDR zeroPoints1,
        GM_ADDR zeroPoints2, GM_ADDR beta, GM_ADDR y1, GM_ADDR y2, const RmsNormQuantV2RegbaseFullLoadTilingData* tilingData)
    {

        // Tiling data
        numA = tilingData->a;
        numR = tilingData->r;
        numQ = tilingData->q;
        blockFactor = tilingData->blockFactor;
        blockTail = tilingData->blockTail;
        ubFactor = tilingData->ubFactor;
        // ubFactor = 1;
        binaryAdd = tilingData->binaryAdd;
        optionMask = tilingData->optionMask & 0xF;
        isScaleDiv = tilingData->divMode == 1;
        epsilon = tilingData->epsilon;
        avgFactor = tilingData->avgFactor;

        // dtype size
        xDtypeSize = blockSize / sizeof(T_X);
        scalesDtypeSize = blockSize / sizeof(T_SCALES);
        zeroPointsDtypeSize = blockSize / sizeof(T_ZEROPOINTS);
        yDtypeSize = blockSize / sizeof(yDtype);
        if constexpr(IsSameType<T_Y, int4b_t>::value) {
            yDtypeSize = yDtypeSize * 2;
        }

        // dtype align
        xGammaBetaAlign = CeilDiv(numR, static_cast<int64_t>(xDtypeSize)) * static_cast<int64_t>(xDtypeSize);
        scalesAlign = CeilDiv(numR, static_cast<int64_t>(scalesDtypeSize)) * static_cast<int64_t>(scalesDtypeSize);
        zeroPointsAlign = CeilDiv(numR, static_cast<int64_t>(zeroPointsDtypeSize)) * static_cast<int64_t>(zeroPointsDtypeSize);
        yAlign = CeilDiv(numR, static_cast<int64_t>(yDtypeSize)) * static_cast<int64_t>(yDtypeSize);
        rstdAlign = CeilDiv(ubFactor, static_cast<int64_t>(blockSizeB32)) * static_cast<int64_t>(blockSizeB32);

        blockNum = GetBlockNum();
        blockIdx = GetBlockIdx();

        // init option
        if ((optionMask&ZEROS_POINTS1_MASK) == ZEROS_POINTS1_MASK) {
            hasZeroPoints1 = true;
        }
        if ((optionMask&SCALES2_MASK) == SCALES2_MASK) {
            hasScales2 = true;
        }
        if ((optionMask&ZEROS_POINTS2_MASK) == ZEROS_POINTS2_MASK) {
            hasZeroPoints2 = true;
        }
        if ((optionMask&BETA_MASK) == BETA_MASK) {
            hasBeta = true;
        }
        hasY2 = hasScales2;

        // init curBlockFactor
        curBlockFactor = blockIdx == (blockNum - 1) ? blockTail : blockFactor;
        curUbLoops = CeilDiv(curBlockFactor, ubFactor);
        ubFactorTail = curBlockFactor - (curUbLoops - 1) * ubFactor;

        InitBuffer(x, gamma, scales1, scales2, zeroPoints1, zeroPoints2, beta, y1, y2);
    }

    __aicore__ inline void InitBuffer(
        GM_ADDR x, GM_ADDR gamma, GM_ADDR scales1, GM_ADDR scales2, GM_ADDR zeroPoints1,
        GM_ADDR zeroPoints2, GM_ADDR beta, GM_ADDR y1, GM_ADDR y2)
    {
        // GM BUFFER
        int64_t xOffset = blockIdx * blockFactor * numR;
        int64_t xLen = curBlockFactor * numR;
        int64_t yOffset = blockIdx * blockFactor * numR;
        int64_t yLen = curBlockFactor * numR;
        if constexpr(IsSameType<T_Y, int4b_t>::value) {
            yOffset = yOffset / 2;
            yLen = yLen / 2;
        }
        xGm.SetGlobalBuffer((__gm__ T_X*)x + xOffset, xLen);
        y1Gm.SetGlobalBuffer((__gm__ yDtype*)y1 + yOffset, yLen);

        gammaGm.SetGlobalBuffer((__gm__ T_X*)gamma, numR);
        scales1Gm.SetGlobalBuffer((__gm__ T_SCALES*)scales1, numR);

        // gamma + scales1
        int64_t preloadDataSize = xGammaBetaAlign * sizeof(T_X) + scalesAlign * sizeof(T_SCALES);
        
        if (hasScales2) {
            scales2Gm.SetGlobalBuffer((__gm__ T_SCALES*)scales2, numR);
            preloadDataSize = preloadDataSize + scalesAlign * sizeof(T_SCALES);
        }
        if (hasZeroPoints1) {
            zeroPoints1Gm.SetGlobalBuffer((__gm__ T_ZEROPOINTS*)zeroPoints1, numR);
            preloadDataSize = preloadDataSize + zeroPointsAlign * sizeof(T_ZEROPOINTS);
        }
        if (hasZeroPoints2) {
            zeroPoints2Gm.SetGlobalBuffer((__gm__ T_ZEROPOINTS*)zeroPoints2, numR);
            preloadDataSize = preloadDataSize + zeroPointsAlign * sizeof(T_ZEROPOINTS);
        }
        if (hasBeta) {
            betaGm.SetGlobalBuffer((__gm__ T_X*)beta, numR);
            preloadDataSize = preloadDataSize + xGammaBetaAlign * sizeof(T_X);
        }
        if (hasY2) {
            y2Gm.SetGlobalBuffer((__gm__ yDtype*)y2 + yOffset, yLen);
        }

        // UB buffer
        // TQue<QuePosition::VECIN, 1> inQueueX;
        // /gamma beta scales0 scales1 zero_points0 zero_point1 all in this queue
        // TQue<QuePosition::VECIN, 1> inQueueOhter;
        // TQue<QuePosition::VECOUT, 1> outQueueY1, outQueueY2;
        // TBuf<TPosition::VECCALC> rstdBuf;
        // TBuf<TPosition::VECCALC> reduceBuf;
        pipe_->InitBuffer(inQueueX, DOUBLE_BUFFER_NUM, ubFactor * xGammaBetaAlign * sizeof(T_X));
        // preload data
        pipe_->InitBuffer(inQueueOhter, 1, preloadDataSize);
        int64_t yQueueSize = ubFactor * yAlign * sizeof(yDtype);
        if constexpr(IsSameType<T_Y, int4b_t>::value) {
            yQueueSize = yQueueSize / 2;
        }
        pipe_->InitBuffer(outQueueY1, DOUBLE_BUFFER_NUM, yQueueSize);
        if (hasScales2) {
            pipe_->InitBuffer(outQueueY2, DOUBLE_BUFFER_NUM, yQueueSize);
        }
        
        pipe_->InitBuffer(rstdBuf, rstdAlign * sizeof(float));
        // reduceTmpBuffer
        int64_t reduceTmpBufferSize = ubFactor * CeilDiv(CeilDiv(binaryAdd, static_cast<int64_t>(vectorLenB32)), static_cast<int64_t>(blockSizeB32)) * blockSizeB32;
        pipe_->InitBuffer(reduceTmpBuf, reduceTmpBufferSize);
    }

    __aicore__ inline void Process()
    {
        // copy other input (gamma scales zeropints beta)
        LocalTensor<uint8_t> otherLocal = inQueueOhter.AllocTensor<uint8_t>();
        CopyInOhters(otherLocal);
        inQueueOhter.EnQue(otherLocal);
        inQueueOhter.DeQue<uint8_t>();

        for (int64_t i = 0; i < curUbLoops; i++) {
            int64_t curUbFactor = (i == (curUbLoops - 1)) ? ubFactorTail : ubFactor; //ubFactorTail 尾部
            // int64_t curUbFactor = (i == (curUbLoops - 1)) ? (curBlockFactor-(curUbLoops-1)*ubFactor) : ubFactor;
            int64_t offsetBase = i * numR * ubFactor;
            // x
            DataCopyPadExtParams<T_X> dataCopyPadExtParamsX;
            dataCopyPadExtParamsX.isPad = false;
            dataCopyPadExtParamsX.leftPadding = 0;
            dataCopyPadExtParamsX.rightPadding = 0;
            dataCopyPadExtParamsX.paddingValue = 0;
            DataCopyExtParams copyInParamsX;
            copyInParamsX.blockCount = curUbFactor;
            // copyInParamsX.blockLen = xGammaBetaAlign * sizeof(T_X);
            copyInParamsX.blockLen = numR * sizeof(T_X);
            copyInParamsX.srcStride = 0;
            copyInParamsX.dstStride = 0;
            LocalTensor<T_X> xLocal = inQueueX.AllocTensor<T_X>();
            DataCopyPad(xLocal, xGm[offsetBase], copyInParamsX, dataCopyPadExtParamsX);
            inQueueX.EnQue(xLocal);
            inQueueX.DeQue<T_X>();
            // compute square reduceSum
            LocalTensor<float> rstdLocal = rstdBuf.Get<float>();
            LocalTensor<float> reduceTmpLocal = reduceTmpBuf.Get<float>();
            ComputeSquareReduceSum(xLocal, reduceTmpLocal, rstdLocal, curUbFactor, numR, xGammaBetaAlign, binaryAdd);
            // compute rstd
            ComputeRstd(rstdLocal, rstdLocal, curUbFactor, epsilon, avgFactor);
            
            
            LocalTensor<yDtype> y1Local = outQueueY1.AllocTensor<yDtype>();
            LocalTensor<yDtype> y2Local;
            if(hasY2){
                y2Local = outQueueY2.AllocTensor<yDtype>();
            }

            // compute quant
            QuantRoute(optionMask,xLocal,rstdLocal,gammaLocal,betaLocal,scales1Local,scales2Local,zeroPoints1Local,zeroPoints2Local,y1Local,y2Local,curUbFactor,numR,numQ,xGammaBetaAlign,scalesAlign,zeroPointsAlign,yAlign);
            inQueueX.FreeTensor(xLocal);

            outQueueY1.EnQue(y1Local);
            outQueueY1.DeQue<yDtype>();

            int64_t yOffsetBase = offsetBase;
            int64_t yBlockLen = numR * sizeof(yDtype);
            if constexpr(IsSameType<T_Y, int4b_t>::value) {
                yOffsetBase = yOffsetBase / 2;
                yBlockLen = yBlockLen / 2;
            }
            DataCopyExtParams copyOutParamsY1;
            copyOutParamsY1.blockCount = curUbFactor;
            copyOutParamsY1.blockLen =  yBlockLen;
            copyOutParamsY1.srcStride = 0;
            copyOutParamsY1.dstStride = 0;
            DataCopyPad(y1Gm[yOffsetBase], y1Local, copyOutParamsY1);
            outQueueY1.FreeTensor(y1Local);

            if(hasY2){
                outQueueY2.EnQue(y2Local);
                outQueueY2.DeQue<yDtype>();
                DataCopyExtParams copyOutParamsY2;
                copyOutParamsY2.blockCount = curUbFactor;
                copyOutParamsY2.blockLen =  yBlockLen;
                copyOutParamsY2.srcStride = 0;
                copyOutParamsY2.dstStride = 0;
                DataCopyPad(y2Gm[yOffsetBase], y2Local, copyOutParamsY2);
                outQueueY2.FreeTensor(y2Local);
            }   
        }
        inQueueOhter.FreeTensor(otherLocal);
    }

private:

    __aicore__ inline void CopyInOhters(LocalTensor<uint8_t> otherLocal)
    {
        uint32_t localOffset = 0;
        // LocalTensor<T_X> gammaLocal
        gammaLocal = otherLocal[localOffset].ReinterpretCast<T_X>();
        localOffset = localOffset + xGammaBetaAlign * sizeof(T_X);
        DataCopyPadExtParams<T_X> dataCopyPadExtParamsGamma;
        dataCopyPadExtParamsGamma.isPad = false;
        dataCopyPadExtParamsGamma.leftPadding = 0;
        dataCopyPadExtParamsGamma.rightPadding = 0;
        dataCopyPadExtParamsGamma.paddingValue = 0;
        DataCopyExtParams copyInParamsGamma;
        copyInParamsGamma.blockCount = 1;
        copyInParamsGamma.blockLen = xGammaBetaAlign * sizeof(T_X);
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

        // zeroPoints 代码
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
                
        if (hasScales2) {
            // LocalTensor<T_SCALES> scales2Local;
            scales2Local = otherLocal[localOffset].ReinterpretCast<T_SCALES>();
            localOffset = localOffset + scalesAlign * sizeof(T_SCALES);
            DataCopyPad(scales2Local, scales2Gm, copyInParamsScales, dataCopyPadExtParamsScales);
        }

        if (hasZeroPoints1) {
            // LocalTensor<T_ZEROPOINTS> zeroPoints1Local;
            zeroPoints1Local = otherLocal[localOffset].ReinterpretCast<T_ZEROPOINTS>();
            localOffset = localOffset + zeroPointsAlign * sizeof(T_ZEROPOINTS);
            DataCopyPad(zeroPoints1Local, zeroPoints1Gm, copyInParamszeroPoints, dataCopyPadExtParamszeroPoints);
        }

        if (hasZeroPoints2) {
            // LocalTensor<T_ZEROPOINTS> zeroPoints2Local;
            zeroPoints2Local = otherLocal[localOffset].ReinterpretCast<T_ZEROPOINTS>();
            localOffset = localOffset + zeroPointsAlign * sizeof(T_ZEROPOINTS);
            DataCopyPad(zeroPoints2Local, zeroPoints2Gm, copyInParamszeroPoints, dataCopyPadExtParamszeroPoints);
        }

        if (hasBeta) {
            // LocalTensor<T_X> betaLocal;
            betaLocal = otherLocal[localOffset].ReinterpretCast<T_X>();
            DataCopyPad(betaLocal, betaGm, copyInParamsGamma, dataCopyPadExtParamsGamma);
        }
    }

    __aicore__ inline void ComputeSquareReduceSum(LocalTensor<T_X> xLocal, LocalTensor<float> xTmpLocal, 
        LocalTensor<float> xReduceTmpLocal, int64_t curUbFactor, int64_t numCol, int64_t numColAlign, int64_t colFlodFactor)
    {
        __local_mem__ T_X* xLocalAddr = (__local_mem__ T_X*)xLocal.GetPhyAddr();
        __local_mem__ float* xTmpLocalUbAddr = (__local_mem__ float*)xTmpLocal.GetPhyAddr();
        __local_mem__ float* xReduceTmpLocalUbAddr = (__local_mem__ float*)xReduceTmpLocal.GetPhyAddr();

        if (numColAlign <= vectorLenB32) {
            CalculateSquareReduceSumRLessThanVL(xLocalAddr, xReduceTmpLocalUbAddr, curUbFactor, numCol, numColAlign);
        } else if (numColAlign <= (vectorLenB32 + vectorLenB32)) {
            CalculateSquareReduceSumRLessThanTwoVL(xLocalAddr, xReduceTmpLocalUbAddr, curUbFactor, numCol, numColAlign);
        } else if (numColAlign <= vectorLenB32 * vectorLenB32 * NUM_TWO){
            CalculateSquareReduceSumRCommon<NUM_ONE>(xLocalAddr, xTmpLocalUbAddr, xReduceTmpLocalUbAddr, curUbFactor, numCol, numColAlign, colFlodFactor);
        } else {
            CalculateSquareReduceSumRCommon<NUM_TWO>(xLocalAddr, xTmpLocalUbAddr, xReduceTmpLocalUbAddr, curUbFactor, numCol, numColAlign, colFlodFactor);
        }
    }

    __aicore__ inline void CalculateSquareReduceSumRLessThanVL(__local_mem__ T_X* xLocalAddr,__local_mem__ float* xReduceTmpLocalUbAddr,
            int64_t curUbFactor, int64_t numCol, int64_t numColAlign)
    {
        uint32_t colNum = static_cast<uint32_t>(numCol);
        uint16_t curAloops = static_cast<uint16_t>(curUbFactor);
        uint32_t colNumAlign = static_cast<uint32_t>(numColAlign);
        __VEC_SCOPE__
        {
            RegTensor<float> xReg;
            RegTensor<float> squareReg;
            RegTensor<float> sumReg;

            // rstd cal
            MaskReg pregOne = CreateMask<float, MaskPattern::VL1>();
            uint32_t colCountSreg = colNum;
            MaskReg pregLoop = UpdateMask<float>(colCountSreg);
            for (uint16_t i = 0; i < curAloops; i++) {
                LoadTensorForDtypeTIn(xLocalAddr, xReg, pregLoop, (i * colNumAlign));
                Mul(squareReg, xReg, xReg, pregLoop);
                ReduceSum(sumReg, squareReg, pregLoop);
                DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(xReduceTmpLocalUbAddr + i, sumReg, pregOne);
            }
        }
    }

    __aicore__ inline void CalculateSquareReduceSumRLessThanTwoVL(__local_mem__ T_X* xLocalAddr, 
        __local_mem__ float* xReduceTmpLocalUbAddr,
        int64_t curUbFactor, int64_t numCol, int64_t numColAlign)
    {
        uint32_t colNum = static_cast<uint32_t>(numCol);
        uint32_t colNumAlign = static_cast<uint32_t>(numColAlign);
        uint16_t curAloops = static_cast<uint16_t>(curUbFactor);
        
        __VEC_SCOPE__
        {
            RegTensor<float> xReg;
            RegTensor<float> xTailReg;
            RegTensor<float> squareReg;
            RegTensor<float> squareTailReg;
            RegTensor<float> addReg;
            RegTensor<float> sumReg;

            MaskReg pregFull = CreateMask<float, MaskPattern::ALL>();
            MaskReg pregOne = CreateMask<float, MaskPattern::VL1>();
            // 64 CHANGE dtype need check
            uint32_t colTailSreg = colNum - vectorLenB32;
            MaskReg pregTail = UpdateMask<float>(colTailSreg);
            for (uint16_t i = 0; i < curAloops; i++) {
                LoadTensorForDtypeTIn(xLocalAddr, xReg, pregFull, (i * colNumAlign));
                Mul(squareReg, xReg, xReg, pregFull);
                LoadTensorForDtypeTIn(xLocalAddr + vectorLenB32, xTailReg, pregTail, (i * colNumAlign));
                Mul(squareTailReg, xTailReg, xTailReg, pregTail);
                Add(addReg, squareReg, squareTailReg, pregFull);
                ReduceSum(sumReg, addReg, pregFull);
                DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(xReduceTmpLocalUbAddr + i, sumReg, pregOne);
            }
        }
    }

    template <int32_t LAST_LOOP_NUMS>
    __aicore__ inline void CalculateSquareReduceSumRCommon(__local_mem__ T_X* xLocalAddr,
        __local_mem__ float* xTmpLocalUbAddr, __local_mem__ float* xReduceTmpLocalUbAddr, int64_t curUbFactor, 
        int64_t numCol, int64_t numColAlign, int64_t colFlodFactor)
    {
        uint32_t colNum = static_cast<uint32_t>(numCol);
        uint32_t colNumAlign = static_cast<uint32_t>(numColAlign);
        uint32_t colFlodNum = static_cast<uint32_t>(colFlodFactor);
        uint16_t curAloops = static_cast<uint16_t>(curUbFactor);

        // first flod
        uint32_t firstFlodTial = static_cast<uint32_t>(colNum - colFlodFactor);
        uint16_t firstFlodAddLoops = static_cast<uint16_t>((firstFlodTial + vectorLenB32-1) / vectorLenB32);
        uint16_t firstFlodWithOutAddLoops = static_cast<uint16_t>((colFlodFactor + vectorLenB32-1) / vectorLenB32) - firstFlodAddLoops;

        // first vcadd
        uint32_t firstVcaddNum = static_cast<uint32_t>((colFlodFactor + vectorLenB32-1 )/ vectorLenB32);
        uint32_t firstVcaddNumCeilAlign = static_cast<uint32_t>((firstVcaddNum + blockSizeB32-1) / blockSizeB32 * blockSizeB32);

        // second flod
        // rstd cal
        uint16_t elewiseLoop = static_cast<uint16_t>((curUbFactor + vectorLenB32-1) / vectorLenB32);

        __VEC_SCOPE__
        {
            RegTensor<float> xReg1;
            RegTensor<float> xReg2;
            RegTensor<float> squareReg1;
            RegTensor<float> squareReg2;
            RegTensor<float> addReg;
            RegTensor<float> sumReg;

            RegTensor<float> xReg3;
            RegTensor<float> squareReg3;
            RegTensor<float> sumReg3;

            RegTensor<float> mulsReg;
            RegTensor<float> addsReg;
            RegTensor<float> sqrtReg;

            MaskReg pregFull = CreateMask<float, MaskPattern::ALL>();
            MaskReg pregOne = CreateMask<float, MaskPattern::VL1>();
            MaskReg pregLoop;

            for (uint16_t i = 0; i < curAloops; i++) {
                uint32_t sregfirstFlodTial = firstFlodTial;
                for (uint16_t j = 0; j < firstFlodAddLoops; j++) {
                    pregLoop = UpdateMask<float>(sregfirstFlodTial);
                    LoadTensorForDtypeTIn(xLocalAddr, xReg1, pregFull, (i * colNumAlign + j * vectorLenB32));
                    Mul(squareReg1, xReg1, xReg1, pregFull);
                    LoadTensorForDtypeTIn(xLocalAddr + colFlodNum, xReg2, pregFull, (i * colNumAlign + j * vectorLenB32));
                    Mul(squareReg2, xReg2, xReg2, pregLoop);
                    Add(addReg, squareReg1, squareReg2, pregFull);
                    ReduceSum(sumReg, addReg, pregFull);
                    DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(
                        xTmpLocalUbAddr + static_cast<uint32_t>(i * firstVcaddNumCeilAlign + j), sumReg, pregOne);
                }
                for (uint16_t j = 0; j < static_cast<uint16_t>(firstFlodWithOutAddLoops); j++) {
                    LoadTensorForDtypeTIn(xLocalAddr + firstFlodAddLoops * vectorLenB32,  xReg3, pregFull, 
                        (i * colNumAlign + j * vectorLenB32));
                    Mul(squareReg3, xReg3, xReg3, pregFull);
                    ReduceSum(sumReg3, squareReg3, pregFull);
                    DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(
                        xTmpLocalUbAddr + static_cast<uint32_t>(i * firstVcaddNumCeilAlign + firstFlodAddLoops + j),sumReg3, pregOne);
                }
            }

            // if need a add to last repeat
            LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
            if constexpr (LAST_LOOP_NUMS == 1) {
                uint32_t sregSecondReduce = firstVcaddNum;
                MaskReg pregLast = UpdateMask<float>(sregSecondReduce);
                for (uint16_t i = 0; i < curAloops; i++) {
                    DataCopy(xReg1, xTmpLocalUbAddr + static_cast<uint32_t>(i * firstVcaddNumCeilAlign));
                    ReduceSum(sumReg, xReg1, pregLast);
                    DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(xReduceTmpLocalUbAddr + i, sumReg, pregOne);
                }
            } else if constexpr (LAST_LOOP_NUMS == 2) {
                uint32_t sregSecondReduce = firstVcaddNum - vectorLenB32;
                MaskReg pregLast = UpdateMask<float>(sregSecondReduce);
                RegTensor<float> shiftLeft;
                for (uint16_t i = 0; i < curAloops; i++) {
                    DataCopy(xReg1, xTmpLocalUbAddr + static_cast<uint32_t>(i * firstVcaddNumCeilAlign));
                    DataCopy(xReg2, xTmpLocalUbAddr + static_cast<uint32_t>(i * firstVcaddNumCeilAlign + vectorLenB32));
                    ShiftLefts((RegTensor<uint32_t> &)shiftLeft, (RegTensor<uint32_t> &)xReg2, static_cast<int16_t>(0), pregLast);
                    Add(addReg, xReg1, shiftLeft, pregFull);
                    ReduceSum(sumReg, addReg, pregFull);
                    DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(xReduceTmpLocalUbAddr + i, sumReg, pregOne);
                }
            }
        }
    }

    __aicore__ inline void ComputeRstd(LocalTensor<float> xReduceTmpLocal, LocalTensor<float> rstdLocal, int64_t curUbFactor, float epsilon, float avgFactor)
    {
        __local_mem__ float* rstdLocalUbAddr = (__local_mem__ float*)rstdLocal.GetPhyAddr();
        __local_mem__ float* xReduceTmpLocalUbAddr = (__local_mem__ float*)xReduceTmpLocal.GetPhyAddr();
        uint16_t aLoop = static_cast<uint16_t>((curUbFactor + vectorLenB32 - 1) / vectorLenB32);
        __VEC_SCOPE__
        {
            MaskReg pregMain = CreateMask<float, MaskPattern::ALL>();
            RegTensor<float> var;
            RegTensor<float> one;
            RegTensor<float> r;
            RegTensor<float> y;
            RegTensor<float> s;
            RegTensor<float> t;
            RegTensor<float> scalar1;
            RegTensor<float> scalarInf;
            RegTensor<float> scalarZero;
            RegTensor<float> t1;
            RegTensor<float> t2;
            RegTensor<float> t3;
            RegTensor<float> t4;
            RegTensor<float> rstd;

            MaskReg cmpRegZero;
            MaskReg cmpRegInf;
            MaskReg pregLoop;

            Duplicate(one, 1.0, pregMain);
            uint32_t sreg0 = static_cast<uint32_t>(curUbFactor);
            for (uint16_t a = 0; a < aLoop; a++) {
                pregLoop = UpdateMask<float>(sreg0);
                Duplicate(scalar1, float(0.5), pregLoop);
                Duplicate(scalarInf, RMS_POS_INF, pregLoop);
                Duplicate(scalarZero, RMS_ZERO, pregLoop);
                Duplicate(t1, float(1.5), pregLoop);
                Duplicate(s, float(1.0), pregLoop);

                // rstd
                DataCopy(var, xReduceTmpLocalUbAddr + a * vectorLenB32);
                Muls(var, var, avgFactor, pregLoop);
                Adds(var, var, epsilon, pregLoop);
                Div(r, one, var, pregLoop);
                Sqrt(y, r, pregLoop);
                Muls(t, var, float(-0.5), pregLoop);
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
                CompareScalar(cmpRegZero, var, RMS_POS_INF, pregLoop);
                Select(rstd, scalarZero, rstd, cmpRegZero);
                CompareScalar(cmpRegInf, var, RMS_ZERO, pregLoop);
                Select(rstd, scalarInf, rstd, cmpRegInf);
                DataCopy(rstdLocalUbAddr + a * vectorLenB32, rstd, pregLoop);
            }
        }
    }

     __aicore__ inline void QuantRoute(uint32_t optionMask,LocalTensor<T_X> xLocal, LocalTensor<float> rstdLocal, 
        LocalTensor<T_X> gammaLocal, LocalTensor<T_X> betaLocal, 
        LocalTensor<T_SCALES> scales1Local, LocalTensor<T_SCALES> scales2Local, 
        LocalTensor<T_ZEROPOINTS> zeroPoints1Local, LocalTensor<T_ZEROPOINTS> zeroPoints2Local, 
        LocalTensor<yDtype> y1Local, LocalTensor<yDtype> y2Local, 
        int64_t curUbFactor, int64_t numR, int64_t numQ, int64_t xGammaBetaAlign, int64_t scalesAlign, 
        int64_t zeroPointsAlign, int64_t yAlign) 
    {    
            // compute quant
            if (!isScaleDiv) {
                if (optionMask == 0b1111) {
                    ComputeQuant<true, true, true, true, false>(xLocal,rstdLocal,gammaLocal,betaLocal,scales1Local,scales2Local,zeroPoints1Local,zeroPoints2Local,
                    y1Local,y2Local,curUbFactor,numR,numQ,xGammaBetaAlign,scalesAlign,zeroPointsAlign,yAlign);
                } else if (optionMask == 0b0111) {
                    ComputeQuant<false, true, true, true, false>(xLocal,rstdLocal,gammaLocal,betaLocal,scales1Local,scales2Local,zeroPoints1Local,zeroPoints2Local,
                    y1Local,y2Local,curUbFactor,numR,numQ,xGammaBetaAlign,scalesAlign,zeroPointsAlign,yAlign);
                } else if (optionMask == 0b1011) {
                    ComputeQuant<true, false, true, true, false>(xLocal,rstdLocal,gammaLocal,betaLocal,scales1Local,scales2Local,zeroPoints1Local,zeroPoints2Local,
                    y1Local,y2Local,curUbFactor,numR,numQ,xGammaBetaAlign,scalesAlign,zeroPointsAlign,yAlign);
                } else if (optionMask == 0b0011) {
                    ComputeQuant<false, false, true, true, false>(xLocal,rstdLocal,gammaLocal,betaLocal,scales1Local,scales2Local,zeroPoints1Local,zeroPoints2Local,
                    y1Local,y2Local,curUbFactor,numR,numQ,xGammaBetaAlign,scalesAlign,zeroPointsAlign,yAlign);
                } else if (optionMask == 0b1101) {
                    ComputeQuant<true, true, false, true, false>(xLocal,rstdLocal,gammaLocal,betaLocal,scales1Local,scales2Local,zeroPoints1Local,zeroPoints2Local,
                    y1Local,y2Local,curUbFactor,numR,numQ,xGammaBetaAlign,scalesAlign,zeroPointsAlign,yAlign);
                } else if (optionMask == 0b0101) {
                    ComputeQuant<false, true, false, true, false>(xLocal,rstdLocal,gammaLocal,betaLocal,scales1Local,scales2Local,zeroPoints1Local,zeroPoints2Local,
                    y1Local,y2Local,curUbFactor,numR,numQ,xGammaBetaAlign,scalesAlign,zeroPointsAlign,yAlign);
                } else if (optionMask == 0b1001) {
                    ComputeQuant<true, false, false, true, false>(xLocal,rstdLocal,gammaLocal,betaLocal,scales1Local,scales2Local,zeroPoints1Local,zeroPoints2Local,
                    y1Local,y2Local,curUbFactor,numR,numQ,xGammaBetaAlign,scalesAlign,zeroPointsAlign,yAlign);
                } else if (optionMask == 0b0001) {
                    ComputeQuant<false, false, false, true, false>(xLocal,rstdLocal,gammaLocal,betaLocal,scales1Local,scales2Local,zeroPoints1Local,zeroPoints2Local,
                    y1Local,y2Local,curUbFactor,numR,numQ,xGammaBetaAlign,scalesAlign,zeroPointsAlign,yAlign);
                } else if (optionMask == 0b1110) {
                    ComputeQuant<true, true, true, false, false>(xLocal,rstdLocal,gammaLocal,betaLocal,scales1Local,scales2Local,zeroPoints1Local,zeroPoints2Local,
                    y1Local,y2Local,curUbFactor,numR,numQ,xGammaBetaAlign,scalesAlign,zeroPointsAlign,yAlign);
                } else if (optionMask == 0b0110) {
                    ComputeQuant<false, true, true, false, false>(xLocal,rstdLocal,gammaLocal,betaLocal,scales1Local,scales2Local,zeroPoints1Local,zeroPoints2Local,
                    y1Local,y2Local,curUbFactor,numR,numQ,xGammaBetaAlign,scalesAlign,zeroPointsAlign,yAlign);
                } else if (optionMask == 0b1010) {
                    ComputeQuant<true, false, true, false, false>(xLocal,rstdLocal,gammaLocal,betaLocal,scales1Local,scales2Local,zeroPoints1Local,zeroPoints2Local,
                    y1Local,y2Local,curUbFactor,numR,numQ,xGammaBetaAlign,scalesAlign,zeroPointsAlign,yAlign);
                } else if (optionMask == 0b0010) {
                    ComputeQuant<false, false, true, false, false>(xLocal,rstdLocal,gammaLocal,betaLocal,scales1Local,scales2Local,zeroPoints1Local,zeroPoints2Local,
                    y1Local,y2Local,curUbFactor,numR,numQ,xGammaBetaAlign,scalesAlign,zeroPointsAlign,yAlign);
                } else if (optionMask == 0b1100) {
                    ComputeQuant<true, true, false, false, false>(xLocal,rstdLocal,gammaLocal,betaLocal,scales1Local,scales2Local,zeroPoints1Local,zeroPoints2Local,
                    y1Local,y2Local,curUbFactor,numR,numQ,xGammaBetaAlign,scalesAlign,zeroPointsAlign,yAlign);
                } else if (optionMask == 0b0100) {
                    ComputeQuant<false, true, false, false, false>(xLocal,rstdLocal,gammaLocal,betaLocal,scales1Local,scales2Local,zeroPoints1Local,zeroPoints2Local,
                    y1Local,y2Local,curUbFactor,numR,numQ,xGammaBetaAlign,scalesAlign,zeroPointsAlign,yAlign);
                } else if (optionMask == 0b1000) {
                    ComputeQuant<true, false, false, false, false>(xLocal,rstdLocal,gammaLocal,betaLocal,scales1Local,scales2Local,zeroPoints1Local,zeroPoints2Local,
                    y1Local,y2Local,curUbFactor,numR,numQ,xGammaBetaAlign,scalesAlign,zeroPointsAlign,yAlign);
                } else if (optionMask == 0b0000) {
                    ComputeQuant<false, false, false, false, false>(xLocal,rstdLocal,gammaLocal,betaLocal,scales1Local,scales2Local,zeroPoints1Local,zeroPoints2Local,
                    y1Local,y2Local,curUbFactor,numR,numQ,xGammaBetaAlign,scalesAlign,zeroPointsAlign,yAlign);
                }

            } else {
                if (optionMask == 0b1111) {
                    ComputeQuant<true, true, true, true, true>(xLocal,rstdLocal,gammaLocal,betaLocal,scales1Local,scales2Local,zeroPoints1Local,zeroPoints2Local,
                    y1Local,y2Local,curUbFactor,numR,numQ,xGammaBetaAlign,scalesAlign,zeroPointsAlign,yAlign);
                } else if (optionMask == 0b0111) {
                    ComputeQuant<false, true, true, true, true>(xLocal,rstdLocal,gammaLocal,betaLocal,scales1Local,scales2Local,zeroPoints1Local,zeroPoints2Local,
                    y1Local,y2Local,curUbFactor,numR,numQ,xGammaBetaAlign,scalesAlign,zeroPointsAlign,yAlign);
                } else if (optionMask == 0b1011) {
                    ComputeQuant<true, false, true, true, true>(xLocal,rstdLocal,gammaLocal,betaLocal,scales1Local,scales2Local,zeroPoints1Local,zeroPoints2Local,
                    y1Local,y2Local,curUbFactor,numR,numQ,xGammaBetaAlign,scalesAlign,zeroPointsAlign,yAlign);
                } else if (optionMask == 0b0011) {
                    ComputeQuant<false, false, true, true, true>(xLocal,rstdLocal,gammaLocal,betaLocal,scales1Local,scales2Local,zeroPoints1Local,zeroPoints2Local,
                    y1Local,y2Local,curUbFactor,numR,numQ,xGammaBetaAlign,scalesAlign,zeroPointsAlign,yAlign);
                } else if (optionMask == 0b1101) {
                    ComputeQuant<true, true, false, true, true>(xLocal,rstdLocal,gammaLocal,betaLocal,scales1Local,scales2Local,zeroPoints1Local,zeroPoints2Local,
                    y1Local,y2Local,curUbFactor,numR,numQ,xGammaBetaAlign,scalesAlign,zeroPointsAlign,yAlign);
                } else if (optionMask == 0b0101) {
                    ComputeQuant<false, true, false, true, true>(xLocal,rstdLocal,gammaLocal,betaLocal,scales1Local,scales2Local,zeroPoints1Local,zeroPoints2Local,
                    y1Local,y2Local,curUbFactor,numR,numQ,xGammaBetaAlign,scalesAlign,zeroPointsAlign,yAlign);
                } else if (optionMask == 0b1001) {
                    ComputeQuant<true, false, false, true, true>(xLocal,rstdLocal,gammaLocal,betaLocal,scales1Local,scales2Local,zeroPoints1Local,zeroPoints2Local,
                    y1Local,y2Local,curUbFactor,numR,numQ,xGammaBetaAlign,scalesAlign,zeroPointsAlign,yAlign);
                } else if (optionMask == 0b0001) {
                    ComputeQuant<false, false, false, true, true>(xLocal,rstdLocal,gammaLocal,betaLocal,scales1Local,scales2Local,zeroPoints1Local,zeroPoints2Local,
                    y1Local,y2Local,curUbFactor,numR,numQ,xGammaBetaAlign,scalesAlign,zeroPointsAlign,yAlign);
                } else if (optionMask == 0b1110) {
                    ComputeQuant<true, true, true, false, true>(xLocal,rstdLocal,gammaLocal,betaLocal,scales1Local,scales2Local,zeroPoints1Local,zeroPoints2Local,
                    y1Local,y2Local,curUbFactor,numR,numQ,xGammaBetaAlign,scalesAlign,zeroPointsAlign,yAlign);
                } else if (optionMask == 0b0110) {
                    ComputeQuant<false, true, true, false, true>(xLocal,rstdLocal,gammaLocal,betaLocal,scales1Local,scales2Local,zeroPoints1Local,zeroPoints2Local,
                    y1Local,y2Local,curUbFactor,numR,numQ,xGammaBetaAlign,scalesAlign,zeroPointsAlign,yAlign);
                } else if (optionMask == 0b1010) {
                    ComputeQuant<true, false, true, false, true>(xLocal,rstdLocal,gammaLocal,betaLocal,scales1Local,scales2Local,zeroPoints1Local,zeroPoints2Local,
                    y1Local,y2Local,curUbFactor,numR,numQ,xGammaBetaAlign,scalesAlign,zeroPointsAlign,yAlign);
                } else if (optionMask == 0b0010) {
                    ComputeQuant<false, false, true, false, true>(xLocal,rstdLocal,gammaLocal,betaLocal,scales1Local,scales2Local,zeroPoints1Local,zeroPoints2Local,
                    y1Local,y2Local,curUbFactor,numR,numQ,xGammaBetaAlign,scalesAlign,zeroPointsAlign,yAlign);
                } else if (optionMask == 0b1100) {
                    ComputeQuant<true, true, false, false, true>(xLocal,rstdLocal,gammaLocal,betaLocal,scales1Local,scales2Local,zeroPoints1Local,zeroPoints2Local,
                    y1Local,y2Local,curUbFactor,numR,numQ,xGammaBetaAlign,scalesAlign,zeroPointsAlign,yAlign);
                } else if (optionMask == 0b0100) {
                    ComputeQuant<false, true, false, false, true>(xLocal,rstdLocal,gammaLocal,betaLocal,scales1Local,scales2Local,zeroPoints1Local,zeroPoints2Local,
                    y1Local,y2Local,curUbFactor,numR,numQ,xGammaBetaAlign,scalesAlign,zeroPointsAlign,yAlign);
                } else if (optionMask == 0b1000) {
                    ComputeQuant<true, false, false, false, true>(xLocal,rstdLocal,gammaLocal,betaLocal,scales1Local,scales2Local,zeroPoints1Local,zeroPoints2Local,
                    y1Local,y2Local,curUbFactor,numR,numQ,xGammaBetaAlign,scalesAlign,zeroPointsAlign,yAlign);
                } else if (optionMask == 0b0000) {
                    ComputeQuant<false, false, false, false, true>(xLocal,rstdLocal,gammaLocal,betaLocal,scales1Local,scales2Local,zeroPoints1Local,zeroPoints2Local,
                    y1Local,y2Local,curUbFactor,numR,numQ,xGammaBetaAlign,scalesAlign,zeroPointsAlign,yAlign);
                }
            }
    }

    template <bool HAS_BETA, bool HAS_ZEROPINTS2, bool HAS_ZEROPINTS1, bool HAS_SCALES2, bool IS_SCALES_DIV>
    __aicore__ inline void ComputeQuant(LocalTensor<T_X> xLocal, LocalTensor<float> rstdLocal, 
        LocalTensor<T_X> gammaLocal, LocalTensor<T_X> betaLocal, 
        LocalTensor<T_SCALES> scales1Local, LocalTensor<T_SCALES> scales2Local, 
        LocalTensor<T_ZEROPOINTS> zeroPoints1Local, LocalTensor<T_ZEROPOINTS> zeroPoints2Local, 
        LocalTensor<yDtype> y1Local, LocalTensor<yDtype> y2Local, 
        int64_t curUbFactor, int64_t numR, int64_t numQ, int64_t xGammaBetaAlign, int64_t scalesAlign, 
        int64_t zeroPointsAlign, int64_t yAlign)
    {
        uint16_t loopsA = static_cast<uint16_t>(curUbFactor);
        uint16_t loopsR = static_cast<uint16_t>(CeilDiv(static_cast<uint32_t>(numR), vectorLenB32));
        uint32_t sregR = static_cast<uint16_t>(numR);
        uint32_t sregxGammaBetaAlign = static_cast<uint16_t>(xGammaBetaAlign);
        uint32_t sregyAlign = static_cast<uint16_t>(yAlign);
        __local_mem__ T_X* xAddr = (__ubuf__ T_X*)xLocal.GetPhyAddr();
        __local_mem__ float* rstdAddr = (__ubuf__ float*)rstdLocal.GetPhyAddr();
        __local_mem__ T_X* gammaAddr = (__ubuf__ T_X*)gammaLocal.GetPhyAddr();
        __local_mem__ T_SCALES* scales1Addr = (__ubuf__ T_SCALES*)scales1Local.GetPhyAddr();
        __local_mem__ T_ZEROPOINTS* zeroPoints1Addr;
        __local_mem__ T_SCALES* scales2Addr;
        __local_mem__ T_ZEROPOINTS* zeroPoints2Addr;
        __local_mem__ T_X* betaAddr;
        __local_mem__ yDtype* y1Addr;
        __local_mem__ yDtype* y2Addr;

        if constexpr(HAS_ZEROPINTS1) {
            zeroPoints1Addr = (__ubuf__ T_ZEROPOINTS*)zeroPoints1Local.GetPhyAddr();
        }
        if constexpr(HAS_SCALES2) {
            scales2Addr = (__ubuf__ T_SCALES*)scales2Local.GetPhyAddr();
        }
        if constexpr(HAS_ZEROPINTS2) {
            zeroPoints2Addr = (__ubuf__ T_ZEROPOINTS*)zeroPoints2Local.GetPhyAddr();
        }
        if constexpr(HAS_BETA) {
            betaAddr = (__ubuf__ T_X*)betaLocal.GetPhyAddr();
        }
        
        y1Addr = (__ubuf__ yDtype*)y1Local.GetPhyAddr();
        if constexpr((HAS_ZEROPINTS2 || HAS_SCALES2)) {
            y2Addr = (__ubuf__ yDtype*)y2Local.GetPhyAddr();
        }
        // y = cast((x * rstd * gamma + beta) * scales + zeropints)
        if (numQ == 1) {
            // scales + zeropints shape [1]
            // rstd shape [A, 1]
            __VEC_SCOPE__
            {
                RegTensor<float> xReg, rstdReg, gammaReg, betaReg;
                RegTensor<float> scales1Reg, zeroPoints1Reg, scales2Reg, zeroPoints2Reg;
                RegTensor<float> mul1Reg, mul2Reg;
                RegTensor<float> scales1ResultReg, scales2ResultReg;
                MaskReg pregFull = CreateMask<float, MaskPattern::ALL>();
                MaskReg pregOne = CreateMask<float, MaskPattern::VL1>();
                MaskReg mask4Int4 = CreateMask<float, MaskPattern::H>();

                // ld scales and zeropoints
                LoadScalarForDtypeTIn(scales1Addr, scales1Reg, pregFull, 0);
                if constexpr(HAS_ZEROPINTS1) {
                    LoadScalarForDtypeTIn(zeroPoints1Addr, zeroPoints1Reg, pregFull, 0);
                }
                if constexpr(HAS_SCALES2) {
                    LoadScalarForDtypeTIn(scales2Addr, scales2Reg, pregFull, 0);
                }
                if constexpr(HAS_ZEROPINTS2) {
                    LoadScalarForDtypeTIn(zeroPoints2Addr, zeroPoints2Reg, pregFull, 0);
                }
                for (uint16_t i = 0; i < loopsA; i++) {
                    // ld rstd
                    uint32_t sregElewiseNum = numR;
                    LoadScalarForDtypeTIn(rstdAddr, rstdReg, pregFull, i);
                    for (uint16_t j = 0; j < loopsR; j++) {
                        MaskReg pregCurLoop = UpdateMask<float>(sregElewiseNum);
                        LoadTensorForDtypeTIn(xAddr, xReg, pregCurLoop, (i * sregxGammaBetaAlign + j * vectorLenB32));
                        Mul(mul1Reg, xReg, rstdReg, pregCurLoop);
                        LoadTensorForDtypeTIn(gammaAddr, gammaReg, pregCurLoop, (j * vectorLenB32));
                        Mul(mul2Reg, gammaReg, mul1Reg, pregCurLoop);
                        if constexpr(HAS_BETA) {
                            LoadTensorForDtypeTIn(betaAddr, betaReg, pregCurLoop, (j * vectorLenB32));
                            Add(mul2Reg, mul2Reg, betaReg, pregCurLoop);
                        }
                        if constexpr(IS_SCALES_DIV) {
                            Div(scales1ResultReg, mul2Reg, scales1Reg, pregCurLoop); 
                        } else {
                            Mul(scales1ResultReg, mul2Reg, scales1Reg, pregCurLoop); 
                        }

                        if constexpr(HAS_ZEROPINTS1) {
                            Add(scales1ResultReg, scales1ResultReg, zeroPoints1Reg, pregCurLoop); 
                        }

                        StoreTensorForDtypeTOut(y1Addr, scales1ResultReg, pregCurLoop, mask4Int4, (i * sregyAlign + j * vectorLenB32));

                        if constexpr((HAS_ZEROPINTS2 || HAS_SCALES2)) {
                            if constexpr(HAS_SCALES2) {
                                if constexpr(IS_SCALES_DIV) {
                                    Div(scales2ResultReg, mul2Reg, scales2Reg, pregCurLoop); 
                                } else {
                                    Mul(scales2ResultReg, mul2Reg, scales2Reg, pregCurLoop); 
                                }
                            }
                            if constexpr(HAS_ZEROPINTS2) {
                                Add(scales2ResultReg, scales2ResultReg, zeroPoints2Reg, pregCurLoop); 
                            }
                            StoreTensorForDtypeTOut(y2Addr, scales2ResultReg, pregCurLoop, mask4Int4, (i * sregyAlign + j * vectorLenB32));
                        }
                    }
                }
            }
        } else {
            __VEC_SCOPE__
            {
                RegTensor<float> xReg, rstdReg, gammaReg, betaReg;
                RegTensor<float> scales1Reg, zeroPoints1Reg, scales2Reg, zeroPoints2Reg;
                RegTensor<float> mul1Reg, mul2Reg;
                RegTensor<float> scales1ResultReg, scales2ResultReg;
                MaskReg pregFull = CreateMask<float, MaskPattern::ALL>();
                MaskReg pregOne = CreateMask<float, MaskPattern::VL1>();
                MaskReg mask4Int4 = CreateMask<float, MaskPattern::H>();
                // ld scales and zeropoints
                for (uint16_t i = 0; i < loopsA; i++) {
                    // ld rstd
                    uint32_t sregElewiseNum = numR;
                    LoadScalarForDtypeTIn(rstdAddr, rstdReg, pregFull, i);
                    for (uint16_t j = 0; j < loopsR; j++) {
                        MaskReg pregCurLoop = UpdateMask<float>(sregElewiseNum);
                        LoadTensorForDtypeTIn(xAddr, xReg, pregCurLoop, (i * sregxGammaBetaAlign + j * vectorLenB32));
                        Mul(mul1Reg, xReg, rstdReg, pregCurLoop);
                        LoadTensorForDtypeTIn(gammaAddr, gammaReg, pregCurLoop, j * vectorLenB32);
                        Mul(mul2Reg, gammaReg, mul1Reg, pregCurLoop);
                        if constexpr(HAS_BETA) {
                            LoadTensorForDtypeTIn(betaAddr, betaReg, pregCurLoop, j * vectorLenB32);
                            Add(mul2Reg, mul2Reg, betaReg, pregCurLoop);
                        }
                        LoadTensorForDtypeTIn(scales1Addr, scales1Reg, pregCurLoop, j * vectorLenB32);
                        if constexpr(IS_SCALES_DIV) {
                            Div(scales1ResultReg, mul2Reg, scales1Reg, pregCurLoop); 
                        } else {
                            Mul(scales1ResultReg, mul2Reg, scales1Reg, pregCurLoop); 
                        }

                        if constexpr(HAS_ZEROPINTS1) {
                            LoadTensorForDtypeTIn(zeroPoints1Addr, zeroPoints1Reg, pregCurLoop, j * vectorLenB32);
                            Add(scales1ResultReg, scales1ResultReg, zeroPoints1Reg, pregCurLoop); 
                        }

                        StoreTensorForDtypeTOut(y1Addr, scales1ResultReg, pregCurLoop, mask4Int4, (i * sregyAlign + j * vectorLenB32));

                        if constexpr((HAS_ZEROPINTS2 || HAS_SCALES2)) {
                            if constexpr(HAS_SCALES2) {
                                LoadTensorForDtypeTIn(scales2Addr, scales2Reg, pregCurLoop, j * vectorLenB32);
                                if constexpr(IS_SCALES_DIV) {
                                    Div(scales2ResultReg, mul2Reg, scales2Reg, pregCurLoop); 
                                } else {
                                    Mul(scales2ResultReg, mul2Reg, scales2Reg, pregCurLoop); 
                                }
                            }
                            if constexpr(HAS_ZEROPINTS2) {
                                LoadTensorForDtypeTIn(zeroPoints2Addr, zeroPoints2Reg, pregCurLoop, j * vectorLenB32);
                                Add(scales2ResultReg, scales2ResultReg, zeroPoints2Reg, pregCurLoop); 
                            }
                            StoreTensorForDtypeTOut(y2Addr, scales2ResultReg, pregCurLoop, mask4Int4, (i * sregyAlign + j * vectorLenB32));
                        }
                    }
                }
            }
        }
       
    }
};
} // namespace RmsNormQuantV2
#endif // RMS_NORM_QUANT_V2_REBASE_H_

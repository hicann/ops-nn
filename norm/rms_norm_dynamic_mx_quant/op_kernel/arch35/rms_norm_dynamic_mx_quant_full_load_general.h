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
 * \file rms_norm_dynamic_mx_quant_full_load_general.h
 * \brief RmsNormDynamicMxQuant full load general template (TilingKey=1000)
 */

#ifndef RMS_NORM_DYNAMIC_MX_QUANT_FULL_LOAD_GENERAL_H
#define RMS_NORM_DYNAMIC_MX_QUANT_FULL_LOAD_GENERAL_H

#include "rms_norm_dynamic_mx_quant_common.h"

namespace RmsNormDynamicMxQuantNs {

using namespace AscendC;
using namespace AscendC::MicroAPI;

template <typename T_X, typename T_GAMMA, typename T_Y>
class KernelRmsNormDynamicMxQuantFullLoadGeneral {
    static constexpr int64_t DOUBLE_BUFFER = 2;
    static constexpr int64_t FP32_UB_BLOCK_ALIGN = 8;

public:
    __aicore__ inline KernelRmsNormDynamicMxQuantFullLoadGeneral(TPipe* pipe)
    {
        pipe_ = pipe;
    }

    __aicore__ inline void Init(
        GM_ADDR x, GM_ADDR gamma, GM_ADDR beta, GM_ADDR y, GM_ADDR mxScale, GM_ADDR rstd, GM_ADDR workspace,
        const RmsNormDynamicMxQuantFullLoadGeneralTilingData* tilingData)
    {
        tilingData_ = tilingData;
        blockIdx_ = GetBlockIdx();

#if (__NPU_ARCH__ == 3510)
        // Init中获取原始SPR配置，避免循环内反复获取
        oriOverflowMode_ = AscendC::GetCtrlSpr<FLOAT_OVERFLOW_MODE_CTRL, FLOAT_OVERFLOW_MODE_CTRL>();
#endif

        // set gm
        gammaGm_.SetGlobalBuffer((__gm__ T_GAMMA*)gamma);
        if (tilingData_->hasInputBeta) {
            betaGm_.SetGlobalBuffer((__gm__ T_GAMMA*)beta);
        }
        if (tilingData_->hasOutputRstd) {
            rstdGm_.SetGlobalBuffer((__gm__ float*)rstd);
        }

        xGm_.SetGlobalBuffer((__gm__ T_X*)x);
        yGm_.SetGlobalBuffer((__gm__ uint8_t*)y);
        mxScaleGm_.SetGlobalBuffer((__gm__ uint8_t*)mxScale);

        // rms norm
        int64_t gammaAlignedSize = CeilAlign(tilingData_->numN * sizeof(T_GAMMA), UB_BLOCK_SIZE);
        pipe_->InitBuffer(gammaInQueue_, 1, gammaAlignedSize);
        if (tilingData_->hasInputBeta) {
            pipe_->InitBuffer(betaInQueue_, 1, gammaAlignedSize);
        }

        int64_t xMxBlockAligned =
            CeilAlign(tilingData_->mUbFactor * tilingData_->totalNMxBlockAligned, UB_BLOCK_SIZE / sizeof(T_X));
        pipe_->InitBuffer(xInQueue_, DOUBLE_BUFFER, xMxBlockAligned * sizeof(T_X));

        int64_t binAddVls = CeilDiv(tilingData_->colFlodFactor, VL_FP32);
        int64_t binAddVlsAligned = CeilAlign(binAddVls, blockSizeB32);
        int64_t binAddTmpBufSize = tilingData_->mUbFactor * binAddVlsAligned * sizeof(float);

        pipe_->InitBuffer(xReduceTmpBuffer_, binAddTmpBufSize);

        int64_t mUbFactorAlignedSize = CeilAlign(tilingData_->mUbFactor * sizeof(float), UB_BLOCK_SIZE);
        pipe_->InitBuffer(rstdOutQueue_, DOUBLE_BUFFER, mUbFactorAlignedSize);
        pipe_->InitBuffer(xReduceOutBuffer_, mUbFactorAlignedSize);

        // 缓存norm中间结果，补pad0
        int64_t yBufSize = tilingData_->mUbFactor * tilingData_->totalNMxBlockAligned;
        pipe_->InitBuffer(yBuffer_, yBufSize * sizeof(T_X));

        // dynamic mx quant
        int64_t mUbFactorAlignedFour = CeilAlign(tilingData_->mUbFactor, DIGIT_FOUR);
        int64_t yOutBufSize =
            CeilAlign(mUbFactorAlignedFour * tilingData_->totalNMxBlockAligned * sizeof(uint8_t), UB_BLOCK_SIZE);
        pipe_->InitBuffer(yOutQueue_, DOUBLE_BUFFER, yOutBufSize);

        int64_t scaleBufSize =
            CeilAlign(tilingData_->mUbFactor * tilingData_->totalNMxBlockNumAlignedTwo * sizeof(T_X), UB_BLOCK_SIZE);

        if (tilingData_->needPadScale) {
            pipe_->InitBuffer(mxScaleOutTmpBuffer_, scaleBufSize);
        }
        pipe_->InitBuffer(mxScaleOutQueue_, DOUBLE_BUFFER, scaleBufSize);
        pipe_->InitBuffer(maxExpBuffer_, scaleBufSize);
        pipe_->InitBuffer(maxhalfScaleBuffer_, scaleBufSize);

        if (tilingData_->needPadN) {
            int64_t yOutTmpBufSize =
                CeilAlign(tilingData_->mUbFactor * tilingData_->totalNMxBlockAligned * sizeof(uint8_t), UB_BLOCK_SIZE);
            pipe_->InitBuffer(yOutTmpBuffer_, yOutTmpBufSize);
        }

        if constexpr (IsSame<T_Y, fp8_e4m3fn_t>::value) {
            f8Emax_ = FP8_E4M3_MAX_EXP;
            dtypeMax = FP8_E4M3_MAX;
        } else {
            f8Emax_ = FP8_E5M2_MAX_EXP;
            dtypeMax = FP8_E5M2_MAX;
        }

        elementAfterReduce_ = Ops::Base::GetVRegSize() / UB_BLOCK_SIZE;
    }

    __aicore__ inline void Process()
    {
        if (blockIdx_ >= tilingData_->usedCoreNum) {
            return;
        }

        CopyInGammaBeta();

        gammaLocal_ = gammaInQueue_.DeQue<T_GAMMA>();

        if (tilingData_->hasInputBeta) {
            betaLocal_ = betaInQueue_.DeQue<T_GAMMA>();
        }

        int64_t usedTailCores = blockIdx_ < tilingData_->mTailCores ? blockIdx_ : tilingData_->mTailCores;

        int64_t xGmOffset = (blockIdx_ * tilingData_->mPerCore + usedTailCores) * tilingData_->numN;
        int64_t rstdGmOffset = blockIdx_ * tilingData_->mPerCore + usedTailCores;
        int64_t mxScaleGmOffset =
            (blockIdx_ * tilingData_->mPerCore + usedTailCores) * tilingData_->totalNMxBlockNumAlignedTwo;

        int64_t curM = tilingData_->mPerCore;
        curM = blockIdx_ < tilingData_->mTailCores ? (curM + 1) : curM;

        int64_t curMLoop = CeilDiv(curM, tilingData_->mUbFactor);
        int64_t mUbTail = curM - (curMLoop - 1) * tilingData_->mUbFactor;

        for (int64_t i = 0; i < curMLoop; i++) {
            int64_t curM = (i == (curMLoop - 1)) ? mUbTail : tilingData_->mUbFactor;
            int64_t xOffset = xGmOffset + i * tilingData_->mUbFactor * tilingData_->numN;
            int64_t rstdOffset = rstdGmOffset + i * tilingData_->mUbFactor;
            int64_t mxScaleOffset =
                mxScaleGmOffset + i * tilingData_->mUbFactor * tilingData_->totalNMxBlockNumAlignedTwo;
            CopyInX(xOffset, curM);

            // 融合 RMSNorm 计算和量化
            ComputeRmsNormAndQuant(curM, rstdOffset, xOffset, mxScaleOffset);
        }

        gammaInQueue_.FreeTensor(gammaLocal_);
        if (tilingData_->hasInputBeta) {
            betaInQueue_.FreeTensor(betaLocal_);
        }
    }

    __aicore__ inline void CopyInGammaBeta()
    {
        DataCopyPadExtParams<T_GAMMA> padParams;
        padParams.isPad = false;
        DataCopyExtParams copyParams;
        copyParams.blockCount = 1;
        copyParams.blockLen = tilingData_->numN * sizeof(T_GAMMA);
        copyParams.srcStride = 0;
        copyParams.dstStride = 0;

        LocalTensor<T_GAMMA> gammaLocal = gammaInQueue_.AllocTensor<T_GAMMA>();
        DataCopyPad(gammaLocal, gammaGm_, copyParams, padParams);
        gammaInQueue_.EnQue(gammaLocal);

        if (tilingData_->hasInputBeta) {
            LocalTensor<T_GAMMA> betaLocal = betaInQueue_.AllocTensor<T_GAMMA>();
            DataCopyPad(betaLocal, betaGm_, copyParams, padParams);
            betaInQueue_.EnQue(betaLocal);
        }
    }

    __aicore__ inline void CopyInX(int64_t offset, int64_t curM)
    {
        DataCopyPadExtParams<T_X> padParams;
        padParams.isPad = false;
        padParams.leftPadding = 0;
        padParams.rightPadding = 0;
        padParams.paddingValue = 0;

        DataCopyExtParams copyParams;
        copyParams.blockCount = curM;
        copyParams.blockLen = tilingData_->numN * sizeof(T_X);
        copyParams.srcStride = 0;
        copyParams.dstStride = 0;

        LocalTensor<T_X> xLocal = xInQueue_.AllocTensor<T_X>();
        DataCopyPad(xLocal, xGm_[offset], copyParams, padParams);
        xInQueue_.EnQue(xLocal);
    }

    __aicore__ inline void ComputeRmsNormAndQuant(
        int64_t curM, int64_t rstdOffset, int64_t xOffset, int64_t mxScaleOffset)
    {
        LocalTensor<T_X> xLocal = xInQueue_.DeQue<T_X>();
        LocalTensor<float> rstdLocal = rstdOutQueue_.AllocTensor<float>();
        LocalTensor<float> xReduceTmpLocal = xReduceTmpBuffer_.Get<float>();
        LocalTensor<float> xReduceOutTmpLocal = xReduceOutBuffer_.Get<float>();
        ComputeSquareReduceSum(xLocal, xReduceTmpLocal, xReduceOutTmpLocal, curM);
        ComputeRstd(xReduceOutTmpLocal, rstdLocal, curM, tilingData_->epsilon, tilingData_->avgFactor);
        rstdOutQueue_.EnQue(rstdLocal);
        rstdLocal = rstdOutQueue_.DeQue<float>();
        if (tilingData_->hasOutputRstd) {
            DataCopyExtParams copyOutParamsRstd;
            copyOutParamsRstd.blockCount = 1;
            copyOutParamsRstd.blockLen = curM * sizeof(float);
            copyOutParamsRstd.srcStride = 0;
            copyOutParamsRstd.dstStride = 0;
            DataCopyPad(rstdGm_[rstdOffset], rstdLocal, copyOutParamsRstd);
        }

        // 融合量化：边计算 Y chunk 边量化
        ComputeYAndQuant<RoundMode::CAST_TRUNC, RoundMode::CAST_RINT>(
            xLocal, gammaLocal_, betaLocal_, rstdLocal, curM, xOffset, mxScaleOffset);

        xInQueue_.FreeTensor(xLocal);
        rstdOutQueue_.FreeTensor(rstdLocal);
    }

    __aicore__ inline void ComputeSquareReduceSum(
        LocalTensor<T_X> xLocal, LocalTensor<float> xTmpLocal, LocalTensor<float> xReduceTmpLocal, uint64_t curUbFactor)
    {
        __local_mem__ T_X* xLocalAddr = (__local_mem__ T_X*)xLocal.GetPhyAddr();
        __local_mem__ float* xTmpLocalUbAddr = (__local_mem__ float*)xTmpLocal.GetPhyAddr();
        __local_mem__ float* xReduceTmpLocalUbAddr = (__local_mem__ float*)xReduceTmpLocal.GetPhyAddr();

        if (tilingData_->numNUbAligned <= VL_FP32) {
            CalculateSquareReduceSumRLessThanVL(
                xLocalAddr, xReduceTmpLocalUbAddr, curUbFactor, tilingData_->numN, tilingData_->numNUbAligned);
        } else if (tilingData_->numNUbAligned <= (VL_FP32 + VL_FP32)) {
            CalculateSquareReduceSumRLessThanTwoVL(
                xLocalAddr, xReduceTmpLocalUbAddr, curUbFactor, tilingData_->numN, tilingData_->numNUbAligned);
        } else if (tilingData_->numNUbAligned <= VL_FP32 * VL_FP32 * NUM_TWO) {
            CalculateSquareReduceSumRCommon<NUM_ONE>(
                xLocalAddr, xTmpLocalUbAddr, xReduceTmpLocalUbAddr, curUbFactor, tilingData_->numN,
                tilingData_->numNUbAligned, tilingData_->colFlodFactor);
        } else {
            CalculateSquareReduceSumRCommon<NUM_TWO>(
                xLocalAddr, xTmpLocalUbAddr, xReduceTmpLocalUbAddr, curUbFactor, tilingData_->numN,
                tilingData_->numNUbAligned, tilingData_->colFlodFactor);
        }
    }

    __aicore__ inline void CalculateSquareReduceSumRLessThanVL(
        __local_mem__ T_X* xLocalAddr, __local_mem__ float* xReduceTmpLocalUbAddr, uint64_t curUbFactor,
        uint64_t numCol, uint64_t numColAlign)
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

    __aicore__ inline void CalculateSquareReduceSumRLessThanTwoVL(
        __local_mem__ T_X* xLocalAddr, __local_mem__ float* xReduceTmpLocalUbAddr, uint64_t curUbFactor,
        uint64_t numCol, uint64_t numColAlign)
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
            uint32_t colTailSreg = colNum - VL_FP32;
            MaskReg pregTail = UpdateMask<float>(colTailSreg);
            for (uint16_t i = 0; i < curAloops; i++) {
                LoadTensorForDtypeTIn(xLocalAddr, xReg, pregFull, (i * colNumAlign));
                Mul(squareReg, xReg, xReg, pregFull);
                LoadTensorForDtypeTIn(xLocalAddr + VL_FP32, xTailReg, pregTail, (i * colNumAlign));
                Mul(squareTailReg, xTailReg, xTailReg, pregTail);
                Add(addReg, squareReg, squareTailReg, pregFull);
                ReduceSum(sumReg, addReg, pregFull);
                DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(xReduceTmpLocalUbAddr + i, sumReg, pregOne);
            }
        }
    }

    template <int32_t LAST_LOOP_NUMS>
    __aicore__ inline void CalculateSquareReduceSumRCommon(
        __local_mem__ T_X* xLocalAddr, __local_mem__ float* xTmpLocalUbAddr, __local_mem__ float* xReduceTmpLocalUbAddr,
        uint64_t curUbFactor, uint64_t numCol, uint64_t numColAlign, uint64_t colFlodFactor)
    {
        uint32_t colNum = static_cast<uint32_t>(numCol);
        uint32_t colNumAlign = static_cast<uint32_t>(numColAlign);
        uint32_t colFlodNum = static_cast<uint32_t>(colFlodFactor);
        uint16_t curAloops = static_cast<uint16_t>(curUbFactor);
        // first flod
        uint32_t firstFlodTial = static_cast<uint32_t>(colNum - colFlodFactor);
        uint16_t firstFlodAddLoops = static_cast<uint16_t>((firstFlodTial + VL_FP32 - 1) / VL_FP32);
        uint16_t firstFlodWithOutAddLoops =
            static_cast<uint16_t>((colFlodNum + VL_FP32 - 1) / VL_FP32) - firstFlodAddLoops;
        // first vcadd
        uint32_t firstVcaddNum = static_cast<uint32_t>((colFlodFactor + VL_FP32 - 1) / VL_FP32);
        uint32_t firstVcaddNumCeilAlign =
            static_cast<uint32_t>((firstVcaddNum + blockSizeB32 - 1) / blockSizeB32 * blockSizeB32);
        // second flod
        // rstd cal
        uint16_t elewiseLoop = static_cast<uint16_t>((curUbFactor + VL_FP32 - 1) / VL_FP32);

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
                    LoadTensorForDtypeTIn(xLocalAddr, xReg1, pregFull, (i * colNumAlign + j * VL_FP32));
                    Mul(squareReg1, xReg1, xReg1, pregFull);
                    LoadTensorForDtypeTIn(xLocalAddr + colFlodNum, xReg2, pregFull, (i * colNumAlign + j * VL_FP32));
                    Mul(squareReg2, xReg2, xReg2, pregLoop);
                    Add(addReg, squareReg1, squareReg2, pregFull);
                    ReduceSum(sumReg, addReg, pregFull);
                    DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(
                        xTmpLocalUbAddr + static_cast<uint32_t>(i * firstVcaddNumCeilAlign + j), sumReg, pregOne);
                }
                for (uint16_t j = 0; j < static_cast<uint16_t>(firstFlodWithOutAddLoops); j++) {
                    LoadTensorForDtypeTIn(
                        xLocalAddr + firstFlodAddLoops * VL_FP32, xReg3, pregFull, (i * colNumAlign + j * VL_FP32));
                    Mul(squareReg3, xReg3, xReg3, pregFull);
                    ReduceSum(sumReg3, squareReg3, pregFull);
                    DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(
                        xTmpLocalUbAddr + static_cast<uint32_t>(i * firstVcaddNumCeilAlign + firstFlodAddLoops + j),
                        sumReg3, pregOne);
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
                uint32_t sregSecondReduce = firstVcaddNum - VL_FP32;
                MaskReg pregLast = UpdateMask<float>(sregSecondReduce);
                RegTensor<float> shiftLeft;
                for (uint16_t i = 0; i < curAloops; i++) {
                    DataCopy(xReg1, xTmpLocalUbAddr + static_cast<uint32_t>(i * firstVcaddNumCeilAlign));
                    DataCopy(xReg2, xTmpLocalUbAddr + static_cast<uint32_t>(i * firstVcaddNumCeilAlign + VL_FP32));
                    ShiftLefts(
                        (RegTensor<uint32_t>&)shiftLeft, (RegTensor<uint32_t>&)xReg2, static_cast<int16_t>(0),
                        pregLast);
                    Add(addReg, xReg1, shiftLeft, pregFull);
                    ReduceSum(sumReg, addReg, pregFull);
                    DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(xReduceTmpLocalUbAddr + i, sumReg, pregOne);
                }
            }
        }
    }

    __aicore__ inline void ComputeRstd(
        LocalTensor<float> xReduceTmpLocal, LocalTensor<float> rstdLocal, uint64_t curUbFactor, float epsilon,
        float avgFactor)
    {
        __local_mem__ float* rstdLocalUbAddr = (__local_mem__ float*)rstdLocal.GetPhyAddr();
        __local_mem__ float* xReduceTmpLocalUbAddr = (__local_mem__ float*)xReduceTmpLocal.GetPhyAddr();
        uint16_t aLoop = static_cast<uint16_t>((curUbFactor + VL_FP32 - 1) / VL_FP32);
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
                DataCopy(var, xReduceTmpLocalUbAddr + a * VL_FP32);
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
                DataCopy(rstdLocalUbAddr + a * VL_FP32, rstd, pregLoop);
            }
        }
    }

    template <typename T_IN>
    __aicore__ inline void LoadTensorForDtypeTIn(
        __local_mem__ T_IN* src, RegTensor<float>& dst, MaskReg& preg, uint32_t offset)
    {
        if constexpr (IsSameType<T_IN, float>::value) {
            DataCopy<float, LoadDist::DIST_NORM>(dst, src + offset);
        } else {
            RegTensor<T_IN> xIn;
            DataCopy<T_IN, LoadDist::DIST_UNPACK_B16>(xIn, src + offset);
            Cast<float, T_IN, CAST_B16_TO_B32>(dst, xIn, preg);
        }
    }

    template <typename T_OUT>
    __aicore__ inline void StoreTensorForDtypeTOut(
        __local_mem__ T_OUT* dst, RegTensor<float>& src, MaskReg& preg, uint32_t offset)
    {
        if constexpr (IsSameType<T_OUT, float>::value) {
            DataCopy<T_OUT, StoreDist::DIST_NORM>(dst + offset, src, preg);
        } else {
            RegTensor<T_OUT> xOut;
            Cast<T_OUT, float, CAST_B32_TO_B16>(xOut, src, preg);
            DataCopy<T_OUT, StoreDist::DIST_PACK_B32>(dst + offset, xOut, preg);
        }
    }

    template <bool hasInputBeta = false>
    __aicore__ inline void ComputeY(
        LocalTensor<T_X> xLocal, LocalTensor<T_GAMMA> gammaLocal, LocalTensor<T_GAMMA> betaLocal,
        LocalTensor<float> rstdLocal, LocalTensor<T_X> yLocal, int64_t curM)
    {
        __local_mem__ T_X* xLocalAddr = (__local_mem__ T_X*)xLocal.GetPhyAddr();
        __local_mem__ T_GAMMA* gammaLocalUbAddr = (__local_mem__ T_GAMMA*)gammaLocal.GetPhyAddr();
        __local_mem__ T_GAMMA* betaLocalUbAddr;
        if constexpr (hasInputBeta) {
            betaLocalUbAddr = (__local_mem__ T_GAMMA*)betaLocal.GetPhyAddr();
        }

        __local_mem__ float* rstdLocalUbAddr = (__local_mem__ float*)rstdLocal.GetPhyAddr();
        __local_mem__ T_X* yLocalUbAddr = (__local_mem__ T_X*)yLocal.GetPhyAddr();
        uint32_t vl_fp32 = static_cast<uint16_t>(VL_FP32);
        uint32_t nNum = static_cast<uint32_t>(tilingData_->numN);
        uint16_t mloops = static_cast<uint16_t>(curM);
        uint16_t nloops = static_cast<uint16_t>(CeilDiv(nNum, VL_FP32));
        uint32_t xInputStride = static_cast<uint32_t>(tilingData_->numN);
        __VEC_SCOPE__
        {
            RegTensor<float> xReg;
            RegTensor<float> RstdReg;
            RegTensor<float> gammaReg;
            RegTensor<float> betaReg;
            RegTensor<float> yReg;
            MaskReg pregMask;
            for (uint16_t i = 0; i < mloops; i++) {
                uint32_t sreg = nNum;
                DataCopy<float, LoadDist::DIST_BRC_B32>(RstdReg, rstdLocalUbAddr + i);
                for (uint16_t j = 0; j < nloops; j++) {
                    pregMask = UpdateMask<float>(sreg);
                    uint32_t gammaElemOffset = j * vl_fp32;
                    uint32_t xElemOffset = i * xInputStride + gammaElemOffset;
                    LoadTensorForDtypeT<T_X>(xLocalAddr, xReg, pregMask, xElemOffset);

                    Mul(yReg, xReg, RstdReg, pregMask);
                    LoadTensorForDtypeT<T_GAMMA>(gammaLocalUbAddr, gammaReg, pregMask, gammaElemOffset);
                    Mul(yReg, yReg, gammaReg, pregMask);
                    if constexpr (hasInputBeta) {
                        LoadTensorForDtypeT<T_GAMMA>(betaLocalUbAddr, betaReg, pregMask, gammaElemOffset);
                        Add(yReg, yReg, betaReg, pregMask);
                    }

                    StoreTensorForDtypeT(yLocalUbAddr, yReg, pregMask, xElemOffset);
                }
            }
        }
    }

    template <bool hasInputBeta = false>
    __aicore__ inline void ComputeYWithPad(
        LocalTensor<T_X> xLocal, LocalTensor<T_GAMMA> gammaLocal, LocalTensor<T_GAMMA> betaLocal,
        LocalTensor<float> rstdLocal, LocalTensor<T_X> yLocal, int64_t curM)
    {
        __local_mem__ T_GAMMA* betaLocalUbAddr;
        if constexpr (hasInputBeta) {
            betaLocalUbAddr = (__local_mem__ T_GAMMA*)betaLocal.GetPhyAddr();
        }
        __local_mem__ T_X* xLocalAddr = (__local_mem__ T_X*)xLocal.GetPhyAddr();
        __local_mem__ T_GAMMA* gammaLocalUbAddr = (__local_mem__ T_GAMMA*)gammaLocal.GetPhyAddr();

        __local_mem__ float* rstdLocalUbAddr = (__local_mem__ float*)rstdLocal.GetPhyAddr();
        __local_mem__ T_X* yLocalUbAddr = (__local_mem__ T_X*)yLocal.GetPhyAddr();
        uint16_t mloops = static_cast<uint16_t>(curM);
        uint32_t nNum = static_cast<uint32_t>(tilingData_->numN);
        uint32_t vl_fp32 = static_cast<uint16_t>(VL_FP32);
        uint16_t nloops = static_cast<uint16_t>(nNum / VL_FP32);
        uint32_t nTail = nNum - nloops * VL_FP32;
        uint32_t padNum = tilingData_->totalNMxBlockAligned - tilingData_->numN;
        uint32_t lastLoopN = nTail + padNum; // <= 32
        uint32_t xInputStride = static_cast<uint32_t>(tilingData_->numNUbAligned);
        uint32_t yOutputStride = static_cast<uint32_t>(tilingData_->totalNMxBlockAligned);
        __VEC_SCOPE__
        {
            RegTensor<float> xReg;
            RegTensor<float> RstdReg;
            RegTensor<float> gammaReg;
            RegTensor<float> betaReg;
            RegTensor<float> yReg;

            MaskReg pregMask;

            RegTensor<float> xRegLast;
            RegTensor<float> gammaRegLast;
            RegTensor<float> betaRegLast;
            RegTensor<float> yRegLast;
            RegTensor<float> zeroReg;
            MaskReg pregLastLoopMask;
            MaskReg pregTailMask;
            for (uint16_t i = 0; i < mloops; i++) {
                uint32_t sreg = nNum;
                uint32_t sregTail = nTail;
                uint32_t sregLastLoop = lastLoopN;
                DataCopy<float, LoadDist::DIST_BRC_B32>(RstdReg, rstdLocalUbAddr + i);
                for (uint16_t j = 0; j < nloops; j++) {
                    pregMask = UpdateMask<float>(sreg);
                    uint32_t gammaElemOffset = j * vl_fp32;
                    uint32_t xElemOffset = i * xInputStride + gammaElemOffset;
                    uint32_t yElemOffset = i * yOutputStride + gammaElemOffset;
                    LoadTensorForDtypeT<T_X>(xLocalAddr, xReg, pregMask, xElemOffset);

                    Mul(yReg, xReg, RstdReg, pregMask);
                    LoadTensorForDtypeT<T_GAMMA>(gammaLocalUbAddr, gammaReg, pregMask, gammaElemOffset);
                    Mul(yReg, yReg, gammaReg, pregMask);
                    if constexpr (hasInputBeta) {
                        LoadTensorForDtypeT<T_GAMMA>(betaLocalUbAddr, betaReg, pregMask, gammaElemOffset);
                        Add(yReg, yReg, betaReg, pregMask);
                    }

                    StoreTensorForDtypeT(yLocalUbAddr, yReg, pregMask, yElemOffset);
                }

                // last loop
                pregLastLoopMask = UpdateMask<float>(sregLastLoop);
                pregTailMask = UpdateMask<float>(sregTail);

                uint32_t gammaLastOffset = nloops * vl_fp32;
                uint32_t xLastOffset = i * xInputStride + gammaLastOffset;
                uint32_t yLastOffset = i * yOutputStride + gammaLastOffset;
                LoadTensorForDtypeT<T_X>(xLocalAddr, xRegLast, pregTailMask, xLastOffset);

                Duplicate(yRegLast, static_cast<float>(0), pregLastLoopMask);
                Mul(yRegLast, xRegLast, RstdReg, pregTailMask);
                LoadTensorForDtypeT<T_GAMMA>(gammaLocalUbAddr, gammaRegLast, pregTailMask, gammaLastOffset);
                Mul(yRegLast, yRegLast, gammaRegLast, pregTailMask);
                if constexpr (hasInputBeta) {
                    LoadTensorForDtypeT<T_GAMMA>(betaLocalUbAddr, betaRegLast, pregTailMask, gammaLastOffset);
                    Add(yRegLast, yRegLast, betaRegLast, pregTailMask);
                }

                StoreTensorForDtypeT(yLocalUbAddr, yRegLast, pregLastLoopMask, yLastOffset);
            }
        }
    }

    template <AscendC::RoundMode toBf16RoundMode, AscendC::RoundMode roundMode>
    __aicore__ inline void ComputeYAndQuant(
        LocalTensor<T_X> xLocal, LocalTensor<T_GAMMA> gammaLocal, LocalTensor<T_GAMMA> betaLocal,
        LocalTensor<float> rstdLocal, int64_t curM, int64_t xOffset, int64_t mxScaleOffset)
    {
        LocalTensor<T_X> yLocal = yBuffer_.Get<T_X>();
        if (tilingData_->needPadN) {
            if (tilingData_->hasInputBeta) {
                ComputeYWithPad<true>(xLocal, gammaLocal, betaLocal, rstdLocal, yLocal, curM);
            } else {
                ComputeYWithPad<false>(xLocal, gammaLocal, betaLocal, rstdLocal, yLocal, curM);
            }
        } else {
            if (tilingData_->hasInputBeta) {
                ComputeY<true>(xLocal, gammaLocal, betaLocal, rstdLocal, yLocal, curM);
            } else {
                ComputeY<false>(xLocal, gammaLocal, betaLocal, rstdLocal, yLocal, curM);
            }
        }

#if (__NPU_ARCH__ == 3510)
        AscendC::SetCtrlSpr<FLOAT_OVERFLOW_MODE_CTRL, FLOAT_OVERFLOW_MODE_CTRL>(0);
#endif
        ComputeMxQuantAndOutput<toBf16RoundMode, roundMode>(yLocal, curM, xOffset, mxScaleOffset);
#if (__NPU_ARCH__ == 3510)
        AscendC::SetCtrlSpr<FLOAT_OVERFLOW_MODE_CTRL, FLOAT_OVERFLOW_MODE_CTRL>(oriOverflowMode_);
#endif
    }

    template <AscendC::RoundMode toBf16RoundMode, AscendC::RoundMode roundMode>
    __aicore__ inline void ComputeMxQuantAndOutput(
        LocalTensor<T_X> yLocal, int64_t curM, int64_t yOutOffset, int64_t scaleOutOffset)
    {
        LocalTensor<uint16_t> maxExpLocal = maxExpBuffer_.Get<uint16_t>();
        uint32_t totalScaleInUB = curM * tilingData_->totalNMxBlockNum;
        uint32_t totalCountInUB = curM * tilingData_->totalNMxBlockNum * tilingData_->mxBlockSize;
        uint16_t loopNum = (totalCountInUB + VL_B16 * DIGIT_TWO - 1) / (VL_B16 * DIGIT_TWO);
        uint16_t loopNumScale = (totalScaleInUB + VL_B16 - 1) / VL_B16;
        uint16_t loopNumScale4NV = (totalScaleInUB + VL_FP32 - 1) / VL_FP32;

        auto srcAddr = reinterpret_cast<__ubuf__ T_X*>(yLocal.GetPhyAddr());
        auto maxExpAddr = reinterpret_cast<__ubuf__ uint16_t*>(maxExpLocal.GetPhyAddr());

        LocalTensor<uint16_t> mxScaleLocal = mxScaleOutQueue_.AllocTensor<uint16_t>();
        auto mxScaleLocalAddr = reinterpret_cast<__ubuf__ uint16_t*>(mxScaleLocal.GetPhyAddr());

        LocalTensor<uint16_t> padMxScaleLocal = mxScaleLocal;
        if (tilingData_->needPadScale) {
            padMxScaleLocal = mxScaleOutTmpBuffer_.Get<uint16_t>();
        }
        auto padMxScaleLocalAddr = reinterpret_cast<__ubuf__ uint16_t*>(padMxScaleLocal.GetPhyAddr());

        LocalTensor<uint16_t> halfScaleLocal = maxhalfScaleBuffer_.Get<uint16_t>();
        auto halfScaleLocalAddr = reinterpret_cast<__ubuf__ uint16_t*>(halfScaleLocal.GetPhyAddr());

        LocalTensor<int8_t> outLocal = yOutQueue_.AllocTensor<int8_t>();
        auto outLocalAddr = reinterpret_cast<__ubuf__ int8_t*>(outLocal.GetPhyAddr());

        if (tilingData_->scaleAlg == 0) {
            ComputeMaxExpOCP(srcAddr, maxExpAddr, totalCountInUB, loopNum);
            ComputeScaleOCP(maxExpAddr, padMxScaleLocalAddr, halfScaleLocalAddr, totalScaleInUB, loopNumScale);
        } else {
            ComputeMaxExpcuBLAS(srcAddr, maxExpAddr, totalCountInUB, loopNum);
            ComputeScalecuBLAS(maxExpAddr, padMxScaleLocalAddr, halfScaleLocalAddr, totalScaleInUB, loopNumScale4NV);
        }

        if (tilingData_->needPadScale) {
            auto tmpLocal = reinterpret_cast<__ubuf__ int8_t*>(padMxScaleLocal.GetPhyAddr());
            auto dstLocal = reinterpret_cast<__ubuf__ int8_t*>(mxScaleLocal.GetPhyAddr());
            PadScaleData(dstLocal, tmpLocal, curM);
            PipeBarrier<PIPE_V>();
        }

        srcAddr = reinterpret_cast<__ubuf__ T_X*>(yLocal.GetPhyAddr());
        halfScaleLocalAddr = reinterpret_cast<__ubuf__ uint16_t*>(halfScaleLocal.GetPhyAddr());

        if (tilingData_->needPadN) {
            LocalTensor<int8_t> outBufferLocal = yOutTmpBuffer_.Get<int8_t>();
            auto outBufferLocalAddr = reinterpret_cast<__ubuf__ int8_t*>(outBufferLocal.GetPhyAddr());
            ComputeData<toBf16RoundMode, roundMode>(
                srcAddr, halfScaleLocalAddr, outBufferLocalAddr, totalCountInUB, loopNum);

            outBufferLocalAddr = reinterpret_cast<__ubuf__ int8_t*>(outBufferLocal.GetPhyAddr());
            DeletePadData(outLocalAddr, outBufferLocalAddr, curM);
        } else {
            ComputeData<toBf16RoundMode, roundMode>(srcAddr, halfScaleLocalAddr, outLocalAddr, totalCountInUB, loopNum);
        }

        yOutQueue_.EnQue(outLocal);
        mxScaleOutQueue_.EnQue(mxScaleLocal);

        // 立即输出
        CopyOutScaleAndY(yOutOffset, scaleOutOffset, curM);
    }

    __aicore__ inline void CopyOutScaleAndY(int64_t yOutOffset, int64_t scaleOutOffset, int64_t curM)
    {
        LocalTensor<uint8_t> outLocal = yOutQueue_.DeQue<uint8_t>();
        DataCopyExtParams copyOutParamData = {0, 0, 0, 0, 0};
        copyOutParamData.blockCount = 1;
        copyOutParamData.blockLen = curM * tilingData_->numN;

        copyOutParamData.srcStride = 0;
        copyOutParamData.dstStride = 0; // tilingData_->numN - copyOutParamData.blockLen;

        DataCopyPad<uint8_t, PaddingMode::Compact>(yGm_[yOutOffset], outLocal, copyOutParamData);
        yOutQueue_.FreeTensor(outLocal);

        LocalTensor<uint8_t> mxScaleLocal = mxScaleOutQueue_.DeQue<uint8_t>();
        DataCopyExtParams copyOutParamScale = {0, 0, 0, 0, 0};
        copyOutParamScale.blockCount = 1;
        copyOutParamScale.blockLen = curM * tilingData_->totalNMxBlockNumAlignedTwo;
        copyOutParamScale.srcStride = 0;
        copyOutParamScale.dstStride = 0; // tilingData_->totalNMxBlockNumAlignedTwo - copyOutParamScale.blockLen;

        DataCopyPad<uint8_t, PaddingMode::Compact>(mxScaleGm_[scaleOutOffset], mxScaleLocal, copyOutParamScale);
        mxScaleOutQueue_.FreeTensor(mxScaleLocal);
    }

    template <AscendC::RoundMode toBf16RoundMode, AscendC::RoundMode roundMode>
    __aicore__ inline void ComputeData( //
        __ubuf__ T_X* srcAddr, __ubuf__ uint16_t* halfScaleLocalAddr, __ubuf__ int8_t* outLocalAddr,
        uint32_t totalCountInUB, uint16_t loopNum)
    {
        uint32_t totalCountInUB2 = totalCountInUB * DIGIT_TWO;
        __VEC_SCOPE__
        {
            AscendC::MicroAPI::MaskReg dataMask1;
            AscendC::MicroAPI::MaskReg dataMask2;
            AscendC::MicroAPI::MaskReg dataMask3;
            AscendC::MicroAPI::MaskReg dataMask4;
            AscendC::MicroAPI::MaskReg dataMask5;
            AscendC::MicroAPI::MaskReg maskAll =
                AscendC::MicroAPI::CreateMask<uint16_t, AscendC::MicroAPI::MaskPattern::ALL>();
            AscendC::MicroAPI::RegTensor<uint16_t> halfScaleForMul;
            AscendC::MicroAPI::RegTensor<float> floatScaleForMul;
            AscendC::MicroAPI::RegTensor<T_X> vdExp0;
            AscendC::MicroAPI::RegTensor<T_X> vdExp1;
            AscendC::MicroAPI::RegTensor<T_X> vdExp0Convert;
            AscendC::MicroAPI::RegTensor<T_X> vdExp1Convert;
            AscendC::MicroAPI::RegTensor<bfloat16_t> vdExp0BF16;
            AscendC::MicroAPI::RegTensor<bfloat16_t> vdExp1BF16;
            AscendC::MicroAPI::RegTensor<float> vdExp0FP32Zero;
            AscendC::MicroAPI::RegTensor<float> vdExp0FP32One;
            AscendC::MicroAPI::RegTensor<float> vdExp1FP32Zero;
            AscendC::MicroAPI::RegTensor<float> vdExp1FP32One;
            AscendC::MicroAPI::RegTensor<T_Y> vdExp0FP8Zero;
            AscendC::MicroAPI::RegTensor<T_Y> vdExp0FP8One;
            AscendC::MicroAPI::RegTensor<T_Y> vdExp1FP8Zero;
            AscendC::MicroAPI::RegTensor<T_Y> vdExp1FP8One;
            AscendC::MicroAPI::RegTensor<bfloat16_t> vdBF16Exp0FP4;
            AscendC::MicroAPI::RegTensor<bfloat16_t> vdBF16Exp1FP4;
            static constexpr AscendC::MicroAPI::CastTrait castTrait = {
                AscendC::MicroAPI::RegLayout::ZERO, AscendC::MicroAPI::SatMode::UNKNOWN,
                AscendC::MicroAPI::MaskMergeMode::ZEROING, roundMode};
            static constexpr AscendC::MicroAPI::CastTrait castTraitHalf2Bf16 = {
                AscendC::MicroAPI::RegLayout::UNKNOWN, AscendC::MicroAPI::SatMode::UNKNOWN,
                AscendC::MicroAPI::MaskMergeMode::ZEROING, toBf16RoundMode};
            static constexpr AscendC::MicroAPI::CastTrait castTraitZero = {
                AscendC::MicroAPI::RegLayout::ZERO, AscendC::MicroAPI::SatMode::UNKNOWN,
                AscendC::MicroAPI::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};
            static constexpr AscendC::MicroAPI::CastTrait castTraitOne = {
                AscendC::MicroAPI::RegLayout::ONE, AscendC::MicroAPI::SatMode::UNKNOWN,
                AscendC::MicroAPI::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};
            static constexpr AscendC::MicroAPI::CastTrait castTrait32to8 = {
                AscendC::MicroAPI::RegLayout::ZERO, AscendC::MicroAPI::SatMode::SAT,
                AscendC::MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
            static constexpr AscendC::MicroAPI::CastTrait castTrait32to80 = {
                AscendC::MicroAPI::RegLayout::ZERO, AscendC::MicroAPI::SatMode::SAT,
                AscendC::MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
            static constexpr AscendC::MicroAPI::CastTrait castTrait32to81 = {
                AscendC::MicroAPI::RegLayout::ONE, AscendC::MicroAPI::SatMode::SAT,
                AscendC::MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
            static constexpr AscendC::MicroAPI::CastTrait castTrait32to82 = {
                AscendC::MicroAPI::RegLayout::TWO, AscendC::MicroAPI::SatMode::SAT,
                AscendC::MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
            static constexpr AscendC::MicroAPI::CastTrait castTrait32to83 = {
                AscendC::MicroAPI::RegLayout::THREE, AscendC::MicroAPI::SatMode::SAT,
                AscendC::MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
            dataMask1 = AscendC::MicroAPI::CreateMask<T_X>();
            dataMask2 = AscendC::MicroAPI::CreateMask<T_X>();
            dataMask3 = AscendC::MicroAPI::CreateMask<T_X>();
            dataMask4 = AscendC::MicroAPI::CreateMask<T_X>();
            dataMask5 = AscendC::MicroAPI::CreateMask<T_Y>();
            for (uint16_t i = 0; i < loopNum; i++) {
                AscendC::MicroAPI::DataCopy<
                    T_X, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE,
                    AscendC::MicroAPI::LoadDist::DIST_DINTLV_B16>(vdExp0, vdExp1, srcAddr, VL_B16 * DIGIT_TWO);
                AscendC::MicroAPI::DataCopy<
                    uint16_t, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE,
                    AscendC::MicroAPI::LoadDist::DIST_E2B_B16>(
                    halfScaleForMul, halfScaleLocalAddr, elementAfterReduce_);
                if constexpr (IsSame<T_X, half>::value) {
                    AscendC::MicroAPI::Cast<float, T_X, castTraitZero>(vdExp0FP32Zero, vdExp0, dataMask1);
                    AscendC::MicroAPI::Cast<float, T_X, castTraitOne>(vdExp0FP32One, vdExp0, dataMask1);
                    AscendC::MicroAPI::Cast<float, bfloat16_t, castTraitZero>(
                        floatScaleForMul, (AscendC::MicroAPI::RegTensor<bfloat16_t>&)halfScaleForMul, maskAll);
                    AscendC::MicroAPI::Mul(vdExp0FP32Zero, vdExp0FP32Zero, floatScaleForMul, dataMask3);
                    AscendC::MicroAPI::Mul(vdExp0FP32One, vdExp0FP32One, floatScaleForMul, dataMask4);

                    AscendC::MicroAPI::Cast<float, T_X, castTraitZero>(vdExp1FP32Zero, vdExp1, dataMask1);
                    AscendC::MicroAPI::Cast<float, T_X, castTraitOne>(vdExp1FP32One, vdExp1, dataMask1);
                    AscendC::MicroAPI::Mul(vdExp1FP32Zero, vdExp1FP32Zero, floatScaleForMul, dataMask3);
                    AscendC::MicroAPI::Mul(vdExp1FP32One, vdExp1FP32One, floatScaleForMul, dataMask4);

                } else {
                    AscendC::MicroAPI::Mul(
                        vdExp0, vdExp0, (AscendC::MicroAPI::RegTensor<T_X>&)halfScaleForMul, dataMask1);
                    AscendC::MicroAPI::Mul(
                        vdExp1, vdExp1, (AscendC::MicroAPI::RegTensor<T_X>&)halfScaleForMul, dataMask1);

                    AscendC::MicroAPI::Cast<float, T_X, castTraitZero>(vdExp0FP32Zero, vdExp0, dataMask1);
                    AscendC::MicroAPI::Cast<float, T_X, castTraitOne>(vdExp0FP32One, vdExp0, dataMask1);
                    AscendC::MicroAPI::Cast<float, T_X, castTraitZero>(vdExp1FP32Zero, vdExp1, dataMask2);
                    AscendC::MicroAPI::Cast<float, T_X, castTraitOne>(vdExp1FP32One, vdExp1, dataMask2);
                }
                AscendC::MicroAPI::Cast<T_Y, float, castTrait32to80>(vdExp0FP8Zero, vdExp0FP32Zero, dataMask3);
                AscendC::MicroAPI::Cast<T_Y, float, castTrait32to82>(vdExp0FP8One, vdExp0FP32One, dataMask3);
                AscendC::MicroAPI::Cast<T_Y, float, castTrait32to81>(vdExp1FP8Zero, vdExp1FP32Zero, dataMask4);
                AscendC::MicroAPI::Cast<T_Y, float, castTrait32to83>(vdExp1FP8One, vdExp1FP32One, dataMask4);

                AscendC::MicroAPI::Add(
                    (AscendC::MicroAPI::RegTensor<uint8_t>&)vdExp0FP8Zero,
                    (AscendC::MicroAPI::RegTensor<uint8_t>&)vdExp0FP8Zero,
                    (AscendC::MicroAPI::RegTensor<uint8_t>&)vdExp0FP8One, dataMask5);
                AscendC::MicroAPI::Add(
                    (AscendC::MicroAPI::RegTensor<uint8_t>&)vdExp0FP8Zero,
                    (AscendC::MicroAPI::RegTensor<uint8_t>&)vdExp0FP8Zero,
                    (AscendC::MicroAPI::RegTensor<uint8_t>&)vdExp1FP8Zero, dataMask5);
                AscendC::MicroAPI::Add(
                    (AscendC::MicroAPI::RegTensor<uint8_t>&)vdExp0FP8Zero,
                    (AscendC::MicroAPI::RegTensor<uint8_t>&)vdExp0FP8Zero,
                    (AscendC::MicroAPI::RegTensor<uint8_t>&)vdExp1FP8One, dataMask5);

                AscendC::MicroAPI::DataCopy<
                    int8_t, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE,
                    AscendC::MicroAPI::StoreDist::DIST_NORM_B8>(
                    outLocalAddr, (AscendC::MicroAPI::RegTensor<int8_t>&)vdExp0FP8Zero, OUT_ALL, dataMask5);
            }
        }
        return;
    }

    __aicore__ inline void ComputeMaxExpOCP( //
        __ubuf__ T_X* srcAddr, __ubuf__ uint16_t* maxExpAddr, uint32_t totalCountInUB, uint16_t loopNum)
    {
        __VEC_SCOPE__
        {
            AscendC::MicroAPI::RegTensor<T_X> vdExp0;
            AscendC::MicroAPI::RegTensor<T_X> vdExp1;
            AscendC::MicroAPI::RegTensor<bfloat16_t> vdExp0BF16;
            AscendC::MicroAPI::RegTensor<bfloat16_t> vdExp1BF16;
            AscendC::MicroAPI::RegTensor<uint16_t> vdExpSelect0;
            AscendC::MicroAPI::RegTensor<uint16_t> vdExpSelect1;
            AscendC::MicroAPI::RegTensor<uint16_t> vdExpExtract0;
            AscendC::MicroAPI::RegTensor<uint16_t> vdExpExtract1;

            AscendC::MicroAPI::RegTensor<uint16_t> expMaskBF16;
            AscendC::MicroAPI::Duplicate(expMaskBF16, MAX_EXP_FOR_BF16);

            AscendC::MicroAPI::RegTensor<uint16_t> invalidMaskFP16;
            AscendC::MicroAPI::Duplicate(invalidMaskFP16, INVALID_FLOAT16);
            AscendC::MicroAPI::RegTensor<uint16_t> vdMaxExp;
            AscendC::MicroAPI::MaskReg scaleMask1;
            AscendC::MicroAPI::MaskReg scaleMask2;
            AscendC::MicroAPI::MaskReg invalidDataMask0;
            AscendC::MicroAPI::MaskReg invalidDataMask1;
            AscendC::MicroAPI::UnalignReg u1;
            static constexpr AscendC::MicroAPI::CastTrait castTraitHalf2Bf16 = {
                AscendC::MicroAPI::RegLayout::UNKNOWN, AscendC::MicroAPI::SatMode::UNKNOWN,
                AscendC::MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_TRUNC};
            for (uint16_t i = 0; i < loopNum; i++) {
                scaleMask1 = AscendC::MicroAPI::UpdateMask<T_X>(totalCountInUB);
                scaleMask2 = AscendC::MicroAPI::UpdateMask<T_X>(totalCountInUB);
                AscendC::MicroAPI::DataCopy<
                    T_X, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE,
                    AscendC::MicroAPI::LoadDist::DIST_DINTLV_B16>(vdExp0, vdExp1, srcAddr, VL_B16 * DIGIT_TWO);
                if constexpr (IsSame<T_X, half>::value) {
                    AscendC::MicroAPI::And(
                        vdExpSelect0, (AscendC::MicroAPI::RegTensor<uint16_t>&)vdExp0, invalidMaskFP16, scaleMask1);
                    AscendC::MicroAPI::And(
                        vdExpSelect1, (AscendC::MicroAPI::RegTensor<uint16_t>&)vdExp1, invalidMaskFP16, scaleMask1);
                    AscendC::MicroAPI::Compare<uint16_t, CMPMODE::NE>(
                        invalidDataMask0, vdExpSelect0, invalidMaskFP16, scaleMask1);
                    AscendC::MicroAPI::Compare<uint16_t, CMPMODE::NE>(
                        invalidDataMask1, vdExpSelect1, invalidMaskFP16, scaleMask1);
                    AscendC::MicroAPI::Cast<bfloat16_t, T_X, castTraitHalf2Bf16>(vdExp0BF16, vdExp0, scaleMask1);
                    AscendC::MicroAPI::Cast<bfloat16_t, T_X, castTraitHalf2Bf16>(vdExp1BF16, vdExp1, scaleMask1);
                    AscendC::MicroAPI::And(
                        vdExpExtract0, (AscendC::MicroAPI::RegTensor<uint16_t>&)vdExp0BF16, expMaskBF16, scaleMask1);
                    AscendC::MicroAPI::And(
                        vdExpExtract1, (AscendC::MicroAPI::RegTensor<uint16_t>&)vdExp1BF16, expMaskBF16, scaleMask1);
                    AscendC::MicroAPI::Select<uint16_t>(vdExpExtract0, vdExpExtract0, expMaskBF16, invalidDataMask0);
                    AscendC::MicroAPI::Select<uint16_t>(vdExpExtract1, vdExpExtract1, expMaskBF16, invalidDataMask1);
                } else {
                    AscendC::MicroAPI::And(
                        vdExpExtract0, (AscendC::MicroAPI::RegTensor<uint16_t>&)vdExp0, expMaskBF16, scaleMask1);
                    AscendC::MicroAPI::And(
                        vdExpExtract1, (AscendC::MicroAPI::RegTensor<uint16_t>&)vdExp1, expMaskBF16, scaleMask1);
                }

                AscendC::MicroAPI::Max(vdMaxExp, vdExpExtract0, vdExpExtract1, scaleMask1);
                AscendC::MicroAPI::ReduceMaxWithDataBlock(vdMaxExp, vdMaxExp, scaleMask1);

                AscendC::MicroAPI::DataCopyUnAlign<uint16_t, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                    maxExpAddr, vdMaxExp, u1, elementAfterReduce_);
            }
            AscendC::MicroAPI::DataCopyUnAlignPost(maxExpAddr, u1, 0);
        }
        return;
    }

    __aicore__ inline void ComputeMaxExpcuBLAS( //
        __ubuf__ T_X* srcAddr, __ubuf__ uint16_t* maxExpAddr, uint32_t totalCountInUB, uint16_t loopNum)
    {
        __VEC_SCOPE__
        {
            AscendC::MicroAPI::RegTensor<T_X> vdExp0;
            AscendC::MicroAPI::RegTensor<T_X> vdExp1;
            AscendC::MicroAPI::RegTensor<uint16_t> absMask16Bit;
            AscendC::MicroAPI::Duplicate(absMask16Bit, ABS_MASK_FOR_16BIT);
            AscendC::MicroAPI::RegTensor<uint16_t> vdMaxExp;
            AscendC::MicroAPI::MaskReg scaleMask1;
            AscendC::MicroAPI::UnalignReg u1;
            for (uint16_t i = 0; i < loopNum; i++) {
                scaleMask1 = AscendC::MicroAPI::UpdateMask<T_X>(totalCountInUB);
                AscendC::MicroAPI::DataCopy<
                    T_X, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE,
                    AscendC::MicroAPI::LoadDist::DIST_DINTLV_B16>(vdExp0, vdExp1, srcAddr, VL_B16 * DIGIT_TWO);
                AscendC::MicroAPI::And(
                    (AscendC::MicroAPI::RegTensor<uint16_t>&)vdExp0, (AscendC::MicroAPI::RegTensor<uint16_t>&)vdExp0,
                    absMask16Bit, scaleMask1);
                AscendC::MicroAPI::And(
                    (AscendC::MicroAPI::RegTensor<uint16_t>&)vdExp1, (AscendC::MicroAPI::RegTensor<uint16_t>&)vdExp1,
                    absMask16Bit, scaleMask1);
                AscendC::MicroAPI::Max(
                    vdMaxExp, (AscendC::MicroAPI::RegTensor<uint16_t>&)vdExp0,
                    (AscendC::MicroAPI::RegTensor<uint16_t>&)vdExp1, scaleMask1);
                AscendC::MicroAPI::ReduceMaxWithDataBlock(vdMaxExp, vdMaxExp, scaleMask1);
                AscendC::MicroAPI::DataCopyUnAlign<uint16_t, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                    maxExpAddr, vdMaxExp, u1, elementAfterReduce_);
            }
            AscendC::MicroAPI::DataCopyUnAlignPost(maxExpAddr, u1, 0);
        }
        return;
    }

    __aicore__ inline void ComputeScaleOCP( //
        __ubuf__ uint16_t* maxExpAddr, __ubuf__ uint16_t* mxScaleLocalAddr, __ubuf__ uint16_t* halfScaleLocalAddr,
        uint32_t totalScaleInUB, uint16_t loopNumScale)
    {
        __VEC_SCOPE__
        {
            AscendC::MicroAPI::RegTensor<uint16_t> expMask;
            AscendC::MicroAPI::Duplicate(expMask, MAX_EXP_FOR_BF16);
            AscendC::MicroAPI::RegTensor<uint16_t> vdMaxExp;

            AscendC::MicroAPI::RegTensor<T_X> vdExp0;
            AscendC::MicroAPI::RegTensor<T_X> vdExp1;

            AscendC::MicroAPI::MaskReg cmpResult;
            AscendC::MicroAPI::MaskReg zeroMask;
            AscendC::MicroAPI::MaskReg cmpResultSub;
            AscendC::MicroAPI::MaskReg preMaskScale;
            AscendC::MicroAPI::RegTensor<uint16_t> maxExpValue;
            AscendC::MicroAPI::Duplicate(maxExpValue, f8Emax_);
            AscendC::MicroAPI::RegTensor<uint16_t> sharedExp;
            AscendC::MicroAPI::RegTensor<uint16_t> scaleValue;
            AscendC::MicroAPI::RegTensor<uint16_t> scaleBias;
            AscendC::MicroAPI::Duplicate(scaleBias, BF16_EXP_BIAS);
            AscendC::MicroAPI::RegTensor<uint16_t> halfScale;
            AscendC::MicroAPI::RegTensor<uint16_t> fp8NanRegTensor;
            AscendC::MicroAPI::Duplicate(fp8NanRegTensor, MAX_EXP_FOR_FP8);
            AscendC::MicroAPI::RegTensor<uint16_t> zeroRegTensor;
            AscendC::MicroAPI::Duplicate(zeroRegTensor, 0);
            AscendC::MicroAPI::RegTensor<uint16_t> nanRegTensor;
            AscendC::MicroAPI::Duplicate(nanRegTensor, NAN_CUSTOMIZATION);
            AscendC::MicroAPI::MaskReg invalidDataMask;
            AscendC::MicroAPI::MaskReg specialDataMask;
            AscendC::MicroAPI::RegTensor<uint16_t> specialExpRegTensor;
            AscendC::MicroAPI::Duplicate(specialExpRegTensor, SPECIAL_EXP_THRESHOLD);
            for (uint16_t i = 0; i < loopNumScale; i++) {
                preMaskScale = AscendC::MicroAPI::UpdateMask<uint16_t>(totalScaleInUB);
                AscendC::MicroAPI::DataCopy<uint16_t, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                    vdMaxExp, maxExpAddr, VL_B16);
                AscendC::MicroAPI::Compare<uint16_t, CMPMODE::NE>(
                    cmpResult, vdMaxExp, expMask, preMaskScale); // INF/NAN
                AscendC::MicroAPI::Compare<uint16_t, CMPMODE::NE>(zeroMask, vdMaxExp, zeroRegTensor, preMaskScale);
                AscendC::MicroAPI::Compare<uint16_t, CMPMODE::LE>(invalidDataMask, vdMaxExp, maxExpValue, preMaskScale);

                AscendC::MicroAPI::Select<uint16_t>(vdMaxExp, maxExpValue, vdMaxExp, invalidDataMask);

                AscendC::MicroAPI::Sub(sharedExp, vdMaxExp, maxExpValue, preMaskScale);
                AscendC::MicroAPI::ShiftRights(scaleValue, sharedExp, SHR_NUM_FOR_BF16, preMaskScale);

                AscendC::MicroAPI::Select<uint16_t>(scaleValue, scaleValue, fp8NanRegTensor, cmpResult);
                AscendC::MicroAPI::Select<uint16_t>(scaleValue, scaleValue, zeroRegTensor, zeroMask);

                AscendC::MicroAPI::DataCopy<
                    uint16_t, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE,
                    AscendC::MicroAPI::StoreDist::DIST_PACK_B16>(
                    mxScaleLocalAddr, scaleValue, VL_B16 / DIGIT_TWO, preMaskScale);

                AscendC::MicroAPI::Compare<uint16_t, CMPMODE::EQ>(specialDataMask, sharedExp, scaleBias, preMaskScale);
                AscendC::MicroAPI::Sub(halfScale, scaleBias, sharedExp, preMaskScale);
                AscendC::MicroAPI::Select<uint16_t>(halfScale, halfScale, nanRegTensor, cmpResult);
                AscendC::MicroAPI::Select<uint16_t>(halfScale, halfScale, zeroRegTensor, zeroMask);
                AscendC::MicroAPI::Select<uint16_t>(halfScale, specialExpRegTensor, halfScale, specialDataMask);

                AscendC::MicroAPI::DataCopy<uint16_t, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                    halfScaleLocalAddr, halfScale, VL_B16, preMaskScale);
            }
        }
        return;
    }

    __aicore__ inline void ComputeScalecuBLAS( //
        __ubuf__ uint16_t* maxExpAddr, __ubuf__ uint16_t* mxScaleLocalAddr, __ubuf__ uint16_t* halfScaleLocalAddr,
        uint32_t totalScaleInUB, uint16_t loopNumScale4NV)
    {
        __VEC_SCOPE__
        {
            AscendC::MicroAPI::RegTensor<uint16_t> max16;
            AscendC::MicroAPI::RegTensor<uint32_t> max32;
            AscendC::MicroAPI::RegTensor<uint32_t> exp32;
            AscendC::MicroAPI::RegTensor<uint32_t> man32;
            AscendC::MicroAPI::RegTensor<uint32_t> normalExp32;
            AscendC::MicroAPI::RegTensor<uint32_t> expAddOne32;
            AscendC::MicroAPI::RegTensor<uint32_t> extractExp;
            AscendC::MicroAPI::RegTensor<uint16_t> expOut;
            AscendC::MicroAPI::RegTensor<uint32_t> halfScale;
            AscendC::MicroAPI::RegTensor<uint16_t> recExpOut;

            AscendC::MicroAPI::RegTensor<uint32_t> invMax;
            AscendC::MicroAPI::Duplicate(invMax, dtypeMax);
            AscendC::MicroAPI::RegTensor<uint32_t> manMaskFP32;
            AscendC::MicroAPI::Duplicate(manMaskFP32, MAN_MASK_FLOAT);
            AscendC::MicroAPI::RegTensor<uint32_t> expMask;
            AscendC::MicroAPI::Duplicate(expMask, MAX_EXP_FOR_FP32);
            AscendC::MicroAPI::RegTensor<uint32_t> zeroRegTensor32;
            AscendC::MicroAPI::Duplicate(zeroRegTensor32, 0);
            AscendC::MicroAPI::RegTensor<uint32_t> scaleBias;
            AscendC::MicroAPI::Duplicate(scaleBias, FP32_EXP_BIAS_CUBLAS);
            AscendC::MicroAPI::RegTensor<uint32_t> nanRegTensor;
            AscendC::MicroAPI::Duplicate(nanRegTensor, NAN_CUSTOMIZATION_PACK);
            AscendC::MicroAPI::RegTensor<uint32_t> fp8NanRegTensor;
            AscendC::MicroAPI::Duplicate(fp8NanRegTensor, MAX_EXP_FOR_FP8_IN_FP32);

            AscendC::MicroAPI::MaskReg cmpResult;
            AscendC::MicroAPI::MaskReg zeroMask;
            AscendC::MicroAPI::MaskReg p0;
            AscendC::MicroAPI::MaskReg p1;
            AscendC::MicroAPI::MaskReg p2;
            AscendC::MicroAPI::MaskReg preMaskScale;
            AscendC::MicroAPI::MaskReg maskHalf;
            preMaskScale = AscendC::MicroAPI::CreateMask<uint32_t>();
            maskHalf = AscendC::MicroAPI::CreateMask<uint16_t>();
            static constexpr AscendC::MicroAPI::CastTrait castTraitHalf2Float = {
                AscendC::MicroAPI::RegLayout::ZERO, AscendC::MicroAPI::SatMode::UNKNOWN,
                AscendC::MicroAPI::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};
            for (uint16_t i = 0; i < loopNumScale4NV; i++) {
                AscendC::MicroAPI::DataCopy<
                    uint16_t, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE,
                    AscendC::MicroAPI::LoadDist::DIST_UNPACK_B16>(max16, maxExpAddr, VL_FP32);

                AscendC::MicroAPI::Cast<float, T_X, castTraitHalf2Float>(
                    (AscendC::MicroAPI::RegTensor<float>&)max32, (AscendC::MicroAPI::RegTensor<T_X>&)max16,
                    preMaskScale);
                AscendC::MicroAPI::Compare<uint32_t, CMPMODE::LT>(cmpResult, max32, expMask, preMaskScale);
                AscendC::MicroAPI::Compare<uint32_t, CMPMODE::NE>(zeroMask, max32, zeroRegTensor32, preMaskScale);

                AscendC::MicroAPI::Mul(
                    (AscendC::MicroAPI::RegTensor<float>&)max32, (AscendC::MicroAPI::RegTensor<float>&)max32,
                    (AscendC::MicroAPI::RegTensor<float>&)invMax, preMaskScale);
                AscendC::MicroAPI::ShiftRights(exp32, max32, SHR_NUM_FOR_FP32, preMaskScale);
                AscendC::MicroAPI::And(man32, max32, manMaskFP32, preMaskScale);

                AscendC::MicroAPI::CompareScalar<uint32_t, CMPMODE::GT>(p0, exp32, zeroForAll, preMaskScale);
                AscendC::MicroAPI::CompareScalar<uint32_t, CMPMODE::LT>(p1, exp32, Exp254, preMaskScale);
                AscendC::MicroAPI::CompareScalar<uint32_t, CMPMODE::GT>(p2, man32, zeroForAll, preMaskScale);
                AscendC::MicroAPI::MaskAnd(p0, p0, p1, preMaskScale);
                AscendC::MicroAPI::MaskAnd(p0, p0, p2, preMaskScale);

                AscendC::MicroAPI::CompareScalar<uint32_t, CMPMODE::EQ>(p1, exp32, zeroForAll, preMaskScale);
                AscendC::MicroAPI::CompareScalar<uint32_t, CMPMODE::GT>(p2, man32, halfForMan, preMaskScale);
                AscendC::MicroAPI::MaskAnd(p1, p1, p2, preMaskScale);
                AscendC::MicroAPI::MaskOr(p0, p0, p1, preMaskScale);

                AscendC::MicroAPI::Adds(expAddOne32, exp32, 1, preMaskScale);
                AscendC::MicroAPI::Select(extractExp, expAddOne32, exp32, p0);
                AscendC::MicroAPI::Select<uint32_t>(extractExp, extractExp, fp8NanRegTensor, cmpResult);
                AscendC::MicroAPI::Select<uint32_t>(extractExp, extractExp, zeroRegTensor32, zeroMask);
                AscendC::MicroAPI::Pack<uint16_t, uint32_t, AscendC::MicroAPI::HighLowPart::LOWEST>(expOut, extractExp);

                AscendC::MicroAPI::DataCopy<
                    uint16_t, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE,
                    AscendC::MicroAPI::StoreDist::DIST_PACK_B16>(
                    mxScaleLocalAddr, expOut, VL_FP32 / DIGIT_TWO, maskHalf);

                AscendC::MicroAPI::ShiftLefts(extractExp, extractExp, SHR_NUM_FOR_BF16, preMaskScale);
                AscendC::MicroAPI::Sub(halfScale, scaleBias, extractExp, preMaskScale);
                AscendC::MicroAPI::Select<uint32_t>(halfScale, halfScale, nanRegTensor, cmpResult);
                AscendC::MicroAPI::Select<uint32_t>(halfScale, halfScale, zeroRegTensor32, zeroMask);
                AscendC::MicroAPI::Pack<uint16_t, uint32_t, AscendC::MicroAPI::HighLowPart::LOWEST>(
                    recExpOut, halfScale);

                AscendC::MicroAPI::DataCopy<uint16_t, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                    halfScaleLocalAddr, recExpOut, VL_FP32, maskHalf);
            }
        }
        return;
    }

    __aicore__ inline void DeletePadData(__ubuf__ int8_t* outLocalAddr, __ubuf__ int8_t* inLocalAddr, uint16_t curM)
    {
        uint32_t inputUpdateStride = tilingData_->totalNMxBlockNum * tilingData_->mxBlockSize;
        uint32_t outputUpdateStride = tilingData_->numN;

        uint16_t mloops = static_cast<uint16_t>(curM);
        uint32_t vl_b8 = static_cast<uint32_t>(VECTOR_LENGTH);
        uint32_t nNum = static_cast<uint32_t>(tilingData_->numN);
        uint16_t nloops = static_cast<uint16_t>(CeilDiv(nNum, vl_b8));
        uint32_t inStride = static_cast<uint32_t>(tilingData_->totalNMxBlockNum * tilingData_->mxBlockSize);
        uint32_t remain = nNum - (nloops - 1) * vl_b8;

        __VEC_SCOPE__
        {
            AscendC::MicroAPI::UnalignReg uIn;
            AscendC::MicroAPI::UnalignReg uOut;
            AscendC::MicroAPI::RegTensor<int8_t> inputRegTensor;

            for (uint16_t i = 0; i < mloops; i++) {
                __ubuf__ int8_t* currInAddr = inLocalAddr + i * inStride;
                __ubuf__ int8_t* currOutAddr = outLocalAddr + i * nNum;
                for (uint16_t j = 0; j < static_cast<uint16_t>(nloops - 1); j++) {
                    AscendC::MicroAPI::DataCopyUnAlignPre(uIn, currInAddr);
                    AscendC::MicroAPI::DataCopyUnAlign<int8_t, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                        inputRegTensor, uIn, currInAddr, vl_b8);
                    AscendC::MicroAPI::DataCopyUnAlign<int8_t, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                        currOutAddr, inputRegTensor, uOut, vl_b8);
                    AscendC::MicroAPI::DataCopyUnAlignPost(currOutAddr, uOut, 0);
                }

                AscendC::MicroAPI::DataCopyUnAlignPre(uIn, currInAddr);
                AscendC::MicroAPI::DataCopyUnAlign<int8_t, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                    inputRegTensor, uIn, currInAddr, remain);
                AscendC::MicroAPI::DataCopyUnAlign<int8_t, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                    currOutAddr, inputRegTensor, uOut, remain);
                AscendC::MicroAPI::DataCopyUnAlignPost(currOutAddr, uOut, 0);
            }
        }
        return;
    }

    __aicore__ inline void PadScaleData(__ubuf__ int8_t* outLocalAddr, __ubuf__ int8_t* inLocalAddr, uint16_t curM)
    {
        uint16_t mloops = static_cast<uint16_t>(curM);
        uint32_t vl_b8 = static_cast<uint32_t>(VECTOR_LENGTH);
        uint32_t nNum = static_cast<uint32_t>(tilingData_->totalNMxBlockNum);
        uint16_t nloops = static_cast<uint16_t>(CeilDiv(nNum, vl_b8));
        uint32_t outStride = static_cast<uint32_t>(nNum + 1);
        uint32_t remain = nNum - (nloops - 1) * vl_b8;

        __VEC_SCOPE__
        {
            AscendC::MicroAPI::UnalignReg uInReg;
            AscendC::MicroAPI::UnalignReg uOutReg;
            AscendC::MicroAPI::RegTensor<int8_t> inputRegTensor;
            AscendC::MicroAPI::RegTensor<int8_t> zeroRegTensor;
            AscendC::MicroAPI::Duplicate<int8_t>(zeroRegTensor, 0);
            for (uint16_t i = 0; i < mloops; i++) {
                __ubuf__ int8_t* currInAddr = inLocalAddr + i * nNum;
                __ubuf__ int8_t* currOutAddr = outLocalAddr + i * outStride;
                for (uint16_t j = 0; j < static_cast<uint16_t>(nloops - 1); j++) {
                    AscendC::MicroAPI::DataCopyUnAlignPre(uInReg, currInAddr);
                    AscendC::MicroAPI::DataCopyUnAlign<int8_t, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                        inputRegTensor, uInReg, currInAddr, vl_b8);
                    AscendC::MicroAPI::DataCopyUnAlign<int8_t, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                        currOutAddr, inputRegTensor, uOutReg, vl_b8);
                    AscendC::MicroAPI::DataCopyUnAlignPost(currOutAddr, uOutReg, 0);
                }

                AscendC::MicroAPI::DataCopyUnAlignPre(uInReg, currInAddr);
                AscendC::MicroAPI::DataCopyUnAlign<int8_t, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                    inputRegTensor, uInReg, currInAddr, remain);
                AscendC::MicroAPI::DataCopyUnAlign<int8_t, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                    currOutAddr, inputRegTensor, uOutReg, remain);
                AscendC::MicroAPI::DataCopyUnAlign<int8_t, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                    currOutAddr, zeroRegTensor, uOutReg, 1);
                AscendC::MicroAPI::DataCopyUnAlignPost(currOutAddr, uOutReg, 0);
            }
        }
        return;
    }

private:
    TPipe* pipe_;
    const RmsNormDynamicMxQuantFullLoadGeneralTilingData* tilingData_;
    GlobalTensor<T_X> xGm_;
    GlobalTensor<T_GAMMA> gammaGm_;
    GlobalTensor<T_GAMMA> betaGm_;
    GlobalTensor<float> rstdGm_;
    GlobalTensor<uint8_t> yGm_;
    GlobalTensor<uint8_t> mxScaleGm_;
    GlobalTensor<uint8_t> workspaceGm_;

    LocalTensor<T_GAMMA> gammaLocal_;
    LocalTensor<T_GAMMA> betaLocal_;

    TQue<QuePosition::VECIN, DOUBLE_BUFFER> gammaInQueue_;
    TQue<QuePosition::VECIN, DOUBLE_BUFFER> betaInQueue_;
    TQue<QuePosition::VECOUT, DOUBLE_BUFFER> rstdOutQueue_;
    TQue<QuePosition::VECIN, DOUBLE_BUFFER> xInQueue_;
    TQue<QuePosition::VECOUT, DOUBLE_BUFFER> yOutQueue_;
    TQue<QuePosition::VECOUT, DOUBLE_BUFFER> mxScaleOutQueue_;

    TBuf<QuePosition::VECCALC> xReduceTmpBuffer_;
    TBuf<QuePosition::VECCALC> xReduceOutBuffer_;
    TBuf<QuePosition::VECCALC> yBuffer_; // 临时存储单个 chunk 的 Y 结果

    TBuf<QuePosition::VECCALC> maxExpBuffer_;
    TBuf<QuePosition::VECCALC> maxhalfScaleBuffer_;
    TBuf<QuePosition::VECCALC> yOutTmpBuffer_;
    TBuf<QuePosition::VECCALC> mxScaleOutTmpBuffer_;

    int64_t blockIdx_ = 0;
    uint16_t f8Emax_ = 0;
    uint32_t dtypeMax = 0;
    uint16_t elementAfterReduce_ = 0;
    uint32_t zeroForAll = 0x00000000;
    uint32_t Exp254 = 0x000000fe;
    uint32_t halfForMan = 0x00400000;
#if (__NPU_ARCH__ == 3510)
    int64_t oriOverflowMode_ = 0; // 存储原始SPR配置
#endif
};
} // namespace RmsNormDynamicMxQuantNs

#endif // RMS_NORM_DYNAMIC_MX_QUANT_FULL_LOAD_GENERAL_H

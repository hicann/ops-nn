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
 * \file rms_norm_dynamic_mx_quant_full_load.h
 * \brief RmsNormDynamicMxQuant full load template (TilingKey=1000, 10000)
 */

#ifndef RMS_NORM_DYNAMIC_MX_QUANT_FULL_LOAD_H
#define RMS_NORM_DYNAMIC_MX_QUANT_FULL_LOAD_H

#include "rms_norm_dynamic_mx_quant_common.h"

namespace RmsNormDynamicMxQuantNs {

using namespace AscendC;
using namespace AscendC::MicroAPI;

template <typename T_X, typename T_GAMMA, typename T_Y, bool isOptimizeMode>
class RmsNormDynamicMxQuantFullLoad {
public:
    __aicore__ inline RmsNormDynamicMxQuantFullLoad(TPipe* pipe)
    {
        pipe_ = pipe;
    }

    __aicore__ inline void Init(
        GM_ADDR x, GM_ADDR gamma, GM_ADDR beta, GM_ADDR y, GM_ADDR mxScale, GM_ADDR rstd,
        const RmsNormDynamicMxQuantFullLoadTilingData* tilingData)
    {
        tilingData_ = tilingData;
        blockIdx_ = GetBlockIdx();

#if (__NPU_ARCH__ == 3510)
        // Init中获取原始SPR配置，避免循环内反复获取
        oriOverflowMode_ = AscendC::GetCtrlSpr<FLOAT_OVERFLOW_MODE_CTRL, FLOAT_OVERFLOW_MODE_CTRL>();
#endif
        // set gm
        gammaGm_.SetGlobalBuffer((__gm__ T_GAMMA*)gamma);
        betaGm_.SetGlobalBuffer((__gm__ T_GAMMA*)beta);
        rstdGm_.SetGlobalBuffer((__gm__ float*)rstd);
        xGm_.SetGlobalBuffer((__gm__ T_X*)x);
        yGm_.SetGlobalBuffer((__gm__ uint8_t*)y);
        mxScaleGm_.SetGlobalBuffer((__gm__ uint8_t*)mxScale);

        // rms norm
        int64_t gammaBufSize = ops::CeilAlign(static_cast<int64_t>(tilingData_->numN * sizeof(T_GAMMA)), UB_BLOCK_SIZE);
        pipe_->InitBuffer(gammaInQueue_, 1, gammaBufSize);
        if (tilingData_->hasInputBeta) {
            pipe_->InitBuffer(betaInQueue_, 1, gammaBufSize);
        }

        int64_t xBufSize = tilingData_->mUbFactor * tilingData_->nMxblockAligned * sizeof(T_X);
        pipe_->InitBuffer(xInQueue_, DOUBLE_BUFFER, xBufSize);
        pipe_->InitBuffer(normOutBuffer_, xBufSize); // 缓存norm中间结果，补pad0

        int64_t binAddVls = ops::CeilDiv(tilingData_->binAddFoldPoint, static_cast<int64_t>(VL_FP32));
        int64_t binAddVlsAligned = ops::CeilAlign(binAddVls, UB_BLOCK_SIZE_FP32);
        int64_t binAddTmpBufSize = tilingData_->mUbFactor * binAddVlsAligned * sizeof(float);
        pipe_->InitBuffer(xReduceTmpBuffer_, binAddTmpBufSize);

        int64_t rstdBufSize =
            ops::CeilAlign(static_cast<int64_t>(tilingData_->mUbFactor * sizeof(float)), UB_BLOCK_SIZE);
        pipe_->InitBuffer(rstdOutQueue_, DOUBLE_BUFFER, rstdBufSize);
        pipe_->InitBuffer(xReduceOutBuffer_, rstdBufSize);

        // dynamic mx quant
        int64_t mUbFactorAlignedFour = ops::CeilAlign(tilingData_->mUbFactor, DIGIT_FOUR);
        int64_t yOutBufSize;
        if constexpr (IsSameType<T_Y, fp4x2_e2m1_t>::value || IsSameType<T_Y, fp4x2_e1m2_t>::value) {
            yOutBufSize = ops::CeilAlign(
                static_cast<int64_t>(mUbFactorAlignedFour * tilingData_->nMxblockAligned / DIGIT_TWO * sizeof(uint8_t)),
                UB_BLOCK_SIZE);
        } else {
            yOutBufSize = ops::CeilAlign(
                static_cast<int64_t>(mUbFactorAlignedFour * tilingData_->nMxblockAligned * sizeof(uint8_t)),
                UB_BLOCK_SIZE);
        }
        pipe_->InitBuffer(yOutQueue_, DOUBLE_BUFFER, yOutBufSize);

        int64_t scaleBufSize = ops::CeilAlign(
            static_cast<int64_t>(tilingData_->mUbFactor * tilingData_->nMxblockNumAlignedTwo * sizeof(T_X)),
            UB_BLOCK_SIZE);

        if (tilingData_->needPadScale) {
            pipe_->InitBuffer(scaleOutTmpBuffer_, scaleBufSize);
        }
        pipe_->InitBuffer(scaleOutQueue_, DOUBLE_BUFFER, scaleBufSize);
        pipe_->InitBuffer(maxExpBuffer_, scaleBufSize);
        pipe_->InitBuffer(maxhalfScaleBuffer_, scaleBufSize);

        if (tilingData_->needPadN) {
            int64_t yOutTmpBufSize = ops::CeilAlign(
                static_cast<int64_t>(tilingData_->mUbFactor * tilingData_->nMxblockAligned * sizeof(uint8_t)),
                UB_BLOCK_SIZE);
            pipe_->InitBuffer(yOutTmpBuffer_, yOutTmpBufSize);
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
        int64_t xBlockOffset = (blockIdx_ * tilingData_->mPerCore + usedTailCores) * tilingData_->numN;
        int64_t rstdBlockOffset = blockIdx_ * tilingData_->mPerCore + usedTailCores;
        int64_t scaleBlockOffset =
            (blockIdx_ * tilingData_->mPerCore + usedTailCores) * tilingData_->nMxblockNumAlignedTwo;

        int64_t curM = tilingData_->mPerCore;
        curM = blockIdx_ < tilingData_->mTailCores ? (curM + 1) : curM;

        int64_t curMLoop = ops::CeilDiv(curM, tilingData_->mUbFactor);
        int64_t mUbTail = curM - (curMLoop - 1) * tilingData_->mUbFactor;

        for (int64_t i = 0; i < curMLoop; i++) {
            int64_t curM = (i == (curMLoop - 1)) ? mUbTail : tilingData_->mUbFactor;
            int64_t xOffset = xBlockOffset + i * tilingData_->mUbFactor * tilingData_->numN;
            int64_t rstdOffset = rstdBlockOffset + i * tilingData_->mUbFactor;
            int64_t scaleOffset = scaleBlockOffset + i * tilingData_->mUbFactor * tilingData_->nMxblockNumAlignedTwo;
            CopyInX(xOffset, curM);
            ComputeRmsNormAndQuant(curM, rstdOffset, xOffset, scaleOffset);
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
        int64_t curM, int64_t rstdOffset, int64_t xOffset, int64_t scaleOffset)
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

        LocalTensor<T_X> yLocal = normOutBuffer_.Get<T_X>();
        if (tilingData_->needPadN) {
            if (tilingData_->hasInputBeta) {
                ComputeYWithPad<true>(xLocal, gammaLocal_, betaLocal_, rstdLocal, yLocal, curM);
            } else {
                ComputeYWithPad<false>(xLocal, gammaLocal_, betaLocal_, rstdLocal, yLocal, curM);
            }
        } else {
            if (tilingData_->hasInputBeta) {
                ComputeY<true>(xLocal, gammaLocal_, betaLocal_, rstdLocal, yLocal, curM);
            } else {
                ComputeY<false>(xLocal, gammaLocal_, betaLocal_, rstdLocal, yLocal, curM);
            }
        }

#if (__NPU_ARCH__ == 3510)
        AscendC::SetCtrlSpr<FLOAT_OVERFLOW_MODE_CTRL, FLOAT_OVERFLOW_MODE_CTRL>(0);
#endif
        if (tilingData_->roundMode == MODE_RINT) {
            ComputeMxQuantAndOutput<RoundMode::CAST_TRUNC, RoundMode::CAST_RINT>(yLocal, curM, xOffset, scaleOffset);
        } else if (tilingData_->roundMode == MODE_ROUND) {
            ComputeMxQuantAndOutput<RoundMode::CAST_TRUNC, RoundMode::CAST_ROUND>(yLocal, curM, xOffset, scaleOffset);
        } else if (tilingData_->roundMode == MODE_FLOOR) {
            ComputeMxQuantAndOutput<RoundMode::CAST_FLOOR, RoundMode::CAST_FLOOR>(yLocal, curM, xOffset, scaleOffset);
        }
#if (__NPU_ARCH__ == 3510)
        AscendC::SetCtrlSpr<FLOAT_OVERFLOW_MODE_CTRL, FLOAT_OVERFLOW_MODE_CTRL>(oriOverflowMode_);
#endif
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
                tilingData_->numNUbAligned, tilingData_->binAddFoldPoint);
        } else {
            CalculateSquareReduceSumRCommon<NUM_TWO>(
                xLocalAddr, xTmpLocalUbAddr, xReduceTmpLocalUbAddr, curUbFactor, tilingData_->numN,
                tilingData_->numNUbAligned, tilingData_->binAddFoldPoint);
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
                LoadTensorForDtypeT(xLocalAddr, xReg, pregLoop, (i * colNumAlign));
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
                LoadTensorForDtypeT(xLocalAddr, xReg, pregFull, (i * colNumAlign));
                Mul(squareReg, xReg, xReg, pregFull);
                LoadTensorForDtypeT(xLocalAddr + VL_FP32, xTailReg, pregTail, (i * colNumAlign));
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
        uint64_t curUbFactor, uint64_t numCol, uint64_t numColAlign, uint64_t binAddFoldPoint)
    {
        uint32_t colNum = static_cast<uint32_t>(numCol);
        uint32_t colNumAlign = static_cast<uint32_t>(numColAlign);
        uint32_t colFlodNum = static_cast<uint32_t>(binAddFoldPoint);
        uint16_t curAloops = static_cast<uint16_t>(curUbFactor);

        // first flod
        uint32_t firstFlodTial = static_cast<uint32_t>(colNum - binAddFoldPoint);
        uint16_t firstFlodAddLoops = static_cast<uint16_t>((firstFlodTial + VL_FP32 - 1) / VL_FP32);
        uint16_t firstFlodWithOutAddLoops =
            static_cast<uint16_t>((colFlodNum + VL_FP32 - 1) / VL_FP32) - firstFlodAddLoops;

        // first vcadd
        uint32_t firstVcaddNum = static_cast<uint32_t>((binAddFoldPoint + VL_FP32 - 1) / VL_FP32);
        uint32_t firstVcaddNumCeilAlign =
            static_cast<uint32_t>((firstVcaddNum + UB_BLOCK_SIZE_FP32 - 1) / UB_BLOCK_SIZE_FP32 * UB_BLOCK_SIZE_FP32);

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
                    LoadTensorForDtypeT(xLocalAddr, xReg1, pregFull, (i * colNumAlign + j * VL_FP32));
                    Mul(squareReg1, xReg1, xReg1, pregFull);
                    LoadTensorForDtypeT(xLocalAddr + colFlodNum, xReg2, pregFull, (i * colNumAlign + j * VL_FP32));
                    Mul(squareReg2, xReg2, xReg2, pregLoop);
                    Add(addReg, squareReg1, squareReg2, pregFull);
                    ReduceSum(sumReg, addReg, pregFull);
                    DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(
                        xTmpLocalUbAddr + static_cast<uint32_t>(i * firstVcaddNumCeilAlign + j), sumReg, pregOne);
                }
                for (uint16_t j = 0; j < static_cast<uint16_t>(firstFlodWithOutAddLoops); j++) {
                    LoadTensorForDtypeT(
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
            } else if constexpr (LAST_LOOP_NUMS == NUM_TWO) {
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
        uint32_t vl_fp32 = VL_FP32;
        uint32_t nNum = static_cast<uint32_t>(tilingData_->numN);
        uint16_t mloops = static_cast<uint16_t>(curM);
        uint16_t nloops = static_cast<uint16_t>(ops::CeilDiv(nNum, VL_FP32));
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
        uint16_t nloops = static_cast<uint16_t>(nNum / VL_FP32);
        uint32_t nTail = nNum - nloops * VL_FP32;
        uint32_t padNum = tilingData_->nMxblockAligned - tilingData_->numN;
        uint32_t lastLoopN = nTail + padNum; // <= 32
        uint32_t xInputStride = static_cast<uint32_t>(tilingData_->numNUbAligned);
        uint32_t yOutputStride = static_cast<uint32_t>(tilingData_->nMxblockAligned);
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
                Mul(yRegLast, xRegLast, RstdReg, pregTailMask);
                LoadTensorForDtypeT<T_GAMMA>(gammaLocalUbAddr, gammaRegLast, pregTailMask, gammaLastOffset);
                // 补pad 0
                Mul<float, MaskMergeMode::ZEROING>(yRegLast, yRegLast, gammaRegLast, pregTailMask);
                if constexpr (hasInputBeta) {
                    LoadTensorForDtypeT<T_GAMMA>(betaLocalUbAddr, betaRegLast, pregTailMask, gammaLastOffset);
                    Add<float, MaskMergeMode::ZEROING>(yRegLast, yRegLast, betaRegLast, pregTailMask);
                }

                StoreTensorForDtypeT(yLocalUbAddr, yRegLast, pregLastLoopMask, yLastOffset);
            }
        }
    }

    template <AscendC::RoundMode toBf16RoundMode, AscendC::RoundMode roundMode>
    __aicore__ inline void ComputeMxQuantAndOutput(
        LocalTensor<T_X>& yLocal, int64_t curM, int64_t xOffset, int64_t scaleOutOffset)
    {
        LocalTensor<uint16_t> scaleOutLocal = scaleOutQueue_.AllocTensor<uint16_t>();
        LocalTensor<int8_t> yOutLocal = yOutQueue_.AllocTensor<int8_t>();

        if constexpr (IsSameType<T_Y, fp4x2_e2m1_t>::value || IsSameType<T_Y, fp4x2_e1m2_t>::value) {
            ComputeFp4Mxquant<toBf16RoundMode, roundMode>(yLocal, curM, xOffset, scaleOutLocal, yOutLocal);
        } else {
            ComputeFp8Mxquant<toBf16RoundMode, roundMode>(yLocal, curM, xOffset, scaleOutLocal, yOutLocal);
        }

        yOutQueue_.EnQue(yOutLocal);
        scaleOutQueue_.EnQue(scaleOutLocal);

        CopyOutScaleAndY(xOffset, scaleOutOffset, curM);
    }

    template <AscendC::RoundMode toBf16RoundMode, AscendC::RoundMode roundMode>
    __aicore__ inline void ComputeFp4Mxquant(
        LocalTensor<T_X>& yLocal, int64_t curM, int64_t xOffset, LocalTensor<uint16_t>& scaleOutLocal,
        LocalTensor<int8_t>& yOutLocal)
    {
        uint32_t totalScaleInUB = curM * tilingData_->nMxblockNum;
        uint32_t totalCountInUB = curM * tilingData_->nMxblockNum * tilingData_->mxBlockSize;
        uint16_t loopNum = (totalCountInUB + VL_B16 * DIGIT_TWO - 1) / (VL_B16 * DIGIT_TWO);
        uint16_t loopNumScale = (totalScaleInUB + VL_B16 - 1) / VL_B16;

        LocalTensor<uint16_t> maxExpLocal = maxExpBuffer_.Get<uint16_t>();
        LocalTensor<uint16_t> halfScaleLocal = maxhalfScaleBuffer_.Get<uint16_t>();
        LocalTensor<uint16_t> padScaleLocal = scaleOutLocal;
        if (tilingData_->needPadScale) {
            padScaleLocal = scaleOutTmpBuffer_.Get<uint16_t>();
        }

        auto srcAddr = reinterpret_cast<__ubuf__ T_X*>(yLocal.GetPhyAddr());
        auto maxExpAddr = reinterpret_cast<__ubuf__ uint16_t*>(maxExpLocal.GetPhyAddr());
        auto scaleLocalAddr = reinterpret_cast<__ubuf__ uint16_t*>(scaleOutLocal.GetPhyAddr());
        auto halfScaleLocalAddr = reinterpret_cast<__ubuf__ uint16_t*>(halfScaleLocal.GetPhyAddr());
        auto padscaleLocalAddr = reinterpret_cast<__ubuf__ uint16_t*>(padScaleLocal.GetPhyAddr());

        ComputeMaxExpOCP<T_X>(srcAddr, maxExpAddr, totalCountInUB, loopNum);
        ComputeScaleOCP<T_X, T_Y>(maxExpAddr, padscaleLocalAddr, halfScaleLocalAddr, totalScaleInUB, loopNumScale);

        if (tilingData_->needPadScale) {
            auto tmpLocal = reinterpret_cast<__ubuf__ int8_t*>(padScaleLocal.GetPhyAddr());
            auto dstLocal = reinterpret_cast<__ubuf__ int8_t*>(scaleOutLocal.GetPhyAddr());
            PadScaleData(dstLocal, tmpLocal, curM, tilingData_->nMxblockNum);
            PipeBarrier<PIPE_V>();
        }

        auto outLocalAddr = reinterpret_cast<__ubuf__ int8_t*>(yOutLocal.GetPhyAddr());

        if (tilingData_->needPadN) {
            LocalTensor<int8_t> outBufferLocal = yOutTmpBuffer_.Get<int8_t>();
            auto outBufferLocalAddr = reinterpret_cast<__ubuf__ int8_t*>(outBufferLocal.GetPhyAddr());
            if constexpr (isOptimizeMode) {
                ComputeDataMxfp4Optimize<toBf16RoundMode, roundMode>(
                    srcAddr, halfScaleLocalAddr, outBufferLocalAddr, totalCountInUB, loopNum);
            } else {
                ComputeDataMxfp4General<toBf16RoundMode, roundMode>(
                    srcAddr, halfScaleLocalAddr, outBufferLocalAddr, totalCountInUB, loopNum);
            }

            uint32_t inputStride = tilingData_->nMxblockNum * tilingData_->mxBlockSize;
            DeletePadData<T_Y>(outLocalAddr, outBufferLocalAddr, curM, tilingData_->numN, inputStride);
        } else {
            if constexpr (isOptimizeMode) {
                ComputeDataMxfp4Optimize<toBf16RoundMode, roundMode>(
                    srcAddr, halfScaleLocalAddr, outLocalAddr, totalCountInUB, loopNum);
            } else {
                ComputeDataMxfp4General<toBf16RoundMode, roundMode>(
                    srcAddr, halfScaleLocalAddr, outLocalAddr, totalCountInUB, loopNum);
            }
        }
    }

    template <AscendC::RoundMode toBf16RoundMode, AscendC::RoundMode roundMode>
    __aicore__ inline void ComputeFp8Mxquant(
        LocalTensor<T_X>& yLocal, int64_t curM, int64_t xOffset, LocalTensor<uint16_t>& scaleOutLocal,
        LocalTensor<int8_t>& yOutLocal)
    {
        uint32_t totalScaleInUB = curM * tilingData_->nMxblockNum;
        uint32_t totalCountInUB = curM * tilingData_->nMxblockNum * tilingData_->mxBlockSize;
        uint16_t loopNum = (totalCountInUB + VL_B16 * DIGIT_TWO - 1) / (VL_B16 * DIGIT_TWO);
        uint16_t loopNumScale = (totalScaleInUB + VL_B16 - 1) / VL_B16;
        uint16_t loopNumScale4NV = (totalScaleInUB + VL_FP32 - 1) / VL_FP32;

        LocalTensor<uint16_t> maxExpLocal = maxExpBuffer_.Get<uint16_t>();
        LocalTensor<uint16_t> padScaleLocal = scaleOutLocal;
        if (tilingData_->needPadScale) {
            padScaleLocal = scaleOutTmpBuffer_.Get<uint16_t>();
        }
        LocalTensor<uint16_t> halfScaleLocal = maxhalfScaleBuffer_.Get<uint16_t>();

        auto srcAddr = reinterpret_cast<__ubuf__ T_X*>(yLocal.GetPhyAddr());
        auto maxExpAddr = reinterpret_cast<__ubuf__ uint16_t*>(maxExpLocal.GetPhyAddr());
        auto scaleLocalAddr = reinterpret_cast<__ubuf__ uint16_t*>(scaleOutLocal.GetPhyAddr());
        auto padscaleLocalAddr = reinterpret_cast<__ubuf__ uint16_t*>(padScaleLocal.GetPhyAddr());
        auto halfScaleLocalAddr = reinterpret_cast<__ubuf__ uint16_t*>(halfScaleLocal.GetPhyAddr());

        if (tilingData_->scaleAlg == 0) {
            ComputeMaxExpOCP<T_X>(srcAddr, maxExpAddr, totalCountInUB, loopNum);
            ComputeScaleOCP<T_X, T_Y>(maxExpAddr, padscaleLocalAddr, halfScaleLocalAddr, totalScaleInUB, loopNumScale);
        } else {
            ComputeMaxExpcuBLAS<T_X>(srcAddr, maxExpAddr, totalCountInUB, loopNum);
            ComputeScalecuBLAS<T_X, T_Y>(
                maxExpAddr, padscaleLocalAddr, halfScaleLocalAddr, totalScaleInUB, loopNumScale4NV);
        }

        if (tilingData_->needPadScale) {
            auto tmpLocal = reinterpret_cast<__ubuf__ int8_t*>(padScaleLocal.GetPhyAddr());
            auto dstLocal = reinterpret_cast<__ubuf__ int8_t*>(scaleOutLocal.GetPhyAddr());
            PadScaleData(dstLocal, tmpLocal, curM, tilingData_->nMxblockNum);
            PipeBarrier<PIPE_V>();
        }

        auto outLocalAddr = reinterpret_cast<__ubuf__ int8_t*>(yOutLocal.GetPhyAddr());

        if (tilingData_->needPadN) {
            LocalTensor<int8_t> outBufferLocal = yOutTmpBuffer_.Get<int8_t>();
            auto outBufferLocalAddr = reinterpret_cast<__ubuf__ int8_t*>(outBufferLocal.GetPhyAddr());
            ComputeData<toBf16RoundMode, roundMode, T_X, T_Y>(
                srcAddr, halfScaleLocalAddr, outBufferLocalAddr, totalCountInUB, loopNum);
            outBufferLocalAddr = reinterpret_cast<__ubuf__ int8_t*>(outBufferLocal.GetPhyAddr());
            uint32_t inputStride = tilingData_->nMxblockNum * tilingData_->mxBlockSize;
            DeletePadData<T_Y>(outLocalAddr, outBufferLocalAddr, curM, tilingData_->numN, inputStride);

        } else {
            ComputeData<toBf16RoundMode, roundMode, T_X, T_Y>(
                srcAddr, halfScaleLocalAddr, outLocalAddr, totalCountInUB, loopNum);
        }
    }

    __aicore__ inline void CopyOutScaleAndY(int64_t xOffset, int64_t scaleOutOffset, int64_t curM)
    {
        LocalTensor<uint8_t> outLocal = yOutQueue_.DeQue<uint8_t>();
        DataCopyExtParams copyOutParamData = {0, 0, 0, 0, 0};
        copyOutParamData.blockCount = 1;
        copyOutParamData.blockLen = curM * tilingData_->numN;
        copyOutParamData.srcStride = 0;
        copyOutParamData.dstStride = 0;
        int64_t yOffset = xOffset;
        if constexpr (IsSameType<T_Y, fp4x2_e2m1_t>::value || IsSameType<T_Y, fp4x2_e1m2_t>::value) {
            yOffset = yOffset / NUM_TWO;
            copyOutParamData.blockLen = copyOutParamData.blockLen / NUM_TWO;
        }
        DataCopyPad<uint8_t, PaddingMode::Compact>(yGm_[yOffset], outLocal, copyOutParamData);
        yOutQueue_.FreeTensor(outLocal);

        LocalTensor<uint8_t> scaleLocal = scaleOutQueue_.DeQue<uint8_t>();
        DataCopyExtParams copyOutParamScale = {0, 0, 0, 0, 0};
        copyOutParamScale.blockCount = 1;
        copyOutParamScale.blockLen = curM * tilingData_->nMxblockNumAlignedTwo;
        copyOutParamScale.srcStride = 0;
        copyOutParamScale.dstStride = 0;
        DataCopyPad<uint8_t, PaddingMode::Compact>(mxScaleGm_[scaleOutOffset], scaleLocal, copyOutParamScale);
        scaleOutQueue_.FreeTensor(scaleLocal);
    }

    template <AscendC::RoundMode toBf16RoundMode, AscendC::RoundMode roundMode>
    __aicore__ inline void ComputeDataMxfp4General(
        __ubuf__ T_X* srcAddr, __ubuf__ uint16_t* halfScaleLocalAddr, __ubuf__ int8_t* outLocalAddr,
        uint32_t totalCountInUB, uint16_t loopNum)
    {
        __VEC_SCOPE__
        {
            AscendC::MicroAPI::MaskReg dataMask1;
            AscendC::MicroAPI::RegTensor<uint16_t> halfScaleForMul;
            AscendC::MicroAPI::RegTensor<T_X> vdExp0;
            AscendC::MicroAPI::RegTensor<T_X> vdExp1;
            AscendC::MicroAPI::RegTensor<T_X> vdExp0Convert;
            AscendC::MicroAPI::RegTensor<T_X> vdExp1Convert;
            MicroAPI::RegTensor<float> halfScaleForMulFP32;
            MicroAPI::RegTensor<float> vdExp0ZeroFP32;
            MicroAPI::RegTensor<float> vdExp0OneFP32;
            MicroAPI::RegTensor<float> vdExp1ZeroFP32;
            MicroAPI::RegTensor<float> vdExp1OneFP32;
            MicroAPI::RegTensor<bfloat16_t> vdExp0ZeroBF16;
            MicroAPI::RegTensor<bfloat16_t> vdExp0OneBF16;
            MicroAPI::RegTensor<bfloat16_t> vdExp1ZeroBF16;
            MicroAPI::RegTensor<bfloat16_t> vdExp1OneBF16;

            AscendC::MicroAPI::RegTensor<bfloat16_t> vdExp0BF16;
            AscendC::MicroAPI::RegTensor<bfloat16_t> vdExp1BF16;

            AscendC::MicroAPI::RegTensor<T_Y> vdExp0FP4;
            AscendC::MicroAPI::RegTensor<T_Y> vdExp1FP4;

            AscendC::MicroAPI::RegTensor<bfloat16_t> vdBF16Exp0FP4;
            AscendC::MicroAPI::RegTensor<bfloat16_t> vdBF16Exp1FP4;

            MicroAPI::MaskReg dataMaskB16 = MicroAPI::CreateMask<half>();
            MicroAPI::MaskReg dataMaskB32 = MicroAPI::CreateMask<float>();

            static constexpr AscendC::MicroAPI::CastTrait castTrait = {
                AscendC::MicroAPI::RegLayout::ZERO, AscendC::MicroAPI::SatMode::UNKNOWN,
                AscendC::MicroAPI::MaskMergeMode::ZEROING, roundMode};
            static constexpr AscendC::MicroAPI::CastTrait castTraitHalf2Bf16 = {
                AscendC::MicroAPI::RegLayout::UNKNOWN, AscendC::MicroAPI::SatMode::UNKNOWN,
                AscendC::MicroAPI::MaskMergeMode::ZEROING, toBf16RoundMode};
            static constexpr MicroAPI::CastTrait castTraitF16toFp32Zero = {
                MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::UNKNOWN, MicroAPI::MaskMergeMode::ZEROING,
                RoundMode::UNKNOWN};
            static constexpr MicroAPI::CastTrait castTraitF16toFp32One = {
                MicroAPI::RegLayout::ONE, MicroAPI::SatMode::UNKNOWN, MicroAPI::MaskMergeMode::ZEROING,
                RoundMode::UNKNOWN};
            static constexpr MicroAPI::CastTrait castTraitFp32toBF16 = {
                MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::NO_SAT, MicroAPI::MaskMergeMode::ZEROING, roundMode};
            for (uint16_t i = 0; i < loopNum; i++) {
                dataMask1 = AscendC::MicroAPI::UpdateMask<T_X>(totalCountInUB);
                AscendC::MicroAPI::DataCopy<
                    T_X, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE,
                    AscendC::MicroAPI::LoadDist::DIST_DINTLV_B16>(vdExp0, vdExp1, srcAddr, VL_B16 * DIGIT_TWO);
                AscendC::MicroAPI::DataCopy<
                    uint16_t, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE,
                    AscendC::MicroAPI::LoadDist::DIST_E2B_B16>(
                    halfScaleForMul, halfScaleLocalAddr, elementAfterReduce_);

                if constexpr (IsSameType<T_X, half>::value) {
                    MicroAPI::Cast<float, bfloat16_t, castTraitF16toFp32Zero>(
                        halfScaleForMulFP32, (MicroAPI::RegTensor<bfloat16_t>&)halfScaleForMul, dataMaskB16);

                    MicroAPI::Cast<float, T_X, castTraitF16toFp32Zero>(vdExp0ZeroFP32, vdExp0, dataMaskB16);
                    MicroAPI::Cast<float, T_X, castTraitF16toFp32One>(vdExp0OneFP32, vdExp0, dataMaskB16);

                    MicroAPI::Mul(vdExp0ZeroFP32, vdExp0ZeroFP32, halfScaleForMulFP32, dataMaskB32);
                    MicroAPI::Mul(vdExp0OneFP32, vdExp0OneFP32, halfScaleForMulFP32, dataMaskB32);
                    ComputeFP4FromHalf<toBf16RoundMode, roundMode, T_Y>(vdExp0ZeroFP32);
                    ComputeFP4FromHalf<toBf16RoundMode, roundMode, T_Y>(vdExp0OneFP32);
                    AscendC::MicroAPI::Cast<bfloat16_t, float, castTraitFp32toBF16>(
                        vdExp0ZeroBF16, vdExp0ZeroFP32, dataMaskB32);
                    AscendC::MicroAPI::Cast<bfloat16_t, float, castTraitFp32toBF16>(
                        vdExp0OneBF16, vdExp0OneFP32, dataMaskB32);
                    MicroAPI::Pack<uint16_t, uint32_t, MicroAPI::HighLowPart::LOWEST>(
                        (MicroAPI::RegTensor<uint16_t>&)vdExp0ZeroBF16, (MicroAPI::RegTensor<uint32_t>&)vdExp0ZeroBF16);
                    MicroAPI::Pack<uint16_t, uint32_t, MicroAPI::HighLowPart::LOWEST>(
                        (MicroAPI::RegTensor<uint16_t>&)vdExp0OneBF16, (MicroAPI::RegTensor<uint32_t>&)vdExp0OneBF16);
                    MicroAPI::Interleave(vdExp0ZeroBF16, vdExp0OneBF16, vdExp0ZeroBF16, vdExp0OneBF16);

                    MicroAPI::Cast<float, T_X, castTraitF16toFp32Zero>(vdExp1ZeroFP32, vdExp1, dataMaskB16);
                    MicroAPI::Cast<float, T_X, castTraitF16toFp32One>(vdExp1OneFP32, vdExp1, dataMaskB16);

                    MicroAPI::Mul(vdExp1ZeroFP32, vdExp1ZeroFP32, halfScaleForMulFP32, dataMaskB32);
                    MicroAPI::Mul(vdExp1OneFP32, vdExp1OneFP32, halfScaleForMulFP32, dataMaskB32);
                    ComputeFP4FromHalf<toBf16RoundMode, roundMode, T_Y>(vdExp1ZeroFP32);
                    ComputeFP4FromHalf<toBf16RoundMode, roundMode, T_Y>(vdExp1OneFP32);
                    AscendC::MicroAPI::Cast<bfloat16_t, float, castTraitFp32toBF16>(
                        vdExp1ZeroBF16, vdExp1ZeroFP32, dataMaskB32);
                    AscendC::MicroAPI::Cast<bfloat16_t, float, castTraitFp32toBF16>(
                        vdExp1OneBF16, vdExp1OneFP32, dataMaskB32);
                    MicroAPI::Pack<uint16_t, uint32_t, MicroAPI::HighLowPart::LOWEST>(
                        (MicroAPI::RegTensor<uint16_t>&)vdExp1ZeroBF16, (MicroAPI::RegTensor<uint32_t>&)vdExp1ZeroBF16);
                    MicroAPI::Pack<uint16_t, uint32_t, MicroAPI::HighLowPart::LOWEST>(
                        (MicroAPI::RegTensor<uint16_t>&)vdExp1OneBF16, (MicroAPI::RegTensor<uint32_t>&)vdExp1OneBF16);
                    MicroAPI::Interleave(vdExp1ZeroBF16, vdExp1OneBF16, vdExp1ZeroBF16, vdExp1OneBF16);

                    AscendC::MicroAPI::Interleave(vdExp0ZeroBF16, vdExp1ZeroBF16, vdExp0ZeroBF16, vdExp1ZeroBF16);
                    AscendC::MicroAPI::Cast<T_Y, bfloat16_t, castTrait>(vdExp0FP4, vdExp0ZeroBF16, dataMask1);
                    AscendC::MicroAPI::Cast<T_Y, bfloat16_t, castTrait>(vdExp1FP4, vdExp1ZeroBF16, dataMask1);
                } else {
                    AscendC::MicroAPI::Mul(
                        vdExp0, vdExp0, (AscendC::MicroAPI::RegTensor<T_X>&)halfScaleForMul, dataMask1);
                    AscendC::MicroAPI::Mul(
                        vdExp1, vdExp1, (AscendC::MicroAPI::RegTensor<T_X>&)halfScaleForMul, dataMask1);
                    AscendC::MicroAPI::Interleave(vdExp0, vdExp1, vdExp0, vdExp1);
                    AscendC::MicroAPI::Cast<T_Y, T_X, castTrait>(vdExp0FP4, vdExp0, dataMask1);
                    AscendC::MicroAPI::Cast<T_Y, T_X, castTrait>(vdExp1FP4, vdExp1, dataMask1);
                }

                AscendC::MicroAPI::DataCopy<
                    int8_t, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE,
                    AscendC::MicroAPI::StoreDist::DIST_PACK4_B32>(
                    outLocalAddr, (AscendC::MicroAPI::RegTensor<int8_t>&)vdExp0FP4, OUT_ELE_NUM_ONE_BLK, dataMask1);
                AscendC::MicroAPI::DataCopy<
                    int8_t, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE,
                    AscendC::MicroAPI::StoreDist::DIST_PACK4_B32>(
                    outLocalAddr, (AscendC::MicroAPI::RegTensor<int8_t>&)vdExp1FP4, OUT_ELE_NUM_ONE_BLK, dataMask1);
            }
        }
        return;
    }

    template <AscendC::RoundMode toBf16RoundMode, AscendC::RoundMode roundMode>
    __aicore__ inline void ComputeDataMxfp4Optimize(
        __ubuf__ T_X* srcAddr, __ubuf__ uint16_t* halfScaleLocalAddr, __ubuf__ int8_t* outLocalAddr,
        uint32_t totalCountInUB, uint16_t loopNum)
    {
        __VEC_SCOPE__
        {
            AscendC::MicroAPI::MaskReg dataMask1;
            AscendC::MicroAPI::RegTensor<uint16_t> halfScaleForMul;
            AscendC::MicroAPI::RegTensor<T_X> vdExp0;
            AscendC::MicroAPI::RegTensor<T_X> vdExp1;
            AscendC::MicroAPI::RegTensor<T_X> vdExp0Convert;
            AscendC::MicroAPI::RegTensor<T_X> vdExp1Convert;

            AscendC::MicroAPI::RegTensor<bfloat16_t> vdExp0BF16;
            AscendC::MicroAPI::RegTensor<bfloat16_t> vdExp1BF16;

            AscendC::MicroAPI::RegTensor<T_Y> vdExp0FP4;
            AscendC::MicroAPI::RegTensor<T_Y> vdExp1FP4;

            AscendC::MicroAPI::RegTensor<bfloat16_t> vdBF16Exp0FP4;
            AscendC::MicroAPI::RegTensor<bfloat16_t> vdBF16Exp1FP4;
            static constexpr AscendC::MicroAPI::CastTrait castTrait = {
                AscendC::MicroAPI::RegLayout::ZERO, AscendC::MicroAPI::SatMode::UNKNOWN,
                AscendC::MicroAPI::MaskMergeMode::ZEROING, roundMode};
            static constexpr AscendC::MicroAPI::CastTrait castTraitHalf2Bf16 = {
                AscendC::MicroAPI::RegLayout::UNKNOWN, AscendC::MicroAPI::SatMode::UNKNOWN,
                AscendC::MicroAPI::MaskMergeMode::ZEROING, toBf16RoundMode};
            for (uint16_t i = 0; i < loopNum; i++) {
                dataMask1 = AscendC::MicroAPI::UpdateMask<T_X>(totalCountInUB);
                AscendC::MicroAPI::DataCopy<
                    T_X, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE,
                    AscendC::MicroAPI::LoadDist::DIST_DINTLV_B16>(vdExp0, vdExp1, srcAddr, VL_B16 * DIGIT_TWO);
                AscendC::MicroAPI::DataCopy<
                    uint16_t, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE,
                    AscendC::MicroAPI::LoadDist::DIST_E2B_B16>(
                    halfScaleForMul, halfScaleLocalAddr, elementAfterReduce_);

                if constexpr (IsSameType<T_X, half>::value) {
                    if constexpr (roundMode == RoundMode::CAST_RINT) {
                        FP16Convert<T_Y>(vdExp0, vdExp0, dataMask1);
                        FP16Convert<T_Y>(vdExp1, vdExp1, dataMask1);
                    }
                    AscendC::MicroAPI::Cast<bfloat16_t, T_X, castTraitHalf2Bf16>(vdExp0BF16, vdExp0, dataMask1);
                    AscendC::MicroAPI::Cast<bfloat16_t, T_X, castTraitHalf2Bf16>(vdExp1BF16, vdExp1, dataMask1);
                    AscendC::MicroAPI::Mul(
                        vdExp0BF16, vdExp0BF16, (AscendC::MicroAPI::RegTensor<bfloat16_t>&)halfScaleForMul, dataMask1);
                    AscendC::MicroAPI::Mul(
                        vdExp1BF16, vdExp1BF16, (AscendC::MicroAPI::RegTensor<bfloat16_t>&)halfScaleForMul, dataMask1);
                    AscendC::MicroAPI::Interleave(vdExp0BF16, vdExp1BF16, vdExp0BF16, vdExp1BF16);
                    AscendC::MicroAPI::Cast<T_Y, bfloat16_t, castTrait>(vdExp0FP4, vdExp0BF16, dataMask1);
                    AscendC::MicroAPI::Cast<T_Y, bfloat16_t, castTrait>(vdExp1FP4, vdExp1BF16, dataMask1);

                } else {
                    AscendC::MicroAPI::Mul(
                        vdExp0, vdExp0, (AscendC::MicroAPI::RegTensor<T_X>&)halfScaleForMul, dataMask1);
                    AscendC::MicroAPI::Mul(
                        vdExp1, vdExp1, (AscendC::MicroAPI::RegTensor<T_X>&)halfScaleForMul, dataMask1);
                    AscendC::MicroAPI::Interleave(vdExp0, vdExp1, vdExp0, vdExp1);
                    AscendC::MicroAPI::Cast<T_Y, T_X, castTrait>(vdExp0FP4, vdExp0, dataMask1);
                    AscendC::MicroAPI::Cast<T_Y, T_X, castTrait>(vdExp1FP4, vdExp1, dataMask1);
                }

                AscendC::MicroAPI::DataCopy<
                    int8_t, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE,
                    AscendC::MicroAPI::StoreDist::DIST_PACK4_B32>(
                    outLocalAddr, (AscendC::MicroAPI::RegTensor<int8_t>&)vdExp0FP4, OUT_ELE_NUM_ONE_BLK, dataMask1);
                AscendC::MicroAPI::DataCopy<
                    int8_t, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE,
                    AscendC::MicroAPI::StoreDist::DIST_PACK4_B32>(
                    outLocalAddr, (AscendC::MicroAPI::RegTensor<int8_t>&)vdExp1FP4, OUT_ELE_NUM_ONE_BLK, dataMask1);
            }
        }
        return;
    }

private:
    TPipe* pipe_;
    const RmsNormDynamicMxQuantFullLoadTilingData* tilingData_;
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
    TQue<QuePosition::VECOUT, DOUBLE_BUFFER> scaleOutQueue_;

    TBuf<QuePosition::VECCALC> xReduceTmpBuffer_;
    TBuf<QuePosition::VECCALC> xReduceOutBuffer_;
    TBuf<QuePosition::VECCALC> normOutBuffer_;

    TBuf<QuePosition::VECCALC> maxExpBuffer_;
    TBuf<QuePosition::VECCALC> maxhalfScaleBuffer_;
    TBuf<QuePosition::VECCALC> yOutTmpBuffer_;
    TBuf<QuePosition::VECCALC> scaleOutTmpBuffer_;

    int64_t blockIdx_ = 0;
    uint16_t elementAfterReduce_ = 0;
#if (__NPU_ARCH__ == 3510)
    int64_t oriOverflowMode_ = 0; // 存储原始SPR配置
#endif
};
} // namespace RmsNormDynamicMxQuantNs

#endif // RMS_NORM_DYNAMIC_MX_QUANT_FULL_LOAD_GENERAL_H

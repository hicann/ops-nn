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
 * \file add_rms_norm_dynamic_mx_quant_fp4_r_full_load.h
 * \brief
 */
#ifndef ADD_RMS_NORM_DYNAMIC_MX_QUANT_FP4_R_FULL_LOAD_H
#define ADD_RMS_NORM_DYNAMIC_MX_QUANT_FP4_R_FULL_LOAD_H
#define FLOAT_OVERFLOW_MODE_CTRL 60

#include "add_rms_norm_dynamic_mx_quant_common.h"

namespace AddRmsNormDynamicMxQuant {

template <typename T_X, typename T_GAMMA, typename T_Y>
class AddRmsNormDynamicMxQuantFP4RFullLoad {
public:
    __aicore__ inline AddRmsNormDynamicMxQuantFP4RFullLoad(TPipe* pipe)
    {
        pPipe = pipe;
    }

    __aicore__ inline void Init(
        GM_ADDR x1, GM_ADDR x2, GM_ADDR gamma, GM_ADDR beta, GM_ADDR y, GM_ADDR x, GM_ADDR mxscale, GM_ADDR workspace,
        GM_ADDR rstd, const AddRmsNormDynamicMxQuantTilingData* tiling)
    {
#if (__NPU_ARCH__ == 3510)
        AscendC::SetCtrlSpr<FLOAT_OVERFLOW_MODE_CTRL, FLOAT_OVERFLOW_MODE_CTRL>(0);
#endif
        ASSERT(GetBlockNum() != 0 && "Block dim can not be zero!");

        numRow_ = tiling->numRow;
        numCol_ = tiling->numCol;
        blockFactor_ = tiling->blockFactor;
        binAddQuotient_ = tiling->binAddQuotient;
        rowFactor_ = tiling->rowFactor;
        epsilon_ = tiling->epsilon;
        numColAlign_ = tiling->numColAlign;
        avgFactor_ = tiling->avgFactor;
        rowWork = (GetBlockIdx() < GetBlockNum() - 1) ? blockFactor_ : numRow_ - (GetBlockNum() - 1) * blockFactor_;
        roundMode_ = tiling->roundMode;
        mxBlockSize_ = tiling->mxBlockSize;
        scaleAlg_ = tiling->scaleAlg;
        blockNumInColAxis_ = tiling->blockNumInColAxis;
        dstStrideUbBlocks_ = tiling->dstStrideUbBlocks;
        mxScaleSize_ = tiling->mxScaleSize;
        betaFlag_ = tiling->betaFlag;
        rstdFlag_ = tiling->rstdFlag;

        vlForHalfNumber = platform::GetVRegSize() / sizeof(T_X);
        elementAfterReduce = platform::GetVRegSize() / UB_BLOCK_SIZE; 

        if constexpr (IsSame<T_Y, fp4x2_e2m1_t>::value) {
            f4Emax_ = FP4_E2M1_BF16_MAX_EXP;
        } else {
            f4Emax_ = FP4_E1M2_MAX_EXP;
        }

        uint64_t blockOffset = GetBlockIdx() * blockFactor_ * numCol_;
        x1Gm.SetGlobalBuffer((__gm__ T_X*)x1 + blockOffset, rowWork * numCol_);
        x2Gm.SetGlobalBuffer((__gm__ T_X*)x2 + blockOffset, rowWork * numCol_);
        gammaGm.SetGlobalBuffer((__gm__ T_GAMMA*)gamma, numCol_);
        if (betaFlag_ != 0) {
            betaGm.SetGlobalBuffer((__gm__ T_GAMMA*)beta, numCol_);
        }
        xOutGm.SetGlobalBuffer((__gm__ T_X*)x + blockOffset, rowWork * numCol_);
        if (rstdFlag_ != 0) {
            rstdGm.SetGlobalBuffer((__gm__ float*)rstd + GetBlockIdx() * blockFactor_, blockFactor_);
        }
        yFp4Gm.SetGlobalBuffer((__gm__ uint8_t*)y + blockOffset / DIGIT_TWO, rowWork * numCol_ / DIGIT_TWO);
        mxScaleGm.SetGlobalBuffer(
            (__gm__ uint8_t*)mxscale + GetBlockIdx() * blockFactor_ * mxScaleSize_, rowWork * mxScaleSize_);

        uint64_t rstdUbSizeAlignSize = CeilAlign(rowFactor_, static_cast<uint64_t>(VL_F32)) * sizeof(float);
        uint16_t binaryAddQuotientLoop = CeilDiv(binAddQuotient_, VL_F32);
        uint32_t binaryAddBufLen =
            CeilAlign(CeilAlign(binaryAddQuotientLoop, BLOCK_F32_ALIGN_NUM) * sizeof(float), UB_BLOCK_SIZE) *
            rowFactor_;

        uint64_t maxExpBufSize = CeilAlign(blockNumInColAxis_ * sizeof(T_X), UB_BLOCK_SIZE) * rowFactor_;
        uint64_t halfScaleBufSize = maxExpBufSize;

        uint64_t quantYBufSize = CeilAlign(CeilDiv(numColAlign_ * rowFactor_ / DIGIT_TWO, MX_STEP_PROCESS_NUM), DIGIT_FOUR) * MX_STEP_PROCESS_NUM;

        uint64_t scaleBufPerIter = CeilAlign(mxScaleSize_ * sizeof(T_X), UB_BLOCK_SIZE) * rowFactor_;

        pPipe->InitBuffer(
            inQueueX1, DOUBLE_BUFFER_NUM, CeilAlign(numColAlign_ * sizeof(T_X), UB_BLOCK_SIZE) * rowFactor_);
        pPipe->InitBuffer(
            inQueueX2, DOUBLE_BUFFER_NUM, CeilAlign(numColAlign_ * sizeof(T_X), UB_BLOCK_SIZE) * rowFactor_);
        if (betaFlag_ != 0) {
            pPipe->InitBuffer(
                inQueueGammabeta, 1,
                DIGIT_TWO * CeilAlign(numCol_, UB_BLOCK_SIZE / sizeof(T_GAMMA)) * sizeof(T_GAMMA));
        } else {
            pPipe->InitBuffer(
                inQueueGammabeta, 1, CeilAlign(numCol_, UB_BLOCK_SIZE / sizeof(T_GAMMA)) * sizeof(T_GAMMA));
        }
        pPipe->InitBuffer(
            outQueueX, DOUBLE_BUFFER_NUM, CeilAlign(numColAlign_ * sizeof(T_X), UB_BLOCK_SIZE) * rowFactor_);
        pPipe->InitBuffer(outQueueRstd, DOUBLE_BUFFER_NUM, rstdUbSizeAlignSize);
        pPipe->InitBuffer(xReduceBuff, rstdUbSizeAlignSize);
        pPipe->InitBuffer(xFp32Buff, CeilAlign(numColAlign_ * sizeof(float), UB_BLOCK_SIZE) * rowFactor_);
        pPipe->InitBuffer(binaryAddBuf, binaryAddBufLen);

        pPipe->InitBuffer(maxExpBuff, maxExpBufSize);
        pPipe->InitBuffer(halfScaleBuff, halfScaleBufSize);
        pPipe->InitBuffer(outQueueQuantY, DOUBLE_BUFFER_NUM, quantYBufSize);
        pPipe->InitBuffer(mxScaleQueue, DOUBLE_BUFFER_NUM, scaleBufPerIter);
    }
    __aicore__ inline void Process()
    {
        LocalTensor<uint8_t> gammabetaLocal = inQueueGammabeta.AllocTensor<uint8_t>();
        CopyInGammabeta(gammabetaLocal);
        inQueueGammabeta.EnQue(gammabetaLocal);
        inQueueGammabeta.DeQue<uint8_t>();

        uint32_t repeatTimes = CeilDiv(rowWork, rowFactor_);
        for (uint32_t repeat = 0; repeat < repeatTimes; repeat++) {
            uint64_t offset = repeat * rowFactor_ * numCol_;
            uint32_t curRows = Min(rowWork - repeat * rowFactor_, rowFactor_);
            Compute(repeat, curRows, offset);
        }
        inQueueGammabeta.FreeTensor(gammabetaLocal);
    }

private:
    __aicore__ inline void Compute(uint32_t rowRepeat, uint32_t curRows, uint64_t offset)
    {
        CopyInXMultiMoveAlign(offset, curRows);
        LocalTensor<T_X> xLocal1 = inQueueX1.DeQue<T_X>();
        LocalTensor<T_X> xLocal2 = inQueueX2.DeQue<T_X>();
        LocalTensor<T_X> xOutLocal = outQueueX.AllocTensor<T_X>();
        LocalTensor<float> xFp32Local = xFp32Buff.Get<float>();

        CalculateXAdd(xLocal1, xLocal2, xOutLocal, xFp32Local, curRows);
        inQueueX1.FreeTensor(xLocal1);
        inQueueX2.FreeTensor(xLocal2);
        outQueueX.EnQue<T_X>(xOutLocal);

        CopyOutX(offset, curRows);

        LocalTensor<float> rstdLocal = outQueueRstd.AllocTensor<float>();
        LocalTensor<float> xReduceLocal = xReduceBuff.Get<float>();
        CalculateSquareReduceSum(xFp32Local, xReduceLocal, curRows);

        CalculateRstd(xReduceLocal, rstdLocal, curRows);
        outQueueRstd.EnQue<float>(rstdLocal);

        rstdLocal = outQueueRstd.DeQue<float>();
        if (rstdFlag_ != 0) {
            DataCopyExtParams rstdCopyParams{1, static_cast<uint32_t>(curRows * sizeof(float)), 0, 0, 0};
            DataCopyPad(rstdGm[rowRepeat * rowFactor_], rstdLocal, rstdCopyParams);
        }

        LocalTensor<T_X> yLocal = outQueueX.AllocTensor<T_X>();
        if (numCol_ != numColAlign_) {
            Duplicate<T_X>(yLocal, static_cast<T_X>(0), curRows * numColAlign_);
            PipeBarrier<PIPE_V>();
        }
        if (betaFlag_ == 1) {
            CalculateY<true>(xFp32Local, yLocal, rstdLocal, curRows);
        } else {
            CalculateY<false>(xFp32Local, yLocal, rstdLocal, curRows);
        }
        outQueueRstd.FreeTensor(rstdLocal);
        outQueueX.EnQue<T_X>(yLocal);
        yLocal = outQueueX.DeQue<T_X>();

        if (roundMode_ == MODE_RINT) {
            DynamicMxQuantPhaseFP4<RoundMode::CAST_TRUNC, RoundMode::CAST_RINT>(yLocal, curRows);
        } else if (roundMode_ == MODE_ROUND) {
            DynamicMxQuantPhaseFP4<RoundMode::CAST_TRUNC, RoundMode::CAST_ROUND>(yLocal, curRows);
        } else if (roundMode_ == MODE_FLOOR) {
            DynamicMxQuantPhaseFP4<RoundMode::CAST_FLOOR, RoundMode::CAST_FLOOR>(yLocal, curRows);
        }
        outQueueX.FreeTensor(yLocal);

        CopyOutQuantYFP4(offset, curRows);
        CopyOutMxScale(rowRepeat, curRows);
    }

    __aicore__ inline void CalculateXAdd(
        LocalTensor<T_X>& xLocal1, LocalTensor<T_X>& xLocal2, LocalTensor<T_X>& xOutLocal,
        LocalTensor<float>& xFp32Local, uint32_t curRows)
    {
        __local_mem__ T_X* x1InUb = (__local_mem__ T_X*)xLocal1.GetPhyAddr();
        __local_mem__ T_X* x2InUb = (__local_mem__ T_X*)xLocal2.GetPhyAddr();
        __local_mem__ T_X* xOutInUb = (__local_mem__ T_X*)xOutLocal.GetPhyAddr();
        __local_mem__ float* xFp32Tmp = (__local_mem__ float*)xFp32Local.GetPhyAddr();

        uint32_t sreg = curRows * numColAlign_;
        uint16_t loopCount = (sreg + VL_F32 - 1) / VL_F32;

        __VEC_SCOPE__
        {
            RegTensor<float> x1;
            RegTensor<float> x2;
            RegTensor<float> xSum;
            MaskReg pregLoop;
            for (uint16_t i = 0; i < loopCount; ++i) {
                uint32_t offset = i * VL_F32;
                pregLoop = UpdateMask<float>(sreg);
                LoadTensorForDtypeTIn<T_X>(x1InUb, x1, pregLoop, offset);
                LoadTensorForDtypeTIn<T_X>(x2InUb, x2, pregLoop, offset);
                AscendC::MicroAPI::Add(xSum, x1, x2, pregLoop);
                StoreTensorForDtypeTOut<T_X>(xOutInUb, xSum, pregLoop, offset);
                AscendC::MicroAPI::DataCopy<float, StoreDist::DIST_NORM_B32>(xFp32Tmp + offset, xSum, pregLoop);
            }
        }
    }

    template <bool hasBeta>
    __aicore__ inline void CalculateY(
        LocalTensor<float>& xFp32Local, LocalTensor<T_X>& yLocal, LocalTensor<float>& rstdLocal, uint32_t curRows)
    {
        uint32_t numColAlign = static_cast<uint32_t>(numColAlign_);
        uint32_t numCol = static_cast<uint32_t>(numCol_);
        __local_mem__ float* xFp32Tmp = (__local_mem__ float*)xFp32Local.GetPhyAddr();
        __local_mem__ T_GAMMA* gammaInUb = (__local_mem__ T_GAMMA*)gammaLocal.GetPhyAddr();
        __local_mem__ T_X* yInUb = (__local_mem__ T_X*)yLocal.GetPhyAddr();
        __local_mem__ float* rstdInUb = (__local_mem__ float*)rstdLocal.GetPhyAddr();
        __local_mem__ T_GAMMA* betaInUb;
        if constexpr (hasBeta) {
            betaInUb = (__local_mem__ T_GAMMA*)betaLocal.GetPhyAddr();
        }

        uint16_t loopRows = static_cast<uint16_t>(curRows);
        uint16_t loopCols = static_cast<uint16_t>((numCol + VL_F32 - 1) / VL_F32);
        uint16_t loopRowsFold = loopRows / 2;
        uint16_t loopRowsHasLast = loopRows % 2;

        __VEC_SCOPE__
        {
            RegTensor<float> x1Reg, x2Reg, gammaReg, betaReg, rstd1Reg, rstd2Reg, mul1Reg, mul1UnrollReg, mul2Reg,
                mul2UnrollReg;

            for (uint16_t i = 0; i < loopRowsFold; ++i) {
                uint32_t sregCount = numCol;
                AscendC::MicroAPI::DataCopy<float, LoadDist::DIST_BRC_B32>(rstd1Reg, rstdInUb + 2 * i);
                AscendC::MicroAPI::DataCopy<float, LoadDist::DIST_BRC_B32>(rstd2Reg, rstdInUb + (2 * i + 1));
                for (uint16_t r = 0; r < loopCols; ++r) {
                    uint32_t offset1 = (2 * i) * numColAlign + r * VL_F32;
                    uint32_t offset2 = (2 * i + 1) * numColAlign + r * VL_F32;
                    MaskReg regCurLoop = UpdateMask<float>(sregCount);
                    LoadTensorForDtypeTIn<float>(xFp32Tmp, x1Reg, regCurLoop, offset1);
                    LoadTensorForDtypeTIn<float>(xFp32Tmp, x2Reg, regCurLoop, offset2);
                    AscendC::MicroAPI::Mul(mul1Reg, x1Reg, rstd1Reg, regCurLoop);
                    AscendC::MicroAPI::Mul(mul1UnrollReg, x2Reg, rstd2Reg, regCurLoop);
                    LoadTensorForDtypeTIn<T_GAMMA>(gammaInUb, gammaReg, regCurLoop, r * VL_F32);
                    AscendC::MicroAPI::Mul(mul2Reg, mul1Reg, gammaReg, regCurLoop);
                    AscendC::MicroAPI::Mul(mul2UnrollReg, mul1UnrollReg, gammaReg, regCurLoop);
                    if constexpr (hasBeta) {
                        LoadTensorForDtypeTIn<T_GAMMA>(betaInUb, betaReg, regCurLoop, r * VL_F32);
                        AscendC::MicroAPI::Add(mul2Reg, mul2Reg, betaReg, regCurLoop);
                        AscendC::MicroAPI::Add(mul2UnrollReg, mul2UnrollReg, betaReg, regCurLoop);
                    }
                    StoreTensorForDtypeTOut<T_X>(yInUb, mul2Reg, regCurLoop, offset1);
                    StoreTensorForDtypeTOut<T_X>(yInUb, mul2UnrollReg, regCurLoop, offset2);
                }
            }
            for (uint16_t i = 0; i < loopRowsHasLast; ++i) {
                uint32_t sregCount = numCol;
                AscendC::MicroAPI::DataCopy<float, LoadDist::DIST_BRC_B32>(rstd1Reg, rstdInUb + 2 * loopRowsFold);
                for (uint16_t r = 0; r < loopCols; ++r) {
                    uint32_t offset = (2 * loopRowsFold) * numColAlign + r * VL_F32;
                    MaskReg regCurLoop = UpdateMask<float>(sregCount);
                    LoadTensorForDtypeTIn<float>(xFp32Tmp, x1Reg, regCurLoop, offset);
                    AscendC::MicroAPI::Mul(mul1Reg, x1Reg, rstd1Reg, regCurLoop);
                    LoadTensorForDtypeTIn<T_GAMMA>(gammaInUb, gammaReg, regCurLoop, r * VL_F32);
                    AscendC::MicroAPI::Mul(mul2Reg, mul1Reg, gammaReg, regCurLoop);
                    if constexpr (hasBeta) {
                        LoadTensorForDtypeTIn<T_GAMMA>(betaInUb, betaReg, regCurLoop, r * VL_F32);
                        AscendC::MicroAPI::Add(mul2Reg, mul2Reg, betaReg, regCurLoop);
                    }
                    StoreTensorForDtypeTOut<T_X>(yInUb, mul2Reg, regCurLoop, offset);
                }
            }
        }
    }

    __aicore__ inline void CalculateSquareReduceSum(
        LocalTensor<float>& xFp32Local, LocalTensor<float>& xReduceLocal, uint32_t curRows)
    {
        LocalTensor<float> binaryAddBuffTmp = binaryAddBuf.Get<float>();
        __local_mem__ float* xReduceUb = (__local_mem__ float*)xReduceLocal.GetPhyAddr();
        __local_mem__ float* tmpUb = (__local_mem__ float*)binaryAddBuffTmp.GetPhyAddr();
        __local_mem__ float* xFp32Tmp = (__local_mem__ float*)xFp32Local.GetPhyAddr();

        if (numCol_ <= VL_F32) {
            CalculateSquareReduceSumLessThanVL(xFp32Tmp, xReduceUb, curRows);
        } else if (numCol_ <= VL_F32 + VL_F32) {
            CalculateSquareReduceSumLessThanTwoVL(xFp32Tmp, xReduceUb, curRows);
        } else if (numCol_ <= VL_F32 * VL_F32 * DIGIT_TWO) {
            CalculateSquareReduceSumCommon<DIGIT_ONE>(xFp32Tmp, xReduceUb, tmpUb, curRows);
        } else {
            CalculateSquareReduceSumCommon<DIGIT_TWO>(xFp32Tmp, xReduceUb, tmpUb, curRows);
        }
    }

    __aicore__ inline void CalculateSquareReduceSumLessThanVL(
        __local_mem__ float* xFp32Tmp, __local_mem__ float* xReduceUb, uint16_t curRows)
    {
        uint32_t numCol = static_cast<uint32_t>(numCol_);
        __VEC_SCOPE__
        {
            RegTensor<float> x;
            RegTensor<float> vMean;
            RegTensor<float> onesReg;

            uint32_t sreg0 = numCol;
            MaskReg pregLoop = UpdateMask<float>(sreg0);
            MaskReg pregOne = CreateMask<float, MaskPattern::VL1>();
            AscendC::MicroAPI::Duplicate(onesReg, float(1.0), pregOne);

            for (uint16_t i = 0; i < curRows; i++) {
                LoadTensorForDtypeTIn<float>(xFp32Tmp, x, pregLoop, i * numColAlign_);
                AscendC::MicroAPI::Mul(x, x, x, pregLoop);
                AscendC::MicroAPI::ReduceSum(vMean, x, pregLoop);
                AscendC::MicroAPI::DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(xReduceUb + i, vMean, pregOne);
            }
        }
    }

    __aicore__ inline void CalculateSquareReduceSumLessThanTwoVL(
        __local_mem__ float* xFp32Tmp, __local_mem__ float* xReduceUb, uint16_t curRows)
    {
        uint32_t tailLen = static_cast<uint32_t>(numCol_) - VL_F32;
        __VEC_SCOPE__
        {
            RegTensor<float> x;
            RegTensor<float> xFold;
            RegTensor<float> sumReg;
            RegTensor<float> vMean;
            RegTensor<float> onesReg;

            MaskReg pregFull = CreateMask<float, MaskPattern::ALL>();
            MaskReg pregOne = CreateMask<float, MaskPattern::VL1>();
            MaskReg pregTail = UpdateMask<float>(tailLen);
            AscendC::MicroAPI::Duplicate(onesReg, float(1.0), pregOne);

            for (uint16_t i = 0; i < curRows; ++i) {
                LoadTensorForDtypeTIn<float>(xFp32Tmp, x, pregFull, i * numColAlign_);
                LoadTensorForDtypeTIn<float>(xFp32Tmp + VL_F32, xFold, pregTail, i * numColAlign_);
                AscendC::MicroAPI::Mul(x, x, x, pregFull);
                AscendC::MicroAPI::Mul(xFold, xFold, xFold, pregTail);
                AscendC::MicroAPI::ShiftLefts(
                    (RegTensor<uint32_t>&)xFold, (RegTensor<uint32_t>&)xFold, static_cast<int16_t>(0), pregTail);
                AscendC::MicroAPI::Add(sumReg, x, xFold, pregFull);
                AscendC::MicroAPI::ReduceSum(vMean, sumReg, pregFull);
                AscendC::MicroAPI::DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(xReduceUb + i, vMean, pregOne);
            }
        }
    }

    template <int32_t LAST_LOOP_NUMS>
    __aicore__ inline void CalculateSquareReduceSumCommon(
        __local_mem__ float* xFp32Tmp, __local_mem__ float* xReduceUb, __local_mem__ float* tmpUb, uint16_t curRows)
    {
        uint32_t binaryAddQuotient = static_cast<uint32_t>(binAddQuotient_);
        uint16_t binaryAddQuotientLoop = (binaryAddQuotient + VL_F32 - 1) / VL_F32;
        uint32_t lastBinaryAddNum = binaryAddQuotient / VL_F32;
        uint32_t lastBinaryAddNumAlign =
            (binaryAddQuotientLoop + BLOCK_F32_ALIGN_NUM - 1) / BLOCK_F32_ALIGN_NUM * BLOCK_F32_ALIGN_NUM;
        uint32_t binaryAddRemainder = static_cast<uint32_t>(numCol_) - binaryAddQuotient;
        uint16_t binaryAddRemainderCeilLoop = (binaryAddRemainder + VL_F32 - 1) / VL_F32;
        uint16_t binaryAddRemainderFloorLoop = binaryAddRemainder / VL_F32;
        __VEC_SCOPE__
        {
            RegTensor<float> x;
            RegTensor<float> xFold;
            RegTensor<float> sumReg;
            RegTensor<float> vMean;
            RegTensor<float> onesReg;
            MaskReg pregFull = CreateMask<float, MaskPattern::ALL>();
            MaskReg pregOne = CreateMask<float, MaskPattern::VL1>();
            MaskReg pregLoop;
            AscendC::MicroAPI::Duplicate(onesReg, float(1.0), pregOne);

            for (uint16_t i = 0; i < curRows; ++i) {
                uint32_t baseOffset = i * numColAlign_;
                for (uint16_t r = 0; r < binaryAddRemainderFloorLoop; ++r) {
                    uint32_t off = r * VL_F32 + baseOffset;
                    LoadTensorForDtypeTIn<float>(xFp32Tmp, x, pregFull, off);
                    LoadTensorForDtypeTIn<float>(xFp32Tmp + binaryAddQuotient, xFold, pregFull, off);
                    AscendC::MicroAPI::Mul(x, x, x, pregFull);
                    AscendC::MicroAPI::Mul(xFold, xFold, xFold, pregFull);
                    AscendC::MicroAPI::Add(sumReg, x, xFold, pregFull);
                    AscendC::MicroAPI::ReduceSum(vMean, sumReg, pregFull);
                    AscendC::MicroAPI::DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(
                        tmpUb + static_cast<uint32_t>(i * lastBinaryAddNumAlign + r), vMean, pregOne);
                }
                uint32_t sregRemainder = binaryAddRemainder - binaryAddRemainderFloorLoop * VL_F32;
                for (uint16_t r = 0;
                     r < static_cast<uint16_t>(binaryAddRemainderCeilLoop - binaryAddRemainderFloorLoop); r++) {
                    uint16_t off = baseOffset;
                    pregLoop = UpdateMask<float>(sregRemainder);
                    LoadTensorForDtypeTIn<float>(xFp32Tmp + binaryAddRemainderFloorLoop * VL_F32, x, pregFull, off);
                    LoadTensorForDtypeTIn<float>(
                        xFp32Tmp + binaryAddRemainderFloorLoop * VL_F32 + binaryAddQuotient, xFold, pregLoop, off);
                    AscendC::MicroAPI::Mul(x, x, x, pregFull);
                    AscendC::MicroAPI::Mul(xFold, xFold, xFold, pregLoop);
                    AscendC::MicroAPI::ShiftLefts(
                        (RegTensor<uint32_t>&)xFold, (RegTensor<uint32_t>&)xFold, static_cast<int16_t>(0), pregLoop);
                    AscendC::MicroAPI::Add(sumReg, x, xFold, pregFull);
                    AscendC::MicroAPI::ReduceSum(vMean, sumReg, pregFull);
                    AscendC::MicroAPI::DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(
                        tmpUb + static_cast<uint32_t>(i * lastBinaryAddNumAlign + binaryAddRemainderFloorLoop), vMean,
                        pregOne);
                }
                for (uint16_t r = 0; r < static_cast<uint16_t>(binaryAddQuotientLoop - binaryAddRemainderCeilLoop);
                     r++) {
                    uint32_t off = r * VL_F32 + baseOffset;
                    LoadTensorForDtypeTIn<float>(xFp32Tmp + binaryAddRemainderCeilLoop * VL_F32, x, pregFull, off);
                    AscendC::MicroAPI::Mul(x, x, x, pregFull);
                    AscendC::MicroAPI::ReduceSum(vMean, x, pregFull);
                    AscendC::MicroAPI::DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(
                        tmpUb + static_cast<uint32_t>(i * lastBinaryAddNumAlign + binaryAddRemainderCeilLoop + r),
                        vMean, pregOne);
                }
            }
            LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
            if constexpr (LAST_LOOP_NUMS == 1) {
                MaskReg pregLast = UpdateMask<float>(lastBinaryAddNum);
                for (uint16_t i = 0; i < curRows; ++i) {
                    AscendC::MicroAPI::DataCopy(x, tmpUb + static_cast<uint32_t>(i * lastBinaryAddNumAlign));
                    AscendC::MicroAPI::ReduceSum(vMean, x, pregLast);
                    AscendC::MicroAPI::DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(
                        xReduceUb + i, vMean, pregOne);
                }
            } else if constexpr (LAST_LOOP_NUMS == 2) {
                lastBinaryAddNum -= VL_F32;
                MaskReg pregLast = UpdateMask<float>(lastBinaryAddNum);
                for (uint16_t i = 0; i < curRows; ++i) {
                    AscendC::MicroAPI::DataCopy(x, tmpUb + static_cast<uint32_t>(i * lastBinaryAddNumAlign));
                    AscendC::MicroAPI::DataCopy(
                        xFold, tmpUb + static_cast<uint32_t>(i * lastBinaryAddNumAlign + VL_F32));
                    AscendC::MicroAPI::ShiftLefts(
                        (RegTensor<uint32_t>&)xFold, (RegTensor<uint32_t>&)xFold, static_cast<int16_t>(0), pregLast);
                    AscendC::MicroAPI::Add(sumReg, x, xFold, pregFull);
                    AscendC::MicroAPI::ReduceSum(vMean, sumReg, pregFull);
                    AscendC::MicroAPI::DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(
                        xReduceUb + i, vMean, pregOne);
                }
            }
        }
    }

    __aicore__ inline void CalculateRstd(
        LocalTensor<float>& xReduceLocal, LocalTensor<float>& rstdLocal, uint32_t curRows)
    {
        static constexpr float POS_INF = 3.40282366920938E+38;
        static constexpr float SCALAR1 = -0.5;
        static constexpr float SCALAR2 = 1.5;
        static constexpr float SCALAR3 = 0.5;
        static constexpr float SCALAR0 = -99.99;

        __local_mem__ float* rstdInUb = (__local_mem__ float*)rstdLocal.GetPhyAddr();
        __local_mem__ float* xReduceUb = (__local_mem__ float*)xReduceLocal.GetPhyAddr();
        uint16_t loopRows = static_cast<uint16_t>((curRows + VL_F32 - 1) / VL_F32);
        __VEC_SCOPE__
        {
            RegTensor<float> var, rstd, r, y, s, t, one, scalar1, t1, t2, t3, t4, scalarInf, scalarZero;
            MaskReg cmpRegZero, cmpRegInf;
            MaskReg pregFull = CreateMask<float, MaskPattern::ALL>();
            MaskReg pregLoop;

            uint32_t sreg = static_cast<uint32_t>(curRows);
            for (uint16_t i = 0; i < loopRows; ++i) {
                pregLoop = UpdateMask<float>(sreg);
                AscendC::MicroAPI::Duplicate(scalarInf, POS_INF, pregLoop);
                AscendC::MicroAPI::Duplicate(scalarZero, float(0.0), pregLoop);
                AscendC::MicroAPI::Duplicate(one, float(1.0), pregLoop);
                AscendC::MicroAPI::Duplicate(scalar1, SCALAR3, pregLoop);
                AscendC::MicroAPI::Duplicate(t1, SCALAR2, pregLoop);
                AscendC::MicroAPI::Duplicate(s, float(1.0), pregLoop);
                AscendC::MicroAPI::DataCopy(var, xReduceUb + i * VL_F32);
                AscendC::MicroAPI::Muls(var, var, avgFactor_, pregLoop);
                AscendC::MicroAPI::Adds(var, var, epsilon_, pregLoop);
                AscendC::MicroAPI::Maxs(var, var, SCALAR0, pregLoop);
                AscendC::MicroAPI::Div(r, one, var, pregLoop);
                AscendC::MicroAPI::Sqrt(y, r, pregLoop);
                AscendC::MicroAPI::Muls(t, var, SCALAR1, pregLoop);
                AscendC::MicroAPI::Mul(t, t, y, pregLoop);
                AscendC::MicroAPI::Mula(t1, t, y, pregLoop);
                AscendC::MicroAPI::Mul(rstd, y, t1, pregLoop);
                AscendC::MicroAPI::Muls(t3, var, float(-1.0), pregLoop);
                AscendC::MicroAPI::Mula(s, t3, r, pregLoop);
                AscendC::MicroAPI::Muls(t4, rstd, float(-1.0), pregLoop);
                AscendC::MicroAPI::Mula(r, t4, rstd, pregLoop);
                AscendC::MicroAPI::Mula(s, var, r, pregLoop);
                AscendC::MicroAPI::Mul(s, s, rstd, pregLoop);
                AscendC::MicroAPI::Mula(rstd, s, scalar1, pregLoop);
                AscendC::MicroAPI::CompareScalar(cmpRegZero, var, POS_INF, pregLoop);
                AscendC::MicroAPI::Select(rstd, scalarZero, rstd, cmpRegZero);
                AscendC::MicroAPI::CompareScalar(cmpRegInf, var, float(0.0), pregLoop);
                AscendC::MicroAPI::Select(rstd, scalarInf, rstd, cmpRegInf);
                AscendC::MicroAPI::DataCopy(rstdInUb + i * VL_F32, rstd, pregLoop);
            }
        }
    }

    template <AscendC::RoundMode toBf16RoundMode, AscendC::RoundMode roundMode>
    __aicore__ inline void DynamicMxQuantPhaseFP4(LocalTensor<T_X>& yLocal, uint32_t curRows)
    {
        LocalTensor<uint16_t> maxExpLocal = maxExpBuff.Get<uint16_t>();

        uint32_t totalScaleInUB = curRows * blockNumInColAxis_;
        uint32_t totalCountInUB = curRows * blockNumInColAxis_ * mxBlockSize_;

        uint16_t loopNum = (totalCountInUB + vlForHalfNumber * DIGIT_TWO - 1) / (vlForHalfNumber * DIGIT_TWO);
        uint16_t loopNumScale = (totalScaleInUB + vlForHalfNumber - 1) / vlForHalfNumber;

        auto srcAddr = reinterpret_cast<__ubuf__ T_X*>(yLocal.GetPhyAddr());
        auto maxExpAddr = reinterpret_cast<__ubuf__ uint16_t*>(maxExpLocal.GetPhyAddr());

        LocalTensor<uint16_t> mxScaleLocal = mxScaleQueue.AllocTensor<uint16_t>();
        auto mxScaleLocalAddr = reinterpret_cast<__ubuf__ uint16_t*>(mxScaleLocal.GetPhyAddr());

        LocalTensor<uint16_t> halfScaleLocal = halfScaleBuff.Get<uint16_t>();
        auto halfScaleLocalAddr = reinterpret_cast<__ubuf__ uint16_t*>(halfScaleLocal.GetPhyAddr());

        LocalTensor<int8_t> outLocal = outQueueQuantY.AllocTensor<int8_t>();
        auto outLocalAddr = reinterpret_cast<__ubuf__ int8_t*>(outLocal.GetPhyAddr());

        maxExpAddr = reinterpret_cast<__ubuf__ uint16_t*>(maxExpLocal.GetPhyAddr());
        MxQuantComputeMaxExpOCP(srcAddr, maxExpAddr, totalCountInUB, loopNum);
        MxQuantComputeScaleOCP(maxExpAddr, mxScaleLocalAddr, halfScaleLocalAddr, totalScaleInUB, loopNumScale);

        srcAddr = reinterpret_cast<__ubuf__ T_X*>(yLocal.GetPhyAddr());
        halfScaleLocalAddr = reinterpret_cast<__ubuf__ uint16_t*>(halfScaleLocal.GetPhyAddr());

        MxQuantComputeDataFP4<toBf16RoundMode, roundMode>(
            srcAddr, halfScaleLocalAddr, outLocalAddr, totalCountInUB, loopNum);

        outQueueQuantY.EnQue(outLocal);
        mxScaleQueue.EnQue(mxScaleLocal);
    }

    __aicore__ inline void MxQuantComputeMaxExpOCP(
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
                    AscendC::MicroAPI::LoadDist::DIST_DINTLV_B16>(vdExp0, vdExp1, srcAddr, vlForHalfNumber * DIGIT_TWO);
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
                    maxExpAddr, vdMaxExp, u1, elementAfterReduce);
            }
            AscendC::MicroAPI::DataCopyUnAlignPost(maxExpAddr, u1, 0);
        }
    }

    __aicore__ inline void MxQuantComputeScaleOCP(
        __ubuf__ uint16_t* maxExpAddr, __ubuf__ uint16_t* mxScaleLocalAddr, __ubuf__ uint16_t* halfScaleLocalAddr,
        uint32_t totalScaleInUB, uint16_t loopNumScale)
    {
        __VEC_SCOPE__
        {
            AscendC::MicroAPI::RegTensor<uint16_t> expMask;
            AscendC::MicroAPI::Duplicate(expMask, MAX_EXP_FOR_BF16);
            AscendC::MicroAPI::RegTensor<uint16_t> vdMaxExp;
            AscendC::MicroAPI::MaskReg cmpResult;
            AscendC::MicroAPI::MaskReg zeroMask;
            AscendC::MicroAPI::MaskReg preMaskScale;
            AscendC::MicroAPI::RegTensor<uint16_t> maxExpValue;
            AscendC::MicroAPI::Duplicate(maxExpValue, f4Emax_);
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
                    vdMaxExp, maxExpAddr, vlForHalfNumber);
                AscendC::MicroAPI::Compare<uint16_t, CMPMODE::NE>(cmpResult, vdMaxExp, expMask, preMaskScale);
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
                    mxScaleLocalAddr, scaleValue, vlForHalfNumber / DIGIT_TWO, preMaskScale);

                AscendC::MicroAPI::Compare<uint16_t, CMPMODE::EQ>(specialDataMask, sharedExp, scaleBias, preMaskScale);
                AscendC::MicroAPI::Sub(halfScale, scaleBias, sharedExp, preMaskScale);
                AscendC::MicroAPI::Select<uint16_t>(halfScale, halfScale, nanRegTensor, cmpResult);
                AscendC::MicroAPI::Select<uint16_t>(halfScale, halfScale, zeroRegTensor, zeroMask);
                AscendC::MicroAPI::Select<uint16_t>(halfScale, specialExpRegTensor, halfScale, specialDataMask);

                AscendC::MicroAPI::DataCopy<uint16_t, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                    halfScaleLocalAddr, halfScale, vlForHalfNumber, preMaskScale);
            }
        }
    }

    template <AscendC::RoundMode toBf16RoundMode, AscendC::RoundMode roundMode>
    __aicore__ inline void ComputeFP4FromHalf(AscendC::MicroAPI::RegTensor<float>& Reg)
    {
        AscendC::MicroAPI::MaskReg pregAll32 =
            AscendC::MicroAPI::CreateMask<uint32_t, AscendC::MicroAPI::MaskPattern::ALL>();
        AscendC::MicroAPI::MaskReg zeroMask;
        AscendC::MicroAPI::MaskReg specialMask;
        AscendC::MicroAPI::MaskReg negInfMask;
        AscendC::MicroAPI::RegTensor<int32_t> negZero;
        AscendC::MicroAPI::RegTensor<int32_t> maxExpFP32;
        AscendC::MicroAPI::RegTensor<int32_t> exp0FP32;
        AscendC::MicroAPI::RegTensor<int32_t> exp1FP32;

        AscendC::MicroAPI::Duplicate(negZero, NEG_ZERO);
        AscendC::MicroAPI::Compare<int32_t, CMPMODE::EQ>(
            negInfMask, (AscendC::MicroAPI::RegTensor<int32_t>&)Reg, negZero, pregAll32);

        if constexpr (IsSame<T_Y, fp4x2_e1m2_t>::value) {
            AscendC::MicroAPI::Muls(Reg, Reg, FOUR, pregAll32);
            AscendC::MicroAPI::CompareScalar<float, CMPMODE::LT>(specialMask, Reg, 0, pregAll32);
            AscendC::MicroAPI::Truncate<float, roundMode>(Reg, Reg, pregAll32);
            AscendC::MicroAPI::Muls(Reg, Reg, ONE_FOURTH, pregAll32);
        } else {
            AscendC::MicroAPI::Duplicate(maxExpFP32, MAX_EXP_FOR_FP32);
            AscendC::MicroAPI::And(exp0FP32, (AscendC::MicroAPI::RegTensor<int32_t>&)Reg, maxExpFP32, pregAll32);
            AscendC::MicroAPI::ShiftRights(exp0FP32, exp0FP32, SHR_NUM_FOR_FP32, pregAll32);
            AscendC::MicroAPI::Adds(exp0FP32, exp0FP32, FP32_BIAS_NEG, pregAll32);
            AscendC::MicroAPI::Maxs(exp0FP32, exp0FP32, 0, pregAll32);
            AscendC::MicroAPI::Adds(exp0FP32, exp0FP32, NEG_ONE, pregAll32);
            AscendC::MicroAPI::Muls(exp1FP32, exp0FP32, NEG_ONE, pregAll32);
            AscendC::MicroAPI::Adds(exp1FP32, exp1FP32, FP32_BIAS, pregAll32);
            AscendC::MicroAPI::ShiftLefts(exp1FP32, exp1FP32, SHR_NUM_FOR_FP32, pregAll32);

            AscendC::MicroAPI::Mul(Reg, Reg, (AscendC::MicroAPI::RegTensor<float>&)exp1FP32, pregAll32);
            AscendC::MicroAPI::Adds(exp0FP32, exp0FP32, FP32_BIAS, pregAll32);
            AscendC::MicroAPI::ShiftLefts(exp0FP32, exp0FP32, SHR_NUM_FOR_FP32, pregAll32);
            AscendC::MicroAPI::CompareScalar<float, CMPMODE::LT>(specialMask, Reg, 0, pregAll32);
            AscendC::MicroAPI::Truncate<float, roundMode>(Reg, Reg, pregAll32);
            AscendC::MicroAPI::Mul(Reg, Reg, (AscendC::MicroAPI::RegTensor<float>&)exp0FP32, pregAll32);
        }

        AscendC::MicroAPI::CompareScalar<float, CMPMODE::EQ>(zeroMask, Reg, 0, pregAll32);
        AscendC::MicroAPI::MaskAnd(zeroMask, specialMask, zeroMask, pregAll32);
        AscendC::MicroAPI::MaskOr(zeroMask, negInfMask, zeroMask, pregAll32);
        AscendC::MicroAPI::Select<int32_t>(
            (AscendC::MicroAPI::RegTensor<int32_t>&)Reg, negZero, (AscendC::MicroAPI::RegTensor<int32_t>&)Reg,
            zeroMask);
    }

    template <AscendC::RoundMode toBf16RoundMode, AscendC::RoundMode roundMode>
    __aicore__ inline void MxQuantComputeDataFP4(
        __ubuf__ T_X* srcAddr, __ubuf__ uint16_t* halfScaleLocalAddr, __ubuf__ int8_t* outLocalAddr,
        uint32_t totalCountInUB, uint16_t loopNum)
    {
        __VEC_SCOPE__
        {
            AscendC::MicroAPI::MaskReg dataMask1;
            AscendC::MicroAPI::RegTensor<uint16_t> halfScaleForMul;
            AscendC::MicroAPI::RegTensor<T_X> vdExp0;
            AscendC::MicroAPI::RegTensor<T_X> vdExp1;
            AscendC::MicroAPI::RegTensor<bfloat16_t> vdExp0BF16;
            AscendC::MicroAPI::RegTensor<bfloat16_t> vdExp1BF16;
            AscendC::MicroAPI::RegTensor<T_Y> vdExp0FP4;
            AscendC::MicroAPI::RegTensor<T_Y> vdExp1FP4;
            AscendC::MicroAPI::RegTensor<float> halfScaleForMulFP32;
            AscendC::MicroAPI::RegTensor<float> vdExp0ZeroFP32;
            AscendC::MicroAPI::RegTensor<float> vdExp0OneFP32;
            AscendC::MicroAPI::RegTensor<float> vdExp1ZeroFP32;
            AscendC::MicroAPI::RegTensor<float> vdExp1OneFP32;
            AscendC::MicroAPI::RegTensor<bfloat16_t> vdExp0ZeroBF16;
            AscendC::MicroAPI::RegTensor<bfloat16_t> vdExp0OneBF16;
            AscendC::MicroAPI::RegTensor<bfloat16_t> vdExp1ZeroBF16;
            AscendC::MicroAPI::RegTensor<bfloat16_t> vdExp1OneBF16;
            AscendC::MicroAPI::MaskReg dataMaskB16 = AscendC::MicroAPI::CreateMask<half>();
            AscendC::MicroAPI::MaskReg dataMaskB32 = AscendC::MicroAPI::CreateMask<float>();

            static constexpr AscendC::MicroAPI::CastTrait castTrait = {
                AscendC::MicroAPI::RegLayout::ZERO, AscendC::MicroAPI::SatMode::UNKNOWN,
                AscendC::MicroAPI::MaskMergeMode::ZEROING, roundMode};
            static constexpr AscendC::MicroAPI::CastTrait castTraitHalf2Bf16 = {
                AscendC::MicroAPI::RegLayout::UNKNOWN, AscendC::MicroAPI::SatMode::UNKNOWN,
                AscendC::MicroAPI::MaskMergeMode::ZEROING, toBf16RoundMode};
            static constexpr AscendC::MicroAPI::CastTrait castTraitF16toFp32Zero = {
                AscendC::MicroAPI::RegLayout::ZERO, AscendC::MicroAPI::SatMode::UNKNOWN,
                AscendC::MicroAPI::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};
            static constexpr AscendC::MicroAPI::CastTrait castTraitF16toFp32One = {
                AscendC::MicroAPI::RegLayout::ONE, AscendC::MicroAPI::SatMode::UNKNOWN,
                AscendC::MicroAPI::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};
            static constexpr AscendC::MicroAPI::CastTrait castTraitFp32toBF16 = {
                AscendC::MicroAPI::RegLayout::ZERO, AscendC::MicroAPI::SatMode::NO_SAT,
                AscendC::MicroAPI::MaskMergeMode::ZEROING, roundMode};

            for (uint16_t i = 0; i < loopNum; i++) {
                dataMask1 = AscendC::MicroAPI::UpdateMask<T_X>(totalCountInUB);
                AscendC::MicroAPI::DataCopy<
                    T_X, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE,
                    AscendC::MicroAPI::LoadDist::DIST_DINTLV_B16>(vdExp0, vdExp1, srcAddr, vlForHalfNumber * DIGIT_TWO);
                AscendC::MicroAPI::DataCopy<
                    uint16_t, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE,
                    AscendC::MicroAPI::LoadDist::DIST_E2B_B16>(halfScaleForMul, halfScaleLocalAddr, elementAfterReduce);

                if constexpr (IsSame<T_X, half>::value) {
                    if constexpr (roundMode == RoundMode::CAST_RINT || roundMode == RoundMode::CAST_ROUND) {
                        // tail_axis_optimize_fp16 -> fp4 (rint、round)
                        AscendC::MicroAPI::Cast<float, bfloat16_t, castTraitF16toFp32Zero>(
                            halfScaleForMulFP32, (AscendC::MicroAPI::RegTensor<bfloat16_t>&)halfScaleForMul,
                            dataMaskB16);
                        AscendC::MicroAPI::Cast<float, T_X, castTraitF16toFp32Zero>(
                            vdExp0ZeroFP32, vdExp0, dataMaskB16);
                        AscendC::MicroAPI::Cast<float, T_X, castTraitF16toFp32One>(vdExp0OneFP32, vdExp0, dataMaskB16);
                        AscendC::MicroAPI::Mul(vdExp0ZeroFP32, vdExp0ZeroFP32, halfScaleForMulFP32, dataMaskB32);
                        AscendC::MicroAPI::Mul(vdExp0OneFP32, vdExp0OneFP32, halfScaleForMulFP32, dataMaskB32);
                        ComputeFP4FromHalf<toBf16RoundMode, roundMode>(vdExp0ZeroFP32);
                        ComputeFP4FromHalf<toBf16RoundMode, roundMode>(vdExp0OneFP32);
                        AscendC::MicroAPI::Cast<bfloat16_t, float, castTraitFp32toBF16>(
                            vdExp0ZeroBF16, vdExp0ZeroFP32, dataMaskB32);
                        AscendC::MicroAPI::Cast<bfloat16_t, float, castTraitFp32toBF16>(
                            vdExp0OneBF16, vdExp0OneFP32, dataMaskB32);
                        AscendC::MicroAPI::Pack<uint16_t, uint32_t, AscendC::MicroAPI::HighLowPart::LOWEST>(
                            (AscendC::MicroAPI::RegTensor<uint16_t>&)vdExp0ZeroBF16,
                            (AscendC::MicroAPI::RegTensor<uint32_t>&)vdExp0ZeroBF16);
                        AscendC::MicroAPI::Pack<uint16_t, uint32_t, AscendC::MicroAPI::HighLowPart::LOWEST>(
                            (AscendC::MicroAPI::RegTensor<uint16_t>&)vdExp0OneBF16,
                            (AscendC::MicroAPI::RegTensor<uint32_t>&)vdExp0OneBF16);
                        AscendC::MicroAPI::Interleave(vdExp0ZeroBF16, vdExp0OneBF16, vdExp0ZeroBF16, vdExp0OneBF16);
                        AscendC::MicroAPI::Cast<float, T_X, castTraitF16toFp32Zero>(
                            vdExp1ZeroFP32, vdExp1, dataMaskB16);
                        AscendC::MicroAPI::Cast<float, T_X, castTraitF16toFp32One>(vdExp1OneFP32, vdExp1, dataMaskB16);
                        AscendC::MicroAPI::Mul(vdExp1ZeroFP32, vdExp1ZeroFP32, halfScaleForMulFP32, dataMaskB32);
                        AscendC::MicroAPI::Mul(vdExp1OneFP32, vdExp1OneFP32, halfScaleForMulFP32, dataMaskB32);
                        ComputeFP4FromHalf<toBf16RoundMode, roundMode>(vdExp1ZeroFP32);
                        ComputeFP4FromHalf<toBf16RoundMode, roundMode>(vdExp1OneFP32);
                        AscendC::MicroAPI::Cast<bfloat16_t, float, castTraitFp32toBF16>(
                            vdExp1ZeroBF16, vdExp1ZeroFP32, dataMaskB32);
                        AscendC::MicroAPI::Cast<bfloat16_t, float, castTraitFp32toBF16>(
                            vdExp1OneBF16, vdExp1OneFP32, dataMaskB32);
                        AscendC::MicroAPI::Pack<uint16_t, uint32_t, AscendC::MicroAPI::HighLowPart::LOWEST>(
                            (AscendC::MicroAPI::RegTensor<uint16_t>&)vdExp1ZeroBF16,
                            (AscendC::MicroAPI::RegTensor<uint32_t>&)vdExp1ZeroBF16);
                        AscendC::MicroAPI::Pack<uint16_t, uint32_t, AscendC::MicroAPI::HighLowPart::LOWEST>(
                            (AscendC::MicroAPI::RegTensor<uint16_t>&)vdExp1OneBF16,
                            (AscendC::MicroAPI::RegTensor<uint32_t>&)vdExp1OneBF16);
                        AscendC::MicroAPI::Interleave(vdExp1ZeroBF16, vdExp1OneBF16, vdExp1ZeroBF16, vdExp1OneBF16);
                        AscendC::MicroAPI::Interleave(vdExp0ZeroBF16, vdExp1ZeroBF16, vdExp0ZeroBF16, vdExp1ZeroBF16);
                        AscendC::MicroAPI::Cast<T_Y, bfloat16_t, castTrait>(vdExp0FP4, vdExp0ZeroBF16, dataMask1);
                        AscendC::MicroAPI::Cast<T_Y, bfloat16_t, castTrait>(vdExp1FP4, vdExp1ZeroBF16, dataMask1);
                    } else {
                        // for fp16 -> fp4 (floor)
                        AscendC::MicroAPI::Cast<bfloat16_t, T_X, castTraitHalf2Bf16>(vdExp0BF16, vdExp0, dataMask1);
                        AscendC::MicroAPI::Cast<bfloat16_t, T_X, castTraitHalf2Bf16>(vdExp1BF16, vdExp1, dataMask1);
                        AscendC::MicroAPI::Mul(
                            vdExp0BF16, vdExp0BF16, (AscendC::MicroAPI::RegTensor<bfloat16_t>&)halfScaleForMul,
                            dataMask1);
                        AscendC::MicroAPI::Mul(
                            vdExp1BF16, vdExp1BF16, (AscendC::MicroAPI::RegTensor<bfloat16_t>&)halfScaleForMul,
                            dataMask1);
                        AscendC::MicroAPI::Interleave(vdExp0BF16, vdExp1BF16, vdExp0BF16, vdExp1BF16);
                        AscendC::MicroAPI::Cast<T_Y, bfloat16_t, castTrait>(vdExp0FP4, vdExp0BF16, dataMask1);
                        AscendC::MicroAPI::Cast<T_Y, bfloat16_t, castTrait>(vdExp1FP4, vdExp1BF16, dataMask1);
                    }
                } else {
                    // for bf16 
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
    }

    __aicore__ inline void CopyInXMultiMoveAlign(uint64_t offset, uint32_t curRows)
    {
        LocalTensor<T_X> xLocal1 = inQueueX1.AllocTensor<T_X>();
        LocalTensor<T_X> xLocal2 = inQueueX2.AllocTensor<T_X>();
        
        DataCopyExtParams extParams{
            static_cast<uint16_t>(curRows),
            static_cast<uint32_t>(numCol_ * sizeof(T_X)),
            static_cast<uint32_t>(0),
            static_cast<uint32_t>(dstStrideUbBlocks_),
            0
        };
        DataCopyPadExtParams<T_X> padParams{false, static_cast<uint8_t>(0), static_cast<uint8_t>(0), static_cast<T_X>(0.0)};

        DataCopyPad(xLocal1, x1Gm[offset], extParams, padParams);
        DataCopyPad(xLocal2, x2Gm[offset], extParams, padParams);
        inQueueX1.EnQue(xLocal1);
        inQueueX2.EnQue(xLocal2);
    }

    __aicore__ inline void CopyInGammabeta(LocalTensor<uint8_t> gammabetaLocal)
    {
        DataCopyExtParams copyParams{
            static_cast<uint16_t>(1), static_cast<uint32_t>(numCol_ * sizeof(T_GAMMA)), static_cast<uint32_t>(0),
            static_cast<uint32_t>(0), 0};
        DataCopyPadExtParams<T_GAMMA> padParams{
            false, static_cast<uint8_t>(0), static_cast<uint8_t>(0), static_cast<T_GAMMA>(0.0)};
        gammaLocal = gammabetaLocal.ReinterpretCast<T_GAMMA>();
        DataCopyPad<T_GAMMA>(gammaLocal, gammaGm, copyParams, padParams);
        if (betaFlag_ != 0) {
            betaLocal = gammabetaLocal[CeilAlign(numCol_, UB_BLOCK_SIZE / sizeof(T_GAMMA)) * sizeof(T_GAMMA)]
                            .ReinterpretCast<T_GAMMA>();
            DataCopyPad<T_GAMMA>(betaLocal, betaGm, copyParams, padParams);
        }
    }

    __aicore__ inline void CopyOutX(uint64_t offset, uint32_t curRows)
    {
        LocalTensor<T_X> xLocal = outQueueX.DeQue<T_X>();
        uint32_t srcStride = (numColAlign_ - numCol_) * sizeof(T_X) / UB_BLOCK_SIZE;
        DataCopyExtParams copyParams{
            static_cast<uint16_t>(curRows), static_cast<uint32_t>(numCol_ * sizeof(T_X)),
            static_cast<uint32_t>(srcStride), static_cast<uint32_t>(0), 0};
        DataCopyPad(xOutGm[offset], xLocal, copyParams);
        outQueueX.FreeTensor(xLocal);
    }

    __aicore__ inline void CopyOutQuantYFP4(uint64_t gmOffset, uint32_t curRows)
    {
        LocalTensor<uint8_t> quantYLocal = outQueueQuantY.DeQue<uint8_t>();
        uint32_t bytesPerRow = numCol_ / DIGIT_TWO;
        uint32_t alignBytesPerRow = numColAlign_ / DIGIT_TWO;
        uint32_t srcStride = (alignBytesPerRow - bytesPerRow) / UB_BLOCK_SIZE;
        DataCopyExtParams copyParams{
            static_cast<uint16_t>(curRows), static_cast<uint32_t>(bytesPerRow), static_cast<uint32_t>(srcStride),
            static_cast<uint32_t>(0), 0};
        DataCopyPad<uint8_t>(yFp4Gm[gmOffset / DIGIT_TWO], quantYLocal, copyParams);
        outQueueQuantY.FreeTensor(quantYLocal);
    }

    __aicore__ inline void CopyOutMxScale(uint32_t rowRepeat, uint32_t curRows)
    {
        LocalTensor<uint8_t> mxScaleLocal = mxScaleQueue.DeQue<uint8_t>();
        DataCopyExtParams copyParams{
            static_cast<uint16_t>(curRows), static_cast<uint32_t>(mxScaleSize_), static_cast<uint32_t>(0),
            static_cast<uint32_t>(0), 0};
        DataCopyPad<uint8_t, PaddingMode::Compact>(
            mxScaleGm[rowRepeat * rowFactor_ * mxScaleSize_], mxScaleLocal, copyParams);
        mxScaleQueue.FreeTensor(mxScaleLocal);
    }

private:

    TPipe* pPipe = nullptr;

    TQue<QuePosition::VECIN, 1> inQueueX1;
    TQue<QuePosition::VECIN, 1> inQueueX2;
    TQue<QuePosition::VECIN, 1> inQueueGammabeta;
    TQue<QuePosition::VECOUT, 1> outQueueX;
    TQue<QuePosition::VECOUT, 1> outQueueRstd;
    TBuf<TPosition::VECCALC> xReduceBuff;
    TBuf<TPosition::VECCALC> xFp32Buff;
    TBuf<TPosition::VECCALC> binaryAddBuf;
    TBuf<TPosition::VECCALC> maxExpBuff;
    TBuf<TPosition::VECCALC> halfScaleBuff;
    TQue<QuePosition::VECOUT, 1> outQueueQuantY;
    TQue<QuePosition::VECOUT, 1> mxScaleQueue;

    LocalTensor<T_GAMMA> gammaLocal;
    LocalTensor<T_GAMMA> betaLocal;

    GlobalTensor<T_X> x1Gm;
    GlobalTensor<T_X> x2Gm;
    GlobalTensor<T_GAMMA> gammaGm;
    GlobalTensor<T_GAMMA> betaGm;
    GlobalTensor<T_X> xOutGm;
    GlobalTensor<float> rstdGm;
    GlobalTensor<uint8_t> yFp4Gm;
    GlobalTensor<uint8_t> mxScaleGm;

    uint64_t numRow_;
    uint64_t numCol_;
    uint64_t numColAlign_;
    uint64_t blockFactor_;
    uint64_t rowFactor_;
    uint64_t binAddQuotient_;
    float epsilon_;
    float avgFactor_;
    uint64_t rowWork{1};
    int64_t roundMode_;
    int64_t mxBlockSize_;
    int64_t scaleAlg_;
    int64_t blockNumInColAxis_;
    uint64_t dstStrideUbBlocks_;
    int64_t mxScaleSize_;
    uint32_t betaFlag_;
    uint32_t rstdFlag_;

    uint16_t f4Emax_;
    uint32_t vlForHalfNumber;
    uint16_t elementAfterReduce;
};
} // namespace AddRmsNormDynamicMxQuant
#endif // ADD_RMS_NORM_DYNAMIC_MX_QUANT_FP4_R_FULL_LOAD_H

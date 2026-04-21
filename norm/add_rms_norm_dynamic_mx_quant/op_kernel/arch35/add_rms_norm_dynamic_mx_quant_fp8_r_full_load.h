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
 * \file add_rms_norm_dynamic_mx_quant_fp8_r_full_load.h
 * \brief
 */
#ifndef ADD_RMS_NORM_DYNAMIC_MX_QUANT_FP8_R_FULL_LOAD_H
#define ADD_RMS_NORM_DYNAMIC_MX_QUANT_FP8_R_FULL_LOAD_H

#include "add_rms_norm_dynamic_mx_quant_common.h"

namespace AddRmsNormDynamicMxQuant {

template <typename T_X, typename T_GAMMA, typename T_Y>
class AddRmsNormDynamicMxQuantFP8RFullLoad {
public:
    __aicore__ inline AddRmsNormDynamicMxQuantFP8RFullLoad(TPipe* pipe)
    {
        pPipe = pipe;
    }

    __aicore__ inline void Init(
        GM_ADDR x1, GM_ADDR x2, GM_ADDR gamma, GM_ADDR beta, GM_ADDR y, 
        GM_ADDR x, GM_ADDR mxscale, GM_ADDR workspace, GM_ADDR rstd,
        const AddRmsNormDynamicMxQuantTilingData* tiling)
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
        rowWork = (GetBlockIdx() < GetBlockNum() - 1) ? blockFactor_
                                                       : numRow_ - (GetBlockNum() - 1) * blockFactor_;
        roundMode_ = tiling->roundMode;
        mxBlockSize_ = tiling->mxBlockSize;
        scaleAlg_ = tiling->scaleAlg;
        blockNumInColAxis_ = tiling->blockNumInColAxis;
        dstStrideUbBlocks_ = tiling->dstStrideUbBlocks;
        mxScaleSize_ = tiling->mxScaleSize;
        betaFlag_ = tiling->betaFlag;
        rstdFlag_ = tiling->rstdFlag;

        // === Setup GM tensors ===
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

        yFp8Gm.SetGlobalBuffer((__gm__ uint8_t*)y + blockOffset, rowWork * numCol_);
        mxScaleGm.SetGlobalBuffer(
            (__gm__ uint8_t*)mxscale + GetBlockIdx() * blockFactor_ * mxScaleSize_, rowWork * mxScaleSize_);

        // === Compute buffer sizes ===
        uint64_t rstdUbSizeAlignSize = CeilAlign(rowFactor_, static_cast<uint64_t>(VL_F32)) * sizeof(float);
        uint32_t binaryAddQuotientLoop = CeilDiv(binAddQuotient_, VL_F32);
        uint32_t binaryAddBufLen = CeilAlign(CeilAlign(binaryAddQuotientLoop, BLOCK_F32_ALIGN_NUM) * sizeof(float), UB_BLOCK_SIZE) * rowFactor_;

        // MxQuant buffer sizes
        uint64_t maxExpBufSize = CeilAlign(blockNumInColAxis_ * sizeof(T_X), UB_BLOCK_SIZE) * rowFactor_;
        uint64_t halfScaleBufSize = maxExpBufSize;

        // outQueueQuantY
        uint64_t quantYBufSize = CeilAlign(CeilDiv(numColAlign_ * rowFactor_, MX_STEP_PROCESS_NUM), DIGIT_FOUR) * MX_STEP_PROCESS_NUM;

        // MxScale output
        uint64_t scaleBufPerIter = CeilAlign(mxScaleSize_ * sizeof(T_X), UB_BLOCK_SIZE) * rowFactor_;

        // === Init buffers ===
        // AddRmsNorm buffers
        pPipe->InitBuffer(inQueueX1, DOUBLE_BUFFER_NUM, CeilAlign(numColAlign_ * sizeof(T_X), UB_BLOCK_SIZE) * rowFactor_);
        pPipe->InitBuffer(inQueueX2, DOUBLE_BUFFER_NUM, CeilAlign(numColAlign_ * sizeof(T_X), UB_BLOCK_SIZE) * rowFactor_);
        if (betaFlag_ != 0) {
            pPipe->InitBuffer(inQueueGammabeta, 1, DIGIT_TWO * CeilAlign(numCol_, UB_BLOCK_SIZE / sizeof(T_GAMMA)) * sizeof(T_GAMMA));
        } else {
            pPipe->InitBuffer(inQueueGammabeta, 1, CeilAlign(numCol_, UB_BLOCK_SIZE / sizeof(T_GAMMA)) * sizeof(T_GAMMA));
        }
        pPipe->InitBuffer(outQueueX, DOUBLE_BUFFER_NUM, CeilAlign(numColAlign_ * sizeof(T_X), UB_BLOCK_SIZE) * rowFactor_);
        pPipe->InitBuffer(outQueueRstd, DOUBLE_BUFFER_NUM, rstdUbSizeAlignSize);
        pPipe->InitBuffer(xReduceBuff, rstdUbSizeAlignSize);
        pPipe->InitBuffer(xFp32Buff, CeilAlign(numColAlign_ * sizeof(float), UB_BLOCK_SIZE) * rowFactor_);
        pPipe->InitBuffer(binaryAddBuf, binaryAddBufLen);

        // MxQuant buffers
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
    __aicore__ inline void Compute(
        uint32_t rowRepeat, uint32_t curRows, uint64_t offset)
    {
        // Phase 1: AddRmsNorm
        // --- CopyIn x1, x2 ---
        CopyInXMultiMoveAlign(offset, curRows);
        LocalTensor<T_X> xLocal1 = inQueueX1.DeQue<T_X>();
        LocalTensor<T_X> xLocal2 = inQueueX2.DeQue<T_X>();
        LocalTensor<T_X> xOutLocal = outQueueX.AllocTensor<T_X>();
        LocalTensor<float> xFp32Local = xFp32Buff.Get<float>();

        // --- x = x1 + x2 ---
        CalculateXAdd(xLocal1, xLocal2, xOutLocal, xFp32Local, curRows);
        inQueueX1.FreeTensor(xLocal1);
        inQueueX2.FreeTensor(xLocal2);
        outQueueX.EnQue<T_X>(xOutLocal);

        // --- CopyOut x ---
        CopyOutX(offset, curRows);

        // --- ReduceSum(x^2) ---
        LocalTensor<float> rstdLocal = outQueueRstd.AllocTensor<float>();
        LocalTensor<float> xReduceLocal = xReduceBuff.Get<float>();
        CalculateSquareReduceSum(xFp32Local, xReduceLocal, curRows);

        // --- Rstd = 1/sqrt(mean + epsilon_) ---
        CalculateRstd(xReduceLocal, rstdLocal, curRows, avgFactor_, epsilon_);
        outQueueRstd.EnQue<float>(rstdLocal);

        // --- CopyOut rstd ---
        rstdLocal = outQueueRstd.DeQue<float>();
        if (rstdFlag_ != 0) {
            DataCopyExtParams rstdCopyParams{1, static_cast<uint32_t>(curRows * sizeof(float)), 0, 0, 0};
            DataCopyPad(rstdGm[rowRepeat * rowFactor_], rstdLocal, rstdCopyParams);
        }

        // --- Y = x * rstd * gamma + beta ---
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

        // Phase 2: MxQuant FP8
        DynamicMxQuantPhase<RoundMode::CAST_RINT>(yLocal, curRows);
        outQueueX.FreeTensor(yLocal);

        // CopyOut FP8 quantized Y
        CopyOutQuantY(offset, curRows);

        // CopyOut MxScale
        CopyOutMxScale(rowRepeat, curRows);
    }

    __aicore__ inline void CalculateXAdd(
        LocalTensor<T_X>& xLocal1, LocalTensor<T_X>& xLocal2, LocalTensor<T_X>& xOutLocal, LocalTensor<float>& xFp32Local,
        uint32_t curRows)
    {
        __local_mem__ T_X* x1InUb = (__local_mem__ T_X*)xLocal1.GetPhyAddr();
        __local_mem__ T_X* x2InUb = (__local_mem__ T_X*)xLocal2.GetPhyAddr();
        __local_mem__ T_X* xOutInUb = (__local_mem__ T_X*)xOutLocal.GetPhyAddr();
        __local_mem__ float* xFp32Tmp = (__local_mem__ float*)xFp32Local.GetPhyAddr();

        uint32_t sreg = curRows * numColAlign_;
        uint16_t loopCount = (sreg + VL_F32 - 1) / VL_F32;

        __VEC_SCOPE__
        {
            RegTensor<float> x1, x2, xSum;
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
    __aicore__ inline void CalculateY(LocalTensor<float>& xFp32Local, LocalTensor<T_X>& yLocal, LocalTensor<float>& rstdLocal, uint32_t curRows)
    {
        uint32_t numColAlign = static_cast<uint32_t>(numColAlign_);
        uint32_t numCol = static_cast<uint32_t>(numCol_);
        __local_mem__ float* xFp32Tmp = (__local_mem__ float*)xFp32Local.GetPhyAddr();
        __local_mem__ T_GAMMA* gammaInUb = (__local_mem__ T_GAMMA*)gammaLocal.GetPhyAddr();
        __local_mem__ T_X* yInUb = (__local_mem__ T_X*)yLocal.GetPhyAddr();
        __local_mem__ float* rstdInUb = (__local_mem__ float*)rstdLocal.GetPhyAddr();
        __local_mem__ T_GAMMA* betaInUb;
        if constexpr(hasBeta) {
            betaInUb = (__local_mem__ T_GAMMA*)betaLocal.GetPhyAddr();
        }

        uint16_t loopRows = static_cast<uint16_t>(curRows);
        uint16_t loopCols = static_cast<uint16_t>((numCol + VL_F32 - 1) / VL_F32);
        uint16_t loopRowsFold = loopRows / 2;
        uint16_t loopRowsHasLast = loopRows % 2;

        __VEC_SCOPE__ {
            RegTensor<float> x1Reg, x2Reg, gammaReg, betaReg, rstd1Reg, rstd2Reg, mul1Reg, mul1UnrollReg, mul2Reg, mul2UnrollReg;

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
                    if constexpr(hasBeta) {
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
                    if constexpr(hasBeta) {
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
            RegTensor<float> x, vMean, onesReg;
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
            RegTensor<float> x, xFold, sumReg, vMean, onesReg;

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
        uint32_t lastBinaryAddNumAlign = (binaryAddQuotientLoop + BLOCK_F32_ALIGN_NUM - 1) / BLOCK_F32_ALIGN_NUM * BLOCK_F32_ALIGN_NUM;

        uint32_t binaryAddRemainder = static_cast<uint32_t>(numCol_) - binaryAddQuotient;
        uint16_t binaryAddRemainderCeilLoop = (binaryAddRemainder + VL_F32 - 1) / VL_F32;
        uint16_t binaryAddRemainderFloorLoop = binaryAddRemainder / VL_F32;
        __VEC_SCOPE__
        {
            RegTensor<float> x, xFold, sumReg, vMean, onesReg;

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
                    AscendC::MicroAPI::DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(xReduceUb + i, vMean, pregOne);
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
                    AscendC::MicroAPI::DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(xReduceUb + i, vMean, pregOne);
                }
            }
        }
    }
    template <AscendC::RoundMode roundMode>
    __aicore__ inline void DynamicMxQuantPhase(LocalTensor<T_X>& yLocal, uint32_t curRows)
    {
        LocalTensor<uint16_t> maxExpLocal = maxExpBuff.Get<uint16_t>();

        uint32_t totalScaleInUB = curRows * blockNumInColAxis_;
        uint32_t totalCountInUB = curRows * blockNumInColAxis_ * mxBlockSize_;

        uint16_t loopNum = (totalCountInUB + VL_B16 * DIGIT_TWO - 1) / (VL_B16 * DIGIT_TWO);
        uint16_t loopNumScale = (totalScaleInUB + VL_B16 - 1) / VL_B16;
        uint16_t loopNumScale4NV = (totalScaleInUB + VL_F32 - 1) / VL_F32;

        auto srcAddr = reinterpret_cast<__ubuf__ T_X*>(yLocal.GetPhyAddr());
        auto maxExpAddr = reinterpret_cast<__ubuf__ uint16_t*>(maxExpLocal.GetPhyAddr());

        LocalTensor<uint16_t> mxScaleLocal = mxScaleQueue.AllocTensor<uint16_t>();
        auto mxScaleLocalAddr = reinterpret_cast<__ubuf__ uint16_t*>(mxScaleLocal.GetPhyAddr());

        LocalTensor<uint16_t> halfScaleLocal = halfScaleBuff.Get<uint16_t>();
        auto halfScaleLocalAddr = reinterpret_cast<__ubuf__ uint16_t*>(halfScaleLocal.GetPhyAddr());

        LocalTensor<int8_t> outLocal = outQueueQuantY.AllocTensor<int8_t>();
        auto outLocalAddr = reinterpret_cast<__ubuf__ int8_t*>(outLocal.GetPhyAddr());
        maxExpAddr = reinterpret_cast<__ubuf__ uint16_t*>(maxExpLocal.GetPhyAddr());
        if (scaleAlg_ == 0) {
            MxQuantComputeMaxExpOCP<T_X>(srcAddr, maxExpAddr, loopNum);
            MxQuantComputeScaleOCP<T_Y>(maxExpAddr, mxScaleLocalAddr, halfScaleLocalAddr, totalScaleInUB, loopNumScale);
        } else {
            MxQuantComputeMaxExpcuBLAS<T_X>(srcAddr, maxExpAddr, loopNum);
            MxQuantComputeScalecuBLAS<T_X, T_Y>(maxExpAddr, mxScaleLocalAddr, halfScaleLocalAddr, totalScaleInUB, loopNumScale4NV);
        }

        srcAddr = reinterpret_cast<__ubuf__ T_X*>(yLocal.GetPhyAddr());
        halfScaleLocalAddr = reinterpret_cast<__ubuf__ uint16_t*>(halfScaleLocal.GetPhyAddr());

        MxQuantComputeData<roundMode, T_X, T_Y>(srcAddr, halfScaleLocalAddr, outLocalAddr, loopNum);

        outQueueQuantY.EnQue(outLocal);
        mxScaleQueue.EnQue(mxScaleLocal);
        return;
    }


    __aicore__ inline void MxQuantDeletePadData(
        __ubuf__ int8_t* outLocalAddr, __ubuf__ int8_t* outBufferLocalAddr, uint16_t loopNum, uint32_t inputUpdateStride,
        uint32_t outputUpdateStride)
    {
        __VEC_SCOPE__
        {
            AscendC::MicroAPI::UnalignReg uIn;
            AscendC::MicroAPI::UnalignReg uOut;
            AscendC::MicroAPI::RegTensor<int8_t> inputRegTensor;
            for (uint16_t i = 0; i < loopNum; i++) {
                AscendC::MicroAPI::DataCopyUnAlignPre(uIn, outBufferLocalAddr);
                AscendC::MicroAPI::DataCopyUnAlign<int8_t, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                    inputRegTensor, uIn, outBufferLocalAddr, inputUpdateStride);
                AscendC::MicroAPI::DataCopyUnAlign<int8_t, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                    outLocalAddr, inputRegTensor, uOut, outputUpdateStride);
                AscendC::MicroAPI::DataCopyUnAlignPost(outLocalAddr, uOut, 0);
            }
        }
        return;
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
            static_cast<uint16_t>(1),
            static_cast<uint32_t>(numCol_ * sizeof(T_GAMMA)),
            static_cast<uint32_t>(0),
            static_cast<uint32_t>(0),
            0
        };
        DataCopyPadExtParams<T_GAMMA> padParams{false, static_cast<uint8_t>(0), static_cast<uint8_t>(0), static_cast<T_GAMMA>(0.0)};
        gammaLocal = gammabetaLocal.ReinterpretCast<T_GAMMA>();
        DataCopyPad<T_GAMMA>(gammaLocal, gammaGm, copyParams, padParams);
        if (betaFlag_ != 0) {
            betaLocal = gammabetaLocal[CeilAlign(numCol_, UB_BLOCK_SIZE / sizeof(T_GAMMA)) * sizeof(T_GAMMA)].ReinterpretCast<T_GAMMA>();
            DataCopyPad<T_GAMMA>(betaLocal, betaGm, copyParams, padParams);
        }
    }

    __aicore__ inline void CopyOutX(uint64_t offset, uint32_t curRows)
    {
        LocalTensor<T_X> xLocal = outQueueX.DeQue<T_X>();
        uint32_t srcStride = (numColAlign_ - numCol_) * sizeof(T_X) / UB_BLOCK_SIZE;
        DataCopyExtParams copyParams{
            static_cast<uint16_t>(curRows),
            static_cast<uint32_t>(numCol_ * sizeof(T_X)),
            static_cast<uint32_t>(srcStride),
            static_cast<uint32_t>(0),
            0
        };
        DataCopyPad(xOutGm[offset], xLocal, copyParams);
        outQueueX.FreeTensor(xLocal);
    }

    __aicore__ inline void CopyOutQuantY(uint64_t gmOffset, uint32_t curRows)
    {
        LocalTensor<uint8_t> quantYLocal = outQueueQuantY.DeQue<uint8_t>();
        // FP8 output
        uint32_t srcStride = (numColAlign_ - numCol_) * sizeof(uint8_t) / UB_BLOCK_SIZE;
        DataCopyExtParams copyParams{
            static_cast<uint16_t>(curRows),
            static_cast<uint32_t>(numCol_),
            static_cast<uint32_t>(srcStride),
            static_cast<uint32_t>(0),
            0
        };
        DataCopyPad<uint8_t>(yFp8Gm[gmOffset], quantYLocal, copyParams);
        outQueueQuantY.FreeTensor(quantYLocal);
    }

    __aicore__ inline void CopyOutMxScale(uint32_t rowRepeat, uint32_t curRows)
    {
        LocalTensor<uint8_t> mxScaleLocal = mxScaleQueue.DeQue<uint8_t>();
        DataCopyExtParams copyParams{
            static_cast<uint16_t>(curRows),
            static_cast<uint32_t>(mxScaleSize_),
            static_cast<uint32_t>(0),
            static_cast<uint32_t>(0),
            0
        };
        DataCopyPad<uint8_t, PaddingMode::Compact>(mxScaleGm[rowRepeat * rowFactor_ * mxScaleSize_], mxScaleLocal, copyParams);
        mxScaleQueue.FreeTensor(mxScaleLocal);
    }
private:
    TPipe* pPipe = nullptr;

    // AddRmsNorm queues and buffers
    TQue<QuePosition::VECIN, 1> inQueueX1;
    TQue<QuePosition::VECIN, 1> inQueueX2;
    TQue<QuePosition::VECIN, 1> inQueueGammabeta;
    TQue<QuePosition::VECOUT, 1> outQueueX;
    TQue<QuePosition::VECOUT, 1> outQueueRstd;
    TBuf<TPosition::VECCALC> xReduceBuff;
    TBuf<TPosition::VECCALC> xFp32Buff;
    TBuf<TPosition::VECCALC> binaryAddBuf;

    // MxQuant buffers
    TBuf<TPosition::VECCALC> maxExpBuff;
    TBuf<TPosition::VECCALC> halfScaleBuff;
    TQue<QuePosition::VECOUT, 1> outQueueQuantY;
    TQue<QuePosition::VECOUT, 1> mxScaleQueue;

    LocalTensor<T_GAMMA> gammaLocal;
    LocalTensor<T_GAMMA> betaLocal;

    // GM tensors
    GlobalTensor<T_X> x1Gm;
    GlobalTensor<T_X> x2Gm;
    GlobalTensor<T_GAMMA> gammaGm;
    GlobalTensor<T_GAMMA> betaGm;
    GlobalTensor<T_X> xOutGm;
    GlobalTensor<float> rstdGm;
    GlobalTensor<uint8_t> yFp8Gm;      // FP8 quantized output
    GlobalTensor<uint8_t> mxScaleGm;   // MX scale output

    // tiling parameters
    uint64_t numRow_;
    uint64_t numCol_;
    uint64_t numColAlign_;
    uint64_t blockFactor_;
    uint64_t rowFactor_;
    uint64_t binAddQuotient_;
    float epsilon_;
    float avgFactor_;
    uint64_t rowWork{1};
    uint64_t roundMode_;
    uint64_t mxBlockSize_;
    int64_t scaleAlg_;
    uint64_t blockNumInColAxis_;
    uint64_t dstStrideUbBlocks_;
    uint64_t mxScaleSize_;
    uint32_t betaFlag_;
    uint32_t rstdFlag_;
};
} // namespace AddRmsNormDynamicMxQuant
#endif // ADD_RMS_NORM_DYNAMIC_MX_QUANT_FP8_R_FULL_LOAD_H

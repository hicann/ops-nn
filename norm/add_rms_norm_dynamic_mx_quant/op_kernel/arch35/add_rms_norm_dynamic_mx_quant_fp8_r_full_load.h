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
#define FLOAT_OVERFLOW_MODE_CTRL 60

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

        vlForHalfNumber = platform::GetVRegSize() / sizeof(T_X);      // VL in T_X elements
        vlForFloat32Number = platform::GetVRegSize() / sizeof(float); // VL in float elements = 64
        elementAfterReduce = platform::GetVRegSize() / UB_BLOCK_SIZE; // 8

        if constexpr (IsSame<T_Y, fp8_e4m3fn_t>::value) {
            f8Emax = MX_FP8_E4M3_MAX_EXP;
            dtypeMax = MX_FP8_E4M3_MAX;
        } else {
            f8Emax = MX_FP8_E5M2_MAX_EXP;
            dtypeMax = MX_FP8_E5M2_MAX;
        }

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
        CalculateRstd(xReduceLocal, rstdLocal, curRows);
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
            RegTensor<float> var, rstd, r, y, s, t, one, scalar1;
            RegTensor<float> t1, t2, t3, t4, scalarInf, scalarZero;
            MaskReg cmpRegZero;
            MaskReg cmpRegInf;
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
                // rstd computation
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

    template <AscendC::RoundMode roundMode>
    __aicore__ inline void DynamicMxQuantPhase(LocalTensor<T_X>& yLocal, uint32_t curRows)
    {
        LocalTensor<uint16_t> maxExpLocal = maxExpBuff.Get<uint16_t>();

        uint32_t totalScaleInUB = curRows * blockNumInColAxis_;
        uint32_t totalCountInUB = curRows * blockNumInColAxis_ * mxBlockSize_;

        uint16_t loopNum = (totalCountInUB + vlForHalfNumber * DIGIT_TWO - 1) / (vlForHalfNumber * DIGIT_TWO);
        uint16_t loopNumScale = (totalScaleInUB + vlForHalfNumber - 1) / vlForHalfNumber;
        uint16_t loopNumScale4NV = (totalScaleInUB + vlForFloat32Number - 1) / vlForFloat32Number;

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
            MxQuantComputeMaxExpOCP(srcAddr, maxExpAddr, totalCountInUB, loopNum);
            MxQuantComputeScaleOCP(maxExpAddr, mxScaleLocalAddr, halfScaleLocalAddr, totalScaleInUB, loopNumScale);
        } else {
            MxQuantComputeMaxExpcuBLAS(srcAddr, maxExpAddr, totalCountInUB, loopNum);
            MxQuantComputeScalecuBLAS(maxExpAddr, mxScaleLocalAddr, halfScaleLocalAddr, totalScaleInUB, loopNumScale4NV);
        }

        srcAddr = reinterpret_cast<__ubuf__ T_X*>(yLocal.GetPhyAddr());
        halfScaleLocalAddr = reinterpret_cast<__ubuf__ uint16_t*>(halfScaleLocal.GetPhyAddr());

        MxQuantComputeData<roundMode>(srcAddr, halfScaleLocalAddr, outLocalAddr, totalCountInUB, loopNum);

        outQueueQuantY.EnQue(outLocal);
        mxScaleQueue.EnQue(mxScaleLocal);
        return;
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
                    T_X, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE, AscendC::MicroAPI::LoadDist::DIST_DINTLV_B16>(
                    vdExp0, vdExp1, srcAddr, vlForHalfNumber * DIGIT_TWO);
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
        return;
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

            AscendC::MicroAPI::RegTensor<T_X> vdExp0;
            AscendC::MicroAPI::RegTensor<T_X> vdExp1;

            AscendC::MicroAPI::MaskReg cmpResult;
            AscendC::MicroAPI::MaskReg zeroMask;
            AscendC::MicroAPI::MaskReg cmpResultSub;
            AscendC::MicroAPI::MaskReg preMaskScale;
            AscendC::MicroAPI::RegTensor<uint16_t> maxExpValue;
            AscendC::MicroAPI::Duplicate(maxExpValue, f8Emax);
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
                AscendC::MicroAPI::Compare<uint16_t, CMPMODE::NE>(cmpResult, vdMaxExp, expMask, preMaskScale); // INF/NAN
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
        return;
    }

    __aicore__ inline void MxQuantComputeMaxExpcuBLAS(
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
                    T_X, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE, AscendC::MicroAPI::LoadDist::DIST_DINTLV_B16>(
                    vdExp0, vdExp1, srcAddr, vlForHalfNumber * DIGIT_TWO);
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
                    maxExpAddr, vdMaxExp, u1, elementAfterReduce);
            }
            AscendC::MicroAPI::DataCopyUnAlignPost(maxExpAddr, u1, 0);
        }
        return;
    }

    __aicore__ inline void MxQuantComputeScalecuBLAS(
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
                // preMaskScale = AscendC::MicroAPI::UpdateMask<uint16_t>(totalScaleInUB);
                AscendC::MicroAPI::DataCopy<
                    uint16_t, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE,
                    AscendC::MicroAPI::LoadDist::DIST_UNPACK_B16>(max16, maxExpAddr, vlForFloat32Number);

                AscendC::MicroAPI::Cast<float, T_X, castTraitHalf2Float>(
                    (AscendC::MicroAPI::RegTensor<float>&)max32, (AscendC::MicroAPI::RegTensor<T_X>&)max16, preMaskScale);
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
                    mxScaleLocalAddr, expOut, vlForFloat32Number / DIGIT_TWO, maskHalf);

                AscendC::MicroAPI::ShiftLefts(extractExp, extractExp, SHR_NUM_FOR_BF16, preMaskScale);
                AscendC::MicroAPI::Sub(halfScale, scaleBias, extractExp, preMaskScale);
                AscendC::MicroAPI::Select<uint32_t>(halfScale, halfScale, nanRegTensor, cmpResult);
                AscendC::MicroAPI::Select<uint32_t>(halfScale, halfScale, zeroRegTensor32, zeroMask);
                AscendC::MicroAPI::Pack<uint16_t, uint32_t, AscendC::MicroAPI::HighLowPart::LOWEST>(recExpOut, halfScale);

                AscendC::MicroAPI::DataCopy<uint16_t, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                    halfScaleLocalAddr, recExpOut, vlForFloat32Number, maskHalf);
            }
        }
        return;
    }

    template <AscendC::RoundMode roundMode>
    __aicore__ inline void MxQuantComputeData(
        __ubuf__ T_X* srcAddr, __ubuf__ uint16_t* halfScaleLocalAddr, __ubuf__ int8_t* outLocalAddr, uint32_t totalCountInUB,
        uint16_t loopNum)
    {
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
                AscendC::MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_TRUNC};
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
                    T_X, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE, AscendC::MicroAPI::LoadDist::DIST_DINTLV_B16>(
                    vdExp0, vdExp1, srcAddr, vlForHalfNumber * DIGIT_TWO);
                AscendC::MicroAPI::DataCopy<
                    uint16_t, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE, AscendC::MicroAPI::LoadDist::DIST_E2B_B16>(
                    halfScaleForMul, halfScaleLocalAddr, elementAfterReduce);
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
                    AscendC::MicroAPI::Mul(vdExp0, vdExp0, (AscendC::MicroAPI::RegTensor<T_X>&)halfScaleForMul, dataMask1);
                    AscendC::MicroAPI::Mul(vdExp1, vdExp1, (AscendC::MicroAPI::RegTensor<T_X>&)halfScaleForMul, dataMask1);

                    AscendC::MicroAPI::Cast<float, T_X, castTraitZero>(vdExp0FP32Zero, vdExp0, dataMask1);
                    AscendC::MicroAPI::Cast<float, T_X, castTraitOne>(vdExp0FP32One, vdExp0, dataMask1);
                    AscendC::MicroAPI::Cast<float, T_X, castTraitZero>(vdExp1FP32Zero, vdExp1, dataMask2);
                    AscendC::MicroAPI::Cast<float, T_X, castTraitOne>(vdExp1FP32One, vdExp1, dataMask2);
                }  
                AscendC::MicroAPI::Cast<T_Y, float, castTrait32to80>(vdExp0FP8Zero, vdExp0FP32Zero, dataMask3);
                AscendC::MicroAPI::Cast<T_Y, float, castTrait32to82>(vdExp0FP8One, vdExp0FP32One, dataMask3);
                AscendC::MicroAPI::Cast<T_Y, float, castTrait32to81>(vdExp1FP8Zero, vdExp1FP32Zero, dataMask4);
                AscendC::MicroAPI::Cast<T_Y, float, castTrait32to83>(vdExp1FP8One, vdExp1FP32One, dataMask4);
            
                AscendC::MicroAPI::Add((AscendC::MicroAPI::RegTensor<uint8_t>&)vdExp0FP8Zero, (AscendC::MicroAPI::RegTensor<uint8_t>&)vdExp0FP8Zero, (AscendC::MicroAPI::RegTensor<uint8_t>&)vdExp0FP8One, dataMask5);
                AscendC::MicroAPI::Add((AscendC::MicroAPI::RegTensor<uint8_t>&)vdExp0FP8Zero, (AscendC::MicroAPI::RegTensor<uint8_t>&)vdExp0FP8Zero, (AscendC::MicroAPI::RegTensor<uint8_t>&)vdExp1FP8Zero, dataMask5);
                AscendC::MicroAPI::Add((AscendC::MicroAPI::RegTensor<uint8_t>&)vdExp0FP8Zero, (AscendC::MicroAPI::RegTensor<uint8_t>&)vdExp0FP8Zero, (AscendC::MicroAPI::RegTensor<uint8_t>&)vdExp1FP8One, dataMask5);

                AscendC::MicroAPI::DataCopy<
                int8_t, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE, AscendC::MicroAPI::StoreDist::DIST_NORM_B8>(
                outLocalAddr, (AscendC::MicroAPI::RegTensor<int8_t>&)vdExp0FP8Zero, OUT_ALL, dataMask5);
            }
        }
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

    uint16_t f8Emax;
    uint32_t vlForHalfNumber;
    uint32_t vlForFloat32Number;
    uint32_t dtypeMax;
    uint16_t elementAfterReduce;
    uint32_t zeroForAll = 0x00000000;
    uint32_t Exp254 = 0x000000fe;
    uint32_t halfForMan = 0x00400000;
};
} // namespace AddRmsNormDynamicMxQuant
#endif // ADD_RMS_NORM_DYNAMIC_MX_QUANT_FP8_R_FULL_LOAD_H

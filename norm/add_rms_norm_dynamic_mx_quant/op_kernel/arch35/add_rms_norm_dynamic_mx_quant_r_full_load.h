/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file add_rms_norm_dynamic_mx_quant_r_full_load.h
 * \brief Unified FULL_LOAD kernel for both FP8 and FP4 output types.
 *        IS_FP4 is derived at compile-time from T_Y via IsFP4Type trait.
 */
#ifndef ADD_RMS_NORM_DYNAMIC_MX_QUANT_R_FULL_LOAD_H
#define ADD_RMS_NORM_DYNAMIC_MX_QUANT_R_FULL_LOAD_H

#include "add_rms_norm_dynamic_mx_quant_common.h"

namespace AddRmsNormDynamicMxQuant {

template <typename T_X, typename T_GAMMA, typename T_Y, bool HAS_X3>
class AddRmsNormDynamicMxQuantRFullLoad {
public:
    __aicore__ inline AddRmsNormDynamicMxQuantRFullLoad(TPipe* pipe) { pPipe_ = pipe; }

    __aicore__ inline void Init(GM_ADDR x1, GM_ADDR x2, GM_ADDR gamma, GM_ADDR beta, GM_ADDR x3, GM_ADDR y, GM_ADDR x,
                                GM_ADDR mxscale, GM_ADDR workspace, GM_ADDR rstd,
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
        rowWork_ = (GetBlockIdx() < GetBlockNum() - 1) ? blockFactor_ : numRow_ - (GetBlockNum() - 1) * blockFactor_;
        roundMode_ = tiling->roundMode;
        mxBlockSize_ = tiling->mxBlockSize;
        scaleAlg_ = tiling->scaleAlg;
        blockNumInColAxis_ = tiling->blockNumInColAxis;
        dstStrideUbBlocks_ = tiling->dstStrideUbBlocks;
        mxScaleSize_ = tiling->mxScaleSize;
        betaFlag_ = tiling->betaFlag;
        rstdFlag_ = tiling->rstdFlag;

        uint64_t blockOffset = GetBlockIdx() * blockFactor_ * numCol_;
        x1Gm_.SetGlobalBuffer((__gm__ T_X*)x1 + blockOffset, rowWork_ * numCol_);
        x2Gm_.SetGlobalBuffer((__gm__ T_X*)x2 + blockOffset, rowWork_ * numCol_);
        if constexpr (HAS_X3) {
            x3Gm_.SetGlobalBuffer((__gm__ T_X*)x3 + blockOffset, rowWork_ * numCol_);
        }
        gammaGm_.SetGlobalBuffer((__gm__ T_GAMMA*)gamma, numCol_);
        if (betaFlag_ != 0) {
            betaGm_.SetGlobalBuffer((__gm__ T_GAMMA*)beta, numCol_);
        }
        xOutGm_.SetGlobalBuffer((__gm__ T_X*)x + blockOffset, rowWork_ * numCol_);
        if (rstdFlag_ != 0) {
            rstdGm_.SetGlobalBuffer((__gm__ float*)rstd + GetBlockIdx() * blockFactor_, blockFactor_);
        }

        if constexpr (IsFP4Type<T_Y>::value) {
            yGm_.SetGlobalBuffer((__gm__ uint8_t*)y + blockOffset / DIGIT_TWO, rowWork_ * numCol_ / DIGIT_TWO);
        } else {
            yGm_.SetGlobalBuffer((__gm__ uint8_t*)y + blockOffset, rowWork_ * numCol_);
        }
        mxScaleGm_.SetGlobalBuffer((__gm__ uint8_t*)mxscale + GetBlockIdx() * blockFactor_ * mxScaleSize_,
                                   rowWork_ * mxScaleSize_);

        uint64_t rstdUbSizeAlignSize = CeilAlign(rowFactor_, static_cast<uint64_t>(VL_F32)) * sizeof(float);
        uint16_t binaryAddQuotientLoop = CeilDiv(binAddQuotient_, VL_F32);
        uint32_t binaryAddBufLen = CeilAlign(CeilAlign(binaryAddQuotientLoop, BLOCK_F32_ALIGN_NUM) * sizeof(float),
                                             UB_BLOCK_SIZE) *
                                   rowFactor_;

        // MxQuantComputeScaleOCP loads VL_F32 uint16_t elements (128 bytes) via LoadAlign per iteration.
        // Ensure buffer can accommodate at least one such load even when blockNumInColAxis is small.
        constexpr uint64_t minScaleBufSize = static_cast<uint64_t>(VL_F32) * sizeof(uint16_t);
        uint64_t maxExpBufSize = CeilAlign(blockNumInColAxis_ * sizeof(T_X), UB_BLOCK_SIZE) * rowFactor_;
        if (maxExpBufSize < minScaleBufSize) {
            maxExpBufSize = minScaleBufSize;
        }
        uint64_t halfScaleBufSize = maxExpBufSize;

        uint64_t quantYBufSize;
        if constexpr (IsFP4Type<T_Y>::value) {
            quantYBufSize = CeilAlign(CeilDiv(numColAlign_ * rowFactor_ / DIGIT_TWO, MX_STEP_PROCESS_NUM), DIGIT_FOUR) *
                            MX_STEP_PROCESS_NUM;
        } else {
            quantYBufSize = CeilAlign(CeilDiv(numColAlign_ * rowFactor_, MX_STEP_PROCESS_NUM), DIGIT_FOUR) *
                            MX_STEP_PROCESS_NUM;
        }

        uint64_t scaleBufPerIter = CeilAlign(mxScaleSize_ * sizeof(T_X), UB_BLOCK_SIZE) * rowFactor_;

        pPipe_->InitBuffer(inQueueX1_, DOUBLE_BUFFER_NUM,
                           CeilAlign(numColAlign_ * sizeof(T_X), UB_BLOCK_SIZE) * rowFactor_);
        pPipe_->InitBuffer(inQueueX2_, DOUBLE_BUFFER_NUM,
                           CeilAlign(numColAlign_ * sizeof(T_X), UB_BLOCK_SIZE) * rowFactor_);
        if constexpr (HAS_X3) {
            pPipe_->InitBuffer(inQueueX3_, DOUBLE_BUFFER_NUM,
                               CeilAlign(numColAlign_ * sizeof(T_X), UB_BLOCK_SIZE) * rowFactor_);
        }
        if (betaFlag_ != 0) {
            pPipe_->InitBuffer(inQueueGammabeta_, 1,
                               DIGIT_TWO * CeilAlign(numCol_, UB_BLOCK_SIZE / sizeof(T_GAMMA)) * sizeof(T_GAMMA));
        } else {
            pPipe_->InitBuffer(inQueueGammabeta_, 1,
                               CeilAlign(numCol_, UB_BLOCK_SIZE / sizeof(T_GAMMA)) * sizeof(T_GAMMA));
        }
        pPipe_->InitBuffer(outQueueX_, DOUBLE_BUFFER_NUM,
                           CeilAlign(numColAlign_ * sizeof(T_X), UB_BLOCK_SIZE) * rowFactor_);
        pPipe_->InitBuffer(outQueueRstd_, DOUBLE_BUFFER_NUM, rstdUbSizeAlignSize);
        pPipe_->InitBuffer(xReduceBuff_, rstdUbSizeAlignSize);
        pPipe_->InitBuffer(xFp32Buff_, CeilAlign(numColAlign_ * sizeof(float), UB_BLOCK_SIZE) * rowFactor_);
        pPipe_->InitBuffer(binaryAddBuf_, binaryAddBufLen);

        pPipe_->InitBuffer(maxExpBuff_, maxExpBufSize);
        pPipe_->InitBuffer(halfScaleBuff_, halfScaleBufSize);
        pPipe_->InitBuffer(outQueueQuantY_, DOUBLE_BUFFER_NUM, quantYBufSize);
        pPipe_->InitBuffer(mxScaleQueue_, DOUBLE_BUFFER_NUM, scaleBufPerIter);
    }

    __aicore__ inline void Process()
    {
        LocalTensor<uint8_t> gammabetaLocal = inQueueGammabeta_.AllocTensor<uint8_t>();
        CopyInGammabeta(gammabetaLocal);
        inQueueGammabeta_.EnQue(gammabetaLocal);
        inQueueGammabeta_.DeQue<uint8_t>();

        uint32_t repeatTimes = CeilDiv(rowWork_, rowFactor_);
        for (uint32_t repeat = 0; repeat < repeatTimes; repeat++) {
            uint64_t offset = repeat * rowFactor_ * numCol_;
            uint32_t curRows = Min(rowWork_ - repeat * rowFactor_, rowFactor_);
            Compute(repeat, curRows, offset);
        }
        inQueueGammabeta_.FreeTensor(gammabetaLocal);
    }

private:
    __aicore__ inline void Compute(uint32_t rowRepeat, uint32_t curRows, uint64_t offset)
    {
        CopyInXMultiMoveAlign(offset, curRows);
        LocalTensor<T_X> xLocal1 = inQueueX1_.DeQue<T_X>();
        LocalTensor<T_X> xLocal2 = inQueueX2_.DeQue<T_X>();
        LocalTensor<T_X> xOutLocal = outQueueX_.AllocTensor<T_X>();
        LocalTensor<float> xFp32Local = xFp32Buff_.Get<float>();

        LocalTensor<T_X> xLocal0;
        if constexpr (HAS_X3) {
            xLocal0 = inQueueX3_.DeQue<T_X>();
        }

        CalculateXAdd<T_X, HAS_X3>(xLocal1, xLocal2, xOutLocal, xFp32Local, curRows * numColAlign_,
                                   HAS_X3 ? &xLocal0 : nullptr);
        inQueueX1_.FreeTensor(xLocal1);
        inQueueX2_.FreeTensor(xLocal2);
        if constexpr (HAS_X3) {
            inQueueX3_.FreeTensor(xLocal0);
        }
        outQueueX_.EnQue<T_X>(xOutLocal);

        CopyOutX(offset, curRows);

        LocalTensor<float> rstdLocal = outQueueRstd_.AllocTensor<float>();
        LocalTensor<float> xReduceLocal = xReduceBuff_.Get<float>();
        NormCommon::NormCommonRegbase::CalculateSquareReduceSum<float>(
            xFp32Local, xReduceLocal, binaryAddBuf_, static_cast<uint16_t>(curRows),
            static_cast<uint32_t>(numColAlign_), static_cast<uint32_t>(numCol_), static_cast<uint32_t>(binAddQuotient_),
            static_cast<uint32_t>(BLOCK_F32_ALIGN_NUM));

        NormCommon::ComputeRstdNewtonRaphson<true, true>(xReduceLocal, rstdLocal, curRows, epsilon_, avgFactor_,
                                                         VL_F32);
        outQueueRstd_.EnQue<float>(rstdLocal);

        rstdLocal = outQueueRstd_.DeQue<float>();
        if (rstdFlag_ != 0) {
            DataCopyExtParams rstdCopyParams{1, static_cast<uint32_t>(curRows * sizeof(float)), 0, 0, 0};
            DataCopyPad(rstdGm_[rowRepeat * rowFactor_], rstdLocal, rstdCopyParams);
        }

        LocalTensor<T_X> yLocal = outQueueX_.AllocTensor<T_X>();
        if (numCol_ != numColAlign_) {
            Duplicate<T_X>(yLocal, static_cast<T_X>(0), curRows * numColAlign_);
            PipeBarrier<PIPE_V>();
        }
        if (betaFlag_ == 1) {
            CalculateY<true>(xFp32Local, yLocal, rstdLocal, curRows);
        } else {
            CalculateY<false>(xFp32Local, yLocal, rstdLocal, curRows);
        }
        outQueueRstd_.FreeTensor(rstdLocal);
        outQueueX_.EnQue<T_X>(yLocal);
        yLocal = outQueueX_.DeQue<T_X>();

        DispatchMxQuant(yLocal, curRows);
        outQueueX_.FreeTensor(yLocal);

        CopyOutQuantY(offset, curRows);
        CopyOutMxScale(rowRepeat, curRows);
    }

    __aicore__ inline void DispatchMxQuant(LocalTensor<T_X>& yLocal, uint32_t curRows)
    {
        if constexpr (IsFP4Type<T_Y>::value) {
            if (roundMode_ == MODE_RINT) {
                MxQuantPhaseFP4<RoundMode::CAST_TRUNC, RoundMode::CAST_RINT>(yLocal, curRows);
            } else if (roundMode_ == MODE_ROUND) {
                MxQuantPhaseFP4<RoundMode::CAST_TRUNC, RoundMode::CAST_ROUND>(yLocal, curRows);
            } else if (roundMode_ == MODE_FLOOR) {
                MxQuantPhaseFP4<RoundMode::CAST_FLOOR, RoundMode::CAST_FLOOR>(yLocal, curRows);
            }
        } else {
            MxQuantPhaseFP8<RoundMode::CAST_RINT>(yLocal, curRows);
        }
    }

    template <bool hasBeta>
    __aicore__ inline void CalculateY(LocalTensor<float>& xFp32Local, LocalTensor<T_X>& yLocal,
                                      LocalTensor<float>& rstdLocal, uint32_t curRows)
    {
        uint32_t numColAlign = static_cast<uint32_t>(numColAlign_);
        uint32_t numCol = static_cast<uint32_t>(numCol_);
        __ubuf__ float* xFp32Tmp = (__ubuf__ float*)xFp32Local.GetPhyAddr();
        __ubuf__ T_GAMMA* gammaInUb = (__ubuf__ T_GAMMA*)gammaLocal_.GetPhyAddr();
        __ubuf__ T_X* yInUb = (__ubuf__ T_X*)yLocal.GetPhyAddr();
        __ubuf__ float* rstdInUb = (__ubuf__ float*)rstdLocal.GetPhyAddr();
        __ubuf__ T_GAMMA* betaInUb;
        if constexpr (hasBeta) {
            betaInUb = (__ubuf__ T_GAMMA*)betaLocal_.GetPhyAddr();
        }

        uint16_t loopRows = static_cast<uint16_t>(curRows);
        uint16_t loopCols = static_cast<uint16_t>((numCol + VL_F32 - 1) / VL_F32);
        uint16_t loopRowsFold = loopRows / DIGIT_TWO;
        uint16_t loopRowsHasLast = loopRows % DIGIT_TWO;

        __VEC_SCOPE__
        {
            RegTensor<float> x1Reg, x2Reg, gammaReg, betaReg, rstd1Reg, rstd2Reg, mul1Reg, mul1UnrollReg, mul2Reg,
                mul2UnrollReg;

            for (uint16_t i = 0; i < loopRowsFold; ++i) {
                uint32_t sregCount = numCol;
                AscendC::MicroAPI::LoadAlign<float, LoadDist::DIST_BRC_B32>(rstd1Reg, rstdInUb + DIGIT_TWO * i);
                AscendC::MicroAPI::LoadAlign<float, LoadDist::DIST_BRC_B32>(rstd2Reg, rstdInUb + (DIGIT_TWO * i + 1));
                for (uint16_t r = 0; r < loopCols; ++r) {
                    uint32_t offset1 = (DIGIT_TWO * i) * numColAlign + r * VL_F32;
                    uint32_t offset2 = (DIGIT_TWO * i + 1) * numColAlign + r * VL_F32;
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
                AscendC::MicroAPI::LoadAlign<float, LoadDist::DIST_BRC_B32>(rstd1Reg,
                                                                            rstdInUb + DIGIT_TWO * loopRowsFold);
                for (uint16_t r = 0; r < loopCols; ++r) {
                    uint32_t offset = (DIGIT_TWO * loopRowsFold) * numColAlign + r * VL_F32;
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

    template <AscendC::RoundMode roundMode>
    __aicore__ inline void MxQuantPhaseFP8(LocalTensor<T_X>& yLocal, uint32_t curRows)
    {
        LocalTensor<uint16_t> maxExpLocal = maxExpBuff_.Get<uint16_t>();

        uint32_t totalScaleInUB = curRows * blockNumInColAxis_;
        uint32_t totalCountInUB = curRows * blockNumInColAxis_ * mxBlockSize_;

        uint16_t loopNum = (totalCountInUB + VL_B16 * DIGIT_TWO - 1) / (VL_B16 * DIGIT_TWO);
        uint16_t loopNumScale = (totalScaleInUB + VL_B16 - 1) / VL_B16;
        uint16_t loopNumScale4NV = (totalScaleInUB + VL_F32 - 1) / VL_F32;

        auto srcAddr = reinterpret_cast<__ubuf__ T_X*>(yLocal.GetPhyAddr());
        auto maxExpAddr = reinterpret_cast<__ubuf__ uint16_t*>(maxExpLocal.GetPhyAddr());

        LocalTensor<uint16_t> mxScaleLocal = mxScaleQueue_.AllocTensor<uint16_t>();
        auto mxScaleLocalAddr = reinterpret_cast<__ubuf__ uint16_t*>(mxScaleLocal.GetPhyAddr());

        LocalTensor<uint16_t> halfScaleLocal = halfScaleBuff_.Get<uint16_t>();
        auto halfScaleLocalAddr = reinterpret_cast<__ubuf__ uint16_t*>(halfScaleLocal.GetPhyAddr());

        LocalTensor<int8_t> outLocal = outQueueQuantY_.AllocTensor<int8_t>();
        auto outLocalAddr = reinterpret_cast<__ubuf__ int8_t*>(outLocal.GetPhyAddr());
        maxExpAddr = reinterpret_cast<__ubuf__ uint16_t*>(maxExpLocal.GetPhyAddr());
        if (scaleAlg_ == 0) {
            MxQuantComputeMaxExpOCP<T_X>(srcAddr, maxExpAddr, loopNum);
            MxQuantComputeScaleOCP<T_Y>(maxExpAddr, mxScaleLocalAddr, halfScaleLocalAddr, totalScaleInUB, loopNumScale);
        } else {
            MxQuantComputeMaxExpcuBLAS<T_X>(srcAddr, maxExpAddr, loopNum);
            MxQuantComputeScalecuBLAS<T_X, T_Y>(maxExpAddr, mxScaleLocalAddr, halfScaleLocalAddr, totalScaleInUB,
                                                loopNumScale4NV);
        }

        srcAddr = reinterpret_cast<__ubuf__ T_X*>(yLocal.GetPhyAddr());
        halfScaleLocalAddr = reinterpret_cast<__ubuf__ uint16_t*>(halfScaleLocal.GetPhyAddr());

        MxQuantComputeData<roundMode, T_X, T_Y>(srcAddr, halfScaleLocalAddr, outLocalAddr, loopNum);

        outQueueQuantY_.EnQue(outLocal);
        mxScaleQueue_.EnQue(mxScaleLocal);
    }

    template <AscendC::RoundMode toBf16RoundMode, AscendC::RoundMode roundMode>
    __aicore__ inline void MxQuantPhaseFP4(LocalTensor<T_X>& yLocal, uint32_t curRows)
    {
        LocalTensor<uint16_t> maxExpLocal = maxExpBuff_.Get<uint16_t>();

        uint32_t totalScaleInUB = curRows * blockNumInColAxis_;
        uint32_t totalCountInUB = curRows * blockNumInColAxis_ * mxBlockSize_;

        uint16_t loopNum = (totalCountInUB + VL_B16 * DIGIT_TWO - 1) / (VL_B16 * DIGIT_TWO);
        uint16_t loopNumScale = (totalScaleInUB + VL_B16 - 1) / VL_B16;

        auto srcAddr = reinterpret_cast<__ubuf__ T_X*>(yLocal.GetPhyAddr());
        auto maxExpAddr = reinterpret_cast<__ubuf__ uint16_t*>(maxExpLocal.GetPhyAddr());

        LocalTensor<uint16_t> mxScaleLocal = mxScaleQueue_.AllocTensor<uint16_t>();
        auto mxScaleLocalAddr = reinterpret_cast<__ubuf__ uint16_t*>(mxScaleLocal.GetPhyAddr());

        LocalTensor<uint16_t> halfScaleLocal = halfScaleBuff_.Get<uint16_t>();
        auto halfScaleLocalAddr = reinterpret_cast<__ubuf__ uint16_t*>(halfScaleLocal.GetPhyAddr());

        LocalTensor<int8_t> outLocal = outQueueQuantY_.AllocTensor<int8_t>();
        auto outLocalAddr = reinterpret_cast<__ubuf__ int8_t*>(outLocal.GetPhyAddr());

        maxExpAddr = reinterpret_cast<__ubuf__ uint16_t*>(maxExpLocal.GetPhyAddr());
        MxQuantComputeMaxExpOCP<T_X>(srcAddr, maxExpAddr, loopNum);
        MxQuantComputeScaleOCP<T_Y>(maxExpAddr, mxScaleLocalAddr, halfScaleLocalAddr, totalScaleInUB, loopNumScale);

        srcAddr = reinterpret_cast<__ubuf__ T_X*>(yLocal.GetPhyAddr());
        halfScaleLocalAddr = reinterpret_cast<__ubuf__ uint16_t*>(halfScaleLocal.GetPhyAddr());

        MxQuantComputeDataFP4<toBf16RoundMode, roundMode, T_X, T_Y>(srcAddr, halfScaleLocalAddr, outLocalAddr,
                                                                    totalCountInUB, loopNum);

        outQueueQuantY_.EnQue(outLocal);
        mxScaleQueue_.EnQue(mxScaleLocal);
    }

    __aicore__ inline void CopyInXMultiMoveAlign(uint64_t offset, uint32_t curRows)
    {
        LocalTensor<T_X> xLocal1 = inQueueX1_.AllocTensor<T_X>();
        LocalTensor<T_X> xLocal2 = inQueueX2_.AllocTensor<T_X>();

        DataCopyExtParams extParams{static_cast<uint16_t>(curRows), static_cast<uint32_t>(numCol_ * sizeof(T_X)),
                                    static_cast<uint32_t>(0), static_cast<uint32_t>(dstStrideUbBlocks_), 0};
        DataCopyPadExtParams<T_X> padParams{false, static_cast<uint8_t>(0), static_cast<uint8_t>(0),
                                            static_cast<T_X>(0.0)};

        DataCopyPad(xLocal1, x1Gm_[offset], extParams, padParams);
        DataCopyPad(xLocal2, x2Gm_[offset], extParams, padParams);
        inQueueX1_.EnQue(xLocal1);
        inQueueX2_.EnQue(xLocal2);

        if constexpr (HAS_X3) {
            LocalTensor<T_X> xLocal0 = inQueueX3_.AllocTensor<T_X>();
            DataCopyPad(xLocal0, x3Gm_[offset], extParams, padParams);
            inQueueX3_.EnQue(xLocal0);
        }
    }

    __aicore__ inline void CopyInGammabeta(LocalTensor<uint8_t> gammabetaLocal)
    {
        DataCopyExtParams copyParams{static_cast<uint16_t>(1), static_cast<uint32_t>(numCol_ * sizeof(T_GAMMA)),
                                     static_cast<uint32_t>(0), static_cast<uint32_t>(0), 0};
        DataCopyPadExtParams<T_GAMMA> padParams{false, static_cast<uint8_t>(0), static_cast<uint8_t>(0),
                                                static_cast<T_GAMMA>(0.0)};
        gammaLocal_ = gammabetaLocal.ReinterpretCast<T_GAMMA>();
        DataCopyPad<T_GAMMA>(gammaLocal_, gammaGm_, copyParams, padParams);
        if (betaFlag_ != 0) {
            betaLocal_ = gammabetaLocal[CeilAlign(numCol_, UB_BLOCK_SIZE / sizeof(T_GAMMA)) * sizeof(T_GAMMA)]
                             .ReinterpretCast<T_GAMMA>();
            DataCopyPad<T_GAMMA>(betaLocal_, betaGm_, copyParams, padParams);
        }
    }

    __aicore__ inline void CopyOutX(uint64_t offset, uint32_t curRows)
    {
        LocalTensor<T_X> xLocal = outQueueX_.DeQue<T_X>();
        uint32_t srcStride = (numColAlign_ - numCol_) * sizeof(T_X) / UB_BLOCK_SIZE;
        DataCopyExtParams copyParams{static_cast<uint16_t>(curRows), static_cast<uint32_t>(numCol_ * sizeof(T_X)),
                                     static_cast<uint32_t>(srcStride), static_cast<uint32_t>(0), 0};
        DataCopyPad(xOutGm_[offset], xLocal, copyParams);
        outQueueX_.FreeTensor(xLocal);
    }

    __aicore__ inline void CopyOutQuantY(uint64_t gmOffset, uint32_t curRows)
    {
        LocalTensor<uint8_t> quantYLocal = outQueueQuantY_.DeQue<uint8_t>();
        if constexpr (IsFP4Type<T_Y>::value) {
            uint32_t bytesPerRow = numCol_ / DIGIT_TWO;
            uint32_t alignBytesPerRow = numColAlign_ / DIGIT_TWO;
            uint32_t srcStride = (alignBytesPerRow - bytesPerRow) / UB_BLOCK_SIZE;
            DataCopyExtParams copyParams{static_cast<uint16_t>(curRows), static_cast<uint32_t>(bytesPerRow),
                                         static_cast<uint32_t>(srcStride), static_cast<uint32_t>(0), 0};
            DataCopyPad<uint8_t>(yGm_[gmOffset / DIGIT_TWO], quantYLocal, copyParams);
        } else {
            uint32_t srcStride = (numColAlign_ - numCol_) * sizeof(uint8_t) / UB_BLOCK_SIZE;
            DataCopyExtParams copyParams{static_cast<uint16_t>(curRows), static_cast<uint32_t>(numCol_),
                                         static_cast<uint32_t>(srcStride), static_cast<uint32_t>(0), 0};
            DataCopyPad<uint8_t>(yGm_[gmOffset], quantYLocal, copyParams);
        }
        outQueueQuantY_.FreeTensor(quantYLocal);
    }

    __aicore__ inline void CopyOutMxScale(uint32_t rowRepeat, uint32_t curRows)
    {
        LocalTensor<uint8_t> mxScaleLocal = mxScaleQueue_.DeQue<uint8_t>();
        DataCopyExtParams copyParams{static_cast<uint16_t>(curRows), static_cast<uint32_t>(mxScaleSize_),
                                     static_cast<uint32_t>(0), static_cast<uint32_t>(0), 0};
        DataCopyPad<uint8_t, PaddingMode::Compact>(mxScaleGm_[rowRepeat * rowFactor_ * mxScaleSize_], mxScaleLocal,
                                                   copyParams);
        mxScaleQueue_.FreeTensor(mxScaleLocal);
    }

private:
    TPipe* pPipe_ = nullptr;

    TQue<QuePosition::VECIN, 1> inQueueX1_;
    TQue<QuePosition::VECIN, 1> inQueueX2_;
    TQue<QuePosition::VECIN, 1> inQueueX3_;
    TQue<QuePosition::VECIN, 1> inQueueGammabeta_;
    TQue<QuePosition::VECOUT, 1> outQueueX_;
    TQue<QuePosition::VECOUT, 1> outQueueRstd_;
    TBuf<TPosition::VECCALC> xReduceBuff_;
    TBuf<TPosition::VECCALC> xFp32Buff_;
    TBuf<TPosition::VECCALC> binaryAddBuf_;
    TBuf<TPosition::VECCALC> maxExpBuff_;
    TBuf<TPosition::VECCALC> halfScaleBuff_;
    TQue<QuePosition::VECOUT, 1> outQueueQuantY_;
    TQue<QuePosition::VECOUT, 1> mxScaleQueue_;

    LocalTensor<T_GAMMA> gammaLocal_;
    LocalTensor<T_GAMMA> betaLocal_;

    GlobalTensor<T_X> x1Gm_;
    GlobalTensor<T_X> x2Gm_;
    GlobalTensor<T_X> x3Gm_;
    GlobalTensor<T_GAMMA> gammaGm_;
    GlobalTensor<T_GAMMA> betaGm_;
    GlobalTensor<T_X> xOutGm_;
    GlobalTensor<float> rstdGm_;
    GlobalTensor<uint8_t> yGm_;
    GlobalTensor<uint8_t> mxScaleGm_;

    uint64_t numRow_{0};
    uint64_t numCol_{0};
    uint64_t numColAlign_{0};
    uint64_t blockFactor_{0};
    uint64_t rowFactor_{0};
    uint64_t binAddQuotient_{0};
    float epsilon_{1e-6};
    float avgFactor_{0.0f};
    uint64_t rowWork_{1};
    uint64_t roundMode_{4};
    uint64_t mxBlockSize_{32};
    int64_t scaleAlg_{0};
    uint64_t blockNumInColAxis_{0};
    uint64_t dstStrideUbBlocks_{0};
    uint64_t mxScaleSize_{0};
    uint32_t betaFlag_{0};
    uint32_t rstdFlag_{0};
};
} // namespace AddRmsNormDynamicMxQuant
#endif // ADD_RMS_NORM_DYNAMIC_MX_QUANT_R_FULL_LOAD_H

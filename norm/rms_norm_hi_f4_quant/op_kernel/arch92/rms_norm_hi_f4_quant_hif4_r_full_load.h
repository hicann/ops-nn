/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !\
 * \file rms_norm_hi_f4_quant_hif4_r_full_load.h
 * \brief Fused RmsNorm + DynamicHiF4Quant kernel for R-axis full load
 *
 * RmsNorm part: x² reduced to variance, then rstd
 * HiF4 Quant part: inherited from DynamicHiF4QuantTailAxis
 */
#ifndef RMS_NORM_HI_F4_QUANT_HIF4_R_FULL_LOAD_H
#define RMS_NORM_HI_F4_QUANT_HIF4_R_FULL_LOAD_H

#include "rms_norm_hi_f4_quant_common.h"

namespace RmsNormHiF4Quant {

template <typename T_X, typename T_GAMMA>
class RmsNormHiF4QuantHif4RFullLoad {
public:
    __aicore__ inline RmsNormHiF4QuantHif4RFullLoad(TPipe* pipe) { pPipe = pipe; }

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR gamma, GM_ADDR beta, GM_ADDR y, GM_ADDR scale, GM_ADDR rstd,
                                GM_ADDR workspace, const RmsNormHiF4QuantTilingData* tiling)
    {
#if (__NPU_ARCH__ == 3510 || __NPU_ARCH__ == 9201 || __CCE_AICORE__ == 920)
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
        mxBlockSize_ = tiling->mxBlockSize;
        blockNumInColAxis_ = tiling->blockNumInColAxis;
        mxScaleSize_ = tiling->mxScaleSize;
        betaFlag_ = tiling->betaFlag;
        rstdFlag_ = tiling->rstdFlag;
        scaleAlg_ = tiling->scaleAlg;
        innerLoopEle_ = tiling->innerLoopEle;
        max8LoopEle_ = tiling->max8LoopEle;
        scaleLoopEle_ = tiling->scaleLoopEle;
        outLoopEle_ = tiling->outLoopEle;
        alignGroupNum_ = tiling->alignGroupNum;
        innerLoopTail_ = tiling->innerLoopTail;
        max8LoopTail_ = tiling->max8LoopTail;
        scaleLoopTail_ = tiling->scaleLoopTail;
        outLoopTail_ = tiling->outLoopTail;

        rowWork = (GetBlockIdx() < GetBlockNum() - 1) ? blockFactor_ : numRow_ - (GetBlockNum() - 1) * blockFactor_;

        vlForHalfNumber = platform::GetVRegSize() / sizeof(T_X);
        elementAfterReduce = platform::GetVRegSize() / UB_BLOCK_SIZE;

        uint64_t blockOffset = GetBlockIdx() * blockFactor_ * numCol_;
        xGm.SetGlobalBuffer((__gm__ T_X*)x + blockOffset, rowWork * numCol_);
        gammaGm.SetGlobalBuffer((__gm__ T_GAMMA*)gamma, numCol_);
        if (betaFlag_ != 0) {
            betaGm.SetGlobalBuffer((__gm__ T_GAMMA*)beta, numCol_);
        }
        if (rstdFlag_ != 0) {
            rstdGm.SetGlobalBuffer((__gm__ float*)rstd + GetBlockIdx() * blockFactor_, blockFactor_);
        }
        yFp4Gm.SetGlobalBuffer((__gm__ uint8_t*)y + blockOffset / DIGIT_TWO, rowWork * numCol_ / DIGIT_TWO);
        scaleGm.SetGlobalBuffer((__gm__ float*)scale + GetBlockIdx() * blockFactor_ * mxScaleSize_,
                                rowWork * mxScaleSize_);

        uint64_t rstdUbSizeAlignSize = CeilAlign(rowFactor_, static_cast<uint64_t>(VL_F32)) * sizeof(float);
        uint16_t binaryAddQuotientLoop = CeilDiv(binAddQuotient_, VL_F32);
        uint32_t binaryAddBufLen = CeilAlign(CeilAlign(binaryAddQuotientLoop, BLOCK_F32_ALIGN_NUM) * sizeof(float),
                                             UB_BLOCK_SIZE) *
                                   rowFactor_;

        pPipe->InitBuffer(inQueueX, DOUBLE_BUFFER_NUM,
                          CeilAlign(numColAlign_ * sizeof(T_X), UB_BLOCK_SIZE) * rowFactor_);
        if (betaFlag_ != 0) {
            pPipe->InitBuffer(inQueueGammabeta, 1,
                              DIGIT_TWO * CeilAlign(numCol_, UB_BLOCK_SIZE / sizeof(T_GAMMA)) * sizeof(T_GAMMA));
        } else {
            pPipe->InitBuffer(inQueueGammabeta, 1,
                              CeilAlign(numCol_, UB_BLOCK_SIZE / sizeof(T_GAMMA)) * sizeof(T_GAMMA));
        }
        pPipe->InitBuffer(outQueueRstd, DOUBLE_BUFFER_NUM, rstdUbSizeAlignSize);
        pPipe->InitBuffer(xReduceBuff, rstdUbSizeAlignSize);
        pPipe->InitBuffer(xFp32Buff, CeilAlign(numColAlign_ * sizeof(float), UB_BLOCK_SIZE) * rowFactor_);
        pPipe->InitBuffer(binaryAddBuf, binaryAddBufLen);

        int64_t totalGroups = rowFactor_ * blockNumInColAxis_;
        int64_t alignTotalGroups = CeilAlign(totalGroups, static_cast<int64_t>(DQ_HIFP4_BLOCK_BYTE));

        bufE8Buffer_ = CeilAlign(totalGroups, static_cast<int64_t>(DQ_HIFP4_BLOCK_BYTE));
        buf1E8Buffer_ = CeilAlign(totalGroups, static_cast<int64_t>(DQ_HIFP4_BLOCK_BYTE));
        buf1E16Buffer_ = CeilAlign(totalGroups, static_cast<int64_t>(DQ_HIFP4_BLOCK_BYTE));
        maxPer4Buffer_ = CeilAlign(totalGroups, static_cast<int64_t>(DQ_HIFP4_MAXPER4_ALIGN));
        maxPer8Buffer_ = CeilAlign(totalGroups, static_cast<int64_t>(DQ_HIFP4_MAXPER8_ALIGN));
        tmpScaleBuffer_ = CeilAlign(totalGroups, static_cast<int64_t>(DQ_HIFP4_SCALETMP_ALIGN));

        pPipe->InitBuffer(bufE8_, bufE8Buffer_ * sizeof(int16_t));
        pPipe->InitBuffer(buf1E8_, buf1E8Buffer_ * sizeof(int32_t));
        pPipe->InitBuffer(buf1E16_, buf1E16Buffer_ * sizeof(int16_t));
        pPipe->InitBuffer(bufMaxPer4_, maxPer4Buffer_ * 16 * sizeof(int16_t));
        pPipe->InitBuffer(bufMaxPer8_, maxPer8Buffer_ * 8 * sizeof(int16_t));
        pPipe->InitBuffer(bufTmpScale_, tmpScaleBuffer_ * 32 * sizeof(int16_t));

        uint64_t quantYBufSize = CeilAlign(CeilDiv(numColAlign_ * rowFactor_ / DIGIT_TWO, MX_STEP_PROCESS_NUM),
                                           DIGIT_FOUR) *
                                 MX_STEP_PROCESS_NUM;
        int64_t alignTo256UbFormer = CeilAlign(alignTotalGroups, static_cast<int64_t>(DQ_HIFP4_SCALE_ALIGN));
        pPipe->InitBuffer(outQueueQuantY, DOUBLE_BUFFER_NUM, quantYBufSize);
        pPipe->InitBuffer(outQueueScale, DOUBLE_BUFFER_NUM, alignTo256UbFormer * sizeof(float));

        sysRegLen_ = 256;
        gmCopyInRegStep_ = sysRegLen_ / sizeof(float) * 2;
        gmCopyOutRegStep_ = sysRegLen_ / sizeof(int32_t) / DQ_HIFP4_CAST_COPYOUT_PACK4_NUM;
        gmCopyOutScaleRegStep_ = sysRegLen_ / sizeof(float);
        gmCopyInRegStep2_ = sysRegLen_ / sizeof(T_X);
        gmCopyTmpE1_ = sysRegLen_ * DIGIT_FOUR / HIF4_BLOCK_SIZE / DIGIT_TWO;
        gmCopyTmpE8_ = sysRegLen_ * DIGIT_FOUR / DIGIT_EIGHT;
        gmCopyTmpE16_ = sysRegLen_ * DIGIT_FOUR / DIGIT_FOUR;
        gmCopyTmpBufE8Step_ = sysRegLen_ * DIGIT_TWO;
        gmCopyTmpBufOtherStep_ = sysRegLen_;
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
        CopyInX(offset, curRows);
        LocalTensor<T_X> xLocal = inQueueX.DeQue<T_X>();
        LocalTensor<float> xFp32Local = xFp32Buff.Get<float>();

        CalculateCopyInToFP32(xLocal, xFp32Local, static_cast<uint32_t>(curRows * numColAlign_));
        inQueueX.FreeTensor(xLocal);

        LocalTensor<float> rstdLocal = outQueueRstd.AllocTensor<float>();
        LocalTensor<float> xReduceLocal = xReduceBuff.Get<float>();
        CalculateSquareReduceSum(xFp32Local, xReduceLocal, curRows);

        CalculateRstd(xReduceLocal, rstdLocal, curRows, avgFactor_, epsilon_);
        outQueueRstd.EnQue<float>(rstdLocal);

        rstdLocal = outQueueRstd.DeQue<float>();
        if (rstdFlag_ != 0) {
            DataCopyExtParams rstdCopyParams{1, static_cast<uint32_t>(curRows * sizeof(float)), 0, 0, 0};
            DataCopyPad(rstdGm[rowRepeat * rowFactor_], rstdLocal, rstdCopyParams);
        }

        LocalTensor<T_X> yLocal = inQueueX.AllocTensor<T_X>();
        if (betaFlag_ == 1) {
            CalculateY<true>(xFp32Local, yLocal, rstdLocal, curRows);
        } else {
            CalculateY<false>(xFp32Local, yLocal, rstdLocal, curRows);
        }
        outQueueRstd.FreeTensor(rstdLocal);

        int64_t curHif4Groups = curRows * blockNumInColAxis_;
        int64_t hif4InnerLoop, hif4Max8Loop, hif4ScaleLoop, hif4OutLoop;
        if (curRows >= rowFactor_) {
            hif4InnerLoop = innerLoopEle_;
            hif4Max8Loop = max8LoopEle_;
            hif4ScaleLoop = scaleLoopEle_;
            hif4OutLoop = outLoopEle_;
        } else {
            hif4InnerLoop = (curHif4Groups + HIF4_GROUP_PER_INNER_LOOP - 1) / HIF4_GROUP_PER_INNER_LOOP;
            hif4Max8Loop = (curHif4Groups + HIF4_GROUP_PER_MAX8_LOOP - 1) / HIF4_GROUP_PER_MAX8_LOOP;
            hif4ScaleLoop = (curHif4Groups + HIF4_GROUP_PER_SCALE_LOOP - 1) / HIF4_GROUP_PER_SCALE_LOOP;
            hif4OutLoop = (curHif4Groups + HIF4_GROUP_PER_OUT_LOOP - 1) / HIF4_GROUP_PER_OUT_LOOP;
        }
        curHif4Groups_ = curHif4Groups;

        HiF4Quant(yLocal, curRows, hif4InnerLoop, hif4Max8Loop, hif4ScaleLoop, hif4OutLoop);
        inQueueX.FreeTensor(yLocal);

        CopyOutQuantYFP4(offset, curRows);
        CopyOutScale(rowRepeat, curRows);
    }

    __aicore__ inline void CalculateSquareReduceSum(LocalTensor<float>& xFp32Local, LocalTensor<float>& xReduceLocal,
                                                    uint32_t curRows)
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

    __aicore__ inline void CalculateSquareReduceSumLessThanVL(__local_mem__ float* xFp32Tmp,
                                                              __local_mem__ float* xReduceUb, uint16_t curRows)
    {
        uint32_t numCol = static_cast<uint32_t>(numCol_);
        __VEC_SCOPE__
        {
            RegTensor<float> x;
            RegTensor<float> vMean;

            uint32_t sreg0 = numCol;
            MaskReg pregLoop = UpdateMask<float>(sreg0);
            MaskReg pregOne = CreateMask<float, MaskPattern::VL1>();

            for (uint16_t i = 0; i < curRows; i++) {
                LoadTensorForDtypeTIn<float>(xFp32Tmp, x, pregLoop, i * numColAlign_);
                AscendC::MicroAPI::Mul(x, x, x, pregLoop);
                AscendC::MicroAPI::ReduceSum(vMean, x, pregLoop);
                AscendC::MicroAPI::DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(xReduceUb + i, vMean, pregOne);
            }
        }
    }

    __aicore__ inline void CalculateSquareReduceSumLessThanTwoVL(__local_mem__ float* xFp32Tmp,
                                                                 __local_mem__ float* xReduceUb, uint16_t curRows)
    {
        uint32_t tailLen = static_cast<uint32_t>(numCol_) - VL_F32;
        __VEC_SCOPE__
        {
            RegTensor<float> x;
            RegTensor<float> xFold;
            RegTensor<float> sumReg;
            RegTensor<float> vMean;

            MaskReg pregFull = CreateMask<float, MaskPattern::ALL>();
            MaskReg pregOne = CreateMask<float, MaskPattern::VL1>();
            MaskReg pregTail = UpdateMask<float>(tailLen);

            for (uint16_t i = 0; i < curRows; ++i) {
                LoadTensorForDtypeTIn<float>(xFp32Tmp, x, pregFull, i * numColAlign_);
                LoadTensorForDtypeTIn<float>(xFp32Tmp + VL_F32, xFold, pregTail, i * numColAlign_);
                AscendC::MicroAPI::Mul(x, x, x, pregFull);
                AscendC::MicroAPI::Mul(xFold, xFold, xFold, pregTail);
                AscendC::MicroAPI::ShiftLefts((RegTensor<uint32_t>&)xFold, (RegTensor<uint32_t>&)xFold,
                                              static_cast<int16_t>(0), pregTail);
                AscendC::MicroAPI::Add(sumReg, x, xFold, pregFull);
                AscendC::MicroAPI::ReduceSum(vMean, sumReg, pregFull);
                AscendC::MicroAPI::DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(xReduceUb + i, vMean, pregOne);
            }
        }
    }

    template <int32_t LAST_LOOP_NUMS>
    __aicore__ inline void CalculateSquareReduceSumCommon(__local_mem__ float* xFp32Tmp, __local_mem__ float* xReduceUb,
                                                          __local_mem__ float* tmpUb, uint16_t curRows)
    {
        uint32_t binaryAddQuotient = static_cast<uint32_t>(binAddQuotient_);
        uint16_t binaryAddQuotientLoop = (binaryAddQuotient + VL_F32 - 1) / VL_F32;
        uint32_t lastBinaryAddNum = binaryAddQuotient / VL_F32;
        uint32_t lastBinaryAddNumAlign = (binaryAddQuotientLoop + BLOCK_F32_ALIGN_NUM - 1) / BLOCK_F32_ALIGN_NUM *
                                         BLOCK_F32_ALIGN_NUM;
        uint32_t binaryAddRemainder = static_cast<uint32_t>(numCol_) - binaryAddQuotient;
        uint16_t binaryAddRemainderCeilLoop = (binaryAddRemainder + VL_F32 - 1) / VL_F32;
        uint16_t binaryAddRemainderFloorLoop = binaryAddRemainder / VL_F32;
        __VEC_SCOPE__
        {
            RegTensor<float> x;
            RegTensor<float> xFold;
            RegTensor<float> sumReg;
            RegTensor<float> vMean;
            MaskReg pregFull = CreateMask<float, MaskPattern::ALL>();
            MaskReg pregOne = CreateMask<float, MaskPattern::VL1>();
            MaskReg pregLoop;

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
                    LoadTensorForDtypeTIn<float>(xFp32Tmp + binaryAddRemainderFloorLoop * VL_F32 + binaryAddQuotient,
                                                 xFold, pregLoop, off);
                    AscendC::MicroAPI::Mul(x, x, x, pregFull);
                    AscendC::MicroAPI::Mul(xFold, xFold, xFold, pregLoop);
                    AscendC::MicroAPI::ShiftLefts((RegTensor<uint32_t>&)xFold, (RegTensor<uint32_t>&)xFold,
                                                  static_cast<int16_t>(0), pregLoop);
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
                    AscendC::MicroAPI::DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(xReduceUb + i, vMean,
                                                                                          pregOne);
                }
            } else if constexpr (LAST_LOOP_NUMS == 2) {
                lastBinaryAddNum -= VL_F32;
                MaskReg pregLast = UpdateMask<float>(lastBinaryAddNum);
                for (uint16_t i = 0; i < curRows; ++i) {
                    AscendC::MicroAPI::DataCopy(x, tmpUb + static_cast<uint32_t>(i * lastBinaryAddNumAlign));
                    AscendC::MicroAPI::DataCopy(xFold,
                                                tmpUb + static_cast<uint32_t>(i * lastBinaryAddNumAlign + VL_F32));
                    AscendC::MicroAPI::ShiftLefts((RegTensor<uint32_t>&)xFold, (RegTensor<uint32_t>&)xFold,
                                                  static_cast<int16_t>(0), pregLast);
                    AscendC::MicroAPI::Add(sumReg, x, xFold, pregFull);
                    AscendC::MicroAPI::ReduceSum(vMean, sumReg, pregFull);
                    AscendC::MicroAPI::DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(xReduceUb + i, vMean,
                                                                                          pregOne);
                }
            }
        }
    }

    template <bool hasBeta>
    __aicore__ inline void CalculateY(LocalTensor<float>& xFp32Local, LocalTensor<T_X>& yLocal,
                                      LocalTensor<float>& rstdLocal, uint32_t curRows)
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

    // ======================== HiF4 Quantization (from DynamicHiF4QuantTailAxis) ========================

    __aicore__ inline void HiF4Quant(LocalTensor<T_X>& yLocal, uint32_t curRows, int64_t innerLoop, int64_t max8Loop,
                                     int64_t scaleLoop, int64_t outLoop)
    {
        LocalTensor<uint16_t> e8buf = bufE8_.Get<uint16_t>();
        LocalTensor<uint8_t> e1p8buf = buf1E8_.Get<uint8_t>();
        LocalTensor<uint8_t> e1p16buf = buf1E16_.Get<uint8_t>();
        LocalTensor<uint16_t> tmpscale = bufTmpScale_.Get<uint16_t>();
        LocalTensor<uint16_t> maxper4 = bufMaxPer4_.Get<uint16_t>();
        LocalTensor<uint16_t> maxper8 = bufMaxPer8_.Get<uint16_t>();

        auto xAddr1 = (__ubuf__ float*)yLocal.GetPhyAddr();
        auto xAddr2 = (__ubuf__ T_X*)yLocal.GetPhyAddr();

        LocalTensor<uint8_t> quantYLocal = outQueueQuantY.AllocTensor<uint8_t>();
        auto yAddr = (__ubuf__ int32_t*)quantYLocal.GetPhyAddr();

        LocalTensor<float> scaleLocal = outQueueScale.AllocTensor<float>();
        auto scaleAddr = (__ubuf__ uint16_t*)scaleLocal.GetPhyAddr();

        auto e8BufAddr = (__ubuf__ uint16_t*)e8buf.GetPhyAddr();
        auto e8BufAddr2 = (__ubuf__ uint8_t*)e8buf.GetPhyAddr();
        auto e1p8BufAddr = (__ubuf__ uint32_t*)e1p8buf.GetPhyAddr();
        auto e1p16BufAddr = (__ubuf__ uint8_t*)e1p16buf.GetPhyAddr();
        auto tmpScaleAddr = (__ubuf__ uint16_t*)tmpscale.GetPhyAddr();
        auto maxPer4BufAddr = (__ubuf__ uint16_t*)maxper4.GetPhyAddr();
        auto maxPer8BufAddr = (__ubuf__ uint16_t*)maxper8.GetPhyAddr();

        auto e8BufRead1Addr = e8BufAddr;
        auto e8BufRead2Addr = e8BufAddr2;
        auto e1p8BufReadAddr = (__ubuf__ uint8_t*)e1p8buf.GetPhyAddr();
        auto e1p16BufReadAddr = e1p16BufAddr;
        auto tmpScaleReadAddr = tmpScaleAddr;
        auto maxPer4BufReadAddr = maxPer4BufAddr;
        auto maxPer4BufRead2Addr = maxPer4BufAddr;
        auto maxPer8BufReadAddr = maxPer8BufAddr;

        ComputeMaxPer4(static_cast<uint16_t>(innerLoop), xAddr1, maxPer4BufAddr);
        ComputeMaxPer8(static_cast<uint16_t>(max8Loop), maxPer4BufReadAddr, e8BufAddr, maxPer8BufAddr);
        ComputeTmpScale(static_cast<uint16_t>(innerLoop), maxPer4BufRead2Addr, maxPer8BufReadAddr, e8BufRead1Addr,
                        e1p8BufAddr, e1p16BufAddr, tmpScaleAddr);
        ComputeScale(static_cast<uint16_t>(scaleLoop), e8BufRead2Addr, e1p8BufReadAddr, e1p16BufReadAddr, scaleAddr);
        ComputeOut(static_cast<uint16_t>(outLoop), xAddr2, tmpScaleReadAddr, yAddr);

        outQueueQuantY.EnQue(quantYLocal);
        outQueueScale.EnQue(scaleLocal);
    }

    __aicore__ inline void ComputeMaxPer4(uint16_t regLoop, __ubuf__ float* xAddr1, __ubuf__ uint16_t* maxPer4BufAddr)
    {
        __VEC_SCOPE__
        {
            MicroAPI::RegTensor<T_X> vregIn1;
            MicroAPI::RegTensor<T_X> vregIn2;
            MicroAPI::RegTensor<T_X> vregIn3;
            MicroAPI::RegTensor<T_X> vregIn4;
            MicroAPI::RegTensor<T_X> vregMax1;
            MicroAPI::RegTensor<T_X> vregMax2;
            MicroAPI::RegTensor<T_X> vregOut;
            MicroAPI::RegTensor<bfloat16_t> vregOutbf16;
            AscendC::MicroAPI::RegTensor<uint16_t> expMaskBF16;
            AscendC::MicroAPI::Duplicate(expMaskBF16, 0x7fff);
            MicroAPI::MaskReg maskAll = MicroAPI::CreateMask<uint8_t>();
            static constexpr AscendC::MicroAPI::CastTrait castTraitHalf2Bf16 = {
                AscendC::MicroAPI::RegLayout::UNKNOWN, AscendC::MicroAPI::SatMode::UNKNOWN,
                AscendC::MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
            for (uint16_t loopIdx = 0; loopIdx < regLoop; loopIdx++) {
                MicroAPI::DataCopy<float, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::LoadDist::DIST_DINTLV_B32>(
                    (MicroAPI::RegTensor<float>&)vregIn1, (MicroAPI::RegTensor<float>&)vregIn2, xAddr1,
                    gmCopyInRegStep_);
                AscendC::MicroAPI::And((AscendC::MicroAPI::RegTensor<uint16_t>&)vregIn1,
                                       (AscendC::MicroAPI::RegTensor<uint16_t>&)vregIn1, expMaskBF16, maskAll);
                AscendC::MicroAPI::And((AscendC::MicroAPI::RegTensor<uint16_t>&)vregIn2,
                                       (AscendC::MicroAPI::RegTensor<uint16_t>&)vregIn2, expMaskBF16, maskAll);
                MicroAPI::DataCopy<float, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::LoadDist::DIST_DINTLV_B32>(
                    (MicroAPI::RegTensor<float>&)vregIn3, (MicroAPI::RegTensor<float>&)vregIn4, xAddr1,
                    gmCopyInRegStep_);
                AscendC::MicroAPI::And((AscendC::MicroAPI::RegTensor<uint16_t>&)vregIn3,
                                       (AscendC::MicroAPI::RegTensor<uint16_t>&)vregIn3, expMaskBF16, maskAll);
                AscendC::MicroAPI::And((AscendC::MicroAPI::RegTensor<uint16_t>&)vregIn4,
                                       (AscendC::MicroAPI::RegTensor<uint16_t>&)vregIn4, expMaskBF16, maskAll);
                MicroAPI::Max(vregMax1, vregIn1, vregIn2, maskAll);
                MicroAPI::Max(vregMax2, vregIn3, vregIn4, maskAll);
                MicroAPI::DeInterleave(vregMax1, vregMax2, vregMax1, vregMax2);
                MicroAPI::Max(vregOut, vregMax1, vregMax2, maskAll);
                if constexpr (IsSame<T_X, half>::value) {
                    AscendC::MicroAPI::Cast<bfloat16_t, T_X, castTraitHalf2Bf16>(vregOutbf16, vregOut, maskAll);
                    AscendC::MicroAPI::DataCopy<uint16_t, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                        maxPer4BufAddr, (MicroAPI::RegTensor<uint16_t>&)vregOutbf16, gmCopyInRegStep2_, maskAll);
                } else {
                    AscendC::MicroAPI::DataCopy<uint16_t, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                        maxPer4BufAddr, (MicroAPI::RegTensor<uint16_t>&)vregOut, gmCopyInRegStep2_, maskAll);
                }
            }
        }
    }

    __aicore__ inline void ComputeMaxPer8(uint16_t maxPer8RegLoop, __ubuf__ uint16_t* maxPer4BufReadAddr,
                                          __ubuf__ uint16_t* e8BufAddr, __ubuf__ uint16_t* maxPer8BufAddr)
    {
        __VEC_SCOPE__
        {
            MicroAPI::RegTensor<bfloat16_t> vreg0Maxper4;
            MicroAPI::RegTensor<bfloat16_t> vreg1Maxper4;
            MicroAPI::RegTensor<bfloat16_t> vreg0Maxper8;
            MicroAPI::RegTensor<bfloat16_t> vreg0Maxper64;
            MicroAPI::RegTensor<bfloat16_t> vreg1Maxper64;
            MicroAPI::RegTensor<uint16_t> vreg0E6m2;
            MicroAPI::RegTensor<uint16_t> vreg1E6m2;
            MicroAPI::RegTensor<uint16_t> vregNumscale;
            MicroAPI::Duplicate(vregNumscale, 0x3e12);
            MicroAPI::UnalignReg ureg1;
            MicroAPI::MaskReg maskAll = MicroAPI::CreateMask<uint8_t>();
            static constexpr AscendC::MicroAPI::CastTrait castTraitE6M2 = {
                AscendC::MicroAPI::RegLayout::ZERO, AscendC::MicroAPI::SatMode::SAT,
                AscendC::MicroAPI::MaskMergeMode::ZEROING, AscendC::RoundMode::CAST_RINT};
            for (uint16_t loopIdx = 0; loopIdx < maxPer8RegLoop; loopIdx++) {
                MicroAPI::DataCopy<uint16_t, MicroAPI::PostLiteral::POST_MODE_UPDATE,
                                   MicroAPI::LoadDist::DIST_DINTLV_B16>(
                    (MicroAPI::RegTensor<uint16_t>&)vreg0Maxper4, (MicroAPI::RegTensor<uint16_t>&)vreg1Maxper4,
                    maxPer4BufReadAddr, gmCopyInRegStep2_ * DIGIT_TWO);
                MicroAPI::Max(vreg0Maxper8, vreg0Maxper4, vreg1Maxper4, maskAll);
                MicroAPI::Interleave(vreg0Maxper4, vreg1Maxper4, vreg0Maxper4, vreg1Maxper4);

                MicroAPI::ReduceMaxWithDataBlock<uint16_t>((MicroAPI::RegTensor<uint16_t>&)vreg0Maxper64,
                                                           (MicroAPI::RegTensor<uint16_t>&)vreg0Maxper4, maskAll);
                MicroAPI::ReduceMaxWithDataBlock<uint16_t>((MicroAPI::RegTensor<uint16_t>&)vreg1Maxper64,
                                                           (MicroAPI::RegTensor<uint16_t>&)vreg1Maxper4, maskAll);

                MicroAPI::Mul(vreg0Maxper64, vreg0Maxper64, (MicroAPI::RegTensor<bfloat16_t>&)vregNumscale, maskAll);
                MicroAPI::Mul(vreg1Maxper64, vreg1Maxper64, (MicroAPI::RegTensor<bfloat16_t>&)vregNumscale, maskAll);

                MicroAPI::Cast<fp8_e6m2_t, bfloat16_t, castTraitE6M2>((MicroAPI::RegTensor<fp8_e6m2_t>&)vreg0E6m2,
                                                                      vreg0Maxper64, maskAll);
                MicroAPI::Cast<fp8_e6m2_t, bfloat16_t, castTraitE6M2>((MicroAPI::RegTensor<fp8_e6m2_t>&)vreg1E6m2,
                                                                      vreg1Maxper64, maskAll);

                AscendC::MicroAPI::DataCopy<uint16_t, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                    maxPer8BufAddr, (MicroAPI::RegTensor<uint16_t>&)vreg0Maxper8, gmCopyInRegStep2_, maskAll);
                AscendC::MicroAPI::DataCopyUnAlign<uint16_t, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                    e8BufAddr, vreg0E6m2, ureg1, 8);
                AscendC::MicroAPI::DataCopyUnAlign<uint16_t, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                    e8BufAddr, vreg1E6m2, ureg1, 8);
            }
            AscendC::MicroAPI::DataCopyUnAlignPost(e8BufAddr, ureg1, 0);
        }
    }

    __aicore__ inline void ComputeTmpScale(uint16_t regLoop, __ubuf__ uint16_t* maxPer4BufRead2Addr,
                                           __ubuf__ uint16_t* maxPer8BufReadAddr, __ubuf__ uint16_t* e8BufRead1Addr,
                                           __ubuf__ uint32_t* e1p8BufAddr, __ubuf__ uint8_t* e1p16BufAddr,
                                           __ubuf__ uint16_t* tmpScaleAddr)
    {
        __VEC_SCOPE__
        {
            MicroAPI::RegTensor<bfloat16_t> vreg0Maxper4;
            MicroAPI::RegTensor<bfloat16_t> vreg0Maxper8;
            MicroAPI::RegTensor<bfloat16_t> vreg0Maxper64;
            MicroAPI::RegTensor<bfloat16_t> vb16_Ea_rec_bf16;
            MicroAPI::RegTensor<bfloat16_t> vreg0E6m2;
            MicroAPI::RegTensor<bfloat16_t> vregTmpScale;

            float number1 = 1.0;
            MicroAPI::RegTensor<bfloat16_t> vregNum1;
            MicroAPI::Duplicate(vregNum1, number1);
            float numberhalf = 0.5;
            MicroAPI::RegTensor<bfloat16_t> vregNumhalf;
            MicroAPI::Duplicate(vregNumhalf, static_cast<bfloat16_t>(numberhalf));

            // Eb/Ec 门限：scale_alg==1(unbiased) 用 3.75/1.875，否则(mse_first)用 4/2。
            float thdEb = (scaleAlg_ == 1) ? 3.75f : 4.0f;
            float thdEc = (scaleAlg_ == 1) ? 1.875f : 2.0f;

            MicroAPI::MaskReg vb1E8Comp;
            MicroAPI::MaskReg vb1E8CompTmp;
            MicroAPI::MaskReg vb1E16Comp;
            MicroAPI::MaskReg vb1E16CompTmp;
            MicroAPI::UnalignReg ureg1;

            MicroAPI::MaskReg maskAll = MicroAPI::CreateMask<uint8_t>();
            for (uint16_t loopIdx = 0; loopIdx < regLoop; loopIdx++) {
                MicroAPI::DataCopy<uint16_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::LoadDist::DIST_E2B_B16>(
                    (MicroAPI::RegTensor<uint16_t>&)vreg0Maxper64, e8BufRead1Addr, 8);
                MicroAPI::DataCopy<uint16_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::LoadDist::DIST_US_B16>(
                    (MicroAPI::RegTensor<uint16_t>&)vreg0Maxper8, maxPer8BufReadAddr, 64);
                MicroAPI::DataCopy<uint16_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::LoadDist::DIST_NORM>(
                    (MicroAPI::RegTensor<uint16_t>&)vreg0Maxper4, maxPer4BufRead2Addr, 128);

                ReciprocalCast<bfloat16_t, fp8_e6m2_t, castTraitFp82B16>(
                    (MicroAPI::RegTensor<bfloat16_t>&)vb16_Ea_rec_bf16, (MicroAPI::RegTensor<fp8_e6m2_t>&)vreg0Maxper64,
                    maskAll);
                MicroAPI::Mul(vreg0Maxper8, vreg0Maxper8, vb16_Ea_rec_bf16, maskAll);
                MicroAPI::CompareScalar<bfloat16_t, CMPMODE::GE>(vb1E8Comp, vreg0Maxper8, thdEb, maskAll);
                MicroAPI::Select(vreg0Maxper8, vregNumhalf, vregNum1, vb1E8Comp);
                MicroAPI::Mul(vreg0Maxper4, vreg0Maxper4, vb16_Ea_rec_bf16, maskAll);
                MicroAPI::Mul(vreg0Maxper4, vreg0Maxper4, vreg0Maxper8, maskAll);
                MicroAPI::CompareScalar<bfloat16_t, CMPMODE::GE>(vb1E16Comp, vreg0Maxper4, thdEc, maskAll);
                MicroAPI::Select(vreg0Maxper4, vregNumhalf, vregNum1, vb1E16Comp);
                MicroAPI::Mul(vreg0Maxper4, vreg0Maxper4, vreg0Maxper8, maskAll);
                MicroAPI::Mul(vregTmpScale, vreg0Maxper4, vb16_Ea_rec_bf16, maskAll);

                MicroAPI::DataCopy<uint16_t, MicroAPI::StoreDist::DIST_INTLV_B16>(
                    tmpScaleAddr + loopIdx * gmCopyTmpBufOtherStep_, (MicroAPI::RegTensor<uint16_t>&)vregTmpScale,
                    (MicroAPI::RegTensor<uint16_t>&)vregTmpScale, maskAll);

                MicroAPI::DataCopyUnAlign<uint32_t>(e1p8BufAddr, vb1E8Comp, ureg1);

                MicroAPI::DataCopy<uint8_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::MaskDist::DIST_PACK>(
                    e1p16BufAddr, vb1E16Comp, 16);
            }
            MicroAPI::DataCopyUnAlignPost<uint32_t, MicroAPI::PostLiteral::POST_MODE_UPDATE>(e1p8BufAddr, ureg1, 0);
        }
    }

    __aicore__ inline void ComputeScale(uint16_t scaleRegLoop, __ubuf__ uint8_t* e8BufRead2Addr,
                                        __ubuf__ uint8_t* e1p8BufReadAddr, __ubuf__ uint8_t* e1p16BufReadAddr,
                                        __ubuf__ uint16_t* scaleAddr)
    {
        __VEC_SCOPE__
        {
            MicroAPI::RegTensor<uint8_t> vregE8;
            MicroAPI::RegTensor<uint8_t> vreg1E8;
            MicroAPI::RegTensor<uint8_t> vreg1E16First;
            MicroAPI::RegTensor<uint8_t> vreg1E16Second;

            MicroAPI::MaskReg maskAll = MicroAPI::CreateMask<uint8_t>();
            for (uint16_t loopIdx = 0; loopIdx < scaleRegLoop; loopIdx++) {
                MicroAPI::DataCopy<uint8_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::LoadDist::DIST_DS_B8>(
                    vregE8, e8BufRead2Addr, gmCopyTmpBufE8Step_);
                MicroAPI::DataCopy<uint8_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::LoadDist::DIST_NORM>(
                    vreg1E8, e1p8BufReadAddr, gmCopyTmpBufOtherStep_);
                MicroAPI::DataCopy<uint8_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::LoadDist::DIST_NORM>(
                    vreg1E16First, e1p16BufReadAddr, gmCopyTmpBufOtherStep_);
                MicroAPI::DataCopy<uint8_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::LoadDist::DIST_NORM>(
                    vreg1E16Second, e1p16BufReadAddr, gmCopyTmpBufOtherStep_);

                MicroAPI::Interleave<uint8_t>(vregE8, vreg1E8, vregE8, vreg1E8);
                MicroAPI::DataCopy<uint16_t, MicroAPI::StoreDist::DIST_INTLV_B16>(
                    scaleAddr + loopIdx * gmCopyTmpBufE8Step_, (MicroAPI::RegTensor<uint16_t>&)vregE8,
                    (MicroAPI::RegTensor<uint16_t>&)vreg1E16First, maskAll);
                MicroAPI::DataCopy<uint16_t, MicroAPI::StoreDist::DIST_INTLV_B16>(
                    scaleAddr + loopIdx * gmCopyTmpBufE8Step_ + gmCopyTmpBufOtherStep_,
                    (MicroAPI::RegTensor<uint16_t>&)vreg1E8, (MicroAPI::RegTensor<uint16_t>&)vreg1E16Second, maskAll);
            }
        }
    }

    __aicore__ inline void ComputeOut(uint16_t outRegLoop, __ubuf__ T_X* xAddr2, __ubuf__ uint16_t* tmpScaleReadAddr,
                                      __ubuf__ int32_t* yAddr)
    {
        __VEC_SCOPE__
        {
            MicroAPI::RegTensor<T_X> vregOut;
            MicroAPI::RegTensor<uint16_t> vregScaleTmp;
            MicroAPI::RegTensor<fp4x2_e1m2_t> vbOut;
            MicroAPI::RegTensor<bfloat16_t> vregOutbf16;
            MicroAPI::MaskReg maskAll = MicroAPI::CreateMask<uint8_t>();
            static constexpr AscendC::MicroAPI::CastTrait castTraitHalf2Bf16 = {
                AscendC::MicroAPI::RegLayout::UNKNOWN, AscendC::MicroAPI::SatMode::UNKNOWN,
                AscendC::MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
            for (uint16_t loopIdx = 0; loopIdx < outRegLoop; loopIdx++) {
                if constexpr (IsSame<T_X, half>::value) {
                    MicroAPI::DataCopy<T_X, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::LoadDist::DIST_NORM>(
                        vregOut, xAddr2, gmCopyInRegStep2_);
                    AscendC::MicroAPI::Cast<bfloat16_t, T_X, castTraitHalf2Bf16>(
                        (MicroAPI::RegTensor<bfloat16_t>&)vregOutbf16, vregOut, maskAll);
                } else {
                    MicroAPI::DataCopy<T_X, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::LoadDist::DIST_NORM>(
                        vregOutbf16, xAddr2, gmCopyInRegStep2_);
                }
                MicroAPI::DataCopy<uint16_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::LoadDist::DIST_US_B16>(
                    vregScaleTmp, tmpScaleReadAddr, gmCopyOutScaleRegStep_);
                MicroAPI::Mul((MicroAPI::RegTensor<bfloat16_t>&)vregOutbf16,
                              (MicroAPI::RegTensor<bfloat16_t>&)vregOutbf16,
                              (MicroAPI::RegTensor<bfloat16_t>&)vregScaleTmp, maskAll);
                MicroAPI::Cast<fp4x2_e1m2_t, bfloat16_t, castTrait_>(vbOut, vregOutbf16, maskAll);
                MicroAPI::DataCopy<int32_t, MicroAPI::PostLiteral::POST_MODE_UPDATE,
                                   MicroAPI::StoreDist::DIST_PACK4_B32>(yAddr, (MicroAPI::RegTensor<int32_t>&)vbOut,
                                                                        gmCopyOutRegStep_, maskAll);
            }
        }
    }

    // ======================== Copy helpers ========================

    __aicore__ inline void CopyInX(uint64_t offset, uint32_t curRows)
    {
        LocalTensor<T_X> xLocal = inQueueX.AllocTensor<T_X>();

        DataCopyExtParams extParams{static_cast<uint16_t>(curRows), static_cast<uint32_t>(numCol_ * sizeof(T_X)),
                                    static_cast<uint32_t>(0), static_cast<uint32_t>(0), 0};
        DataCopyPadExtParams<T_X> padParams{false, static_cast<uint8_t>(0), static_cast<uint8_t>(0),
                                            static_cast<T_X>(0.0)};

        DataCopyPad(xLocal, xGm[offset], extParams, padParams);
        inQueueX.EnQue(xLocal);
    }

    __aicore__ inline void CopyInGammabeta(LocalTensor<uint8_t> gammabetaLocal)
    {
        DataCopyExtParams copyParams{static_cast<uint16_t>(1), static_cast<uint32_t>(numCol_ * sizeof(T_GAMMA)),
                                     static_cast<uint32_t>(0), static_cast<uint32_t>(0), 0};
        DataCopyPadExtParams<T_GAMMA> padParams{false, static_cast<uint8_t>(0), static_cast<uint8_t>(0),
                                                static_cast<T_GAMMA>(0.0)};
        gammaLocal = gammabetaLocal.ReinterpretCast<T_GAMMA>();
        DataCopyPad<T_GAMMA>(gammaLocal, gammaGm, copyParams, padParams);
        if (betaFlag_ != 0) {
            betaLocal = gammabetaLocal[CeilAlign(numCol_, UB_BLOCK_SIZE / sizeof(T_GAMMA)) * sizeof(T_GAMMA)]
                            .ReinterpretCast<T_GAMMA>();
            DataCopyPad<T_GAMMA>(betaLocal, betaGm, copyParams, padParams);
        }
    }

    __aicore__ inline void CopyOutQuantYFP4(uint64_t gmOffset, uint32_t curRows)
    {
        LocalTensor<uint8_t> quantYLocal = outQueueQuantY.DeQue<uint8_t>();
        uint32_t bytesPerRow = numCol_ / DIGIT_TWO;
        DataCopyExtParams copyParams{static_cast<uint16_t>(curRows), static_cast<uint32_t>(bytesPerRow),
                                     static_cast<uint32_t>(0), static_cast<uint32_t>(0), 0};
        DataCopyPad<uint8_t>(yFp4Gm[gmOffset / DIGIT_TWO], quantYLocal, copyParams);
        outQueueQuantY.FreeTensor(quantYLocal);
    }

    __aicore__ inline void CopyOutScale(uint32_t rowRepeat, uint32_t curRows)
    {
        LocalTensor<float> scaleLocal = outQueueScale.DeQue<float>();
        DataCopyExtParams copyParams{static_cast<uint16_t>(curRows),
                                     static_cast<uint32_t>(mxScaleSize_ * sizeof(float)), static_cast<uint32_t>(0),
                                     static_cast<uint32_t>(0), 0};
        DataCopyPad<float, PaddingMode::Compact>(scaleGm[rowRepeat * rowFactor_ * mxScaleSize_], scaleLocal,
                                                 copyParams);
        outQueueScale.FreeTensor(scaleLocal);
    }

private:
    TPipe* pPipe = nullptr;

    TQue<QuePosition::VECIN, 1> inQueueX;
    TQue<QuePosition::VECIN, 1> inQueueGammabeta;
    TQue<QuePosition::VECOUT, 1> outQueueRstd;
    TBuf<TPosition::VECCALC> xReduceBuff;
    TBuf<TPosition::VECCALC> xFp32Buff;
    TBuf<TPosition::VECCALC> binaryAddBuf;

    TBuf<TPosition::VECCALC> bufE8_;
    TBuf<TPosition::VECCALC> buf1E8_;
    TBuf<TPosition::VECCALC> buf1E16_;
    TBuf<TPosition::VECCALC> bufTmpScale_;
    TBuf<TPosition::VECCALC> bufMaxPer4_;
    TBuf<TPosition::VECCALC> bufMaxPer8_;
    TQue<QuePosition::VECOUT, DOUBLE_BUFFER_NUM> outQueueQuantY;
    TQue<QuePosition::VECOUT, DOUBLE_BUFFER_NUM> outQueueScale;

    LocalTensor<T_GAMMA> gammaLocal;
    LocalTensor<T_GAMMA> betaLocal;

    GlobalTensor<T_X> xGm;
    GlobalTensor<T_GAMMA> gammaGm;
    GlobalTensor<T_GAMMA> betaGm;
    GlobalTensor<float> rstdGm;
    GlobalTensor<uint8_t> yFp4Gm;
    GlobalTensor<float> scaleGm;

    uint64_t numRow_;
    uint64_t numCol_;
    uint64_t numColAlign_;
    uint64_t blockFactor_;
    uint64_t rowFactor_;
    uint64_t binAddQuotient_;
    float epsilon_;
    float avgFactor_;
    uint64_t rowWork{1};
    int64_t mxBlockSize_;
    int64_t blockNumInColAxis_;
    int64_t mxScaleSize_;
    uint32_t betaFlag_;
    uint32_t rstdFlag_;
    int64_t scaleAlg_{0};
    int64_t innerLoopEle_;
    int64_t max8LoopEle_;
    int64_t scaleLoopEle_;
    int64_t outLoopEle_;
    int64_t alignGroupNum_;
    int64_t innerLoopTail_;
    int64_t max8LoopTail_;
    int64_t scaleLoopTail_;
    int64_t outLoopTail_;
    int64_t curHif4Groups_ = 0;

    uint32_t vlForHalfNumber;
    uint16_t elementAfterReduce;

    int64_t sysRegLen_ = 256;
    int64_t gmCopyInRegStep_ = 0;
    int64_t gmCopyOutRegStep_ = 0;
    int64_t gmCopyOutScaleRegStep_ = 0;
    int64_t gmCopyInRegStep2_ = 0;
    int64_t gmCopyTmpE1_ = 0;
    int64_t gmCopyTmpE8_ = 0;
    int64_t gmCopyTmpE16_ = 0;
    int64_t gmCopyTmpBufE8Step_ = 0;
    int64_t gmCopyTmpBufOtherStep_ = 0;

    int64_t bufE8Buffer_ = 0;
    int64_t buf1E8Buffer_ = 0;
    int64_t buf1E16Buffer_ = 0;
    int64_t maxPer4Buffer_ = 0;
    int64_t maxPer8Buffer_ = 0;
    int64_t tmpScaleBuffer_ = 0;

    constexpr static MicroAPI::CastTrait castTrait_ = {MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::UNKNOWN,
                                                       MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
};
} // namespace RmsNormHiF4Quant
#endif // RMS_NORM_HI_F4_QUANT_HIF4_R_FULL_LOAD_H

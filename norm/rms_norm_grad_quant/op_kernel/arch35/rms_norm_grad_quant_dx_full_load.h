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
 * \file rms_norm_grad_quant_regbase_dx_full_load.h
 * \brief RmsNormGradQuant Regbase DX Full Load kernel File
 */

#ifndef RMS_NORM_GRAD_Quant_DX_FULL_LOAD_H
#define RMS_NORM_GRAD_Quant_DX_FULL_LOAD_H

#include "kernel_tiling/kernel_tiling.h"
#include "kernel_operator.h"
#include "rms_norm_grad_quant_common.h"

namespace RmsNormGradQuant {
using namespace AscendC;
template <typename T_DY, typename T_X, typename T_GAMMA, typename T_DX, typename T_DGAMMA, typename T_SCALES_X, typename T_OFFSET_X, bool HAS_OFFSET_X, bool DIV_MODE>
class RegbaseDxFullLoad {
public:
     __aicore__ inline RegbaseDxFullLoad(TPipe* pipe, const RmsNormGradQuantRegbaseDxTilingData* tilingData)
        : Ppipe_(pipe), tiling_(tilingData)
    {}

    __aicore__ inline void Init(
        __gm__ uint8_t* dy, __gm__ uint8_t* x, __gm__ uint8_t* rstd, __gm__ uint8_t* gamma, __gm__ uint8_t* scales_x,
        __gm__ uint8_t* offset_x, __gm__ uint8_t* dx, __gm__ uint8_t* dgamma)
    {
    #if (__NPU_ARCH__ == 3510)
        if constexpr (
            IsSameType<T_DX, hifloat8_t>::value) {
            AscendC::SetCtrlSpr<FLOAT_OVERFLOW_MODE_CTRL, FLOAT_OVERFLOW_MODE_CTRL>(0);
        }
    #endif
        usedCoreNum_ = tiling_->usedCoreNumDx;
        uint32_t coreIdx = GetBlockIdx();
        if (coreIdx >= usedCoreNum_) {
            return;
        }
        rows_ = tiling_->rows;
        cols_ = tiling_->cols;
        blockFactor_ = tiling_->blockFactorDx;

        colsAlignBlock_ =
            IsSameType<T_X, float>::value ? AlignUp(cols_, FLOAT_NUM_BLOCK) : AlignUp(cols_, HALF_NUM_BLOCK);
        colsAlignB8_ = AlignUp(cols_, HIFP8_NUM_BLOCK);
        colsAlignB32_ = AlignUp(cols_, FLOAT_NUM_BLOCK);

        ubFactor_ = tiling_->ubFactor;
        colFlodFactor_ = tiling_->bodyPart;
        avgFactor1_ = 1.0f / cols_;

        dyGm_.SetGlobalBuffer((__gm__ T_DY*)dy + coreIdx * blockFactor_ * cols_);
        xGm_.SetGlobalBuffer((__gm__ T_X*)x + coreIdx * blockFactor_ * cols_);
        rstdGm_.SetGlobalBuffer((__gm__ float*)rstd + coreIdx * blockFactor_);
        gammaGm_.SetGlobalBuffer((__gm__ T_GAMMA*)gamma);
        dxGm_.SetGlobalBuffer((__gm__ T_DX*)dx + coreIdx * blockFactor_ * cols_);

        Ppipe_->InitBuffer(inQueueDy_, DB_NUM, ubFactor_ * colsAlignBlock_ * sizeof(T_DY));
        Ppipe_->InitBuffer(inQueueX_, DB_NUM, ubFactor_ * colsAlignBlock_ * sizeof(T_X));
        Ppipe_->InitBuffer(inQueueRstd_, DB_NUM, AlignUp(ubFactor_, V_LENGTH) * sizeof(float));
        Ppipe_->InitBuffer(outQueueDx_, DB_NUM, ubFactor_ * colsAlignB8_ * sizeof(T_DX));
        Ppipe_->InitBuffer(inQueueGamma_, 1, colsAlignBlock_ * sizeof(T_GAMMA));
        Ppipe_->InitBuffer(reduceBuf_, ubFactor_ * colsAlignB32_ * sizeof(float));
        Ppipe_->InitBuffer(tmpSumBuf_, AlignUp(ubFactor_, V_LENGTH) * sizeof(float));
        
        // init tmp buffer
        uint64_t firstVcaddResult =
            ubFactor_ *
            (((colFlodFactor_ + V_LENGTH - 1) / V_LENGTH + BLOCKSIZEB32 - 1) / BLOCKSIZEB32 * BLOCKSIZEB32) *
            sizeof(float);
        Ppipe_->InitBuffer(reduceTmpBuf_, firstVcaddResult);

        scalesXGm_.SetGlobalBuffer((__gm__ T_SCALES_X*)scales_x);
        Ppipe_->InitBuffer(inQueueScalesX_, 1, sizeof(T_SCALES_X));
        if constexpr (HAS_OFFSET_X) {
            offsetXGm_.SetGlobalBuffer((__gm__ T_OFFSET_X*)offset_x);
            Ppipe_->InitBuffer(inQueueOffsetX_, 1, sizeof(T_OFFSET_X));
        }
    }
    __aicore__ inline void Process()
    {
        uint32_t coreIdx = GetBlockIdx();
        if (coreIdx >= usedCoreNum_) {
            return;
        }
        // copyInScalesX
        CopyInScalesX();
        if constexpr (HAS_OFFSET_X) {
            CopyInOffsetX();
        }
        int64_t blockTail = rows_ - (usedCoreNum_ - 1) * blockFactor_;
        int64_t calcRowNum = coreIdx == usedCoreNum_ - 1 ? blockTail : blockFactor_;
        int64_t calcRowNumRemain = calcRowNum;
        for (int64_t rowIdx = 0; rowIdx < calcRowNum; rowIdx += ubFactor_) {
            int64_t calcRowNumSub = Min(ubFactor_, calcRowNumRemain);
            SubProcess(rowIdx, calcRowNumSub);
            calcRowNumRemain -= ubFactor_;
        }
        if (calcRowNum > 0) {
            inQueueGamma_.FreeTensor(gammaLocal_);
        }
        inQueueScalesX_.FreeTensor(scalesXLocal_);
        if constexpr (HAS_OFFSET_X) {
            inQueueOffsetX_.FreeTensor(offsetXLocal_);
        }
    }

    __aicore__ inline void SubProcess(int64_t rowIdx, int64_t calcRowNumSub)
    {
        if (rowIdx == 0) {
            CopyInGamma();
        }
        CopyInDy(rowIdx, calcRowNumSub);
        LocalTensor<T_DY> dyLocal = inQueueDy_.DeQue<T_DY>();
        CopyInX(rowIdx, calcRowNumSub);
        LocalTensor<T_X> xLocal = inQueueX_.DeQue<T_X>();
        CopyInRstd(rowIdx, calcRowNumSub);
        LocalTensor<float> rstdLocal = inQueueRstd_.DeQue<float>();
        LocalTensor<T_GAMMA> gammaLocal = gammaLocal_;
        LocalTensor<float> tmpSumLocal = tmpSumBuf_.Get<float>();
        LocalTensor<float> reduceLocal = reduceBuf_.Get<float>();
        LocalTensor<float> reduceTmpLocal = reduceTmpBuf_.Get<float>();
        uint16_t loopRow = calcRowNumSub;
        constexpr uint32_t oneRepeat = V_LENGTH;
        uint32_t cols = static_cast<uint32_t>(colsAlignBlock_);
        uint32_t colsAlignB32 = static_cast<uint32_t>(colsAlignB32_);
        uint32_t colsReal = static_cast<uint32_t>(cols_);
        uint16_t repeatCount = DivCeil(cols_, oneRepeat);
        __local_mem__ T_GAMMA* gammaAddr = (__ubuf__ T_GAMMA*)gammaLocal.GetPhyAddr();
        __local_mem__ T_DY* dyAddr = (__ubuf__ T_DY*)dyLocal.GetPhyAddr();
        __local_mem__ T_X* xAddr = (__ubuf__ T_X*)xLocal.GetPhyAddr();
        __local_mem__ float* rstdAddr = (__ubuf__ float*)rstdLocal.GetPhyAddr();
        __local_mem__ float* reduceAddr = (__ubuf__ float*)reduceLocal.GetPhyAddr();
        __VEC_SCOPE__
        {
            RegTensor<float> gammaReg, dyReg, xReg, rstdReg, mulReg0, mulReg2, mulReg3;
            for (uint16_t r = 0; r < loopRow; r++) {
                uint32_t sreg = colsReal;
                MaskReg maskReg = CreateMask<float, MaskPattern::ALL>();
                DataCopy<float, LoadDist::DIST_BRC_B32>(rstdReg, rstdAddr + r);
                for (uint16_t i = 0; i < repeatCount; i++) {
                    maskReg = UpdateMask<float>(sreg);
                    LoadAndCast(gammaReg, gammaAddr, maskReg, i * oneRepeat);
                    LoadAndCast(dyReg, dyAddr, maskReg, r * cols + i * oneRepeat);
                    Mul(mulReg2, dyReg, gammaReg, maskReg);
                    LoadAndCast(xReg, xAddr, maskReg, r * cols + i * oneRepeat);
                    Mul(mulReg0, xReg, rstdReg, maskReg);
                    Mul(mulReg3, mulReg2, mulReg0, maskReg);
                    DataCopy(reduceAddr + (r * colsAlignB32 + i * oneRepeat), mulReg3, maskReg);
                }
            }
        }

        ComputeReduceSum(reduceLocal, reduceTmpLocal, tmpSumLocal, calcRowNumSub);

        LocalTensor<T_DX> dxLocal = outQueueDx_.AllocTensor<T_DX>();
        LocalTensor<T_SCALES_X> scalesXLocal;
        LocalTensor<T_OFFSET_X> offsetXLocal;
        __local_mem__ float* meanAddr = (__ubuf__ float*)tmpSumLocal.GetPhyAddr();
        __local_mem__ T_DX* dxAddr = (__ubuf__ T_DX*)dxLocal.GetPhyAddr();
        __local_mem__ T_SCALES_X* scalesXAddr;
        __local_mem__ T_OFFSET_X* offsetXAddr;

        scalesXLocal = scalesXLocal_;
        scalesXAddr = (__ubuf__ T_SCALES_X*)scalesXLocal.GetPhyAddr();
        if constexpr (HAS_OFFSET_X) {
            offsetXLocal = offsetXLocal_;
            offsetXAddr = (__ubuf__ T_OFFSET_X*)offsetXLocal.GetPhyAddr();
        }

        uint32_t colsAlignHiFP8 = static_cast<uint32_t>(colsAlignB8_);

        __VEC_SCOPE__
        {
            RegTensor<float> gammaReg, dyReg, xReg, rstdReg, meanReg, dxReg, mulReg0, mulReg2, mulReg4, subReg;
            RegTensor<float> scalesXReg, scalesXResultReg, offsetXReg;
            for (uint16_t r = 0; r < loopRow; r++) {
                uint32_t sreg = colsReal;
                MaskReg maskReg = CreateMask<float, AscendC::MicroAPI::MaskPattern::ALL>();
                DataCopy<float, LoadDist::DIST_BRC_B32>(rstdReg, rstdAddr + (r));
                DataCopy<float, LoadDist::DIST_BRC_B32>(meanReg, meanAddr + (r));
                Muls(meanReg, meanReg, avgFactor1_, maskReg);
                for (uint16_t i = 0; i < repeatCount; i++) {
                    maskReg = UpdateMask<float>(sreg);
                    LoadAndCast(gammaReg, gammaAddr, maskReg, i * oneRepeat);

                    LoadAndCast(dyReg, dyAddr, maskReg, r * cols + i * oneRepeat);
                    Mul(mulReg2, dyReg, gammaReg, maskReg);

                    LoadAndCast(xReg, xAddr, maskReg, r * cols + i * oneRepeat);
                    Mul(mulReg0, xReg, rstdReg, maskReg);
                    Mul(mulReg4, mulReg0, meanReg, maskReg);
                    Sub(subReg, mulReg2, mulReg4, maskReg);
                    Mul(dxReg, subReg, rstdReg, maskReg);
                    // cal quant
                    LoadTensorForDtypeTIn(scalesXAddr, scalesXReg, maskReg);
                    if constexpr (DIV_MODE) {
                        Div(scalesXResultReg, dxReg, scalesXReg, maskReg);
                    } else {
                        Mul(scalesXResultReg, dxReg, scalesXReg, maskReg);
                    }
                    if constexpr (HAS_OFFSET_X){
                        LoadTensorForDtypeTIn(offsetXAddr, offsetXReg, maskReg);
                        Add(scalesXResultReg, scalesXResultReg, offsetXReg, maskReg);
                    }
                    if constexpr(IsSameType<T_DX, hifloat8_t>::value){
                        RegTensor<T_DX> dxRegHif8;
                        Cast<T_DX, float, castTraitFp322Hifp8>(dxRegHif8, scalesXResultReg, maskReg);
                        DataCopy<T_DX, StoreDist::DIST_PACK4_B32>(dxAddr + (r * colsAlignHiFP8 + i * oneRepeat), dxRegHif8, maskReg);
                    } else if constexpr(IsSameType<T_DX, int8_t>::value){
                        RegTensor<T_DX> dxRegInt8;
                        RegTensor<half> dxRegFp16;
                        RegTensor<int32_t> dxRegInt32;
                        Cast<int32_t, float, castTraitFp322Int32>(dxRegInt32, scalesXResultReg, maskReg);
                        Cast<float, int32_t, castTraitInt322Fp32>(scalesXResultReg, dxRegInt32, maskReg);
                        Cast<half, float, castTraitFp322Fp16>(dxRegFp16, scalesXResultReg, maskReg);
                        Cast<T_DX, half, castTraitFp162Int8>(dxRegInt8, dxRegFp16, maskReg);
                        DataCopy<T_DX, StoreDist::DIST_PACK4_B32>(dxAddr + (r * colsAlignHiFP8 + i * oneRepeat), dxRegInt8, maskReg);
                    }
                }
            }
        }
        inQueueDy_.FreeTensor(dyLocal);
        inQueueX_.FreeTensor(xLocal);
        inQueueRstd_.FreeTensor(rstdLocal);
        outQueueDx_.EnQue(dxLocal);
        CopyOutDx(rowIdx, calcRowNumSub);
    }

    __aicore__ inline void CopyInRstd(int64_t rowIdx, int64_t count)
    {
        LocalTensor<float> rstdLocal = inQueueRstd_.AllocTensor<float>();
        DataCopyExtParams copyParams{
            1,                                            // blockCount
            static_cast<uint32_t>(count * sizeof(float)), // blockLen
            0,                                            // srcStride
            0,                                            // dstStride
            0                                             // rsv
        };
        DataCopyPad(rstdLocal, rstdGm_[rowIdx], copyParams, {true, 0, 0, 0});
        inQueueRstd_.EnQue(rstdLocal);
    }

    __aicore__ inline void CopyInGamma()
    {
        LocalTensor<T_GAMMA> gammaLocal = inQueueGamma_.AllocTensor<T_GAMMA>();
        DataCopyExtParams copyParams{
            1,                                              // blockCount
            static_cast<uint32_t>(cols_ * sizeof(T_GAMMA)), // blockLen
            0,                                              // srcStride
            0,                                              // dstStride
            0                                               // rsv
        };

        DataCopyPad(gammaLocal, gammaGm_, copyParams, {true, 0, 0, 0});
        inQueueGamma_.EnQue(gammaLocal);
        gammaLocal_ = inQueueGamma_.DeQue<T_GAMMA>();
    }

    __aicore__ inline void CopyInScalesX()
    {
        LocalTensor<T_SCALES_X> scalesXLocal = inQueueScalesX_.AllocTensor<T_SCALES_X>();
        DataCopyExtParams copyParams{
            1,                                              // blockCount
            static_cast<uint32_t>(1 * sizeof(T_SCALES_X)), // blockLen
            0,                                              // srcStride
            0,                                              // dstStride
            0                                               // rsv
        };

        DataCopyPad(scalesXLocal, scalesXGm_, copyParams, {true, 0, 0, 0});
        inQueueScalesX_.EnQue(scalesXLocal);
        scalesXLocal_ = inQueueScalesX_.DeQue<T_SCALES_X>();
    }

    __aicore__ inline void CopyInOffsetX()
    {
        LocalTensor<T_OFFSET_X> offsetXLocal = inQueueOffsetX_.AllocTensor<T_OFFSET_X>();
        DataCopyExtParams copyParams{
            1,                                                 // blockCount
            static_cast<uint32_t>(1 * sizeof(T_OFFSET_X)), // blockLen
            0,                                                 // srcStride
            0,                                                 // dstStride
            0                                                  // rsv
        };

        DataCopyPad(offsetXLocal, offsetXGm_, copyParams, {true, 0, 0, 0});
        inQueueOffsetX_.EnQue(offsetXLocal);
        offsetXLocal_ = inQueueOffsetX_.DeQue<T_OFFSET_X>();
    }

    __aicore__ inline void CopyInDy(int64_t rowIdx, int64_t calcRow)
    {
        LocalTensor<T_DY> dyLocal = inQueueDy_.AllocTensor<T_DY>();
        DataCopyExtParams copyParams{
            static_cast<uint16_t>(calcRow),              // blockCount
            static_cast<uint32_t>(cols_ * sizeof(T_DY)), // blockLen
            0,                                           // srcStride
            0,                                           // dstStride
            0                                            // rsv
        };

        DataCopyPad(dyLocal, dyGm_[rowIdx * cols_], copyParams, {true, 0, 0, 0});
        inQueueDy_.EnQue(dyLocal);
    }

    __aicore__ inline void CopyInX(int64_t rowIdx, int64_t calcRow)
    {
        LocalTensor<T_X> xLocal = inQueueX_.AllocTensor<T_X>();
        DataCopyExtParams copyParams{
            static_cast<uint16_t>(calcRow),             // blockCount
            static_cast<uint32_t>(cols_ * sizeof(T_X)), // blockLen
            0,                                          // srcStride
            0,                                          // dstStride
            0                                           // rsv
        };

        DataCopyPad(xLocal, xGm_[rowIdx * cols_], copyParams, {true, 0, 0, 0});
        inQueueX_.EnQue(xLocal);
    }

    __aicore__ inline void ComputeReduceSum(
        LocalTensor<float> xLocal, LocalTensor<float> xTmpLocal, LocalTensor<float> xReduceTmpLocal, uint64_t curUbFactor)
    {
        __local_mem__ float* xLocalAddr = (__local_mem__ float*)xLocal.GetPhyAddr();
        __local_mem__ float* xTmpLocalUbAddr = (__local_mem__ float*)xTmpLocal.GetPhyAddr();
        __local_mem__ float* xReduceTmpLocalUbAddr = (__local_mem__ float*)xReduceTmpLocal.GetPhyAddr();

        if (colsAlignB32_ <= V_LENGTH) {
            CalculateReduceSumRLessThanVL(xLocalAddr, xReduceTmpLocalUbAddr, curUbFactor, cols_, colsAlignB32_);
        } else if (colsAlignB32_ <= (V_LENGTH + V_LENGTH)) {
            CalculateReduceSumRLessThanTwoVL(xLocalAddr, xReduceTmpLocalUbAddr, curUbFactor, cols_, colsAlignB32_);
        } else if (colsAlignB32_ <= V_LENGTH * V_LENGTH * NUM_TWO) {
            CalculateReduceSumRCommon<NUM_ONE>(
                xLocalAddr, xTmpLocalUbAddr, xReduceTmpLocalUbAddr, curUbFactor, cols_, colsAlignB32_, colFlodFactor_);
        } else {
            CalculateReduceSumRCommon<NUM_TWO>(
                xLocalAddr, xTmpLocalUbAddr, xReduceTmpLocalUbAddr, curUbFactor, cols_, colsAlignB32_, colFlodFactor_);
        }
    }

    __aicore__ inline void CalculateReduceSumRLessThanVL(
        __local_mem__ float* xLocalAddr, __local_mem__ float* xReduceTmpLocalUbAddr, uint64_t curUbFactor, uint64_t numCol,
        uint64_t numColAlign)
    {
        uint32_t colNum = static_cast<uint32_t>(numCol);
        uint16_t curAloops = static_cast<uint16_t>(curUbFactor);
        uint32_t colNumAlign = static_cast<uint32_t>(numColAlign);
        __VEC_SCOPE__
        {
            RegTensor<float> xReg;
            RegTensor<float> sumReg;

            // rstd cal
            MaskReg pregOne = CreateMask<float, MaskPattern::VL1>();
            uint32_t colCountSreg = colNum;
            MaskReg pregLoop = UpdateMask<float>(colCountSreg);
            for (uint16_t i = 0; i < curAloops; i++) {
                LoadAndCast(xReg, xLocalAddr, pregLoop, (i * colNumAlign));
                ReduceSum(sumReg, xReg, pregLoop);
                DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(xReduceTmpLocalUbAddr + i, sumReg, pregOne);
            }
        }
    }

    __aicore__ inline void CalculateReduceSumRLessThanTwoVL(
        __local_mem__ float* xLocalAddr, __local_mem__ float* xReduceTmpLocalUbAddr, uint64_t curUbFactor, uint64_t numCol,
        uint64_t numColAlign)
    {
        uint32_t colNum = static_cast<uint32_t>(numCol);
        uint32_t colNumAlign = static_cast<uint32_t>(numColAlign);
        uint16_t curAloops = static_cast<uint16_t>(curUbFactor);

        __VEC_SCOPE__
        {
            RegTensor<float> xReg;
            RegTensor<float> xTailReg;
            RegTensor<float> addReg;
            RegTensor<float> sumReg;
            RegTensor<float> xTailRegshiftLeft;
            MaskReg pregFull = CreateMask<float, MaskPattern::ALL>();
            MaskReg pregOne = CreateMask<float, MaskPattern::VL1>();

            uint32_t colTailSreg = colNum - V_LENGTH;
            MaskReg pregTail = UpdateMask<float>(colTailSreg);
            for (uint16_t i = 0; i < curAloops; i++) {
                LoadAndCast(xReg, xLocalAddr, pregFull, (i * colNumAlign));
                LoadAndCast(xTailReg, xLocalAddr + V_LENGTH, pregTail, (i * colNumAlign));
                ShiftLefts(
                    (RegTensor<uint32_t>&)xTailRegshiftLeft, (RegTensor<uint32_t>&)xTailReg, static_cast<int16_t>(0),
                    pregTail);
                Add(addReg, xReg, xTailRegshiftLeft, pregFull);
                ReduceSum(sumReg, addReg, pregFull);
                DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(xReduceTmpLocalUbAddr + i, sumReg, pregOne);
            }
        }
    }

    template <int32_t LAST_LOOP_NUMS>
    __aicore__ inline void CalculateReduceSumRCommon(
        __local_mem__ float* xLocalAddr, __local_mem__ float* xTmpLocalUbAddr, __local_mem__ float* xReduceTmpLocalUbAddr,
        uint64_t curUbFactor, uint64_t numCol, uint64_t numColAlign, uint64_t colFlodFactor_)
    {
        uint32_t colNum = static_cast<uint32_t>(numCol);
        uint32_t colNumAlign = static_cast<uint32_t>(numColAlign);
        uint32_t colFlodNum = static_cast<uint32_t>(colFlodFactor_);
        uint16_t curAloops = static_cast<uint16_t>(curUbFactor);

        // first flod
        uint32_t firstFlodTial = static_cast<uint32_t>(colNum - colFlodFactor_);
        uint16_t firstFlodAddLoops = static_cast<uint16_t>((firstFlodTial + V_LENGTH - 1) / V_LENGTH);
        uint16_t firstFlodWithOutAddLoops =
            static_cast<uint16_t>((colFlodNum + V_LENGTH - 1) / V_LENGTH) - firstFlodAddLoops;

        // first vcadd
        uint32_t firstVcaddNum = static_cast<uint32_t>((colFlodFactor_ + V_LENGTH - 1) / V_LENGTH);
        uint32_t firstVcaddNumCeilAlign =
            static_cast<uint32_t>((firstVcaddNum + BLOCKSIZEB32 - 1) / BLOCKSIZEB32 * BLOCKSIZEB32);

        // second flod
        // rstd cal
        uint16_t elewiseLoop = static_cast<uint16_t>((curUbFactor + V_LENGTH - 1) / V_LENGTH);

        __VEC_SCOPE__
        {
            RegTensor<float> xReg1;
            RegTensor<float> xReg2;
            RegTensor<float> xReg2shiftLeft;
            RegTensor<float> addReg;
            RegTensor<float> sumReg;

            RegTensor<float> xReg3;
            RegTensor<float> sumReg3;

            MaskReg pregFull = CreateMask<float, MaskPattern::ALL>();
            MaskReg pregOne = CreateMask<float, MaskPattern::VL1>();
            MaskReg pregLoop;

            for (uint16_t i = 0; i < curAloops; i++) {
                uint32_t sregfirstFlodTial = firstFlodTial;
                for (uint16_t j = 0; j < firstFlodAddLoops; j++) {
                    pregLoop = UpdateMask<float>(sregfirstFlodTial);
                    LoadAndCast(xReg1, xLocalAddr, pregFull, (i * colNumAlign + j * V_LENGTH));
                    LoadAndCast(
                        xReg2, xLocalAddr + colFlodNum, pregFull, (i * colNumAlign + j * V_LENGTH));
                    ShiftLefts(
                        (RegTensor<uint32_t>&)xReg2shiftLeft, (RegTensor<uint32_t>&)xReg2, static_cast<int16_t>(0),
                        pregLoop);
                    Add(addReg, xReg1, xReg2shiftLeft, pregFull);
                    ReduceSum(sumReg, addReg, pregFull);
                    DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(
                        xTmpLocalUbAddr + (i * firstVcaddNumCeilAlign + j), sumReg, pregOne);
                }
                for (uint16_t j = 0; j < static_cast<uint16_t>(firstFlodWithOutAddLoops); j++) {
                    LoadAndCast(
                        xReg3, xLocalAddr + firstFlodAddLoops * V_LENGTH, pregFull,
                        (i * colNumAlign + j * V_LENGTH));
                    ReduceSum(sumReg3, xReg3, pregFull);
                    DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(
                        xTmpLocalUbAddr + (i * firstVcaddNumCeilAlign + firstFlodAddLoops + j),
                        sumReg3, pregOne);
                }
            }

            // if need a add to last repeat
            LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
            if constexpr (LAST_LOOP_NUMS == 1) {
                uint32_t sregSecondReduce = firstVcaddNum;
                MaskReg pregLast = UpdateMask<float>(sregSecondReduce);
                for (uint16_t i = 0; i < curAloops; i++) {
                    DataCopy(xReg1, xTmpLocalUbAddr + (i * firstVcaddNumCeilAlign));
                    ReduceSum(sumReg, xReg1, pregLast);
                    DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(xReduceTmpLocalUbAddr + i, sumReg, pregOne);
                }
            } else if constexpr (LAST_LOOP_NUMS == 2) {
                uint32_t sregSecondReduce = firstVcaddNum - V_LENGTH;
                MaskReg pregLast = UpdateMask<float>(sregSecondReduce);
                RegTensor<float> shiftLeft;
                for (uint16_t i = 0; i < curAloops; i++) {
                    DataCopy(xReg1, xTmpLocalUbAddr + (i * firstVcaddNumCeilAlign));
                    DataCopy(xReg2, xTmpLocalUbAddr + (i * firstVcaddNumCeilAlign + V_LENGTH));
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

    template <typename T_IN>
    __aicore__ inline void LoadTensorForDtypeTIn(
        __local_mem__ T_IN* src, RegTensor<float>& dst, MaskReg& preg)
    {
        if constexpr (IsSameType<T_IN, float>::value) {
            DataCopy<float, LoadDist::DIST_BRC_B32>(dst, src);
        } else if constexpr (IsSameType<T_IN, int32_t>::value) {
            RegTensor<T_IN> xIn;
            DataCopy<int32_t, LoadDist::DIST_BRC_B32>(xIn, src);
            Cast<float, T_IN, castTraitInt322Fp32>(dst, xIn, preg);
        } else {
            RegTensor<T_IN> xIn;
            DataCopy<T_IN, LoadDist::DIST_BRC_B16>(xIn, src);
            Cast<float, T_IN, castTraitB162B32>(dst, xIn, preg);
        }
    }

    __aicore__ inline void CopyOutDx(int64_t rowIdx, int64_t calcRow)
    {
        LocalTensor<T_DX> dxLocal = outQueueDx_.DeQue<T_DX>();
        DataCopyExtParams copyParams{
            static_cast<uint16_t>(calcRow),             // blockCount
            static_cast<uint32_t>(cols_ * sizeof(T_DX)), // blockLen
            0,                                          // srcStride
            0,                                          // dstStride
            0                                           // rsv
        };
        DataCopyPad(dxGm_[rowIdx * cols_], dxLocal, copyParams);
        outQueueDx_.FreeTensor(dxLocal);
    }

private:
    TPipe* Ppipe_;
    const RmsNormGradQuantRegbaseDxTilingData* tiling_;
    GlobalTensor<T_DY> dyGm_;
    GlobalTensor<T_X> xGm_;
    GlobalTensor<T_GAMMA> gammaGm_;
    GlobalTensor<float> rstdGm_;
    GlobalTensor<T_DX> dxGm_;
    GlobalTensor<T_SCALES_X> scalesXGm_;
    GlobalTensor<T_OFFSET_X> offsetXGm_;
    TQue<QuePosition::VECIN, DEPTH_TWO> inQueueDy_;
    TQue<QuePosition::VECIN, DEPTH_TWO> inQueueX_;
    TQue<QuePosition::VECIN, DEPTH_TWO> inQueueRstd_;
    TQue<QuePosition::VECOUT, DEPTH_TWO> outQueueDx_;
    TQue<QuePosition::VECIN, DEPTH_TWO> inQueueGamma_;
    TQue<QuePosition::VECIN, DEPTH_TWO> inQueueScalesX_;
    TQue<QuePosition::VECIN, DEPTH_TWO> inQueueOffsetX_;
    TBuf<TPosition::VECCALC> reduceTmpBuf_;
    TBuf<TPosition::VECCALC> reduceBuf_;
    TBuf<TPosition::VECCALC> tmpSumBuf_;
    LocalTensor<T_GAMMA> gammaLocal_;
    LocalTensor<T_SCALES_X> scalesXLocal_;
    LocalTensor<T_OFFSET_X> offsetXLocal_;

    uint32_t usedCoreNum_;
    int64_t rows_;
    int64_t cols_;
    int64_t colsAlignBlock_;
    int64_t colsAlignB32_;
    int64_t colsAlignB8_;
    int64_t blockFactor_;
    int64_t ubFactor_;
    int64_t colFlodFactor_;
    float avgFactor1_;
};
} // namespace RmsNormGradQuant
#endif // RMS_NORM_GRAD_REGBASE_DX_FULL_LOAD_H
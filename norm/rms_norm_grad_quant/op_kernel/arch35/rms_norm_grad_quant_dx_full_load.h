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
        if constexpr (IsSameType<T_DX, hifloat8_t>::value || IsSameType<T_DX, int8_t>::value) {
            colsAlignHiFP8_ = AlignUp(cols_, HIFP8_NUM_BLOCK);
        }
        colsAlign2VL_ = AlignUp(cols_, FLOAT_NUM_2VL);

        ubFactor_ = UB_FACTOR_DX_FULL_LOAD;
        ubFactorD_ = colsAlign2VL_;
        ubFactorN_ = ubFactor_ / ubFactorD_;
        avgFactor1_ = 1.0f / cols_;

        dyGm_.SetGlobalBuffer((__gm__ T_DY*)dy + coreIdx * blockFactor_ * cols_);
        xGm_.SetGlobalBuffer((__gm__ T_X*)x + coreIdx * blockFactor_ * cols_);
        rstdGm_.SetGlobalBuffer((__gm__ float*)rstd + coreIdx * blockFactor_);
        gammaGm_.SetGlobalBuffer((__gm__ T_GAMMA*)gamma);
        dxGm_.SetGlobalBuffer((__gm__ T_DX*)dx + coreIdx * blockFactor_ * cols_);

        Ppipe_->InitBuffer(inQueueDy_, DB_NUM, ubFactor_ * sizeof(float));
        Ppipe_->InitBuffer(inQueueX_, DB_NUM, ubFactor_ * sizeof(float));
        Ppipe_->InitBuffer(inQueueRstd_, DB_NUM, AlignUp(ubFactorN_, V_LENGTH) * sizeof(float));
        Ppipe_->InitBuffer(outQueueDx_, DB_NUM, ubFactor_ * sizeof(float));
        Ppipe_->InitBuffer(inQueueGamma_, 1, ubFactor_ * sizeof(float));
        Ppipe_->InitBuffer(reduceBuf_, ubFactorN_ * colsAlign2VL_ * sizeof(float));
        Ppipe_->InitBuffer(tmpSumBuf_, AlignUp(ubFactorN_, V_LENGTH) * sizeof(float));
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
        for (int64_t rowIdx = 0; rowIdx < calcRowNum; rowIdx += ubFactorN_) {
            int64_t calcRowNumSub = Min(ubFactorN_, calcRowNumRemain);
            SubProcess(rowIdx, calcRowNumSub);
            calcRowNumRemain -= ubFactorN_;
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
        LocalTensor<float> dyLocal = inQueueDy_.DeQue<float>();
        CopyInX(rowIdx, calcRowNumSub);
        LocalTensor<float> xLocal = inQueueX_.DeQue<float>();
        CopyInRstd(rowIdx, calcRowNumSub);
        LocalTensor<float> rstdLocal = inQueueRstd_.DeQue<float>();
        LocalTensor<T_GAMMA> gammaLocal = gammaLocal_;
        LocalTensor<float> tmpSumLocal = tmpSumBuf_.Get<float>();

        LocalTensor<float> reduceLocal = reduceBuf_.Get<float>();
        uint16_t loopRow = calcRowNumSub;

        constexpr uint32_t oneRepeat = V_LENGTH;
        int64_t cols = colsAlignBlock_;
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
                uint32_t sreg = cols_;
                MaskReg maskReg = CreateMask<float, MaskPattern::ALL>();
                DataCopy<float, LoadDist::DIST_BRC_B32>(rstdReg, rstdAddr + static_cast<uint32_t>(r));
                for (uint16_t i = 0; i < repeatCount; i++) {
                    maskReg = UpdateMask<float>(sreg);
                    LoadAndCast(gammaReg, gammaAddr, maskReg, i * oneRepeat);
                    LoadAndCast(dyReg, dyAddr, maskReg, r * cols + i * oneRepeat);
                    Mul(mulReg2, dyReg, gammaReg, maskReg);
                    LoadAndCast(xReg, xAddr, maskReg, r * cols + i * oneRepeat);
                    Mul(mulReg0, xReg, rstdReg, maskReg);
                    Mul(mulReg3, mulReg2, mulReg0, maskReg);
                    DataCopy(reduceAddr + static_cast<uint32_t>(r * colsAlign2VL_ + i * oneRepeat), mulReg3, maskReg);
                }
            }
        }

        MultiReduceSum(tmpSumLocal, reduceLocal, calcRowNumSub);
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

        __VEC_SCOPE__
        {
            RegTensor<float> gammaReg, dyReg, xReg, rstdReg, meanReg, dxReg, mulReg0, mulReg2, mulReg4, subReg;
            RegTensor<float> scalesXReg, scalesXResultReg, offsetXReg;
            for (uint16_t r = 0; r < loopRow; r++) {
                uint32_t sreg = cols_;
                int64_t cols = colsAlignBlock_;
                int64_t colsAlignHiFP8 = colsAlignHiFP8_;
                MaskReg maskReg = CreateMask<float, AscendC::MicroAPI::MaskPattern::ALL>();
                DataCopy<float, LoadDist::DIST_BRC_B32>(rstdReg, rstdAddr + static_cast<uint32_t>(r));
                DataCopy<float, LoadDist::DIST_BRC_B32>(meanReg, meanAddr + static_cast<uint32_t>(r));
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
                        DataCopy<T_DX, StoreDist::DIST_PACK4_B32>(dxAddr + static_cast<uint32_t>(r * colsAlignHiFP8 + i * oneRepeat), dxRegHif8, maskReg);
                    } else if constexpr(IsSameType<T_DX, int8_t>::value){
                        RegTensor<T_DX> dxRegInt8;
                        RegTensor<half> dxRegFp16;
                        RegTensor<int32_t> dxRegInt32;
                        Cast<int32_t, float, castTraitFp322Int32>(dxRegInt32, scalesXResultReg, maskReg);
                        Cast<float, int32_t, castTraitInt322Fp32>(scalesXResultReg, dxRegInt32, maskReg);
                        Cast<half, float, castTraitFp322Fp16>(dxRegFp16, scalesXResultReg, maskReg);
                        Cast<T_DX, half, castTraitFp162Int8>(dxRegInt8, dxRegFp16, maskReg);
                        DataCopy<T_DX, StoreDist::DIST_PACK4_B32>(dxAddr + static_cast<uint32_t>(r * colsAlignHiFP8 + i * oneRepeat), dxRegInt8, maskReg);
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

    __aicore__ inline void MultiReduceSum(LocalTensor<float>& dstLocal, LocalTensor<float>& srcLocal, int64_t rows)
    {
        __local_mem__ float* srcAddr = (__ubuf__ float*)srcLocal.GetPhyAddr();
        uint32_t colsTail = colsAlign2VL_ - cols_;
        if (colsTail > V_LENGTH) {
            // 当要补的个数大于64时，需要两个寄存器进行填充（一个完整的全0 regtensor 加上 利用shiftleft将非对齐位置补0）
            uint32_t colsStartLastTwoVL = colsAlign2VL_ - V_LENGTH * NUM_TWO;
            uint32_t colsStartLastOneVL = colsAlign2VL_ - V_LENGTH;
            uint32_t colsValidLastTwoVL = V_LENGTH * NUM_TWO - colsTail;
            __VEC_SCOPE__
            {
                RegTensor<float> xTailReg;
                RegTensor<float> xTailRegshiftLeft;
                RegTensor<float> srcReg;
                MaskReg pregTail = UpdateMask<float>(colsValidLastTwoVL);
                MaskReg maskRegAll = CreateMask<float, AscendC::MicroAPI::MaskPattern::ALL>();
                Duplicate(srcReg, 0.0f, maskRegAll);
                for (uint16_t r = 0; r < (uint16_t)rows; r++) {
                    DataCopy(xTailReg, srcAddr + static_cast<uint32_t>(r * colsAlign2VL_ + colsStartLastTwoVL));
                    // 利用shiftleft将非对齐位置补0
                    ShiftLefts(
                        (RegTensor<uint32_t>&)xTailRegshiftLeft, (RegTensor<uint32_t>&)xTailReg, static_cast<int16_t>(0),
                        pregTail);
                    DataCopy(srcAddr + static_cast<uint32_t>(r * colsAlign2VL_ + colsStartLastTwoVL), xTailRegshiftLeft, maskRegAll);
                    DataCopy(srcAddr + static_cast<uint32_t>(r * colsAlign2VL_ + colsStartLastOneVL), srcReg, maskRegAll);
                }
            }
        } else if (colsTail == V_LENGTH) {
            // 当要补的个数等于64时，直接dup一个全0的regtensor进行填充
            uint32_t colsStartLastOneVL = colsAlign2VL_ - V_LENGTH;
            __VEC_SCOPE__
            {
                RegTensor<float> srcReg;
                MaskReg maskRegAll = CreateMask<float, AscendC::MicroAPI::MaskPattern::ALL>();
                Duplicate(srcReg, 0.0f, maskRegAll);
                for (uint16_t r = 0; r < (uint16_t)rows; r++) {
                    DataCopy(srcAddr + static_cast<uint32_t>(r * colsAlign2VL_ + colsStartLastOneVL), srcReg, maskRegAll);
                }
            }
        } else if (colsTail > 0) {
            // 当要补的个数小于64时，利用shiftleft将非对齐位置补0
            uint32_t colsStartLastOneVL = colsAlign2VL_ - V_LENGTH;
            uint32_t colsValidLastOneVL = V_LENGTH - colsTail;
            __VEC_SCOPE__
            {
                RegTensor<float> xTailReg;
                RegTensor<float> xTailRegshiftLeft;
                MaskReg pregTail = UpdateMask<float>(colsValidLastOneVL);
                MaskReg maskRegAll = CreateMask<float, AscendC::MicroAPI::MaskPattern::ALL>();
                for (uint16_t r = 0; r < (uint16_t)rows; r++) {
                    DataCopy(xTailReg, srcAddr + static_cast<uint32_t>(r * colsAlign2VL_ + colsStartLastOneVL));
                    // 利用shiftleft将非对齐位置补0
                    ShiftLefts(
                        (RegTensor<uint32_t>&)xTailRegshiftLeft, (RegTensor<uint32_t>&)xTailReg, static_cast<int16_t>(0),
                        pregTail);
                    DataCopy(srcAddr + static_cast<uint32_t>(r * colsAlign2VL_ + colsStartLastOneVL), xTailRegshiftLeft, maskRegAll);
                }
            }
        }
        uint32_t srcShape[2] = {uint32_t(rows), uint32_t(colsAlign2VL_)};
        AscendC::ReduceSum<float, AscendC::Pattern::Reduce::AR, true>(dstLocal, srcLocal, srcShape, false);
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
    TBuf<TPosition::VECCALC> reduceBuf_;
    TBuf<TPosition::VECCALC> tmpSumBuf_;
    LocalTensor<T_GAMMA> gammaLocal_;
    LocalTensor<T_SCALES_X> scalesXLocal_;
    LocalTensor<T_OFFSET_X> offsetXLocal_;

    uint32_t usedCoreNum_;
    int64_t rows_;
    int64_t cols_;
    int64_t colsAlignBlock_;
    int64_t colsAlign2VL_;
    int64_t colsAlignHiFP8_;
    int64_t blockFactor_;
    int64_t ubFactor_;
    int64_t ubFactorN_;
    int64_t ubFactorD_;
    float avgFactor1_;
};
} // namespace RmsNormGradQuant
#endif // RMS_NORM_GRAD_REGBASE_DX_FULL_LOAD_H
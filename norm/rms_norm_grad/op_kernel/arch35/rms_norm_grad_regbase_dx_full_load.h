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
 * \file rms_norm_grad_regbase_dx_full_load.h
 * \brief RmsNormGrad Regbase DX Full Load kernel File
 */

#ifndef RMS_NORM_GRAD_REGBASE_DX_FULL_LOAD_H
#define RMS_NORM_GRAD_REGBASE_DX_FULL_LOAD_H

#include "kernel_tiling/kernel_tiling.h"
#include "kernel_operator.h"
#include "rms_norm_grad_regbase_common.h"

namespace RmsNormGrad {
using namespace AscendC;
template <typename T_DY, typename T_X, typename T_GAMMA, typename T_DGAMMA>
class RegbaseDxFullLoad {
public:
     __aicore__ inline RegbaseDxFullLoad(TPipe* pipe, const RmsNormGradRegbaseDxTilingData* tilingData)
        : Ppipe_(pipe), tiling_(tilingData)
    {}

    __aicore__ inline void Init(
        __gm__ uint8_t* dy, __gm__ uint8_t* x, __gm__ uint8_t* rstd, __gm__ uint8_t* gamma, __gm__ uint8_t* dx,
        __gm__ uint8_t* dgamma)
    {
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
        colsAlign2VL_ = AlignUp(cols_, FLOAT_NUM_2VL);

        ubFactor_ = UB_FACTOR_DX_FULL_LOAD;
        ubFactorD_ = colsAlign2VL_;
        ubFactorN_ = ubFactor_ / ubFactorD_;
        avgFactor1_ = 1.0f / cols_;

        dyGm_.SetGlobalBuffer((__gm__ T_DY*)dy + coreIdx * blockFactor_ * cols_);
        xGm_.SetGlobalBuffer((__gm__ T_X*)x + coreIdx * blockFactor_ * cols_);
        rstdGm_.SetGlobalBuffer((__gm__ float*)rstd + coreIdx * blockFactor_);
        gammaGm_.SetGlobalBuffer((__gm__ T_GAMMA*)gamma);
        dxGm_.SetGlobalBuffer((__gm__ T_X*)dx + coreIdx * blockFactor_ * cols_);

        Ppipe_->InitBuffer(inQueueDy_, DB_NUM, ubFactor_ * sizeof(float));
        Ppipe_->InitBuffer(inQueueX_, DB_NUM, ubFactor_ * sizeof(float));
        Ppipe_->InitBuffer(inQueueRstd_, DB_NUM, AlignUp(ubFactorN_, V_LENGTH) * sizeof(float));
        Ppipe_->InitBuffer(outQueueDx_, DB_NUM, ubFactor_ * sizeof(float));
        Ppipe_->InitBuffer(gammaBuf_, ubFactor_ * sizeof(float));
        Ppipe_->InitBuffer(reduceBuf_, ubFactorN_ * colsAlign2VL_ * sizeof(float));
        Ppipe_->InitBuffer(tmpSumBuf_, AlignUp(ubFactorN_, V_LENGTH) * sizeof(float));
    }
    __aicore__ inline void Process()
    {
        uint32_t coreIdx = GetBlockIdx();
        if (coreIdx >= usedCoreNum_) {
            return;
        }
        int64_t blockTail = rows_ - (usedCoreNum_ - 1) * blockFactor_;
        int64_t calcRowNum = coreIdx == usedCoreNum_ - 1 ? blockTail : blockFactor_;
        int64_t calcRowNumRemain = calcRowNum;
        for (int64_t rowIdx = 0; rowIdx < calcRowNum; rowIdx += ubFactorN_) {
            int64_t calcRowNumSub = Min(ubFactorN_, calcRowNumRemain);
            SubProcess(rowIdx, calcRowNumSub);
            calcRowNumRemain -= ubFactorN_;
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
        LocalTensor<T_GAMMA> gammaLocal = gammaBuf_.Get<T_GAMMA>();
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
                MaskReg maskReg = CreateMask<float, MaskPattern::ALL>();
                DataCopy<float, LoadDist::DIST_BRC_B32>(rstdReg, rstdAddr + static_cast<uint32_t>(r));
                for (uint16_t i = 0; i < repeatCount; i++) {
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

        LocalTensor<float> dxLocal = outQueueDx_.AllocTensor<float>();
        __local_mem__ float* meanAddr = (__ubuf__ float*)tmpSumLocal.GetPhyAddr();
        __local_mem__ T_X* dxAddr = (__ubuf__ T_X*)dxLocal.GetPhyAddr();
        __VEC_SCOPE__
        {
            RegTensor<float> gammaReg, dyReg, xReg, rstdReg, meanReg, dxReg, mulReg0, mulReg2, mulReg4, subReg;
            for (uint16_t r = 0; r < loopRow; r++) {
                uint32_t sreg = cols_;
                int64_t cols = colsAlignBlock_;
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
                    if constexpr (IsSameType<T_X, float>::value) {
                        DataCopy(dxAddr + static_cast<uint32_t>(r * cols + i * oneRepeat), dxReg, maskReg);
                    } else {
                        RegTensor<T_X> dxRegB16;
                        Cast<T_X, float, castTraitB322B16>(dxRegB16, dxReg, maskReg);
                        DataCopy<T_X, StoreDist::DIST_PACK_B32>(dxAddr + static_cast<uint32_t>(r * cols + i * oneRepeat), dxRegB16, maskReg);
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
        DataCopyPad(rstdLocal, rstdGm_[rowIdx], copyParams, {true, 0, 0, 0}); // dummy值为0，待确认
        inQueueRstd_.EnQue(rstdLocal);
    }

    __aicore__ inline void CopyInGamma()
    {
        LocalTensor<T_GAMMA> gammaLocal = gammaBuf_.Get<T_GAMMA>();
        DataCopyExtParams copyParams{
            1,                                              // blockCount
            static_cast<uint32_t>(cols_ * sizeof(T_GAMMA)), // blockLen
            0,                                              // srcStride
            0,                                              // dstStride
            0                                               // rsv
        };

        DataCopyPad(gammaLocal, gammaGm_, copyParams, {true, 0, 0, 0});
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
        uint32_t colsTail = colsAlign2VL_ - colsAlignBlock_;
        constexpr uint32_t oneRepeat = V_LENGTH;
        uint16_t repeatCount = DivCeil(colsTail, oneRepeat);
        int64_t cols = colsAlignBlock_;
        if (colsTail > 0) {
            __VEC_SCOPE__
            {
                RegTensor<float> srcReg;
                for (uint16_t r = 0; r < (uint16_t)rows; r++) {
                    MaskReg maskReg = CreateMask<float, AscendC::MicroAPI::MaskPattern::ALL>();
                    Duplicate(srcReg, 0.0f, maskReg);
                    uint32_t sreg = colsTail;
                    for (uint16_t i = 0; i < repeatCount; i++) {
                        maskReg = UpdateMask<float>(sreg);
                        DataCopy(srcAddr + static_cast<uint32_t>(r * colsAlign2VL_ + cols + i * oneRepeat), srcReg, maskReg);
                    }
                }
            }
        }
        uint32_t srcShape[2] = {uint32_t(rows), uint32_t(colsAlign2VL_)};
        AscendC::ReduceSum<float, AscendC::Pattern::Reduce::AR, true>(dstLocal, srcLocal, srcShape, false);
    }

    __aicore__ inline void CopyOutDx(int64_t rowIdx, int64_t calcRow)
    {
        LocalTensor<T_X> dxLocal = outQueueDx_.DeQue<T_X>();
        DataCopyExtParams copyParams{
            static_cast<uint16_t>(calcRow),             // blockCount
            static_cast<uint32_t>(cols_ * sizeof(T_X)), // blockLen
            0,                                          // srcStride
            0,                                          // dstStride
            0                                           // rsv
        };
        DataCopyPad(dxGm_[rowIdx * cols_], dxLocal, copyParams);
        outQueueDx_.FreeTensor(dxLocal);
    }

private:
    TPipe* Ppipe_;
    const RmsNormGradRegbaseDxTilingData* tiling_;
    GlobalTensor<T_DY> dyGm_;
    GlobalTensor<T_X> xGm_;
    GlobalTensor<T_GAMMA> gammaGm_;
    GlobalTensor<float> rstdGm_;
    GlobalTensor<T_X> dxGm_;
    TQue<QuePosition::VECIN, DEPTH_TWO> inQueueDy_;
    TQue<QuePosition::VECIN, DEPTH_TWO> inQueueX_;
    TQue<QuePosition::VECIN, DEPTH_TWO> inQueueRstd_;
    TQue<QuePosition::VECOUT, DEPTH_TWO> outQueueDx_;
    TBuf<TPosition::VECCALC> gammaBuf_;
    TBuf<TPosition::VECCALC> reduceBuf_;
    TBuf<TPosition::VECCALC> tmpSumBuf_;

    uint32_t usedCoreNum_;
    int64_t rows_;
    int64_t cols_;
    int64_t colsAlignBlock_;
    int64_t colsAlign2VL_;
    int64_t blockFactor_;
    int64_t ubFactor_;
    int64_t ubFactorN_;
    int64_t ubFactorD_;
    float avgFactor1_;
};
} // namespace RmsNormGrad
#endif // RMS_NORM_GRAD_REGBASE_DX_FULL_LOAD_H
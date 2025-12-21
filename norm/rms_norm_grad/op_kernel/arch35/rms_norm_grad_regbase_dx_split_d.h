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
 * \file rms_norm_grad_regbase_dx_split_d.h
 * \brief RmsNormGrad Regbase DX Split D kernel File
 */

#ifndef RMS_NORM_GRAD_REGBASE_DX_SPLIT_D_H
#define RMS_NORM_GRAD_REGBASE_DX_SPLIT_D_H

#include "kernel_tiling/kernel_tiling.h"
#include "kernel_operator.h"
#include "rms_norm_grad_regbase_common.h"

namespace RmsNormGrad {
using namespace AscendC;
template <typename T_DY, typename T_X, typename T_GAMMA, typename T_DGAMMA>
class RegbaseDxSplitD {
public:
    __aicore__ inline RegbaseDxSplitD(TPipe* pipe, const RmsNormGradRegbaseTilingData* tilingData)
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
        blockFactor_ = tiling_->blockFactorDx; // ceilDiv(rows_, usedCoreNum)
        ubFactorD_ = UB_FACTOR_DX_SPLIT_D;     // 固定值
        bodyPart_ = tiling_->bodyPart;         // 小于cols的最大二次幂
        avgFactor1_ = 1.0f / cols_;
        dyGm_.SetGlobalBuffer((__gm__ T_DY*)dy + coreIdx * blockFactor_ * cols_);
        xGm_.SetGlobalBuffer((__gm__ T_X*)x + coreIdx * blockFactor_ * cols_);
        rstdGm_.SetGlobalBuffer((__gm__ float*)rstd + coreIdx * blockFactor_);
        gammaGm_.SetGlobalBuffer((__gm__ T_GAMMA*)gamma);
        dxGm_.SetGlobalBuffer((__gm__ T_X*)dx + coreIdx * blockFactor_ * cols_);

        Ppipe_->InitBuffer(inQueueDy_, DB_NUM, ubFactorD_ * sizeof(float));
        Ppipe_->InitBuffer(inQueueX_, DB_NUM, ubFactorD_ * sizeof(float));
        Ppipe_->InitBuffer(inQueueRstd_, DB_NUM, V_LENGTH * sizeof(float));
        Ppipe_->InitBuffer(inQueueGamma_, DB_NUM, ubFactorD_ * sizeof(float));
        Ppipe_->InitBuffer(outQueueDx_, DB_NUM, ubFactorD_ * sizeof(float));
        Ppipe_->InitBuffer(reduceBuf_, DB_NUM * ubFactorD_ * sizeof(float));
        Ppipe_->InitBuffer(level0Buf_, ONCE_VECTOR_SIZE * sizeof(float));
        Ppipe_->InitBuffer(level1Buf_, ONCE_VECTOR_SIZE * sizeof(float));
        Ppipe_->InitBuffer(level2Buf_, ONCE_VECTOR_SIZE * sizeof(float));
        Ppipe_->InitBuffer(tmpSumBuf_, V_LENGTH * sizeof(float));
        Ppipe_->InitBuffer(workBuf_, ONCE_VECTOR_SIZE * sizeof(float));
    }
    __aicore__ inline void Process()
    {
        uint32_t coreIdx = GetBlockIdx();
        if (coreIdx >= usedCoreNum_) {
            return;
        }
        int64_t blockTail = rows_ - (usedCoreNum_ - 1) * blockFactor_;
        int64_t calcRowNum = coreIdx == usedCoreNum_ - 1 ? blockTail : blockFactor_;
        for (int64_t rowIdx = 0; rowIdx < calcRowNum; rowIdx++) {
            SubProcess(rowIdx);
        }
    }

    __aicore__ inline void SubProcess(int64_t rowIdx)
    {
        CopyInRstd(rowIdx, 1);
        LocalTensor<float> rstdLocal = inQueueRstd_.DeQue<float>();
        FormerProcess(rstdLocal, rowIdx);
        LocalTensor<float> tmpSumLocal = tmpSumBuf_.Get<float>();
        Muls(tmpSumLocal, tmpSumLocal, avgFactor1_, 1);
        LatterProcess(rstdLocal, rowIdx);
        inQueueRstd_.FreeTensor(rstdLocal);
    }

    __aicore__ inline void FormerProcess(LocalTensor<float>& rstdLocal, int64_t rowIdx)
    {
        uint32_t level0Offset = 0;
        uint32_t level1Offset = 0;
        uint32_t level2Offset = 0;
        InitLevelLocal();
        for (int64_t colIdx = 0; colIdx < bodyPart_; colIdx += ubFactorD_) {
            CopyInGamma(colIdx, ubFactorD_);
            CopyInDy(rowIdx, colIdx, ubFactorD_);
            CopyInX(rowIdx, colIdx, ubFactorD_);
            ComputeMul<true>(rstdLocal, ubFactorD_);
            int64_t tailCount = 0;
            if (bodyPart_ + colIdx < cols_) {
                int64_t remainCount = cols_ - bodyPart_ - colIdx; // must > 0
                tailCount = Min(remainCount, ubFactorD_);
                CopyInGamma(bodyPart_ + colIdx, tailCount);
                CopyInDy(rowIdx, bodyPart_ + colIdx, tailCount);
                CopyInX(rowIdx, bodyPart_ + colIdx, tailCount);
                ComputeMul<false>(rstdLocal, tailCount);
            }
            int64_t reduceCount = ubFactorD_ + tailCount; // [ubFactorD, 2*ubFactorD_]
            ComputeIntoMultiLevel(reduceCount, level0Offset, level1Offset, level2Offset);
        }
        FinalLevelReduce(level0Offset, level1Offset);
    }

    __aicore__ inline void InitLevelLocal()
    {
        LocalTensor<float> level0Local = level0Buf_.Get<float>();
        LocalTensor<float> level1Local = level1Buf_.Get<float>();
        LocalTensor<float> level2Local = level2Buf_.Get<float>();

        Duplicate(level0Local, 0.0f, ONCE_VECTOR_SIZE);
        Duplicate(level1Local, 0.0f, ONCE_VECTOR_SIZE);
        Duplicate(level2Local, 0.0f, ONCE_VECTOR_SIZE);
    }

    __aicore__ inline void ComputeIntoMultiLevel(
        int64_t count, uint32_t& level0Offset, uint32_t& level1Offset, uint32_t& level2Offset)
    {
        LocalTensor<float> level0Local = level0Buf_.Get<float>();
        LocalTensor<float> level1Local = level1Buf_.Get<float>();
        LocalTensor<float> level2Local = level2Buf_.Get<float>();
        LocalTensor<float> reduceLocal = reduceBuf_.Get<float>();
        WholeReduceSum(level0Local, reduceLocal, count, level0Offset);
        level0Offset++;
        ComputeMultiLevelReduce(level0Local, level1Local, level2Local, level0Offset, level1Offset, level2Offset);
    }

    __aicore__ inline void FinalLevelReduce(uint32_t& level0Offset, uint32_t& level1Offset)
    {
        LocalTensor<float> level0Local = level0Buf_.Get<float>();
        LocalTensor<float> level1Local = level1Buf_.Get<float>();
        LocalTensor<float> level2Local = level2Buf_.Get<float>();
        LocalTensor<float> tmpSumLocal = tmpSumBuf_.Get<float>();
        ComputeMultiLevelMean(tmpSumLocal, 0, level0Local, level1Local, level2Local, level0Offset, level1Offset);
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

    __aicore__ inline void CopyInGamma(int64_t colIdx, int64_t count)
    {
        LocalTensor<T_GAMMA> gammaLocal = inQueueGamma_.AllocTensor<T_GAMMA>();
        DataCopyExtParams copyParams{
            1,                                              // blockCount
            static_cast<uint32_t>(count * sizeof(T_GAMMA)), // blockLen
            0,                                              // srcStride
            0,                                              // dstStride
            0                                               // rsv
        };
        DataCopyPad(gammaLocal, gammaGm_[colIdx], copyParams, {true, 0, 0, 0});
        inQueueGamma_.EnQue(gammaLocal);
    }

    __aicore__ inline void CopyInDy(int64_t rowIdx, int64_t colIdx, int64_t count)
    {
        LocalTensor<T_DY> dyLocal = inQueueDy_.AllocTensor<T_DY>();
        DataCopyExtParams copyParams{
            1,                                           // blockCount
            static_cast<uint32_t>(count * sizeof(T_DY)), // blockLen
            0,                                           // srcStride
            0,                                           // dstStride
            0                                            // rsv
        };
        DataCopyPad(dyLocal, dyGm_[rowIdx * cols_ + colIdx], copyParams, {true, 0, 0, 0});
        inQueueDy_.EnQue(dyLocal);
    }

    __aicore__ inline void CopyInX(int64_t rowIdx, int64_t colIdx, int64_t count)
    {
        LocalTensor<T_X> xLocal = inQueueX_.AllocTensor<T_X>();
        DataCopyExtParams copyParams{
            1,                                          // blockCount
            static_cast<uint32_t>(count * sizeof(T_X)), // blockLen
            0,                                          // srcStride
            0,                                          // dstStride
            0                                           // rsv
        };
        DataCopyPad(xLocal, xGm_[rowIdx * cols_ + colIdx], copyParams, {true, 0, 0, 0});
        inQueueX_.EnQue(xLocal);
    }

    template <bool IsBody>
    __aicore__ inline void ComputeMul(LocalTensor<float>& rstdLocal, int64_t count)
    {
        LocalTensor<float> reduceLocal = reduceBuf_.Get<float>();
        LocalTensor<float> gammaLocal = inQueueGamma_.DeQue<float>();
        LocalTensor<float> dyLocal = inQueueDy_.DeQue<float>();
        LocalTensor<float> xLocal = inQueueX_.DeQue<float>();

        uint32_t sreg = count;
        constexpr uint32_t oneRepeat = V_LENGTH;
        uint16_t repeatCount = DivCeil(count, oneRepeat);
        __local_mem__ T_GAMMA* gammaAddr = (__ubuf__ T_GAMMA*)gammaLocal.GetPhyAddr();
        __local_mem__ T_DY* dyAddr = (__ubuf__ T_DY*)dyLocal.GetPhyAddr();
        __local_mem__ T_X* xAddr = (__ubuf__ T_X*)xLocal.GetPhyAddr();
        __local_mem__ float* rstdAddr = (__ubuf__ float*)rstdLocal.GetPhyAddr();
        __local_mem__ float* reduceAddr = (__ubuf__ float*)reduceLocal.GetPhyAddr();
        __VEC_SCOPE__
        {
            RegTensor<float> gammaReg, dyReg, xReg, rstdReg, mulReg0, mulReg2, mulReg3;
            MaskReg maskReg = CreateMask<float, MaskPattern::ALL>();
            DataCopy<float, LoadDist::DIST_BRC_B32>(rstdReg, rstdAddr);
            for (uint16_t i = 0; i < repeatCount; i++) {
                LoadAndCast(gammaReg, gammaAddr, maskReg, i * oneRepeat);
                LoadAndCast(dyReg, dyAddr, maskReg, i * oneRepeat);
                Mul(mulReg2, dyReg, gammaReg, maskReg);
                LoadAndCast(xReg, xAddr, maskReg, i * oneRepeat);
                Mul(mulReg0, xReg, rstdReg, maskReg);
                Mul(mulReg3, mulReg2, mulReg0, maskReg);
                if constexpr (IsBody) {
                    DataCopy(reduceAddr + i * oneRepeat, mulReg3, maskReg);
                } else {
                    DataCopy(reduceAddr + ubFactorD_ + i * oneRepeat, mulReg3, maskReg); // 注意补零
                }
            }
        }

        inQueueGamma_.FreeTensor(gammaLocal);
        inQueueDy_.FreeTensor(dyLocal);
        inQueueX_.FreeTensor(xLocal);
    }

    __aicore__ inline void WholeReduceSum(
        LocalTensor<float>& dstLocal, LocalTensor<float>& srcLocal, int64_t count, int32_t dstOffset)
    {
        // 对齐到512BYTE, reduce需要
        int64_t countBlockAlign = AlignUp(count, FLOAT_NUM_BLOCK); // 搬入已对齐
        int64_t count2VLAlign = AlignUp(count, FLOAT_NUM_2VL);
        if (count2VLAlign - countBlockAlign > 0) {
            Duplicate(srcLocal[countBlockAlign], 0.0f, count2VLAlign - countBlockAlign);
        }
        int64_t power = count2VLAlign < NUM_TWO * ubFactorD_ ?
                            ubFactorD_ :
                            NUM_TWO * ubFactorD_; // 等于2*UbFactorD_时设为相同大小，否则为其一半
        LocalTensor<float> workLocal = workBuf_.Get<float>();
        ReduceSumImpl(dstLocal, srcLocal, workLocal, dstOffset, count2VLAlign, power);
    }

    __aicore__ inline void LatterProcess(LocalTensor<float>& rstdLocal, int64_t rowIdx)
    {
        for (int64_t colIdx = 0; colIdx < cols_; colIdx += ubFactorD_) {
            int64_t remainCount = cols_ - colIdx;
            int64_t calcCount = Min(remainCount, ubFactorD_);
            CopyInGamma(colIdx, calcCount);
            CopyInDy(rowIdx, colIdx, calcCount);
            CopyInX(rowIdx, colIdx, calcCount);
            ComputeLatter(rstdLocal, calcCount);
            CopyOutDx(rowIdx, colIdx, calcCount);
        }
    }

    __aicore__ inline void ComputeLatter(LocalTensor<float>& rstdLocal, int64_t count)
    {
        LocalTensor<float> gammaLocal = inQueueGamma_.DeQue<float>();
        LocalTensor<float> dyLocal = inQueueDy_.DeQue<float>();
        LocalTensor<float> xLocal = inQueueX_.DeQue<float>();
        LocalTensor<float> dxLocal = outQueueDx_.AllocTensor<float>();
        LocalTensor<float> tmpSumLocal = tmpSumBuf_.Get<float>();

        uint32_t sreg = count;
        constexpr uint32_t oneRepeat = V_LENGTH;
        uint16_t repeatCount = DivCeil(count, oneRepeat); // 可能会报错
        __local_mem__ T_GAMMA* gammaAddr = (__ubuf__ T_GAMMA*)gammaLocal.GetPhyAddr();
        __local_mem__ T_DY* dyAddr = (__ubuf__ T_DY*)dyLocal.GetPhyAddr();
        __local_mem__ T_X* xAddr = (__ubuf__ T_X*)xLocal.GetPhyAddr();
        __local_mem__ float* rstdAddr = (__ubuf__ float*)rstdLocal.GetPhyAddr();
        __local_mem__ float* meanAddr = (__ubuf__ float*)tmpSumLocal.GetPhyAddr();
        __local_mem__ T_X* dxAddr = (__ubuf__ T_X*)dxLocal.GetPhyAddr();
        __VEC_SCOPE__
        {
            RegTensor<float> gammaReg, dyReg, xReg, rstdReg, meanReg, dxReg, mulReg0, mulReg2, mulReg4, subReg;
            MaskReg maskReg;
            DataCopy<float, LoadDist::DIST_BRC_B32>(rstdReg, rstdAddr);
            DataCopy<float, LoadDist::DIST_BRC_B32>(meanReg, meanAddr);
            for (uint16_t i = 0; i < repeatCount; i++) {
                maskReg = UpdateMask<float>(sreg);
                LoadAndCast(gammaReg, gammaAddr, maskReg, i * oneRepeat);
                LoadAndCast(dyReg, dyAddr, maskReg, i * oneRepeat);
                Mul(mulReg2, dyReg, gammaReg, maskReg);
                LoadAndCast(xReg, xAddr, maskReg, i * oneRepeat);
                Mul(mulReg0, xReg, rstdReg, maskReg);
                Mul(mulReg4, mulReg0, meanReg, maskReg);
                Sub(subReg, mulReg2, mulReg4, maskReg);
                Mul(dxReg, subReg, rstdReg, maskReg);
                if constexpr (IsSameType<T_X, float>::value) {
                    DataCopy(dxAddr + i * oneRepeat, dxReg, maskReg);
                } else {
                    RegTensor<T_X> dxRegB16;
                    Cast<T_X, float, castTraitB322B16>(dxRegB16, dxReg, maskReg);
                    DataCopy<T_X, StoreDist::DIST_PACK_B32>(dxAddr + i * oneRepeat, dxRegB16, maskReg);
                }
            }
        }

        inQueueGamma_.FreeTensor(gammaLocal);
        inQueueX_.FreeTensor(xLocal);
        inQueueDy_.FreeTensor(dyLocal);
        outQueueDx_.EnQue(dxLocal);
        // Mul mean
    }

    __aicore__ inline void CopyOutDx(int64_t rowIdx, int64_t colIdx, int64_t count)
    {
        LocalTensor<T_X> dxLocal = outQueueDx_.DeQue<T_X>();
        DataCopyExtParams copyParams{
            1,                                          // blockCount
            static_cast<uint32_t>(count * sizeof(T_X)), // blockLen
            0,                                          // srcStride
            0,                                          // dstStride
            0                                           // rsv
        };
        DataCopyPad(dxGm_[rowIdx * cols_ + colIdx], dxLocal, copyParams);
        outQueueDx_.FreeTensor(dxLocal);
    }

private:
    TPipe* Ppipe_;
    const RmsNormGradRegbaseTilingData* tiling_;
    GlobalTensor<T_DY> dyGm_;
    GlobalTensor<T_X> xGm_;
    GlobalTensor<T_GAMMA> gammaGm_;
    GlobalTensor<float> rstdGm_;
    GlobalTensor<T_X> dxGm_;
    TQue<QuePosition::VECIN, DEPTH_TWO> inQueueDy_;
    TQue<QuePosition::VECIN, DEPTH_TWO> inQueueX_;
    TQue<QuePosition::VECIN, DEPTH_TWO> inQueueRstd_;
    TQue<QuePosition::VECIN, DEPTH_TWO> inQueueGamma_;
    TQue<QuePosition::VECOUT, DEPTH_TWO> outQueueDx_;
    TBuf<TPosition::VECCALC> reduceBuf_;
    TBuf<TPosition::VECCALC> level0Buf_;
    TBuf<TPosition::VECCALC> level1Buf_;
    TBuf<TPosition::VECCALC> level2Buf_;
    TBuf<TPosition::VECCALC> tmpSumBuf_;
    TBuf<TPosition::VECCALC> workBuf_;
    uint32_t usedCoreNum_;
    int64_t rows_;
    int64_t cols_;
    int64_t blockFactor_;
    int64_t ubFactorD_;
    int64_t bodyPart_;
    float avgFactor1_;
};
} // namespace RmsNormGrad
#endif // RMS_NORM_GRAD_REGBASE_DX_SPLIT_D_H
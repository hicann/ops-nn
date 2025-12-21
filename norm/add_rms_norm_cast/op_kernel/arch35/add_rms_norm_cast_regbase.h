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
 * \file add_rms_norm_cast_regbase_normal.h
 * \brief AddRmsNormCast regbase normal template.
 */
#ifndef OPS_BUILT_IN_TBE_IMPL_ASCENDC_ADD_RMS_NORM_CAST_REGBASE_H
#define OPS_BUILT_IN_TBE_IMPL_ASCENDC_ADD_RMS_NORM_CAST_REGBASE_H
#include "add_rms_norm_cast_regbase_common.h"

namespace AddRmsNormCast {
template <typename T_X>
class KernelAddRmsNormCastRegBaseNormal {
public:
    __aicore__ inline KernelAddRmsNormCastRegBaseNormal(TPipe* pipe)
    {
        pipe_ = pipe;
    }

    __aicore__ inline void Init(
        GM_ADDR x1, GM_ADDR x2, GM_ADDR gamma, GM_ADDR y1, GM_ADDR y2, GM_ADDR rstd, GM_ADDR x, GM_ADDR workspace,
        const AddRmsNormCastRegbaseTilingData* tilingData)
    {
        numM_ = tilingData->numM;
        numN_ = tilingData->numN;
        baseM_ = tilingData->baseM;
        baseN_ = tilingData->baseN;
        baseNDtypeAlign_ = tilingData->baseNDtypeAlign;
        baseNReduceAlign_ = tilingData->baseNReduceAlign;
        powerSplit_ = tilingData->powerSplit;
        mPerCore_ = tilingData->mPerCore;
        mLastCore_ = tilingData->mLastCore;
        epsilon_ = tilingData->epsilon;
        avgFactor_ = tilingData->avgFactor;
        isNddma = tilingData->isNddma;

        blockNum_ = GetBlockNum();
        blockIdx_ = GetBlockIdx();

        CalBlockTail();
        InitBuffer(x1, x2, gamma, y1, y2, rstd, x);
    }

    __aicore__ inline void CalBlockTail()
    {
        mCore_ = blockIdx_ == (blockNum_ - 1) ? mLastCore_ : mPerCore_;
        mOuterCnt_ = CeilDiv(mCore_, baseM_);
        tailMOuter_ = mCore_ - (mOuterCnt_ - 1) * baseM_;

        uint64_t reduceSumBufLen = baseNReduceAlign_ / (2 * V_LENGTH);
        reduceSumBufAlign_ = CeilAlign(reduceSumBufLen, AddRmsNormCast::B32_BLOCK_NUM);
        baseNB32Align_ = CeilAlign(baseN_, AddRmsNormCast::B32_BLOCK_NUM);
        dtypeBlockStride_ = (baseNReduceAlign_ - baseNDtypeAlign_) * sizeof(T_X) / ALIGN_32_FACTOR; // Block
        b32BlockStride_ = (baseNReduceAlign_ - baseNB32Align_) * sizeof(float) / ALIGN_32_FACTOR;   // Block
    }

    __aicore__ inline void InitBuffer(
        GM_ADDR x1, GM_ADDR x2, GM_ADDR gamma, GM_ADDR y1, GM_ADDR y2, GM_ADDR rstd, GM_ADDR x)
    {
        // Init GM
        uint64_t gmOffset = blockIdx_ * mPerCore_ * numN_;
        uint64_t gmLen = mCore_ * numN_;
        uint64_t rstdGmOffset = blockIdx_ * mPerCore_;
        x1Gm_.SetGlobalBuffer((__gm__ T_X*)x1 + gmOffset, gmLen);
        x2Gm_.SetGlobalBuffer((__gm__ T_X*)x2 + gmOffset, gmLen);
        y1Gm_.SetGlobalBuffer((__gm__ float*)y1 + gmOffset, gmLen);
        y2Gm_.SetGlobalBuffer((__gm__ T_X*)y2 + gmOffset, gmLen);
        xGm_.SetGlobalBuffer((__gm__ T_X*)x + gmOffset, gmLen);
        gammaGm_.SetGlobalBuffer((__gm__ T_X*)gamma, numN_);
        rstdGm_.SetGlobalBuffer((__gm__ float*)rstd + rstdGmOffset, mCore_);

        InitUBBuffer();
    }

    __aicore__ inline void InitUBBuffer()
    {
        // Init Buffer
        uint64_t ubFactorXY = baseM_ * baseNReduceAlign_;
        uint64_t ubFactorRstd = CeilAlign(baseM_, AddRmsNormCast::B32_BLOCK_NUM);
        pipe_->InitBuffer(inQueueX1_, 1, ubFactorXY * sizeof(T_X));
        pipe_->InitBuffer(inQueueX2_, 1, ubFactorXY * sizeof(T_X));
        pipe_->InitBuffer(outQueueX_, 1, ubFactorXY * sizeof(T_X));
        pipe_->InitBuffer(inQueueGamma_, 1, baseNReduceAlign_ * sizeof(T_X));
        pipe_->InitBuffer(outQueueY1_, 1, ubFactorXY * sizeof(float));
        pipe_->InitBuffer(outQueueY2_, 1, ubFactorXY * sizeof(T_X));
        pipe_->InitBuffer(outQueueRstd_, 1, ubFactorRstd * sizeof(float));
        pipe_->InitBuffer(xOutTmpBuf_, ubFactorXY * sizeof(float));
        pipe_->InitBuffer(reduceSumBuf_, reduceSumBufAlign_ * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        CopyInGamma();
        LocalTensor<T_X> gammaLocal = inQueueGamma_.DeQue<T_X>();
        for (uint64_t mOuterIdx = 0; mOuterIdx < mOuterCnt_; mOuterIdx++) {
            uint64_t realM = mOuterIdx == (mOuterCnt_ - 1) ? tailMOuter_ : baseM_;
            uint64_t mOuterOffset = mOuterIdx * baseM_;
            Compute(gammaLocal, mOuterOffset, realM);
        }
        inQueueGamma_.FreeTensor(gammaLocal);
    }

private:
    template <typename T>
    __aicore__ inline void CopyInXMultiNddma(
        TQue<QuePosition::VECIN, 1>& inQueueX, GlobalTensor<T>& srcGm, uint64_t gmOffset, uint64_t repeatTimes,
        uint64_t blockLen, uint64_t blockLenAlign)
    {
        LocalTensor<T> xLocal = inQueueX.AllocTensor<T>();
        MultiCopyParams<T, NDDMA_DIM> dmaParam = {
            {
                {1, 1, 1, (uint32_t)blockLen, 1},                     // SrcStride
                {1, 1, 1, (uint32_t)blockLenAlign, 1},                // DstStride
                {1, 1, 1, (uint32_t)repeatTimes, (uint32_t)blockLen}, // repeatTimes
                {0, 0, 0, 0, 0},                                      // leftPad = 0
                {0, 0, 0, 0, (uint8_t)(blockLenAlign - blockLen)}     // rightPad, can pad bigger than 32B.
            },
            0}; // pad value
        DataCopy(xLocal, srcGm[gmOffset], dmaParam);
        inQueueX.EnQue(xLocal);
    }

    template <typename T>
    __aicore__ inline void CopyInXMultiMoveAlign(
        TQue<QuePosition::VECIN, 1>& inQueueX, GlobalTensor<T>& srcGm, uint64_t gmOffset, uint64_t repeatTimes,
        uint64_t blockLen, uint64_t blockLenDtypeAlign)
    {
        LocalTensor<T> xLocal = inQueueX.AllocTensor<T>();
        DataCopyExtParams extParams{
            static_cast<uint16_t>(repeatTimes),          // blockCount
            static_cast<uint32_t>(blockLen * sizeof(T)), // blockLen
            static_cast<uint32_t>(0),                    // srcStride
            static_cast<uint32_t>(dtypeBlockStride_),    // dstStride
            0                                            // rsv
        };
        DataCopyPadExtParams<T> padParams{
            true,                                                // isPad
            static_cast<uint8_t>(0),                             // leftPadding
            static_cast<uint8_t>(blockLenDtypeAlign - blockLen), // rightPadding
            static_cast<T>(0.0)                                  // paddingValue
        };
        DataCopyPad(xLocal, srcGm[gmOffset], extParams, padParams);
        inQueueX.EnQue(xLocal);
    }

    __aicore__ inline void CopyInGamma()
    {
        LocalTensor<T_X> gammaLocal = inQueueGamma_.AllocTensor<T_X>();
        RmsNorm::DataCopyImpl<T_X>(gammaLocal, gammaGm_, 1, numN_);
        inQueueGamma_.EnQue(gammaLocal);
    }

    template <typename T>
    __aicore__ inline void CopyOutXYMulti(
        GlobalTensor<T>& yGm, TQue<QuePosition::VECOUT, 1>& outQueueY, uint64_t gmOffset, uint32_t repeatTimes,
        uint32_t srcStride, uint32_t dstStride)
    {
        LocalTensor<T> yLocal = outQueueY.DeQue<T>();
        RmsNorm::DataCopyImpl<T>(yGm[gmOffset], yLocal, repeatTimes, numN_, srcStride, dstStride);
        outQueueY.FreeTensor(yLocal);
    }

    __aicore__ inline void CopyOutRstd(uint64_t gmOffset, uint32_t num)
    {
        LocalTensor<float> rstdLocal = outQueueRstd_.DeQue<float>();
        RmsNorm::DataCopyImpl<float>(rstdGm_[gmOffset], rstdLocal, 1, num, 0, 0);
        outQueueRstd_.FreeTensor(rstdLocal);
    }

    __aicore__ inline void Compute(LocalTensor<T_X>& gammaLocal, uint64_t mOuterOffset, uint64_t realM)
    {
        LocalTensor<float> reduceLocal = reduceSumBuf_.Get<float>();
        LocalTensor<float> xOutTmpLocal = xOutTmpBuf_.Get<float>();
        uint64_t gmOffsetXY = mOuterOffset * numN_;

        if (isNddma) {
            CopyInXMultiNddma(inQueueX1_, x1Gm_, gmOffsetXY, realM, numN_, baseNReduceAlign_);
            CopyInXMultiNddma(inQueueX2_, x2Gm_, gmOffsetXY, realM, numN_, baseNReduceAlign_);
        } else {
            CopyInXMultiMoveAlign(inQueueX1_, x1Gm_, gmOffsetXY, realM, numN_, baseNDtypeAlign_);
            CopyInXMultiMoveAlign(inQueueX2_, x2Gm_, gmOffsetXY, realM, numN_, baseNDtypeAlign_);
        }
        LocalTensor<T_X> x1Local = inQueueX1_.DeQue<T_X>();
        LocalTensor<T_X> x2Local = inQueueX2_.DeQue<T_X>();

        uint64_t dupLen = baseNReduceAlign_ - baseNDtypeAlign_;
        if (dupLen > 0 && !isNddma) {
            for (uint32_t mIdx = 0; mIdx < realM; mIdx++) {
                Duplicate(x1Local[mIdx * baseNReduceAlign_ + baseNDtypeAlign_], (T_X)0.0, dupLen);
                Duplicate(x2Local[mIdx * baseNReduceAlign_ + baseNDtypeAlign_], (T_X)0.0, dupLen);
            }
            PipeBarrier<PIPE_V>();
        }

        // 1. Calc xOut
        LocalTensor<float> rstdLocal = outQueueRstd_.AllocTensor<float>();
        LocalTensor<T_X> xOutLocal = outQueueX_.AllocTensor<T_X>();
        ReduceSumRstdMulti<T_X, true, true, true>(
            rstdLocal, xOutLocal, xOutTmpLocal, x1Local, x2Local, reduceLocal, 0, baseNReduceAlign_, powerSplit_, realM,
            avgFactor_, epsilon_);
        inQueueX1_.FreeTensor(x1Local);
        inQueueX2_.FreeTensor(x2Local);
        outQueueX_.EnQue<T_X>(xOutLocal);
        CopyOutXYMulti(xGm_, outQueueX_, gmOffsetXY, realM, dtypeBlockStride_, 0);

        // 2. Calc y1/y2
        LocalTensor<float> y1Local = outQueueY1_.AllocTensor<float>();
        LocalTensor<T_X> y2Local = outQueueY2_.AllocTensor<T_X>();
        AddRmsNormCast::ComputeYMulti<T_X, T_X>(
            y1Local, y2Local, xOutTmpLocal, gammaLocal, rstdLocal, 0, baseNReduceAlign_, realM);
        outQueueRstd_.EnQue<float>(rstdLocal);
        outQueueY1_.EnQue<float>(y1Local);
        outQueueY2_.EnQue<T_X>(y2Local);
        CopyOutRstd(mOuterOffset, realM);
        CopyOutXYMulti(y1Gm_, outQueueY1_, gmOffsetXY, realM, b32BlockStride_, 0);
        CopyOutXYMulti(y2Gm_, outQueueY2_, gmOffsetXY, realM, dtypeBlockStride_, 0);
    }

private:
    TPipe* pipe_ = nullptr;
    // GM Buffer
    GlobalTensor<T_X> x1Gm_;
    GlobalTensor<T_X> x2Gm_;
    GlobalTensor<T_X> gammaGm_;
    GlobalTensor<float> y1Gm_;
    GlobalTensor<T_X> y2Gm_;
    GlobalTensor<float> rstdGm_;
    GlobalTensor<T_X> xGm_;
    // UB Buffer
    TQue<QuePosition::VECIN, 1> inQueueX1_;
    TQue<QuePosition::VECIN, 1> inQueueX2_;
    TQue<QuePosition::VECIN, 1> inQueueGamma_;
    TQue<QuePosition::VECOUT, 1> outQueueY1_;
    TQue<QuePosition::VECOUT, 1> outQueueY2_;
    TQue<QuePosition::VECOUT, 1> outQueueRstd_;
    TQue<QuePosition::VECOUT, 1> outQueueX_;
    TBuf<TPosition::VECCALC> y1TmpBuf_;
    TBuf<TPosition::VECCALC> y2TmpBuf_;
    TBuf<TPosition::VECCALC> xOutTmpBuf_;
    TBuf<TPosition::VECCALC> reduceSumBuf_;

    // Tiling data
    uint64_t numN_{0};
    uint64_t numM_{0};
    uint64_t baseM_{0};
    uint64_t baseN_{0};
    uint64_t baseNDtypeAlign_{0};
    uint64_t baseNReduceAlign_{0};
    uint64_t baseNB32Align_{0};
    uint64_t reduceSumBufAlign_{0};
    uint64_t powerSplit_{0};
    uint64_t mPerCore_{0};
    uint64_t mLastCore_{0};
    float epsilon_{0};
    float avgFactor_{0};
    // Platform
    int64_t blockIdx_{0};
    int64_t blockNum_{0};
    // Cal params
    uint64_t mCore_;
    uint64_t mOuterCnt_;
    uint64_t tailMOuter_;
    uint32_t dtypeBlockStride_;
    uint32_t b32BlockStride_;
    // Other
    uint32_t isNddma{0};
    uint32_t vfLength_{0};
};
} // namespace AddRmsNormCast
#endif // OPS_BUILT_IN_TBE_IMPL_ASCENDC_ADD_RMS_NORM_CAST_REGBASE_H

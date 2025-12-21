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
 * \file add_rms_norm_quant_regbase_split_reduce.h
 * \brief
 */
#ifndef ADD_RMS_NORM_QUANT_REBASE_REDUCE_H_
#define ADD_RMS_NORM_QUANT_REBASE_REDUCE_H_

#include "add_rms_norm_quant_regbase_common.h"

namespace AddRmsNormQuant {

template <typename T_X, typename T_Y, typename T_SCALES, typename T_ZEROPOINTS, uint64_t TILING_KEY>
class KernelAddRmsNormQuantRegbaseSpiltReduce {
#define DIV_MODE ((TILING_KEY / 100) == 1)
#define INPUT_KEY ((TILING_KEY % 100) / 10)
#define HAS_ZEROPOINTS1 ((INPUT_KEY >> 2) % 2 == 1)
#define HAS_SCALE2 ((INPUT_KEY >> 1) % 2 == 1)
#define HAS_ZEROPOINTS2 (INPUT_KEY % 2 == 1)
#define HAS_Y2 (HAS_SCALE2 || HAS_ZEROPOINTS2)
public:
    __aicore__ inline KernelAddRmsNormQuantRegbaseSpiltReduce(TPipe* pipe)
    {
        pipe_ = pipe;
    }

    __aicore__ inline void Init(
        GM_ADDR x1, GM_ADDR x2, GM_ADDR gamma, GM_ADDR scales1, GM_ADDR scales2, GM_ADDR zeroPoints1,
        GM_ADDR zeroPoints2, GM_ADDR y1, GM_ADDR y2, GM_ADDR x, const AddRmsNormQuantRegbaseTilingData* tilingData)
    {
        numM_ = tilingData->numM;
        numN_ = tilingData->numN;
        baseM_ = tilingData->baseM;
        baseN_ = tilingData->baseN;
        baseNDtypeAlign_ = tilingData->baseNDtypeAlign;
        baseNReduceAlign_ = tilingData->baseNReduceAlign;
        powerSplit_ = tilingData->powerSplit;
        powerLoop_ = tilingData->powerLoop;
        mPerCore_ = tilingData->mPerCore;
        mLastCore_ = tilingData->mLastCore;
        epsilon_ = tilingData->epsilon;
        avgFactor_ = tilingData->avgFactor;

        blockNum_ = GetBlockNum();
        blockIdx_ = GetBlockIdx();

        CalBlockTail();
        InitBuffer(x1, x2, gamma, scales1, scales2, zeroPoints1, zeroPoints2, y1, y2, x);
    }

    __aicore__ inline void CalBlockTail()
    {
        mCore_ = blockIdx_ == (blockNum_ - 1) ? mLastCore_ : mPerCore_;
        nCnt_ = CeilDiv(numN_, baseN_);
        tailN_ = numN_ - (nCnt_ - 1) * baseN_;

        tailNDtypeAlign_ = CeilAlign(tailN_, BLOCK_SIZE / sizeof(T_X));
        baseNB8Align_ = CeilAlign(baseN_, B8_BLOCK_NUM);
        uint64_t reduceBufLen = baseNReduceAlign_ / (2 * V_LENGTH);
        reduceBufAlign_ = CeilAlign(reduceBufLen, B32_BLOCK_NUM);
    }

    __aicore__ inline void InitBuffer(
        GM_ADDR x1, GM_ADDR x2, GM_ADDR gamma, GM_ADDR scales1, GM_ADDR scales2, GM_ADDR zeroPoints1,
        GM_ADDR zeroPoints2, GM_ADDR y1, GM_ADDR y2, GM_ADDR x)
    {
        uint64_t gmOffset = blockIdx_ * mPerCore_ * numN_;
        uint64_t gmLen = mCore_ * numN_;
        x1Gm_.SetGlobalBuffer((__gm__ T_X*)x1 + gmOffset, gmLen);
        x2Gm_.SetGlobalBuffer((__gm__ T_X*)x2 + gmOffset, gmLen);
        gammaGm_.SetGlobalBuffer((__gm__ T_X*)gamma, numN_);
        scales1Gm_.SetGlobalBuffer((__gm__ T_SCALES*)scales1, numN_);
        y1Gm_.SetGlobalBuffer((__gm__ T_Y*)y1 + gmOffset, gmLen);
        y2Gm_.SetGlobalBuffer((__gm__ T_Y*)y2 + gmOffset, gmLen);
        xGm_.SetGlobalBuffer((__gm__ T_X*)x + gmOffset, gmLen);
        if constexpr (HAS_SCALE2) {
            scales2Gm_.SetGlobalBuffer((__gm__ T_SCALES*)scales2, numN_);
        }
        if constexpr (HAS_ZEROPOINTS1) {
            zeroPoints1Gm_.SetGlobalBuffer((__gm__ T_ZEROPOINTS*)zeroPoints1, numN_);
        }
        if constexpr (HAS_ZEROPOINTS2) {
            zeroPoints2Gm_.SetGlobalBuffer((__gm__ T_ZEROPOINTS*)zeroPoints2, numN_);
        }

        uint64_t ubFactorQuant = CeilAlign(baseN_, BLOCK_SIZE / sizeof(T_SCALES));
        uint64_t ubFactorRstd = B32_BLOCK_NUM;
        pipe_->InitBuffer(inQueueX1_, 1, baseNReduceAlign_ * sizeof(T_X));
        pipe_->InitBuffer(inQueueX2_, 1, baseNReduceAlign_ * sizeof(T_X));
        pipe_->InitBuffer(outQueueX_, 1, baseNReduceAlign_ * sizeof(T_X));
        pipe_->InitBuffer(inQueueGamma_, 1, baseNDtypeAlign_ * sizeof(T_X));
        pipe_->InitBuffer(inQueueScales1_, 1, ubFactorQuant * sizeof(T_SCALES));
        pipe_->InitBuffer(outQueueY1_, 1, baseNB8Align_ * sizeof(T_Y));
        if constexpr (HAS_SCALE2) {
            pipe_->InitBuffer(inQueueScales2_, 1, ubFactorQuant * sizeof(T_SCALES));
        }
        if constexpr (HAS_ZEROPOINTS1) {
            pipe_->InitBuffer(inQueueZeroPoints1_, 1, ubFactorQuant * sizeof(T_ZEROPOINTS));
        }
        if constexpr (HAS_ZEROPOINTS2) {
            pipe_->InitBuffer(inQueueZeroPoints2_, 1, ubFactorQuant * sizeof(T_ZEROPOINTS));
        }
        if constexpr (HAS_Y2) {
            pipe_->InitBuffer(outQueueY2_, 1, baseNB8Align_ * sizeof(T_Y));
        }

        pipe_->InitBuffer(rstdBuf_, ubFactorRstd * sizeof(float));
        pipe_->InitBuffer(reduceBuf_, reduceBufAlign_ * sizeof(float));
        pipe_->InitBuffer(level1Buf_, RmsNorm::ONCE_VECTOR_SIZE * sizeof(float));
        pipe_->InitBuffer(level2Buf_, RmsNorm::ONCE_VECTOR_SIZE * sizeof(float));
        pipe_->InitBuffer(level3Buf_, RmsNorm::ONCE_VECTOR_SIZE * sizeof(float));
        pipe_->InitBuffer(tempBuf_, V_LENGTH * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        uint64_t powerMain = powerSplit_ * powerLoop_;
        uint64_t powerTail = numN_ - powerMain;
        uint64_t powerTailLoop = powerTail / powerSplit_;
        uint64_t powerTailTail = powerTail % powerSplit_;
        uint64_t masterLoop = powerTailTail != 0 ? 1 : 0;
        masterLoop = powerLoop_ - powerTailLoop - masterLoop;
        LocalTensor<float> rstdLocal = rstdBuf_.Get<float>();
        for (uint64_t mIdx = 0; mIdx < mCore_; mIdx++) {
            uint64_t xGmOffset = mIdx * numN_;
            // Cal Sum and Copy out xOut(x1+x2)
            ComputeFormer<true>(
                rstdLocal, mIdx, xGmOffset, masterLoop, powerTailLoop, powerTailTail, avgFactor_, epsilon_);

            for (uint32_t nIdx = 0; nIdx < nCnt_; nIdx++) {
                uint64_t realN = (nIdx == nCnt_ - 1) ? tailN_ : baseN_;
                uint64_t realNDtypeAlign = (nIdx == nCnt_ - 1) ? tailNDtypeAlign_ : baseNDtypeAlign_;
                CopyInGamma(nIdx * baseN_, realN);
                CopyInQuant(nIdx * baseN_, realN);
                LocalTensor<T_X> gammaLocal = inQueueGamma_.DeQue<T_X>();
                LocalTensor<T_SCALES> scales1Local = inQueueScales1_.DeQue<T_SCALES>();
                LocalTensor<T_SCALES> scales2Local;
                LocalTensor<T_ZEROPOINTS> zeroPoints1Local, zeroPoints2Local;
                if constexpr (HAS_SCALE2) {
                    scales2Local = inQueueScales2_.DeQue<T_SCALES>();
                }
                if constexpr (HAS_ZEROPOINTS1) {
                    zeroPoints1Local = inQueueZeroPoints1_.DeQue<T_ZEROPOINTS>();
                }
                if constexpr (HAS_ZEROPOINTS2) {
                    zeroPoints2Local = inQueueZeroPoints2_.DeQue<T_ZEROPOINTS>();
                }

                uint64_t gmOffset = mIdx * numN_ + nIdx * baseN_;
                CopyInX(inQueueX1_, xGm_, gmOffset, realN, 0, realNDtypeAlign - realN);
                Compute(
                    rstdLocal, gammaLocal, scales1Local, scales2Local, zeroPoints1Local, zeroPoints2Local, mIdx, nIdx);
                CopyOutY(y1Gm_, outQueueY1_, gmOffset, realN);
                if constexpr (HAS_Y2) {
                    CopyOutY(y2Gm_, outQueueY2_, gmOffset, realN);
                }

                inQueueGamma_.FreeTensor(gammaLocal);
                inQueueScales1_.FreeTensor(scales1Local);
                if constexpr (HAS_SCALE2) {
                    inQueueScales2_.FreeTensor(scales2Local);
                }
                if constexpr (HAS_ZEROPOINTS1) {
                    inQueueZeroPoints1_.FreeTensor(zeroPoints1Local);
                }
                if constexpr (HAS_ZEROPOINTS2) {
                    inQueueZeroPoints2_.FreeTensor(zeroPoints2Local);
                }
            }
        }
    }

private:
    __aicore__ inline void CopyInGamma(uint64_t gmOffset, uint64_t blockLen)
    {
        LocalTensor<T_X> gammaLocal = inQueueGamma_.AllocTensor<T_X>();
        RmsNorm::DataCopyImpl<T_X>(gammaLocal, gammaGm_[gmOffset], 1, blockLen);
        inQueueGamma_.EnQue(gammaLocal);
    }

    __aicore__ inline void CopyInQuant(uint64_t gmOffset, uint64_t blockLen)
    {
        LocalTensor<T_SCALES> scales1Local = inQueueScales1_.AllocTensor<T_SCALES>();
        RmsNorm::DataCopyImpl<T_SCALES>(scales1Local, scales1Gm_[gmOffset], 1, blockLen);
        inQueueScales1_.EnQue(scales1Local);
        if constexpr (HAS_SCALE2) {
            LocalTensor<T_SCALES> scales2Local = inQueueScales2_.AllocTensor<T_SCALES>();
            RmsNorm::DataCopyImpl<T_SCALES>(scales2Local, scales2Gm_[gmOffset], 1, blockLen);
            inQueueScales2_.EnQue(scales2Local);
        }
        if constexpr (HAS_ZEROPOINTS1) {
            LocalTensor<T_ZEROPOINTS> zeroPoints1Local = inQueueZeroPoints1_.AllocTensor<T_ZEROPOINTS>();
            RmsNorm::DataCopyImpl<T_ZEROPOINTS>(zeroPoints1Local, zeroPoints1Gm_[gmOffset], 1, blockLen);
            inQueueZeroPoints1_.EnQue(zeroPoints1Local);
        }
        if constexpr (HAS_ZEROPOINTS2) {
            LocalTensor<T_ZEROPOINTS> zeroPoints2Local = inQueueZeroPoints2_.AllocTensor<T_ZEROPOINTS>();
            RmsNorm::DataCopyImpl<T_ZEROPOINTS>(zeroPoints2Local, zeroPoints2Gm_[gmOffset], 1, blockLen);
            inQueueZeroPoints2_.EnQue(zeroPoints2Local);
        }
    }

    /**
     * @brief Copy in xGm_[srcGmOffset:srcGmOffset+count] and reduce to dst[dstOffset].
     */
    __aicore__ inline void ComputeFormerHandle(
        LocalTensor<float>& dstLocal, uint64_t srcGmOffset, uint64_t dstOffset, uint32_t count, uint32_t powerSplit)
    {
        uint32_t calCount = Aligned((uint64_t)(count * sizeof(T_X)), ALIGN_32_FACTOR) / sizeof(T_X);
        CopyInX(inQueueX1_, x1Gm_, srcGmOffset, count, 0, calCount - count);
        CopyInX(inQueueX2_, x2Gm_, srcGmOffset, count, 0, calCount - count);
        LocalTensor<T_X> x1Local = inQueueX1_.DeQue<T_X>();
        LocalTensor<T_X> x2Local = inQueueX2_.DeQue<T_X>();

        LocalTensor<float> workLocal = reduceBuf_.Get<float>();
        uint32_t calNum = Aligned((uint64_t)(count * sizeof(T_X)), ALIGN_512_FACTOR) / sizeof(T_X);
        uint64_t dupLen = calNum - calCount;
        if (dupLen > 0) {
            Duplicate(x1Local[calCount], (T_X)0.0, dupLen);
            Duplicate(x2Local[calCount], (T_X)0.0, dupLen);
        }

        LocalTensor<T_X> xOutLocal = outQueueX_.AllocTensor<T_X>();
        NormCommon::ReduceSumRstd<T_X, true, false, false>(
            dstLocal, xOutLocal, workLocal, x1Local, x2Local, workLocal, dstOffset, calNum, powerSplit);
        outQueueX_.EnQue<T_X>(xOutLocal);
        inQueueX1_.FreeTensor(x1Local);
        inQueueX2_.FreeTensor(x2Local);

        CopyOutX(xGm_, outQueueX_, srcGmOffset, count);
    }

    template <bool IS_RSTD>
    __aicore__ inline void ComputeFormer(
        LocalTensor<float> dstLocal, uint64_t position, uint64_t curRow, uint64_t masterLoop, uint64_t tailLoop,
        uint64_t tail, float avgFactor = 1.0f, float epsilon = 0.0f)
    {
        uint64_t offset{curRow};
        uint32_t level1{0};
        uint32_t level2{0};
        uint32_t level3{0};
        LocalTensor<float> level1Local = level1Buf_.Get<float>();
        LocalTensor<float> level2Local = level2Buf_.Get<float>();
        LocalTensor<float> level3Local = level3Buf_.Get<float>();
        LocalTensor<float> tempLocal = tempBuf_.Get<float>();
        Duplicate(level1Local, (float)0.0, RmsNorm::ONCE_VECTOR_SIZE);
        Duplicate(level2Local, (float)0.0, RmsNorm::ONCE_VECTOR_SIZE);
        Duplicate(level3Local, (float)0.0, RmsNorm::ONCE_VECTOR_SIZE);
        // Stage1: Cal TailLoop
        // Use TailBlock as MainBlock and reduce 2 nearby Mainblock twice.
        for (uint32_t repeat = 0; repeat < tailLoop; repeat++) {
            // Tail
            ComputeFormerHandle(tempLocal, offset, 0, powerSplit_, powerSplit_);
            offset += powerSplit_;
            // Main
            ComputeFormerHandle(tempLocal, offset, 1, powerSplit_, powerSplit_);
            offset += powerSplit_;

            RmsNorm::ComputeSum(level1Local, tempLocal, level1, SUM_COUNT);
            level1 += 1;
            RmsNorm::ComputeMultiLevelReduce(level1Local, level2Local, level3Local, level1, level2, level3);
        }
        // Stage2: Cal TailTail
        // Use TailBlock as MainBlock and reduce 2 nearby Mainblock twice.
        uint64_t powerSplitHalf = powerSplit_ / CONST_FACTOR_2;
        if (tail > 0 && tail <= powerSplitHalf) {
            // Tail
            ComputeFormerHandle(tempLocal, offset, 0, powerSplitHalf + tail, powerSplitHalf);
            offset += powerSplitHalf + tail;
            // Main
            ComputeFormerHandle(tempLocal, offset, 1, powerSplitHalf, powerSplitHalf);
            offset += powerSplitHalf;
        } else if (tail > powerSplitHalf) {
            // Half Main & Half Tail
            ComputeFormerHandle(tempLocal, offset, 0, powerSplit_, powerSplit_);
            offset += powerSplit_;
            // Other Tail
            ComputeFormerHandle(tempLocal, offset, 1, tail, powerSplitHalf);
            offset += tail;
        }
        RmsNorm::ComputeSum(level1Local, tempLocal, level1, SUM_COUNT);
        level1 += 1;
        RmsNorm::ComputeMultiLevelReduce(level1Local, level2Local, level3Local, level1, level2, level3);
        // Stage3: Cal MasterLoop
        for (uint32_t repeat = 0; repeat < masterLoop; repeat++) {
            ComputeFormerHandle(level1Local, offset, level1, powerSplit_, powerSplit_);
            offset += powerSplit_;
            level1 += 1;
            RmsNorm::ComputeMultiLevelReduce(level1Local, level2Local, level3Local, level1, level2, level3);
        }
        ComputeMultiLevelRstd<IS_RSTD>(
            dstLocal, position, level1Local, level2Local, level3Local, level1, level2, avgFactor, epsilon);
    }

    __aicore__ inline void Compute(
        LocalTensor<float>& rstdLocal, LocalTensor<T_X>& gammaLocal, LocalTensor<T_SCALES>& scales1Local,
        LocalTensor<T_SCALES>& scales2Local, LocalTensor<T_ZEROPOINTS>& zeroPoints1Local,
        LocalTensor<T_ZEROPOINTS>& zeroPoints2Local, uint64_t mIdx, uint64_t nIdx)
    {
        LocalTensor<T_X> x1Local = inQueueX1_.DeQue<T_X>();
        LocalTensor<T_Y> y1Local = outQueueY1_.AllocTensor<T_Y>();
        LocalTensor<T_Y> y2Local;
        if constexpr (HAS_Y2) {
            y2Local = outQueueY2_.AllocTensor<T_Y>();
        }

        ComputeY<T_X, T_Y, T_SCALES, T_ZEROPOINTS, true, HAS_ZEROPOINTS1, DIV_MODE>(
            y1Local, x1Local, rstdLocal, gammaLocal, scales1Local, zeroPoints1Local, mIdx, baseNDtypeAlign_);
        if constexpr (HAS_Y2) {
            ComputeY<T_X, T_Y, T_SCALES, T_ZEROPOINTS, HAS_SCALE2, HAS_ZEROPOINTS2, DIV_MODE>(
                y2Local, x1Local, rstdLocal, gammaLocal, scales2Local, zeroPoints2Local, mIdx, baseNDtypeAlign_);
        }

        inQueueX1_.FreeTensor(x1Local);
        outQueueY1_.EnQue<T_Y>(y1Local);
        if constexpr (HAS_Y2) {
            outQueueY2_.EnQue<T_Y>(y2Local);
        }
    }

private:
    TPipe* pipe_ = nullptr;
    // GM Buffer
    GlobalTensor<T_X> x1Gm_;
    GlobalTensor<T_X> x2Gm_;
    GlobalTensor<T_X> gammaGm_;
    GlobalTensor<T_X> xGm_;
    GlobalTensor<T_SCALES> scales1Gm_, scales2Gm_;
    GlobalTensor<T_ZEROPOINTS> zeroPoints1Gm_, zeroPoints2Gm_;
    GlobalTensor<T_Y> y1Gm_;
    GlobalTensor<T_Y> y2Gm_;
    // UB Buffer
    TQue<QuePosition::VECIN, 1> inQueueX1_, inQueueX2_, inQueueGamma_;
    TQue<QuePosition::VECIN, 1> inQueueScales1_, inQueueScales2_, inQueueZeroPoints1_, inQueueZeroPoints2_;
    TQue<QuePosition::VECOUT, 1> outQueueY1_, outQueueY2_, outQueueX_;
    TBuf<TPosition::VECCALC> rstdBuf_;
    TBuf<TPosition::VECCALC> reduceBuf_;
    TBuf<TPosition::VECCALC> level1Buf_;
    TBuf<TPosition::VECCALC> level2Buf_;
    TBuf<TPosition::VECCALC> level3Buf_;
    TBuf<TPosition::VECCALC> tempBuf_;

    // Tiling data
    uint64_t numN_{0};
    uint64_t numM_{0};
    uint64_t baseM_{0};
    uint64_t baseN_{0};
    uint64_t baseNDtypeAlign_{0};
    uint64_t tailNDtypeAlign_{0};
    uint64_t baseNReduceAlign_{0};
    uint64_t reduceBufAlign_{0};
    uint64_t powerSplit_{0};
    uint64_t powerLoop_{0};
    uint64_t mPerCore_{0};
    uint64_t mLastCore_{0};
    float epsilon_{0};
    float avgFactor_{0};
    // Platform
    int64_t blockIdx_{0};
    int64_t blockNum_{0};
    // Cal params
    uint64_t mCore_;
    uint64_t nCnt_;
    uint64_t tailN_;
    uint64_t baseNB8Align_;
    // Other
    uint32_t vfLength_{0};
};
} // namespace AddRmsNormQuant
#endif // ADD_RMS_NORM_QUANT_REGBASE_SPILT_REDUCE_H_

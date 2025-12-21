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
 * \file add_rms_norm_dynamic_quant_regbase_split_reduce.h
 * \brief
 */
#ifndef ADD_RMS_NORM_DYNAMIC_QUANT_REGBASE_SPILT_REDUCE_H_
#define ADD_RMS_NORM_DYNAMIC_QUANT_REGBASE_SPILT_REDUCE_H_

#include "add_rms_norm_dynamic_quant_regbase_common.h"

namespace AddRmsNormDynamicQuant {

template <typename T_X, typename T_Y, uint64_t TILING_KEY>
class KernelAddRmsNormDynamicQuantRegbaseSpiltReduce {
#define INPUT_KEY ((TILING_KEY % 100) / 10)
#define HAS_SMOOTH_SCALE1 ((INPUT_KEY >> 1) % 2 == 1)
#define HAS_SMOOTH_SCALE2 (INPUT_KEY % 2 == 1)
#define HAS_Y2_SCALE2 HAS_SMOOTH_SCALE2
#define T_SMOOTH_SCALE T_X
public:
    __aicore__ inline KernelAddRmsNormDynamicQuantRegbaseSpiltReduce(TPipe* pipe)
    {
        pipe_ = pipe;
    }

    __aicore__ inline void Init(
        GM_ADDR x1, GM_ADDR x2, GM_ADDR gamma, GM_ADDR smooathScale1, GM_ADDR smooathScale2, GM_ADDR y1, GM_ADDR y2,
        GM_ADDR x, GM_ADDR scale1, GM_ADDR scale2, GM_ADDR workspace,
        const AddRmsNormDynamicQuantRegbaseTilingData* tilingData)
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
        InitBuffer(x1, x2, gamma, smooathScale1, smooathScale2, y1, y2, x, scale1, scale2, workspace);
    }

    __aicore__ inline void CalBlockTail()
    {
        mCore_ = blockIdx_ == (blockNum_ - 1) ? mLastCore_ : mPerCore_;
        nCnt_ = CeilDiv(numN_, baseN_);
        tailN_ = numN_ - (nCnt_ - 1) * baseN_;

        tailNDtypeAlign_ = CeilAlign(tailN_, BLOCK_SIZE / sizeof(T_X));
        baseNB8Align_ = CeilAlign(baseN_, B8_BLOCK_NUM);
        baseNB32Align_ = CeilAlign(baseN_, B32_BLOCK_NUM);
        uint64_t reduceSumBufLen = baseNReduceAlign_ / (2 * V_LENGTH);
        reduceSumBufAlign_ = CeilAlign(reduceSumBufLen, B32_BLOCK_NUM);
    }

    __aicore__ inline void InitBuffer(
        GM_ADDR x1, GM_ADDR x2, GM_ADDR gamma, GM_ADDR smooathScale1, GM_ADDR smooathScale2, GM_ADDR y1, GM_ADDR y2,
        GM_ADDR x, GM_ADDR scale1, GM_ADDR scale2, GM_ADDR workspace)
    {
        uint64_t gmOffset = blockIdx_ * mPerCore_ * numN_;
        uint64_t gmLen = mCore_ * numN_;
        uint64_t scalesGmOffset = blockIdx_ * mPerCore_;
        uint64_t workSpaceOffset = blockIdx_ * numN_;
        uint64_t workSpaceStart = blockNum_ * numN_;
        x1Gm_.SetGlobalBuffer((__gm__ T_X*)x1 + gmOffset, gmLen);
        x2Gm_.SetGlobalBuffer((__gm__ T_X*)x2 + gmOffset, gmLen);
        gammaGm_.SetGlobalBuffer((__gm__ T_X*)gamma, numN_);
        y1Gm_.SetGlobalBuffer((__gm__ T_Y*)y1 + gmOffset, gmLen);
        xGm_.SetGlobalBuffer((__gm__ T_X*)x + gmOffset, gmLen);
        scale1Gm_.SetGlobalBuffer((__gm__ float*)scale1 + scalesGmOffset, mCore_);
        y1TmpGm_.SetGlobalBuffer((__gm__ float*)workspace + workSpaceOffset, numN_);
        xOutTmpGm_.SetGlobalBuffer((__gm__ float*)workspace + workSpaceStart + workSpaceOffset, numN_);
        if constexpr (HAS_SMOOTH_SCALE1) {
            smoothScale1Gm_.SetGlobalBuffer((__gm__ T_SMOOTH_SCALE*)smooathScale1, numN_);
        }
        if constexpr (HAS_SMOOTH_SCALE2) {
            smoothScale2Gm_.SetGlobalBuffer((__gm__ T_SMOOTH_SCALE*)smooathScale2, numN_);
        }
        if constexpr (HAS_Y2_SCALE2) {
            y2Gm_.SetGlobalBuffer((__gm__ T_Y*)y2 + gmOffset, gmLen);
            scale2Gm_.SetGlobalBuffer((__gm__ float*)scale2 + scalesGmOffset, mCore_);
            y2TmpGm_.SetGlobalBuffer((__gm__ float*)workspace + 2 * workSpaceStart + workSpaceOffset, numN_);
        }

        uint64_t ubFactorQuant = CeilAlign(baseN_, BLOCK_SIZE / sizeof(T_SMOOTH_SCALE));
        uint64_t ubFactorRstd = B32_BLOCK_NUM;
        pipe_->InitBuffer(inQueueX1_, 1, baseNReduceAlign_ * sizeof(T_X));
        pipe_->InitBuffer(inQueueX2_, 1, baseNReduceAlign_ * sizeof(T_X));
        pipe_->InitBuffer(outQueueX_, 1, baseNReduceAlign_ * sizeof(T_X));
        pipe_->InitBuffer(inQueueGamma_, 1, baseNDtypeAlign_ * sizeof(T_X));
        pipe_->InitBuffer(outQueueY1_, 1, baseNB8Align_ * sizeof(T_Y));
        pipe_->InitBuffer(outQueueScale1_, 1, ubFactorRstd * sizeof(float));
        if constexpr (HAS_SMOOTH_SCALE1) {
            pipe_->InitBuffer(inQueueSmoothScale1_, 1, ubFactorQuant * sizeof(T_SMOOTH_SCALE));
        }
        if constexpr (HAS_SMOOTH_SCALE2) {
            pipe_->InitBuffer(inQueueSmoothScale2_, 1, ubFactorQuant * sizeof(T_SMOOTH_SCALE));
        }
        if constexpr (HAS_Y2_SCALE2) {
            pipe_->InitBuffer(outQueueY2_, 1, baseNB8Align_ * sizeof(T_Y));
            pipe_->InitBuffer(outQueueScale2_, 1, ubFactorRstd * sizeof(float));
        }

        pipe_->InitBuffer(xOutTmpBuf_, baseNReduceAlign_ * sizeof(float));
        pipe_->InitBuffer(y1TmpBuf_, baseNB32Align_ * sizeof(float));
        if constexpr (HAS_Y2_SCALE2) {
            pipe_->InitBuffer(y2TmpBuf_, baseNB32Align_ * sizeof(float));
        }
        pipe_->InitBuffer(rstdBuf_, ubFactorRstd * sizeof(float));
        pipe_->InitBuffer(reduceSumBuf_, reduceSumBufAlign_ * sizeof(float));
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

            LocalTensor<float> scale1Local = outQueueScale1_.AllocTensor<float>();
            LocalTensor<float> scale2Local;
            if constexpr (HAS_Y2_SCALE2) {
                scale2Local = outQueueScale2_.AllocTensor<float>();
            }
            Duplicate(scale1Local, 0.0f, B32_BLOCK_NUM);
            if constexpr (HAS_Y2_SCALE2) {
                Duplicate(scale2Local, 0.0f, B32_BLOCK_NUM);
            }
            // Process rstd & Copy out xOut(x1+x2)
            SubProcessRstd<true>(
                rstdLocal, 0, xGmOffset, masterLoop, powerTailLoop, powerTailTail, avgFactor_, epsilon_);
            // Process reducemax & Store (rmsout*smooth) to workSpace
            SubProcessReduceMax(scale1Local, scale2Local, rstdLocal, mIdx);
            // Process scale
            PipeBarrier<PIPE_V>();
            SubProcessScale(scale1Local, scale2Local);
            // Copy out y
            PipeBarrier<PIPE_V>();
            SubProcessY(scale1Local, scale2Local, mIdx);
            // Copy out scale
            outQueueScale1_.EnQue<float>(scale1Local);
            if constexpr (HAS_Y2_SCALE2) {
                outQueueScale2_.EnQue<float>(scale2Local);
            }
            CopyOutScale(scale1Gm_, outQueueScale1_, mIdx, 1);
            if constexpr (HAS_Y2_SCALE2) {
                CopyOutScale(scale2Gm_, outQueueScale2_, mIdx, 1);
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
        if constexpr (HAS_SMOOTH_SCALE1) {
            LocalTensor<T_SMOOTH_SCALE> smoothScale1Local = inQueueSmoothScale1_.AllocTensor<T_SMOOTH_SCALE>();
            RmsNorm::DataCopyImpl<T_SMOOTH_SCALE>(smoothScale1Local, smoothScale1Gm_[gmOffset], 1, blockLen);
            inQueueSmoothScale1_.EnQue(smoothScale1Local);
        }
        if constexpr (HAS_SMOOTH_SCALE2) {
            LocalTensor<T_SMOOTH_SCALE> smoothScale2Local = inQueueSmoothScale2_.AllocTensor<T_SMOOTH_SCALE>();
            RmsNorm::DataCopyImpl<T_SMOOTH_SCALE>(smoothScale2Local, smoothScale2Gm_[gmOffset], 1, blockLen);
            inQueueSmoothScale2_.EnQue(smoothScale2Local);
        }
    }

    /**
     * @brief Copy in xGm_[srcGmOffset:srcGmOffset+count] and reduce to dst[dstOffset].
     */
    __aicore__ inline void ComputeFormerHandle(
        LocalTensor<float>& dstLocal, uint64_t srcGmOffset, uint64_t dstOffset, uint64_t workSpaceOffset,
        uint32_t count, uint32_t powerSplit)
    {
        LocalTensor<float> xOutTmpLocal = xOutTmpBuf_.Get<float>();
        uint32_t calCount = Aligned((uint64_t)(count * sizeof(T_X)), ALIGN_32_FACTOR) / sizeof(T_X);
        CopyInX(inQueueX1_, x1Gm_, srcGmOffset, count, 0, calCount - count);
        CopyInX(inQueueX2_, x2Gm_, srcGmOffset, count, 0, calCount - count);
        LocalTensor<T_X> x1Local = inQueueX1_.DeQue<T_X>();
        LocalTensor<T_X> x2Local = inQueueX2_.DeQue<T_X>();

        LocalTensor<float> workLocal = reduceSumBuf_.Get<float>();
        uint32_t calNum = Aligned((uint64_t)(count * sizeof(T_X)), ALIGN_512_FACTOR) / sizeof(T_X);
        uint64_t dupLen = calNum - calCount;
        if (dupLen > 0) {
            Duplicate(x1Local[calCount], (T_X)0.0, dupLen);
            Duplicate(x2Local[calCount], (T_X)0.0, dupLen);
        }

        LocalTensor<T_X> xOutLocal = outQueueX_.AllocTensor<T_X>();
        NormCommon::ReduceSumRstd<T_X, true, true, false>(
            dstLocal, xOutLocal, xOutTmpLocal, x1Local, x2Local, workLocal, dstOffset, calNum, powerSplit);
        outQueueX_.EnQue<T_X>(xOutLocal);
        inQueueX1_.FreeTensor(x1Local);
        inQueueX2_.FreeTensor(x2Local);

        CopyOutX(xGm_, outQueueX_, srcGmOffset, count);
        // Copy xOut to workspace
        event_t eventVMTE3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
        event_t eventMTE3V = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));
        SetFlag<HardEvent::V_MTE3>(eventVMTE3);
        WaitFlag<HardEvent::V_MTE3>(eventVMTE3);
        RmsNorm::DataCopyImpl<float>(xOutTmpGm_[workSpaceOffset], xOutTmpLocal, 1, count);
        SetFlag<HardEvent::MTE3_V>(eventMTE3V);
        WaitFlag<HardEvent::MTE3_V>(eventMTE3V);
    }

    template <bool IS_RSTD>
    __aicore__ inline void SubProcessRstd(
        LocalTensor<float> dstLocal, uint64_t position, uint64_t curRow, uint64_t masterLoop, uint64_t tailLoop,
        uint64_t tail, float avgFactor = 1.0f, float epsilon = 0.0f)
    {
        uint64_t workSpaceOffset{0};
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
            ComputeFormerHandle(tempLocal, offset, 0, workSpaceOffset, powerSplit_, powerSplit_);
            offset += powerSplit_;
            workSpaceOffset += powerSplit_;
            // Main
            ComputeFormerHandle(tempLocal, offset, 1, workSpaceOffset, powerSplit_, powerSplit_);
            offset += powerSplit_;
            workSpaceOffset += powerSplit_;

            RmsNorm::ComputeSum(level1Local, tempLocal, level1, SUM_COUNT);
            level1 += 1;
            RmsNorm::ComputeMultiLevelReduce(level1Local, level2Local, level3Local, level1, level2, level3);
        }
        // Stage2: Cal TailTail
        // Use TailBlock as MainBlock and reduce 2 nearby Mainblock twice.
        uint64_t powerSplitHalf = powerSplit_ / CONST_FACTOR_2;
        if (tail > 0 && tail <= powerSplitHalf) {
            // Tail
            ComputeFormerHandle(tempLocal, offset, 0, workSpaceOffset, powerSplitHalf + tail, powerSplitHalf);
            offset += powerSplitHalf + tail;
            workSpaceOffset += powerSplitHalf + tail;
            // Main
            ComputeFormerHandle(tempLocal, offset, 1, workSpaceOffset, powerSplitHalf, powerSplitHalf);
            offset += powerSplitHalf;
            workSpaceOffset += powerSplitHalf;
        } else if (tail > powerSplitHalf) {
            // Half Main & Half Tail
            ComputeFormerHandle(tempLocal, offset, 0, workSpaceOffset, powerSplit_, powerSplit_);
            offset += powerSplit_;
            workSpaceOffset += powerSplit_;
            // Other Tail
            ComputeFormerHandle(tempLocal, offset, 1, workSpaceOffset, tail, powerSplitHalf);
            offset += tail;
            workSpaceOffset += tail;
        }
        RmsNorm::ComputeSum(level1Local, tempLocal, level1, SUM_COUNT);
        level1 += 1;
        RmsNorm::ComputeMultiLevelReduce(level1Local, level2Local, level3Local, level1, level2, level3);
        // Stage3: Cal MasterLoop
        for (uint32_t repeat = 0; repeat < masterLoop; repeat++) {
            ComputeFormerHandle(level1Local, offset, level1, workSpaceOffset, powerSplit_, powerSplit_);
            offset += powerSplit_;
            workSpaceOffset += powerSplit_;
            level1 += 1;
            RmsNorm::ComputeMultiLevelReduce(level1Local, level2Local, level3Local, level1, level2, level3);
        }
        ComputeMultiLevelRstd<IS_RSTD>(
            dstLocal, position, level1Local, level2Local, level3Local, level1, level2, avgFactor, epsilon);
    }

    __aicore__ inline void SubProcessReduceMax(
        LocalTensor<float>& scale1Local, LocalTensor<float>& scale2Local, LocalTensor<float>& rstdLocal, uint64_t mIdx)
    {
        LocalTensor<float> xOutTmpLocal = xOutTmpBuf_.Get<float>();
        LocalTensor<float> y1TmpLocal = y1TmpBuf_.Get<float>();
        LocalTensor<float> y2TmpLocal;
        if constexpr (HAS_Y2_SCALE2) {
            y2TmpLocal = y2TmpBuf_.Get<float>();
        }
        for (uint32_t nIdx = 0; nIdx < nCnt_; nIdx++) {
            uint64_t realN = (nIdx == nCnt_ - 1) ? tailN_ : baseN_;
            uint64_t realNDtypeAlign = (nIdx == nCnt_ - 1) ? tailNDtypeAlign_ : baseNDtypeAlign_;
            CopyInGamma(nIdx * baseN_, realN);
            CopyInQuant(nIdx * baseN_, realN);
            LocalTensor<T_X> gammaLocal = inQueueGamma_.DeQue<T_X>();
            LocalTensor<T_SMOOTH_SCALE> smoothScale1Local;
            LocalTensor<T_SMOOTH_SCALE> smoothScale2Local;
            if constexpr (HAS_SMOOTH_SCALE1) {
                smoothScale1Local = inQueueSmoothScale1_.DeQue<T_SMOOTH_SCALE>();
            }
            if constexpr (HAS_SMOOTH_SCALE2) {
                smoothScale2Local = inQueueSmoothScale2_.DeQue<T_SMOOTH_SCALE>();
            }

            uint64_t gmOffsetX = mIdx * numN_ + nIdx * baseN_;
            uint64_t gmOffsetYTmp = nIdx * baseN_;
            uint64_t gmOffsetScale = 0;
            DataCopyPadExtParams<float> padParams{
                true,                    // isPad
                static_cast<uint8_t>(0), // leftPadding
                static_cast<uint8_t>(0), // rightPadding
                static_cast<float>(0.0)  // paddingValue
            };
            event_t eventVMTE2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE2));
            event_t eventMTE2V = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
            SetFlag<HardEvent::V_MTE2>(eventVMTE2);
            WaitFlag<HardEvent::V_MTE2>(eventVMTE2);
            RmsNorm::DataCopyImpl<float>(xOutTmpLocal, xOutTmpGm_[gmOffsetYTmp], 1, realN, 0, 0, padParams);
            SetFlag<HardEvent::MTE2_V>(eventMTE2V);
            WaitFlag<HardEvent::MTE2_V>(eventMTE2V);

            ComputeReduceMax<float, T_X, T_SMOOTH_SCALE, HAS_SMOOTH_SCALE1, T_Y>(
                scale1Local, y1TmpLocal, xOutTmpLocal, rstdLocal, gammaLocal, smoothScale1Local, 0, baseNDtypeAlign_);
            if constexpr (HAS_Y2_SCALE2) {
                ComputeReduceMax<float, T_X, T_SMOOTH_SCALE, HAS_SMOOTH_SCALE2, T_Y>(
                    scale2Local, y2TmpLocal, xOutTmpLocal, rstdLocal, gammaLocal, smoothScale2Local, 0,
                    baseNDtypeAlign_);
            }

            inQueueGamma_.FreeTensor(gammaLocal);
            if constexpr (HAS_SMOOTH_SCALE1) {
                inQueueSmoothScale1_.FreeTensor(smoothScale1Local);
            }
            if constexpr (HAS_SMOOTH_SCALE2) {
                inQueueSmoothScale2_.FreeTensor(smoothScale2Local);
            }

            // Copy (rmsOut * smooth) to workspace
            event_t eventVMTE3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
            event_t eventMTE3V = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));
            SetFlag<HardEvent::V_MTE3>(eventVMTE3);
            WaitFlag<HardEvent::V_MTE3>(eventVMTE3);
            RmsNorm::DataCopyImpl<float>(y1TmpGm_[gmOffsetYTmp], y1TmpLocal, 1, realN);
            if constexpr (HAS_Y2_SCALE2) {
                RmsNorm::DataCopyImpl<float>(y2TmpGm_[gmOffsetYTmp], y2TmpLocal, 1, realN);
            }
            SetFlag<HardEvent::MTE3_V>(eventMTE3V);
            WaitFlag<HardEvent::MTE3_V>(eventMTE3V);
        }
    }

    __aicore__ inline void SubProcessScale(LocalTensor<float>& scale1Local, LocalTensor<float>& scale2Local)
    {
        ComputeScale<T_Y>(scale1Local, 0);
        if constexpr (HAS_Y2_SCALE2) {
            ComputeScale<T_Y>(scale2Local, 0);
        }
    }

    __aicore__ inline void SubProcessY(LocalTensor<float>& scale1Local, LocalTensor<float>& scale2Local, uint64_t mIdx)
    {
        LocalTensor<float> y1TmpLocal = y1TmpBuf_.Get<float>();
        LocalTensor<float> y2TmpLocal;
        if constexpr (HAS_Y2_SCALE2) {
            y2TmpLocal = y2TmpBuf_.Get<float>();
        }

        for (uint32_t nIdx = 0; nIdx < nCnt_; nIdx++) {
            uint64_t realN = (nIdx == nCnt_ - 1) ? tailN_ : baseN_;
            uint64_t realNDtypeAlign = (nIdx == nCnt_ - 1) ? tailNDtypeAlign_ : baseNDtypeAlign_;
            uint64_t gmOffsetXY = mIdx * numN_ + nIdx * baseN_;
            uint64_t gmOffsetYTmp = nIdx * baseN_;

            LocalTensor<T_Y> y1Local = outQueueY1_.AllocTensor<T_Y>();
            LocalTensor<T_Y> y2Local;
            if constexpr (HAS_Y2_SCALE2) {
                y2Local = outQueueY2_.AllocTensor<T_Y>();
            }
            // Copy (rmsOut * smooth) from workspace
            DataCopyPadExtParams<float> padParams{
                true,                    // isPad
                static_cast<uint8_t>(0), // leftPadding
                static_cast<uint8_t>(0), // rightPadding
                static_cast<float>(0.0)  // paddingValue
            };
            event_t eventVMTE2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE2));
            event_t eventMTE2V = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
            SetFlag<HardEvent::V_MTE2>(eventVMTE2);
            WaitFlag<HardEvent::V_MTE2>(eventVMTE2);
            RmsNorm::DataCopyImpl<float>(y1TmpLocal, y1TmpGm_[gmOffsetYTmp], 1, realN, 0, 0, padParams);
            if constexpr (HAS_Y2_SCALE2) {
                RmsNorm::DataCopyImpl<float>(y2TmpLocal, y2TmpGm_[gmOffsetYTmp], 1, realN, 0, 0, padParams);
            }
            SetFlag<HardEvent::MTE2_V>(eventMTE2V);
            WaitFlag<HardEvent::MTE2_V>(eventMTE2V);

            ComputeY(y1Local, scale1Local, y1TmpLocal, 0, baseNDtypeAlign_);
            if constexpr (HAS_Y2_SCALE2) {
                ComputeY(y2Local, scale2Local, y2TmpLocal, 0, baseNDtypeAlign_);
            }

            outQueueY1_.EnQue<T_Y>(y1Local);
            if constexpr (HAS_Y2_SCALE2) {
                outQueueY2_.EnQue<T_Y>(y2Local);
            }
            CopyOutY(y1Gm_, outQueueY1_, gmOffsetXY, realN);
            if constexpr (HAS_Y2_SCALE2) {
                CopyOutY(y2Gm_, outQueueY2_, gmOffsetXY, realN);
            }
        }
    }

private:
    TPipe* pipe_ = nullptr;
    // GM Buffer
    GlobalTensor<T_X> x1Gm_;
    GlobalTensor<T_X> x2Gm_;
    GlobalTensor<T_X> gammaGm_;
    GlobalTensor<T_X> xGm_;
    GlobalTensor<T_SMOOTH_SCALE> smoothScale1Gm_;
    GlobalTensor<T_SMOOTH_SCALE> smoothScale2Gm_;
    GlobalTensor<T_Y> y1Gm_;
    GlobalTensor<T_Y> y2Gm_;
    GlobalTensor<float> scale1Gm_;
    GlobalTensor<float> scale2Gm_;
    GlobalTensor<float> y1TmpGm_;
    GlobalTensor<float> y2TmpGm_;
    GlobalTensor<float> xOutTmpGm_;
    // UB Buffer
    TQue<QuePosition::VECIN, 1> inQueueX1_;
    TQue<QuePosition::VECIN, 1> inQueueX2_;
    TQue<QuePosition::VECIN, 1> inQueueGamma_;
    TQue<QuePosition::VECIN, 1> inQueueSmoothScale1_;
    TQue<QuePosition::VECIN, 1> inQueueSmoothScale2_;
    TQue<QuePosition::VECOUT, 1> outQueueY1_;
    TQue<QuePosition::VECOUT, 1> outQueueY2_;
    TQue<QuePosition::VECOUT, 1> outQueueX_;
    TQue<QuePosition::VECOUT, 1> outQueueScale1_;
    TQue<QuePosition::VECOUT, 1> outQueueScale2_;
    TBuf<TPosition::VECCALC> rstdBuf_;
    TBuf<TPosition::VECCALC> reduceSumBuf_;
    TBuf<TPosition::VECCALC> level1Buf_;
    TBuf<TPosition::VECCALC> level2Buf_;
    TBuf<TPosition::VECCALC> level3Buf_;
    TBuf<TPosition::VECCALC> tempBuf_;
    TBuf<TPosition::VECCALC> y1TmpBuf_;
    TBuf<TPosition::VECCALC> y2TmpBuf_;
    TBuf<TPosition::VECCALC> xOutTmpBuf_;

    // Tiling data
    uint64_t numN_{0};
    uint64_t numM_{0};
    uint64_t baseM_{0};
    uint64_t baseN_{0};
    uint64_t baseNDtypeAlign_{0};
    uint64_t tailNDtypeAlign_{0};
    uint64_t baseNReduceAlign_{0};
    uint64_t reduceSumBufAlign_{0};
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
    uint64_t baseNB32Align_;
    // Other
    uint32_t vfLength_{0};
};
} // namespace AddRmsNormDynamicQuant
#endif // _ADD_RMS_NORM_DYNAMIC_QUANT_REGBASE_SPILT_REDUCE_H_

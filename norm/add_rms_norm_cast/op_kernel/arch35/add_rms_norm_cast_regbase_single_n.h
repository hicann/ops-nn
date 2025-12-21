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
 * \file add_rms_norm_cast_regbase_single_n.h
 * \brief AddRmsNormCast regbase normal template.
 */
#ifndef OPS_BUILT_IN_TBE_IMPL_ASCENDC_ADD_RMS_NORM_CAST_SINGLE_N_REGBASE_H
#define OPS_BUILT_IN_TBE_IMPL_ASCENDC_ADD_RMS_NORM_CAST_SINGLE_N_REGBASE_H
#include "add_rms_norm_cast_regbase_common.h"

namespace AddRmsNormCast {
template <typename T_X>
class KernelAddRmsNormCastRegBaseSingleN {
#define NUM_M_ (tilingData_->numM)
#define NUM_N_ (tilingData_->numN)
#define BASE_M_ (tilingData_->baseM)
#define BASE_N_ (tilingData_->baseN)
#define BASE_N_DTYPE_ALIGN_ (tilingData_->baseNDtypeAlign)
#define BASE_N_REDUCE_ALIGN_ (tilingData_->baseNReduceAlign)
#define POWER_SPLIT_ (tilingData_->powerSplit)
#define M_PER_CORE_ (tilingData_->mPerCore)
#define M_LAST_CORE_ (tilingData_->mLastCore)
#define EPSILON_ (tilingData_->epsilon)
#define AVG_FACTOR_ (tilingData_->avgFactor)

#define MAGIC_CONDITION (GetBlockNum() > 0x7)

public:
    __aicore__ inline KernelAddRmsNormCastRegBaseSingleN(TPipe* pipe)
    {
        pipe_ = pipe;
    }

    __aicore__ inline void Init(
        GM_ADDR x1, GM_ADDR x2, GM_ADDR gamma, GM_ADDR y1, GM_ADDR y2, GM_ADDR rstd, GM_ADDR x, GM_ADDR workspace,
        const AddRmsNormCastRegbaseTilingData* tilingData)
    {
        tilingData_ = tilingData;
        InitBuffer(x1, x2, gamma, y1, y2, rstd, x, workspace);
    }

    __aicore__ inline void InitBuffer(
        GM_ADDR x1, GM_ADDR x2, GM_ADDR gamma, GM_ADDR y1, GM_ADDR y2, GM_ADDR rstd, GM_ADDR x, GM_ADDR workspace)
    {
        // Init GM
        uint64_t gmOffset = GetBlockIdx() * NUM_N_;
        x1Gm_.SetGlobalBuffer((__gm__ T_X*)x1 + gmOffset, NUM_N_);
        x2Gm_.SetGlobalBuffer((__gm__ T_X*)x2 + gmOffset, NUM_N_);
        y1Gm_.SetGlobalBuffer((__gm__ float*)y1 + gmOffset, NUM_N_);
        y2Gm_.SetGlobalBuffer((__gm__ T_X*)y2 + gmOffset, NUM_N_);
        xGm_.SetGlobalBuffer((__gm__ T_X*)x + gmOffset, NUM_N_);
        gammaGm_.SetGlobalBuffer((__gm__ T_X*)gamma, NUM_N_);
        rstdGm_.SetGlobalBuffer((__gm__ float*)rstd + GetBlockIdx());
        wsGm_.SetGlobalBuffer((__gm__ float*)(workspace + MIGIC_AND_WONDERFUL_BYTES * GetBlockIdx()));
        InitUBBuffer();
    }

    __aicore__ inline void InitUBBuffer()
    {
        // Init Buffer
        uint64_t reduceSumBufLen = BASE_N_REDUCE_ALIGN_ / (CONST_FACTOR_2 * V_LENGTH);
        uint64_t reduceSumBufAlign_ = CeilAlign(reduceSumBufLen, AddRmsNormCast::B32_BLOCK_NUM);

        pipe_->InitBuffer(inQueueX1X2_, 1, CONST_FACTOR_2 * BASE_N_REDUCE_ALIGN_ * sizeof(T_X));
        pipe_->InitBuffer(outQueueX_, 1, BASE_N_REDUCE_ALIGN_ * sizeof(T_X));
        pipe_->InitBuffer(inQueueGamma_, BASE_N_REDUCE_ALIGN_ * sizeof(T_X));
        pipe_->InitBuffer(outQueueY1_, 1, BASE_N_REDUCE_ALIGN_ * sizeof(float));
        pipe_->InitBuffer(outQueueY2_, 1, BASE_N_REDUCE_ALIGN_ * sizeof(T_X));
        pipe_->InitBuffer(outQueueRstd_, 1, BLOCK_SIZE);
        pipe_->InitBuffer(xOutTmpBuf_, BASE_N_REDUCE_ALIGN_ * sizeof(float));
        pipe_->InitBuffer(reduceSumBuf_, reduceSumBufAlign_ * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        uint64_t baseNB32Align_ = CeilAlign(BASE_N_, AddRmsNormCast::B32_BLOCK_NUM);
        uint32_t dtypeBlockStride_ = (BASE_N_REDUCE_ALIGN_ - BASE_N_DTYPE_ALIGN_) * sizeof(T_X) / ALIGN_32_FACTOR;
        uint32_t b32BlockStride_ = (BASE_N_REDUCE_ALIGN_ - baseNB32Align_) * sizeof(float) / ALIGN_32_FACTOR;

        LocalTensor<float> reduceLocal = reduceSumBuf_.Get<float>();
        LocalTensor<float> xOutTmpLocal = xOutTmpBuf_.Get<float>();
        LocalTensor<T_X> gammaLocal = inQueueGamma_.Get<T_X>();

        CopyInXMultiMoveAlign<T_X>(NUM_N_, BASE_N_DTYPE_ALIGN_, dtypeBlockStride_);
        LocalTensor<T_X> xLocal = inQueueX1X2_.DeQue<T_X>();
        LocalTensor<T_X> x1Local = xLocal[0];
        LocalTensor<T_X> x2Local = xLocal[BASE_N_REDUCE_ALIGN_];

        uint64_t dupLen = BASE_N_REDUCE_ALIGN_ - BASE_N_DTYPE_ALIGN_;
        if (dupLen > 0) {
            Duplicate(x1Local[BASE_N_DTYPE_ALIGN_], (T_X)0.0, dupLen);
            Duplicate(x2Local[BASE_N_DTYPE_ALIGN_], (T_X)0.0, dupLen);
        }

        // 1. Calc xOut
        LocalTensor<float> rstdLocal = outQueueRstd_.AllocTensor<float>();
        LocalTensor<T_X> xOutLocal = outQueueX_.AllocTensor<T_X>();
        ReduceSumRstdMulti<T_X, true, true, true>(
            rstdLocal, xOutLocal, xOutTmpLocal, x1Local, x2Local, reduceLocal, 0, BASE_N_REDUCE_ALIGN_, POWER_SPLIT_, 1,
            AVG_FACTOR_, EPSILON_);
        inQueueX1X2_.FreeTensor(xLocal);
        outQueueX_.EnQue<T_X>(xOutLocal);
        outQueueRstd_.EnQue<float>(rstdLocal);
        CopyOutXYMulti(xGm_, outQueueX_, dtypeBlockStride_);
        CopyOutRstd(0, 1);

        RmsNorm::DataCopyImpl<T_X>(gammaLocal, gammaGm_, 1, NUM_N_);
        SimpleNativePipeSync<HardEvent::MTE2_V>();

        // 2. Calc y1/y2
        LocalTensor<float> y1Local = outQueueY1_.AllocTensor<float>();
        LocalTensor<T_X> y2Local = outQueueY2_.AllocTensor<T_X>();
        AddRmsNormCast::ComputeYMulti<T_X, T_X>(
            y1Local, y2Local, xOutTmpLocal, gammaLocal, rstdLocal, 0, BASE_N_REDUCE_ALIGN_, 1);
        outQueueY1_.EnQue<float>(y1Local);
        outQueueY2_.EnQue<T_X>(y2Local);
        CopyOutXYMulti(y1Gm_, outQueueY1_, b32BlockStride_);
        CopyOutXYMulti(y2Gm_, outQueueY2_, dtypeBlockStride_);

        if (MAGIC_CONDITION) {
            SyncAll();
            LocalTensor<float> tmpLocal = xOutTmpBuf_.Get<float>();
            MergeOutScales(tmpLocal, wsGm_, rstdGm_);
        }
    }

    __aicore__ inline void MergeOutScales(
        LocalTensor<float>& tmpLocal, GlobalTensor<float>& wsGm, GlobalTensor<float>& outRstdGm)
    {
        if (GetBlockIdx() == 0) {
            constexpr int64_t MAGIC_PAGE_BYTES = 128;

            DataCopyPadExtParams<float> padParams{false, 0, 0, 0};

            DataCopyExtParams cpInParam;
            cpInParam.blockCount = GetBlockNum();
            cpInParam.blockLen = sizeof(float) * 1;
            cpInParam.srcStride = MAGIC_PAGE_BYTES - sizeof(float);
            cpInParam.dstStride = 0;

            DataCopyPad<float, PaddingMode::Compact>(tmpLocal, wsGm, cpInParam, padParams);

            SimpleNativePipeSync<HardEvent::MTE2_MTE3>();

            DataCopyExtParams cpOutParam;
            cpOutParam.blockCount = 1;
            cpOutParam.blockLen = sizeof(float) * GetBlockNum();
            cpOutParam.srcStride = 0;
            cpOutParam.dstStride = 0;

            DataCopyPad(outRstdGm, tmpLocal, cpOutParam);
        }
    }

private:
    template <typename T>
    __aicore__ inline void CopyInXMultiMoveAlign(
        uint64_t blockLen, uint64_t blockLenDtypeAlign, uint64_t dtypeBlockStride)
    {
        LocalTensor<T> xLocal = inQueueX1X2_.AllocTensor<T>();

        DataCopyExtParams extParams{
            static_cast<uint16_t>(1),                    // blockCount
            static_cast<uint32_t>(blockLen * sizeof(T)), // blockLen
            static_cast<uint32_t>(0),                    // srcStride
            static_cast<uint32_t>(dtypeBlockStride),     // dstStride
            0                                            // rsv
        };
        DataCopyPadExtParams<T> padParams{
            true,                                                // isPad
            static_cast<uint8_t>(0),                             // leftPadding
            static_cast<uint8_t>(blockLenDtypeAlign - blockLen), // rightPadding
            static_cast<T>(0.0)                                  // paddingValue
        };
        DataCopyPad(xLocal[0], x1Gm_[0], extParams, padParams);
        DataCopyPad(xLocal[BASE_N_REDUCE_ALIGN_], x2Gm_[0], extParams, padParams);
        inQueueX1X2_.EnQue(xLocal);
    }

    template <typename T>
    __aicore__ inline void CopyOutXYMulti(
        GlobalTensor<T>& yGm, TQue<QuePosition::VECOUT, 1>& outQueueY, uint32_t srcStride)
    {
        LocalTensor<T> yLocal = outQueueY.DeQue<T>();
        RmsNorm::DataCopyImpl<T>(yGm[0], yLocal, 1, NUM_N_, srcStride, 0);
        outQueueY.FreeTensor(yLocal);
    }

    __aicore__ inline void CopyOutRstd(uint64_t gmOffset, uint32_t num)
    {
        LocalTensor<float> rstdLocal = outQueueRstd_.DeQue<float>();
        GlobalTensor<float> outGm = (MAGIC_CONDITION) ? wsGm_ : rstdGm_;
        RmsNorm::DataCopyImpl<float>(outGm[gmOffset], rstdLocal, 1, num, 0, 0);
        outQueueRstd_.FreeTensor(rstdLocal);
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
    GlobalTensor<float> wsGm_;

    // UB Buffer
    TQue<QuePosition::VECIN, 1> inQueueX1X2_;

    TQue<QuePosition::VECOUT, 1> outQueueY1_;
    TQue<QuePosition::VECOUT, 1> outQueueY2_;
    TQue<QuePosition::VECOUT, 1> outQueueRstd_;
    TQue<QuePosition::VECOUT, 1> outQueueX_;
    TBuf<TPosition::VECCALC> y1TmpBuf_;
    TBuf<TPosition::VECCALC> y2TmpBuf_;
    TBuf<TPosition::VECCALC> xOutTmpBuf_;
    TBuf<TPosition::VECCALC> reduceSumBuf_;
    TBuf<TPosition::VECCALC> inQueueGamma_;

    const AddRmsNormCastRegbaseTilingData* tilingData_;
};

#undef NUM_M_
#undef NUM_N_
#undef BASE_M_
#undef BASE_N_
#undef BASE_N_DTYPE_ALIGN_
#undef BASE_N_REDUCE_ALIGN_
#undef POWER_SPLIT_
#undef M_PER_CORE_
#undef M_LAST_CORE_
#undef EPSILON_
#undef AVG_FACTOR_

} // namespace AddRmsNormCast
#endif // OPS_BUILT_IN_TBE_IMPL_ASCENDC_ADD_RMS_NORM_CAST_SINGLE_N_REGBASE_H

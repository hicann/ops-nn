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
 * \file apply_came_part1_fp16.h
 * \brief
 */
#ifndef APPLY_CAME_PART1_FP16
#define APPLY_CAME_PART1_FP16

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "apply_came_part1_common.h"

namespace ApplyCamePart1 {

using namespace AscendC;

template <typename T>
class ApplyCamePart1FP16 {
public:
    __aicore__ inline ApplyCamePart1FP16(){};
    __aicore__ inline void Init(GM_ADDR grad, GM_ADDR eps, GM_ADDR sum_grad_r, GM_ADDR sum_grad_c, GM_ADDR sum_grad_rc,
                                GM_ADDR workspace, const ApplyCamePart1TilingData* tilingData, int64_t batchIdx);
    __aicore__ inline void Process();
    __aicore__ inline void ProcessTile(int64_t nLoopIdx, int64_t mLoopIdx, int64_t curRepeatTimes);

private:
    __aicore__ inline void SyncAllCore(GM_ADDR sum_grad_r, GM_ADDR sum_grad_c, GM_ADDR sum_grad_rc);
    __aicore__ inline void ParseTilingData(const ApplyCamePart1TilingData* tilingData);
    __aicore__ inline void ClearAcculateMatrix();
    __aicore__ inline void CopyIn(int64_t nLoopIdx, int64_t mLoopIdx, int64_t curRepeatTimes);
    __aicore__ inline void Compute(int64_t nLoopIdx, int64_t mLoopIdx, int64_t curRepeatTimes);
    __aicore__ inline void CopyOut(int64_t nLoopIdx, int64_t mLoopIdx, int64_t curRepeatTimes);
    __aicore__ inline void CopyInLast(int64_t nLoopIdx, int64_t mLoopIdx, int64_t curRepeatTimes,
                                      LocalTensor<T> gradLocal);
    __aicore__ inline void CopyInNormal(int64_t nLoopIdx, int64_t mLoopIdx, int64_t curRepeatTimes,
                                        LocalTensor<T> gradLocal);
    __aicore__ inline void ComputeR(int64_t mLoopIdx, int64_t curRepeatTimes, LocalTensor<float> gradSqrtTmpUb,
                                    LocalTensor<float> rowTree, LocalTensor<float> workLocal);
    __aicore__ inline void ComputeC(int64_t curRepeatTimes, LocalTensor<float> gradSqrtTmpUb);

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, 1> gradQueue_;
    TQue<QuePosition::VECIN, 1> epsQueue_;

    TQue<QuePosition::VECOUT, 1> sumGradRQueue_;
    TQue<QuePosition::VECOUT, 1> sumGradCQueue_;
    TQue<QuePosition::VECOUT, 1> sumGradRCQueue_;

    TBuf<QuePosition::VECCALC> accuComTmpBuf_;
    TBuf<QuePosition::VECCALC> gradCastTmpBuf_;
    TBuf<QuePosition::VECCALC> gradSqrtTmpBuf_;
    TBuf<QuePosition::VECCALC> mComTmpBuf_;

    GlobalTensor<T> gmGrad_;
    GlobalTensor<float> gmEps_;
    GlobalTensor<float> gmSumGradR_;
    GlobalTensor<float> gmSumGradC_;
    GlobalTensor<float> gmSumGradRC_;

    GlobalTensor<float> workspaceSumGradR_;
    GlobalTensor<float> workspaceSumGradC_;
    GlobalTensor<float> workspaceSumGradRC_;
    GlobalTensor<float> workspaceSumGradRCLow_;
    // multi-core sync
    TQue<QuePosition::VECIN, 1> syncWorkQueue_;
    GM_ADDR workspaceAddr_;

    // tiling params
    int64_t N{0};
    int64_t M{0};

    int64_t nLoopNormCore_{0};
    int64_t nLoopTailCore_{0};

    int64_t nNormalCoreNum_{0};
    int64_t nTailCoreNum_{0};

    int64_t mNormalCoreNum_{0};
    int64_t mTailCoreNum_{0};

    int64_t totalCoreNum_{0};
    int64_t usedCoreNum_{0};

    int64_t nCoreNum_{0};
    int64_t mCoreNum_{0};
    int64_t mLoopNumCore_{0};

    bool hasColumnTail_{false};
    int64_t inputBase_{0};

    const int64_t ONCE_HANDLE_NUM64{64};
    const int64_t ONCE_ONE_SIZE8{8};
};

template <typename T>
__aicore__ inline void ApplyCamePart1FP16<T>::ParseTilingData(const ApplyCamePart1TilingData* tilingData)
{
    // 总维度[N, M]
    N = tilingData->N;
    M = tilingData->M;

    // 单核矩阵维度 [nNormalCoreNum_, nTailCoreNum_]
    nNormalCoreNum_ = tilingData->nNormalCoreNum;
    nTailCoreNum_ = tilingData->nTailCoreNum;

    // 单核矩阵维度 [mNormalCoreNum_, mTailCoreNum_]
    mNormalCoreNum_ = tilingData->mNormalCoreNum;
    mTailCoreNum_ = tilingData->mTailCoreNum;

    // 循环次数
    nLoopNormCore_ = tilingData->nLoopNormCore;
    nLoopTailCore_ = tilingData->nLoopTailCore;
    mLoopNumCore_ = tilingData->mLoopNumCore;

    // 使用核数 && 总核数 [totalCoreNum_, usedCoreNum_]
    totalCoreNum_ = tilingData->totalCoreNum;
    usedCoreNum_ = tilingData->usedCoreNum;

    // 行列方向的核 [nCoreNum_, mCoreNum_]
    nCoreNum_ = tilingData->nCoreNum;
    mCoreNum_ = tilingData->mCoreNum;
}

template <typename T>
__aicore__ inline void ApplyCamePart1FP16<T>::Init(GM_ADDR grad, GM_ADDR eps, GM_ADDR sum_grad_r, GM_ADDR sum_grad_c,
                                                   GM_ADDR sum_grad_rc, GM_ADDR workspace,
                                                   const ApplyCamePart1TilingData* tilingData, int64_t batchIdx)
{
    // 初始化tiling
    ParseTilingData(tilingData);
    const int64_t batchElements = batchIdx * N * M;
    const int64_t batchBytes = batchElements * sizeof(T);
    const int64_t alignedBatchBytes = batchBytes / 32 * 32;
    inputBase_ = (batchBytes - alignedBatchBytes) / sizeof(T);

    // workspace地址
    workspaceAddr_ = workspace;

    // 清零gmOutput
    SyncAllCore(sum_grad_r, sum_grad_c, sum_grad_rc);
    gmSumGradR_.SetGlobalBuffer((__gm__ float*)sum_grad_r + batchIdx * N);
    gmSumGradC_.SetGlobalBuffer((__gm__ float*)sum_grad_c + batchIdx * M);
    gmSumGradRC_.SetGlobalBuffer((__gm__ float*)sum_grad_rc + batchIdx);

    // gmInput分核 && 输入偏移初始化
    GM_ADDR alignedGrad = grad + alignedBatchBytes;
    gmGrad_.SetGlobalBuffer((__gm__ T*)alignedGrad +
                            GetBlockIdx() / mCoreNum_ * ONCE_HANDLE_NUM64 * nLoopNormCore_ * M +
                            GetBlockIdx() % mCoreNum_ * mNormalCoreNum_);
    gmEps_.SetGlobalBuffer((__gm__ float*)eps, 1);

    // buffer申请初始化
    pipe.InitBuffer(gradQueue_, 1, ONCE_HANDLE_NUM64 * ONCE_HANDLE_NUM64 * sizeof(T));
    pipe.InitBuffer(epsQueue_, 1, ONCE_ONE_SIZE8 * sizeof(float));

    pipe.InitBuffer(sumGradRQueue_, 1, ONCE_HANDLE_NUM64 * sizeof(float));
    pipe.InitBuffer(sumGradCQueue_, 1, ONCE_HANDLE_NUM64 * sizeof(float));
    pipe.InitBuffer(sumGradRCQueue_, 1, 2 * ONCE_ONE_SIZE8 * sizeof(float));
    hasColumnTail_ = HasApplyCamePart1Tail(mNormalCoreNum_, ONCE_HANDLE_NUM64) || (mTailCoreNum_ > 0);

    // 缓存buf空间清零
    ClearAcculateMatrix();
}

template <typename T>
__aicore__ inline void ApplyCamePart1FP16<T>::SyncAllCore(GM_ADDR sum_grad_r, GM_ADDR sum_grad_c, GM_ADDR sum_grad_rc)
{
    InitApplyCamePart1OutputBuffers(gmSumGradR_, gmSumGradC_, gmSumGradRC_, workspaceSumGradR_, workspaceSumGradRC_,
                                    workspaceSumGradRCLow_, workspaceSumGradC_, sum_grad_r, sum_grad_c, sum_grad_rc,
                                    workspaceAddr_, mCoreNum_, nLoopNormCore_, usedCoreNum_, nLoopTailCore_,
                                    mLoopNumCore_, ONCE_HANDLE_NUM64);
    pipe.InitBuffer(syncWorkQueue_, 1, totalCoreNum_ * 8 * sizeof(int32_t));
    SyncAll();
}

template <typename T>
__aicore__ inline void ApplyCamePart1FP16<T>::CopyInLast(int64_t nLoopIdx, int64_t mLoopIdx, int64_t curRepeatTimes,
                                                         LocalTensor<T> gradLocal)
{
    CopyInApplyCamePart1Last(gmGrad_, gradLocal, nLoopIdx, mLoopIdx, curRepeatTimes, mNormalCoreNum_, mTailCoreNum_, M,
                             ONCE_HANDLE_NUM64, inputBase_);
}

template <typename T>
__aicore__ inline void ApplyCamePart1FP16<T>::CopyInNormal(int64_t nLoopIdx, int64_t mLoopIdx, int64_t curRepeatTimes,
                                                           LocalTensor<T> gradLocal)
{
    CopyInApplyCamePart1Normal(gmGrad_, gradLocal, nLoopIdx, mLoopIdx, curRepeatTimes, M, ONCE_HANDLE_NUM64,
                               inputBase_);
}

template <typename T>
__aicore__ inline void ApplyCamePart1FP16<T>::CopyIn(int64_t nLoopIdx, int64_t mLoopIdx, int64_t curRepeatTimes)
{
    LocalTensor<T> gradLocal = gradQueue_.AllocTensor<T>();
    LocalTensor<float> epsLocal = epsQueue_.AllocTensor<float>();

    if (mLoopIdx == (mLoopNumCore_ - 1) && hasColumnTail_) {
        CopyInLast(nLoopIdx, mLoopIdx, curRepeatTimes, gradLocal);
    } else {
        CopyInNormal(nLoopIdx, mLoopIdx, curRepeatTimes, gradLocal);
    }

    DataCopyPad(epsLocal, gmEps_, {1, sizeof(float), 0, 0, 0}, {false, 0, 0, 0});
    gradQueue_.EnQue(gradLocal);
    epsQueue_.EnQue(epsLocal);
}

template <typename T>
__aicore__ inline void ApplyCamePart1FP16<T>::Compute(int64_t nLoopIdx, int64_t mLoopIdx, int64_t curRepeatTimes)
{
    LocalTensor<T> gradLocal = gradQueue_.DeQue<T>();
    LocalTensor<float> epsLocal = epsQueue_.DeQue<float>();

    LocalTensor<float> gradCastTmpUb = gradCastTmpBuf_.Get<float>();
    LocalTensor<float> gradSqrtTmpUb = gradSqrtTmpBuf_.Get<float>();
    LocalTensor<float> accuComTmpUb = accuComTmpBuf_.Get<float>();
    LocalTensor<float> reductionTmpUb = mComTmpBuf_.Get<float>();
    Duplicate(gradCastTmpUb, (float)0.0, ONCE_HANDLE_NUM64 * ONCE_HANDLE_NUM64);
    PipeBarrier<PIPE_V>();
    event_t eventIdMte2ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
    SetFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
    WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToV);

    event_t eventIdMte2ToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_S));
    SetFlag<HardEvent::MTE2_S>(eventIdMte2ToS);
    WaitFlag<HardEvent::MTE2_S>(eventIdMte2ToS);
    float eps = epsLocal.GetValue(0);

    int64_t calCount = curRepeatTimes * ONCE_HANDLE_NUM64;
    Cast(gradCastTmpUb, gradLocal, RoundMode::CAST_NONE, calCount);
    PipeBarrier<PIPE_V>();
    Duplicate(gradSqrtTmpUb, (float)0.0, ONCE_HANDLE_NUM64 * ONCE_HANDLE_NUM64);
    PipeBarrier<PIPE_V>();
    Mul(gradSqrtTmpUb, gradCastTmpUb, gradCastTmpUb, calCount);

    PipeBarrier<PIPE_V>();
    AddEpsApplyCamePart1(gradSqrtTmpUb, eps, mLoopIdx, mLoopNumCore_, hasColumnTail_, mTailCoreNum_, mNormalCoreNum_,
                         curRepeatTimes);

    PipeBarrier<PIPE_V>();
    (void)accuComTmpUb;

    ComputeR(mLoopIdx, curRepeatTimes, gradSqrtTmpUb, accuComTmpUb, reductionTmpUb);
    LocalTensor<float> sumGradRCLocal = sumGradRCQueue_.AllocTensor<float>();
    ComputeC(curRepeatTimes, gradSqrtTmpUb);
    LocalTensor<float> sumGradCLocal = sumGradCQueue_.DeQue<float>();
    FinishApplyCamePart1Reduction(sumGradRCLocal, sumGradCLocal, reductionTmpUb, ONCE_HANDLE_NUM64);

    if (mLoopIdx == (mLoopNumCore_ - 1)) {
        PipeBarrier<PIPE_V>();
        Duplicate(accuComTmpUb, static_cast<float>(0), ONCE_HANDLE_NUM64 * ONCE_HANDLE_NUM64);
        Duplicate(reductionTmpUb, static_cast<float>(0), 4 * ONCE_HANDLE_NUM64);
    }

    sumGradCQueue_.EnQue<float>(sumGradCLocal);
    sumGradRCQueue_.EnQue<float>(sumGradRCLocal);

    gradQueue_.FreeTensor(gradLocal);
    epsQueue_.FreeTensor(epsLocal);
}

template <typename T>
__aicore__ inline void ApplyCamePart1FP16<T>::ComputeR(int64_t mLoopIdx, int64_t curRepeatTimes,
                                                       LocalTensor<float> gradSqrtTmpUb, LocalTensor<float> rowTree,
                                                       LocalTensor<float> workLocal)
{
    ComputeRApplyCamePart1(sumGradRQueue_, curRepeatTimes, mLoopIdx, mLoopNumCore_, gradSqrtTmpUb, rowTree, workLocal,
                           ONCE_HANDLE_NUM64);
}

template <typename T>
__aicore__ inline void ApplyCamePart1FP16<T>::ComputeC(int64_t curRepeatTimes, LocalTensor<float> gradSqrtTmpUb)
{
    ComputeCApplyCamePart1(sumGradCQueue_, curRepeatTimes, gradSqrtTmpUb, mCoreNum_, nCoreNum_, ONCE_HANDLE_NUM64);
}

template <typename T>
__aicore__ inline void ApplyCamePart1FP16<T>::CopyOut(int64_t nLoopIdx, int64_t mLoopIdx, int64_t curRepeatTimes)
{
    if (mLoopIdx == (mLoopNumCore_ - 1)) {
        CopyOutSumGradRWorkspaceApplyCamePart1(sumGradRQueue_, workspaceSumGradR_, nLoopIdx, curRepeatTimes, mCoreNum_,
                                               nLoopNormCore_, ONCE_HANDLE_NUM64);
    }
    const int64_t nCoreIdx = GetBlockIdx() / mCoreNum_;
    const int64_t mCoreIdx = GetBlockIdx() % mCoreNum_;
    int64_t offset = ((nCoreIdx * nLoopNormCore_ + nLoopIdx) * mCoreNum_ * mLoopNumCore_ + mCoreIdx * mLoopNumCore_ +
                      mLoopIdx);

    CopyOutReductionWorkspaceApplyCamePart1(sumGradRCQueue_, sumGradCQueue_, workspaceSumGradRC_,
                                            workspaceSumGradRCLow_, workspaceSumGradC_, offset, ONCE_ONE_SIZE8,
                                            ONCE_HANDLE_NUM64);
}

template <typename T>
__aicore__ inline void ApplyCamePart1FP16<T>::Process()
{
    ProcessApplyCamePart1(*this, GetBlockIdx(), usedCoreNum_, nCoreNum_, mCoreNum_, nLoopNormCore_, nLoopTailCore_,
                          nTailCoreNum_, mLoopNumCore_, ONCE_HANDLE_NUM64);
    event_t eventIdMte3ToMte2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
    SetFlag<HardEvent::MTE3_MTE2>(eventIdMte3ToMte2);
    WaitFlag<HardEvent::MTE3_MTE2>(eventIdMte3ToMte2);
    SyncAll();
}

template <typename T>
__aicore__ inline void ApplyCamePart1FP16<T>::ProcessTile(int64_t nLoopIdx, int64_t mLoopIdx, int64_t curRepeatTimes)
{
    PipeBarrier<PIPE_ALL>();
    CopyIn(nLoopIdx, mLoopIdx, curRepeatTimes);
    Compute(nLoopIdx, mLoopIdx, curRepeatTimes);
    CopyOut(nLoopIdx, mLoopIdx, curRepeatTimes);
}

template <typename T>
__aicore__ inline void ApplyCamePart1FP16<T>::ClearAcculateMatrix()
{
    constexpr float scalarValue = 0;

    pipe.InitBuffer(gradCastTmpBuf_, ONCE_HANDLE_NUM64 * ONCE_HANDLE_NUM64 * sizeof(float));
    LocalTensor<float> gradCastTmpUb = gradCastTmpBuf_.Get<float>(ONCE_HANDLE_NUM64 * ONCE_HANDLE_NUM64);
    Duplicate(gradCastTmpUb, scalarValue, ONCE_HANDLE_NUM64 * ONCE_HANDLE_NUM64);

    pipe.InitBuffer(gradSqrtTmpBuf_, ONCE_HANDLE_NUM64 * ONCE_HANDLE_NUM64 * sizeof(float));
    LocalTensor<float> gradSqrtTmpUb = gradSqrtTmpBuf_.Get<float>(ONCE_HANDLE_NUM64 * ONCE_HANDLE_NUM64);
    Duplicate(gradSqrtTmpUb, scalarValue, ONCE_HANDLE_NUM64 * ONCE_HANDLE_NUM64);

    pipe.InitBuffer(accuComTmpBuf_, ONCE_HANDLE_NUM64 * ONCE_HANDLE_NUM64 * sizeof(float));
    LocalTensor<float> accuComTmpUb = accuComTmpBuf_.Get<float>(ONCE_HANDLE_NUM64 * ONCE_HANDLE_NUM64);
    Duplicate(accuComTmpUb, scalarValue, ONCE_HANDLE_NUM64 * ONCE_HANDLE_NUM64);

    pipe.InitBuffer(mComTmpBuf_, 4 * ONCE_HANDLE_NUM64 * sizeof(float));
    LocalTensor<float> reductionTmpUb = mComTmpBuf_.Get<float>();
    Duplicate(reductionTmpUb, scalarValue, 4 * ONCE_HANDLE_NUM64);
}

} // namespace ApplyCamePart1
#endif // APPLY_CAME_PART1_FP16

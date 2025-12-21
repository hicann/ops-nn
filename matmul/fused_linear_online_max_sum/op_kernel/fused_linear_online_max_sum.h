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
 * \file fused_linear_online_max_sum.h
 * \brief
 */
#ifndef _OP_KERNEL_FUSED_LINEAR_ONLINE_MAX_SUM_H_
#define _OP_KERNEL_FUSED_LINEAR_ONLINE_MAX_SUM_H_

#include "kernel_operator.h"
#include "lib/matmul_intf.h"

namespace FusedLinearOnlineMaxSum {
using namespace AscendC;

constexpr uint32_t BITS_B32 = 32;
constexpr uint32_t DATA_PER_UINT8 = 8;
constexpr uint64_t BLOCK_BYTES = 32;
constexpr uint64_t BYTES_B32 = 4;
constexpr uint32_t DATA_PER_BLOCK_B32 = 8;
constexpr uint32_t DATA_PER_BLOCK_B16 = 16;
constexpr uint32_t DATA_PER_BLOCK_B8 = 32;
constexpr uint64_t DOUBLE_BUFFER = 2;
constexpr uint64_t BASE_M = 128;
constexpr uint64_t BASE_N = 256;
constexpr uint64_t BASE_BLOCK_SIZE = 32768;
constexpr uint64_t DOUBLE_BASE_BLOCK_SIZE = 65536;

constexpr int32_t NUM_TWO = 2;
constexpr int32_t NUM_FOUR = 4;
constexpr int32_t QUARTER_BASIC_BLOCK_SIZE = 8192;
constexpr int32_t MAX_REPEAT_TIMES_FOR_REDUCE = 255;
constexpr int32_t DATA_PER_REPEAT_MAX_OUT = 8;
constexpr uint32_t MAX_LOOP_NUM = 64;

constexpr uint64_t DATA_PER_REPEAT_B32 = 64;
constexpr uint64_t DATA_TWO_REPEAT_B32 = 128;
constexpr uint64_t DATA_THREE_REPEAT_B32 = 192;
constexpr uint64_t THRESHOLD_BLOCK_NUM = 8;
constexpr uint32_t THRESHOLD_DIM_M = 5;
constexpr uint64_t SYNC_MODE2 = 2;
constexpr uint64_t SYNC_AIC_AIV_FLAG_FIVE = 5;
constexpr uint64_t SYNC_AIV_AIC_FLAG_SIX = 6;
constexpr uint64_t SYNC_AIV_AIC_FLAG_SEVEN = 7;
constexpr int32_t FLOAT32_NEG_INF = 0xFF800000;
constexpr uint16_t FLOAT16_NEG_INF = 0xFC00;
constexpr uint16_t BF16_NEG_INF = 0xFF80;
constexpr MatmulConfig matmulCFGUnitFlag{false, false, true, 0, 0, 0, false, false, false, false, false, 0, 0, 0,
                                         0, 0, 0, 0, true};

struct MNConfig {
    uint64_t m = 0;
    uint64_t k = 0;
    uint64_t n = 0;
    uint64_t baseM = 0;
    uint64_t baseN = 0;
    uint64_t mIdx = 0;
    uint64_t nIdx = 0;
    uint64_t blockDimM = 0;
    uint64_t blockDimN = 0;
    uint64_t singleM = 0;
    uint64_t singleN = 0;
    uint64_t rowSize = 0;
};

template <typename inputT, typename targetT, typename mmT, uint64_t mmOutFlag>
class FusedLinearOnlineMaxSumOp {
public:
    __aicore__ inline FusedLinearOnlineMaxSumOp(){};
    __aicore__ inline void Init(
        const FusedLinearOnlineMaxSumTilingData &__restrict tilingData, GM_ADDR input, GM_ADDR weight, GM_ADDR target,
        GM_ADDR logitsMaxLocal, GM_ADDR sumExpLogitsLocal, GM_ADDR predictedLogitsLocal, GM_ADDR targetMask,
        GM_ADDR maskedTarget, GM_ADDR vocabParallelLogitsOut, GM_ADDR userWorkspace, TPipe *inputPipe);
    __aicore__ inline void Process();

protected:
    template <typename T>
    __aicore__ inline T GreatestCommonDivisor(T a, T b) {
        T c = a;
        if (a < b) {
            a = b;
            b = c;
        }
        while (b != 0) {
            c = a;
            a = b;
            b = c % b;
        }
        return a;
    }
    template <typename T>
    __aicore__ inline T LeastCommonMultiple(T a, T b) {
        return a * b / GreatestCommonDivisor(a, b);
    }
    template <typename T>
    __aicore__ inline T AlignDown(T a, T base) {
        if (unlikely(base == 0)) {
            return a;
        }
        return a / base * base;
    }

    __aicore__ inline void InitTilingData(const FusedLinearOnlineMaxSumTilingData &__restrict tilingData);
    __aicore__ inline void InitOutputAndWorkspace();
    __aicore__ inline void MatmulInputEmptyProcess();
    __aicore__ inline void MatmulInputEmptyMmOut(uint64_t mLoopNum, uint64_t mOffset);

    __aicore__ inline void TargetProcess();
    __aicore__ inline void TargetInnerProcess(uint64_t targetGmOffset, uint64_t tasks, uint64_t copyCount);
    
    __aicore__ inline void CVProcess();

    __aicore__ inline void SetMNConfig(MNConfig &mnConfig);
    __aicore__ inline void SetMKN(MNConfig &mnConfig);
    __aicore__ inline void MNBlockIdxCompute(
        MNConfig &mnConfig, const uint64_t curBlock, const uint64_t count, const uint64_t thresholdM_dimN);
    __aicore__ inline void MMCompute(MNConfig& mnConfig, uint64_t tailN, uint64_t outOffset, bool enSequentialWrite);

    __aicore__ inline void OnlineMaxSumInitBuffer();
    __aicore__ inline void OnlineMaxSumProcess(MNConfig& mnConfig, uint64_t baseN, uint64_t tailN, uint64_t outOffset);
    __aicore__ inline void DataCopyCastVocabParallelLogitsOut(
        MNConfig& mnConfig, uint32_t loopNum, uint32_t curSingleN, uint64_t outOffset);
    __aicore__ inline void SetDataCopyPadParamsAndInit(
        LocalTensor<uint16_t>& tmpTensor, DataCopyExtParams& extParams, DataCopyPadExtParams<mmT>& padExtParams,
        MNConfig& mnConfig, uint32_t curSingleN);
    __aicore__ inline void OnlineCopyIn(
        MNConfig& mnConfig, uint64_t loopStart, uint32_t loopNum, uint32_t curSingleN, uint64_t outOffset);
    __aicore__ inline void OnlineMaxCompute(MNConfig& mnConfig, uint32_t loopNum);
    __aicore__ inline void OnlineGetPredictedLogits(MNConfig& mnConfig, uint32_t loopNum, uint32_t curSingleN);
    __aicore__ inline void OnlineSumCompute(uint32_t loopNum);
    
    __aicore__ inline void AllocEventIDs();
    __aicore__ inline void ReleaseEventIDs();
    __aicore__ inline void vecIndexCompute(LocalTensor<int32_t> &vecIndexTensor);
    __aicore__ inline void AllReduceMaxSumInitBuffer();
    __aicore__ inline void AllReduceMaxSumInnerTiling(
        uint32_t& loopOffset, uint32_t& outerLoop, uint32_t& outerLoopTail, uint32_t& batchNum);
    __aicore__ inline void AllReduceMaxSumProcess();
    __aicore__ inline void AllReduceMaxSumSetCopyInParams(
        DataCopyExtParams& extCopyFromWsParams, DataCopyExtParams& extCopyFromWsTailParams, uint32_t outerLoopTail);
    __aicore__ inline void AllReduceMaxSumCopyIn(
        DataCopyExtParams& extCopyFromWsParams, uint64_t gmOffset, uint32_t calCount, uint64_t pp);
    __aicore__ inline void AllReduceMaxSumCompute(uint64_t gmOffset, uint32_t calCount, uint64_t pp);
    __aicore__ inline void AllReduceMaxSumCopyOut(uint64_t gmOffset, uint32_t calCount, uint64_t pp);

    __aicore__ inline void PIPE_MTE2_V() {
        event_t eventIdMte2ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        SetFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
        WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
    }
    __aicore__ inline void PIPE_V_MTE3() {
        event_t eventIdVToMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
        SetFlag<HardEvent::V_MTE3>(eventIdVToMte3);
        WaitFlag<HardEvent::V_MTE3>(eventIdVToMte3);
    }
    __aicore__ inline void PIPE_V_MTE2() {
        event_t eventIdVToMte2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE2));
        SetFlag<HardEvent::V_MTE2>(eventIdVToMte2);
        WaitFlag<HardEvent::V_MTE2>(eventIdVToMte2);
    }
    __aicore__ inline void PIPE_MTE3_MTE2() {
        event_t eventIdMte3ToMte2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
        SetFlag<HardEvent::MTE3_MTE2>(eventIdMte3ToMte2);
        WaitFlag<HardEvent::MTE3_MTE2>(eventIdMte3ToMte2);
    }
protected:
    TPipe *pipe_;

    using inputType = matmul::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, inputT>;
    using weightType = matmul::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, inputT, true>;
    using mmType = matmul::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, mmT>;
    using biasType = matmul::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, inputT>;
    matmul::MatmulImpl<inputType, weightType, mmType, biasType, matmulCFGUnitFlag> mm_;
    // gm Input
    GlobalTensor<inputT> gmInput_;
    GlobalTensor<inputT> gmWeight_;
    GlobalTensor<targetT> gmTarget_;
    // gm Output
    GlobalTensor<float> gmLogitsMaxLocal_;
    GlobalTensor<float> gmSumExpLogitsLocal_;
    GlobalTensor<float> gmPredictedLogitsLocal_;
    GlobalTensor<uint8_t> gmTargetMask_;
    GlobalTensor<targetT> gmMaskedTarget_;
    GlobalTensor<mmT> mmOutTensor_;
    // gm workspace
    GlobalTensor<float> gmOnlineMax_;
    GlobalTensor<float> gmOnlineSum_;

    // tilingData
    uint64_t m_ = 0;
    uint64_t k_ = 0;
    uint64_t n_ = 0;
    uint64_t bufSize_ = 0;
    uint64_t cubeCoreNum_ = 0;
    uint64_t vecCoreNum_ = 0;
    float vocabStartIndex_ = 0;
    float vocabEndIndex_ = 0;
    uint64_t initWorkspaceLength_ = 0;
    uint64_t matmulInputEmptyFlag_ = 0;
    uint64_t targetTasksPerLoop_ = 0;
    const TCubeTiling* __restrict mmTilingData_;

    int64_t blockIdx_ = 0;
    int64_t coreIdx_ = 0;

    uint64_t curVecTasks_ = 0;
    uint64_t batchOffset_ = 0;
    uint64_t targetPadNum_ = 0;
    uint32_t cubeCoreNumAligned_ = 0;
    BinaryRepeatParams selectRepeatParams{1, 0, 1, DEFAULT_REPEAT_STRIDE, 0, DEFAULT_REPEAT_STRIDE};
    event_t eventIdMte2ToVList[DOUBLE_BUFFER];
    event_t eventIdVToMte3ToVList[DOUBLE_BUFFER];
    event_t eventIdMte3ToMte2List[DOUBLE_BUFFER];
    uint64_t SYNC_AIV_AIC_FLAG_LIST[DOUBLE_BUFFER] = {SYNC_AIV_AIC_FLAG_SIX, SYNC_AIV_AIC_FLAG_SEVEN};

    TBuf<TPosition::VECCALC> calcBuf_;
    LocalTensor<targetT> targetTensor;
    LocalTensor<uint8_t> targetMaskOutTensor;
    LocalTensor<targetT> maskedTargetTensor;
    LocalTensor<uint8_t> compareTensor;
    LocalTensor<float> zeroTensor;

    LocalTensor<int32_t> vecIndexTensor;
    LocalTensor<float> vocabParallelLogitsOutTensor;
    LocalTensor<float> maxTensor;
    LocalTensor<float> brcbMaxTensor;
    LocalTensor<float> brcbSumTensor;
    LocalTensor<float> sumTensor;
    LocalTensor<float> preMaxTensor;
    LocalTensor<float> preSumTensor;
    LocalTensor<float> predictedLogitsTensor;
    LocalTensor<uint8_t> compareLtTensor;
    LocalTensor<uint8_t> compareGeTensor;

    LocalTensor<float> maxCopyTensor;
    LocalTensor<float> sumCopyTensor;
    LocalTensor<float> maxOutTensor;
    LocalTensor<float> sumOutTensor;
    LocalTensor<float> brcbTensor;
    LocalTensor<float> predictedLogitsOutTensor;
};

template <typename inputT, typename targetT, typename mmT, uint64_t mmOutFlag>
__aicore__ inline void FusedLinearOnlineMaxSumOp<inputT, targetT, mmT, mmOutFlag>::InitTilingData(
    const FusedLinearOnlineMaxSumTilingData &__restrict tilingData) {
    m_ = tilingData.m;
    k_ = tilingData.k;
    n_ = tilingData.n;
    bufSize_ = tilingData.bufSize;
    cubeCoreNum_ = tilingData.cubeCoreNum;
    vecCoreNum_ = tilingData.vecCoreNum;
    uint64_t batchTaksPerVecCore = tilingData.batchTaksPerVecCore;
    uint64_t batchTaksTailVecCore = tilingData.batchTaksTailVecCore;
    vocabStartIndex_ = tilingData.vocabStartIndex;
    vocabEndIndex_ = tilingData.vocabEndIndex;
    initWorkspaceLength_ = tilingData.initWorkspaceLength;
    targetTasksPerLoop_ = tilingData.targetTasksPerLoop;
    matmulInputEmptyFlag_ = tilingData.matmulInputEmptyFlag;
    mmTilingData_ = &tilingData.mmTiling;
    blockIdx_ = GetBlockIdx();
    cubeCoreNumAligned_ = ((cubeCoreNum_ + DATA_PER_BLOCK_B32 - 1) / DATA_PER_BLOCK_B32 * DATA_PER_BLOCK_B32);
    if (matmulInputEmptyFlag_ == 0) {
        mm_.Init(mmTilingData_, pipe_);
    }

    coreIdx_ = blockIdx_;
    int64_t coreRation = GetTaskRation();
    if ASCEND_IS_AIV {
        if (coreRation > 1) {
            coreIdx_ /= coreRation;
        }
    }

    if (blockIdx_ < batchTaksTailVecCore) {
        curVecTasks_ = batchTaksPerVecCore + 1;
        batchOffset_ = (blockIdx_ * curVecTasks_) * DATA_PER_UINT8;
    } else {
        curVecTasks_ = batchTaksPerVecCore;
        batchOffset_ = (blockIdx_ * batchTaksPerVecCore + batchTaksTailVecCore) * DATA_PER_UINT8;
    }
}

template <typename inputT, typename targetT, typename mmT, uint64_t mmOutFlag>
__aicore__ inline void FusedLinearOnlineMaxSumOp<inputT, targetT, mmT, mmOutFlag>::Init(
    const FusedLinearOnlineMaxSumTilingData &__restrict tilingData, GM_ADDR input, GM_ADDR weight, GM_ADDR target,
    GM_ADDR logitsMaxLocal, GM_ADDR sumExpLogitsLocal, GM_ADDR predictedLogitsLocal, GM_ADDR targetMask,
    GM_ADDR maskedTarget, GM_ADDR vocabParallelLogitsOut, GM_ADDR userWorkspace, TPipe *inputPipe) {
    pipe_ = inputPipe;
    InitTilingData(tilingData);

    gmInput_.SetGlobalBuffer(reinterpret_cast<__gm__ inputT *>(input));
    gmWeight_.SetGlobalBuffer(reinterpret_cast<__gm__ inputT *>(weight));
    gmTarget_.SetGlobalBuffer(reinterpret_cast<__gm__ targetT *>(target));
    gmLogitsMaxLocal_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(logitsMaxLocal));
    gmSumExpLogitsLocal_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(sumExpLogitsLocal));
    gmPredictedLogitsLocal_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(predictedLogitsLocal));
    gmTargetMask_.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t *>(targetMask));
    gmMaskedTarget_.SetGlobalBuffer(reinterpret_cast<__gm__ targetT *>(maskedTarget));
    if constexpr (mmOutFlag) {
        mmOutTensor_.SetGlobalBuffer(reinterpret_cast<__gm__ mmT *>(vocabParallelLogitsOut));
    } else {
        mmOutTensor_.SetGlobalBuffer(reinterpret_cast<__gm__ mmT *>(userWorkspace) + m_ * cubeCoreNum_ * NUM_FOUR);
    }

    gmOnlineMax_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(userWorkspace));
    gmOnlineSum_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(userWorkspace) + m_ * cubeCoreNum_);
    if ASCEND_IS_AIV {
        pipe_->InitBuffer(calcBuf_, bufSize_);
    }
}

template <typename inputT, typename targetT, typename mmT, uint64_t mmOutFlag>
__aicore__ inline void FusedLinearOnlineMaxSumOp<inputT, targetT, mmT, mmOutFlag>::MatmulInputEmptyProcess() {
    zeroTensor = calcBuf_.GetWithOffset<float>(initWorkspaceLength_, 0);
    LocalTensor<float> nTensor = calcBuf_.GetWithOffset<float>(initWorkspaceLength_,
                                                               sizeof(float) * initWorkspaceLength_);
    Duplicate(zeroTensor, float(0), initWorkspaceLength_);
    Duplicate(nTensor, static_cast<float>(static_cast<int64_t>(n_)), initWorkspaceLength_);

    uint64_t mCalPerVecCore = m_ / vecCoreNum_;
    uint64_t mCalPerVecCoreTail = m_ % vecCoreNum_;
    uint64_t mStartOffset = 0;
    if (blockIdx_ < mCalPerVecCoreTail) {
        mCalPerVecCore += 1;
        mStartOffset = blockIdx_ * mCalPerVecCore;
    } else {
        mStartOffset = blockIdx_ * mCalPerVecCore + mCalPerVecCoreTail;
    }

    uint64_t mLoopNum = mCalPerVecCore / initWorkspaceLength_;
    uint64_t mLoopNumTail = mCalPerVecCore % initWorkspaceLength_;
    DataCopyExtParams copyB32Params{1, static_cast<uint32_t>(sizeof(float) * initWorkspaceLength_), 0, 0, 0};
    PIPE_V_MTE3();
    for (uint64_t lp = 0; lp < mLoopNum; lp++){
        uint64_t mOffset = mStartOffset + lp * initWorkspaceLength_;
        DataCopyPad(gmLogitsMaxLocal_[mOffset], zeroTensor, copyB32Params);
        DataCopyPad(gmSumExpLogitsLocal_[mOffset], nTensor, copyB32Params);
        DataCopyPad(gmPredictedLogitsLocal_[mOffset], zeroTensor, copyB32Params);
        if constexpr (mmOutFlag) {
            MatmulInputEmptyMmOut(mLoopNum, mOffset);
        }
    }

    if (mLoopNumTail > 0) {
        DataCopyExtParams copyB32TailParams{1, static_cast<uint32_t>(sizeof(float) * mLoopNumTail), 0, 0, 0};
        uint64_t mOffset = mStartOffset + mLoopNum * initWorkspaceLength_;
        DataCopyPad(gmLogitsMaxLocal_[mOffset], zeroTensor, copyB32TailParams);
        DataCopyPad(gmSumExpLogitsLocal_[mOffset], nTensor, copyB32TailParams);
        DataCopyPad(gmPredictedLogitsLocal_[mOffset], zeroTensor, copyB32TailParams);
        if constexpr (mmOutFlag) {
            MatmulInputEmptyMmOut(mLoopNumTail, mOffset);
        }
    }
}

template <typename inputT, typename targetT, typename mmT, uint64_t mmOutFlag>
__aicore__ inline void FusedLinearOnlineMaxSumOp<inputT, targetT, mmT, mmOutFlag>::MatmulInputEmptyMmOut(
    uint64_t mLoopNum, uint64_t mOffset) {
    uint64_t mnLoopNum = mLoopNum * n_ / (initWorkspaceLength_ * NUM_TWO);
    uint64_t mnLoopNumTail = mLoopNum * n_ % (initWorkspaceLength_ * NUM_TWO);
    uint64_t mmStartOffset = mOffset * n_;
    DataCopyExtParams copyMmParams{1, static_cast<uint32_t>(sizeof(mmT) * initWorkspaceLength_ * NUM_TWO), 0, 0, 0};
    for (uint64_t lpMm = 0; lpMm < mnLoopNum; lpMm++) {
        uint64_t mmOffset = mmStartOffset + lpMm * initWorkspaceLength_ * NUM_TWO;
        DataCopyPad(mmOutTensor_[mmOffset], zeroTensor.template ReinterpretCast<mmT>(), copyMmParams);
    }

    if (mnLoopNumTail > 0) {
        DataCopyExtParams copyMmTailParams{1, static_cast<uint32_t>(sizeof(mmT) * mnLoopNumTail), 0, 0, 0};
        uint64_t mmOffset = mmStartOffset + mnLoopNum * initWorkspaceLength_ * NUM_TWO;
        DataCopyPad(mmOutTensor_[mmOffset], zeroTensor.template ReinterpretCast<mmT>(), copyMmTailParams);
        PipeBarrier<PIPE_ALL>();
    }
}

template <typename inputT, typename targetT, typename mmT, uint64_t mmOutFlag>
__aicore__ inline void FusedLinearOnlineMaxSumOp<inputT, targetT, mmT, mmOutFlag>::Process() {
    if ASCEND_IS_AIV {
        if (matmulInputEmptyFlag_ == 1) {
            MatmulInputEmptyProcess();
        } else {
            InitOutputAndWorkspace();
        }
    }
    if (matmulInputEmptyFlag_ == 0) {
        SyncAll<false>();
    }
    if ASCEND_IS_AIV {
        TargetProcess();
    }
    if (matmulInputEmptyFlag_ == 1) {
        return;
    }

    SyncAll<false>();

    CVProcess();

    SyncAll<false>();
    if ASCEND_IS_AIV {
        AllReduceMaxSumInitBuffer();
        Duplicate(zeroTensor, float(0), DATA_PER_BLOCK_B32);
        vecIndexCompute(vecIndexTensor);
        AllocEventIDs();
        AllReduceMaxSumProcess();
        ReleaseEventIDs();
    }
}

template <typename inputT, typename targetT, typename mmT, uint64_t mmOutFlag>
__aicore__ inline void FusedLinearOnlineMaxSumOp<inputT, targetT, mmT, mmOutFlag>::CVProcess() {
    MNConfig mnConfig;
    SetMNConfig(mnConfig);
    mm_.DisableBias();
    uint64_t curCount = mnConfig.blockDimM * mnConfig.blockDimN;
    uint64_t curBlock = coreIdx_;
    uint64_t thresholdM_dimN = THRESHOLD_BLOCK_NUM * mnConfig.blockDimN;
    uint64_t pp = 0;
    uint64_t tailN = 0;
    uint64_t outOffset = 0;

    uint64_t ppCount = static_cast<uint64_t>(NUM_TWO);
    uint64_t loopCount = cubeCoreNum_ == 0 ? 1 : curCount / cubeCoreNum_;
    uint64_t loopRemain = cubeCoreNum_ == 0 ? 0 : curCount % cubeCoreNum_;
    if (coreIdx_ < loopRemain) {
        loopCount += 1;
    }

    for (uint64_t count = 0; count < loopCount; count++, curBlock += cubeCoreNum_) {
        MNBlockIdxCompute(mnConfig, curBlock, 0, thresholdM_dimN);
        if ASCEND_IS_AIC {
            if (count >= ppCount) {
                CrossCoreWaitFlag(SYNC_AIV_AIC_FLAG_LIST[pp]);
            }
            tailN = mnConfig.nIdx * mnConfig.singleN;
            if constexpr (mmOutFlag) {
                outOffset = mnConfig.mIdx * mnConfig.singleM * mnConfig.n + tailN;
                MMCompute(mnConfig, tailN, outOffset, false);
            } else {
                outOffset = blockIdx_ * DOUBLE_BASE_BLOCK_SIZE +  BASE_BLOCK_SIZE * pp;
                MMCompute(mnConfig, tailN, outOffset, true);
            }
            CrossCoreSetFlag<SYNC_MODE2, PIPE_FIX>(SYNC_AIC_AIV_FLAG_FIVE);
        }
        if ASCEND_IS_AIV {
            CrossCoreWaitFlag(SYNC_AIC_AIV_FLAG_FIVE);
            tailN = mnConfig.nIdx * mnConfig.singleN;
            if constexpr (mmOutFlag) {
                outOffset = mnConfig.mIdx * mnConfig.singleM * mnConfig.n + tailN;
            } else {
                outOffset = this->blockIdx_ / NUM_TWO * DOUBLE_BASE_BLOCK_SIZE +  BASE_BLOCK_SIZE * pp;
            }
            OnlineMaxSumProcess(mnConfig, mnConfig.rowSize, tailN, outOffset);
            if ((count + ppCount < loopCount) && (loopCount > ppCount)) {
                CrossCoreSetFlag<SYNC_MODE2, PIPE_MTE3>(SYNC_AIV_AIC_FLAG_LIST[pp]);
            }
        }
        pp = 1 - pp;
    }
}

template <typename inputT, typename targetT, typename mmT, uint64_t mmOutFlag>
__aicore__ inline void FusedLinearOnlineMaxSumOp<inputT, targetT, mmT, mmOutFlag>::InitOutputAndWorkspace() {
    zeroTensor = calcBuf_.GetWithOffset<float>(initWorkspaceLength_, 0);
    LocalTensor<float> negInfTensor = calcBuf_.GetWithOffset<float>(initWorkspaceLength_,
                                                                   sizeof(float) * initWorkspaceLength_);
    
    Duplicate(negInfTensor.template ReinterpretCast<int32_t>(), FLOAT32_NEG_INF, initWorkspaceLength_);
    Duplicate(zeroTensor, float(0), initWorkspaceLength_);
    
    uint64_t workspaceStartIdx = 0;
    uint64_t workspaceTotalCopyNum = 0;
    uint64_t gmStartIdx = 0;
    if (blockIdx_ % NUM_TWO == 0) {
        workspaceStartIdx = blockIdx_ / NUM_TWO * m_;
        workspaceTotalCopyNum = m_ / NUM_TWO;
    } else {
        workspaceStartIdx = blockIdx_ / NUM_TWO * m_ + m_ / NUM_TWO;
        workspaceTotalCopyNum = m_ - m_ / NUM_TWO;
        gmStartIdx = m_ / NUM_TWO;
    }
    uint64_t loopNumWorkspace = workspaceTotalCopyNum / initWorkspaceLength_;
    uint64_t loopTailNumWorkspace = workspaceTotalCopyNum % initWorkspaceLength_;
    DataCopyExtParams copyWorkspaceParams{1, static_cast<uint32_t>(sizeof(float) * initWorkspaceLength_), 0, 0, 0};
    PIPE_V_MTE3();
    for (uint64_t loop = 0; loop < loopNumWorkspace; loop++) {
        uint64_t workspaceOffset = workspaceStartIdx + loop * initWorkspaceLength_;
        if (blockIdx_ < NUM_TWO) {
            DataCopyPad(gmPredictedLogitsLocal_[gmStartIdx + loop * initWorkspaceLength_], zeroTensor,
                        copyWorkspaceParams);
        }
        DataCopyPad(gmOnlineMax_[workspaceOffset], negInfTensor, copyWorkspaceParams);
        DataCopyPad(gmOnlineSum_[workspaceOffset], zeroTensor, copyWorkspaceParams);
    }

    if (loopTailNumWorkspace > 0) {
        DataCopyExtParams copyTailParams{1, static_cast<uint32_t>(sizeof(float) * loopTailNumWorkspace), 0, 0, 0};
        uint64_t workspaceOffset = workspaceStartIdx + loopNumWorkspace * initWorkspaceLength_;
        if (blockIdx_ < NUM_TWO) {
            DataCopyPad(gmPredictedLogitsLocal_[gmStartIdx + loopNumWorkspace * initWorkspaceLength_], zeroTensor,
                        copyTailParams);
        }
        DataCopyPad(gmOnlineMax_[workspaceOffset], negInfTensor, copyTailParams);
        DataCopyPad(gmOnlineSum_[workspaceOffset], zeroTensor, copyTailParams);
    }
}

template <typename inputT, typename targetT, typename mmT, uint64_t mmOutFlag>
__aicore__ inline void FusedLinearOnlineMaxSumOp<inputT, targetT, mmT, mmOutFlag>::TargetInnerProcess(
    uint64_t targetGmOffset, uint64_t tasks, uint64_t copyCount) {
    uint64_t targetMaskGmOffset = targetGmOffset / DATA_PER_UINT8;
    uint64_t calCount = tasks * DATA_PER_UINT8;
    uint8_t repeatTimes = static_cast<uint8_t>((calCount + DATA_PER_REPEAT_B32 - 1) / DATA_PER_REPEAT_B32);
    uint64_t calCountAlignedRepeat = 
        (calCount + DATA_PER_REPEAT_B32 - 1) / DATA_PER_REPEAT_B32 * DATA_PER_REPEAT_B32;

    DataCopyPad(targetTensor, gmTarget_[targetGmOffset],
                {1, static_cast<uint32_t>(sizeof(targetT) * copyCount), 0, 0, 0},
                {true, 0, static_cast<uint8_t>(targetPadNum_), targetT(0)});
    PIPE_MTE2_V();
    LocalTensor<float> targetTensorFp32 = targetTensor.template ReinterpretCast<float>();
    Cast(targetTensorFp32, targetTensor, RoundMode::CAST_RINT, calCount);
    PipeBarrier<PIPE_V>();

    CompareScalar(compareTensor, targetTensorFp32, vocabStartIndex_, CMPMODE::LT, calCountAlignedRepeat);
    CompareScalar(targetMaskOutTensor, targetTensorFp32, vocabEndIndex_, CMPMODE::GE, calCountAlignedRepeat);
    Adds(targetTensorFp32, targetTensorFp32, float(-vocabStartIndex_), calCount);
    PipeBarrier<PIPE_V>();
    Add(targetMaskOutTensor.template ReinterpretCast<int32_t>(),
        targetMaskOutTensor.template ReinterpretCast<int32_t>(),
        compareTensor.template ReinterpretCast<int32_t>(), calCountAlignedRepeat / BITS_B32);

    PipeBarrier<PIPE_V>();
    Select(targetTensorFp32, targetMaskOutTensor, zeroTensor, targetTensorFp32, SELMODE::VSEL_TENSOR_TENSOR_MODE,
           uint64_t(DATA_PER_REPEAT_B32), repeatTimes, selectRepeatParams);
    PipeBarrier<PIPE_V>();
    Cast(maskedTargetTensor, targetTensorFp32, RoundMode::CAST_RINT, calCount);
    PIPE_V_MTE3();
    DataCopyPad(gmTargetMask_[targetMaskGmOffset], targetMaskOutTensor, {1, static_cast<uint32_t>(tasks), 0, 0, 0});
    DataCopyPad(gmMaskedTarget_[targetGmOffset], maskedTargetTensor,
                {1, static_cast<uint32_t>(copyCount * sizeof(targetT)), 0, 0, 0});
}

template <typename inputT, typename targetT, typename mmT, uint64_t mmOutFlag>
__aicore__ inline void FusedLinearOnlineMaxSumOp<inputT, targetT, mmT, mmOutFlag>::TargetProcess() {
    uint32_t bufOffset = 0;
    uint64_t calCount = targetTasksPerLoop_ * DATA_PER_UINT8;
    uint64_t calCountAlignedRepeat = (calCount + DATA_PER_REPEAT_B32 - 1) / DATA_PER_REPEAT_B32 * DATA_PER_REPEAT_B32;
    zeroTensor = calcBuf_.GetWithOffset<float>(DATA_PER_BLOCK_B32, bufOffset);
    bufOffset += sizeof(float) * DATA_PER_BLOCK_B32;
    targetTensor = calcBuf_.GetWithOffset<targetT>(calCountAlignedRepeat, bufOffset);
    bufOffset += sizeof(targetT) * calCountAlignedRepeat;
    maskedTargetTensor = calcBuf_.GetWithOffset<targetT>(calCountAlignedRepeat, bufOffset);
    bufOffset += sizeof(targetT) * calCountAlignedRepeat;
    uint64_t numU8Align32B = (calCountAlignedRepeat / DATA_PER_UINT8 + BLOCK_BYTES - 1) / BLOCK_BYTES * BLOCK_BYTES;
    targetMaskOutTensor = calcBuf_.GetWithOffset<uint8_t>(numU8Align32B, bufOffset);
    bufOffset += sizeof(uint8_t) * numU8Align32B;
    compareTensor = calcBuf_.GetWithOffset<uint8_t>(numU8Align32B, bufOffset);

    int64_t endBlockIdx = (m_ + DATA_PER_BLOCK_B32 - 1) / DATA_PER_BLOCK_B32;
    endBlockIdx = endBlockIdx > vecCoreNum_ ? vecCoreNum_ : endBlockIdx;
    endBlockIdx = endBlockIdx - 1;
    if (blockIdx_ > endBlockIdx) {
        return;
    }

    uint64_t targetLoop = curVecTasks_ / targetTasksPerLoop_;
    uint64_t targetTailTasks = curVecTasks_ % targetTasksPerLoop_;
    Duplicate(zeroTensor, float(0), DATA_PER_BLOCK_B32);
    uint64_t copyCount = targetTasksPerLoop_ * DATA_PER_UINT8;
    for (uint64_t loopTarget = 0; loopTarget < targetLoop; loopTarget++) {
        uint64_t targetGmOffset = batchOffset_ + loopTarget * targetTasksPerLoop_ * DATA_PER_UINT8;
        TargetInnerProcess(targetGmOffset, targetTasksPerLoop_, copyCount);
        PIPE_MTE3_MTE2();
    }
    
    if (targetTailTasks > 0) {
        uint64_t targetGmOffset = batchOffset_ + targetLoop * targetTasksPerLoop_ * DATA_PER_UINT8;
        copyCount = targetTailTasks * DATA_PER_UINT8;
        if (blockIdx_ == endBlockIdx) {
            uint64_t targetNumPerBlock = BLOCK_BYTES / sizeof(targetT);
            copyCount = m_ - targetGmOffset;
            uint64_t copyCountAligned = (copyCount + targetNumPerBlock - 1) / targetNumPerBlock * targetNumPerBlock;
            targetPadNum_ = copyCountAligned - copyCount;
            if constexpr (IsSameType<targetT, int64_t>::value) {
                uint64_t dupOffset = copyCountAligned - targetNumPerBlock;
                uint64_t copyCountAlignedB32 =
                    (copyCount + DATA_PER_BLOCK_B32 - 1) / DATA_PER_BLOCK_B32 * DATA_PER_BLOCK_B32;
                uint64_t dupCount = (copyCountAlignedB32 - dupOffset) * NUM_TWO;
                Duplicate(targetTensor[dupOffset].template ReinterpretCast<int32_t>(), int32_t(0), dupCount);
                PIPE_V_MTE2();
            }
        }
        TargetInnerProcess(targetGmOffset, targetTailTasks, copyCount);
    }
}

template <typename inputT, typename targetT, typename mmT, uint64_t mmOutFlag>
__aicore__ inline void FusedLinearOnlineMaxSumOp<inputT, targetT, mmT, mmOutFlag>::SetMNConfig(MNConfig &mnConfig) {
    SetMKN(mnConfig);
    mnConfig.baseM = mmTilingData_->baseM;
    mnConfig.baseN = mmTilingData_->baseN;
    mnConfig.singleM = mnConfig.baseM;
    mnConfig.singleN = mnConfig.baseN;
    mnConfig.blockDimM = Ceil(mnConfig.m, mnConfig.singleM);
    mnConfig.blockDimN = Ceil(mnConfig.n, mnConfig.singleN);
    if constexpr (mmOutFlag) {
        mnConfig.rowSize = mnConfig.n;
    } else {
        mnConfig.rowSize = mnConfig.baseN;
    }
}

template <typename inputT, typename targetT, typename mmT, uint64_t mmOutFlag>
__aicore__ inline void FusedLinearOnlineMaxSumOp<inputT, targetT, mmT, mmOutFlag>::SetMKN(MNConfig &mnConfig) {
    mnConfig.m = m_;
    mnConfig.k = k_;
    mnConfig.n = n_;
}

template <typename inputT, typename targetT, typename mmT, uint64_t mmOutFlag>
__aicore__ inline void FusedLinearOnlineMaxSumOp<inputT, targetT, mmT, mmOutFlag>::MNBlockIdxCompute(
    MNConfig &mnConfig, const uint64_t curBlock, const uint64_t count, const uint64_t thresholdM_dimN) {
    if (mnConfig.blockDimM <= THRESHOLD_DIM_M || THRESHOLD_DIM_M == 1) {
        mnConfig.mIdx = (curBlock - count) / mnConfig.blockDimN;
        mnConfig.nIdx = (curBlock - count) % mnConfig.blockDimN;
    } else {
        uint64_t relativeBlock = curBlock - count;
        uint64_t curThresholdM = relativeBlock >= AlignDown(mnConfig.blockDimM * mnConfig.blockDimN, thresholdM_dimN) ?
            mnConfig.blockDimM % THRESHOLD_BLOCK_NUM : THRESHOLD_BLOCK_NUM;
        uint64_t curThresholdM_thresholdN = curThresholdM * THRESHOLD_BLOCK_NUM;
        uint64_t curThresholdN = relativeBlock % thresholdM_dimN >= AlignDown(curThresholdM * mnConfig.blockDimN,
            curThresholdM_thresholdN) ? mnConfig.blockDimN % THRESHOLD_BLOCK_NUM : THRESHOLD_BLOCK_NUM;

        uint64_t localRelativeBlock = relativeBlock % thresholdM_dimN % curThresholdM_thresholdN;
        mnConfig.mIdx = localRelativeBlock % curThresholdM + relativeBlock / thresholdM_dimN * THRESHOLD_BLOCK_NUM;
        mnConfig.nIdx = (localRelativeBlock + localRelativeBlock /
            LeastCommonMultiple(curThresholdM, curThresholdN)) % curThresholdN + relativeBlock %
            thresholdM_dimN / curThresholdM_thresholdN * THRESHOLD_BLOCK_NUM;
    }
}

template <typename inputT, typename targetT, typename mmT, uint64_t mmOutFlag>
__aicore__ inline void FusedLinearOnlineMaxSumOp<inputT, targetT, mmT, mmOutFlag>::MMCompute(
    MNConfig& mnConfig, uint64_t tailN, uint64_t outOffset, bool enSequentialWrite) {
    uint64_t curSingleN = mnConfig.nIdx < mnConfig.blockDimN - 1 ? mnConfig.singleN : mnConfig.n - tailN;
    uint64_t curSingleM = mnConfig.mIdx < mnConfig.blockDimM - 1 ? mnConfig.singleM
                                                                 : mnConfig.m - mnConfig.mIdx * mnConfig.singleM;
    uint64_t xOffset = mnConfig.mIdx * mnConfig.singleM * mnConfig.k;
    if constexpr (mmOutFlag) {
        mm_.SetOrgShape(mnConfig.m, mnConfig.n, mnConfig.k);
    } else {
        mm_.SetOrgShape(BASE_M, BASE_N, mnConfig.k);
    }
    mm_.SetSingleShape(curSingleM, curSingleN, mnConfig.k);
    mm_.SetTensorA(gmInput_[xOffset], false);
    mm_.SetTensorB(gmWeight_[mnConfig.nIdx * mnConfig.k * mnConfig.baseN], true);
    mm_.template IterateAll<false>(mmOutTensor_[outOffset], 0, enSequentialWrite);
}

template <typename inputT, typename targetT, typename mmT, uint64_t mmOutFlag>
__aicore__ inline void FusedLinearOnlineMaxSumOp<inputT, targetT, mmT, mmOutFlag>::SetDataCopyPadParamsAndInit(
    LocalTensor<uint16_t>& tmpTensor, DataCopyExtParams& extParams, DataCopyPadExtParams<mmT>& padExtParams,
    MNConfig& mnConfig, uint32_t curSingleN) {
    if (curSingleN != mnConfig.baseN) {
        if constexpr (IsSameType<mmT, half>::value) {
            Duplicate(tmpTensor, FLOAT16_NEG_INF, QUARTER_BASIC_BLOCK_SIZE);
            Duplicate(tmpTensor[QUARTER_BASIC_BLOCK_SIZE], FLOAT16_NEG_INF, QUARTER_BASIC_BLOCK_SIZE);
            uint16_t negInfFp16 = 0xFC00;
            mmT negInfValue = *reinterpret_cast<mmT *>(&negInfFp16);
            padExtParams.paddingValue = negInfValue;
        } else {
            Duplicate(tmpTensor, BF16_NEG_INF, QUARTER_BASIC_BLOCK_SIZE);
            Duplicate(tmpTensor[QUARTER_BASIC_BLOCK_SIZE], BF16_NEG_INF, QUARTER_BASIC_BLOCK_SIZE);
            uint16_t negInfBf16 = 0xFF80;
            mmT negInfValue = *reinterpret_cast<mmT *>(&negInfBf16);
            padExtParams.paddingValue = negInfValue;
        }
        event_t eventIdVToMte2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE2));
        SetFlag<HardEvent::V_MTE2>(eventIdVToMte2);
        WaitFlag<HardEvent::V_MTE2>(eventIdVToMte2);

        uint32_t curSingleNAligned = (curSingleN + DATA_PER_BLOCK_B16 - 1) / DATA_PER_BLOCK_B16 * DATA_PER_BLOCK_B16;
        uint8_t rightPadding = curSingleNAligned - curSingleN;
        uint32_t dstStride = (mnConfig.baseN - curSingleNAligned) / DATA_PER_BLOCK_B16;
        uint32_t srcStride = (mnConfig.rowSize - curSingleN) * sizeof(mmT);
        extParams.srcStride = srcStride;
        extParams.dstStride = dstStride;

        padExtParams.isPad = true;
        padExtParams.rightPadding = rightPadding;
        
    } else {
        uint32_t srcStride = (mnConfig.rowSize - curSingleN) * sizeof(mmT);
        uint32_t dstStride = 0;
        extParams.srcStride = srcStride;
        extParams.dstStride = dstStride;

        padExtParams.isPad = false;
        padExtParams.rightPadding = 0;
        padExtParams.paddingValue = 0;
    }
}

template <typename inputT, typename targetT, typename mmT, uint64_t mmOutFlag>
__aicore__ inline void FusedLinearOnlineMaxSumOp<inputT, targetT, mmT, mmOutFlag>::DataCopyCastVocabParallelLogitsOut(
    MNConfig& mnConfig, uint32_t loopNum, uint32_t curSingleN, uint64_t outOffset) {
    DataCopyPadExtParams<mmT> padExtParams;
    DataCopyExtParams extParams;
    extParams.blockCount = loopNum;
    uint64_t mmOutputOffset = loopNum * mnConfig.baseN;
    uint32_t blockLen = curSingleN * sizeof(mmT);
    extParams.blockLen = blockLen;
    padExtParams.leftPadding = 0;
    LocalTensor<uint16_t> tmpTensor = vocabParallelLogitsOutTensor.template ReinterpretCast<uint16_t>()[mmOutputOffset];
    SetDataCopyPadParamsAndInit(tmpTensor, extParams, padExtParams, mnConfig, curSingleN);
    DataCopyPad(vocabParallelLogitsOutTensor.template ReinterpretCast<mmT>()[mmOutputOffset],
                mmOutTensor_[outOffset], extParams, padExtParams);
    PIPE_MTE2_V();
    Cast(vocabParallelLogitsOutTensor,
         vocabParallelLogitsOutTensor.template ReinterpretCast<mmT>()[mmOutputOffset],
         RoundMode::CAST_NONE, mmOutputOffset);
}

template <typename inputT, typename targetT, typename mmT, uint64_t mmOutFlag>
__aicore__ inline void FusedLinearOnlineMaxSumOp<inputT, targetT, mmT, mmOutFlag>::OnlineMaxSumInitBuffer() {
    uint32_t bufOffset = 0;
    uint32_t baseMDivTwo = mmTilingData_->baseM / NUM_TWO;
    uint32_t basicBlockSize = mmTilingData_->baseN * mmTilingData_->baseM;
    zeroTensor = calcBuf_.GetWithOffset<float>(DATA_PER_BLOCK_B32, bufOffset);
    bufOffset += sizeof(float) * DATA_PER_BLOCK_B32;

    vecIndexTensor = calcBuf_.GetWithOffset<int32_t>(baseMDivTwo, bufOffset);
    bufOffset += sizeof(int32_t) * baseMDivTwo;
    vocabParallelLogitsOutTensor = calcBuf_.GetWithOffset<float>(mmTilingData_->baseN * baseMDivTwo, bufOffset);
    bufOffset += sizeof(float) * mmTilingData_->baseN * baseMDivTwo;
    preMaxTensor = calcBuf_.GetWithOffset<float>(baseMDivTwo, bufOffset);
    bufOffset += sizeof(float) * baseMDivTwo;
    preSumTensor = calcBuf_.GetWithOffset<float>(baseMDivTwo, bufOffset);
    bufOffset += sizeof(float) * baseMDivTwo;
    maxTensor = calcBuf_.GetWithOffset<float>(basicBlockSize  / DATA_PER_REPEAT_B32, bufOffset);
    bufOffset += sizeof(float) * basicBlockSize  / DATA_PER_REPEAT_B32;
    brcbMaxTensor = calcBuf_.GetWithOffset<float>(baseMDivTwo * DATA_PER_BLOCK_B32, bufOffset);
    bufOffset += sizeof(float) * baseMDivTwo * DATA_PER_BLOCK_B32;
    brcbSumTensor = calcBuf_.GetWithOffset<float>(baseMDivTwo * DATA_PER_BLOCK_B32, bufOffset);
    bufOffset += sizeof(float) * baseMDivTwo * DATA_PER_BLOCK_B32;
    sumTensor = calcBuf_.GetWithOffset<float>(basicBlockSize / DATA_PER_REPEAT_B32, bufOffset);
    bufOffset += sizeof(float) * basicBlockSize / DATA_PER_REPEAT_B32;

    predictedLogitsTensor = calcBuf_.GetWithOffset<float>(baseMDivTwo, bufOffset);
    bufOffset += sizeof(float) * baseMDivTwo;
    maskedTargetTensor = calcBuf_.GetWithOffset<targetT>(baseMDivTwo, bufOffset);
    bufOffset += sizeof(targetT) * baseMDivTwo;
    compareLtTensor = calcBuf_.GetWithOffset<uint8_t>(DATA_PER_BLOCK_B8, bufOffset);
    bufOffset += sizeof(uint8_t) * DATA_PER_BLOCK_B8;
    compareGeTensor = calcBuf_.GetWithOffset<uint8_t>(DATA_PER_BLOCK_B8, bufOffset);
    bufOffset += sizeof(uint8_t) * DATA_PER_BLOCK_B8;
}

template <typename inputT, typename targetT, typename mmT, uint64_t mmOutFlag>
__aicore__ inline void FusedLinearOnlineMaxSumOp<inputT, targetT, mmT, mmOutFlag>::OnlineCopyIn(
    MNConfig& mnConfig, uint64_t loopStart, uint32_t loopNum, uint32_t curSingleN, uint64_t outOffset) {
    uint64_t wsOffset = blockIdx_ / NUM_TWO * m_;
    DataCopyExtParams copyParams{1, static_cast<uint32_t>(sizeof(float) * loopNum), 0, 0, 0};
    uint64_t gmOffset = mnConfig.mIdx * mnConfig.baseM + loopStart;
    uint64_t onlineOffset = wsOffset + gmOffset;
    DataCopyPad(maskedTargetTensor, gmMaskedTarget_[gmOffset],
                {1, static_cast<uint32_t>(sizeof(targetT) * loopNum), 0, 0, 0}, {false, 0, 0, 0});
    DataCopyPad(preMaxTensor, gmOnlineMax_[onlineOffset], copyParams, {false, 0, 0, 0});
    DataCopyPad(preSumTensor, gmOnlineSum_[onlineOffset], copyParams, {false, 0, 0, 0});
    DataCopyCastVocabParallelLogitsOut(mnConfig, loopNum, curSingleN, outOffset);
}

template <typename inputT, typename targetT, typename mmT, uint64_t mmOutFlag>
__aicore__ inline void FusedLinearOnlineMaxSumOp<inputT, targetT, mmT, mmOutFlag>::OnlineMaxCompute(
    MNConfig& mnConfig, uint32_t loopNum) {
    int32_t repeatTimes = (loopNum * mnConfig.baseN + DATA_PER_REPEAT_B32 - 1) / DATA_PER_REPEAT_B32;
    int32_t blockMask = DATA_PER_REPEAT_B32;
    int32_t dstRepStride = 1;
    if (repeatTimes > MAX_REPEAT_TIMES_FOR_REDUCE) {
        int32_t repeatTimesFisrt =
            (repeatTimes / NUM_TWO + DATA_PER_REPEAT_MAX_OUT - 1) / DATA_PER_REPEAT_MAX_OUT * DATA_PER_REPEAT_MAX_OUT;
        int32_t repeatTimesSecond = repeatTimes - repeatTimesFisrt;
        WholeReduceMax(maxTensor, vocabParallelLogitsOutTensor, blockMask, repeatTimesFisrt, dstRepStride,
                       DEFAULT_BLK_STRIDE, DEFAULT_REPEAT_STRIDE);

        int32_t localOffset = repeatTimesFisrt * NUM_TWO;
        int64_t gmOffset = repeatTimesFisrt * DATA_PER_REPEAT_B32;
        WholeReduceMax(maxTensor[localOffset], vocabParallelLogitsOutTensor[gmOffset], blockMask, repeatTimesSecond,
                       dstRepStride, DEFAULT_BLK_STRIDE, DEFAULT_REPEAT_STRIDE);
    } else {
        WholeReduceMax(maxTensor, vocabParallelLogitsOutTensor, blockMask, repeatTimes, dstRepStride,
                       DEFAULT_BLK_STRIDE, DEFAULT_REPEAT_STRIDE);
    }
    PipeBarrier<PIPE_V>();
    uint8_t dupRepeatTimes =
        (loopNum * DATA_PER_REPEAT_MAX_OUT + DATA_PER_REPEAT_B32 - 1) / DATA_PER_REPEAT_B32;
    uint64_t dupMask[2] = {0b1010101010101010101010101010101010101010101010101010101010101010, 0};
    Duplicate(maxTensor.template ReinterpretCast<int32_t>(), FLOAT32_NEG_INF, dupMask, dupRepeatTimes, 1,
              DEFAULT_REPEAT_STRIDE);
    PipeBarrier<PIPE_V>();
    BlockReduceMax(maxTensor, maxTensor, dupRepeatTimes, blockMask, dstRepStride, DEFAULT_BLK_STRIDE,
                   DEFAULT_REPEAT_STRIDE);
    PipeBarrier<PIPE_V>();
    Max(maxTensor, maxTensor, preMaxTensor, loopNum);
}

template <typename inputT, typename targetT, typename mmT, uint64_t mmOutFlag>
__aicore__ inline void FusedLinearOnlineMaxSumOp<inputT, targetT, mmT, mmOutFlag>::OnlineGetPredictedLogits(
    MNConfig& mnConfig, uint32_t loopNum, uint32_t curSingleN) {
    float startIndex = -static_cast<int32_t>(mnConfig.nIdx * mnConfig.baseN);
    LocalTensor<float> maskedTargetTensorFp32 = maskedTargetTensor.template ReinterpretCast<float>();
    Cast(maskedTargetTensorFp32, maskedTargetTensor, RoundMode::CAST_RINT, loopNum);
    PipeBarrier<PIPE_V>();
    Adds(maskedTargetTensorFp32, maskedTargetTensorFp32, startIndex, loopNum);
    Muls(vecIndexTensor, vecIndexTensor, int32_t(BYTES_B32 * mnConfig.baseN), loopNum);
    PipeBarrier<PIPE_V>();
    UnaryRepeatParams unaryParams{DEFAULT_BLK_STRIDE, DEFAULT_BLK_STRIDE, DEFAULT_REPEAT_STRIDE, DEFAULT_REPEAT_STRIDE};
    CompareScalar(compareLtTensor, maskedTargetTensorFp32, float(0), CMPMODE::LT, uint64_t(loopNum), 1,
                  unaryParams);
    CompareScalar(compareGeTensor, maskedTargetTensorFp32, float(static_cast<int32_t>(curSingleN)), CMPMODE::GE,
                  uint64_t(loopNum), 1, unaryParams);
    PipeBarrier<PIPE_V>();
    Add(compareLtTensor.template ReinterpretCast<int32_t>(), compareLtTensor.template ReinterpretCast<int32_t>(),
        compareGeTensor.template ReinterpretCast<int32_t>(), mmTilingData_->baseM / NUM_TWO / BITS_B32);
    PipeBarrier<PIPE_V>();
    Select(maskedTargetTensorFp32, compareLtTensor, zeroTensor, maskedTargetTensorFp32,
           SELMODE::VSEL_TENSOR_TENSOR_MODE, uint64_t(loopNum), 1, selectRepeatParams);
    PipeBarrier<PIPE_V>();
    LocalTensor<int32_t> maskedTargetTensorInt32 = maskedTargetTensor.template ReinterpretCast<int32_t>();
    Cast(maskedTargetTensorInt32, maskedTargetTensorFp32, RoundMode::CAST_RINT, loopNum);
    PipeBarrier<PIPE_V>();
    Muls(maskedTargetTensorInt32, maskedTargetTensorInt32, int32_t(BYTES_B32), loopNum);
    PipeBarrier<PIPE_V>();
    Add(maskedTargetTensorInt32, maskedTargetTensorInt32, vecIndexTensor, loopNum);
    PipeBarrier<PIPE_V>();
    Gather(predictedLogitsTensor, vocabParallelLogitsOutTensor, maskedTargetTensor.template ReinterpretCast<uint32_t>(),
           0, loopNum);
    PipeBarrier<PIPE_V>();
    Select(predictedLogitsTensor, compareLtTensor, zeroTensor, predictedLogitsTensor,
           SELMODE::VSEL_TENSOR_TENSOR_MODE, uint64_t(loopNum), 1, selectRepeatParams);
}

template <typename inputT, typename targetT, typename mmT, uint64_t mmOutFlag>
__aicore__ inline void FusedLinearOnlineMaxSumOp<inputT, targetT, mmT, mmOutFlag>::OnlineSumCompute(
    uint32_t loopNum) {
    Brcb(brcbMaxTensor, maxTensor, mmTilingData_->baseM / NUM_TWO / DATA_PER_BLOCK_B32,
         {DEFAULT_BLK_STRIDE, DEFAULT_REPEAT_STRIDE});
    Sub(preMaxTensor, preMaxTensor, maxTensor, loopNum);
    PipeBarrier<PIPE_V>();
    uint64_t mask = DATA_PER_REPEAT_B32;
    uint8_t repeatTimesSub = loopNum * mmTilingData_->baseN / DATA_PER_REPEAT_B32 / NUM_FOUR;
    uint8_t subRepeatStride = NUM_FOUR * DEFAULT_REPEAT_STRIDE;
    BinaryRepeatParams repeatParams{DEFAULT_BLK_STRIDE, DEFAULT_BLK_STRIDE, 0, subRepeatStride, subRepeatStride, 1};
    Sub(vocabParallelLogitsOutTensor, vocabParallelLogitsOutTensor, brcbMaxTensor, mask, repeatTimesSub, repeatParams);
    Sub(vocabParallelLogitsOutTensor[DATA_PER_REPEAT_B32], vocabParallelLogitsOutTensor[DATA_PER_REPEAT_B32],
        brcbMaxTensor, mask, repeatTimesSub, repeatParams);
    Sub(vocabParallelLogitsOutTensor[DATA_TWO_REPEAT_B32], vocabParallelLogitsOutTensor[DATA_TWO_REPEAT_B32],
        brcbMaxTensor, mask, repeatTimesSub, repeatParams);
    Sub(vocabParallelLogitsOutTensor[DATA_THREE_REPEAT_B32], vocabParallelLogitsOutTensor[DATA_THREE_REPEAT_B32],
        brcbMaxTensor, mask, repeatTimesSub, repeatParams);
    Duplicate(sumTensor, float(0), mmTilingData_->baseN * mmTilingData_->baseM / DATA_PER_REPEAT_B32);
    Exp(preMaxTensor, preMaxTensor, loopNum);
    PipeBarrier<PIPE_V>();
    Exp(vocabParallelLogitsOutTensor, vocabParallelLogitsOutTensor, loopNum * mmTilingData_->baseN);
    Mul(preSumTensor, preSumTensor, preMaxTensor, loopNum);
    PipeBarrier<PIPE_V>();
    int32_t wholeReduceSumRepeatTimes = loopNum * NUM_TWO;
    WholeReduceSum(sumTensor, vocabParallelLogitsOutTensor, int32_t(DATA_PER_REPEAT_B32), wholeReduceSumRepeatTimes,
                   NUM_TWO, DEFAULT_BLK_STRIDE, DEFAULT_REPEAT_STRIDE);
    WholeReduceSum(sumTensor[wholeReduceSumRepeatTimes * NUM_TWO],
                   vocabParallelLogitsOutTensor[wholeReduceSumRepeatTimes * DATA_PER_REPEAT_B32],
                   int32_t(DATA_PER_REPEAT_B32), wholeReduceSumRepeatTimes, NUM_TWO, DEFAULT_BLK_STRIDE,
                   DEFAULT_REPEAT_STRIDE);
    PipeBarrier<PIPE_V>();
    WholeReduceSum(sumTensor, sumTensor, DATA_PER_REPEAT_MAX_OUT, loopNum, 1, 1, 1);
    PipeBarrier<PIPE_V>();
    Add(sumTensor, preSumTensor, sumTensor, loopNum);
}

template <typename inputT, typename targetT, typename mmT, uint64_t mmOutFlag>
__aicore__ inline void FusedLinearOnlineMaxSumOp<inputT, targetT, mmT, mmOutFlag>::OnlineMaxSumProcess(
    MNConfig& mnConfig, uint64_t baseN, uint64_t tailN, uint64_t outOffset) {
    uint64_t curSingleN = mnConfig.nIdx < mnConfig.blockDimN - 1 ? mnConfig.singleN : mnConfig.n - tailN;
    uint64_t curSingleM = mnConfig.mIdx < mnConfig.blockDimM - 1 ? mnConfig.singleM
                                                                 : mnConfig.m - mnConfig.mIdx * mnConfig.singleM;

    uint32_t loopStart = curSingleM / NUM_TWO;
    uint32_t loopCount = curSingleM;
    uint32_t loopNum = loopCount - loopStart;
    if (blockIdx_ % NUM_TWO == 0) {
        loopCount = loopStart;
        loopNum = loopStart;
        loopStart = 0;
    } else {
        outOffset = outOffset + loopStart * baseN;
    }
    
    uint64_t wsOffset = blockIdx_ / NUM_TWO * m_;

    OnlineMaxSumInitBuffer();
    Duplicate(zeroTensor, float(0), DATA_PER_BLOCK_B32);
    CreateVecIndex(vecIndexTensor, int32_t(0), loopNum);
    OnlineCopyIn(mnConfig, loopStart, loopNum, curSingleN, outOffset);
    PIPE_MTE2_V();
    OnlineMaxCompute(mnConfig, loopNum);
    PIPE_V_MTE3();
    DataCopyPad(gmOnlineMax_[wsOffset + mnConfig.mIdx * mnConfig.baseM + loopStart], maxTensor,
                {1, static_cast<uint32_t>(sizeof(float) * loopNum), 0, 0, 0});
    OnlineGetPredictedLogits(mnConfig, loopNum, curSingleN);
    PIPE_V_MTE3();
    SetAtomicAdd<float>();
    DataCopyPad(gmPredictedLogitsLocal_[mnConfig.mIdx * mnConfig.baseM + loopStart], predictedLogitsTensor,
                {1, static_cast<uint32_t>(sizeof(float) * loopNum), 0, 0, 0});
    SetAtomicNone();
    PipeBarrier<PIPE_V>();
    OnlineSumCompute(loopNum);
    PIPE_V_MTE3();
    DataCopyPad(gmOnlineSum_[wsOffset + mnConfig.mIdx * mnConfig.baseM + loopStart], sumTensor,
                {1, static_cast<uint32_t>(sizeof(float) * loopNum), 0, 0, 0});
}

template <typename inputT, typename targetT, typename mmT, uint64_t mmOutFlag>
__aicore__ inline void FusedLinearOnlineMaxSumOp<inputT, targetT, mmT, mmOutFlag>::vecIndexCompute(
    LocalTensor<int32_t> &vecIndexTensor) {
    int32_t addsNum = 4;
    CreateVecIndex(vecIndexTensor, int32_t(0), cubeCoreNumAligned_);
    PipeBarrier<PIPE_V>();
    Muls(vecIndexTensor, vecIndexTensor, int32_t(BYTES_B32 * MAX_LOOP_NUM), cubeCoreNumAligned_);
    PipeBarrier<PIPE_V>();
    for (int32_t cof = 1; cof < MAX_LOOP_NUM;) {
        Adds(vecIndexTensor[cubeCoreNumAligned_ * cof], vecIndexTensor, addsNum * cof, cubeCoreNumAligned_ * cof);
        PipeBarrier<PIPE_V>();
        cof = cof * 2;
    }
}

template <typename inputT, typename targetT, typename mmT, uint64_t mmOutFlag>
__aicore__ inline void FusedLinearOnlineMaxSumOp<inputT, targetT, mmT, mmOutFlag>::AllReduceMaxSumInitBuffer() {
    uint32_t maxLoopNumDoubleBuffer = MAX_LOOP_NUM * DOUBLE_BUFFER;
    uint32_t maxLoopNumAllCoreDoubleBuffer = maxLoopNumDoubleBuffer * cubeCoreNumAligned_;
    uint32_t bufOffset = 0;
    zeroTensor = calcBuf_.GetWithOffset<float>(DATA_PER_BLOCK_B32 * DOUBLE_BUFFER, bufOffset);
    bufOffset += sizeof(float) * DATA_PER_BLOCK_B32 * DOUBLE_BUFFER;
    vecIndexTensor = calcBuf_.GetWithOffset<int32_t>(maxLoopNumAllCoreDoubleBuffer, bufOffset);
    bufOffset += sizeof(int32_t) * maxLoopNumAllCoreDoubleBuffer;

    maxCopyTensor = calcBuf_.GetWithOffset<float>(maxLoopNumAllCoreDoubleBuffer, bufOffset);
    bufOffset += sizeof(float) * maxLoopNumAllCoreDoubleBuffer;
    sumCopyTensor = calcBuf_.GetWithOffset<float>(maxLoopNumAllCoreDoubleBuffer, bufOffset);
    bufOffset += sizeof(float) * maxLoopNumAllCoreDoubleBuffer;
    maxTensor = calcBuf_.GetWithOffset<float>(maxLoopNumAllCoreDoubleBuffer, bufOffset);
    bufOffset += sizeof(float) * maxLoopNumAllCoreDoubleBuffer;
    sumTensor = calcBuf_.GetWithOffset<float>(maxLoopNumAllCoreDoubleBuffer, bufOffset);
    bufOffset += sizeof(float) * maxLoopNumAllCoreDoubleBuffer;
    maxOutTensor = calcBuf_.GetWithOffset<float>(maxLoopNumDoubleBuffer, bufOffset);
    bufOffset += sizeof(float) * maxLoopNumDoubleBuffer;
    sumOutTensor = calcBuf_.GetWithOffset<float>(maxLoopNumDoubleBuffer, bufOffset);
    bufOffset += sizeof(float) * maxLoopNumDoubleBuffer;
    brcbTensor = calcBuf_.GetWithOffset<float>(maxLoopNumDoubleBuffer * DATA_PER_BLOCK_B32, bufOffset);
    bufOffset += sizeof(float) * maxLoopNumDoubleBuffer * DATA_PER_BLOCK_B32;

    targetTensor = calcBuf_.GetWithOffset<targetT>(maxLoopNumDoubleBuffer, bufOffset);
    bufOffset += sizeof(targetT) * maxLoopNumDoubleBuffer;
    compareLtTensor = calcBuf_.GetWithOffset<uint8_t>(DATA_PER_BLOCK_B8 * DOUBLE_BUFFER, bufOffset);
    bufOffset += sizeof(uint8_t) * DATA_PER_BLOCK_B8 * DOUBLE_BUFFER;
    compareGeTensor = calcBuf_.GetWithOffset<uint8_t>(DATA_PER_BLOCK_B8 * DOUBLE_BUFFER, bufOffset);
    bufOffset += sizeof(uint8_t) * DATA_PER_BLOCK_B8 * DOUBLE_BUFFER;
    predictedLogitsTensor = calcBuf_.GetWithOffset<float>(maxLoopNumDoubleBuffer, bufOffset);
    bufOffset += sizeof(float) * maxLoopNumDoubleBuffer;
    predictedLogitsOutTensor = calcBuf_.GetWithOffset<float>(maxLoopNumDoubleBuffer, bufOffset);
}

template <typename inputT, typename targetT, typename mmT, uint64_t mmOutFlag>
__aicore__ inline void FusedLinearOnlineMaxSumOp<inputT, targetT, mmT, mmOutFlag>::AllReduceMaxSumSetCopyInParams(
    DataCopyExtParams& extCopyFromWsParams, DataCopyExtParams& extCopyFromWsTailParams, uint32_t outerLoopTail) {
    extCopyFromWsParams.blockCount = cubeCoreNum_;
    extCopyFromWsParams.blockLen =  MAX_LOOP_NUM * sizeof(float);
    extCopyFromWsParams.srcStride = (m_ - MAX_LOOP_NUM) * sizeof(float);
    extCopyFromWsParams.dstStride = 0;

    extCopyFromWsTailParams.blockCount = cubeCoreNum_;
    extCopyFromWsTailParams.blockLen =  outerLoopTail * sizeof(float);
    extCopyFromWsTailParams.srcStride = (m_ - outerLoopTail) * sizeof(float);
    extCopyFromWsTailParams.dstStride = (MAX_LOOP_NUM - outerLoopTail) / DATA_PER_BLOCK_B32;
}

template <typename inputT, typename targetT, typename mmT, uint64_t mmOutFlag>
__aicore__ inline void FusedLinearOnlineMaxSumOp<inputT, targetT, mmT, mmOutFlag>::AllReduceMaxSumCopyIn(
    DataCopyExtParams& extCopyFromWsParams, uint64_t gmOffset, uint32_t calCount, uint64_t pp) {
    uint32_t maxLoopNumPpOffset = pp * MAX_LOOP_NUM;
    uint32_t maxLoopNumAllCorePpOffset = maxLoopNumPpOffset * cubeCoreNumAligned_;
    DataCopyPad(maxCopyTensor[maxLoopNumAllCorePpOffset], gmOnlineMax_[gmOffset],
                extCopyFromWsParams, {false, 0, 0, 0});
    DataCopyPad(sumCopyTensor[maxLoopNumAllCorePpOffset], gmOnlineSum_[gmOffset],
                extCopyFromWsParams, {false, 0, 0, 0});
    DataCopyPad(predictedLogitsTensor[maxLoopNumPpOffset], gmPredictedLogitsLocal_[gmOffset],
                {1, static_cast<uint32_t>(sizeof(float) * calCount), 0, 0, 0}, {false, 0, 0, 0});
    DataCopyPad(targetTensor[maxLoopNumPpOffset], gmTarget_[gmOffset],
                {1, static_cast<uint32_t>(sizeof(targetT) * calCount), 0, 0, 0}, {false, 0, 0, 0});
}

template <typename inputT, typename targetT, typename mmT, uint64_t mmOutFlag>
__aicore__ inline void FusedLinearOnlineMaxSumOp<inputT, targetT, mmT, mmOutFlag>::AllReduceMaxSumCompute(
    uint64_t gmOffset, uint32_t calCount, uint64_t pp) {
    uint32_t cmpResPpOffset = pp * DATA_PER_BLOCK_B8;
    uint32_t loopPpOffset = pp * MAX_LOOP_NUM;
    uint32_t loopPpAllCoreOffset = loopPpOffset * cubeCoreNumAligned_;
    int32_t cubeCoreNumCeilDiv = (cubeCoreNum_ + DATA_PER_BLOCK_B32 - 1) / DATA_PER_BLOCK_B32;
    uint32_t calCountAllCore = calCount * cubeCoreNumAligned_;
    uint8_t subStride = cubeCoreNumAligned_ / DATA_PER_BLOCK_B32;
    UnaryRepeatParams unaryParams{DEFAULT_BLK_STRIDE, DEFAULT_BLK_STRIDE, DEFAULT_REPEAT_STRIDE, DEFAULT_REPEAT_STRIDE};
    PipeBarrier<PIPE_V>();
    Gather(maxTensor[loopPpAllCoreOffset], maxCopyTensor[loopPpAllCoreOffset],
           vecIndexTensor.template ReinterpretCast<uint32_t>(), uint32_t(0), calCountAllCore);
    PipeBarrier<PIPE_V>();
    WholeReduceMax(maxOutTensor[loopPpOffset], maxTensor[loopPpAllCoreOffset], int32_t(cubeCoreNum_), calCount,
                   1, 1, cubeCoreNumCeilDiv, ReduceOrder::ORDER_ONLY_VALUE);
    Cast(targetTensor[loopPpOffset].template ReinterpretCast<float>(), targetTensor[loopPpOffset],
         RoundMode::CAST_RINT, calCount);
    PipeBarrier<PIPE_ALL>();
    CompareScalar(compareLtTensor[cmpResPpOffset], targetTensor[loopPpOffset].template ReinterpretCast<float>(),
                  vocabStartIndex_, CMPMODE::LT, uint64_t(calCount), 1, unaryParams);
    CompareScalar(compareGeTensor[cmpResPpOffset], targetTensor[loopPpOffset].template ReinterpretCast<float>(),
                  vocabEndIndex_, CMPMODE::GE, uint64_t(calCount), 1, unaryParams);
    PipeBarrier<PIPE_V>();
    Sub(predictedLogitsTensor[loopPpOffset], predictedLogitsTensor[loopPpOffset], maxOutTensor[loopPpOffset], calCount);
    Add(compareLtTensor[cmpResPpOffset].template ReinterpretCast<int32_t>(),
        compareLtTensor[cmpResPpOffset].template ReinterpretCast<int32_t>(),
        compareGeTensor[cmpResPpOffset].template ReinterpretCast<int32_t>(), MAX_LOOP_NUM / BITS_B32);
    PipeBarrier<PIPE_V>();
    Select(predictedLogitsOutTensor[loopPpOffset], compareLtTensor[cmpResPpOffset], zeroTensor,
           predictedLogitsTensor[loopPpOffset], SELMODE::VSEL_TENSOR_TENSOR_MODE, uint64_t(DATA_PER_REPEAT_B32),
           MAX_LOOP_NUM / DATA_PER_REPEAT_B32, selectRepeatParams);
    Brcb(brcbTensor[loopPpOffset * DATA_PER_BLOCK_B32], maxOutTensor[loopPpOffset], MAX_LOOP_NUM / DATA_PER_BLOCK_B32,
         {DEFAULT_BLK_STRIDE, DEFAULT_REPEAT_STRIDE});
    PipeBarrier<PIPE_V>();
    Sub(maxTensor[loopPpAllCoreOffset], maxTensor[loopPpAllCoreOffset],
        brcbTensor[pp * MAX_LOOP_NUM * DATA_PER_BLOCK_B32], uint64_t(cubeCoreNum_), calCount,
        {1, 1, 0, subStride, subStride, 1});
    PipeBarrier<PIPE_V>();
    Exp(maxTensor[loopPpAllCoreOffset], maxTensor[loopPpAllCoreOffset], calCountAllCore);
    Gather(sumTensor[loopPpAllCoreOffset], sumCopyTensor[loopPpAllCoreOffset],
           vecIndexTensor.template ReinterpretCast<uint32_t>(), uint32_t(0), calCountAllCore);
    PipeBarrier<PIPE_V>();
    Mul(sumTensor[loopPpAllCoreOffset], sumTensor[loopPpAllCoreOffset], maxTensor[loopPpAllCoreOffset],
        calCountAllCore);
    PipeBarrier<PIPE_V>();
    WholeReduceSum(sumOutTensor[loopPpOffset], sumTensor[loopPpAllCoreOffset], int32_t(cubeCoreNum_), calCount, 1, 1,
                   cubeCoreNumCeilDiv);
}

template <typename inputT, typename targetT, typename mmT, uint64_t mmOutFlag>
__aicore__ inline void FusedLinearOnlineMaxSumOp<inputT, targetT, mmT, mmOutFlag>::AllReduceMaxSumCopyOut(
    uint64_t gmOffset, uint32_t calCount, uint64_t pp) {
    uint32_t ubOffset = pp * MAX_LOOP_NUM;
    DataCopyExtParams copyParams{1, static_cast<uint32_t>(sizeof(float) * calCount), 0, 0, 0};
    DataCopyPad(gmLogitsMaxLocal_[gmOffset], maxOutTensor[ubOffset], copyParams);
    DataCopyPad(gmSumExpLogitsLocal_[gmOffset], sumOutTensor[ubOffset], copyParams);
    DataCopyPad(gmPredictedLogitsLocal_[gmOffset], predictedLogitsOutTensor[ubOffset], copyParams);
}

template <typename inputT, typename targetT, typename mmT, uint64_t mmOutFlag>
__aicore__ inline void FusedLinearOnlineMaxSumOp<inputT, targetT, mmT, mmOutFlag>::AllocEventIDs() {
    event_t eventIdMte2ToVPing = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::MTE2_V>());
    event_t eventIdMte2ToVPong = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::MTE2_V>());
    event_t eventIdVToMte3Ping = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::V_MTE3>());
    event_t eventIdVToMte3Pong = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::V_MTE3>());
    event_t eventIdMte3ToMte2Ping = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::MTE3_MTE2>());
    event_t eventIdMte3ToMte2Pong = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::MTE3_MTE2>());

    eventIdMte2ToVList[0] = eventIdMte2ToVPing;
    eventIdMte2ToVList[1] = eventIdMte2ToVPong;
    eventIdVToMte3ToVList[0] = eventIdVToMte3Ping;
    eventIdVToMte3ToVList[1] = eventIdVToMte3Pong;
    eventIdMte3ToMte2List[0] = eventIdMte3ToMte2Ping;
    eventIdMte3ToMte2List[1] = eventIdMte3ToMte2Pong;
}

template <typename inputT, typename targetT, typename mmT, uint64_t mmOutFlag>
__aicore__ inline void FusedLinearOnlineMaxSumOp<inputT, targetT, mmT, mmOutFlag>::ReleaseEventIDs() {
    GetTPipePtr()->ReleaseEventID<AscendC::HardEvent::MTE2_V>(eventIdMte2ToVList[0]);
    GetTPipePtr()->ReleaseEventID<AscendC::HardEvent::MTE2_V>(eventIdMte2ToVList[1]);
    GetTPipePtr()->ReleaseEventID<AscendC::HardEvent::V_MTE3>(eventIdVToMte3ToVList[0]);
    GetTPipePtr()->ReleaseEventID<AscendC::HardEvent::V_MTE3>(eventIdVToMte3ToVList[1]);
    GetTPipePtr()->ReleaseEventID<AscendC::HardEvent::MTE3_MTE2>(eventIdMte3ToMte2List[0]);
    GetTPipePtr()->ReleaseEventID<AscendC::HardEvent::MTE3_MTE2>(eventIdMte3ToMte2List[1]);
}

template <typename inputT, typename targetT, typename mmT, uint64_t mmOutFlag>
__aicore__ inline void FusedLinearOnlineMaxSumOp<inputT, targetT, mmT, mmOutFlag>::AllReduceMaxSumInnerTiling(
    uint32_t& loopOffset, uint32_t& outerLoop, uint32_t& outerLoopTail, uint32_t& batchNum) {
    uint32_t mPerCore = m_ / vecCoreNum_;
    uint32_t mPerCoreTail = m_ % vecCoreNum_;
    if (blockIdx_ < mPerCoreTail) {
        batchNum = mPerCore + 1;
        loopOffset = batchNum * blockIdx_;
    } else {
        batchNum = mPerCore;
        loopOffset = batchNum * blockIdx_ + mPerCoreTail;
    }

    outerLoop = batchNum / MAX_LOOP_NUM;
    outerLoopTail = batchNum % MAX_LOOP_NUM;

    if (outerLoopTail > 0) {
        outerLoop = outerLoop + 1; 
    } else {
        outerLoopTail = MAX_LOOP_NUM;
    }
}

template <typename inputT, typename targetT, typename mmT, uint64_t mmOutFlag>
__aicore__ inline void FusedLinearOnlineMaxSumOp<inputT, targetT, mmT, mmOutFlag>::AllReduceMaxSumProcess() {
    uint32_t calCount = MAX_LOOP_NUM;
    uint32_t loopOffset = 0;
    uint32_t outerLoop = 0;
    uint32_t outerLoopTail = 0;
    uint32_t batchNum = 0;
    AllReduceMaxSumInnerTiling(loopOffset, outerLoop, outerLoopTail, batchNum);
    if (batchNum == 0) {
        return;
    }

    DataCopyExtParams extCopyFromWsParams;
    DataCopyExtParams extCopyFromWsTailParams;
    AllReduceMaxSumSetCopyInParams(extCopyFromWsParams, extCopyFromWsTailParams, outerLoopTail);

    uint64_t pp = 0;
    uint64_t gmOffsetPre = loopOffset;
    PIPE_MTE3_MTE2();
    if (outerLoop == 1) {
        AllReduceMaxSumCopyIn(extCopyFromWsTailParams, gmOffsetPre, outerLoopTail, pp);
    } else {
        AllReduceMaxSumCopyIn(extCopyFromWsParams, gmOffsetPre, calCount, pp);
    }
    SetFlag<HardEvent::MTE2_V>(eventIdMte2ToVList[pp]);
    for (uint32_t loopNum = 1; loopNum < outerLoop; loopNum++) {
        uint64_t gmOffset = loopOffset + loopNum * MAX_LOOP_NUM;
        WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToVList[pp]);
        AllReduceMaxSumCompute(gmOffsetPre, calCount, pp);
        SetFlag<HardEvent::V_MTE3>(eventIdVToMte3ToVList[pp]);
        WaitFlag<HardEvent::V_MTE3>(eventIdVToMte3ToVList[pp]);
        if (loopNum > 1) {
            WaitFlag<HardEvent::MTE3_MTE2>(eventIdMte3ToMte2List[1 - pp]);
        }
        if (loopNum == outerLoop - 1) {
            AllReduceMaxSumCopyIn(extCopyFromWsTailParams, gmOffset, outerLoopTail, 1 - pp);
        } else {
            AllReduceMaxSumCopyIn(extCopyFromWsParams, gmOffset, calCount, 1 - pp);
        }
        SetFlag<HardEvent::MTE2_V>(eventIdMte2ToVList[1 - pp]);
        AllReduceMaxSumCopyOut(gmOffsetPre, calCount, pp);
        if (loopNum < outerLoop - 1) {
            SetFlag<HardEvent::MTE3_MTE2>(eventIdMte3ToMte2List[pp]);
        }
        pp = 1 - pp;
        gmOffsetPre = gmOffset;
    }
    WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToVList[pp]);
    AllReduceMaxSumCompute(gmOffsetPre, outerLoopTail, pp);
    SetFlag<HardEvent::V_MTE3>(eventIdVToMte3ToVList[pp]);
    WaitFlag<HardEvent::V_MTE3>(eventIdVToMte3ToVList[pp]);
    AllReduceMaxSumCopyOut(gmOffsetPre, outerLoopTail, pp);
}

} // FusedLinearOnlineMaxSum
#endif // _OP_KERNEL_FUSED_LINEAR_ONLINE_MAX_SUM_H_
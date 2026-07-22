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
 * \file fused_cross_entropy_loss_with_max_sum_ar.h
 * \brief FusedCrossEntropyLossWithMaxSum arch35(regbase) kernel:
 *        loss[b]      = log(sum_exp_logits[b]) - predicted_logits[b]
 *        softmax[b,v] = exp(vocab_parallel_logits[b,v] - logits_max[b]) / sum_exp_logits[b]
 */

#ifndef FUSED_CROSS_ENTROPY_LOSS_WITH_MAX_SUM_AR_H
#define FUSED_CROSS_ENTROPY_LOSS_WITH_MAX_SUM_AR_H

#include "kernel_tiling/kernel_tiling.h"
#include "kernel_operator.h"
#include "../inc/kernel_utils.h"
#include "op_kernel/platform_util.h"
#include "fused_cross_entropy_loss_with_max_sum_tiling_data.h"

namespace FusedCrossEntropyLossWithMaxSumOps {
using namespace AscendC;

constexpr static AscendC::MicroAPI::CastTrait castTraitB16ToFp32 = {
    AscendC::MicroAPI::RegLayout::ZERO,
    AscendC::MicroAPI::SatMode::UNKNOWN,
    AscendC::MicroAPI::MaskMergeMode::ZEROING,
    AscendC::RoundMode::UNKNOWN,
};

template <typename T, bool fullPath>
class FusedCrossEntropyLossWithMaxSumRegBase {
public:
    __aicore__ inline FusedCrossEntropyLossWithMaxSumRegBase() = default;

    __aicore__ inline void Init(GM_ADDR logitsMax, GM_ADDR sumExpLogits, GM_ADDR predictedLogits,
                                GM_ADDR vocabParallelLogits, GM_ADDR loss, GM_ADDR softmaxLogits,
                                const FusedCrossEntropyLossWithMaxSumRegBaseTilingData* tilingData, TPipe* pipe);
    __aicore__ inline void Process();

private:
    // 完整路径：loss + softmax
    __aicore__ inline void ProcessFull();
    __aicore__ inline void ProcessRowTile(int64_t rowBase, int64_t aRows);
    __aicore__ inline void CopyInScalar(const LocalTensor<float>& ub, const GlobalTensor<float>& gm, int64_t offset,
                                        int64_t len);
    __aicore__ inline void ComputeLossInvSum(const LocalTensor<float>& sumExpUb, const LocalTensor<float>& predictedUb,
                                             const LocalTensor<float>& lossUb, const LocalTensor<float>& invSumUb,
                                             int64_t aRows);
    __aicore__ inline void CopyInVocabTile(const LocalTensor<T>& ub, int64_t rowBase, int64_t vOff, int64_t aRows,
                                           int64_t vCur);
    __aicore__ inline void ComputeSoftmaxTile(const LocalTensor<T>& vocabUb, const LocalTensor<float>& outUb,
                                              const LocalTensor<float>& maxUb, const LocalTensor<float>& invSumUb,
                                              int64_t aRows, int64_t vCur);
    __aicore__ inline void CopyOutSoftmaxTile(const LocalTensor<float>& ub, int64_t rowBase, int64_t vOff,
                                              int64_t aRows, int64_t vCur);
    // 省显存路径：仅loss
    __aicore__ inline void ProcessMemory();
    __aicore__ inline void ComputeLossChunk(const LocalTensor<float>& sumExpUb, const LocalTensor<float>& predictedUb,
                                            const LocalTensor<float>& lossUb, int64_t len);
    __aicore__ inline void LoadVocabToFp32(__ubuf__ T* src, AscendC::MicroAPI::RegTensor<float>& dst,
                                           AscendC::MicroAPI::MaskReg& preg, uint32_t offset);

private:
    static constexpr uint32_t VL_FP32 = Ops::Base::GetVRegSize() / sizeof(float);
    static constexpr int64_t A_PER_LOOP = FUSED_CE_MAX_SUM_A_PER_LOOP;
    static constexpr uint32_t UB_BLOCK_SIZE = 32U;
    static constexpr uint32_t SCALAR_BUF_BYTES = VL_FP32 * sizeof(float);

    TPipe* pipe_ = nullptr;
    const FusedCrossEntropyLossWithMaxSumRegBaseTilingData* tl_ = nullptr;

    GlobalTensor<T> vocabGm_;
    GlobalTensor<float> logitsMaxGm_;
    GlobalTensor<float> sumExpGm_;
    GlobalTensor<float> predictedGm_;
    GlobalTensor<float> softmaxGm_;
    GlobalTensor<float> lossGm_;

    // 完整路径buffer
    TQue<QuePosition::VECIN, 2> vocabQue_;
    TQue<QuePosition::VECOUT, 2> softmaxQue_;
    TQue<QuePosition::VECIN, 1> logitsMaxQue_;
    TQue<QuePosition::VECIN, 1> sumExpScalarQue_;
    TQue<QuePosition::VECIN, 1> predictedScalarQue_;
    TQue<QuePosition::VECOUT, 1> lossScalarQue_;
    TBuf<> invSumBuf_;
    // 省显存路径buffer
    TQue<QuePosition::VECIN, 2> sumExpQue_;
    TQue<QuePosition::VECIN, 2> predictedQue_;
    TQue<QuePosition::VECOUT, 2> lossQue_;

    int64_t myRows_ = 0;
    int64_t rowStart_ = 0;
    int64_t vPartIdx_ = 0; // v切分时本核在一行内负责的v分片序号（0..vCores-1），不切分恒为0
};

template <typename T, bool fullPath>
__aicore__ inline void FusedCrossEntropyLossWithMaxSumRegBase<T, fullPath>::Init(
    GM_ADDR logitsMax, GM_ADDR sumExpLogits, GM_ADDR predictedLogits, GM_ADDR vocabParallelLogits, GM_ADDR loss,
    GM_ADDR softmaxLogits, const FusedCrossEntropyLossWithMaxSumRegBaseTilingData* tilingData, TPipe* pipe)
{
    pipe_ = pipe;
    tl_ = tilingData;
    int64_t blockIdx = static_cast<int64_t>(AscendC::GetBlockIdx());
    // v切分时blockIdx按（行块, v分片）二维排布：blockIdx = rowBlockIdx * vCores + vPartIdx
    int64_t rowBlockIdx = tl_->vCores > 1 ? blockIdx / tl_->vCores : blockIdx;
    vPartIdx_ = tl_->vCores > 1 ? blockIdx % tl_->vCores : 0;
    if (rowBlockIdx < tl_->formerCoreNum) {
        myRows_ = tl_->formerRows;
        rowStart_ = rowBlockIdx * tl_->formerRows;
    } else {
        myRows_ = tl_->latterRows;
        rowStart_ = tl_->formerCoreNum * tl_->formerRows + (rowBlockIdx - tl_->formerCoreNum) * tl_->latterRows;
    }

    sumExpGm_.SetGlobalBuffer((__gm__ float*)sumExpLogits);
    predictedGm_.SetGlobalBuffer((__gm__ float*)predictedLogits);
    lossGm_.SetGlobalBuffer((__gm__ float*)loss);
    if constexpr (fullPath) {
        vocabGm_.SetGlobalBuffer((__gm__ T*)vocabParallelLogits);
        logitsMaxGm_.SetGlobalBuffer((__gm__ float*)logitsMax);
        softmaxGm_.SetGlobalBuffer((__gm__ float*)softmaxLogits);
        pipe_->InitBuffer(vocabQue_, 2, A_PER_LOOP * tl_->vPerLoop * sizeof(T));
        pipe_->InitBuffer(softmaxQue_, 2, A_PER_LOOP * tl_->vPerLoop * sizeof(float));
        pipe_->InitBuffer(logitsMaxQue_, 1, SCALAR_BUF_BYTES);
        pipe_->InitBuffer(sumExpScalarQue_, 1, SCALAR_BUF_BYTES);
        pipe_->InitBuffer(predictedScalarQue_, 1, SCALAR_BUF_BYTES);
        pipe_->InitBuffer(lossScalarQue_, 1, SCALAR_BUF_BYTES);
        pipe_->InitBuffer(invSumBuf_, SCALAR_BUF_BYTES);
    } else {
        pipe_->InitBuffer(sumExpQue_, 2, tl_->elementsNumber * sizeof(float));
        pipe_->InitBuffer(predictedQue_, 2, tl_->elementsNumber * sizeof(float));
        pipe_->InitBuffer(lossQue_, 2, tl_->elementsNumber * sizeof(float));
    }
}

template <typename T, bool fullPath>
__aicore__ inline void FusedCrossEntropyLossWithMaxSumRegBase<T, fullPath>::Process()
{
    if (myRows_ <= 0) {
        return;
    }
    if constexpr (fullPath) {
        ProcessFull();
    } else {
        ProcessMemory();
    }
}

template <typename T, bool fullPath>
__aicore__ inline void FusedCrossEntropyLossWithMaxSumRegBase<T, fullPath>::ProcessFull()
{
    int64_t aLoops = ops::CeilDiv(myRows_, A_PER_LOOP);
    for (int64_t i = 0; i < aLoops; i++) {
        int64_t rowBase = rowStart_ + i * A_PER_LOOP;
        int64_t aRows = (i == aLoops - 1) ? (myRows_ - i * A_PER_LOOP) : A_PER_LOOP;
        ProcessRowTile(rowBase, aRows);
    }
}

template <typename T, bool fullPath>
__aicore__ inline void FusedCrossEntropyLossWithMaxSumRegBase<T, fullPath>::ProcessRowTile(int64_t rowBase,
                                                                                           int64_t aRows)
{
    // 1. 搬入本行块的 logits_max / sum_exp_logits / predicted_logits 标量行向量
    LocalTensor<float> maxUb = logitsMaxQue_.AllocTensor<float>();
    LocalTensor<float> sumUb = sumExpScalarQue_.AllocTensor<float>();
    LocalTensor<float> predUb = predictedScalarQue_.AllocTensor<float>();
    CopyInScalar(maxUb, logitsMaxGm_, rowBase, aRows);
    CopyInScalar(sumUb, sumExpGm_, rowBase, aRows);
    CopyInScalar(predUb, predictedGm_, rowBase, aRows);
    logitsMaxQue_.EnQue(maxUb);
    sumExpScalarQue_.EnQue(sumUb);
    predictedScalarQue_.EnQue(predUb);
    maxUb = logitsMaxQue_.DeQue<float>();
    sumUb = sumExpScalarQue_.DeQue<float>();
    predUb = predictedScalarQue_.DeQue<float>();

    // 2. 计算本行块的 loss = log(sum_exp) - predicted，以及 softmax 要用的 inv_sum = 1 / sum_exp
    //    v切分时每个v分片核都需要inv_sum（各自独立计算），但loss只由每行vPartIdx==0的核写出，避免重复写
    LocalTensor<float> invSumUb = invSumBuf_.Get<float>();
    LocalTensor<float> lossUb = lossScalarQue_.AllocTensor<float>();
    ComputeLossInvSum(sumUb, predUb, lossUb, invSumUb, aRows);
    lossScalarQue_.EnQue(lossUb);
    lossUb = lossScalarQue_.DeQue<float>();
    if (vPartIdx_ == 0) {
        DataCopyExtParams lossCopyParams{1, static_cast<uint32_t>(aRows * sizeof(float)), 0, 0, 0};
        DataCopyPad(lossGm_[rowBase], lossUb, lossCopyParams);
    }
    lossScalarQue_.FreeTensor(lossUb);

    // 3. v维分块计算 softmax = exp(vocab - max) * inv_sum（v切分时只处理本核的v分片[vBegin, vEnd)）
    int64_t vBegin = vPartIdx_ * tl_->vChunk;
    int64_t vEnd = (vBegin + tl_->vChunk) < tl_->vLen ? (vBegin + tl_->vChunk) : tl_->vLen;
    int64_t vLoops = ops::CeilDiv(vEnd - vBegin, tl_->vPerLoop);
    for (int64_t v = 0; v < vLoops; v++) {
        int64_t vOff = vBegin + v * tl_->vPerLoop;
        int64_t vCur = (v == vLoops - 1) ? (vEnd - vOff) : tl_->vPerLoop;
        LocalTensor<T> vocabUb = vocabQue_.AllocTensor<T>();
        CopyInVocabTile(vocabUb, rowBase, vOff, aRows, vCur);
        vocabQue_.EnQue(vocabUb);
        vocabUb = vocabQue_.DeQue<T>();

        LocalTensor<float> outUb = softmaxQue_.AllocTensor<float>();
        ComputeSoftmaxTile(vocabUb, outUb, maxUb, invSumUb, aRows, vCur);
        vocabQue_.FreeTensor(vocabUb);

        softmaxQue_.EnQue(outUb);
        outUb = softmaxQue_.DeQue<float>();
        CopyOutSoftmaxTile(outUb, rowBase, vOff, aRows, vCur);
        softmaxQue_.FreeTensor(outUb);
    }

    logitsMaxQue_.FreeTensor(maxUb);
    sumExpScalarQue_.FreeTensor(sumUb);
    predictedScalarQue_.FreeTensor(predUb);
}

template <typename T, bool fullPath>
__aicore__ inline void FusedCrossEntropyLossWithMaxSumRegBase<T, fullPath>::CopyInScalar(const LocalTensor<float>& ub,
                                                                                         const GlobalTensor<float>& gm,
                                                                                         int64_t offset, int64_t len)
{
    DataCopyExtParams copyParams{1, static_cast<uint32_t>(len * sizeof(float)), 0, 0, 0};
    DataCopyPadExtParams<float> padParams{false, 0, 0, 0};
    DataCopyPad(ub, gm[offset], copyParams, padParams);
}

template <typename T, bool fullPath>
__aicore__ inline void FusedCrossEntropyLossWithMaxSumRegBase<T, fullPath>::ComputeLossInvSum(
    const LocalTensor<float>& sumExpUb, const LocalTensor<float>& predictedUb, const LocalTensor<float>& lossUb,
    const LocalTensor<float>& invSumUb, int64_t aRows)
{
    __ubuf__ float* sumAddr = (__ubuf__ float*)sumExpUb.GetPhyAddr();
    __ubuf__ float* predAddr = (__ubuf__ float*)predictedUb.GetPhyAddr();
    __ubuf__ float* lossAddr = (__ubuf__ float*)lossUb.GetPhyAddr();
    __ubuf__ float* invAddr = (__ubuf__ float*)invSumUb.GetPhyAddr();
    __VEC_SCOPE__
    {
        uint32_t count = static_cast<uint32_t>(aRows);
        AscendC::MicroAPI::RegTensor<float> sumReg, predReg, logReg, oneReg, invReg;
        AscendC::MicroAPI::MaskReg pMask = AscendC::MicroAPI::UpdateMask<float>(count);
        AscendC::MicroAPI::DataCopy(sumReg, sumAddr);
        AscendC::MicroAPI::DataCopy(predReg, predAddr);
        AscendC::MicroAPI::Log(logReg, sumReg, pMask);
        AscendC::MicroAPI::Sub(logReg, logReg, predReg, pMask);
        AscendC::MicroAPI::DataCopy(lossAddr, logReg, pMask);
        AscendC::MicroAPI::Duplicate(oneReg, 1.0f, pMask);
        AscendC::MicroAPI::Div(invReg, oneReg, sumReg, pMask);
        AscendC::MicroAPI::DataCopy(invAddr, invReg, pMask);
    }
}

template <typename T, bool fullPath>
__aicore__ inline void FusedCrossEntropyLossWithMaxSumRegBase<T, fullPath>::CopyInVocabTile(const LocalTensor<T>& ub,
                                                                                            int64_t rowBase,
                                                                                            int64_t vOff, int64_t aRows,
                                                                                            int64_t vCur)
{
    // GM行距为vLen，UB行距为vPerLoop（32B对齐）
    int64_t blockLen = vCur * static_cast<int64_t>(sizeof(T));
    DataCopyExtParams copyParams;
    copyParams.blockCount = static_cast<uint16_t>(aRows);
    copyParams.blockLen = static_cast<uint32_t>(blockLen);
    copyParams.srcStride = static_cast<uint32_t>((tl_->vLen - vCur) * sizeof(T));
    copyParams.dstStride = static_cast<uint32_t>((tl_->vPerLoop * static_cast<int64_t>(sizeof(T)) -
                                                  ops::Aligned(blockLen, static_cast<int64_t>(UB_BLOCK_SIZE))) /
                                                 static_cast<int64_t>(UB_BLOCK_SIZE));
    DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
    DataCopyPad(ub, vocabGm_[rowBase * tl_->vLen + vOff], copyParams, padParams);
}

template <typename T, bool fullPath>
__aicore__ inline void FusedCrossEntropyLossWithMaxSumRegBase<T, fullPath>::LoadVocabToFp32(
    __ubuf__ T* src, AscendC::MicroAPI::RegTensor<float>& dst, AscendC::MicroAPI::MaskReg& preg, uint32_t offset)
{
    if constexpr (IsSameType<T, float>::value) {
        AscendC::MicroAPI::DataCopy<float, AscendC::MicroAPI::LoadDist::DIST_NORM>(dst, src + offset);
    } else {
        AscendC::MicroAPI::RegTensor<T> b16Reg;
        AscendC::MicroAPI::DataCopy<T, AscendC::MicroAPI::LoadDist::DIST_UNPACK_B16>(b16Reg, src + offset);
        AscendC::MicroAPI::Cast<float, T, castTraitB16ToFp32>(dst, b16Reg, preg);
    }
}

template <typename T, bool fullPath>
__aicore__ inline void FusedCrossEntropyLossWithMaxSumRegBase<T, fullPath>::ComputeSoftmaxTile(
    const LocalTensor<T>& vocabUb, const LocalTensor<float>& outUb, const LocalTensor<float>& maxUb,
    const LocalTensor<float>& invSumUb, int64_t aRows, int64_t vCur)
{
    __ubuf__ T* vocabAddr = (__ubuf__ T*)vocabUb.GetPhyAddr();
    __ubuf__ float* outAddr = (__ubuf__ float*)outUb.GetPhyAddr();
    __ubuf__ float* maxAddr = (__ubuf__ float*)maxUb.GetPhyAddr();
    __ubuf__ float* invAddr = (__ubuf__ float*)invSumUb.GetPhyAddr();
    uint16_t aRowsU16 = static_cast<uint16_t>(aRows);
    // vPerLoop按VL_FP32对齐，满块迭代复用编译期生成的满掩码；尾块写成0/1次迭代的for，避免VF内if分支
    uint16_t fullLoops = static_cast<uint16_t>(vCur / static_cast<int64_t>(VL_FP32));
    uint16_t totalLoops = static_cast<uint16_t>(ops::CeilDiv(vCur, static_cast<int64_t>(VL_FP32)));
    uint32_t tailCount = static_cast<uint32_t>(vCur) - fullLoops * VL_FP32;
    uint32_t rowStride = static_cast<uint32_t>(tl_->vPerLoop);
    __VEC_SCOPE__
    {
        AscendC::MicroAPI::RegTensor<float> maxReg, invReg, xReg, yReg;
        AscendC::MicroAPI::MaskReg
            fullMask = AscendC::MicroAPI::CreateMask<float, AscendC::MicroAPI::MaskPattern::ALL>();
        for (uint16_t j = 0; j < aRowsU16; j++) {
            // 每行的 max / inv_sum 广播加载进寄存器
            AscendC::MicroAPI::DataCopy<float, AscendC::MicroAPI::LoadDist::DIST_BRC_B32>(maxReg, maxAddr + j);
            AscendC::MicroAPI::DataCopy<float, AscendC::MicroAPI::LoadDist::DIST_BRC_B32>(invReg, invAddr + j);
            uint32_t offset = j * rowStride;
            for (uint16_t i = 0; i < fullLoops; i++) {
                LoadVocabToFp32(vocabAddr, xReg, fullMask, offset);
                AscendC::MicroAPI::Sub(yReg, xReg, maxReg, fullMask);
                AscendC::MicroAPI::Exp(yReg, yReg, fullMask);
                AscendC::MicroAPI::Mul(yReg, yReg, invReg, fullMask);
                AscendC::MicroAPI::DataCopy(outAddr + offset, yReg, fullMask);
                offset += VL_FP32;
            }
            for (uint16_t i = fullLoops; i < totalLoops; i++) {
                uint32_t tail = tailCount;
                AscendC::MicroAPI::MaskReg tailMask = AscendC::MicroAPI::UpdateMask<float>(tail);
                LoadVocabToFp32(vocabAddr, xReg, tailMask, offset);
                AscendC::MicroAPI::Sub(yReg, xReg, maxReg, tailMask);
                AscendC::MicroAPI::Exp(yReg, yReg, tailMask);
                AscendC::MicroAPI::Mul(yReg, yReg, invReg, tailMask);
                AscendC::MicroAPI::DataCopy(outAddr + offset, yReg, tailMask);
            }
        }
    }
}

template <typename T, bool fullPath>
__aicore__ inline void FusedCrossEntropyLossWithMaxSumRegBase<T, fullPath>::CopyOutSoftmaxTile(
    const LocalTensor<float>& ub, int64_t rowBase, int64_t vOff, int64_t aRows, int64_t vCur)
{
    // UB行距为vPerLoop，GM行距为vLen
    int64_t blockLen = vCur * static_cast<int64_t>(sizeof(float));
    DataCopyExtParams copyParams;
    copyParams.blockCount = static_cast<uint16_t>(aRows);
    copyParams.blockLen = static_cast<uint32_t>(blockLen);
    copyParams.dstStride = static_cast<uint32_t>((tl_->vLen - vCur) * sizeof(float));
    copyParams.srcStride = static_cast<uint32_t>((tl_->vPerLoop * static_cast<int64_t>(sizeof(float)) -
                                                  ops::Aligned(blockLen, static_cast<int64_t>(UB_BLOCK_SIZE))) /
                                                 static_cast<int64_t>(UB_BLOCK_SIZE));
    DataCopyPad(softmaxGm_[rowBase * tl_->vLen + vOff], ub, copyParams);
}

template <typename T, bool fullPath>
__aicore__ inline void FusedCrossEntropyLossWithMaxSumRegBase<T, fullPath>::ProcessMemory()
{
    int64_t loops = ops::CeilDiv(myRows_, tl_->elementsNumber);
    for (int64_t i = 0; i < loops; i++) {
        int64_t offset = rowStart_ + i * tl_->elementsNumber;
        int64_t len = (i == loops - 1) ? (myRows_ - i * tl_->elementsNumber) : tl_->elementsNumber;

        LocalTensor<float> sumUb = sumExpQue_.AllocTensor<float>();
        LocalTensor<float> predUb = predictedQue_.AllocTensor<float>();
        CopyInScalar(sumUb, sumExpGm_, offset, len);
        CopyInScalar(predUb, predictedGm_, offset, len);
        sumExpQue_.EnQue(sumUb);
        predictedQue_.EnQue(predUb);
        sumUb = sumExpQue_.DeQue<float>();
        predUb = predictedQue_.DeQue<float>();

        LocalTensor<float> lossUb = lossQue_.AllocTensor<float>();
        ComputeLossChunk(sumUb, predUb, lossUb, len);
        sumExpQue_.FreeTensor(sumUb);
        predictedQue_.FreeTensor(predUb);

        lossQue_.EnQue(lossUb);
        lossUb = lossQue_.DeQue<float>();
        DataCopyExtParams copyParams{1, static_cast<uint32_t>(len * sizeof(float)), 0, 0, 0};
        DataCopyPad(lossGm_[offset], lossUb, copyParams);
        lossQue_.FreeTensor(lossUb);
    }
}

template <typename T, bool fullPath>
__aicore__ inline void FusedCrossEntropyLossWithMaxSumRegBase<T, fullPath>::ComputeLossChunk(
    const LocalTensor<float>& sumExpUb, const LocalTensor<float>& predictedUb, const LocalTensor<float>& lossUb,
    int64_t len)
{
    __ubuf__ float* sumAddr = (__ubuf__ float*)sumExpUb.GetPhyAddr();
    __ubuf__ float* predAddr = (__ubuf__ float*)predictedUb.GetPhyAddr();
    __ubuf__ float* lossAddr = (__ubuf__ float*)lossUb.GetPhyAddr();
    // elementsNumber按VL_FP32对齐，满块迭代复用编译期生成的满掩码；尾块写成0/1次迭代的for，避免VF内if分支
    uint16_t fullLoops = static_cast<uint16_t>(len / static_cast<int64_t>(VL_FP32));
    uint16_t totalLoops = static_cast<uint16_t>(ops::CeilDiv(len, static_cast<int64_t>(VL_FP32)));
    uint32_t tailCount = static_cast<uint32_t>(len) - fullLoops * VL_FP32;
    __VEC_SCOPE__
    {
        AscendC::MicroAPI::RegTensor<float> sumReg, predReg, logReg;
        AscendC::MicroAPI::MaskReg
            fullMask = AscendC::MicroAPI::CreateMask<float, AscendC::MicroAPI::MaskPattern::ALL>();
        uint32_t offset = 0;
        for (uint16_t i = 0; i < fullLoops; i++) {
            AscendC::MicroAPI::DataCopy(sumReg, sumAddr + offset);
            AscendC::MicroAPI::DataCopy(predReg, predAddr + offset);
            AscendC::MicroAPI::Log(logReg, sumReg, fullMask);
            AscendC::MicroAPI::Sub(logReg, logReg, predReg, fullMask);
            AscendC::MicroAPI::DataCopy(lossAddr + offset, logReg, fullMask);
            offset += VL_FP32;
        }
        for (uint16_t i = fullLoops; i < totalLoops; i++) {
            uint32_t tail = tailCount;
            AscendC::MicroAPI::MaskReg tailMask = AscendC::MicroAPI::UpdateMask<float>(tail);
            AscendC::MicroAPI::DataCopy(sumReg, sumAddr + offset);
            AscendC::MicroAPI::DataCopy(predReg, predAddr + offset);
            AscendC::MicroAPI::Log(logReg, sumReg, tailMask);
            AscendC::MicroAPI::Sub(logReg, logReg, predReg, tailMask);
            AscendC::MicroAPI::DataCopy(lossAddr + offset, logReg, tailMask);
        }
    }
}

} // namespace FusedCrossEntropyLossWithMaxSumOps
#endif // FUSED_CROSS_ENTROPY_LOSS_WITH_MAX_SUM_AR_H

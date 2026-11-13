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

constexpr static AscendC::Reg::CastTrait castTraitB16ToFp32 = {
    AscendC::Reg::RegLayout::ZERO,
    AscendC::Reg::SatMode::UNKNOWN,
    AscendC::Reg::MaskMergeMode::ZEROING,
    AscendC::RoundMode::UNKNOWN,
};
// log(次正规) / 1÷次正规 需显式走 PRECISION_1ULP_FTZ_FALSE 补偿路径：
// 默认 INTRINSIC 是裸 vln/vdiv，对 fp32 次正规数(<1.175e-38)下溢为 -inf/inf，
// 与 torch golden 的 log(1e-38)=-87.5 不一致（910b 传统 Log 默认已补偿，950 regbase 需显式指定）。
constexpr static AscendC::Reg::LogSpecificMode castLogFt32False = {
    AscendC::Reg::MaskMergeMode::ZEROING,
    AscendC::LogAlgo::PRECISION_1ULP_FTZ_FALSE,
};
constexpr static AscendC::Reg::DivSpecificMode castDivFt32False = {
    AscendC::Reg::MaskMergeMode::ZEROING,
    false,
    AscendC::DivAlgo::PRECISION_1ULP_FTZ_FALSE,
};
// 小V快路的 inv_sum=1/sum_exp 用裸 vdiv：sum_exp 非正常(次正规)时 inv_sum 本就溢出为 inf，
// 补偿与否不影响输出；裸 vdiv 省掉补偿序列，吞吐显著更高
constexpr static AscendC::Reg::DivSpecificMode castDivIntrinsic = {
    AscendC::Reg::MaskMergeMode::ZEROING,
    false,
    AscendC::DivAlgo::INTRINSIC,
};
// 小V快路Gather索引构建用CastTrait（仿softmax_focal_loss_grad_nd_vec.h惯用法）
constexpr static AscendC::Reg::CastTrait castTraitS32ToFp32 = {
    AscendC::Reg::RegLayout::ZERO,
    AscendC::Reg::SatMode::NO_SAT,
    AscendC::Reg::MaskMergeMode::ZEROING,
    AscendC::RoundMode::CAST_NONE,
};
constexpr static AscendC::Reg::CastTrait castTraitFp32ToS32Floor = {
    AscendC::Reg::RegLayout::ZERO,
    AscendC::Reg::SatMode::NO_SAT,
    AscendC::Reg::MaskMergeMode::ZEROING,
    AscendC::RoundMode::CAST_FLOOR,
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
    // 小V快路（vLen<=FUSED_CE_MAX_SUM_SMALL_V_MAX）：多行打包成平坦向量，
    // max/inv按Gather(l/V)展开，vocab与softmax均按整块平坦搬运；
    // 循环按"预取下一块输入→再等本块输出"排布，使MTE与VEC跨tile重叠
    __aicore__ inline void ProcessFullSmallV();
    __aicore__ inline void PrefetchSmallVTile(int64_t tileIdx);
    __aicore__ inline int64_t SmallVTileRows(int64_t tileIdx);
    __aicore__ inline void ComputeLossInvSumLoop(const LocalTensor<float>& sumExpUb,
                                                 const LocalTensor<float>& predictedUb,
                                                 const LocalTensor<float>& lossUb, const LocalTensor<float>& invSumUb,
                                                 int64_t len);
    __aicore__ inline void CastVocabToFp32(const LocalTensor<T>& srcUb, const LocalTensor<float>& dstUb, int64_t len);
    // softmax平坦计算：源为fp32（原样/预转副本）或b16（内联cast），由S区分
    template <typename S>
    __aicore__ inline void SoftmaxFlatLoop(__ubuf__ S* src, const LocalTensor<float>& maxUb,
                                           const LocalTensor<float>& invSumUb, const LocalTensor<float>& outUb,
                                           int64_t aRows, int64_t g);
    __aicore__ inline void ComputeSoftmaxFlat(const LocalTensor<T>& vocabUb, const LocalTensor<float>& maxUb,
                                              const LocalTensor<float>& invSumUb, const LocalTensor<float>& outUb,
                                              int64_t aRows);
    // 省显存路径：仅loss
    __aicore__ inline void ProcessMemory();
    __aicore__ inline void ComputeLossChunk(const LocalTensor<float>& sumExpUb, const LocalTensor<float>& predictedUb,
                                            const LocalTensor<float>& lossUb, int64_t len);
    // 加载len个元素并转fp32：S=float直接DIST_NORM，S=b16则UNPACK_B16+Cast（源地址须32B对齐）
    template <typename S>
    __aicore__ inline void LoadVocabToFp32(__ubuf__ S* src, AscendC::Reg::RegTensor<float>& dst,
                                           AscendC::Reg::MaskReg& preg, uint32_t offset);

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

    // 完整路径buffer（小V快路四标量队列按深度2做跨tile预取流水；大V通路单缓冲，num=1在[1,bufNum]内合法）
    TQue<QuePosition::VECIN, 2> vocabQue_;
    TQue<QuePosition::VECOUT, 2> softmaxQue_;
    TQue<QuePosition::VECIN, 2> logitsMaxQue_;
    TQue<QuePosition::VECIN, 2> sumExpScalarQue_;
    TQue<QuePosition::VECIN, 2> predictedScalarQue_;
    TQue<QuePosition::VECOUT, 2> lossScalarQue_;
    TBuf<> invSumBuf_;
    // 小V快路buffer：vocab的fp32平坦副本（Gather索引在寄存器内经Arange/Cast/Div构建）
    TBuf<> vocabFp32Buf_;
    // 省显存路径buffer
    TQue<QuePosition::VECIN, 2> sumExpQue_;
    TQue<QuePosition::VECIN, 2> predictedQue_;
    TQue<QuePosition::VECOUT, 2> lossQue_;

    int64_t myRows_ = 0;
    int64_t rowStart_ = 0;
    int64_t vPartIdx_ = 0; // v切分时本核在一行内负责的v分片序号（0..vCores-1），不切分恒为0
    bool smallV_ = false;  // vLen<=FUSED_CE_MAX_SUM_SMALL_V_MAX时走小V快路
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
        smallV_ = tl_->vLen > 0 && tl_->vLen <= FUSED_CE_MAX_SUM_SMALL_V_MAX;
        if (smallV_) {
            // tile按VL对齐预留，保证尾块向量访存的越界读/零写在已分配区内；
            // vocab源缓冲额外多留一个VL：V>1时G*v<VL（如V=3组步长48<64），末组满载
            // 向量读会越过tileElems，padding把这次越界读收敛在自有buffer内（掩码位不参与计算）。
            // 该开销已含在host预算的DOUBLE_BUFFER_NUM*V_ALIGN*(dtypeSize+4B)余量中，不挤占aPerLoop；
            // max/sum/pred/loss四个行向量队列深2，支撑跨tile预取流水
            int64_t tileElems = ops::CeilDiv(tl_->aPerLoop * tl_->vLen, static_cast<int64_t>(VL_FP32)) * VL_FP32;
            int64_t rowPad = ops::CeilDiv(tl_->aPerLoop, static_cast<int64_t>(VL_FP32)) * VL_FP32;
            pipe_->InitBuffer(vocabQue_, 2, (tileElems + VL_FP32) * sizeof(T));
            pipe_->InitBuffer(softmaxQue_, 2, tileElems * sizeof(float));
            pipe_->InitBuffer(logitsMaxQue_, 2, rowPad * sizeof(float));
            pipe_->InitBuffer(sumExpScalarQue_, 2, rowPad * sizeof(float));
            pipe_->InitBuffer(predictedScalarQue_, 2, rowPad * sizeof(float));
            pipe_->InitBuffer(lossScalarQue_, 2, rowPad * sizeof(float));
            pipe_->InitBuffer(invSumBuf_, rowPad * sizeof(float));
            pipe_->InitBuffer(vocabFp32Buf_, (tileElems + VL_FP32) * sizeof(float));
        } else {
            pipe_->InitBuffer(vocabQue_, 2, A_PER_LOOP * tl_->vPerLoop * sizeof(T));
            pipe_->InitBuffer(softmaxQue_, 2, A_PER_LOOP * tl_->vPerLoop * sizeof(float));
            pipe_->InitBuffer(logitsMaxQue_, 1, SCALAR_BUF_BYTES);
            pipe_->InitBuffer(sumExpScalarQue_, 1, SCALAR_BUF_BYTES);
            pipe_->InitBuffer(predictedScalarQue_, 1, SCALAR_BUF_BYTES);
            pipe_->InitBuffer(lossScalarQue_, 1, SCALAR_BUF_BYTES);
            pipe_->InitBuffer(invSumBuf_, SCALAR_BUF_BYTES);
        }
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
        if (smallV_) {
            ProcessFullSmallV();
            return;
        }
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

// 小V快路的tile行数（尾块不足aPerLoop）
template <typename T, bool fullPath>
__aicore__ inline int64_t FusedCrossEntropyLossWithMaxSumRegBase<T, fullPath>::SmallVTileRows(int64_t tileIdx)
{
    int64_t rows = myRows_ - tileIdx * tl_->aPerLoop;
    return rows < tl_->aPerLoop ? rows : tl_->aPerLoop;
}

// 下发tileIdx块的全部搬入（3个标量行向量 + vocab整块平坦拷贝），仅入队不等待
template <typename T, bool fullPath>
__aicore__ inline void FusedCrossEntropyLossWithMaxSumRegBase<T, fullPath>::PrefetchSmallVTile(int64_t tileIdx)
{
    int64_t rowBase = rowStart_ + tileIdx * tl_->aPerLoop;
    int64_t aRows = SmallVTileRows(tileIdx);
    LocalTensor<float> maxUb = logitsMaxQue_.AllocTensor<float>();
    LocalTensor<float> sumUb = sumExpScalarQue_.AllocTensor<float>();
    LocalTensor<float> predUb = predictedScalarQue_.AllocTensor<float>();
    CopyInScalar(maxUb, logitsMaxGm_, rowBase, aRows);
    CopyInScalar(sumUb, sumExpGm_, rowBase, aRows);
    CopyInScalar(predUb, predictedGm_, rowBase, aRows);
    logitsMaxQue_.EnQue(maxUb);
    sumExpScalarQue_.EnQue(sumUb);
    predictedScalarQue_.EnQue(predUb);

    LocalTensor<T> vocabUb = vocabQue_.AllocTensor<T>();
    DataCopyExtParams vocabCopyParams{1, static_cast<uint32_t>(aRows * tl_->vLen * sizeof(T)), 0, 0, 0};
    DataCopyPadExtParams<T> vocabPadParams{false, 0, 0, 0};
    DataCopyPad(vocabUb, vocabGm_[rowBase * tl_->vLen], vocabCopyParams, vocabPadParams);
    vocabQue_.EnQue(vocabUb);
}

// 小V快路主循环：先下发本块全部VEC计算、再预取下一块输入（MTE与VEC重叠）、最后等本块输出
template <typename T, bool fullPath>
__aicore__ inline void FusedCrossEntropyLossWithMaxSumRegBase<T, fullPath>::ProcessFullSmallV()
{
    int64_t aLoops = ops::CeilDiv(myRows_, tl_->aPerLoop);
    PrefetchSmallVTile(0);
    for (int64_t i = 0; i < aLoops; i++) {
        int64_t rowBase = rowStart_ + i * tl_->aPerLoop;
        int64_t aRows = SmallVTileRows(i);

        // 1. 本块输入就绪
        LocalTensor<float> maxUb = logitsMaxQue_.DeQue<float>();
        LocalTensor<float> sumUb = sumExpScalarQue_.DeQue<float>();
        LocalTensor<float> predUb = predictedScalarQue_.DeQue<float>();

        // 2. loss = log(sum_exp) - predicted 与 inv_sum = 1 / sum_exp（按VL分块向量化）
        LocalTensor<float> invSumUb = invSumBuf_.Get<float>();
        LocalTensor<float> lossUb = lossScalarQue_.AllocTensor<float>();
        ComputeLossInvSumLoop(sumUb, predUb, lossUb, invSumUb, aRows);
        lossScalarQue_.EnQue(lossUb);

        // 3. softmax = exp(vocab - max) * inv_sum，G行一组打包成G*vLen条lane平坦计算；
        //    b16且组内偏移满足32B对齐时加载即内联cast，否则先整tile预转fp32
        LocalTensor<T> vocabUb = vocabQue_.DeQue<T>();
        LocalTensor<float> outUb = softmaxQue_.AllocTensor<float>();
        ComputeSoftmaxFlat(vocabUb, maxUb, invSumUb, outUb, aRows);
        vocabQue_.FreeTensor(vocabUb);
        softmaxQue_.EnQue(outUb);

        // 5. 本块输入已消费完（VEC侧由事件保证读完才复用），释放后立即预取下一块，
        //    使下一块的MTE搬入与本块仍在执行的VEC计算重叠
        logitsMaxQue_.FreeTensor(maxUb);
        sumExpScalarQue_.FreeTensor(sumUb);
        predictedScalarQue_.FreeTensor(predUb);
        if (i + 1 < aLoops) {
            PrefetchSmallVTile(i + 1);
        }

        // 6. 等本块输出并写回
        LocalTensor<float> lossOutUb = lossScalarQue_.DeQue<float>();
        DataCopyExtParams lossCopyParams{1, static_cast<uint32_t>(aRows * sizeof(float)), 0, 0, 0};
        DataCopyPad(lossGm_[rowBase], lossOutUb, lossCopyParams);
        lossScalarQue_.FreeTensor(lossOutUb);

        LocalTensor<float> outGmUb = softmaxQue_.DeQue<float>();
        DataCopyExtParams outCopyParams{1, static_cast<uint32_t>(aRows * tl_->vLen * sizeof(float)), 0, 0, 0};
        DataCopyPad(softmaxGm_[rowBase * tl_->vLen], outGmUb, outCopyParams);
        softmaxQue_.FreeTensor(outGmUb);
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
        AscendC::Reg::RegTensor<float> sumReg, predReg, logReg, oneReg, invReg;
        AscendC::Reg::MaskReg pMask = AscendC::Reg::UpdateMask<float>(count);
        AscendC::Reg::DataCopy(sumReg, sumAddr);
        AscendC::Reg::DataCopy(predReg, predAddr);
        AscendC::Reg::Log<float, &castLogFt32False>(logReg, sumReg, pMask);
        AscendC::Reg::Sub(logReg, logReg, predReg, pMask);
        AscendC::Reg::DataCopy(lossAddr, logReg, pMask);
        AscendC::Reg::Duplicate(oneReg, 1.0f, pMask);
        AscendC::Reg::Div<float, &castDivFt32False>(invReg, oneReg, sumReg, pMask);
        AscendC::Reg::DataCopy(invAddr, invReg, pMask);
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
template <typename S>
__aicore__ inline void FusedCrossEntropyLossWithMaxSumRegBase<T, fullPath>::LoadVocabToFp32(
    __ubuf__ S* src, AscendC::Reg::RegTensor<float>& dst, AscendC::Reg::MaskReg& preg, uint32_t offset)
{
    if constexpr (IsSameType<S, float>::value) {
        AscendC::Reg::DataCopy<float, AscendC::Reg::LoadDist::DIST_NORM>(dst, src + offset);
    } else {
        AscendC::Reg::RegTensor<S> b16Reg;
        AscendC::Reg::DataCopy<S, AscendC::Reg::LoadDist::DIST_UNPACK_B16>(b16Reg, src + offset);
        AscendC::Reg::Cast<float, S, castTraitB16ToFp32>(dst, b16Reg, preg);
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
        AscendC::Reg::RegTensor<float> maxReg, invReg, xReg, yReg;
        AscendC::Reg::MaskReg fullMask = AscendC::Reg::CreateMask<float, AscendC::Reg::MaskPattern::ALL>();
        for (uint16_t j = 0; j < aRowsU16; j++) {
            // 每行的 max / inv_sum 广播加载进寄存器
            AscendC::Reg::DataCopy<float, AscendC::Reg::LoadDist::DIST_BRC_B32>(maxReg, maxAddr + j);
            AscendC::Reg::DataCopy<float, AscendC::Reg::LoadDist::DIST_BRC_B32>(invReg, invAddr + j);
            uint32_t offset = j * rowStride;
            for (uint16_t i = 0; i < fullLoops; i++) {
                LoadVocabToFp32(vocabAddr, xReg, fullMask, offset);
                AscendC::Reg::Sub(yReg, xReg, maxReg, fullMask);
                AscendC::Reg::Exp(yReg, yReg, fullMask);
                AscendC::Reg::Mul(yReg, yReg, invReg, fullMask);
                AscendC::Reg::DataCopy(outAddr + offset, yReg, fullMask);
                offset += VL_FP32;
            }
            for (uint16_t i = fullLoops; i < totalLoops; i++) {
                uint32_t tail = tailCount;
                AscendC::Reg::MaskReg tailMask = AscendC::Reg::UpdateMask<float>(tail);
                LoadVocabToFp32(vocabAddr, xReg, tailMask, offset);
                AscendC::Reg::Sub(yReg, xReg, maxReg, tailMask);
                AscendC::Reg::Exp(yReg, yReg, tailMask);
                AscendC::Reg::Mul(yReg, yReg, invReg, tailMask);
                AscendC::Reg::DataCopy(outAddr + offset, yReg, tailMask);
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
__aicore__ inline void FusedCrossEntropyLossWithMaxSumRegBase<T, fullPath>::ComputeLossInvSumLoop(
    const LocalTensor<float>& sumExpUb, const LocalTensor<float>& predictedUb, const LocalTensor<float>& lossUb,
    const LocalTensor<float>& invSumUb, int64_t len)
{
    __ubuf__ float* sumAddr = (__ubuf__ float*)sumExpUb.GetPhyAddr();
    __ubuf__ float* predAddr = (__ubuf__ float*)predictedUb.GetPhyAddr();
    __ubuf__ float* lossAddr = (__ubuf__ float*)lossUb.GetPhyAddr();
    __ubuf__ float* invAddr = (__ubuf__ float*)invSumUb.GetPhyAddr();
    // 与ComputeLossChunk同构：满块迭代复用编译期满掩码，尾块单独0/1次迭代
    uint16_t fullLoops = static_cast<uint16_t>(len / static_cast<int64_t>(VL_FP32));
    uint16_t totalLoops = static_cast<uint16_t>(ops::CeilDiv(len, static_cast<int64_t>(VL_FP32)));
    uint32_t tailCount = static_cast<uint32_t>(len) - fullLoops * VL_FP32;
    __VEC_SCOPE__
    {
        AscendC::Reg::RegTensor<float> sumReg, predReg, logReg, oneReg, invReg;
        AscendC::Reg::MaskReg fullMask = AscendC::Reg::CreateMask<float, AscendC::Reg::MaskPattern::ALL>();
        uint32_t offset = 0;
        // oneReg为常量1向量，与循环无关，外提避免每轮重复广播
        AscendC::Reg::Duplicate(oneReg, 1.0f, fullMask);
        for (uint16_t i = 0; i < fullLoops; i++) {
            AscendC::Reg::DataCopy(sumReg, sumAddr + offset);
            AscendC::Reg::DataCopy(predReg, predAddr + offset);
            AscendC::Reg::Log<float, &castLogFt32False>(logReg, sumReg, fullMask);
            AscendC::Reg::Sub(logReg, logReg, predReg, fullMask);
            AscendC::Reg::DataCopy(lossAddr + offset, logReg, fullMask);
            AscendC::Reg::Div<float, &castDivIntrinsic>(invReg, oneReg, sumReg, fullMask);
            AscendC::Reg::DataCopy(invAddr + offset, invReg, fullMask);
            offset += VL_FP32;
        }
        for (uint16_t i = fullLoops; i < totalLoops; i++) {
            uint32_t tail = tailCount;
            AscendC::Reg::MaskReg tailMask = AscendC::Reg::UpdateMask<float>(tail);
            AscendC::Reg::DataCopy(sumReg, sumAddr + offset);
            AscendC::Reg::DataCopy(predReg, predAddr + offset);
            AscendC::Reg::Log<float, &castLogFt32False>(logReg, sumReg, tailMask);
            AscendC::Reg::Sub(logReg, logReg, predReg, tailMask);
            AscendC::Reg::DataCopy(lossAddr + offset, logReg, tailMask);
            AscendC::Reg::Div<float, &castDivIntrinsic>(invReg, oneReg, sumReg, tailMask);
            AscendC::Reg::DataCopy(invAddr + offset, invReg, tailMask);
        }
    }
}

template <typename T, bool fullPath>
__aicore__ inline void FusedCrossEntropyLossWithMaxSumRegBase<T, fullPath>::CastVocabToFp32(
    const LocalTensor<T>& srcUb, const LocalTensor<float>& dstUb, int64_t len)
{
    __ubuf__ T* srcAddr = (__ubuf__ T*)srcUb.GetPhyAddr();
    __ubuf__ float* dstAddr = (__ubuf__ float*)dstUb.GetPhyAddr();
    uint16_t fullLoops = static_cast<uint16_t>(len / static_cast<int64_t>(VL_FP32));
    uint16_t totalLoops = static_cast<uint16_t>(ops::CeilDiv(len, static_cast<int64_t>(VL_FP32)));
    uint32_t tailCount = static_cast<uint32_t>(len) - fullLoops * VL_FP32;
    __VEC_SCOPE__
    {
        AscendC::Reg::RegTensor<float> fp32Reg;
        AscendC::Reg::MaskReg fullMask = AscendC::Reg::CreateMask<float, AscendC::Reg::MaskPattern::ALL>();
        uint32_t offset = 0;
        for (uint16_t i = 0; i < fullLoops; i++) {
            LoadVocabToFp32(srcAddr, fp32Reg, fullMask, offset);
            AscendC::Reg::DataCopy(dstAddr + offset, fp32Reg, fullMask);
            offset += VL_FP32;
        }
        for (uint16_t i = fullLoops; i < totalLoops; i++) {
            uint32_t tail = tailCount;
            AscendC::Reg::MaskReg tailMask = AscendC::Reg::UpdateMask<float>(tail);
            LoadVocabToFp32(srcAddr, fp32Reg, tailMask, offset);
            AscendC::Reg::DataCopy(dstAddr + offset, fp32Reg, tailMask);
        }
    }
}

template <typename T, bool fullPath>
__aicore__ inline void FusedCrossEntropyLossWithMaxSumRegBase<T, fullPath>::ComputeSoftmaxFlat(
    const LocalTensor<T>& vocabUb, const LocalTensor<float>& maxUb, const LocalTensor<float>& invSumUb,
    const LocalTensor<float>& outUb, int64_t aRows)
{
    const int64_t v = tl_->vLen;
    int64_t g = VL_FP32 / v;
    while ((g * v) % (UB_BLOCK_SIZE / sizeof(float)) != 0) {
        g--;
    }
    if constexpr (IsSameType<T, float>::value) {
        SoftmaxFlatLoop<float>((__ubuf__ float*)vocabUb.GetPhyAddr(), maxUb, invSumUb, outUb, aRows, g);
    } else if ((g * v) % (UB_BLOCK_SIZE / sizeof(T)) == 0) {
        // b16组内偏移 r0*v*2B 满足32B对齐（V=5/7不满足）：加载时内联cast，
        // 省掉整tile预转fp32的一趟UB读写，数值与预转bit级一致（同一条Cast指令）
        SoftmaxFlatLoop<T>((__ubuf__ T*)vocabUb.GetPhyAddr(), maxUb, invSumUb, outUb, aRows, g);
    } else {
        LocalTensor<float> vocabFp32Ub = vocabFp32Buf_.Get<float>();
        CastVocabToFp32(vocabUb, vocabFp32Ub, aRows * v);
        SoftmaxFlatLoop<float>((__ubuf__ float*)vocabFp32Ub.GetPhyAddr(), maxUb, invSumUb, outUb, aRows, g);
    }
}

template <typename T, bool fullPath>
template <typename S>
__aicore__ inline void FusedCrossEntropyLossWithMaxSumRegBase<T, fullPath>::SoftmaxFlatLoop(
    __ubuf__ S* src, const LocalTensor<float>& maxUb, const LocalTensor<float>& invSumUb,
    const LocalTensor<float>& outUb, int64_t aRows, int64_t g)
{
    __ubuf__ float* maxAddr = (__ubuf__ float*)maxUb.GetPhyAddr();
    __ubuf__ float* invAddr = (__ubuf__ float*)invSumUb.GetPhyAddr();
    __ubuf__ float* outAddr = (__ubuf__ float*)outUb.GetPhyAddr();
    const int64_t v = tl_->vLen;
    // g由调用方算好传入：每组G行打包成G*v条lane，组内偏移r0*v*4B须保持32B对齐（G*v为8的倍数），
    // 且G*v<=VL；G>=8保证maxAddr+r0（4B*r0）同样32B对齐
    uint16_t gU16 = static_cast<uint16_t>(g);
    uint16_t aRowsU16 = static_cast<uint16_t>(aRows);
    // VF内不写运行时if分支（见ComputeSoftmaxTile处注释）：V=1与V>1分两个独立的VEC域
    if (v == 1) {
        __VEC_SCOPE__
        {
            AscendC::Reg::RegTensor<float> maxReg, invReg, xReg, yReg;
            for (uint16_t r0 = 0; r0 < aRowsU16; r0 += gU16) {
                uint16_t rows = (aRowsU16 - r0) < gU16 ? (aRowsU16 - r0) : gU16;
                uint32_t lanes = static_cast<uint32_t>(rows);
                AscendC::Reg::MaskReg mask = AscendC::Reg::UpdateMask<float>(lanes);
                // V=1无需展开：max/inv与vocab逐元素对位，直接连续加载
                AscendC::Reg::DataCopy(maxReg, maxAddr + r0);
                AscendC::Reg::DataCopy(invReg, invAddr + r0);
                // 满载VL（UB tile按VL对齐预留，越界读在已分配区内），尾块由compute/store掩码收敛
                LoadVocabToFp32(src, xReg, mask, r0);
                AscendC::Reg::Sub(yReg, xReg, maxReg, mask);
                AscendC::Reg::Exp(yReg, yReg, mask);
                AscendC::Reg::Mul(yReg, yReg, invReg, mask);
                AscendC::Reg::DataCopy(outAddr + r0, yReg, mask);
            }
        }
    } else {
        __VEC_SCOPE__
        {
            AscendC::Reg::RegTensor<float> maxReg, invReg, xReg, yReg;
            AscendC::Reg::RegTensor<int32_t> idxReg;
            AscendC::Reg::MaskReg fullMask = AscendC::Reg::CreateMask<float, AscendC::Reg::MaskPattern::ALL>();
            // 索引在寄存器内构建：idx[l] = floor(l / v)。整型 Reg::Div 无截断语义保证（全仓无
            // 先例），按 softmax_focal_loss_grad 惯用法走 fp32 真除 + CAST_FLOOR：l < VL 且
            // v <= 8 时 l/v 与最近整数的距离 >= 1/8，远超 0.5ulp，floor 结果必精确
            AscendC::Reg::RegTensor<float> idxFReg;
            AscendC::Reg::RegTensor<float> vFReg;
            AscendC::Reg::Arange(idxReg, 0);
            AscendC::Reg::Cast<float, int32_t, castTraitS32ToFp32>(idxFReg, idxReg, fullMask);
            AscendC::Reg::Duplicate(vFReg, static_cast<float>(v), fullMask);
            AscendC::Reg::Div(idxFReg, idxFReg, vFReg, fullMask);
            AscendC::Reg::Cast<int32_t, float, castTraitFp32ToS32Floor>(idxReg, idxFReg, fullMask);
            for (uint16_t r0 = 0; r0 < aRowsU16; r0 += gU16) {
                uint16_t rows = (aRowsU16 - r0) < gU16 ? (aRowsU16 - r0) : gU16;
                uint32_t lanes = static_cast<uint32_t>(rows) * static_cast<uint32_t>(v);
                AscendC::Reg::MaskReg mask = AscendC::Reg::UpdateMask<float>(lanes);
                uint32_t off = static_cast<uint32_t>(r0) * static_cast<uint32_t>(v);
                // V>1用Gather按idx[l]=l/v把每行的max/inv展开到G*v条lane
                AscendC::Reg::Gather<float, float, uint32_t>(maxReg, maxAddr + r0,
                                                             (AscendC::Reg::RegTensor<uint32_t>&)idxReg, mask);
                AscendC::Reg::Gather<float, float, uint32_t>(invReg, invAddr + r0,
                                                             (AscendC::Reg::RegTensor<uint32_t>&)idxReg, mask);
                LoadVocabToFp32(src, xReg, mask, off);
                AscendC::Reg::Sub(yReg, xReg, maxReg, mask);
                AscendC::Reg::Exp(yReg, yReg, mask);
                AscendC::Reg::Mul(yReg, yReg, invReg, mask);
                AscendC::Reg::DataCopy(outAddr + off, yReg, mask);
            }
        }
    }
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
        AscendC::Reg::RegTensor<float> sumReg, predReg, logReg;
        AscendC::Reg::MaskReg fullMask = AscendC::Reg::CreateMask<float, AscendC::Reg::MaskPattern::ALL>();
        uint32_t offset = 0;
        for (uint16_t i = 0; i < fullLoops; i++) {
            AscendC::Reg::DataCopy(sumReg, sumAddr + offset);
            AscendC::Reg::DataCopy(predReg, predAddr + offset);
            AscendC::Reg::Log<float, &castLogFt32False>(logReg, sumReg, fullMask);
            AscendC::Reg::Sub(logReg, logReg, predReg, fullMask);
            AscendC::Reg::DataCopy(lossAddr + offset, logReg, fullMask);
            offset += VL_FP32;
        }
        for (uint16_t i = fullLoops; i < totalLoops; i++) {
            uint32_t tail = tailCount;
            AscendC::Reg::MaskReg tailMask = AscendC::Reg::UpdateMask<float>(tail);
            AscendC::Reg::DataCopy(sumReg, sumAddr + offset);
            AscendC::Reg::DataCopy(predReg, predAddr + offset);
            AscendC::Reg::Log<float, &castLogFt32False>(logReg, sumReg, tailMask);
            AscendC::Reg::Sub(logReg, logReg, predReg, tailMask);
            AscendC::Reg::DataCopy(lossAddr + offset, logReg, tailMask);
        }
    }
}

} // namespace FusedCrossEntropyLossWithMaxSumOps
#endif // FUSED_CROSS_ENTROPY_LOSS_WITH_MAX_SUM_AR_H

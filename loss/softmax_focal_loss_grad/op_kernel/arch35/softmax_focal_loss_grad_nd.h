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
 * \file softmax_focal_loss_grad_nd.h
 * \brief softmax_focal_loss_grad regbase kernel
 *
 * 计算语义(对齐 A2 tbe.dsl softmax_focal_loss_grad_compute):
 *   wf = alpha * exp(gamma       * log(1-p)) * t     WF = sum_j wf
 *   wb = alpha * exp((gamma-1)   * log(1-p)) * t     WB = sum_j wb
 *   ce = -log(p) * t * w                             CE = sum_j ce
 *   wt = w * t                                       W  = sum_j wt
 *   d_ce = p * W - t * w
 *   d_wf = -gamma * ((WF - WB) + wb) * p
 *   grad = (d_ce * WF + d_wf * CE) * dout * coef     coef: mean 为 1/numel, 否则 1.0
 *
 * 调度: 行分核, 核内按行块推进, 行内按列块推进(列块数为 1 时即全载路径)。
 * 两趟: 第一趟求四个行标量, 第二趟逐元素组合出 grad(wb 在第二趟重算, 不占额外缓冲)。
 */

#ifndef SOFTMAX_FOCAL_LOSS_GRAD_ND_H
#define SOFTMAX_FOCAL_LOSS_GRAD_ND_H

#include "kernel_tiling/kernel_tiling.h"
#include "kernel_operator.h"
#include "../inc/platform.h"
#include "softmax_focal_loss_grad_tiling_data.h"

namespace SoftmaxFocalLossGrad {
using namespace AscendC;
using AscendC::MicroAPI::LoadDist;
using AscendC::MicroAPI::MaskPattern;
using AscendC::MicroAPI::MaskReg;
using AscendC::MicroAPI::RegTensor;
using AscendC::MicroAPI::UpdateMask;

template <typename T, typename TW, uint64_t hasWeight>
class SoftmaxFocalLossGradND {
public:
    __aicore__ inline SoftmaxFocalLossGradND() {}

    __aicore__ inline void Init(GM_ADDR pred, GM_ADDR target, GM_ADDR dout, GM_ADDR weight, GM_ADDR grad,
                                const SoftmaxFocalLossGradArch35TilingData* tiling, TPipe* pipe);
    __aicore__ inline void Process();

private:
    __aicore__ inline void ProcessTile(int64_t rowGm, int64_t rows);
    __aicore__ inline void CopyIn(int64_t rowGm, int64_t colOff, int64_t rows, int64_t len);
    // 搬运参数按 dtype 组装(blockLen/srcStride 走各自 sizeof, rightPadding 走各自行对齐)
    template <typename TE>
    __aicore__ inline void MakeCopyParams(DataCopyExtParams& params, DataCopyPadExtParams<TE>& pad, int64_t rows,
                                          int64_t len, int64_t stride, TE padValue);
    __aicore__ inline void ComputeSums(int64_t rows, int64_t len);
    // VF 域内复用: 取一段 pred/target/weight 到 fp32 寄存器 / 由它们算出四路行和的逐元素项
    __aicore__ inline void LoadPTW(RegTensor<float>& p32, RegTensor<float>& t32, RegTensor<float>& w32,
                                   __ubuf__ T* predAddr, __ubuf__ int32_t* targetAddr, __ubuf__ TW* weightAddr,
                                   AscendC::MicroAPI::AddrReg offT, AscendC::MicroAPI::AddrReg offW,
                                   AscendC::MicroAPI::AddrReg offF, MaskReg preg);
    __aicore__ inline void SumsOfSeg(__ubuf__ float* wfAddr, __ubuf__ float* wbAddr, __ubuf__ float* ceAddr,
                                     __ubuf__ float* wtAddr, RegTensor<float>& p32, RegTensor<float>& t32,
                                     RegTensor<float>& w32, AscendC::MicroAPI::AddrReg offF, MaskReg preg);
    __aicore__ inline void SumsVec(__ubuf__ T* predAddr, __ubuf__ int32_t* targetAddr, __ubuf__ TW* weightAddr,
                                   __ubuf__ float* wfAddr, __ubuf__ float* wbAddr, __ubuf__ float* ceAddr,
                                   __ubuf__ float* wtAddr, int64_t rows, int64_t len);
    __aicore__ inline void ComputeGrad(int64_t rows, int64_t len);
    __aicore__ inline void GradOfSeg(__ubuf__ float* gradAddr, RegTensor<float>& p32, RegTensor<float>& t32,
                                     RegTensor<float>& w32, RegTensor<float>& d32, RegTensor<float>& wfB,
                                     RegTensor<float>& wbB, RegTensor<float>& ceB, RegTensor<float>& wB,
                                     AscendC::MicroAPI::AddrReg offO, MaskReg preg);
    __aicore__ inline void MakeSegOffsets(AscendC::MicroAPI::AddrReg& offT, AscendC::MicroAPI::AddrReg& offW,
                                          AscendC::MicroAPI::AddrReg& offF, AscendC::MicroAPI::AddrReg& offO,
                                          uint16_t i, uint16_t j, int64_t strideT, int64_t strideW, int64_t strideF,
                                          uint32_t vfLen);
    __aicore__ inline void LoadRowScalars(RegTensor<float>& wfB, RegTensor<float>& wbB, RegTensor<float>& ceB,
                                          RegTensor<float>& wB, __ubuf__ float* accAddr, int64_t accStride,
                                          uint16_t rowIdx);
    __aicore__ inline void GradVec(__ubuf__ T* predAddr, __ubuf__ int32_t* targetAddr, __ubuf__ T* doutAddr,
                                   __ubuf__ TW* weightAddr, __ubuf__ float* accAddr, __ubuf__ float* gradAddr,
                                   int64_t rows, int64_t len);
    __aicore__ inline void CopyOut(int64_t rowGm, int64_t colOff, int64_t rows, int64_t len);

    __aicore__ inline int64_t AlignUp(int64_t x, int64_t base) const { return (x + base - 1) / base * base; }
    __aicore__ inline int64_t StrideOfT(int64_t len) const
    {
        return AlignUp(len, static_cast<int64_t>(BLOCK_SIZE / sizeof(T)));
    }
    __aicore__ inline int64_t StrideOfW(int64_t len) const
    {
        return AlignUp(len, static_cast<int64_t>(BLOCK_SIZE / sizeof(TW)));
    }
    __aicore__ inline int64_t StrideOfF32(int64_t len) const
    {
        return AlignUp(len, static_cast<int64_t>(BLOCK_SIZE / sizeof(float)));
    }

protected:
    constexpr static AscendC::MicroAPI::CastTrait castB16ToB32 = {
        AscendC::MicroAPI::RegLayout::ZERO, AscendC::MicroAPI::SatMode::UNKNOWN,
        AscendC::MicroAPI::MaskMergeMode::ZEROING, AscendC::RoundMode::UNKNOWN};
    constexpr static AscendC::MicroAPI::CastTrait castI32ToF32 = {
        AscendC::MicroAPI::RegLayout::ZERO, AscendC::MicroAPI::SatMode::SAT, AscendC::MicroAPI::MaskMergeMode::ZEROING,
        AscendC::RoundMode::CAST_RINT};

    constexpr static uint32_t BLOCK_SIZE = platform::GetUbBlockSize();
    constexpr static uint32_t VL_FP32 = platform::GetVRegSize() / sizeof(float);

    // 见前向同名常量: pad 值取 (0,1) 内的 0.5 使 padding 位上四路和均为 0(求和单位元)
    constexpr static float PRED_PAD_VALUE = 0.5f;

private:
    GlobalTensor<T> predGm_;
    GlobalTensor<int32_t> targetGm_;
    GlobalTensor<T> doutGm_;
    GlobalTensor<TW> weightGm_;
    GlobalTensor<T> gradGm_;

    TPipe* pipe_ = nullptr;
    TQue<QuePosition::VECIN, 1> predQue_;
    TQue<QuePosition::VECIN, 1> targetQue_;
    TQue<QuePosition::VECIN, 1> doutQue_;
    TQue<QuePosition::VECIN, 1> weightQue_;
    TQue<QuePosition::VECOUT, 1> gradQue_;

    TBuf<QuePosition::VECCALC> wfBuf_; // 第二趟复用为 grad 的 fp32 暂存
    TBuf<QuePosition::VECCALC> wbBuf_;
    TBuf<QuePosition::VECCALC> ceBuf_;
    TBuf<QuePosition::VECCALC> wtBuf_;
    TBuf<QuePosition::VECCALC> accBuf_;

    int64_t a_ = 0;
    int64_t r_ = 0;
    int64_t realCoreNum_ = 0;
    int64_t blockFactor_ = 0;
    int64_t tailBlockFactor_ = 0;
    int64_t rowsPerTile_ = 0;
    int64_t colsPerChunk_ = 0;
    int64_t chunkNum_ = 0;
    int64_t accStride_ = 0;
    float gamma_ = 0.0f;
    float alpha_ = 0.0f;
    float coef_ = 1.0f;
};

template <typename T, typename TW, uint64_t hasWeight>
__aicore__ inline void SoftmaxFocalLossGradND<T, TW, hasWeight>::Init(
    GM_ADDR pred, GM_ADDR target, GM_ADDR dout, GM_ADDR weight, GM_ADDR grad,
    const SoftmaxFocalLossGradArch35TilingData* tiling, TPipe* pipe)
{
    a_ = tiling->a;
    r_ = tiling->r;
    realCoreNum_ = tiling->realCoreNum;
    blockFactor_ = tiling->blockFactor;
    tailBlockFactor_ = tiling->tailBlockFactor;
    rowsPerTile_ = tiling->rowsPerTile;
    colsPerChunk_ = tiling->colsPerChunk;
    chunkNum_ = tiling->chunkNum;
    gamma_ = tiling->gamma;
    alpha_ = tiling->alpha;
    coef_ = tiling->reductionCoef;
    pipe_ = pipe;

    predGm_.SetGlobalBuffer((__gm__ T*)pred);
    targetGm_.SetGlobalBuffer((__gm__ int32_t*)target);
    doutGm_.SetGlobalBuffer((__gm__ T*)dout);
    gradGm_.SetGlobalBuffer((__gm__ T*)grad);
    if constexpr (hasWeight == 1) {
        weightGm_.SetGlobalBuffer((__gm__ TW*)weight);
    }

    int64_t maxElem = rowsPerTile_ * colsPerChunk_ + static_cast<int64_t>(VL_FP32);
    accStride_ = AlignUp(rowsPerTile_, static_cast<int64_t>(BLOCK_SIZE / sizeof(float)));

    pipe_->InitBuffer(predQue_, 1, maxElem * sizeof(T));
    pipe_->InitBuffer(targetQue_, 1, maxElem * sizeof(int32_t));
    pipe_->InitBuffer(doutQue_, 1, maxElem * sizeof(T));
    if constexpr (hasWeight == 1) {
        pipe_->InitBuffer(weightQue_, 1, maxElem * sizeof(TW));
    }
    pipe_->InitBuffer(gradQue_, 1, maxElem * sizeof(T));
    pipe_->InitBuffer(wfBuf_, maxElem * sizeof(float));
    pipe_->InitBuffer(wbBuf_, maxElem * sizeof(float));
    pipe_->InitBuffer(ceBuf_, maxElem * sizeof(float));
    pipe_->InitBuffer(wtBuf_, maxElem * sizeof(float));
    // wfAcc | wbAcc | ceAcc | wtAcc | redTmp
    pipe_->InitBuffer(accBuf_, 5 * accStride_ * sizeof(float));
}

template <typename T, typename TW, uint64_t hasWeight>
__aicore__ inline void SoftmaxFocalLossGradND<T, TW, hasWeight>::Process()
{
    int64_t blockIdx = static_cast<int64_t>(GetBlockIdx());
    if (blockIdx >= realCoreNum_) {
        return;
    }
    int64_t rowStart = blockIdx * blockFactor_;
    int64_t rowNum = (blockIdx == realCoreNum_ - 1) ? tailBlockFactor_ : blockFactor_;

    for (int64_t off = 0; off < rowNum; off += rowsPerTile_) {
        int64_t rows = rowNum - off;
        rows = rows > rowsPerTile_ ? rowsPerTile_ : rows;
        ProcessTile(rowStart + off, rows);
    }
}

template <typename T, typename TW, uint64_t hasWeight>
__aicore__ inline void SoftmaxFocalLossGradND<T, TW, hasWeight>::ProcessTile(int64_t rowGm, int64_t rows)
{
    LocalTensor<float> accBuf = accBuf_.template Get<float>();
    LocalTensor<float> wfAcc = accBuf[0];
    LocalTensor<float> wbAcc = accBuf[accStride_];
    LocalTensor<float> ceAcc = accBuf[2 * accStride_];
    LocalTensor<float> wtAcc = accBuf[3 * accStride_];
    LocalTensor<float> redTmp = accBuf[4 * accStride_];
    int32_t rowCnt = static_cast<int32_t>(rows);

    AscendC::Duplicate(wfAcc, 0.0f, rowCnt);
    AscendC::Duplicate(wbAcc, 0.0f, rowCnt);
    AscendC::Duplicate(ceAcc, 0.0f, rowCnt);
    AscendC::Duplicate(wtAcc, 0.0f, rowCnt);

    LocalTensor<float> wfBuf = wfBuf_.template Get<float>();
    LocalTensor<float> wbBuf = wbBuf_.template Get<float>();
    LocalTensor<float> ceBuf = ceBuf_.template Get<float>();
    LocalTensor<float> wtBuf = wtBuf_.template Get<float>();

    // 第一趟: 求 WF / WB / CE / W 四个行标量
    for (int64_t c = 0; c < chunkNum_; ++c) {
        int64_t colOff = c * colsPerChunk_;
        int64_t len = r_ - colOff;
        len = len > colsPerChunk_ ? colsPerChunk_ : len;

        CopyIn(rowGm, colOff, rows, len);
        ComputeSums(rows, len);

        uint32_t srcShape[2] = {static_cast<uint32_t>(rows), static_cast<uint32_t>(StrideOfF32(len))};
        AscendC::ReduceSum<float, AscendC::Pattern::Reduce::AR, true>(redTmp, wfBuf, srcShape, false);
        AscendC::Add(wfAcc, wfAcc, redTmp, rowCnt);
        AscendC::ReduceSum<float, AscendC::Pattern::Reduce::AR, true>(redTmp, wbBuf, srcShape, false);
        AscendC::Add(wbAcc, wbAcc, redTmp, rowCnt);
        AscendC::ReduceSum<float, AscendC::Pattern::Reduce::AR, true>(redTmp, ceBuf, srcShape, false);
        AscendC::Add(ceAcc, ceAcc, redTmp, rowCnt);
        AscendC::ReduceSum<float, AscendC::Pattern::Reduce::AR, true>(redTmp, wtBuf, srcShape, false);
        AscendC::Add(wtAcc, wtAcc, redTmp, rowCnt);
    }

    // 第二趟: 逐元素组合出 grad
    for (int64_t c = 0; c < chunkNum_; ++c) {
        int64_t colOff = c * colsPerChunk_;
        int64_t len = r_ - colOff;
        len = len > colsPerChunk_ ? colsPerChunk_ : len;

        CopyIn(rowGm, colOff, rows, len);
        ComputeGrad(rows, len);
        CopyOut(rowGm, colOff, rows, len);
    }
}

template <typename T, typename TW, uint64_t hasWeight>
template <typename TE>
__aicore__ inline void SoftmaxFocalLossGradND<T, TW, hasWeight>::MakeCopyParams(DataCopyExtParams& params,
                                                                                DataCopyPadExtParams<TE>& pad,
                                                                                int64_t rows, int64_t len,
                                                                                int64_t stride, TE padValue)
{
    params.blockCount = static_cast<uint16_t>(rows);
    params.blockLen = static_cast<uint32_t>(len * sizeof(TE));
    params.srcStride = static_cast<uint32_t>((r_ - len) * sizeof(TE));
    params.dstStride = 0;
    pad.isPad = true;
    pad.leftPadding = 0;
    pad.rightPadding = static_cast<uint8_t>(stride - len);
    pad.paddingValue = padValue;
}

template <typename T, typename TW, uint64_t hasWeight>
__aicore__ inline void SoftmaxFocalLossGradND<T, TW, hasWeight>::CopyIn(int64_t rowGm, int64_t colOff, int64_t rows,
                                                                        int64_t len)
{
    int64_t gmOffset = rowGm * r_ + colOff;

    LocalTensor<T> predBuf = predQue_.template AllocTensor<T>();
    DataCopyExtParams tParams;
    DataCopyPadExtParams<T> predPad;
    MakeCopyParams<T>(tParams, predPad, rows, len, StrideOfT(len), static_cast<T>(PRED_PAD_VALUE));
    AscendC::DataCopyPad(predBuf, predGm_[gmOffset], tParams, predPad);
    predQue_.template EnQue<T>(predBuf);

    LocalTensor<int32_t> targetBuf = targetQue_.template AllocTensor<int32_t>();
    DataCopyExtParams tgtParams;
    DataCopyPadExtParams<int32_t> tgtPad;
    MakeCopyParams<int32_t>(tgtParams, tgtPad, rows, len, StrideOfF32(len), 0);
    AscendC::DataCopyPad(targetBuf, targetGm_[gmOffset], tgtParams, tgtPad);
    targetQue_.template EnQue<int32_t>(targetBuf);

    LocalTensor<T> doutBuf = doutQue_.template AllocTensor<T>();
    DataCopyExtParams dParams;
    DataCopyPadExtParams<T> doutPad;
    MakeCopyParams<T>(dParams, doutPad, rows, len, StrideOfT(len), static_cast<T>(0));
    AscendC::DataCopyPad(doutBuf, doutGm_[gmOffset], dParams, doutPad);
    doutQue_.template EnQue<T>(doutBuf);

    if constexpr (hasWeight == 1) {
        LocalTensor<TW> weightBuf = weightQue_.template AllocTensor<TW>();
        DataCopyExtParams wParams;
        DataCopyPadExtParams<TW> wPad;
        MakeCopyParams<TW>(wParams, wPad, rows, len, StrideOfW(len), static_cast<TW>(0));
        AscendC::DataCopyPad(weightBuf, weightGm_[gmOffset], wParams, wPad);
        weightQue_.template EnQue<TW>(weightBuf);
    }
}

template <typename T, typename TW, uint64_t hasWeight>
__aicore__ inline void SoftmaxFocalLossGradND<T, TW, hasWeight>::CopyOut(int64_t rowGm, int64_t colOff, int64_t rows,
                                                                         int64_t len)
{
    LocalTensor<T> outBuf = gradQue_.template DeQue<T>();
    DataCopyExtParams outParams;
    outParams.blockCount = static_cast<uint16_t>(rows);
    outParams.blockLen = static_cast<uint32_t>(len * sizeof(T));
    outParams.srcStride = 0;
    outParams.dstStride = static_cast<uint32_t>((r_ - len) * sizeof(T));
    AscendC::DataCopyPad(gradGm_[rowGm * r_ + colOff], outBuf, outParams);
    gradQue_.FreeTensor(outBuf);
}

} // namespace SoftmaxFocalLossGrad

#endif // SOFTMAX_FOCAL_LOSS_GRAD_ND_H

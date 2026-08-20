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
 * \file softmax_focal_loss_nd.h
 * \brief softmax_focal_loss regbase kernel
 *
 * 计算语义(对齐 A2 tbe.dsl softmax_focal_loss_compute):
 *   ce[b][j] = -log(pred) * target * weight
 *   CE[b]    = sum_j ce[b][j]
 *   fw[b][j] = alpha * exp(gamma * log(1 - pred)) * target
 *   FW[b]    = sum_j fw[b][j]
 *   y[b][j]  = CE[b] * FW[b]          // 整行同值
 *
 * 调度: 行分核, 核内按行块推进, 行内按列块推进(列块数为 1 时即全载路径)。
 */

#ifndef SOFTMAX_FOCAL_LOSS_ND_H
#define SOFTMAX_FOCAL_LOSS_ND_H

#include "kernel_tiling/kernel_tiling.h"
#include "kernel_operator.h"
#include "../inc/platform.h"
#include "softmax_focal_loss_tiling_data.h"

namespace SoftmaxFocalLoss {
using namespace AscendC;
using AscendC::MicroAPI::LoadDist;
using AscendC::MicroAPI::MaskPattern;
using AscendC::MicroAPI::MaskReg;
using AscendC::MicroAPI::RegTensor;
using AscendC::MicroAPI::UpdateMask;

template <typename T, typename TW, uint64_t hasWeight>
class SoftmaxFocalLossND {
public:
    __aicore__ inline SoftmaxFocalLossND() {}

    __aicore__ inline void Init(GM_ADDR pred, GM_ADDR target, GM_ADDR weight, GM_ADDR y,
                                const SoftmaxFocalLossArch35TilingData* tiling, TPipe* pipe);
    __aicore__ inline void Process();

private:
    __aicore__ inline void ProcessTile(int64_t rowGm, int64_t rows);
    __aicore__ inline void CopyIn(int64_t rowGm, int64_t colOff, int64_t rows, int64_t len);
    __aicore__ inline void ComputeCeFw(int64_t rows, int64_t len);
    __aicore__ inline void CeFwTailSeg(__ubuf__ T* predAddr, __ubuf__ int32_t* targetAddr, __ubuf__ TW* weightAddr,
                                       __ubuf__ float* ceAddr, __ubuf__ float* fwAddr, int64_t rowIdx, int64_t strideT,
                                       int64_t strideW, int64_t strideF, uint32_t doneCols, MaskReg preg);
    __aicore__ inline void CeFwVec(__ubuf__ T* predAddr, __ubuf__ int32_t* targetAddr, __ubuf__ TW* weightAddr,
                                   __ubuf__ float* ceAddr, __ubuf__ float* fwAddr, int64_t rows, int64_t len);
    __aicore__ inline void CopyOutRow(int64_t rowGm, int64_t colOff, int64_t rows, int64_t len);

    // VF 域内复用: 取一段 pred/target/weight 到 fp32 寄存器 / 由它们算出 ce、fw
    template <bool byAddrReg>
    __aicore__ inline void LoadPredTargetWeight(RegTensor<float>& p32, RegTensor<float>& t32, RegTensor<float>& w32,
                                                __ubuf__ T* predPtr, __ubuf__ int32_t* targetPtr,
                                                __ubuf__ TW* weightPtr, AscendC::MicroAPI::AddrReg offT,
                                                AscendC::MicroAPI::AddrReg offW, AscendC::MicroAPI::AddrReg offF,
                                                MaskReg preg);
    __aicore__ inline void CeFwOfSeg(RegTensor<float>& ceReg, RegTensor<float>& fwReg, RegTensor<float>& p32,
                                     RegTensor<float>& t32, RegTensor<float>& w32, float gamma, float alpha,
                                     MaskReg preg);

    __aicore__ inline int64_t AlignUp(int64_t x, int64_t base) const { return (x + base - 1) / base * base; }

    // 各 buffer 在 UB 中的行间距: 按各自 dtype 的 32B 块对齐, 与 DataCopyPad 的落盘布局一致
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

    // pred 的 padding 值取开区间 (0,1) 内的 0.5, target 的 padding 值取 0。
    // 于是 padding 位上 ce = -log(0.5)*0*w = 0, fw = alpha*exp(gamma*log(0.5))*0 = 0,
    // 恰为求和的单位元, 对任意 gamma 均成立, 无需再对归约缓冲显式清零。
    // (不能 pad 0: -log(0)*0 = inf*0 = NaN; 不能 pad 1: gamma=0 时 0*log(0) = NaN)
    constexpr static float PRED_PAD_VALUE = 0.5f;

private:
    GlobalTensor<T> predGm_;
    GlobalTensor<int32_t> targetGm_;
    GlobalTensor<TW> weightGm_;
    GlobalTensor<T> yGm_;

    TPipe* pipe_ = nullptr;
    TQue<QuePosition::VECIN, 1> predQue_;
    TQue<QuePosition::VECIN, 1> targetQue_;
    TQue<QuePosition::VECIN, 1> weightQue_;
    TQue<QuePosition::VECOUT, 1> yQue_;

    TBuf<QuePosition::VECCALC> ceBuf_;
    TBuf<QuePosition::VECCALC> fwBuf_;
    TBuf<QuePosition::VECCALC> yF32Buf_;
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
};

template <typename T, typename TW, uint64_t hasWeight>
__aicore__ inline void SoftmaxFocalLossND<T, TW, hasWeight>::Init(GM_ADDR pred, GM_ADDR target, GM_ADDR weight,
                                                                  GM_ADDR y,
                                                                  const SoftmaxFocalLossArch35TilingData* tiling,
                                                                  TPipe* pipe)
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
    pipe_ = pipe;

    predGm_.SetGlobalBuffer((__gm__ T*)pred);
    targetGm_.SetGlobalBuffer((__gm__ int32_t*)target);
    yGm_.SetGlobalBuffer((__gm__ T*)y);
    if constexpr (hasWeight == 1) {
        weightGm_.SetGlobalBuffer((__gm__ TW*)weight);
    }

    // VF 的 LoadAlign 按整向量取数, 末行可能越过有效区, 故每块留一个向量的余量
    int64_t maxElem = rowsPerTile_ * colsPerChunk_ + static_cast<int64_t>(VL_FP32);
    accStride_ = AlignUp(rowsPerTile_, static_cast<int64_t>(BLOCK_SIZE / sizeof(float)));

    pipe_->InitBuffer(predQue_, 1, maxElem * sizeof(T));
    pipe_->InitBuffer(targetQue_, 1, maxElem * sizeof(int32_t));
    if constexpr (hasWeight == 1) {
        pipe_->InitBuffer(weightQue_, 1, maxElem * sizeof(TW));
    }
    pipe_->InitBuffer(yQue_, 1, maxElem * sizeof(T));
    pipe_->InitBuffer(ceBuf_, maxElem * sizeof(float));
    pipe_->InitBuffer(fwBuf_, maxElem * sizeof(float));
    pipe_->InitBuffer(yF32Buf_, maxElem * sizeof(float));
    // ceAcc | fwAcc | redTmp | rowVal
    pipe_->InitBuffer(accBuf_, 4 * accStride_ * sizeof(float));
}

template <typename T, typename TW, uint64_t hasWeight>
__aicore__ inline void SoftmaxFocalLossND<T, TW, hasWeight>::Process()
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
__aicore__ inline void SoftmaxFocalLossND<T, TW, hasWeight>::ProcessTile(int64_t rowGm, int64_t rows)
{
    LocalTensor<float> accBuf = accBuf_.template Get<float>();
    LocalTensor<float> ceAcc = accBuf[0];
    LocalTensor<float> fwAcc = accBuf[accStride_];
    LocalTensor<float> redTmp = accBuf[2 * accStride_];
    LocalTensor<float> rowVal = accBuf[3 * accStride_];

    AscendC::Duplicate(ceAcc, 0.0f, static_cast<int32_t>(rows));
    AscendC::Duplicate(fwAcc, 0.0f, static_cast<int32_t>(rows));

    LocalTensor<float> ceBuf = ceBuf_.template Get<float>();
    LocalTensor<float> fwBuf = fwBuf_.template Get<float>();

    for (int64_t c = 0; c < chunkNum_; ++c) {
        int64_t colOff = c * colsPerChunk_;
        int64_t len = r_ - colOff;
        len = len > colsPerChunk_ ? colsPerChunk_ : len;

        CopyIn(rowGm, colOff, rows, len);
        ComputeCeFw(rows, len);

        uint32_t srcShape[2] = {static_cast<uint32_t>(rows), static_cast<uint32_t>(StrideOfF32(len))};
        AscendC::ReduceSum<float, AscendC::Pattern::Reduce::AR, true>(redTmp, ceBuf, srcShape, false);
        AscendC::Add(ceAcc, ceAcc, redTmp, static_cast<int32_t>(rows));
        AscendC::ReduceSum<float, AscendC::Pattern::Reduce::AR, true>(redTmp, fwBuf, srcShape, false);
        AscendC::Add(fwAcc, fwAcc, redTmp, static_cast<int32_t>(rows));
    }

    // y 整行同值 = CE * FW
    AscendC::Mul(rowVal, ceAcc, fwAcc, static_cast<int32_t>(rows));

    for (int64_t c = 0; c < chunkNum_; ++c) {
        int64_t colOff = c * colsPerChunk_;
        int64_t len = r_ - colOff;
        len = len > colsPerChunk_ ? colsPerChunk_ : len;
        CopyOutRow(rowGm, colOff, rows, len);
    }
}

template <typename T, typename TW, uint64_t hasWeight>
__aicore__ inline void SoftmaxFocalLossND<T, TW, hasWeight>::CopyIn(int64_t rowGm, int64_t colOff, int64_t rows,
                                                                    int64_t len)
{
    int64_t gmOffset = rowGm * r_ + colOff;

    LocalTensor<T> predBuf = predQue_.template AllocTensor<T>();
    DataCopyExtParams predParams;
    predParams.blockCount = static_cast<uint16_t>(rows);
    predParams.blockLen = static_cast<uint32_t>(len * sizeof(T));
    predParams.srcStride = static_cast<uint32_t>((r_ - len) * sizeof(T));
    predParams.dstStride = 0;
    DataCopyPadExtParams<T> predPad;
    predPad.isPad = true;
    predPad.leftPadding = 0;
    predPad.rightPadding = static_cast<uint8_t>(StrideOfT(len) - len);
    predPad.paddingValue = static_cast<T>(PRED_PAD_VALUE);
    AscendC::DataCopyPad(predBuf, predGm_[gmOffset], predParams, predPad);
    predQue_.template EnQue<T>(predBuf);

    LocalTensor<int32_t> targetBuf = targetQue_.template AllocTensor<int32_t>();
    DataCopyExtParams tgtParams;
    tgtParams.blockCount = static_cast<uint16_t>(rows);
    tgtParams.blockLen = static_cast<uint32_t>(len * sizeof(int32_t));
    tgtParams.srcStride = static_cast<uint32_t>((r_ - len) * sizeof(int32_t));
    tgtParams.dstStride = 0;
    DataCopyPadExtParams<int32_t> tgtPad;
    tgtPad.isPad = true;
    tgtPad.leftPadding = 0;
    tgtPad.rightPadding = static_cast<uint8_t>(StrideOfF32(len) - len);
    tgtPad.paddingValue = 0;
    AscendC::DataCopyPad(targetBuf, targetGm_[gmOffset], tgtParams, tgtPad);
    targetQue_.template EnQue<int32_t>(targetBuf);

    if constexpr (hasWeight == 1) {
        LocalTensor<TW> weightBuf = weightQue_.template AllocTensor<TW>();
        DataCopyExtParams wParams;
        wParams.blockCount = static_cast<uint16_t>(rows);
        wParams.blockLen = static_cast<uint32_t>(len * sizeof(TW));
        wParams.srcStride = static_cast<uint32_t>((r_ - len) * sizeof(TW));
        wParams.dstStride = 0;
        DataCopyPadExtParams<TW> wPad;
        wPad.isPad = true;
        wPad.leftPadding = 0;
        wPad.rightPadding = static_cast<uint8_t>(StrideOfW(len) - len);
        wPad.paddingValue = static_cast<TW>(0);
        AscendC::DataCopyPad(weightBuf, weightGm_[gmOffset], wParams, wPad);
        weightQue_.template EnQue<TW>(weightBuf);
    }
}

template <typename T, typename TW, uint64_t hasWeight>
__aicore__ inline void SoftmaxFocalLossND<T, TW, hasWeight>::CopyOutRow(int64_t rowGm, int64_t colOff, int64_t rows,
                                                                        int64_t len)
{
    LocalTensor<float> accBuf = accBuf_.template Get<float>();
    LocalTensor<float> rowVal = accBuf[3 * accStride_];
    LocalTensor<float> yF32 = yF32Buf_.template Get<float>();
    LocalTensor<T> yBuf = yQue_.template AllocTensor<T>();

    auto rowValAddr = (__ubuf__ float*)rowVal.GetPhyAddr();
    auto yF32Addr = (__ubuf__ float*)yF32.GetPhyAddr();

    int64_t strideT = StrideOfT(len);
    uint16_t aTimes = static_cast<uint16_t>(rows);
    uint32_t vfLen = VL_FP32;
    // 整行同值, 按整向量铺满 strideT(末行溢出落在预留余量内), 无需尾掩码
    uint16_t repeatTimes = static_cast<uint16_t>((strideT + vfLen - 1) / vfLen);

    __VEC_SCOPE__
    {
        RegTensor<float> valReg;
        MaskReg pregMain = AscendC::MicroAPI::CreateMask<float, MaskPattern::ALL>();
        for (uint16_t i = 0; i < aTimes; ++i) {
            AscendC::MicroAPI::LoadAlign<float, LoadDist::DIST_BRC_B32>(valReg, rowValAddr + i);
            for (uint16_t j = 0; j < repeatTimes; ++j) {
                AscendC::MicroAPI::AddrReg offF = AscendC::MicroAPI::CreateAddrReg<float>(
                    i, static_cast<uint32_t>(strideT), j, vfLen);
                AscendC::MicroAPI::StoreAlign(yF32Addr, valReg, offF, pregMain);
            }
        }
    }

    int32_t total = static_cast<int32_t>(rows * strideT);
    if constexpr (sizeof(T) == sizeof(half)) {
        AscendC::Cast(yBuf, yF32, AscendC::RoundMode::CAST_RINT, total);
    } else {
        AscendC::Copy(yBuf, yF32, total);
    }
    yQue_.template EnQue<T>(yBuf);

    LocalTensor<T> outBuf = yQue_.template DeQue<T>();
    DataCopyExtParams outParams;
    outParams.blockCount = static_cast<uint16_t>(rows);
    outParams.blockLen = static_cast<uint32_t>(len * sizeof(T));
    outParams.srcStride = 0;
    outParams.dstStride = static_cast<uint32_t>((r_ - len) * sizeof(T));
    AscendC::DataCopyPad(yGm_[rowGm * r_ + colOff], outBuf, outParams);
    yQue_.FreeTensor(outBuf);
}

} // namespace SoftmaxFocalLoss

#endif // SOFTMAX_FOCAL_LOSS_ND_H

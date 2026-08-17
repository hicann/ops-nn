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
 * \file adaptive_avg_pool2d_split_c.h
 * \brief SplitC template: downsampling fallback for large kernels. Streams input
 *        H-rows in hiFactor-row batches and reduces arbitrarily large H/W windows
 *        per output point, NC-vectorized. Same streaming machinery as SplitW; this
 *        template is selected (lower priority) when SplitW's IsCapable rejects
 *        (e.g. NC below the SplitW threshold), keeping such cases off the slow
 *        scalar BigKernel path.
 *
 *        For H-downsampling with hiFactor > 1, uses AccumulateWBatchedReg to
 *        batch multiple hi rows per ho in a single VEC_SCOPE, reducing
 *        repeated outBuf loads/stores.
 *
 *        For H-upsampling (hOut > hIn, hoNum > 1), uses a two-phase approach:
 *        WReduceChunkToSum (W-reduce one hi row's chunk to sumBuf_) +
 *        ScatterSumToHo (scatter sumBuf_ to all covering ho positions in
 *        outBuf_). This avoids redundant W-reduction when one hi maps to
 *        multiple ho. Pattern mirrors UpsampleH's WReduceAndScatter.
 *
 *        Hides base CopyInputBatch/TransInputBatch with wi-chunk versions
 *        (different signatures: wiBase/wiLen/wChunkAlign parameters).
 *
 * \arch Ascend950 / A5 / DAV_3510 only, RegBase (MicroAPI) main path, VL = 256 Byte.
 *       [RegBase-native] Host-side gate: AdaptivePool2dBaseTiling::GetShapeAttrsInfo
 *       rejects GetCurNpuArch() != NpuArch::DAV_3510.
 */

#ifndef ADAPTIVE_AVG_POOL2D_SPLIT_C_H_
#define ADAPTIVE_AVG_POOL2D_SPLIT_C_H_

#include "adaptive_avg_pool2d_pooling_base.h"

namespace AdaptivePool2dSplitCNamespace {
using namespace AscendC;
using namespace ops;
using namespace AdaptiveAvgPool2dOp;
using namespace AdaptiveAvgPool2dPoolingBaseNs;

template <typename T, const uint32_t NC_FACTOR>
__simd_vf__ inline void SplitCAccumulateWVf(__ubuf__ T* inputAddr, __ubuf__ float* outAddr,
                                            __ubuf__ int32_t* wStartAddr, __ubuf__ int32_t* wKerSizeAddr,
                                            uint32_t rowBase, uint32_t outBase, uint32_t vlNum, uint32_t vfLenFp32,
                                            int32_t chunkStart, int32_t chunkEnd, uint16_t woChunkFirst,
                                            uint16_t woChunkLast)
{
    MicroAPI::RegTensor<float> inputReg;
    MicroAPI::RegTensor<float> sumReg;
    MicroAPI::MaskReg preg = MicroAPI::CreateMask<float, MicroAPI::MaskPattern::ALL>();

    uint16_t woFirst = woChunkFirst;
    uint16_t woLast = woChunkLast;

    for (uint16_t wo = woFirst; wo < woLast; wo++) {
        int32_t wStart = wStartAddr[wo];
        int32_t wEnd = wStart + wKerSizeAddr[wo];
        int32_t ovStart = wStart > chunkStart ? wStart : chunkStart;
        int32_t ovEnd = wEnd < chunkEnd ? wEnd : chunkEnd;
        uint16_t cnt = ovEnd > ovStart ? static_cast<uint16_t>(ovEnd - ovStart) : 0;
        uint32_t baseOffset = (rowBase + static_cast<uint32_t>(ovStart - chunkStart)) * vlNum;
        uint32_t sumOffset = outBase + static_cast<uint32_t>(wo) * vlNum;

        MicroAPI::LoadAlign(sumReg, outAddr + sumOffset);
        for (uint16_t k = 0; k < cnt; k++) {
            uint32_t inputOffset = baseOffset + static_cast<uint32_t>(k) * vlNum;
            ops_vf::LoadOneTensorForDtypeT<T>(inputAddr, inputReg, preg, inputOffset);
            MicroAPI::Add(sumReg, sumReg, inputReg, preg);
        }
        MicroAPI::StoreAlign(outAddr + sumOffset, sumReg, preg);

        if constexpr (NC_FACTOR == TPL_NC_FACTOR_128) {
            MicroAPI::LoadAlign(sumReg, outAddr + sumOffset + vfLenFp32);
            for (uint16_t k = 0; k < cnt; k++) {
                uint32_t inputOffsetC = baseOffset + static_cast<uint32_t>(k) * vlNum + vfLenFp32;
                ops_vf::LoadOneTensorForDtypeT<T>(inputAddr, inputReg, preg, inputOffsetC);
                MicroAPI::Add(sumReg, sumReg, inputReg, preg);
            }
            MicroAPI::StoreAlign(outAddr + sumOffset + vfLenFp32, sumReg, preg);
        }
    }
}

template <typename T, const uint32_t NC_FACTOR>
__simd_vf__ inline void SplitCAccumulateWBatchedRegVf(__ubuf__ T* inputAddr, __ubuf__ float* outAddr,
                                                      __ubuf__ int32_t* wStartAddr, __ubuf__ int32_t* wKerSizeAddr,
                                                      uint32_t wChunkAlign, uint16_t hiStart, uint16_t hiEnd,
                                                      uint32_t outBase, uint32_t vlNum, uint32_t vfLenFp32,
                                                      int32_t chunkStart, int32_t chunkEnd, uint16_t woChunkFirst,
                                                      uint16_t woChunkLast)
{
    MicroAPI::RegTensor<float> inputReg;
    MicroAPI::RegTensor<float> sumReg;
    MicroAPI::MaskReg preg = MicroAPI::CreateMask<float, MicroAPI::MaskPattern::ALL>();

    uint16_t woFirst = woChunkFirst;
    uint16_t woLast = woChunkLast;

    for (uint16_t wo = woFirst; wo < woLast; wo++) {
        int32_t wStart = wStartAddr[wo];
        int32_t wEnd = wStart + wKerSizeAddr[wo];
        int32_t ovStart = wStart > chunkStart ? wStart : chunkStart;
        int32_t ovEnd = wEnd < chunkEnd ? wEnd : chunkEnd;
        uint16_t cnt = ovEnd > ovStart ? static_cast<uint16_t>(ovEnd - ovStart) : 0;
        uint32_t wiOffset = static_cast<uint32_t>(ovStart - chunkStart);
        uint32_t sumOffset = outBase + static_cast<uint32_t>(wo) * vlNum;

        MicroAPI::LoadAlign(sumReg, outAddr + sumOffset);
        for (uint16_t hiOff = hiStart; hiOff < hiEnd; hiOff++) {
            uint32_t baseOffset = (static_cast<uint32_t>(hiOff) * wChunkAlign + wiOffset) * vlNum;
            for (uint16_t k = 0; k < cnt; k++) {
                uint32_t inputOffset = baseOffset + static_cast<uint32_t>(k) * vlNum;
                ops_vf::LoadOneTensorForDtypeT<T>(inputAddr, inputReg, preg, inputOffset);
                MicroAPI::Add(sumReg, sumReg, inputReg, preg);
            }
        }
        MicroAPI::StoreAlign(outAddr + sumOffset, sumReg, preg);

        if constexpr (NC_FACTOR == TPL_NC_FACTOR_128) {
            MicroAPI::LoadAlign(sumReg, outAddr + sumOffset + vfLenFp32);
            for (uint16_t hiOff = hiStart; hiOff < hiEnd; hiOff++) {
                uint32_t baseOffset = (static_cast<uint32_t>(hiOff) * wChunkAlign + wiOffset) * vlNum;
                for (uint16_t k = 0; k < cnt; k++) {
                    uint32_t inputOffsetC = baseOffset + static_cast<uint32_t>(k) * vlNum + vfLenFp32;
                    ops_vf::LoadOneTensorForDtypeT<T>(inputAddr, inputReg, preg, inputOffsetC);
                    MicroAPI::Add(sumReg, sumReg, inputReg, preg);
                }
            }
            MicroAPI::StoreAlign(outAddr + sumOffset + vfLenFp32, sumReg, preg);
        }
    }
}

template <typename T, const uint32_t NC_FACTOR>
__simd_vf__ inline void SplitCWReduceChunkToSumVf(__ubuf__ T* inputAddr, __ubuf__ float* sumAddr,
                                                  __ubuf__ int32_t* wStartAddr, __ubuf__ int32_t* wKerSizeAddr,
                                                  uint32_t rowBase, uint32_t vlNum, uint32_t vfLenFp32,
                                                  int32_t chunkStart, int32_t chunkEnd, uint16_t woFirst,
                                                  uint16_t woLast)
{
    MicroAPI::RegTensor<float> inputReg;
    MicroAPI::RegTensor<float> sumReg;
    MicroAPI::MaskReg preg = MicroAPI::CreateMask<float, MicroAPI::MaskPattern::ALL>();

    for (uint16_t wo = woFirst; wo < woLast; wo++) {
        int32_t wStart = wStartAddr[wo];
        int32_t wEnd = wStart + wKerSizeAddr[wo];
        int32_t ovStart = wStart > chunkStart ? wStart : chunkStart;
        int32_t ovEnd = wEnd < chunkEnd ? wEnd : chunkEnd;
        uint16_t cnt = ovEnd > ovStart ? static_cast<uint16_t>(ovEnd - ovStart) : 0;
        uint32_t baseOffset = (rowBase + static_cast<uint32_t>(ovStart - chunkStart)) * vlNum;
        uint32_t sumOffset = static_cast<uint32_t>(wo) * vlNum;

        MicroAPI::Duplicate(sumReg, 0.0f);
        for (uint16_t k = 0; k < cnt; k++) {
            uint32_t inputOffset = baseOffset + static_cast<uint32_t>(k) * vlNum;
            ops_vf::LoadOneTensorForDtypeT<T>(inputAddr, inputReg, preg, inputOffset);
            MicroAPI::Add(sumReg, sumReg, inputReg, preg);
        }
        MicroAPI::StoreAlign(sumAddr + sumOffset, sumReg, preg);

        if constexpr (NC_FACTOR == TPL_NC_FACTOR_128) {
            MicroAPI::Duplicate(sumReg, 0.0f);
            for (uint16_t k = 0; k < cnt; k++) {
                uint32_t inputOffsetC = baseOffset + static_cast<uint32_t>(k) * vlNum + vfLenFp32;
                ops_vf::LoadOneTensorForDtypeT<T>(inputAddr, inputReg, preg, inputOffsetC);
                MicroAPI::Add(sumReg, sumReg, inputReg, preg);
            }
            MicroAPI::StoreAlign(sumAddr + sumOffset + vfLenFp32, sumReg, preg);
        }
    }
}

template <const uint32_t NC_FACTOR>
__simd_vf__ inline void SplitCScatterSumToHoVf(__ubuf__ float* sumAddr, __ubuf__ float* outAddr, uint32_t outBase,
                                               uint32_t vlNum, uint32_t vfLenFp32, uint16_t woFirst, uint16_t woLast)
{
    MicroAPI::RegTensor<float> sumReg;
    MicroAPI::RegTensor<float> outReg;
    MicroAPI::MaskReg preg = MicroAPI::CreateMask<float, MicroAPI::MaskPattern::ALL>();

    for (uint16_t wo = woFirst; wo < woLast; wo++) {
        uint32_t sumOffset = static_cast<uint32_t>(wo) * vlNum;
        uint32_t outOffset = outBase + sumOffset;

        MicroAPI::LoadAlign(sumReg, sumAddr + sumOffset);
        MicroAPI::LoadAlign(outReg, outAddr + outOffset);
        MicroAPI::Add(outReg, outReg, sumReg, preg);
        MicroAPI::StoreAlign(outAddr + outOffset, outReg, preg);

        if constexpr (NC_FACTOR == TPL_NC_FACTOR_128) {
            MicroAPI::LoadAlign(sumReg, sumAddr + sumOffset + vfLenFp32);
            MicroAPI::LoadAlign(outReg, outAddr + outOffset + vfLenFp32);
            MicroAPI::Add(outReg, outReg, sumReg, preg);
            MicroAPI::StoreAlign(outAddr + outOffset + vfLenFp32, outReg, preg);
        }
    }
}

template <typename T, typename ID_T, const uint32_t NC_FACTOR>
class AdaptiveAvgPool2dSplitC
    : protected AdaptiveAvgPool2dPoolingBase<T, ID_T, NC_FACTOR, AdaptivePool2dSplitCTilingData> {
    using Base = AdaptiveAvgPool2dPoolingBase<T, ID_T, NC_FACTOR, AdaptivePool2dSplitCTilingData>;

public:
    __aicore__ inline AdaptiveAvgPool2dSplitC(const AdaptivePool2dSplitCTilingData* tilingData, TPipe* pipe)
        : Base(tilingData, pipe){};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y);
    __aicore__ inline void Process();

private:
    // Shadows base CopyIn/Transpose with wi-chunk versions (different signatures)
    __aicore__ inline void CopyInputBatch(int64_t ncIdx, int64_t ncNum, int64_t hiStart, int64_t hiBatch,
                                          int64_t wiBase, int64_t wiLen, uint32_t wChunkAlign);
    __aicore__ inline void TransInputBatch(int64_t hiBatch, uint32_t wChunkAlign);
    __aicore__ inline void AccumulateW(int64_t rowOffset, int64_t hoLocal, int64_t wiBase, int64_t wiLen);
    __aicore__ inline void AccumulateWBatchedReg(uint32_t wChunkAlign, uint16_t hiStart, uint16_t hiEnd,
                                                 int64_t hoLocal, int64_t wiBase, int64_t wiLen);
    __aicore__ inline void WReduceChunkToSum(int64_t rowOffset, int64_t wiBase, int64_t wiLen);
    __aicore__ inline void ScatterSumToHo(int64_t hoLocal);
    __aicore__ inline void CalWoChunkRange(int64_t wiBase, int64_t wiLen);
    __aicore__ inline void ProcessOneBlock(const BlockParam& blockPara);

    // Uses wInFactor (chunk width) alignment, not full wIn
    uint32_t wInFactorAlign_;
    // wo range [woChunkFirst_, woChunkLast_) overlapping with current wi chunk,
    // set by WReduceChunkToSum and consumed by ScatterSumToHo
    uint16_t woChunkFirst_;
    uint16_t woChunkLast_;
};

template <typename T, typename ID_T, const uint32_t NC_FACTOR>
__aicore__ inline void AdaptiveAvgPool2dSplitC<T, ID_T, NC_FACTOR>::Init(GM_ADDR x, GM_ADDR y)
{
    if (!this->InitCommon(x, y)) {
        return;
    }
    wInFactorAlign_ = ops::CeilAlign(static_cast<uint32_t>(this->tilingData_->wInFactor), this->ubAlignNum_);
    uint64_t dataBlock = AAP_UB_BLOCK_SIZE;
    uint64_t wBufSize = ops::CeilAlign(static_cast<uint64_t>(this->tilingData_->wOut) * sizeof(int32_t), dataBlock);
    this->InitBuffers(wBufSize, wInFactorAlign_);
}

// CopyInputBatch: [hiBatch, wiLen] tile for VL NC channels
template <typename T, typename ID_T, const uint32_t NC_FACTOR>
__aicore__ inline void AdaptiveAvgPool2dSplitC<T, ID_T, NC_FACTOR>::CopyInputBatch(int64_t ncIdx, int64_t ncNum,
                                                                                   int64_t hiStart, int64_t hiBatch,
                                                                                   int64_t wiBase, int64_t wiLen,
                                                                                   uint32_t wChunkAlign)
{
    LocalTensor<T> xLocal = this->inputQue_.template AllocTensor<T>();

    int64_t gmOffset = ncIdx * this->tilingData_->ncFactor * this->inHW_ + hiStart * this->tilingData_->wIn + wiBase;

    DataCopyExtParams paramsIn = {static_cast<uint16_t>(hiBatch), static_cast<uint32_t>(wiLen * sizeof(T)),
                                  static_cast<uint32_t>((this->tilingData_->wIn - wiLen) * sizeof(T)),
                                  static_cast<uint32_t>(0), static_cast<uint32_t>(0)};
    DataCopyPadExtParams<T> padParams = {false, 0, 0, 0};

    LoopModeParams loopModeParams;
    loopModeParams.loop1Size = ncNum;
    loopModeParams.loop2Size = 1;
    loopModeParams.loop1SrcStride = this->inHW_ * sizeof(T);
    loopModeParams.loop2SrcStride = 0;
    loopModeParams.loop1DstStride = hiBatch * wChunkAlign * sizeof(T);
    loopModeParams.loop2DstStride = 0;

    SetLoopModePara(loopModeParams, DataCopyMVType::OUT_TO_UB);
    DataCopyPad(xLocal, this->xGm_[gmOffset], paramsIn, padParams);
    ResetLoopModePara(DataCopyMVType::OUT_TO_UB);
    this->inputQue_.EnQue(xLocal);
}

// TransInputBatch: [VL, hiBatch*wChunkAlign] → [hiBatch*wChunkAlign, VL]
template <typename T, typename ID_T, const uint32_t NC_FACTOR>
__aicore__ inline void AdaptiveAvgPool2dSplitC<T, ID_T, NC_FACTOR>::TransInputBatch(int64_t hiBatch,
                                                                                    uint32_t wChunkAlign)
{
    LocalTensor<T> xLocal = this->inputQue_.template DeQue<T>();
    LocalTensor<T> transLocal = this->transBuf_.template Get<T>();
    uint32_t colNum = static_cast<uint32_t>(hiBatch) * wChunkAlign;
    if constexpr (IsSameType<T, float>::value) {
        this->template TransposeB32<T>(transLocal, xLocal, this->vlNum_, colNum);
    } else {
        this->TransposeB16(transLocal, xLocal, this->vlNum_, colNum);
    }
    this->inputQue_.FreeTensor(xLocal);
}

template <typename T, typename ID_T, const uint32_t NC_FACTOR>
__aicore__ inline void AdaptiveAvgPool2dSplitC<T, ID_T, NC_FACTOR>::AccumulateW(int64_t rowOffset, int64_t hoLocal,
                                                                                int64_t wiBase, int64_t wiLen)
{
    LocalTensor<T> transLocal = this->transBuf_.template Get<T>();
    LocalTensor<float> outLocal = this->outBuf_.template Get<float>();
    __ubuf__ T* inputAddr = (__ubuf__ T*)transLocal.GetPhyAddr();
    __ubuf__ float* outAddr = (__ubuf__ float*)outLocal.GetPhyAddr();
    LocalTensor<int32_t> wStartLocal = this->wStartBuf_.template Get<int32_t>();
    __ubuf__ int32_t* wStartAddr = (__ubuf__ int32_t*)wStartLocal.GetPhyAddr();
    LocalTensor<int32_t> wKerSizeLocal = this->wKerSizeBuf_.template Get<int32_t>();
    __ubuf__ int32_t* wKerSizeAddr = (__ubuf__ int32_t*)wKerSizeLocal.GetPhyAddr();

    uint32_t vfLenFp32 = AAP_V_REG_SIZE / sizeof(float);
    uint32_t rowBase = static_cast<uint32_t>(rowOffset);
    uint32_t outBase = static_cast<uint32_t>(hoLocal) * this->wOutAlign_ * this->vlNum_;
    uint16_t woNum = static_cast<uint16_t>(this->tilingData_->wOut);
    int32_t chunkStart = static_cast<int32_t>(wiBase);
    int32_t chunkEnd = static_cast<int32_t>(wiBase + wiLen);

    SplitCAccumulateWVf<T, NC_FACTOR>(inputAddr, outAddr, wStartAddr, wKerSizeAddr, rowBase, outBase, this->vlNum_,
                                      vfLenFp32, chunkStart, chunkEnd, woChunkFirst_, woChunkLast_);
}

template <typename T, typename ID_T, const uint32_t NC_FACTOR>
__aicore__ inline void AdaptiveAvgPool2dSplitC<T, ID_T, NC_FACTOR>::AccumulateWBatchedReg(
    uint32_t wChunkAlign, uint16_t hiStart, uint16_t hiEnd, int64_t hoLocal, int64_t wiBase, int64_t wiLen)
{
    LocalTensor<T> transLocal = this->transBuf_.template Get<T>();
    LocalTensor<float> outLocal = this->outBuf_.template Get<float>();
    __ubuf__ T* inputAddr = (__ubuf__ T*)transLocal.GetPhyAddr();
    __ubuf__ float* outAddr = (__ubuf__ float*)outLocal.GetPhyAddr();
    LocalTensor<int32_t> wStartLocal = this->wStartBuf_.template Get<int32_t>();
    __ubuf__ int32_t* wStartAddr = (__ubuf__ int32_t*)wStartLocal.GetPhyAddr();
    LocalTensor<int32_t> wKerSizeLocal = this->wKerSizeBuf_.template Get<int32_t>();
    __ubuf__ int32_t* wKerSizeAddr = (__ubuf__ int32_t*)wKerSizeLocal.GetPhyAddr();

    uint32_t vfLenFp32 = AAP_V_REG_SIZE / sizeof(float);
    uint32_t outBase = static_cast<uint32_t>(hoLocal) * this->wOutAlign_ * this->vlNum_;
    uint16_t woNum = static_cast<uint16_t>(this->tilingData_->wOut);
    int32_t chunkStart = static_cast<int32_t>(wiBase);
    int32_t chunkEnd = static_cast<int32_t>(wiBase + wiLen);

    SplitCAccumulateWBatchedRegVf<T, NC_FACTOR>(inputAddr, outAddr, wStartAddr, wKerSizeAddr, wChunkAlign, hiStart,
                                                hiEnd, outBase, this->vlNum_, vfLenFp32, chunkStart, chunkEnd,
                                                woChunkFirst_, woChunkLast_);
}

// W-reduce one hi row's chunk to sumBuf, clipped to overlapping wo range.
// Stores [woChunkFirst_, woChunkLast_) for ScatterSumToHo.
template <typename T, typename ID_T, const uint32_t NC_FACTOR>
__aicore__ inline void AdaptiveAvgPool2dSplitC<T, ID_T, NC_FACTOR>::WReduceChunkToSum(int64_t rowOffset, int64_t wiBase,
                                                                                      int64_t wiLen)
{
    LocalTensor<T> transLocal = this->transBuf_.template Get<T>();
    LocalTensor<float> sumLocal = this->sumBuf_.template Get<float>();
    __ubuf__ T* inputAddr = (__ubuf__ T*)transLocal.GetPhyAddr();
    __ubuf__ float* sumAddr = (__ubuf__ float*)sumLocal.GetPhyAddr();
    LocalTensor<int32_t> wStartLocal = this->wStartBuf_.template Get<int32_t>();
    __ubuf__ int32_t* wStartAddr = (__ubuf__ int32_t*)wStartLocal.GetPhyAddr();
    LocalTensor<int32_t> wKerSizeLocal = this->wKerSizeBuf_.template Get<int32_t>();
    __ubuf__ int32_t* wKerSizeAddr = (__ubuf__ int32_t*)wKerSizeLocal.GetPhyAddr();

    uint32_t vfLenFp32 = AAP_V_REG_SIZE / sizeof(float);
    uint32_t rowBase = static_cast<uint32_t>(rowOffset);
    int32_t chunkStart = static_cast<int32_t>(wiBase);
    int32_t chunkEnd = static_cast<int32_t>(wiBase + wiLen);

    uint16_t woFirst = woChunkFirst_;
    uint16_t woLast = woChunkLast_;

    SplitCWReduceChunkToSumVf<T, NC_FACTOR>(inputAddr, sumAddr, wStartAddr, wKerSizeAddr, rowBase, this->vlNum_,
                                            vfLenFp32, chunkStart, chunkEnd, woFirst, woLast);
}

template <typename T, typename ID_T, const uint32_t NC_FACTOR>
__aicore__ inline void AdaptiveAvgPool2dSplitC<T, ID_T, NC_FACTOR>::CalWoChunkRange(int64_t wiBase, int64_t wiLen)
{
    LocalTensor<int32_t> wStartLocal = this->wStartBuf_.template Get<int32_t>();
    __ubuf__ int32_t* wStartAddr = (__ubuf__ int32_t*)wStartLocal.GetPhyAddr();
    LocalTensor<int32_t> wKerSizeLocal = this->wKerSizeBuf_.template Get<int32_t>();
    __ubuf__ int32_t* wKerSizeAddr = (__ubuf__ int32_t*)wKerSizeLocal.GetPhyAddr();

    uint16_t woNum = static_cast<uint16_t>(this->tilingData_->wOut);
    int32_t chunkStart = static_cast<int32_t>(wiBase);
    int32_t chunkEnd = static_cast<int32_t>(wiBase + wiLen);

    uint16_t lo = 0;
    uint16_t hi = woNum;
    while (lo < hi) {
        uint16_t mid = lo + (hi - lo) / 2;
        if (wStartAddr[mid] + wKerSizeAddr[mid] <= chunkStart) {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    woChunkFirst_ = lo;

    uint16_t lo2 = lo;
    uint16_t hi2 = woNum;
    while (lo2 < hi2) {
        uint16_t mid = lo2 + (hi2 - lo2) / 2;
        if (wStartAddr[mid] < chunkEnd) {
            lo2 = mid + 1;
        } else {
            hi2 = mid;
        }
    }
    woChunkLast_ = lo2;
}

// Scatter sumBuf to one ho position in outBuf: outBuf[ho][wo] += sumBuf[wo].
// Only iterates [woChunkFirst_, woChunkLast_) set by CalWoChunkRange.
template <typename T, typename ID_T, const uint32_t NC_FACTOR>
__aicore__ inline void AdaptiveAvgPool2dSplitC<T, ID_T, NC_FACTOR>::ScatterSumToHo(int64_t hoLocal)
{
    LocalTensor<float> sumLocal = this->sumBuf_.template Get<float>();
    LocalTensor<float> outLocal = this->outBuf_.template Get<float>();
    __ubuf__ float* sumAddr = (__ubuf__ float*)sumLocal.GetPhyAddr();
    __ubuf__ float* outAddr = (__ubuf__ float*)outLocal.GetPhyAddr();

    uint32_t vfLenFp32 = AAP_V_REG_SIZE / sizeof(float);
    uint32_t outBase = static_cast<uint32_t>(hoLocal) * this->wOutAlign_ * this->vlNum_;
    uint16_t woFirst = woChunkFirst_;
    uint16_t woLast = woChunkLast_;

    SplitCScatterSumToHoVf<NC_FACTOR>(sumAddr, outAddr, outBase, this->vlNum_, vfLenFp32, woFirst, woLast);
}

template <typename T, typename ID_T, const uint32_t NC_FACTOR>
__aicore__ inline void AdaptiveAvgPool2dSplitC<T, ID_T, NC_FACTOR>::ProcessOneBlock(const BlockParam& blockPara)
{
    int64_t ncIdx = blockPara.ncIdx;
    int64_t ncNum = blockPara.ncNum;
    int64_t hoGlobalStart = blockPara.hoIdx * this->tilingData_->hoFactor;
    int64_t hIn = this->tilingData_->hIn;
    int64_t hoNum = blockPara.hoNum;
    int64_t hOut = this->tilingData_->hOut;

    this->ClearOutBuf();

    int64_t hiMin = 0;
    int64_t hiMax = 0;
    this->CalHiRange(hoGlobalStart, hoNum, hiMin, hiMax);
    int64_t wIn = this->tilingData_->wIn;
    int64_t wInTile = this->tilingData_->wInFactor;
    bool isHUpsample = (hOut > hIn && hoNum > 1);

    for (int64_t hiBase = hiMin; hiBase < hiMax; hiBase += this->tilingData_->hiFactor) {
        int64_t hiBatch = (hiBase + this->tilingData_->hiFactor > hiMax) ? (hiMax - hiBase) :
                                                                           this->tilingData_->hiFactor;

        for (int64_t wiBase = 0; wiBase < wIn; wiBase += wInTile) {
            int64_t wiLen = wInTile;
            if (wiBase + wiLen > wIn) {
                wiLen = wIn - wiBase;
            }
            uint32_t wChunkAlign = ops::CeilAlign(static_cast<uint32_t>(wiLen), this->ubAlignNum_);

            CopyInputBatch(ncIdx, ncNum, hiBase, hiBatch, wiBase, wiLen, wChunkAlign);
            TransInputBatch(hiBatch, wChunkAlign);
            CalWoChunkRange(wiBase, wiLen);

            if (isHUpsample) {
                for (int64_t hiOff = 0; hiOff < hiBatch; hiOff++) {
                    int64_t hi = hiBase + hiOff;
                    int64_t hoScatterStart = -1;
                    int64_t hoScatterCount = 0;
                    for (int64_t hoLocal = 0; hoLocal < hoNum; hoLocal++) {
                        int64_t hoGlobal = hoGlobalStart + hoLocal;
                        int64_t hStart = (hoGlobal * hIn) / hOut;
                        int64_t hEnd = ((hoGlobal + 1) * hIn + hOut - 1) / hOut;
                        if (hi >= hStart && hi < hEnd) {
                            if (hoScatterStart < 0) {
                                hoScatterStart = hoLocal;
                            }
                            hoScatterCount++;
                        } else if (hoScatterStart >= 0) {
                            break;
                        }
                    }
                    if (hoScatterCount > 1) {
                        WReduceChunkToSum(hiOff * wChunkAlign, wiBase, wiLen);
                        for (int64_t j = 0; j < hoScatterCount; j++) {
                            ScatterSumToHo(hoScatterStart + j);
                        }
                    } else if (hoScatterCount == 1) {
                        AccumulateW(hiOff * wChunkAlign, hoScatterStart, wiBase, wiLen);
                    }
                }
            }
            if (!isHUpsample) {
                for (int64_t hoLocal = 0; hoLocal < hoNum; hoLocal++) {
                    int64_t hoGlobal = hoGlobalStart + hoLocal;
                    int64_t hStart = (hoGlobal * hIn) / hOut;
                    int64_t hEnd = ((hoGlobal + 1) * hIn + hOut - 1) / hOut;
                    int64_t hiStartClamped = hStart > hiBase ? hStart : hiBase;
                    int64_t hiEndClamped = hEnd < (hiBase + hiBatch) ? hEnd : (hiBase + hiBatch);
                    if (hiStartClamped < hiEndClamped) {
                        uint16_t hiStart16 = static_cast<uint16_t>(hiStartClamped - hiBase);
                        uint16_t hiEnd16 = static_cast<uint16_t>(hiEndClamped - hiBase);
                        AccumulateWBatchedReg(wChunkAlign, hiStart16, hiEnd16, hoLocal, wiBase, wiLen);
                    }
                }
            }
        }
    }

    for (int64_t hoLocal = 0; hoLocal < hoNum; hoLocal++) {
        int64_t hoGlobal = hoGlobalStart + hoLocal;
        int64_t hStart = (hoGlobal * hIn) / hOut;
        int64_t kernelH = ((hoGlobal + 1) * hIn + hOut - 1) / hOut - hStart;
        this->CalAvgOneHo(kernelH, hoLocal);
    }

    this->TransOut(hoNum);
    this->CopyOut(ncIdx, ncNum, hoGlobalStart, hoNum);
}

template <typename T, typename ID_T, const uint32_t NC_FACTOR>
__aicore__ inline void AdaptiveAvgPool2dSplitC<T, ID_T, NC_FACTOR>::Process()
{
    if (GetBlockIdx() >= this->tilingData_->useCoreNum) {
        return;
    }

    // wStart/wKerSize depend only on wIn/wOut, so compute them once for all blocks.
    // The V_S fence is required because CalWoChunkRange binary-searches these buffers
    // with scalar loads, and there is no hardware ordering between the VF stores above
    // and those scalar reads.
    this->CalWKernelInfo();
    event_t eventIdVToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
    SetFlag<HardEvent::V_S>(eventIdVToS);
    WaitFlag<HardEvent::V_S>(eventIdVToS);

    for (int64_t curIdx = this->startBlockIdx_; curIdx < this->endBlockIdx_; curIdx++) {
        BlockParam blockPara;
        this->CalBlockPara(curIdx, blockPara);
        ProcessOneBlock(blockPara);
    }
}

} // namespace AdaptivePool2dSplitCNamespace
#endif // ADAPTIVE_AVG_POOL2D_SPLIT_C_H_

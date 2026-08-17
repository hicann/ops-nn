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
 * \file adaptive_avg_pool2d_upsample_h.h
 * \brief UpsampleH template: optimized for H upsampling + W downsampling.
 *        H upsampling means each input row is reused by many output rows.
 *        We batch-load input H rows (hiFactor), transpose to [hiFactor*wIn, VL],
 *        then for each input row accumulate its W-reduced contribution into all
 *        Ho outputs that cover it. This avoids SmallKernel's totalOuter explosion.
 *
 *        Algorithm per block (ncIdx, hoIdx):
 *          1. Clear outBuf (per-Ho accumulators)
 *          2. For each hiBatch in [hiMin, hiMax):
 *             a. CopyIn [VL, hiBatch, wInAlign]
 *             b. Transpose to [hiBatch*wInAlign, VL]
 *             c. For each hi in batch:
 *                - Find all Ho whose window covers this hi (sliding cursor)
 *                - AccumulateW: sum W window for each such Ho into outBuf[ho]
 *          3. CalAvgOneHo: divide outBuf by kernelH * kernelW
 *          4. TransOut + CopyOut
 *
 * \arch Ascend950 / A5 / DAV_3510 only, RegBase (MicroAPI) main path, VL = 256 Byte.
 *       [RegBase-native] Host-side gate: AdaptivePool2dBaseTiling::GetShapeAttrsInfo
 *       rejects GetCurNpuArch() != NpuArch::DAV_3510.
 */

#ifndef ADAPTIVE_AVG_POOL2D_UPSAMPLE_H_H_
#define ADAPTIVE_AVG_POOL2D_UPSAMPLE_H_H_

#include "adaptive_avg_pool2d_pooling_base.h"

namespace AdaptivePool2dUpsampleHNamespace {
using namespace AscendC;
using namespace ops;
using namespace AdaptiveAvgPool2dOp;
using namespace AdaptiveAvgPool2dPoolingBaseNs;

template <typename T, const uint32_t NC_FACTOR>
__simd_vf__ inline void UpsampleHAccumulateWVf(__ubuf__ T* inputAddr, __ubuf__ float* outAddr,
                                               __ubuf__ int32_t* wStartAddr, __ubuf__ int32_t* wKerSizeAddr,
                                               uint32_t rowBase, uint32_t outBase, uint16_t woNum, uint32_t vlNum,
                                               uint32_t vfLenFp32)
{
    MicroAPI::RegTensor<float> inputReg;
    MicroAPI::RegTensor<float> sumReg;
    MicroAPI::MaskReg preg = MicroAPI::CreateMask<float, MicroAPI::MaskPattern::ALL>();

    for (uint16_t wo = 0; wo < woNum; wo++) {
        uint32_t baseOffset = (rowBase + static_cast<uint32_t>(wStartAddr[wo])) * vlNum;
        uint16_t kernelW = static_cast<uint16_t>(wKerSizeAddr[wo]);
        uint32_t sumOffset = outBase + static_cast<uint32_t>(wo) * vlNum;

        MicroAPI::LoadAlign(sumReg, outAddr + sumOffset);
        for (uint16_t k = 0; k < kernelW; k++) {
            uint32_t inputOffset = baseOffset + static_cast<uint32_t>(k) * vlNum;
            ops_vf::LoadOneTensorForDtypeT<T>(inputAddr, inputReg, preg, inputOffset);
            MicroAPI::Add(sumReg, sumReg, inputReg, preg);
        }
        MicroAPI::StoreAlign(outAddr + sumOffset, sumReg, preg);

        if constexpr (NC_FACTOR == TPL_NC_FACTOR_128) {
            MicroAPI::LoadAlign(sumReg, outAddr + sumOffset + vfLenFp32);
            for (uint16_t k = 0; k < kernelW; k++) {
                uint32_t inputOffset1 = baseOffset + static_cast<uint32_t>(k) * vlNum + vfLenFp32;
                ops_vf::LoadOneTensorForDtypeT<T>(inputAddr, inputReg, preg, inputOffset1);
                MicroAPI::Add(sumReg, sumReg, inputReg, preg);
            }
            MicroAPI::StoreAlign(outAddr + sumOffset + vfLenFp32, sumReg, preg);
        }
    }
}

template <typename T, const uint32_t NC_FACTOR>
__simd_vf__ inline void UpsampleHAccumulateWBulkVf(__ubuf__ T* inputAddr, __ubuf__ float* outAddr, uint32_t rowBase,
                                                   uint32_t outBase, uint16_t woNum, uint32_t vlNum, uint32_t vfLenFp32)
{
    MicroAPI::RegTensor<float> inputReg;
    MicroAPI::RegTensor<float> sumReg;
    MicroAPI::RegTensor<float> outReg;
    MicroAPI::MaskReg preg = MicroAPI::CreateMask<float, MicroAPI::MaskPattern::ALL>();

    for (uint16_t wo = 0; wo < woNum; wo++) {
        uint32_t inputOffset = (rowBase + static_cast<uint32_t>(wo)) * vlNum;
        uint32_t outOffset = outBase + static_cast<uint32_t>(wo) * vlNum;

        MicroAPI::Duplicate(sumReg, 0.0f);
        ops_vf::LoadOneTensorForDtypeT<T>(inputAddr, inputReg, preg, inputOffset);
        MicroAPI::Add(sumReg, sumReg, inputReg, preg);
        ops_vf::LoadOneTensorForDtypeT<T>(inputAddr, inputReg, preg, inputOffset + vlNum);
        MicroAPI::Add(sumReg, sumReg, inputReg, preg);
        MicroAPI::LoadAlign(outReg, outAddr + outOffset);
        MicroAPI::Add(outReg, outReg, sumReg, preg);
        MicroAPI::StoreAlign(outAddr + outOffset, outReg, preg);

        if constexpr (NC_FACTOR == TPL_NC_FACTOR_128) {
            MicroAPI::Duplicate(sumReg, 0.0f);
            ops_vf::LoadOneTensorForDtypeT<T>(inputAddr, inputReg, preg, inputOffset + vfLenFp32);
            MicroAPI::Add(sumReg, sumReg, inputReg, preg);
            ops_vf::LoadOneTensorForDtypeT<T>(inputAddr, inputReg, preg, inputOffset + vlNum + vfLenFp32);
            MicroAPI::Add(sumReg, sumReg, inputReg, preg);
            MicroAPI::LoadAlign(outReg, outAddr + outOffset + vfLenFp32);
            MicroAPI::Add(outReg, outReg, sumReg, preg);
            MicroAPI::StoreAlign(outAddr + outOffset + vfLenFp32, outReg, preg);
        }
    }
}

template <typename T, const uint32_t NC_FACTOR>
__simd_vf__ inline void UpsampleHAccumulateWUpsampleVf(__ubuf__ T* inputAddr, __ubuf__ float* outAddr,
                                                       __ubuf__ int32_t* wiWoStartAddr, __ubuf__ int32_t* wiWoCountAddr,
                                                       uint32_t rowBase, uint32_t outBase, uint16_t wiNum,
                                                       uint32_t vlNum, uint32_t vfLenFp32)
{
    MicroAPI::RegTensor<float> inputReg;
    MicroAPI::RegTensor<float> outReg;
    MicroAPI::MaskReg preg = MicroAPI::CreateMask<float, MicroAPI::MaskPattern::ALL>();

    for (uint16_t wi = 0; wi < wiNum; wi++) {
        uint32_t inputOffset = (rowBase + static_cast<uint32_t>(wi)) * vlNum;
        ops_vf::LoadOneTensorForDtypeT<T>(inputAddr, inputReg, preg, inputOffset);

        uint32_t woStart = static_cast<uint32_t>(wiWoStartAddr[wi]);
        uint16_t woCount = static_cast<uint16_t>(wiWoCountAddr[wi]);
        for (uint16_t j = 0; j < woCount; j++) {
            uint32_t wo = woStart + static_cast<uint32_t>(j);
            uint32_t outOffset = outBase + wo * vlNum;
            MicroAPI::LoadAlign(outReg, outAddr + outOffset);
            MicroAPI::Add(outReg, outReg, inputReg, preg);
            MicroAPI::StoreAlign(outAddr + outOffset, outReg, preg);
        }

        if constexpr (NC_FACTOR == TPL_NC_FACTOR_128) {
            uint32_t inputOffset1 = inputOffset + vfLenFp32;
            ops_vf::LoadOneTensorForDtypeT<T>(inputAddr, inputReg, preg, inputOffset1);
            for (uint16_t j = 0; j < woCount; j++) {
                uint32_t wo = woStart + static_cast<uint32_t>(j);
                uint32_t outOffset1 = outBase + wo * vlNum + vfLenFp32;
                MicroAPI::LoadAlign(outReg, outAddr + outOffset1);
                MicroAPI::Add(outReg, outReg, inputReg, preg);
                MicroAPI::StoreAlign(outAddr + outOffset1, outReg, preg);
            }
        }
    }
}

template <typename ID_T>
__simd_vf__ inline void UpsampleHCalWiToWoInfoVf(__ubuf__ int32_t* wiWoStartAddr, __ubuf__ int32_t* wiWoCountAddr,
                                                 uint16_t loopSize, uint32_t dataLen, uint16_t vfLen, int64_t wInDim,
                                                 int64_t wOutDim)
{
    AapIndexRegType<ID_T> startIdxReg;
    AapIndexRegType<ID_T> endIdxReg;
    AapIndexRegType<ID_T> countReg;
    AapIndexRegType<ID_T> dupReg;
    // See CalWKernelInfo: Pack narrows b64->b32 and requires an unsigned destination.
    MicroAPI::RegTensor<uint32_t> startDstReg;
    MicroAPI::RegTensor<uint32_t> countDstReg;
    MicroAPI::MaskReg calMask;

    MicroAPI::Duplicate(dupReg, static_cast<ID_T>(wInDim));
    for (uint16_t i = 0; i < loopSize; i++) {
        if constexpr (IsSameType<ID_T, int64_t>::value) {
            calMask = MicroAPI::UpdateMask<ID_T, MicroAPI::RegTraitNumTwo>(dataLen);
        } else {
            calMask = MicroAPI::UpdateMask<ID_T>(dataLen);
        }
        ID_T startIdx = i * vfLen;
        MicroAPI::Arange(startIdxReg, startIdx);
        MicroAPI::Adds(endIdxReg, startIdxReg, static_cast<ID_T>(1), calMask);
        MicroAPI::Muls(startIdxReg, startIdxReg, static_cast<ID_T>(wOutDim), calMask);
        MicroAPI::Muls(endIdxReg, endIdxReg, static_cast<ID_T>(wOutDim), calMask);
        MicroAPI::Adds(endIdxReg, endIdxReg, static_cast<ID_T>(wInDim - 1), calMask);
        MicroAPI::Div(startIdxReg, startIdxReg, dupReg, calMask);
        MicroAPI::Div(endIdxReg, endIdxReg, dupReg, calMask);
        MicroAPI::Sub(countReg, endIdxReg, startIdxReg, calMask);

        if constexpr (IsSameType<ID_T, int64_t>::value) {
            // Narrow b64 indices to b32; see CalWKernelInfo. Under RegTraitNumTwo
            // reg[0] already holds the low words of all 64 elements.
            MicroAPI::Pack<uint32_t, ID_T, MicroAPI::HighLowPart::LOWEST>(startDstReg, startIdxReg);
            MicroAPI::Pack<uint32_t, ID_T, MicroAPI::HighLowPart::LOWEST>(countDstReg, countReg);
            MicroAPI::StoreAlign((__ubuf__ uint32_t*)wiWoStartAddr + i * vfLen, startDstReg, calMask);
            MicroAPI::StoreAlign((__ubuf__ uint32_t*)wiWoCountAddr + i * vfLen, countDstReg, calMask);
        } else {
            MicroAPI::StoreAlign(wiWoStartAddr + i * vfLen, startIdxReg, calMask);
            MicroAPI::StoreAlign(wiWoCountAddr + i * vfLen, countReg, calMask);
        }
    }
}

__simd_vf__ inline void UpsampleHBroadcastRowVf(__ubuf__ float* outAddr, uint32_t srcOffset, uint32_t dstOffset,
                                                uint16_t loopSize, uint32_t remaining, uint32_t vfLenFp32)
{
    MicroAPI::RegTensor<float> rowReg;
    for (uint16_t i = 0; i < loopSize; i++) {
        MicroAPI::MaskReg mask = MicroAPI::UpdateMask<float>(remaining);
        MicroAPI::AddrReg offset = MicroAPI::CreateAddrReg<float>(i, vfLenFp32);
        MicroAPI::LoadAlign(rowReg, outAddr + srcOffset, offset);
        MicroAPI::StoreAlign(outAddr + dstOffset, rowReg, offset, mask);
    }
}

template <typename T, const uint32_t NC_FACTOR>
__simd_vf__ inline void UpsampleHWReduceAndScatterVf(__ubuf__ T* inputAddr, __ubuf__ float* outAddr,
                                                     __ubuf__ int32_t* wStartAddr, __ubuf__ int32_t* wKerSizeAddr,
                                                     uint32_t rowBase, uint16_t woNum, uint16_t hoCnt, uint32_t vlNum,
                                                     uint32_t wOutAlign, uint32_t vfLenFp32, int64_t hoStart)
{
    MicroAPI::RegTensor<float> inputReg;
    MicroAPI::RegTensor<float> sumReg;
    MicroAPI::RegTensor<float> outReg;
    MicroAPI::MaskReg preg = MicroAPI::CreateMask<float, MicroAPI::MaskPattern::ALL>();

    for (uint16_t wo = 0; wo < woNum; wo++) {
        uint32_t baseOffset = (rowBase + static_cast<uint32_t>(wStartAddr[wo])) * vlNum;
        uint16_t kernelW = static_cast<uint16_t>(wKerSizeAddr[wo]);

        MicroAPI::Duplicate(sumReg, 0.0f);
        for (uint16_t k = 0; k < kernelW; k++) {
            uint32_t inputOffset = baseOffset + static_cast<uint32_t>(k) * vlNum;
            ops_vf::LoadOneTensorForDtypeT<T>(inputAddr, inputReg, preg, inputOffset);
            MicroAPI::Add(sumReg, sumReg, inputReg, preg);
        }

        for (uint16_t j = 0; j < hoCnt; j++) {
            uint32_t outOffset = (static_cast<uint32_t>(hoStart) + static_cast<uint32_t>(j)) * wOutAlign * vlNum +
                                 static_cast<uint32_t>(wo) * vlNum;
            MicroAPI::LoadAlign(outReg, outAddr + outOffset);
            MicroAPI::Add(outReg, outReg, sumReg, preg);
            MicroAPI::StoreAlign(outAddr + outOffset, outReg, preg);
        }

        if constexpr (NC_FACTOR == TPL_NC_FACTOR_128) {
            MicroAPI::Duplicate(sumReg, 0.0f);
            for (uint16_t k = 0; k < kernelW; k++) {
                uint32_t inputOffset = baseOffset + static_cast<uint32_t>(k) * vlNum + vfLenFp32;
                ops_vf::LoadOneTensorForDtypeT<T>(inputAddr, inputReg, preg, inputOffset);
                MicroAPI::Add(sumReg, sumReg, inputReg, preg);
            }
            for (uint16_t j = 0; j < hoCnt; j++) {
                uint32_t outOffset = (static_cast<uint32_t>(hoStart) + static_cast<uint32_t>(j)) * wOutAlign * vlNum +
                                     static_cast<uint32_t>(wo) * vlNum + vfLenFp32;
                MicroAPI::LoadAlign(outReg, outAddr + outOffset);
                MicroAPI::Add(outReg, outReg, sumReg, preg);
                MicroAPI::StoreAlign(outAddr + outOffset, outReg, preg);
            }
        }
    }
}

template <typename T, const uint32_t NC_FACTOR>
__simd_vf__ inline void UpsampleHWReduceAndScatterBulkVf(__ubuf__ T* inputAddr, __ubuf__ float* outAddr,
                                                         uint32_t rowBase, uint32_t rowStride, uint16_t woNum,
                                                         uint16_t hoCnt, uint32_t vlNum, uint32_t vfLenFp32,
                                                         int64_t hoStart)
{
    MicroAPI::RegTensor<float> inputReg;
    MicroAPI::RegTensor<float> sumReg;
    MicroAPI::RegTensor<float> outReg;
    MicroAPI::MaskReg preg = MicroAPI::CreateMask<float, MicroAPI::MaskPattern::ALL>();

    for (uint16_t wo = 0; wo < woNum; wo++) {
        uint32_t inputOffset = (rowBase + static_cast<uint32_t>(wo)) * vlNum;

        MicroAPI::Duplicate(sumReg, 0.0f);
        ops_vf::LoadOneTensorForDtypeT<T>(inputAddr, inputReg, preg, inputOffset);
        MicroAPI::Add(sumReg, sumReg, inputReg, preg);
        ops_vf::LoadOneTensorForDtypeT<T>(inputAddr, inputReg, preg, inputOffset + vlNum);
        MicroAPI::Add(sumReg, sumReg, inputReg, preg);
        for (uint16_t j = 0; j < hoCnt; j++) {
            uint32_t outOffset = (static_cast<uint32_t>(hoStart) + static_cast<uint32_t>(j)) * rowStride +
                                 static_cast<uint32_t>(wo) * vlNum;
            MicroAPI::LoadAlign(outReg, outAddr + outOffset);
            MicroAPI::Add(outReg, outReg, sumReg, preg);
            MicroAPI::StoreAlign(outAddr + outOffset, outReg, preg);
        }

        if constexpr (NC_FACTOR == TPL_NC_FACTOR_128) {
            MicroAPI::Duplicate(sumReg, 0.0f);
            ops_vf::LoadOneTensorForDtypeT<T>(inputAddr, inputReg, preg, inputOffset + vfLenFp32);
            MicroAPI::Add(sumReg, sumReg, inputReg, preg);
            ops_vf::LoadOneTensorForDtypeT<T>(inputAddr, inputReg, preg, inputOffset + vlNum + vfLenFp32);
            MicroAPI::Add(sumReg, sumReg, inputReg, preg);
            for (uint16_t j = 0; j < hoCnt; j++) {
                uint32_t outOffset = (static_cast<uint32_t>(hoStart) + static_cast<uint32_t>(j)) * rowStride +
                                     static_cast<uint32_t>(wo) * vlNum + vfLenFp32;
                MicroAPI::LoadAlign(outReg, outAddr + outOffset);
                MicroAPI::Add(outReg, outReg, sumReg, preg);
                MicroAPI::StoreAlign(outAddr + outOffset, outReg, preg);
            }
        }
    }
}

template <typename T, typename ID_T, const uint32_t NC_FACTOR>
class AdaptiveAvgPool2dUpsampleH
    : protected AdaptiveAvgPool2dPoolingBase<T, ID_T, NC_FACTOR, AdaptivePool2dUpsampleHTilingData> {
    using Base = AdaptiveAvgPool2dPoolingBase<T, ID_T, NC_FACTOR, AdaptivePool2dUpsampleHTilingData>;

public:
    __aicore__ inline AdaptiveAvgPool2dUpsampleH(const AdaptivePool2dUpsampleHTilingData* tilingData, TPipe* pipe)
        : Base(tilingData, pipe){};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y);
    __aicore__ inline void Process();

private:
    __aicore__ inline void AccumulateW(int64_t rowOffset, int64_t hoLocal);
    __aicore__ inline void AccumulateWBulk(int64_t rowOffset, int64_t hoLocal);
    __aicore__ inline void AccumulateWUpsample(int64_t rowOffset, int64_t hoLocal);
    __aicore__ inline void WReduceAndScatter(int64_t rowOffset, int64_t hoStart, int64_t hoCount);
    __aicore__ inline void WReduceAndScatterBulk(int64_t rowOffset, int64_t hoStart, int64_t hoCount);
    __aicore__ inline void CalWiToWoInfo();
    __aicore__ inline void BroadcastOutBufRow(int64_t srcHoLocal, int64_t dstHoStart, int64_t dstHoCount);
    __aicore__ inline void ClearOutBufRows(int64_t hoStart, int64_t hoCount);
    __aicore__ inline void EarlyCastTransOut(int64_t hoNum);
    __aicore__ inline void EarlyCastCopyOut(int64_t ncIdx, int64_t ncNum, int64_t hoGlobal, int64_t hoNum);
    __aicore__ inline void CompactCopyOut(int64_t ncIdx, int64_t ncNum, int64_t compactHoNum, int64_t hoGlobalStart,
                                          int64_t hoNum);
    __aicore__ inline void CompactEarlyCastCopyOut(int64_t ncIdx, int64_t ncNum, int64_t compactHoNum,
                                                   int64_t hoGlobalStart, int64_t hoNum);
    __aicore__ inline void ProcessOneBlock(const BlockParam& blockPara);

    // ProcessOneBlock helpers. The block splits into two parallel strategies -- plain
    // (one outBuf row per ho) and compact (kernelH==1 runs share a row) -- each of which
    // scans input rows, averages, then writes out.
    __aicore__ inline void LoadHiBatchCached(int64_t ncIdx, int64_t ncNum, int64_t hiBase, int64_t hiBatch);
    __aicore__ inline void FindHoScatterRange(int64_t hi, int64_t hoGlobalStart, int64_t hoNum, int64_t& hoCursor,
                                              int64_t& hoScatterStart, int64_t& hoScatterCount) const;
    __aicore__ inline int64_t CountKh1Group(int64_t hoGlobalStart, int64_t hoLocal, int64_t hoNum) const;
    __aicore__ inline void ReduceRowToHo(int64_t rowOffset, int64_t hoLocal);
    __aicore__ inline void ReduceRowToHoRange(int64_t rowOffset, int64_t hoStart, int64_t hoCount);
    __aicore__ inline int64_t BuildCompactMap(int64_t hoGlobalStart, int64_t hoNum, int64_t* hoToCompact) const;
    __aicore__ inline void ProcessBlockPlain(const BlockParam& blockPara, int64_t hiMin, int64_t hiMax);
    __aicore__ inline void ProcessBlockCompact(const BlockParam& blockPara, int64_t hiMin, int64_t hiMax,
                                               const int64_t* hoToCompact, int64_t compactHoNum);

    TBuf<QuePosition::VECCALC> wiWoStartBuf_;
    TBuf<QuePosition::VECCALC> wiWoCountBuf_;
    bool isWUpsample_ = false;
    bool isBulkWReduce_ = false;
    int64_t cachedNcIdx_ = -1;
    int64_t cachedHiBase_ = -1;
    int64_t cachedHiBatch_ = 0;
};

template <typename T, typename ID_T, const uint32_t NC_FACTOR>
__aicore__ inline void AdaptiveAvgPool2dUpsampleH<T, ID_T, NC_FACTOR>::Init(GM_ADDR x, GM_ADDR y)
{
    if (!this->InitCommon(x, y)) {
        return;
    }
    isWUpsample_ = (this->tilingData_->wOut > this->tilingData_->wIn);
    isBulkWReduce_ = IsSameType<T, float>::value && !isWUpsample_ &&
                     (this->tilingData_->wIn == this->tilingData_->wOut + 1);
    uint64_t vfLenI32 = AAP_V_REG_SIZE / sizeof(int32_t);
    uint64_t wBufSize = ops::CeilAlign(static_cast<uint64_t>(this->tilingData_->wOut), vfLenI32) * sizeof(int32_t);
    uint64_t transRowAlign = ops::CeilAlign(static_cast<uint64_t>(this->tilingData_->hiFactor) * this->wInAlign_,
                                            static_cast<uint64_t>(AAP_TRANS_ADDR_LEN));
    uint64_t transBufSize = transRowAlign * this->vlNum_ * sizeof(T);
    uint64_t outRowAlign = ops::CeilAlign(static_cast<uint64_t>(this->tilingData_->hoFactor) * this->wOutAlign_,
                                          static_cast<uint64_t>(AAP_TRANS_ADDR_LEN));
    uint64_t outBufSize = outRowAlign * this->vlNum_ * sizeof(float);
    this->pipe_->InitBuffer(this->inputQue_, 1, this->tilingData_->inputQueSize);
    this->pipe_->InitBuffer(this->resQue1_, 1, this->tilingData_->resQue1Size);
    if (this->tilingData_->resQue2Size > 0) {
        this->pipe_->InitBuffer(this->resQue2_, 1, this->tilingData_->resQue2Size);
    }
    this->pipe_->InitBuffer(this->transBuf_, transBufSize);
    this->pipe_->InitBuffer(this->outBuf_, outBufSize);
    this->pipe_->InitBuffer(this->wStartBuf_, wBufSize);
    this->pipe_->InitBuffer(this->wKerSizeBuf_, wBufSize);
    if (isWUpsample_) {
        uint64_t dataBlock = AAP_UB_BLOCK_SIZE;
        uint64_t wiBufSize = ops::CeilAlign(static_cast<uint64_t>(this->tilingData_->wIn) * sizeof(int32_t), dataBlock);
        this->pipe_->InitBuffer(wiWoStartBuf_, wiBufSize);
        this->pipe_->InitBuffer(wiWoCountBuf_, wiBufSize);
    }
}

template <typename T, typename ID_T, const uint32_t NC_FACTOR>
__aicore__ inline void AdaptiveAvgPool2dUpsampleH<T, ID_T, NC_FACTOR>::AccumulateW(int64_t rowOffset, int64_t hoLocal)
{
    LocalTensor<T> transLocal = this->transBuf_.template Get<T>();
    __ubuf__ T* inputAddr = (__ubuf__ T*)transLocal.GetPhyAddr();
    LocalTensor<float> outLocal = this->outBuf_.template Get<float>();
    __ubuf__ float* outAddr = (__ubuf__ float*)outLocal.GetPhyAddr();
    LocalTensor<int32_t> wStartLocal = this->wStartBuf_.template Get<int32_t>();
    __ubuf__ int32_t* wStartAddr = (__ubuf__ int32_t*)wStartLocal.GetPhyAddr();
    LocalTensor<int32_t> wKerSizeLocal = this->wKerSizeBuf_.template Get<int32_t>();
    __ubuf__ int32_t* wKerSizeAddr = (__ubuf__ int32_t*)wKerSizeLocal.GetPhyAddr();

    uint32_t vfLenFp32 = AAP_V_REG_SIZE / sizeof(float);
    uint32_t rowBase = static_cast<uint32_t>(rowOffset);
    uint32_t outBase = static_cast<uint32_t>(hoLocal) * this->wOutAlign_ * this->vlNum_;
    uint16_t woNum = static_cast<uint16_t>(this->tilingData_->wOut);

    UpsampleHAccumulateWVf<T, NC_FACTOR>(inputAddr, outAddr, wStartAddr, wKerSizeAddr, rowBase, outBase, woNum,
                                         this->vlNum_, vfLenFp32);
}

template <typename T, typename ID_T, const uint32_t NC_FACTOR>
__aicore__ inline void AdaptiveAvgPool2dUpsampleH<T, ID_T, NC_FACTOR>::AccumulateWBulk(int64_t rowOffset,
                                                                                       int64_t hoLocal)
{
    if constexpr (IsSameType<T, float>::value) {
        LocalTensor<T> transLocal = this->transBuf_.template Get<T>();
        __ubuf__ T* inputAddr = (__ubuf__ T*)transLocal.GetPhyAddr();
        LocalTensor<float> outLocal = this->outBuf_.template Get<float>();
        __ubuf__ float* outAddr = (__ubuf__ float*)outLocal.GetPhyAddr();

        uint32_t vfLenFp32 = AAP_V_REG_SIZE / sizeof(float);
        uint32_t rowBase = static_cast<uint32_t>(rowOffset);
        uint32_t outBase = static_cast<uint32_t>(hoLocal) * this->wOutAlign_ * this->vlNum_;
        uint16_t woNum = static_cast<uint16_t>(this->tilingData_->wOut);

        UpsampleHAccumulateWBulkVf<T, NC_FACTOR>(inputAddr, outAddr, rowBase, outBase, woNum, this->vlNum_, vfLenFp32);
    } else {
        AccumulateW(rowOffset, hoLocal);
    }
}

// W-upsampling: for each wi, scatter its value to all covering wo positions, then add to outBuf[hoLocal].
template <typename T, typename ID_T, const uint32_t NC_FACTOR>
__aicore__ inline void AdaptiveAvgPool2dUpsampleH<T, ID_T, NC_FACTOR>::AccumulateWUpsample(int64_t rowOffset,
                                                                                           int64_t hoLocal)
{
    LocalTensor<T> transLocal = this->transBuf_.template Get<T>();
    __ubuf__ T* inputAddr = (__ubuf__ T*)transLocal.GetPhyAddr();
    LocalTensor<float> outLocal = this->outBuf_.template Get<float>();
    __ubuf__ float* outAddr = (__ubuf__ float*)outLocal.GetPhyAddr();
    LocalTensor<int32_t> wiWoStartLocal = wiWoStartBuf_.template Get<int32_t>();
    __ubuf__ int32_t* wiWoStartAddr = (__ubuf__ int32_t*)wiWoStartLocal.GetPhyAddr();
    LocalTensor<int32_t> wiWoCountLocal = wiWoCountBuf_.template Get<int32_t>();
    __ubuf__ int32_t* wiWoCountAddr = (__ubuf__ int32_t*)wiWoCountLocal.GetPhyAddr();

    uint32_t vfLenFp32 = AAP_V_REG_SIZE / sizeof(float);
    uint32_t rowBase = static_cast<uint32_t>(rowOffset);
    uint32_t outBase = static_cast<uint32_t>(hoLocal) * this->wOutAlign_ * this->vlNum_;
    uint16_t wiNum = static_cast<uint16_t>(this->tilingData_->wIn);

    UpsampleHAccumulateWUpsampleVf<T, NC_FACTOR>(inputAddr, outAddr, wiWoStartAddr, wiWoCountAddr, rowBase, outBase,
                                                 wiNum, this->vlNum_, vfLenFp32);
}

template <typename T, typename ID_T, const uint32_t NC_FACTOR>
__aicore__ inline void AdaptiveAvgPool2dUpsampleH<T, ID_T, NC_FACTOR>::CalWiToWoInfo()
{
    LocalTensor<int32_t> wiWoStartLocal = wiWoStartBuf_.template Get<int32_t>();
    LocalTensor<int32_t> wiWoCountLocal = wiWoCountBuf_.template Get<int32_t>();
    __ubuf__ int32_t* wiWoStartAddr = (__ubuf__ int32_t*)wiWoStartLocal.GetPhyAddr();
    __ubuf__ int32_t* wiWoCountAddr = (__ubuf__ int32_t*)wiWoCountLocal.GetPhyAddr();

    int32_t wIn = this->tilingData_->wIn;
    int64_t wOutDim = this->tilingData_->wOut;
    int64_t wInDim = this->tilingData_->wIn;
    uint16_t vfLen = AAP_V_REG_SIZE / sizeof(int32_t);
    uint16_t loopSize = ops::CeilDiv(static_cast<uint16_t>(wIn), vfLen);
    uint32_t dataLen = wIn;

    UpsampleHCalWiToWoInfoVf<ID_T>(wiWoStartAddr, wiWoCountAddr, loopSize, dataLen, vfLen, wInDim, wOutDim);
}

template <typename T, typename ID_T, const uint32_t NC_FACTOR>
__aicore__ inline void AdaptiveAvgPool2dUpsampleH<T, ID_T, NC_FACTOR>::BroadcastOutBufRow(int64_t srcHoLocal,
                                                                                          int64_t dstHoStart,
                                                                                          int64_t dstHoCount)
{
    LocalTensor<float> outLocal = this->outBuf_.template Get<float>();
    __ubuf__ float* outAddr = (__ubuf__ float*)outLocal.GetPhyAddr();
    uint32_t rowElems = this->wOutAlign_ * this->vlNum_;
    uint32_t srcOffset = static_cast<uint32_t>(srcHoLocal) * rowElems;

    // Plain row copy in the RegBase idiom. The previous Adds(src, 0.0f) form also
    // normalised -0.0f to +0.0f; a load/store pair preserves the sign bit.
    uint32_t vfLenFp32 = AAP_V_REG_SIZE / sizeof(float);
    uint16_t loopSize = ops::CeilDiv(rowElems, vfLenFp32);

    for (int64_t j = 0; j < dstHoCount; j++) {
        uint32_t dstOffset = static_cast<uint32_t>(dstHoStart + j) * rowElems;
        uint32_t remaining = rowElems;
        UpsampleHBroadcastRowVf(outAddr, srcOffset, dstOffset, loopSize, remaining, vfLenFp32);
    }
}

template <typename T, typename ID_T, const uint32_t NC_FACTOR>
__aicore__ inline void AdaptiveAvgPool2dUpsampleH<T, ID_T, NC_FACTOR>::ClearOutBufRows(int64_t hoStart, int64_t hoCount)
{
    LocalTensor<float> outLocal = this->outBuf_.template Get<float>();
    __ubuf__ float* outAddr = (__ubuf__ float*)outLocal.GetPhyAddr();
    uint32_t rowElems = this->wOutAlign_ * this->vlNum_;
    uint32_t startOffset = static_cast<uint32_t>(hoStart) * rowElems;
    uint32_t totalElems = static_cast<uint32_t>(hoCount) * rowElems;

    // Identical to the base ClearOutBuf loop, just restricted to [hoStart, hoStart+hoCount),
    // so reuse its VF instead of emitting a second copy.
    uint32_t vfLenFp32 = AAP_V_REG_SIZE / sizeof(float);
    uint16_t loopSize = ops::CeilDiv(totalElems, vfLenFp32);
    uint32_t remaining = totalElems;

    ClearOutBufVf(outAddr + startOffset, loopSize, remaining, vfLenFp32);
}

// Fused W-reduction + scatter: keeps sumReg in register across Ho iterations.
template <typename T, typename ID_T, const uint32_t NC_FACTOR>
__aicore__ inline void AdaptiveAvgPool2dUpsampleH<T, ID_T, NC_FACTOR>::WReduceAndScatter(int64_t rowOffset,
                                                                                         int64_t hoStart,
                                                                                         int64_t hoCount)
{
    LocalTensor<int32_t> wKerSizeLocal = this->wKerSizeBuf_.template Get<int32_t>();
    __ubuf__ int32_t* wKerSizeAddr = (__ubuf__ int32_t*)wKerSizeLocal.GetPhyAddr();
    LocalTensor<int32_t> wStartLocal = this->wStartBuf_.template Get<int32_t>();
    __ubuf__ int32_t* wStartAddr = (__ubuf__ int32_t*)wStartLocal.GetPhyAddr();
    LocalTensor<float> outLocal = this->outBuf_.template Get<float>();
    __ubuf__ float* outAddr = (__ubuf__ float*)outLocal.GetPhyAddr();
    LocalTensor<T> transLocal = this->transBuf_.template Get<T>();
    __ubuf__ T* inputAddr = (__ubuf__ T*)transLocal.GetPhyAddr();

    uint32_t vfLenFp32 = AAP_V_REG_SIZE / sizeof(float);
    uint32_t rowBase = static_cast<uint32_t>(rowOffset);
    uint16_t woNum = static_cast<uint16_t>(this->tilingData_->wOut);
    uint16_t hoCnt = static_cast<uint16_t>(hoCount);

    UpsampleHWReduceAndScatterVf<T, NC_FACTOR>(inputAddr, outAddr, wStartAddr, wKerSizeAddr, rowBase, woNum, hoCnt,
                                               this->vlNum_, this->wOutAlign_, vfLenFp32, hoStart);
}

template <typename T, typename ID_T, const uint32_t NC_FACTOR>
__aicore__ inline void AdaptiveAvgPool2dUpsampleH<T, ID_T, NC_FACTOR>::WReduceAndScatterBulk(int64_t rowOffset,
                                                                                             int64_t hoStart,
                                                                                             int64_t hoCount)
{
    if constexpr (IsSameType<T, float>::value) {
        LocalTensor<T> transLocal = this->transBuf_.template Get<T>();
        __ubuf__ T* inputAddr = (__ubuf__ T*)transLocal.GetPhyAddr();
        LocalTensor<float> outLocal = this->outBuf_.template Get<float>();
        __ubuf__ float* outAddr = (__ubuf__ float*)outLocal.GetPhyAddr();

        uint32_t vfLenFp32 = AAP_V_REG_SIZE / sizeof(float);
        uint32_t rowBase = static_cast<uint32_t>(rowOffset);
        uint32_t rowStride = this->wOutAlign_ * this->vlNum_;
        uint16_t woNum = static_cast<uint16_t>(this->tilingData_->wOut);
        uint16_t hoCnt = static_cast<uint16_t>(hoCount);

        UpsampleHWReduceAndScatterBulkVf<T, NC_FACTOR>(inputAddr, outAddr, rowBase, rowStride, woNum, hoCnt,
                                                       this->vlNum_, vfLenFp32, hoStart);
    } else {
        WReduceAndScatter(rowOffset, hoStart, hoCount);
    }
}

template <typename T, typename ID_T, const uint32_t NC_FACTOR>
__aicore__ inline void AdaptiveAvgPool2dUpsampleH<T, ID_T, NC_FACTOR>::EarlyCastTransOut(int64_t hoNum)
{
    int64_t rowNum = hoNum * this->wOutAlign_;
    uint64_t rowNumAlign = ops::CeilAlign(static_cast<uint64_t>(rowNum), static_cast<uint64_t>(AAP_TRANS_ADDR_LEN));
    uint64_t castLen = rowNumAlign * this->vlNum_;

    LocalTensor<float> srcFp32 = this->outBuf_.template Get<float>();
    LocalTensor<T> srcT = srcFp32.template ReinterpretCast<T>();
    if constexpr (IsSameType<T, half>::value) {
        Cast(srcT, srcFp32, RoundMode::CAST_NONE, castLen);
    } else {
        Cast(srcT, srcFp32, RoundMode::CAST_RINT, castLen);
    }

    LocalTensor<T> dstLocal = this->resQue1_.template AllocTensor<T>();
    this->TransposeB16(dstLocal, srcT, rowNumAlign, this->vlNum_);
    this->resQue1_.EnQue(dstLocal);
}

template <typename T, typename ID_T, const uint32_t NC_FACTOR>
__aicore__ inline void AdaptiveAvgPool2dUpsampleH<T, ID_T, NC_FACTOR>::EarlyCastCopyOut(int64_t ncIdx, int64_t ncNum,
                                                                                        int64_t hoGlobal, int64_t hoNum)
{
    uint64_t hwOutStride = ops::CeilAlign(static_cast<uint64_t>(hoNum * this->wOutAlign_),
                                          static_cast<uint64_t>(AAP_TRANS_ADDR_LEN));
    int64_t ncBase = ncIdx * this->tilingData_->ncFactor;

    LocalTensor<T> resOutLocal = this->resQue1_.template DeQue<T>();

    DataCopyExtParams valueParams;
    valueParams.blockCount = 1;
    valueParams.blockLen = static_cast<uint32_t>(this->tilingData_->wOut * sizeof(T));
    valueParams.srcStride = 0;
    valueParams.dstStride = 0;

    LoopModeParams loopModeParams;
    loopModeParams.loop1Size = ncNum;
    loopModeParams.loop2Size = 1;
    loopModeParams.loop1SrcStride = hwOutStride * sizeof(T);
    loopModeParams.loop2SrcStride = 0;
    loopModeParams.loop1DstStride = this->outHW_ * sizeof(T);
    loopModeParams.loop2DstStride = 0;

    SetLoopModePara(loopModeParams, DataCopyMVType::UB_TO_OUT);
    for (int64_t ho = 0; ho < hoNum; ho++) {
        int64_t yOff = ncBase * this->outHW_ + (hoGlobal + ho) * this->tilingData_->wOut;
        uint64_t srcOff = static_cast<uint64_t>(ho) * this->wOutAlign_;
        DataCopyPad(this->yGm_[yOff], resOutLocal[srcOff], valueParams);
    }
    ResetLoopModePara(DataCopyMVType::UB_TO_OUT);
    this->resQue1_.FreeTensor(resOutLocal);
}

template <typename T, typename ID_T, const uint32_t NC_FACTOR>
__aicore__ inline void AdaptiveAvgPool2dUpsampleH<T, ID_T, NC_FACTOR>::CompactCopyOut(int64_t ncIdx, int64_t ncNum,
                                                                                      int64_t compactHoNum,
                                                                                      int64_t hoGlobalStart,
                                                                                      int64_t hoNum)
{
    int64_t hIn = this->tilingData_->hIn;
    int64_t hOut = this->tilingData_->hOut;
    uint64_t hwOutStride = ops::CeilAlign(static_cast<uint64_t>(compactHoNum * this->wOutAlign_),
                                          static_cast<uint64_t>(AAP_TRANS_ADDR_LEN));
    int64_t ncBase = ncIdx * this->tilingData_->ncFactor;

    LocalTensor<float> resOutLocal = this->resQue1_.template DeQue<float>();

    DataCopyExtParams valueParams;
    valueParams.blockCount = 1;
    valueParams.blockLen = static_cast<uint32_t>(this->tilingData_->wOut * sizeof(T));
    valueParams.srcStride = 0;
    valueParams.dstStride = 0;

    LoopModeParams loopModeParams;
    loopModeParams.loop1Size = ncNum;
    loopModeParams.loop2Size = 1;
    loopModeParams.loop1SrcStride = hwOutStride * sizeof(T);
    loopModeParams.loop2SrcStride = 0;
    loopModeParams.loop1DstStride = this->outHW_ * sizeof(T);
    loopModeParams.loop2DstStride = 0;

    if constexpr (IsSameType<T, float>::value) {
        SetLoopModePara(loopModeParams, DataCopyMVType::UB_TO_OUT);
        int64_t ci = 0;
        int64_t prevHStart = -1;
        int64_t prevKH = -1;
        for (int64_t ho = 0; ho < hoNum; ho++) {
            int64_t hoGlobal = hoGlobalStart + ho;
            int64_t hStart = (hoGlobal * hIn) / hOut;
            int64_t kH = ((hoGlobal + 1) * hIn + hOut - 1) / hOut - hStart;
            if (ho > 0 && (kH != 1 || prevKH != 1 || hStart != prevHStart)) {
                ci++;
            }
            uint64_t srcOff = static_cast<uint64_t>(ci) * this->wOutAlign_;
            int64_t yOff = ncBase * this->outHW_ + hoGlobal * this->tilingData_->wOut;
            DataCopyPad(this->yGm_[yOff], resOutLocal[srcOff], valueParams);
            prevHStart = hStart;
            prevKH = kH;
        }
        ResetLoopModePara(DataCopyMVType::UB_TO_OUT);
    } else {
        LocalTensor<T> castOutLocal = this->resQue2_.template AllocTensor<T>();
        if constexpr (IsSameType<T, half>::value) {
            Cast(castOutLocal, resOutLocal, RoundMode::CAST_NONE, ncNum * hwOutStride);
        } else {
            Cast(castOutLocal, resOutLocal, RoundMode::CAST_RINT, ncNum * hwOutStride);
        }
        this->resQue2_.EnQue(castOutLocal);
        castOutLocal = this->resQue2_.template DeQue<T>();
        SetLoopModePara(loopModeParams, DataCopyMVType::UB_TO_OUT);
        int64_t ci = 0;
        int64_t prevHStart = -1;
        int64_t prevKH = -1;
        for (int64_t ho = 0; ho < hoNum; ho++) {
            int64_t hoGlobal = hoGlobalStart + ho;
            int64_t hStart = (hoGlobal * hIn) / hOut;
            int64_t kH = ((hoGlobal + 1) * hIn + hOut - 1) / hOut - hStart;
            if (ho > 0 && (kH != 1 || prevKH != 1 || hStart != prevHStart)) {
                ci++;
            }
            uint64_t srcOff = static_cast<uint64_t>(ci) * this->wOutAlign_;
            int64_t yOff = ncBase * this->outHW_ + hoGlobal * this->tilingData_->wOut;
            DataCopyPad(this->yGm_[yOff], castOutLocal[srcOff], valueParams);
            prevHStart = hStart;
            prevKH = kH;
        }
        ResetLoopModePara(DataCopyMVType::UB_TO_OUT);
        this->resQue2_.FreeTensor(castOutLocal);
    }
    this->resQue1_.FreeTensor(resOutLocal);
}

template <typename T, typename ID_T, const uint32_t NC_FACTOR>
__aicore__ inline void AdaptiveAvgPool2dUpsampleH<T, ID_T, NC_FACTOR>::CompactEarlyCastCopyOut(
    int64_t ncIdx, int64_t ncNum, int64_t compactHoNum, int64_t hoGlobalStart, int64_t hoNum)
{
    int64_t hIn = this->tilingData_->hIn;
    int64_t hOut = this->tilingData_->hOut;
    uint64_t hwOutStride = ops::CeilAlign(static_cast<uint64_t>(compactHoNum * this->wOutAlign_),
                                          static_cast<uint64_t>(AAP_TRANS_ADDR_LEN));
    int64_t ncBase = ncIdx * this->tilingData_->ncFactor;

    LocalTensor<T> resOutLocal = this->resQue1_.template DeQue<T>();

    DataCopyExtParams valueParams;
    valueParams.blockCount = 1;
    valueParams.blockLen = static_cast<uint32_t>(this->tilingData_->wOut * sizeof(T));
    valueParams.srcStride = 0;
    valueParams.dstStride = 0;

    LoopModeParams loopModeParams;
    loopModeParams.loop1Size = ncNum;
    loopModeParams.loop2Size = 1;
    loopModeParams.loop1SrcStride = hwOutStride * sizeof(T);
    loopModeParams.loop2SrcStride = 0;
    loopModeParams.loop1DstStride = this->outHW_ * sizeof(T);
    loopModeParams.loop2DstStride = 0;

    SetLoopModePara(loopModeParams, DataCopyMVType::UB_TO_OUT);
    int64_t ci = 0;
    int64_t prevHStart = -1;
    int64_t prevKH = -1;
    for (int64_t ho = 0; ho < hoNum; ho++) {
        int64_t hoGlobal = hoGlobalStart + ho;
        int64_t hStart = (hoGlobal * hIn) / hOut;
        int64_t kH = ((hoGlobal + 1) * hIn + hOut - 1) / hOut - hStart;
        if (ho > 0 && (kH != 1 || prevKH != 1 || hStart != prevHStart)) {
            ci++;
        }
        uint64_t srcOff = static_cast<uint64_t>(ci) * this->wOutAlign_;
        int64_t yOff = ncBase * this->outHW_ + hoGlobal * this->tilingData_->wOut;
        DataCopyPad(this->yGm_[yOff], resOutLocal[srcOff], valueParams);
        prevHStart = hStart;
        prevKH = kH;
    }
    ResetLoopModePara(DataCopyMVType::UB_TO_OUT);
    this->resQue1_.FreeTensor(resOutLocal);
}

// CopyIn+Transpose the [hiBase, hiBase+hiBatch) rows unless they are already resident.
template <typename T, typename ID_T, const uint32_t NC_FACTOR>
__aicore__ inline void AdaptiveAvgPool2dUpsampleH<T, ID_T, NC_FACTOR>::LoadHiBatchCached(int64_t ncIdx, int64_t ncNum,
                                                                                         int64_t hiBase,
                                                                                         int64_t hiBatch)
{
    if (hiBase == cachedHiBase_ && hiBatch == cachedHiBatch_) {
        return;
    }
    this->CopyInputBatch(ncIdx, ncNum, hiBase, hiBatch);
    this->TransInputBatch(hiBatch);
    cachedHiBase_ = hiBase;
    cachedHiBatch_ = hiBatch;
}

// Advance hoCursor past the Ho windows that already ended, then count the ones covering hi.
// Ho windows are monotonic in hi, so the cursor never moves backwards across a block.
template <typename T, typename ID_T, const uint32_t NC_FACTOR>
__aicore__ inline void AdaptiveAvgPool2dUpsampleH<T, ID_T, NC_FACTOR>::FindHoScatterRange(
    int64_t hi, int64_t hoGlobalStart, int64_t hoNum, int64_t& hoCursor, int64_t& hoScatterStart,
    int64_t& hoScatterCount) const
{
    while (hoCursor < hoNum && this->CalHoEnd(hoGlobalStart + hoCursor) <= hi) {
        hoCursor++;
    }
    hoScatterStart = hoCursor;
    hoScatterCount = 0;
    for (int64_t hoLocal = hoCursor; hoLocal < hoNum; hoLocal++) {
        if (this->CalHoStart(hoGlobalStart + hoLocal) > hi) {
            break;
        }
        hoScatterCount++;
    }
}

// Length of the run of consecutive Ho starting at hoLocal that share one input row (all kernelH==1).
template <typename T, typename ID_T, const uint32_t NC_FACTOR>
__aicore__ inline int64_t AdaptiveAvgPool2dUpsampleH<T, ID_T, NC_FACTOR>::CountKh1Group(int64_t hoGlobalStart,
                                                                                        int64_t hoLocal,
                                                                                        int64_t hoNum) const
{
    int64_t hStart = this->CalHoStart(hoGlobalStart + hoLocal);
    int64_t groupCount = 1;
    for (int64_t j = hoLocal + 1; j < hoNum; j++) {
        int64_t hoG = hoGlobalStart + j;
        if (this->CalHoStart(hoG) == hStart && this->CalKernelH(hoG) == 1) {
            groupCount++;
        } else {
            break;
        }
    }
    return groupCount;
}

// W-reduce one input row into a single Ho accumulator, picking the W variant for this shape.
template <typename T, typename ID_T, const uint32_t NC_FACTOR>
__aicore__ inline void AdaptiveAvgPool2dUpsampleH<T, ID_T, NC_FACTOR>::ReduceRowToHo(int64_t rowOffset, int64_t hoLocal)
{
    if (isWUpsample_) {
        AccumulateWUpsample(rowOffset, hoLocal);
    } else if (isBulkWReduce_) {
        AccumulateWBulk(rowOffset, hoLocal);
    } else {
        AccumulateW(rowOffset, hoLocal);
    }
}

// Same as ReduceRowToHo but spread over hoCount consecutive accumulators. The non-upsample
// paths reduce once and scatter; W upsampling has no shared intermediate, so it loops.
template <typename T, typename ID_T, const uint32_t NC_FACTOR>
__aicore__ inline void AdaptiveAvgPool2dUpsampleH<T, ID_T, NC_FACTOR>::ReduceRowToHoRange(int64_t rowOffset,
                                                                                          int64_t hoStart,
                                                                                          int64_t hoCount)
{
    if (isWUpsample_) {
        for (int64_t j = 0; j < hoCount; j++) {
            AccumulateWUpsample(rowOffset, hoStart + j);
        }
    } else if (isBulkWReduce_) {
        WReduceAndScatterBulk(rowOffset, hoStart, hoCount);
    } else {
        WReduceAndScatter(rowOffset, hoStart, hoCount);
    }
}

// Map each Ho to an outBuf row, collapsing runs of kernelH==1 Ho that read the same input row
// into one shared row. Returns the compact row count (== hoNum when nothing can be shared).
template <typename T, typename ID_T, const uint32_t NC_FACTOR>
__aicore__ inline int64_t AdaptiveAvgPool2dUpsampleH<T, ID_T, NC_FACTOR>::BuildCompactMap(int64_t hoGlobalStart,
                                                                                          int64_t hoNum,
                                                                                          int64_t* hoToCompact) const
{
    int64_t compactHoNum = 0;
    int64_t prevHStart = -1;
    int64_t prevKH = -1;
    for (int64_t ho = 0; ho < hoNum; ho++) {
        int64_t hoGlobal = hoGlobalStart + ho;
        int64_t hStart = this->CalHoStart(hoGlobal);
        int64_t kH = this->CalKernelH(hoGlobal);
        if (ho == 0 || kH != 1 || prevKH != 1 || hStart != prevHStart) {
            compactHoNum++;
        }
        hoToCompact[ho] = compactHoNum - 1;
        prevHStart = hStart;
        prevKH = kH;
    }
    return compactHoNum;
}

// One outBuf row per Ho.
template <typename T, typename ID_T, const uint32_t NC_FACTOR>
__aicore__ inline void AdaptiveAvgPool2dUpsampleH<T, ID_T, NC_FACTOR>::ProcessBlockPlain(const BlockParam& blockPara,
                                                                                         int64_t hiMin, int64_t hiMax)
{
    int64_t ncIdx = blockPara.ncIdx;
    int64_t ncNum = blockPara.ncNum;
    int64_t hoNum = blockPara.hoNum;
    int64_t hoGlobalStart = blockPara.hoIdx * this->tilingData_->hoFactor;

    int64_t hoCursor = 0;
    int64_t hoCleared = 0;
    for (int64_t hiBase = hiMin; hiBase < hiMax; hiBase += this->tilingData_->hiFactor) {
        int64_t hiBatch = (hiBase + this->tilingData_->hiFactor > hiMax) ? (hiMax - hiBase) :
                                                                           this->tilingData_->hiFactor;
        LoadHiBatchCached(ncIdx, ncNum, hiBase, hiBatch);

        for (int64_t hiOff = 0; hiOff < hiBatch; hiOff++) {
            int64_t hoScatterStart = 0;
            int64_t hoScatterCount = 0;
            FindHoScatterRange(hiBase + hiOff, hoGlobalStart, hoNum, hoCursor, hoScatterStart, hoScatterCount);
            if (hoScatterCount == 0) {
                continue;
            }

            // Clear accumulators lazily, the first time a row is scattered into.
            int64_t scatterEnd = hoScatterStart + hoScatterCount;
            if (scatterEnd > hoCleared) {
                int64_t clearStart = (hoCleared > hoScatterStart) ? hoCleared : hoScatterStart;
                ClearOutBufRows(clearStart, scatterEnd - clearStart);
                hoCleared = scatterEnd;
            }

            int64_t rowOffset = hiOff * this->wInAlign_;
            if (hoScatterCount == 1) {
                ReduceRowToHo(rowOffset, hoScatterStart);
            } else if (this->CalKernelH(hoGlobalStart + hoScatterStart) == 1) {
                // Every covered Ho reads only this row, so reduce once and broadcast.
                ReduceRowToHo(rowOffset, hoScatterStart);
                BroadcastOutBufRow(hoScatterStart, hoScatterStart + 1, hoScatterCount - 1);
            } else {
                ReduceRowToHoRange(rowOffset, hoScatterStart, hoScatterCount);
            }
        }
    }

    for (int64_t hoLocal = 0; hoLocal < hoNum;) {
        int64_t kernelH = this->CalKernelH(hoGlobalStart + hoLocal);
        this->CalAvgOneHo(kernelH, hoLocal);
        int64_t groupCount = 1;
        if (kernelH == 1) {
            groupCount = CountKh1Group(hoGlobalStart, hoLocal, hoNum);
            if (groupCount > 1) {
                BroadcastOutBufRow(hoLocal, hoLocal + 1, groupCount - 1);
            }
        }
        hoLocal += groupCount;
    }

    if constexpr (IsSameType<T, float>::value) {
        this->TransOut(hoNum);
        this->CopyOut(ncIdx, ncNum, hoGlobalStart, hoNum);
    } else {
        EarlyCastTransOut(hoNum);
        EarlyCastCopyOut(ncIdx, ncNum, hoGlobalStart, hoNum);
    }
}

// Runs of kernelH==1 Ho share one outBuf row, so the averaged row is expanded only at CopyOut.
template <typename T, typename ID_T, const uint32_t NC_FACTOR>
__aicore__ inline void AdaptiveAvgPool2dUpsampleH<T, ID_T, NC_FACTOR>::ProcessBlockCompact(const BlockParam& blockPara,
                                                                                           int64_t hiMin, int64_t hiMax,
                                                                                           const int64_t* hoToCompact,
                                                                                           int64_t compactHoNum)
{
    int64_t ncIdx = blockPara.ncIdx;
    int64_t ncNum = blockPara.ncNum;
    int64_t hoNum = blockPara.hoNum;
    int64_t hoGlobalStart = blockPara.hoIdx * this->tilingData_->hoFactor;

    int64_t hoCursor = 0;
    int64_t compactCleared = 0;
    for (int64_t hiBase = hiMin; hiBase < hiMax; hiBase += this->tilingData_->hiFactor) {
        int64_t hiBatch = (hiBase + this->tilingData_->hiFactor > hiMax) ? (hiMax - hiBase) :
                                                                           this->tilingData_->hiFactor;
        LoadHiBatchCached(ncIdx, ncNum, hiBase, hiBatch);

        for (int64_t hiOff = 0; hiOff < hiBatch; hiOff++) {
            int64_t hoScatterStart = 0;
            int64_t hoScatterCount = 0;
            FindHoScatterRange(hiBase + hiOff, hoGlobalStart, hoNum, hoCursor, hoScatterStart, hoScatterCount);
            if (hoScatterCount == 0) {
                continue;
            }

            int64_t cStart = hoToCompact[hoScatterStart];
            int64_t cEnd = hoToCompact[hoScatterStart + hoScatterCount - 1] + 1;
            if (cEnd > compactCleared) {
                int64_t clearFrom = (compactCleared > cStart) ? compactCleared : cStart;
                ClearOutBufRows(clearFrom, cEnd - clearFrom);
                compactCleared = cEnd;
            }

            int64_t rowOffset = hiOff * this->wInAlign_;
            int64_t compactCount = cEnd - cStart;
            if (compactCount > 1) {
                ReduceRowToHoRange(rowOffset, cStart, compactCount);
            } else {
                ReduceRowToHo(rowOffset, cStart);
            }
        }
    }

    // Average each compact row once, using the kernelH of the Ho run it represents.
    int64_t ci = 0;
    for (int64_t ho = 0; ho < hoNum;) {
        int64_t kH = this->CalKernelH(hoGlobalStart + ho);
        this->CalAvgOneHo(kH, ci);
        int64_t groupCount = (kH == 1) ? CountKh1Group(hoGlobalStart, ho, hoNum) : 1;
        ho += groupCount;
        ci++;
    }

    if constexpr (IsSameType<T, float>::value) {
        this->TransOut(compactHoNum);
        CompactCopyOut(ncIdx, ncNum, compactHoNum, hoGlobalStart, hoNum);
    } else {
        EarlyCastTransOut(compactHoNum);
        CompactEarlyCastCopyOut(ncIdx, ncNum, compactHoNum, hoGlobalStart, hoNum);
    }
}

template <typename T, typename ID_T, const uint32_t NC_FACTOR>
__aicore__ inline void AdaptiveAvgPool2dUpsampleH<T, ID_T, NC_FACTOR>::ProcessOneBlock(const BlockParam& blockPara)
{
    int64_t hoNum = blockPara.hoNum;
    int64_t hoGlobalStart = blockPara.hoIdx * this->tilingData_->hoFactor;

    if (blockPara.ncIdx != cachedNcIdx_) {
        cachedHiBase_ = -1;
        cachedHiBatch_ = 0;
        cachedNcIdx_ = blockPara.ncIdx;
    }

    int64_t hiMin = 0;
    int64_t hiMax = 0;
    this->CalHiRange(hoGlobalStart, hoNum, hiMin, hiMax);

    // Prefer the compact layout, but only when it actually merges rows and the map fits.
    constexpr int64_t MAX_COMPACT_HO = 128;
    int64_t hoToCompact[MAX_COMPACT_HO];
    if (hoNum <= MAX_COMPACT_HO) {
        int64_t compactHoNum = BuildCompactMap(hoGlobalStart, hoNum, hoToCompact);
        if (compactHoNum != hoNum) {
            ProcessBlockCompact(blockPara, hiMin, hiMax, hoToCompact, compactHoNum);
            return;
        }
    }
    ProcessBlockPlain(blockPara, hiMin, hiMax);
}

template <typename T, typename ID_T, const uint32_t NC_FACTOR>
__aicore__ inline void AdaptiveAvgPool2dUpsampleH<T, ID_T, NC_FACTOR>::Process()
{
    if (GetBlockIdx() >= this->tilingData_->useCoreNum) {
        return;
    }

    // Compute W kernel info (vector ops) then sync V->S before scalar reads.
    this->CalWKernelInfo();
    if (isWUpsample_) {
        CalWiToWoInfo();
    }
    event_t eventIdVToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
    SetFlag<HardEvent::V_S>(eventIdVToS);
    WaitFlag<HardEvent::V_S>(eventIdVToS);

    for (int64_t curIdx = this->startBlockIdx_; curIdx < this->endBlockIdx_; curIdx++) {
        BlockParam blockPara;
        this->CalBlockPara(curIdx, blockPara);
        ProcessOneBlock(blockPara);
    }
}

} // namespace AdaptivePool2dUpsampleHNamespace
#endif // ADAPTIVE_AVG_POOL2D_UPSAMPLE_H_H_

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
 * \file adaptive_avg_pool2d_pooling_base.h
 * \brief Common base class for vectorized AdaptiveAvgPool2d kernel templates
 *        (SplitW, SplitC, SplitH, UpsampleH). Consolidates identical helper
 *        functions (CalBlockPara, TransposeB32, TransposeB16, ClearOutBuf,
 *        CalWKernelInfo, CalAvgOneHo) and shared member declarations.
 *        Derived classes shadow TransOut/CopyOut/CopyInputBatch/TransInputBatch
 *        when their layouts differ from the default padded-wOutAlign version.
 *
 * \arch  Ascend950 / A5 / DAV_3510 only, RegBase (MicroAPI) main path, VL = 256 Byte.
 *        [RegBase-native] Enforced on the host side: AdaptivePool2dBaseTiling::
 *        GetShapeAttrsInfo rejects GetCurNpuArch() != NpuArch::DAV_3510, so these
 *        templates are never dispatched on another architecture.
 */

#ifndef ADAPTIVE_AVG_POOL2D_POOLING_BASE_H_
#define ADAPTIVE_AVG_POOL2D_POOLING_BASE_H_

#include "kernel_operator.h"
#include "../inc/kernel_utils.h"
#include "../inc/platform.h"
#include "../inc/load_store_utils.h"
#include "kernel_tiling/kernel_tiling.h"
#include "adaptive_avg_pool2d_struct.h"

namespace AdaptiveAvgPool2dPoolingBaseNs {
using namespace AscendC;
using namespace ops;
using namespace AdaptiveAvgPool2dOp;

// TransDataTo5HD operates on a fixed 16x16 b16 / 16x8 b32 fractal, so these two are
// instruction-level constants rather than tunable sizes. [general]
constexpr static uint64_t AAP_TRANS_ADDR_LEN = 16;
constexpr static uint64_t AAP_TRANS_LEN_B32 = 8;
// UB block size and VL come from the platform interface, so they follow the target chip.
constexpr static uint32_t AAP_UB_BLOCK_SIZE = platform::GetUbBlockSize();
constexpr static uint32_t AAP_V_REG_SIZE = platform::GetVRegSize();

constexpr AscendC::MicroAPI::CastTrait aapCastTraitI32F32 = {
    AscendC::MicroAPI::RegLayout::UNKNOWN, AscendC::MicroAPI::SatMode::UNKNOWN,
    AscendC::MicroAPI::MaskMergeMode::ZEROING, AscendC::RoundMode::CAST_RINT};

struct BlockParam {
    int64_t ncIdx;
    int64_t hoIdx;
    int64_t ncNum;
    int64_t hoNum;
};

// int64 index math needs a two-register tensor; int32 fits one. Declared at namespace scope so
// the free VF functions below can name it too (the class re-exports it as IndexRegType).
template <typename ID_T>
using AapIndexRegType = typename std::conditional<
    IsSameType<ID_T, int64_t>::value, typename AscendC::MicroAPI::RegTensor<int64_t, AscendC::MicroAPI::RegTraitNumTwo>,
    typename AscendC::MicroAPI::RegTensor<int32_t>>::type;

// ---------------------------------------------------------------------------
// VF functions extracted from the base class vector-scope blocks. They must be
// defined before the class, and take plain scalars/__ubuf__ pointers because a VF
// cannot reach `this`.
// ---------------------------------------------------------------------------

__simd_vf__ inline void ClearOutBufVf(__ubuf__ float* outAddr, uint16_t loopSize, uint32_t remaining,
                                      uint32_t vfLenFp32)
{
    MicroAPI::RegTensor<float> zeroReg;
    MicroAPI::Duplicate(zeroReg, 0.0f);
    for (uint16_t i = 0; i < loopSize; i++) {
        MicroAPI::MaskReg mask = MicroAPI::UpdateMask<float>(remaining);
        MicroAPI::AddrReg offset = MicroAPI::CreateAddrReg<float>(i, vfLenFp32);
        MicroAPI::DataCopy(outAddr, zeroReg, offset, mask);
    }
}

template <typename ID_T>
__simd_vf__ inline void CalWKernelInfoVf(__ubuf__ int32_t* wStartAddr, __ubuf__ int32_t* wKerSizeAddr,
                                         uint16_t loopSize, uint32_t dataLen, uint16_t vfLen, int64_t wIn,
                                         int64_t wOutDim)
{
    AapIndexRegType<ID_T> startIdxReg;
    AapIndexRegType<ID_T> endIdxReg;
    AapIndexRegType<ID_T> kerSizeReg;
    AapIndexRegType<ID_T> dupReg;
    // Pack narrows b64->b32 and only supports unsigned destinations; wStart/wKerSize
    // are non-negative and far below 2^31, so the uint32->int32 store is value-preserving.
    MicroAPI::RegTensor<uint32_t> startDstReg;
    MicroAPI::RegTensor<uint32_t> kSizeDstReg;
    MicroAPI::MaskReg calMask;

    MicroAPI::Duplicate(dupReg, static_cast<ID_T>(wOutDim));
    for (uint16_t i = 0; i < loopSize; i++) {
        if constexpr (IsSameType<ID_T, int64_t>::value) {
            calMask = MicroAPI::UpdateMask<ID_T, MicroAPI::RegTraitNumTwo>(dataLen);
        } else {
            calMask = MicroAPI::UpdateMask<ID_T>(dataLen);
        }
        ID_T startIdx = i * vfLen;
        MicroAPI::Arange(startIdxReg, startIdx);
        MicroAPI::Adds(endIdxReg, startIdxReg, static_cast<ID_T>(1), calMask);
        MicroAPI::Muls(startIdxReg, startIdxReg, static_cast<ID_T>(wIn), calMask);
        MicroAPI::Muls(endIdxReg, endIdxReg, static_cast<ID_T>(wIn), calMask);
        MicroAPI::Adds(endIdxReg, endIdxReg, static_cast<ID_T>(wOutDim - 1), calMask);

        MicroAPI::Div(startIdxReg, startIdxReg, dupReg, calMask);
        MicroAPI::Div(endIdxReg, endIdxReg, dupReg, calMask);
        MicroAPI::Sub(kerSizeReg, endIdxReg, startIdxReg, calMask);

        if constexpr (IsSameType<ID_T, int64_t>::value) {
            // Narrow b64 indices to b32 for the int32 UB buffers. Under RegTraitNumTwo the
            // low words of all 64 elements live in reg[0] (reg[1] holds the high words), so
            // Pack reduces to a copy of reg[0] -- matching the vfLen=64 store stride and the
            // b32 mask that UpdateMask<int64_t, RegTraitNumTwo> produces.
            MicroAPI::Pack<uint32_t, ID_T, MicroAPI::HighLowPart::LOWEST>(startDstReg, startIdxReg);
            MicroAPI::Pack<uint32_t, ID_T, MicroAPI::HighLowPart::LOWEST>(kSizeDstReg, kerSizeReg);
            MicroAPI::DataCopy((__ubuf__ uint32_t*)wStartAddr + i * vfLen, startDstReg, calMask);
            MicroAPI::DataCopy((__ubuf__ uint32_t*)wKerSizeAddr + i * vfLen, kSizeDstReg, calMask);
        } else {
            MicroAPI::DataCopy(wStartAddr + i * vfLen, startIdxReg, calMask);
            MicroAPI::DataCopy(wKerSizeAddr + i * vfLen, kerSizeReg, calMask);
        }
    }
}

template <const uint32_t NC_FACTOR>
__simd_vf__ inline void CalAvgOneHoVf(__ubuf__ float* outAddr, __ubuf__ int32_t* wKerSizeAddr, uint16_t woNum,
                                      uint32_t outBaseOffset, uint32_t vlNum, uint32_t vfLenFp32, int32_t kh)
{
    MicroAPI::RegTensor<float> sumReg;
    MicroAPI::RegTensor<float> avgReg;
    MicroAPI::RegTensor<int32_t> divisorReg;
    MicroAPI::RegTensor<float> divisorCastReg;
    MicroAPI::MaskReg calMask = MicroAPI::CreateMask<float, MicroAPI::MaskPattern::ALL>();

    for (uint16_t wo = 0; wo < woNum; wo++) {
        uint32_t offset = outBaseOffset + static_cast<uint32_t>(wo) * vlNum;
        int32_t totalKer = kh * wKerSizeAddr[wo];

        MicroAPI::Duplicate(divisorReg, totalKer);
        MicroAPI::Cast<float, int32_t, aapCastTraitI32F32>(divisorCastReg, divisorReg, calMask);
        MicroAPI::DataCopy(sumReg, outAddr + offset);
        MicroAPI::Div(avgReg, sumReg, divisorCastReg, calMask);
        MicroAPI::DataCopy(outAddr + offset, avgReg, calMask);

        if constexpr (NC_FACTOR == TPL_NC_FACTOR_128) {
            MicroAPI::DataCopy(sumReg, outAddr + offset + vfLenFp32);
            MicroAPI::Div(avgReg, sumReg, divisorCastReg, calMask);
            MicroAPI::DataCopy(outAddr + offset + vfLenFp32, avgReg, calMask);
        }
    }
}

template <typename T, typename ID_T, const uint32_t NC_FACTOR, typename TilingT>
class AdaptiveAvgPool2dPoolingBase {
protected:
    __aicore__ inline AdaptiveAvgPool2dPoolingBase(const TilingT* tilingData, TPipe* pipe)
        : tilingData_(tilingData), pipe_(pipe){};

    __aicore__ inline void CalBlockPara(int64_t curBlockIdx, BlockParam& blockPara);

    // Returns false if this core should skip (blockIdx >= useCoreNum).
    __aicore__ inline bool InitCommon(GM_ADDR x, GM_ADDR y)
    {
        if (GetBlockIdx() >= tilingData_->useCoreNum) {
            return false;
        }
        inHW_ = tilingData_->hIn * tilingData_->wIn;
        outHW_ = tilingData_->hOut * tilingData_->wOut;
        vlNum_ = tilingData_->ncFactor;
        ubAlignNum_ = AAP_UB_BLOCK_SIZE / sizeof(T);
        wInAlign_ = ops::CeilAlign(static_cast<uint32_t>(tilingData_->wIn), ubAlignNum_);
        wOutAlign_ = ops::CeilAlign(static_cast<uint32_t>(tilingData_->wOut), ubAlignNum_);
        int64_t curHandleBlockNum = tilingData_->blockFactor;
        if (GetBlockIdx() == tilingData_->useCoreNum - 1) {
            curHandleBlockNum = tilingData_->blockTail;
        }
        startBlockIdx_ = GetBlockIdx() * tilingData_->blockFactor;
        endBlockIdx_ = startBlockIdx_ + curHandleBlockNum;
        xGm_.SetGlobalBuffer((__gm__ T*)x);
        yGm_.SetGlobalBuffer((__gm__ T*)y);
        return true;
    }

    // wBufSize alignment varies by strategy; wInAlignForTrans is wInFactorAlign_ for SplitC.
    __aicore__ inline void InitBuffers(uint64_t wBufSize, uint32_t wInAlignForTrans)
    {
        uint64_t transRowAlign = ops::CeilAlign(static_cast<uint64_t>(tilingData_->hiFactor) * wInAlignForTrans,
                                                static_cast<uint64_t>(AAP_TRANS_ADDR_LEN));
        uint64_t transBufSize = transRowAlign * vlNum_ * sizeof(T);
        uint64_t sumBufSize = static_cast<uint64_t>(wOutAlign_) * vlNum_ * sizeof(float);
        uint64_t outRowAlign = ops::CeilAlign(static_cast<uint64_t>(tilingData_->hoFactor) * wOutAlign_,
                                              static_cast<uint64_t>(AAP_TRANS_ADDR_LEN));
        uint64_t outBufSize = outRowAlign * vlNum_ * sizeof(float);
        pipe_->InitBuffer(inputQue_, 1, tilingData_->inputQueSize);
        pipe_->InitBuffer(resQue1_, 1, tilingData_->resQue1Size);
        if (tilingData_->resQue2Size > 0) {
            pipe_->InitBuffer(resQue2_, 1, tilingData_->resQue2Size);
        }
        pipe_->InitBuffer(transBuf_, transBufSize);
        pipe_->InitBuffer(sumBuf_, sumBufSize);
        pipe_->InitBuffer(outBuf_, outBufSize);
        pipe_->InitBuffer(wStartBuf_, wBufSize);
        pipe_->InitBuffer(wKerSizeBuf_, wBufSize);
    }

    template <typename U>
    __aicore__ inline void TransposeB32(LocalTensor<U> dst, LocalTensor<U> src, uint32_t rowNum, uint32_t colNum);
    __aicore__ inline void TransposeB16(LocalTensor<T> dst, LocalTensor<T> src, uint32_t rowNum, uint32_t colNum);
    __aicore__ inline void ClearOutBuf();
    __aicore__ inline void CalWKernelInfo();
    __aicore__ inline void CalAvgOneHo(int64_t kernelH, int64_t hoLocal, uint32_t wStride);
    __aicore__ inline void CalAvgOneHo(int64_t kernelH, int64_t hoLocal) { CalAvgOneHo(kernelH, hoLocal, wOutAlign_); }

    // Default TransOut/CopyOut use padded wOutAlign_ row stride. SplitH shadows with compact version.
    __aicore__ inline void TransOut(int64_t hoNum);
    __aicore__ inline void CopyOut(int64_t ncIdx, int64_t ncNum, int64_t hoGlobal, int64_t hoNum);

    // Default CopyInputBatch/TransInputBatch load full-width wIn. SplitC shadows with wi-chunk version.
    __aicore__ inline void CopyInputBatch(int64_t ncIdx, int64_t ncNum, int64_t hiStart, int64_t hiBatch);
    __aicore__ inline void TransInputBatch(int64_t hiBatch);

    // Adaptive-pooling H window arithmetic, shared by every split template:
    //   ho covers input rows [hStart, hEnd) with hStart = ho*hIn/hOut, hEnd = ceil((ho+1)*hIn/hOut).
    __aicore__ inline int64_t CalHoStart(int64_t hoGlobal) const
    {
        return (hoGlobal * tilingData_->hIn) / tilingData_->hOut;
    }

    __aicore__ inline int64_t CalHoEnd(int64_t hoGlobal) const
    {
        int64_t hOut = tilingData_->hOut;
        return ((hoGlobal + 1) * tilingData_->hIn + hOut - 1) / hOut;
    }

    __aicore__ inline int64_t CalKernelH(int64_t hoGlobal) const { return CalHoEnd(hoGlobal) - CalHoStart(hoGlobal); }

    // Half-open input-row range touched by [hoGlobalStart, hoGlobalStart+hoNum).
    __aicore__ inline void CalHiRange(int64_t hoGlobalStart, int64_t hoNum, int64_t& hiMin, int64_t& hiMax) const
    {
        hiMin = CalHoStart(hoGlobalStart);
        hiMax = CalHoEnd(hoGlobalStart + hoNum - 1);
    }

    TPipe* pipe_ = nullptr;
    TQue<QuePosition::VECIN, 1> inputQue_;
    TQue<QuePosition::VECOUT, 1> resQue1_;
    TQue<QuePosition::VECOUT, 1> resQue2_;
    TBuf<QuePosition::VECCALC> transBuf_;
    TBuf<QuePosition::VECCALC> sumBuf_;
    TBuf<QuePosition::VECCALC> outBuf_;
    TBuf<QuePosition::VECCALC> wStartBuf_;
    TBuf<QuePosition::VECCALC> wKerSizeBuf_;

    GlobalTensor<T> xGm_;
    GlobalTensor<T> yGm_;

    int64_t inHW_ = 1;
    int64_t outHW_ = 1;
    int64_t startBlockIdx_ = 0;
    int64_t endBlockIdx_ = 0;

    uint32_t vlNum_;
    uint32_t ubAlignNum_;
    uint32_t wInAlign_;
    uint32_t wOutAlign_;

    const TilingT* tilingData_;
    using IndexRegType = AapIndexRegType<ID_T>;
};

template <typename T, typename ID_T, const uint32_t NC_FACTOR, typename TilingT>
__aicore__ inline void AdaptiveAvgPool2dPoolingBase<T, ID_T, NC_FACTOR, TilingT>::CalBlockPara(int64_t curBlockIdx,
                                                                                               BlockParam& blockPara)
{
    blockPara.ncIdx = curBlockIdx / tilingData_->hoOuter;
    blockPara.ncNum = (blockPara.ncIdx == (tilingData_->ncOuter - 1)) ? tilingData_->ncTail : tilingData_->ncFactor;

    blockPara.hoIdx = curBlockIdx % tilingData_->hoOuter;
    blockPara.hoNum = (blockPara.hoIdx == (tilingData_->hoOuter - 1)) ? tilingData_->hoTail : tilingData_->hoFactor;
}

// TransposeB32: [rowNum, colNum] → [colNum, rowNum]
template <typename T, typename ID_T, const uint32_t NC_FACTOR, typename TilingT>
template <typename U>
__aicore__ inline void AdaptiveAvgPool2dPoolingBase<T, ID_T, NC_FACTOR, TilingT>::TransposeB32(LocalTensor<U> dst,
                                                                                               LocalTensor<U> src,
                                                                                               uint32_t rowNum,
                                                                                               uint32_t colNum)
{
    uint64_t dstList[AAP_TRANS_ADDR_LEN];
    uint64_t srcList[AAP_TRANS_ADDR_LEN];
    TransDataTo5HDParams transDataParams;
    transDataParams.dstHighHalf = false;
    transDataParams.srcHighHalf = false;
    uint64_t transPoseAlign = AAP_UB_BLOCK_SIZE / sizeof(U);

    if (colNum == transPoseAlign) {
        transDataParams.repeatTimes = rowNum / AAP_TRANS_ADDR_LEN;
        transDataParams.dstRepStride = AAP_TRANS_ADDR_LEN * sizeof(U) / AAP_UB_BLOCK_SIZE;
        transDataParams.srcRepStride = AAP_TRANS_ADDR_LEN;
        for (int32_t i = 0; i < AAP_TRANS_ADDR_LEN; i++) {
            srcList[i] = static_cast<uint64_t>(src[i * transPoseAlign].GetPhyAddr());
        }
        for (int32_t i = 0; i < AAP_TRANS_LEN_B32; i++) {
            dstList[i * 2] = static_cast<uint64_t>(dst[i * rowNum].GetPhyAddr());
            dstList[i * 2 + 1] = static_cast<uint64_t>(dst[i * rowNum + transPoseAlign].GetPhyAddr());
        }
        // TransDataTo5HD requires both RepStride fields to be 0 when repeatTimes == 1;
        // a non-zero stride is rejected in that case. [general]
        if (transDataParams.repeatTimes == 1) {
            transDataParams.srcRepStride = 0;
            transDataParams.dstRepStride = 0;
        }
        TransDataTo5HD<U>(dstList, srcList, transDataParams);
    } else {
        transDataParams.repeatTimes = colNum / transPoseAlign;
        transDataParams.dstRepStride = rowNum;
        transDataParams.srcRepStride = 1;
        for (int32_t rowLoopIdx = 0; rowLoopIdx < rowNum / AAP_TRANS_ADDR_LEN; rowLoopIdx++) {
            for (int32_t i = 0; i < AAP_TRANS_ADDR_LEN; i++) {
                srcList[i] = static_cast<uint64_t>(
                    src[rowLoopIdx * AAP_TRANS_ADDR_LEN * colNum + i * colNum].GetPhyAddr());
            }
            for (int32_t i = 0; i < AAP_TRANS_LEN_B32; i++) {
                dstList[i * 2] = static_cast<uint64_t>(dst[rowLoopIdx * AAP_TRANS_ADDR_LEN + i * rowNum].GetPhyAddr());
                dstList[i * 2 + 1] = static_cast<uint64_t>(
                    dst[rowLoopIdx * AAP_TRANS_ADDR_LEN + i * rowNum + transPoseAlign].GetPhyAddr());
            }
            TransDataTo5HD<U>(dstList, srcList, transDataParams);
        }
    }
}

// TransposeB16: [rowNum, colNum] → [colNum, rowNum]
template <typename T, typename ID_T, const uint32_t NC_FACTOR, typename TilingT>
__aicore__ inline void AdaptiveAvgPool2dPoolingBase<T, ID_T, NC_FACTOR, TilingT>::TransposeB16(LocalTensor<T> dst,
                                                                                               LocalTensor<T> src,
                                                                                               uint32_t rowNum,
                                                                                               uint32_t colNum)
{
    uint64_t dstList[AAP_TRANS_ADDR_LEN];
    uint64_t srcList[AAP_TRANS_ADDR_LEN];
    TransDataTo5HDParams transDataParams;
    transDataParams.dstHighHalf = false;
    transDataParams.srcHighHalf = false;
    uint64_t transPoseAlign = ubAlignNum_;

    if (colNum == transPoseAlign) {
        transDataParams.repeatTimes = rowNum / AAP_TRANS_ADDR_LEN;
        transDataParams.dstRepStride = AAP_TRANS_ADDR_LEN * sizeof(T) / AAP_UB_BLOCK_SIZE;
        transDataParams.srcRepStride = AAP_TRANS_ADDR_LEN;
        for (int32_t i = 0; i < AAP_TRANS_ADDR_LEN; i++) {
            srcList[i] = static_cast<uint64_t>(src[i * transPoseAlign].GetPhyAddr());
            dstList[i] = static_cast<uint64_t>(dst[i * rowNum].GetPhyAddr());
        }
        // Both RepStride fields must be 0 when repeatTimes == 1. [general]
        if (transDataParams.repeatTimes == 1) {
            transDataParams.srcRepStride = 0;
            transDataParams.dstRepStride = 0;
        }
        TransDataTo5HD<T>(dstList, srcList, transDataParams);
    } else {
        transDataParams.repeatTimes = colNum / transPoseAlign;
        transDataParams.dstRepStride = rowNum;
        transDataParams.srcRepStride = 1;
        for (int32_t rowLoopIdx = 0; rowLoopIdx < rowNum / AAP_TRANS_ADDR_LEN; rowLoopIdx++) {
            for (int32_t i = 0; i < AAP_TRANS_ADDR_LEN; i++) {
                srcList[i] = static_cast<uint64_t>(
                    src[rowLoopIdx * AAP_TRANS_ADDR_LEN * colNum + i * colNum].GetPhyAddr());
                dstList[i] = static_cast<uint64_t>(dst[rowLoopIdx * AAP_TRANS_ADDR_LEN + i * rowNum].GetPhyAddr());
            }
            TransDataTo5HD<T>(dstList, srcList, transDataParams);
        }
    }
}

template <typename T, typename ID_T, const uint32_t NC_FACTOR, typename TilingT>
__aicore__ inline void AdaptiveAvgPool2dPoolingBase<T, ID_T, NC_FACTOR, TilingT>::ClearOutBuf()
{
    LocalTensor<float> outLocal = outBuf_.Get<float>();
    __ubuf__ float* outAddr = (__ubuf__ float*)outLocal.GetPhyAddr();

    uint32_t totalCount = static_cast<uint32_t>(tilingData_->hoFactor) * wOutAlign_ * vlNum_;
    uint32_t vfLenFp32 = AAP_V_REG_SIZE / sizeof(float);
    uint16_t loopSize = ops::CeilDiv(totalCount, vfLenFp32);
    uint32_t remaining = totalCount;

    ClearOutBufVf(outAddr, loopSize, remaining, vfLenFp32);
}

template <typename T, typename ID_T, const uint32_t NC_FACTOR, typename TilingT>
__aicore__ inline void AdaptiveAvgPool2dPoolingBase<T, ID_T, NC_FACTOR, TilingT>::CalWKernelInfo()
{
    LocalTensor<int32_t> wStartLocal = wStartBuf_.Get<int32_t>();
    LocalTensor<int32_t> wKerSizeLocal = wKerSizeBuf_.Get<int32_t>();
    __ubuf__ int32_t* wStartAddr = (__ubuf__ int32_t*)wStartLocal.GetPhyAddr();
    __ubuf__ int32_t* wKerSizeAddr = (__ubuf__ int32_t*)wKerSizeLocal.GetPhyAddr();

    int32_t wOut = tilingData_->wOut;
    int64_t wIn = tilingData_->wIn;
    int64_t wOutDim = tilingData_->wOut;
    uint16_t vfLen = AAP_V_REG_SIZE / sizeof(int32_t);
    uint16_t loopSize = ops::CeilDiv(static_cast<uint16_t>(wOut), vfLen);
    uint32_t dataLen = wOut;

    CalWKernelInfoVf<ID_T>(wStartAddr, wKerSizeAddr, loopSize, dataLen, vfLen, wIn, wOutDim);
}

template <typename T, typename ID_T, const uint32_t NC_FACTOR, typename TilingT>
__aicore__ inline void AdaptiveAvgPool2dPoolingBase<T, ID_T, NC_FACTOR, TilingT>::CalAvgOneHo(int64_t kernelH,
                                                                                              int64_t hoLocal,
                                                                                              uint32_t wStride)
{
    LocalTensor<float> outLocal = outBuf_.Get<float>();
    LocalTensor<int32_t> wKerSizeLocal = wKerSizeBuf_.Get<int32_t>();

    __ubuf__ float* outAddr = (__ubuf__ float*)outLocal.GetPhyAddr();
    __ubuf__ int32_t* wKerSizeAddr = (__ubuf__ int32_t*)wKerSizeLocal.GetPhyAddr();

    uint32_t vfLenFp32 = AAP_V_REG_SIZE / sizeof(float);
    uint32_t outBaseOffset = static_cast<uint32_t>(hoLocal) * wStride * vlNum_;
    uint16_t woNum = static_cast<uint16_t>(tilingData_->wOut);
    int32_t kh = static_cast<int32_t>(kernelH);

    CalAvgOneHoVf<NC_FACTOR>(outAddr, wKerSizeAddr, woNum, outBaseOffset, vlNum_, vfLenFp32, kh);
}

// SplitH shadows with compact wOut stride + early Cast.
template <typename T, typename ID_T, const uint32_t NC_FACTOR, typename TilingT>
__aicore__ inline void AdaptiveAvgPool2dPoolingBase<T, ID_T, NC_FACTOR, TilingT>::TransOut(int64_t hoNum)
{
    int64_t rowNum = hoNum * wOutAlign_;
    uint64_t rowNumAlign = ops::CeilAlign(static_cast<uint64_t>(rowNum), AAP_TRANS_ADDR_LEN);

    LocalTensor<float> srcLocal = outBuf_.Get<float>();
    LocalTensor<float> dstLocal = resQue1_.AllocTensor<float>();
    this->template TransposeB32<float>(dstLocal, srcLocal, rowNumAlign, vlNum_);
    resQue1_.EnQue(dstLocal);
}

// SplitH shadows with single-block compact CopyOut.
template <typename T, typename ID_T, const uint32_t NC_FACTOR, typename TilingT>
__aicore__ inline void AdaptiveAvgPool2dPoolingBase<T, ID_T, NC_FACTOR, TilingT>::CopyOut(int64_t ncIdx, int64_t ncNum,
                                                                                          int64_t hoGlobal,
                                                                                          int64_t hoNum)
{
    uint64_t hwOutStride = ops::CeilAlign(static_cast<uint64_t>(hoNum * wOutAlign_),
                                          static_cast<uint64_t>(AAP_TRANS_ADDR_LEN));
    int64_t ncBase = ncIdx * tilingData_->ncFactor;

    LocalTensor<float> resOutLocal = resQue1_.DeQue<float>();

    DataCopyExtParams valueParams;
    valueParams.blockCount = 1;
    valueParams.blockLen = static_cast<uint32_t>(tilingData_->wOut * sizeof(T));
    valueParams.srcStride = 0;
    valueParams.dstStride = 0;

    LoopModeParams loopModeParams;
    loopModeParams.loop1Size = ncNum;
    loopModeParams.loop2Size = 1;
    loopModeParams.loop1SrcStride = hwOutStride * sizeof(T);
    loopModeParams.loop2SrcStride = 0;
    loopModeParams.loop1DstStride = outHW_ * sizeof(T);
    loopModeParams.loop2DstStride = 0;

    if constexpr (IsSameType<T, float>::value) {
        SetLoopModePara(loopModeParams, DataCopyMVType::UB_TO_OUT);
        for (int64_t ho = 0; ho < hoNum; ho++) {
            int64_t yOff = ncBase * outHW_ + (hoGlobal + ho) * tilingData_->wOut;
            uint64_t srcOff = static_cast<uint64_t>(ho) * wOutAlign_;
            DataCopyPad(yGm_[yOff], resOutLocal[srcOff], valueParams);
        }
        ResetLoopModePara(DataCopyMVType::UB_TO_OUT);
    } else {
        LocalTensor<T> castOutLocal = resQue2_.AllocTensor<T>();
        if constexpr (IsSameType<T, half>::value) {
            Cast(castOutLocal, resOutLocal, RoundMode::CAST_NONE, ncNum * hwOutStride);
        } else {
            Cast(castOutLocal, resOutLocal, RoundMode::CAST_RINT, ncNum * hwOutStride);
        }
        resQue2_.EnQue(castOutLocal);
        castOutLocal = resQue2_.DeQue<T>();
        SetLoopModePara(loopModeParams, DataCopyMVType::UB_TO_OUT);
        for (int64_t ho = 0; ho < hoNum; ho++) {
            int64_t yOff = ncBase * outHW_ + (hoGlobal + ho) * tilingData_->wOut;
            uint64_t srcOff = static_cast<uint64_t>(ho) * wOutAlign_;
            DataCopyPad(yGm_[yOff], castOutLocal[srcOff], valueParams);
        }
        ResetLoopModePara(DataCopyMVType::UB_TO_OUT);
        resQue2_.FreeTensor(castOutLocal);
    }
    resQue1_.FreeTensor(resOutLocal);
}

// SplitC shadows with wi-chunk version (wiBase/wiLen/wChunkAlign parameters).
template <typename T, typename ID_T, const uint32_t NC_FACTOR, typename TilingT>
__aicore__ inline void AdaptiveAvgPool2dPoolingBase<T, ID_T, NC_FACTOR, TilingT>::CopyInputBatch(int64_t ncIdx,
                                                                                                 int64_t ncNum,
                                                                                                 int64_t hiStart,
                                                                                                 int64_t hiBatch)
{
    LocalTensor<T> xLocal = inputQue_.AllocTensor<T>();

    int64_t gmOffset = ncIdx * tilingData_->ncFactor * inHW_ + hiStart * tilingData_->wIn;

    DataCopyExtParams paramsIn = {static_cast<uint16_t>(hiBatch), static_cast<uint32_t>(tilingData_->wIn * sizeof(T)),
                                  static_cast<uint32_t>(0), static_cast<uint32_t>(0), static_cast<uint32_t>(0)};
    DataCopyPadExtParams<T> padParams = {false, 0, 0, 0};

    LoopModeParams loopModeParams;
    loopModeParams.loop1Size = ncNum;
    loopModeParams.loop2Size = 1;
    loopModeParams.loop1SrcStride = inHW_ * sizeof(T);
    loopModeParams.loop2SrcStride = 0;
    loopModeParams.loop1DstStride = hiBatch * wInAlign_ * sizeof(T);
    loopModeParams.loop2DstStride = 0;

    SetLoopModePara(loopModeParams, DataCopyMVType::OUT_TO_UB);
    DataCopyPad(xLocal, xGm_[gmOffset], paramsIn, padParams);
    ResetLoopModePara(DataCopyMVType::OUT_TO_UB);
    inputQue_.EnQue(xLocal);
}

// TransInputBatch: [VL, hiBatch*wInAlign] → [hiBatch*wInAlign, VL]. SplitC shadows with wChunkAlign version.
template <typename T, typename ID_T, const uint32_t NC_FACTOR, typename TilingT>
__aicore__ inline void AdaptiveAvgPool2dPoolingBase<T, ID_T, NC_FACTOR, TilingT>::TransInputBatch(int64_t hiBatch)
{
    LocalTensor<T> xLocal = inputQue_.DeQue<T>();
    LocalTensor<T> transLocal = transBuf_.Get<T>();
    uint32_t colNum = static_cast<uint32_t>(hiBatch) * wInAlign_;
    if constexpr (IsSameType<T, float>::value) {
        this->template TransposeB32<T>(transLocal, xLocal, vlNum_, colNum);
    } else {
        this->TransposeB16(transLocal, xLocal, vlNum_, colNum);
    }
    inputQue_.FreeTensor(xLocal);
}

} // namespace AdaptiveAvgPool2dPoolingBaseNs
#endif // ADAPTIVE_AVG_POOL2D_POOLING_BASE_H_

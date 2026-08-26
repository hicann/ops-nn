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
 * \file adaptive_avg_pool3d_gather_trans.h
 * \brief
 */

#ifndef ADAPTIVE_AVG_POOL3D_GATHER_TRANS_H_
#define ADAPTIVE_AVG_POOL3D_GATHER_TRANS_H_

#include "kernel_operator.h"
#include "../inc/kernel_utils.h"
#include "../inc/platform.h"
#include "../inc/load_store_utils.h"
#include "kernel_tiling/kernel_tiling.h"
#include "adaptive_pool3d_tiling_struct.h"
#include "op_kernel/platform_util.h"

namespace AdaptivePool3d {
using namespace AscendC;
using namespace ops;

constexpr AscendC::MicroAPI::CastTrait castTraitGatherI32Fp32 = {
    AscendC::MicroAPI::RegLayout::UNKNOWN,
    AscendC::MicroAPI::SatMode::UNKNOWN,
    AscendC::MicroAPI::MaskMergeMode::ZEROING,
    AscendC::RoundMode::CAST_RINT,
};
constexpr uint32_t GT_UNROLL = 4;

template <typename T>
class AdaptiveAvgPool3dGatherTrans {
public:
    __aicore__ inline AdaptiveAvgPool3dGatherTrans(
        const AdaptivePool3DTiling::AdaptivePool3dGatherTransTilingData& tilingData, TPipe& pipe)
        : tilingData_(tilingData), pipe_(pipe){};

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y);
    __aicore__ inline void GenIndices(int64_t spatialBlock, int64_t outBlock);
    __aicore__ inline void GenChannelIndices(int64_t spatialBlock);
    __aicore__ inline void CopyIn(int64_t ncTotal, int64_t ncStart, int64_t dLo, int64_t spatialBlock, bool isSplit);
    __aicore__ inline void GatherTranspose(const LocalTensor<T>& xLocal, int64_t groupIdx, int64_t ncNum,
                                           int64_t spatialBlock);
    __aicore__ inline void Compute(int64_t groupIdx, int64_t ncNum, int64_t odStart, int64_t odNum, int64_t dLo,
                                   int64_t spatialBlock, int64_t outBlock);
    __aicore__ inline void CopyOut(int64_t ncTotal, int64_t ncStart, int64_t odStart, int64_t outBlock, bool isSplit);
    __aicore__ inline void Process();

protected:
    using RangeType_ = typename std::conditional<sizeof(T) <= sizeof(int16_t), int16_t, int32_t>::type;
    using IdxType_ = typename std::conditional<sizeof(T) <= sizeof(int16_t), uint16_t, uint32_t>::type;

    TPipe pipe_;
    TBuf<TPosition::VECCALC> transBuf_;
    TBuf<TPosition::VECCALC> shareBuf_;
    TBuf<TPosition::VECCALC> idxChBuf_;
    TBuf<TPosition::VECCALC> idxScatterBuf_;
    GlobalTensor<T> xGm_, yGm_;

    int64_t spatialIn_ = 1;
    int64_t inHW_ = 1;
    int64_t outDHW_ = 1;
    int64_t outHW_ = 1;
    int64_t tileNum_ = 1;
    int64_t startTaskIdx_ = 0;
    int64_t endTaskIdx_ = 0;

    uint32_t vfLen_ = 0;
    uint32_t vfLenFp32_ = 0;
    uint32_t idxVLSize_ = 0;

    const AdaptivePool3DTiling::AdaptivePool3dGatherTransTilingData tilingData_;
};

template <typename T>
__aicore__ inline void AdaptiveAvgPool3dGatherTrans<T>::Init(GM_ADDR x, GM_ADDR y)
{
    if (GetBlockIdx() >= tilingData_.useCoreNum) {
        return;
    }

    spatialIn_ = tilingData_.dIn * tilingData_.hIn * tilingData_.wIn;
    inHW_ = tilingData_.hIn * tilingData_.wIn;
    outDHW_ = tilingData_.dOut * tilingData_.hOut * tilingData_.wOut;
    outHW_ = tilingData_.hOut * tilingData_.wOut;
    vfLen_ = platform::GetVRegSize() / sizeof(T);
    vfLenFp32_ = platform::GetVRegSize() / sizeof(float);
    idxVLSize_ = platform::GetVRegSize() / sizeof(RangeType_);
    tileNum_ = ops::CeilDiv(tilingData_.ncOuter, tilingData_.ncBatch);
    int64_t totalTasks = tilingData_.doOuter * tileNum_;
    int64_t curTaskNum = tilingData_.blockFactor;
    uint32_t ubBlockSize = Ops::Base::GetUbBlockSize();

    if (GetBlockIdx() == tilingData_.useCoreNum - 1) {
        curTaskNum = tilingData_.blockTail;
    }

    startTaskIdx_ = GetBlockIdx() * tilingData_.blockFactor;
    endTaskIdx_ = startTaskIdx_ + curTaskNum;
    if (endTaskIdx_ > totalTasks) {
        endTaskIdx_ = totalTasks;
    }

    xGm_.SetGlobalBuffer((__gm__ T*)x);
    yGm_.SetGlobalBuffer((__gm__ T*)y);

    uint32_t maxNcTotal = tilingData_.ncBatch * tilingData_.ncFactor;
    uint32_t maxSpatialBlock = tilingData_.maxDInBlock * inHW_;
    uint32_t maxOutBlock = tilingData_.maxDoBlock * outHW_;
    uint32_t transBufSize = ops::CeilAlign(
        static_cast<uint32_t>(tilingData_.ncBatch * maxSpatialBlock * vfLen_ * sizeof(T)), ubBlockSize);

    if constexpr (!IsSameType<T, float>::value) {
        uint32_t castBufSize = ops::CeilAlign(static_cast<uint32_t>(maxNcTotal * maxOutBlock * sizeof(T)), ubBlockSize);
        transBufSize = transBufSize > castBufSize ? transBufSize : castBufSize;
    }

    pipe_.InitBuffer(transBuf_, transBufSize);
    uint32_t stageInSize = ops::CeilAlign(static_cast<uint32_t>(maxNcTotal * maxSpatialBlock * sizeof(T)), ubBlockSize);
    uint32_t stageOutSize = ops::CeilAlign(static_cast<uint32_t>(maxNcTotal * maxOutBlock * sizeof(float)),
                                           ubBlockSize);
    uint32_t indexBufSize = ops::CeilAlign(static_cast<uint32_t>(vfLen_ * sizeof(int32_t)), ubBlockSize);
    pipe_.InitBuffer(shareBuf_, stageInSize > stageOutSize ? stageInSize : stageOutSize);
    pipe_.InitBuffer(idxChBuf_, indexBufSize);
    pipe_.InitBuffer(idxScatterBuf_, indexBufSize);
}

template <typename T>
__aicore__ inline void AdaptiveAvgPool3dGatherTrans<T>::GenChannelIndices(int64_t spatialBlock)
{
    constexpr bool twoHalf = !IsSameType<T, float>::value;
    if constexpr (twoHalf) {
        LocalTensor<IdxType_> idxChLocal = idxChBuf_.Get<IdxType_>();
        __local_mem__ IdxType_* idxChAddr = (__local_mem__ IdxType_*)idxChLocal.GetPhyAddr();
        IdxType_ chStride = static_cast<IdxType_>(spatialBlock);
        __VEC_SCOPE__
        {
            AscendC::MicroAPI::RegTensor<RangeType_> srcReg;
            AscendC::MicroAPI::RegTensor<IdxType_> dstReg;
            AscendC::MicroAPI::MaskReg genMask = AscendC::MicroAPI::CreateMask<IdxType_>();

            AscendC::Reg::Arange(srcReg, 0);
            AscendC::MicroAPI::Muls(dstReg, (AscendC::MicroAPI::RegTensor<IdxType_>&)srcReg, chStride, genMask);
            AscendC::MicroAPI::DataCopy(idxChAddr, dstReg, genMask);
        }
    } else {
        LocalTensor<RangeType_> idxChLocal = idxChBuf_.Get<RangeType_>();
        __local_mem__ RangeType_* idxChAddr = (__local_mem__ RangeType_*)idxChLocal.GetPhyAddr();
        RangeType_ chStride = static_cast<RangeType_>(spatialBlock);
        __VEC_SCOPE__
        {
            AscendC::MicroAPI::RegTensor<RangeType_> srcReg;
            AscendC::MicroAPI::RegTensor<RangeType_> dstReg;
            AscendC::MicroAPI::MaskReg genMask = AscendC::MicroAPI::CreateMask<RangeType_>();

            AscendC::Reg::Arange(srcReg, 0);
            AscendC::Reg::Muls(dstReg, srcReg, chStride, genMask);
            AscendC::Reg::DataCopy(idxChAddr, dstReg, genMask);
        }
    }
}

template <typename T>
__aicore__ inline void AdaptiveAvgPool3dGatherTrans<T>::GenIndices(int64_t spatialBlock, int64_t outBlock)
{
    GenChannelIndices(spatialBlock);
    LocalTensor<int32_t> idxScatterLocal = idxScatterBuf_.Get<int32_t>();
    __local_mem__ int32_t* idxScatterAddr = (__local_mem__ int32_t*)idxScatterLocal.GetPhyAddr();
    int32_t scatterStride = static_cast<int32_t>(outBlock);
    uint16_t scatterLoop = static_cast<uint16_t>(ops::CeilDiv(vfLen_, vfLenFp32_));
    __VEC_SCOPE__
    {
        AscendC::MicroAPI::RegTensor<int32_t> srcScatter;
        AscendC::MicroAPI::RegTensor<int32_t> dstScatter;
        AscendC::MicroAPI::MaskReg scatterMask = AscendC::MicroAPI::CreateMask<int32_t>();

        AscendC::Reg::Arange(srcScatter, 0);

        for (uint16_t i = 0; i < scatterLoop; ++i) {
            AscendC::MicroAPI::Muls(dstScatter, srcScatter, scatterStride, scatterMask);
            AscendC::MicroAPI::DataCopy(idxScatterAddr + static_cast<uint32_t>(i) * vfLenFp32_, dstScatter,
                                        scatterMask);
            AscendC::MicroAPI::Adds(srcScatter, srcScatter, static_cast<int32_t>(vfLenFp32_), scatterMask);
        }
    }
}

template <typename T>
__aicore__ inline void AdaptiveAvgPool3dGatherTrans<T>::CopyIn(int64_t ncTotal, int64_t ncStart, int64_t dLo,
                                                               int64_t spatialBlock, bool isSplit)
{
    LocalTensor<T> xLocal = shareBuf_.Get<T>();
    DataCopyExtParams params;
    DataCopyPadExtParams<T> padParams = {false, 0, 0, 0};
    if (!isSplit) {
        params.blockCount = 1;
        params.blockLen = static_cast<uint32_t>(ncTotal * spatialBlock * sizeof(T));
        params.srcStride = 0;
        params.dstStride = 0;
        params.rsv = 0;
        DataCopyPad(xLocal, xGm_[ncStart * spatialIn_], params, padParams);
    } else {
        params.blockCount = static_cast<uint16_t>(ncTotal);
        params.blockLen = static_cast<uint32_t>(spatialBlock * sizeof(T));
        params.srcStride = static_cast<uint32_t>((spatialIn_ - spatialBlock) * sizeof(T));
        params.dstStride = 0;
        params.rsv = 0;
        DataCopyPad<T, PaddingMode::Compact>(xLocal, xGm_[ncStart * spatialIn_ + dLo * inHW_], params, padParams);
    }
    event_t eventMte2ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
    SetFlag<HardEvent::MTE2_V>(eventMte2ToV);
    WaitFlag<HardEvent::MTE2_V>(eventMte2ToV);
}

template <typename T>
__aicore__ inline void AdaptiveAvgPool3dGatherTrans<T>::GatherTranspose(const LocalTensor<T>& xLocal, int64_t groupIdx,
                                                                        int64_t ncNum, int64_t spatialBlock)
{
    LocalTensor<T> transLocal = transBuf_.Get<T>();
    LocalTensor<RangeType_> idxLocal = idxChBuf_.Get<RangeType_>();
    __local_mem__ T* xInAddr = (__local_mem__ T*)xLocal.GetPhyAddr() + groupIdx * tilingData_.ncFactor * spatialBlock;
    __local_mem__ T* transAddr = (__local_mem__ T*)transLocal.GetPhyAddr() +
                                 groupIdx * spatialBlock * static_cast<int64_t>(vfLen_);
    __local_mem__ RangeType_* idxAddr = (__local_mem__ RangeType_*)idxLocal.GetPhyAddr();
    uint32_t ncMask = static_cast<uint32_t>(ncNum);
    uint32_t spatial = static_cast<uint32_t>(spatialBlock);
    uint16_t mainLoop = static_cast<uint16_t>(spatial / GT_UNROLL);
    uint16_t tailLoop = static_cast<uint16_t>(spatial - mainLoop * GT_UNROLL);
    uint32_t tailStart = static_cast<uint32_t>(mainLoop) * GT_UNROLL;
    __VEC_SCOPE__
    {
        AscendC::MicroAPI::RegTensor<T> xReg0;
        AscendC::MicroAPI::RegTensor<T> xReg1;
        AscendC::MicroAPI::RegTensor<T> xReg2;
        AscendC::MicroAPI::RegTensor<T> xReg3;
        AscendC::MicroAPI::RegTensor<RangeType_> idxReg;
        AscendC::MicroAPI::MaskReg mask = AscendC::MicroAPI::UpdateMask<T>(ncMask);
        AscendC::MicroAPI::DataCopy(idxReg, idxAddr);
        for (uint16_t i = 0; i < mainLoop; ++i) {
            uint32_t s = static_cast<uint32_t>(i) * GT_UNROLL;
            AscendC::MicroAPI::DataCopyGather((AscendC::MicroAPI::RegTensor<T>&)xReg0, xInAddr + s,
                                              (AscendC::MicroAPI::RegTensor<IdxType_>&)idxReg, mask);
            AscendC::MicroAPI::DataCopyGather((AscendC::MicroAPI::RegTensor<T>&)xReg1, xInAddr + s + 1,
                                              (AscendC::MicroAPI::RegTensor<IdxType_>&)idxReg, mask);
            AscendC::MicroAPI::DataCopyGather((AscendC::MicroAPI::RegTensor<T>&)xReg2, xInAddr + s + 2,
                                              (AscendC::MicroAPI::RegTensor<IdxType_>&)idxReg, mask);
            AscendC::MicroAPI::DataCopyGather((AscendC::MicroAPI::RegTensor<T>&)xReg3, xInAddr + s + 3,
                                              (AscendC::MicroAPI::RegTensor<IdxType_>&)idxReg, mask);
            AscendC::MicroAPI::DataCopy(transAddr + s * vfLen_, xReg0, mask);
            AscendC::MicroAPI::DataCopy(transAddr + (s + 1) * vfLen_, xReg1, mask);
            AscendC::MicroAPI::DataCopy(transAddr + (s + 2) * vfLen_, xReg2, mask);
            AscendC::MicroAPI::DataCopy(transAddr + (s + 3) * vfLen_, xReg3, mask);
        }
        for (uint16_t t = 0; t < tailLoop; ++t) {
            uint32_t s = tailStart + t;
            AscendC::MicroAPI::DataCopyGather((AscendC::MicroAPI::RegTensor<T>&)xReg0, xInAddr + s,
                                              (AscendC::MicroAPI::RegTensor<IdxType_>&)idxReg, mask);
            AscendC::MicroAPI::DataCopy(transAddr + s * vfLen_, xReg0, mask);
        }
    }
}

template <typename T>
__aicore__ inline void AdaptiveAvgPool3dGatherTrans<T>::Compute(int64_t groupIdx, int64_t ncNum, int64_t odStart,
                                                                int64_t odNum, int64_t dLo, int64_t spatialBlock,
                                                                int64_t outBlock)
{
    LocalTensor<T> transLocal = transBuf_.Get<T>();
    LocalTensor<float> yTransLocal = shareBuf_.Get<float>();
    LocalTensor<int32_t> idxScatterLocal = idxScatterBuf_.Get<int32_t>();
    __ubuf__ T* transAddr = (__ubuf__ T*)transLocal.GetPhyAddr() +
                            groupIdx * spatialBlock * static_cast<int64_t>(vfLen_);
    __ubuf__ float* yTransAddr = (__ubuf__ float*)yTransLocal.GetPhyAddr() + groupIdx * tilingData_.ncFactor * outBlock;
    __local_mem__ int32_t* idxScatterAddr = (__local_mem__ int32_t*)idxScatterLocal.GetPhyAddr();
    int64_t dIn = tilingData_.dIn;
    int64_t hIn = tilingData_.hIn;
    int64_t wIn = tilingData_.wIn;
    int64_t dOut = tilingData_.dOut;
    int64_t hOut = tilingData_.hOut;
    int64_t wOut = tilingData_.wOut;
    constexpr bool twoHalf = !IsSameType<T, float>::value;
    uint32_t half2Off = vfLenFp32_;
    for (int32_t od = odStart; od < odStart + odNum; ++od) {
        int32_t d0 = od * dIn / dOut;
        int32_t d1 = ((od + 1) * dIn + dOut - 1) / dOut;
        for (int32_t oh = 0; oh < hOut; ++oh) {
            int32_t h0 = oh * hIn / hOut;
            int32_t h1 = ((oh + 1) * hIn + hOut - 1) / hOut;
            for (int32_t ow = 0; ow < wOut; ++ow) {
                int32_t w0 = ow * wIn / wOut;
                int32_t w1 = ((ow + 1) * wIn + wOut - 1) / wOut;
                int32_t cnt = static_cast<int32_t>((d1 - d0) * (h1 - h0) * (w1 - w0));
                int32_t outIdx = ((od - odStart) * hOut + oh) * wOut + ow;
                __VEC_SCOPE__
                {
                    AscendC::MicroAPI::RegTensor<float> sumLo;
                    AscendC::MicroAPI::RegTensor<float> sumHi;
                    AscendC::MicroAPI::RegTensor<float> inLo;
                    AscendC::MicroAPI::RegTensor<float> inHi;
                    AscendC::MicroAPI::RegTensor<float> avgReg;
                    AscendC::MicroAPI::RegTensor<float> cntReg;
                    AscendC::MicroAPI::RegTensor<int32_t> cntRegInt;
                    AscendC::MicroAPI::RegTensor<int32_t> idxScatterReg;
                    AscendC::MicroAPI::MaskReg mask = AscendC::MicroAPI::CreateMask<int32_t>();
                    AscendC::MicroAPI::DataCopy(idxScatterReg, idxScatterAddr);
                    AscendC::MicroAPI::Duplicate(cntRegInt, cnt);
                    AscendC::MicroAPI::Cast<float, int32_t, castTraitGatherI32Fp32>(cntReg, cntRegInt, mask);
                    AscendC::MicroAPI::Duplicate(sumLo, static_cast<float>(0));
                    if constexpr (twoHalf) {
                        AscendC::MicroAPI::Duplicate(sumHi, static_cast<float>(0));
                    }
                    for (uint16_t dd = 0; dd < static_cast<uint16_t>(d1 - d0); ++dd) {
                        int32_t d = d0 + dd - dLo;
                        for (uint16_t hh = 0; hh < static_cast<uint16_t>(h1 - h0); ++hh) {
                            int32_t h = h0 + hh;
                            for (uint16_t ww = 0; ww < static_cast<uint16_t>(w1 - w0); ++ww) {
                                int32_t w = w0 + ww;
                                uint32_t s = static_cast<uint32_t>((d * hIn + h) * wIn + w);
                                if constexpr (twoHalf) {
                                    ops::LoadTwoTensorForDtypeT<T>(transAddr, transAddr, inLo, inHi, mask, mask,
                                                                   s * vfLen_, s * vfLen_ + half2Off);
                                    AscendC::MicroAPI::Add(sumLo, inLo, sumLo, mask);
                                    AscendC::MicroAPI::Add(sumHi, inHi, sumHi, mask);
                                } else {
                                    ops::LoadOneTensorForDtypeT<T>(transAddr, inLo, mask, s * vfLen_);
                                    AscendC::MicroAPI::Add(sumLo, inLo, sumLo, mask);
                                }
                            }
                        }
                    }
                    AscendC::MicroAPI::Div(avgReg, sumLo, cntReg, mask);
                    AscendC::MicroAPI::DataCopyScatter(yTransAddr + outIdx, avgReg,
                                                       (AscendC::MicroAPI::RegTensor<uint32_t>&)idxScatterReg, mask);
                    if constexpr (twoHalf) {
                        AscendC::MicroAPI::Div(avgReg, sumHi, cntReg, mask);
                        AscendC::MicroAPI::DataCopyScatter(yTransAddr + outIdx + half2Off * outBlock, avgReg,
                                                           (AscendC::MicroAPI::RegTensor<uint32_t>&)idxScatterReg,
                                                           mask);
                    }
                }
            }
        }
    }
}

template <typename T>
__aicore__ inline void AdaptiveAvgPool3dGatherTrans<T>::CopyOut(int64_t ncTotal, int64_t ncStart, int64_t odStart,
                                                                int64_t outBlock, bool isSplit)
{
    LocalTensor<float> yTransLocal = shareBuf_.Get<float>();
    int64_t total = ncTotal * outBlock;
    DataCopyExtParams params;
    if (!isSplit) {
        params.blockCount = 1;
        params.blockLen = static_cast<uint32_t>(total * sizeof(T));
        params.srcStride = 0;
        params.dstStride = 0;
    } else {
        params.blockCount = static_cast<uint16_t>(ncTotal);
        params.blockLen = static_cast<uint32_t>(outBlock * sizeof(T));
        params.srcStride = 0;
        params.dstStride = static_cast<uint32_t>((outDHW_ - outBlock) * sizeof(T));
    }
    params.rsv = 0;
    int64_t yOffset = ncStart * outDHW_ + odStart * outHW_;
    event_t eventVToMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
    event_t eventMte3ToMte2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
    if constexpr (IsSameType<T, float>::value) {
        SetFlag<HardEvent::V_MTE3>(eventVToMte3);
        WaitFlag<HardEvent::V_MTE3>(eventVToMte3);
        if (isSplit) {
            DataCopyPad<T, PaddingMode::Compact>(yGm_[yOffset], yTransLocal, params);
        } else {
            DataCopyPad(yGm_[yOffset], yTransLocal, params);
        }
    } else {
        LocalTensor<T> castLocal = transBuf_.Get<T>();
        __local_mem__ float* srcAddr = (__local_mem__ float*)yTransLocal.GetPhyAddr();
        __local_mem__ T* dstAddr = (__local_mem__ T*)castLocal.GetPhyAddr();
        uint16_t castLoop = static_cast<uint16_t>(ops::CeilDiv(total, static_cast<int64_t>(vfLenFp32_)));
        uint32_t remain = static_cast<uint32_t>(total);
        __VEC_SCOPE__
        {
            AscendC::MicroAPI::RegTensor<float> srcReg;
            AscendC::MicroAPI::MaskReg castMask;
            for (uint16_t i = 0; i < castLoop; ++i) {
                castMask = AscendC::MicroAPI::UpdateMask<float>(remain);
                AscendC::MicroAPI::DataCopy(srcReg, srcAddr + static_cast<uint32_t>(i) * vfLenFp32_);
                ops::StoreOneTensorForDtypeT<T>(dstAddr, srcReg, castMask, static_cast<uint32_t>(i) * vfLenFp32_);
            }
        }
        SetFlag<HardEvent::V_MTE3>(eventVToMte3);
        WaitFlag<HardEvent::V_MTE3>(eventVToMte3);
        if (isSplit) {
            DataCopyPad<T, PaddingMode::Compact>(yGm_[yOffset], castLocal, params);
        } else {
            DataCopyPad(yGm_[yOffset], castLocal, params);
        }
    }
    SetFlag<HardEvent::MTE3_MTE2>(eventMte3ToMte2);
    WaitFlag<HardEvent::MTE3_MTE2>(eventMte3ToMte2);
    if constexpr (!IsSameType<T, float>::value) {
        event_t eventMte3ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));
        SetFlag<HardEvent::MTE3_V>(eventMte3ToV);
        WaitFlag<HardEvent::MTE3_V>(eventMte3ToV);
    }
}

template <typename T>
__aicore__ inline void AdaptiveAvgPool3dGatherTrans<T>::Process()
{
    if (GetBlockIdx() >= tilingData_.useCoreNum) {
        return;
    }
    int64_t ncOuter = tilingData_.ncOuter;
    int64_t dIn = tilingData_.dIn;
    int64_t dOut = tilingData_.dOut;
    for (int64_t task = startTaskIdx_; task < endTaskIdx_; ++task) {
        int64_t doBlockIdx = task / tileNum_;
        int64_t ncTileIdx = task - doBlockIdx * tileNum_;
        int64_t odStart = doBlockIdx * tilingData_.doFactor;
        int64_t odNum = (doBlockIdx == tilingData_.doOuter - 1) ? tilingData_.doTail : tilingData_.doFactor;
        int64_t dLo = odStart * dIn / dOut;
        int64_t dHi = ((odStart + odNum) * dIn + dOut - 1) / dOut;
        int64_t dInBlock = dHi - dLo;
        int64_t spatialBlock = dInBlock * inHW_;
        int64_t outBlock = odNum * outHW_;
        bool isSplit = (tilingData_.doOuter > 1);
        GenIndices(spatialBlock, outBlock);
        int64_t groupStart = ncTileIdx * tilingData_.ncBatch;
        int64_t groupNum = ncOuter - groupStart;
        if (groupNum > tilingData_.ncBatch) {
            groupNum = tilingData_.ncBatch;
        }
        bool hasTail = (groupStart + groupNum == ncOuter);
        int64_t ncTotal = (groupNum - 1) * tilingData_.ncFactor + (hasTail ? tilingData_.ncTail : tilingData_.ncFactor);
        int64_t ncStart = groupStart * tilingData_.ncFactor;
        CopyIn(ncTotal, ncStart, dLo, spatialBlock, isSplit);
        LocalTensor<T> xLocal = shareBuf_.Get<T>();
        for (int64_t g = 0; g < groupNum; ++g) {
            int64_t ncNum = (hasTail && g == groupNum - 1) ? tilingData_.ncTail : tilingData_.ncFactor;
            GatherTranspose(xLocal, g, ncNum, spatialBlock);
        }
        for (int64_t g = 0; g < groupNum; ++g) {
            int64_t ncNum = (hasTail && g == groupNum - 1) ? tilingData_.ncTail : tilingData_.ncFactor;
            Compute(g, ncNum, odStart, odNum, dLo, spatialBlock, outBlock);
        }
        CopyOut(ncTotal, ncStart, odStart, outBlock, isSplit);
    }
}
} // namespace AdaptivePool3d
#endif // ADAPTIVE_AVG_POOL3D_GATHER_TRANS_H_

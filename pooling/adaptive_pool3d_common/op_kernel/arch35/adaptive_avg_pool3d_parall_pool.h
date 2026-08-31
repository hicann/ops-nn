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
 * \file adaptive_avg_pool3d_parall_pool.h
 * \brief
 */

#ifndef ADAPTIVE_AVG_POOL3D_PARALL_POOL_H_
#define ADAPTIVE_AVG_POOL3D_PARALL_POOL_H_

#include "kernel_operator.h"
#include "../inc/kernel_utils.h"
#include "../inc/platform.h"
#include "../inc/load_store_utils.h"
#include "kernel_tiling/kernel_tiling.h"
#include "adaptive_pool3d_tiling_struct.h"
#include "pool_utils/arch35/compute/adaptive_pool3d_parall_pool_compute.h"
#include "pool_utils/arch35/data_move/adaptive_pool3d_parall_pool_data_move.h"
#include "pool_utils/arch35/data_move/adaptive_pool_transpose_data_move.h"

namespace AdaptivePool3d {
using namespace AscendC;
using namespace ops;

constexpr uint64_t TRANS_ADDR_LEN = 16;
constexpr uint64_t TRANS_LEN_B32 = 8;

struct BlockSplitParam {
    int64_t ncIdx;
    int64_t doIdx;
    int64_t hoIdx;
    int64_t woIdx;
    int64_t ncNum;
    int64_t doNum;
    int64_t hoNum;
    int64_t woNum;

    int64_t kerDStartIdx;
    int64_t kerHStartIdx;
    int64_t kerWStartIdx;

    int64_t diDataLen;
    int64_t hiDataLen;
    int64_t wiDataLen;
    int64_t xOffset;
};
constexpr AscendC::Reg::CastTrait castTraitI32Fp32 = {
    AscendC::Reg::RegLayout::UNKNOWN,
    AscendC::Reg::SatMode::UNKNOWN,
    AscendC::Reg::MaskMergeMode::ZEROING,
    AscendC::RoundMode::CAST_RINT,
};

template <typename T, typename ID_T>
class AdaptiveAvgPool3dParaPool {
public:
    __aicore__ inline AdaptiveAvgPool3dParaPool(
        const AdaptivePool3DTiling::AdaptivePool3dParaKernelTilingData& tilingData, TPipe& pipe)
        : tilingData_(tilingData), pipe_(pipe){};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y);
    __aicore__ inline void TransInput(uint32_t diDataLen, uint32_t hiDataLen, uint32_t wiDataLen);
    __aicore__ inline void CalKernelSize(int64_t kernelIdx, int32_t kernelNum, int64_t dimIn, int64_t dimOut,
                                         LocalTensor<int32_t> startIdxLocal, LocalTensor<int32_t> kernelSizeLocal);
    template <typename U>
    __aicore__ inline void CustomSelect(LocalTensor<U> inputLocal, LocalTensor<float> outLocal, uint16_t repeatTimes,
                                        uint16_t kernelSize, uint16_t srcStride);
    __aicore__ inline void AvgPoolOnW(uint32_t diDataLen, uint32_t hiDataLen, uint32_t wiDataLen, int64_t woIdx,
                                      uint32_t woNum);
    __aicore__ inline void AvgPoolOnH(uint32_t diDataLen, uint32_t hiDataLen, uint32_t woNum, int64_t hoIdx,
                                      uint32_t hoNum);
    __aicore__ inline void AvgPoolOnD(uint32_t diDataLen, uint32_t hoNum, uint32_t woNum, int64_t doIdx,
                                      uint32_t doNum);
    __aicore__ inline void CalAvg(int64_t doNum, int64_t hoNum, int64_t woNum);
    __aicore__ inline void TransOut(int64_t doNum, int64_t hoNum, int64_t woNum);
    __aicore__ inline void CopyOut(int64_t ncNum, int64_t doNum, int64_t hoNum, int64_t woNum, int64_t yGmOffset);
    __aicore__ inline void Process();

protected:
    TPipe pipe_;
    TQue<QuePosition::VECIN, 1> inputQue;
    TQue<QuePosition::VECOUT, 1> avgQue;
    TQue<QuePosition::VECOUT, 1> avgTransQue;
    TBuf<TPosition::VECCALC> startIdxBuf;
    TBuf<TPosition::VECCALC> dKerSizeBuf;
    TBuf<TPosition::VECCALC> hKerSizeBuf;
    TBuf<TPosition::VECCALC> wKerSizeBuf;
    GlobalTensor<T> xGm, yGm;

    int64_t inDHW_ = 1;
    int64_t inHW_ = 1;
    int64_t outDHW_ = 1;
    int64_t outHW_ = 1;
    int64_t startBlockIdx_ = 0;
    int64_t endBlockIdx_ = 0;

    uint32_t vlNum_;
    uint32_t ubBlockSize_;
    uint32_t ubAlignNum_;
    BlockSplitParam blockPara_;
    const AdaptivePool3DTiling::AdaptivePool3dParaKernelTilingData tilingData_;
    using IndexRegType = typename std::conditional<
        IsSameType<ID_T, int64_t>::value, typename AscendC::Reg::RegTensor<int64_t, AscendC::Reg::RegTraitNumTwo>,
        typename AscendC::Reg::RegTensor<int32_t>>::type;
};

template <typename T, typename ID_T>
__aicore__ inline void AdaptiveAvgPool3dParaPool<T, ID_T>::Init(GM_ADDR x, GM_ADDR y)
{
    if (GetBlockIdx() >= tilingData_.useCoreNum) {
        return;
    }
    inDHW_ = tilingData_.dIn * tilingData_.hIn * tilingData_.wIn;
    inHW_ = tilingData_.hIn * tilingData_.wIn;
    outDHW_ = tilingData_.dOut * tilingData_.hOut * tilingData_.wOut;
    outHW_ = tilingData_.hOut * tilingData_.wOut;

    vlNum_ = platform::GetVRegSize() / sizeof(T);
    ubBlockSize_ = platform::GetUbBlockSize();
    ubAlignNum_ = ubBlockSize_ / sizeof(T);

    int64_t curHandleBlockNum = tilingData_.blockFactor;
    if (GetBlockIdx() == tilingData_.useCoreNum - 1) {
        curHandleBlockNum = tilingData_.blockTail;
    }
    startBlockIdx_ = GetBlockIdx() * tilingData_.blockFactor;
    endBlockIdx_ = startBlockIdx_ + curHandleBlockNum;

    xGm.SetGlobalBuffer((__gm__ T*)x);
    yGm.SetGlobalBuffer((__gm__ T*)y);

    auto startBufSize = ops::CeilAlign(tilingData_.maxDimOut * sizeof(int32_t), static_cast<uint64_t>(ubBlockSize_));
    auto dBufSize = ops::CeilAlign(tilingData_.doFactor * sizeof(int32_t), static_cast<uint64_t>(ubBlockSize_));
    auto hBufSize = ops::CeilAlign(tilingData_.hoFactor * sizeof(int32_t), static_cast<uint64_t>(ubBlockSize_));
    auto wBufSize = ops::CeilAlign(tilingData_.woFactor * sizeof(int32_t), static_cast<uint64_t>(ubBlockSize_));

    pipe_.InitBuffer(startIdxBuf, startBufSize);
    pipe_.InitBuffer(dKerSizeBuf, dBufSize);
    pipe_.InitBuffer(hKerSizeBuf, hBufSize);
    pipe_.InitBuffer(wKerSizeBuf, wBufSize);

    pipe_.InitBuffer(inputQue, 1, tilingData_.maxInputSize * sizeof(T));
    pipe_.InitBuffer(avgQue, 1, tilingData_.maxInputSize * sizeof(float));
    pipe_.InitBuffer(avgTransQue, 1, tilingData_.maxInputSize * sizeof(float));
}

template <typename T, typename ID_T>
__aicore__ inline void AdaptiveAvgPool3dParaPool<T, ID_T>::TransInput(uint32_t diDataLen, uint32_t hiDataLen,
                                                                      uint32_t wiDataLen)
{
    uint32_t wiDataAlign = ops::CeilAlign(wiDataLen, ubAlignNum_);
    LocalTensor<T> xLocal = inputQue.DeQue<T>();
    LocalTensor<T> xLocalTransVL = avgQue.AllocTensor<T>();
    if constexpr (IsSameType<T, float>::value) {
        PoolUtils::DataMove::TransposeB32<T>(xLocalTransVL, xLocal, vlNum_, diDataLen * hiDataLen * wiDataAlign,
                                             ubBlockSize_);
    } else {
        PoolUtils::DataMove::TransposeB16<T>(xLocalTransVL, xLocal, vlNum_, diDataLen * hiDataLen * wiDataAlign,
                                             ubBlockSize_);
    }
    inputQue.FreeTensor(xLocal);
    avgQue.EnQue(xLocalTransVL);
}

template <typename T, typename ID_T>
__aicore__ inline void AdaptiveAvgPool3dParaPool<T, ID_T>::CalKernelSize(int64_t kernelIdx, int32_t kernelNum,
                                                                         int64_t dimIn, int64_t dimOut,
                                                                         LocalTensor<int32_t> startIdxLocal,
                                                                         LocalTensor<int32_t> kernelSizeLocal)
{
    __ubuf__ int32_t* startIdxAddr = (__ubuf__ int32_t*)startIdxLocal.GetPhyAddr();
    __ubuf__ int32_t* kernelSizeAddr = (__ubuf__ int32_t*)kernelSizeLocal.GetPhyAddr();

    uint32_t dataLen = kernelNum;
    uint16_t vfLen = platform::GetVRegSize() / sizeof(int32_t);
    uint16_t loopSize = ops::CeilDiv(static_cast<uint16_t>(kernelNum), vfLen);
    __VEC_SCOPE__
    {
        IndexRegType startIdxReg;
        IndexRegType endIdxReg;
        IndexRegType kerSizeReg;
        IndexRegType dupReg;
        IndexRegType gblStIdxReg;
        Reg::RegTensor<int32_t> startDstReg;
        Reg::RegTensor<int32_t> kSizeDstReg;
        Reg::MaskReg calMask;

        ID_T globalStartIdx = kernelIdx * dimIn / dimOut;
        Reg::Duplicate(dupReg, static_cast<ID_T>(dimOut));
        Reg::Duplicate(gblStIdxReg, globalStartIdx);
        for (uint16_t i = 0; i < loopSize; i++) {
            if constexpr (IsSameType<ID_T, int64_t>::value) {
                calMask = Reg::UpdateMask<ID_T, Reg::RegTraitNumTwo>(dataLen);
            } else {
                calMask = Reg::UpdateMask<ID_T>(dataLen);
            }
            auto startIdx = kernelIdx + i * vfLen;
            Reg::Arange(startIdxReg, startIdx);
            Reg::Adds(endIdxReg, startIdxReg, static_cast<ID_T>(1), calMask);
            Reg::Muls(startIdxReg, startIdxReg, static_cast<ID_T>(dimIn), calMask);
            Reg::Muls(endIdxReg, endIdxReg, static_cast<ID_T>(dimIn), calMask);
            Reg::Adds(endIdxReg, endIdxReg, static_cast<ID_T>(dimOut - 1), calMask);

            Reg::Div(startIdxReg, startIdxReg, dupReg, calMask);
            Reg::Div(endIdxReg, endIdxReg, dupReg, calMask);
            Reg::Sub(kerSizeReg, endIdxReg, startIdxReg, calMask);
            Reg::Sub(startIdxReg, startIdxReg, gblStIdxReg, calMask);

            if constexpr (IsSameType<ID_T, int64_t>::value) {
                startDstReg = (AscendC::Reg::RegTensor<int32_t>&)startIdxReg.reg[0];
                kSizeDstReg = (AscendC::Reg::RegTensor<int32_t>&)kerSizeReg.reg[0];
                Reg::StoreAlign(startIdxAddr + i * vfLen, startDstReg, calMask);
                Reg::StoreAlign(kernelSizeAddr + i * vfLen, kSizeDstReg, calMask);
            } else {
                Reg::StoreAlign(startIdxAddr + i * vfLen, startIdxReg, calMask);
                Reg::StoreAlign(kernelSizeAddr + i * vfLen, kerSizeReg, calMask);
            }
        }
    }
}

template <typename T, typename ID_T>
template <typename U>
__aicore__ inline void AdaptiveAvgPool3dParaPool<T, ID_T>::CustomSelect(LocalTensor<U> inputLocal,
                                                                        LocalTensor<float> outLocal,
                                                                        uint16_t repeatTimes, uint16_t kernelSize,
                                                                        uint16_t srcStride)
{
    __ubuf__ U* inputAddr = (__ubuf__ U*)inputLocal.GetPhyAddr();
    __ubuf__ float* outAddr = (__ubuf__ float*)outLocal.GetPhyAddr();

    int64_t vfLen = platform::GetVRegSize() / sizeof(T);
    int64_t vfLenFp32 = platform::GetVRegSize() / sizeof(float);
    __VEC_SCOPE__
    {
        Reg::RegTensor<float> inputReg;
        Reg::RegTensor<float> sumReg;

        Reg::MaskReg calMask = Reg::CreateMask<int32_t>();
        for (uint16_t i = 0; i < repeatTimes; i++) {
            auto srcStrideOfset = i * srcStride * vfLen;
            auto dstStrideOfset = i * vfLen;
            auto srcOfset = inputAddr + srcStrideOfset;
            auto dstOfset = outAddr + dstStrideOfset;

            ops::LoadOneTensorForDtypeT<U>(srcOfset, sumReg, calMask, 0);
            for (uint16_t k = 1; k < kernelSize; k++) {
                ops::LoadOneTensorForDtypeT<U>(srcOfset, inputReg, calMask, k * vfLen);
                Reg::Add(sumReg, inputReg, sumReg, calMask);
            }
            Reg::StoreAlign(dstOfset, sumReg, calMask);

            if constexpr (!IsSameType<T, float>::value) {
                ops::LoadOneTensorForDtypeT<U>(srcOfset, sumReg, calMask, vfLenFp32);
                for (uint16_t k = 1; k < kernelSize; k++) {
                    ops::LoadOneTensorForDtypeT<U>(srcOfset, inputReg, calMask, k * vfLen + vfLenFp32);
                    Reg::Add(sumReg, inputReg, sumReg, calMask);
                }
                Reg::StoreAlign(dstOfset + vfLenFp32, sumReg, calMask);
            }
        }
    }
}

template <typename T, typename ID_T>
__aicore__ inline void AdaptiveAvgPool3dParaPool<T, ID_T>::AvgPoolOnW(uint32_t diDataLen, uint32_t hiDataLen,
                                                                      uint32_t wiDataLen, int64_t woIdx, uint32_t woNum)
{
    LocalTensor<T> xTransLocal = avgQue.DeQue<T>();
    LocalTensor<float> avgOutLocal = avgTransQue.AllocTensor<float>();
    LocalTensor<int32_t> startIdxLocal = startIdxBuf.Get<int32_t>();
    LocalTensor<int32_t> kernelSizeLocal = wKerSizeBuf.Get<int32_t>();

    uint32_t wiDataAlign = ops::CeilAlign(wiDataLen, ubAlignNum_);
    auto repeat = diDataLen * hiDataLen;
    int64_t kerWStartIdxGlobal = woIdx * tilingData_.woFactor;
    CalKernelSize(kerWStartIdxGlobal, woNum, tilingData_.wIn, tilingData_.wOut, startIdxLocal, kernelSizeLocal);
    event_t eventIdVToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
    SetFlag<HardEvent::V_S>(eventIdVToS);
    WaitFlag<HardEvent::V_S>(eventIdVToS);
    for (int32_t kernelIdx = 0; kernelIdx < woNum; kernelIdx++) {
        auto kernelSize = kernelSizeLocal(kernelIdx);
        auto kerWOffset = startIdxLocal(kernelIdx);
        auto inputOffset = kerWOffset * vlNum_;
        auto avgBufferOffset = kernelIdx * repeat * vlNum_;
        CustomSelect<T>(xTransLocal[inputOffset], avgOutLocal[avgBufferOffset], repeat, kernelSize, wiDataAlign);
    }
    avgQue.FreeTensor(xTransLocal);
    avgTransQue.EnQue(avgOutLocal);
}

template <typename T, typename ID_T>
__aicore__ inline void AdaptiveAvgPool3dParaPool<T, ID_T>::AvgPoolOnH(uint32_t diDataLen, uint32_t hiDataLen,
                                                                      uint32_t woNum, int64_t hoIdx, uint32_t hoNum)
{
    LocalTensor<float> wTransLocal = avgTransQue.DeQue<float>();
    LocalTensor<float> avgOutLocal = avgQue.AllocTensor<float>();
    LocalTensor<int32_t> startIdxLocal = startIdxBuf.Get<int32_t>();
    LocalTensor<int32_t> kernelSizeLocal = hKerSizeBuf.Get<int32_t>();

    uint32_t woNumAlign = ops::CeilAlign(woNum, ubAlignNum_);
    auto repeat = woNum * diDataLen;
    int64_t kerHStartIdxGlobal = hoIdx * tilingData_.hoFactor;
    CalKernelSize(kerHStartIdxGlobal, hoNum, tilingData_.hIn, tilingData_.hOut, startIdxLocal, kernelSizeLocal);
    event_t eventIdVToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
    SetFlag<HardEvent::V_S>(eventIdVToS);
    WaitFlag<HardEvent::V_S>(eventIdVToS);
    for (int kernelIdx = 0; kernelIdx < hoNum; kernelIdx++) {
        auto kernelSize = kernelSizeLocal(kernelIdx);
        auto kerHOffset = startIdxLocal(kernelIdx);
        auto inputOffset = kerHOffset * vlNum_;
        auto avgBufferOffset = kernelIdx * woNumAlign * diDataLen * vlNum_;
        CustomSelect<float>(wTransLocal[inputOffset], avgOutLocal[avgBufferOffset], repeat, kernelSize, hiDataLen);
    }
    avgTransQue.EnQue(wTransLocal);
    avgQue.EnQue(avgOutLocal);
}

template <typename T, typename ID_T>
__aicore__ inline void AdaptiveAvgPool3dParaPool<T, ID_T>::AvgPoolOnD(uint32_t diDataLen, uint32_t hoNum,
                                                                      uint32_t woNum, int64_t doIdx, uint32_t doNum)
{
    LocalTensor<float> hTransLocal = avgQue.DeQue<float>();
    LocalTensor<float> avgOutLocal = avgTransQue.DeQue<float>();
    LocalTensor<int32_t> startIdxLocal = startIdxBuf.Get<int32_t>();
    LocalTensor<int32_t> kernelSizeLocal = dKerSizeBuf.Get<int32_t>();

    uint32_t woNumAlign = ops::CeilAlign(woNum, ubAlignNum_);
    auto repeat = hoNum * woNumAlign;
    int64_t kerDStartIdxGlobal = doIdx * tilingData_.doFactor;
    CalKernelSize(kerDStartIdxGlobal, doNum, tilingData_.dIn, tilingData_.dOut, startIdxLocal, kernelSizeLocal);
    event_t eventIdVToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
    SetFlag<HardEvent::V_S>(eventIdVToS);
    WaitFlag<HardEvent::V_S>(eventIdVToS);
    for (int kernelIdx = 0; kernelIdx < doNum; kernelIdx++) {
        auto kernelSize = kernelSizeLocal(kernelIdx);
        auto kerDOffset = startIdxLocal(kernelIdx);
        auto inputOffset = kerDOffset * vlNum_;
        auto avgBufferOffset = kernelIdx * repeat * vlNum_;
        CustomSelect<float>(hTransLocal[inputOffset], avgOutLocal[avgBufferOffset], repeat, kernelSize, diDataLen);
    }
    avgQue.EnQue(hTransLocal);
    avgTransQue.EnQue(avgOutLocal);
}

template <typename T, typename ID_T>
__aicore__ inline void AdaptiveAvgPool3dParaPool<T, ID_T>::CalAvg(int64_t doNum, int64_t hoNum, int64_t woNum)
{
    LocalTensor<float> sumLocal = avgTransQue.DeQue<float>();
    LocalTensor<int32_t> dKerSizeLocal = dKerSizeBuf.Get<int32_t>();
    LocalTensor<int32_t> hKerSizeLocal = hKerSizeBuf.Get<int32_t>();
    LocalTensor<int32_t> wKerSizeLocal = wKerSizeBuf.Get<int32_t>();

    __ubuf__ float* sumAddr = (__ubuf__ float*)sumLocal.GetPhyAddr();
    __ubuf__ int32_t* dKerSizeAddr = (__ubuf__ int32_t*)dKerSizeLocal.GetPhyAddr();
    __ubuf__ int32_t* hKerSizeAddr = (__ubuf__ int32_t*)hKerSizeLocal.GetPhyAddr();
    __ubuf__ int32_t* wKerSizeAddr = (__ubuf__ int32_t*)wKerSizeLocal.GetPhyAddr();

    uint32_t woNumAlign = ops::CeilAlign(woNum, static_cast<int64_t>(ubAlignNum_));
    uint16_t vfLenFp32 = platform::GetVRegSize() / sizeof(float);

    __VEC_SCOPE__
    {
        Reg::RegTensor<float> sumReg;
        Reg::RegTensor<float> avgReg;
        Reg::RegTensor<float> cntReg;
        Reg::RegTensor<int32_t> cntRegInt;
        Reg::MaskReg calMask = Reg::CreateMask<int32_t>();

        for (uint16_t dIdx = 0; dIdx < static_cast<uint16_t>(doNum); dIdx++) {
            int32_t dSize = dKerSizeAddr[dIdx];
            auto dOfset = dIdx * hoNum * woNumAlign;
            for (uint16_t hIdx = 0; hIdx < static_cast<uint16_t>(hoNum); hIdx++) {
                int32_t hSize = hKerSizeAddr[hIdx];
                auto hOfset = hIdx * woNumAlign;
                for (uint16_t wIdx = 0; wIdx < static_cast<uint16_t>(woNum); wIdx++) {
                    int32_t wSize = wKerSizeAddr[wIdx];
                    auto srcOfset = sumAddr + (dOfset + hOfset + wIdx) * vlNum_;
                    int32_t kerSize = dSize * hSize * wSize;
                    Reg::Duplicate(cntRegInt, kerSize);
                    Reg::Cast<float, int32_t, castTraitI32Fp32>(cntReg, cntRegInt, calMask);
                    Reg::LoadAlign(sumReg, srcOfset);
                    Reg::Div(avgReg, sumReg, cntReg, calMask);
                    Reg::StoreAlign(srcOfset, avgReg, calMask);
                    if constexpr (!IsSameType<T, float>::value) {
                        srcOfset += vfLenFp32;
                        Reg::LoadAlign(sumReg, srcOfset);
                        Reg::Div(avgReg, sumReg, cntReg, calMask);
                        Reg::StoreAlign(srcOfset, avgReg, calMask);
                    }
                }
            }
        }
    }
    avgTransQue.EnQue(sumLocal);
}

template <typename T, typename ID_T>
__aicore__ inline void AdaptiveAvgPool3dParaPool<T, ID_T>::TransOut(int64_t doNum, int64_t hoNum, int64_t woNum)
{
    int64_t woNumAlign = ops::CeilAlign(woNum, static_cast<int64_t>(ubAlignNum_));
    int64_t rowNum = doNum * hoNum * woNumAlign;

    LocalTensor<float> dTransLocal = avgTransQue.DeQue<float>();
    LocalTensor<float> avgOutLocal = avgQue.DeQue<float>();

    /* 5HD transpose时rowNum按16对齐 */
    auto rowNumAlign = ops::CeilAlign(static_cast<uint64_t>(rowNum), TRANS_ADDR_LEN);
    PoolUtils::DataMove::TransposeB32<float>(avgOutLocal, dTransLocal, rowNumAlign, vlNum_, ubBlockSize_);
    avgTransQue.FreeTensor(dTransLocal);
    avgQue.EnQue(avgOutLocal);
}

template <typename T, typename ID_T>
__aicore__ inline void AdaptiveAvgPool3dParaPool<T, ID_T>::CopyOut(int64_t ncNum, int64_t doNum, int64_t hoNum,
                                                                   int64_t woNum, int64_t yGmOffset)
{
    uint32_t woNumAlign = ops::CeilAlign(woNum, static_cast<int64_t>(ubAlignNum_));
    LocalTensor<float> avgOutLocal = avgQue.DeQue<float>();

    DataCopyExtParams valueParams;
    valueParams.blockCount = hoNum;
    valueParams.blockLen = woNum * sizeof(T);
    valueParams.srcStride = 0;
    valueParams.dstStride = (tilingData_.wOut - woNum) * sizeof(T);

    auto dhwInStride = ops::CeilAlign(static_cast<uint64_t>(doNum * hoNum * woNumAlign), TRANS_ADDR_LEN);
    LoopModeParams loopModeParams;
    loopModeParams.loop1Size = doNum;
    loopModeParams.loop2Size = ncNum;
    loopModeParams.loop1SrcStride = hoNum * woNumAlign * sizeof(T);
    loopModeParams.loop2SrcStride = dhwInStride * sizeof(T);
    loopModeParams.loop1DstStride = outHW_ * sizeof(T);
    loopModeParams.loop2DstStride = outDHW_ * sizeof(T);
    SetLoopModePara(loopModeParams, DataCopyMVType::UB_TO_OUT);

    if constexpr (IsSameType<T, float>::value) {
        DataCopyPad(yGm[yGmOffset], avgOutLocal, valueParams);
    } else {
        LocalTensor<T> castOutLocal = avgTransQue.AllocTensor<T>();
        if constexpr (IsSameType<T, half>::value) {
            Cast(castOutLocal, avgOutLocal, RoundMode::CAST_NONE, ncNum * dhwInStride);
        } else {
            Cast(castOutLocal, avgOutLocal, RoundMode::CAST_RINT, ncNum * dhwInStride);
        }
        avgTransQue.EnQue(castOutLocal);
        castOutLocal = avgTransQue.DeQue<T>();
        DataCopyPad(yGm[yGmOffset], castOutLocal, valueParams);
        avgTransQue.FreeTensor(castOutLocal);
    }
    ResetLoopModePara(DataCopyMVType::UB_TO_OUT);

    avgQue.FreeTensor(avgOutLocal);
}

template <typename T, typename ID_T>
__aicore__ inline void AdaptiveAvgPool3dParaPool<T, ID_T>::Process()
{
    if (GetBlockIdx() >= tilingData_.useCoreNum) {
        return;
    }
    if (startBlockIdx_ >= endBlockIdx_) {
        return;
    }
    PoolUtils::Compute::CalInputBlockPara(startBlockIdx_, blockPara_, tilingData_, inDHW_, inHW_);
    PoolUtils::DataMove::CopyInput<T>(inputQue, xGm, blockPara_.ncNum, blockPara_.diDataLen, blockPara_.hiDataLen,
                                      blockPara_.wiDataLen, blockPara_.xOffset, ubAlignNum_, tilingData_.wIn, inHW_,
                                      inDHW_);
    for (auto curIdx = startBlockIdx_; curIdx < endBlockIdx_ - 1; curIdx++) {
        auto ncIdx = blockPara_.ncIdx;
        auto doIdx = blockPara_.doIdx;
        auto hoIdx = blockPara_.hoIdx;
        auto woIdx = blockPara_.woIdx;
        auto ncNum = blockPara_.ncNum;
        auto doNum = blockPara_.doNum;
        auto hoNum = blockPara_.hoNum;
        auto woNum = blockPara_.woNum;

        auto kerDStartIdx = blockPara_.kerDStartIdx;
        auto kerHStartIdx = blockPara_.kerHStartIdx;
        auto kerWStartIdx = blockPara_.kerWStartIdx;

        auto diDataLen = blockPara_.diDataLen;
        auto hiDataLen = blockPara_.hiDataLen;
        auto wiDataLen = blockPara_.wiDataLen;
        auto xOffset = blockPara_.xOffset;

        TransInput(diDataLen, hiDataLen, wiDataLen);

        PoolUtils::Compute::CalInputBlockPara(curIdx + 1, blockPara_, tilingData_, inDHW_, inHW_);
        PoolUtils::DataMove::CopyInput<T>(inputQue, xGm, blockPara_.ncNum, blockPara_.diDataLen, blockPara_.hiDataLen,
                                          blockPara_.wiDataLen, blockPara_.xOffset, ubAlignNum_, tilingData_.wIn, inHW_,
                                          inDHW_);

        /* outshape: [woNum, hiDataLen, diDataLen, vl] */
        AvgPoolOnW(diDataLen, hiDataLen, wiDataLen, woIdx, woNum);

        /* outshape: [hoNum, woNumAlign, diDataLen, vl] */
        AvgPoolOnH(diDataLen, hiDataLen, woNum, hoIdx, hoNum);

        /* outshape: [doNum, hoNum, woNumAlign, vl] */
        AvgPoolOnD(diDataLen, hoNum, woNum, doIdx, doNum);

        int64_t yOffset = ncIdx * tilingData_.ncFactor * outDHW_ + doIdx * tilingData_.doFactor * outHW_ +
                          hoIdx * tilingData_.hoFactor * tilingData_.wOut + woIdx * tilingData_.woFactor;

        CalAvg(doNum, hoNum, woNum);
        TransOut(doNum, hoNum, woNum);
        CopyOut(ncNum, doNum, hoNum, woNum, yOffset);
    }
    /* 单独处理最后一个block */
    TransInput(blockPara_.diDataLen, blockPara_.hiDataLen, blockPara_.wiDataLen);
    /* outshape: [woNum, hiDataLen, diDataLen, vl] */
    AvgPoolOnW(blockPara_.diDataLen, blockPara_.hiDataLen, blockPara_.wiDataLen, blockPara_.woIdx, blockPara_.woNum);
    /* outshape: [hoNum, woNumAlign, diDataLen, vl] */
    AvgPoolOnH(blockPara_.diDataLen, blockPara_.hiDataLen, blockPara_.woNum, blockPara_.hoIdx, blockPara_.hoNum);
    /* outshape: [doNum, hoNum, woNumAlign, vl] */
    AvgPoolOnD(blockPara_.diDataLen, blockPara_.hoNum, blockPara_.woNum, blockPara_.doIdx, blockPara_.doNum);
    int64_t yOffset = blockPara_.ncIdx * tilingData_.ncFactor * outDHW_ +
                      blockPara_.doIdx * tilingData_.doFactor * outHW_ +
                      blockPara_.hoIdx * tilingData_.hoFactor * tilingData_.wOut +
                      blockPara_.woIdx * tilingData_.woFactor;
    CalAvg(blockPara_.doNum, blockPara_.hoNum, blockPara_.woNum);
    TransOut(blockPara_.doNum, blockPara_.hoNum, blockPara_.woNum);
    CopyOut(blockPara_.ncNum, blockPara_.doNum, blockPara_.hoNum, blockPara_.woNum, yOffset);
}
} // namespace AdaptivePool3d
#endif // ADAPTIVE_AVG_POOL3D_PARALL_POOL_H_

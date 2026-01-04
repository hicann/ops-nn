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
 * \file pool_3d_ncdhw_big_kernel.h
 * \brief
 */
#ifndef POOL_3D_BIG_KERNEL_H_
#define POOL_3D_BIG_KERNEL_H_

#include "pool_3d_common.h"
#include "../inc/platform.h"
#include "../inc/kernel_utils.h"

namespace Pool3D
{
using namespace AscendC;

static constexpr int32_t OUT_BUFFER_LEN = 1024;

template <typename T, int32_t OP_TYPE>
class Pool3DNcdhwBigKernel
{
public:
    __aicore__ inline Pool3DNcdhwBigKernel(TPipe* pipe, const Pool3DNcdhwBigKernelTilingData* __restrict tiling)
        : pipe_(pipe), tilingData_(tiling){};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CalcKernelSize(int64_t curIdx, int64_t& curkD, int64_t& curkH, int64_t& curkW,
                                          int64_t& curInOffset);
    template <bool SPLIT_KERNEL>
    __aicore__ inline void BaseCompute(int64_t beginIdx, int64_t endIdx, int64_t maxCount);
    __aicore__ inline void CopyInMultiItems(int64_t offset, int64_t blockLen, int64_t blockCount, int64_t loopCount);
    __aicore__ inline int64_t AdapterKsize(int64_t dimIndex);
    __aicore__ inline void CopyMaxOut(int64_t curIdx);
    __aicore__ inline void CopyResultToUb(int64_t curIdx);
    __aicore__ inline void NoSplitKernelProcess(int32_t localCurIdx, int64_t curkD, int64_t curkH, int64_t curkW,
                                                int64_t curInOffset, int64_t maxCount);
    __aicore__ inline void ComputeSum(int64_t length);
    __aicore__ inline void ComputeAvg();
    __aicore__ inline void SplitKernelProcess(int32_t localCurIdx, int64_t curkD, int64_t curkH, int64_t curkW,
                                              int64_t curInOffset, int64_t maxCount);
    template <bool CLEAR>
    __aicore__ inline void InitOutLocal(int32_t localCurIdx);
    __aicore__ inline int64_t min(int64_t a, int64_t b)
    {
        return (a > b) ? b : a;
    }

    TPipe* pipe_;
    // 输入队列
    TQue<QuePosition::VECIN, BUFFER_NUM> inputQue_;
    // cast 输入转换
    TBuf<> ubSizeCast_;
    // cast 循环累加结果
    TBuf<> ubLoopReduce_;
    // cast 最终结果
    TBuf<> ubLoopResult_;
    // 输出ub
    TBuf<> uBOutput_;

    GlobalTensor<T> xGm_;
    GlobalTensor<T> maxGm_;

    const Pool3DNcdhwBigKernelTilingData* tilingData_;

    float mulsFactor_ = 0;

    int64_t inDHW_ = 1;
    int64_t curOriginD_ = 0;
    int64_t curOriginH_ = 0;
    int64_t curOriginW_ = 0;
    int64_t curOriginIndex_ = 0;
    int64_t beginIdx_ = 0;
    int64_t endIdx_ = 0;
};

template <typename T, int32_t OP_TYPE>
__aicore__ inline int64_t Pool3DNcdhwBigKernel<T, OP_TYPE>::AdapterKsize(int64_t dimIndex)
{
    if (dimIndex == DIM_D) {
        return (tilingData_->kD - ONE) * tilingData_->dD + 1;
    }
    if (dimIndex == DIM_H) {
        return (tilingData_->kH - ONE) * tilingData_->dH + 1;
    }
    if (dimIndex == DIM_W) {
        return (tilingData_->kW - ONE) * tilingData_->dW + 1;
    }
    return 0;
}

template <typename T, int32_t OP_TYPE>
__aicore__ inline void Pool3DNcdhwBigKernel<T, OP_TYPE>::Init(GM_ADDR x, GM_ADDR y)
{
    inDHW_ = tilingData_->dInDim * tilingData_->hInDim * tilingData_->wInDim;
    if (GetBlockIdx() < tilingData_->blockTail) {
        beginIdx_ = GetBlockIdx() * (tilingData_->blockFactor + ONE);
        endIdx_ = beginIdx_ + tilingData_->blockFactor + 1;
    } else {
        beginIdx_ = GetBlockIdx() * tilingData_->blockFactor + tilingData_->blockTail;
        endIdx_ = beginIdx_ + tilingData_->blockFactor;
    }
    // GM
    xGm_.SetGlobalBuffer((__gm__ T*)x);
    maxGm_.SetGlobalBuffer((__gm__ T*)y);

    pipe_->InitBuffer(inputQue_, BUFFER_NUM, tilingData_->maxCount * sizeof(T));
    pipe_->InitBuffer(uBOutput_, OUT_BUFFER_LEN * sizeof(T));
    pipe_->InitBuffer(ubLoopReduce_, NUM128);
    pipe_->InitBuffer(ubLoopResult_, NUM128);

    if constexpr (OP_TYPE == OP_TYPE_AVG_POOL_3D && !std::is_same<T, float>::value) {
        pipe_->InitBuffer(ubSizeCast_, tilingData_->maxCount * sizeof(float));
    }
}

template <typename T, int32_t OP_TYPE>
__aicore__ inline void Pool3DNcdhwBigKernel<T, OP_TYPE>::Process()
{
    int64_t tailTwoDimLen = AdapterKsize(DIM_H) * AdapterKsize(DIM_W);

    if (AdapterKsize(DIM_D) * tailTwoDimLen <= tilingData_->maxCount) {
        BaseCompute<false>(beginIdx_, endIdx_, tilingData_->maxCount);
    } else {
        BaseCompute<true>(beginIdx_, endIdx_, tilingData_->maxCount);
    }
}

template <typename T, int32_t OP_TYPE>
__aicore__ inline void Pool3DNcdhwBigKernel<T, OP_TYPE>::CalcKernelSize(int64_t curIdx, int64_t& curkD, int64_t& curkH,
                                                                        int64_t& curkW, int64_t& curInOffset)
{
    int64_t outDHW = tilingData_->dOutDim * tilingData_->hOutDim * tilingData_->wOutDim;
    int64_t outHW = tilingData_->hOutDim * tilingData_->wOutDim;
    int64_t cur3D = curIdx % outDHW;
    int64_t curNc = curIdx / outDHW;
    int64_t curDo = cur3D / outHW;
    int64_t cur2D = cur3D % outHW;
    int64_t curHo = cur2D / tilingData_->wOutDim;
    int64_t curWo = cur2D % tilingData_->wOutDim;
    int64_t curkPadD = 0;
    CalcKernelSizeCore(ParamsForDim{tilingData_->dInDim, curDo, tilingData_->kD, tilingData_->sD, tilingData_->dD,
                                    tilingData_->fPad, tilingData_->bkPad},
                       curkD, curkPadD, curOriginD_);
    int64_t curkPadH = 0;
    CalcKernelSizeCore(ParamsForDim{tilingData_->hInDim, curHo, tilingData_->kH, tilingData_->sH, tilingData_->dH,
                                    tilingData_->tPad, tilingData_->bPad},
                       curkH, curkPadH, curOriginH_);
    int64_t curkPadW = 0;
    CalcKernelSizeCore(ParamsForDim{tilingData_->wInDim, curWo, tilingData_->kW, tilingData_->sW, tilingData_->dW,
                                    tilingData_->lPad, tilingData_->rPad},
                       curkW, curkPadW, curOriginW_);
    curOriginIndex_ =
        curOriginD_ * tilingData_->hInDim * tilingData_->wInDim + curOriginH_ * tilingData_->wInDim + curOriginW_;
    curInOffset = curNc * inDHW_ + curOriginIndex_;

    if (OP_TYPE == OP_TYPE_AVG_POOL_3D) {
        if (tilingData_->divisorOverride > 0) {
            mulsFactor_ = 1.0f / static_cast<float>(tilingData_->divisorOverride);
        } else if (tilingData_->countIncludePad == 0) {
            mulsFactor_ = curkD * curkH * curkW == 0 ? 0 : 1.0f / static_cast<float>(curkD * curkH * curkW);
        } else {
            mulsFactor_ = 1.0f / static_cast<float>(curkPadD * curkPadH * curkPadW);
        }
    }
}

template <typename T, int32_t OP_TYPE>
template <bool SPLIT_KERNEL>
__aicore__ inline void Pool3DNcdhwBigKernel<T, OP_TYPE>::BaseCompute(int64_t beginIdx, int64_t endIdx, int64_t maxCount)
{
    int64_t curkD = 1;
    int64_t curkH = 1;
    int64_t curkW = 1;
    int64_t curInOffset = 0;
    // current blockdim range
    for (int64_t idx = beginIdx; idx < endIdx; idx++) {
        CalcKernelSize(idx, curkD, curkH, curkW, curInOffset);
        int32_t localCurIdx = (idx - beginIdx) % OUT_BUFFER_LEN;
        if constexpr (SPLIT_KERNEL) {
            InitOutLocal<true>(localCurIdx);
            SplitKernelProcess(localCurIdx, curkD, curkH, curkW, curInOffset, maxCount);
        } else {
            InitOutLocal<true>(localCurIdx);
            NoSplitKernelProcess(localCurIdx, curkD, curkH, curkW, curInOffset, maxCount);
        }
        CopyResultToUb(localCurIdx);
        CopyMaxOut(idx);
    }
}

template <typename T, int32_t OP_TYPE>
__aicore__ inline void Pool3DNcdhwBigKernel<T, OP_TYPE>::CopyInMultiItems(int64_t offset, int64_t blockLen,
                                                                          int64_t blockCount, int64_t loopCount)
{
    LocalTensor<T> xLocal = inputQue_.AllocTensor<T>();
    // NDDMA loopInfo init
    MultiCopyLoopInfo<DIM3> loopInfo;
    loopInfo.loopSize[ZERO] = blockLen;
    loopInfo.loopSize[ONE] = blockCount;
    loopInfo.loopSize[TWO] = loopCount;

    loopInfo.loopSrcStride[ZERO] = tilingData_->dW;
    loopInfo.loopSrcStride[ONE] = tilingData_->dH * tilingData_->wInDim;
    loopInfo.loopSrcStride[TWO] = tilingData_->dD * tilingData_->hInDim * tilingData_->wInDim;

    loopInfo.loopDstStride[ZERO] = ONE;
    loopInfo.loopDstStride[ONE] = blockLen;
    loopInfo.loopDstStride[TWO] = blockLen * blockCount;

    static constexpr MultiCopyConfig config = {false};
    MultiCopyParams<T, DIM3> paramsMain = {loopInfo};
    DataCopy<T, DIM3, config>(xLocal, xGm_[offset], paramsMain);
    inputQue_.EnQue(xLocal);
}

template <typename T, int32_t OP_TYPE>
__aicore__ inline void Pool3DNcdhwBigKernel<T, OP_TYPE>::CopyResultToUb(int64_t curIdx)
{
    LocalTensor<T> uboutLocal = uBOutput_.Get<T>();
    __local_mem__ T* dstAddr = (__local_mem__ T*)uboutLocal.GetPhyAddr() + curIdx;

    LocalTensor<T> ubResult = ubLoopResult_.Get<T>();
    __local_mem__ T* srcAddr = (__local_mem__ T*)ubResult.GetPhyAddr();

    __VEC_SCOPE__
    {
        MicroAPI::RegTensor<T> res;
        MicroAPI::UnalignReg u0;
        MicroAPI::DataCopyUnAlignPre(u0, srcAddr);
        MicroAPI::DataCopyUnAlign(res, u0, srcAddr, ONE);

        MicroAPI::UnalignReg u1;
        MicroAPI::DataCopyUnAlign(dstAddr, res, u1, ONE);
        MicroAPI::DataCopyUnAlignPost(dstAddr, u1, 0);
    }
}

template <typename T, int32_t OP_TYPE>
__aicore__ inline void Pool3DNcdhwBigKernel<T, OP_TYPE>::CopyMaxOut(int64_t curIdx)
{
    constexpr int32_t maxLocalLen = OUT_BUFFER_LEN;
    int32_t localCurIdx = (curIdx - beginIdx_) % maxLocalLen;

    if (localCurIdx == maxLocalLen - 1 || curIdx == endIdx_ - 1) {
        LocalTensor<T> uboutLocal = uBOutput_.Get<T>();
        DataCopyExtParams extParams;
        extParams.blockCount = 1;
        extParams.blockLen = (localCurIdx + ONE) * sizeof(T);
        extParams.srcStride = 0;
        extParams.dstStride = 0;
        event_t eventIdVtoMTE3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
        SetFlag<HardEvent::V_MTE3>(eventIdVtoMTE3);
        WaitFlag<HardEvent::V_MTE3>(eventIdVtoMTE3);
        DataCopyPad(maxGm_[curIdx - localCurIdx], uboutLocal, extParams);
    }
}

template <typename T, int32_t OP_TYPE>
__aicore__ inline void Pool3DNcdhwBigKernel<T, OP_TYPE>::ComputeAvg()
{
    if constexpr (OP_TYPE == OP_TYPE_AVG_POOL_3D) {
        if constexpr (std::is_same<T, float>::value) {
            LocalTensor<T> loopReduce = ubLoopReduce_.Get<T>();
            LocalTensor<T> resultLocal = ubLoopResult_.Get<T>();
            Muls(resultLocal, loopReduce, mulsFactor_, ONE);
        } else {
            LocalTensor<float> loopReduce = ubLoopReduce_.Get<float>();
            Muls(loopReduce, loopReduce, mulsFactor_, ONE);
            LocalTensor<T> resultLocal = ubLoopResult_.Get<T>();
            Cast(resultLocal, loopReduce, RoundMode::CAST_ROUND, ONE);
        }
    }
}

template <typename T, int32_t OP_TYPE>
__aicore__ inline void Pool3DNcdhwBigKernel<T, OP_TYPE>::ComputeSum(int64_t length)
{
    LocalTensor<T> xLocal = inputQue_.DeQue<T>();
    if (OP_TYPE == OP_TYPE_MAX_POOL_3D) {
        LocalTensor<T> resultLocal = ubLoopResult_.Get<T>();
        uint32_t srcShape[TWO] = {static_cast<uint32_t>(ONE), static_cast<uint32_t>(length)};
        ReduceMax<T, AscendC::Pattern::Reduce::AR, true>(xLocal, xLocal, srcShape, false);
        Max(resultLocal, resultLocal, xLocal, ONE);
    } else if constexpr (OP_TYPE == OP_TYPE_AVG_POOL_3D) {
        if constexpr (std::is_same<T, float>::value) {
            LocalTensor<T> loopReduce = ubLoopReduce_.Get<T>();
            ReduceSum<T>(xLocal, xLocal, xLocal, length);
            Add(loopReduce, loopReduce, xLocal, ONE);
        } else {
            LocalTensor<float> ubCast = ubSizeCast_.Get<float>();
            Cast(ubCast, xLocal, RoundMode::CAST_NONE, length);
            ReduceSum<float>(ubCast, ubCast, ubCast, length);
            LocalTensor<float> loopReduce = ubLoopReduce_.Get<float>();
            Add(loopReduce, loopReduce, ubCast, ONE);
        }
    }
    inputQue_.FreeTensor<T>(xLocal);
}

template <typename T, int32_t OP_TYPE>
__aicore__ inline void Pool3DNcdhwBigKernel<T, OP_TYPE>::NoSplitKernelProcess(int32_t localCurIdx, int64_t curkD,
                                                                              int64_t curkH, int64_t curkW,
                                                                              int64_t curInOffset, int64_t maxCount)
{
    if (curkD * curkH * curkW == 0) {
        return;
    }
    CopyInMultiItems(curInOffset, curkW, curkH, curkD);
    ComputeSum(curkW * curkH * curkD);
    ComputeAvg();
}

template <typename T, int32_t OP_TYPE>
__aicore__ inline void Pool3DNcdhwBigKernel<T, OP_TYPE>::SplitKernelProcess(int32_t localCurIdx, int64_t curkD,
                                                                            int64_t curkH, int64_t curkW,
                                                                            int64_t curInOffset, int64_t maxCount)
{
    if (curkD * curkH * curkW == 0) {
        return;
    }
    int64_t realIndex = 0;
    int64_t maxIndex = 0;
    int64_t inputOffset = curInOffset;
    T padValue = GetPadValue<T, OP_TYPE>();
    int64_t curkHW = curkH * curkW;
    if (curkHW <= maxCount) {  // ZHENGHANG行搬入
        int64_t dFactor = maxCount / curkHW;
        int64_t dLoops = (curkD + dFactor - ONE) / dFactor;
        int64_t dTail = curkD - (dLoops - ONE) * dFactor;
        for (int64_t dLoop = 0; dLoop < dLoops; dLoop++) {
            int32_t curdFactor = dLoop == dLoops - 1 ? dTail : dFactor;
            CopyInMultiItems(inputOffset, curkW, curkH, curdFactor);
            ComputeSum(curkW * curkH * curdFactor);
            inputOffset += dFactor * tilingData_->hInDim * tilingData_->wInDim * tilingData_->dW;
        }
    } else if (curkW <= maxCount) { // 整行搬入
        int64_t hFactor = maxCount / curkW;
        int64_t hLoops = (curkH + hFactor - ONE) / hFactor;
        int64_t hTail = curkH - (hLoops - ONE) * hFactor;
        for (int64_t dLoop = 0; dLoop < curkD; dLoop++) {
            inputOffset = curInOffset + dLoop * tilingData_->hInDim * tilingData_->wInDim * tilingData_->dD;
            for (int64_t hLoop = 0; hLoop < hLoops; hLoop++) {
                int32_t curhFactor = hLoop == hLoops - 1 ? hTail : hFactor;
                CopyInMultiItems(inputOffset, curkW, curhFactor, ONE);
                ComputeSum(curkW * curhFactor);
                inputOffset += hFactor * tilingData_->wInDim * tilingData_->dH;
            }
        }
    } else { // 单行很大，单行循环搬
        int64_t wFactor = maxCount;
        int64_t wLoops = (curkW + wFactor - ONE) / wFactor;
        int64_t wTail = curkW - (wLoops - ONE) * wFactor;
        for (int64_t dLoop = 0; dLoop < curkD; dLoop++) {
            int64_t dOffSet = curInOffset + dLoop * tilingData_->hInDim * tilingData_->wInDim * tilingData_->dD;
            for (int64_t hLoop = 0; hLoop < curkH; hLoop++) {
                inputOffset = dOffSet + hLoop * tilingData_->wInDim * tilingData_->dH;
                for (int64_t wLoop = 0; wLoop < wLoops; wLoop++) {
                    int32_t curFactor = wLoop == wLoops - 1 ? wTail : wFactor;
                    CopyInMultiItems(inputOffset, curFactor, ONE, ONE);
                    ComputeSum(curFactor);
                    inputOffset += wFactor * tilingData_->dW;
                }
            }
        }
    }
    ComputeAvg();
}

template <typename T, int32_t OP_TYPE>
template <bool CLEAR>
__aicore__ inline void Pool3DNcdhwBigKernel<T, OP_TYPE>::InitOutLocal(int32_t localCurIdx)
{
    T padValue = GetPadValue<T, OP_TYPE>();
    if constexpr (OP_TYPE == OP_TYPE_AVG_POOL_3D) {
        LocalTensor<float> loopReduce = ubLoopReduce_.Get<float>();
        Duplicate(loopReduce, 0.0f, ONE);
    }
    if constexpr (std::is_same<T, bfloat16_t>::value) {
        uint16_t ValueHalf = (OP_TYPE == OP_TYPE_MAX_POOL_3D) ? BFLOAT16_NEG_INF : 0;
        LocalTensor<uint16_t> resultLocal = ubLoopResult_.Get<uint16_t>();
        Duplicate(resultLocal, ValueHalf, ONE);
    } else {
        LocalTensor<T> resultLocal = ubLoopResult_.Get<T>();
        Duplicate(resultLocal, padValue, ONE);
    }
    if (localCurIdx != 0) {
        return;
    }
    constexpr int32_t maxLocalLen = OUT_BUFFER_LEN;
    event_t eventIdMTE3toV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));
    SetFlag<HardEvent::MTE3_V>(eventIdMTE3toV);
    WaitFlag<HardEvent::MTE3_V>(eventIdMTE3toV);
    if constexpr (!CLEAR) {  // kerel 全载场景无需merge，因此无需初始化output
        return;
    }
    LocalTensor<T> uboutLocal = uBOutput_.Get<T>();
    __local_mem__ T* dstAddr = (__local_mem__ T*)uboutLocal.GetPhyAddr();
    constexpr uint32_t repeatElm = platform::GetVRegSize() / sizeof(T);
    uint16_t repeatTimes = CeilDivision(maxLocalLen, repeatElm);
    uint32_t num = maxLocalLen;
    __local_mem__ T* addr = (__local_mem__ T*)dstAddr;
    __VEC_SCOPE__
    {
        MicroAPI::RegTensor<T> v0;
        if constexpr (std::is_same<T, bfloat16_t>::value) {
            uint16_t value = (OP_TYPE == OP_TYPE_MAX_POOL_3D) ? BFLOAT16_NEG_INF : 0;
            MicroAPI::Duplicate((MicroAPI::RegTensor<uint16_t>&)v0, value);
        } else {
            MicroAPI::Duplicate(v0, padValue);
        }
        for (uint16_t i = 0; i < repeatTimes; i++) {
            MicroAPI::MaskReg p0 = MicroAPI::UpdateMask<T>(num);
            if constexpr (sizeof(T) == B64) {
                MicroAPI::DataCopy(addr + i * repeatElm, v0, p0);
            } else {
                MicroAPI::AddrReg offsetReg = MicroAPI::CreateAddrReg<T>(i, repeatElm);
                MicroAPI::DataCopy(addr, v0, offsetReg, p0);
            }
        }
    }
}

}  // namespace Pool3D
#endif  // MAX_POOL_3D_BIG_KERNEL_H_

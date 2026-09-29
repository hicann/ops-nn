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
 * @file max_pool3d_grad_ndhwc_impl_scatter.h
 * @brief NDHWC 共享反向基类 BackwardBase（大小 kernel 模板共用）
 *
 * argmax 协议: dense [n][d'][h'][w'][C], 前向产出, 反向消费。
 * 叶子散射原语见 max_pool3d_grad_small_kernel_scatter.h。
 */

#ifndef MAX_POOL3D_GRAD_NDHWC_IMPL_SCATTER_H
#define MAX_POOL3D_GRAD_NDHWC_IMPL_SCATTER_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "../pool_grad_common/arch35/pool3d_grad_struct_common.h"
#include "max_pool3d_grad_small_kernel_scatter.h"

namespace MaxPool3DGradNDHWCNameSpace {

using namespace AscendC;
using namespace Pool3DGradNameSpace;
using MaxPool3DSmallKernelNameSpace::DivMagic;
using MaxPool3DSmallKernelNameSpace::DoMulNCNdhwcFastDiv;
using MaxPool3DSmallKernelNameSpace::DoSingleNCNdhwcFastDiv;
using MaxPool3DSmallKernelNameSpace::PrecomputeDiv;
using computeType = float;

constexpr int32_t BUFFER_NUM = 2;
constexpr int64_t DOUBLE_HEAD = 2;
constexpr int32_t HELP_BUFFER = 5120;

template <typename T, typename INDEX_T, const uint32_t IS_CHECK_RANGE>
class MaxPool3DGradNDHWCBackwardBase {
public:
    __aicore__ inline MaxPool3DGradNDHWCBackwardBase() {}

protected:
    __aicore__ inline void ParseTilingData(const Pool3DGradNDHWCTilingData& tilingData);
    __aicore__ inline void ScalarCompute(int64_t loopNum);
    __aicore__ inline void CopyInGrad();
    __aicore__ inline void CopyOut();
    __aicore__ inline void ProcessNoArgmaxBlock();
    __aicore__ inline void Backward();
    __aicore__ inline void VDrainToMte2();
    __aicore__ inline void Mte3Drain();
    __aicore__ inline void BackwardCompute(__ubuf__ float* yAddr, __ubuf__ T* gradAddr, __ubuf__ INDEX_T* argmaxAddr);

    __aicore__ inline void GenPattern2D(AscendC::Reg::RegTensor<int32_t>& indexReg, int32_t groupStride, int32_t cCnt,
                                        int32_t slot);
    __aicore__ inline void GenPattern3D(AscendC::Reg::RegTensor<int32_t>& indexReg, int32_t hiGroupStride,
                                        int32_t loGroupStride, int32_t loGroupCount, int32_t cCnt, int32_t slot);
    __aicore__ inline void GenPattern4D(AscendC::Reg::RegTensor<int32_t>& indexReg, int32_t dGroupStride,
                                        int32_t hGroupStride, int32_t wGroupStride, int32_t wGroupCount,
                                        int32_t hGroupCount, int32_t cCnt, int32_t slot);

    __aicore__ inline __ubuf__ int32_t* PatternSlotAddr(int32_t slot);

    __aicore__ inline void SingleLineProcessVF(__ubuf__ float* yAddr, __ubuf__ T* gradAddr,
                                               __ubuf__ INDEX_T* argmaxAddr);
    __aicore__ inline void MultipleLineHwProcessVF(__ubuf__ float* yAddr, __ubuf__ T* gradAddr,
                                                   __ubuf__ INDEX_T* argmaxAddr);
    __aicore__ inline void MultipleLineDhwProcessVF(__ubuf__ float* yAddr, __ubuf__ T* gradAddr,
                                                    __ubuf__ INDEX_T* argmaxAddr);
    __aicore__ inline void MultipleLineProcessVF2(__ubuf__ float* yAddr, __ubuf__ T* gradAddr,
                                                  __ubuf__ INDEX_T* argmaxAddr);

    __aicore__ inline int64_t PStart(int64_t index, int64_t pad, int64_t kernel, int64_t dilation, int64_t stride);
    __aicore__ inline int64_t PEnd(int64_t index, int64_t pad, int64_t stride, int64_t pooledSize);

    Pool3DGradNDHWCTilingData tilingData_;

    GlobalTensor<T> gradGm_;
    GlobalTensor<T> yGm_;

    TQue<QuePosition::VECIN, BUFFER_NUM> gradQue_;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outputQue_;
    TBuf<QuePosition::VECCALC> argmaxBuff_;
    TBuf<QuePosition::VECCALC> helpBuf_;

    int64_t blockIdx_ = 0;
    int64_t curCoreProcessNum_ = 0;
    event_t vToMte2Event_;
    event_t mte3ToMte2Event_;

    int64_t nAxisIndex_ = 0;
    int64_t cAxisIndex_ = 0;
    int64_t dAxisIndex_ = 0;
    int64_t hAxisIndex_ = 0;
    int64_t wAxisIndex_ = 0;

    int64_t nOutputActual_ = 0;
    int64_t cOutputActual_ = 0;
    int64_t dOutputActual_ = 0;
    int64_t hOutputActual_ = 0;
    int64_t wOutputActual_ = 0;

    int64_t cAxisGradOffset_ = 0;

    int64_t dArgmaxActualStart_ = 0;
    int64_t dArgmaxActualEnd_ = 0;
    int64_t hArgmaxActualStart_ = 0;
    int64_t hArgmaxActualEnd_ = 0;
    int64_t wArgmaxActualStart_ = 0;
    int64_t wArgmaxActualEnd_ = 0;
    int64_t dArgmaxActual_ = 0;
    int64_t hArgmaxActual_ = 0;
    int64_t wArgmaxActual_ = 0;

    int64_t cAligned_ = 0;
    int64_t cDim_ = 0;
    int64_t argmaxBufferSize_ = 0;

    bool isPad_ = false;
    int64_t vlT2_ = 0;

    int64_t curDProBatchSize_ = 0;
    int64_t curHProBatchSize_ = 0;
    int64_t curWProBatchSize_ = 0;

    constexpr static int32_t BLOCK_SIZE = platform::GetUbBlockSize();
    constexpr static int32_t V_REG_SIZE = platform::GetVRegSize();
};

template <typename T, typename INDEX_T, const uint32_t IS_CHECK_RANGE>
__aicore__ inline void MaxPool3DGradNDHWCBackwardBase<T, INDEX_T, IS_CHECK_RANGE>::ParseTilingData(
    const Pool3DGradNDHWCTilingData& tilingData)
{
    tilingData_ = tilingData;
    cDim_ = tilingData.cDim;
    argmaxBufferSize_ = tilingData.base.argmaxBufferSize;
    isPad_ = (tilingData.base.padD != 0 || tilingData.base.padH != 0 || tilingData.base.padW != 0 ||
              tilingData.base.padDBack != 0 || tilingData.base.padHBack != 0 || tilingData.base.padWBack != 0);
    vlT2_ = static_cast<int64_t>(V_REG_SIZE) / sizeof(INDEX_T);
}

template <typename T, typename INDEX_T, const uint32_t IS_CHECK_RANGE>
__aicore__ inline int64_t MaxPool3DGradNDHWCBackwardBase<T, INDEX_T, IS_CHECK_RANGE>::PStart(int64_t index, int64_t pad,
                                                                                             int64_t kernel,
                                                                                             int64_t dilation,
                                                                                             int64_t stride)
{
    if (stride == 0) {
        return 0;
    }
    return (index + pad < (kernel - 1) * dilation + 1) ? 0 : (index + pad - ((kernel - 1) * dilation + 1)) / stride + 1;
}

template <typename T, typename INDEX_T, const uint32_t IS_CHECK_RANGE>
__aicore__ inline int64_t MaxPool3DGradNDHWCBackwardBase<T, INDEX_T, IS_CHECK_RANGE>::PEnd(int64_t index, int64_t pad,
                                                                                           int64_t stride,
                                                                                           int64_t pooledSize)
{
    if (stride == 0) {
        return 0;
    }
    int64_t end = (index + pad) / stride + 1;
    return end < pooledSize ? end : pooledSize;
}

template <typename T, typename INDEX_T, const uint32_t IS_CHECK_RANGE>
__aicore__ inline void MaxPool3DGradNDHWCBackwardBase<T, INDEX_T, IS_CHECK_RANGE>::ScalarCompute(int64_t loopNum)
{
    int64_t baseBlockIdx = blockIdx_ * tilingData_.base.normalCoreProcessNum + loopNum;

    int64_t cOuter = tilingData_.cOutputOuter;
    int64_t dOuter = tilingData_.base.dOutputOuter;
    int64_t hOuter = tilingData_.base.hOutputOuter;
    int64_t wOuter = tilingData_.base.wOutputOuter;

    nAxisIndex_ = baseBlockIdx / (cOuter * dOuter * hOuter * wOuter);
    int64_t rem1 = baseBlockIdx % (cOuter * dOuter * hOuter * wOuter);
    cAxisIndex_ = rem1 / (dOuter * hOuter * wOuter);
    int64_t rem2 = rem1 % (dOuter * hOuter * wOuter);
    dAxisIndex_ = rem2 / (hOuter * wOuter);
    int64_t rem3 = rem2 % (hOuter * wOuter);
    hAxisIndex_ = rem3 / wOuter;
    wAxisIndex_ = rem3 % wOuter;

    nOutputActual_ = (nAxisIndex_ == tilingData_.base.highAxisOuter - 1) ? tilingData_.base.highAxisTail :
                                                                           tilingData_.base.highAxisInner;
    cOutputActual_ = (cAxisIndex_ == cOuter - 1) ? tilingData_.cOutputTail : tilingData_.cOutputInner;
    dOutputActual_ = (dAxisIndex_ == dOuter - 1) ? tilingData_.base.dOutputTail : tilingData_.base.dOutputInner;
    hOutputActual_ = (hAxisIndex_ == hOuter - 1) ? tilingData_.base.hOutputTail : tilingData_.base.hOutputInner;
    wOutputActual_ = (wAxisIndex_ == wOuter - 1) ? tilingData_.base.wOutputTail : tilingData_.base.wOutputInner;

    dArgmaxActualStart_ = PStart(dAxisIndex_ * tilingData_.base.dOutputInner, tilingData_.base.padD,
                                 tilingData_.base.dKernel, tilingData_.base.dilationD, tilingData_.base.dStride);
    dArgmaxActualEnd_ = PEnd(dAxisIndex_ * tilingData_.base.dOutputInner + dOutputActual_ - 1, tilingData_.base.padD,
                             tilingData_.base.dStride, tilingData_.base.dArgmax);
    hArgmaxActualStart_ = PStart(hAxisIndex_ * tilingData_.base.hOutputInner, tilingData_.base.padH,
                                 tilingData_.base.hKernel, tilingData_.base.dilationH, tilingData_.base.hStride);
    hArgmaxActualEnd_ = PEnd(hAxisIndex_ * tilingData_.base.hOutputInner + hOutputActual_ - 1, tilingData_.base.padH,
                             tilingData_.base.hStride, tilingData_.base.hArgmax);
    wArgmaxActualStart_ = PStart(wAxisIndex_ * tilingData_.base.wOutputInner, tilingData_.base.padW,
                                 tilingData_.base.wKernel, tilingData_.base.dilationW, tilingData_.base.wStride);
    wArgmaxActualEnd_ = PEnd(wAxisIndex_ * tilingData_.base.wOutputInner + wOutputActual_ - 1, tilingData_.base.padW,
                             tilingData_.base.wStride, tilingData_.base.wArgmax);

    dArgmaxActual_ = dArgmaxActualEnd_ - dArgmaxActualStart_;
    hArgmaxActual_ = hArgmaxActualEnd_ - hArgmaxActualStart_;
    wArgmaxActual_ = wArgmaxActualEnd_ - wArgmaxActualStart_;

    cAxisGradOffset_ = cAxisIndex_ * tilingData_.cOutputInner;
    int64_t blockElems = BLOCK_SIZE / sizeof(T);
    cAligned_ = (cOutputActual_ + blockElems - 1) / blockElems * blockElems;

    curDProBatchSize_ = tilingData_.base.dProBatchSize > dArgmaxActual_ ? dArgmaxActual_ :
                                                                          tilingData_.base.dProBatchSize;
    curHProBatchSize_ = tilingData_.base.hProBatchSize > hArgmaxActual_ ? hArgmaxActual_ :
                                                                          tilingData_.base.hProBatchSize;
    curWProBatchSize_ = tilingData_.base.wProBatchSize > wArgmaxActual_ ? wArgmaxActual_ :
                                                                          tilingData_.base.wProBatchSize;
}

// ==================== CopyInGrad ====================

template <typename T, typename INDEX_T, const uint32_t IS_CHECK_RANGE>
__aicore__ inline void MaxPool3DGradNDHWCBackwardBase<T, INDEX_T, IS_CHECK_RANGE>::CopyInGrad()
{
    LocalTensor<T> gradLocal = gradQue_.AllocTensor<T>();
    const int64_t C = cOutputActual_;
    const int64_t rowElt = wArgmaxActual_ * cAligned_;
    const int64_t planeElt = hArgmaxActual_ * rowElt;
    const int64_t batchElt = dArgmaxActual_ * planeElt;

    const int64_t gradStrideW = cDim_;
    const int64_t gradStrideH = tilingData_.base.wArgmax * gradStrideW;
    const int64_t gradStrideD = tilingData_.base.hArgmax * gradStrideH;
    const int64_t gradBase = nAxisIndex_ * tilingData_.base.highAxisInner * tilingData_.base.dArgmax * gradStrideD +
                             dArgmaxActualStart_ * gradStrideD + hArgmaxActualStart_ * gradStrideH +
                             wArgmaxActualStart_ * gradStrideW + cAxisGradOffset_;

    constexpr uint32_t NDDMA_DIMS = 5;
    constexpr uint32_t DIM_C = 0;
    constexpr uint32_t DIM_W = 1;
    constexpr uint32_t DIM_H = 2;
    constexpr uint32_t DIM_D = 3;
    constexpr uint32_t DIM_N = 4;
    MultiCopyLoopInfo<NDDMA_DIMS> loopInfo;
    loopInfo.loopSize[DIM_C] = C;
    loopInfo.loopSize[DIM_W] = wArgmaxActual_;
    loopInfo.loopSize[DIM_H] = hArgmaxActual_;
    loopInfo.loopSize[DIM_D] = dArgmaxActual_;
    loopInfo.loopSize[DIM_N] = nOutputActual_;
    loopInfo.loopSrcStride[DIM_C] = 1;
    loopInfo.loopSrcStride[DIM_W] = gradStrideW;
    loopInfo.loopSrcStride[DIM_H] = gradStrideH;
    loopInfo.loopSrcStride[DIM_D] = gradStrideD;
    loopInfo.loopSrcStride[DIM_N] = tilingData_.base.dArgmax * gradStrideD;
    loopInfo.loopDstStride[DIM_C] = 1;
    loopInfo.loopDstStride[DIM_W] = cAligned_;
    loopInfo.loopDstStride[DIM_H] = rowElt;
    loopInfo.loopDstStride[DIM_D] = planeElt;
    loopInfo.loopDstStride[DIM_N] = batchElt;
    static constexpr MultiCopyConfig config = {false};
    MultiCopyParams<T, NDDMA_DIMS> paramsMain = {loopInfo};
    DataCopy<T, NDDMA_DIMS, config>(gradLocal, gradGm_[gradBase], paramsMain);
    gradQue_.EnQue(gradLocal);
}

// ==================== CopyOut ====================

template <typename T, typename INDEX_T, const uint32_t IS_CHECK_RANGE>
__aicore__ inline void MaxPool3DGradNDHWCBackwardBase<T, INDEX_T, IS_CHECK_RANGE>::CopyOut()
{
    LocalTensor<T> yLocal = outputQue_.DeQue<T>();
    const int64_t C = cOutputActual_;
    const int64_t rowElt = wOutputActual_ * cAligned_;
    const int64_t planeElt = hOutputActual_ * rowElt;

    const int64_t yStrideW = cDim_;
    const int64_t yStrideH = tilingData_.base.wOutput * yStrideW;
    const int64_t yStrideD = tilingData_.base.hOutput * yStrideH;
    const int64_t yBase = nAxisIndex_ * tilingData_.base.highAxisInner * tilingData_.base.dOutput * yStrideD +
                          dAxisIndex_ * tilingData_.base.dOutputInner * yStrideD +
                          hAxisIndex_ * tilingData_.base.hOutputInner * yStrideH +
                          wAxisIndex_ * tilingData_.base.wOutputInner * yStrideW + cAxisGradOffset_;

    for (int64_t n = 0; n < nOutputActual_; n++) {
        int64_t gmOffset = yBase + n * tilingData_.base.dOutput * yStrideD;
        int64_t ubOffset = n * dOutputActual_ * planeElt;

        DataCopyExtParams copyParam;
        copyParam.blockCount = static_cast<uint16_t>(wOutputActual_);
        copyParam.blockLen = static_cast<uint32_t>(C * sizeof(T));
        copyParam.srcStride = 0;
        copyParam.dstStride = static_cast<uint32_t>((cDim_ - C) * sizeof(T));
        copyParam.rsv = 0;

        LoopModeParams loopParam;
        loopParam.loop1Size = static_cast<uint16_t>(hOutputActual_);
        loopParam.loop2Size = static_cast<uint16_t>(dOutputActual_);
        loopParam.loop1SrcStride = static_cast<uint32_t>(rowElt * sizeof(T));
        loopParam.loop2SrcStride = static_cast<uint32_t>(planeElt * sizeof(T));
        loopParam.loop1DstStride = static_cast<uint32_t>(yStrideH * sizeof(T));
        loopParam.loop2DstStride = static_cast<uint32_t>(yStrideD * sizeof(T));

        SetLoopModePara(loopParam, DataCopyMVType::UB_TO_OUT);
        DataCopyPad(yGm_[gmOffset], yLocal[static_cast<uint32_t>(ubOffset)], copyParam);
        ResetLoopModePara(DataCopyMVType::UB_TO_OUT);
    }
    outputQue_.FreeTensor(yLocal);
}

// ==================== ProcessNoArgmaxBlock ====================

template <typename T, typename INDEX_T, const uint32_t IS_CHECK_RANGE>
__aicore__ inline void MaxPool3DGradNDHWCBackwardBase<T, INDEX_T, IS_CHECK_RANGE>::ProcessNoArgmaxBlock()
{
    uint32_t calcCount = static_cast<uint32_t>(nOutputActual_ * dOutputActual_ * hOutputActual_ * wOutputActual_ *
                                               cAligned_);
    LocalTensor<T> yLocal = outputQue_.AllocTensor<T>();
    Duplicate(yLocal, T(0), calcCount);
    outputQue_.EnQue(yLocal);
    CopyOut();
}

// ==================== Backward ====================

template <typename T, typename INDEX_T, const uint32_t IS_CHECK_RANGE>
__aicore__ inline void MaxPool3DGradNDHWCBackwardBase<T, INDEX_T, IS_CHECK_RANGE>::Backward()
{
    uint32_t calcCount = static_cast<uint32_t>(nOutputActual_ * dOutputActual_ * hOutputActual_ * wOutputActual_ *
                                               cAligned_);
    LocalTensor<float> yLocal = outputQue_.AllocTensor<float>();
    Duplicate(yLocal, 0.f, calcCount);
    SetFlag<HardEvent::V_MTE2>(vToMte2Event_);
    WaitFlag<HardEvent::V_MTE2>(vToMte2Event_);
    LocalTensor<T> gradLocal = gradQue_.DeQue<T>();
    LocalTensor<INDEX_T> argmaxLocal = argmaxBuff_.Get<INDEX_T>();

    BackwardCompute((__ubuf__ float*)yLocal.GetPhyAddr(), (__ubuf__ T*)gradLocal.GetPhyAddr(),
                    (__ubuf__ INDEX_T*)argmaxLocal.GetPhyAddr());
    SetFlag<HardEvent::V_MTE2>(vToMte2Event_);
    WaitFlag<HardEvent::V_MTE2>(vToMte2Event_);

    if constexpr (std::negation<std::is_same<T, float>>::value) {
        Cast(yLocal.ReinterpretCast<T>(), yLocal, RoundMode::CAST_RINT, calcCount);
    }
    outputQue_.EnQue(yLocal);
    gradQue_.FreeTensor(gradLocal);
}

// ==================== VDrainToMte2: 前向 V loads → 反向 grad MTE2 写排序 ====================
template <typename T, typename INDEX_T, const uint32_t IS_CHECK_RANGE>
__aicore__ inline void MaxPool3DGradNDHWCBackwardBase<T, INDEX_T, IS_CHECK_RANGE>::VDrainToMte2()
{
    SetFlag<HardEvent::V_MTE2>(vToMte2Event_);
    WaitFlag<HardEvent::V_MTE2>(vToMte2Event_);
}

// ==================== Mte3Drain ====================
template <typename T, typename INDEX_T, const uint32_t IS_CHECK_RANGE>
__aicore__ inline void MaxPool3DGradNDHWCBackwardBase<T, INDEX_T, IS_CHECK_RANGE>::Mte3Drain()
{
    SetFlag<HardEvent::MTE3_MTE2>(mte3ToMte2Event_);
    WaitFlag<HardEvent::MTE3_MTE2>(mte3ToMte2Event_);
}

// ==================== 模式生成（张量级） ====================
constexpr int32_t PATTERN_BASE_WORD = 64;
constexpr int32_t PATTERN_STRIDE_WORD = 64;

constexpr int32_t SLOT_PATTERN_ONE = 2;       // 3D (H,W batch, C) -> initialReg{Index,Argmax}One
constexpr int32_t SLOT_PATTERN_TAIL = 4;      // 2D (H batch, C) -> initialReg{Index,Argmax}Tail
constexpr int32_t SLOT_PATTERN_W_TAIL = 6;    // 3D (D,H batch, C) -> initialReg{Index,Argmax}WTail
constexpr int32_t SLOT_PATTERN_DH_TAIL = 8;   // 2D (DH batch, C) -> initialReg{Index,Argmax}DHTail
constexpr int32_t SLOT_PATTERN_W_BATCH = 10;  // 2D (W batch, C) -> initialReg{Index,Argmax}WBatch
constexpr int32_t SLOT_PATTERN_H_TAIL = 12;   // 3D (D,W batch, C) -> initialReg{Index,Argmax}HTail
constexpr int32_t SLOT_PATTERN_ONE_TAIL = 14; // 1D arange -> initialRegIndexOneTail

template <typename T, typename INDEX_T, const uint32_t IS_CHECK_RANGE>
__aicore__ inline void MaxPool3DGradNDHWCBackwardBase<T, INDEX_T, IS_CHECK_RANGE>::GenPattern2D(
    AscendC::Reg::RegTensor<int32_t>& indexReg, int32_t groupStride, int32_t cCnt, int32_t slot)
{
    // lane = g*cCnt + c → offset = g*groupStride + c
    __ubuf__ int32_t* dstAddr = (__ubuf__ int32_t*)helpBuf_.template Get<int32_t>().GetPhyAddr() + PATTERN_BASE_WORD +
                                slot * PATTERN_STRIDE_WORD;
    AscendC::Reg::RegTensor<int32_t> laneReg;
    AscendC::Reg::RegTensor<int32_t> gReg;
    AscendC::Reg::RegTensor<int32_t> cReg;
    AscendC::Reg::RegTensor<int32_t> cCntReg;
    AscendC::Reg::MaskReg p0 = AscendC::Reg::CreateMask<int32_t, AscendC::Reg::MaskPattern::ALL>();
    AscendC::Reg::Arange(laneReg, 0);
    AscendC::Reg::Duplicate(cCntReg, cCnt, p0);
    AscendC::Reg::Div(gReg, laneReg, cCntReg, p0);
    AscendC::Reg::Mul(cReg, gReg, cCntReg, p0);
    AscendC::Reg::Sub(cReg, laneReg, cReg, p0);
    AscendC::Reg::Muls(gReg, gReg, groupStride, p0);
    AscendC::Reg::Add(gReg, gReg, cReg, p0);
    AscendC::Reg::StoreAlign(dstAddr, gReg, p0);
    AscendC::Reg::LocalMemBar<AscendC::Reg::MemType::VEC_STORE, AscendC::Reg::MemType::VEC_LOAD>();
    AscendC::Reg::LoadAlign(indexReg, dstAddr);
}

template <typename T, typename INDEX_T, const uint32_t IS_CHECK_RANGE>
__aicore__ inline void MaxPool3DGradNDHWCBackwardBase<T, INDEX_T, IS_CHECK_RANGE>::GenPattern3D(
    AscendC::Reg::RegTensor<int32_t>& indexReg, int32_t hiGroupStride, int32_t loGroupStride, int32_t loGroupCount,
    int32_t cCnt, int32_t slot)
{
    // lane = hi*(loCnt*cCnt) + lo*cCnt + c → offset = hi*hiStride + lo*loStride + c
    __ubuf__ int32_t* dstAddr = (__ubuf__ int32_t*)helpBuf_.template Get<int32_t>().GetPhyAddr() + PATTERN_BASE_WORD +
                                slot * PATTERN_STRIDE_WORD;
    AscendC::Reg::RegTensor<int32_t> laneReg;
    AscendC::Reg::RegTensor<int32_t> hiReg;
    AscendC::Reg::RegTensor<int32_t> loReg;
    AscendC::Reg::RegTensor<int32_t> remReg;
    AscendC::Reg::RegTensor<int32_t> cReg;
    AscendC::Reg::RegTensor<int32_t> cCntReg;
    AscendC::Reg::RegTensor<int32_t> loCntReg;
    AscendC::Reg::MaskReg p0 = AscendC::Reg::CreateMask<int32_t, AscendC::Reg::MaskPattern::ALL>();
    AscendC::Reg::Arange(laneReg, 0);
    AscendC::Reg::Duplicate(cCntReg, cCnt, p0);
    AscendC::Reg::Duplicate(loCntReg, loGroupCount, p0);
    AscendC::Reg::Mul(remReg, loCntReg, cCntReg, p0);
    AscendC::Reg::Div(hiReg, laneReg, remReg, p0);
    AscendC::Reg::Mul(remReg, hiReg, remReg, p0);
    AscendC::Reg::Sub(remReg, laneReg, remReg, p0);
    AscendC::Reg::Div(loReg, remReg, cCntReg, p0);
    AscendC::Reg::Mul(cReg, loReg, cCntReg, p0);
    AscendC::Reg::Sub(cReg, remReg, cReg, p0);
    AscendC::Reg::Muls(hiReg, hiReg, hiGroupStride, p0);
    AscendC::Reg::Muls(loReg, loReg, loGroupStride, p0);
    AscendC::Reg::Add(hiReg, hiReg, loReg, p0);
    AscendC::Reg::Add(hiReg, hiReg, cReg, p0);
    AscendC::Reg::StoreAlign(dstAddr, hiReg, p0);
    AscendC::Reg::LocalMemBar<AscendC::Reg::MemType::VEC_STORE, AscendC::Reg::MemType::VEC_LOAD>();
    AscendC::Reg::LoadAlign(indexReg, dstAddr);
}

template <typename T, typename INDEX_T, const uint32_t IS_CHECK_RANGE>
__aicore__ inline void MaxPool3DGradNDHWCBackwardBase<T, INDEX_T, IS_CHECK_RANGE>::GenPattern4D(
    AscendC::Reg::RegTensor<int32_t>& indexReg, int32_t dGroupStride, int32_t hGroupStride, int32_t wGroupStride,
    int32_t wGroupCount, int32_t hGroupCount, int32_t cCnt, int32_t slot)
{
    // lane = d*(hC*wC*C) + h*(wC*C) + w*C + c → offset = d*dS + h*hS + w*wS + c
    __ubuf__ int32_t* dstAddr = (__ubuf__ int32_t*)helpBuf_.template Get<int32_t>().GetPhyAddr() + PATTERN_BASE_WORD +
                                slot * PATTERN_STRIDE_WORD;
    AscendC::Reg::RegTensor<int32_t> laneReg;
    AscendC::Reg::RegTensor<int32_t> dReg;
    AscendC::Reg::RegTensor<int32_t> hReg;
    AscendC::Reg::RegTensor<int32_t> wReg;
    AscendC::Reg::RegTensor<int32_t> remReg;
    AscendC::Reg::RegTensor<int32_t> rem2Reg;
    AscendC::Reg::RegTensor<int32_t> cReg;
    AscendC::Reg::RegTensor<int32_t> tmpReg;
    AscendC::Reg::RegTensor<int32_t> cCntReg;
    AscendC::Reg::RegTensor<int32_t> wCntReg;
    AscendC::Reg::RegTensor<int32_t> hCntReg;
    AscendC::Reg::RegTensor<int32_t> wcReg;
    AscendC::Reg::RegTensor<int32_t> hwcReg;
    AscendC::Reg::MaskReg p0 = AscendC::Reg::CreateMask<int32_t, AscendC::Reg::MaskPattern::ALL>();
    AscendC::Reg::Arange(laneReg, 0);
    AscendC::Reg::Duplicate(cCntReg, cCnt, p0);
    AscendC::Reg::Duplicate(wCntReg, wGroupCount, p0);
    AscendC::Reg::Duplicate(hCntReg, hGroupCount, p0);
    AscendC::Reg::Mul(wcReg, wCntReg, cCntReg, p0);
    AscendC::Reg::Mul(hwcReg, wcReg, hCntReg, p0);
    AscendC::Reg::Div(dReg, laneReg, hwcReg, p0);
    AscendC::Reg::Mul(tmpReg, dReg, hwcReg, p0);
    AscendC::Reg::Sub(remReg, laneReg, tmpReg, p0);
    AscendC::Reg::Div(hReg, remReg, wcReg, p0);
    AscendC::Reg::Mul(tmpReg, hReg, wcReg, p0);
    AscendC::Reg::Sub(rem2Reg, remReg, tmpReg, p0);
    AscendC::Reg::Div(wReg, rem2Reg, cCntReg, p0);
    AscendC::Reg::Mul(tmpReg, wReg, cCntReg, p0);
    AscendC::Reg::Sub(cReg, rem2Reg, tmpReg, p0);
    AscendC::Reg::Muls(dReg, dReg, dGroupStride, p0);
    AscendC::Reg::Muls(hReg, hReg, hGroupStride, p0);
    AscendC::Reg::Muls(wReg, wReg, wGroupStride, p0);
    AscendC::Reg::Add(dReg, dReg, hReg, p0);
    AscendC::Reg::Add(dReg, dReg, wReg, p0);
    AscendC::Reg::Add(dReg, dReg, cReg, p0);
    AscendC::Reg::StoreAlign(dstAddr, dReg, p0);
    AscendC::Reg::LocalMemBar<AscendC::Reg::MemType::VEC_STORE, AscendC::Reg::MemType::VEC_LOAD>();
    AscendC::Reg::LoadAlign(indexReg, dstAddr);
}

template <typename T, typename INDEX_T, const uint32_t IS_CHECK_RANGE>
__aicore__ inline __ubuf__ int32_t* MaxPool3DGradNDHWCBackwardBase<T, INDEX_T, IS_CHECK_RANGE>::PatternSlotAddr(
    int32_t slot)
{
    return (__ubuf__ int32_t*)helpBuf_.template Get<int32_t>().GetPhyAddr() + PATTERN_BASE_WORD +
           slot * PATTERN_STRIDE_WORD;
}

// ==================== Scatter 策略 ====================

template <typename T, typename INDEX_T, const uint32_t IS_CHECK_RANGE>
__aicore__ inline void MaxPool3DGradNDHWCBackwardBase<T, INDEX_T, IS_CHECK_RANGE>::SingleLineProcessVF(
    __ubuf__ float* yAddr, __ubuf__ T* gradAddr, __ubuf__ INDEX_T* argmaxAddr)
{
    constexpr int32_t V_REG_SIZE = platform::GetVRegSize();

    int32_t wInput = static_cast<int32_t>(tilingData_.base.wOutput);
    int32_t hInput = static_cast<int32_t>(tilingData_.base.hOutput);
    int32_t hwInput = hInput * wInput;
    int32_t wOutput = static_cast<int32_t>(tilingData_.base.wArgmax);
    int32_t hOutput = static_cast<int32_t>(tilingData_.base.hArgmax);
    int32_t hwOutput = hOutput * wOutput;
    int32_t wOutputActual = static_cast<int32_t>(wOutputActual_);
    int32_t hOutputActual = static_cast<int32_t>(hOutputActual_);
    int32_t dOutputActual = static_cast<int32_t>(dOutputActual_);
    int32_t dhOutputActual = hOutputActual * wOutputActual;
    int32_t curDIndex = static_cast<int32_t>(dAxisIndex_ * tilingData_.base.dOutputInner);
    int32_t curHIndex = static_cast<int32_t>(hAxisIndex_ * tilingData_.base.hOutputInner);
    int32_t curWIndex = static_cast<int32_t>(wAxisIndex_ * tilingData_.base.wOutputInner);
    int32_t cOutputAligned = static_cast<int32_t>(cAligned_);
    int32_t cStride = cOutputAligned;
    int32_t baseOffsetConst = -(curDIndex * dhOutputActual + curHIndex * wOutputActual + curWIndex) * cStride;

    uint16_t nOutputActual = static_cast<uint16_t>(nOutputActual_);
    uint16_t dArgmaxActual = static_cast<uint16_t>(dArgmaxActual_);
    uint16_t hArgmaxActual = static_cast<uint16_t>(hArgmaxActual_);
    uint16_t wArgmaxActual = static_cast<uint16_t>(wArgmaxActual_);
    int32_t cOutputActual = static_cast<int32_t>(cOutputActual_);
    int32_t cActual = cOutputActual;

    uint16_t computeSizeFP32 = V_REG_SIZE / sizeof(float);
    uint16_t cRepeatimes = static_cast<uint16_t>(cOutputActual / computeSizeFP32);
    uint16_t cRemain = static_cast<uint16_t>(cOutputActual - cRepeatimes * computeSizeFP32);

    uint32_t magicHW = 0;
    uint32_t shiftHW = 0;
    uint32_t magicW = 0;
    uint32_t shiftW = 0;
    GetUintDivMagicAndShift<uint32_t>(magicHW, shiftHW, static_cast<uint32_t>(hwInput));
    GetUintDivMagicAndShift<uint32_t>(magicW, shiftW, static_cast<uint32_t>(wInput));

    __VEC_SCOPE__
    {
        AscendC::Reg::RegTensor<int32_t> dLowerReg;
        AscendC::Reg::RegTensor<int32_t> wLowerReg;
        AscendC::Reg::RegTensor<int32_t> hLowerReg;
        AscendC::Reg::RegTensor<int32_t> dUpperReg;
        AscendC::Reg::RegTensor<int32_t> hUpperReg;
        AscendC::Reg::RegTensor<int32_t> wUpperReg;
        if constexpr (IS_CHECK_RANGE == 1) {
            AscendC::Reg::Duplicate(dLowerReg, int32_t(curDIndex));
            AscendC::Reg::Duplicate(hLowerReg, int32_t(curHIndex));
            AscendC::Reg::Duplicate(wLowerReg, int32_t(curWIndex));
            AscendC::Reg::Duplicate(dUpperReg, int32_t(dOutputActual + curDIndex));
            AscendC::Reg::Duplicate(hUpperReg, int32_t(hOutputActual + curHIndex));
            AscendC::Reg::Duplicate(wUpperReg, int32_t(wOutputActual + curWIndex));
        }
        AscendC::Reg::RegTensor<uint32_t> magicHWReg;
        AscendC::Reg::RegTensor<uint32_t> magicWReg;
        AscendC::Reg::Duplicate(magicHWReg, magicHW);
        AscendC::Reg::Duplicate(magicWReg, magicW);

        AscendC::Reg::RegTensor<int32_t> parallelRegIndex;
        AscendC::Reg::RegTensor<int32_t> parallelRegArgmax;
        AscendC::Reg::RegTensor<int32_t> initialRegIndex;
        AscendC::Reg::Arange(initialRegIndex, 0);
        AscendC::Reg::MaskReg allMaskU32 = AscendC::Reg::CreateMask<uint32_t, AscendC::Reg::MaskPattern::ALL>();
        AscendC::Reg::RegTensor<int32_t> cIncReg;

        for (uint16_t nIdx = 0; nIdx < nOutputActual; ++nIdx) {
            uint32_t nGradOffset = nIdx * dArgmaxActual * hArgmaxActual * wArgmaxActual * cStride;
            uint32_t nArgmaxOffset = nIdx * dArgmaxActual * hArgmaxActual * wArgmaxActual * cActual;
            int32_t baseOffset = int32_t(nIdx * dOutputActual * dhOutputActual * cStride) + baseOffsetConst;
            for (uint16_t dIdx = 0; dIdx < dArgmaxActual; ++dIdx) {
                uint32_t dGradBase = dIdx * hArgmaxActual * wArgmaxActual * cStride + nGradOffset;
                uint32_t dArgmaxBase = dIdx * hArgmaxActual * wArgmaxActual * cActual + nArgmaxOffset;
                for (uint16_t hIdx = 0; hIdx < hArgmaxActual; ++hIdx) {
                    uint32_t dhGradBase = hIdx * wArgmaxActual * cStride + dGradBase;
                    uint32_t dhArgmaxBase = hIdx * wArgmaxActual * cActual + dArgmaxBase;
                    for (uint16_t wIdx = 0; wIdx < wArgmaxActual; ++wIdx) {
                        uint32_t dhwGradBase = wIdx * cStride + dhGradBase;
                        uint32_t dhwArgmaxBase = wIdx * cActual + dhArgmaxBase;
                        for (uint16_t cRepeatIdx = 0; cRepeatIdx < cRepeatimes; ++cRepeatIdx) {
                            uint32_t cOffset = cRepeatIdx * computeSizeFP32;
                            uint32_t gradOffset = dhwGradBase + cOffset;
                            uint32_t argmaxOffset = dhwArgmaxBase + cOffset;
                            AscendC::Reg::Adds(parallelRegIndex, initialRegIndex, static_cast<int32_t>(gradOffset),
                                               allMaskU32);
                            AscendC::Reg::Adds(parallelRegArgmax, initialRegIndex, static_cast<int32_t>(argmaxOffset),
                                               allMaskU32);
                            AscendC::Reg::Arange(cIncReg, static_cast<int32_t>(cOffset));
                            DoSingleNCNdhwcFastDiv<T, INDEX_T, IS_CHECK_RANGE>(
                                (__local_mem__ computeType*)yAddr, (__local_mem__ T*)gradAddr,
                                (__local_mem__ INDEX_T*)argmaxAddr, parallelRegArgmax, parallelRegIndex,
                                computeSizeFP32, magicHWReg, static_cast<int16_t>(shiftHW), magicWReg,
                                static_cast<int16_t>(shiftW), dhOutputActual, wOutputActual, wOutput, hwOutput, wInput,
                                hwInput, baseOffset, cOutputAligned, cIncReg, dLowerReg, hLowerReg, wLowerReg,
                                dUpperReg, hUpperReg, wUpperReg);
                        }
                        if (cRemain > 0) {
                            uint32_t cOffset = cRepeatimes * computeSizeFP32;
                            uint32_t gradOffset = dhwGradBase + cOffset;
                            uint32_t argmaxOffset = dhwArgmaxBase + cOffset;
                            AscendC::Reg::Adds(parallelRegIndex, initialRegIndex, static_cast<int32_t>(gradOffset),
                                               allMaskU32);
                            AscendC::Reg::Adds(parallelRegArgmax, initialRegIndex, static_cast<int32_t>(argmaxOffset),
                                               allMaskU32);
                            AscendC::Reg::Arange(cIncReg, static_cast<int32_t>(cOffset));
                            DoSingleNCNdhwcFastDiv<T, INDEX_T, IS_CHECK_RANGE>(
                                (__local_mem__ computeType*)yAddr, (__local_mem__ T*)gradAddr,
                                (__local_mem__ INDEX_T*)argmaxAddr, parallelRegArgmax, parallelRegIndex, cRemain,
                                magicHWReg, static_cast<int16_t>(shiftHW), magicWReg, static_cast<int16_t>(shiftW),
                                dhOutputActual, wOutputActual, wOutput, hwOutput, wInput, hwInput, baseOffset,
                                cOutputAligned, cIncReg, dLowerReg, hLowerReg, wLowerReg, dUpperReg, hUpperReg,
                                wUpperReg);
                        }
                    }
                }
            }
        }
        AscendC::Reg::LocalMemBar<AscendC::Reg::MemType::VEC_STORE, AscendC::Reg::MemType::VEC_LOAD>();
    }
}

// ==================== BackwardCompute ====================

template <typename T, typename INDEX_T, const uint32_t IS_CHECK_RANGE>
__aicore__ inline void MaxPool3DGradNDHWCBackwardBase<T, INDEX_T, IS_CHECK_RANGE>::BackwardCompute(
    __ubuf__ float* yAddr, __ubuf__ T* gradAddr, __ubuf__ INDEX_T* argmaxAddr)
{
    uint32_t wConcurrentCount = curWProBatchSize_ > 0 ? static_cast<uint32_t>(wArgmaxActual_ / curWProBatchSize_) : 0;
    uint32_t hConcurrentCount = curHProBatchSize_ > 0 ? static_cast<uint32_t>(hArgmaxActual_ / curHProBatchSize_) : 0;
    uint32_t dConcurrentCount = curDProBatchSize_ > 0 ? static_cast<uint32_t>(dArgmaxActual_ / curDProBatchSize_) : 0;

    const uint32_t computeSizeFP32 = static_cast<uint32_t>(V_REG_SIZE) / sizeof(float);
    const uint32_t cActualU32 = static_cast<uint32_t>(cOutputActual_);
    const bool fitWC = cActualU32 > 0 && cActualU32 <= computeSizeFP32;
    const bool fitHWC = fitWC && wConcurrentCount * cActualU32 <= computeSizeFP32;
    const bool fitDHWC = fitHWC && hConcurrentCount * wConcurrentCount * cActualU32 <= computeSizeFP32;

    const uint32_t concW = computeSizeFP32 / cActualU32;
    const bool fillW = fitWC && wConcurrentCount >= concW && wConcurrentCount > 0;
    const uint32_t concH = fillW ? concW / wConcurrentCount : 0;
    const bool fillH = fillW && hConcurrentCount >= concH && hConcurrentCount > 0;
    const uint32_t concD = fillH ? concH / hConcurrentCount : 0;
    const bool fillD = fillH && dConcurrentCount >= concD;

    if (concW <= 1 || wConcurrentCount * DOUBLE_HEAD * sizeof(INDEX_T) > V_REG_SIZE || !fitWC || !fillW) {
        SingleLineProcessVF(yAddr, gradAddr, argmaxAddr);
    } else if (wConcurrentCount * hConcurrentCount * DOUBLE_HEAD * sizeof(INDEX_T) > V_REG_SIZE || !fitHWC || !fillH) {
        MultipleLineHwProcessVF(yAddr, gradAddr, argmaxAddr);
    } else if (wConcurrentCount * hConcurrentCount * dConcurrentCount * DOUBLE_HEAD * sizeof(INDEX_T) > V_REG_SIZE ||
               !fitDHWC || !fillD) {
        MultipleLineDhwProcessVF(yAddr, gradAddr, argmaxAddr);
    } else {
        MultipleLineProcessVF2(yAddr, gradAddr, argmaxAddr);
    }
}

template <typename T, typename INDEX_T, const uint32_t IS_CHECK_RANGE>
__aicore__ inline void MaxPool3DGradNDHWCBackwardBase<T, INDEX_T, IS_CHECK_RANGE>::MultipleLineHwProcessVF(
    __ubuf__ float* yAddr, __ubuf__ T* gradAddr, __ubuf__ INDEX_T* argmaxAddr)
{
    constexpr int32_t V_REG_SIZE = platform::GetVRegSize();

    int32_t wInput = static_cast<int32_t>(tilingData_.base.wOutput);
    int32_t hInput = static_cast<int32_t>(tilingData_.base.hOutput);
    int32_t hwInput = hInput * wInput;
    int32_t wOutput = static_cast<int32_t>(tilingData_.base.wArgmax);
    int32_t hOutput = static_cast<int32_t>(tilingData_.base.hArgmax);
    int32_t hwOutput = hOutput * wOutput;
    int32_t wOutputActual = static_cast<int32_t>(wOutputActual_);
    int32_t hOutputActual = static_cast<int32_t>(hOutputActual_);
    int32_t dOutputActual = static_cast<int32_t>(dOutputActual_);
    int32_t dhOutputActual = hOutputActual * wOutputActual;
    int32_t curDIndex = static_cast<int32_t>(dAxisIndex_ * tilingData_.base.dOutputInner);
    int32_t curHIndex = static_cast<int32_t>(hAxisIndex_ * tilingData_.base.hOutputInner);
    int32_t curWIndex = static_cast<int32_t>(wAxisIndex_ * tilingData_.base.wOutputInner);
    int32_t cOutputAligned = static_cast<int32_t>(cAligned_);
    int32_t cOutputActual = static_cast<int32_t>(cOutputActual_);
    int32_t cStride = cOutputAligned;
    int32_t cActual = cOutputActual;
    int32_t baseOffsetConst = -(curDIndex * dhOutputActual + curHIndex * wOutputActual + curWIndex) * cStride;

    uint16_t nOutputActual = static_cast<uint16_t>(nOutputActual_);
    uint16_t dArgmaxActual = static_cast<uint16_t>(dArgmaxActual_);
    uint16_t hArgmaxActual = static_cast<uint16_t>(hArgmaxActual_);
    int32_t wArgmaxActual = static_cast<int32_t>(wArgmaxActual_);
    uint16_t wProBatchSize = static_cast<uint16_t>(tilingData_.base.wProBatchSize);
    wProBatchSize = (wProBatchSize > wArgmaxActual) ? static_cast<uint16_t>(wArgmaxActual) : wProBatchSize;

    uint32_t wFullBatchCount = static_cast<uint32_t>(wArgmaxActual) / wProBatchSize;
    uint16_t computeSizeFP32 = V_REG_SIZE / sizeof(float);
    uint16_t concurrencyCount = computeSizeFP32 / static_cast<uint16_t>(cOutputActual);
    uint16_t repeatimes = static_cast<uint16_t>(wFullBatchCount / concurrencyCount);
    uint32_t wRemain = static_cast<uint32_t>(wArgmaxActual) - repeatimes * wProBatchSize * concurrencyCount;
    uint32_t wRemainBatch = wRemain / wProBatchSize;
    uint16_t wRemainTail = static_cast<uint16_t>(wRemain - wRemainBatch * wProBatchSize);
    uint32_t mask0 = concurrencyCount * cOutputActual;
    uint32_t mask1 = wRemainBatch * cOutputActual;
    uint32_t mask2 = cOutputActual;

    uint32_t magicHW = 0;
    uint32_t shiftHW = 0;
    uint32_t magicW = 0;
    uint32_t shiftW = 0;
    GetUintDivMagicAndShift<uint32_t>(magicHW, shiftHW, static_cast<uint32_t>(hwInput));
    GetUintDivMagicAndShift<uint32_t>(magicW, shiftW, static_cast<uint32_t>(wInput));
    DivMagic divC = PrecomputeDiv(static_cast<uint32_t>(cOutputActual));
    int32_t wBatchStride = wProBatchSize * cStride;
    int32_t wBatchStrideA = wProBatchSize * cActual;

    for (uint16_t nIdx = 0; nIdx < nOutputActual; ++nIdx) {
        uint32_t nGradOffset = nIdx * dArgmaxActual * hArgmaxActual * wArgmaxActual * cStride;
        uint32_t nArgmaxOffset = nIdx * dArgmaxActual * hArgmaxActual * wArgmaxActual * cActual;
        int32_t baseOffset = int32_t(nIdx * dOutputActual * dhOutputActual * cStride) + baseOffsetConst;
        for (uint16_t dIdx = 0; dIdx < dArgmaxActual; ++dIdx) {
            uint32_t dGradBase = dIdx * hArgmaxActual * wArgmaxActual * cStride + nGradOffset;
            uint32_t dArgmaxBase = dIdx * hArgmaxActual * wArgmaxActual * cActual + nArgmaxOffset;
            for (uint16_t hIdx = 0; hIdx < hArgmaxActual; ++hIdx) {
                uint32_t dhGradBase = hIdx * wArgmaxActual * cStride + dGradBase;
                uint32_t dhArgmaxBase = hIdx * wArgmaxActual * cActual + dArgmaxBase;
                __VEC_SCOPE__
                {
                    AscendC::Reg::RegTensor<int32_t> dLowerReg;
                    AscendC::Reg::RegTensor<int32_t> wLowerReg;
                    AscendC::Reg::RegTensor<int32_t> hLowerReg;
                    AscendC::Reg::RegTensor<int32_t> dUpperReg;
                    AscendC::Reg::RegTensor<int32_t> hUpperReg;
                    AscendC::Reg::RegTensor<int32_t> wUpperReg;
                    if constexpr (IS_CHECK_RANGE == 1) {
                        AscendC::Reg::Duplicate(dLowerReg, int32_t(curDIndex));
                        AscendC::Reg::Duplicate(hLowerReg, int32_t(curHIndex));
                        AscendC::Reg::Duplicate(wLowerReg, int32_t(curWIndex));
                        AscendC::Reg::Duplicate(dUpperReg, int32_t(dOutputActual + curDIndex));
                        AscendC::Reg::Duplicate(hUpperReg, int32_t(hOutputActual + curHIndex));
                        AscendC::Reg::Duplicate(wUpperReg, int32_t(wOutputActual + curWIndex));
                    }
                    AscendC::Reg::RegTensor<uint32_t> magicHWReg;
                    AscendC::Reg::RegTensor<uint32_t> magicWReg;
                    AscendC::Reg::Duplicate(magicHWReg, magicHW);
                    AscendC::Reg::Duplicate(magicWReg, magicW);
                    AscendC::Reg::RegTensor<int32_t> initialRegIndex;
                    GenPattern2D(initialRegIndex, wBatchStride, cOutputActual, 0);
                    AscendC::Reg::RegTensor<int32_t> initialRegArgmax;
                    GenPattern2D(initialRegArgmax, wBatchStrideA, cOutputActual, 1);
                    AscendC::Reg::RegTensor<int32_t> initialRegIndexOne;
                    PoolGradCommon::Gen2DIndexOne((AscendC::Reg::RegTensor<int32_t>&)initialRegIndexOne, 1, 1);
                    AscendC::Reg::RegTensor<int32_t> parallelRegIndex;
                    AscendC::Reg::RegTensor<int32_t> parallelRegArgmax;
                    AscendC::Reg::RegTensor<int32_t> cIncReg;
                    AscendC::Reg::Arange(cIncReg, 0);
                    AscendC::Reg::MaskReg
                        allMaskU32 = AscendC::Reg::CreateMask<uint32_t, AscendC::Reg::MaskPattern::ALL>();

                    for (uint16_t wRepeatIdx = 0; wRepeatIdx < repeatimes; ++wRepeatIdx) {
                        for (uint16_t wBatchIdx = 0; wBatchIdx < wProBatchSize; ++wBatchIdx) {
                            uint32_t offset = (wBatchIdx + wRepeatIdx * concurrencyCount * wProBatchSize) * cStride +
                                              dhGradBase;
                            uint32_t argmaxOff = (wBatchIdx + wRepeatIdx * concurrencyCount * wProBatchSize) * cActual +
                                                 dhArgmaxBase;
                            AscendC::Reg::Adds(parallelRegIndex, initialRegIndex, static_cast<int32_t>(offset),
                                               allMaskU32);
                            AscendC::Reg::Adds(parallelRegArgmax, initialRegArgmax, static_cast<int32_t>(argmaxOff),
                                               allMaskU32);
                            DoMulNCNdhwcFastDiv<T, INDEX_T, IS_CHECK_RANGE>(
                                (__local_mem__ computeType*)yAddr, (__local_mem__ T*)gradAddr,
                                (__local_mem__ INDEX_T*)argmaxAddr, parallelRegArgmax, parallelRegIndex, mask0,
                                magicHWReg, static_cast<int16_t>(shiftHW), magicWReg, static_cast<int16_t>(shiftW),
                                dhOutputActual, wOutputActual, wOutput, hwOutput, wInput, hwInput, baseOffset,
                                cOutputAligned, cOutputActual, divC, cIncReg, dLowerReg, hLowerReg, wLowerReg,
                                dUpperReg, hUpperReg, wUpperReg);
                        }
                    }
                    for (uint16_t wBatchIdx = 0; wBatchIdx < wProBatchSize; ++wBatchIdx) {
                        uint32_t offset = (wBatchIdx + repeatimes * concurrencyCount * wProBatchSize) * cStride +
                                          dhGradBase;
                        uint32_t argmaxOff = (wBatchIdx + repeatimes * concurrencyCount * wProBatchSize) * cActual +
                                             dhArgmaxBase;
                        AscendC::Reg::Adds(parallelRegIndex, initialRegIndex, static_cast<int32_t>(offset), allMaskU32);
                        AscendC::Reg::Adds(parallelRegArgmax, initialRegArgmax, static_cast<int32_t>(argmaxOff),
                                           allMaskU32);
                        DoMulNCNdhwcFastDiv<T, INDEX_T, IS_CHECK_RANGE>(
                            (__local_mem__ computeType*)yAddr, (__local_mem__ T*)gradAddr,
                            (__local_mem__ INDEX_T*)argmaxAddr, parallelRegArgmax, parallelRegIndex, mask1, magicHWReg,
                            static_cast<int16_t>(shiftHW), magicWReg, static_cast<int16_t>(shiftW), dhOutputActual,
                            wOutputActual, wOutput, hwOutput, wInput, hwInput, baseOffset, cOutputAligned,
                            cOutputActual, divC, cIncReg, dLowerReg, hLowerReg, wLowerReg, dUpperReg, hUpperReg,
                            wUpperReg);
                    }
                    for (uint16_t wBatchIdx = 0; wBatchIdx < wRemainTail; ++wBatchIdx) {
                        uint32_t offset = (wBatchIdx + wRemainBatch * wProBatchSize +
                                           repeatimes * concurrencyCount * wProBatchSize) *
                                              cStride +
                                          dhGradBase;
                        uint32_t argmaxOff = (wBatchIdx + wRemainBatch * wProBatchSize +
                                              repeatimes * concurrencyCount * wProBatchSize) *
                                                 cActual +
                                             dhArgmaxBase;
                        AscendC::Reg::Adds(parallelRegIndex, initialRegIndexOne, static_cast<int32_t>(offset),
                                           allMaskU32);
                        AscendC::Reg::Adds(parallelRegArgmax, initialRegIndexOne, static_cast<int32_t>(argmaxOff),
                                           allMaskU32);
                        DoSingleNCNdhwcFastDiv<T, INDEX_T, IS_CHECK_RANGE>(
                            (__local_mem__ computeType*)yAddr, (__local_mem__ T*)gradAddr,
                            (__local_mem__ INDEX_T*)argmaxAddr, parallelRegArgmax, parallelRegIndex, mask2, magicHWReg,
                            static_cast<int16_t>(shiftHW), magicWReg, static_cast<int16_t>(shiftW), dhOutputActual,
                            wOutputActual, wOutput, hwOutput, wInput, hwInput, baseOffset, cOutputAligned, cIncReg,
                            dLowerReg, hLowerReg, wLowerReg, dUpperReg, hUpperReg, wUpperReg);
                    }
                    AscendC::Reg::LocalMemBar<AscendC::Reg::MemType::VEC_STORE, AscendC::Reg::MemType::VEC_LOAD>();
                }
            }
        }
    }
}

template <typename T, typename INDEX_T, const uint32_t IS_CHECK_RANGE>
__aicore__ inline void MaxPool3DGradNDHWCBackwardBase<T, INDEX_T, IS_CHECK_RANGE>::MultipleLineDhwProcessVF(
    __ubuf__ float* yAddr, __ubuf__ T* gradAddr, __ubuf__ INDEX_T* argmaxAddr)
{
    constexpr int32_t V_REG_SIZE = platform::GetVRegSize();

    int32_t wInput = static_cast<int32_t>(tilingData_.base.wOutput);
    int32_t hInput = static_cast<int32_t>(tilingData_.base.hOutput);
    int32_t hwInput = hInput * wInput;
    int32_t wOutput = static_cast<int32_t>(tilingData_.base.wArgmax);
    int32_t hOutput = static_cast<int32_t>(tilingData_.base.hArgmax);
    int32_t hwOutput = hOutput * wOutput;
    int32_t wOutputActual = static_cast<int32_t>(wOutputActual_);
    int32_t hOutputActual = static_cast<int32_t>(hOutputActual_);
    int32_t dOutputActual = static_cast<int32_t>(dOutputActual_);
    int32_t dhOutputActual = hOutputActual * wOutputActual;
    int32_t curDIndex = static_cast<int32_t>(dAxisIndex_ * tilingData_.base.dOutputInner);
    int32_t curHIndex = static_cast<int32_t>(hAxisIndex_ * tilingData_.base.hOutputInner);
    int32_t curWIndex = static_cast<int32_t>(wAxisIndex_ * tilingData_.base.wOutputInner);
    int32_t cOutputAligned = static_cast<int32_t>(cAligned_);
    int32_t cOutputActual = static_cast<int32_t>(cOutputActual_);
    int32_t cStride = cOutputAligned;
    int32_t cActual = cOutputActual;
    int32_t baseOffsetConst = -(curDIndex * dhOutputActual + curHIndex * wOutputActual + curWIndex) * cStride;

    uint16_t nOutputActual = static_cast<uint16_t>(nOutputActual_);
    uint16_t dArgmaxActual = static_cast<uint16_t>(dArgmaxActual_);
    uint16_t hArgmaxActual = static_cast<uint16_t>(hArgmaxActual_);
    int32_t wArgmaxActual = static_cast<int32_t>(wArgmaxActual_);
    uint16_t hProBatchSize = static_cast<uint16_t>(tilingData_.base.hProBatchSize);
    hProBatchSize = (hProBatchSize > hArgmaxActual) ? hArgmaxActual : hProBatchSize;
    uint16_t wProBatchSize = static_cast<uint16_t>(tilingData_.base.wProBatchSize);
    wProBatchSize = (wProBatchSize > wArgmaxActual) ? static_cast<uint16_t>(wArgmaxActual) : wProBatchSize;

    uint32_t wFullBatchCount = static_cast<uint32_t>(wArgmaxActual) / wProBatchSize;
    uint16_t hFullBatchCount = hArgmaxActual / hProBatchSize;
    uint16_t wRemainTail = static_cast<uint16_t>(wArgmaxActual) - wProBatchSize * wFullBatchCount;
    uint16_t computeSizeFP32 = V_REG_SIZE / sizeof(float);
    uint16_t concurrencyCount = computeSizeFP32 / static_cast<uint16_t>(cOutputActual);
    uint16_t hConcurrentCount = concurrencyCount / static_cast<uint16_t>(wFullBatchCount);
    uint16_t blockConcurrentCount = hFullBatchCount / hConcurrentCount;
    uint16_t hRemain = hArgmaxActual - blockConcurrentCount * hConcurrentCount * hProBatchSize;
    uint16_t hRemainBatchCount = hRemain / hProBatchSize;
    uint16_t hRemainTail = hRemain - hRemainBatchCount * hProBatchSize;

    uint32_t maskBlock = wFullBatchCount * hConcurrentCount * cOutputActual;
    uint32_t maskRemainBatch = wFullBatchCount * hRemainBatchCount * cOutputActual;
    uint32_t maskRemainTail = wFullBatchCount * cOutputActual;
    uint32_t blockOne = hConcurrentCount * cOutputActual;
    uint32_t remainBatchOne = hRemainBatchCount * cOutputActual;
    uint32_t remainTailOne = cOutputActual;

    uint32_t magicHW = 0;
    uint32_t shiftHW = 0;
    uint32_t magicW = 0;
    uint32_t shiftW = 0;
    GetUintDivMagicAndShift<uint32_t>(magicHW, shiftHW, static_cast<uint32_t>(hwInput));
    GetUintDivMagicAndShift<uint32_t>(magicW, shiftW, static_cast<uint32_t>(wInput));
    DivMagic divC = PrecomputeDiv(static_cast<uint32_t>(cOutputActual));
    int32_t hBatchStride = hProBatchSize * wArgmaxActual * cStride;
    int32_t wBatchStride = wProBatchSize * cStride;
    int32_t hBatchStrideA = hProBatchSize * wArgmaxActual * cActual;
    int32_t wBatchStrideA = wProBatchSize * cActual;

    for (uint16_t nIdx = 0; nIdx < nOutputActual; ++nIdx) {
        uint32_t nGradOffset = nIdx * dArgmaxActual * hArgmaxActual * wArgmaxActual * cStride;
        uint32_t nArgmaxOffset = nIdx * dArgmaxActual * hArgmaxActual * wArgmaxActual * cActual;
        int32_t baseOffset = int32_t(nIdx * dOutputActual * dhOutputActual * cStride) + baseOffsetConst;
        for (uint16_t dIdx = 0; dIdx < dArgmaxActual; ++dIdx) {
            uint32_t dGradBase = dIdx * hArgmaxActual * wArgmaxActual * cStride + nGradOffset;
            uint32_t dArgmaxBase = dIdx * hArgmaxActual * wArgmaxActual * cActual + nArgmaxOffset;
            __VEC_SCOPE__
            {
                AscendC::Reg::RegTensor<int32_t> dLowerReg;
                AscendC::Reg::RegTensor<int32_t> wLowerReg;
                AscendC::Reg::RegTensor<int32_t> hLowerReg;
                AscendC::Reg::RegTensor<int32_t> dUpperReg;
                AscendC::Reg::RegTensor<int32_t> hUpperReg;
                AscendC::Reg::RegTensor<int32_t> wUpperReg;
                if constexpr (IS_CHECK_RANGE == 1) {
                    AscendC::Reg::Duplicate(dLowerReg, int32_t(curDIndex));
                    AscendC::Reg::Duplicate(hLowerReg, int32_t(curHIndex));
                    AscendC::Reg::Duplicate(wLowerReg, int32_t(curWIndex));
                    AscendC::Reg::Duplicate(dUpperReg, int32_t(dOutputActual + curDIndex));
                    AscendC::Reg::Duplicate(hUpperReg, int32_t(hOutputActual + curHIndex));
                    AscendC::Reg::Duplicate(wUpperReg, int32_t(wOutputActual + curWIndex));
                }
                AscendC::Reg::RegTensor<uint32_t> magicHWReg;
                AscendC::Reg::RegTensor<uint32_t> magicWReg;
                AscendC::Reg::Duplicate(magicHWReg, magicHW);
                AscendC::Reg::Duplicate(magicWReg, magicW);
                AscendC::Reg::RegTensor<int32_t> initialRegIndex;
                GenPattern3D(initialRegIndex, hBatchStride, wBatchStride, static_cast<int32_t>(wFullBatchCount),
                             cOutputActual, 0);
                AscendC::Reg::RegTensor<int32_t> initialRegArgmax;
                GenPattern3D(initialRegArgmax, hBatchStrideA, wBatchStrideA, static_cast<int32_t>(wFullBatchCount),
                             cOutputActual, 1);
                AscendC::Reg::RegTensor<int32_t> initialRegIndexOne;
                GenPattern2D(initialRegIndexOne, hBatchStride, cOutputActual, SLOT_PATTERN_ONE);
                AscendC::Reg::RegTensor<int32_t> initialRegArgmaxOne;
                GenPattern2D(initialRegArgmaxOne, hBatchStrideA, cOutputActual, SLOT_PATTERN_ONE + 1);
                AscendC::Reg::RegTensor<int32_t> initialRegIndexWBatch;
                GenPattern2D(initialRegIndexWBatch, wBatchStride, cOutputActual, SLOT_PATTERN_W_BATCH);
                AscendC::Reg::RegTensor<int32_t> initialRegArgmaxWBatch;
                GenPattern2D(initialRegArgmaxWBatch, wBatchStrideA, cOutputActual, SLOT_PATTERN_W_BATCH + 1);
                AscendC::Reg::RegTensor<int32_t> parallelRegIndex;
                AscendC::Reg::RegTensor<int32_t> parallelRegArgmax;
                AscendC::Reg::RegTensor<int32_t> cIncReg;
                AscendC::Reg::Arange(cIncReg, 0);
                AscendC::Reg::MaskReg allMaskU32 = AscendC::Reg::CreateMask<uint32_t, AscendC::Reg::MaskPattern::ALL>();

                for (uint16_t hIdx = 0; hIdx < blockConcurrentCount; ++hIdx) {
                    for (uint16_t hProBatchIdx = 0; hProBatchIdx < hProBatchSize; ++hProBatchIdx) {
                        for (uint16_t wBatchIdx = 0; wBatchIdx < wProBatchSize; ++wBatchIdx) {
                            uint32_t offset = (wBatchIdx + hProBatchIdx * wArgmaxActual +
                                               hIdx * wArgmaxActual * hProBatchSize * hConcurrentCount) *
                                                  cStride +
                                              dGradBase;
                            uint32_t argmaxOff = (wBatchIdx + hProBatchIdx * wArgmaxActual +
                                                  hIdx * wArgmaxActual * hProBatchSize * hConcurrentCount) *
                                                     cActual +
                                                 dArgmaxBase;
                            AscendC::Reg::Adds(parallelRegIndex, initialRegIndex, static_cast<int32_t>(offset),
                                               allMaskU32);
                            AscendC::Reg::Adds(parallelRegArgmax, initialRegArgmax, static_cast<int32_t>(argmaxOff),
                                               allMaskU32);
                            DoMulNCNdhwcFastDiv<T, INDEX_T, IS_CHECK_RANGE>(
                                (__local_mem__ computeType*)yAddr, (__local_mem__ T*)gradAddr,
                                (__local_mem__ INDEX_T*)argmaxAddr, parallelRegArgmax, parallelRegIndex, maskBlock,
                                magicHWReg, static_cast<int16_t>(shiftHW), magicWReg, static_cast<int16_t>(shiftW),
                                dhOutputActual, wOutputActual, wOutput, hwOutput, wInput, hwInput, baseOffset,
                                cOutputAligned, cOutputActual, divC, cIncReg, dLowerReg, hLowerReg, wLowerReg,
                                dUpperReg, hUpperReg, wUpperReg);
                        }
                        for (uint16_t wBatchIdx = 0; wBatchIdx < wRemainTail; ++wBatchIdx) {
                            uint32_t offset = (wBatchIdx + wProBatchSize * wFullBatchCount +
                                               hProBatchIdx * wArgmaxActual +
                                               hIdx * wArgmaxActual * hProBatchSize * hConcurrentCount) *
                                                  cStride +
                                              dGradBase;
                            uint32_t argmaxOff = (wBatchIdx + wProBatchSize * wFullBatchCount +
                                                  hProBatchIdx * wArgmaxActual +
                                                  hIdx * wArgmaxActual * hProBatchSize * hConcurrentCount) *
                                                     cActual +
                                                 dArgmaxBase;
                            AscendC::Reg::Adds(parallelRegIndex, initialRegIndexOne, static_cast<int32_t>(offset),
                                               allMaskU32);
                            AscendC::Reg::Adds(parallelRegArgmax, initialRegArgmaxOne, static_cast<int32_t>(argmaxOff),
                                               allMaskU32);
                            DoMulNCNdhwcFastDiv<T, INDEX_T, IS_CHECK_RANGE>(
                                (__local_mem__ computeType*)yAddr, (__local_mem__ T*)gradAddr,
                                (__local_mem__ INDEX_T*)argmaxAddr, parallelRegArgmax, parallelRegIndex, blockOne,
                                magicHWReg, static_cast<int16_t>(shiftHW), magicWReg, static_cast<int16_t>(shiftW),
                                dhOutputActual, wOutputActual, wOutput, hwOutput, wInput, hwInput, baseOffset,
                                cOutputAligned, cOutputActual, divC, cIncReg, dLowerReg, hLowerReg, wLowerReg,
                                dUpperReg, hUpperReg, wUpperReg);
                        }
                    }
                }
                for (uint16_t hProBatchIdx = 0; hProBatchIdx < hRemainBatchCount; ++hProBatchIdx) {
                    for (uint16_t wBatchIdx = 0; wBatchIdx < wProBatchSize; ++wBatchIdx) {
                        uint32_t offset = (wBatchIdx + hProBatchIdx * wArgmaxActual +
                                           blockConcurrentCount * hConcurrentCount * hProBatchSize * wArgmaxActual) *
                                              cStride +
                                          dGradBase;
                        uint32_t argmaxOff = (wBatchIdx + hProBatchIdx * wArgmaxActual +
                                              blockConcurrentCount * hConcurrentCount * hProBatchSize * wArgmaxActual) *
                                                 cActual +
                                             dArgmaxBase;
                        AscendC::Reg::Adds(parallelRegIndex, initialRegIndex, static_cast<int32_t>(offset), allMaskU32);
                        AscendC::Reg::Adds(parallelRegArgmax, initialRegArgmax, static_cast<int32_t>(argmaxOff),
                                           allMaskU32);
                        DoMulNCNdhwcFastDiv<T, INDEX_T, IS_CHECK_RANGE>(
                            (__local_mem__ computeType*)yAddr, (__local_mem__ T*)gradAddr,
                            (__local_mem__ INDEX_T*)argmaxAddr, parallelRegArgmax, parallelRegIndex, maskRemainBatch,
                            magicHWReg, static_cast<int16_t>(shiftHW), magicWReg, static_cast<int16_t>(shiftW),
                            dhOutputActual, wOutputActual, wOutput, hwOutput, wInput, hwInput, baseOffset,
                            cOutputAligned, cOutputActual, divC, cIncReg, dLowerReg, hLowerReg, wLowerReg, dUpperReg,
                            hUpperReg, wUpperReg);
                    }
                    for (uint16_t wBatchIdx = 0; wBatchIdx < wRemainTail; ++wBatchIdx) {
                        uint32_t offset = (wBatchIdx + wProBatchSize * wFullBatchCount + hProBatchIdx * wArgmaxActual +
                                           blockConcurrentCount * hConcurrentCount * hProBatchSize * wArgmaxActual) *
                                              cStride +
                                          dGradBase;
                        uint32_t argmaxOff = (wBatchIdx + wProBatchSize * wFullBatchCount +
                                              hProBatchIdx * wArgmaxActual +
                                              blockConcurrentCount * hConcurrentCount * hProBatchSize * wArgmaxActual) *
                                                 cActual +
                                             dArgmaxBase;
                        AscendC::Reg::Adds(parallelRegIndex, initialRegIndexOne, static_cast<int32_t>(offset),
                                           allMaskU32);
                        AscendC::Reg::Adds(parallelRegArgmax, initialRegArgmaxOne, static_cast<int32_t>(argmaxOff),
                                           allMaskU32);
                        DoMulNCNdhwcFastDiv<T, INDEX_T, IS_CHECK_RANGE>(
                            (__local_mem__ computeType*)yAddr, (__local_mem__ T*)gradAddr,
                            (__local_mem__ INDEX_T*)argmaxAddr, parallelRegArgmax, parallelRegIndex, remainBatchOne,
                            magicHWReg, static_cast<int16_t>(shiftHW), magicWReg, static_cast<int16_t>(shiftW),
                            dhOutputActual, wOutputActual, wOutput, hwOutput, wInput, hwInput, baseOffset,
                            cOutputAligned, cOutputActual, divC, cIncReg, dLowerReg, hLowerReg, wLowerReg, dUpperReg,
                            hUpperReg, wUpperReg);
                    }
                }
                for (uint16_t hProBatchIdx = 0; hProBatchIdx < hRemainTail; ++hProBatchIdx) {
                    for (uint16_t wBatchIdx = 0; wBatchIdx < wProBatchSize; ++wBatchIdx) {
                        uint32_t offset = (wBatchIdx + hProBatchIdx * wArgmaxActual +
                                           hRemainBatchCount * hProBatchSize * wArgmaxActual +
                                           blockConcurrentCount * hConcurrentCount * hProBatchSize * wArgmaxActual) *
                                              cStride +
                                          dGradBase;
                        uint32_t argmaxOff = (wBatchIdx + hProBatchIdx * wArgmaxActual +
                                              hRemainBatchCount * hProBatchSize * wArgmaxActual +
                                              blockConcurrentCount * hConcurrentCount * hProBatchSize * wArgmaxActual) *
                                                 cActual +
                                             dArgmaxBase;
                        AscendC::Reg::Adds(parallelRegIndex, initialRegIndexWBatch, static_cast<int32_t>(offset),
                                           allMaskU32);
                        AscendC::Reg::Adds(parallelRegArgmax, initialRegArgmaxWBatch, static_cast<int32_t>(argmaxOff),
                                           allMaskU32);
                        DoMulNCNdhwcFastDiv<T, INDEX_T, IS_CHECK_RANGE>(
                            (__local_mem__ computeType*)yAddr, (__local_mem__ T*)gradAddr,
                            (__local_mem__ INDEX_T*)argmaxAddr, parallelRegArgmax, parallelRegIndex, maskRemainTail,
                            magicHWReg, static_cast<int16_t>(shiftHW), magicWReg, static_cast<int16_t>(shiftW),
                            dhOutputActual, wOutputActual, wOutput, hwOutput, wInput, hwInput, baseOffset,
                            cOutputAligned, cOutputActual, divC, cIncReg, dLowerReg, hLowerReg, wLowerReg, dUpperReg,
                            hUpperReg, wUpperReg);
                    }
                    for (uint16_t wBatchIdx = 0; wBatchIdx < wRemainTail; ++wBatchIdx) {
                        uint32_t offset = (wBatchIdx + wProBatchSize * wFullBatchCount + hProBatchIdx * wArgmaxActual +
                                           hRemainBatchCount * hProBatchSize * wArgmaxActual +
                                           blockConcurrentCount * hConcurrentCount * hProBatchSize * wArgmaxActual) *
                                              cStride +
                                          dGradBase;
                        uint32_t argmaxOff = (wBatchIdx + wProBatchSize * wFullBatchCount +
                                              hProBatchIdx * wArgmaxActual +
                                              hRemainBatchCount * hProBatchSize * wArgmaxActual +
                                              blockConcurrentCount * hConcurrentCount * hProBatchSize * wArgmaxActual) *
                                                 cActual +
                                             dArgmaxBase;
                        AscendC::Reg::RegTensor<int32_t> initialRegIndexTailOne;
                        PoolGradCommon::Gen2DIndexOne((AscendC::Reg::RegTensor<int32_t>&)initialRegIndexTailOne, 1, 1);
                        AscendC::Reg::Adds(parallelRegIndex, initialRegIndexTailOne, static_cast<int32_t>(offset),
                                           allMaskU32);
                        AscendC::Reg::Adds(parallelRegArgmax, initialRegIndexTailOne, static_cast<int32_t>(argmaxOff),
                                           allMaskU32);
                        DoSingleNCNdhwcFastDiv<T, INDEX_T, IS_CHECK_RANGE>(
                            (__local_mem__ computeType*)yAddr, (__local_mem__ T*)gradAddr,
                            (__local_mem__ INDEX_T*)argmaxAddr, parallelRegArgmax, parallelRegIndex, remainTailOne,
                            magicHWReg, static_cast<int16_t>(shiftHW), magicWReg, static_cast<int16_t>(shiftW),
                            dhOutputActual, wOutputActual, wOutput, hwOutput, wInput, hwInput, baseOffset,
                            cOutputAligned, cIncReg, dLowerReg, hLowerReg, wLowerReg, dUpperReg, hUpperReg, wUpperReg);
                    }
                }
                AscendC::Reg::LocalMemBar<AscendC::Reg::MemType::VEC_STORE, AscendC::Reg::MemType::VEC_LOAD>();
            }
        }
    }
}

template <typename T, typename INDEX_T, const uint32_t IS_CHECK_RANGE>
__aicore__ inline void MaxPool3DGradNDHWCBackwardBase<T, INDEX_T, IS_CHECK_RANGE>::MultipleLineProcessVF2(
    __ubuf__ float* yAddr, __ubuf__ T* gradAddr, __ubuf__ INDEX_T* argmaxAddr)
{
    constexpr int32_t V_REG_SIZE = platform::GetVRegSize();

    int32_t wInput = static_cast<int32_t>(tilingData_.base.wOutput);
    int32_t hInput = static_cast<int32_t>(tilingData_.base.hOutput);
    int32_t hwInput = hInput * wInput;
    int32_t wOutput = static_cast<int32_t>(tilingData_.base.wArgmax);
    int32_t hOutput = static_cast<int32_t>(tilingData_.base.hArgmax);
    int32_t hwOutput = hOutput * wOutput;
    int32_t wOutputActual = static_cast<int32_t>(wOutputActual_);
    int32_t hOutputActual = static_cast<int32_t>(hOutputActual_);
    int32_t dOutputActual = static_cast<int32_t>(dOutputActual_);
    int32_t dhOutputActual = hOutputActual * wOutputActual;
    int32_t curDIndex = static_cast<int32_t>(dAxisIndex_ * tilingData_.base.dOutputInner);
    int32_t curHIndex = static_cast<int32_t>(hAxisIndex_ * tilingData_.base.hOutputInner);
    int32_t curWIndex = static_cast<int32_t>(wAxisIndex_ * tilingData_.base.wOutputInner);
    int32_t cOutputAligned = static_cast<int32_t>(cAligned_);
    int32_t cOutputActual = static_cast<int32_t>(cOutputActual_);
    int32_t cStride = cOutputAligned;
    int32_t cActual = cOutputActual;
    int32_t baseOffsetConst = -(curDIndex * dhOutputActual + curHIndex * wOutputActual + curWIndex) * cStride;

    uint16_t nOutputActual = static_cast<uint16_t>(nOutputActual_);
    uint16_t dArgmaxActual = static_cast<uint16_t>(dArgmaxActual_);
    uint16_t hArgmaxActual = static_cast<uint16_t>(hArgmaxActual_);
    int32_t wArgmaxActual = static_cast<int32_t>(wArgmaxActual_);
    uint16_t dProBatchSize = static_cast<uint16_t>(tilingData_.base.dProBatchSize);
    dProBatchSize = (dProBatchSize > dArgmaxActual) ? dArgmaxActual : dProBatchSize;
    uint16_t hProBatchSize = static_cast<uint16_t>(tilingData_.base.hProBatchSize);
    hProBatchSize = (hProBatchSize > hArgmaxActual) ? hArgmaxActual : hProBatchSize;
    uint16_t wProBatchSize = static_cast<uint16_t>(tilingData_.base.wProBatchSize);
    wProBatchSize = (wProBatchSize > wArgmaxActual) ? static_cast<uint16_t>(wArgmaxActual) : wProBatchSize;

    uint32_t wFullBatchCount = static_cast<uint32_t>(wArgmaxActual) / wProBatchSize;
    uint16_t hFullBatchCount = hArgmaxActual / hProBatchSize;
    uint16_t dFullBatchCount = dArgmaxActual / dProBatchSize;
    uint16_t wRemainTail = static_cast<uint16_t>(wArgmaxActual) - wProBatchSize * wFullBatchCount;
    uint16_t hRemain = hArgmaxActual - hFullBatchCount * hProBatchSize;
    uint16_t hRemainBatchCount = hRemain / hProBatchSize;
    uint16_t hRemainTail = hRemain - hRemainBatchCount * hProBatchSize;

    uint16_t computeSizeFP32 = V_REG_SIZE / sizeof(float);
    uint16_t concurrencyCount = computeSizeFP32 / static_cast<uint16_t>(cOutputActual);
    uint16_t hConcurrentCount = concurrencyCount / static_cast<uint16_t>(wFullBatchCount);
    uint16_t dConcurrentCount = hConcurrentCount / hFullBatchCount;
    uint16_t dBlockConcurrentCount = dFullBatchCount / dConcurrentCount;
    uint16_t dRemain = dArgmaxActual - dBlockConcurrentCount * dConcurrentCount * dProBatchSize;
    uint16_t dRemainBatchCount = dRemain / dProBatchSize;
    uint16_t dRemainTail = dRemain - dRemainBatchCount * dProBatchSize;

    uint32_t mask0 = dConcurrentCount * hFullBatchCount * wFullBatchCount * cOutputActual;
    uint32_t mask1 = dConcurrentCount * hFullBatchCount * cOutputActual;
    uint32_t mask2 = dConcurrentCount * hRemainBatchCount * wFullBatchCount * cOutputActual;
    uint32_t mask3 = dConcurrentCount * hRemainBatchCount * cOutputActual;
    uint32_t mask4 = dConcurrentCount * wFullBatchCount * cOutputActual;
    uint32_t mask5 = dConcurrentCount * cOutputActual;
    uint32_t mask6 = dRemainBatchCount * hFullBatchCount * wFullBatchCount * cOutputActual;
    uint32_t mask7 = dRemainBatchCount * hFullBatchCount * cOutputActual;
    uint32_t mask8 = dRemainBatchCount * hRemainBatchCount * wFullBatchCount * cOutputActual;
    uint32_t mask9 = dRemainBatchCount * hRemainBatchCount * cOutputActual;
    uint32_t mask10 = dRemainBatchCount * wFullBatchCount * cOutputActual;
    uint32_t mask11 = dRemainBatchCount * cOutputActual;
    uint32_t mask12 = hFullBatchCount * wFullBatchCount * cOutputActual;
    uint32_t mask13 = hFullBatchCount * cOutputActual;
    uint32_t mask14 = hRemainBatchCount * wFullBatchCount * cOutputActual;
    uint32_t mask15 = hRemainBatchCount * cOutputActual;
    uint32_t mask16 = wFullBatchCount * cOutputActual;
    uint32_t mask17 = cOutputActual;

    uint32_t magicHW = 0;
    uint32_t shiftHW = 0;
    uint32_t magicW = 0;
    uint32_t shiftW = 0;
    GetUintDivMagicAndShift<uint32_t>(magicHW, shiftHW, static_cast<uint32_t>(hwInput));
    GetUintDivMagicAndShift<uint32_t>(magicW, shiftW, static_cast<uint32_t>(wInput));
    DivMagic divC = PrecomputeDiv(static_cast<uint32_t>(cOutputActual));
    int32_t dBatchStride = dProBatchSize * hArgmaxActual * wArgmaxActual * cStride;
    int32_t hBatchStride = hProBatchSize * wArgmaxActual * cStride;
    int32_t wBatchStride = wProBatchSize * cStride;
    int32_t dhTailStride = hArgmaxActual * wArgmaxActual * cStride;
    int32_t dBatchStrideA = dProBatchSize * hArgmaxActual * wArgmaxActual * cActual;
    int32_t hBatchStrideA = hProBatchSize * wArgmaxActual * cActual;
    int32_t wBatchStrideA = wProBatchSize * cActual;
    int32_t dhTailStrideA = hArgmaxActual * wArgmaxActual * cActual;

    __VEC_SCOPE__
    {
        AscendC::Reg::RegTensor<int32_t> initialRegIndex;
        GenPattern4D(initialRegIndex, dBatchStride, hBatchStride, wBatchStride, static_cast<int32_t>(wFullBatchCount),
                     static_cast<int32_t>(hFullBatchCount), cOutputActual, 0);
        AscendC::Reg::RegTensor<int32_t> initialRegArgmax;
        GenPattern4D(initialRegArgmax, dBatchStrideA, hBatchStrideA, wBatchStrideA,
                     static_cast<int32_t>(wFullBatchCount), static_cast<int32_t>(hFullBatchCount), cOutputActual, 1);
        AscendC::Reg::RegTensor<int32_t> initialRegIndexOne;
        GenPattern3D(initialRegIndexOne, hBatchStride, wBatchStride, static_cast<int32_t>(wFullBatchCount),
                     cOutputActual, SLOT_PATTERN_ONE);
        AscendC::Reg::RegTensor<int32_t> initialRegArgmaxOne;
        GenPattern3D(initialRegArgmaxOne, hBatchStrideA, wBatchStrideA, static_cast<int32_t>(wFullBatchCount),
                     cOutputActual, SLOT_PATTERN_ONE + 1);
        AscendC::Reg::RegTensor<int32_t> initialRegIndexTail;
        GenPattern2D(initialRegIndexTail, hBatchStride, cOutputActual, SLOT_PATTERN_TAIL);
        AscendC::Reg::RegTensor<int32_t> initialRegArgmaxTail;
        GenPattern2D(initialRegArgmaxTail, hBatchStrideA, cOutputActual, SLOT_PATTERN_TAIL + 1);
        AscendC::Reg::RegTensor<int32_t> initialRegIndexWTail;
        GenPattern3D(initialRegIndexWTail, dBatchStride, hBatchStride, static_cast<int32_t>(hFullBatchCount),
                     cOutputActual, SLOT_PATTERN_W_TAIL);
        AscendC::Reg::RegTensor<int32_t> initialRegArgmaxWTail;
        GenPattern3D(initialRegArgmaxWTail, dBatchStrideA, hBatchStrideA, static_cast<int32_t>(hFullBatchCount),
                     cOutputActual, SLOT_PATTERN_W_TAIL + 1);
        AscendC::Reg::RegTensor<int32_t> initialRegIndexDHTail;
        GenPattern2D(initialRegIndexDHTail, dhTailStride, cOutputActual, SLOT_PATTERN_DH_TAIL);
        AscendC::Reg::RegTensor<int32_t> initialRegArgmaxDHTail;
        GenPattern2D(initialRegArgmaxDHTail, dhTailStrideA, cOutputActual, SLOT_PATTERN_DH_TAIL + 1);
        AscendC::Reg::RegTensor<int32_t> initialRegIndexWBatch;
        GenPattern2D(initialRegIndexWBatch, wBatchStride, cOutputActual, SLOT_PATTERN_W_BATCH);
        AscendC::Reg::RegTensor<int32_t> initialRegArgmaxWBatch;
        GenPattern2D(initialRegArgmaxWBatch, wBatchStrideA, cOutputActual, SLOT_PATTERN_W_BATCH + 1);
        AscendC::Reg::RegTensor<int32_t> initialRegIndexHTail;
        GenPattern3D(initialRegIndexHTail, dBatchStride, wBatchStride, static_cast<int32_t>(wFullBatchCount),
                     cOutputActual, SLOT_PATTERN_H_TAIL);
        AscendC::Reg::RegTensor<int32_t> initialRegArgmaxHTail;
        GenPattern3D(initialRegArgmaxHTail, dBatchStrideA, wBatchStrideA, static_cast<int32_t>(wFullBatchCount),
                     cOutputActual, SLOT_PATTERN_H_TAIL + 1);
        AscendC::Reg::RegTensor<int32_t> initialRegIndexOneTail;
        PoolGradCommon::Gen2DIndexOne((AscendC::Reg::RegTensor<int32_t>&)initialRegIndexOneTail, 1, 1);
        AscendC::Reg::MaskReg storeMask = AscendC::Reg::CreateMask<int32_t, AscendC::Reg::MaskPattern::ALL>();
        AscendC::Reg::StoreAlign(PatternSlotAddr(SLOT_PATTERN_ONE_TAIL), initialRegIndexOneTail, storeMask);
        AscendC::Reg::LocalMemBar<AscendC::Reg::MemType::VEC_STORE, AscendC::Reg::MemType::VEC_LOAD>();
    }

    for (uint16_t nIdx = 0; nIdx < nOutputActual; ++nIdx) {
        uint32_t nGradOffset = nIdx * dArgmaxActual * hArgmaxActual * wArgmaxActual * cStride;
        uint32_t nArgmaxOffset = nIdx * dArgmaxActual * hArgmaxActual * wArgmaxActual * cActual;
        int32_t baseOffset = int32_t(nIdx * dOutputActual * dhOutputActual * cStride) + baseOffsetConst;

        __VEC_SCOPE__
        {
            AscendC::Reg::RegTensor<uint32_t> magicHWReg;
            AscendC::Reg::RegTensor<uint32_t> magicWReg;
            AscendC::Reg::Duplicate(magicHWReg, magicHW);
            AscendC::Reg::Duplicate(magicWReg, magicW);
            AscendC::Reg::RegTensor<int32_t> dLowerReg;
            AscendC::Reg::RegTensor<int32_t> hLowerReg;
            AscendC::Reg::RegTensor<int32_t> wLowerReg;
            AscendC::Reg::RegTensor<int32_t> dUpperReg;
            AscendC::Reg::RegTensor<int32_t> hUpperReg;
            AscendC::Reg::RegTensor<int32_t> wUpperReg;
            if constexpr (IS_CHECK_RANGE == 1) {
                AscendC::Reg::Duplicate(dLowerReg, int32_t(curDIndex));
                AscendC::Reg::Duplicate(hLowerReg, int32_t(curHIndex));
                AscendC::Reg::Duplicate(wLowerReg, int32_t(curWIndex));
                AscendC::Reg::Duplicate(dUpperReg, int32_t(dOutputActual + curDIndex));
                AscendC::Reg::Duplicate(hUpperReg, int32_t(hOutputActual + curHIndex));
                AscendC::Reg::Duplicate(wUpperReg, int32_t(wOutputActual + curWIndex));
            }
            AscendC::Reg::RegTensor<int32_t> initialRegIndex;
            AscendC::Reg::LoadAlign(initialRegIndex, PatternSlotAddr(0));
            AscendC::Reg::RegTensor<int32_t> initialRegArgmax;
            AscendC::Reg::LoadAlign(initialRegArgmax, PatternSlotAddr(1));
            AscendC::Reg::RegTensor<int32_t> initialRegIndexWTail;
            AscendC::Reg::LoadAlign(initialRegIndexWTail, PatternSlotAddr(SLOT_PATTERN_W_TAIL));
            AscendC::Reg::RegTensor<int32_t> initialRegArgmaxWTail;
            AscendC::Reg::LoadAlign(initialRegArgmaxWTail, PatternSlotAddr(SLOT_PATTERN_W_TAIL + 1));
            AscendC::Reg::RegTensor<int32_t> initialRegIndexDHTail;
            AscendC::Reg::LoadAlign(initialRegIndexDHTail, PatternSlotAddr(SLOT_PATTERN_DH_TAIL));
            AscendC::Reg::RegTensor<int32_t> initialRegArgmaxDHTail;
            AscendC::Reg::LoadAlign(initialRegArgmaxDHTail, PatternSlotAddr(SLOT_PATTERN_DH_TAIL + 1));
            AscendC::Reg::RegTensor<int32_t> initialRegIndexHTail;
            AscendC::Reg::LoadAlign(initialRegIndexHTail, PatternSlotAddr(SLOT_PATTERN_H_TAIL));
            AscendC::Reg::RegTensor<int32_t> initialRegArgmaxHTail;
            AscendC::Reg::LoadAlign(initialRegArgmaxHTail, PatternSlotAddr(SLOT_PATTERN_H_TAIL + 1));
            AscendC::Reg::RegTensor<int32_t> parallelRegIndex;
            AscendC::Reg::RegTensor<int32_t> parallelRegArgmax;
            AscendC::Reg::RegTensor<int32_t> cIncReg;
            AscendC::Reg::Arange(cIncReg, 0);
            AscendC::Reg::MaskReg allMaskU32 = AscendC::Reg::CreateMask<uint32_t, AscendC::Reg::MaskPattern::ALL>();
            for (uint16_t dIdx = 0; dIdx < dBlockConcurrentCount; ++dIdx) {
                for (uint16_t dProBatchIdx = 0; dProBatchIdx < dProBatchSize; ++dProBatchIdx) {
                    // --- H full ---
                    for (uint16_t hProBatchIdx = 0; hProBatchIdx < hProBatchSize; ++hProBatchIdx) {
                        for (uint16_t wBatchIdx = 0; wBatchIdx < wProBatchSize; ++wBatchIdx) {
                            uint32_t offset = (wBatchIdx + hProBatchIdx * wArgmaxActual +
                                               dProBatchIdx * hArgmaxActual * wArgmaxActual +
                                               dIdx * dProBatchSize * hArgmaxActual * wArgmaxActual *
                                                   dConcurrentCount) *
                                                  cStride +
                                              nGradOffset;
                            uint32_t argmaxOff = (wBatchIdx + hProBatchIdx * wArgmaxActual +
                                                  dProBatchIdx * hArgmaxActual * wArgmaxActual +
                                                  dIdx * dProBatchSize * hArgmaxActual * wArgmaxActual *
                                                      dConcurrentCount) *
                                                     cActual +
                                                 nArgmaxOffset;
                            AscendC::Reg::Adds(parallelRegIndex, initialRegIndex, static_cast<int32_t>(offset),
                                               allMaskU32);
                            AscendC::Reg::Adds(parallelRegArgmax, initialRegArgmax, static_cast<int32_t>(argmaxOff),
                                               allMaskU32);
                            DoMulNCNdhwcFastDiv<T, INDEX_T, IS_CHECK_RANGE>(
                                (__local_mem__ computeType*)yAddr, (__local_mem__ T*)gradAddr,
                                (__local_mem__ INDEX_T*)argmaxAddr, parallelRegArgmax, parallelRegIndex, mask0,
                                magicHWReg, static_cast<int16_t>(shiftHW), magicWReg, static_cast<int16_t>(shiftW),
                                dhOutputActual, wOutputActual, wOutput, hwOutput, wInput, hwInput, baseOffset,
                                cOutputAligned, cOutputActual, divC, cIncReg, dLowerReg, hLowerReg, wLowerReg,
                                dUpperReg, hUpperReg, wUpperReg);
                        }
                        for (uint16_t wBatchIdx = 0; wBatchIdx < wRemainTail; ++wBatchIdx) {
                            uint32_t offset = (wBatchIdx + wProBatchSize * wFullBatchCount +
                                               hProBatchIdx * wArgmaxActual +
                                               dProBatchIdx * hArgmaxActual * wArgmaxActual +
                                               dIdx * dProBatchSize * hArgmaxActual * wArgmaxActual *
                                                   dConcurrentCount) *
                                                  cStride +
                                              nGradOffset;
                            uint32_t argmaxOff = (wBatchIdx + wProBatchSize * wFullBatchCount +
                                                  hProBatchIdx * wArgmaxActual +
                                                  dProBatchIdx * hArgmaxActual * wArgmaxActual +
                                                  dIdx * dProBatchSize * hArgmaxActual * wArgmaxActual *
                                                      dConcurrentCount) *
                                                     cActual +
                                                 nArgmaxOffset;
                            AscendC::Reg::Adds(parallelRegIndex, initialRegIndexWTail, static_cast<int32_t>(offset),
                                               allMaskU32);
                            AscendC::Reg::Adds(parallelRegArgmax, initialRegArgmaxWTail,
                                               static_cast<int32_t>(argmaxOff), allMaskU32);
                            DoMulNCNdhwcFastDiv<T, INDEX_T, IS_CHECK_RANGE>(
                                (__local_mem__ computeType*)yAddr, (__local_mem__ T*)gradAddr,
                                (__local_mem__ INDEX_T*)argmaxAddr, parallelRegArgmax, parallelRegIndex, mask1,
                                magicHWReg, static_cast<int16_t>(shiftHW), magicWReg, static_cast<int16_t>(shiftW),
                                dhOutputActual, wOutputActual, wOutput, hwOutput, wInput, hwInput, baseOffset,
                                cOutputAligned, cOutputActual, divC, cIncReg, dLowerReg, hLowerReg, wLowerReg,
                                dUpperReg, hUpperReg, wUpperReg);
                        }
                    }
                    // --- H 余 batch ---
                    for (uint16_t hProBatchIdx = 0; hProBatchIdx < hRemainBatchCount; ++hProBatchIdx) {
                        for (uint16_t wBatchIdx = 0; wBatchIdx < wProBatchSize; ++wBatchIdx) {
                            uint32_t offset = (wBatchIdx +
                                               (hProBatchIdx + hFullBatchCount * hProBatchSize) * wArgmaxActual +
                                               dProBatchIdx * hArgmaxActual * wArgmaxActual +
                                               dIdx * dProBatchSize * hArgmaxActual * wArgmaxActual *
                                                   dConcurrentCount) *
                                                  cStride +
                                              nGradOffset;
                            uint32_t argmaxOff = (wBatchIdx +
                                                  (hProBatchIdx + hFullBatchCount * hProBatchSize) * wArgmaxActual +
                                                  dProBatchIdx * hArgmaxActual * wArgmaxActual +
                                                  dIdx * dProBatchSize * hArgmaxActual * wArgmaxActual *
                                                      dConcurrentCount) *
                                                     cActual +
                                                 nArgmaxOffset;
                            AscendC::Reg::Adds(parallelRegIndex, initialRegIndex, static_cast<int32_t>(offset),
                                               allMaskU32);
                            AscendC::Reg::Adds(parallelRegArgmax, initialRegArgmax, static_cast<int32_t>(argmaxOff),
                                               allMaskU32);
                            DoMulNCNdhwcFastDiv<T, INDEX_T, IS_CHECK_RANGE>(
                                (__local_mem__ computeType*)yAddr, (__local_mem__ T*)gradAddr,
                                (__local_mem__ INDEX_T*)argmaxAddr, parallelRegArgmax, parallelRegIndex, mask2,
                                magicHWReg, static_cast<int16_t>(shiftHW), magicWReg, static_cast<int16_t>(shiftW),
                                dhOutputActual, wOutputActual, wOutput, hwOutput, wInput, hwInput, baseOffset,
                                cOutputAligned, cOutputActual, divC, cIncReg, dLowerReg, hLowerReg, wLowerReg,
                                dUpperReg, hUpperReg, wUpperReg);
                        }
                        for (uint16_t wBatchIdx = 0; wBatchIdx < wRemainTail; ++wBatchIdx) {
                            uint32_t offset = (wBatchIdx + wProBatchSize * wFullBatchCount +
                                               (hProBatchIdx + hFullBatchCount * hProBatchSize) * wArgmaxActual +
                                               dProBatchIdx * hArgmaxActual * wArgmaxActual +
                                               dIdx * dProBatchSize * hArgmaxActual * wArgmaxActual *
                                                   dConcurrentCount) *
                                                  cStride +
                                              nGradOffset;
                            uint32_t argmaxOff = (wBatchIdx + wProBatchSize * wFullBatchCount +
                                                  (hProBatchIdx + hFullBatchCount * hProBatchSize) * wArgmaxActual +
                                                  dProBatchIdx * hArgmaxActual * wArgmaxActual +
                                                  dIdx * dProBatchSize * hArgmaxActual * wArgmaxActual *
                                                      dConcurrentCount) *
                                                     cActual +
                                                 nArgmaxOffset;
                            AscendC::Reg::Adds(parallelRegIndex, initialRegIndexWTail, static_cast<int32_t>(offset),
                                               allMaskU32);
                            AscendC::Reg::Adds(parallelRegArgmax, initialRegArgmaxWTail,
                                               static_cast<int32_t>(argmaxOff), allMaskU32);
                            DoMulNCNdhwcFastDiv<T, INDEX_T, IS_CHECK_RANGE>(
                                (__local_mem__ computeType*)yAddr, (__local_mem__ T*)gradAddr,
                                (__local_mem__ INDEX_T*)argmaxAddr, parallelRegArgmax, parallelRegIndex, mask3,
                                magicHWReg, static_cast<int16_t>(shiftHW), magicWReg, static_cast<int16_t>(shiftW),
                                dhOutputActual, wOutputActual, wOutput, hwOutput, wInput, hwInput, baseOffset,
                                cOutputAligned, cOutputActual, divC, cIncReg, dLowerReg, hLowerReg, wLowerReg,
                                dUpperReg, hUpperReg, wUpperReg);
                        }
                    }
                    // --- H 余 tail ---
                    for (uint16_t hProBatchIdx = 0; hProBatchIdx < hRemainTail; ++hProBatchIdx) {
                        for (uint16_t wBatchIdx = 0; wBatchIdx < wProBatchSize; ++wBatchIdx) {
                            uint32_t offset = (wBatchIdx +
                                               (hProBatchIdx + hRemainBatchCount * hProBatchSize +
                                                hFullBatchCount * hProBatchSize) *
                                                   wArgmaxActual +
                                               dProBatchIdx * hArgmaxActual * wArgmaxActual +
                                               dIdx * dProBatchSize * hArgmaxActual * wArgmaxActual *
                                                   dConcurrentCount) *
                                                  cStride +
                                              nGradOffset;
                            uint32_t argmaxOff = (wBatchIdx +
                                                  (hProBatchIdx + hRemainBatchCount * hProBatchSize +
                                                   hFullBatchCount * hProBatchSize) *
                                                      wArgmaxActual +
                                                  dProBatchIdx * hArgmaxActual * wArgmaxActual +
                                                  dIdx * dProBatchSize * hArgmaxActual * wArgmaxActual *
                                                      dConcurrentCount) *
                                                     cActual +
                                                 nArgmaxOffset;
                            AscendC::Reg::Adds(parallelRegIndex, initialRegIndexHTail, static_cast<int32_t>(offset),
                                               allMaskU32);
                            AscendC::Reg::Adds(parallelRegArgmax, initialRegArgmaxHTail,
                                               static_cast<int32_t>(argmaxOff), allMaskU32);
                            DoMulNCNdhwcFastDiv<T, INDEX_T, IS_CHECK_RANGE>(
                                (__local_mem__ computeType*)yAddr, (__local_mem__ T*)gradAddr,
                                (__local_mem__ INDEX_T*)argmaxAddr, parallelRegArgmax, parallelRegIndex, mask4,
                                magicHWReg, static_cast<int16_t>(shiftHW), magicWReg, static_cast<int16_t>(shiftW),
                                dhOutputActual, wOutputActual, wOutput, hwOutput, wInput, hwInput, baseOffset,
                                cOutputAligned, cOutputActual, divC, cIncReg, dLowerReg, hLowerReg, wLowerReg,
                                dUpperReg, hUpperReg, wUpperReg);
                        }
                        for (uint16_t wBatchIdx = 0; wBatchIdx < wRemainTail; ++wBatchIdx) {
                            uint32_t offset = (wBatchIdx + wProBatchSize * wFullBatchCount +
                                               (hProBatchIdx + hRemainBatchCount * hProBatchSize +
                                                hFullBatchCount * hProBatchSize) *
                                                   wArgmaxActual +
                                               dProBatchIdx * hArgmaxActual * wArgmaxActual +
                                               dIdx * dProBatchSize * hArgmaxActual * wArgmaxActual *
                                                   dConcurrentCount) *
                                                  cStride +
                                              nGradOffset;
                            uint32_t argmaxOff = (wBatchIdx + wProBatchSize * wFullBatchCount +
                                                  (hProBatchIdx + hRemainBatchCount * hProBatchSize +
                                                   hFullBatchCount * hProBatchSize) *
                                                      wArgmaxActual +
                                                  dProBatchIdx * hArgmaxActual * wArgmaxActual +
                                                  dIdx * dProBatchSize * hArgmaxActual * wArgmaxActual *
                                                      dConcurrentCount) *
                                                     cActual +
                                                 nArgmaxOffset;
                            AscendC::Reg::Adds(parallelRegIndex, initialRegIndexDHTail, static_cast<int32_t>(offset),
                                               allMaskU32);
                            AscendC::Reg::Adds(parallelRegArgmax, initialRegArgmaxDHTail,
                                               static_cast<int32_t>(argmaxOff), allMaskU32);
                            DoMulNCNdhwcFastDiv<T, INDEX_T, IS_CHECK_RANGE>(
                                (__local_mem__ computeType*)yAddr, (__local_mem__ T*)gradAddr,
                                (__local_mem__ INDEX_T*)argmaxAddr, parallelRegArgmax, parallelRegIndex, mask5,
                                magicHWReg, static_cast<int16_t>(shiftHW), magicWReg, static_cast<int16_t>(shiftW),
                                dhOutputActual, wOutputActual, wOutput, hwOutput, wInput, hwInput, baseOffset,
                                cOutputAligned, cOutputActual, divC, cIncReg, dLowerReg, hLowerReg, wLowerReg,
                                dUpperReg, hUpperReg, wUpperReg);
                        }
                    }
                }
            }
            AscendC::Reg::LocalMemBar<AscendC::Reg::MemType::VEC_STORE, AscendC::Reg::MemType::VEC_LOAD>();
        }

        __VEC_SCOPE__
        {
            AscendC::Reg::RegTensor<uint32_t> magicHWReg;
            AscendC::Reg::RegTensor<uint32_t> magicWReg;
            AscendC::Reg::Duplicate(magicHWReg, magicHW);
            AscendC::Reg::Duplicate(magicWReg, magicW);
            AscendC::Reg::RegTensor<int32_t> dLowerReg;
            AscendC::Reg::RegTensor<int32_t> hLowerReg;
            AscendC::Reg::RegTensor<int32_t> wLowerReg;
            AscendC::Reg::RegTensor<int32_t> dUpperReg;
            AscendC::Reg::RegTensor<int32_t> hUpperReg;
            AscendC::Reg::RegTensor<int32_t> wUpperReg;
            if constexpr (IS_CHECK_RANGE == 1) {
                AscendC::Reg::Duplicate(dLowerReg, int32_t(curDIndex));
                AscendC::Reg::Duplicate(hLowerReg, int32_t(curHIndex));
                AscendC::Reg::Duplicate(wLowerReg, int32_t(curWIndex));
                AscendC::Reg::Duplicate(dUpperReg, int32_t(dOutputActual + curDIndex));
                AscendC::Reg::Duplicate(hUpperReg, int32_t(hOutputActual + curHIndex));
                AscendC::Reg::Duplicate(wUpperReg, int32_t(wOutputActual + curWIndex));
            }
            AscendC::Reg::RegTensor<int32_t> initialRegIndex;
            AscendC::Reg::LoadAlign(initialRegIndex, PatternSlotAddr(0));
            AscendC::Reg::RegTensor<int32_t> initialRegArgmax;
            AscendC::Reg::LoadAlign(initialRegArgmax, PatternSlotAddr(1));
            AscendC::Reg::RegTensor<int32_t> initialRegIndexWTail;
            AscendC::Reg::LoadAlign(initialRegIndexWTail, PatternSlotAddr(SLOT_PATTERN_W_TAIL));
            AscendC::Reg::RegTensor<int32_t> initialRegArgmaxWTail;
            AscendC::Reg::LoadAlign(initialRegArgmaxWTail, PatternSlotAddr(SLOT_PATTERN_W_TAIL + 1));
            AscendC::Reg::RegTensor<int32_t> initialRegIndexDHTail;
            AscendC::Reg::LoadAlign(initialRegIndexDHTail, PatternSlotAddr(SLOT_PATTERN_DH_TAIL));
            AscendC::Reg::RegTensor<int32_t> initialRegArgmaxDHTail;
            AscendC::Reg::LoadAlign(initialRegArgmaxDHTail, PatternSlotAddr(SLOT_PATTERN_DH_TAIL + 1));
            AscendC::Reg::RegTensor<int32_t> initialRegIndexHTail;
            AscendC::Reg::LoadAlign(initialRegIndexHTail, PatternSlotAddr(SLOT_PATTERN_H_TAIL));
            AscendC::Reg::RegTensor<int32_t> initialRegArgmaxHTail;
            AscendC::Reg::LoadAlign(initialRegArgmaxHTail, PatternSlotAddr(SLOT_PATTERN_H_TAIL + 1));
            AscendC::Reg::RegTensor<int32_t> parallelRegIndex;
            AscendC::Reg::RegTensor<int32_t> parallelRegArgmax;
            AscendC::Reg::RegTensor<int32_t> cIncReg;
            AscendC::Reg::Arange(cIncReg, 0);
            AscendC::Reg::MaskReg allMaskU32 = AscendC::Reg::CreateMask<uint32_t, AscendC::Reg::MaskPattern::ALL>();
            for (uint16_t dProBatchIdx = 0; dProBatchIdx < dProBatchSize; ++dProBatchIdx) {
                // --- H full ---
                for (uint16_t hProBatchIdx = 0; hProBatchIdx < hProBatchSize; ++hProBatchIdx) {
                    for (uint16_t wBatchIdx = 0; wBatchIdx < wProBatchSize; ++wBatchIdx) {
                        uint32_t offset = (wBatchIdx + hProBatchIdx * wArgmaxActual +
                                           dProBatchIdx * hArgmaxActual * wArgmaxActual +
                                           dBlockConcurrentCount * dConcurrentCount * dProBatchSize * hArgmaxActual *
                                               wArgmaxActual) *
                                              cStride +
                                          nGradOffset;
                        uint32_t argmaxOff = (wBatchIdx + hProBatchIdx * wArgmaxActual +
                                              dProBatchIdx * hArgmaxActual * wArgmaxActual +
                                              dBlockConcurrentCount * dConcurrentCount * dProBatchSize * hArgmaxActual *
                                                  wArgmaxActual) *
                                                 cActual +
                                             nArgmaxOffset;
                        AscendC::Reg::Adds(parallelRegIndex, initialRegIndex, static_cast<int32_t>(offset), allMaskU32);
                        AscendC::Reg::Adds(parallelRegArgmax, initialRegArgmax, static_cast<int32_t>(argmaxOff),
                                           allMaskU32);
                        DoMulNCNdhwcFastDiv<T, INDEX_T, IS_CHECK_RANGE>(
                            (__local_mem__ computeType*)yAddr, (__local_mem__ T*)gradAddr,
                            (__local_mem__ INDEX_T*)argmaxAddr, parallelRegArgmax, parallelRegIndex, mask6, magicHWReg,
                            static_cast<int16_t>(shiftHW), magicWReg, static_cast<int16_t>(shiftW), dhOutputActual,
                            wOutputActual, wOutput, hwOutput, wInput, hwInput, baseOffset, cOutputAligned,
                            cOutputActual, divC, cIncReg, dLowerReg, hLowerReg, wLowerReg, dUpperReg, hUpperReg,
                            wUpperReg);
                    }
                    for (uint16_t wBatchIdx = 0; wBatchIdx < wRemainTail; ++wBatchIdx) {
                        uint32_t offset = (wBatchIdx + wProBatchSize * wFullBatchCount + hProBatchIdx * wArgmaxActual +
                                           dProBatchIdx * hArgmaxActual * wArgmaxActual +
                                           dBlockConcurrentCount * dConcurrentCount * dProBatchSize * hArgmaxActual *
                                               wArgmaxActual) *
                                              cStride +
                                          nGradOffset;
                        uint32_t argmaxOff = (wBatchIdx + wProBatchSize * wFullBatchCount +
                                              hProBatchIdx * wArgmaxActual +
                                              dProBatchIdx * hArgmaxActual * wArgmaxActual +
                                              dBlockConcurrentCount * dConcurrentCount * dProBatchSize * hArgmaxActual *
                                                  wArgmaxActual) *
                                                 cActual +
                                             nArgmaxOffset;
                        AscendC::Reg::Adds(parallelRegIndex, initialRegIndexWTail, static_cast<int32_t>(offset),
                                           allMaskU32);
                        AscendC::Reg::Adds(parallelRegArgmax, initialRegArgmaxWTail, static_cast<int32_t>(argmaxOff),
                                           allMaskU32);
                        DoMulNCNdhwcFastDiv<T, INDEX_T, IS_CHECK_RANGE>(
                            (__local_mem__ computeType*)yAddr, (__local_mem__ T*)gradAddr,
                            (__local_mem__ INDEX_T*)argmaxAddr, parallelRegArgmax, parallelRegIndex, mask7, magicHWReg,
                            static_cast<int16_t>(shiftHW), magicWReg, static_cast<int16_t>(shiftW), dhOutputActual,
                            wOutputActual, wOutput, hwOutput, wInput, hwInput, baseOffset, cOutputAligned,
                            cOutputActual, divC, cIncReg, dLowerReg, hLowerReg, wLowerReg, dUpperReg, hUpperReg,
                            wUpperReg);
                    }
                }
                // --- H 余 batch ---
                for (uint16_t hProBatchIdx = 0; hProBatchIdx < hRemainBatchCount; ++hProBatchIdx) {
                    for (uint16_t wBatchIdx = 0; wBatchIdx < wProBatchSize; ++wBatchIdx) {
                        uint32_t offset = (wBatchIdx +
                                           (hProBatchIdx + hFullBatchCount * hProBatchSize) * wArgmaxActual +
                                           dProBatchIdx * hArgmaxActual * wArgmaxActual +
                                           dBlockConcurrentCount * dConcurrentCount * dProBatchSize * hArgmaxActual *
                                               wArgmaxActual) *
                                              cStride +
                                          nGradOffset;
                        uint32_t argmaxOff = (wBatchIdx +
                                              (hProBatchIdx + hFullBatchCount * hProBatchSize) * wArgmaxActual +
                                              dProBatchIdx * hArgmaxActual * wArgmaxActual +
                                              dBlockConcurrentCount * dConcurrentCount * dProBatchSize * hArgmaxActual *
                                                  wArgmaxActual) *
                                                 cActual +
                                             nArgmaxOffset;
                        AscendC::Reg::Adds(parallelRegIndex, initialRegIndex, static_cast<int32_t>(offset), allMaskU32);
                        AscendC::Reg::Adds(parallelRegArgmax, initialRegArgmax, static_cast<int32_t>(argmaxOff),
                                           allMaskU32);
                        DoMulNCNdhwcFastDiv<T, INDEX_T, IS_CHECK_RANGE>(
                            (__local_mem__ computeType*)yAddr, (__local_mem__ T*)gradAddr,
                            (__local_mem__ INDEX_T*)argmaxAddr, parallelRegArgmax, parallelRegIndex, mask8, magicHWReg,
                            static_cast<int16_t>(shiftHW), magicWReg, static_cast<int16_t>(shiftW), dhOutputActual,
                            wOutputActual, wOutput, hwOutput, wInput, hwInput, baseOffset, cOutputAligned,
                            cOutputActual, divC, cIncReg, dLowerReg, hLowerReg, wLowerReg, dUpperReg, hUpperReg,
                            wUpperReg);
                    }
                    for (uint16_t wBatchIdx = 0; wBatchIdx < wRemainTail; ++wBatchIdx) {
                        uint32_t offset = (wBatchIdx + wProBatchSize * wFullBatchCount +
                                           (hProBatchIdx + hFullBatchCount * hProBatchSize) * wArgmaxActual +
                                           dProBatchIdx * hArgmaxActual * wArgmaxActual +
                                           dBlockConcurrentCount * dConcurrentCount * dProBatchSize * hArgmaxActual *
                                               wArgmaxActual) *
                                              cStride +
                                          nGradOffset;
                        uint32_t argmaxOff = (wBatchIdx + wProBatchSize * wFullBatchCount +
                                              (hProBatchIdx + hFullBatchCount * hProBatchSize) * wArgmaxActual +
                                              dProBatchIdx * hArgmaxActual * wArgmaxActual +
                                              dBlockConcurrentCount * dConcurrentCount * dProBatchSize * hArgmaxActual *
                                                  wArgmaxActual) *
                                                 cActual +
                                             nArgmaxOffset;
                        AscendC::Reg::Adds(parallelRegIndex, initialRegIndexWTail, static_cast<int32_t>(offset),
                                           allMaskU32);
                        AscendC::Reg::Adds(parallelRegArgmax, initialRegArgmaxWTail, static_cast<int32_t>(argmaxOff),
                                           allMaskU32);
                        DoMulNCNdhwcFastDiv<T, INDEX_T, IS_CHECK_RANGE>(
                            (__local_mem__ computeType*)yAddr, (__local_mem__ T*)gradAddr,
                            (__local_mem__ INDEX_T*)argmaxAddr, parallelRegArgmax, parallelRegIndex, mask9, magicHWReg,
                            static_cast<int16_t>(shiftHW), magicWReg, static_cast<int16_t>(shiftW), dhOutputActual,
                            wOutputActual, wOutput, hwOutput, wInput, hwInput, baseOffset, cOutputAligned,
                            cOutputActual, divC, cIncReg, dLowerReg, hLowerReg, wLowerReg, dUpperReg, hUpperReg,
                            wUpperReg);
                    }
                }
                // --- H 余 tail ---
                for (uint16_t hProBatchIdx = 0; hProBatchIdx < hRemainTail; ++hProBatchIdx) {
                    for (uint16_t wBatchIdx = 0; wBatchIdx < wProBatchSize; ++wBatchIdx) {
                        uint32_t offset = (wBatchIdx +
                                           (hProBatchIdx + hRemainBatchCount * hProBatchSize +
                                            hFullBatchCount * hProBatchSize) *
                                               wArgmaxActual +
                                           dProBatchIdx * hArgmaxActual * wArgmaxActual +
                                           dBlockConcurrentCount * dConcurrentCount * dProBatchSize * hArgmaxActual *
                                               wArgmaxActual) *
                                              cStride +
                                          nGradOffset;
                        uint32_t argmaxOff = (wBatchIdx +
                                              (hProBatchIdx + hRemainBatchCount * hProBatchSize +
                                               hFullBatchCount * hProBatchSize) *
                                                  wArgmaxActual +
                                              dProBatchIdx * hArgmaxActual * wArgmaxActual +
                                              dBlockConcurrentCount * dConcurrentCount * dProBatchSize * hArgmaxActual *
                                                  wArgmaxActual) *
                                                 cActual +
                                             nArgmaxOffset;
                        AscendC::Reg::Adds(parallelRegIndex, initialRegIndexHTail, static_cast<int32_t>(offset),
                                           allMaskU32);
                        AscendC::Reg::Adds(parallelRegArgmax, initialRegArgmaxHTail, static_cast<int32_t>(argmaxOff),
                                           allMaskU32);
                        DoMulNCNdhwcFastDiv<T, INDEX_T, IS_CHECK_RANGE>(
                            (__local_mem__ computeType*)yAddr, (__local_mem__ T*)gradAddr,
                            (__local_mem__ INDEX_T*)argmaxAddr, parallelRegArgmax, parallelRegIndex, mask10, magicHWReg,
                            static_cast<int16_t>(shiftHW), magicWReg, static_cast<int16_t>(shiftW), dhOutputActual,
                            wOutputActual, wOutput, hwOutput, wInput, hwInput, baseOffset, cOutputAligned,
                            cOutputActual, divC, cIncReg, dLowerReg, hLowerReg, wLowerReg, dUpperReg, hUpperReg,
                            wUpperReg);
                    }
                    for (uint16_t wBatchIdx = 0; wBatchIdx < wRemainTail; ++wBatchIdx) {
                        uint32_t offset = (wBatchIdx + wProBatchSize * wFullBatchCount +
                                           (hProBatchIdx + hRemainBatchCount * hProBatchSize +
                                            hFullBatchCount * hProBatchSize) *
                                               wArgmaxActual +
                                           dProBatchIdx * hArgmaxActual * wArgmaxActual +
                                           dBlockConcurrentCount * dConcurrentCount * dProBatchSize * hArgmaxActual *
                                               wArgmaxActual) *
                                              cStride +
                                          nGradOffset;
                        uint32_t argmaxOff = (wBatchIdx + wProBatchSize * wFullBatchCount +
                                              (hProBatchIdx + hRemainBatchCount * hProBatchSize +
                                               hFullBatchCount * hProBatchSize) *
                                                  wArgmaxActual +
                                              dProBatchIdx * hArgmaxActual * wArgmaxActual +
                                              dBlockConcurrentCount * dConcurrentCount * dProBatchSize * hArgmaxActual *
                                                  wArgmaxActual) *
                                                 cActual +
                                             nArgmaxOffset;
                        AscendC::Reg::Adds(parallelRegIndex, initialRegIndexDHTail, static_cast<int32_t>(offset),
                                           allMaskU32);
                        AscendC::Reg::Adds(parallelRegArgmax, initialRegArgmaxDHTail, static_cast<int32_t>(argmaxOff),
                                           allMaskU32);
                        DoMulNCNdhwcFastDiv<T, INDEX_T, IS_CHECK_RANGE>(
                            (__local_mem__ computeType*)yAddr, (__local_mem__ T*)gradAddr,
                            (__local_mem__ INDEX_T*)argmaxAddr, parallelRegArgmax, parallelRegIndex, mask11, magicHWReg,
                            static_cast<int16_t>(shiftHW), magicWReg, static_cast<int16_t>(shiftW), dhOutputActual,
                            wOutputActual, wOutput, hwOutput, wInput, hwInput, baseOffset, cOutputAligned,
                            cOutputActual, divC, cIncReg, dLowerReg, hLowerReg, wLowerReg, dUpperReg, hUpperReg,
                            wUpperReg);
                    }
                }
            }
            AscendC::Reg::LocalMemBar<AscendC::Reg::MemType::VEC_STORE, AscendC::Reg::MemType::VEC_LOAD>();
        }

        __VEC_SCOPE__
        {
            AscendC::Reg::RegTensor<uint32_t> magicHWReg;
            AscendC::Reg::RegTensor<uint32_t> magicWReg;
            AscendC::Reg::Duplicate(magicHWReg, magicHW);
            AscendC::Reg::Duplicate(magicWReg, magicW);
            AscendC::Reg::RegTensor<int32_t> dLowerReg;
            AscendC::Reg::RegTensor<int32_t> hLowerReg;
            AscendC::Reg::RegTensor<int32_t> wLowerReg;
            AscendC::Reg::RegTensor<int32_t> dUpperReg;
            AscendC::Reg::RegTensor<int32_t> hUpperReg;
            AscendC::Reg::RegTensor<int32_t> wUpperReg;
            if constexpr (IS_CHECK_RANGE == 1) {
                AscendC::Reg::Duplicate(dLowerReg, int32_t(curDIndex));
                AscendC::Reg::Duplicate(hLowerReg, int32_t(curHIndex));
                AscendC::Reg::Duplicate(wLowerReg, int32_t(curWIndex));
                AscendC::Reg::Duplicate(dUpperReg, int32_t(dOutputActual + curDIndex));
                AscendC::Reg::Duplicate(hUpperReg, int32_t(hOutputActual + curHIndex));
                AscendC::Reg::Duplicate(wUpperReg, int32_t(wOutputActual + curWIndex));
            }
            AscendC::Reg::RegTensor<int32_t> initialRegIndexOne;
            AscendC::Reg::LoadAlign(initialRegIndexOne, PatternSlotAddr(SLOT_PATTERN_ONE));
            AscendC::Reg::RegTensor<int32_t> initialRegArgmaxOne;
            AscendC::Reg::LoadAlign(initialRegArgmaxOne, PatternSlotAddr(SLOT_PATTERN_ONE + 1));
            AscendC::Reg::RegTensor<int32_t> initialRegIndexTail;
            AscendC::Reg::LoadAlign(initialRegIndexTail, PatternSlotAddr(SLOT_PATTERN_TAIL));
            AscendC::Reg::RegTensor<int32_t> initialRegArgmaxTail;
            AscendC::Reg::LoadAlign(initialRegArgmaxTail, PatternSlotAddr(SLOT_PATTERN_TAIL + 1));
            AscendC::Reg::RegTensor<int32_t> initialRegIndexWBatch;
            AscendC::Reg::LoadAlign(initialRegIndexWBatch, PatternSlotAddr(SLOT_PATTERN_W_BATCH));
            AscendC::Reg::RegTensor<int32_t> initialRegArgmaxWBatch;
            AscendC::Reg::LoadAlign(initialRegArgmaxWBatch, PatternSlotAddr(SLOT_PATTERN_W_BATCH + 1));
            AscendC::Reg::RegTensor<int32_t> initialRegIndexOneTail;
            AscendC::Reg::LoadAlign(initialRegIndexOneTail, PatternSlotAddr(SLOT_PATTERN_ONE_TAIL));
            AscendC::Reg::RegTensor<int32_t> parallelRegIndex;
            AscendC::Reg::RegTensor<int32_t> parallelRegArgmax;
            AscendC::Reg::RegTensor<int32_t> cIncReg;
            AscendC::Reg::Arange(cIncReg, 0);
            AscendC::Reg::MaskReg allMaskU32 = AscendC::Reg::CreateMask<uint32_t, AscendC::Reg::MaskPattern::ALL>();
            for (uint16_t dProBatchIdx = 0; dProBatchIdx < dRemainTail; ++dProBatchIdx) {
                // --- H full ---
                for (uint16_t hProBatchIdx = 0; hProBatchIdx < hProBatchSize; ++hProBatchIdx) {
                    for (uint16_t wBatchIdx = 0; wBatchIdx < wProBatchSize; ++wBatchIdx) {
                        uint32_t offset = (wBatchIdx + hProBatchIdx * wArgmaxActual +
                                           dProBatchIdx * hArgmaxActual * wArgmaxActual +
                                           (dRemainBatchCount + dBlockConcurrentCount * dConcurrentCount) *
                                               dProBatchSize * hArgmaxActual * wArgmaxActual) *
                                              cStride +
                                          nGradOffset;
                        uint32_t argmaxOff = (wBatchIdx + hProBatchIdx * wArgmaxActual +
                                              dProBatchIdx * hArgmaxActual * wArgmaxActual +
                                              (dRemainBatchCount + dBlockConcurrentCount * dConcurrentCount) *
                                                  dProBatchSize * hArgmaxActual * wArgmaxActual) *
                                                 cActual +
                                             nArgmaxOffset;
                        AscendC::Reg::Adds(parallelRegIndex, initialRegIndexOne, static_cast<int32_t>(offset),
                                           allMaskU32);
                        AscendC::Reg::Adds(parallelRegArgmax, initialRegArgmaxOne, static_cast<int32_t>(argmaxOff),
                                           allMaskU32);
                        DoMulNCNdhwcFastDiv<T, INDEX_T, IS_CHECK_RANGE>(
                            (__local_mem__ computeType*)yAddr, (__local_mem__ T*)gradAddr,
                            (__local_mem__ INDEX_T*)argmaxAddr, parallelRegArgmax, parallelRegIndex, mask12, magicHWReg,
                            static_cast<int16_t>(shiftHW), magicWReg, static_cast<int16_t>(shiftW), dhOutputActual,
                            wOutputActual, wOutput, hwOutput, wInput, hwInput, baseOffset, cOutputAligned,
                            cOutputActual, divC, cIncReg, dLowerReg, hLowerReg, wLowerReg, dUpperReg, hUpperReg,
                            wUpperReg);
                    }
                    for (uint16_t wBatchIdx = 0; wBatchIdx < wRemainTail; ++wBatchIdx) {
                        uint32_t offset = (wBatchIdx + wProBatchSize * wFullBatchCount + hProBatchIdx * wArgmaxActual +
                                           dProBatchIdx * hArgmaxActual * wArgmaxActual +
                                           (dRemainBatchCount + dBlockConcurrentCount * dConcurrentCount) *
                                               dProBatchSize * hArgmaxActual * wArgmaxActual) *
                                              cStride +
                                          nGradOffset;
                        uint32_t argmaxOff = (wBatchIdx + wProBatchSize * wFullBatchCount +
                                              hProBatchIdx * wArgmaxActual +
                                              dProBatchIdx * hArgmaxActual * wArgmaxActual +
                                              (dRemainBatchCount + dBlockConcurrentCount * dConcurrentCount) *
                                                  dProBatchSize * hArgmaxActual * wArgmaxActual) *
                                                 cActual +
                                             nArgmaxOffset;
                        AscendC::Reg::Adds(parallelRegIndex, initialRegIndexTail, static_cast<int32_t>(offset),
                                           allMaskU32);
                        AscendC::Reg::Adds(parallelRegArgmax, initialRegArgmaxTail, static_cast<int32_t>(argmaxOff),
                                           allMaskU32);
                        DoMulNCNdhwcFastDiv<T, INDEX_T, IS_CHECK_RANGE>(
                            (__local_mem__ computeType*)yAddr, (__local_mem__ T*)gradAddr,
                            (__local_mem__ INDEX_T*)argmaxAddr, parallelRegArgmax, parallelRegIndex, mask13, magicHWReg,
                            static_cast<int16_t>(shiftHW), magicWReg, static_cast<int16_t>(shiftW), dhOutputActual,
                            wOutputActual, wOutput, hwOutput, wInput, hwInput, baseOffset, cOutputAligned,
                            cOutputActual, divC, cIncReg, dLowerReg, hLowerReg, wLowerReg, dUpperReg, hUpperReg,
                            wUpperReg);
                    }
                }
                // --- H 余 batch ---
                for (uint16_t hProBatchIdx = 0; hProBatchIdx < hRemainBatchCount; ++hProBatchIdx) {
                    for (uint16_t wBatchIdx = 0; wBatchIdx < wProBatchSize; ++wBatchIdx) {
                        uint32_t offset = (wBatchIdx +
                                           (hProBatchIdx + hFullBatchCount * hProBatchSize) * wArgmaxActual +
                                           dProBatchIdx * hArgmaxActual * wArgmaxActual +
                                           (dRemainBatchCount + dBlockConcurrentCount * dConcurrentCount) *
                                               dProBatchSize * hArgmaxActual * wArgmaxActual) *
                                              cStride +
                                          nGradOffset;
                        uint32_t argmaxOff = (wBatchIdx +
                                              (hProBatchIdx + hFullBatchCount * hProBatchSize) * wArgmaxActual +
                                              dProBatchIdx * hArgmaxActual * wArgmaxActual +
                                              (dRemainBatchCount + dBlockConcurrentCount * dConcurrentCount) *
                                                  dProBatchSize * hArgmaxActual * wArgmaxActual) *
                                                 cActual +
                                             nArgmaxOffset;
                        AscendC::Reg::Adds(parallelRegIndex, initialRegIndexOne, static_cast<int32_t>(offset),
                                           allMaskU32);
                        AscendC::Reg::Adds(parallelRegArgmax, initialRegArgmaxOne, static_cast<int32_t>(argmaxOff),
                                           allMaskU32);
                        DoMulNCNdhwcFastDiv<T, INDEX_T, IS_CHECK_RANGE>(
                            (__local_mem__ computeType*)yAddr, (__local_mem__ T*)gradAddr,
                            (__local_mem__ INDEX_T*)argmaxAddr, parallelRegArgmax, parallelRegIndex, mask14, magicHWReg,
                            static_cast<int16_t>(shiftHW), magicWReg, static_cast<int16_t>(shiftW), dhOutputActual,
                            wOutputActual, wOutput, hwOutput, wInput, hwInput, baseOffset, cOutputAligned,
                            cOutputActual, divC, cIncReg, dLowerReg, hLowerReg, wLowerReg, dUpperReg, hUpperReg,
                            wUpperReg);
                    }
                    for (uint16_t wBatchIdx = 0; wBatchIdx < wRemainTail; ++wBatchIdx) {
                        uint32_t offset = (wBatchIdx + wProBatchSize * wFullBatchCount +
                                           (hProBatchIdx + hFullBatchCount * hProBatchSize) * wArgmaxActual +
                                           dProBatchIdx * hArgmaxActual * wArgmaxActual +
                                           (dRemainBatchCount + dBlockConcurrentCount * dConcurrentCount) *
                                               dProBatchSize * hArgmaxActual * wArgmaxActual) *
                                              cStride +
                                          nGradOffset;
                        uint32_t argmaxOff = (wBatchIdx + wProBatchSize * wFullBatchCount +
                                              (hProBatchIdx + hFullBatchCount * hProBatchSize) * wArgmaxActual +
                                              dProBatchIdx * hArgmaxActual * wArgmaxActual +
                                              (dRemainBatchCount + dBlockConcurrentCount * dConcurrentCount) *
                                                  dProBatchSize * hArgmaxActual * wArgmaxActual) *
                                                 cActual +
                                             nArgmaxOffset;
                        AscendC::Reg::Adds(parallelRegIndex, initialRegIndexTail, static_cast<int32_t>(offset),
                                           allMaskU32);
                        AscendC::Reg::Adds(parallelRegArgmax, initialRegArgmaxTail, static_cast<int32_t>(argmaxOff),
                                           allMaskU32);
                        DoMulNCNdhwcFastDiv<T, INDEX_T, IS_CHECK_RANGE>(
                            (__local_mem__ computeType*)yAddr, (__local_mem__ T*)gradAddr,
                            (__local_mem__ INDEX_T*)argmaxAddr, parallelRegArgmax, parallelRegIndex, mask15, magicHWReg,
                            static_cast<int16_t>(shiftHW), magicWReg, static_cast<int16_t>(shiftW), dhOutputActual,
                            wOutputActual, wOutput, hwOutput, wInput, hwInput, baseOffset, cOutputAligned,
                            cOutputActual, divC, cIncReg, dLowerReg, hLowerReg, wLowerReg, dUpperReg, hUpperReg,
                            wUpperReg);
                    }
                }
                // --- H 余 tail ---
                for (uint16_t hProBatchIdx = 0; hProBatchIdx < hRemainTail; ++hProBatchIdx) {
                    for (uint16_t wBatchIdx = 0; wBatchIdx < wProBatchSize; ++wBatchIdx) {
                        uint32_t offset = (wBatchIdx +
                                           (hProBatchIdx + hRemainBatchCount * hProBatchSize +
                                            hFullBatchCount * hProBatchSize) *
                                               wArgmaxActual +
                                           dProBatchIdx * hArgmaxActual * wArgmaxActual +
                                           (dRemainBatchCount + dBlockConcurrentCount * dConcurrentCount) *
                                               dProBatchSize * hArgmaxActual * wArgmaxActual) *
                                              cStride +
                                          nGradOffset;
                        uint32_t argmaxOff = (wBatchIdx +
                                              (hProBatchIdx + hRemainBatchCount * hProBatchSize +
                                               hFullBatchCount * hProBatchSize) *
                                                  wArgmaxActual +
                                              dProBatchIdx * hArgmaxActual * wArgmaxActual +
                                              (dRemainBatchCount + dBlockConcurrentCount * dConcurrentCount) *
                                                  dProBatchSize * hArgmaxActual * wArgmaxActual) *
                                                 cActual +
                                             nArgmaxOffset;
                        AscendC::Reg::Adds(parallelRegIndex, initialRegIndexWBatch, static_cast<int32_t>(offset),
                                           allMaskU32);
                        AscendC::Reg::Adds(parallelRegArgmax, initialRegArgmaxWBatch, static_cast<int32_t>(argmaxOff),
                                           allMaskU32);
                        DoMulNCNdhwcFastDiv<T, INDEX_T, IS_CHECK_RANGE>(
                            (__local_mem__ computeType*)yAddr, (__local_mem__ T*)gradAddr,
                            (__local_mem__ INDEX_T*)argmaxAddr, parallelRegArgmax, parallelRegIndex, mask16, magicHWReg,
                            static_cast<int16_t>(shiftHW), magicWReg, static_cast<int16_t>(shiftW), dhOutputActual,
                            wOutputActual, wOutput, hwOutput, wInput, hwInput, baseOffset, cOutputAligned,
                            cOutputActual, divC, cIncReg, dLowerReg, hLowerReg, wLowerReg, dUpperReg, hUpperReg,
                            wUpperReg);
                    }
                    for (uint16_t wBatchIdx = 0; wBatchIdx < wRemainTail; ++wBatchIdx) {
                        uint32_t offset = (wBatchIdx + wProBatchSize * wFullBatchCount +
                                           (hProBatchIdx + hRemainBatchCount * hProBatchSize +
                                            hFullBatchCount * hProBatchSize) *
                                               wArgmaxActual +
                                           dProBatchIdx * hArgmaxActual * wArgmaxActual +
                                           (dRemainBatchCount + dBlockConcurrentCount * dConcurrentCount) *
                                               dProBatchSize * hArgmaxActual * wArgmaxActual) *
                                              cStride +
                                          nGradOffset;
                        uint32_t argmaxOff = (wBatchIdx + wProBatchSize * wFullBatchCount +
                                              (hProBatchIdx + hRemainBatchCount * hProBatchSize +
                                               hFullBatchCount * hProBatchSize) *
                                                  wArgmaxActual +
                                              dProBatchIdx * hArgmaxActual * wArgmaxActual +
                                              (dRemainBatchCount + dBlockConcurrentCount * dConcurrentCount) *
                                                  dProBatchSize * hArgmaxActual * wArgmaxActual) *
                                                 cActual +
                                             nArgmaxOffset;
                        AscendC::Reg::Adds(parallelRegIndex, initialRegIndexOneTail, static_cast<int32_t>(offset),
                                           allMaskU32);
                        AscendC::Reg::Adds(parallelRegArgmax, initialRegIndexOneTail, static_cast<int32_t>(argmaxOff),
                                           allMaskU32);
                        DoMulNCNdhwcFastDiv<T, INDEX_T, IS_CHECK_RANGE>(
                            (__local_mem__ computeType*)yAddr, (__local_mem__ T*)gradAddr,
                            (__local_mem__ INDEX_T*)argmaxAddr, parallelRegArgmax, parallelRegIndex, mask17, magicHWReg,
                            static_cast<int16_t>(shiftHW), magicWReg, static_cast<int16_t>(shiftW), dhOutputActual,
                            wOutputActual, wOutput, hwOutput, wInput, hwInput, baseOffset, cOutputAligned,
                            cOutputActual, divC, cIncReg, dLowerReg, hLowerReg, wLowerReg, dUpperReg, hUpperReg,
                            wUpperReg);
                    }
                }
            }
            AscendC::Reg::LocalMemBar<AscendC::Reg::MemType::VEC_STORE, AscendC::Reg::MemType::VEC_LOAD>();
        }
    }
}

} // namespace MaxPool3DGradNDHWCNameSpace

#endif // MAX_POOL3D_GRAD_NDHWC_IMPL_SCATTER_H

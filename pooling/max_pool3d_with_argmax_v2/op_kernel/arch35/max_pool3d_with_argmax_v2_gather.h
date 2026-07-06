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
 * \file max_pool3d_with_argmax_v2_gather_kernel.h
 * \brief
 */
#ifndef MAX_POOL_3D_WITH_ARGMAX_V2_GATHER_KERNEL_H_
#define MAX_POOL_3D_WITH_ARGMAX_V2_GATHER_KERNEL_H_

#include "max_pool3d_with_argmax_v2_base.h"
#include "max_pool3d_with_argmax_v2_tiling_struct.h"

namespace MaxPool3DWithArgmaxV2GatherNameSpace {
using namespace AscendC;

constexpr uint32_t BUFFER_NUM = 2;
constexpr int64_t HELP_BUFFER = 1024;
constexpr int64_t RATIO = 2;
static constexpr AscendC::MicroAPI::CastTrait castTrait = {MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::UNKNOWN,
                                                           MicroAPI::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};

template <typename T1, typename T2, const uint32_t IS_PAD = 1, typename UB_INDEXT = int16_t,
          typename CAL_INDEXT = int32_t>
class MaxPool3DWithArgmaxV2GatherKernel {
public:
    __aicore__ inline MaxPool3DWithArgmaxV2GatherKernel(
        TPipe& pipeIn, const MaxPool3DWithArgmaxV2Tiling::MaxPool3DWithArgmaxV2GatherTilingData& tilingData)
        : pipe_(pipeIn), tilingData_(tilingData){};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR argmax);
    __aicore__ inline void Process();

private:
    __aicore__ inline void ScalarCompute(int64_t loopNum);
    __aicore__ inline void ProcessEachLoop();
    __aicore__ inline void CopyIn();
    __aicore__ inline void Compute();
    __aicore__ inline void CopyOut();
    __aicore__ inline void DupBufferNegInf(__local_mem__ T1* dstAddr, uint32_t repeatElm, uint16_t loop, uint32_t tail);
    __aicore__ inline void CopyToCalcBuffer(__local_mem__ T1* dstAddr, __local_mem__ T1* srcAddr, uint16_t batch,
                                            uint16_t deps, uint16_t rows, uint16_t loopCols, uint16_t tailCols,
                                            uint32_t repeatElm, uint32_t srcBatchStride, uint32_t srcDepStride,
                                            uint32_t srcRowStride, uint32_t dstBatchStride, uint32_t dstDepStride,
                                            uint32_t dstRowStride, uint32_t dstDepOffset, uint32_t dstRowOffset,
                                            uint32_t dstColOffset);
    __aicore__ inline void DupAndCopyToCalcBuffer(__local_mem__ T1* dstAddr, __local_mem__ T1* srcAddr);
    __aicore__ inline void ComputeBaseIndex(MicroAPI::RegTensor<UB_INDEXT>& srcReg, uint32_t wStrideOffset,
                                            uint32_t hInputActualPad, int32_t ncInputOffset,
                                            MicroAPI::RegTensor<UB_INDEXT>& t1Reg,
                                            MicroAPI::RegTensor<UB_INDEXT>& divResultReg,
                                            MicroAPI::RegTensor<UB_INDEXT>& wIndexRegUnpack);
    __aicore__ inline void ConvertIndexWithoutPadAlign(MicroAPI::RegTensor<CAL_INDEXT>& t1Reg,
                                                       MicroAPI::RegTensor<CAL_INDEXT>& divResultReg,
                                                       MicroAPI::RegTensor<CAL_INDEXT>& wIndexRegUnpack,
                                                       CAL_INDEXT left, CAL_INDEXT wInput, CAL_INDEXT hIndexBase,
                                                       CAL_INDEXT hInput, CAL_INDEXT dIndexBase,
                                                       MicroAPI::RegTensor<CAL_INDEXT>& dstReg);

    __aicore__ inline void ProcessW(__local_mem__ T1* computeAddr, __local_mem__ T1* maxValueAddr, UB_INDEXT hOffset,
                                    uint32_t wStrideOffset, uint32_t hInputActualPad,
                                    MicroAPI::RegTensor<UB_INDEXT>& indexReg, uint16_t dKernel, uint16_t hKernel,
                                    uint16_t wKernel, uint16_t repeatElem, UB_INDEXT outputOffset,
                                    MicroAPI::RegTensor<UB_INDEXT>& maxIndexReg, uint16_t dDilation, uint16_t hDilation,
                                    uint16_t wDilation);

    __aicore__ inline void ConvertBaseIndexAndWriteArgmax(MicroAPI::RegTensor<UB_INDEXT>& maxIndexReg,
                                                          uint32_t wInputActualAlignedPad, uint32_t hInputActualPad,
                                                          UB_INDEXT ncInputOffset, CAL_INDEXT left, CAL_INDEXT wInput,
                                                          CAL_INDEXT hIndexBase, CAL_INDEXT hInput,
                                                          CAL_INDEXT dIndexBase, __local_mem__ T2*& argmaxAddrLocal,
                                                          const uint16_t* v2Part, uint16_t totalElem);
    __aicore__ inline void ConvertNcAndWriteArgmax(
        MicroAPI::RegTensor<UB_INDEXT>& maxIndexReg, MicroAPI::RegTensor<int32_t>& maxIndexConvertRegTep,
        MicroAPI::RegTensor<T2>& maxIndexConvertRegT2, MicroAPI::UnalignReg& u1, uint32_t wInputActualAlignedPad,
        CAL_INDEXT hInputActualPad, CAL_INDEXT left, CAL_INDEXT wInput, CAL_INDEXT hIndexBase, CAL_INDEXT hInput,
        CAL_INDEXT dIndexBase, CAL_INDEXT ncInputOffset, CAL_INDEXT num3D, CAL_INDEXT rate4D,
        __local_mem__ T2*& argmaxAddrLocal, const uint16_t* v2Part, uint16_t totalElem);

    __aicore__ inline void ComputeV2Parts(uint16_t* v2Part, uint16_t totalElem);
    __aicore__ inline void SingleRowGather(__local_mem__ T1* computeAddr, __local_mem__ T1* maxValueAddr,
                                           __local_mem__ T2* argmaxAddr);
    __aicore__ inline void MultiRowGather(__local_mem__ T1* computeAddr, __local_mem__ T1* maxValueAddr,
                                          __local_mem__ T2* argmaxAddr);
    __aicore__ inline void MultiDepGather(__local_mem__ T1* computeAddr, __local_mem__ T1* maxValueAddr,
                                          __local_mem__ T2* argmaxAddr);
    __aicore__ inline void MultiNcGather(__local_mem__ T1* computeAddr, __local_mem__ T1* maxValueAddr,
                                         __local_mem__ T2* argmaxAddr);

    __aicore__ inline void SingleRowGatherForInt64(__local_mem__ T1* computeAddr, __local_mem__ T1* maxValueAddr,
                                                   __local_mem__ T2* argmaxAddr);
    __aicore__ inline void MultiRowGatherForInt64(__local_mem__ T1* computeAddr, __local_mem__ T1* maxValueAddr,
                                                  __local_mem__ T2* argmaxAddr);
    __aicore__ inline void MultiDepGatherForInt64(__local_mem__ T1* computeAddr, __local_mem__ T1* maxValueAddr,
                                                  __local_mem__ T2* argmaxAddr);
    __aicore__ inline void MultiNcGatherForInt64(__local_mem__ T1* computeAddr, __local_mem__ T1* maxValueAddr,
                                                 __local_mem__ T2* argmaxAddr);

private:
    TPipe& pipe_;
    const MaxPool3DWithArgmaxV2Tiling::MaxPool3DWithArgmaxV2GatherTilingData& tilingData_;
    TQue<QuePosition::VECIN, BUFFER_NUM> inputQue_;
    TQue<QuePosition::VECOUT, BUFFER_NUM> maxValueQue_;
    TQue<QuePosition::VECOUT, BUFFER_NUM> argmaxQue_;
    TBuf<TPosition::VECCALC> inputCalcBuff_;

    GlobalTensor<T1> xGm_;
    GlobalTensor<T1> yGm_;
    GlobalTensor<T2> argmaxGm_;

    uint32_t blockIdx_ = 0;
    int64_t highAxisActual_ = 0;
    int64_t dOutputReal_ = 0;
    int64_t hOutputReal_ = 0;
    int64_t wOutputReal_ = 0;
    int64_t curCoreProcessNum_ = 0;
    int64_t dInputActualPad_ = 0;
    int64_t hInputActualPad_ = 0;
    int64_t wInputActualPad_ = 0;
    int64_t wInputActualAlignedPad_ = 0;
    int64_t frontOffsetToInputFront_ = 0;
    int64_t backOffsetToInputBack_ = 0;
    int64_t leftOffsetToInputLeft_ = 0;
    int64_t rightOffsetToInputRight_ = 0;
    int64_t topOffsetToInputTop_ = 0;
    int64_t downOffsetToInputDown_ = 0;

    int64_t highAxisIndex_ = 0;
    int64_t dAxisIndex_ = 0;
    int64_t hAxisIndex_ = 0;
    int64_t wAxisIndex_ = 0;

    int64_t highInputOffset_ = 0;
    int64_t dInputOffset_ = 0;
    int64_t hInputOffset_ = 0;
    int64_t wInputOffset_ = 0;

    int64_t dInputActualNoPad_ = 0;
    int64_t hInputActualNoPad_ = 0;
    int64_t wInputActualNoPad_ = 0;

    constexpr static int32_t blockSize_ = platform::GetUbBlockSize();

    constexpr static int64_t maxDataNumOneBlock_ = blockSize_ / sizeof(T1) >= blockSize_ / sizeof(T2) ?
                                                       blockSize_ / sizeof(T1) :
                                                       blockSize_ / sizeof(T2);
    constexpr static uint16_t vlT2_ = platform::GetVRegSize() / sizeof(T2);
    constexpr static uint16_t vlT1_ = platform::GetVRegSize() / sizeof(T1);
};

template <typename T1, typename T2, const uint32_t IS_PAD, typename UB_INDEXT, typename CAL_INDEXT>
__aicore__ inline void MaxPool3DWithArgmaxV2GatherKernel<T1, T2, IS_PAD, UB_INDEXT, CAL_INDEXT>::Init(GM_ADDR x,
                                                                                                      GM_ADDR y,
                                                                                                      GM_ADDR argmax)
{
    blockIdx_ = GetBlockIdx();
    if (blockIdx_ >= tilingData_.usedCoreNum) {
        return;
    }
    curCoreProcessNum_ = (blockIdx_ + 1 == tilingData_.usedCoreNum) ? tilingData_.tailCoreProcessNum :
                                                                      tilingData_.normalCoreProcessNum;
    xGm_.SetGlobalBuffer((__gm__ T1*)x);
    yGm_.SetGlobalBuffer((__gm__ T1*)y);
    argmaxGm_.SetGlobalBuffer((__gm__ T2*)argmax);

    pipe_.InitBuffer(inputQue_, BUFFER_NUM, tilingData_.inputBufferSize);
    if constexpr (IS_PAD == 1) {
        pipe_.InitBuffer(inputCalcBuff_, tilingData_.inputBufferSize);
    }
    pipe_.InitBuffer(maxValueQue_, BUFFER_NUM, tilingData_.maxValueBufferSize);
    pipe_.InitBuffer(argmaxQue_, BUFFER_NUM, tilingData_.argmaxBufferSize);
    return;
}

template <typename T1, typename T2, const uint32_t IS_PAD, typename UB_INDEXT, typename CAL_INDEXT>
__aicore__ inline void MaxPool3DWithArgmaxV2GatherKernel<T1, T2, IS_PAD, UB_INDEXT, CAL_INDEXT>::Process()
{
    if (blockIdx_ >= tilingData_.usedCoreNum) {
        return;
    }
    for (int64_t loopNum = 0; loopNum < curCoreProcessNum_; loopNum++) {
        ScalarCompute(loopNum);
        ProcessEachLoop();
    }
}

template <typename T1, typename T2, const uint32_t IS_PAD, typename UB_INDEXT, typename CAL_INDEXT>
__aicore__ inline void MaxPool3DWithArgmaxV2GatherKernel<T1, T2, IS_PAD, UB_INDEXT, CAL_INDEXT>::ScalarCompute(
    int64_t loopNum)
{
    int64_t d_h_w_outer = tilingData_.dOutputOuter * tilingData_.hOutputOuter * tilingData_.wOutputOuter;
    int64_t h_w_outer = tilingData_.hOutputOuter * tilingData_.wOutputOuter;
    int64_t w_outer = tilingData_.wOutputOuter;

    int64_t baseBlockIdx = blockIdx_ * tilingData_.normalCoreProcessNum + loopNum;

    highAxisIndex_ = baseBlockIdx / d_h_w_outer;
    highAxisActual_ = highAxisIndex_ == (tilingData_.highAxisOuter - 1) ? tilingData_.highAxisTail :
                                                                          tilingData_.highAxisInner;

    int64_t base_mod_dhw = baseBlockIdx - d_h_w_outer * highAxisIndex_;
    dAxisIndex_ = base_mod_dhw / h_w_outer;
    dOutputReal_ = dAxisIndex_ == (tilingData_.dOutputOuter - 1) ? tilingData_.dOutputTail : tilingData_.dOutputInner;

    int64_t tempTail = baseBlockIdx - h_w_outer * (baseBlockIdx / h_w_outer);
    hAxisIndex_ = tempTail / w_outer;
    hOutputReal_ = hAxisIndex_ == (tilingData_.hOutputOuter - 1) ? tilingData_.hOutputTail : tilingData_.hOutputInner;

    wAxisIndex_ = tempTail - w_outer * (tempTail / w_outer);
    wOutputReal_ = wAxisIndex_ == (tilingData_.wOutputOuter - 1) ? tilingData_.wOutputTail : tilingData_.wOutputInner;

    dInputActualPad_ = (dOutputReal_ - 1) * tilingData_.dStride + (tilingData_.dKernel - 1) * tilingData_.dDilation + 1;
    hInputActualPad_ = (hOutputReal_ - 1) * tilingData_.hStride + (tilingData_.hKernel - 1) * tilingData_.hDilation + 1;
    wInputActualPad_ = (wOutputReal_ - 1) * tilingData_.wStride + (tilingData_.wKernel - 1) * tilingData_.wDilation + 1;

    wInputActualAlignedPad_ = CeilDivision(wInputActualPad_, blockSize_ / sizeof(T1)) * (blockSize_ / sizeof(T1));
    int64_t hw = tilingData_.hInput * tilingData_.wInput;
    int64_t inputPlaneSize = tilingData_.dInput * hw;
    highInputOffset_ = highAxisIndex_ * tilingData_.highAxisInner * inputPlaneSize;
    dInputOffset_ = dAxisIndex_ * tilingData_.dOutputInner * tilingData_.dStride * hw;
    hInputOffset_ = hAxisIndex_ * tilingData_.hOutputInner * tilingData_.hStride * tilingData_.wInput;
    wInputOffset_ = wAxisIndex_ * tilingData_.wOutputInner * tilingData_.wStride;

    if constexpr (IS_PAD == 1) {
        int64_t hBase = hAxisIndex_ * tilingData_.hOutputInner * tilingData_.hStride;
        int64_t wBase = wAxisIndex_ * tilingData_.wOutputInner * tilingData_.wStride;
        int64_t dBase = dAxisIndex_ * tilingData_.dOutputInner * tilingData_.dStride;

        int64_t tRelBoundDistance = hBase - tilingData_.padTop;
        int64_t dRelBoundDistance = hBase + (hOutputReal_ - 1) * tilingData_.hStride + tilingData_.hKernel -
                                    tilingData_.hInput - tilingData_.padTop;
        int64_t lRelBoundDistance = wBase - tilingData_.padLeft;
        int64_t rRelBoundDistance = wBase + (wOutputReal_ - 1) * tilingData_.wStride + tilingData_.wKernel -
                                    tilingData_.wInput - tilingData_.padLeft;
        int64_t frontRelBoundDistance = dBase - tilingData_.padFront;
        int64_t backRelBoundDistance = dBase + (dOutputReal_ - 1) * tilingData_.dStride + tilingData_.dKernel -
                                       tilingData_.dInput - tilingData_.padFront;
        leftOffsetToInputLeft_ = lRelBoundDistance >= 0 ? 0 : -lRelBoundDistance;
        rightOffsetToInputRight_ = rRelBoundDistance >= 0 ? rRelBoundDistance : 0;
        topOffsetToInputTop_ = tRelBoundDistance >= 0 ? 0 : -tRelBoundDistance;
        downOffsetToInputDown_ = dRelBoundDistance >= 0 ? dRelBoundDistance : 0;

        frontOffsetToInputFront_ = frontRelBoundDistance >= 0 ? 0 : -frontRelBoundDistance;
        backOffsetToInputBack_ = backRelBoundDistance >= 0 ? backRelBoundDistance : 0;

        dInputActualNoPad_ = dInputActualPad_ - frontOffsetToInputFront_ - backOffsetToInputBack_;
        hInputActualNoPad_ = hInputActualPad_ - topOffsetToInputTop_ - downOffsetToInputDown_;
        wInputActualNoPad_ = wInputActualPad_ - leftOffsetToInputLeft_ - rightOffsetToInputRight_;

        dInputOffset_ = frontOffsetToInputFront_ == 0 ? dInputOffset_ - tilingData_.padFront * hw : 0;
        hInputOffset_ = topOffsetToInputTop_ == 0 ? hInputOffset_ - tilingData_.padTop * tilingData_.wInput : 0;
        wInputOffset_ = leftOffsetToInputLeft_ == 0 ? wInputOffset_ - tilingData_.padLeft : 0;
    }
}

template <typename T1, typename T2, const uint32_t IS_PAD, typename UB_INDEXT, typename CAL_INDEXT>
__aicore__ inline void MaxPool3DWithArgmaxV2GatherKernel<T1, T2, IS_PAD, UB_INDEXT, CAL_INDEXT>::ProcessEachLoop()
{
    CopyIn();
    Compute();
    CopyOut();
}

template <typename T1, typename T2, const uint32_t IS_PAD, typename UB_INDEXT, typename CAL_INDEXT>
__aicore__ inline void MaxPool3DWithArgmaxV2GatherKernel<T1, T2, IS_PAD, UB_INDEXT, CAL_INDEXT>::CopyIn()
{
    LocalTensor<T1> xLocal = inputQue_.AllocTensor<T1>();
    int64_t xGmOffset = highInputOffset_ + dInputOffset_ + hInputOffset_ + wInputOffset_;

    LoopModeParams loopModeParamsT1;
    if constexpr (IS_PAD == 1) {
        int64_t wInputActualAlignedNoPad = CeilDivision(wInputActualNoPad_, blockSize_ / sizeof(T1)) *
                                           (blockSize_ / sizeof(T1));
        loopModeParamsT1.loop1Size = highAxisActual_;
        loopModeParamsT1.loop2Size = dInputActualNoPad_;
        loopModeParamsT1.loop1SrcStride = tilingData_.dInput * tilingData_.hInput * tilingData_.wInput * sizeof(T1);
        loopModeParamsT1.loop2SrcStride = tilingData_.hInput * tilingData_.wInput * sizeof(T1);
        loopModeParamsT1.loop1DstStride = dInputActualNoPad_ * hInputActualNoPad_ * wInputActualAlignedNoPad *
                                          sizeof(T1);
        loopModeParamsT1.loop2DstStride = hInputActualNoPad_ * wInputActualAlignedNoPad * sizeof(T1);
    } else {
        loopModeParamsT1.loop1Size = highAxisActual_;
        loopModeParamsT1.loop2Size = dInputActualPad_;
        loopModeParamsT1.loop1SrcStride = tilingData_.dInput * tilingData_.hInput * tilingData_.wInput * sizeof(T1);
        loopModeParamsT1.loop2SrcStride = tilingData_.hInput * tilingData_.wInput * sizeof(T1);
        loopModeParamsT1.loop1DstStride = dInputActualPad_ * hInputActualPad_ * wInputActualAlignedPad_ * sizeof(T1);
        loopModeParamsT1.loop2DstStride = hInputActualPad_ * wInputActualAlignedPad_ * sizeof(T1);
    }

    SetLoopModePara(loopModeParamsT1, DataCopyMVType::OUT_TO_UB);
    DataCopyPadExtParams<T1> paramsT1 = {false, 0, 0, 0};
    DataCopyExtParams copyOutParamT1;
    if constexpr (IS_PAD == 1) {
        copyOutParamT1.blockCount = static_cast<uint16_t>(hInputActualNoPad_);
        copyOutParamT1.blockLen = static_cast<uint32_t>(wInputActualNoPad_ * sizeof(T1));
        copyOutParamT1.srcStride = static_cast<uint32_t>((tilingData_.wInput - wInputActualNoPad_) * sizeof(T1));
        copyOutParamT1.dstStride = 0;
        copyOutParamT1.rsv = 0;
    } else {
        copyOutParamT1.blockCount = static_cast<uint16_t>(hInputActualPad_);
        copyOutParamT1.blockLen = static_cast<uint32_t>(wInputActualPad_ * sizeof(T1));
        copyOutParamT1.srcStride = static_cast<uint32_t>((tilingData_.wInput - wInputActualPad_) * sizeof(T1));
        copyOutParamT1.dstStride = 0;
        copyOutParamT1.rsv = 0;
    }
    DataCopyPad(xLocal, xGm_[xGmOffset], copyOutParamT1, paramsT1);
    inputQue_.EnQue(xLocal);
    ResetLoopModePara(DataCopyMVType::OUT_TO_UB);
}

template <typename T1, typename T2, const uint32_t IS_PAD, typename UB_INDEXT, typename CAL_INDEXT>
__aicore__ inline void MaxPool3DWithArgmaxV2GatherKernel<T1, T2, IS_PAD, UB_INDEXT, CAL_INDEXT>::DupBufferNegInf(
    __local_mem__ T1* dstAddr, uint32_t repeatElm, uint16_t loop, uint32_t tail)
{
    MicroAPI::RegTensor<T1> v0;
    SetNegInfReg<T1>(v0);
    MicroAPI::MaskReg preg = MicroAPI::CreateMask<T1, MicroAPI::MaskPattern::ALL>();
    uint32_t maskCount = tail;
    for (uint16_t i = 0; i < loop; i++) {
        MicroAPI::DataCopy<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(dstAddr, v0, repeatElm, preg);
    }
    preg = MicroAPI::UpdateMask<T1>(maskCount);
    MicroAPI::DataCopy<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(dstAddr, v0, repeatElm, preg);
}

template <typename T1, typename T2, const uint32_t IS_PAD, typename UB_INDEXT, typename CAL_INDEXT>
__aicore__ inline void MaxPool3DWithArgmaxV2GatherKernel<T1, T2, IS_PAD, UB_INDEXT, CAL_INDEXT>::CopyToCalcBuffer(
    __local_mem__ T1* dstAddr, __local_mem__ T1* srcAddr, uint16_t batch, uint16_t deps, uint16_t rows,
    uint16_t loopCols, uint16_t tailCols, uint32_t repeatElm, uint32_t srcBatchStride, uint32_t srcDepStride,
    uint32_t srcRowStride, uint32_t dstBatchStride, uint32_t dstDepStride, uint32_t dstRowStride, uint32_t dstDepOffset,
    uint32_t dstRowOffset, uint32_t dstColOffset)
{
    MicroAPI::RegTensor<T1> v0;
    MicroAPI::UnalignReg u0;

    uint32_t dstFixedOffset = dstDepOffset * dstDepStride + dstRowOffset * dstRowStride + dstColOffset;
    for (uint16_t i = 0; i < batch; i++) {
        __local_mem__ T1* batchSrcBase = srcAddr + i * srcBatchStride;
        __local_mem__ T1* batchDstBase = dstAddr + i * dstBatchStride + dstFixedOffset;

        for (uint16_t t = 0; t < deps; t++) {
            __local_mem__ T1* depSrcBase = batchSrcBase + t * srcDepStride;
            __local_mem__ T1* depDstBase = batchDstBase + t * dstDepStride;

            __local_mem__ T1* rowSrcBase = depSrcBase;
            __local_mem__ T1* rowDstBase = depDstBase;

            for (uint16_t j = 0; j < rows; j++) {
                __local_mem__ T1* curSrcAddr = rowSrcBase;
                __local_mem__ T1* curDstAddr = rowDstBase;

                for (uint16_t k = 0; k < loopCols; k++) {
                    MicroAPI::DataCopy<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(v0, curSrcAddr, repeatElm);
                    MicroAPI::DataCopyUnAlign(curDstAddr, v0, u0, repeatElm);
                }
                MicroAPI::DataCopy<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(v0, curSrcAddr, 0);
                MicroAPI::DataCopyUnAlign(curDstAddr, v0, u0, tailCols);
                MicroAPI::DataCopyUnAlignPost(curDstAddr, u0, 0);
                rowSrcBase += srcRowStride;
                rowDstBase += dstRowStride;
            }
        }
    }
}

template <typename T1, typename T2, const uint32_t IS_PAD, typename UB_INDEXT, typename CAL_INDEXT>
__aicore__ inline void MaxPool3DWithArgmaxV2GatherKernel<T1, T2, IS_PAD, UB_INDEXT, CAL_INDEXT>::DupAndCopyToCalcBuffer(
    __local_mem__ T1* dstAddr, __local_mem__ T1* srcAddr)
{
    uint16_t loopCols = wInputActualNoPad_ / vlT1_;
    uint16_t tailCols = wInputActualNoPad_ - loopCols * vlT1_;
    uint32_t wInputActualNoPadAlign = CeilDivision(wInputActualNoPad_, blockSize_ / sizeof(T1)) * blockSize_ /
                                      sizeof(T1);
    uint32_t totalInputPad = tilingData_.highAxisInner * dInputActualPad_ * hInputActualPad_ * wInputActualAlignedPad_;
    uint16_t loopDup = totalInputPad / vlT1_;
    uint32_t tailDup = totalInputPad - loopDup * vlT1_;
    uint32_t dstDepOffset = frontOffsetToInputFront_;
    uint32_t dstRowOffset = topOffsetToInputTop_;
    uint32_t dstColOffset = leftOffsetToInputLeft_;
    uint32_t srcDepStride = hInputActualNoPad_ * wInputActualNoPadAlign;
    uint32_t srcBatchStride = dInputActualNoPad_ * hInputActualNoPad_ * wInputActualNoPadAlign;
    uint32_t dstDepStride = hInputActualPad_ * wInputActualAlignedPad_;
    uint32_t dstBatchStride = dInputActualPad_ * hInputActualPad_ * wInputActualAlignedPad_;
    __VEC_SCOPE__
    {
        DupBufferNegInf(dstAddr, vlT1_, loopDup, tailDup);
        CopyToCalcBuffer(dstAddr, srcAddr, highAxisActual_, dInputActualNoPad_, hInputActualNoPad_, loopCols, tailCols,
                         vlT1_, srcBatchStride, srcDepStride, wInputActualNoPadAlign, dstBatchStride, dstDepStride,
                         wInputActualAlignedPad_, dstDepOffset, dstRowOffset, dstColOffset);
    }
    return;
}

template <typename T1, typename T2, const uint32_t IS_PAD, typename UB_INDEXT, typename CAL_INDEXT>
__aicore__ inline void MaxPool3DWithArgmaxV2GatherKernel<T1, T2, IS_PAD, UB_INDEXT, CAL_INDEXT>::Compute()
{
    LocalTensor<T1> inputLocal = inputQue_.DeQue<T1>();
    LocalTensor<T1> caclBuffLocal;
    __local_mem__ T1* inputBuffAddr;
    __local_mem__ T1* inputQueAddr = (__local_mem__ T1*)inputLocal.GetPhyAddr();
    __local_mem__ T1* computeAddr = inputQueAddr;
    if constexpr (IS_PAD == 1) {
        caclBuffLocal = inputCalcBuff_.Get<T1>();
        inputBuffAddr = (__local_mem__ T1*)caclBuffLocal.GetPhyAddr();
        DupAndCopyToCalcBuffer(inputBuffAddr, inputQueAddr);
        computeAddr = inputBuffAddr;
    }
    LocalTensor<T1> maxValueLocal = maxValueQue_.AllocTensor<T1>();
    LocalTensor<T2> argmaxLocal = argmaxQue_.AllocTensor<T2>();
    __local_mem__ T1* maxValueAddr = (__local_mem__ T1*)maxValueLocal.GetPhyAddr();
    __local_mem__ T2* argmaxAddr = (__local_mem__ T2*)argmaxLocal.GetPhyAddr();

    if (wOutputReal_ * RATIO > vlT1_) {
        if constexpr (std::is_same<CAL_INDEXT, int64_t>::value) {
            SingleRowGatherForInt64(computeAddr, maxValueAddr, argmaxAddr);
        } else {
            SingleRowGather(computeAddr, maxValueAddr, argmaxAddr);
        }
    } else if (hOutputReal_ * wOutputReal_ * RATIO > vlT1_) {
        if constexpr (std::is_same<CAL_INDEXT, int64_t>::value) {
            MultiRowGatherForInt64(computeAddr, maxValueAddr, argmaxAddr);
        } else {
            MultiRowGather(computeAddr, maxValueAddr, argmaxAddr);
        }
    } else if (dOutputReal_ * hOutputReal_ * wOutputReal_ * RATIO > vlT1_) {
        if constexpr (std::is_same<CAL_INDEXT, int64_t>::value) {
            MultiDepGatherForInt64(computeAddr, maxValueAddr, argmaxAddr);
        } else {
            MultiDepGather(computeAddr, maxValueAddr, argmaxAddr);
        }
    } else {
        if constexpr (std::is_same<CAL_INDEXT, int64_t>::value) {
            MultiNcGatherForInt64(computeAddr, maxValueAddr, argmaxAddr);
        } else {
            MultiNcGather(computeAddr, maxValueAddr, argmaxAddr);
        }
    }

    inputQue_.FreeTensor(inputLocal);
    maxValueQue_.EnQue(maxValueLocal);
    argmaxQue_.EnQue(argmaxLocal);
    return;
}

template <typename T1, typename T2, const uint32_t IS_PAD, typename UB_INDEXT, typename CAL_INDEXT>
__aicore__ inline void MaxPool3DWithArgmaxV2GatherKernel<T1, T2, IS_PAD, UB_INDEXT, CAL_INDEXT>::ComputeBaseIndex(
    MicroAPI::RegTensor<UB_INDEXT>& srcReg, uint32_t wStrideOffset, uint32_t hInputActualPad, int32_t ncInputOffset,
    MicroAPI::RegTensor<UB_INDEXT>& t1Reg, MicroAPI::RegTensor<UB_INDEXT>& divResultReg,
    MicroAPI::RegTensor<UB_INDEXT>& wIndexRegUnpack)
{
    MicroAPI::RegTensor<UB_INDEXT> constReg;
    MicroAPI::RegTensor<UB_INDEXT> t3Reg;
    MicroAPI::RegTensor<UB_INDEXT> t2Reg;
    MicroAPI::MaskReg allMask = MicroAPI::CreateMask<UB_INDEXT, MicroAPI::MaskPattern::ALL>();
    MicroAPI::MaskReg geMask;

    MicroAPI::Duplicate(constReg, static_cast<UB_INDEXT>(wStrideOffset));
    MicroAPI::Duplicate(t3Reg, static_cast<UB_INDEXT>(wStrideOffset * hInputActualPad));
    MicroAPI::Adds(srcReg, srcReg, -ncInputOffset, allMask);
    MicroAPI::Div(t1Reg, srcReg, t3Reg, allMask);
    MicroAPI::Muls(t2Reg, t1Reg, static_cast<UB_INDEXT>(-1 * wStrideOffset * hInputActualPad), allMask);
    MicroAPI::Add(t2Reg, srcReg, t2Reg, allMask);

    MicroAPI::Compare<UB_INDEXT, CMPMODE::GE>(geMask, t2Reg, t3Reg, allMask);
    MicroAPI::Sub(t3Reg, t2Reg, t3Reg, allMask);
    MicroAPI::Select(t2Reg, t3Reg, t2Reg, geMask);
    MicroAPI::Adds(t3Reg, t1Reg, static_cast<UB_INDEXT>(1), allMask);
    MicroAPI::Select(t1Reg, t3Reg, t1Reg, geMask);

    MicroAPI::Div(divResultReg, t2Reg, constReg, allMask);
    MicroAPI::Mul(t3Reg, divResultReg, constReg, allMask);
    MicroAPI::Sub(wIndexRegUnpack, t2Reg, t3Reg, allMask);

    MicroAPI::Compare<UB_INDEXT, CMPMODE::GE>(geMask, wIndexRegUnpack, constReg, allMask);
    MicroAPI::Sub(t3Reg, wIndexRegUnpack, constReg, allMask);
    MicroAPI::Select(wIndexRegUnpack, t3Reg, wIndexRegUnpack, geMask);
    MicroAPI::Adds(t3Reg, divResultReg, static_cast<UB_INDEXT>(1), allMask);
    MicroAPI::Select(divResultReg, t3Reg, divResultReg, geMask);
}

template <typename T1, typename T2, const uint32_t IS_PAD, typename UB_INDEXT, typename CAL_INDEXT>
__aicore__ inline void
MaxPool3DWithArgmaxV2GatherKernel<T1, T2, IS_PAD, UB_INDEXT, CAL_INDEXT>::ConvertIndexWithoutPadAlign(
    MicroAPI::RegTensor<CAL_INDEXT>& t1Reg, MicroAPI::RegTensor<CAL_INDEXT>& divResultReg,
    MicroAPI::RegTensor<CAL_INDEXT>& wIndexRegUnpack, CAL_INDEXT left, CAL_INDEXT wInput, CAL_INDEXT hIndexBase,
    CAL_INDEXT hInput, CAL_INDEXT dIndexBase, MicroAPI::RegTensor<CAL_INDEXT>& dstReg)
{
    MicroAPI::RegTensor<CAL_INDEXT> hIndexReg;
    MicroAPI::RegTensor<CAL_INDEXT> dIndexReg;
    MicroAPI::RegTensor<CAL_INDEXT> wIndexReg;
    MicroAPI::RegTensor<CAL_INDEXT> zeroReg;
    MicroAPI::MaskReg negInfMask;
    MicroAPI::MaskReg allMaskB32 = MicroAPI::CreateMask<CAL_INDEXT, MicroAPI::MaskPattern::ALL>();

    MicroAPI::Duplicate(zeroReg, static_cast<CAL_INDEXT>(0));

    MicroAPI::Adds(dIndexReg, t1Reg, dIndexBase, allMaskB32);
    MicroAPI::Adds(hIndexReg, divResultReg, hIndexBase, allMaskB32);
    MicroAPI::Adds(wIndexReg, wIndexRegUnpack, left, allMaskB32);
    if constexpr (IS_PAD == 1) {
        MicroAPI::Compare<CAL_INDEXT, CMPMODE::LT>(negInfMask, dIndexReg, zeroReg, allMaskB32);
        MicroAPI::Select(dIndexReg, zeroReg, dIndexReg, negInfMask);
        MicroAPI::Compare<CAL_INDEXT, CMPMODE::LT>(negInfMask, hIndexReg, zeroReg, allMaskB32);
        MicroAPI::Select(hIndexReg, zeroReg, hIndexReg, negInfMask);
        MicroAPI::Compare<CAL_INDEXT, CMPMODE::LT>(negInfMask, wIndexReg, zeroReg, allMaskB32);
        MicroAPI::Select(wIndexReg, zeroReg, wIndexReg, negInfMask);
    }
    MicroAPI::Muls(dIndexReg, dIndexReg, (wInput * hInput), allMaskB32);
    MicroAPI::Muls(hIndexReg, hIndexReg, wInput, allMaskB32);
    MicroAPI::Add(dstReg, hIndexReg, dIndexReg, allMaskB32);
    MicroAPI::Add(dstReg, dstReg, wIndexReg, allMaskB32);
}

template <typename T1, typename T2, const uint32_t IS_PAD, typename UB_INDEXT, typename CAL_INDEXT>
__aicore__ inline void
MaxPool3DWithArgmaxV2GatherKernel<T1, T2, IS_PAD, UB_INDEXT, CAL_INDEXT>::ConvertBaseIndexAndWriteArgmax(
    MicroAPI::RegTensor<UB_INDEXT>& maxIndexReg, uint32_t wInputActualAlignedPad, uint32_t hInputActualPad,
    UB_INDEXT ncInputOffset, CAL_INDEXT left, CAL_INDEXT wInput, CAL_INDEXT hIndexBase, CAL_INDEXT hInput,
    CAL_INDEXT dIndexBase, __local_mem__ T2*& argmaxAddrLocal, const uint16_t* v2Part, uint16_t totalElem)
{
    if constexpr (sizeof(UB_INDEXT) == sizeof(CAL_INDEXT)) {
        MicroAPI::RegTensor<UB_INDEXT> t1Reg, divResultReg, wIndexRegUnpack;
        MicroAPI::RegTensor<CAL_INDEXT> maxIndexConvertReg;
        MicroAPI::UnalignReg u1;

        ComputeBaseIndex(maxIndexReg, wInputActualAlignedPad, hInputActualPad, ncInputOffset, t1Reg, divResultReg,
                         wIndexRegUnpack);
        ConvertIndexWithoutPadAlign(t1Reg, divResultReg, wIndexRegUnpack, left, wInput, hIndexBase, hInput, dIndexBase,
                                    maxIndexConvertReg);
        if constexpr (sizeof(CAL_INDEXT) == sizeof(T2)) {
            MicroAPI::DataCopyUnAlign(argmaxAddrLocal, maxIndexConvertReg, u1, totalElem);
            MicroAPI::DataCopyUnAlignPost(argmaxAddrLocal, u1, 0);
        } else if constexpr (sizeof(CAL_INDEXT) * 2 == sizeof(T2)) {
            MicroAPI::RegTensor<T2> maxIndexConvertRegT2;
            AscendC::MicroAPI::UnPack<T2, CAL_INDEXT, AscendC::MicroAPI::HighLowPart::LOWEST>(maxIndexConvertRegT2,
                                                                                              maxIndexConvertReg);
            MicroAPI::DataCopyUnAlign(argmaxAddrLocal, maxIndexConvertRegT2, u1, v2Part[0]);
            MicroAPI::DataCopyUnAlignPost(argmaxAddrLocal, u1, 0);
            AscendC::MicroAPI::UnPack<T2, CAL_INDEXT, AscendC::MicroAPI::HighLowPart::HIGHEST>(maxIndexConvertRegT2,
                                                                                               maxIndexConvertReg);
            MicroAPI::DataCopyUnAlign(argmaxAddrLocal, maxIndexConvertRegT2, u1, v2Part[1]);
            MicroAPI::DataCopyUnAlignPost(argmaxAddrLocal, u1, 0);
        } else {
            MicroAPI::RegTensor<int32_t> maxIndexConvertRegTep;
            MicroAPI::RegTensor<T2> maxIndexConvertRegT2;
            AscendC::MicroAPI::UnPack<int32_t, CAL_INDEXT, AscendC::MicroAPI::HighLowPart::LOWEST>(
                maxIndexConvertRegTep, maxIndexConvertReg);
            AscendC::MicroAPI::UnPack<T2, int32_t, AscendC::MicroAPI::HighLowPart::LOWEST>(maxIndexConvertRegT2,
                                                                                           maxIndexConvertRegTep);
            MicroAPI::DataCopyUnAlign(argmaxAddrLocal, maxIndexConvertRegT2, u1, v2Part[0]);
            MicroAPI::DataCopyUnAlignPost(argmaxAddrLocal, u1, 0);
            AscendC::MicroAPI::UnPack<T2, int32_t, AscendC::MicroAPI::HighLowPart::HIGHEST>(maxIndexConvertRegT2,
                                                                                            maxIndexConvertRegTep);
            MicroAPI::DataCopyUnAlign(argmaxAddrLocal, maxIndexConvertRegT2, u1, v2Part[1]);
            MicroAPI::DataCopyUnAlignPost(argmaxAddrLocal, u1, 0);
            AscendC::MicroAPI::UnPack<int32_t, CAL_INDEXT, AscendC::MicroAPI::HighLowPart::HIGHEST>(
                maxIndexConvertRegTep, maxIndexConvertReg);
            AscendC::MicroAPI::UnPack<T2, int32_t, AscendC::MicroAPI::HighLowPart::LOWEST>(maxIndexConvertRegT2,
                                                                                           maxIndexConvertRegTep);
            MicroAPI::DataCopyUnAlign(argmaxAddrLocal, maxIndexConvertRegT2, u1, v2Part[2]);
            MicroAPI::DataCopyUnAlignPost(argmaxAddrLocal, u1, 0);
            AscendC::MicroAPI::UnPack<T2, int32_t, AscendC::MicroAPI::HighLowPart::HIGHEST>(maxIndexConvertRegT2,
                                                                                            maxIndexConvertRegTep);
            MicroAPI::DataCopyUnAlign(argmaxAddrLocal, maxIndexConvertRegT2, u1, v2Part[3]);
            MicroAPI::DataCopyUnAlignPost(argmaxAddrLocal, u1, 0);
        }
    } else if constexpr (sizeof(UB_INDEXT) * 2 == sizeof(CAL_INDEXT)) {
        MicroAPI::RegTensor<UB_INDEXT> t1Reg, divResultReg, wIndexRegUnpack;
        MicroAPI::RegTensor<CAL_INDEXT> t1RegCal, divResultRegCal, wIndexRegUnpackCal;
        MicroAPI::RegTensor<CAL_INDEXT> maxIndexConvertReg;
        MicroAPI::UnalignReg u1;

        ComputeBaseIndex(maxIndexReg, wInputActualAlignedPad, hInputActualPad, ncInputOffset, t1Reg, divResultReg,
                         wIndexRegUnpack);

        AscendC::MicroAPI::UnPack<CAL_INDEXT, UB_INDEXT, AscendC::MicroAPI::HighLowPart::LOWEST>(t1RegCal, t1Reg);
        AscendC::MicroAPI::UnPack<CAL_INDEXT, UB_INDEXT, AscendC::MicroAPI::HighLowPart::LOWEST>(divResultRegCal,
                                                                                                 divResultReg);
        AscendC::MicroAPI::UnPack<CAL_INDEXT, UB_INDEXT, AscendC::MicroAPI::HighLowPart::LOWEST>(wIndexRegUnpackCal,
                                                                                                 wIndexRegUnpack);
        ConvertIndexWithoutPadAlign(t1RegCal, divResultRegCal, wIndexRegUnpackCal, left, wInput, hIndexBase, hInput,
                                    dIndexBase, maxIndexConvertReg);
        if constexpr (sizeof(CAL_INDEXT) == sizeof(T2)) {
            MicroAPI::DataCopyUnAlign(argmaxAddrLocal, maxIndexConvertReg, u1, v2Part[0]);
            MicroAPI::DataCopyUnAlignPost(argmaxAddrLocal, u1, 0);
        } else if constexpr (sizeof(CAL_INDEXT) * 2 == sizeof(T2)) {
            MicroAPI::RegTensor<T2> maxIndexConvertRegT2;
            AscendC::MicroAPI::UnPack<T2, CAL_INDEXT, AscendC::MicroAPI::HighLowPart::LOWEST>(maxIndexConvertRegT2,
                                                                                              maxIndexConvertReg);
            MicroAPI::DataCopyUnAlign(argmaxAddrLocal, maxIndexConvertRegT2, u1, v2Part[0]);
            MicroAPI::DataCopyUnAlignPost(argmaxAddrLocal, u1, 0);
            AscendC::MicroAPI::UnPack<T2, CAL_INDEXT, AscendC::MicroAPI::HighLowPart::HIGHEST>(maxIndexConvertRegT2,
                                                                                               maxIndexConvertReg);
            MicroAPI::DataCopyUnAlign(argmaxAddrLocal, maxIndexConvertRegT2, u1, v2Part[1]);
            MicroAPI::DataCopyUnAlignPost(argmaxAddrLocal, u1, 0);
        }

        AscendC::MicroAPI::UnPack<CAL_INDEXT, UB_INDEXT, AscendC::MicroAPI::HighLowPart::HIGHEST>(t1RegCal, t1Reg);
        AscendC::MicroAPI::UnPack<CAL_INDEXT, UB_INDEXT, AscendC::MicroAPI::HighLowPart::HIGHEST>(divResultRegCal,
                                                                                                  divResultReg);
        AscendC::MicroAPI::UnPack<CAL_INDEXT, UB_INDEXT, AscendC::MicroAPI::HighLowPart::HIGHEST>(wIndexRegUnpackCal,
                                                                                                  wIndexRegUnpack);
        ConvertIndexWithoutPadAlign(t1RegCal, divResultRegCal, wIndexRegUnpackCal, left, wInput, hIndexBase, hInput,
                                    dIndexBase, maxIndexConvertReg);
        if constexpr (sizeof(CAL_INDEXT) == sizeof(T2)) {
            MicroAPI::DataCopyUnAlign(argmaxAddrLocal, maxIndexConvertReg, u1, v2Part[1]);
            MicroAPI::DataCopyUnAlignPost(argmaxAddrLocal, u1, 0);
        } else if constexpr (sizeof(CAL_INDEXT) * 2 == sizeof(T2)) {
            MicroAPI::RegTensor<T2> maxIndexConvertRegT2;
            AscendC::MicroAPI::UnPack<T2, CAL_INDEXT, AscendC::MicroAPI::HighLowPart::LOWEST>(maxIndexConvertRegT2,
                                                                                              maxIndexConvertReg);
            MicroAPI::DataCopyUnAlign(argmaxAddrLocal, maxIndexConvertRegT2, u1, v2Part[2]);
            MicroAPI::DataCopyUnAlignPost(argmaxAddrLocal, u1, 0);
            AscendC::MicroAPI::UnPack<T2, CAL_INDEXT, AscendC::MicroAPI::HighLowPart::HIGHEST>(maxIndexConvertRegT2,
                                                                                               maxIndexConvertReg);
            MicroAPI::DataCopyUnAlign(argmaxAddrLocal, maxIndexConvertRegT2, u1, v2Part[3]);
            MicroAPI::DataCopyUnAlignPost(argmaxAddrLocal, u1, 0);
        }
    } else if constexpr (sizeof(UB_INDEXT) * 4 == sizeof(CAL_INDEXT)) {
        MicroAPI::RegTensor<UB_INDEXT> t1Reg, divResultReg, wIndexRegUnpack;

        ComputeBaseIndex(maxIndexReg, wInputActualAlignedPad, hInputActualPad, ncInputOffset, t1Reg, divResultReg,
                         wIndexRegUnpack);

        MicroAPI::RegTensor<int32_t> t1RegTep, divResultRegTep, wIndexRegUnpackTep;
        MicroAPI::RegTensor<CAL_INDEXT> t1RegCal, divResultRegCal, wIndexRegUnpackCal;
        MicroAPI::RegTensor<CAL_INDEXT> maxIndexConvertReg;
        MicroAPI::UnalignReg u1;

        AscendC::MicroAPI::UnPack<int32_t, UB_INDEXT, AscendC::MicroAPI::HighLowPart::LOWEST>(t1RegTep, t1Reg);
        AscendC::MicroAPI::UnPack<int32_t, UB_INDEXT, AscendC::MicroAPI::HighLowPart::LOWEST>(divResultRegTep,
                                                                                              divResultReg);
        AscendC::MicroAPI::UnPack<int32_t, UB_INDEXT, AscendC::MicroAPI::HighLowPart::LOWEST>(wIndexRegUnpackTep,
                                                                                              wIndexRegUnpack);

        AscendC::MicroAPI::UnPack<CAL_INDEXT, int32_t, AscendC::MicroAPI::HighLowPart::LOWEST>(t1RegCal, t1RegTep);
        AscendC::MicroAPI::UnPack<CAL_INDEXT, int32_t, AscendC::MicroAPI::HighLowPart::LOWEST>(divResultRegCal,
                                                                                               divResultRegTep);
        AscendC::MicroAPI::UnPack<CAL_INDEXT, int32_t, AscendC::MicroAPI::HighLowPart::LOWEST>(wIndexRegUnpackCal,
                                                                                               wIndexRegUnpackTep);
        ConvertIndexWithoutPadAlign(t1RegCal, divResultRegCal, wIndexRegUnpackCal, left, wInput, hIndexBase, hInput,
                                    dIndexBase, maxIndexConvertReg);
        MicroAPI::DataCopyUnAlign(argmaxAddrLocal, (MicroAPI::RegTensor<T2>&)maxIndexConvertReg, u1, v2Part[0]);
        MicroAPI::DataCopyUnAlignPost(argmaxAddrLocal, u1, 0);

        AscendC::MicroAPI::UnPack<CAL_INDEXT, int32_t, AscendC::MicroAPI::HighLowPart::HIGHEST>(t1RegCal, t1RegTep);
        AscendC::MicroAPI::UnPack<CAL_INDEXT, int32_t, AscendC::MicroAPI::HighLowPart::HIGHEST>(divResultRegCal,
                                                                                                divResultRegTep);
        AscendC::MicroAPI::UnPack<CAL_INDEXT, int32_t, AscendC::MicroAPI::HighLowPart::HIGHEST>(wIndexRegUnpackCal,
                                                                                                wIndexRegUnpackTep);
        ConvertIndexWithoutPadAlign(t1RegCal, divResultRegCal, wIndexRegUnpackCal, left, wInput, hIndexBase, hInput,
                                    dIndexBase, maxIndexConvertReg);
        MicroAPI::DataCopyUnAlign(argmaxAddrLocal, (MicroAPI::RegTensor<T2>&)maxIndexConvertReg, u1, v2Part[1]);
        MicroAPI::DataCopyUnAlignPost(argmaxAddrLocal, u1, 0);

        AscendC::MicroAPI::UnPack<int32_t, UB_INDEXT, AscendC::MicroAPI::HighLowPart::HIGHEST>(t1RegTep, t1Reg);
        AscendC::MicroAPI::UnPack<int32_t, UB_INDEXT, AscendC::MicroAPI::HighLowPart::HIGHEST>(divResultRegTep,
                                                                                               divResultReg);
        AscendC::MicroAPI::UnPack<int32_t, UB_INDEXT, AscendC::MicroAPI::HighLowPart::HIGHEST>(wIndexRegUnpackTep,
                                                                                               wIndexRegUnpack);

        AscendC::MicroAPI::UnPack<CAL_INDEXT, int32_t, AscendC::MicroAPI::HighLowPart::LOWEST>(t1RegCal, t1RegTep);
        AscendC::MicroAPI::UnPack<CAL_INDEXT, int32_t, AscendC::MicroAPI::HighLowPart::LOWEST>(divResultRegCal,
                                                                                               divResultRegTep);
        AscendC::MicroAPI::UnPack<CAL_INDEXT, int32_t, AscendC::MicroAPI::HighLowPart::LOWEST>(wIndexRegUnpackCal,
                                                                                               wIndexRegUnpackTep);
        ConvertIndexWithoutPadAlign(t1RegCal, divResultRegCal, wIndexRegUnpackCal, left, wInput, hIndexBase, hInput,
                                    dIndexBase, maxIndexConvertReg);
        MicroAPI::DataCopyUnAlign(argmaxAddrLocal, (MicroAPI::RegTensor<T2>&)maxIndexConvertReg, u1, v2Part[2]);
        MicroAPI::DataCopyUnAlignPost(argmaxAddrLocal, u1, 0);

        AscendC::MicroAPI::UnPack<CAL_INDEXT, int32_t, AscendC::MicroAPI::HighLowPart::HIGHEST>(t1RegCal, t1RegTep);
        AscendC::MicroAPI::UnPack<CAL_INDEXT, int32_t, AscendC::MicroAPI::HighLowPart::HIGHEST>(divResultRegCal,
                                                                                                divResultRegTep);
        AscendC::MicroAPI::UnPack<CAL_INDEXT, int32_t, AscendC::MicroAPI::HighLowPart::HIGHEST>(wIndexRegUnpackCal,
                                                                                                wIndexRegUnpackTep);
        ConvertIndexWithoutPadAlign(t1RegCal, divResultRegCal, wIndexRegUnpackCal, left, wInput, hIndexBase, hInput,
                                    dIndexBase, maxIndexConvertReg);
        MicroAPI::DataCopyUnAlign(argmaxAddrLocal, (MicroAPI::RegTensor<T2>&)maxIndexConvertReg, u1, v2Part[3]);
        MicroAPI::DataCopyUnAlignPost(argmaxAddrLocal, u1, 0);
    }
}

template <typename T1, typename T2, const uint32_t IS_PAD, typename UB_INDEXT, typename CAL_INDEXT>
__aicore__ inline void
MaxPool3DWithArgmaxV2GatherKernel<T1, T2, IS_PAD, UB_INDEXT, CAL_INDEXT>::ConvertNcAndWriteArgmax(
    MicroAPI::RegTensor<UB_INDEXT>& maxIndexReg, MicroAPI::RegTensor<int32_t>& maxIndexConvertRegTep,
    MicroAPI::RegTensor<T2>& maxIndexConvertRegT2, MicroAPI::UnalignReg& u1, uint32_t wInputActualAlignedPad,
    CAL_INDEXT hInputActualPad, CAL_INDEXT left, CAL_INDEXT wInput, CAL_INDEXT hIndexBase, CAL_INDEXT hInput,
    CAL_INDEXT dIndexBase, CAL_INDEXT ncInputOffset, CAL_INDEXT num3D, CAL_INDEXT rate4D,
    __local_mem__ T2*& argmaxAddrLocal, const uint16_t* v2Part, uint16_t totalElem)
{
    MicroAPI::RegTensor<UB_INDEXT> ncIndexReg;
    MicroAPI::RegTensor<UB_INDEXT> ncDivResultReg;
    MicroAPI::RegTensor<UB_INDEXT> ncConstReg;
    MicroAPI::MaskReg allMaskUb = MicroAPI::CreateMask<UB_INDEXT, MicroAPI::MaskPattern::ALL>();
    MicroAPI::Arange(ncIndexReg, static_cast<UB_INDEXT>(0));
    MicroAPI::Duplicate(ncConstReg, static_cast<UB_INDEXT>(num3D));
    MicroAPI::Div(ncDivResultReg, ncIndexReg, ncConstReg, allMaskUb);
    MicroAPI::Muls(ncDivResultReg, ncDivResultReg, static_cast<UB_INDEXT>(rate4D), allMaskUb);
    MicroAPI::Sub(maxIndexReg, maxIndexReg, ncDivResultReg, allMaskUb);

    MicroAPI::RegTensor<UB_INDEXT> t1Reg, divResultReg, wIndexRegUnpack;
    ComputeBaseIndex(maxIndexReg, wInputActualAlignedPad, hInputActualPad, ncInputOffset, t1Reg, divResultReg,
                     wIndexRegUnpack);

    if constexpr (sizeof(UB_INDEXT) == sizeof(CAL_INDEXT)) {
        MicroAPI::RegTensor<CAL_INDEXT> maxIndexCalReg;
        ConvertIndexWithoutPadAlign(t1Reg, divResultReg, wIndexRegUnpack, left, wInput, hIndexBase, hInput, dIndexBase,
                                    maxIndexCalReg);
        if constexpr (sizeof(CAL_INDEXT) == sizeof(T2)) {
            MicroAPI::DataCopyUnAlign(argmaxAddrLocal, maxIndexCalReg, u1, totalElem);
            MicroAPI::DataCopyUnAlignPost(argmaxAddrLocal, u1, 0);
        } else if constexpr (sizeof(CAL_INDEXT) * 2 == sizeof(T2)) {
            AscendC::MicroAPI::UnPack<T2, CAL_INDEXT, AscendC::MicroAPI::HighLowPart::LOWEST>(maxIndexConvertRegT2,
                                                                                              maxIndexCalReg);
            MicroAPI::DataCopyUnAlign(argmaxAddrLocal, maxIndexConvertRegT2, u1, v2Part[0]);
            MicroAPI::DataCopyUnAlignPost(argmaxAddrLocal, u1, 0);
            AscendC::MicroAPI::UnPack<T2, CAL_INDEXT, AscendC::MicroAPI::HighLowPart::HIGHEST>(maxIndexConvertRegT2,
                                                                                               maxIndexCalReg);
            MicroAPI::DataCopyUnAlign(argmaxAddrLocal, maxIndexConvertRegT2, u1, v2Part[1]);
            MicroAPI::DataCopyUnAlignPost(argmaxAddrLocal, u1, 0);
        } else if constexpr (sizeof(CAL_INDEXT) * 4 == sizeof(T2)) {
            AscendC::MicroAPI::UnPack<int32_t, CAL_INDEXT, AscendC::MicroAPI::HighLowPart::LOWEST>(
                maxIndexConvertRegTep, maxIndexCalReg);

            AscendC::MicroAPI::UnPack<T2, int32_t, AscendC::MicroAPI::HighLowPart::LOWEST>(maxIndexConvertRegT2,
                                                                                           maxIndexConvertRegTep);
            MicroAPI::DataCopyUnAlign(argmaxAddrLocal, maxIndexConvertRegT2, u1, v2Part[0]);
            MicroAPI::DataCopyUnAlignPost(argmaxAddrLocal, u1, 0);

            AscendC::MicroAPI::UnPack<T2, int32_t, AscendC::MicroAPI::HighLowPart::HIGHEST>(maxIndexConvertRegT2,
                                                                                            maxIndexConvertRegTep);
            MicroAPI::DataCopyUnAlign(argmaxAddrLocal, maxIndexConvertRegT2, u1, v2Part[1]);
            MicroAPI::DataCopyUnAlignPost(argmaxAddrLocal, u1, 0);

            AscendC::MicroAPI::UnPack<int32_t, CAL_INDEXT, AscendC::MicroAPI::HighLowPart::HIGHEST>(
                maxIndexConvertRegTep, maxIndexCalReg);

            AscendC::MicroAPI::UnPack<T2, int32_t, AscendC::MicroAPI::HighLowPart::LOWEST>(maxIndexConvertRegT2,
                                                                                           maxIndexConvertRegTep);
            MicroAPI::DataCopyUnAlign(argmaxAddrLocal, maxIndexConvertRegT2, u1, v2Part[2]);
            MicroAPI::DataCopyUnAlignPost(argmaxAddrLocal, u1, 0);

            AscendC::MicroAPI::UnPack<T2, int32_t, AscendC::MicroAPI::HighLowPart::HIGHEST>(maxIndexConvertRegT2,
                                                                                            maxIndexConvertRegTep);
            MicroAPI::DataCopyUnAlign(argmaxAddrLocal, maxIndexConvertRegT2, u1, v2Part[3]);
            MicroAPI::DataCopyUnAlignPost(argmaxAddrLocal, u1, 0);
        }
    } else if constexpr (sizeof(UB_INDEXT) * 2 == sizeof(CAL_INDEXT)) {
        MicroAPI::RegTensor<CAL_INDEXT> t1RegCal, divResultRegCal, wIndexRegUnpackCal;
        MicroAPI::RegTensor<CAL_INDEXT> maxIndexCalReg;

        AscendC::MicroAPI::UnPack<CAL_INDEXT, UB_INDEXT, AscendC::MicroAPI::HighLowPart::LOWEST>(t1RegCal, t1Reg);
        AscendC::MicroAPI::UnPack<CAL_INDEXT, UB_INDEXT, AscendC::MicroAPI::HighLowPart::LOWEST>(divResultRegCal,
                                                                                                 divResultReg);
        AscendC::MicroAPI::UnPack<CAL_INDEXT, UB_INDEXT, AscendC::MicroAPI::HighLowPart::LOWEST>(wIndexRegUnpackCal,
                                                                                                 wIndexRegUnpack);
        ConvertIndexWithoutPadAlign(t1RegCal, divResultRegCal, wIndexRegUnpackCal, left, wInput, hIndexBase, hInput,
                                    dIndexBase, maxIndexCalReg);
        if constexpr (sizeof(CAL_INDEXT) == sizeof(T2)) {
            MicroAPI::DataCopyUnAlign(argmaxAddrLocal, maxIndexCalReg, u1, v2Part[0]);
            MicroAPI::DataCopyUnAlignPost(argmaxAddrLocal, u1, 0);
        } else if constexpr (sizeof(CAL_INDEXT) * 2 == sizeof(T2)) {
            AscendC::MicroAPI::UnPack<T2, CAL_INDEXT, AscendC::MicroAPI::HighLowPart::LOWEST>(maxIndexConvertRegT2,
                                                                                              maxIndexCalReg);
            MicroAPI::DataCopyUnAlign(argmaxAddrLocal, maxIndexConvertRegT2, u1, v2Part[0]);
            MicroAPI::DataCopyUnAlignPost(argmaxAddrLocal, u1, 0);
            AscendC::MicroAPI::UnPack<T2, CAL_INDEXT, AscendC::MicroAPI::HighLowPart::HIGHEST>(maxIndexConvertRegT2,
                                                                                               maxIndexCalReg);
            MicroAPI::DataCopyUnAlign(argmaxAddrLocal, maxIndexConvertRegT2, u1, v2Part[1]);
            MicroAPI::DataCopyUnAlignPost(argmaxAddrLocal, u1, 0);
        }

        AscendC::MicroAPI::UnPack<CAL_INDEXT, UB_INDEXT, AscendC::MicroAPI::HighLowPart::HIGHEST>(t1RegCal, t1Reg);
        AscendC::MicroAPI::UnPack<CAL_INDEXT, UB_INDEXT, AscendC::MicroAPI::HighLowPart::HIGHEST>(divResultRegCal,
                                                                                                  divResultReg);
        AscendC::MicroAPI::UnPack<CAL_INDEXT, UB_INDEXT, AscendC::MicroAPI::HighLowPart::HIGHEST>(wIndexRegUnpackCal,
                                                                                                  wIndexRegUnpack);
        ConvertIndexWithoutPadAlign(t1RegCal, divResultRegCal, wIndexRegUnpackCal, left, wInput, hIndexBase, hInput,
                                    dIndexBase, maxIndexCalReg);
        if constexpr (sizeof(CAL_INDEXT) == sizeof(T2)) {
            MicroAPI::DataCopyUnAlign(argmaxAddrLocal, maxIndexCalReg, u1, v2Part[1]);
            MicroAPI::DataCopyUnAlignPost(argmaxAddrLocal, u1, 0);
        } else if constexpr (sizeof(CAL_INDEXT) * 2 == sizeof(T2)) {
            AscendC::MicroAPI::UnPack<T2, CAL_INDEXT, AscendC::MicroAPI::HighLowPart::LOWEST>(maxIndexConvertRegT2,
                                                                                              maxIndexCalReg);
            MicroAPI::DataCopyUnAlign(argmaxAddrLocal, maxIndexConvertRegT2, u1, v2Part[2]);
            MicroAPI::DataCopyUnAlignPost(argmaxAddrLocal, u1, 0);
            AscendC::MicroAPI::UnPack<T2, CAL_INDEXT, AscendC::MicroAPI::HighLowPart::HIGHEST>(maxIndexConvertRegT2,
                                                                                               maxIndexCalReg);
            MicroAPI::DataCopyUnAlign(argmaxAddrLocal, maxIndexConvertRegT2, u1, v2Part[3]);
            MicroAPI::DataCopyUnAlignPost(argmaxAddrLocal, u1, 0);
        }
    } else if constexpr (sizeof(UB_INDEXT) * 4 == sizeof(CAL_INDEXT)) {
        MicroAPI::RegTensor<int32_t> t1RegTep, divResultRegTep, wIndexRegUnpackTep;
        MicroAPI::RegTensor<CAL_INDEXT> t1RegCal, divResultRegCal, wIndexRegUnpackCal;
        MicroAPI::RegTensor<CAL_INDEXT> maxIndexCalReg;

        AscendC::MicroAPI::UnPack<int32_t, UB_INDEXT, AscendC::MicroAPI::HighLowPart::LOWEST>(t1RegTep, t1Reg);
        AscendC::MicroAPI::UnPack<int32_t, UB_INDEXT, AscendC::MicroAPI::HighLowPart::LOWEST>(divResultRegTep,
                                                                                              divResultReg);
        AscendC::MicroAPI::UnPack<int32_t, UB_INDEXT, AscendC::MicroAPI::HighLowPart::LOWEST>(wIndexRegUnpackTep,
                                                                                              wIndexRegUnpack);

        AscendC::MicroAPI::UnPack<CAL_INDEXT, int32_t, AscendC::MicroAPI::HighLowPart::LOWEST>(t1RegCal, t1RegTep);
        AscendC::MicroAPI::UnPack<CAL_INDEXT, int32_t, AscendC::MicroAPI::HighLowPart::LOWEST>(divResultRegCal,
                                                                                               divResultRegTep);
        AscendC::MicroAPI::UnPack<CAL_INDEXT, int32_t, AscendC::MicroAPI::HighLowPart::LOWEST>(wIndexRegUnpackCal,
                                                                                               wIndexRegUnpackTep);
        ConvertIndexWithoutPadAlign(t1RegCal, divResultRegCal, wIndexRegUnpackCal, left, wInput, hIndexBase, hInput,
                                    dIndexBase, maxIndexCalReg);
        MicroAPI::DataCopyUnAlign(argmaxAddrLocal, (MicroAPI::RegTensor<T2>&)maxIndexCalReg, u1, v2Part[0]);
        MicroAPI::DataCopyUnAlignPost(argmaxAddrLocal, u1, 0);

        AscendC::MicroAPI::UnPack<CAL_INDEXT, int32_t, AscendC::MicroAPI::HighLowPart::HIGHEST>(t1RegCal, t1RegTep);
        AscendC::MicroAPI::UnPack<CAL_INDEXT, int32_t, AscendC::MicroAPI::HighLowPart::HIGHEST>(divResultRegCal,
                                                                                                divResultRegTep);
        AscendC::MicroAPI::UnPack<CAL_INDEXT, int32_t, AscendC::MicroAPI::HighLowPart::HIGHEST>(wIndexRegUnpackCal,
                                                                                                wIndexRegUnpackTep);
        ConvertIndexWithoutPadAlign(t1RegCal, divResultRegCal, wIndexRegUnpackCal, left, wInput, hIndexBase, hInput,
                                    dIndexBase, maxIndexCalReg);
        MicroAPI::DataCopyUnAlign(argmaxAddrLocal, (MicroAPI::RegTensor<T2>&)maxIndexCalReg, u1, v2Part[1]);
        MicroAPI::DataCopyUnAlignPost(argmaxAddrLocal, u1, 0);

        AscendC::MicroAPI::UnPack<int32_t, UB_INDEXT, AscendC::MicroAPI::HighLowPart::HIGHEST>(t1RegTep, t1Reg);
        AscendC::MicroAPI::UnPack<int32_t, UB_INDEXT, AscendC::MicroAPI::HighLowPart::HIGHEST>(divResultRegTep,
                                                                                               divResultReg);
        AscendC::MicroAPI::UnPack<int32_t, UB_INDEXT, AscendC::MicroAPI::HighLowPart::HIGHEST>(wIndexRegUnpackTep,
                                                                                               wIndexRegUnpack);

        AscendC::MicroAPI::UnPack<CAL_INDEXT, int32_t, AscendC::MicroAPI::HighLowPart::LOWEST>(t1RegCal, t1RegTep);
        AscendC::MicroAPI::UnPack<CAL_INDEXT, int32_t, AscendC::MicroAPI::HighLowPart::LOWEST>(divResultRegCal,
                                                                                               divResultRegTep);
        AscendC::MicroAPI::UnPack<CAL_INDEXT, int32_t, AscendC::MicroAPI::HighLowPart::LOWEST>(wIndexRegUnpackCal,
                                                                                               wIndexRegUnpackTep);
        ConvertIndexWithoutPadAlign(t1RegCal, divResultRegCal, wIndexRegUnpackCal, left, wInput, hIndexBase, hInput,
                                    dIndexBase, maxIndexCalReg);
        MicroAPI::DataCopyUnAlign(argmaxAddrLocal, (MicroAPI::RegTensor<T2>&)maxIndexCalReg, u1, v2Part[2]);
        MicroAPI::DataCopyUnAlignPost(argmaxAddrLocal, u1, 0);

        AscendC::MicroAPI::UnPack<CAL_INDEXT, int32_t, AscendC::MicroAPI::HighLowPart::HIGHEST>(t1RegCal, t1RegTep);
        AscendC::MicroAPI::UnPack<CAL_INDEXT, int32_t, AscendC::MicroAPI::HighLowPart::HIGHEST>(divResultRegCal,
                                                                                                divResultRegTep);
        AscendC::MicroAPI::UnPack<CAL_INDEXT, int32_t, AscendC::MicroAPI::HighLowPart::HIGHEST>(wIndexRegUnpackCal,
                                                                                                wIndexRegUnpackTep);
        ConvertIndexWithoutPadAlign(t1RegCal, divResultRegCal, wIndexRegUnpackCal, left, wInput, hIndexBase, hInput,
                                    dIndexBase, maxIndexCalReg);
        MicroAPI::DataCopyUnAlign(argmaxAddrLocal, (MicroAPI::RegTensor<T2>&)maxIndexCalReg, u1, v2Part[3]);
        MicroAPI::DataCopyUnAlignPost(argmaxAddrLocal, u1, 0);
    }
}

template <typename T1, typename T2, const uint32_t IS_PAD, typename UB_INDEXT, typename CAL_INDEXT>
__aicore__ inline void MaxPool3DWithArgmaxV2GatherKernel<T1, T2, IS_PAD, UB_INDEXT, CAL_INDEXT>::ComputeV2Parts(
    uint16_t* v2Part, uint16_t totalElem)
{
    uint16_t tmp = totalElem;
    for (uint16_t i = 0; i < 4; i++) {
        if (tmp > vlT2_) {
            v2Part[i] = vlT2_;
            tmp -= vlT2_;
        } else {
            v2Part[i] = tmp;
            break;
        }
    }
}

template <typename T1, typename T2, const uint32_t IS_PAD, typename UB_INDEXT, typename CAL_INDEXT>
__aicore__ inline void MaxPool3DWithArgmaxV2GatherKernel<T1, T2, IS_PAD, UB_INDEXT, CAL_INDEXT>::ProcessW(
    __local_mem__ T1* computeAddr, __local_mem__ T1* maxValueAddr, UB_INDEXT hOffset, uint32_t wStrideOffset,
    uint32_t hInputActualPad, MicroAPI::RegTensor<UB_INDEXT>& indexReg, uint16_t dKernel, uint16_t hKernel,
    uint16_t wKernel, uint16_t repeatElem, UB_INDEXT outputOffset, MicroAPI::RegTensor<UB_INDEXT>& maxIndexReg,
    uint16_t dDilation, uint16_t hDilation, uint16_t wDilation)
{
    MicroAPI::RegTensor<UB_INDEXT> indexWithOffset;
    MicroAPI::RegTensor<T1> calcReg;
    uint32_t maskCount = repeatElem;
    MicroAPI::MaskReg allMaskU16 = MicroAPI::CreateMask<UB_INDEXT, MicroAPI::MaskPattern::ALL>();
    MicroAPI::MaskReg gatherMask = MicroAPI::UpdateMask<T1>(maskCount);
    MicroAPI::RegTensor<T1> maxReg;
    MicroAPI::MaskReg neMask;
    MicroAPI::MaskReg gtMask;
    MicroAPI::UnalignReg u0;

    __local_mem__ T1* maxValueAddrLocal = maxValueAddr + outputOffset;
    SetNegInfReg<T1>(maxReg);
    MicroAPI::Adds(maxIndexReg, indexReg, hOffset, allMaskU16);
    for (uint16_t d = 0; d < dKernel; d++) {
        for (uint16_t i = 0; i < hKernel; i++) {
            for (uint16_t j = 0; j < wKernel; j++) {
                UB_INDEXT relIndex = d * hInputActualPad * wStrideOffset * dDilation + i * wStrideOffset * hDilation +
                                     j * wDilation;
                UB_INDEXT offset = static_cast<UB_INDEXT>(hOffset + relIndex);
                MicroAPI::Adds(indexWithOffset, indexReg, offset, allMaskU16);
                if constexpr (std::is_same<T1, bfloat16_t>::value || std::is_same<T1, float16_t>::value) {
                    MicroAPI::DataCopyGather(calcReg, computeAddr, (MicroAPI::RegTensor<uint16_t>&)indexWithOffset,
                                             gatherMask);
                } else {
                    MicroAPI::DataCopyGather(calcReg, computeAddr, (MicroAPI::RegTensor<uint32_t>&)indexWithOffset,
                                             gatherMask);
                }
                MicroAPI::Compare<T1, CMPMODE::GT>(gtMask, calcReg, maxReg, gatherMask);
                MicroAPI::Compare<T1, CMPMODE::NE>(neMask, calcReg, calcReg, gatherMask);
                MicroAPI::MaskOr(gtMask, gtMask, neMask, gatherMask);
                MicroAPI::Select(maxIndexReg, indexWithOffset, maxIndexReg, gtMask);

                MicroAPI::Max(maxReg, maxReg, calcReg, gatherMask);
            }
        }
    }
    MicroAPI::DataCopyUnAlign(maxValueAddrLocal, maxReg, u0, repeatElem);
    MicroAPI::DataCopyUnAlignPost(maxValueAddrLocal, u0, 0);
    return;
}

template <typename T1, typename T2, const uint32_t IS_PAD, typename UB_INDEXT, typename CAL_INDEXT>
__aicore__ inline void MaxPool3DWithArgmaxV2GatherKernel<T1, T2, IS_PAD, UB_INDEXT, CAL_INDEXT>::SingleRowGather(
    __local_mem__ T1* computeAddr, __local_mem__ T1* maxValueAddr, __local_mem__ T2* argmaxAddr)
{
    uint16_t loopW = wOutputReal_ / vlT1_;
    uint16_t repeatsElem = vlT1_;
    uint16_t tailRepeatsElem = wOutputReal_ - loopW * vlT1_;

    uint16_t dKernel = tilingData_.dKernel;
    uint16_t hKernel = tilingData_.hKernel;
    uint16_t wKernel = tilingData_.wKernel;
    if (tailRepeatsElem == 0) {
        loopW = loopW - 1;
        tailRepeatsElem = repeatsElem;
    }

    uint16_t v2Part[4] = {0, 0, 0, 0};
    uint16_t v2PartTail[4] = {0, 0, 0, 0};
    ComputeV2Parts(v2Part, repeatsElem);
    ComputeV2Parts(v2PartTail, tailRepeatsElem);

    uint32_t dStride = tilingData_.dStride;
    uint32_t hStride = tilingData_.hStride;
    uint32_t wStride = tilingData_.wStride;
    CAL_INDEXT left = wAxisIndex_ * tilingData_.wOutputInner * tilingData_.wStride - tilingData_.padLeft;
    CAL_INDEXT hIndexBase = hAxisIndex_ * tilingData_.hOutputInner * tilingData_.hStride - tilingData_.padTop;
    CAL_INDEXT dIndexBase = dAxisIndex_ * tilingData_.dOutputInner * tilingData_.dStride - tilingData_.padFront;
    CAL_INDEXT hInput = tilingData_.hInput;
    CAL_INDEXT wInput = tilingData_.wInput;
    uint32_t highAxisActual = highAxisActual_;
    uint32_t dInputActualPad = dInputActualPad_;
    uint32_t hInputActualPad = hInputActualPad_;
    uint32_t dOutputActual = dOutputReal_;
    uint32_t hOutputActual = hOutputReal_;
    uint32_t wOutputActual = wOutputReal_;
    uint16_t dDilation = tilingData_.dDilation;
    uint16_t hDilation = tilingData_.hDilation;
    uint16_t wDilation = tilingData_.wDilation;

    uint32_t inputPlaneStride = dInputActualPad * hInputActualPad * wInputActualAlignedPad_;
    uint32_t outputPlaneStride = dOutputActual * hOutputActual * wOutputActual;
    uint32_t dInputStrideVal = dStride * hInputActualPad * wInputActualAlignedPad_;
    uint32_t dOutputStrideVal = hOutputActual * wOutputActual;
    uint32_t hInputStrideVal = wInputActualAlignedPad_ * hStride;
    uint32_t wInputStep = static_cast<uint32_t>(repeatsElem) * wStride;
    uint32_t tailInputOffset = static_cast<uint32_t>(loopW) * wInputStep;
    uint32_t tailOutputOffset = static_cast<uint32_t>(loopW) * repeatsElem;
    for (uint16_t nc = 0; nc < static_cast<uint16_t>(highAxisActual); nc++) {
        uint32_t ncInputBase = nc * inputPlaneStride;
        uint32_t ncOutBase = nc * outputPlaneStride;
        for (uint16_t dLoop = 0; dLoop < static_cast<uint16_t>(dOutputActual); dLoop++) {
            uint32_t dInBase = dLoop * dInputStrideVal;
            uint32_t dOutBase = dLoop * dOutputStrideVal;
            for (uint16_t hLoop = 0; hLoop < static_cast<uint16_t>(hOutputActual); hLoop++) {
                uint32_t hInBase = hLoop * hInputStrideVal;
                uint32_t hOutBase = hLoop * wOutputActual;
                uint32_t inputBase = ncInputBase + dInBase + hInBase;
                uint32_t outputBase = ncOutBase + dOutBase + hOutBase;
                __VEC_SCOPE__
                {
                    MicroAPI::RegTensor<UB_INDEXT> indexReg;
                    MicroAPI::RegTensor<UB_INDEXT> maxIndexReg;
                    UB_INDEXT ncInputOffset = static_cast<UB_INDEXT>(ncInputBase);
                    __local_mem__ T2* argmaxAddrLocal = argmaxAddr + outputBase;
                    MicroAPI::Arange(indexReg, static_cast<UB_INDEXT>(0));
                    MicroAPI::MaskReg preg = MicroAPI::CreateMask<UB_INDEXT, MicroAPI::MaskPattern::ALL>();
                    MicroAPI::Muls(indexReg, indexReg, static_cast<UB_INDEXT>(wStride), preg);

                    for (uint16_t wLoop = 0; wLoop < loopW; wLoop++) {
                        UB_INDEXT wOffset = static_cast<UB_INDEXT>(inputBase + wLoop * wInputStep);
                        UB_INDEXT wOutOffset = static_cast<UB_INDEXT>(outputBase + wLoop * repeatsElem);

                        ProcessW(computeAddr, maxValueAddr, wOffset, wInputActualAlignedPad_, hInputActualPad, indexReg,
                                 dKernel, hKernel, wKernel, repeatsElem, wOutOffset, maxIndexReg, dDilation, hDilation,
                                 wDilation);

                        ConvertBaseIndexAndWriteArgmax(maxIndexReg, wInputActualAlignedPad_, hInputActualPad,
                                                       ncInputOffset, left, wInput, hIndexBase, hInput, dIndexBase,
                                                       argmaxAddrLocal, v2Part, repeatsElem);
                    }

                    UB_INDEXT wOffsetTail = static_cast<UB_INDEXT>(inputBase + tailInputOffset);
                    UB_INDEXT wOutOffsetTail = static_cast<UB_INDEXT>(outputBase + tailOutputOffset);
                    ProcessW(computeAddr, maxValueAddr, wOffsetTail, wInputActualAlignedPad_, hInputActualPad, indexReg,
                             dKernel, hKernel, wKernel, tailRepeatsElem, wOutOffsetTail, maxIndexReg, dDilation,
                             hDilation, wDilation);

                    ConvertBaseIndexAndWriteArgmax(maxIndexReg, wInputActualAlignedPad_, hInputActualPad, ncInputOffset,
                                                   left, wInput, hIndexBase, hInput, dIndexBase, argmaxAddrLocal,
                                                   v2PartTail, tailRepeatsElem);
                }
            }
        }
    }
    return;
}

template <typename T1, typename T2, const uint32_t IS_PAD, typename UB_INDEXT, typename CAL_INDEXT>
__aicore__ inline void MaxPool3DWithArgmaxV2GatherKernel<T1, T2, IS_PAD, UB_INDEXT, CAL_INDEXT>::MultiDepGather(
    __local_mem__ T1* computeAddr, __local_mem__ T1* maxValueAddr, __local_mem__ T2* argmaxAddr)
{
    uint16_t wKernel = tilingData_.wKernel;
    uint16_t hKernel = tilingData_.hKernel;
    uint16_t dKernel = tilingData_.dKernel;
    uint32_t wStride = tilingData_.wStride;
    uint32_t rate3D = tilingData_.dStride * hInputActualPad_ * wInputActualAlignedPad_;
    uint16_t num2D = hOutputReal_ * wOutputReal_;
    uint32_t rate2D = tilingData_.hStride * wInputActualAlignedPad_;
    uint16_t wOutputActual = wOutputReal_;
    uint16_t eachDepCount = hOutputReal_ * wOutputReal_;
    uint16_t dBatchCount = vlT1_ / eachDepCount;
    uint16_t dLoopTimes = dOutputReal_ / dBatchCount;
    uint16_t dTail = dOutputReal_ - dLoopTimes * dBatchCount;
    if (dTail == 0) {
        dLoopTimes = dLoopTimes - 1;
        dTail = dBatchCount;
    }
    uint16_t repeatsElem = dBatchCount * eachDepCount;
    uint16_t tailRepeatsElem = dTail * eachDepCount;

    uint16_t v2Part[4] = {0, 0, 0, 0};
    uint16_t v2PartTail[4] = {0, 0, 0, 0};
    ComputeV2Parts(v2Part, repeatsElem);
    ComputeV2Parts(v2PartTail, tailRepeatsElem);

    CAL_INDEXT left = wAxisIndex_ * tilingData_.wOutputInner * tilingData_.wStride - tilingData_.padLeft;
    CAL_INDEXT hIndexBase = hAxisIndex_ * tilingData_.hOutputInner * tilingData_.hStride - tilingData_.padTop;
    CAL_INDEXT dIndexBase = dAxisIndex_ * tilingData_.dOutputInner * tilingData_.dStride - tilingData_.padFront;
    CAL_INDEXT hInput = tilingData_.hInput;
    CAL_INDEXT wInput = tilingData_.wInput;
    uint32_t highAxisActual = highAxisActual_;
    uint32_t dInputActualPad = dInputActualPad_;
    uint32_t hInputActualPad = hInputActualPad_;
    uint32_t wInputActualAlignedPad = wInputActualAlignedPad_;
    uint32_t dOutputActual = dOutputReal_;
    uint32_t hOutputActual = hOutputReal_;
    uint32_t dStride = tilingData_.dStride;
    uint32_t hStride = tilingData_.hStride;
    uint16_t dDilation = tilingData_.dDilation;
    uint16_t hDilation = tilingData_.hDilation;
    uint16_t wDilation = tilingData_.wDilation;

    uint32_t inputPlaneStride = dInputActualPad * hInputActualPad * wInputActualAlignedPad;
    uint32_t outputPlaneStride = dOutputActual * hOutputActual * wOutputActual;
    uint32_t dBatchInputStride = static_cast<uint32_t>(dBatchCount) * dStride * hInputActualPad *
                                 wInputActualAlignedPad;
    uint32_t dBatchOutputStride = static_cast<uint32_t>(dBatchCount) * hOutputActual * wOutputActual;
    uint32_t tailDInputOffset = static_cast<uint32_t>(dLoopTimes) * dBatchInputStride;
    uint32_t tailDOutputOffset = static_cast<uint32_t>(dLoopTimes) * dBatchOutputStride;
    for (uint16_t nc = 0; nc < static_cast<uint16_t>(highAxisActual); nc++) {
        uint32_t ncInputBase = nc * inputPlaneStride;
        uint32_t ncOutBase = nc * outputPlaneStride;
        __VEC_SCOPE__
        {
            MicroAPI::RegTensor<UB_INDEXT> indexReg;
            MicroAPI::RegTensor<UB_INDEXT> maxIndexReg;
            UB_INDEXT ncInputOffset = static_cast<UB_INDEXT>(ncInputBase);
            __local_mem__ T2* argmaxAddrLocal = argmaxAddr + ncOutBase;
            CalGatterIndex3D<UB_INDEXT>(indexReg, rate3D, num2D, rate2D, wOutputActual, wStride);
            for (uint16_t dLoop = 0; dLoop < static_cast<uint16_t>(dLoopTimes); dLoop++) {
                UB_INDEXT wOffset = static_cast<UB_INDEXT>(ncInputBase + dLoop * dBatchInputStride);
                UB_INDEXT wOutputOffset = static_cast<UB_INDEXT>(ncOutBase + dLoop * dBatchOutputStride);
                ProcessW(computeAddr, maxValueAddr, wOffset, wInputActualAlignedPad, hInputActualPad, indexReg, dKernel,
                         hKernel, wKernel, repeatsElem, wOutputOffset, maxIndexReg, dDilation, hDilation, wDilation);
                ConvertBaseIndexAndWriteArgmax(maxIndexReg, wInputActualAlignedPad, hInputActualPad, ncInputOffset,
                                               left, wInput, hIndexBase, hInput, dIndexBase, argmaxAddrLocal, v2Part,
                                               repeatsElem);
            }
            UB_INDEXT wOffsetTail = static_cast<UB_INDEXT>(ncInputBase + tailDInputOffset);
            UB_INDEXT wOutputOffsetTail = static_cast<UB_INDEXT>(ncOutBase + tailDOutputOffset);
            ProcessW(computeAddr, maxValueAddr, wOffsetTail, wInputActualAlignedPad, hInputActualPad, indexReg, dKernel,
                     hKernel, wKernel, tailRepeatsElem, wOutputOffsetTail, maxIndexReg, dDilation, hDilation,
                     wDilation);
            ConvertBaseIndexAndWriteArgmax(maxIndexReg, wInputActualAlignedPad, hInputActualPad, ncInputOffset, left,
                                           wInput, hIndexBase, hInput, dIndexBase, argmaxAddrLocal, v2PartTail,
                                           tailRepeatsElem);
        }
    }
    return;
}

template <typename T1, typename T2, const uint32_t IS_PAD, typename UB_INDEXT, typename CAL_INDEXT>
__aicore__ inline void MaxPool3DWithArgmaxV2GatherKernel<T1, T2, IS_PAD, UB_INDEXT, CAL_INDEXT>::MultiRowGather(
    __local_mem__ T1* computeAddr, __local_mem__ T1* maxValueAddr, __local_mem__ T2* argmaxAddr)
{
    uint16_t dKernel = tilingData_.dKernel;
    uint16_t wStride = tilingData_.wStride;
    uint16_t wOutputActual = wOutputReal_;
    uint16_t wKernel = tilingData_.wKernel;
    uint16_t hKernel = tilingData_.hKernel;

    uint16_t hBatchCount = vlT1_ / wOutputReal_;
    uint16_t rate2D = wInputActualAlignedPad_ * tilingData_.hStride;
    uint16_t hLoopTimes = hOutputReal_ / hBatchCount;
    uint16_t hTail = hOutputReal_ - hLoopTimes * hBatchCount;
    if (hTail == 0) {
        hLoopTimes = hLoopTimes - 1;
        hTail = hBatchCount;
    }
    uint16_t repeatsElem = hBatchCount * wOutputReal_;
    uint16_t tailRepeatsElem = hTail * wOutputReal_;
    uint16_t v2Part[4] = {0, 0, 0, 0};
    uint16_t v2PartTail[4] = {0, 0, 0, 0};
    ComputeV2Parts(v2Part, repeatsElem);
    ComputeV2Parts(v2PartTail, tailRepeatsElem);

    CAL_INDEXT left = wAxisIndex_ * tilingData_.wOutputInner * tilingData_.wStride - tilingData_.padLeft;
    CAL_INDEXT hIndexBase = hAxisIndex_ * tilingData_.hOutputInner * tilingData_.hStride - tilingData_.padTop;
    CAL_INDEXT dIndexBase = dAxisIndex_ * tilingData_.dOutputInner * tilingData_.dStride - tilingData_.padFront;
    CAL_INDEXT hInput = tilingData_.hInput;
    CAL_INDEXT wInput = tilingData_.wInput;
    uint16_t highAxisActual = highAxisActual_;
    uint16_t dInputActualPad = dInputActualPad_;
    uint16_t hInputActualPad = hInputActualPad_;
    uint16_t wInputActualAlignedPad = wInputActualAlignedPad_;
    uint16_t dOutputActual = dOutputReal_;
    uint16_t hOutputActual = hOutputReal_;
    uint16_t dStride = tilingData_.dStride;
    uint16_t hStride = tilingData_.hStride;
    uint16_t dDilation = tilingData_.dDilation;
    uint16_t hDilation = tilingData_.hDilation;
    uint16_t wDilation = tilingData_.wDilation;
    uint32_t inputPlaneStride = static_cast<uint32_t>(dInputActualPad) * hInputActualPad * wInputActualAlignedPad;
    uint32_t outputPlaneStride = static_cast<uint32_t>(dOutputActual) * hOutputActual * wOutputActual;
    uint32_t dInputStride = static_cast<uint32_t>(dStride) * hInputActualPad * wInputActualAlignedPad;
    uint32_t dOutputStride = static_cast<uint32_t>(hOutputActual) * wOutputActual;
    uint32_t hInputStride = static_cast<uint32_t>(hBatchCount) * hStride * wInputActualAlignedPad;
    uint32_t hOutputStride = static_cast<uint32_t>(hBatchCount) * wOutputActual;
    uint32_t tailHInputOffset = static_cast<uint32_t>(hLoopTimes) * hInputStride;
    uint32_t tailHOutputOffset = static_cast<uint32_t>(hLoopTimes) * hOutputStride;
    for (uint16_t nc = 0; nc < highAxisActual; nc++) {
        uint32_t ncInputBase = nc * inputPlaneStride;
        uint32_t ncOutBase = nc * outputPlaneStride;
        for (uint16_t dLoop = 0; dLoop < dOutputActual; dLoop++) {
            uint32_t ncDInputBase = ncInputBase + dLoop * dInputStride;
            uint32_t ncDOutBase = ncOutBase + dLoop * dOutputStride;
            __VEC_SCOPE__
            {
                MicroAPI::RegTensor<UB_INDEXT> indexReg;
                MicroAPI::RegTensor<UB_INDEXT> maxIndexReg;
                UB_INDEXT ncInputOffset = static_cast<UB_INDEXT>(ncInputBase);
                __local_mem__ T2* argmaxAddrLocal = argmaxAddr + ncDOutBase;
                CalGatterIndex2D<UB_INDEXT>(indexReg, rate2D, wOutputActual, wStride);
                for (uint16_t hLoop = 0; hLoop < hLoopTimes; hLoop++) {
                    UB_INDEXT wOffset = static_cast<UB_INDEXT>(ncDInputBase + hLoop * hInputStride);
                    UB_INDEXT wOutputOffset = static_cast<UB_INDEXT>(ncDOutBase + hLoop * hOutputStride);
                    ProcessW(computeAddr, maxValueAddr, wOffset, wInputActualAlignedPad, hInputActualPad, indexReg,
                             dKernel, hKernel, wKernel, repeatsElem, wOutputOffset, maxIndexReg, dDilation, hDilation,
                             wDilation);
                    ConvertBaseIndexAndWriteArgmax(maxIndexReg, wInputActualAlignedPad, hInputActualPad, ncInputOffset,
                                                   left, wInput, hIndexBase, hInput, dIndexBase, argmaxAddrLocal,
                                                   v2Part, repeatsElem);
                }

                UB_INDEXT wOffsetTail = static_cast<UB_INDEXT>(ncDInputBase + tailHInputOffset);
                UB_INDEXT wOutputOffsetTail = static_cast<UB_INDEXT>(ncDOutBase + tailHOutputOffset);
                ProcessW(computeAddr, maxValueAddr, wOffsetTail, wInputActualAlignedPad, hInputActualPad, indexReg,
                         dKernel, hKernel, wKernel, tailRepeatsElem, wOutputOffsetTail, maxIndexReg, dDilation,
                         hDilation, wDilation);
                ConvertBaseIndexAndWriteArgmax(maxIndexReg, wInputActualAlignedPad, hInputActualPad, ncInputOffset,
                                               left, wInput, hIndexBase, hInput, dIndexBase, argmaxAddrLocal,
                                               v2PartTail, tailRepeatsElem);
            }
        }
    }
    return;
}

template <typename T1, typename T2, const uint32_t IS_PAD, typename UB_INDEXT, typename CAL_INDEXT>
__aicore__ inline void MaxPool3DWithArgmaxV2GatherKernel<T1, T2, IS_PAD, UB_INDEXT, CAL_INDEXT>::MultiNcGather(
    __local_mem__ T1* computeAddr, __local_mem__ T1* maxValueAddr, __local_mem__ T2* argmaxAddr)
{
    uint16_t dKernel = tilingData_.dKernel;
    uint16_t hKernel = tilingData_.hKernel;
    uint16_t wKernel = tilingData_.wKernel;
    uint32_t wStride = tilingData_.wStride;
    uint32_t rate4D = dInputActualPad_ * hInputActualPad_ * wInputActualAlignedPad_;
    uint32_t num3D = dOutputReal_ * hOutputReal_ * wOutputReal_;
    uint32_t rate3D = tilingData_.dStride * hInputActualPad_ * wInputActualAlignedPad_;
    uint16_t num2D = hOutputReal_ * wOutputReal_;
    uint32_t rate2D = tilingData_.hStride * wInputActualAlignedPad_;
    uint16_t wOutputActual = wOutputReal_;
    uint16_t eachBatchCount = dOutputReal_ * hOutputReal_ * wOutputReal_;
    uint16_t ncBatchCount = vlT1_ / eachBatchCount;
    uint16_t ncLoopTimes = highAxisActual_ / ncBatchCount;
    uint16_t ncTail = highAxisActual_ - ncLoopTimes * ncBatchCount;
    if (ncTail == 0) {
        ncLoopTimes = ncLoopTimes - 1;
        ncTail = ncBatchCount;
    }
    uint16_t repeatsElem = ncBatchCount * eachBatchCount;
    uint16_t tailRepeatsElem = ncTail * eachBatchCount;

    uint16_t v2Part[4] = {0, 0, 0, 0};
    uint16_t v2PartTail[4] = {0, 0, 0, 0};
    ComputeV2Parts(v2Part, repeatsElem);
    ComputeV2Parts(v2PartTail, tailRepeatsElem);

    CAL_INDEXT left = wAxisIndex_ * tilingData_.wOutputInner * tilingData_.wStride - tilingData_.padLeft;
    CAL_INDEXT hIndexBase = hAxisIndex_ * tilingData_.hOutputInner * tilingData_.hStride - tilingData_.padTop;
    CAL_INDEXT dIndexBase = dAxisIndex_ * tilingData_.dOutputInner * tilingData_.dStride - tilingData_.padFront;
    CAL_INDEXT hInput = tilingData_.hInput;
    CAL_INDEXT wInput = tilingData_.wInput;
    uint32_t dInputActualPad = dInputActualPad_;
    uint32_t hInputActualPad = hInputActualPad_;
    uint32_t wInputActualAlignedPad = wInputActualAlignedPad_;
    uint32_t dOutputActual = dOutputReal_;
    uint32_t hOutputActual = hOutputReal_;
    uint32_t dStride = tilingData_.dStride;
    uint32_t hStride = tilingData_.hStride;
    uint16_t dDilation = tilingData_.dDilation;
    uint16_t hDilation = tilingData_.hDilation;
    uint16_t wDilation = tilingData_.wDilation;

    uint32_t wInputStrideU32 = ncBatchCount * dInputActualPad * hInputActualPad * wInputActualAlignedPad;
    uint32_t wOutStrideU32 = ncBatchCount * dOutputActual * hOutputActual * wOutputActual;
    uint32_t tailNcInputOffset = static_cast<uint32_t>(ncLoopTimes) * wInputStrideU32;
    uint32_t tailNcOutputOffset = static_cast<uint32_t>(ncLoopTimes) * wOutStrideU32;
    __VEC_SCOPE__
    {
        MicroAPI::RegTensor<UB_INDEXT> indexReg;
        MicroAPI::RegTensor<UB_INDEXT> maxIndexReg;
        MicroAPI::RegTensor<int32_t> maxIndexConvertRegTep;
        MicroAPI::RegTensor<T2> maxIndexConvertRegT2;
        MicroAPI::UnalignReg u1;
        __local_mem__ T2* argmaxAddrLocal = argmaxAddr;
        CalGatterIndex4D<UB_INDEXT>(indexReg, rate4D, num3D, rate3D, num2D, rate2D, wOutputActual, wStride);

        for (uint16_t ncLoop = 0; ncLoop < ncLoopTimes; ncLoop++) {
            UB_INDEXT wOffset = static_cast<UB_INDEXT>(ncLoop * wInputStrideU32);
            UB_INDEXT wOutputOffset = static_cast<UB_INDEXT>(ncLoop * wOutStrideU32);
            ProcessW(computeAddr, maxValueAddr, wOffset, wInputActualAlignedPad, hInputActualPad, indexReg, dKernel,
                     hKernel, wKernel, repeatsElem, wOutputOffset, maxIndexReg, dDilation, hDilation, wDilation);
            ConvertNcAndWriteArgmax(maxIndexReg, maxIndexConvertRegTep, maxIndexConvertRegT2, u1,
                                    wInputActualAlignedPad, hInputActualPad, left, wInput, hIndexBase, hInput,
                                    dIndexBase, wOffset, num3D, rate4D, argmaxAddrLocal, v2Part, repeatsElem);
        }

        UB_INDEXT wOffsetTail = static_cast<UB_INDEXT>(tailNcInputOffset);
        UB_INDEXT wOutputOffsetTail = static_cast<UB_INDEXT>(tailNcOutputOffset);
        ProcessW(computeAddr, maxValueAddr, wOffsetTail, wInputActualAlignedPad, hInputActualPad, indexReg, dKernel,
                 hKernel, wKernel, tailRepeatsElem, wOutputOffsetTail, maxIndexReg, dDilation, hDilation, wDilation);
        ConvertNcAndWriteArgmax(maxIndexReg, maxIndexConvertRegTep, maxIndexConvertRegT2, u1, wInputActualAlignedPad,
                                hInputActualPad, left, wInput, hIndexBase, hInput, dIndexBase, wOffsetTail, num3D,
                                rate4D, argmaxAddrLocal, v2PartTail, tailRepeatsElem);
    }
    return;
}

template <typename T1, typename T2, const uint32_t IS_PAD, typename UB_INDEXT, typename CAL_INDEXT>
__aicore__ inline void
MaxPool3DWithArgmaxV2GatherKernel<T1, T2, IS_PAD, UB_INDEXT, CAL_INDEXT>::SingleRowGatherForInt64(
    __local_mem__ T1* computeAddr, __local_mem__ T1* maxValueAddr, __local_mem__ T2* argmaxAddr)
{
    uint16_t loopW = wOutputReal_ / vlT1_;
    uint16_t repeatsElem = vlT1_;
    uint16_t tailRepeatsElem = wOutputReal_ - loopW * vlT1_;

    uint16_t dKernel = tilingData_.dKernel;
    uint16_t hKernel = tilingData_.hKernel;
    uint16_t wKernel = tilingData_.wKernel;
    if (tailRepeatsElem == 0) {
        loopW = loopW - 1;
        tailRepeatsElem = repeatsElem;
    }

    uint16_t v2Part[4] = {0, 0, 0, 0};
    uint16_t v2PartTail[4] = {0, 0, 0, 0};
    ComputeV2Parts(v2Part, repeatsElem);
    ComputeV2Parts(v2PartTail, tailRepeatsElem);

    uint32_t dStride = tilingData_.dStride;
    uint32_t hStride = tilingData_.hStride;
    uint32_t wStride = tilingData_.wStride;
    CAL_INDEXT left = wAxisIndex_ * tilingData_.wOutputInner * tilingData_.wStride - tilingData_.padLeft;
    CAL_INDEXT hIndexBase = hAxisIndex_ * tilingData_.hOutputInner * tilingData_.hStride - tilingData_.padTop;
    CAL_INDEXT dIndexBase = dAxisIndex_ * tilingData_.dOutputInner * tilingData_.dStride - tilingData_.padFront;
    CAL_INDEXT hInput = tilingData_.hInput;
    CAL_INDEXT wInput = tilingData_.wInput;
    uint32_t highAxisActual = highAxisActual_;
    uint32_t dInputActualPad = dInputActualPad_;
    uint32_t hInputActualPad = hInputActualPad_;
    uint32_t wInputActualAlignedPad = wInputActualAlignedPad_;
    uint32_t dOutputActual = dOutputReal_;
    uint32_t hOutputActual = hOutputReal_;
    uint32_t wOutputActual = wOutputReal_;
    uint16_t dDilation = tilingData_.dDilation;
    uint16_t hDilation = tilingData_.hDilation;
    uint16_t wDilation = tilingData_.wDilation;

    uint32_t inputPlaneStride = dInputActualPad * hInputActualPad * wInputActualAlignedPad;
    uint32_t outputPlaneStride = dOutputActual * hOutputActual * wOutputActual;
    uint32_t dInputStrideVal = dStride * hInputActualPad * wInputActualAlignedPad;
    uint32_t dOutputStrideVal = hOutputActual * wOutputActual;
    uint32_t hInputStrideVal = wInputActualAlignedPad * hStride;
    uint32_t wInputStep = static_cast<uint32_t>(repeatsElem) * wStride;
    uint32_t tailInputOffset = static_cast<uint32_t>(loopW) * wInputStep;
    uint32_t tailOutputOffset = static_cast<uint32_t>(loopW) * repeatsElem;
    for (uint16_t nc = 0; nc < static_cast<uint16_t>(highAxisActual); nc++) {
        uint32_t ncInputBase = nc * inputPlaneStride;
        uint32_t ncOutBase = nc * outputPlaneStride;
        for (uint16_t dLoop = 0; dLoop < static_cast<uint16_t>(dOutputActual); dLoop++) {
            uint32_t dInBase = dLoop * dInputStrideVal;
            uint32_t dOutBase = dLoop * dOutputStrideVal;
            for (uint16_t hLoop = 0; hLoop < static_cast<uint16_t>(hOutputActual); hLoop++) {
                uint32_t hInBase = hLoop * hInputStrideVal;
                uint32_t hOutBase = hLoop * wOutputActual;
                uint32_t inputBase = ncInputBase + dInBase + hInBase;
                uint32_t outputBase = ncOutBase + dOutBase + hOutBase;
                for (uint16_t wLoop = 0; wLoop < loopW; wLoop++) {
                    UB_INDEXT wOffset = static_cast<UB_INDEXT>(inputBase + wLoop * wInputStep);
                    UB_INDEXT wOutOffset = static_cast<UB_INDEXT>(outputBase + wLoop * repeatsElem);
                    __VEC_SCOPE__
                    {
                        MicroAPI::RegTensor<UB_INDEXT> indexReg;
                        MicroAPI::RegTensor<UB_INDEXT> maxIndexReg;
                        UB_INDEXT ncInputOffset = static_cast<UB_INDEXT>(ncInputBase);
                        __local_mem__ T2* argmaxAddrLocal = argmaxAddr + static_cast<T2>(outputBase) +
                                                            static_cast<T2>(wLoop * repeatsElem);
                        MicroAPI::Arange(indexReg, static_cast<UB_INDEXT>(0));
                        MicroAPI::MaskReg preg = MicroAPI::CreateMask<UB_INDEXT, MicroAPI::MaskPattern::ALL>();
                        MicroAPI::Muls(indexReg, indexReg, static_cast<UB_INDEXT>(wStride), preg);
                        ProcessW(computeAddr, maxValueAddr, wOffset, wInputActualAlignedPad, hInputActualPad, indexReg,
                                 dKernel, hKernel, wKernel, repeatsElem, wOutOffset, maxIndexReg, dDilation, hDilation,
                                 wDilation);

                        ConvertBaseIndexAndWriteArgmax(maxIndexReg, wInputActualAlignedPad, hInputActualPad,
                                                       ncInputOffset, left, wInput, hIndexBase, hInput, dIndexBase,
                                                       argmaxAddrLocal, v2Part, repeatsElem);
                    }
                }
                __VEC_SCOPE__
                {
                    MicroAPI::RegTensor<UB_INDEXT> indexReg;
                    MicroAPI::RegTensor<UB_INDEXT> maxIndexReg;
                    UB_INDEXT ncInputOffset = static_cast<UB_INDEXT>(ncInputBase);
                    __local_mem__ T2* argmaxAddrLocal = argmaxAddr + static_cast<T2>(outputBase) +
                                                        static_cast<T2>(loopW * repeatsElem);
                    MicroAPI::Arange(indexReg, static_cast<UB_INDEXT>(0));
                    MicroAPI::MaskReg preg = MicroAPI::CreateMask<UB_INDEXT, MicroAPI::MaskPattern::ALL>();
                    MicroAPI::Muls(indexReg, indexReg, static_cast<UB_INDEXT>(wStride), preg);

                    UB_INDEXT wOffsetTail = static_cast<UB_INDEXT>(inputBase + tailInputOffset);
                    UB_INDEXT wOutOffsetTail = static_cast<UB_INDEXT>(outputBase + tailOutputOffset);
                    ProcessW(computeAddr, maxValueAddr, wOffsetTail, wInputActualAlignedPad, hInputActualPad, indexReg,
                             dKernel, hKernel, wKernel, tailRepeatsElem, wOutOffsetTail, maxIndexReg, dDilation,
                             hDilation, wDilation);

                    ConvertBaseIndexAndWriteArgmax(maxIndexReg, wInputActualAlignedPad, hInputActualPad, ncInputOffset,
                                                   left, wInput, hIndexBase, hInput, dIndexBase, argmaxAddrLocal,
                                                   v2PartTail, tailRepeatsElem);
                }
            }
        }
    }
    return;
}

template <typename T1, typename T2, const uint32_t IS_PAD, typename UB_INDEXT, typename CAL_INDEXT>
__aicore__ inline void MaxPool3DWithArgmaxV2GatherKernel<T1, T2, IS_PAD, UB_INDEXT, CAL_INDEXT>::MultiRowGatherForInt64(
    __local_mem__ T1* computeAddr, __local_mem__ T1* maxValueAddr, __local_mem__ T2* argmaxAddr)
{
    uint16_t dKernel = tilingData_.dKernel;
    uint32_t wStride = tilingData_.wStride;
    uint16_t wOutputActual = wOutputReal_;
    uint16_t wKernel = tilingData_.wKernel;
    uint16_t hKernel = tilingData_.hKernel;

    uint16_t hBatchCount = vlT1_ / wOutputReal_;
    uint32_t rate2D = wInputActualAlignedPad_ * tilingData_.hStride;
    uint16_t hLoopTimes = hOutputReal_ / hBatchCount;
    uint16_t hTail = hOutputReal_ - hLoopTimes * hBatchCount;
    if (hTail == 0) {
        hLoopTimes = hLoopTimes - 1;
        hTail = hBatchCount;
    }
    uint16_t repeatsElem = hBatchCount * wOutputReal_;
    uint16_t tailRepeatsElem = hTail * wOutputReal_;
    uint16_t v2Part[4] = {0, 0, 0, 0};
    uint16_t v2PartTail[4] = {0, 0, 0, 0};
    ComputeV2Parts(v2Part, repeatsElem);
    ComputeV2Parts(v2PartTail, tailRepeatsElem);

    CAL_INDEXT left = wAxisIndex_ * tilingData_.wOutputInner * tilingData_.wStride - tilingData_.padLeft;
    CAL_INDEXT hIndexBase = hAxisIndex_ * tilingData_.hOutputInner * tilingData_.hStride - tilingData_.padTop;
    CAL_INDEXT dIndexBase = dAxisIndex_ * tilingData_.dOutputInner * tilingData_.dStride - tilingData_.padFront;
    CAL_INDEXT hInput = tilingData_.hInput;
    CAL_INDEXT wInput = tilingData_.wInput;
    uint32_t highAxisActual = highAxisActual_;
    uint32_t dInputActualPad = dInputActualPad_;
    uint32_t hInputActualPad = hInputActualPad_;
    uint32_t wInputActualAlignedPad = wInputActualAlignedPad_;
    uint32_t dOutputActual = dOutputReal_;
    uint32_t hOutputActual = hOutputReal_;
    uint32_t dStride = tilingData_.dStride;
    uint32_t hStride = tilingData_.hStride;
    uint16_t dDilation = tilingData_.dDilation;
    uint16_t hDilation = tilingData_.hDilation;
    uint16_t wDilation = tilingData_.wDilation;
    uint32_t inputPlaneStride = dInputActualPad * hInputActualPad * wInputActualAlignedPad;
    uint32_t outputPlaneStride = dOutputActual * hOutputActual * wOutputActual;
    uint32_t dInputStride = dStride * hInputActualPad * wInputActualAlignedPad;
    uint32_t dOutputStride = hOutputActual * wOutputActual;
    uint32_t hInputStride = static_cast<uint32_t>(hBatchCount) * hStride * wInputActualAlignedPad;
    uint32_t hOutputStride = static_cast<uint32_t>(hBatchCount) * wOutputActual;
    uint32_t tailHInputOffset = static_cast<uint32_t>(hLoopTimes) * hInputStride;
    uint32_t tailHOutputOffset = static_cast<uint32_t>(hLoopTimes) * hOutputStride;
    for (uint16_t nc = 0; nc < static_cast<uint16_t>(highAxisActual); nc++) {
        uint32_t ncInputBase = nc * inputPlaneStride;
        uint32_t ncOutBase = nc * outputPlaneStride;
        for (uint16_t dLoop = 0; dLoop < static_cast<uint16_t>(dOutputActual); dLoop++) {
            uint32_t ncDInputBase = ncInputBase + dLoop * dInputStride;
            uint32_t ncDOutBase = ncOutBase + dLoop * dOutputStride;
            for (uint16_t hLoop = 0; hLoop < hLoopTimes; hLoop++) {
                UB_INDEXT wOffset = static_cast<UB_INDEXT>(ncDInputBase + hLoop * hInputStride);
                UB_INDEXT wOutputOffset = static_cast<UB_INDEXT>(ncDOutBase + hLoop * hOutputStride);
                __VEC_SCOPE__
                {
                    MicroAPI::RegTensor<UB_INDEXT> indexReg;
                    MicroAPI::RegTensor<UB_INDEXT> maxIndexReg;
                    UB_INDEXT ncInputOffset = static_cast<UB_INDEXT>(ncInputBase);
                    __local_mem__ T2* argmaxAddrLocal = argmaxAddr + static_cast<T2>(ncDOutBase) +
                                                        static_cast<T2>(hLoop * hOutputStride);
                    CalGatterIndex2D<UB_INDEXT>(indexReg, rate2D, wOutputActual, wStride);
                    ProcessW(computeAddr, maxValueAddr, wOffset, wInputActualAlignedPad, hInputActualPad, indexReg,
                             dKernel, hKernel, wKernel, repeatsElem, wOutputOffset, maxIndexReg, dDilation, hDilation,
                             wDilation);
                    ConvertBaseIndexAndWriteArgmax(maxIndexReg, wInputActualAlignedPad, hInputActualPad, ncInputOffset,
                                                   left, wInput, hIndexBase, hInput, dIndexBase, argmaxAddrLocal,
                                                   v2Part, repeatsElem);
                }
            }
            __VEC_SCOPE__
            {
                MicroAPI::RegTensor<UB_INDEXT> indexReg;
                MicroAPI::RegTensor<UB_INDEXT> maxIndexReg;
                UB_INDEXT ncInputOffset = static_cast<UB_INDEXT>(ncInputBase);
                __local_mem__ T2* argmaxAddrLocal = argmaxAddr + static_cast<T2>(ncDOutBase) +
                                                    static_cast<T2>(hLoopTimes * hOutputStride);
                CalGatterIndex2D<UB_INDEXT>(indexReg, rate2D, wOutputActual, wStride);

                UB_INDEXT wOffsetTail = static_cast<UB_INDEXT>(ncDInputBase + tailHInputOffset);
                UB_INDEXT wOutputOffsetTail = static_cast<UB_INDEXT>(ncDOutBase + tailHOutputOffset);
                ProcessW(computeAddr, maxValueAddr, wOffsetTail, wInputActualAlignedPad, hInputActualPad, indexReg,
                         dKernel, hKernel, wKernel, tailRepeatsElem, wOutputOffsetTail, maxIndexReg, dDilation,
                         hDilation, wDilation);
                ConvertBaseIndexAndWriteArgmax(maxIndexReg, wInputActualAlignedPad, hInputActualPad, ncInputOffset,
                                               left, wInput, hIndexBase, hInput, dIndexBase, argmaxAddrLocal,
                                               v2PartTail, tailRepeatsElem);
            }
        }
    }
    return;
}

template <typename T1, typename T2, const uint32_t IS_PAD, typename UB_INDEXT, typename CAL_INDEXT>
__aicore__ inline void MaxPool3DWithArgmaxV2GatherKernel<T1, T2, IS_PAD, UB_INDEXT, CAL_INDEXT>::MultiDepGatherForInt64(
    __local_mem__ T1* computeAddr, __local_mem__ T1* maxValueAddr, __local_mem__ T2* argmaxAddr)
{
    uint16_t wKernel = tilingData_.wKernel;
    uint16_t hKernel = tilingData_.hKernel;
    uint16_t dKernel = tilingData_.dKernel;
    uint32_t wStride = tilingData_.wStride;
    uint32_t rate3D = tilingData_.dStride * hInputActualPad_ * wInputActualAlignedPad_;
    uint16_t num2D = hOutputReal_ * wOutputReal_;
    uint32_t rate2D = tilingData_.hStride * wInputActualAlignedPad_;
    uint16_t wOutputActual = wOutputReal_;
    uint16_t eachDepCount = hOutputReal_ * wOutputReal_;
    uint16_t dBatchCount = vlT1_ / eachDepCount;
    uint16_t dLoopTimes = dOutputReal_ / dBatchCount;
    uint16_t dTail = dOutputReal_ - dLoopTimes * dBatchCount;
    if (dTail == 0) {
        dLoopTimes = dLoopTimes - 1;
        dTail = dBatchCount;
    }
    uint16_t repeatsElem = dBatchCount * eachDepCount;
    uint16_t tailRepeatsElem = dTail * eachDepCount;

    uint16_t v2Part[4] = {0, 0, 0, 0};
    uint16_t v2PartTail[4] = {0, 0, 0, 0};
    ComputeV2Parts(v2Part, repeatsElem);
    ComputeV2Parts(v2PartTail, tailRepeatsElem);

    CAL_INDEXT left = wAxisIndex_ * tilingData_.wOutputInner * tilingData_.wStride - tilingData_.padLeft;
    CAL_INDEXT hIndexBase = hAxisIndex_ * tilingData_.hOutputInner * tilingData_.hStride - tilingData_.padTop;
    CAL_INDEXT dIndexBase = dAxisIndex_ * tilingData_.dOutputInner * tilingData_.dStride - tilingData_.padFront;
    CAL_INDEXT hInput = tilingData_.hInput;
    CAL_INDEXT wInput = tilingData_.wInput;
    uint32_t highAxisActual = highAxisActual_;
    uint32_t dInputActualPad = dInputActualPad_;
    uint32_t hInputActualPad = hInputActualPad_;
    uint32_t wInputActualAlignedPad = wInputActualAlignedPad_;
    uint32_t dOutputActual = dOutputReal_;
    uint32_t hOutputActual = hOutputReal_;
    uint32_t dStride = tilingData_.dStride;
    uint32_t hStride = tilingData_.hStride;
    uint16_t dDilation = tilingData_.dDilation;
    uint16_t hDilation = tilingData_.hDilation;
    uint16_t wDilation = tilingData_.wDilation;

    uint32_t inputPlaneStride = dInputActualPad * hInputActualPad * wInputActualAlignedPad;
    uint32_t outputPlaneStride = dOutputActual * hOutputActual * wOutputActual;
    uint32_t dBatchInputStride = static_cast<uint32_t>(dBatchCount) * dStride * hInputActualPad *
                                 wInputActualAlignedPad;
    uint32_t dBatchOutputStride = static_cast<uint32_t>(dBatchCount) * hOutputActual * wOutputActual;
    uint32_t tailDInputOffset = static_cast<uint32_t>(dLoopTimes) * dBatchInputStride;
    uint32_t tailDOutputOffset = static_cast<uint32_t>(dLoopTimes) * dBatchOutputStride;
    for (uint16_t nc = 0; nc < static_cast<uint16_t>(highAxisActual); nc++) {
        uint32_t ncInputBase = nc * inputPlaneStride;
        uint32_t ncOutBase = nc * outputPlaneStride;

        for (uint16_t dLoop = 0; dLoop < static_cast<uint16_t>(dLoopTimes); dLoop++) {
            UB_INDEXT wOffset = static_cast<UB_INDEXT>(ncInputBase + dLoop * dBatchInputStride);
            UB_INDEXT wOutputOffset = static_cast<UB_INDEXT>(ncOutBase + dLoop * dBatchOutputStride);
            __VEC_SCOPE__
            {
                MicroAPI::RegTensor<UB_INDEXT> indexReg;
                MicroAPI::RegTensor<UB_INDEXT> maxIndexReg;
                UB_INDEXT ncInputOffset = static_cast<UB_INDEXT>(ncInputBase);
                __local_mem__ T2* argmaxAddrLocal = argmaxAddr + static_cast<T2>(ncOutBase) +
                                                    static_cast<T2>(dLoop * dBatchOutputStride);
                CalGatterIndex3D<UB_INDEXT>(indexReg, rate3D, num2D, rate2D, wOutputActual, wStride);
                ProcessW(computeAddr, maxValueAddr, wOffset, wInputActualAlignedPad, hInputActualPad, indexReg, dKernel,
                         hKernel, wKernel, repeatsElem, wOutputOffset, maxIndexReg, dDilation, hDilation, wDilation);
                ConvertBaseIndexAndWriteArgmax(maxIndexReg, wInputActualAlignedPad, hInputActualPad, ncInputOffset,
                                               left, wInput, hIndexBase, hInput, dIndexBase, argmaxAddrLocal, v2Part,
                                               repeatsElem);
            }
        }
        __VEC_SCOPE__
        {
            MicroAPI::RegTensor<UB_INDEXT> indexReg;
            MicroAPI::RegTensor<UB_INDEXT> maxIndexReg;
            UB_INDEXT ncInputOffset = static_cast<UB_INDEXT>(ncInputBase);
            __local_mem__ T2* argmaxAddrLocal = argmaxAddr + static_cast<T2>(ncOutBase) +
                                                static_cast<T2>(dLoopTimes * dBatchOutputStride);
            CalGatterIndex3D<UB_INDEXT>(indexReg, rate3D, num2D, rate2D, wOutputActual, wStride);
            UB_INDEXT wOffsetTail = static_cast<UB_INDEXT>(ncInputBase + tailDInputOffset);
            UB_INDEXT wOutputOffsetTail = static_cast<UB_INDEXT>(ncOutBase + tailDOutputOffset);
            ProcessW(computeAddr, maxValueAddr, wOffsetTail, wInputActualAlignedPad, hInputActualPad, indexReg, dKernel,
                     hKernel, wKernel, tailRepeatsElem, wOutputOffsetTail, maxIndexReg, dDilation, hDilation,
                     wDilation);
            ConvertBaseIndexAndWriteArgmax(maxIndexReg, wInputActualAlignedPad, hInputActualPad, ncInputOffset, left,
                                           wInput, hIndexBase, hInput, dIndexBase, argmaxAddrLocal, v2PartTail,
                                           tailRepeatsElem);
        }
    }
    return;
}

template <typename T1, typename T2, const uint32_t IS_PAD, typename UB_INDEXT, typename CAL_INDEXT>
__aicore__ inline void MaxPool3DWithArgmaxV2GatherKernel<T1, T2, IS_PAD, UB_INDEXT, CAL_INDEXT>::MultiNcGatherForInt64(
    __local_mem__ T1* computeAddr, __local_mem__ T1* maxValueAddr, __local_mem__ T2* argmaxAddr)
{
    uint16_t dKernel = tilingData_.dKernel;
    uint16_t hKernel = tilingData_.hKernel;
    uint16_t wKernel = tilingData_.wKernel;
    uint32_t wStride = tilingData_.wStride;
    uint32_t rate4D = dInputActualPad_ * hInputActualPad_ * wInputActualAlignedPad_;
    uint32_t num3D = dOutputReal_ * hOutputReal_ * wOutputReal_;
    uint32_t rate3D = tilingData_.dStride * hInputActualPad_ * wInputActualAlignedPad_;
    uint16_t num2D = hOutputReal_ * wOutputReal_;
    uint32_t rate2D = tilingData_.hStride * wInputActualAlignedPad_;
    uint16_t wOutputActual = wOutputReal_;
    uint16_t eachBatchCount = dOutputReal_ * hOutputReal_ * wOutputReal_;
    uint16_t ncBatchCount = vlT1_ / eachBatchCount;
    uint16_t ncLoopTimes = highAxisActual_ / ncBatchCount;
    uint16_t ncTail = highAxisActual_ - ncLoopTimes * ncBatchCount;
    if (ncTail == 0) {
        ncLoopTimes = ncLoopTimes - 1;
        ncTail = ncBatchCount;
    }
    uint16_t repeatsElem = ncBatchCount * eachBatchCount;
    uint16_t tailRepeatsElem = ncTail * eachBatchCount;

    uint16_t v2Part[4] = {0, 0, 0, 0};
    uint16_t v2PartTail[4] = {0, 0, 0, 0};
    ComputeV2Parts(v2Part, repeatsElem);
    ComputeV2Parts(v2PartTail, tailRepeatsElem);

    CAL_INDEXT left = wAxisIndex_ * tilingData_.wOutputInner * tilingData_.wStride - tilingData_.padLeft;
    CAL_INDEXT hIndexBase = hAxisIndex_ * tilingData_.hOutputInner * tilingData_.hStride - tilingData_.padTop;
    CAL_INDEXT dIndexBase = dAxisIndex_ * tilingData_.dOutputInner * tilingData_.dStride - tilingData_.padFront;
    CAL_INDEXT hInput = tilingData_.hInput;
    CAL_INDEXT wInput = tilingData_.wInput;
    uint32_t dInputActualPad = dInputActualPad_;
    uint32_t hInputActualPad = hInputActualPad_;
    uint32_t wInputActualAlignedPad = wInputActualAlignedPad_;
    uint32_t dOutputActual = dOutputReal_;
    uint32_t hOutputActual = hOutputReal_;
    uint32_t dStride = tilingData_.dStride;
    uint32_t hStride = tilingData_.hStride;
    uint16_t dDilation = tilingData_.dDilation;
    uint16_t hDilation = tilingData_.hDilation;
    uint16_t wDilation = tilingData_.wDilation;

    uint32_t wInputStrideU32 = ncBatchCount * dInputActualPad * hInputActualPad * wInputActualAlignedPad;
    uint32_t wOutStrideU32 = ncBatchCount * dOutputActual * hOutputActual * wOutputActual;
    uint32_t tailNcInputOffset = static_cast<uint32_t>(ncLoopTimes) * wInputStrideU32;
    uint32_t tailNcOutputOffset = static_cast<uint32_t>(ncLoopTimes) * wOutStrideU32;
    for (uint16_t ncLoop = 0; ncLoop < ncLoopTimes; ncLoop++) {
        UB_INDEXT wOffset = static_cast<UB_INDEXT>(ncLoop * wInputStrideU32);
        UB_INDEXT wOutputOffset = static_cast<UB_INDEXT>(ncLoop * wOutStrideU32);
        __VEC_SCOPE__
        {
            MicroAPI::RegTensor<UB_INDEXT> indexReg;
            MicroAPI::RegTensor<UB_INDEXT> maxIndexReg;
            MicroAPI::RegTensor<int32_t> maxIndexConvertRegTep;
            MicroAPI::RegTensor<T2> maxIndexConvertRegT2;
            MicroAPI::UnalignReg u1;
            __local_mem__ T2* argmaxAddrLocal = argmaxAddr + static_cast<T2>(ncLoop * wOutStrideU32);
            CalGatterIndex4D<UB_INDEXT>(indexReg, rate4D, num3D, rate3D, num2D, rate2D, wOutputActual, wStride);
            ProcessW(computeAddr, maxValueAddr, wOffset, wInputActualAlignedPad, hInputActualPad, indexReg, dKernel,
                     hKernel, wKernel, repeatsElem, wOutputOffset, maxIndexReg, dDilation, hDilation, wDilation);
            ConvertNcAndWriteArgmax(maxIndexReg, maxIndexConvertRegTep, maxIndexConvertRegT2, u1,
                                    wInputActualAlignedPad, hInputActualPad, left, wInput, hIndexBase, hInput,
                                    dIndexBase, wOffset, num3D, rate4D, argmaxAddrLocal, v2Part, repeatsElem);
        }
    }
    __VEC_SCOPE__
    {
        MicroAPI::RegTensor<UB_INDEXT> indexReg;
        MicroAPI::RegTensor<UB_INDEXT> maxIndexReg;
        MicroAPI::RegTensor<int32_t> maxIndexConvertRegTep;
        MicroAPI::RegTensor<T2> maxIndexConvertRegT2;
        MicroAPI::UnalignReg u1;
        __local_mem__ T2* argmaxAddrLocal = argmaxAddr + static_cast<T2>(ncLoopTimes * wOutStrideU32);
        CalGatterIndex4D<UB_INDEXT>(indexReg, rate4D, num3D, rate3D, num2D, rate2D, wOutputActual, wStride);

        UB_INDEXT wOffsetTail = static_cast<UB_INDEXT>(tailNcInputOffset);
        UB_INDEXT wOutputOffsetTail = static_cast<UB_INDEXT>(tailNcOutputOffset);
        ProcessW(computeAddr, maxValueAddr, wOffsetTail, wInputActualAlignedPad, hInputActualPad, indexReg, dKernel,
                 hKernel, wKernel, tailRepeatsElem, wOutputOffsetTail, maxIndexReg, dDilation, hDilation, wDilation);
        ConvertNcAndWriteArgmax(maxIndexReg, maxIndexConvertRegTep, maxIndexConvertRegT2, u1, wInputActualAlignedPad,
                                hInputActualPad, left, wInput, hIndexBase, hInput, dIndexBase, wOffsetTail, num3D,
                                rate4D, argmaxAddrLocal, v2PartTail, tailRepeatsElem);
    }
    return;
}

template <typename T1, typename T2, const uint32_t IS_PAD, typename UB_INDEXT, typename CAL_INDEXT>
__aicore__ inline void MaxPool3DWithArgmaxV2GatherKernel<T1, T2, IS_PAD, UB_INDEXT, CAL_INDEXT>::CopyOut()
{
    LocalTensor<T1> maxValueLocal = maxValueQue_.DeQue<T1>();
    LocalTensor<T2> argmaxLocal = argmaxQue_.DeQue<T2>();
    int64_t outPlaneSize = tilingData_.dOutput * tilingData_.hOutput * tilingData_.wOutput;
    int64_t highOutOffset = highAxisIndex_ * tilingData_.highAxisInner * outPlaneSize;
    int64_t dOutputOffset = dAxisIndex_ * tilingData_.dOutputInner * tilingData_.hOutput * tilingData_.wOutput;
    int64_t hOutputOffset = hAxisIndex_ * tilingData_.hOutputInner * tilingData_.wOutput;
    int64_t wOutputOffset = wAxisIndex_ * tilingData_.wOutputInner;
    int64_t outGmOffset = highOutOffset + dOutputOffset + hOutputOffset + wOutputOffset;

    DataCopyExtParams copyOutParamT1 = {
        static_cast<uint16_t>(1),
        static_cast<uint32_t>(highAxisActual_ * dOutputReal_ * hOutputReal_ * wOutputReal_ * sizeof(T1)),
        static_cast<uint32_t>(0), static_cast<uint32_t>(0), static_cast<uint32_t>(0)};

    DataCopyPad(yGm_[outGmOffset], maxValueLocal, copyOutParamT1);
    DataCopyExtParams copyOutParamT2 = {
        static_cast<uint16_t>(1),
        static_cast<uint32_t>(highAxisActual_ * dOutputReal_ * hOutputReal_ * wOutputReal_ * sizeof(T2)),
        static_cast<uint32_t>(0), static_cast<uint32_t>(0), static_cast<uint32_t>(0)};
    DataCopyPad(argmaxGm_[outGmOffset], argmaxLocal, copyOutParamT2);
    maxValueQue_.FreeTensor(maxValueLocal);
    argmaxQue_.FreeTensor(argmaxLocal);
    return;
}

} // namespace MaxPool3DWithArgmaxV2GatherNameSpace
#endif // MAX_POOL3D_WITH_ARGMAX_V2_GATHER_H_
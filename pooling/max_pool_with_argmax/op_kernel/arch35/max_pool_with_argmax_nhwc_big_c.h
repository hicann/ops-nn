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
 * \file max_pool_with_argmax_nhwc_big_c.h
 * \brief
 */

#ifndef MAX_POOL_WITH_ARGMAX_NHWC_BIG_C_H_
#define MAX_POOL_WITH_ARGMAX_NHWC_BIG_C_H_

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "../inc/platform.h"
#include "max_pool_with_argmax_base.h"
#include "max_pool_with_argmax_struct_common.h"

namespace MaxPoolWithArgmaxNHWC
{
using namespace AscendC;

constexpr uint32_t BUFFER_NUM = 2;
constexpr int64_t HELPER_BUFFER_SIZE = platform::GetVRegSize() * 5;
constexpr int64_t HELPER_BUFFER_SIZE_512 = 512;
constexpr int64_t THREE_DIM = 3;
constexpr int64_t DIGIT_1 = 1;
constexpr int64_t DIGIT_2 = 2;

template <typename T1, typename T2, uint32_t IS_PAD = 0, uint32_t NANPROP = 0>
class MaxPoolWithArgmaxNhwCKernel
{
public:
    __aicore__ inline MaxPoolWithArgmaxNhwCKernel(
        TPipe* pipe,
        const MaxPoolWithArgmaxCommonStructNameSpace::MaxPoolWithArgmaxNHWCTilingCommonData* __restrict tiling)
        : pipe_(pipe), tilingData_(tiling){};

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR argmax);
    __aicore__ inline void ParseTilingData(
        const MaxPoolWithArgmaxCommonStructNameSpace::MaxPoolWithArgmaxNHWCTilingCommonData& tilingData);
    __aicore__ inline void Process();
    __aicore__ inline void ScalarCompute(int64_t loopNum);
    __aicore__ inline void CopyIn();
    __aicore__ inline void FillPadNegVF(__local_mem__ T1* xLocalAddr);
    __aicore__ inline void Compute(__local_mem__ T1* maxValueLocal, __local_mem__ T2* argmaxLocal);

    __aicore__ inline void InitHelpBuf();
    __aicore__ inline void CopyResultToUb(__local_mem__ T1* maxValueLocal, __local_mem__ T2* argmaxLocal);

    template <const bool IS_SPLIT_KERNEL>
    __aicore__ inline void MaxPoolAndArgmaxVF(__local_mem__ T1* xLocal, __local_mem__ T1* maxValueLocal,
                                                __local_mem__ T2* argmaxLocal);
    template <const bool IS_SPLIT_KERNEL>
    __aicore__ inline void MaxPoolAndArgmaxVFForPad(__local_mem__ T1* xLocal, __local_mem__ T1* maxValueLocal,
                                                      __local_mem__ T2* argmaxLocal);
    __aicore__ inline void CopyOut();

    TPipe* pipe_;
    TQue<QuePosition::VECIN, BUFFER_NUM> inputQue_;
    TQue<QuePosition::VECOUT, BUFFER_NUM> maxValueQue_;
    TQue<QuePosition::VECOUT, BUFFER_NUM> argmaxQue_;
    TBuf<TPosition::VECCALC> helperTBuf_;

    GlobalTensor<T1> xGm_;
    GlobalTensor<T1> yGm_;
    GlobalTensor<T2> argmaxGm_;

    const MaxPoolWithArgmaxCommonStructNameSpace::MaxPoolWithArgmaxNHWCTilingCommonData* tilingData_;
    uint32_t blockIdx_ = 0;

    constexpr static int32_t BLOCK_SIZE = platform::GetUbBlockSize();
    constexpr static int64_t MAX_DATA_NUM_IN_ONE_BLOCK =
        BLOCK_SIZE / sizeof(T1) >= BLOCK_SIZE / sizeof(T2) ? BLOCK_SIZE / sizeof(T1) : BLOCK_SIZE / sizeof(T2);
    constexpr static int64_t VREG_LENGTH_DATA_NUM_T2 = platform::GetVRegSize() / sizeof(T2);

    // tilingdata
    int64_t cInput_ = 0;
    int64_t hInput_ = 0;
    int64_t wInput_ = 0;
    int64_t hOutput_ = 0;
    int64_t wOutput_ = 0;
    int64_t hKernel_ = 0;
    int64_t wKernel_ = 0;
    int64_t hStride_ = 0;
    int64_t wStride_ = 0;
    int64_t padLeft_ = 0;
    int64_t padTop_ = 0;
    int64_t nOutputInner_ = 0;
    int64_t nOutputTail_ = 0;
    int64_t nOutputOuter_ = 0;
    int64_t hOutputInner_ = 0;
    int64_t hOutputTail_ = 0;
    int64_t hOutputOuter_ = 0;
    int64_t wOutputInner_ = 0;
    int64_t wOutputTail_ = 0;
    int64_t wOutputOuter_ = 0;
    int64_t cOutputInner_ = 0;
    int64_t cOutputTail_ = 0;
    int64_t cOutputOuter_ = 0;
    int64_t normalCoreProcessNum_ = 0;
    int64_t tailCoreProcessNum_ = 0;
    int64_t usedCoreNum_ = 0;
    int64_t inputBufferSize_ = 0;
    int64_t maxValueBufferSize_ = 0;
    int64_t argmaxBufferSize_ = 0;
    int64_t isPad_ = 0;
    int64_t isSplitKernel_ = 0;
    int64_t hKernelInner_ = 0;
    int64_t hKernelTail_ = 0;
    int64_t hKernelOuter_ = 0;
    int64_t wKernelInner_ = 0;
    int64_t wKernelTail_ = 0;
    int64_t wKernelOuter_ = 0;

    // 输出域大小
    int64_t nOutputActual_ = 1;
    int64_t hOutputActual_ = 1;
    int64_t wOutputActual_ = 1;
    int64_t cOutputActual_ = 1;

    // c轴对齐到BlockSize
    int64_t cOutputActualAlign_ = 0;

    // 输入域大小
    int64_t hInputActual_ = 1;
    int64_t wInputActual_ = 1;
    // 输入域大小包含前后pad大小
    int64_t hInputActualPad_ = 1;
    int64_t wInputActualPad_ = 1;

    // 输入相对偏移
    int64_t nInputAxisOffset_ = 0;
    int64_t cInputAxisOffset_ = 0;
    int64_t hInputAxisOffset_ = 0;
    int64_t wInputAxisOffset_ = 0;
    // 输出相对偏移
    int64_t nOutputAxisOffset_ = 0;
    int64_t cOutputAxisOffset_ = 0;
    int64_t hOutputAxisOffset_ = 0;
    int64_t wOutputAxisOffset_ = 0;

    // WH轴索引起始点
    int64_t indexWHplanOffset_ = 0;
    // 轴循环index
    int64_t nAxisIndex_ = 0;
    int64_t cAxisIndex_ = 0;
    int64_t hAxisIndex_ = 0;
    int64_t wAxisIndex_ = 0;

    // 切kernel时kernel循环、大小
    int64_t hKernelIndex_ = 0;
    int64_t wKernelIndex_ = 0;
    int64_t hKernelActual_ = 0;
    int64_t wKernelActual_ = 0;

    // 存在pad时，上下左右偏移
    int64_t baseBlockLeftOffsetInOcean_ = 0;
    int64_t baseBlockRightOffsetInOcean_ = 0;
    int64_t baseBlockTopOffsetInOcean_ = 0;
    int64_t baseBlockDownOffsetInOcean_ = 0;

    // 存在pad时，上下左右偏移
    int64_t xLocalUbOffset = 0;
};

template <typename T1, typename T2, uint32_t IS_PAD, uint32_t NANPROP>
__aicore__ inline void MaxPoolWithArgmaxNhwCKernel<T1, T2, IS_PAD, NANPROP>::ParseTilingData(
    const MaxPoolWithArgmaxCommonStructNameSpace::MaxPoolWithArgmaxNHWCTilingCommonData& tilingData)
{
    cInput_ = tilingData.cInput;
    hInput_ = tilingData.hInput;
    wInput_ = tilingData.wInput;
    hOutput_ = tilingData.hOutput;
    wOutput_ = tilingData.wOutput;
    hKernel_ = tilingData.hKernel;
    wKernel_ = tilingData.wKernel;
    hStride_ = tilingData.hStride;
    wStride_ = tilingData.wStride;
    padLeft_ = tilingData.padLeft;
    padTop_ = tilingData.padTop;
    nOutputInner_ = tilingData.nOutputInner;
    nOutputTail_ = tilingData.nOutputTail;
    nOutputOuter_ = tilingData.nOutputOuter;
    hOutputInner_ = tilingData.hOutputInner;
    hOutputTail_ = tilingData.hOutputTail;
    hOutputOuter_ = tilingData.hOutputOuter;
    wOutputInner_ = tilingData.wOutputInner;
    wOutputTail_ = tilingData.wOutputTail;
    wOutputOuter_ = tilingData.wOutputOuter;
    cOutputInner_ = tilingData.cOutputInner;
    cOutputTail_ = tilingData.cOutputTail;
    cOutputOuter_ = tilingData.cOutputOuter;
    normalCoreProcessNum_ = tilingData.normalCoreProcessNum;
    tailCoreProcessNum_ = tilingData.tailCoreProcessNum;
    usedCoreNum_ = tilingData.usedCoreNum;
    inputBufferSize_ = tilingData.inputBufferSize;
    maxValueBufferSize_ = tilingData.maxValueBufferSize;
    argmaxBufferSize_ = tilingData.argmaxBufferSize;
    isPad_ = tilingData.isPad;
    isSplitKernel_ = tilingData.isSplitKernel;
    hKernelInner_ = tilingData.hKernelInner;
    hKernelTail_ = tilingData.hKernelTail;
    hKernelOuter_ = tilingData.hKernelOuter;
    wKernelInner_ = tilingData.wKernelInner;
    wKernelTail_ = tilingData.wKernelTail;
    wKernelOuter_ = tilingData.wKernelOuter;
}

template <typename T1, typename T2, uint32_t IS_PAD, uint32_t NANPROP>
__aicore__ inline void MaxPoolWithArgmaxNhwCKernel<T1, T2, IS_PAD, NANPROP>::Init(GM_ADDR x, GM_ADDR y, GM_ADDR argmax)
{
    ParseTilingData(*tilingData_);
    blockIdx_ = GetBlockIdx();
    if (blockIdx_ >= usedCoreNum_) {
        return;
    }

    xGm_.SetGlobalBuffer((__gm__ T1*)x);
    yGm_.SetGlobalBuffer((__gm__ T1*)y);
    argmaxGm_.SetGlobalBuffer((__gm__ T2*)argmax);

    pipe_->InitBuffer(inputQue_, BUFFER_NUM, inputBufferSize_);
    pipe_->InitBuffer(maxValueQue_, BUFFER_NUM, maxValueBufferSize_);
    pipe_->InitBuffer(argmaxQue_, BUFFER_NUM, argmaxBufferSize_);
    pipe_->InitBuffer(helperTBuf_, HELPER_BUFFER_SIZE);
}

template <typename T1, typename T2, uint32_t IS_PAD, uint32_t NANPROP>
__aicore__ inline void MaxPoolWithArgmaxNhwCKernel<T1, T2, IS_PAD, NANPROP>::ScalarCompute(int64_t loopNum)
{
    int64_t baseBlockIdx = blockIdx_ * normalCoreProcessNum_ + loopNum;
    int64_t hwc = hOutputOuter_ * wOutputOuter_ * cOutputOuter_;
    int64_t wc = wOutputOuter_ * cOutputOuter_;

    nAxisIndex_ = baseBlockIdx / hwc;
    baseBlockIdx = baseBlockIdx % hwc;
    hAxisIndex_ = baseBlockIdx / wc;
    baseBlockIdx = baseBlockIdx % wc;
    wAxisIndex_ = baseBlockIdx / cOutputOuter_;
    cAxisIndex_ = baseBlockIdx % cOutputOuter_;

    nOutputActual_ = nAxisIndex_ == (nOutputOuter_ - 1) ? nOutputTail_ : nOutputInner_;
    hOutputActual_ = hAxisIndex_ == (hOutputOuter_ - 1) ? hOutputTail_ : hOutputInner_;
    wOutputActual_ = wAxisIndex_ == (wOutputOuter_ - 1) ? wOutputTail_ : wOutputInner_;
    cOutputActual_ = cAxisIndex_ == (cOutputOuter_ - 1) ? cOutputTail_ : cOutputInner_;

    cOutputActualAlign_ = ops::Aligned(cOutputActual_, int64_t(BLOCK_SIZE / sizeof(T1)));

    hInputActual_ = (hOutputActual_ - 1) * hStride_ + hKernel_;
    wInputActual_ = (wOutputActual_ - 1) * wStride_ + wKernel_;
    // 输入相对偏移
    cInputAxisOffset_ = cAxisIndex_ * cOutputInner_;
    wInputAxisOffset_ = (wAxisIndex_ * wStride_ * wOutputInner_ + wKernelIndex_ * wKernelInner_) * cInput_;
    hInputAxisOffset_ = (hAxisIndex_ * hStride_ * hOutputInner_ + hKernelIndex_ * hKernelInner_) * wInput_ * cInput_;
    nInputAxisOffset_ = nAxisIndex_ * nOutputInner_ * hInput_ * wInput_ * cInput_;

    // 输出相对偏移
    cOutputAxisOffset_ = cAxisIndex_ * cOutputInner_;
    wOutputAxisOffset_ = wAxisIndex_ * wOutputInner_ * cInput_;
    hOutputAxisOffset_ = hAxisIndex_ * hOutputInner_ * wOutput_ * cInput_;
    nOutputAxisOffset_ = nAxisIndex_ * nOutputInner_ * hOutput_ * wOutput_ * cInput_;

    hKernelActual_ = hKernel_;
    wKernelActual_ = wKernel_;

    // kernel切分
    if (isSplitKernel_ == 1) {
        wInputActual_ = wKernelIndex_ == (wKernelOuter_ - 1) ? wKernelTail_ : wKernelInner_;
        hInputActual_ = hKernelIndex_ == (hKernelOuter_ - 1) ? hKernelTail_ : hKernelInner_;
        hKernelActual_ = hInputActual_;
        wKernelActual_ = wInputActual_;
    }
    hInputActualPad_ = hInputActual_;
    wInputActualPad_ = wInputActual_;

    if constexpr (IS_PAD == 1) {
        int64_t topOffset = hAxisIndex_ * hOutputInner_ * hStride_ + hKernelIndex_ * hKernelInner_ - padTop_;
        int64_t downOffset = hAxisIndex_ * hOutputInner_ * hStride_ + hKernelIndex_ * hKernelInner_ +
                             (hOutputActual_ - 1) * hStride_ + hKernelActual_ - hInput_ - padTop_;
        int64_t leftOffset = wAxisIndex_ * wOutputInner_ * wStride_ + wKernelIndex_ * wKernelInner_ - padLeft_;
        int64_t rightOffset = wAxisIndex_ * wOutputInner_ * wStride_ + wKernelIndex_ * wKernelInner_ +
                              (wOutputActual_ - 1) * wStride_ + wKernelActual_ - wInput_ - padLeft_;

        baseBlockLeftOffsetInOcean_ = leftOffset >= 0 ? 0 : -leftOffset;
        baseBlockRightOffsetInOcean_ = rightOffset >= 0 ? rightOffset : 0;
        baseBlockTopOffsetInOcean_ = topOffset >= 0 ? 0 : -topOffset;
        baseBlockDownOffsetInOcean_ = downOffset >= 0 ? downOffset : 0;
        // PAD时输入偏移
        xLocalUbOffset = baseBlockTopOffsetInOcean_ * wInputActual_ * cOutputActualAlign_ +
                         baseBlockLeftOffsetInOcean_ * cOutputActualAlign_;

        hInputActual_ = hInputActual_ - baseBlockTopOffsetInOcean_ - baseBlockDownOffsetInOcean_;
        wInputActual_ = wInputActual_ - baseBlockLeftOffsetInOcean_ - baseBlockRightOffsetInOcean_;

        hInputAxisOffset_ = baseBlockTopOffsetInOcean_ == 0 ? hInputAxisOffset_ - padTop_ * wInput_ * cInput_ : 0;
        wInputAxisOffset_ = baseBlockLeftOffsetInOcean_ == 0 ? wInputAxisOffset_ - padLeft_ * cInput_ : 0;
    }
    indexWHplanOffset_ = (hInputAxisOffset_ + wInputAxisOffset_) / cInput_;
}

template <typename T1, typename T2, uint32_t IS_PAD, uint32_t NANPROP>
__aicore__ inline void MaxPoolWithArgmaxNhwCKernel<T1, T2, IS_PAD, NANPROP>::InitHelpBuf()
{
    __local_mem__ T1* maxValueHelp = (__local_mem__ T1*)helperTBuf_.Get<T1>().GetPhyAddr();
    __local_mem__ T2* argmaxHelp =
        (__local_mem__ T2*)helperTBuf_.Get<T2>().GetPhyAddr() + HELPER_BUFFER_SIZE_512 / sizeof(T1);

    __VEC_SCOPE__
    {
        AscendC::MicroAPI::RegTensor<T1> negInfReg;
        AscendC::MicroAPI::RegTensor<T2> negOne;
        DuplicateLowestReg(negInfReg);
        AscendC::MicroAPI::Duplicate(negOne, T2(-1));
        AscendC::MicroAPI::MaskReg pregAll = AscendC::MicroAPI::CreateMask<T1, AscendC::MicroAPI::MaskPattern::ALL>();
        AscendC::MicroAPI::MaskReg pregAllT2 = AscendC::MicroAPI::CreateMask<T2, AscendC::MicroAPI::MaskPattern::ALL>();
        AscendC::MicroAPI::DataCopy(maxValueHelp, negInfReg, pregAll);
        AscendC::MicroAPI::DataCopy(argmaxHelp, negOne, pregAllT2);
    }
}

template <typename T1, typename T2, uint32_t IS_PAD, uint32_t NANPROP>
__aicore__ inline void MaxPoolWithArgmaxNhwCKernel<T1, T2, IS_PAD, NANPROP>::CopyResultToUb(__local_mem__ T1* maxValueLocal,
                                                                                     __local_mem__ T2* argmaxLocal)
{
    __local_mem__ T1* maxValueHelp = (__local_mem__ T1*)helperTBuf_.Get<T1>().GetPhyAddr();
    __local_mem__ T2* argmaxHelp =
        (__local_mem__ T2*)helperTBuf_.Get<T2>().GetPhyAddr() + HELPER_BUFFER_SIZE_512 / sizeof(T1);

    __VEC_SCOPE__
    {
        AscendC::MicroAPI::RegTensor<T1> vreg0;
        AscendC::MicroAPI::RegTensor<T2> argmaxUpdateVreg;
        AscendC::MicroAPI::MaskReg pregAllT1 = AscendC::MicroAPI::CreateMask<T1, AscendC::MicroAPI::MaskPattern::ALL>();
        AscendC::MicroAPI::MaskReg pregAllT2 = AscendC::MicroAPI::CreateMask<T2, AscendC::MicroAPI::MaskPattern::ALL>();
        AscendC::MicroAPI::DataCopy(vreg0, maxValueHelp);
        AscendC::MicroAPI::DataCopy(argmaxUpdateVreg, argmaxHelp);
        AscendC::MicroAPI::DataCopy(maxValueLocal, vreg0, pregAllT1);
        AscendC::MicroAPI::DataCopy(argmaxLocal, argmaxUpdateVreg, pregAllT2);
    }
}

template <typename T1, typename T2, uint32_t IS_PAD, uint32_t NANPROP>
__aicore__ inline void MaxPoolWithArgmaxNhwCKernel<T1, T2, IS_PAD, NANPROP>::Process()
{
    if (blockIdx_ >= usedCoreNum_) {
        return;
    }

    int64_t curCoreProcessNum = (blockIdx_ + 1 == usedCoreNum_) ? tailCoreProcessNum_ : normalCoreProcessNum_;

    for (int64_t loopNum = 0; loopNum < curCoreProcessNum; loopNum++) {
        LocalTensor<T1> maxValueLocal = maxValueQue_.AllocTensor<T1>();
        LocalTensor<T2> argmaxLocal = argmaxQue_.AllocTensor<T2>();
        __local_mem__ T1* maxValueAddr = (__local_mem__ T1*)maxValueLocal.GetPhyAddr();
        __local_mem__ T2* argmaxAddr = (__local_mem__ T2*)argmaxLocal.GetPhyAddr();
        if (isSplitKernel_ == 1) {
            InitHelpBuf();
            for (hKernelIndex_ = 0; hKernelIndex_ < tilingData_->hKernelOuter; ++hKernelIndex_) {
                for (wKernelIndex_ = 0; wKernelIndex_ < tilingData_->wKernelOuter; ++wKernelIndex_) {
                    ScalarCompute(loopNum);
                    if (hInputActual_ <= 0 || wInputActual_ <= 0) {
                        continue;
                    }
                    CopyIn();
                    Compute(maxValueAddr, argmaxAddr);
                }
            }
            CopyResultToUb(maxValueAddr, argmaxAddr);
        } else {
            ScalarCompute(loopNum);
            CopyIn();
            Compute(maxValueAddr, argmaxAddr);
        }

        maxValueQue_.EnQue(maxValueLocal);
        argmaxQue_.EnQue(argmaxLocal);
        CopyOut();
    }
}

template <typename T1, typename T2, uint32_t IS_PAD, uint32_t NANPROP>
__aicore__ inline void MaxPoolWithArgmaxNhwCKernel<T1, T2, IS_PAD, NANPROP>::Compute(__local_mem__ T1* maxValueLocal,
                                                                              __local_mem__ T2* argmaxLocal)
{
    LocalTensor<T1> xLocal = inputQue_.DeQue<T1>();
    __local_mem__ T1* xAddr = (__local_mem__ T1*)xLocal.GetPhyAddr();

    if constexpr (IS_PAD == 1) {
        if (isSplitKernel_ == 1) {
            MaxPoolAndArgmaxVFForPad<true>(xAddr, maxValueLocal, argmaxLocal);
        } else {
            MaxPoolAndArgmaxVFForPad<false>(xAddr, maxValueLocal, argmaxLocal);
        }
    } else {
        if (isSplitKernel_ == 1) {
            MaxPoolAndArgmaxVF<true>(xAddr, maxValueLocal, argmaxLocal);
        } else {
            MaxPoolAndArgmaxVF<false>(xAddr, maxValueLocal, argmaxLocal);
        }
    }
    inputQue_.FreeTensor(xLocal);
}

template <typename T1, typename T2, uint32_t IS_PAD, uint32_t NANPROP>
template <const bool IS_SPLIT_KERNEL>
__aicore__ inline void MaxPoolWithArgmaxNhwCKernel<T1, T2, IS_PAD, NANPROP>::MaxPoolAndArgmaxVF(
    __local_mem__ T1* xLocal, __local_mem__ T1* maxValueLocal, __local_mem__ T2* argmaxLocal)
{
    __local_mem__ T1* maxValueHelp = (__local_mem__ T1*)helperTBuf_.Get<T1>().GetPhyAddr();
    __local_mem__ T2* argmaxHelp =
        (__local_mem__ T2*)helperTBuf_.Get<T2>().GetPhyAddr() + HELPER_BUFFER_SIZE_512 / sizeof(T1);

    int64_t nOutputActual = nOutputActual_;
    int64_t hOutputActual = hOutputActual_;
    int64_t wOutputActual = wOutputActual_;
    int64_t cLoop = ops::CeilDiv(cOutputActual_, int64_t(VREG_LENGTH_DATA_NUM_T2));
    int64_t cOutputActualAlign = cOutputActualAlign_;
    int64_t cOutputActual = cOutputActual_;
    int64_t wInputActual = wInputActual_;
    int64_t hInputActual = hInputActual_;

    int64_t hStride = hStride_;
    int64_t wStride = wStride_;
    int64_t hKernel = hKernelActual_;
    int64_t wKernel = wKernelActual_;
    int64_t wInput = wInput_;

    // hwc平面偏移
    int64_t indexNHWCOffset = hInputAxisOffset_ + wInputAxisOffset_ + cInputAxisOffset_;

    int64_t cOutputTail = (cOutputActual_ % VREG_LENGTH_DATA_NUM_T2) == 0 ? VREG_LENGTH_DATA_NUM_T2
                                                                          : (cOutputActual_ % VREG_LENGTH_DATA_NUM_T2);

    for (uint16_t nIndex = 0; nIndex < nOutputActual; ++nIndex) {
        for (uint16_t cIndex = 0; cIndex < cLoop; ++cIndex) {
            uint32_t computeLoopTmp = cIndex == (cLoop - 1) ? cOutputTail : VREG_LENGTH_DATA_NUM_T2;
            uint32_t computeLoopVL = computeLoopTmp;
            uint32_t computeLoopVLT2 = computeLoopTmp;
            __VEC_SCOPE__
            {
                AscendC::MicroAPI::RegTensor<T1> vreg0;
                AscendC::MicroAPI::RegTensor<T1> vreg1;
                AscendC::MicroAPI::RegTensor<T2> argmaxUpdateVreg;
                AscendC::MicroAPI::RegTensor<T2> argmaxResVreg;
                AscendC::MicroAPI::MaskReg neMask;
                AscendC::MicroAPI::MaskReg gtMask;
                AscendC::MicroAPI::MaskReg gtMaskT2;
                AscendC::MicroAPI::MaskReg gtMaskT4;

                AscendC::MicroAPI::MaskReg computeMaskT1 = AscendC::MicroAPI::UpdateMask<T1>(computeLoopVL);
                AscendC::MicroAPI::MaskReg computeMaskT2 = AscendC::MicroAPI::UpdateMask<T2>(computeLoopVLT2);
                for (uint16_t hIndex = 0; hIndex < static_cast<uint16_t>(hOutputActual); ++hIndex) {
                    for (uint16_t wIndex = 0; wIndex < static_cast<uint16_t>(wOutputActual); ++wIndex) {
                        int64_t outputOffset = nIndex * hOutputActual * wOutputActual * cOutputActualAlign +
                                               hIndex * wOutputActual * cOutputActualAlign +
                                               wIndex * cOutputActualAlign + cIndex * VREG_LENGTH_DATA_NUM_T2;
                        // UB内偏移
                        int64_t offsetC = cIndex * VREG_LENGTH_DATA_NUM_T2;
                        int64_t offsetW = wIndex * wStride * cOutputActualAlign;
                        int64_t offsetH = hIndex * hStride * wInputActual * cOutputActualAlign;
                        int64_t offsetN = nIndex * hInputActual * wInputActual * cOutputActualAlign;
                        int64_t startInUb = offsetC + offsetW + offsetH + offsetN;

                        // 起始点hwc偏移
                        int64_t scopeHWCOffset = indexNHWCOffset +
                                                 (hIndex * hStride * wInput + wIndex * wStride) * cInput_ +
                                                 offsetC;

                        if constexpr (IS_SPLIT_KERNEL == 1) {
                            AscendC::MicroAPI::DataCopy(vreg0, maxValueHelp);
                            AscendC::MicroAPI::DataCopy(argmaxResVreg, argmaxHelp);
                        } else {
                            DuplicateLowestReg(vreg0);
                            AscendC::MicroAPI::Duplicate(argmaxResVreg, T2(-1));
                        }

                        for (uint16_t hKernelIdx = 0; hKernelIdx < static_cast<uint16_t>(hKernel); ++hKernelIdx) {
                            for (uint16_t wKernelIdx = 0; wKernelIdx < static_cast<uint16_t>(wKernel); wKernelIdx++) {
                                AscendC::MicroAPI::DataCopy(
                                    vreg1,
                                    xLocal + startInUb + (hKernelIdx * wInputActual + wKernelIdx) * cOutputActualAlign);
                                AscendC::MicroAPI::Arange(argmaxUpdateVreg, scopeHWCOffset + (hKernelIdx * wInput + wKernelIdx) * cInput_);

                                if constexpr (NANPROP == 1) {
                                    AscendC::MicroAPI::Compare<T1, CMPMODE::LE>(gtMask, vreg1, vreg0, computeMaskT1);
                                    AscendC::MicroAPI::MaskNot(gtMask, gtMask, computeMaskT1);
                                } else {
                                    AscendC::MicroAPI::Compare<T1, CMPMODE::GT>(gtMask, vreg1, vreg0, computeMaskT1);
                                }

                                if constexpr (sizeof(T2) / sizeof(T1) == DIGIT_1) {
                                    AscendC::MicroAPI::Select(argmaxResVreg, argmaxUpdateVreg, argmaxResVreg, gtMask);
                                } else if constexpr (sizeof(T2) / sizeof(T1) == DIGIT_2) {
                                    AscendC::MicroAPI::MaskUnPack(gtMaskT2, gtMask);
                                    AscendC::MicroAPI::Select(argmaxResVreg, argmaxUpdateVreg, argmaxResVreg, gtMaskT2);
                                } else {
                                    AscendC::MicroAPI::MaskUnPack(gtMaskT2, gtMask);
                                    AscendC::MicroAPI::MaskUnPack(gtMaskT4, gtMaskT2);
                                    AscendC::MicroAPI::Select(argmaxResVreg, argmaxUpdateVreg, argmaxResVreg, gtMaskT4);
                                }

                                AscendC::MicroAPI::Select(vreg0, vreg1, vreg0, gtMask);
                            }
                        }

                        if constexpr (IS_SPLIT_KERNEL == 1) {
                            AscendC::MicroAPI::DataCopy(maxValueHelp, vreg0, computeMaskT1);
                            AscendC::MicroAPI::DataCopy(argmaxHelp, argmaxResVreg, computeMaskT2);
                        } else {
                            AscendC::MicroAPI::DataCopy(maxValueLocal + outputOffset, vreg0, computeMaskT1);
                            AscendC::MicroAPI::DataCopy(argmaxLocal + outputOffset, argmaxResVreg, computeMaskT2);
                        }
                    }
                }
            }
        }
    }
}

template <typename T1, typename T2, uint32_t IS_PAD, uint32_t NANPROP>
template <const bool IS_SPLIT_KERNEL>
__aicore__ inline void MaxPoolWithArgmaxNhwCKernel<T1, T2, IS_PAD, NANPROP>::MaxPoolAndArgmaxVFForPad(
    __local_mem__ T1* xLocal, __local_mem__ T1* maxValueLocal, __local_mem__ T2* argmaxLocal)
{
    __local_mem__ T1* maxValueHelp = (__local_mem__ T1*)helperTBuf_.Get<T1>().GetPhyAddr();
    __local_mem__ T2* argmaxHelp =
        (__local_mem__ T2*)helperTBuf_.Get<T2>().GetPhyAddr() + HELPER_BUFFER_SIZE_512 / sizeof(T1);
    int64_t nOutputActual = nOutputActual_;
    int64_t hOutputActual = hOutputActual_;
    int64_t wOutputActual = wOutputActual_;
    int64_t cLoop = ops::CeilDiv(cOutputActual_, int64_t(VREG_LENGTH_DATA_NUM_T2));
    int64_t cOutputActualAlign = cOutputActualAlign_;
    int64_t cOutputActual = cOutputActual_;
    int64_t wInputActual = wInputActual_;
    int64_t hInputActual = hInputActual_;
    int64_t hStride = hStride_;
    int64_t wStride = wStride_;
    int64_t wInput = wInput_;
    int64_t wInputActualPad = wInputActualPad_;
    int64_t indexNHWCOffset = cInputAxisOffset_;
    int64_t cOutputTail = (cOutputActual_ % VREG_LENGTH_DATA_NUM_T2) == 0 ? VREG_LENGTH_DATA_NUM_T2
                                                                          : (cOutputActual_ % VREG_LENGTH_DATA_NUM_T2);
    for (uint16_t nIndex = 0; nIndex < nOutputActual; ++nIndex) {
        for (uint16_t hIndex = 0; hIndex < hOutputActual; ++hIndex) {
            for (uint16_t wIndex = 0; wIndex < wOutputActual; ++wIndex) {
                for (uint16_t cIndex = 0; cIndex < cLoop; ++cIndex) {
                    uint32_t computeLoopTmp = cIndex == (cLoop - 1) ? cOutputTail : VREG_LENGTH_DATA_NUM_T2;
                    uint32_t computeLoopVL = computeLoopTmp;
                    uint32_t computeLoopVLT2 = computeLoopTmp;
                    uint32_t correctHKernel = hKernelActual_;
                    uint32_t correctWKernel = wKernelActual_;
                    int64_t topOffset = hAxisIndex_ * hOutputInner_ * hStride_ + hIndex * hStride_ +
                                        hKernelIndex_ * hKernelInner_ - padTop_;
                    correctHKernel = topOffset >= 0 ? correctHKernel : correctHKernel + topOffset;
                    int64_t downOffset = hAxisIndex_ * hOutputInner_ * hStride_ + hIndex * hStride_ +
                                         hKernelIndex_ * hKernelInner_ + hKernelActual_ - padTop_ - hInput_;
                    correctHKernel = downOffset >= 0 ? correctHKernel - downOffset : correctHKernel;

                    int64_t leftOffset = wAxisIndex_ * wOutputInner_ * wStride_ + wIndex * wStride_ +
                                         wKernelIndex_ * wKernelInner_ - padLeft_;
                    correctWKernel = leftOffset >= 0 ? correctWKernel : correctWKernel + leftOffset;
                    int64_t rightOffset = wAxisIndex_ * wOutputInner_ * wStride_ + wIndex * wStride_ +
                                          wKernelIndex_ * wKernelInner_ + wKernelActual_ - padLeft_ - wInput_;
                    correctWKernel = rightOffset >= 0 ? correctWKernel - rightOffset : correctWKernel;

                    int64_t outputOffset = nIndex * hOutputActual * wOutputActual * cOutputActualAlign +
                                           hIndex * wOutputActual * cOutputActualAlign + wIndex * cOutputActualAlign +
                                           cIndex * VREG_LENGTH_DATA_NUM_T2;
                    // UB内偏移
                    int64_t kernelTopOffsetOnLand = hIndex * hStride_;
                    if (baseBlockTopOffsetInOcean_ != 0) {
                        kernelTopOffsetOnLand = kernelTopOffsetOnLand >= baseBlockTopOffsetInOcean_
                                                    ? kernelTopOffsetOnLand
                                                    : baseBlockTopOffsetInOcean_;
                    }
                    int64_t kernelLeftOffsetOnLand = wIndex * wStride_;
                    if (baseBlockLeftOffsetInOcean_ != 0) {
                        kernelLeftOffsetOnLand = kernelLeftOffsetOnLand >= baseBlockLeftOffsetInOcean_
                                                    ? kernelLeftOffsetOnLand
                                                    : baseBlockLeftOffsetInOcean_;
                    }

                    int64_t offsetC = cIndex * VREG_LENGTH_DATA_NUM_T2;
                    int64_t offsetW = kernelLeftOffsetOnLand * cOutputActualAlign;
                    int64_t offsetH = kernelTopOffsetOnLand * wInputActualPad_ * cOutputActualAlign;
                    int64_t offsetN = nIndex * hInputActualPad_ * wInputActualPad_ * cOutputActualAlign;
                    int64_t startInUb = offsetC + offsetW + offsetH + offsetN;

                    // w, h, c 偏移.
                    int64_t topOffsetCoast = topOffset >= 0 ? topOffset : 0;
                    int64_t leftOffsetCoast = leftOffset >= 0 ? leftOffset : 0;
                    int64_t kernelStartArgmaxOffset = indexNHWCOffset +
                                                      (topOffsetCoast * wInput_ + leftOffsetCoast) * cInput_+
                                                      offsetC;

                    __VEC_SCOPE__
                    {
                        AscendC::MicroAPI::RegTensor<T1> vreg0;
                        AscendC::MicroAPI::RegTensor<T1> vreg1;

                        AscendC::MicroAPI::RegTensor<T2> argmaxUpdateVreg;
                        AscendC::MicroAPI::RegTensor<T2> argmaxResVreg;

                        AscendC::MicroAPI::RegTensor<uint32_t> startOffsetRegU32;
                        AscendC::MicroAPI::RegTensor<uint32_t> separateOffsetRegU32;
                        AscendC::MicroAPI::RegTensor<uint16_t> separateOffsetRegU16;
                        AscendC::MicroAPI::MaskReg computeMaskT1 = AscendC::MicroAPI::UpdateMask<T1>(computeLoopVL);
                        AscendC::MicroAPI::MaskReg computeMaskT2 = AscendC::MicroAPI::UpdateMask<T2>(computeLoopVLT2);
                        AscendC::MicroAPI::MaskReg neMask;
                        AscendC::MicroAPI::MaskReg gtMask;
                        AscendC::MicroAPI::MaskReg gtMaskT2;
                        AscendC::MicroAPI::MaskReg gtMaskT4;

                        if constexpr (IS_SPLIT_KERNEL == 1) {
                            AscendC::MicroAPI::DataCopy(vreg0, maxValueHelp);
                            AscendC::MicroAPI::DataCopy(argmaxResVreg, argmaxHelp);
                        } else {
                            DuplicateLowestReg(vreg0);
                            AscendC::MicroAPI::Duplicate(argmaxResVreg, T2(-1));
                        }

                        for (uint16_t hKernelIdx = 0; hKernelIdx < static_cast<uint16_t>(correctHKernel);
                             ++hKernelIdx) {
                            for (uint16_t wKernelIdx = 0; wKernelIdx < static_cast<uint16_t>(correctWKernel);
                                 ++wKernelIdx) {
                                AscendC::MicroAPI::DataCopy(
                                    vreg1, xLocal + startInUb +
                                               (hKernelIdx * wInputActualPad + wKernelIdx) * cOutputActualAlign);
                                AscendC::MicroAPI::Arange(argmaxUpdateVreg, (hKernelIdx * wInput + wKernelIdx) * cInput_ + kernelStartArgmaxOffset);

                                if constexpr (NANPROP == 1) {
                                    AscendC::MicroAPI::Compare<T1, CMPMODE::LE>(gtMask, vreg1, vreg0, computeMaskT1);
                                    AscendC::MicroAPI::MaskNot(gtMask, gtMask, computeMaskT1);
                                } else {
                                    AscendC::MicroAPI::Compare<T1, CMPMODE::GT>(gtMask, vreg1, vreg0, computeMaskT1);
                                }

                                if constexpr (sizeof(T2) / sizeof(T1) == DIGIT_1) {
                                    AscendC::MicroAPI::Select(argmaxResVreg, argmaxUpdateVreg, argmaxResVreg, gtMask);
                                } else if constexpr (sizeof(T2) / sizeof(T1) == DIGIT_2) {
                                    AscendC::MicroAPI::MaskUnPack(gtMaskT2, gtMask);
                                    AscendC::MicroAPI::Select(argmaxResVreg, argmaxUpdateVreg, argmaxResVreg, gtMaskT2);
                                } else {
                                    AscendC::MicroAPI::MaskUnPack(gtMaskT2, gtMask);
                                    AscendC::MicroAPI::MaskUnPack(gtMaskT4, gtMaskT2);
                                    AscendC::MicroAPI::Select(argmaxResVreg, argmaxUpdateVreg, argmaxResVreg, gtMaskT4);
                                }

                                AscendC::MicroAPI::Select(vreg0, vreg1, vreg0, gtMask);
                            }
                        }

                        if constexpr (IS_SPLIT_KERNEL == 1) {
                            AscendC::MicroAPI::DataCopy(maxValueHelp, vreg0, computeMaskT1);
                            AscendC::MicroAPI::DataCopy(argmaxHelp, argmaxResVreg, computeMaskT2);
                        } else {
                            AscendC::MicroAPI::DataCopy(maxValueLocal + outputOffset, vreg0, computeMaskT1);
                            AscendC::MicroAPI::DataCopy(argmaxLocal + outputOffset, argmaxResVreg, computeMaskT2);
                        }
                    }
                }
            }
        }
    }
}

template <typename T1, typename T2, uint32_t IS_PAD, uint32_t NANPROP>
__aicore__ inline void MaxPoolWithArgmaxNhwCKernel<T1, T2, IS_PAD, NANPROP>::CopyOut()
{
    LocalTensor<T1> maxValueLocal = maxValueQue_.DeQue<T1>();
    LocalTensor<T2> argmaxLocal = argmaxQue_.DeQue<T2>();

    {
        DataCopyExtParams copyOutParamT;
        copyOutParamT.blockCount = nOutputActual_ * hOutputActual_ * wOutputActual_;
        copyOutParamT.blockLen = cOutputActual_ * sizeof(T1);
        copyOutParamT.srcStride = (cOutputActualAlign_ - cOutputActual_) * sizeof(T1) / BLOCK_SIZE;
        copyOutParamT.dstStride = 0;

        DataCopyPad(yGm_[nOutputAxisOffset_ + hOutputAxisOffset_ + wOutputAxisOffset_ + cOutputAxisOffset_],
                    maxValueLocal, copyOutParamT);
    }

    {
        DataCopyExtParams copyOutParamT;
        copyOutParamT.blockCount = nOutputActual_ * hOutputActual_ * wOutputActual_;
        copyOutParamT.blockLen = cOutputActual_ * sizeof(T2);
        copyOutParamT.srcStride = (cOutputActualAlign_ - cOutputActual_) * sizeof(T2) / BLOCK_SIZE;
        copyOutParamT.dstStride = 0;
        DataCopyPad(argmaxGm_[nOutputAxisOffset_ + hOutputAxisOffset_ + wOutputAxisOffset_ + cOutputAxisOffset_],
                    argmaxLocal, copyOutParamT);
    }
    maxValueQue_.FreeTensor(maxValueLocal);
    argmaxQue_.FreeTensor(argmaxLocal);
}

template <typename T1, typename T2, uint32_t IS_PAD, uint32_t NANPROP>
__aicore__ inline void MaxPoolWithArgmaxNhwCKernel<T1, T2, IS_PAD, NANPROP>::FillPadNegVF(__local_mem__ T1* xLocalAddr)
{
    int32_t top = baseBlockTopOffsetInOcean_;
    int32_t left = baseBlockLeftOffsetInOcean_;
    int32_t right = baseBlockRightOffsetInOcean_;
    int32_t down = baseBlockDownOffsetInOcean_;
    int64_t wInputActual = wInputActual_;
    int64_t hInputActual = hInputActual_;
    int32_t cOutputActualAlign = cOutputActualAlign_;
    int32_t hInputActualAmend = (hOutputActual_ - 1) * hStride_ + hKernel_;
    int32_t wInputActualAmend = (wOutputActual_ - 1) * wStride_ + wKernel_;
    uint32_t computeSize = platform::GetVRegSize() / sizeof(T1);

    uint32_t topCount = top * wInputActualAmend * cOutputActualAlign;
    uint16_t topRepeatTimes = (topCount + computeSize - 1) / computeSize;

    int32_t leftSingleRowCount = left * cOutputActualAlign;
    uint16_t leftSingleRowRepeatTimes = (leftSingleRowCount + computeSize - 1) / computeSize;
    int32_t leftStartOffset = topCount;

    int32_t rightSingleRowCount = right * cOutputActualAlign;
    uint16_t rightSingleRowRepeatTimes = (rightSingleRowCount + computeSize - 1) / computeSize;
    int32_t rightStartOffset = topCount + (wInputActual + left) * cOutputActualAlign;

    uint32_t downCount = down * wInputActualAmend * cOutputActualAlign;
    uint16_t downRepeatTimes = (downCount + computeSize - 1) / computeSize;
    int32_t downStartOffset = (hInputActual + top) * wInputActualAmend * cOutputActualAlign;
    uint16_t nOutputActual = nOutputActual_;
    int32_t nStartOffset = hInputActualAmend * wInputActualAmend * cOutputActualAlign;
    __VEC_SCOPE__

    {
        AscendC::MicroAPI::RegTensor<T1> negInfReg;
        DuplicateNegInfReg(negInfReg);
        for (uint16_t n = 0; n < nOutputActual; n++) {
            int32_t nOffset = n * nStartOffset;
            // top
            uint32_t topCountTmp = topCount;
            for (uint16_t i = 0; i < topRepeatTimes; i++) {
                AscendC::MicroAPI::MaskReg preg = AscendC::MicroAPI::UpdateMask<T1>(topCountTmp);
                AscendC::MicroAPI::DataCopy(xLocalAddr + nOffset + i * computeSize, negInfReg, preg);
            }

            // left
            for (uint16_t hIndex = 0; hIndex < static_cast<uint16_t>(hInputActual); hIndex++) {
                int32_t leftOffset = hIndex * wInputActualAmend * cOutputActualAlign + leftStartOffset;
                uint32_t leftCount = leftSingleRowCount;
                for (uint16_t i = 0; i < leftSingleRowRepeatTimes; i++) {
                    AscendC::MicroAPI::MaskReg preg = AscendC::MicroAPI::UpdateMask<T1>(leftCount);
                    AscendC::MicroAPI::DataCopy(xLocalAddr + nOffset + leftOffset + i * computeSize, negInfReg, preg);
                }
            }

            // right
            for (uint16_t hIndex = 0; hIndex < static_cast<uint16_t>(hInputActual); hIndex++) {
                int32_t rightOffset = hIndex * wInputActualAmend * cOutputActualAlign + rightStartOffset;
                uint32_t rightCount = rightSingleRowCount;
                for (uint16_t i = 0; i < rightSingleRowRepeatTimes; i++) {
                    AscendC::MicroAPI::MaskReg preg = AscendC::MicroAPI::UpdateMask<T1>(rightCount);
                    AscendC::MicroAPI::DataCopy(xLocalAddr + nOffset + rightOffset + i * computeSize, negInfReg, preg);
                }
            }

            // down
            uint32_t downCountTmp = downCount;
            for (uint16_t i = 0; i < downRepeatTimes; i++) {
                AscendC::MicroAPI::MaskReg preg = AscendC::MicroAPI::UpdateMask<T1>(downCountTmp);
                AscendC::MicroAPI::DataCopy(xLocalAddr + nOffset + downStartOffset + i * computeSize, negInfReg, preg);
            }
        }
    }
}

template <typename T1, typename T2, uint32_t IS_PAD, uint32_t NANPROP>
__aicore__ inline void MaxPoolWithArgmaxNhwCKernel<T1, T2, IS_PAD, NANPROP>::CopyIn()
{
    LocalTensor<T1> xLocal = inputQue_.AllocTensor<T1>();
    __local_mem__ T1* xLocalAddr = (__local_mem__ T1*)xLocal.GetPhyAddr();

    int64_t nOutputActual = nOutputActual_;
    int64_t hInputWithPad = (hInputActual_ + baseBlockTopOffsetInOcean_ + baseBlockDownOffsetInOcean_);
    int64_t wInputWithPad = (wInputActual_ + baseBlockLeftOffsetInOcean_ + baseBlockRightOffsetInOcean_);
    int64_t cOutputActualAlign = cOutputActualAlign_;
    int64_t xGmOffset = nInputAxisOffset_ + hInputAxisOffset_ + wInputAxisOffset_ + cInputAxisOffset_;

    LoopModeParams loopParams;
    loopParams.loop2Size = nOutputActual_;
    loopParams.loop2SrcStride = hInput_ * wInput_ * cInput_ * sizeof(T1);
    loopParams.loop2DstStride = hInputWithPad * wInputWithPad * cOutputActualAlign_ * sizeof(T1);

    loopParams.loop1Size = hInputActual_;
    loopParams.loop1SrcStride = wInput_ * cInput_ * sizeof(T1);
    loopParams.loop1DstStride = wInputWithPad * cOutputActualAlign_ * sizeof(T1);

    SetLoopModePara(loopParams, DataCopyMVType::OUT_TO_UB);
    DataCopyExtParams copyExtParams;
    copyExtParams.blockCount = wInputActual_;
    copyExtParams.blockLen = cOutputActual_ * sizeof(T1);
    copyExtParams.srcStride = (cInput_ - cOutputActual_) * sizeof(T1);
    copyExtParams.dstStride = 0;
    DataCopyPadExtParams<T1> copyPadExtparams;
    copyPadExtparams.isPad = false;
    DataCopyPad(xLocal[xLocalUbOffset], xGm_[xGmOffset], copyExtParams, copyPadExtparams);
    ResetLoopModePara(DataCopyMVType::OUT_TO_UB);

    inputQue_.EnQue(xLocal);
}

}  // namespace MaxPoolWithArgmaxNHWC
#endif  // MAX_POOL_WITH_ARGMAX_NHWC_BIG_C_H_
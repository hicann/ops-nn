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
 * \file max_pool_grad_with_argmax_nhwc_kernel_common.h
 * \brief
 */

#ifndef MAX_POOL_GRAD_WITH_ARGMAX_NHWC_KERNEL_BASE_H_
#define MAX_POOL_GRAD_WITH_ARGMAX_NHWC_KERNEL_BASE_H_

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "../inc/platform.h"
#include "max_pool_grad_with_argmax_base_common.h"
#include "pool_utils/arch35/index/pool_grad_nhwc_scatter_index.h"

namespace MaxPoolGradWithArgmaxNHWCNameSpace {
using PoolUtils::Index::DoMulCNhwc;
using PoolUtils::Index::DoSingleCNhwc;
using PoolUtils::Index::Gen3DIndexOneNhwc;
using PoolUtils::Index::GenInitial3DIndicesNhwc;
using PoolUtils::Index::IndexConvNhwc;

template <typename T1, typename T2, typename T3, const uint32_t IS_CHECK_RANGE = 0, int32_t VER = 0,
          typename DerivedT = void>
class MaxPoolGradWithArgmaxKernelNHWCBase {
public:
    __aicore__ inline MaxPoolGradWithArgmaxKernelNHWCBase(void){};
    __aicore__ inline void Init(
        GM_ADDR x, GM_ADDR grad, GM_ADDR argmax, GM_ADDR y, TPipe& pipeIn,
        const MaxPoolGradWithArgmaxNHWCNameSpace::MaxPoolGradWithArgmaxNHWCTilingCommonData& tilingData);
    __aicore__ inline void ParseTilingData(
        const MaxPoolGradWithArgmaxNHWCNameSpace::MaxPoolGradWithArgmaxNHWCTilingCommonData& tilingData);
    __aicore__ inline void ScalarCompute(int64_t loopNum);
    __aicore__ inline void CopyIn();
    __aicore__ inline void ProcessNoArgmaxBlock();
    __aicore__ inline void CopyOut();
    // 公共生命周期流程 (CRTP 模板方法): 派生类通过继承获得 Compute/ProcessPerLoop/Process,
    // 并提供同签名入口 ConCProcVF(yAddr, gradAddr, argmaxAddr, helpAddr)。
    // Compute 内经 static_cast<DerivedT*> 编译期派发到派生类实现, 无虚函数开销。
    __aicore__ inline void Compute();
    __aicore__ inline void ProcessPerLoop();
    __aicore__ inline void Process();

    TPipe pipe_;
    TQue<QuePosition::VECIN, BUFFER_NUM> gradQue_;
    TQue<QuePosition::VECIN, BUFFER_NUM> argmaxQue_;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outputQue_;
    TBuf<QuePosition::VECCALC> helpBuf_;

    GlobalTensor<T1> yGm_;
    GlobalTensor<T1> gradGm_;
    GlobalTensor<T2> argmaxGm_;

    uint32_t blockIdx_ = 0;

    int64_t hArgmax_ = 1;
    int64_t wArgmax_ = 1;

    int64_t cOutput_ = 1;
    int64_t hOutput_ = 1;
    int64_t wOutput_ = 1;

    int64_t kernelH_ = 1;
    int64_t kernelW_ = 1;

    int64_t strideH_ = 1;
    int64_t strideW_ = 1;

    int64_t padH_ = 0;
    int64_t padW_ = 0;

    int64_t dilationH_ = 1;
    int64_t dilationW_ = 1;

    int64_t nOutputInner_ = 1;
    int64_t nOutputTail_ = 1;
    int64_t nOutputOuter_ = 1;
    int64_t nOutputActual_ = 1;

    int64_t hOutputInner_ = 1;
    int64_t hOutputTail_ = 1;
    int64_t hOutputOuter_ = 1;
    int64_t hOutputActual_ = 1;

    int64_t wOutputInner_ = 1;
    int64_t wOutputTail_ = 1;
    int64_t wOutputOuter_ = 1;
    int64_t wOutputActual_ = 1;

    int64_t cOutputInner_ = 1;
    int64_t cOutputTail_ = 1;
    int64_t cOutputOuter_ = 1;
    int64_t cOutputActual_ = 1;
    int64_t cOutputAligned_ = 1;

    int64_t normalCoreProcessNum_ = 1;
    int64_t tailCoreProcessNum_ = 1;
    int64_t curCoreProcessNum_ = 1;
    int64_t usedCoreNum_ = 1;

    int64_t outputBufferSize_ = 1;
    int64_t gradBufferSize_ = 1;
    int64_t argmaxBufferSize_ = 1;

    int64_t nAxisIndex_ = 0;
    int64_t hAxisIndex_ = 0;
    int64_t wAxisIndex_ = 0;
    int64_t cAxisIndex_ = 0;

    int64_t hArgmaxActual_ = 0;
    int64_t wArgmaxActual_ = 0;

    int64_t nOutputArgmaxOffset_ = 0;
    int64_t hAxisArgmaxOffset_ = 0;
    int64_t wAxisArgmaxOffset_ = 0;
    int64_t cAxisArgmaxOffset_ = 0;

    int64_t argmaxPlaneSize_ = 1;
    int64_t hProBatchSize_ = 1;
    int64_t wProBatchSize_ = 1;
    int64_t curHProBatchSize_ = 1;
    int64_t curWProBatchSize_ = 1;
    constexpr static int32_t BLOCK_SIZE = platform::GetUbBlockSize();
    constexpr static int32_t V_REG_SIZE = platform::GetVRegSize();

    constexpr static int64_t MAX_DATA_NUM_IN_ONE_BLOCK = BLOCK_SIZE / sizeof(T1) >= BLOCK_SIZE / sizeof(T2) ?
                                                             BLOCK_SIZE / sizeof(T1) :
                                                             BLOCK_SIZE / sizeof(T2);
    constexpr static int64_t VREG_LENGTH_DATA_NUM_T2 = platform::GetVRegSize() / sizeof(T2);
};

template <typename T1, typename T2, typename T3, const uint32_t IS_CHECK_RANGE, int32_t VER, typename DerivedT>
__aicore__ inline void MaxPoolGradWithArgmaxKernelNHWCBase<T1, T2, T3, IS_CHECK_RANGE, VER, DerivedT>::ParseTilingData(
    const MaxPoolGradWithArgmaxNHWCNameSpace::MaxPoolGradWithArgmaxNHWCTilingCommonData& tilingData)
{
    hArgmax_ = tilingData.hArgmax;
    wArgmax_ = tilingData.wArgmax;

    cOutput_ = tilingData.cOutput;
    hOutput_ = tilingData.hOutput;
    wOutput_ = tilingData.wOutput;

    kernelH_ = tilingData.hKernel;
    kernelW_ = tilingData.wKernel;

    strideH_ = tilingData.hStride;
    strideW_ = tilingData.wStride;

    padH_ = tilingData.padH;
    padW_ = tilingData.padW;

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

    outputBufferSize_ = tilingData.outputBufferSize;
    gradBufferSize_ = tilingData.gradBufferSize;
    argmaxBufferSize_ = tilingData.argmaxBufferSize;

    hProBatchSize_ = tilingData.hProBatchSize;
    wProBatchSize_ = tilingData.wProBatchSize;
    curHProBatchSize_ = hProBatchSize_;
    curWProBatchSize_ = wProBatchSize_;
}

template <typename T1, typename T2, typename T3, const uint32_t IS_CHECK_RANGE, int32_t VER, typename DerivedT>
__aicore__ inline void MaxPoolGradWithArgmaxKernelNHWCBase<T1, T2, T3, IS_CHECK_RANGE, VER, DerivedT>::Init(
    GM_ADDR x, GM_ADDR grad, GM_ADDR argmax, GM_ADDR y, TPipe& pipeIn,
    const MaxPoolGradWithArgmaxNHWCNameSpace::MaxPoolGradWithArgmaxNHWCTilingCommonData& tilingData)
{
    ParseTilingData(tilingData);

    blockIdx_ = GetBlockIdx();
    argmaxPlaneSize_ = hArgmax_ * wArgmax_;

    if (blockIdx_ >= usedCoreNum_) {
        return;
    }

    curCoreProcessNum_ = (blockIdx_ + 1 == usedCoreNum_) ? tailCoreProcessNum_ : normalCoreProcessNum_;
    gradGm_.SetGlobalBuffer((__gm__ T1*)grad);
    argmaxGm_.SetGlobalBuffer((__gm__ T2*)argmax);
    yGm_.SetGlobalBuffer((__gm__ T1*)y);

    pipe_ = pipeIn;
    pipe_.InitBuffer(outputQue_, BUFFER_NUM, outputBufferSize_);
    pipe_.InitBuffer(gradQue_, BUFFER_NUM, gradBufferSize_);
    pipe_.InitBuffer(argmaxQue_, BUFFER_NUM, argmaxBufferSize_);
    pipe_.InitBuffer(helpBuf_, HELP_BUFFER);
}

template <typename T1, typename T2, typename T3, const uint32_t IS_CHECK_RANGE, int32_t VER, typename DerivedT>
__aicore__ inline void MaxPoolGradWithArgmaxKernelNHWCBase<T1, T2, T3, IS_CHECK_RANGE, VER, DerivedT>::ScalarCompute(
    int64_t loopNum)
{
    int64_t baseBlockIdx = blockIdx_ * normalCoreProcessNum_ + loopNum;
    nAxisIndex_ = baseBlockIdx / (hOutputOuter_ * wOutputOuter_ * cOutputOuter_);
    nOutputActual_ = nAxisIndex_ == (nOutputOuter_ - 1) ? nOutputTail_ : nOutputInner_;

    int64_t tempNTail = baseBlockIdx % (hOutputOuter_ * wOutputOuter_ * cOutputOuter_);
    cAxisIndex_ = tempNTail / (hOutputOuter_ * wOutputOuter_);
    cOutputActual_ = cAxisIndex_ == (cOutputOuter_ - 1) ? cOutputTail_ : cOutputInner_;
    cOutputAligned_ = (cOutputActual_ + MAX_DATA_NUM_IN_ONE_BLOCK - 1) / MAX_DATA_NUM_IN_ONE_BLOCK *
                      MAX_DATA_NUM_IN_ONE_BLOCK;

    int64_t tempCTail = tempNTail % (hOutputOuter_ * wOutputOuter_);
    hAxisIndex_ = tempCTail / (wOutputOuter_);
    hOutputActual_ = hAxisIndex_ == (hOutputOuter_ - 1) ? hOutputTail_ : hOutputInner_;

    wAxisIndex_ = tempCTail % wOutputOuter_;
    wOutputActual_ = wAxisIndex_ == (wOutputOuter_ - 1) ? wOutputTail_ : wOutputInner_;

    int64_t hArgmaxActualStart = PStart(hAxisIndex_ * hOutputInner_, padH_, kernelH_, dilationH_, strideH_);
    int64_t hArgmaxActualEnd = PEnd(hAxisIndex_ * hOutputInner_ + hOutputActual_ - 1, padH_, strideH_, hArgmax_);
    int64_t wArgmaxActualStart = PStart(wAxisIndex_ * wOutputInner_, padW_, kernelW_, dilationW_, strideW_);
    int64_t wArgmaxActualEnd = PEnd(wAxisIndex_ * wOutputInner_ + wOutputActual_ - 1, padW_, strideW_, wArgmax_);
    hArgmaxActual_ = hArgmaxActualEnd - hArgmaxActualStart;
    wArgmaxActual_ = wArgmaxActualEnd - wArgmaxActualStart;

    curHProBatchSize_ = hProBatchSize_ > hArgmaxActual_ ? hArgmaxActual_ : hProBatchSize_;
    curWProBatchSize_ = wProBatchSize_ > wArgmaxActual_ ? wArgmaxActual_ : wProBatchSize_;

    nOutputArgmaxOffset_ = nAxisIndex_ * nOutputInner_ * argmaxPlaneSize_ * cOutput_;
    hAxisArgmaxOffset_ = hArgmaxActualStart * wArgmax_ * cOutput_;
    wAxisArgmaxOffset_ = wArgmaxActualStart * cOutput_;
    cAxisArgmaxOffset_ = cAxisIndex_ * cOutputInner_;
}

template <typename T1, typename T2, typename T3, const uint32_t IS_CHECK_RANGE, int32_t VER, typename DerivedT>
__aicore__ inline void
MaxPoolGradWithArgmaxKernelNHWCBase<T1, T2, T3, IS_CHECK_RANGE, VER, DerivedT>::ProcessNoArgmaxBlock()
{
    uint32_t calcCount = static_cast<uint32_t>(outputBufferSize_) / sizeof(T1);
    LocalTensor<T1> yLocal = outputQue_.AllocTensor<T1>();
    Duplicate(yLocal, T1(0), calcCount);
    outputQue_.EnQue(yLocal);
    CopyOut();
    return;
}

template <typename T1, typename T2, typename T3, const uint32_t IS_CHECK_RANGE, int32_t VER, typename DerivedT>
__aicore__ inline void MaxPoolGradWithArgmaxKernelNHWCBase<T1, T2, T3, IS_CHECK_RANGE, VER, DerivedT>::CopyIn()
{
    LocalTensor<T1> gradLocal = gradQue_.AllocTensor<T1>();
    LocalTensor<T2> argmaxLocal = argmaxQue_.AllocTensor<T2>();

    int64_t argmaxGmOffset = nOutputArgmaxOffset_ + hAxisArgmaxOffset_ + wAxisArgmaxOffset_ + cAxisArgmaxOffset_;

    DataCopyPadExtParams<T1> paramsT1 = {false, 0, 0, 0};
    LoopModeParams loopModeParamsT1;
    loopModeParamsT1.loop1Size = hArgmaxActual_;
    loopModeParamsT1.loop2Size = nOutputActual_;
    loopModeParamsT1.loop1SrcStride = wArgmax_ * cOutput_ * sizeof(T1);
    loopModeParamsT1.loop2SrcStride = argmaxPlaneSize_ * cOutput_ * sizeof(T1);
    loopModeParamsT1.loop1DstStride = wArgmaxActual_ * cOutputAligned_ * sizeof(T1);
    loopModeParamsT1.loop2DstStride = hArgmaxActual_ * wArgmaxActual_ * cOutputAligned_ * sizeof(T1);

    SetLoopModePara(loopModeParamsT1, DataCopyMVType::OUT_TO_UB);
    DataCopyExtParams copyOutParamT1 = {static_cast<uint16_t>(wArgmaxActual_),
                                        static_cast<uint32_t>(cOutputActual_ * sizeof(T1)),
                                        static_cast<uint32_t>((cOutput_ - cOutputActual_) * sizeof(T1)),
                                        static_cast<uint32_t>(0), static_cast<uint32_t>(0)};
    DataCopyPad(gradLocal, gradGm_[argmaxGmOffset], copyOutParamT1, paramsT1);

    DataCopyPadExtParams<T2> paramsT2 = {false, 0, 0, 0};
    LoopModeParams loopModeParamsT2;
    loopModeParamsT2.loop1Size = hArgmaxActual_;
    loopModeParamsT2.loop2Size = nOutputActual_;
    loopModeParamsT2.loop1SrcStride = wArgmax_ * cOutput_ * sizeof(T2);
    loopModeParamsT2.loop2SrcStride = argmaxPlaneSize_ * cOutput_ * sizeof(T2);
    loopModeParamsT2.loop1DstStride = wArgmaxActual_ * cOutputAligned_ * sizeof(T2);
    loopModeParamsT2.loop2DstStride = hArgmaxActual_ * wArgmaxActual_ * cOutputAligned_ * sizeof(T2);

    uint32_t dstStrideT2 = (cOutputAligned_ - cOutputActual_) * sizeof(T2) / BLOCK_SIZE;
    SetLoopModePara(loopModeParamsT2, DataCopyMVType::OUT_TO_UB);
    DataCopyExtParams copyOutParamT2 = {static_cast<uint16_t>(wArgmaxActual_),
                                        static_cast<uint32_t>(cOutputActual_ * sizeof(T2)),
                                        static_cast<uint32_t>((cOutput_ - cOutputActual_) * sizeof(T2)),
                                        static_cast<uint32_t>(dstStrideT2), static_cast<uint32_t>(0)};

    DataCopyPad(argmaxLocal, argmaxGm_[argmaxGmOffset], copyOutParamT2, paramsT2);
    ResetLoopModePara(DataCopyMVType::OUT_TO_UB);
    gradQue_.EnQue(gradLocal);
    argmaxQue_.EnQue(argmaxLocal);
}

template <typename T1, typename T2, typename T3, const uint32_t IS_CHECK_RANGE, int32_t VER, typename DerivedT>
__aicore__ inline void MaxPoolGradWithArgmaxKernelNHWCBase<T1, T2, T3, IS_CHECK_RANGE, VER, DerivedT>::CopyOut()
{
    LocalTensor<T1> yLocal = outputQue_.DeQue<T1>();

    int64_t outputPlaneSize = hOutput_ * wOutput_ * cOutput_;
    int64_t nOutputAxisOffset = nAxisIndex_ * nOutputInner_ * outputPlaneSize;
    int64_t hOutputAxisOffset = hAxisIndex_ * hOutputInner_ * wOutput_ * cOutput_;
    int64_t wOutputAxisOffset = wAxisIndex_ * wOutputInner_ * cOutput_;
    int64_t cOutputAxisOffset = cAxisIndex_ * cOutputInner_;
    int64_t outputGmOffset = nOutputAxisOffset + hOutputAxisOffset + wOutputAxisOffset + cOutputAxisOffset;

    LoopModeParams loopModeParamsT1;
    loopModeParamsT1.loop1Size = hOutputActual_;
    loopModeParamsT1.loop2Size = nOutputActual_;
    loopModeParamsT1.loop1SrcStride = wOutputActual_ * cOutputAligned_ * sizeof(T1);
    loopModeParamsT1.loop2SrcStride = hOutputActual_ * wOutputActual_ * cOutputAligned_ * sizeof(T1);
    loopModeParamsT1.loop1DstStride = wOutput_ * cOutput_ * sizeof(T1);
    loopModeParamsT1.loop2DstStride = outputPlaneSize * sizeof(T1);

    SetLoopModePara(loopModeParamsT1, DataCopyMVType::UB_TO_OUT);
    DataCopyExtParams copyOutParamT1 = {static_cast<uint16_t>(wOutputActual_),
                                        static_cast<uint32_t>(cOutputActual_ * sizeof(T1)), static_cast<uint32_t>(0),
                                        static_cast<uint32_t>((cOutput_ - cOutputActual_) * sizeof(T1)),
                                        static_cast<uint32_t>(0)};

    DataCopyPad(yGm_[outputGmOffset], yLocal, copyOutParamT1);
    ResetLoopModePara(DataCopyMVType::UB_TO_OUT);
    outputQue_.FreeTensor(yLocal);
}

template <typename T1, typename T2, typename T3, const uint32_t IS_CHECK_RANGE, int32_t VER, typename DerivedT>
__aicore__ inline void MaxPoolGradWithArgmaxKernelNHWCBase<T1, T2, T3, IS_CHECK_RANGE, VER, DerivedT>::Compute()
{
    uint32_t calCount = outputBufferSize_ / sizeof(computeType);
    LocalTensor<computeType> yLocal = outputQue_.AllocTensor<computeType>();
    Duplicate(yLocal, computeType(0), calCount);

    LocalTensor<T1> gradLocal = gradQue_.DeQue<T1>();
    LocalTensor<T2> argmaxLocal = argmaxQue_.DeQue<T2>();

    __local_mem__ computeType* yAddr = (__local_mem__ computeType*)yLocal.GetPhyAddr();
    __local_mem__ T1* gradAddr = (__local_mem__ T1*)gradLocal.GetPhyAddr();
    __local_mem__ T2* argmaxAddr = (__local_mem__ T2*)argmaxLocal.GetPhyAddr();

    LocalTensor<uint32_t> helpTensor = helpBuf_.Get<uint32_t>();
    __local_mem__ uint32_t* helpAddr = (__local_mem__ uint32_t*)helpTensor.GetPhyAddr();

    static_cast<DerivedT*>(this)->ConCProcVF(yAddr, gradAddr, argmaxAddr, helpAddr);

    if constexpr (std::negation<std::is_same<T1, float>>::value) {
        Cast(yLocal.ReinterpretCast<T1>(), yLocal, RoundMode::CAST_RINT, calCount);
    }

    outputQue_.EnQue(yLocal);
    gradQue_.FreeTensor(gradLocal);
    argmaxQue_.FreeTensor(argmaxLocal);
}

template <typename T1, typename T2, typename T3, const uint32_t IS_CHECK_RANGE, int32_t VER, typename DerivedT>
__aicore__ inline void MaxPoolGradWithArgmaxKernelNHWCBase<T1, T2, T3, IS_CHECK_RANGE, VER, DerivedT>::ProcessPerLoop()
{
    if (hArgmaxActual_ <= 0 || wArgmaxActual_ <= 0) {
        ProcessNoArgmaxBlock(); // ceilMode为false时，最后的尾块可能是这种情况
        return;
    }

    CopyIn();
    Compute();
    CopyOut();
}

template <typename T1, typename T2, typename T3, const uint32_t IS_CHECK_RANGE, int32_t VER, typename DerivedT>
__aicore__ inline void MaxPoolGradWithArgmaxKernelNHWCBase<T1, T2, T3, IS_CHECK_RANGE, VER, DerivedT>::Process()
{
    if (blockIdx_ >= usedCoreNum_) {
        return;
    }

    for (int64_t loopNum = 0; loopNum < curCoreProcessNum_; loopNum++) {
        ScalarCompute(loopNum);
        ProcessPerLoop();
    }
}
} // namespace MaxPoolGradWithArgmaxNHWCNameSpace
#endif // MAX_POOL_GRAD_WITH_ARGMAX__NHWC_KERNEL_H_

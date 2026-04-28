/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef MAX_POOL_GRAD_BIG_KERNEL_H_
#define MAX_POOL_GRAD_BIG_KERNEL_H_

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "max_pool_grad_struct.h"
#include "../inc/platform.h"
#include "max_pool_grad_nchw_backward_base.h"
#include "../pool_3d_common/arch35/pool_big_kernel_utils.h"
#include "../../max_pool_with_argmax_v3/arch35/max_pool_with_argmax_v3_base.h"
#include <algorithm>
#include <type_traits>

namespace MaxPoolGradNCHWBigKernelNameSpace {
using namespace AscendC;
using MaxPoolGradNCHWTilingCommonData = MaxPoolGradWithArgmaxNHWCNameSpace::MaxPoolGradWithArgmaxNCHWTilingCommonData;
using PoolBigKernelUtils::CalcRealIndex;   
using PoolBigKernelUtils::DuplicateNegInf; 
using PoolBigKernelUtils::LoadOneElement;  
using PoolBigKernelUtils::LoadOneTensor;
using PoolBigKernelUtils::ReduceMaxWithIndex; 
using PoolBigKernelUtils::StoreOneElement;    

constexpr int32_t BK_BUFFER_NUM = 2;
constexpr int32_t MERGE_BUF_ALIGN = 32;

template <typename T1, typename T2, typename T3, const uint32_t IS_CHECK_RANGE>
class MaxPoolGradNCHWBigKernel
    : public MaxPoolGradNCHWBackwardBaseNameSpace::MaxPoolGradNCHWBackwardBase<T1, T2, T3, IS_CHECK_RANGE> {
    using Base = MaxPoolGradNCHWBackwardBaseNameSpace::MaxPoolGradNCHWBackwardBase<T1, T2, T3, IS_CHECK_RANGE>;

public:
    __aicore__ inline MaxPoolGradNCHWBigKernel()
    {}

    __aicore__ inline void Init(
        GM_ADDR x, GM_ADDR origY, GM_ADDR grad, GM_ADDR y, TPipe& pipeIn,
        const MaxPoolGradNCHWTilingCommonData& tilingData)
    {
        Base::ParseTilingData(tilingData);
        Base::blockIdx_ = GetBlockIdx();
        if (Base::blockIdx_ >= Base::usedCoreNum_) {
            return;
        }

        Base::curCoreProcessNum_ =
            (Base::blockIdx_ + 1 == Base::usedCoreNum_) ? Base::tailCoreProcessNum_ : Base::normalCoreProcessNum_;

        xGm_.SetGlobalBuffer((__gm__ T1*)x);
        origYGm_.SetGlobalBuffer((__gm__ T1*)origY);
        Base::gradGm_.SetGlobalBuffer((__gm__ T1*)grad);
        Base::yGm_.SetGlobalBuffer((__gm__ T1*)y);

        inHW_ = Base::hOutput_ * Base::wOutput_;
        int64_t maxCountBySize = Base::inputBufferSize_ / static_cast<int64_t>(sizeof(T1));
        maxCount_ = (maxCountBySize < 1) ? 1 : maxCountBySize;

        const int64_t forwardStageBufferSize = Base::inputBufferSize_ * BK_BUFFER_NUM + MERGE_BUF_ALIGN;
        const int64_t backwardStageBufferSize =
            Base::gradBufferSize_ * BK_BUFFER_NUM + Base::outputBufferSize_ * BK_BUFFER_NUM;
        totalStageBufferSize_ =
            (forwardStageBufferSize > backwardStageBufferSize) ? forwardStageBufferSize : backwardStageBufferSize;

        pipeIn.InitBuffer(Base::argmaxBuf_, Base::argmaxBufferSize_);
        pipeIn.InitBuffer(Base::helpBuf_, MaxPoolGradNCHWBackwardBaseNameSpace::HELP_BUFFER);
        pipeIn.InitBufPool(forwardBufPool_, static_cast<uint32_t>(totalStageBufferSize_));
        pipeIn.InitBufPool(backwardBufPool_, static_cast<uint32_t>(totalStageBufferSize_), forwardBufPool_);
    }

    __aicore__ inline void Process()
    {
        if (Base::blockIdx_ >= Base::usedCoreNum_) {
            return;
        }

        for (int64_t loopNum = 0; loopNum < Base::curCoreProcessNum_; ++loopNum) {
            Base::ScalarCompute(loopNum);

            if (Base::hArgmaxActual_ <= 0 || Base::wArgmaxActual_ <= 0) {
                InitBackwardStageBuffers();
                Base::ProcessNoArgmaxBlock();
                backwardBufPool_.Reset();
                continue;
            }

            InitForwardStageBuffers();
            ForwardComputeTile();
            forwardBufPool_.Reset();

            InitBackwardStageBuffers();
            Base::CopyInGrad();
            BackwardScatter();
            Base::CopyOut();
            backwardBufPool_.Reset();
        }
    }

private:
    TBufPool<TPosition::VECCALC> forwardBufPool_;
    TBufPool<TPosition::VECCALC> backwardBufPool_;
    TQue<QuePosition::VECIN, BK_BUFFER_NUM> inputQue_;
    TBuf<TPosition::VECCALC> maxValBuf_;
    GlobalTensor<T1> xGm_;
    GlobalTensor<T1> origYGm_;
    int64_t inHW_ = 1;
    int64_t maxCount_ = 1;
    int64_t totalStageBufferSize_ = 0;

private:
    __aicore__ inline void InitForwardStageBuffers()
    {
        forwardBufPool_.InitBuffer(inputQue_, BK_BUFFER_NUM, static_cast<uint32_t>(Base::inputBufferSize_));
        forwardBufPool_.InitBuffer(maxValBuf_, static_cast<uint32_t>(MERGE_BUF_ALIGN));
    }

    __aicore__ inline void InitBackwardStageBuffers()
    {
        backwardBufPool_.InitBuffer(Base::outputQue_, BK_BUFFER_NUM, static_cast<uint32_t>(Base::outputBufferSize_));
        backwardBufPool_.InitBuffer(Base::gradQue_, BK_BUFFER_NUM, static_cast<uint32_t>(Base::gradBufferSize_));
    }

    __aicore__ inline void ForwardComputeTile()
    {
        LocalTensor<T2> argmaxLocal = Base::argmaxBuf_.template Get<T2>();
        Duplicate(argmaxLocal, T2(-1), Base::argmaxBufferSize_ / sizeof(T2));
        PipeBarrier<PIPE_ALL>();

        const int64_t tileHW = Base::hArgmaxActual_ * Base::wArgmaxActual_;
        for (int64_t highIdx = 0; highIdx < Base::highAxisActual_; ++highIdx) {
            for (int64_t hIdx = 0; hIdx < Base::hArgmaxActual_; ++hIdx) {
                for (int64_t wIdx = 0; wIdx < Base::wArgmaxActual_; ++wIdx) {
                    int64_t curkH = 0;
                    int64_t curkW = 0;
                    int64_t curInOffset = 0;
                    int64_t curOriginIndex = 0;
                    CalcKernelSize(highIdx, hIdx, wIdx, curkH, curkW, curInOffset, curOriginIndex);

                    if (curkH <= 0 || curkW <= 0) {
                        continue;
                    }

                    const int64_t bufferOffset = highIdx * tileHW + hIdx * Base::wArgmaxActual_ + wIdx;
                    if (curkH * curkW <= maxCount_) {
                        NoSplitKernelProcess(curkH, curkW, curInOffset, curOriginIndex, bufferOffset);
                    } else {
                        SplitKernelProcess(curkH, curkW, curInOffset, curOriginIndex, bufferOffset);
                    }
                    PipeBarrier<PIPE_ALL>();
                }
            }
        }
    }

    __aicore__ inline void BackwardScatter()
    {
        const uint32_t calcCount = Base::outputBufferSize_ / sizeof(computeType);

        LocalTensor<computeType> yLocal = Base::outputQue_.template AllocTensor<computeType>();
        Duplicate(yLocal, computeType(0), calcCount);
        PipeBarrier<PIPE_ALL>();

        LocalTensor<T1> gradLocal = Base::gradQue_.template DeQue<T1>();
        LocalTensor<T2> argmaxLocal = Base::argmaxBuf_.template Get<T2>();
        __local_mem__ computeType* yAddr = (__local_mem__ computeType*)yLocal.GetPhyAddr();
        __local_mem__ T1* gradAddr = (__local_mem__ T1*)gradLocal.GetPhyAddr();
        __local_mem__ T2* argmaxAddr = (__local_mem__ T2*)argmaxLocal.GetPhyAddr();

        Base::BackwardCompute(yAddr, gradAddr, argmaxAddr);
        PipeBarrier<PIPE_ALL>();

        if constexpr (!std::is_same<T1, float>::value) {
            Cast(yLocal.ReinterpretCast<T1>(), yLocal, RoundMode::CAST_RINT, calcCount);
            PipeBarrier<PIPE_ALL>();
        }

        Base::outputQue_.EnQue(yLocal);
        Base::gradQue_.FreeTensor(gradLocal);
    }

    __aicore__ inline void CalcKernelSize(
        int64_t highIdx, int64_t hIdx, int64_t wIdx, int64_t& curkH, int64_t& curkW, int64_t& curInOffset,
        int64_t& curOriginIndex)
    {
        const int64_t ncOffset = Base::highAxisIndex_ * Base::highAxisInner_ + highIdx;
        const int64_t ho = Base::hArgmaxActualStart_ + hIdx;
        const int64_t wo = Base::wArgmaxActualStart_ + wIdx;

        int64_t curOriginH = ho * Base::strideH_ - Base::padH_;
        int64_t curOriginW = wo * Base::strideW_ - Base::padW_;
        curkH = Base::kernelH_;
        curkW = Base::kernelW_;

        if (curOriginH < 0) {
            curkH += curOriginH;
            curOriginH = 0;
        }
        if (curOriginH + curkH > Base::hOutput_) {
            curkH = Base::hOutput_ - curOriginH;
        }

        if (curOriginW < 0) {
            curkW += curOriginW;
            curOriginW = 0;
        }
        if (curOriginW + curkW > Base::wOutput_) {
            curkW = Base::wOutput_ - curOriginW;
        }

        curOriginIndex = curOriginH * Base::wOutput_ + curOriginW; 
        curInOffset = ncOffset * inHW_ + curOriginIndex;           
    }

    __aicore__ inline void CopyInMultiRows(int64_t offset, int64_t blockLen, int64_t blockCount)
    {
        LocalTensor<T1> xLocal = inputQue_.AllocTensor<T1>();

        DataCopyPadExtParams<T1> padExtParams = {false, 0, 0, 0};
        DataCopyExtParams extParams;
        extParams.blockCount = blockCount;
        extParams.blockLen = blockLen * sizeof(T1);
        extParams.srcStride = (Base::wOutput_ - blockLen) * sizeof(T1);
        extParams.dstStride = 0;

        DataCopyPad<T1, PaddingMode::Compact>(xLocal, xGm_[offset], extParams, padExtParams);
        PipeBarrier<PIPE_ALL>();
        inputQue_.EnQue(xLocal);
    }

    __aicore__ inline void CopyInSingleRow(int64_t offset, int64_t blockLen)
    {
        LocalTensor<T1> xLocal = inputQue_.AllocTensor<T1>();

        DataCopyPadExtParams<T1> padExtParams = {false, 0, 0, 0};
        DataCopyExtParams extParams;
        extParams.blockCount = 1;
        extParams.blockLen = blockLen * sizeof(T1);
        extParams.srcStride = 0;
        extParams.dstStride = 0;

        DataCopyPad(xLocal, xGm_[offset], extParams, padExtParams);
        PipeBarrier<PIPE_ALL>();
        inputQue_.EnQue(xLocal);
    }

    __aicore__ inline void NoSplitKernelProcess(
        int64_t curkH, int64_t curkW, int64_t curInOffset, int64_t curOriginIndex, int64_t bufferOffset)
    {
        CopyInMultiRows(curInOffset, curkW, curkH);
        ComputeSingleArgmax<false, false>(curkW * curkH, curkW, curOriginIndex, bufferOffset);
    }

    __aicore__ inline void SplitKernelProcess(
        int64_t curkH, int64_t curkW, int64_t curInOffset, int64_t curOriginIndex, int64_t bufferOffset)
    {
        InitMergeBuffer(bufferOffset, curOriginIndex);

        if (curkW <= 0 || curkH <= 0 || maxCount_ <= 0) {
            return;
        }

        if (curkW <= maxCount_) {
            const int64_t hFactor = maxCount_ / ((curkW > 0) ? curkW : 1);
            const int64_t hLoops = (curkH + hFactor - 1) / hFactor;
            const int64_t hTail = curkH - (hLoops - 1) * hFactor;

            int64_t inputOffset = curInOffset;
            int64_t kernelOffset = curOriginIndex;
            for (int64_t hLoop = 0; hLoop < hLoops; ++hLoop) {
                const int64_t curhFactor = (hLoop == hLoops - 1) ? hTail : hFactor;
                CopyInMultiRows(inputOffset, curkW, curhFactor);
                ComputeSingleArgmax<true, false>(
                    curkW * curhFactor, ((curkW > 0) ? curkW : 1), kernelOffset, bufferOffset);
                PipeBarrier<PIPE_ALL>();
                inputOffset += curhFactor * Base::wOutput_;
                kernelOffset += curhFactor * Base::wOutput_;
            }
        } else {
            const int64_t hLoops = curkH;
            const int64_t wFactor = maxCount_;
            const int64_t wLoops = (curkW + wFactor - 1) / wFactor;
            const int64_t wTail = curkW - (wLoops - 1) * wFactor;

            for (int64_t hLoop = 0; hLoop < hLoops; ++hLoop) {
                int64_t inputOffset = curInOffset + hLoop * Base::wOutput_;
                int64_t kernelOffset = curOriginIndex + hLoop * Base::wOutput_;
                for (int64_t wLoop = 0; wLoop < wLoops; ++wLoop) {
                    const int64_t curFactor = (wLoop == wLoops - 1) ? wTail : wFactor;
                    CopyInSingleRow(inputOffset, curFactor);
                    ComputeSingleArgmax<true, true>(
                        curFactor, ((curkW > 0) ? curkW : 1), kernelOffset, bufferOffset);
                    PipeBarrier<PIPE_ALL>();
                    inputOffset += curFactor;
                    kernelOffset += curFactor;
                }
            }
        }
    }

    __aicore__ inline void InitMergeBuffer(int64_t bufferOffset, int64_t initIndex)
    {
        LocalTensor<T2> argmaxLocal = Base::argmaxBuf_.template Get<T2>();
        __local_mem__ T2* argmaxAddr = (__local_mem__ T2*)argmaxLocal.GetPhyAddr();

        LocalTensor<float> maxValLocal = maxValBuf_.template Get<float>();
        __local_mem__ float* maxValAddr = (__local_mem__ float*)maxValLocal.GetPhyAddr();

        const float negInf = AscendC::NumericLimits<float>::NegativeInfinity();

        __VEC_SCOPE__
        {
            MicroAPI::RegTensor<float> negInfVal;
            MicroAPI::Duplicate(negInfVal, negInf);
            MicroAPI::MaskReg pregOne = MicroAPI::CreateMask<float, MicroAPI::MaskPattern::VL1>();
            StoreOneElement<float, float>(maxValAddr, negInfVal, pregOne, 0);

            MicroAPI::RegTensor<T2> initResIndex;
            MicroAPI::Duplicate(initResIndex, static_cast<T2>(initIndex));
            StoreOneElement<T2, T2>(argmaxAddr, initResIndex, pregOne, bufferOffset);
        }
        PipeBarrier<PIPE_ALL>();
    }

    template <bool MERGE, bool SPLITKW>
    __aicore__ inline void ComputeSingleArgmax(
        int64_t dataCount, int64_t curKw, int64_t curOriginIndex, int64_t bufferOffset)
    {
        LocalTensor<T1> xLocal = inputQue_.DeQue<T1>();

        __local_mem__ T1* xLocalAddr = (__local_mem__ T1*)xLocal.GetPhyAddr();

        LocalTensor<T2> argmaxLocal = Base::argmaxBuf_.template Get<T2>();
        __local_mem__ T2* argmaxAddr = (__local_mem__ T2*)argmaxLocal.GetPhyAddr();

        LocalTensor<float> maxValLocal = maxValBuf_.template Get<float>();
        __local_mem__ float* maxValAddr = (__local_mem__ float*)maxValLocal.GetPhyAddr();

        const float negInf = AscendC::NumericLimits<float>::NegativeInfinity();

        constexpr uint32_t repeatElm = platform::GetVRegSize() / sizeof(float); 
        const uint16_t repeatTimes =
            static_cast<uint16_t>((dataCount + repeatElm - 1) / repeatElm); 
        uint32_t num = repeatTimes * repeatElm;                            
        const uint32_t padNum = num - dataCount;                            
        constexpr int32_t padIndex = -1;

        __VEC_SCOPE__
        {
            DuplicateNegInf<T1>(xLocalAddr, padNum, dataCount);

            MicroAPI::RegTensor<float> res;        
            MicroAPI::RegTensor<int32_t> resIndex; 
            MicroAPI::RegTensor<int32_t> index;
            MicroAPI::RegTensor<float> vd0;

            MicroAPI::MaskReg nanMaskReg; 
            MicroAPI::MaskReg cmpMaskReg; 
            MicroAPI::MaskReg maskAll = MicroAPI::CreateMask<float, MicroAPI::MaskPattern::ALL>();

            MicroAPI::Duplicate(resIndex, padIndex);
            MicroAPI::Duplicate(res, negInf);
            MicroAPI::Arange(index, 0);

            for (uint16_t i = 0; i < repeatTimes; ++i) {
                uint32_t maskNum = num;
                MicroAPI::MaskReg p0 = MicroAPI::UpdateMask<float>(maskNum);
                MicroAPI::AddrReg offset = MicroAPI::CreateAddrReg<T1>(i, repeatElm);
                LoadOneTensor<T1>(xLocalAddr, vd0, p0, offset);

                MicroAPI::Compare<float, CMPMODE::NE>(nanMaskReg, vd0, vd0, maskAll); 
                MicroAPI::Compare<float, CMPMODE::GT>(cmpMaskReg, vd0, res, maskAll); 
                MicroAPI::MaskXor(cmpMaskReg, cmpMaskReg, nanMaskReg, maskAll);      
                MicroAPI::Select(res, vd0, res, cmpMaskReg);
                MicroAPI::Select(resIndex, index, resIndex, cmpMaskReg);
                MicroAPI::Adds(index, index, repeatElm, maskAll);
            }

            ReduceMaxWithIndex<float>(res, index, res, resIndex, padIndex);

            MicroAPI::MaskReg pregOne = MicroAPI::CreateMask<float, MicroAPI::MaskPattern::VL1>();
            MicroAPI::RegTensor<T2> realResIndex;

            CalcRealIndex<T2, SPLITKW>(realResIndex, index, curKw, Base::wOutput_, curOriginIndex);

            if constexpr (MERGE) {
                MicroAPI::RegTensor<T2> lastResIndex;
                LoadOneElement<T2, T2>(argmaxAddr, lastResIndex, pregOne, bufferOffset);

                MicroAPI::RegTensor<float> lastRes;
                LoadOneElement<float, float>(maxValAddr, lastRes, pregOne, 0);

                MicroAPI::MaskReg curNanMaskReg;
                MicroAPI::MaskReg selReg;

                MicroAPI::Compare<float, CMPMODE::NE>(curNanMaskReg, res, res, maskAll);
                MicroAPI::Compare<float, CMPMODE::GT>(selReg, res, lastRes, maskAll);
                MicroAPI::MaskXor(selReg, selReg, curNanMaskReg, maskAll);

                MicroAPI::Select(res, res, lastRes, selReg);
                MicroAPI::Select(realResIndex, realResIndex, lastResIndex, selReg);

                MicroAPI::LocalMemBar<MicroAPI::MemType::VEC_LOAD, MicroAPI::MemType::VEC_STORE>();
            }

            StoreOneElement<T2, T2>(argmaxAddr, realResIndex, pregOne, bufferOffset);
            if constexpr (MERGE) {
                StoreOneElement<float, float>(maxValAddr, res, pregOne, 0);
            }
        }

        PipeBarrier<PIPE_ALL>();
        inputQue_.FreeTensor(xLocal);
    }
};

} // namespace MaxPoolGradNCHWBigKernelNameSpace
#endif // MAX_POOL_GRAD_BIG_KERNEL_H_
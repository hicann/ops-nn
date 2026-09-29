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
 * \file max_pool3d_grad_ncdhw_impl_big_kernel.h
 * \brief NCDHW Big Kernel 正向实现（逐位置 MaxPool + argmax）
 */

#ifndef MAX_POOL3D_GRAD_NCDHW_IMPL_BIG_KERNEL_H
#define MAX_POOL3D_GRAD_NCDHW_IMPL_BIG_KERNEL_H

#include "max_pool3d_grad_ncdhw_kernel.h"
#include "../../pool_3d_common/arch35/pool_big_kernel_utils.h"
#include "../inc/kernel_utils.h"

namespace MaxPool3DSmallKernelNameSpace {
using PoolBigKernelUtils::CalcRealIndex;
using PoolBigKernelUtils::CalcRealIndex3D;
using PoolBigKernelUtils::DuplicateNegInf;
using PoolBigKernelUtils::LoadOneElement;
using PoolBigKernelUtils::LoadOneTensor;
using PoolBigKernelUtils::ReduceMaxWithIndex;
using PoolBigKernelUtils::StoreOneElement;

template <typename TYPE_ORIG_X, typename TYPE_ARGMAX, typename T3, const uint32_t IS_CHECK_RANGE>
__aicore__ inline void Pool3DGradSmallKernel<TYPE_ORIG_X, TYPE_ARGMAX, T3, IS_CHECK_RANGE>::ForwardBigKernel()
{
    LocalTensor<TYPE_ARGMAX> argmaxLocal = argmaxBuff_.Get<TYPE_ARGMAX>();
    Duplicate(argmaxLocal, TYPE_ARGMAX(-1), argmaxBufferSize_ / sizeof(TYPE_ARGMAX));

    const int64_t tileDHW = dArgmaxActual_ * hArgmaxActual_ * wArgmaxActual_;
    for (int64_t highIdx = 0; highIdx < highAxisActual_; ++highIdx) {
        for (int64_t dIdx = 0; dIdx < dArgmaxActual_; ++dIdx) {
            for (int64_t hIdx = 0; hIdx < hArgmaxActual_; ++hIdx) {
                for (int64_t wIdx = 0; wIdx < wArgmaxActual_; ++wIdx) {
                    int64_t curkD = 0;
                    int64_t curkH = 0;
                    int64_t curkW = 0;
                    int64_t curInOffset = 0;
                    int64_t curOriginIndex = 0;
                    int64_t curOriginD = 0;
                    int64_t curOriginH = 0;
                    int64_t curOriginW = 0;
                    CalcKernelSize(highIdx, dIdx, hIdx, wIdx, curkD, curkH, curkW, curInOffset, curOriginIndex,
                                   curOriginD, curOriginH, curOriginW);
                    if (curkD <= 0 || curkH <= 0 || curkW <= 0) {
                        continue;
                    }

                    const int64_t bufferOffset = highIdx * tileDHW + dIdx * hArgmaxActual_ * wArgmaxActual_ +
                                                 hIdx * wArgmaxActual_ + wIdx;
                    // 判据用对齐后的装载量 (与 tiling 侧 fullLoadCount 一致), 避免封顶后 NoSplit 越界
                    const int64_t alignElems = static_cast<int64_t>(BLOCK_SIZE) /
                                               static_cast<int64_t>(sizeof(TYPE_ORIG_X));
                    const int64_t hwAligned = ops::CeilAlign(curkH * curkW, alignElems);
                    if (curkD * hwAligned <= maxCount_) {
                        NoSplitKernelProcess(curkD, curkH, curkW, curInOffset, curOriginIndex, bufferOffset);
                    } else {
                        SplitKernelProcess(curkD, curkH, curkW, curInOffset, curOriginIndex, bufferOffset);
                    }
                }
            }
        }
    }
}

template <typename TYPE_ORIG_X, typename TYPE_ARGMAX, typename T3, const uint32_t IS_CHECK_RANGE>
__aicore__ inline void Pool3DGradSmallKernel<TYPE_ORIG_X, TYPE_ARGMAX, T3, IS_CHECK_RANGE>::CalcKernelSize(
    int64_t highIdx, int64_t dIdx, int64_t hIdx, int64_t wIdx, int64_t& curkD, int64_t& curkH, int64_t& curkW,
    int64_t& curInOffset, int64_t& curOriginIndex, int64_t& curOriginD, int64_t& curOriginH, int64_t& curOriginW)
{
    const int64_t ncOffset = highAxisIndex_ * highAxisInner_ + highIdx;
    const int64_t do_ = dArgmaxActualStart + dIdx;
    const int64_t ho = hArgmaxActualStart + hIdx;
    const int64_t wo = wArgmaxActualStart + wIdx;

    curOriginD = do_ * strideD_ - padD_;
    curOriginH = ho * strideH_ - padH_;
    curOriginW = wo * strideW_ - padW_;
    curkD = kernelD_;
    curkH = kernelH_;
    curkW = kernelW_;

    if (curOriginD < 0) {
        curkD += curOriginD;
        curOriginD = 0;
    }
    if (curOriginD + curkD > dOutput_) {
        curkD = dOutput_ - curOriginD;
    }

    if (curOriginH < 0) {
        curkH += curOriginH;
        curOriginH = 0;
    }
    if (curOriginH + curkH > hOutput_) {
        curkH = hOutput_ - curOriginH;
    }

    if (curOriginW < 0) {
        curkW += curOriginW;
        curOriginW = 0;
    }
    if (curOriginW + curkW > wOutput_) {
        curkW = wOutput_ - curOriginW;
    }

    curOriginIndex = curOriginD * hOutput_ * wOutput_ + curOriginH * wOutput_ + curOriginW;
    curInOffset = ncOffset * inDHW_ + curOriginIndex;
}

template <typename TYPE_ORIG_X, typename TYPE_ARGMAX, typename T3, const uint32_t IS_CHECK_RANGE>
__aicore__ inline void Pool3DGradSmallKernel<TYPE_ORIG_X, TYPE_ARGMAX, T3, IS_CHECK_RANGE>::CopyInMultiRows(
    int64_t offset, int64_t blockLen, int64_t blockCount)
{
    LocalTensor<TYPE_ORIG_X> xLocal = inputQue_.AllocTensor<TYPE_ORIG_X>();

    DataCopyPadExtParams<TYPE_ORIG_X> padExtParams = {false, 0, 0, 0};
    DataCopyExtParams extParams;
    extParams.blockCount = static_cast<uint16_t>(blockCount);
    extParams.blockLen = static_cast<uint32_t>(blockLen * sizeof(TYPE_ORIG_X));
    extParams.srcStride = static_cast<uint32_t>((wOutput_ - blockLen) * sizeof(TYPE_ORIG_X));
    extParams.dstStride = 0;

    DataCopyPad<TYPE_ORIG_X, PaddingMode::Compact>(xLocal, xGm_[offset], extParams, padExtParams);
    inputQue_.EnQue(xLocal);
}

template <typename TYPE_ORIG_X, typename TYPE_ARGMAX, typename T3, const uint32_t IS_CHECK_RANGE>
__aicore__ inline void Pool3DGradSmallKernel<TYPE_ORIG_X, TYPE_ARGMAX, T3, IS_CHECK_RANGE>::CopyInSingleRow(
    int64_t offset, int64_t blockLen)
{
    LocalTensor<TYPE_ORIG_X> xLocal = inputQue_.AllocTensor<TYPE_ORIG_X>();

    DataCopyPadExtParams<TYPE_ORIG_X> padExtParams = {false, 0, 0, 0};
    DataCopyExtParams extParams;
    extParams.blockCount = 1;
    extParams.blockLen = static_cast<uint32_t>(blockLen * sizeof(TYPE_ORIG_X));
    extParams.srcStride = 0;
    extParams.dstStride = 0;

    DataCopyPad(xLocal, xGm_[offset], extParams, padExtParams);
    inputQue_.EnQue(xLocal);
}

template <typename TYPE_ORIG_X, typename TYPE_ARGMAX, typename T3, const uint32_t IS_CHECK_RANGE>
__aicore__ inline void Pool3DGradSmallKernel<TYPE_ORIG_X, TYPE_ARGMAX, T3, IS_CHECK_RANGE>::CopyInMultiRowsWithD(
    int64_t curkD, int64_t curkH, int64_t curkW, int64_t curInOffset, int64_t hwAligned)
{
    LocalTensor<TYPE_ORIG_X> xLocal = inputQue_.AllocTensor<TYPE_ORIG_X>();
    const TYPE_ORIG_X negInf = static_cast<TYPE_ORIG_X>(AscendC::NumericLimits<float>::NegativeInfinity());
    DataCopyPadExtParams<TYPE_ORIG_X> padExtParams = {true, 0, 0, negInf};
    DataCopyExtParams extParams;
    extParams.blockCount = static_cast<uint16_t>(curkH);
    extParams.blockLen = static_cast<uint32_t>(curkW * sizeof(TYPE_ORIG_X));
    extParams.srcStride = static_cast<uint32_t>((wOutput_ - curkW) * sizeof(TYPE_ORIG_X));
    extParams.dstStride = 0;

    // D 维用 loop mode 合并为单条搬运: GM/UB 层间距均为起点 pitch, 层尾对齐 gap 由 isPad 的 -inf 填充
    LoopModeParams loopParams;
    loopParams.loop2Size = 1;
    loopParams.loop2SrcStride = 0;
    loopParams.loop2DstStride = 0;
    loopParams.loop1Size = static_cast<uint32_t>(curkD);
    loopParams.loop1SrcStride = static_cast<uint64_t>(hOutput_ * wOutput_ * sizeof(TYPE_ORIG_X));
    loopParams.loop1DstStride = static_cast<uint64_t>(hwAligned * sizeof(TYPE_ORIG_X));

    SetLoopModePara(loopParams, DataCopyMVType::OUT_TO_UB);
    DataCopyPad<TYPE_ORIG_X, PaddingMode::Compact>(xLocal, xGm_[curInOffset], extParams, padExtParams);
    ResetLoopModePara(DataCopyMVType::OUT_TO_UB);
    inputQue_.EnQue(xLocal);
}

template <typename TYPE_ORIG_X, typename TYPE_ARGMAX, typename T3, const uint32_t IS_CHECK_RANGE>
__aicore__ inline void Pool3DGradSmallKernel<TYPE_ORIG_X, TYPE_ARGMAX, T3, IS_CHECK_RANGE>::NoSplitKernelProcess(
    int64_t curkD, int64_t curkH, int64_t curkW, int64_t curInOffset, int64_t curOriginIndex, int64_t bufferOffset)
{
    const int64_t alignElems = BLOCK_SIZE / sizeof(TYPE_ORIG_X);
    const int64_t hwAligned = ops::CeilAlign(curkH * curkW, alignElems);
    const int64_t totalAligned = curkD * hwAligned;

    CopyInMultiRowsWithD(curkD, curkH, curkW, curInOffset, hwAligned);
    ComputeSingleArgmax<false, false, true>(totalAligned, curkW, curOriginIndex, bufferOffset, hwAligned);
}

template <typename TYPE_ORIG_X, typename TYPE_ARGMAX, typename T3, const uint32_t IS_CHECK_RANGE>
__aicore__ inline void Pool3DGradSmallKernel<TYPE_ORIG_X, TYPE_ARGMAX, T3, IS_CHECK_RANGE>::SplitKernelProcess(
    int64_t curkD, int64_t curkH, int64_t curkW, int64_t curInOffset, int64_t curOriginIndex, int64_t bufferOffset)
{
    InitMergeBuffer(bufferOffset, curOriginIndex);

    if (curkW <= 0 || curkH <= 0 || curkD <= 0 || maxCount_ <= 0) {
        return;
    }

    if (curkW <= maxCount_) {
        const int64_t hFactor = maxCount_ / ((curkW > 0) ? curkW : 1);
        const int64_t dhLoops = curkD * ((curkH + hFactor - 1) / hFactor);
        int64_t inputOffset = curInOffset;
        int64_t kernelOffset = curOriginIndex;

        for (int64_t dhLoop = 0; dhLoop < dhLoops; ++dhLoop) {
            int64_t dIdx = dhLoop / ((curkH + hFactor - 1) / hFactor);
            int64_t hIdx = dhLoop % ((curkH + hFactor - 1) / hFactor);
            int64_t hLoops = (curkH + hFactor - 1) / hFactor;
            int64_t curhFactor = (hIdx == hLoops - 1) ? (curkH - (hLoops - 1) * hFactor) : hFactor;

            inputOffset = curInOffset + dIdx * hOutput_ * wOutput_ + hIdx * hFactor * wOutput_;
            kernelOffset = curOriginIndex + dIdx * hOutput_ * wOutput_ + hIdx * hFactor * wOutput_;

            CopyInMultiRows(inputOffset, curkW, curhFactor);
            ComputeSingleArgmax<true, false>(curkW * curhFactor, ((curkW > 0) ? curkW : 1), kernelOffset, bufferOffset);
        }
    } else {
        const int64_t wFactor = maxCount_;
        const int64_t wLoops = (curkW + wFactor - 1) / wFactor;
        const int64_t wTail = curkW - (wLoops - 1) * wFactor;

        for (int64_t dLoop = 0; dLoop < curkD; ++dLoop) {
            for (int64_t hLoop = 0; hLoop < curkH; ++hLoop) {
                int64_t inputOffset = curInOffset + dLoop * hOutput_ * wOutput_ + hLoop * wOutput_;
                int64_t kernelOffset = curOriginIndex + dLoop * hOutput_ * wOutput_ + hLoop * wOutput_;
                for (int64_t wLoop = 0; wLoop < wLoops; ++wLoop) {
                    const int64_t curFactor = (wLoop == wLoops - 1) ? wTail : wFactor;
                    CopyInSingleRow(inputOffset, curFactor);
                    ComputeSingleArgmax<true, true>(curFactor, ((curkW > 0) ? curkW : 1), kernelOffset, bufferOffset);
                    inputOffset += curFactor;
                    kernelOffset += curFactor;
                }
            }
        }
    }
}

template <typename TYPE_ORIG_X, typename TYPE_ARGMAX, typename T3, const uint32_t IS_CHECK_RANGE>
__aicore__ inline void Pool3DGradSmallKernel<TYPE_ORIG_X, TYPE_ARGMAX, T3, IS_CHECK_RANGE>::InitMergeBuffer(
    int64_t bufferOffset, int64_t initIndex)
{
    LocalTensor<TYPE_ARGMAX> argmaxLocal = argmaxBuff_.Get<TYPE_ARGMAX>();
    __local_mem__ TYPE_ARGMAX* argmaxAddr = (__local_mem__ TYPE_ARGMAX*)argmaxLocal.GetPhyAddr();

    LocalTensor<float> maxValLocal = maxValBuf_.Get<float>();
    __local_mem__ float* maxValAddr = (__local_mem__ float*)maxValLocal.GetPhyAddr();
    const float negInf = AscendC::NumericLimits<float>::NegativeInfinity();

    __VEC_SCOPE__
    {
        Reg::RegTensor<float> negInfVal;
        Reg::Duplicate(negInfVal, negInf);
        Reg::MaskReg pregOne = Reg::CreateMask<float, Reg::MaskPattern::VL1>();
        StoreOneElement<float, float>(maxValAddr, negInfVal, pregOne, 0);

        Reg::RegTensor<int32_t> initResIndex;
        Reg::Duplicate(initResIndex, static_cast<int32_t>(initIndex));
        StoreOneElement<int32_t, int32_t>(argmaxAddr, initResIndex, pregOne, static_cast<uint32_t>(bufferOffset));
    }
}

template <typename TYPE_ORIG_X, typename TYPE_ARGMAX, typename T3, const uint32_t IS_CHECK_RANGE>
template <bool MERGE, bool SPLITKW, bool IS_3D>
__aicore__ inline void Pool3DGradSmallKernel<TYPE_ORIG_X, TYPE_ARGMAX, T3, IS_CHECK_RANGE>::ComputeSingleArgmax(
    int64_t dataCount, int64_t curKw, int64_t curOriginIndex, int64_t bufferOffset, int64_t hwStride)
{
    LocalTensor<TYPE_ORIG_X> xLocal = inputQue_.DeQue<TYPE_ORIG_X>();
    __local_mem__ TYPE_ORIG_X* xLocalAddr = (__local_mem__ TYPE_ORIG_X*)xLocal.GetPhyAddr();

    LocalTensor<TYPE_ARGMAX> argmaxLocal = argmaxBuff_.Get<TYPE_ARGMAX>();
    __local_mem__ TYPE_ARGMAX* argmaxAddr = (__local_mem__ TYPE_ARGMAX*)argmaxLocal.GetPhyAddr();

    LocalTensor<float> maxValLocal = maxValBuf_.Get<float>();
    __local_mem__ float* maxValAddr = (__local_mem__ float*)maxValLocal.GetPhyAddr();
    const float negInf = AscendC::NumericLimits<float>::NegativeInfinity();

    constexpr uint32_t repeatElm = platform::GetVRegSize() / sizeof(float);
    const uint16_t repeatTimes = static_cast<uint16_t>((dataCount + repeatElm - 1) / repeatElm);
    uint32_t num = repeatTimes * repeatElm;
    const uint32_t padNum = num - dataCount;
    constexpr int32_t padIndex = -1;

    __VEC_SCOPE__
    {
        DuplicateNegInf<TYPE_ORIG_X>(xLocalAddr, padNum, dataCount);

        Reg::RegTensor<float> res;
        Reg::RegTensor<int32_t> resIndex;
        Reg::RegTensor<int32_t> index;
        Reg::RegTensor<float> vd0;

        Reg::MaskReg nanMaskReg;
        Reg::MaskReg cmpMaskReg;
        Reg::MaskReg maskAll = Reg::CreateMask<float, Reg::MaskPattern::ALL>();

        Reg::Duplicate(resIndex, padIndex);
        Reg::Duplicate(res, negInf);
        Reg::Arange(index, 0);

        for (uint16_t i = 0; i < repeatTimes; ++i) {
            uint32_t maskNum = num;
            Reg::MaskReg p0 = Reg::UpdateMask<float>(maskNum);
            Reg::AddrReg offset = Reg::CreateAddrReg<TYPE_ORIG_X>(i, repeatElm);
            LoadOneTensor<TYPE_ORIG_X>(xLocalAddr, vd0, p0, offset);

            Reg::Compare<float, CMPMODE::NE>(nanMaskReg, vd0, vd0, maskAll);
            Reg::Compare<float, CMPMODE::GT>(cmpMaskReg, vd0, res, maskAll);
            Reg::MaskXor(cmpMaskReg, cmpMaskReg, nanMaskReg, maskAll);
            Reg::Select(res, vd0, res, cmpMaskReg);
            Reg::Select(resIndex, index, resIndex, cmpMaskReg);
            Reg::Adds(index, index, repeatElm, maskAll);
        }
        ReduceMaxWithIndex<float>(res, index, res, resIndex, padIndex);
        Reg::MaskReg pregOne = Reg::CreateMask<float, Reg::MaskPattern::VL1>();
        Reg::RegTensor<int32_t> realResIndex;

        if constexpr (IS_3D) {
            CalcRealIndex3D<int32_t>(realResIndex, index, curKw, hwStride, hOutput_, wOutput_, curOriginIndex);
        } else {
            CalcRealIndex<int32_t, SPLITKW>(realResIndex, index, curKw, wOutput_, curOriginIndex);
        }

        if constexpr (MERGE) {
            Reg::RegTensor<int32_t> lastResIndex;
            LoadOneElement<int32_t, int32_t>(argmaxAddr, lastResIndex, pregOne, static_cast<uint32_t>(bufferOffset));

            Reg::RegTensor<float> lastRes;
            LoadOneElement<float, float>(maxValAddr, lastRes, pregOne, 0);

            Reg::MaskReg curNanMaskReg;
            Reg::MaskReg selReg;
            Reg::Compare<float, CMPMODE::NE>(curNanMaskReg, res, res, maskAll);
            Reg::Compare<float, CMPMODE::GT>(selReg, res, lastRes, maskAll);
            Reg::MaskXor(selReg, selReg, curNanMaskReg, maskAll);

            Reg::Select(res, res, lastRes, selReg);
            Reg::Select(realResIndex, realResIndex, lastResIndex, selReg);
            Reg::LocalMemBar<Reg::MemType::VEC_LOAD, Reg::MemType::VEC_STORE>();
            StoreOneElement<float, float>(maxValAddr, res, pregOne, 0);
        }

        StoreOneElement<int32_t, int32_t>(argmaxAddr, realResIndex, pregOne, static_cast<uint32_t>(bufferOffset));
    }
    inputQue_.FreeTensor(xLocal);
    // rls_buf 可能先于 VF 完成而提前释放 bufId, 需屏障保证 VF 读完成后再允许 MTE2 复写
    event_t eventIDVToMTE2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE2));
    SetFlag<HardEvent::V_MTE2>(eventIDVToMTE2);
    WaitFlag<HardEvent::V_MTE2>(eventIDVToMTE2);
}

} // namespace MaxPool3DSmallKernelNameSpace
#endif // MAX_POOL3D_GRAD_NCDHW_IMPL_BIG_KERNEL_H

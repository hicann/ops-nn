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
 * \file max_pool3d_grad_small_kernel_impl_gather.h
 * \brief
 */
#ifndef MAX_POOL3D_GRAD_SMALL_KERNEL_IMPL_GATHER_H
#define MAX_POOL3D_GRAD_SMALL_KERNEL_IMPL_GATHER_H

#include "max_pool3d_grad_small_kernel_gather.h"
#include "max_pool3d_grad_small_kernel.h"

namespace MaxPool3DSmallKernelNameSpace {
template <typename TYPE_ORIG_X, typename TYPE_ARGMAX, typename T3, const uint32_t IS_CHECK_RANGE>
__aicore__ inline void Pool3DGradSmallKernel<TYPE_ORIG_X, TYPE_ARGMAX, T3, IS_CHECK_RANGE>::ConvertIndexWithoutPadAlign(
    MicroAPI::RegTensor<int32_t>& srcReg, uint32_t wStrideOffset, uint32_t hInputActualPad, TYPE_ARGMAX left, TYPE_ARGMAX wInput, TYPE_ARGMAX hIndexBase, TYPE_ARGMAX hInput,TYPE_ARGMAX dIndexBase,
    MicroAPI::RegTensor<TYPE_ARGMAX>& dstReg, int32_t ncInputOffset)
{
    MicroAPI::RegTensor<TYPE_ARGMAX> hIndexReg;
    MicroAPI::RegTensor<TYPE_ARGMAX> dIndexReg;
    MicroAPI::RegTensor<int32_t> constReg;
    MicroAPI::RegTensor<int32_t> t3Reg;
    MicroAPI::RegTensor<int32_t> divResultReg;
    MicroAPI::RegTensor<int32_t> t1Reg;
    MicroAPI::RegTensor<int32_t> t2Reg;
    MicroAPI::RegTensor<TYPE_ARGMAX> divResultRegUnpack;
    MicroAPI::RegTensor<TYPE_ARGMAX> dIndexRegUnpack;
    MicroAPI::RegTensor<TYPE_ARGMAX> wIndexReg;
    MicroAPI::RegTensor<int32_t> wIndexRegUnpack;
    MicroAPI::RegTensor<TYPE_ARGMAX> zeroReg;
    MicroAPI::MaskReg negInfMask;
    MicroAPI::MaskReg allMaskB32 = MicroAPI::CreateMask<int32_t, MicroAPI::MaskPattern::ALL>();
    MicroAPI::MaskReg allMaskT2 = MicroAPI::CreateMask<TYPE_ARGMAX, MicroAPI::MaskPattern::ALL>();
    MicroAPI::Duplicate(constReg, static_cast<int32_t>(wStrideOffset));
    MicroAPI::Duplicate(t3Reg, static_cast<int32_t>(wStrideOffset * hInputActualPad));
    MicroAPI::Duplicate(zeroReg, static_cast<TYPE_ARGMAX>(0));
    MicroAPI::Adds(srcReg, srcReg, -ncInputOffset, allMaskB32);
    MicroAPI::Div(t1Reg, srcReg, t3Reg, allMaskB32);
    MicroAPI::Muls(t2Reg, t1Reg, static_cast<int32_t>(-1 * wStrideOffset * hInputActualPad), allMaskB32);
    MicroAPI::Add(t2Reg, srcReg, t2Reg, allMaskB32);
    MicroAPI::Div(divResultReg, t2Reg, constReg, allMaskB32);
    MicroAPI::Mul(t3Reg, divResultReg, constReg, allMaskB32);
    MicroAPI::Sub(wIndexRegUnpack, t2Reg, t3Reg, allMaskB32);
    if constexpr (std::is_same<TYPE_ARGMAX, int64_t>::value) {
        MicroAPI::UnPack(dIndexRegUnpack, t1Reg);
        MicroAPI::UnPack(divResultRegUnpack, divResultReg);
        MicroAPI::UnPack(wIndexReg, wIndexRegUnpack);
        MicroAPI::Adds(dIndexReg, dIndexRegUnpack, dIndexBase, allMaskT2);
        MicroAPI::Adds(hIndexReg, divResultRegUnpack, hIndexBase, allMaskT2);
        MicroAPI::Adds(wIndexReg, wIndexReg, left, allMaskT2);
    } else {
        MicroAPI::Adds(dIndexReg, t1Reg, dIndexBase, allMaskB32);
        MicroAPI::Adds(hIndexReg, divResultReg, hIndexBase, allMaskB32);
        MicroAPI::Adds(wIndexReg, wIndexRegUnpack, left, allMaskB32);
    }
    if (IS_PAD) {
        MicroAPI::Compare<TYPE_ARGMAX, CMPMODE::LT>(negInfMask, dIndexReg, zeroReg, allMaskT2);
        MicroAPI::Select(dIndexReg, zeroReg, dIndexReg, negInfMask);
        MicroAPI::Compare<TYPE_ARGMAX, CMPMODE::LT>(negInfMask, hIndexReg, zeroReg, allMaskT2);
        MicroAPI::Select(hIndexReg, zeroReg, hIndexReg, negInfMask);
        MicroAPI::Compare<TYPE_ARGMAX, CMPMODE::LT>(negInfMask, wIndexReg, zeroReg, allMaskT2);
        MicroAPI::Select(wIndexReg, zeroReg, wIndexReg, negInfMask);
    }
    MicroAPI::Muls(dIndexReg, dIndexReg, (wInput * hInput), allMaskT2);
    MicroAPI::Muls(hIndexReg, hIndexReg, wInput, allMaskT2);
    MicroAPI::Add(dstReg, hIndexReg, dIndexReg, allMaskT2);
    MicroAPI::Add(dstReg, dstReg, wIndexReg, allMaskT2);
    return;
}

template <typename TYPE_ORIG_X, typename TYPE_ARGMAX, typename T3, const uint32_t IS_CHECK_RANGE>
__aicore__ inline void Pool3DGradSmallKernel<TYPE_ORIG_X, TYPE_ARGMAX, T3, IS_CHECK_RANGE>::ProcessW(
    __local_mem__ TYPE_ORIG_X* computeAddr, int32_t hOffset, uint16_t wStrideOffset, uint16_t hInputActualPad,
    MicroAPI::RegTensor<int32_t>& indexReg, uint16_t dKernel, uint16_t hKernel, uint16_t wKernel, uint16_t repeatElem,
    MicroAPI::RegTensor<int32_t>& maxIndexReg, uint32_t dDilation ,uint32_t hDilation, uint32_t wDilation)
{
    MicroAPI::RegTensor<int32_t> indexWithOffset;
    MicroAPI::RegTensor<TYPE_ORIG_X> calcReg;
    MicroAPI::RegTensor<int32_t> calcMaxIndexReg;
    uint32_t maskCount = repeatElem;
    MicroAPI::MaskReg allMaskU32 = MicroAPI::CreateMask<int32_t, MicroAPI::MaskPattern::ALL>();
    MicroAPI::MaskReg gatherMask = MicroAPI::UpdateMask<TYPE_ORIG_X>(maskCount);
    MicroAPI::RegTensor<TYPE_ORIG_X> maxReg; 
    MicroAPI::MaskReg neMask;
    MicroAPI::MaskReg gtMask;
    MicroAPI::MaskReg tmpMask;
    MicroAPI::UnalignReg u0;

    SetNegInfReg<TYPE_ORIG_X>(maxReg);

    MicroAPI::Adds(maxIndexReg, indexReg, hOffset, allMaskU32);

    for (uint16_t d = 0; d < dKernel; d++) {
        for (uint16_t i = 0; i < hKernel; i++) {
            for (uint16_t j = 0; j < wKernel; j++) {
                int32_t relIndex = d * hInputActualPad * wStrideOffset * dDilation  + i * wStrideOffset * hDilation + j * wDilation;
                int32_t offset = static_cast<int32_t>(hOffset + relIndex);
                MicroAPI::Adds(indexWithOffset, indexReg, offset, allMaskU32);
                if constexpr (std::is_same<TYPE_ORIG_X, float>::value) {
                    MicroAPI::DataCopyGather(calcReg, computeAddr, (MicroAPI::RegTensor<uint32_t>&)indexWithOffset,
                                            gatherMask);
                } else {
                    MicroAPI::RegTensor<uint16_t> indexConvert;
                    MicroAPI::Pack(indexConvert, indexWithOffset);
                    MicroAPI::DataCopyGather(calcReg, computeAddr, indexConvert, gatherMask);
                }
                MicroAPI::Compare<TYPE_ORIG_X, CMPMODE::GT>(gtMask, calcReg, maxReg, gatherMask);
                MicroAPI::Compare<TYPE_ORIG_X, CMPMODE::NE>(neMask, calcReg, calcReg, gatherMask);
                MicroAPI::MaskOr(gtMask, gtMask, neMask, gatherMask);
                if constexpr (sizeof(int32_t) / sizeof(TYPE_ORIG_X) == 1) {
                    MicroAPI::Select(maxIndexReg, indexWithOffset, maxIndexReg, gtMask);
                } else {
                    MicroAPI::MaskUnPack(tmpMask, gtMask);
                    MicroAPI::Select(maxIndexReg, indexWithOffset, maxIndexReg, tmpMask);
                }
                MicroAPI::Max(maxReg, maxReg, calcReg, gatherMask);
            }
        }
    }
    return;
}

template <typename TYPE_ORIG_X, typename TYPE_ARGMAX, typename T3, const uint32_t IS_CHECK_RANGE>
__aicore__ inline void Pool3DGradSmallKernel<TYPE_ORIG_X, TYPE_ARGMAX, T3, IS_CHECK_RANGE>::ConvertIndexWithoutPadAlignNc(
    MicroAPI::RegTensor<int32_t>& srcReg, uint32_t wStrideOffset,int32_t hInputActualPad, TYPE_ARGMAX left, TYPE_ARGMAX wInput, TYPE_ARGMAX hIndexBase,TYPE_ARGMAX hInput, TYPE_ARGMAX dIndexBase,
    MicroAPI::RegTensor<TYPE_ARGMAX>& dstReg, int32_t ncInputOffset, int32_t ncOutputCount, int32_t inputNcSize)
{
    MicroAPI::RegTensor<int32_t> ncIndexReg;
    MicroAPI::RegTensor<int32_t> divResultReg;
    MicroAPI::RegTensor<int32_t> constReg;
    MicroAPI::MaskReg allMaskB32 = MicroAPI::CreateMask<int32_t, MicroAPI::MaskPattern::ALL>();
    MicroAPI::Arange(ncIndexReg, static_cast<int32_t>(0));
    MicroAPI::Duplicate(constReg, static_cast<int32_t>(ncOutputCount));
    MicroAPI::Div(divResultReg, ncIndexReg, constReg, allMaskB32);
    MicroAPI::Muls(divResultReg, divResultReg, inputNcSize, allMaskB32);
    MicroAPI::Sub(srcReg, srcReg, divResultReg, allMaskB32);
    ConvertIndexWithoutPadAlign(srcReg, wStrideOffset, hInputActualPad, left, wInput, hIndexBase, hInput, dIndexBase,
                                                dstReg, ncInputOffset);
}

template <typename TYPE_ORIG_X, typename TYPE_ARGMAX, typename T3, const uint32_t IS_CHECK_RANGE>
__aicore__ inline void Pool3DGradSmallKernel<TYPE_ORIG_X, TYPE_ARGMAX, T3, IS_CHECK_RANGE>::MultiNcGather(__local_mem__ TYPE_ORIG_X* computeAddr,
                                                                                      __local_mem__ TYPE_ARGMAX* argmaxAddr)
{
    uint16_t dKernel = tilingData_.dKernel;
    uint16_t hKernel = tilingData_.hKernel;
    uint16_t wKernel = tilingData_.wKernel;
    uint32_t wStride = tilingData_.wStride;
    uint16_t rate4D = dInputActualPad_ * hInputActualPad_ * wInputActualAlignedPad_;
    uint16_t num3D = dArgmaxActual_ * hArgmaxActual_ * wArgmaxActual_;
    uint16_t rate3D = tilingData_.dStride *  hInputActualPad_ * wInputActualAlignedPad_;
    uint16_t num2D = hArgmaxActual_ * wArgmaxActual_;
    uint16_t rate2D = tilingData_.hStride * wInputActualAlignedPad_;
    uint16_t wOutputActual = wArgmaxActual_;
    uint16_t eachBatchCount = dArgmaxActual_ * hArgmaxActual_ * wArgmaxActual_;
    uint16_t ncBatchCount = vlT2_ / eachBatchCount;
    uint16_t ncLoopTimes = highAxisActual_ / ncBatchCount;
    uint16_t ncTail = highAxisActual_ - ncLoopTimes * ncBatchCount;
    if (ncTail == 0) {
        ncLoopTimes = ncLoopTimes - 1;
        ncTail = ncBatchCount;
    }
    uint16_t repeatsElem = ncBatchCount * eachBatchCount;
    uint16_t tailRepeatsElem = ncTail * eachBatchCount;
    TYPE_ARGMAX left = wArgmaxActualStart * tilingData_.wStride - tilingData_.padW;
    TYPE_ARGMAX hIndexBase = hArgmaxActualStart * tilingData_.hStride - tilingData_.padH;
    TYPE_ARGMAX dIndexBase = dArgmaxActualStart * tilingData_.dStride - tilingData_.padD;
    TYPE_ARGMAX hInput = tilingData_.hOutput;
    TYPE_ARGMAX wInput = tilingData_.wOutput;
    uint32_t dInputActualPad = dInputActualPad_;
    uint32_t hInputActualPad = hInputActualPad_;
    uint32_t wInputActualAlignedPad = wInputActualAlignedPad_;
    uint32_t dOutputActual = dArgmaxActual_;
    uint32_t hOutputActual = hArgmaxActual_;
    uint32_t wOutputActualAligned = wOutputAligned_;
    uint32_t dDilation = tilingData_.dilationD;
    uint32_t hDilation = tilingData_.dilationH;
    uint32_t wDilation = tilingData_.dilationW;
    __VEC_SCOPE__
    {
        MicroAPI::RegTensor<int32_t> indexReg;
        MicroAPI::RegTensor<int32_t> maxIndexReg;
        MicroAPI::RegTensor<TYPE_ARGMAX> maxIndexConvertReg;
        MicroAPI::UnalignReg u1;
        __local_mem__ TYPE_ARGMAX* argmaxAddrLocal = argmaxAddr;
        CalGatterIndex4D<int32_t>(indexReg, rate4D, num3D, rate3D, num2D, rate2D, wOutputActual, wStride);
        for (uint16_t nc = 0; nc < ncLoopTimes; nc++) {
            uint32_t wOffset = nc * ncBatchCount * dInputActualPad * hInputActualPad * wInputActualAlignedPad;
            ProcessW(computeAddr, wOffset, wInputActualAlignedPad, hInputActualPad, 
                            indexReg, dKernel,hKernel, wKernel,repeatsElem, 
                            maxIndexReg, dDilation, hDilation, wDilation);
            ConvertIndexWithoutPadAlignNc(maxIndexReg, wInputActualAlignedPad, hInputActualPad,left, wInput, hIndexBase, hInput, dIndexBase,
                                          maxIndexConvertReg, wOffset, num3D, rate4D);
            MicroAPI::DataCopyUnAlign(argmaxAddrLocal, maxIndexConvertReg, u1, repeatsElem);
            MicroAPI::DataCopyUnAlignPost(argmaxAddrLocal, u1, 0);
        }
        uint32_t wOffsetTail = ncLoopTimes * ncBatchCount * dInputActualPad * hInputActualPad * wInputActualAlignedPad;
        ProcessW(computeAddr, wOffsetTail, wInputActualAlignedPad, hInputActualPad, 
                            indexReg, dKernel,hKernel, wKernel,tailRepeatsElem, 
                            maxIndexReg, dDilation, hDilation, wDilation);
        ConvertIndexWithoutPadAlignNc(maxIndexReg, wInputActualAlignedPad, hInputActualPad, left, wInput, hIndexBase, hInput, dIndexBase,
                                          maxIndexConvertReg, wOffsetTail, num3D, rate4D);
        MicroAPI::DataCopyUnAlign(argmaxAddrLocal, maxIndexConvertReg, u1, tailRepeatsElem);
        MicroAPI::DataCopyUnAlignPost(argmaxAddrLocal, u1, 0);
    }
    return;
}

template <typename TYPE_ORIG_X, typename TYPE_ARGMAX, typename T3, const uint32_t IS_CHECK_RANGE>
__aicore__ inline void Pool3DGradSmallKernel<TYPE_ORIG_X, TYPE_ARGMAX, T3, IS_CHECK_RANGE>::MultiDepGather(__local_mem__ TYPE_ORIG_X* computeAddr,
                                                                                       __local_mem__ TYPE_ARGMAX* argmaxAddr)
{
    uint16_t wKernel = tilingData_.wKernel;
    uint16_t hKernel = tilingData_.hKernel;
    uint16_t dKernel = tilingData_.dKernel;
    uint32_t wStride = tilingData_.wStride;
    uint16_t rate3D = tilingData_.dStride *  hInputActualPad_ * wInputActualAlignedPad_;
    uint16_t num2D = hArgmaxActual_ * wArgmaxActual_;
    uint16_t rate2D = tilingData_.hStride * wInputActualAlignedPad_;
    uint16_t wOutputActual = wArgmaxActual_;
    uint16_t eachDepCount = hArgmaxActual_ * wArgmaxActual_;
    uint16_t dBatchCount = vlT2_ / eachDepCount;
    uint16_t dLoopTimes = dArgmaxActual_ / dBatchCount;
    uint16_t dTail = dArgmaxActual_ - dLoopTimes * dBatchCount;
    if (dTail == 0) {
        dLoopTimes = dLoopTimes - 1;
        dTail = dBatchCount;
    }
    uint16_t repeatsElem = dBatchCount * eachDepCount;
    uint16_t tailRepeatsElem = dTail * eachDepCount;
    TYPE_ARGMAX left = wArgmaxActualStart * tilingData_.wStride - tilingData_.padW;
    TYPE_ARGMAX hIndexBase = hArgmaxActualStart * tilingData_.hStride - tilingData_.padH;
    TYPE_ARGMAX dIndexBase = dArgmaxActualStart * tilingData_.dStride - tilingData_.padD;
    TYPE_ARGMAX hInput = tilingData_.hOutput;
    TYPE_ARGMAX wInput = tilingData_.wOutput;
    uint32_t highAxisActual = highAxisActual_;
    uint32_t dInputActualPad = dInputActualPad_;
    uint32_t hInputActualPad = hInputActualPad_;
    uint32_t wInputActualAlignedPad = wInputActualAlignedPad_;
    uint32_t wOutputActualAligned = wOutputAligned_;
    uint32_t dOutputActual = dArgmaxActual_;
    uint32_t hOutputActual = hArgmaxActual_;
    uint32_t dStride = tilingData_.dStride;
    uint32_t hStride = tilingData_.hStride;
    uint32_t dDilation = tilingData_.dilationD;
    uint32_t hDilation = tilingData_.dilationH;
    uint32_t wDilation = tilingData_.dilationW;
    for (uint16_t nc = 0; nc < static_cast<uint16_t>(highAxisActual); nc++) {
        __VEC_SCOPE__
        {
            MicroAPI::RegTensor<int32_t> indexReg;
            MicroAPI::RegTensor<int32_t> maxIndexReg;
            MicroAPI::RegTensor<TYPE_ARGMAX> maxIndexConvertReg;
            MicroAPI::UnalignReg u1;
            __local_mem__ TYPE_ARGMAX* argmaxAddrLocal = argmaxAddr;
            CalGatterIndex3D<int32_t>(indexReg, rate3D, num2D, rate2D, wOutputActual, wStride);
            argmaxAddrLocal = argmaxAddr + nc * dOutputActual * hOutputActual * wOutputActual;
            int32_t ncInputOffset = nc * dInputActualPad * hInputActualPad * wInputActualAlignedPad;
            for(uint16_t dLoop = 0; dLoop < static_cast<uint16_t>(dLoopTimes); dLoop++) {
                int32_t wOffset = ncInputOffset + dLoop * dBatchCount * dStride * hInputActualPad * wInputActualAlignedPad;
                ProcessW(computeAddr, wOffset, wInputActualAlignedPad, hInputActualPad, 
                            indexReg, dKernel,hKernel, wKernel,repeatsElem, 
                            maxIndexReg, dDilation, hDilation, wDilation);
                ConvertIndexWithoutPadAlign(maxIndexReg, wInputActualAlignedPad, hInputActualPad, left, wInput, hIndexBase, hInput, dIndexBase,
                                            maxIndexConvertReg, ncInputOffset);
                MicroAPI::DataCopyUnAlign(argmaxAddrLocal, maxIndexConvertReg, u1, repeatsElem);
                MicroAPI::DataCopyUnAlignPost(argmaxAddrLocal, u1, 0);
            }
            int32_t wOffsetTail = ncInputOffset + dLoopTimes * dBatchCount  *  dStride * hInputActualPad * wInputActualAlignedPad;
            ProcessW(computeAddr, wOffsetTail, wInputActualAlignedPad, hInputActualPad, 
                            indexReg, dKernel ,hKernel, wKernel,tailRepeatsElem, 
                            maxIndexReg, dDilation, hDilation, wDilation);
            ConvertIndexWithoutPadAlign(maxIndexReg, wInputActualAlignedPad, hInputActualPad, left, wInput, hIndexBase, hInput, dIndexBase,
                                            maxIndexConvertReg, ncInputOffset);
            MicroAPI::DataCopyUnAlign(argmaxAddrLocal, maxIndexConvertReg, u1, tailRepeatsElem);
            MicroAPI::DataCopyUnAlignPost(argmaxAddrLocal, u1, 0);
        }
    }
    return;
}

template <typename TYPE_ORIG_X, typename TYPE_ARGMAX, typename T3, const uint32_t IS_CHECK_RANGE>
__aicore__ inline void Pool3DGradSmallKernel<TYPE_ORIG_X, TYPE_ARGMAX, T3, IS_CHECK_RANGE>::MultiRowGather(__local_mem__ TYPE_ORIG_X* computeAddr,
                                                                                         __local_mem__ TYPE_ARGMAX* argmaxAddr)
{
    uint16_t dKernel = tilingData_.dKernel;
    uint32_t wStride = tilingData_.wStride;
    uint32_t wOutputActual = wArgmaxActual_;
    uint16_t wKernel = tilingData_.wKernel;
    uint16_t hKernel = tilingData_.hKernel;

    uint16_t hBatchCount = vlT2_ / wArgmaxActual_;
    uint32_t rate2D = wInputActualAlignedPad_ * tilingData_.hStride;
    uint16_t hLoopTimes = hArgmaxActual_ / hBatchCount;
    uint16_t hTail = hArgmaxActual_ - hLoopTimes * hBatchCount;
    if (hTail == 0) {
        hLoopTimes = hLoopTimes - 1;
        hTail = hBatchCount;
    }
    uint16_t repeatsElem = hBatchCount * wArgmaxActual_;
    uint16_t tailRepeatsElem = hTail * wArgmaxActual_;
    TYPE_ARGMAX left = wArgmaxActualStart * tilingData_.wStride - tilingData_.padW;
    TYPE_ARGMAX hIndexBase = hArgmaxActualStart * tilingData_.hStride - tilingData_.padH;
    TYPE_ARGMAX dIndexBase = dArgmaxActualStart * tilingData_.dStride - tilingData_.padD;
    TYPE_ARGMAX hInput = tilingData_.hOutput;
    TYPE_ARGMAX wInput = tilingData_.wOutput;
    uint32_t highAxisActual = highAxisActual_;
    uint32_t dInputActualPad = dInputActualPad_;
    uint32_t hInputActualPad = hInputActualPad_;
    uint32_t wInputActualAlignedPad = wInputActualAlignedPad_;
    uint32_t wOutputActualAligned = wOutputAligned_;
    uint32_t dOutputActual = dArgmaxActual_;
    uint32_t hOutputActual = hArgmaxActual_;
    uint32_t dStride = tilingData_.dStride;
    uint32_t hStride = tilingData_.hStride;
    uint32_t dDilation = tilingData_.dilationD;
    uint32_t hDilation = tilingData_.dilationH;
    uint32_t wDilation = tilingData_.dilationW;
    for (uint16_t nc = 0; nc < static_cast<uint16_t>(highAxisActual); nc++) {
        for(uint16_t dLoop = 0; dLoop < static_cast<uint16_t>(dOutputActual); dLoop++) {
            __VEC_SCOPE__
            {
                MicroAPI::RegTensor<int32_t> indexReg;
                MicroAPI::RegTensor<int32_t> maxIndexReg;
                MicroAPI::RegTensor<TYPE_ARGMAX> maxIndexConvertReg;
                MicroAPI::UnalignReg u1;
                __local_mem__ TYPE_ARGMAX* argmaxAddrLocal;
                int32_t ncInputOffset = nc * dInputActualPad * hInputActualPad * wInputActualAlignedPad;
                argmaxAddrLocal = argmaxAddr + nc * dOutputActual * hOutputActual * wOutputActual + dLoop * hOutputActual * wOutputActual;
                CalGatterIndex2D<int32_t>(indexReg, rate2D, wOutputActual, wStride);
                for (uint16_t hLoop = 0; hLoop < hLoopTimes; hLoop++) {
                    int32_t wOffset = ncInputOffset + dLoop * dStride * hInputActualPad * wInputActualAlignedPad  +
                    hLoop * hBatchCount * hStride * wInputActualAlignedPad;
                    ProcessW(computeAddr, wOffset, wInputActualAlignedPad, hInputActualPad, 
                                indexReg, dKernel,hKernel, wKernel,repeatsElem, 
                                maxIndexReg, dDilation, hDilation, wDilation);
                    ConvertIndexWithoutPadAlign(maxIndexReg, wInputActualAlignedPad, hInputActualPad, left, wInput, hIndexBase, hInput, dIndexBase,
                                                maxIndexConvertReg, ncInputOffset);
                    MicroAPI::DataCopyUnAlign(argmaxAddrLocal, maxIndexConvertReg, u1, repeatsElem);
                    MicroAPI::DataCopyUnAlignPost(argmaxAddrLocal, u1, 0);
                }
                int32_t wOffsetTail = ncInputOffset + dLoop * dStride * hInputActualPad * wInputActualAlignedPad  +
                                            hLoopTimes * hBatchCount * hStride * wInputActualAlignedPad;
                ProcessW(computeAddr, wOffsetTail, wInputActualAlignedPad, hInputActualPad, 
                                indexReg, dKernel ,hKernel, wKernel,tailRepeatsElem, 
                                maxIndexReg, dDilation, hDilation, wDilation);
                ConvertIndexWithoutPadAlign(maxIndexReg, wInputActualAlignedPad, hInputActualPad, left, wInput, hIndexBase, hInput, dIndexBase,
                                                maxIndexConvertReg, ncInputOffset);
                MicroAPI::DataCopyUnAlign(argmaxAddrLocal, maxIndexConvertReg, u1, tailRepeatsElem);
                MicroAPI::DataCopyUnAlignPost(argmaxAddrLocal, u1, 0);
            }
        }
    }
    return;
}

template <typename TYPE_ORIG_X, typename TYPE_ARGMAX, typename T3, const uint32_t IS_CHECK_RANGE>
__aicore__ inline void Pool3DGradSmallKernel<TYPE_ORIG_X, TYPE_ARGMAX, T3, IS_CHECK_RANGE>::SingleRowGather(__local_mem__ TYPE_ORIG_X* computeAddr,
                                                                                        __local_mem__ TYPE_ARGMAX* argmaxAddr)
{
    uint16_t loopW = wArgmaxActual_ / vlT2_;
    uint16_t repeatsElem = vlT2_;   
    uint16_t tailRepeatsElem = wArgmaxActual_ - loopW * vlT2_;
    uint16_t dKernel = tilingData_.dKernel;
    uint16_t hKernel = tilingData_.hKernel;
    uint16_t wKernel = tilingData_.wKernel;
    if (tailRepeatsElem == 0) {
        loopW = loopW - 1;
        tailRepeatsElem = repeatsElem;
    }
    uint32_t dStride = tilingData_.dStride;
    uint32_t hStride = tilingData_.hStride;
    uint32_t wStride = tilingData_.wStride;
    TYPE_ARGMAX dIndexBase = dArgmaxActualStart * tilingData_.dStride - tilingData_.padD;
    TYPE_ARGMAX left = wArgmaxActualStart * tilingData_.wStride - tilingData_.padW;
    TYPE_ARGMAX wInput = tilingData_.wOutput;
    TYPE_ARGMAX hInput = tilingData_.hOutput;
    TYPE_ARGMAX hIndexBase = hArgmaxActualStart * tilingData_.hStride - tilingData_.padH;
    uint32_t highAxisActual = highAxisActual_;
    uint32_t dOutputActual = dArgmaxActual_;
    uint32_t hOutputActual = hArgmaxActual_;
    uint32_t wOutputActual = wArgmaxActual_;
    uint32_t dInputActualPad = dInputActualPad_;
    uint32_t hInputActualPad = hInputActualPad_;
    uint32_t wInputActualAlignedPad = wInputActualAlignedPad_;
    uint32_t wOutputActualAligned = wOutputAligned_;
    uint32_t dDilation = tilingData_.dilationD;
    uint32_t hDilation = tilingData_.dilationH;
    uint32_t wDilation = tilingData_.dilationW;
    for (uint16_t nc = 0; nc < static_cast<uint16_t>(highAxisActual); nc++) {
        for(uint16_t dLoop = 0; dLoop < static_cast<uint16_t>(dOutputActual); dLoop++) {
            for (uint16_t hLoop = 0; hLoop < static_cast<uint16_t>(hOutputActual); hLoop++) {
                __VEC_SCOPE__
                {
                    MicroAPI::RegTensor<int32_t> indexReg; 
                    MicroAPI::RegTensor<int32_t> maxIndexReg; 
                    MicroAPI::RegTensor<TYPE_ARGMAX> maxIndexConvertReg; 
                    MicroAPI::UnalignReg u1;  
                    MicroAPI::Arange(indexReg, static_cast<int32_t>(0));
                    MicroAPI::MaskReg preg = MicroAPI::CreateMask<TYPE_ORIG_X, MicroAPI::MaskPattern::ALL>();
                    MicroAPI::Muls(indexReg, indexReg, static_cast<int32_t>(wStride), preg);

                    int32_t ncInOffset = nc * dInputActualPad * hInputActualPad * wInputActualAlignedPad;
                    int32_t ncOutOffset = nc * dOutputActual * hOutputActual * wOutputActual;
                    int32_t vfMaxAddrOffset = ncOutOffset +  dLoop * hOutputActual * wOutputActual  + hLoop * wOutputActual;
                    __local_mem__ TYPE_ARGMAX* argmaxAddrLocal = argmaxAddr + vfMaxAddrOffset;
                    for (uint16_t wLoop = 0; wLoop < loopW; wLoop++) {
                        int32_t wOffset =
                            ncInOffset + dLoop * dStride * hInputActualPad * wInputActualAlignedPad + hLoop * wInputActualAlignedPad * hStride 
                            + wLoop * repeatsElem * wStride;
                        ProcessW(
                            computeAddr, wOffset, wInputActualAlignedPad, hInputActualPad, indexReg, dKernel, hKernel,
                            wKernel, repeatsElem, maxIndexReg, dDilation, hDilation, wDilation);

                        ConvertIndexWithoutPadAlign(maxIndexReg, wInputActualAlignedPad, hInputActualPad, left, wInput, hIndexBase, hInput, dIndexBase,
                                                    maxIndexConvertReg, ncInOffset);
                        MicroAPI::DataCopyUnAlign(argmaxAddrLocal, maxIndexConvertReg, u1, repeatsElem);
                        MicroAPI::DataCopyUnAlignPost(argmaxAddrLocal, u1, 0);
                    }
                    int32_t wOffsetTail =
                            ncInOffset + dLoop * dStride * hInputActualPad * wInputActualAlignedPad + hLoop * wInputActualAlignedPad * hStride 
                            + loopW * repeatsElem * wStride;
                    ProcessW(
                        computeAddr, wOffsetTail, wInputActualAlignedPad, hInputActualPad, indexReg, dKernel, hKernel,
                        wKernel, tailRepeatsElem, maxIndexReg, dDilation, hDilation, wDilation);
                    ConvertIndexWithoutPadAlign(maxIndexReg, wInputActualAlignedPad, hInputActualPad, left, wInput, hIndexBase, hInput, dIndexBase,
                                                maxIndexConvertReg, ncInOffset);
                    MicroAPI::DataCopyUnAlign(argmaxAddrLocal, maxIndexConvertReg, u1, tailRepeatsElem);
                    MicroAPI::DataCopyUnAlignPost(argmaxAddrLocal, u1, 0); 
                }
            }
        }
    }
    return;
}

template <typename TYPE_ORIG_X, typename TYPE_ARGMAX, typename T3, const uint32_t IS_CHECK_RANGE>
__aicore__ inline void Pool3DGradSmallKernel<TYPE_ORIG_X, TYPE_ARGMAX, T3, IS_CHECK_RANGE>::DupBufferNegInf(__local_mem__ TYPE_ORIG_X* dstAddr,
                                                                                        uint32_t repeatElm,
                                                                                        uint16_t loop, uint32_t tail)
{
    MicroAPI::RegTensor<TYPE_ORIG_X> v0;
    SetNegInfReg<TYPE_ORIG_X>(v0);
    MicroAPI::MaskReg preg = MicroAPI::CreateMask<TYPE_ORIG_X, MicroAPI::MaskPattern::ALL>();
    uint32_t maskCount = tail;
    for (uint16_t i = 0; i < loop; i++) {
        MicroAPI::DataCopy<TYPE_ORIG_X, MicroAPI::PostLiteral::POST_MODE_UPDATE>(dstAddr, v0, repeatElm, preg);
    }
    preg = MicroAPI::UpdateMask<TYPE_ORIG_X>(maskCount);
    MicroAPI::DataCopy<TYPE_ORIG_X, MicroAPI::PostLiteral::POST_MODE_UPDATE>(dstAddr, v0, repeatElm, preg);
}

template <typename TYPE_ORIG_X, typename TYPE_ARGMAX, typename T3, const uint32_t IS_CHECK_RANGE>
__aicore__ inline void Pool3DGradSmallKernel<TYPE_ORIG_X, TYPE_ARGMAX, T3, IS_CHECK_RANGE>::CopyToCalcBuffer(
    __local_mem__ TYPE_ORIG_X* dstAddr, __local_mem__ TYPE_ORIG_X* srcAddr, uint16_t batch,uint16_t deps ,uint16_t rows, uint16_t loopCols,
    uint16_t tailCols, uint32_t repeatElm, uint32_t srcBatchStride, uint32_t srcDepStride,uint32_t srcRowStride, 
    uint32_t dstBatchStride, uint32_t dstDepStride, uint32_t dstRowStride, uint32_t dstDepOffset, uint32_t dstRowOffset, uint32_t dstColOffset)
{
    MicroAPI::RegTensor<TYPE_ORIG_X> v0;
    MicroAPI::UnalignReg u0;
    for (uint16_t i = 0; i < batch; i++) {
        for (uint16_t t = 0; t < deps; t++) {
            for (uint16_t j = 0; j < rows; j++) {
                __local_mem__ TYPE_ORIG_X* curSrcAddr = srcAddr + i * srcBatchStride + t * srcDepStride + j * srcRowStride;
                __local_mem__ TYPE_ORIG_X* curDstAddr =
                    dstAddr + i * dstBatchStride + (t + dstDepOffset) * dstDepStride +(j + dstRowOffset) * dstRowStride + dstColOffset;
                for (uint16_t k = 0; k < loopCols; k++) {
                    MicroAPI::DataCopy<TYPE_ORIG_X, MicroAPI::PostLiteral::POST_MODE_UPDATE>(v0, curSrcAddr, repeatElm);
                    MicroAPI::DataCopyUnAlign(curDstAddr, v0, u0, repeatElm);
                }
                MicroAPI::DataCopy<TYPE_ORIG_X, MicroAPI::PostLiteral::POST_MODE_UPDATE>(v0, curSrcAddr, repeatElm);
                MicroAPI::DataCopyUnAlign(curDstAddr, v0, u0, tailCols);
                MicroAPI::DataCopyUnAlignPost(curDstAddr, u0, 0);
            }
        }
    }
}

template <typename TYPE_ORIG_X, typename TYPE_ARGMAX, typename T3, const uint32_t IS_CHECK_RANGE>
__aicore__ inline void Pool3DGradSmallKernel<TYPE_ORIG_X, TYPE_ARGMAX, T3, IS_CHECK_RANGE>::DupAndCopyToCalcBuffer(
    __local_mem__ TYPE_ORIG_X* dstAddr, __local_mem__ TYPE_ORIG_X* srcAddr)
{
    uint16_t loopCols = wInputActualNoPad_ / vlT1_;
    uint16_t tailCols = wInputActualNoPad_ - loopCols * vlT1_;
    uint32_t wInputActualNoPadAlign =
        CeilDivision(wInputActualNoPad_, BLOCK_SIZE / sizeof(TYPE_ORIG_X)) * BLOCK_SIZE / sizeof(TYPE_ORIG_X);
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
        CopyToCalcBuffer(
            dstAddr, srcAddr, highAxisActual_, dInputActualNoPad_, hInputActualNoPad_, loopCols, tailCols, vlT1_,
            srcBatchStride, srcDepStride, wInputActualNoPadAlign, dstBatchStride, dstDepStride, wInputActualAlignedPad_,
            dstDepOffset, dstRowOffset, dstColOffset);
    }
    return;
}
}
#endif // MAX_POOL3D_GRAD_SMALL_KERNEL_IMPL_GATHER_H

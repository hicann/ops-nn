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
 * \file max_pool_with_argmax_v3_nhwc_small_c_.h
 * \brief
 */

#ifndef MAX_POOL_WITH_ARGMAX_V3_NHWC_SMALL_C__H_
#define MAX_POOL_WITH_ARGMAX_V3_NHWC_SMALL_C__H_

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "../inc/platform.h"
#include "max_pool_with_argmax_v3_base.h"
#include "max_pool_with_argmax_v3_nhwc_big_c.h"

namespace MaxPoolWithArgmaxV3SmallCNameSpace {
using namespace AscendC;

constexpr int64_t THREE = 3;
constexpr int64_t DOUBLE = 2;

constexpr AscendC::MicroAPI::CastTrait castTraitU32U16 = {
    AscendC::MicroAPI::RegLayout::ZERO,
    AscendC::MicroAPI::SatMode::NO_SAT,
    AscendC::MicroAPI::MaskMergeMode::ZEROING,
    AscendC::RoundMode::CAST_RINT,
};

template <typename T1, typename T2, const uint32_t IS_PAD = 0>
__aicore__ inline void MaxPoolWithArgMaxV3GatherImpl(
    __local_mem__ T1* xAddr, __local_mem__ T1* maxValueAddr, __local_mem__ T2* argmaxAddr, uint16_t kH, uint16_t kW,
    uint32_t rowStrideInUb, uint16_t alignedC, int32_t gatterIndexOffset, MicroAPI::RegTensor<uint32_t>& gatterStartIdx,
    int32_t count, MicroAPI::RegTensor<T2>& argmaxHStart, MicroAPI::RegTensor<T2>& argmaxWStart, int32_t argmaxHOffset,
    int32_t argmaxWOffset, MicroAPI::RegTensor<uint32_t>& scatterStartIdx, int32_t scatterOffset, int32_t padH,
    int32_t padW, int32_t wInput)
{
    MicroAPI::RegTensor<T1> vd0;
    MicroAPI::RegTensor<T1> vd1;
    MicroAPI::RegTensor<uint32_t> gatterIndexReg;
    MicroAPI::RegTensor<uint32_t> scatterIndexReg;
    MicroAPI::RegTensor<uint16_t> gatterIdxU16Reg;
    MicroAPI::RegTensor<uint16_t> scatterIdxU16Reg;

    AscendC::MicroAPI::RegTensor<T2> argmaxUpdateHVreg;
    AscendC::MicroAPI::RegTensor<T2> argmaxUpdateWVreg;
    AscendC::MicroAPI::RegTensor<T2> argmaxHRes;
    AscendC::MicroAPI::RegTensor<T2> argmaxWRes;

    AscendC::MicroAPI::MaskReg neMask;
    AscendC::MicroAPI::MaskReg gtMask;
    AscendC::MicroAPI::MaskReg gtMaskT2;
    AscendC::MicroAPI::MaskReg gtMaskT4;

    DuplicateNegInfReg<T1>(vd0);

    uint32_t numU32 = count;
    uint32_t numT1 = count;
    uint32_t numT2 = count;

    MicroAPI::MaskReg computeT1 = MicroAPI::UpdateMask<T1>(numT1);
    MicroAPI::MaskReg computeT2 = MicroAPI::UpdateMask<T2>(numT2);
    MicroAPI::MaskReg computeU32 = MicroAPI::UpdateMask<uint32_t>(numU32);
    MicroAPI::MaskReg maskAllU32 = MicroAPI::CreateMask<uint32_t, MicroAPI::MaskPattern::ALL>();
    MicroAPI::MaskReg maskAllT2 = MicroAPI::CreateMask<T2, MicroAPI::MaskPattern::ALL>();

    MicroAPI::Adds(argmaxHRes, argmaxHStart, argmaxHOffset, computeT2);
    MicroAPI::Adds(argmaxWRes, argmaxWStart, argmaxWOffset, computeT2);

    for (uint16_t hIdx = 0; hIdx < kH; hIdx++) {
        int32_t hKernelOffset = hIdx * rowStrideInUb;
        int32_t argmaxHKernelOffset = hIdx + argmaxHOffset;

        for (uint16_t wIdx = 0; wIdx < kW; wIdx++) {
            int32_t wKernelOffset = wIdx * alignedC;
            int32_t argmaxWKernelOffset = wIdx + argmaxWOffset;

            int32_t gatterIndexOffsetTotal = gatterIndexOffset + hKernelOffset + wKernelOffset;
            MicroAPI::Adds(gatterIndexReg, gatterStartIdx, gatterIndexOffsetTotal, computeU32);

            if constexpr (std::is_same<T1, float>::value) {
                MicroAPI::DataCopyGather(vd1, xAddr, gatterIndexReg, computeT1);
            } else {
                AscendC::MicroAPI::Cast<uint16_t, uint32_t, castTraitU32U16>(
                    gatterIdxU16Reg, gatterIndexReg, computeU32);
                AscendC::MicroAPI::Pack(gatterIdxU16Reg, (AscendC::MicroAPI::RegTensor<uint32_t>&)gatterIdxU16Reg);
                AscendC::MicroAPI::DataCopyGather(vd1, xAddr, gatterIdxU16Reg, computeT1);
            }

            AscendC::MicroAPI::Compare<T1, CMPMODE::GT>(gtMask, vd1, vd0, computeT1);
            AscendC::MicroAPI::Compare<T1, CMPMODE::NE>(neMask, vd1, vd1, computeT1);
            AscendC::MicroAPI::MaskOr(gtMask, gtMask, neMask, computeT1);

            MicroAPI::Adds(argmaxUpdateHVreg, argmaxHStart, argmaxHKernelOffset, computeT2);
            MicroAPI::Adds(argmaxUpdateWVreg, argmaxWStart, argmaxWKernelOffset, computeT2);
            if constexpr (sizeof(T2) / sizeof(T1) == 1) {
                AscendC::MicroAPI::Select(argmaxHRes, argmaxUpdateHVreg, argmaxHRes, gtMask);
                AscendC::MicroAPI::Select(argmaxWRes, argmaxUpdateWVreg, argmaxWRes, gtMask);
            } else if constexpr (sizeof(T2) / sizeof(T1) == DOUBLE) {
                AscendC::MicroAPI::MaskUnPack(gtMaskT2, gtMask);
                AscendC::MicroAPI::Select(argmaxHRes, argmaxUpdateHVreg, argmaxHRes, gtMaskT2);
                AscendC::MicroAPI::Select(argmaxWRes, argmaxUpdateWVreg, argmaxWRes, gtMaskT2);
            } else {
                AscendC::MicroAPI::MaskUnPack(gtMaskT2, gtMask);
                AscendC::MicroAPI::MaskUnPack(gtMaskT4, gtMaskT2);
                AscendC::MicroAPI::Select(argmaxHRes, argmaxUpdateHVreg, argmaxHRes, gtMaskT4);
                AscendC::MicroAPI::Select(argmaxWRes, argmaxUpdateWVreg, argmaxWRes, gtMaskT4);
            }

            MicroAPI::Max(vd0, vd1, vd0, computeT1);
        }
    }

    if constexpr (IS_PAD == 1) {
        // 修正argmax
        MicroAPI::Adds(argmaxHRes, argmaxHRes, -padH, computeT2);
        MicroAPI::Adds(argmaxWRes, argmaxWRes, -padW, computeT2);

        AscendC::MicroAPI::MaskReg hMask;
        AscendC::MicroAPI::MaskReg wMask;
        MicroAPI::RegTensor<T2> argmaxZero;
        AscendC::MicroAPI::Duplicate(argmaxZero, 0);

        AscendC::MicroAPI::Compare<T2, CMPMODE::GE>(hMask, argmaxHRes, argmaxZero, computeT2);
        AscendC::MicroAPI::Select(argmaxHRes, argmaxHRes, argmaxZero, hMask);
        AscendC::MicroAPI::Compare<T2, CMPMODE::GE>(wMask, argmaxWRes, argmaxZero, computeT2);
        AscendC::MicroAPI::Select(argmaxWRes, argmaxWRes, argmaxZero, wMask);
    }

    MicroAPI::RegTensor<T2> argmaxRes;
    MicroAPI::Muls(argmaxRes, argmaxHRes, wInput, computeT2);
    MicroAPI::Add(argmaxRes, argmaxRes, argmaxWRes, computeT2);

    MicroAPI::Adds(scatterIndexReg, scatterStartIdx, scatterOffset, computeU32);

    AscendC::MicroAPI::DataCopyScatter(argmaxAddr, argmaxRes, scatterIndexReg, computeT2);
    if constexpr (std::is_same<T1, float>::value) {
        AscendC::MicroAPI::DataCopyScatter(maxValueAddr, vd0, scatterIndexReg, computeT1);
    } else {
        AscendC::MicroAPI::Cast<uint16_t, uint32_t, castTraitU32U16>(scatterIdxU16Reg, scatterIndexReg, computeU32);
        AscendC::MicroAPI::Pack(scatterIdxU16Reg, (AscendC::MicroAPI::RegTensor<uint32_t>&)scatterIdxU16Reg);
        AscendC::MicroAPI::DataCopyScatter(maxValueAddr, vd0, scatterIdxU16Reg, computeT1);
    }
}

template <typename T1, typename T2, const uint32_t IS_PAD = 0>
class MaxPoolWithArgmaxV3SmallC : public MaxPoolWithArgmaxV3NHWC::MaxPoolWithArgmaxV3NhwCKernel<T1, T2, IS_PAD> {
public:
    __aicore__ inline MaxPoolWithArgmaxV3SmallC(TPipe* pipe, const MaxPoolWithArgmaxV3NhwcTilingData* tiling)
        : MaxPoolWithArgmaxV3NHWC::MaxPoolWithArgmaxV3NhwCKernel<T1, T2, IS_PAD>(pipe, tiling){};
    __aicore__ inline void MaxPoolWithArgmaxV3SmallCProcess();
    __aicore__ inline void MaxPoolWithArgmaxV3SmallCCompute();
    __aicore__ inline void ComputeSingleRow(
        __local_mem__ T1* xAddr, __local_mem__ T1* maxValueAddr, __local_mem__ T2* argmaxAddr);
    __aicore__ inline void ComputeMultiRow(
        __local_mem__ T1* xAddr, __local_mem__ T1* maxValueAddr, __local_mem__ T2* argmaxAddr);
    __aicore__ inline void ComputeMultiRowForInt64(
        __local_mem__ T1* xAddr, __local_mem__ T1* maxValueAddr, __local_mem__ T2* argmaxAddr,
        __local_mem__ uint32_t* helpAddr);
    __aicore__ inline void ComputeMultiBatch(
        __local_mem__ T1* xAddr, __local_mem__ T1* maxValueAddr, __local_mem__ T2* argmaxAddr);
    __aicore__ inline void ComputeMultiBatchForInt64(
        __local_mem__ T1* xAddr, __local_mem__ T1* maxValueAddr, __local_mem__ T2* argmaxAddr,
        __local_mem__ uint32_t* helpAddr);

public:
    constexpr static uint32_t V_REG_SIZE = platform::GetVRegSize();
};

template <typename T1, typename T2, const uint32_t IS_PAD>
__aicore__ inline void MaxPoolWithArgmaxV3SmallC<T1, T2, IS_PAD>::MaxPoolWithArgmaxV3SmallCProcess()
{
    if (this->blockIdx_ >= this->usedCoreNum_) {
        return;
    }

    int64_t curCoreProcessNum =
        (this->blockIdx_ + 1 == this->usedCoreNum_) ? this->tailCoreProcessNum_ : this->normalCoreProcessNum_;
    for (int64_t loopNum = 0; loopNum < curCoreProcessNum; loopNum++) {
        this->ScalarCompute(loopNum);
        this->CopyIn();
        MaxPoolWithArgmaxV3SmallCCompute();
        this->CopyOut();
    }
}

template <typename T1, typename T2, const uint32_t IS_PAD>
__aicore__ inline void MaxPoolWithArgmaxV3SmallC<T1, T2, IS_PAD>::MaxPoolWithArgmaxV3SmallCCompute()
{
    LocalTensor<T1> xLocal = this->inputQue_.template DeQue<T1>();
    LocalTensor<T1> maxValueLocal = this->maxValueQue_.template AllocTensor<T1>();
    LocalTensor<T2> argmaxLocal = this->argmaxQue_.template AllocTensor<T2>();
    LocalTensor<uint32_t> helpTensor = this->helperTBuf_.template Get<uint32_t>();

    __local_mem__ T1* xAddr = (__local_mem__ T1*)xLocal.GetPhyAddr();
    __local_mem__ T1* maxValueAddr = (__local_mem__ T1*)maxValueLocal.GetPhyAddr();
    __local_mem__ T2* argmaxAddr = (__local_mem__ T2*)argmaxLocal.GetPhyAddr();
    __local_mem__ uint32_t* helpAddr = (__local_mem__ uint32_t*)helpTensor.GetPhyAddr();

    if constexpr (IS_PAD == 1) {
        this->FillPadNegVF(xAddr);
    }

    uint16_t repeatElm = platform::GetVRegSize() / sizeof(T2);
    if (repeatElm >= DOUBLE * this->hOutputActual_ * this->wOutputActual_ * this->cInput_) {
        if constexpr (std::is_same<T2, int64_t>::value) { // 拼nhw，并发nhwc
            ComputeMultiBatchForInt64(xAddr, maxValueAddr, argmaxAddr, helpAddr);
        } else {
            ComputeMultiBatch(xAddr, maxValueAddr, argmaxAddr);
        }
    } else if (repeatElm >= DOUBLE * this->wOutputActual_ * this->cInput_) {
        if constexpr (std::is_same<T2, int64_t>::value) { // 拼hw，并发hwc
            ComputeMultiRowForInt64(xAddr, maxValueAddr, argmaxAddr, helpAddr);
        } else {
            ComputeMultiRow(xAddr, maxValueAddr, argmaxAddr);
        }
    } else { // 拼w，并发wc
        ComputeSingleRow(xAddr, maxValueAddr, argmaxAddr);
    }

    this->inputQue_.template FreeTensor(xLocal);
    this->maxValueQue_.template EnQue(maxValueLocal);
    this->argmaxQue_.template EnQue(argmaxLocal);
    this->helperTBuf_.template FreeTensor(helpTensor);
}

template <typename T1, typename T2, const uint32_t IS_PAD>
__aicore__ inline void MaxPoolWithArgmaxV3SmallC<T1, T2, IS_PAD>::ComputeMultiBatch(
    __local_mem__ T1* xAddr, __local_mem__ T1* maxValueAddr, __local_mem__ T2* argmaxAddr)
{
    uint16_t kH = static_cast<uint16_t>(this->hKernel_);
    uint16_t kW = static_cast<uint16_t>(this->wKernel_);
    uint16_t hStride = static_cast<uint16_t>(this->hStride_);
    uint16_t padH = static_cast<uint16_t>(this->padTop_);
    uint16_t padW = static_cast<uint16_t>(this->padLeft_);
    int32_t wInput = static_cast<int32_t>(this->wInput_);
    uint16_t alignedC = static_cast<uint16_t>(this->cOutputActualAlign_);

    constexpr uint16_t repeatElm = platform::GetVRegSize() / sizeof(T2);
    uint16_t nFactor = static_cast<uint16_t>(repeatElm / (this->hOutputActual_ * this->wOutputActual_ * this->cInput_));
    nFactor = nFactor > this->nOutputActual_ ? this->nOutputActual_ : nFactor;
    uint16_t loopN = static_cast<uint16_t>(this->nOutputActual_ / nFactor);
    uint16_t tailN = static_cast<uint16_t>(this->nOutputActual_ - loopN * nFactor);

    int32_t hInputActualAmend = (this->hOutputActual_ - 1) * this->hStride_ + this->hKernel_;
    int32_t wInputActualAmend = (this->wOutputActual_ - 1) * this->wStride_ + this->wKernel_;
    int32_t ubNumHWC = hInputActualAmend * wInputActualAmend * this->cOutputActualAlign_;

    int32_t wBlockArgmaxOffset = this->wAxisIndex_ * this->wStride_ * this->wOutputInner_;
    int32_t hBlockArgmaxOffset = this->hAxisIndex_ * this->hStride_ * this->hOutputInner_;

    uint32_t oneLoopElements = static_cast<uint32_t>(
        nFactor * this->hOutputActual_ * this->wOutputActual_ * this->cInput_); // 一次循环处理的输出元素
    uint32_t tailLoopElements =
        static_cast<uint32_t>(tailN * this->hOutputActual_ * this->wOutputActual_ * this->cInput_); // 尾循环处理输出
    uint32_t rowStrideInUb = static_cast<uint32_t>(wInputActualAmend * this->cOutputActualAlign_);
    uint32_t oneNOutScatterElements =
        static_cast<uint32_t>(this->hOutputActual_ * this->wOutputActual_ * this->cOutputActualAlign_);

    int32_t num1D = this->cInput_;
    int32_t rate2D = this->wStride_ * this->cOutputActualAlign_;
    int32_t num2D = this->wOutputActual_ * this->cInput_;
    int32_t rate3D = this->hStride_ * wInputActualAmend * this->cOutputActualAlign_;
    int32_t num3D = this->hOutputActual_ * this->wOutputActual_ * this->cInput_;
    int32_t rate4D = hInputActualAmend * wInputActualAmend * this->cOutputActualAlign_;

    T2 argNum1D = this->cInput_;
    T2 argRate2D = this->wStride_;
    T2 argNum2D = this->wOutputActual_ * this->cInput_;
    T2 argNum3D = this->hOutputActual_ * this->wOutputActual_ * this->cInput_;
    int32_t scatterIdxNum1D = this->cInput_;
    int32_t scatterIdxRate2D = this->cOutputActualAlign_;

    // 产生N的输出索引的索引
    __VEC_SCOPE__
    {
        MicroAPI::RegTensor<uint32_t> gatterStartIdx;
        MicroAPI::RegTensor<uint32_t> gatterNStartIdx;
        MicroAPI::RegTensor<T2> argmaxHStart;
        MicroAPI::RegTensor<T2> argmaxWStart;
        MicroAPI::RegTensor<uint32_t> scatterStartIdx;
        MicroAPI::RegTensor<uint32_t> scatterNStartIdx;
        MicroAPI::MaskReg maskAllU32 = MicroAPI::CreateMask<uint32_t, MicroAPI::MaskPattern::ALL>();
        MicroAPI::MaskReg maskAllT2 = MicroAPI::CreateMask<T2, MicroAPI::MaskPattern::ALL>();

        GenGatterIndex4D<int32_t>(
            (MicroAPI::RegTensor<int32_t>&)gatterStartIdx, rate4D, num3D, rate3D, num2D, rate2D, num1D);
        GenGatterIndex4D<T2>(argmaxWStart, 0, argNum3D, 0, argNum2D, argRate2D, argNum1D, 0);
        GenGatterIndex3D<T2>(argmaxHStart, 0, argNum3D, static_cast<T2>(hStride), argNum2D, 0);
        GenGatterIndex2D<int32_t>((MicroAPI::RegTensor<int32_t>&)scatterStartIdx, scatterIdxRate2D, scatterIdxNum1D);

        for (uint16_t nIdex = 0; nIdex < loopN; nIdex++) {
            // 校正N
            MicroAPI::Adds(gatterNStartIdx, gatterStartIdx, nIdex * nFactor * ubNumHWC, maskAllU32);
            MicroAPI::Adds(scatterNStartIdx, scatterStartIdx, nIdex * nFactor * oneNOutScatterElements, maskAllU32);

            int32_t gatterIndexOffset = 0;
            int32_t argmaxHOffset = hBlockArgmaxOffset;
            int32_t argmaxWOffset = wBlockArgmaxOffset;
            int32_t scatterOffset = 0;

            MaxPoolWithArgMaxV3GatherImpl<T1, T2, IS_PAD>(
                xAddr, maxValueAddr, argmaxAddr, kH, kW, rowStrideInUb, alignedC, gatterIndexOffset, gatterNStartIdx,
                oneLoopElements, argmaxHStart, argmaxWStart, argmaxHOffset, argmaxWOffset, scatterNStartIdx,
                scatterOffset, padH, padW, wInput);
        }

        // tail N
        MicroAPI::Adds(gatterNStartIdx, gatterStartIdx, loopN * nFactor * ubNumHWC, maskAllU32);
        MicroAPI::Adds(scatterNStartIdx, scatterStartIdx, loopN * nFactor * oneNOutScatterElements, maskAllU32);

        int32_t gatterIndexOffset = 0;
        int32_t argmaxHOffset = hBlockArgmaxOffset;
        int32_t argmaxWOffset = wBlockArgmaxOffset;
        int32_t scatterOffset = 0;

        MaxPoolWithArgMaxV3GatherImpl<T1, T2, IS_PAD>(
            xAddr, maxValueAddr, argmaxAddr, kH, kW, rowStrideInUb, alignedC, gatterIndexOffset, gatterNStartIdx,
            tailLoopElements, argmaxHStart, argmaxWStart, argmaxHOffset, argmaxWOffset, scatterNStartIdx, scatterOffset,
            padH, padW, wInput);
    }
}

template <typename T1, typename T2, const uint32_t IS_PAD>
__aicore__ inline void MaxPoolWithArgmaxV3SmallC<T1, T2, IS_PAD>::ComputeMultiBatchForInt64(
    __local_mem__ T1* xAddr, __local_mem__ T1* maxValueAddr, __local_mem__ T2* argmaxAddr,
    __local_mem__ uint32_t* helpAddr)
{
    uint16_t kH = static_cast<uint16_t>(this->hKernel_);
    uint16_t kW = static_cast<uint16_t>(this->wKernel_);
    uint16_t hStride = static_cast<uint16_t>(this->hStride_);
    uint16_t padH = static_cast<uint16_t>(this->padTop_);
    uint16_t padW = static_cast<uint16_t>(this->padLeft_);
    int32_t wInput = static_cast<int32_t>(this->wInput_);
    uint16_t alignedC = static_cast<uint16_t>(this->cOutputActualAlign_);

    constexpr uint16_t repeatElm = platform::GetVRegSize() / sizeof(T2);
    uint16_t nFactor = static_cast<uint16_t>(repeatElm / (this->hOutputActual_ * this->wOutputActual_ * this->cInput_));
    nFactor = nFactor > this->nOutputActual_ ? this->nOutputActual_ : nFactor;
    uint16_t loopN = static_cast<uint16_t>(this->nOutputActual_ / nFactor);
    uint16_t tailN = static_cast<uint16_t>(this->nOutputActual_ - loopN * nFactor);

    int32_t hInputActualAmend = (this->hOutputActual_ - 1) * this->hStride_ + this->hKernel_;
    int32_t wInputActualAmend = (this->wOutputActual_ - 1) * this->wStride_ + this->wKernel_;
    int32_t ubNumHWC = hInputActualAmend * wInputActualAmend * this->cOutputActualAlign_;

    int32_t wBlockArgmaxOffset = this->wAxisIndex_ * this->wStride_ * this->wOutputInner_;
    int32_t hBlockArgmaxOffset = this->hAxisIndex_ * this->hStride_ * this->hOutputInner_;

    uint32_t oneLoopElements = static_cast<uint32_t>(
        nFactor * this->hOutputActual_ * this->wOutputActual_ * this->cInput_); // 一次循环处理的输出元素
    uint32_t tailLoopElements =
        static_cast<uint32_t>(tailN * this->hOutputActual_ * this->wOutputActual_ * this->cInput_); // 尾循环处理输出
    uint32_t rowStrideInUb = static_cast<uint32_t>(wInputActualAmend * this->cOutputActualAlign_);
    uint32_t oneNOutScatterElements =
        static_cast<uint32_t>(this->hOutputActual_ * this->wOutputActual_ * this->cOutputActualAlign_);

    int32_t num1D = this->cInput_;
    int32_t rate2D = this->wStride_ * this->cOutputActualAlign_;
    int32_t num2D = this->wOutputActual_ * this->cInput_;
    int32_t rate3D = this->hStride_ * wInputActualAmend * this->cOutputActualAlign_;
    int32_t num3D = this->hOutputActual_ * this->wOutputActual_ * this->cInput_;
    int32_t rate4D = hInputActualAmend * wInputActualAmend * this->cOutputActualAlign_;

    T2 argNum1D = this->cInput_;
    T2 argRate2D = this->wStride_;
    T2 argNum2D = this->wOutputActual_ * this->cInput_;
    T2 argNum3D = this->hOutputActual_ * this->wOutputActual_ * this->cInput_;
    int32_t scatterIdxNum1D = this->cInput_;
    int32_t scatterIdxRate2D = this->cOutputActualAlign_;

    // 产生N的输出索引的索引
    __VEC_SCOPE__
    {
        MicroAPI::RegTensor<uint32_t> gatterStartIdx;
        MicroAPI::RegTensor<T2> argmaxHStart;
        MicroAPI::MaskReg maskAllU32 = MicroAPI::CreateMask<uint32_t, MicroAPI::MaskPattern::ALL>();

        GenGatterIndex4D<int32_t>(
            (MicroAPI::RegTensor<int32_t>&)gatterStartIdx, rate4D, num3D, rate3D, num2D, rate2D, num1D);
        GenGatterIndex3D<T2>(argmaxHStart, 0, argNum3D, hStride, argNum2D, 0);

        AscendC::MicroAPI::DataCopy(helpAddr, gatterStartIdx, maskAllU32);
        AscendC::MicroAPI::DataCopy(
            helpAddr + V_REG_SIZE / sizeof(uint32_t), (MicroAPI::RegTensor<uint32_t>&)argmaxHStart, maskAllU32);
    }

    __VEC_SCOPE__
    {
        MicroAPI::RegTensor<T2> argmaxWStart;
        MicroAPI::RegTensor<uint32_t> scatterStartIdx;
        MicroAPI::MaskReg maskAllU32 = MicroAPI::CreateMask<uint32_t, MicroAPI::MaskPattern::ALL>();

        GenGatterIndex4D<T2>(argmaxWStart, 0, argNum3D, 0, argNum2D, argRate2D, argNum1D, 0);
        GenGatterIndex2D<int32_t>((MicroAPI::RegTensor<int32_t>&)scatterStartIdx, scatterIdxRate2D, scatterIdxNum1D);

        AscendC::MicroAPI::DataCopy(
            helpAddr + V_REG_SIZE / sizeof(uint32_t) * DOUBLE, (MicroAPI::RegTensor<uint32_t>&)argmaxWStart,
            maskAllU32);
        AscendC::MicroAPI::DataCopy(helpAddr + V_REG_SIZE / sizeof(uint32_t) * THREE, scatterStartIdx, maskAllU32);
    }

    __VEC_SCOPE__
    {
        MicroAPI::RegTensor<uint32_t> gatterStartIdx;
        MicroAPI::RegTensor<uint32_t> gatterNStartIdx;
        MicroAPI::RegTensor<T2> argmaxHStart;
        MicroAPI::RegTensor<T2> argmaxWStart;
        MicroAPI::RegTensor<uint32_t> scatterStartIdx;
        MicroAPI::RegTensor<uint32_t> scatterNStartIdx;
        MicroAPI::MaskReg maskAllU32 = MicroAPI::CreateMask<uint32_t, MicroAPI::MaskPattern::ALL>();
        MicroAPI::MaskReg maskAllT2 = MicroAPI::CreateMask<T2, MicroAPI::MaskPattern::ALL>();

        AscendC::MicroAPI::DataCopy(gatterStartIdx, helpAddr);
        AscendC::MicroAPI::DataCopy(
            (MicroAPI::RegTensor<uint32_t>&)argmaxHStart, helpAddr + V_REG_SIZE / sizeof(uint32_t));
        AscendC::MicroAPI::DataCopy(
            (MicroAPI::RegTensor<uint32_t>&)argmaxWStart, helpAddr + V_REG_SIZE / sizeof(uint32_t) * DOUBLE);
        AscendC::MicroAPI::DataCopy(scatterStartIdx, helpAddr + V_REG_SIZE / sizeof(uint32_t) * THREE);

        for (uint16_t nIdex = 0; nIdex < loopN; nIdex++) {
            // 校正N
            MicroAPI::Adds(gatterNStartIdx, gatterStartIdx, nIdex * nFactor * ubNumHWC, maskAllU32);
            MicroAPI::Adds(scatterNStartIdx, scatterStartIdx, nIdex * nFactor * oneNOutScatterElements, maskAllU32);

            int32_t gatterIndexOffset = 0;
            int32_t argmaxHOffset = hBlockArgmaxOffset;
            int32_t argmaxWOffset = wBlockArgmaxOffset;
            int32_t scatterOffset = 0;

            MaxPoolWithArgMaxV3GatherImpl<T1, T2, IS_PAD>(
                xAddr, maxValueAddr, argmaxAddr, kH, kW, rowStrideInUb, alignedC, gatterIndexOffset, gatterNStartIdx,
                oneLoopElements, argmaxHStart, argmaxWStart, argmaxHOffset, argmaxWOffset, scatterNStartIdx,
                scatterOffset, padH, padW, wInput);
        }

        // tail N
        MicroAPI::Adds(gatterNStartIdx, gatterStartIdx, loopN * nFactor * ubNumHWC, maskAllU32);
        MicroAPI::Adds(scatterNStartIdx, scatterStartIdx, loopN * nFactor * oneNOutScatterElements, maskAllU32);

        int32_t gatterIndexOffset = 0;
        int32_t argmaxHOffset = hBlockArgmaxOffset;
        int32_t argmaxWOffset = wBlockArgmaxOffset;
        int32_t scatterOffset = 0;

        MaxPoolWithArgMaxV3GatherImpl<T1, T2, IS_PAD>(
            xAddr, maxValueAddr, argmaxAddr, kH, kW, rowStrideInUb, alignedC, gatterIndexOffset, gatterNStartIdx,
            tailLoopElements, argmaxHStart, argmaxWStart, argmaxHOffset, argmaxWOffset, scatterNStartIdx, scatterOffset,
            padH, padW, wInput);
    }
}

template <typename T1, typename T2, const uint32_t IS_PAD>
__aicore__ inline void MaxPoolWithArgmaxV3SmallC<T1, T2, IS_PAD>::ComputeMultiRow(
    __local_mem__ T1* xAddr, __local_mem__ T1* maxValueAddr, __local_mem__ T2* argmaxAddr)
{
    uint16_t kH = static_cast<uint16_t>(this->hKernel_);
    uint16_t kW = static_cast<uint16_t>(this->wKernel_);
    uint16_t hStride = static_cast<uint16_t>(this->hStride_);
    uint16_t padH = static_cast<uint16_t>(this->padTop_);
    uint16_t padW = static_cast<uint16_t>(this->padLeft_);
    int32_t wInput = static_cast<int32_t>(this->wInput_);
    uint16_t wOutputActual = static_cast<uint16_t>(this->wOutputActual_);
    uint16_t alignedC = static_cast<uint16_t>(this->cOutputActualAlign_);

    uint16_t loopN = static_cast<uint16_t>(this->nOutputActual_);
    constexpr uint32_t repeatElm = platform::GetVRegSize() / sizeof(T2);
    uint16_t hFactor = static_cast<uint16_t>(repeatElm / (this->wOutputActual_ * this->cInput_));
    hFactor = hFactor > this->hOutputActual_ ? this->hOutputActual_ : hFactor;
    uint16_t loopH = static_cast<uint16_t>(this->hOutputActual_ / hFactor);
    uint16_t tailH = static_cast<uint16_t>(this->hOutputActual_ - loopH * hFactor);

    int32_t hInputActualAmend = (this->hOutputActual_ - 1) * this->hStride_ + this->hKernel_;
    int32_t wInputActualAmend = (this->wOutputActual_ - 1) * this->wStride_ + this->wKernel_;
    int32_t ubNumHWC = hInputActualAmend * wInputActualAmend * this->cOutputActualAlign_;

    int32_t wBlockArgmaxOffset = this->wAxisIndex_ * this->wStride_ * this->wOutputInner_;
    int32_t hBlockArgmaxOffset = this->hAxisIndex_ * this->hStride_ * this->hOutputInner_;

    uint32_t oneLoopStrideH =
        static_cast<uint32_t>(hFactor * this->hStride_ * wInputActualAmend * this->cOutputActualAlign_);
    uint32_t oneLoopElements = static_cast<uint32_t>(hFactor * this->wOutputActual_ * this->cInput_);
    uint32_t tailLoopElements = static_cast<uint32_t>(tailH * this->wOutputActual_ * this->cInput_);
    uint32_t rowStrideInUb = static_cast<uint32_t>(wInputActualAmend * this->cOutputActualAlign_);
    uint32_t oneNOutScatterElements =
        static_cast<uint32_t>(this->hOutputActual_ * this->wOutputActual_ * this->cOutputActualAlign_);

    int32_t num1D = this->cInput_;
    int32_t rate2D = this->wStride_ * this->cOutputActualAlign_;
    int32_t num2D = this->wOutputActual_ * this->cInput_;
    int32_t rate3D = this->hStride_ * wInputActualAmend * this->cOutputActualAlign_;
    T2 argmaxNum1D = this->cInput_;
    T2 argMaxRate2D = this->wStride_;
    T2 argMaxNum2D = this->wOutputActual_ * this->cInput_;
    T2 argHRate3D = this->hStride_;
    int32_t scatterIdxNum1D = this->cInput_;
    int32_t scatterIdxRate2D = this->cOutputActualAlign_;

    __VEC_SCOPE__
    {
        MicroAPI::RegTensor<uint32_t> gatterStartIdx;
        MicroAPI::RegTensor<uint32_t> gatterNStartIdx;
        MicroAPI::RegTensor<T2> argmaxHStart;
        MicroAPI::RegTensor<T2> argmaxWStart;
        MicroAPI::RegTensor<uint32_t> scatterStartIdx;
        MicroAPI::RegTensor<uint32_t> scatterNStartIdx;
        MicroAPI::MaskReg maskAllU32 = MicroAPI::CreateMask<uint32_t, MicroAPI::MaskPattern::ALL>();
        MicroAPI::MaskReg maskAllT2 = MicroAPI::CreateMask<T2, MicroAPI::MaskPattern::ALL>();

        GenGatterIndex3D<int32_t>((MicroAPI::RegTensor<int32_t>&)gatterStartIdx, rate3D, num2D, rate2D, num1D);
        GenGatterIndex3D<T2>(argmaxWStart, 0, argMaxNum2D, argMaxRate2D, argmaxNum1D, 0);
        GenGatterIndex2D<T2>(argmaxHStart, argHRate3D, argMaxNum2D, 0);
        GenGatterIndex2D<int32_t>((MicroAPI::RegTensor<int32_t>&)scatterStartIdx, scatterIdxRate2D, scatterIdxNum1D);

        for (uint16_t nIdex = 0; nIdex < loopN; nIdex++) {
            // 校正N
            MicroAPI::Adds(gatterNStartIdx, gatterStartIdx, nIdex * ubNumHWC, maskAllU32);
            MicroAPI::Adds(scatterNStartIdx, scatterStartIdx, nIdex * oneNOutScatterElements, maskAllU32);

            for (uint16_t j = 0; j < loopH; j++) {
                int32_t gatterIndexOffset = j * oneLoopStrideH;
                int32_t argmaxHOffset = j * hStride * hFactor + hBlockArgmaxOffset;
                int32_t argmaxWOffset = wBlockArgmaxOffset;
                int32_t scatterOffset = j * hFactor * wOutputActual * alignedC;

                MaxPoolWithArgMaxV3GatherImpl<T1, T2, IS_PAD>(
                    xAddr, maxValueAddr, argmaxAddr, kH, kW, rowStrideInUb, alignedC, gatterIndexOffset,
                    gatterNStartIdx, oneLoopElements, argmaxHStart, argmaxWStart, argmaxHOffset, argmaxWOffset,
                    scatterNStartIdx, scatterOffset, padH, padW, wInput);
            }

            // tail H
            int32_t gatterIndexOffset = loopH * oneLoopStrideH;
            int32_t argmaxHOffset = loopH * hStride * hFactor + hBlockArgmaxOffset;
            int32_t argmaxWOffset = wBlockArgmaxOffset;
            int32_t scatterOffset = loopH * hFactor * wOutputActual * alignedC;

            MaxPoolWithArgMaxV3GatherImpl<T1, T2, IS_PAD>(
                xAddr, maxValueAddr, argmaxAddr, kH, kW, rowStrideInUb, alignedC, gatterIndexOffset, gatterNStartIdx,
                tailLoopElements, argmaxHStart, argmaxWStart, argmaxHOffset, argmaxWOffset, scatterNStartIdx,
                scatterOffset, padH, padW, wInput);
        }
    }
}

template <typename T1, typename T2, const uint32_t IS_PAD>
__aicore__ inline void MaxPoolWithArgmaxV3SmallC<T1, T2, IS_PAD>::ComputeMultiRowForInt64(
    __local_mem__ T1* xAddr, __local_mem__ T1* maxValueAddr, __local_mem__ T2* argmaxAddr,
    __local_mem__ uint32_t* helpAddr)
{
    uint16_t kH = static_cast<uint16_t>(this->hKernel_);
    uint16_t kW = static_cast<uint16_t>(this->wKernel_);
    uint16_t hStride = static_cast<uint16_t>(this->hStride_);
    uint16_t padH = static_cast<uint16_t>(this->padTop_);
    uint16_t padW = static_cast<uint16_t>(this->padLeft_);
    int32_t wInput = static_cast<int32_t>(this->wInput_);
    uint16_t wOutputActual = static_cast<uint16_t>(this->wOutputActual_);
    uint16_t alignedC = static_cast<uint16_t>(this->cOutputActualAlign_);

    uint16_t loopN = static_cast<uint16_t>(this->nOutputActual_);
    constexpr uint32_t repeatElm = platform::GetVRegSize() / sizeof(T2);
    uint16_t hFactor = static_cast<uint16_t>(repeatElm / (this->wOutputActual_ * this->cInput_));
    hFactor = hFactor > this->hOutputActual_ ? this->hOutputActual_ : hFactor;
    uint16_t loopH = static_cast<uint16_t>(this->hOutputActual_ / hFactor);
    uint16_t tailH = static_cast<uint16_t>(this->hOutputActual_ - loopH * hFactor);

    int32_t hInputActualAmend = (this->hOutputActual_ - 1) * this->hStride_ + this->hKernel_;
    int32_t wInputActualAmend = (this->wOutputActual_ - 1) * this->wStride_ + this->wKernel_;
    int32_t ubNumHWC = hInputActualAmend * wInputActualAmend * this->cOutputActualAlign_;

    int32_t wBlockArgmaxOffset = this->wAxisIndex_ * this->wStride_ * this->wOutputInner_;
    int32_t hBlockArgmaxOffset = this->hAxisIndex_ * this->hStride_ * this->hOutputInner_;

    uint32_t oneLoopStrideH =
        static_cast<uint32_t>(hFactor * this->hStride_ * wInputActualAmend * this->cOutputActualAlign_);
    uint32_t oneLoopElements = static_cast<uint32_t>(hFactor * this->wOutputActual_ * this->cInput_);
    uint32_t tailLoopElements = static_cast<uint32_t>(tailH * this->wOutputActual_ * this->cInput_);
    uint32_t rowStrideInUb = static_cast<uint32_t>(wInputActualAmend * this->cOutputActualAlign_);
    uint32_t oneNOutScatterElements =
        static_cast<uint32_t>(this->hOutputActual_ * this->wOutputActual_ * this->cOutputActualAlign_);

    int32_t num1D = this->cInput_;
    int32_t rate2D = this->wStride_ * this->cOutputActualAlign_;
    int32_t num2D = this->wOutputActual_ * this->cInput_;
    int32_t rate3D = this->hStride_ * wInputActualAmend * this->cOutputActualAlign_;
    T2 argmaxNum1D = this->cInput_;
    T2 argMaxRate2D = this->wStride_;
    T2 argMaxNum2D = this->wOutputActual_ * this->cInput_;
    T2 argHRate3D = this->hStride_;
    int32_t scatterIdxNum1D = this->cInput_;
    int32_t scatterIdxRate2D = this->cOutputActualAlign_;

    __VEC_SCOPE__
    {
        MicroAPI::RegTensor<uint32_t> gatterStartIdx;
        MicroAPI::RegTensor<uint32_t> scatterStartIdx;
        MicroAPI::MaskReg maskAllU32 = MicroAPI::CreateMask<uint32_t, MicroAPI::MaskPattern::ALL>();

        GenGatterIndex3D<int32_t>((MicroAPI::RegTensor<int32_t>&)gatterStartIdx, rate3D, num2D, rate2D, num1D);
        GenGatterIndex2D<int32_t>((MicroAPI::RegTensor<int32_t>&)scatterStartIdx, scatterIdxRate2D, scatterIdxNum1D);

        AscendC::MicroAPI::DataCopy(helpAddr, gatterStartIdx, maskAllU32);
        AscendC::MicroAPI::DataCopy(helpAddr + V_REG_SIZE / sizeof(uint32_t), scatterStartIdx, maskAllU32);
    }

    __VEC_SCOPE__
    {
        MicroAPI::RegTensor<T2> argmaxHStart;
        MicroAPI::RegTensor<T2> argmaxWStart;
        MicroAPI::MaskReg maskAllU32 = MicroAPI::CreateMask<uint32_t, MicroAPI::MaskPattern::ALL>();

        GenGatterIndex3D<T2>(argmaxWStart, 0, argMaxNum2D, argMaxRate2D, argmaxNum1D, 0);
        GenGatterIndex2D<T2>(argmaxHStart, argHRate3D, argMaxNum2D, 0);

        AscendC::MicroAPI::DataCopy(
            helpAddr + V_REG_SIZE / sizeof(uint32_t) * DOUBLE, (MicroAPI::RegTensor<uint32_t>&)argmaxHStart,
            maskAllU32);
        AscendC::MicroAPI::DataCopy(
            helpAddr + V_REG_SIZE / sizeof(uint32_t) * THREE, (MicroAPI::RegTensor<uint32_t>&)argmaxWStart, maskAllU32);
    }

    __VEC_SCOPE__
    {
        MicroAPI::RegTensor<uint32_t> gatterStartIdx;
        MicroAPI::RegTensor<uint32_t> gatterNStartIdx;
        MicroAPI::RegTensor<T2> argmaxHStart;
        MicroAPI::RegTensor<T2> argmaxWStart;
        MicroAPI::RegTensor<uint32_t> scatterStartIdx;
        MicroAPI::RegTensor<uint32_t> scatterNStartIdx;
        MicroAPI::MaskReg maskAllU32 = MicroAPI::CreateMask<uint32_t, MicroAPI::MaskPattern::ALL>();
        MicroAPI::MaskReg maskAllT2 = MicroAPI::CreateMask<T2, MicroAPI::MaskPattern::ALL>();

        AscendC::MicroAPI::DataCopy(gatterStartIdx, helpAddr);
        AscendC::MicroAPI::DataCopy(scatterStartIdx, helpAddr + V_REG_SIZE / sizeof(uint32_t));
        AscendC::MicroAPI::DataCopy(
            (MicroAPI::RegTensor<uint32_t>&)argmaxHStart, helpAddr + V_REG_SIZE / sizeof(uint32_t) * DOUBLE);
        AscendC::MicroAPI::DataCopy(
            (MicroAPI::RegTensor<uint32_t>&)argmaxWStart, helpAddr + V_REG_SIZE / sizeof(uint32_t) * THREE);

        for (uint16_t nIdex = 0; nIdex < loopN; nIdex++) {
            // 校正N
            MicroAPI::Adds(gatterNStartIdx, gatterStartIdx, nIdex * ubNumHWC, maskAllU32);
            MicroAPI::Adds(scatterNStartIdx, scatterStartIdx, nIdex * oneNOutScatterElements, maskAllU32);

            for (uint16_t j = 0; j < loopH; j++) {
                int32_t gatterIndexOffset = j * oneLoopStrideH;
                int32_t argmaxHOffset = j * hStride * hFactor + hBlockArgmaxOffset;
                int32_t argmaxWOffset = wBlockArgmaxOffset;
                int32_t scatterOffset = j * hFactor * wOutputActual * alignedC;

                MaxPoolWithArgMaxV3GatherImpl<T1, T2, IS_PAD>(
                    xAddr, maxValueAddr, argmaxAddr, kH, kW, rowStrideInUb, alignedC, gatterIndexOffset,
                    gatterNStartIdx, oneLoopElements, argmaxHStart, argmaxWStart, argmaxHOffset, argmaxWOffset,
                    scatterNStartIdx, scatterOffset, padH, padW, wInput);
            }

            // tail H
            int32_t gatterIndexOffset = loopH * oneLoopStrideH;
            int32_t argmaxHOffset = loopH * hStride * hFactor + hBlockArgmaxOffset;
            int32_t argmaxWOffset = wBlockArgmaxOffset;
            int32_t scatterOffset = loopH * hFactor * wOutputActual * alignedC;

            MaxPoolWithArgMaxV3GatherImpl<T1, T2, IS_PAD>(
                xAddr, maxValueAddr, argmaxAddr, kH, kW, rowStrideInUb, alignedC, gatterIndexOffset, gatterNStartIdx,
                tailLoopElements, argmaxHStart, argmaxWStart, argmaxHOffset, argmaxWOffset, scatterNStartIdx,
                scatterOffset, padH, padW, wInput);
        }
    }
}

template <typename T1, typename T2, const uint32_t IS_PAD>
__aicore__ inline void MaxPoolWithArgmaxV3SmallC<T1, T2, IS_PAD>::ComputeSingleRow(
    __local_mem__ T1* xAddr, __local_mem__ T1* maxValueAddr, __local_mem__ T2* argmaxAddr)
{
    uint16_t kH = static_cast<uint16_t>(this->hKernel_);
    uint16_t kW = static_cast<uint16_t>(this->wKernel_);
    uint16_t hStride = static_cast<uint16_t>(this->hStride_);
    uint16_t wStride = static_cast<uint16_t>(this->hStride_);
    uint16_t padH = static_cast<uint16_t>(this->padTop_);
    uint16_t padW = static_cast<uint16_t>(this->padLeft_);
    int32_t wInput = static_cast<int32_t>(this->wInput_);
    uint16_t wOutputActual = static_cast<uint16_t>(this->wOutputActual_);
    uint16_t alignedC = static_cast<uint16_t>(this->cOutputActualAlign_);

    uint16_t loopN = this->nOutputActual_;
    uint16_t loopH = this->hOutputActual_;

    constexpr uint32_t repeatElm = platform::GetVRegSize() / sizeof(T2);
    uint16_t wFactor = repeatElm / this->cInput_;
    wFactor = wFactor > this->wOutputActual_ ? this->wOutputActual_ : wFactor;
    uint16_t loopW = static_cast<uint16_t>(this->wOutputActual_ / wFactor);
    uint16_t tailW = static_cast<uint16_t>(this->wOutputActual_ - loopW * wFactor);

    int32_t hInputActualAmend = (this->hOutputActual_ - 1) * this->hStride_ + this->hKernel_;
    int32_t wInputActualAmend = (this->wOutputActual_ - 1) * this->wStride_ + this->wKernel_;
    int32_t ubNumHWC = hInputActualAmend * wInputActualAmend * this->cOutputActualAlign_;

    int32_t wBlockArgmaxOffset = this->wAxisIndex_ * this->wStride_ * this->wOutputInner_;
    int32_t hBlockArgmaxOffset = this->hAxisIndex_ * this->hStride_ * this->hOutputInner_;

    uint32_t oneLoopStrideH = static_cast<uint32_t>(this->hStride_ * wInputActualAmend * this->cOutputActualAlign_);
    uint32_t oneLoopStrideW = static_cast<uint32_t>(this->wStride_ * wFactor * this->cOutputActualAlign_);
    uint32_t oneLoopElements = static_cast<uint32_t>(wFactor * this->cInput_);
    uint32_t tailLoopElements = tailW * this->cInput_;

    uint32_t oneNOutScatterElements =
        static_cast<uint32_t>(this->hOutputActual_ * this->wOutputActual_ * this->cOutputActualAlign_);
    uint32_t rowStrideInUb = static_cast<uint32_t>(wInputActualAmend * this->cOutputActualAlign_);

    int32_t num1D = this->cInput_;
    int32_t rate2D = this->wStride_ * this->cOutputActualAlign_;
    int32_t argmaxNum1D = this->cInput_;
    T2 argmaxRate2D = this->wStride_;
    int32_t scatterIdxNum1D = this->cInput_;
    int32_t scatterIdxRate2D = this->cOutputActualAlign_;

    for (uint16_t nIdex = 0; nIdex < loopN; nIdex++) {
        __VEC_SCOPE__
        {
            MicroAPI::RegTensor<uint32_t> gatterStartIdx;
            MicroAPI::RegTensor<T2> argmaxHStart;
            MicroAPI::RegTensor<T2> argmaxWStart;
            MicroAPI::RegTensor<uint32_t> scatterStartIdx;
            MicroAPI::MaskReg maskAllU32 = MicroAPI::CreateMask<uint32_t, MicroAPI::MaskPattern::ALL>();
            MicroAPI::MaskReg maskAllT2 = MicroAPI::CreateMask<T2, MicroAPI::MaskPattern::ALL>();

            GenGatterIndex2D<int32_t>((MicroAPI::RegTensor<int32_t>&)gatterStartIdx, rate2D, num1D);
            GenGatterIndex2D<T2>(argmaxWStart, argmaxRate2D, static_cast<T2>(argmaxNum1D), 0);
            AscendC::MicroAPI::Duplicate(argmaxHStart, 0);
            GenGatterIndex2D<int32_t>(
                (MicroAPI::RegTensor<int32_t>&)scatterStartIdx, scatterIdxRate2D, scatterIdxNum1D);

            MicroAPI::Adds(gatterStartIdx, gatterStartIdx, nIdex * ubNumHWC, maskAllU32);
            MicroAPI::Adds(scatterStartIdx, scatterStartIdx, nIdex * oneNOutScatterElements, maskAllU32);

            for (uint16_t i = 0; i < loopH; i++) {
                int32_t hOffset = i * oneLoopStrideH;
                int32_t argmaxHOffset = i * hStride + hBlockArgmaxOffset;

                for (uint16_t j = 0; j < loopW; j++) {
                    int32_t wOffset = j * oneLoopStrideW;
                    int32_t argmaxWOffset = j * wStride * wFactor + wBlockArgmaxOffset;
                    int32_t gatterIndexOffset = hOffset + wOffset;
                    int32_t scatterOffset = (j * wFactor + i * wOutputActual) * alignedC;

                    MaxPoolWithArgMaxV3GatherImpl<T1, T2, IS_PAD>(
                        xAddr, maxValueAddr, argmaxAddr, kH, kW, rowStrideInUb, alignedC, gatterIndexOffset,
                        gatterStartIdx, oneLoopElements, argmaxHStart, argmaxWStart, argmaxHOffset, argmaxWOffset,
                        scatterStartIdx, scatterOffset, padH, padW, wInput);
                }

                // tail w
                int32_t wOffset = loopW * oneLoopStrideW;
                int32_t argmaxWOffset = loopW * wStride * wFactor + wBlockArgmaxOffset;
                int32_t gatterIndexOffset = hOffset + wOffset;
                int32_t scatterOffset = (loopW * wFactor + i * wOutputActual) * alignedC;

                MaxPoolWithArgMaxV3GatherImpl<T1, T2, IS_PAD>(
                    xAddr, maxValueAddr, argmaxAddr, kH, kW, rowStrideInUb, alignedC, gatterIndexOffset, gatterStartIdx,
                    tailLoopElements, argmaxHStart, argmaxWStart, argmaxHOffset, argmaxWOffset, scatterStartIdx,
                    scatterOffset, padH, padW, wInput);
            }
        }
    }
}
} // namespace MaxPoolWithArgmaxV3SmallCNameSpace
#endif // MAX_POOL_WITH_ARGMAX_V3_NHWC_SMALL_C__H_
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
 * \file glu_common.h
 * \brief GLU operator common definitions for arch35
 */

#ifndef GLU_COMMON_ARCH35_H
#define GLU_COMMON_ARCH35_H

#include "kernel_operator.h"

#ifdef __CCE_AICORE__
#include "op_kernel/platform_util.h"
#endif

namespace Glu {
namespace Common {

using namespace AscendC;

constexpr static int32_t BUFFER_NUM = 2;
constexpr static int64_t BUFFER_SIZE = 10 * 1024;
constexpr static int32_t BLOCK_BYTES = 32;

constexpr static AscendC::MicroAPI::CastTrait castTrait00 = {
    AscendC::MicroAPI::RegLayout::ZERO, AscendC::MicroAPI::SatMode::UNKNOWN,
    AscendC::MicroAPI::MaskMergeMode::ZEROING, AscendC::RoundMode::UNKNOWN};

constexpr static AscendC::MicroAPI::CastTrait castTrait01 = {
    AscendC::MicroAPI::RegLayout::ONE, AscendC::MicroAPI::SatMode::UNKNOWN,
    AscendC::MicroAPI::MaskMergeMode::ZEROING, AscendC::RoundMode::UNKNOWN};

constexpr static AscendC::MicroAPI::CastTrait castTrait11 = {
    AscendC::MicroAPI::RegLayout::ZERO, AscendC::MicroAPI::SatMode::NO_SAT,
    AscendC::MicroAPI::MaskMergeMode::ZEROING, AscendC::RoundMode::CAST_RINT};

constexpr static AscendC::MicroAPI::CastTrait castTrait12 = {
    AscendC::MicroAPI::RegLayout::ONE, AscendC::MicroAPI::SatMode::NO_SAT,
    AscendC::MicroAPI::MaskMergeMode::ZEROING, AscendC::RoundMode::CAST_RINT};

template <typename T>
__aicore__ inline void SetGlobalBufferForGlu(GlobalTensor<T>& xGm, GlobalTensor<T>& yGm, GM_ADDR x, GM_ADDR y)
{
    xGm.SetGlobalBuffer((__gm__ T*)x);
    yGm.SetGlobalBuffer((__gm__ T*)y);
}

#ifdef __CCE_AICORE__

template <typename T>
__aicore__ inline void ComputeSigmoidAndMulCore(
    AscendC::MicroAPI::RegTensor<T>& vregA,
    AscendC::MicroAPI::RegTensor<T>& vregB,
    AscendC::MicroAPI::RegTensor<T>& vregOutput,
    __local_mem__ T* outLocalPtr,
    uint16_t loopIdx,
    uint32_t vlSize,
    AscendC::MicroAPI::MaskReg& preg0,
    AscendC::MicroAPI::MaskReg& maskAll8,
    AscendC::MicroAPI::RegTensor<float>& vreg0)
{
    AscendC::MicroAPI::RegTensor<float> vreg1;
    AscendC::MicroAPI::RegTensor<float> vreg2;
    AscendC::MicroAPI::RegTensor<float> vreg3;
    AscendC::MicroAPI::RegTensor<float> vreg4;
    AscendC::MicroAPI::RegTensor<float> vreg5;
    AscendC::MicroAPI::RegTensor<float> vreg6;
    AscendC::MicroAPI::RegTensor<float> vreg7;
    AscendC::MicroAPI::RegTensor<float> vreg8;
    AscendC::MicroAPI::RegTensor<float> vreg9;
    AscendC::MicroAPI::RegTensor<float> vreg10;
    AscendC::MicroAPI::RegTensor<float> vreg11;
    AscendC::MicroAPI::RegTensor<float> vreg12;
    AscendC::MicroAPI::RegTensor<float> vreg13;
    AscendC::MicroAPI::RegTensor<float> vreg14;
    AscendC::MicroAPI::RegTensor<T> vreg15;
    AscendC::MicroAPI::RegTensor<T> vreg16;

    if constexpr (std::is_same_v<T, bfloat16_t> || std::is_same_v<T, half>) {
        AscendC::MicroAPI::Cast<float, T, castTrait00>(vreg5, vregA, maskAll8);
        AscendC::MicroAPI::Cast<float, T, castTrait01>(vreg6, vregA, maskAll8);
        AscendC::MicroAPI::Cast<float, T, castTrait00>(vreg8, vregB, maskAll8);
        AscendC::MicroAPI::Cast<float, T, castTrait01>(vreg9, vregB, maskAll8);

        AscendC::MicroAPI::Muls<float, float, AscendC::MicroAPI::MaskMergeMode::ZEROING>(
            vreg1, vreg8, static_cast<float>(-1), maskAll8);
        AscendC::MicroAPI::Exp<float, AscendC::MicroAPI::MaskMergeMode::ZEROING>(vreg2, vreg1, maskAll8);
        AscendC::MicroAPI::Adds<float, float, AscendC::MicroAPI::MaskMergeMode::ZEROING>(
            vreg3, vreg2, static_cast<float>(1), maskAll8);
        AscendC::MicroAPI::Div<float, AscendC::MicroAPI::MaskMergeMode::ZEROING>(
            vreg4, vreg0, vreg3, maskAll8);

        AscendC::MicroAPI::Mul<float, AscendC::MicroAPI::MaskMergeMode::ZEROING>(
            vreg7, vreg4, vreg5, maskAll8);

        AscendC::MicroAPI::Muls<float, float, AscendC::MicroAPI::MaskMergeMode::ZEROING>(
            vreg11, vreg9, static_cast<float>(-1), maskAll8);
        AscendC::MicroAPI::Exp<float, AscendC::MicroAPI::MaskMergeMode::ZEROING>(vreg12, vreg11, maskAll8);
        AscendC::MicroAPI::Adds<float, float, AscendC::MicroAPI::MaskMergeMode::ZEROING>(
            vreg13, vreg12, static_cast<float>(1), maskAll8);
        AscendC::MicroAPI::Div<float, AscendC::MicroAPI::MaskMergeMode::ZEROING>(
            vreg14, vreg0, vreg13, maskAll8);

        AscendC::MicroAPI::Mul<float, AscendC::MicroAPI::MaskMergeMode::ZEROING>(
            vreg10, vreg14, vreg6, maskAll8);

        AscendC::MicroAPI::Cast<T, float, castTrait11>(vreg15, vreg7, maskAll8);
        AscendC::MicroAPI::Cast<T, float, castTrait12>(vreg16, vreg10, maskAll8);
        AscendC::Reg::Or(
            (MicroAPI::RegTensor<uint16_t>&)vregOutput,
            (MicroAPI::RegTensor<uint16_t>&)vreg15,
            (MicroAPI::RegTensor<uint16_t>&)vreg16,
            maskAll8);

        AscendC::MicroAPI::DataCopy(outLocalPtr + loopIdx * vlSize, vregOutput, preg0);
    } else {
        AscendC::MicroAPI::Muls<T, T, AscendC::MicroAPI::MaskMergeMode::ZEROING>(
            vreg1, vregB, static_cast<T>(-1), preg0);
        AscendC::MicroAPI::Exp<T, AscendC::MicroAPI::MaskMergeMode::ZEROING>(vreg2, vreg1, preg0);
        AscendC::MicroAPI::Adds<T, T, AscendC::MicroAPI::MaskMergeMode::ZEROING>(
            vreg3, vreg2, static_cast<T>(1), preg0);
        AscendC::MicroAPI::Div<T, AscendC::MicroAPI::MaskMergeMode::ZEROING>(
            vreg4, vreg0, vreg3, preg0);

        AscendC::MicroAPI::Mul<T, AscendC::MicroAPI::MaskMergeMode::ZEROING>(
            vregOutput, vreg4, vregA, preg0);

        AscendC::MicroAPI::DataCopy(outLocalPtr + loopIdx * vlSize, vregOutput, preg0);
    }
}

template <typename T>
__aicore__ inline void ComputeSigmoidAndMulImpl(
    __local_mem__ T* x1LocalPtr,
    __local_mem__ T* x2LocalPtr,
    __local_mem__ T* outLocalPtr,
    const int64_t& count)
{
    using namespace Ops::Base;
    constexpr uint32_t VECTOR_LENGTH = GetVRegSize();
    uint32_t dtypeSize = sizeof(T);
    uint32_t vl = VECTOR_LENGTH / dtypeSize;
    uint16_t loopNum = CeilDivision(count, vl);
    uint32_t vlSize = vl;

    __VEC_SCOPE__
    {
        AscendC::MicroAPI::RegTensor<float> vreg0;
        AscendC::MicroAPI::RegTensor<T> vregInput1;
        AscendC::MicroAPI::RegTensor<T> vregInput2;
        AscendC::MicroAPI::RegTensor<T> vregOutput;

        AscendC::MicroAPI::MaskReg preg0;
        uint32_t size = count;
        preg0 = AscendC::MicroAPI::CreateMask<T>();
        AscendC::MicroAPI::Duplicate<float, AscendC::MicroAPI::MaskMergeMode::ZEROING, float>(
            vreg0, static_cast<float>(1), preg0);
        AscendC::MicroAPI::MaskReg maskAll8 = MicroAPI::CreateMask<uint8_t, MicroAPI::MaskPattern::ALL>();

        for (uint16_t loopIdx = 0; loopIdx < loopNum; loopIdx++) {
            preg0 = AscendC::MicroAPI::UpdateMask<T>(size);
            AscendC::MicroAPI::DataCopy(vregInput1, (__ubuf__ T*)(x1LocalPtr + loopIdx * vlSize));
            AscendC::MicroAPI::DataCopy(vregInput2, (__ubuf__ T*)(x2LocalPtr + loopIdx * vlSize));

            ComputeSigmoidAndMulCore<T>(vregInput1, vregInput2, vregOutput, outLocalPtr,
                loopIdx, vlSize, preg0, maskAll8, vreg0);
        }
    }
}

template <typename T>
__aicore__ inline void ComputeSigmoidAndMulWithDeInterleave(
    __local_mem__ T* xLocalPtr,
    __local_mem__ T* outLocalPtr,
    const int64_t& count)
{
    using namespace Ops::Base;
    constexpr uint32_t VECTOR_LENGTH = GetVRegSize();
    uint32_t dtypeSize = sizeof(T);
    uint32_t vl = VECTOR_LENGTH / dtypeSize;
    uint16_t loopNum = CeilDivision(count, vl);
    uint32_t vlSize = vl;

    __VEC_SCOPE__
    {
        AscendC::MicroAPI::RegTensor<float> vreg0;
        AscendC::MicroAPI::RegTensor<T> vregInput1;
        AscendC::MicroAPI::RegTensor<T> vregInput2;
        AscendC::MicroAPI::RegTensor<T> vregOutput;
        AscendC::MicroAPI::RegTensor<T> vreg1;
        AscendC::MicroAPI::RegTensor<T> vreg2;

        AscendC::MicroAPI::MaskReg preg0;
        uint32_t size = count;
        preg0 = AscendC::MicroAPI::CreateMask<T>();
        AscendC::MicroAPI::Duplicate<float, AscendC::MicroAPI::MaskMergeMode::ZEROING, float>(
            vreg0, static_cast<float>(1), preg0);

        MicroAPI::MaskReg maskAll8 = MicroAPI::CreateMask<uint8_t, MicroAPI::MaskPattern::ALL>();
        
        for (uint16_t loopIdx = 0; loopIdx < loopNum; loopIdx++) {
            preg0 = AscendC::MicroAPI::UpdateMask<T>(size);
            AscendC::MicroAPI::DataCopy(vregInput1, (__ubuf__ T*)(xLocalPtr + loopIdx * 2 * vlSize));
            AscendC::MicroAPI::DataCopy(vregInput2, (__ubuf__ T*)(xLocalPtr + loopIdx * 2 * vlSize + vlSize));

            MicroAPI::DeInterleave<T>(vreg1, vreg2, vregInput1, vregInput2);

            ComputeSigmoidAndMulCore<T>(vreg1, vreg2, vregOutput, outLocalPtr,
                loopIdx, vlSize, preg0, maskAll8, vreg0);
        }
    }
}

#endif

} // namespace Common
} // namespace Glu

#endif // GLU_COMMON_ARCH35_H
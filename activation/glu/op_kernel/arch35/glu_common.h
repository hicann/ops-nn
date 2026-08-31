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

constexpr static AscendC::Reg::CastTrait castTrait00 = {AscendC::Reg::RegLayout::ZERO, AscendC::Reg::SatMode::UNKNOWN,
                                                        AscendC::Reg::MaskMergeMode::ZEROING,
                                                        AscendC::RoundMode::UNKNOWN};

constexpr static AscendC::Reg::CastTrait castTrait01 = {AscendC::Reg::RegLayout::ONE, AscendC::Reg::SatMode::UNKNOWN,
                                                        AscendC::Reg::MaskMergeMode::ZEROING,
                                                        AscendC::RoundMode::UNKNOWN};

constexpr static AscendC::Reg::CastTrait castTrait11 = {AscendC::Reg::RegLayout::ZERO, AscendC::Reg::SatMode::NO_SAT,
                                                        AscendC::Reg::MaskMergeMode::ZEROING,
                                                        AscendC::RoundMode::CAST_RINT};

constexpr static AscendC::Reg::CastTrait castTrait12 = {AscendC::Reg::RegLayout::ONE, AscendC::Reg::SatMode::NO_SAT,
                                                        AscendC::Reg::MaskMergeMode::ZEROING,
                                                        AscendC::RoundMode::CAST_RINT};

template <typename T>
__aicore__ inline void SetGlobalBufferForGlu(GlobalTensor<T>& xGm, GlobalTensor<T>& yGm, GM_ADDR x, GM_ADDR y)
{
    xGm.SetGlobalBuffer((__gm__ T*)x);
    yGm.SetGlobalBuffer((__gm__ T*)y);
}

#ifdef __CCE_AICORE__

template <typename T>
__aicore__ inline void ComputeSigmoidAndMulCore(AscendC::Reg::RegTensor<T>& vregA, AscendC::Reg::RegTensor<T>& vregB,
                                                AscendC::Reg::RegTensor<T>& vregOutput, __local_mem__ T* outLocalPtr,
                                                uint16_t loopIdx, uint32_t vlSize, AscendC::Reg::MaskReg& preg0,
                                                AscendC::Reg::MaskReg& maskAll8, AscendC::Reg::RegTensor<float>& vreg0)
{
    AscendC::Reg::RegTensor<float> vreg1;
    AscendC::Reg::RegTensor<float> vreg2;
    AscendC::Reg::RegTensor<float> vreg3;
    AscendC::Reg::RegTensor<float> vreg4;
    AscendC::Reg::RegTensor<float> vreg5;
    AscendC::Reg::RegTensor<float> vreg6;
    AscendC::Reg::RegTensor<float> vreg7;
    AscendC::Reg::RegTensor<float> vreg8;
    AscendC::Reg::RegTensor<float> vreg9;
    AscendC::Reg::RegTensor<float> vreg10;
    AscendC::Reg::RegTensor<float> vreg11;
    AscendC::Reg::RegTensor<float> vreg12;
    AscendC::Reg::RegTensor<float> vreg13;
    AscendC::Reg::RegTensor<float> vreg14;
    AscendC::Reg::RegTensor<T> vreg15;
    AscendC::Reg::RegTensor<T> vreg16;

    if constexpr (std::is_same_v<T, bfloat16_t> || std::is_same_v<T, half>) {
        AscendC::Reg::Cast<float, T, castTrait00>(vreg5, vregA, maskAll8);
        AscendC::Reg::Cast<float, T, castTrait01>(vreg6, vregA, maskAll8);
        AscendC::Reg::Cast<float, T, castTrait00>(vreg8, vregB, maskAll8);
        AscendC::Reg::Cast<float, T, castTrait01>(vreg9, vregB, maskAll8);

        AscendC::Reg::Muls<float, float, AscendC::Reg::MaskMergeMode::ZEROING>(vreg1, vreg8, static_cast<float>(-1),
                                                                               maskAll8);
        AscendC::Reg::Exp<float, AscendC::Reg::MaskMergeMode::ZEROING>(vreg2, vreg1, maskAll8);
        AscendC::Reg::Adds<float, float, AscendC::Reg::MaskMergeMode::ZEROING>(vreg3, vreg2, static_cast<float>(1),
                                                                               maskAll8);
        AscendC::Reg::Div<float, AscendC::Reg::MaskMergeMode::ZEROING>(vreg4, vreg0, vreg3, maskAll8);

        AscendC::Reg::Mul<float, AscendC::Reg::MaskMergeMode::ZEROING>(vreg7, vreg4, vreg5, maskAll8);

        AscendC::Reg::Muls<float, float, AscendC::Reg::MaskMergeMode::ZEROING>(vreg11, vreg9, static_cast<float>(-1),
                                                                               maskAll8);
        AscendC::Reg::Exp<float, AscendC::Reg::MaskMergeMode::ZEROING>(vreg12, vreg11, maskAll8);
        AscendC::Reg::Adds<float, float, AscendC::Reg::MaskMergeMode::ZEROING>(vreg13, vreg12, static_cast<float>(1),
                                                                               maskAll8);
        AscendC::Reg::Div<float, AscendC::Reg::MaskMergeMode::ZEROING>(vreg14, vreg0, vreg13, maskAll8);

        AscendC::Reg::Mul<float, AscendC::Reg::MaskMergeMode::ZEROING>(vreg10, vreg14, vreg6, maskAll8);

        AscendC::Reg::Cast<T, float, castTrait11>(vreg15, vreg7, maskAll8);
        AscendC::Reg::Cast<T, float, castTrait12>(vreg16, vreg10, maskAll8);
        AscendC::Reg::Or((Reg::RegTensor<uint16_t>&)vregOutput, (Reg::RegTensor<uint16_t>&)vreg15,
                         (Reg::RegTensor<uint16_t>&)vreg16, maskAll8);

        AscendC::Reg::StoreAlign(outLocalPtr + loopIdx * vlSize, vregOutput, preg0);
    } else {
        AscendC::Reg::Muls<T, T, AscendC::Reg::MaskMergeMode::ZEROING>(vreg1, vregB, static_cast<T>(-1), preg0);
        AscendC::Reg::Exp<T, AscendC::Reg::MaskMergeMode::ZEROING>(vreg2, vreg1, preg0);
        AscendC::Reg::Adds<T, T, AscendC::Reg::MaskMergeMode::ZEROING>(vreg3, vreg2, static_cast<T>(1), preg0);
        AscendC::Reg::Div<T, AscendC::Reg::MaskMergeMode::ZEROING>(vreg4, vreg0, vreg3, preg0);

        AscendC::Reg::Mul<T, AscendC::Reg::MaskMergeMode::ZEROING>(vregOutput, vreg4, vregA, preg0);

        AscendC::Reg::StoreAlign(outLocalPtr + loopIdx * vlSize, vregOutput, preg0);
    }
}

template <typename T>
__aicore__ inline void ComputeSigmoidAndMulImpl(__ubuf__ T* x1LocalPtr, __ubuf__ T* x2LocalPtr, __ubuf__ T* outLocalPtr,
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
        AscendC::Reg::RegTensor<float> vreg0;
        AscendC::Reg::RegTensor<T> vregInput1;
        AscendC::Reg::RegTensor<T> vregInput2;
        AscendC::Reg::RegTensor<T> vregOutput;

        AscendC::Reg::MaskReg preg0;
        uint32_t size = count;
        preg0 = AscendC::Reg::CreateMask<T>();
        AscendC::Reg::Duplicate<float, AscendC::Reg::MaskMergeMode::ZEROING, float>(vreg0, static_cast<float>(1),
                                                                                    preg0);
        AscendC::Reg::MaskReg maskAll8 = Reg::CreateMask<uint8_t, Reg::MaskPattern::ALL>();

        for (uint16_t loopIdx = 0; loopIdx < loopNum; loopIdx++) {
            preg0 = AscendC::Reg::UpdateMask<T>(size);
            AscendC::Reg::LoadAlign(vregInput1, (__ubuf__ T*)(x1LocalPtr + loopIdx * vlSize));
            AscendC::Reg::LoadAlign(vregInput2, (__ubuf__ T*)(x2LocalPtr + loopIdx * vlSize));

            ComputeSigmoidAndMulCore<T>(vregInput1, vregInput2, vregOutput, outLocalPtr, loopIdx, vlSize, preg0,
                                        maskAll8, vreg0);
        }
    }
}

template <typename T>
__aicore__ inline void ComputeSigmoidAndMulWithDeInterleave(__ubuf__ T* xLocalPtr, __ubuf__ T* outLocalPtr,
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
        AscendC::Reg::RegTensor<float> vreg0;
        AscendC::Reg::RegTensor<T> vregInput1;
        AscendC::Reg::RegTensor<T> vregInput2;
        AscendC::Reg::RegTensor<T> vregOutput;
        AscendC::Reg::RegTensor<T> vreg1;
        AscendC::Reg::RegTensor<T> vreg2;

        AscendC::Reg::MaskReg preg0;
        uint32_t size = count;
        preg0 = AscendC::Reg::CreateMask<T>();
        AscendC::Reg::Duplicate<float, AscendC::Reg::MaskMergeMode::ZEROING, float>(vreg0, static_cast<float>(1),
                                                                                    preg0);

        Reg::MaskReg maskAll8 = Reg::CreateMask<uint8_t, Reg::MaskPattern::ALL>();

        for (uint16_t loopIdx = 0; loopIdx < loopNum; loopIdx++) {
            preg0 = AscendC::Reg::UpdateMask<T>(size);
            AscendC::Reg::LoadAlign(vregInput1, (__ubuf__ T*)(xLocalPtr + loopIdx * 2 * vlSize));
            AscendC::Reg::LoadAlign(vregInput2, (__ubuf__ T*)(xLocalPtr + loopIdx * 2 * vlSize + vlSize));

            Reg::DeInterleave<T>(vreg1, vreg2, vregInput1, vregInput2);

            ComputeSigmoidAndMulCore<T>(vreg1, vreg2, vregOutput, outLocalPtr, loopIdx, vlSize, preg0, maskAll8, vreg0);
        }
    }
}

#endif

} // namespace Common
} // namespace Glu

#endif // GLU_COMMON_ARCH35_H

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
 * \file max_pool3d_with_argmax_v2_base.h
 * \brief
 */

#ifndef MAX_POOL3D_WITH_ARGMAX_V2_BASE_H_
#define MAX_POOL3D_WITH_ARGMAX_V2_BASE_H_

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "../inc/platform.h"

using namespace AscendC;

template <typename T>
__aicore__ inline void CalGatterIndex2D(MicroAPI::RegTensor<T>& indexReg, T rate2D, T num1D, T rate1D = 1)
{
    AscendC::MicroAPI::Arange(indexReg, 0);
    AscendC::MicroAPI::RegTensor<T> segScalarReg;
    AscendC::MicroAPI::RegTensor<T> tmpReg;
    AscendC::MicroAPI::RegTensor<T> constReg;
    AscendC::MicroAPI::MaskReg preg = AscendC::MicroAPI::CreateMask<T, AscendC::MicroAPI::MaskPattern::ALL>();
    AscendC::MicroAPI::Duplicate(constReg, T(num1D));
    AscendC::MicroAPI::Div(segScalarReg, indexReg, constReg, preg);
    AscendC::MicroAPI::Muls(tmpReg, segScalarReg, T(num1D), preg);
    AscendC::MicroAPI::Sub(indexReg, indexReg, tmpReg, preg);
    AscendC::MicroAPI::Muls(indexReg, indexReg, T(rate1D), preg);
    AscendC::MicroAPI::Muls(segScalarReg, segScalarReg, T(rate2D), preg);
    AscendC::MicroAPI::Add(indexReg, indexReg, segScalarReg, preg);
}

template <typename T>
__aicore__ inline void CalGatterIndex3D(MicroAPI::RegTensor<T>& indexReg, T rate3D, T num2D, T rate2D, T num1D, T rate1D = 1)
{
    AscendC::MicroAPI::Arange(indexReg, 0);
    AscendC::MicroAPI::RegTensor<T> segScalarReg;
    AscendC::MicroAPI::RegTensor<T> segScalarReg2;
    AscendC::MicroAPI::RegTensor<T> tmpReg;
    AscendC::MicroAPI::RegTensor<T> constReg;
    AscendC::MicroAPI::MaskReg preg = AscendC::MicroAPI::CreateMask<T, AscendC::MicroAPI::MaskPattern::ALL>();
    AscendC::MicroAPI::Duplicate(constReg, T(num2D));
    AscendC::MicroAPI::Div(segScalarReg2, indexReg, constReg, preg);
    AscendC::MicroAPI::Muls(tmpReg, segScalarReg2, T(num2D), preg);
    AscendC::MicroAPI::Sub(indexReg, indexReg, tmpReg, preg);
    AscendC::MicroAPI::Muls(segScalarReg2, segScalarReg2, T(rate3D), preg);

    AscendC::MicroAPI::Duplicate(constReg, T(num1D));
    AscendC::MicroAPI::Div(segScalarReg, indexReg, constReg, preg);
    AscendC::MicroAPI::Muls(tmpReg, segScalarReg, T(num1D), preg);
    AscendC::MicroAPI::Sub(indexReg, indexReg, tmpReg, preg);
    AscendC::MicroAPI::Muls(indexReg, indexReg, T(rate1D), preg);
    AscendC::MicroAPI::Muls(segScalarReg, segScalarReg, T(rate2D), preg);

    AscendC::MicroAPI::Add(indexReg, indexReg, segScalarReg, preg);
    AscendC::MicroAPI::Add(indexReg, indexReg, segScalarReg2, preg);
}

template <typename T>
__aicore__ inline void CalGatterIndex4D(MicroAPI::RegTensor<T>& indexReg, T rate4D, T num3D, T rate3D, T num2D, T rate2D, T num1D, T rate1D = 1)
{
    AscendC::MicroAPI::Arange(indexReg, 0);
    AscendC::MicroAPI::RegTensor<T> segScalarReg;
    AscendC::MicroAPI::RegTensor<T> segScalarReg2;
    AscendC::MicroAPI::RegTensor<T> segScalarReg3;
    AscendC::MicroAPI::RegTensor<T> tmpReg;
    AscendC::MicroAPI::RegTensor<T> constReg;
    AscendC::MicroAPI::MaskReg preg = AscendC::MicroAPI::CreateMask<T, AscendC::MicroAPI::MaskPattern::ALL>();
    AscendC::MicroAPI::Duplicate(constReg, T(num3D));
    AscendC::MicroAPI::Div(segScalarReg3, indexReg, constReg, preg);
    AscendC::MicroAPI::Muls(tmpReg, segScalarReg3, T(num3D), preg);
    AscendC::MicroAPI::Sub(indexReg, indexReg, tmpReg, preg);
    AscendC::MicroAPI::Muls(segScalarReg3, segScalarReg3, T(rate4D), preg);

    AscendC::MicroAPI::Duplicate(constReg, T(num2D));
    AscendC::MicroAPI::Div(segScalarReg2, indexReg, constReg, preg);
    AscendC::MicroAPI::Muls(tmpReg, segScalarReg2, T(num2D), preg);
    AscendC::MicroAPI::Sub(indexReg, indexReg, tmpReg, preg);
    AscendC::MicroAPI::Muls(segScalarReg2, segScalarReg2, T(rate3D), preg);

    AscendC::MicroAPI::Duplicate(constReg, T(num1D));
    AscendC::MicroAPI::Div(segScalarReg, indexReg, constReg, preg);
    AscendC::MicroAPI::Muls(tmpReg, segScalarReg, T(num1D), preg);
    AscendC::MicroAPI::Sub(indexReg, indexReg, tmpReg, preg);
    AscendC::MicroAPI::Muls(indexReg, indexReg, T(rate1D), preg);
    AscendC::MicroAPI::Muls(segScalarReg, segScalarReg, T(rate2D), preg);

    AscendC::MicroAPI::Add(indexReg, indexReg, segScalarReg, preg);
    AscendC::MicroAPI::Add(indexReg, indexReg, segScalarReg2, preg);
    AscendC::MicroAPI::Add(indexReg, indexReg, segScalarReg3, preg);
}

template <typename T>
__aicore__ inline void SetNegInfReg(MicroAPI::RegTensor<T>& negInfReg)
{
     // -inf 
    constexpr uint32_t FLOAT32_NEG_INF = 0xFF800000;
    constexpr uint16_t FLOAT16_NEG_INF = 0xFC00;
    constexpr uint16_t BFLOAT16_NEG_INF = 0xFF80;
    using computeType = std::conditional_t<std::is_same<T, float>::value, uint32_t, uint16_t>;

    if constexpr(std::is_same<T, float>::value) {
        AscendC::MicroAPI::Duplicate((AscendC::MicroAPI::RegTensor<computeType>&)negInfReg, (FLOAT32_NEG_INF));
    } else if constexpr(std::is_same<T, half>::value) {
        AscendC::MicroAPI::Duplicate((AscendC::MicroAPI::RegTensor<computeType>&)negInfReg, (FLOAT16_NEG_INF));
    } else {
        AscendC::MicroAPI::Duplicate((AscendC::MicroAPI::RegTensor<computeType>&)negInfReg, (BFLOAT16_NEG_INF));
    }
}

#endif  // MAX_POOL_WITH_ARGMAX_V3_BASE_H_

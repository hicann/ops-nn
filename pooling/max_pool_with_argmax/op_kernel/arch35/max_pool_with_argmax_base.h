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
 * \file max_pool_with_argmax_base.h
 * \brief
 */

#ifndef MAX_POOL_WITH_ARGMAX_BASE_H_
#define MAX_POOL_WITH_ARGMAX_BASE_H_

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "../inc/platform.h"

using namespace AscendC;

// 默认 rate1D = 1 生成 0 1 2 3 ...         rate1D = 0 生成  0 0 0 0 ...
template <typename T>
__aicore__ inline void GenGatterIndex2D(Reg::RegTensor<T>& indexReg, T rate2D, T num1D, T rate1D = 1)
{
    AscendC::Reg::Arange(indexReg, 0);
    AscendC::Reg::RegTensor<T> segmentScalarReg;
    AscendC::Reg::RegTensor<T> tmpReg;
    AscendC::Reg::RegTensor<T> constReg;
    AscendC::Reg::MaskReg preg = AscendC::Reg::CreateMask<T, AscendC::Reg::MaskPattern::ALL>();
    AscendC::Reg::Duplicate(constReg, T(num1D));
    AscendC::Reg::Div(segmentScalarReg, indexReg, constReg, preg);
    AscendC::Reg::Muls(tmpReg, segmentScalarReg, T(num1D), preg);
    AscendC::Reg::Sub(indexReg, indexReg, tmpReg, preg);
    AscendC::Reg::Muls(indexReg, indexReg, T(rate1D), preg);
    AscendC::Reg::Muls(segmentScalarReg, segmentScalarReg, T(rate2D), preg);

    AscendC::Reg::Add(indexReg, indexReg, segmentScalarReg, preg);
}

template <typename T>
__simd_callee__ inline void GenGatterIndex2DVF(Reg::RegTensor<T>& indexReg, T rate2D, T num1D, T rate1D = 1)
{
    Reg::Arange(indexReg, 0);
    Reg::RegTensor<T> segmentScalarReg;
    Reg::RegTensor<T> tmpReg;
    Reg::RegTensor<T> constReg;
    Reg::MaskReg preg = Reg::CreateMask<T, Reg::MaskPattern::ALL>();
    Reg::Duplicate(constReg, T(num1D));
    Reg::Div(segmentScalarReg, indexReg, constReg, preg);
    Reg::Muls(tmpReg, segmentScalarReg, T(num1D), preg);
    Reg::Sub(indexReg, indexReg, tmpReg, preg);
    Reg::Muls(indexReg, indexReg, T(rate1D), preg);
    Reg::Muls(segmentScalarReg, segmentScalarReg, T(rate2D), preg);

    Reg::Add(indexReg, indexReg, segmentScalarReg, preg);
}

template <typename T>
__simd_callee__ inline void GenGatterIndex3DVF(Reg::RegTensor<T>& indexReg, T rate3D, T num2D, T rate2D, T num1D,
                                               T rate1D = 1)
{
    Reg::Arange(indexReg, 0);
    Reg::RegTensor<T> segmentScalarReg;
    Reg::RegTensor<T> segmentScalarReg2;
    Reg::RegTensor<T> tmpReg;
    Reg::RegTensor<T> constReg;
    Reg::MaskReg preg = Reg::CreateMask<T, Reg::MaskPattern::ALL>();
    Reg::Duplicate(constReg, T(num2D));
    Reg::Div(segmentScalarReg2, indexReg, constReg, preg);
    Reg::Muls(tmpReg, segmentScalarReg2, T(num2D), preg);
    Reg::Sub(indexReg, indexReg, tmpReg, preg);
    Reg::Muls(segmentScalarReg2, segmentScalarReg2, T(rate3D), preg);

    Reg::Duplicate(constReg, T(num1D));
    Reg::Div(segmentScalarReg, indexReg, constReg, preg);
    Reg::Muls(tmpReg, segmentScalarReg, T(num1D), preg);
    Reg::Sub(indexReg, indexReg, tmpReg, preg);
    Reg::Muls(indexReg, indexReg, T(rate1D), preg);
    Reg::Muls(segmentScalarReg, segmentScalarReg, T(rate2D), preg);

    Reg::Add(indexReg, indexReg, segmentScalarReg, preg);
    Reg::Add(indexReg, indexReg, segmentScalarReg2, preg);
}

template <typename T>
__aicore__ inline void GenGatterIndex3D(Reg::RegTensor<T>& indexReg, T rate3D, T num2D, T rate2D, T num1D, T rate1D = 1)
{
    AscendC::Reg::Arange(indexReg, 0);
    AscendC::Reg::RegTensor<T> segmentScalarReg;
    AscendC::Reg::RegTensor<T> segmentScalarReg2;
    AscendC::Reg::RegTensor<T> tmpReg;
    AscendC::Reg::RegTensor<T> constReg;
    AscendC::Reg::MaskReg preg = AscendC::Reg::CreateMask<T, AscendC::Reg::MaskPattern::ALL>();
    AscendC::Reg::Duplicate(constReg, T(num2D));
    AscendC::Reg::Div(segmentScalarReg2, indexReg, constReg, preg);
    AscendC::Reg::Muls(tmpReg, segmentScalarReg2, T(num2D), preg);
    AscendC::Reg::Sub(indexReg, indexReg, tmpReg, preg);
    AscendC::Reg::Muls(segmentScalarReg2, segmentScalarReg2, T(rate3D), preg);

    AscendC::Reg::Duplicate(constReg, T(num1D));
    AscendC::Reg::Div(segmentScalarReg, indexReg, constReg, preg);
    AscendC::Reg::Muls(tmpReg, segmentScalarReg, T(num1D), preg);
    AscendC::Reg::Sub(indexReg, indexReg, tmpReg, preg);
    AscendC::Reg::Muls(indexReg, indexReg, T(rate1D), preg);
    AscendC::Reg::Muls(segmentScalarReg, segmentScalarReg, T(rate2D), preg);

    AscendC::Reg::Add(indexReg, indexReg, segmentScalarReg, preg);
    AscendC::Reg::Add(indexReg, indexReg, segmentScalarReg2, preg);
}

template <typename T>
__aicore__ inline void GenGatterIndex4D(Reg::RegTensor<T>& indexReg, T rate4D, T num3D, T rate3D, T num2D, T rate2D,
                                        T num1D, T rate1D = 1)
{
    AscendC::Reg::Arange(indexReg, 0);
    AscendC::Reg::RegTensor<T> segmentScalarReg;
    AscendC::Reg::RegTensor<T> segmentScalarReg2;
    AscendC::Reg::RegTensor<T> segmentScalarReg3;
    AscendC::Reg::RegTensor<T> tmpReg;
    AscendC::Reg::RegTensor<T> constReg;
    AscendC::Reg::MaskReg preg = AscendC::Reg::CreateMask<T, AscendC::Reg::MaskPattern::ALL>();
    AscendC::Reg::Duplicate(constReg, T(num3D));
    AscendC::Reg::Div(segmentScalarReg3, indexReg, constReg, preg);
    AscendC::Reg::Muls(tmpReg, segmentScalarReg3, T(num3D), preg);
    AscendC::Reg::Sub(indexReg, indexReg, tmpReg, preg);
    AscendC::Reg::Muls(segmentScalarReg3, segmentScalarReg3, T(rate4D), preg);

    AscendC::Reg::Duplicate(constReg, T(num2D));
    AscendC::Reg::Div(segmentScalarReg2, indexReg, constReg, preg);
    AscendC::Reg::Muls(tmpReg, segmentScalarReg2, T(num2D), preg);
    AscendC::Reg::Sub(indexReg, indexReg, tmpReg, preg);
    AscendC::Reg::Muls(segmentScalarReg2, segmentScalarReg2, T(rate3D), preg);

    AscendC::Reg::Duplicate(constReg, T(num1D));
    AscendC::Reg::Div(segmentScalarReg, indexReg, constReg, preg);
    AscendC::Reg::Muls(tmpReg, segmentScalarReg, T(num1D), preg);
    AscendC::Reg::Sub(indexReg, indexReg, tmpReg, preg);
    AscendC::Reg::Muls(indexReg, indexReg, T(rate1D), preg);
    AscendC::Reg::Muls(segmentScalarReg, segmentScalarReg, T(rate2D), preg);

    AscendC::Reg::Add(indexReg, indexReg, segmentScalarReg, preg);
    AscendC::Reg::Add(indexReg, indexReg, segmentScalarReg2, preg);
    AscendC::Reg::Add(indexReg, indexReg, segmentScalarReg3, preg);
}

template <typename T>
__aicore__ inline void DuplicateNegInfReg(Reg::RegTensor<T>& negInfReg)
{
    // -inf
    constexpr uint32_t FLOAT32_NEG_INF = 0xFF800000;
    constexpr uint16_t FLOAT16_NEG_INF = 0xFC00;
    constexpr uint16_t BFLOAT16_NEG_INF = 0xFF80;
    using computeType = std::conditional_t<std::is_same<T, float>::value, uint32_t, uint16_t>;

    if constexpr (std::is_same<T, float>::value) {
        AscendC::Reg::Duplicate((AscendC::Reg::RegTensor<computeType>&)negInfReg, (FLOAT32_NEG_INF));
    } else if constexpr (std::is_same<T, half>::value) {
        AscendC::Reg::Duplicate((AscendC::Reg::RegTensor<computeType>&)negInfReg, (FLOAT16_NEG_INF));
    } else {
        AscendC::Reg::Duplicate((AscendC::Reg::RegTensor<computeType>&)negInfReg, (BFLOAT16_NEG_INF));
    }
}

template <typename T>
__simd_callee__ inline void DuplicateNegInfRegVF(Reg::RegTensor<T>& negInfReg)
{
    constexpr uint32_t FLOAT32_NEG_INF = 0xFF800000;
    constexpr uint16_t FLOAT16_NEG_INF = 0xFC00;
    constexpr uint16_t BFLOAT16_NEG_INF = 0xFF80;
    using computeType = std::conditional_t<std::is_same<T, float>::value, uint32_t, uint16_t>;

    if constexpr (std::is_same<T, float>::value) {
        Reg::Duplicate((Reg::RegTensor<computeType>&)negInfReg, (FLOAT32_NEG_INF));
    } else if constexpr (std::is_same<T, half>::value) {
        Reg::Duplicate((Reg::RegTensor<computeType>&)negInfReg, (FLOAT16_NEG_INF));
    } else {
        Reg::Duplicate((Reg::RegTensor<computeType>&)negInfReg, (BFLOAT16_NEG_INF));
    }
}

template <typename T>
__aicore__ inline void DuplicateLowestReg(Reg::RegTensor<T>& negInfReg)
{
    // min
    constexpr uint32_t FLOAT32_MIN = 0xFF7FFFFF;
    constexpr uint16_t FLOAT16_MIN = 0xFBFF;
    constexpr uint16_t BFLOAT16_MIN = 0xFF7F;
    using computeType = std::conditional_t<std::is_same<T, float>::value, uint32_t, uint16_t>;

    if constexpr (std::is_same<T, float>::value) {
        AscendC::Reg::Duplicate((AscendC::Reg::RegTensor<computeType>&)negInfReg, (FLOAT32_MIN));
    } else if constexpr (std::is_same<T, half>::value) {
        AscendC::Reg::Duplicate((AscendC::Reg::RegTensor<computeType>&)negInfReg, (FLOAT16_MIN));
    } else {
        AscendC::Reg::Duplicate((AscendC::Reg::RegTensor<computeType>&)negInfReg, (BFLOAT16_MIN));
    }
}

#endif // MAX_POOL_WITH_ARGMAX_BASE_H_

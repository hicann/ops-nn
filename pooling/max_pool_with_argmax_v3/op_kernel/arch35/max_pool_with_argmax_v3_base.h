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
 * \file max_pool_with_argmax_v3_base.h
 * \brief
 */

#ifndef MAX_POOL_WITH_ARGMAX_V3_BASE_H_
#define MAX_POOL_WITH_ARGMAX_V3_BASE_H_

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "../inc/platform.h"

using namespace AscendC;

// 默认 rate1D = 1 生成 0 1 2 3 ...         rate1D = 0 生成  0 0 0 0 ...
template <typename T>
__aicore__ inline void GenGatterIndex2D(MicroAPI::RegTensor<T>& indexReg, T rate2D, T num1D, T rate1D = 1)
{
    AscendC::MicroAPI::Arange(indexReg, 0);
    AscendC::MicroAPI::RegTensor<T> segmentScalarReg;
    AscendC::MicroAPI::RegTensor<T> tmpReg;
    AscendC::MicroAPI::RegTensor<T> constReg;
    AscendC::MicroAPI::MaskReg preg = AscendC::MicroAPI::CreateMask<T, AscendC::MicroAPI::MaskPattern::ALL>();
    AscendC::MicroAPI::Duplicate(constReg, T(num1D));
    AscendC::MicroAPI::Div(segmentScalarReg, indexReg, constReg, preg);
    AscendC::MicroAPI::Muls(tmpReg, segmentScalarReg, T(num1D), preg);
    AscendC::MicroAPI::Sub(indexReg, indexReg, tmpReg, preg);
    AscendC::MicroAPI::Muls(indexReg, indexReg, T(rate1D), preg);
    AscendC::MicroAPI::Muls(segmentScalarReg, segmentScalarReg, T(rate2D), preg);

    AscendC::MicroAPI::Add(indexReg, indexReg, segmentScalarReg, preg);
}

template <typename T>
__aicore__ inline void GenGatterIndex3D(
    MicroAPI::RegTensor<T>& indexReg, T rate3D, T num2D, T rate2D, T num1D, T rate1D = 1)
{
    AscendC::MicroAPI::Arange(indexReg, 0);
    AscendC::MicroAPI::RegTensor<T> segmentScalarReg;
    AscendC::MicroAPI::RegTensor<T> segmentScalarReg2;
    AscendC::MicroAPI::RegTensor<T> tmpReg;
    AscendC::MicroAPI::RegTensor<T> constReg;
    AscendC::MicroAPI::MaskReg preg = AscendC::MicroAPI::CreateMask<T, AscendC::MicroAPI::MaskPattern::ALL>();
    AscendC::MicroAPI::Duplicate(constReg, T(num2D));
    AscendC::MicroAPI::Div(segmentScalarReg2, indexReg, constReg, preg);
    AscendC::MicroAPI::Muls(tmpReg, segmentScalarReg2, T(num2D), preg);
    AscendC::MicroAPI::Sub(indexReg, indexReg, tmpReg, preg);
    AscendC::MicroAPI::Muls(segmentScalarReg2, segmentScalarReg2, T(rate3D), preg);

    AscendC::MicroAPI::Duplicate(constReg, T(num1D));
    AscendC::MicroAPI::Div(segmentScalarReg, indexReg, constReg, preg);
    AscendC::MicroAPI::Muls(tmpReg, segmentScalarReg, T(num1D), preg);
    AscendC::MicroAPI::Sub(indexReg, indexReg, tmpReg, preg);
    AscendC::MicroAPI::Muls(indexReg, indexReg, T(rate1D), preg);
    AscendC::MicroAPI::Muls(segmentScalarReg, segmentScalarReg, T(rate2D), preg);

    AscendC::MicroAPI::Add(indexReg, indexReg, segmentScalarReg, preg);
    AscendC::MicroAPI::Add(indexReg, indexReg, segmentScalarReg2, preg);
}

template <typename T>
__aicore__ inline void GenGatterIndex4D(
    MicroAPI::RegTensor<T>& indexReg, T rate4D, T num3D, T rate3D, T num2D, T rate2D, T num1D, T rate1D = 1)
{
    AscendC::MicroAPI::Arange(indexReg, 0);
    AscendC::MicroAPI::RegTensor<T> segmentScalarReg;
    AscendC::MicroAPI::RegTensor<T> segmentScalarReg2;
    AscendC::MicroAPI::RegTensor<T> segmentScalarReg3;
    AscendC::MicroAPI::RegTensor<T> tmpReg;
    AscendC::MicroAPI::RegTensor<T> constReg;
    AscendC::MicroAPI::MaskReg preg = AscendC::MicroAPI::CreateMask<T, AscendC::MicroAPI::MaskPattern::ALL>();
    AscendC::MicroAPI::Duplicate(constReg, T(num3D));
    AscendC::MicroAPI::Div(segmentScalarReg3, indexReg, constReg, preg);
    AscendC::MicroAPI::Muls(tmpReg, segmentScalarReg3, T(num3D), preg);
    AscendC::MicroAPI::Sub(indexReg, indexReg, tmpReg, preg);
    AscendC::MicroAPI::Muls(segmentScalarReg3, segmentScalarReg3, T(rate4D), preg);

    AscendC::MicroAPI::Duplicate(constReg, T(num2D));
    AscendC::MicroAPI::Div(segmentScalarReg2, indexReg, constReg, preg);
    AscendC::MicroAPI::Muls(tmpReg, segmentScalarReg2, T(num2D), preg);
    AscendC::MicroAPI::Sub(indexReg, indexReg, tmpReg, preg);
    AscendC::MicroAPI::Muls(segmentScalarReg2, segmentScalarReg2, T(rate3D), preg);

    AscendC::MicroAPI::Duplicate(constReg, T(num1D));
    AscendC::MicroAPI::Div(segmentScalarReg, indexReg, constReg, preg);
    AscendC::MicroAPI::Muls(tmpReg, segmentScalarReg, T(num1D), preg);
    AscendC::MicroAPI::Sub(indexReg, indexReg, tmpReg, preg);
    AscendC::MicroAPI::Muls(indexReg, indexReg, T(rate1D), preg);
    AscendC::MicroAPI::Muls(segmentScalarReg, segmentScalarReg, T(rate2D), preg);

    AscendC::MicroAPI::Add(indexReg, indexReg, segmentScalarReg, preg);
    AscendC::MicroAPI::Add(indexReg, indexReg, segmentScalarReg2, preg);
    AscendC::MicroAPI::Add(indexReg, indexReg, segmentScalarReg3, preg);
}

template <typename T>
__aicore__ inline void DuplicateNegInfReg(MicroAPI::RegTensor<T>& negInfReg)
{
    // -inf
    constexpr uint32_t FLOAT32_NEG_INF = 0xFF800000;
    constexpr uint16_t FLOAT16_NEG_INF = 0xFC00;
    constexpr uint16_t BFLOAT16_NEG_INF = 0xFF80;
    using computeType = std::conditional_t<std::is_same<T, float>::value, uint32_t, uint16_t>;

    if constexpr (std::is_same<T, float>::value) {
        AscendC::MicroAPI::Duplicate((AscendC::MicroAPI::RegTensor<computeType>&)negInfReg, (FLOAT32_NEG_INF));
    } else if constexpr (std::is_same<T, half>::value) {
        AscendC::MicroAPI::Duplicate((AscendC::MicroAPI::RegTensor<computeType>&)negInfReg, (FLOAT16_NEG_INF));
    } else {
        AscendC::MicroAPI::Duplicate((AscendC::MicroAPI::RegTensor<computeType>&)negInfReg, (BFLOAT16_NEG_INF));
    }
}

#endif // MAX_POOL_WITH_ARGMAX_V3_BASE_H_

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
 * \file max_pool3d_grad_small_kernel_gather.h
 * \brief
 */

#ifndef MAX_POOL3D_GRAD_SMALL_KERNEL_GATHER_H
#define MAX_POOL3D_GRAD_SMALL_KERNEL_GATHER_H

namespace MaxPool3DSmallKernelNameSpace {

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "../inc/platform.h"

using namespace AscendC;

template <typename T>
__aicore__ inline void CalGatterIndex2D(Reg::RegTensor<T>& indexReg, T rate2D, T num1D, T rate1D = 1)
{
    AscendC::Reg::Arange(indexReg, 0);
    AscendC::Reg::RegTensor<T> segScalarReg;
    AscendC::Reg::RegTensor<T> tmpReg;
    AscendC::Reg::RegTensor<T> constReg;
    AscendC::Reg::MaskReg preg = AscendC::Reg::CreateMask<T, AscendC::Reg::MaskPattern::ALL>();
    AscendC::Reg::Duplicate(constReg, T(num1D));
    AscendC::Reg::Div(segScalarReg, indexReg, constReg, preg);
    AscendC::Reg::Muls(tmpReg, segScalarReg, T(num1D), preg);
    AscendC::Reg::Sub(indexReg, indexReg, tmpReg, preg);
    AscendC::Reg::Muls(indexReg, indexReg, T(rate1D), preg);
    AscendC::Reg::Muls(segScalarReg, segScalarReg, T(rate2D), preg);
    AscendC::Reg::Add(indexReg, indexReg, segScalarReg, preg);
}

template <typename T>
__aicore__ inline void CalGatterIndex3D(Reg::RegTensor<T>& indexReg, T rate3D, T num2D, T rate2D, T num1D, T rate1D = 1)
{
    AscendC::Reg::Arange(indexReg, 0);
    AscendC::Reg::RegTensor<T> segScalarReg;
    AscendC::Reg::RegTensor<T> segScalarReg2;
    AscendC::Reg::RegTensor<T> tmpReg;
    AscendC::Reg::RegTensor<T> constReg;
    AscendC::Reg::MaskReg preg = AscendC::Reg::CreateMask<T, AscendC::Reg::MaskPattern::ALL>();
    AscendC::Reg::Duplicate(constReg, T(num2D));
    AscendC::Reg::Div(segScalarReg2, indexReg, constReg, preg);
    AscendC::Reg::Muls(tmpReg, segScalarReg2, T(num2D), preg);
    AscendC::Reg::Sub(indexReg, indexReg, tmpReg, preg);
    AscendC::Reg::Muls(segScalarReg2, segScalarReg2, T(rate3D), preg);

    AscendC::Reg::Duplicate(constReg, T(num1D));
    AscendC::Reg::Div(segScalarReg, indexReg, constReg, preg);
    AscendC::Reg::Muls(tmpReg, segScalarReg, T(num1D), preg);
    AscendC::Reg::Sub(indexReg, indexReg, tmpReg, preg);
    AscendC::Reg::Muls(indexReg, indexReg, T(rate1D), preg);
    AscendC::Reg::Muls(segScalarReg, segScalarReg, T(rate2D), preg);

    AscendC::Reg::Add(indexReg, indexReg, segScalarReg, preg);
    AscendC::Reg::Add(indexReg, indexReg, segScalarReg2, preg);
}

template <typename T>
__aicore__ inline void CalGatterIndex4D(Reg::RegTensor<T>& indexReg, T rate4D, T num3D, T rate3D, T num2D, T rate2D,
                                        T num1D, T rate1D = 1)
{
    AscendC::Reg::Arange(indexReg, 0);
    AscendC::Reg::RegTensor<T> segScalarReg;
    AscendC::Reg::RegTensor<T> segScalarReg2;
    AscendC::Reg::RegTensor<T> segScalarReg3;
    AscendC::Reg::RegTensor<T> tmpReg;
    AscendC::Reg::RegTensor<T> constReg;
    AscendC::Reg::MaskReg preg = AscendC::Reg::CreateMask<T, AscendC::Reg::MaskPattern::ALL>();
    AscendC::Reg::Duplicate(constReg, T(num3D));
    AscendC::Reg::Div(segScalarReg3, indexReg, constReg, preg);
    AscendC::Reg::Muls(tmpReg, segScalarReg3, T(num3D), preg);
    AscendC::Reg::Sub(indexReg, indexReg, tmpReg, preg);
    AscendC::Reg::Muls(segScalarReg3, segScalarReg3, T(rate4D), preg);

    AscendC::Reg::Duplicate(constReg, T(num2D));
    AscendC::Reg::Div(segScalarReg2, indexReg, constReg, preg);
    AscendC::Reg::Muls(tmpReg, segScalarReg2, T(num2D), preg);
    AscendC::Reg::Sub(indexReg, indexReg, tmpReg, preg);
    AscendC::Reg::Muls(segScalarReg2, segScalarReg2, T(rate3D), preg);

    AscendC::Reg::Duplicate(constReg, T(num1D));
    AscendC::Reg::Div(segScalarReg, indexReg, constReg, preg);
    AscendC::Reg::Muls(tmpReg, segScalarReg, T(num1D), preg);
    AscendC::Reg::Sub(indexReg, indexReg, tmpReg, preg);
    AscendC::Reg::Muls(indexReg, indexReg, T(rate1D), preg);
    AscendC::Reg::Muls(segScalarReg, segScalarReg, T(rate2D), preg);

    AscendC::Reg::Add(indexReg, indexReg, segScalarReg, preg);
    AscendC::Reg::Add(indexReg, indexReg, segScalarReg2, preg);
    AscendC::Reg::Add(indexReg, indexReg, segScalarReg3, preg);
}

template <typename T>
__aicore__ inline void SetNegInfReg(Reg::RegTensor<T>& negInfReg)
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
} // namespace MaxPool3DSmallKernelNameSpace
#endif // MAX_POOL3D_GRAD_SMALL_KERNEL_GATHER_H

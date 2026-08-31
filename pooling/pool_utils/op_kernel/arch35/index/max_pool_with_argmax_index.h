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
 * \file max_pool_with_argmax_index.h
 * \brief MaxPoolWithArgmax 系列算子共用的 2D/3D/4D gather 索引生成接口，含标量与 VF 版本。
 */

#ifndef POOL_UTILS_ARCH35_INDEX_MAX_POOL_WITH_ARGMAX_INDEX_H_
#define POOL_UTILS_ARCH35_INDEX_MAX_POOL_WITH_ARGMAX_INDEX_H_

#include "kernel_operator.h"

namespace PoolUtils {
namespace Index {

// 默认 rate1D = 1 生成 0 1 2 3 ...         rate1D = 0 生成  0 0 0 0 ...
template <typename T>
__aicore__ inline void GenGatterIndex2D(AscendC::Reg::RegTensor<T>& indexReg, T rate2D, T num1D, T rate1D = 1)
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
__simd_callee__ inline void GenGatterIndex2DVF(AscendC::Reg::RegTensor<T>& indexReg, T rate2D, T num1D, T rate1D = 1)
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
__aicore__ inline void GenGatterIndex3D(AscendC::Reg::RegTensor<T>& indexReg, T rate3D, T num2D, T rate2D, T num1D,
                                        T rate1D = 1)
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
__simd_callee__ inline void GenGatterIndex3DVF(AscendC::Reg::RegTensor<T>& indexReg, T rate3D, T num2D, T rate2D,
                                               T num1D, T rate1D = 1)
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
__aicore__ inline void GenGatterIndex4D(AscendC::Reg::RegTensor<T>& indexReg, T rate4D, T num3D, T rate3D, T num2D,
                                        T rate2D, T num1D, T rate1D = 1)
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

} // namespace Index
} // namespace PoolUtils

#endif // POOL_UTILS_ARCH35_INDEX_MAX_POOL_WITH_ARGMAX_INDEX_H_

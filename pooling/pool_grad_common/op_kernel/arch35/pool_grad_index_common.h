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
 * \file pool_grad_index_common.h
 * \brief 池化反向系列 kernel 共用的 1D/2D 起始索引生成接口。
 */

#ifndef POOL_GRAD_INDEX_COMMON_H_
#define POOL_GRAD_INDEX_COMMON_H_

#include <cstdint>

#include "kernel_operator.h"

namespace PoolGradCommon {

template <typename T>
__aicore__ inline void GenInitial1DIndices(AscendC::Reg::RegTensor<T>& indexReg, int64_t colGenRate)
{
    AscendC::Reg::Arange(indexReg, 0);
    AscendC::Reg::MaskReg preg = AscendC::Reg::CreateMask<T, AscendC::Reg::MaskPattern::ALL>();
    AscendC::Reg::Muls(indexReg, indexReg, T(colGenRate), preg);
}

template <typename T>
__aicore__ inline void GenInitial2DIndices(AscendC::Reg::RegTensor<T>& indexReg, int64_t colGenRate, int64_t rowGenRate,
                                           int64_t colNumAligned, int64_t fullBatchColNum)
{
    AscendC::Reg::Arange(indexReg, 0);
    AscendC::Reg::RegTensor<T> segmentScalarReg;
    AscendC::Reg::RegTensor<T> segmentIncReg;
    AscendC::Reg::RegTensor<T> constReg;
    AscendC::Reg::Duplicate(constReg, T(fullBatchColNum));
    AscendC::Reg::MaskReg preg = AscendC::Reg::CreateMask<T, AscendC::Reg::MaskPattern::ALL>();

    AscendC::Reg::Div(segmentScalarReg, indexReg, constReg, preg);

    AscendC::Reg::Muls(segmentIncReg, segmentScalarReg, T(fullBatchColNum), preg);
    AscendC::Reg::Sub(segmentIncReg, indexReg, segmentIncReg, preg);

    AscendC::Reg::Muls(segmentIncReg, segmentIncReg, T(colGenRate), preg);
    AscendC::Reg::Muls(segmentScalarReg, segmentScalarReg, T(rowGenRate * colNumAligned), preg);
    AscendC::Reg::Add(indexReg, segmentScalarReg, segmentIncReg, preg);
}

template <typename T>
__aicore__ inline void Gen2DIndexOne(AscendC::Reg::RegTensor<T>& indexReg, int64_t rowGenRate, int64_t colNumAligned)
{
    AscendC::Reg::Arange(indexReg, 0);
    AscendC::Reg::MaskReg preg = AscendC::Reg::CreateMask<T, AscendC::Reg::MaskPattern::ALL>();
    AscendC::Reg::Muls(indexReg, indexReg, T(rowGenRate * colNumAligned), preg);
}

} // namespace PoolGradCommon

#endif // POOL_GRAD_INDEX_COMMON_H_

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
 * \file max_pool_3d_scatter_index.h
 * \brief MaxPool 3D 系列算子 scatter 模板共用的多维索引生成接口。
 */

#ifndef POOL_UTILS_ARCH35_INDEX_MAX_POOL_3D_SCATTER_INDEX_H_
#define POOL_UTILS_ARCH35_INDEX_MAX_POOL_3D_SCATTER_INDEX_H_

#include <cstdint>

#include "kernel_operator.h"

namespace PoolUtils {
namespace Index {

template <typename T>
__aicore__ inline void GenInitial3DIndices(AscendC::Reg::RegTensor<T>& indexReg, int64_t dGenRate, int64_t rowGenRate,
                                           int64_t colGenRate, int64_t fullBatchRowNum, int64_t rowNumCount,
                                           int64_t fullBatchColNum, int64_t colNumAligned)
{
    AscendC::Reg::Arange(indexReg, 0);
    AscendC::Reg::RegTensor<T> segmentScalarReg;
    AscendC::Reg::RegTensor<T> segmentIncReg;
    AscendC::Reg::RegTensor<T> segmentScalarReg2;
    AscendC::Reg::RegTensor<T> segmentIncReg2;
    AscendC::Reg::RegTensor<T> constReg;
    AscendC::Reg::MaskReg preg = AscendC::Reg::CreateMask<T, AscendC::Reg::MaskPattern::ALL>();

    AscendC::Reg::Duplicate(constReg, T(fullBatchColNum * fullBatchRowNum));
    AscendC::Reg::Div(segmentScalarReg, indexReg, constReg, preg);
    AscendC::Reg::Muls(segmentIncReg, segmentScalarReg, T(fullBatchColNum * fullBatchRowNum), preg);
    AscendC::Reg::Sub(segmentIncReg, indexReg, segmentIncReg, preg);

    AscendC::Reg::Muls(segmentScalarReg, segmentScalarReg, T(dGenRate * rowNumCount * colNumAligned), preg);

    AscendC::Reg::Duplicate(constReg, T(fullBatchColNum));
    AscendC::Reg::Div(segmentScalarReg2, segmentIncReg, constReg, preg);
    AscendC::Reg::Muls(segmentIncReg2, segmentScalarReg2, T(fullBatchColNum), preg);
    AscendC::Reg::Sub(segmentIncReg2, segmentIncReg, segmentIncReg2, preg);
    AscendC::Reg::Muls(segmentIncReg2, segmentIncReg2, colGenRate, preg);

    AscendC::Reg::Muls(segmentScalarReg2, segmentScalarReg2, T(rowGenRate * colNumAligned), preg);

    AscendC::Reg::Add(indexReg, segmentIncReg2, segmentScalarReg2, preg);
    AscendC::Reg::Add(indexReg, indexReg, segmentScalarReg, preg);
}

template <typename T>
__aicore__ inline void Gen3DIndexOne(AscendC::Reg::RegTensor<T>& indexReg, int64_t dGenRate, int64_t rowGenRate,
                                     int64_t colNumAligned, int64_t fullBatchRowNum, int64_t rowNumCount)
{
    AscendC::Reg::Arange(indexReg, 0);
    AscendC::Reg::RegTensor<T> segmentScalarReg;
    AscendC::Reg::RegTensor<T> segmentIncReg;
    AscendC::Reg::RegTensor<T> segmentScalarReg2;
    AscendC::Reg::RegTensor<T> segmentIncReg2;
    AscendC::Reg::RegTensor<T> constReg;
    AscendC::Reg::MaskReg preg = AscendC::Reg::CreateMask<T, AscendC::Reg::MaskPattern::ALL>();

    AscendC::Reg::Duplicate(constReg, T(1 * fullBatchRowNum));
    AscendC::Reg::Div(segmentScalarReg, indexReg, constReg, preg);
    AscendC::Reg::Muls(segmentIncReg, segmentScalarReg, T(1 * fullBatchRowNum), preg);
    AscendC::Reg::Sub(segmentIncReg, indexReg, segmentIncReg, preg);

    AscendC::Reg::Muls(segmentScalarReg, segmentScalarReg, T(dGenRate * rowNumCount * colNumAligned), preg);

    AscendC::Reg::Muls(segmentIncReg, segmentIncReg, T(rowGenRate * colNumAligned), preg);

    AscendC::Reg::Add(indexReg, segmentIncReg, segmentScalarReg, preg);
}

template <typename T>
__aicore__ inline void GenInitial3DHighIndices(AscendC::Reg::RegTensor<T>& indexReg, int64_t highStride,
                                               int64_t colGenRate, int64_t rowGenRate, int64_t colNumAligned,
                                               int64_t fullBatchColNum, int64_t fullBatchRowNum)
{
    AscendC::Reg::Arange(indexReg, 0);
    AscendC::Reg::RegTensor<T> constReg;
    AscendC::Reg::RegTensor<T> highReg;
    AscendC::Reg::RegTensor<T> hwReg;
    AscendC::Reg::RegTensor<T> hReg;
    AscendC::Reg::RegTensor<T> wReg;
    AscendC::Reg::RegTensor<T> tmpReg;
    AscendC::Reg::MaskReg preg = AscendC::Reg::CreateMask<T, AscendC::Reg::MaskPattern::ALL>();
    const uint64_t hStride = rowGenRate * colNumAligned;
    const uint64_t wStride = colGenRate;

    AscendC::Reg::Duplicate(constReg, T(fullBatchColNum * fullBatchRowNum));
    AscendC::Reg::Div(highReg, indexReg, constReg, preg);
    AscendC::Reg::Muls(tmpReg, highReg, T(fullBatchColNum * fullBatchRowNum), preg);
    AscendC::Reg::Sub(hwReg, indexReg, tmpReg, preg);

    AscendC::Reg::Duplicate(constReg, T(fullBatchColNum));
    AscendC::Reg::Div(hReg, hwReg, constReg, preg);
    AscendC::Reg::Muls(tmpReg, hReg, T(fullBatchColNum), preg);
    AscendC::Reg::Sub(wReg, hwReg, tmpReg, preg);

    AscendC::Reg::RegTensor<T> highPartReg;
    AscendC::Reg::RegTensor<T> hPartReg;
    AscendC::Reg::RegTensor<T> wPartReg;
    AscendC::Reg::Muls(highPartReg, highReg, T(highStride), preg);
    AscendC::Reg::Muls(hPartReg, hReg, T(hStride), preg);
    AscendC::Reg::Muls(wPartReg, wReg, T(wStride), preg);

    AscendC::Reg::Add(indexReg, highPartReg, hPartReg, preg);
    AscendC::Reg::Add(indexReg, indexReg, wPartReg, preg);
}

template <typename T>
__aicore__ inline void Gen3DHighIndexOne(AscendC::Reg::RegTensor<T>& indexReg, int64_t highStride, int64_t rowGenRate,
                                         int64_t colNumAligned, int64_t fullBatchRowNum)
{
    AscendC::Reg::Arange(indexReg, 0);
    AscendC::Reg::RegTensor<T> constReg;
    AscendC::Reg::RegTensor<T> highReg;
    AscendC::Reg::RegTensor<T> hReg;
    AscendC::Reg::RegTensor<T> tmpReg;
    AscendC::Reg::MaskReg preg = AscendC::Reg::CreateMask<T, AscendC::Reg::MaskPattern::ALL>();
    const uint64_t hStride = rowGenRate * colNumAligned;

    AscendC::Reg::Duplicate(constReg, T(1 * fullBatchRowNum));
    AscendC::Reg::Div(highReg, indexReg, constReg, preg);
    AscendC::Reg::Muls(tmpReg, highReg, T(1 * fullBatchRowNum), preg);
    AscendC::Reg::Sub(hReg, indexReg, tmpReg, preg);

    AscendC::Reg::RegTensor<T> highPartReg;
    AscendC::Reg::RegTensor<T> hPartReg;
    AscendC::Reg::Muls(highPartReg, highReg, T(highStride), preg);
    AscendC::Reg::Muls(hPartReg, hReg, T(hStride), preg);
    AscendC::Reg::Add(indexReg, highPartReg, hPartReg, preg);
}

template <typename T>
__aicore__ inline void GenInitial4DIndices(AscendC::Reg::RegTensor<T>& indexReg, int64_t colGenRate, int64_t rowGenRate,
                                           int64_t colNumAligned, int64_t fullBatchColNum, int64_t fullBatchRowNum,
                                           int64_t fullBatchDepthNum, int64_t depthStride, int64_t highStride)
{
    AscendC::Reg::Arange(indexReg, 0);
    AscendC::Reg::MaskReg preg = AscendC::Reg::CreateMask<T, AscendC::Reg::MaskPattern::ALL>();

    AscendC::Reg::RegTensor<T> highReg;
    AscendC::Reg::RegTensor<T> dReg;
    AscendC::Reg::RegTensor<T> constReg;
    AscendC::Reg::RegTensor<T> dhwReg;
    AscendC::Reg::Duplicate(constReg, T(fullBatchColNum * fullBatchRowNum * fullBatchDepthNum));
    AscendC::Reg::Div(highReg, indexReg, constReg, preg);
    AscendC::Reg::Muls(dReg, highReg, T(fullBatchColNum * fullBatchRowNum * fullBatchDepthNum), preg);
    AscendC::Reg::Sub(dhwReg, indexReg, dReg, preg);

    AscendC::Reg::RegTensor<T> hwReg;
    AscendC::Reg::Duplicate(constReg, T(fullBatchRowNum * fullBatchColNum));
    AscendC::Reg::Div(dReg, dhwReg, constReg, preg);
    AscendC::Reg::Muls(hwReg, dReg, T(fullBatchRowNum * fullBatchColNum), preg);
    AscendC::Reg::Sub(hwReg, dhwReg, hwReg, preg);

    AscendC::Reg::RegTensor<T> hReg;
    AscendC::Reg::RegTensor<T> wReg;

    AscendC::Reg::Duplicate(constReg, T(fullBatchColNum));
    AscendC::Reg::Div(hReg, hwReg, constReg, preg);
    AscendC::Reg::Muls(wReg, hReg, T(fullBatchColNum), preg);
    AscendC::Reg::Sub(wReg, hwReg, wReg, preg);

    // 组装offset
    AscendC::Reg::RegTensor<T> highPartReg;
    AscendC::Reg::RegTensor<T> dPartReg;
    AscendC::Reg::RegTensor<T> hPartReg;
    AscendC::Reg::RegTensor<T> wPartReg;

    AscendC::Reg::Muls(highPartReg, highReg, T(highStride), preg);
    AscendC::Reg::Muls(dPartReg, dReg, T(depthStride), preg);
    AscendC::Reg::Muls(hPartReg, hReg, T(rowGenRate * colNumAligned), preg);
    AscendC::Reg::Muls(wPartReg, wReg, T(colGenRate), preg);
    AscendC::Reg::Add(indexReg, highPartReg, dPartReg, preg);
    AscendC::Reg::Add(indexReg, indexReg, hPartReg, preg);
    AscendC::Reg::Add(indexReg, indexReg, wPartReg, preg);
}

template <typename T>
__aicore__ inline void Gen4DIndexOne(AscendC::Reg::RegTensor<T>& indexReg, int64_t rowGenRate, int64_t colNumAligned,
                                     int64_t fullBatchRowNum, int64_t fullBatchDepthNum, int64_t depthStride,
                                     int64_t highStride)
{
    AscendC::Reg::Arange(indexReg, 0);
    AscendC::Reg::MaskReg preg = AscendC::Reg::CreateMask<T, AscendC::Reg::MaskPattern::ALL>();
    AscendC::Reg::RegTensor<T> highReg;
    AscendC::Reg::RegTensor<T> dReg;
    AscendC::Reg::RegTensor<T> dhwReg;
    AscendC::Reg::RegTensor<T> constReg;

    AscendC::Reg::Duplicate(constReg, T(1 * fullBatchRowNum * fullBatchDepthNum));
    AscendC::Reg::Div(highReg, indexReg, constReg, preg);
    AscendC::Reg::Muls(dReg, highReg, T(1 * fullBatchRowNum * fullBatchDepthNum), preg);
    AscendC::Reg::Sub(dhwReg, indexReg, dReg, preg);

    AscendC::Reg::RegTensor<T> tmpReg;
    AscendC::Reg::RegTensor<T> hReg;

    AscendC::Reg::Duplicate(constReg, T(1 * fullBatchRowNum));
    AscendC::Reg::Div(dReg, dhwReg, constReg, preg);
    AscendC::Reg::Muls(tmpReg, dReg, T(1 * fullBatchRowNum), preg);
    AscendC::Reg::Sub(hReg, dhwReg, tmpReg, preg);
    AscendC::Reg::RegTensor<T> highPartReg;
    AscendC::Reg::RegTensor<T> dPartReg;
    AscendC::Reg::RegTensor<T> hPartReg;

    AscendC::Reg::Muls(highPartReg, highReg, T(highStride), preg);
    AscendC::Reg::Muls(dPartReg, dReg, T(depthStride), preg);
    AscendC::Reg::Muls(hPartReg, hReg, T(rowGenRate * colNumAligned), preg);
    AscendC::Reg::Add(indexReg, highPartReg, dPartReg, preg);
    AscendC::Reg::Add(indexReg, indexReg, hPartReg, preg);
}

} // namespace Index
} // namespace PoolUtils

#endif // POOL_UTILS_ARCH35_INDEX_MAX_POOL_3D_SCATTER_INDEX_H_

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
 * \file pool_2d_row_data_move.h
 * \brief AvgPool/MaxPoolV3 二维池化 big kernel 共用的按行输入搬入接口，区分单行与多行搬入。
 */

#ifndef POOL_UTILS_ARCH35_DATA_MOVE_POOL_2D_ROW_DATA_MOVE_H_
#define POOL_UTILS_ARCH35_DATA_MOVE_POOL_2D_ROW_DATA_MOVE_H_

#include <cstdint>

#include "kernel_operator.h"
#include "op_kernel/platform_util.h"
#include "op_kernel/math_util.h"

namespace PoolUtils {
namespace DataMove {

template <typename T>
__aicore__ inline void CopyInSingleRow(AscendC::TQue<AscendC::QuePosition::VECIN, 2>& inputQue,
                                       const AscendC::GlobalTensor<T>& xGm, int64_t offset, int64_t blockLen)
{
    AscendC::LocalTensor<T> xLocal = inputQue.AllocTensor<T>();

    AscendC::DataCopyPadExtParams<T> padExtParams;
    padExtParams.isPad = false;
    padExtParams.leftPadding = 0;
    padExtParams.rightPadding = 0;
    padExtParams.paddingValue = 0;

    AscendC::DataCopyExtParams extParams;
    extParams.blockCount = 1;
    extParams.blockLen = blockLen * sizeof(T);
    extParams.srcStride = 0;
    extParams.dstStride = 0;
    AscendC::DataCopyPad(xLocal, xGm[offset], extParams, padExtParams);
    inputQue.EnQue(xLocal);
}

namespace BigKernel {

template <typename T>
__aicore__ inline void CopyInMultiRows(AscendC::TQue<AscendC::QuePosition::VECIN, 2>& inputQue,
                                       const AscendC::GlobalTensor<T>& xGm, int64_t offset, int64_t blockLen,
                                       int64_t blockCount, int64_t wInDim)
{
    AscendC::LocalTensor<T> xLocal = inputQue.AllocTensor<T>();

    AscendC::DataCopyPadExtParams<T> padExtParams;
    padExtParams.isPad = false;
    padExtParams.leftPadding = 0;
    padExtParams.rightPadding = 0;
    padExtParams.paddingValue = 0;

    AscendC::DataCopyExtParams extParams;
    extParams.blockCount = blockCount;
    extParams.blockLen = blockLen * sizeof(T);
    extParams.srcStride = (wInDim - blockLen) * sizeof(T);
    extParams.dstStride = 0;
    AscendC::DataCopyPad<T, AscendC::PaddingMode::Compact>(xLocal, xGm[offset], extParams, padExtParams);
    inputQue.EnQue(xLocal);
}

} // namespace BigKernel

namespace SmallKernel {

/*
 * 功能：按 n 方向多通道紧凑搬入多行输入。
 * 说明：通道内按 blockCount * blockLen 对齐后紧凑排布，通道间按 channelStride 跳转。
 */
template <typename T, int32_t BUFFER_NUM>
__aicore__ inline void CopyInMultiRowsCompact(AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM>& inputQue,
                                              const AscendC::GlobalTensor<T>& xGm, int64_t offset, int64_t n,
                                              int64_t blockCount, int64_t blockLen, int64_t wInDim, int64_t hInDim)
{
    AscendC::LocalTensor<T> xLocal = inputQue.template AllocTensor<T>();
    AscendC::DataCopyPadExtParams<T> padExtParams;
    padExtParams.isPad = false;
    padExtParams.leftPadding = 0;
    padExtParams.rightPadding = 0;
    padExtParams.paddingValue = 0;
    int32_t elemNum = Ops::Base::GetUbBlockSize() / sizeof(T);
    int64_t channelStride = Ops::Base::CeilAlign(static_cast<int32_t>(blockCount * blockLen), elemNum);
    AscendC::DataCopyExtParams extParams;
    extParams.blockCount = blockCount;
    extParams.blockLen = blockLen * sizeof(T);
    extParams.srcStride = (wInDim - blockLen) * sizeof(T);
    extParams.dstStride = 0;

    AscendC::LoopModeParams loopParams;
    loopParams.loop2Size = 1;
    loopParams.loop1Size = n;
    loopParams.loop2SrcStride = 0;
    loopParams.loop2DstStride = 0;
    loopParams.loop1SrcStride = wInDim * hInDim * sizeof(T);
    loopParams.loop1DstStride = channelStride * sizeof(T);
    AscendC::SetLoopModePara(loopParams, AscendC::DataCopyMVType::OUT_TO_UB);
    AscendC::DataCopyPad<T, AscendC::PaddingMode::Compact>(xLocal, xGm[offset], extParams, padExtParams);
    AscendC::ResetLoopModePara(AscendC::DataCopyMVType::OUT_TO_UB);

    inputQue.EnQue(xLocal);
}

/*
 * 功能：稀疏（kH < sH）场景按行搬入多行输入。
 * 说明：非切列场景按 outRows 整块搬入；切列场景按 kH 行、inCols 列搬入并按 colsInUb 对齐。
 */
template <typename T, int32_t BUFFER_NUM>
__aicore__ inline void CopyInMultiRowsSparse(AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM>& inputQue,
                                             const AscendC::GlobalTensor<T>& xGm, int64_t offset, int64_t paramN,
                                             int64_t paramOutRows, int32_t inCols, int64_t colsInUb,
                                             int64_t channelStride, int32_t splitMode, int32_t splitColsMode,
                                             int64_t kH, int64_t sH, int64_t wInDim, int64_t hInDim)
{
    AscendC::LocalTensor<T> xLocal = inputQue.template AllocTensor<T>();
    AscendC::DataCopyExtParams extParams;
    AscendC::DataCopyPadExtParams<T> padExtParams;
    AscendC::LoopModeParams loopParams;
    padExtParams.isPad = false;
    padExtParams.leftPadding = 0;
    padExtParams.rightPadding = 0;
    padExtParams.paddingValue = 0;
    if (splitMode != splitColsMode) {
        uint32_t elemNum = Ops::Base::GetUbBlockSize() / sizeof(T);
        extParams.blockCount = paramOutRows;
        extParams.blockLen = kH * wInDim * sizeof(T);
        extParams.srcStride = (sH - kH) * wInDim * sizeof(T);
        extParams.dstStride = 0;
        loopParams.loop2Size = 1;
        loopParams.loop1Size = paramN;
        loopParams.loop2SrcStride = 0;
        loopParams.loop2DstStride = 0;
        loopParams.loop1SrcStride = wInDim * hInDim * sizeof(T);
        loopParams.loop1DstStride = channelStride * sizeof(T);
        AscendC::SetLoopModePara(loopParams, AscendC::DataCopyMVType::OUT_TO_UB);
        AscendC::DataCopyPad<T, AscendC::PaddingMode::Compact>(xLocal, xGm[offset], extParams, padExtParams);
        AscendC::ResetLoopModePara(AscendC::DataCopyMVType::OUT_TO_UB);
    } else {
        uint32_t dstStride = (colsInUb - inCols) * sizeof(T) / Ops::Base::GetUbBlockSize();
        extParams.blockCount = kH;
        extParams.blockLen = inCols * sizeof(T);
        extParams.srcStride = (wInDim - inCols) * sizeof(T);
        extParams.dstStride = dstStride;
        loopParams.loop2Size = paramN;
        loopParams.loop1Size = paramOutRows;
        loopParams.loop2SrcStride = wInDim * hInDim * sizeof(T);
        loopParams.loop2DstStride = paramOutRows * kH * colsInUb * sizeof(T);
        loopParams.loop1SrcStride = wInDim * sH * sizeof(T);
        loopParams.loop1DstStride = kH * colsInUb * sizeof(T);
        AscendC::SetLoopModePara(loopParams, AscendC::DataCopyMVType::OUT_TO_UB);
        AscendC::DataCopyPad<T>(xLocal, xGm[offset], extParams, padExtParams);
        AscendC::ResetLoopModePara(AscendC::DataCopyMVType::OUT_TO_UB);
    }
    inputQue.EnQue(xLocal);
}

} // namespace SmallKernel

} // namespace DataMove
} // namespace PoolUtils

#endif // POOL_UTILS_ARCH35_DATA_MOVE_POOL_2D_ROW_DATA_MOVE_H_

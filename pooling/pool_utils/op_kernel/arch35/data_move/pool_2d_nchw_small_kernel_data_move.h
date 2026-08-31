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
 * \file pool_2d_nchw_small_kernel_data_move.h
 * \brief AvgPool/MaxPoolV3 NCHW small kernel 共用的多行输入搬入与结果搬出接口，区分无 pad 与 pad 两种模板。
 */

#ifndef POOL_UTILS_ARCH35_DATA_MOVE_POOL_2D_NCHW_SMALL_KERNEL_DATA_MOVE_H_
#define POOL_UTILS_ARCH35_DATA_MOVE_POOL_2D_NCHW_SMALL_KERNEL_DATA_MOVE_H_

#include <cstdint>

#include "op_kernel/platform_util.h"
#include "op_kernel/math_util.h"
#include "kernel_operator.h"

namespace PoolUtils {
namespace DataMove {

template <typename T>
__aicore__ inline void CopyMaxOut(AscendC::TQue<AscendC::QuePosition::VECOUT, 2>& maxUBOutput,
                                  const AscendC::GlobalTensor<T>& maxGm, int64_t offset, int64_t n, int64_t blockCount,
                                  int64_t blockLen)
{
    AscendC::LocalTensor<T> maxOutLocal = maxUBOutput.DeQue<T>();
    AscendC::DataCopyExtParams extParams;
    extParams.blockCount = 1;
    extParams.blockLen = (n * blockCount * blockLen) * sizeof(T);
    extParams.srcStride = 0;
    extParams.dstStride = 0;
    AscendC::DataCopyPad<T>(maxGm[offset], maxOutLocal, extParams);
    maxUBOutput.FreeTensor<T>(maxOutLocal);
}

namespace NoPad {

template <typename T>
__aicore__ inline void CopyInMultiRows(AscendC::TQue<AscendC::QuePosition::VECIN, 2>& inputQue,
                                       const AscendC::GlobalTensor<T>& xGm, int64_t offset, int64_t n,
                                       int64_t blockCount, int64_t blockLen, uint32_t colsInUb, int64_t splitMode,
                                       int64_t wInDim)
{
    AscendC::LocalTensor<T> xLocal = inputQue.AllocTensor<T>();
    int64_t channelStride = blockCount * colsInUb;
    AscendC::DataCopyPadExtParams<T> padExtParams;
    padExtParams.isPad = false;
    padExtParams.leftPadding = 0;
    padExtParams.rightPadding = 0;
    padExtParams.paddingValue = 0;

    if (splitMode != 1) { // SPLIT_COLS
        AscendC::DataCopyExtParams extParams;
        extParams.blockCount = 1;
        extParams.blockLen = n * blockCount * blockLen * sizeof(T);
        extParams.srcStride = 0;
        extParams.dstStride = 0;
        AscendC::DataCopyPad<T>(xLocal, xGm[offset], extParams, padExtParams);
    } else {
        uint32_t dstStride = (colsInUb - blockLen) * sizeof(T) / Ops::Base::GetUbBlockSize();
        AscendC::DataCopyExtParams extParams;
        extParams.blockCount = blockCount;
        extParams.blockLen = blockLen * sizeof(T);
        extParams.srcStride = (wInDim - blockLen) * sizeof(T);
        extParams.dstStride = dstStride;
        AscendC::DataCopyPad<T>(xLocal, xGm[offset], extParams, padExtParams);
    }
    inputQue.EnQue(xLocal);
}

} // namespace NoPad

namespace Pad {

template <typename T>
__aicore__ inline void CopyInMultiRows(AscendC::TQue<AscendC::QuePosition::VECIN, 2>& inputQue,
                                       const AscendC::GlobalTensor<T>& xGm, int64_t offset, int64_t n,
                                       int64_t blockCount, int64_t blockLen, int64_t wInDim, int64_t hInDim)
{
    AscendC::LocalTensor<T> xLocal = inputQue.AllocTensor<T>();
    int32_t elemNum = Ops::Base::GetUbBlockSize() / sizeof(T);
    int64_t channelStride = blockCount * Ops::Base::CeilAlign(static_cast<int32_t>(blockLen), elemNum);
    AscendC::DataCopyPadExtParams<T> padExtParams;
    padExtParams.isPad = false;
    padExtParams.leftPadding = 0;
    padExtParams.rightPadding = 0;
    padExtParams.paddingValue = 0;
    uint32_t dstStride = 0;
    AscendC::DataCopyExtParams extParams;
    extParams.blockCount = blockCount;
    extParams.blockLen = blockLen * sizeof(T);
    extParams.srcStride = (wInDim - blockLen) * sizeof(T);
    extParams.dstStride = dstStride;
    if (n > 1) {
        AscendC::LoopModeParams loopParams;
        loopParams.loop2Size = 1;
        loopParams.loop1Size = n;
        loopParams.loop2SrcStride = 0;
        loopParams.loop2DstStride = 0;
        loopParams.loop1SrcStride = wInDim * hInDim * sizeof(T);
        loopParams.loop1DstStride = channelStride * sizeof(T);
        AscendC::SetLoopModePara(loopParams, AscendC::DataCopyMVType::OUT_TO_UB);
        AscendC::DataCopyPad<T>(xLocal, xGm[offset], extParams, padExtParams);
        AscendC::ResetLoopModePara(AscendC::DataCopyMVType::OUT_TO_UB);
    } else {
        AscendC::DataCopyPad<T>(xLocal, xGm[offset], extParams, padExtParams);
    }
    inputQue.EnQue(xLocal);
}

} // namespace Pad

} // namespace DataMove
} // namespace PoolUtils

#endif // POOL_UTILS_ARCH35_DATA_MOVE_POOL_2D_NCHW_SMALL_KERNEL_DATA_MOVE_H_

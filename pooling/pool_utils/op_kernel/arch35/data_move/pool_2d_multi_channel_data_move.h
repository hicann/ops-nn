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
 * \file pool_2d_multi_channel_data_move.h
 * \brief AvgPool/MaxPoolV3 NHWC 多通道场景共用的输入搬入与结果搬出接口。
 */

#ifndef POOL_UTILS_ARCH35_DATA_MOVE_POOL_2D_MULTI_CHANNEL_DATA_MOVE_H_
#define POOL_UTILS_ARCH35_DATA_MOVE_POOL_2D_MULTI_CHANNEL_DATA_MOVE_H_

#include <cstdint>

#include "op_kernel/platform_util.h"
#include "kernel_operator.h"

namespace PoolUtils {
namespace DataMove {

struct MultiChannelCopyParams {
    int64_t srcOffset;
    int64_t n;
    int64_t rows;
    int64_t cols;
    int64_t channels;
    int64_t alignChannels;
    int64_t winDim;
    int64_t splitMode;
};

struct MultiChannelCopyOutParams {
    int64_t gmOffset;
    int64_t n;
    int64_t rows;
    int64_t cols;
    int64_t channels;
};

template <typename T>
__aicore__ inline void CopyInMultiChannels(AscendC::TQue<AscendC::QuePosition::VECIN, 2>& inputQue,
                                           const AscendC::GlobalTensor<T>& xGm, const MultiChannelCopyParams& params)
{
    AscendC::LocalTensor<T> xLocal = inputQue.AllocTensor<T>();
    AscendC::DataCopyPadExtParams<T> padExtParams;
    padExtParams.isPad = false;
    padExtParams.leftPadding = 0;
    padExtParams.rightPadding = 0;
    padExtParams.paddingValue = 0;
    uint32_t dstStride = (params.alignChannels - params.channels) * sizeof(T) / Ops::Base::GetUbBlockSize();
    AscendC::DataCopyExtParams extParams;
    extParams.blockCount = params.n * params.rows * params.cols;
    extParams.blockLen = params.channels * sizeof(T);
    extParams.srcStride = 0;
    extParams.dstStride = dstStride;
    // w不切，地址连续
    if (params.splitMode != 1) {
        AscendC::DataCopyPad<T>(xLocal, xGm[params.srcOffset], extParams, padExtParams);
    } else {
        AscendC::LoopModeParams loopParams;
        loopParams.loop2Size = 1;
        loopParams.loop1Size = params.rows;
        loopParams.loop2SrcStride = 0;
        loopParams.loop2DstStride = 0;
        loopParams.loop1SrcStride = params.winDim * params.channels * sizeof(T);
        loopParams.loop1DstStride = params.cols * params.alignChannels * sizeof(T);
        AscendC::SetLoopModePara(loopParams, AscendC::DataCopyMVType::OUT_TO_UB);
        extParams.blockCount = params.cols;
        AscendC::DataCopyPad<T>(xLocal, xGm[params.srcOffset], extParams, padExtParams);
        AscendC::ResetLoopModePara(AscendC::DataCopyMVType::OUT_TO_UB);
    }
    inputQue.EnQue(xLocal);
}

template <typename T>
__aicore__ inline void CopyOutMultiChannels(AscendC::TQue<AscendC::QuePosition::VECOUT, 2>& maxUBOutput,
                                            const AscendC::GlobalTensor<T>& maxGm,
                                            const MultiChannelCopyOutParams& params)
{
    AscendC::LocalTensor<T> maxOutLocal = maxUBOutput.DeQue<T>();
    AscendC::DataCopyExtParams extParams;
    extParams.blockCount = params.n * params.rows * params.cols;
    extParams.blockLen = params.channels * sizeof(T);
    extParams.srcStride = 0;
    extParams.dstStride = 0;
    AscendC::DataCopyPad<T>(maxGm[params.gmOffset], maxOutLocal, extParams);
    maxUBOutput.FreeTensor<T>(maxOutLocal);
}

} // namespace DataMove
} // namespace PoolUtils

#endif // POOL_UTILS_ARCH35_DATA_MOVE_POOL_2D_MULTI_CHANNEL_DATA_MOVE_H_

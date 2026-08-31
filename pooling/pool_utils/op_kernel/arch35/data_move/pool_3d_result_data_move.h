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
 * \file pool_3d_result_data_move.h
 * \brief Pool3D 系列 kernel（pad 与非 pad）共用的结果搬出接口。
 */

#ifndef POOL_UTILS_ARCH35_DATA_MOVE_POOL_3D_RESULT_DATA_MOVE_H_
#define POOL_UTILS_ARCH35_DATA_MOVE_POOL_3D_RESULT_DATA_MOVE_H_

#include <cstdint>

#include "kernel_operator.h"

namespace PoolUtils {
namespace DataMove {

// Pool3D NCDHW small kernel：连续搬出 blockLen 个元素。
template <typename T>
__aicore__ inline void CopyMaxOutNcdhwSmall(AscendC::TQue<AscendC::QuePosition::VECOUT, 2>& maxUBOutput,
                                            const AscendC::GlobalTensor<T>& maxGm, int64_t offset, int64_t blockLen)
{
    AscendC::LocalTensor<T> maxOutLocal = maxUBOutput.DeQue<T>();
    AscendC::DataCopyExtParams extParams;
    extParams.blockCount = 1;
    extParams.blockLen = blockLen * sizeof(T);
    extParams.srcStride = 0;
    extParams.dstStride = 0;
    AscendC::DataCopyPad<T>(maxGm[offset], maxOutLocal, extParams);
    maxUBOutput.FreeTensor<T>(maxOutLocal);
}

// Pool3D NDHWC small kernel：连续搬出 n * blockCount * blockLen * channels 个元素。
template <typename T>
__aicore__ inline void CopyMaxOutNdhwcSmall(AscendC::TQue<AscendC::QuePosition::VECOUT, 2>& maxUBOutput,
                                            const AscendC::GlobalTensor<T>& maxGm, int64_t offset, int64_t n,
                                            int64_t blockCount, int64_t blockLen, int64_t channels)
{
    AscendC::LocalTensor<T> maxOutLocal = maxUBOutput.DeQue<T>();
    AscendC::DataCopyExtParams extParams;
    extParams.blockCount = 1;
    extParams.blockLen = (n * blockCount * blockLen * channels) * sizeof(T);
    extParams.srcStride = 0;
    extParams.dstStride = 0;
    AscendC::DataCopyPad<T>(maxGm[offset], maxOutLocal, extParams);
    maxUBOutput.FreeTensor<T>(maxOutLocal);
}

// Pool3D NDHWC big channel kernel：按 n * deps * rows * cols 个 block 搬出，每 block channels 个元素。
template <typename T>
__aicore__ inline void CopyOutMultiChannelsNdhwc(AscendC::TQue<AscendC::QuePosition::VECOUT, 2>& maxUBOutput,
                                                 const AscendC::GlobalTensor<T>& maxGm, int64_t offset, int64_t n,
                                                 int64_t deps, int64_t rows, int64_t cols, int64_t channels)
{
    AscendC::LocalTensor<T> maxOutLocal = maxUBOutput.DeQue<T>();
    AscendC::DataCopyExtParams extParams;
    extParams.blockCount = n * deps * rows * cols;
    extParams.blockLen = channels * sizeof(T);
    extParams.srcStride = 0;
    extParams.dstStride = 0;
    AscendC::DataCopyPad<T>(maxGm[offset], maxOutLocal, extParams);
    maxUBOutput.FreeTensor<T>(maxOutLocal);
}

} // namespace DataMove
} // namespace PoolUtils

#endif // POOL_UTILS_ARCH35_DATA_MOVE_POOL_3D_RESULT_DATA_MOVE_H_

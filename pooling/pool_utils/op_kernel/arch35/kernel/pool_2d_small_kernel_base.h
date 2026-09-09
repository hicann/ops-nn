/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software: you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file pool_2d_small_kernel_base.h
 * \brief AvgPool/MaxPoolV3 二维池化 small kernel 共用的 pipe/队列/buffer 字段与 Init 搬运逻辑。
 */

#ifndef POOL_UTILS_ARCH35_KERNEL_POOL_2D_SMALL_KERNEL_BASE_H_
#define POOL_UTILS_ARCH35_KERNEL_POOL_2D_SMALL_KERNEL_BASE_H_

#include <cstdint>

#include "kernel_tiling/kernel_tiling.h"
#include "kernel_operator.h"

namespace PoolUtils {
namespace Kernel {

constexpr int32_t POOL_2D_SMALL_KERNEL_BUFFER_NUM = 2;

/*
 * 功能：AvgPool/MaxPoolV3 small kernel 共用基类，收编与 dtype 无关的 pipe/输入队列/输出队列/index buffer 字段。
 * 说明：GM tensor 与 tiling 指针依赖派生类模板参数，保留在派生类中声明。
 */
class Pool2DSmallKernelIoBase {
protected:
    __aicore__ inline Pool2DSmallKernelIoBase(AscendC::TPipe* pipe) : pipe_(pipe) {}

    /*
     * 功能：small kernel 共用的 Init 搬运逻辑，绑定 GM、初始化输入/输出队列与 index buffer。
     * 说明：indexBufferSize 由派生类注入（NCHW 使用 tiling 的 indiceUbSize，NHWC 使用固定 INDEX_SIZE）。
     */
    template <typename T, typename TilingT>
    __aicore__ inline void InitIo(GM_ADDR x, GM_ADDR y, AscendC::GlobalTensor<T>& xGm, AscendC::GlobalTensor<T>& maxGm,
                                  const TilingT* tilingData, int64_t indexBufferSize)
    {
        // GM
        xGm.SetGlobalBuffer((__gm__ T*)x);
        maxGm.SetGlobalBuffer((__gm__ T*)y);

        pipe_->InitBuffer(inputQue_, POOL_2D_SMALL_KERNEL_BUFFER_NUM, tilingData->inUbSize * sizeof(T));
        pipe_->InitBuffer(maxUBOutput_, POOL_2D_SMALL_KERNEL_BUFFER_NUM, tilingData->outUbSize * sizeof(T));
        pipe_->InitBuffer(indexBuf_, indexBufferSize);
    }

    AscendC::TPipe* pipe_;
    // 输入队列
    AscendC::TQue<AscendC::QuePosition::VECIN, POOL_2D_SMALL_KERNEL_BUFFER_NUM> inputQue_;
    // 输出ub
    AscendC::TQue<AscendC::QuePosition::VECOUT, POOL_2D_SMALL_KERNEL_BUFFER_NUM> maxUBOutput_;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> indexBuf_;
};

} // namespace Kernel
} // namespace PoolUtils

#endif // POOL_UTILS_ARCH35_KERNEL_POOL_2D_SMALL_KERNEL_BASE_H_

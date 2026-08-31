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
 * \file adaptive_pool_big_kernel_data_move.h
 * \brief AdaptivePool2D/AdaptivePool3D big kernel 共用的批量结果回搬接口。
 */

#ifndef POOL_UTILS_ARCH35_DATA_MOVE_ADAPTIVE_POOL_BIG_KERNEL_DATA_MOVE_H_
#define POOL_UTILS_ARCH35_DATA_MOVE_ADAPTIVE_POOL_BIG_KERNEL_DATA_MOVE_H_

#include <cstdint>

#include "kernel_operator.h"

namespace PoolUtils {
namespace DataMove {

/*
 * 功能：将输出 UB 中已攒批的 copyCount 个结果一次性搬出到 GM 的指定偏移。
 */
template <typename T>
__aicore__ inline void CopyOut(AscendC::TBuf<AscendC::QuePosition::VECCALC>& outputUB,
                               const AscendC::GlobalTensor<T>& yGm, int64_t copyCount, int64_t offset)
{
    event_t eventIdVtoMTE3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::V_MTE3));
    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(eventIdVtoMTE3);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(eventIdVtoMTE3);
    AscendC::LocalTensor<T> outputLocal = outputUB.Get<T>();
    AscendC::DataCopyExtParams extParams;
    extParams.blockCount = 1;
    extParams.blockLen = copyCount * sizeof(T);
    extParams.srcStride = 0;
    extParams.dstStride = 0;
    AscendC::DataCopyPad(yGm[offset], outputLocal, extParams);
}

} // namespace DataMove
} // namespace PoolUtils

#endif // POOL_UTILS_ARCH35_DATA_MOVE_ADAPTIVE_POOL_BIG_KERNEL_DATA_MOVE_H_

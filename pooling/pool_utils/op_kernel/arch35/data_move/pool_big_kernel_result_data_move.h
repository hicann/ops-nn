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
 * \file pool_big_kernel_result_data_move.h
 * \brief AvgPool/Pool3D big kernel 共用的单点归约结果回搬接口，将本轮结果写入输出 UB 的指定位置。
 */

#ifndef POOL_UTILS_ARCH35_DATA_MOVE_POOL_BIG_KERNEL_RESULT_DATA_MOVE_H_
#define POOL_UTILS_ARCH35_DATA_MOVE_POOL_BIG_KERNEL_RESULT_DATA_MOVE_H_

#include <cstdint>

#include "kernel_operator.h"

namespace PoolUtils {
namespace DataMove {
namespace BigKernel {

template <typename T>
__aicore__ inline void CopyResultToUb(AscendC::TBuf<>& uBOutput, AscendC::TBuf<>& ubLoopResult, int64_t curIdx,
                                      int32_t one)
{
    AscendC::LocalTensor<T> uboutLocal = uBOutput.Get<T>();
    __ubuf__ T* dstAddr = (__ubuf__ T*)uboutLocal.GetPhyAddr() + curIdx;

    AscendC::LocalTensor<T> ubResult = ubLoopResult.Get<T>();
    __ubuf__ T* srcAddr = (__ubuf__ T*)ubResult.GetPhyAddr();

    __VEC_SCOPE__
    {
        AscendC::Reg::RegTensor<T> res;
        AscendC::Reg::UnalignRegForLoad u0;
        AscendC::Reg::LoadUnAlignPre(u0, srcAddr);
        AscendC::Reg::LoadUnAlign(res, u0, srcAddr, one);

        AscendC::Reg::UnalignRegForStore u1;
        AscendC::Reg::StoreUnAlign(dstAddr, res, u1, one);
        AscendC::Reg::StoreUnAlignPost(dstAddr, u1, 0);
    }
}

} // namespace BigKernel
} // namespace DataMove
} // namespace PoolUtils

#endif // POOL_UTILS_ARCH35_DATA_MOVE_POOL_BIG_KERNEL_RESULT_DATA_MOVE_H_

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
 * \file pool_3d_ndhwc_gather_index_common.h
 * \brief Pool3D NDHWC small kernel（pad 与非 pad）共用的 gather 索引生成分派入口。
 */

#ifndef POOL_3D_NDHWC_GATHER_INDEX_COMMON_H_
#define POOL_3D_NDHWC_GATHER_INDEX_COMMON_H_

#include <cstdint>

#include "kernel_operator.h"
#include "gather_index_impl.h"
#include "pool_3d_common.h"

namespace Pool3D {

template <typename U, int32_t GATHER_MODE>
__aicore__ inline void GenNdhwcSmallGatherIndex(const GatherIndexImpl::ShapeInfo& param,
                                                AscendC::LocalTensor<U>& indexLocal, uint16_t loopNum)
{
    if constexpr (GATHER_MODE == GATHER_SINGLE_ROW) {
        GatherIndexImpl::GenGatherIndex<U, GatherIndexImpl::TWO>(param, indexLocal, loopNum);
    } else if constexpr (GATHER_MODE == GATHER_MULTI_ROW) {
        GatherIndexImpl::GenGatherIndex<U, GatherIndexImpl::THREE>(param, indexLocal, loopNum);
    } else if constexpr (GATHER_MODE == GATHER_MULTI_PLANE) {
        GatherIndexImpl::GenGatherIndex<U, GatherIndexImpl::FOUR>(param, indexLocal, loopNum);
    } else {
        GatherIndexImpl::GenGatherIndex<U, GatherIndexImpl::FIVE>(param, indexLocal, loopNum);
    }
}

} // namespace Pool3D

#endif // POOL_3D_NDHWC_GATHER_INDEX_COMMON_H_

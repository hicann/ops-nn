/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file gemm_v3_base_kernel.h
 * \brief
 */

#ifndef GEMM_V3_BASE_KERNEL_H
#define GEMM_V3_BASE_KERNEL_H

#ifdef __CCE_KT_TEST__
#include "stub_def.h"
#include "stub_fun.h"
#else
#define __aicore__ [aicore]
#endif

#include "gemm_v3_tiling_data.h"
#include "kernel_tensor.h"
#include "../transpose_batch_mat_mul/utils/common.h"

namespace PpMatMulNS {
template <uint32_t swizzleDirect,
          bool transA,
          bool transB,
          typename InDtype,
          typename OutDtype,
          typename AccumDtype,
          DataFormat formatA = DataFormat::ND,
          DataFormat formatB = DataFormat::ND>
class GemmV3BaseKernel {
public:
    __aicore__ explicit GemmV3BaseKernel(){};
    __aicore__ FORCE_INLINE void Init(__gm__ uint8_t* __restrict__ a,
                                      __gm__ uint8_t* __restrict__ b,
                                      __gm__ uint8_t* __restrict__ c,
                                      __gm__ uint8_t* __restrict__ y,
                                      __gm__ uint8_t* __restrict__ workspace,
                                      const GemmV3TilingData& tilingData);
    __aicore__ FORCE_INLINE void RunCube();
    __aicore__ FORCE_INLINE void RunVector();
};

template <uint32_t swizzleDirect,
          bool transA,
          bool transB,
          typename InDtype,
          typename OutDtype,
          typename AccumDtype,
          DataFormat formatA,
          DataFormat formatB>
__aicore__ FORCE_INLINE void
GemmV3BaseKernel<swizzleDirect, transA, transB, InDtype, OutDtype, AccumDtype, formatA, formatB>::Init(
    __gm__ uint8_t* __restrict__ a,
    __gm__ uint8_t* __restrict__ b,
    __gm__ uint8_t* __restrict__ c,
    __gm__ uint8_t* __restrict__ y,
    __gm__ uint8_t* __restrict__ workspace,
    const GemmV3TilingData& tilingData)
{
}

template <uint32_t swizzleDirect,
          bool transA,
          bool transB,
          typename InDtype,
          typename OutDtype,
          typename AccumDtype,
          DataFormat formatA,
          DataFormat formatB>
__aicore__ FORCE_INLINE void
GemmV3BaseKernel<swizzleDirect, transA, transB, InDtype, OutDtype, AccumDtype, formatA, formatB>::RunCube()
{
}

template <uint32_t swizzleDirect,
          bool transA,
          bool transB,
          typename InDtype,
          typename OutDtype,
          typename AccumDtype,
          DataFormat formatA,
          DataFormat formatB>
__aicore__ FORCE_INLINE void
GemmV3BaseKernel<swizzleDirect, transA, transB, InDtype, OutDtype, AccumDtype, formatA, formatB>::RunVector()
{
}
} // namespace PpMatMulNS
#endif // GEMM_V3_BASE_KERNEL_H
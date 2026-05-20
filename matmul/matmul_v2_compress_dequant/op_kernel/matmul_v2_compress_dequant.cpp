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
 * \file matmul_v2_compress_dequant.cpp
 * \brief matmul_v2_compress_dequant kernel entry, ported from pp_matmul_i8_nz_compress.cce
 */
#include "kernel_operator.h"
#include "pp_matmul_i8_nz_compress.h"

#if defined(__DAV_C100__) || defined(__DAV_M200__)

extern "C" __global__ __aicore__ void mat_mul_v2_compress_dequant(
    GM_ADDR x1,
    GM_ADDR x2,
    GM_ADDR compress_index,
    GM_ADDR deq_scale,
    GM_ADDR bias,
    GM_ADDR offset_w,
    GM_ADDR out,
    GM_ADDR workspace,
    GM_ADDR tiling)
{
    SetPadding<uint64_t>((uint64_t)0x0);
    SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
    SetAtomicnone();

    PpMatmulI8NzCompress<0, false, true, false, int8_t, uint64_t, int32_t> kernel;

    AscendC::GlobalTensor<int32_t> gm_tiling;
    gm_tiling.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(tiling));
    AscendC::LocalTensor<int32_t> ub_tiling =
        kernel.buf.GetBuffer<BufferType::ASCEND_UB, int32_t>(0);
    gm_to_ub<ArchType::ASCEND_V200, int32_t>(ub_tiling, gm_tiling,
                                             0,
                                             1,
                                             512 / 32,
                                             0,
                                             0);
    SET_FLAG(MTE2, S, EVENT_ID0);
    WAIT_FLAG(MTE2, S, EVENT_ID0);

    kernel.Init(
        reinterpret_cast<__gm__ uint8_t *>(x1),
        reinterpret_cast<__gm__ uint8_t *>(x2),
        reinterpret_cast<__gm__ uint8_t *>(bias),
        reinterpret_cast<__gm__ uint8_t *>(deq_scale),
        reinterpret_cast<__gm__ uint8_t *>(compress_index),
        reinterpret_cast<__gm__ uint8_t *>(out),
        ub_tiling.GetValue(0),   // batchSize
        ub_tiling.GetValue(1),   // m
        ub_tiling.GetValue(2),   // k
        ub_tiling.GetValue(3),   // n
        ub_tiling.GetValue(4),   // m0
        ub_tiling.GetValue(5),   // k0
        ub_tiling.GetValue(6),   // n0
        ub_tiling.GetValue(7),   // mLoop
        ub_tiling.GetValue(8),   // kLoop
        ub_tiling.GetValue(9),   // nLoop
        ub_tiling.GetValue(10),  // coreLoop
        ub_tiling.GetValue(11),  // swizzlCount
        ub_tiling.GetValue(12),  // tilingK
        ub_tiling.GetValue(13),  // tilingN
        ub_tiling.GetValue(14)); // compressOverlapN

    kernel.Process();
}

#endif

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
 * \file matmul_compress.cpp
 * \brief
 */  
#include "matmul_compress.h"
#include "lib/matmul_intf.h"
#include "kernel_operator.h"
#include "matmul_compress_tiling_key.h"
#include "matmul_compress_tiling_data.h"

template <int PP_MAT_MUL_MODE, int TRANS>
__global__ __aicore__ void matmul_compress(
    GM_ADDR aGM, GM_ADDR bGM, GM_ADDR biasGM, GM_ADDR compressIndexGM, GM_ADDR cGM, GM_ADDR workspaceGM, GM_ADDR tilingGM)
{
    REGISTER_TILING_DEFAULT(MatmulCompressTilingDataArch20);
    GET_TILING_DATA(tilingData, tilingGM);
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 2002)
    SetPadding<uint64_t>((uint64_t)0x0);
    SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
    SetAtomicnone();
    if constexpr (PP_MAT_MUL_MODE == MATMUL_COMPRESS_PP_MAT_MUL_MODE_TRUE && TRANS == MATMUL_COMPRESS_B_TRANS){
        MatmulCompressSpace::MatmulCompress<0, false, true, false, half, uint64_t, float, float, true, false> kernel;
        SET_FLAG(MTE2, S, EVENT_ID0);
        WAIT_FLAG(MTE2, S, EVENT_ID0);
        kernel.Init(aGM, bGM, biasGM, compressIndexGM, cGM, &tilingData);
        kernel.Process();
    }
#endif
}

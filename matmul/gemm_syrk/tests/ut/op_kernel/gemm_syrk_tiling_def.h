/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software: you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef _TEST_GEMM_SYRK_TILING_DEF_H_
#define _TEST_GEMM_SYRK_TILING_DEF_H_

#include "blaze_kernel_stub.h"
#include "kernel_tiling/kernel_tiling.h"
#include <string>

inline void InitGemmSyrkTilingData(void* tiling, void* const_data, size_t size) { memcpy(const_data, tiling, size); }

#define GET_TILING_DATA_WITH_STRUCT(tiling_struct, tiling_data, tiling_arg) \
    tiling_struct tiling_data;                                              \
    InitGemmSyrkTilingData(tiling_arg, &tiling_data, sizeof(tiling_struct));

#define X_LOG(format, ...)

#define KERNEL_LOG_KERNEL_EORROR(format, ...)

// CANN 9.1.0 CPU debug compatibility: the tikicpulib stub only provides the
// 8-arg copy_gm_to_cbuf_v2, while the Blaze fixpipe block's ND GM2L1 copy
// (tensor_api asc_copy_gm2l1_impl) calls the 9-arg cache_mode overload.
#ifndef COPY_GM_TO_CBUF_V2_WITH_CACHE_MODE
#define COPY_GM_TO_CBUF_V2_WITH_CACHE_MODE
#include <cstdint>
inline void copy_gm_to_cbuf_v2(void* dst, void* src, uint8_t sid, uint32_t n_burst, uint32_t len_burst,
                               uint8_t pad_func_mode, uint8_t cache_mode, uint64_t src_stride, uint32_t dst_stride)
{}
#endif

#endif // _TEST_GEMM_SYRK_TILING_DEF_H_

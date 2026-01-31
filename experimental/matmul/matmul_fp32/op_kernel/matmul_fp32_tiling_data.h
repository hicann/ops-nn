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
 * \file matmul_fp32_tiling_data.h
 * \brief tiling data struct
 */

#ifndef __MATMUL_FP32_TILING_DATA_H__
#define __MATMUL_FP32_TILING_DATA_H__

#include "kernel_tiling/kernel_tiling.h"

struct MatmulFp32RunInfo {
    uint32_t transA;
    uint32_t transB;
};

struct MatmulFp32TilingData {
    TCubeTiling tCubeTiling;
    MatmulFp32RunInfo matmulFp32RunInfo;
};
#endif

/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file transpose_batch_mat_mul_tiling_data.h
 * \brief Basic tiling data for TBMM arch35.
 */
#pragma once

#if defined(__CCE_AICORE__)
#include "../../mat_mul_v3/arch35/mat_mul_tiling_data.h"
#else
#include "matmul/mat_mul_v3/op_kernel/arch35/mat_mul_tiling_data.h"
#endif

#pragma pack(push, 8)
struct TransposeBatchMatMulBasicTilingData {
    BatchMatMulV3BasicTilingData matMulTilingData;
    uint32_t batchSplitFactor = 1;
};
#pragma pack(pop)

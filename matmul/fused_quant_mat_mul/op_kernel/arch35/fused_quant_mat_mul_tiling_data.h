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
 * \file fused_quant_mat_mul_tiling_data.h
 * \brief
 */
#ifndef __OP_KERNEL_FUSED_QUANT_MAT_MUL_TILING_DATA_H__
#define __OP_KERNEL_FUSED_QUANT_MAT_MUL_TILING_DATA_H__
#include "kernel_tiling/kernel_tiling.h"

#ifndef __CCE_AICORE__
#include <cstdint>
#endif

#pragma pack(push, 8)
struct FusedQuantMatmulSwigluBaseParams {
    uint8_t isDeqQuant;
    uint8_t isQuant;
    uint8_t isFullLoadA;
    uint8_t reserved0 = 0;
    uint32_t singleCoreM;
    uint32_t singleCoreN;
    uint32_t mLoops;
    uint32_t nLoops;
    uint32_t singleMTail;
    uint32_t singleNTail;
    uint32_t reserved1;
};
#pragma pack(pop)

#pragma pack(push, 8)
struct FusedQuantMatmulSwigluTilingData {
    FusedQuantMatmulSwigluBaseParams baseParams;
    TCubeTiling mmTilingData;
};
#pragma pack(pop)


#endif // __OP_KERNEL_FUSED_QUANT_MAT_MUL_TILING_DATA_H__
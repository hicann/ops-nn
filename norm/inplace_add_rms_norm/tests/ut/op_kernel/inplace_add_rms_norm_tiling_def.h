/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef INPLACE_ADD_RMS_NORM_TILING_H_
#define INPLACE_ADD_RMS_NORM_TILING_H_

#include "kernel_tiling/kernel_tiling.h"

#define DT_BF16 bfloat16_t
#define ORIG_DTYPE_START DT_BF16
#define __CCE_UT_TEST__

#pragma pack(1)

struct AddRMSNormTilingData {
    uint32_t num_row;
    uint32_t num_col;
    uint32_t block_factor;
    uint32_t row_factor;
    uint32_t ub_factor;
    float epsilon;
    float avg_factor;
    uint32_t num_col_align;
    uint32_t last_block_factor;
    uint32_t row_loop;
    uint32_t last_block_row_loop;
    uint32_t row_tail;
    uint32_t last_block_row_tail;
    uint32_t mul_loop_fp32;
    uint32_t mul_tail_fp32;
    uint32_t dst_rep_stride_fp32;
    uint32_t mul_loop_fp16;
    uint32_t mul_tail_fp16;
    uint32_t dst_rep_stride_fp16;
    uint32_t is_performance;
};

#pragma pack()

#define CONVERT_TILING_DATA(tilingStruct, tilingDataPointer, tilingPointer) \
    __ubuf__ tilingStruct* tilingDataPointer =                              \
        reinterpret_cast<__ubuf__ tilingStruct*>((__ubuf__ uint8_t*)(tilingPointer));

#define INIT_TILING_DATA(tilingStruct, tilingDataPointer, tilingPointer) \
    CONVERT_TILING_DATA(tilingStruct, tilingDataPointer, tilingPointer);

#define GET_TILING_DATA(tilingData, tilingPointer)                            \
    AddRMSNormTilingData tilingData;                                          \
    INIT_TILING_DATA(AddRMSNormTilingData, tilingDataPointer, tilingPointer); \
    (tilingData).num_row = tilingDataPointer->num_row;                        \
    (tilingData).num_col = tilingDataPointer->num_col;                        \
    (tilingData).block_factor = tilingDataPointer->block_factor;              \
    (tilingData).row_factor = tilingDataPointer->row_factor;                  \
    (tilingData).ub_factor = tilingDataPointer->ub_factor;                    \
    (tilingData).epsilon = tilingDataPointer->epsilon;                        \
    (tilingData).avg_factor = tilingDataPointer->avg_factor;                  \
    (tilingData).num_col_align = tilingDataPointer->num_col_align;            \
    (tilingData).last_block_factor = tilingDataPointer->last_block_factor;    \
    (tilingData).row_loop = tilingDataPointer->row_loop;                      \
    (tilingData).last_block_row_loop = tilingDataPointer->last_block_row_loop;            \
    (tilingData).row_tail = tilingDataPointer->row_tail;                                  \
    (tilingData).last_block_row_tail = tilingDataPointer->last_block_row_tail;            \
    (tilingData).mul_loop_fp32 = tilingDataPointer->mul_loop_fp32;                        \
    (tilingData).mul_tail_fp32 = tilingDataPointer->mul_tail_fp32;                        \
    (tilingData).dst_rep_stride_fp32 = tilingDataPointer->dst_rep_stride_fp32;            \
    (tilingData).mul_loop_fp16 = tilingDataPointer->mul_loop_fp16;                        \
    (tilingData).mul_tail_fp16 = tilingDataPointer->mul_tail_fp16;                        \
    (tilingData).dst_rep_stride_fp16 = tilingDataPointer->dst_rep_stride_fp16;            \
    (tilingData).is_performance = tilingDataPointer->is_performance;
#endif
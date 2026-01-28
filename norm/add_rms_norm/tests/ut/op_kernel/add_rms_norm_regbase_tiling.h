/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef ADD_RMS_NORM_REGBASE_TILING_H_
#define ADD_RMS_NORM_REGBASE_TILING_H_

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

struct AddRMSNormRegbaseTilingData {
    uint32_t numRow;
    uint32_t numCol;
    uint32_t numColAlign;
    uint32_t blockFactor;
    uint32_t rowFactor;
    uint32_t ubFactor;
    float epsilon;
    float avgFactor;
    uint32_t ubLoop;
    uint32_t colBuferLength;
    uint32_t multiNNum;
    uint32_t isNddma;
};

struct AddRMSNormRegbaseRFullLoadTilingData {
    uint64_t numRow;
    uint64_t numCol;
    uint64_t numColAlign;
    uint64_t blockFactor;
    uint64_t rowFactor;
    uint64_t binAddQuotient;
    float epsilon;
    float avgFactor;
};
#pragma pack()

template <typename T>
inline void InitTilingData(uint8_t* tiling, T* const_data)
{
    memcpy(const_data, tiling, sizeof(T));
};

#define GET_TILING_DATA_WITH_STRUCT(tiling_struct, tiling_data, tiling_arg) \
    tiling_struct tiling_data;                                              \
    InitTilingData<tiling_struct>(tiling_arg, &tiling_data)

#endif

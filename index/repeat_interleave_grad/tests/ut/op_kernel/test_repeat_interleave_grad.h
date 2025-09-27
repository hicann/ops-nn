/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file repeat_interleave_grad_proto.h
 * \brief
 */

#ifndef _REPEAT_INTERLEAVE_GRAD_TILING_H_
#define _REPEAT_INTERLEAVE_GRAD_TILING_H_

#include "kernel_tiling/kernel_tiling.h"

#define DT_BF16 bfloat16_t
#define ORIG_DTYPE_START DT_BF16
#define __CCE_UT_TEST__

#pragma pack(1)

struct RepeatInterleaveGradTilingDataDef {
    int64_t batch_dim_num;
    int64_t repeats_i_grad_dim_num;
    int64_t repeats_o_grad_dim_num;
    int64_t data_dim_num;

    int64_t core_num;
    int64_t batch_dim_num_each_core;
    int64_t batch_dim_num_last_core;
    int64_t core_num_each_batch;
    int64_t element_num_each_core;
    int64_t element_num_last_core;

    int64_t element_num_each_loop;
};

#pragma pack()

#define CONVERT_TILING_DATA(tilingStruct, tilingDataPointer, tilingPointer) \
    __ubuf__ tilingStruct* tilingDataPointer =                              \
        reinterpret_cast<__ubuf__ tilingStruct*>((__ubuf__ uint8_t*)(tilingPointer));

#define INIT_TILING_DATA(tilingStruct, tilingDataPointer, tilingPointer) \
    CONVERT_TILING_DATA(tilingStruct, tilingDataPointer, tilingPointer);

#define GET_TILING_DATA(tilingData, tilingPointer)                                      \
    RepeatInterleaveGradTilingDataDef tilingData;                                          \
    INIT_TILING_DATA(RepeatInterleaveGradTilingDataDef, tilingDataPointer, tilingPointer); \
    (tilingData).batch_dim_num = tilingDataPointer->batch_dim_num;                      \
    (tilingData).repeats_i_grad_dim_num = tilingDataPointer->repeats_i_grad_dim_num;    \
    (tilingData).repeats_o_grad_dim_num = tilingDataPointer->repeats_o_grad_dim_num;    \
    (tilingData).data_dim_num = tilingDataPointer->data_dim_num;                        \
    (tilingData).core_num = tilingDataPointer->core_num;                                \
    (tilingData).batch_dim_num_each_core = tilingDataPointer->batch_dim_num_each_core;  \
    (tilingData).batch_dim_num_last_core = tilingDataPointer->batch_dim_num_last_core;  \
    (tilingData).core_num_each_batch = tilingDataPointer->core_num_each_batch;          \
    (tilingData).element_num_each_core = tilingDataPointer->element_num_each_core;      \
    (tilingData).element_num_last_core = tilingDataPointer->element_num_last_core;      \
    (tilingData).element_num_each_loop = tilingDataPointer->element_num_each_loop;
#endif
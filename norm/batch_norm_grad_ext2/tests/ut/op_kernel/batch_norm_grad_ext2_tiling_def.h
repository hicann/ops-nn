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
 * \file batch_norm_grad_ext2_v3_tiling_def.h
 * \brief
 */

#ifndef __BATCH_NORM_GRAD_EXT2_V3_TILING_H__
#define __BATCH_NORM_GRAD_EXT2_V3_TILING_H__

#include <cstdint>
#include <cstring>
#include "kernel_tiling/kernel_tiling.h"

#define DTYPE_DY float_t
#define DTYPE_WEIGHT float_t
#define DTYPE_RUNNING_VAR float_t
#define DTYPE_SAVE_MEAN float_t
#define __CCE_UT_TEST__

#include "../../../op_kernel/arch35/batch_norm_grad_ext2_tiling_data.h"

template <typename T>
inline void InitTilingData(uint8_t* tiling, T* const_data)
{
    memcpy(const_data, tiling, sizeof(T));
};

#define GET_TILING_DATA_WITH_STRUCT(tiling_struct, tiling_data, tiling_arg) \
    tiling_struct tiling_data;                                              \
    InitTilingData<tiling_struct>(tiling_arg, &tiling_data)

#endif

/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*
 * The op-kernel UT build force-includes this file before the generated A5
 * kernel source. Reuse the production tiling definition so the UT and the
 * kernel always agree on field names and layout.
 */
#ifndef __RENORM_TILING_H__
#define __RENORM_TILING_H__

#include <cstdint>
#include <cstring>

#include "../../../op_kernel/arch35/renorm_tiling_data.h"

template <typename T>
inline void InitTilingData(const uint8_t* tiling, T* const_data)
{
    std::memcpy(const_data, tiling, sizeof(T));
}

#ifndef GET_TILING_DATA_WITH_STRUCT
#define GET_TILING_DATA_WITH_STRUCT(tiling_struct, tiling_data, tiling_arg) \
    tiling_struct tiling_data;                                              \
    InitTilingData<tiling_struct>(tiling_arg, &tiling_data)
#endif

#endif

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
 * \file max_pool_with_argmax_tiling.h
 * \brief
 */

#ifndef _GE_MAX_POOL_WITH_ARGMAX_TILING_DEF_H_
#define _GE_MAX_POOL_WITH_ARGMAX_TILING_DEF_H_

#include "kernel_tiling/kernel_tiling.h"
#include "securec.h"

#include "../../../op_kernel/arch35/max_pool_with_argmax_struct_common.h"

inline void InitTilingData(uint8_t* tiling, MaxPoolWithArgmaxCommonStructNameSpace::MaxPoolWithArgmaxNHWCTilingCommonData* constData)
{
    memcpy_s(
        constData, sizeof(MaxPoolWithArgmaxCommonStructNameSpace::MaxPoolWithArgmaxNHWCTilingCommonData), tiling,
        sizeof(MaxPoolWithArgmaxCommonStructNameSpace::MaxPoolWithArgmaxNHWCTilingCommonData));
}

inline void InitTilingData(uint8_t* tiling, MaxPoolWithArgmaxCommonStructNameSpace::MaxPoolWithArgmaxSimtTilingCommonData* constData)
{
    memcpy_s(
        constData, sizeof(MaxPoolWithArgmaxCommonStructNameSpace::MaxPoolWithArgmaxSimtTilingCommonData), tiling,
        sizeof(MaxPoolWithArgmaxCommonStructNameSpace::MaxPoolWithArgmaxSimtTilingCommonData));
}

#define GET_TILING_DATA_WITH_STRUCT(tiling_struct, tiling_data, tiling_arg) \
    do {                                                                    \
        tiling_struct tiling_data;                                          \
        InitTilingData(tiling_arg, &tiling_data)                            \
    } while (0)

#endif // MAX_POOL_WITH_ARGMAX_TILING_DEF_H_
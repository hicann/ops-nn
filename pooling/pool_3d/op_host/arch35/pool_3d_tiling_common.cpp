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
 * \file pool_3d_tiling_common.cpp
 * \brief
 */
 
#include "pool_3d_tiling_common.h"

namespace optiling
{
    
// Integer division rounding to -Infinity
int64_t DivRtn(const int64_t x, const int64_t y)
{
    if (y == 0) {
        return 0;
    }
    int64_t q = x / y;
    int64_t r = x % y;
    if ((r != 0) && ((r < 0) != (y < 0))) {
        --q;
    };
    return q;
}

}  // namespace optiling
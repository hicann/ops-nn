/*
 * Copyright (c) 2026 联通（广东）产业互联网有限公司.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#ifndef MATMUL_ADD_TILING_DEF_H_
#define MATMUL_ADD_TILING_DEF_H_

#include "kernel_tiling/kernel_tiling.h"
#include <algorithm>
#include <cstdint>
#include "securec.h"

#define __CCE_UT_TEST__
#define __aicore__

using std::min;

#include "../../../op_kernel/matmul_add_tiling.h"

inline void IMatmulAddTilingData(
    uint8_t* tiling,
    optiling::MatmulAddTilingData* constData)
{
    auto ret = memcpy_s(constData, sizeof(optiling::MatmulAddTilingData),
        tiling, sizeof(optiling::MatmulAddTilingData));
    if (ret != EOK) {
        return;
    }
}

#define GET_TILING_DATA(tilingData, tilingPointer) \
    optiling::MatmulAddTilingData tilingData; \
    IMatmulAddTilingData(tilingPointer, &tilingData)

#endif

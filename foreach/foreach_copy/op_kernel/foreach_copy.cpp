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
 * \file foreach_copy.cpp
 * \brief
 */

#include "foreach_copy.h"
#include "foreach_cast.h"

using namespace ForeachCopy;
using namespace ForeachCast;

#define INIT_AND_PROCESS                \
    op.Init(x, y, userWS, &tilingData); \
    op.Process()

#define FOREACH_COPY_OP(T) \
    ForeachCopyND<T> op; \
    op.Init(x, y, userWS, &tilingData); \
    op.Process()

extern "C" __global__ __aicore__ void foreach_copy(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tilingData, tiling);

    // foreach(vector) not need workspace
    GM_ADDR userWS = nullptr;
    if (TILING_KEY_IS(1)) {
        FOREACH_COPY_OP(half);
    } else if (TILING_KEY_IS(2)) {
        FOREACH_COPY_OP(float);
    } else if (TILING_KEY_IS(3)) {
        FOREACH_COPY_OP(int);
#if !(defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3003 || __NPU_ARCH__ == 3113))
    } else if (TILING_KEY_IS(4)) {
        FOREACH_COPY_OP(bfloat16_t);
#endif
    } else if (TILING_KEY_IS(5)) {
        FOREACH_COPY_OP(int16_t);
    } else if (TILING_KEY_IS(6)) {
        FOREACH_COPY_OP(uint16_t);
    } else if (TILING_KEY_IS(7)) {
        FOREACH_COPY_OP(int8_t);
    } else if (TILING_KEY_IS(8)) {
        FOREACH_COPY_OP(uint8_t);
    } else if (TILING_KEY_IS(9)) {
        FOREACH_COPY_OP(uint32_t);
    } else if (TILING_KEY_IS(10)) {
        FOREACH_COPY_OP(uint64_t);
    } else if (TILING_KEY_IS(11)) {
        FOREACH_COPY_OP(double);
    } else if (TILING_KEY_IS(12)) {
        FOREACH_COPY_OP(bool);
    } else if (TILING_KEY_IS(13)) {
        ForeachCastND<float, half> op;
        INIT_AND_PROCESS;
    } else if (TILING_KEY_IS(14)) {
        ForeachCastND<half, float> op;
        INIT_AND_PROCESS;
#if !(defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3003 || __NPU_ARCH__ == 3113))
    } else if (TILING_KEY_IS(15)) {
        ForeachCastND<float, bfloat16_t> op;
        INIT_AND_PROCESS;
    } else if (TILING_KEY_IS(16)) {
        ForeachCastND<bfloat16_t, float> op;
        INIT_AND_PROCESS;
#endif
    }
}

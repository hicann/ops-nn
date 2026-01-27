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
 * \file hashtable_common.h
 * \brief
 */

#pragma once

#include "kernel_operator.h"

namespace Hashtbl {

__aicore__ inline uint32_t MurmurHash3(__gm__ int64_t* key, int len, uint32_t seed)
{
    const uint32_t c1 = 0xcc9e2d51;
    const uint32_t c2 = 0x1b873593;
    const int r1 = 15;
    const int r2 = 13;
    const int m = 5;
    const uint32_t n = 0xe6546b64;
    uint32_t hash = seed;
    const int nblocks = len / 4;
    __gm__ int32_t* blocks = (__gm__ int32_t*)key;

    for (uint16_t i = 0; i < nblocks; ++i) {
        uint32_t k = blocks[i];
        k *= c1;
        k = (k << r1) | (k >> (32 - r1));
        k *= c2;
        hash ^= k;
        hash = ((hash << r2) | (hash >> (32 - r2))) * m + n;
    }
    hash ^= len;
    hash ^= hash >> 16;
    hash *= 0x85ebca6b;
    hash ^= hash >> 13;
    hash *= 0xc2b2ae35;
    hash ^= hash >> 16;
    return hash;
}

template <typename T>
__aicore__ inline T RoundUpTo8(T x)
{
    constexpr T ROUND_SIZE = static_cast<T>(8);
    if (x % ROUND_SIZE != 0) {
        return (x / ROUND_SIZE + 1) * ROUND_SIZE;
    }
    return x;
}

} // namespace Hashtbl
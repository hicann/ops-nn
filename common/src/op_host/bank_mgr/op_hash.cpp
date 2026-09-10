/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "op_hash.h"
#include "kb_log.h"
#include "kb_common.h"
namespace RuntimeKb {
uint32_t CommonHash(const void* src, uint32_t len, uint32_t seed)
{
    RTKB_CHECK(src == nullptr, CANNKB_LOGE("Src is nullptr."), return seed);
    const uint32_t* ky = static_cast<const uint32_t*>(src);
    uint32_t hash_ky = seed;
    uint32_t tmp_ky = 0U;
    for (uint32_t i = len >> 2; i > 0; i--) {
        tmp_ky = *ky;
        ky++;
        HashCombine(tmp_ky, hash_ky);
    }
    const uint8_t* rest_ky = static_cast<const uint8_t*>(src);
    for (uint32_t i = len & 3; i; i--) {
        tmp_ky <<= kEightBitShift;
        tmp_ky |= rest_ky[i - 1];
    }
    HashCombine(tmp_ky, hash_ky);

    hash_ky ^= len;
    hash_ky ^= hash_ky >> kSixtennBitShift;
    return hash_ky;
}
} // namespace RuntimeKb

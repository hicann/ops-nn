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
 * \file masked_scatter_v2_tiling_key.h
 * \brief tiling key for masked_scatter_v2
 */
#ifndef __MASKED_SCATTER_V2_TILING_KEY_H__
#define __MASKED_SCATTER_V2_TILING_KEY_H__

namespace NsMaskedScatterV2 {
// bit0: mask broadcast (1 = native broadcast read enabled)
constexpr uint64_t TILING_KEY_MASK_BCAST = 1UL;
// bit1: multi-core soft-sync enabled
constexpr uint64_t TILING_KEY_USE_SYNC = 1UL << 1;
} // namespace NsMaskedScatterV2

#endif // __MASKED_SCATTER_V2_TILING_KEY_H__

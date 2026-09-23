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
 * \file turbo_quant_tiling.h
 * \brief
 */
#ifndef TURBO_QUANT_TILING_H
#define TURBO_QUANT_TILING_H

#include <cstdint>
#include "register/tilingdata_base.h"
#include "register/op_impl_registry.h"
#include "log/log.h"
#include "error_util.h"
#include "tiling/platform/platform_ascendc.h"

namespace optiling {

// TurboQuant 4-bit codebook: one nibble per element, so exactly 16 centroids.
constexpr int64_t TQ_N_CENT = 16;
constexpr int64_t TQ_ELEMENTS_PER_BYTE = 2;
// UB buffers are aligned to this size for efficient vector access.
constexpr int64_t TQ_SLOT_ALIGN = 64;
// The current production path uses the MLA kv_lora_rank of 512.
constexpr int64_t TQ_SUPPORTED_HEAD_DIM = 512;

// The codebook scan is a chain of narrow vector instructions whose fixed issue cost dominates, so the
// kernel processes several tokens per instruction. Measured on Atlas A2: a 4x wider loop costs only
// 1.82x the time, i.e. 2.2x better efficiency; the gain flattens out past 8 and the buffers grow
// linearly, so the batch is capped here. Must stay <= TQ_MAX_TOKENS_PER_BATCH in the kernel.
constexpr int64_t TQ_MAX_TOKENS_PER_BATCH = 12;
// Each token's L2 norm is reduced into its own 64B-aligned slot so one V->S sync covers the whole batch.
constexpr int64_t TQ_NORM_SLOT_FLOATS = 16;
constexpr int64_t TQ_UB_RESERVE = 1024;

inline int64_t TqAlign64(int64_t value) { return (value + TQ_SLOT_ALIGN - 1) / TQ_SLOT_ALIGN * TQ_SLOT_ALIGN; }

// UB held per token in a batch: inQ + u + nib + tmp + sel + one, plus y, the half staging
// buffer and the compare mask.
inline int64_t TqBytesPerToken(int64_t headDim, int64_t packedBytes)
{
    return TqAlign64(headDim * static_cast<int64_t>(sizeof(float))) * 6 + TqAlign64(packedBytes) +
           TqAlign64(headDim * 2) + TqAlign64(headDim / 8);
}

// UB that does not scale with the batch: the ReduceSum work area, the codebook and the norm slots.
inline int64_t TqFixedBytes(int64_t headDim)
{
    return TqAlign64(headDim * static_cast<int64_t>(sizeof(float))) +
           TqAlign64(TQ_N_CENT * static_cast<int64_t>(sizeof(float))) +
           TqAlign64(TQ_MAX_TOKENS_PER_BATCH * TQ_NORM_SLOT_FLOATS * static_cast<int64_t>(sizeof(float)));
}

BEGIN_TILING_DATA_DEF(TurboQuantTilingData)
TILING_DATA_FIELD_DEF(uint32_t, numTokens);
TILING_DATA_FIELD_DEF(uint32_t, tokensPerCore);
TILING_DATA_FIELD_DEF(uint32_t, headDim);
TILING_DATA_FIELD_DEF(uint32_t, packedBytes);
TILING_DATA_FIELD_DEF(uint32_t, tokensPerBatch);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(TurboQuant, TurboQuantTilingData)

struct TurboQuantCompileInfo {};

} // namespace optiling
#endif // TURBO_QUANT_TILING_H

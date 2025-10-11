/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file kv_rms_norm_rope_cache_apt.cpp
 * \brief
 */

#include "arch35/kv_rms_norm_rope_cache_regbase_full_load.h"

using namespace KvRmsNormRopeCache;
extern "C" __global__ __aicore__ void kv_rms_norm_rope_cache(
    GM_ADDR kv, GM_ADDR gamma, GM_ADDR cos, GM_ADDR sin, GM_ADDR index, GM_ADDR k_cache, GM_ADDR ckv_cache,
    GM_ADDR k_rope_scale, GM_ADDR c_kv_scale, GM_ADDR k_rope_offset, GM_ADDR c_kv_offset, GM_ADDR k_cache_out,
    GM_ADDR c_kv_cache_out, GM_ADDR k_rope, GM_ADDR c_kv, GM_ADDR workspace, GM_ADDR tiling)
{
    TPipe pipe;
    if (TILING_KEY_IS(10000)) {
        GET_TILING_DATA_WITH_STRUCT(KvRmsNormRopeCacheRegbaseFullLoadTilingData, tiling_data_in, tiling);
        const KvRmsNormRopeCacheRegbaseFullLoadTilingData* __restrict tilingData = &tiling_data_in;
        KvRmsNormRopeCacheRegbaseFullLoad<DTYPE_KV, DTYPE_K_CACHE, DTYPE_CKV_CACHE> op(&pipe, tilingData);
        op.Init(
            kv, gamma, cos, sin, index, k_cache, ckv_cache, k_rope_scale, c_kv_scale, k_rope_offset, c_kv_offset,
            k_rope, c_kv);
        op.Process();
    }
}
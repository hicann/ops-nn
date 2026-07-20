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
 * ile weight_quant_batch_matmul_v2_kernel_util.h
 * rief Common helpers for weight_quant_batch_matmul_v2 kernel templates.
 */

#pragma once

#include "weight_quant_batch_matmul_v2_arch35_tiling_data.h"
#include "../weight_quant_batch_matmul_v2_constant.h"
#include "../tool.h"

namespace WeightQuantBatchMatmulV2 {

template <typename xType, typename wType, typename biasType, typename yType>
__aicore__ inline void SetL2CacheHintHelper(wqbmmv2_tiling::L2CacheMode l2CacheDisable,
                                            AscendC::GlobalTensor<xType>& aGlobal,
                                            AscendC::GlobalTensor<wType>& bGlobal)
{
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 5102)
    if (l2CacheDisable == wqbmmv2_tiling::L2CacheMode::ALL_L2_CACHE_DISABLE ||
        l2CacheDisable == wqbmmv2_tiling::L2CacheMode::A_L2_CACHE_DISABLE) {
        aGlobal.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
    }
    if (l2CacheDisable == wqbmmv2_tiling::L2CacheMode::ALL_L2_CACHE_DISABLE ||
        l2CacheDisable == wqbmmv2_tiling::L2CacheMode::B_L2_CACHE_DISABLE) {
        bGlobal.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
    }
#endif
}

template <typename xType, typename wType, typename biasType, typename yType, QuantType antiQuantType, typename BlockT>
__aicore__ inline void UpdateGlobalAddrHelper(
    GM_ADDR x, GM_ADDR weight, GM_ADDR antiquantScale, GM_ADDR antiquantOffset, GM_ADDR quantScale, GM_ADDR quantOffset,
    GM_ADDR bias, GM_ADDR y, GM_ADDR workspace, BlockT& block,
    const wqbmmv2_tiling::WeightQuantBatchMatmulV2ASWTilingDataParams* tiling, uint32_t blockIdx,
    AscendC::GlobalTensor<xType>& aGlobal, AscendC::GlobalTensor<wType>& bGlobal, AscendC::GlobalTensor<yType>& cGlobal,
    AscendC::GlobalTensor<biasType>& biasGlobal, AscendC::GlobalTensor<uint64_t>& scaleGlobal,
    wqbmmv2_tiling::L2CacheMode l2CacheDisable)
{
    block.Init(tiling, blockIdx);

    if constexpr (antiQuantType == QuantType::PER_TENSOR) { // pertensor
        block.offset_.scaleScalar = *((__gm__ uint64_t*)antiquantScale);
    } else {
        scaleGlobal.SetGlobalBuffer((__gm__ uint64_t*)antiquantScale);
    }

    // update global buffer
    aGlobal.SetGlobalBuffer((__gm__ xType*)x);
    bGlobal.SetGlobalBuffer((__gm__ wType*)weight);
    SetL2CacheHintHelper<xType, wType, biasType, yType>(l2CacheDisable, aGlobal, bGlobal);
    cGlobal.SetGlobalBuffer((__gm__ yType*)y);
    if (static_cast<bool>(tiling->matmulTiling.isBias)) {
        biasGlobal.SetGlobalBuffer((__gm__ biasType*)bias);
    }
}

} // namespace WeightQuantBatchMatmulV2

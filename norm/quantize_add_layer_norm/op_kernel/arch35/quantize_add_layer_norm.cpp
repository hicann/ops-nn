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
 * \file quantize_add_layer_norm.cpp
 * \brief ascend950 (arch35/regbase) kernel entry for QuantizeAddLayerNorm.
 *        Tiling key layout: 8000(prefix) + 100(welford) + bias(1 elewise / 2 brc)
 *                           + quantmode(0 mul / 10 per_channel div / 20 per_tensor scalar mul)
 */
#include "quantize_add_layer_norm_static_quant_regbase_full_load_kernel.h"
#include "quantize_add_layer_norm_static_quant_regbase_welford_kernel.h"
#include "quantize_add_layer_norm_tiling_key.h"

#define TILING_REGBASE_PREFIX 8000
#define TILING_WELFORD_OFFSET 100
#define TILING_BIAS_ELEWISE_OFFSET 1
#define TILING_BIAS_BROADCAST_OFFSET 2
#define TILING_QUANT_MODE_STRIDE 10

#define KERNEL_BUFFER_NUM 1

// Scales is a required input; DTYPE_SCALES is always injected by the build script.
#ifdef DTYPE_SCALES
#else
#define DTYPE_SCALES DTYPE_X1
#endif

#define RUN_STATIC_QUANT_KERNEL(tilingKeyNum, KernelClass, optStatus)                                  \
    do {                                                                                               \
        KernelClass<DTYPE_X1, DTYPE_SCALES, tilingKeyNum, optStatus, KERNEL_BUFFER_NUM> qalnOp(&pipe); \
        qalnOp.Init(x1, x2, gamma, beta, bias, scales, zeroPoints, y, x, usrWorkspace, tilingData);    \
        qalnOp.Process();                                                                              \
    } while (0)

#define CREATE_STATIC_QUANT_KERNEL(tilingKeyNum, KernelClass)                \
    do {                                                                     \
        if (zeroPoints == nullptr) {                                         \
            RUN_STATIC_QUANT_KERNEL(tilingKeyNum, KernelClass, 0);           \
        } else {                                                             \
            RUN_STATIC_QUANT_KERNEL(tilingKeyNum, KernelClass, OFFSET_CODE); \
        }                                                                    \
    } while (0)

template <bool isWelford, bool isBiasBroadcast, uint32_t quantMode, bool isMembase, bool additionalOutput>
__global__ __aicore__ void quantize_add_layer_norm(GM_ADDR x1, GM_ADDR x2, GM_ADDR gamma, GM_ADDR beta, GM_ADDR bias,
                                                   GM_ADDR scales, GM_ADDR zeroPoints, GM_ADDR y, GM_ADDR x,
                                                   GM_ADDR workspace, GM_ADDR tiling)
{
    static_assert(!isMembase && !additionalOutput, "arch35 template selection must use the RegBase key space");
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIV_1_0);
    REGISTER_TILING_DEFAULT(QuantizeAddLayerNormRegbaseTilingData);
    AscendC::TPipe pipe;
    GM_ADDR usrWorkspace = AscendC::GetUserWorkspace(workspace);
    GET_TILING_DATA_WITH_STRUCT(QuantizeAddLayerNormRegbaseTilingData, tilingDataIn, tiling);
    const QuantizeAddLayerNormRegbaseTilingData* __restrict tilingData = &tilingDataIn;

    constexpr int32_t legacyTilingKey = TILING_REGBASE_PREFIX + (isWelford ? TILING_WELFORD_OFFSET : 0) +
                                        (isBiasBroadcast ? TILING_BIAS_BROADCAST_OFFSET : TILING_BIAS_ELEWISE_OFFSET) +
                                        static_cast<int32_t>(quantMode) * TILING_QUANT_MODE_STRIDE;
    if constexpr (isWelford) {
        CREATE_STATIC_QUANT_KERNEL(legacyTilingKey,
                                   NsQuantizeAddLayerNorm::KernelQuantizeAddLayerNormStaticQuantRegbaseWelford);
    } else {
        CREATE_STATIC_QUANT_KERNEL(legacyTilingKey,
                                   NsQuantizeAddLayerNorm::KernelQuantizeAddLayerNormStaticQuantRegbaseFullLoad);
    }
}

/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)

#include "kernel_operator.h"
#include "matmul_emu_split_weight_tiling_key.h"
#include "../matmul_emu_split_weight_tiling_data.h"
#include "blaze/gemm/kernel/kernel_matmul_emu_split_weight.h"

using namespace AscendC;

using ProblemShape = AscendC::Te::Shape<int64_t, int64_t, int64_t, int64_t>;

#define MATMUL_EMU_SPLIT_WEIGHT_DISPATCH(layoutA, layoutB)                                                            \
    do {                                                                                                              \
        using BlockMmad = Blaze::Gemm::Block::BlockMmad<Blaze::Gemm::MatmulEmuSplitWeightPolicy, bfloat16_t, layoutA, \
                                                        bfloat16_t, layoutB, float, void, void, void>;                \
        using BlockEpilogue = Blaze::Gemm::Block::BlockEpilogueMulsAdd;                                               \
        using Kernel = Blaze::Gemm::Kernel::GemmUniversal<ProblemShape, BlockMmad, BlockEpilogue, void>;              \
        typename BlockMmad::Params mmadParams{x,                                                                      \
                                              wHigh,                                                                  \
                                              wLow,                                                                   \
                                              static_cast<uint64_t>(tilingData.k),                                    \
                                              static_cast<uint64_t>(tilingData.kL1),                                  \
                                              tilingData.baseM,                                                       \
                                              tilingData.baseN,                                                       \
                                              tilingData.baseK,                                                       \
                                              tilingData.usedCoreNum};                                                \
        typename BlockEpilogue::Params epilogueParams{y, tilingData.scale};                                           \
        typename Kernel::Params params{{tilingData.m, tilingData.n, tilingData.k, 1L}, mmadParams, epilogueParams};   \
        Kernel kernel;                                                                                                \
        kernel(params);                                                                                               \
    } while (0)

template <int8_t API_LEVEL, int8_t A_TRANS, int8_t B_TRANS>
__global__ __aicore__ void matmul_emu_split_weight(GM_ADDR x, GM_ADDR wHigh, GM_ADDR wLow, GM_ADDR y, GM_ADDR workspace,
                                                   GM_ADDR tilingGm)
{
    REGISTER_TILING_DEFAULT(MatmulEmuSplitWeightTilingData);
    GET_TILING_DATA_WITH_STRUCT(MatmulEmuSplitWeightTilingData, tilingData, tilingGm);

    constexpr bool aTran = (A_TRANS == MATMUL_EMU_SPLIT_WEIGHT_TRANS);
    constexpr bool bTran = (B_TRANS == MATMUL_EMU_SPLIT_WEIGHT_TRANS);

    using layoutA = AscendC::Std::conditional_t<aTran, AscendC::Te::DNExtLayoutPtn, AscendC::Te::NDExtLayoutPtn>;
    using layoutB = AscendC::Std::conditional_t<bTran, AscendC::Te::DNExtLayoutPtn, AscendC::Te::NDExtLayoutPtn>;
    MATMUL_EMU_SPLIT_WEIGHT_DISPATCH(layoutA, layoutB);
}

#endif

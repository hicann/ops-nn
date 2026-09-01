/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#if (defined(__CCE_AICORE__) && (__CCE_AICORE__ == 220)) || (defined(__NPU_ARCH__) && (__NPU_ARCH__ == 2201))

#ifndef CATLASS_ARCH
#define CATLASS_ARCH 2201
#endif

#include "kernel_operator.h"

#include "lib/matmul_intf.h"
#include "matmul_emu_split_weight_tiling_key.h"
#include "../matmul_emu_split_weight_tiling_data.h"

#include "./catlass/arch/matmul_emu_arch.hpp"
#include "./catlass/matmul_emu_catlass.hpp"
#include "./catlass/epilogue/block/block_epilogue_decomposition_gm.hpp"
#include "./catlass/gemm/block/matmul_emu_block_mmad.hpp"
#include "./catlass/gemm/block/matmul_emu_block_swizzle.hpp"
#include "./catlass/gemm/matmul_emu_dispatch_policy.hpp"
#include "./catlass/gemm/matmul_emu_gemm_type.hpp"
#include "./catlass/gemm/kernel/matmul_decomposition_gm.hpp"
#include "./catlass/gemm/tile/matmul_emu_tile_copy.hpp"
#include "./catlass/matmul_emu_gemm_coord.hpp"
#include "./catlass/layout/matmul_emu_layout.hpp"
#include "./tla/matmul_emu_layout.hpp"

using namespace Catlass;
using namespace tla;

using ArchTag = Arch::AtlasA2;
using ElementA = bfloat16_t;
using ElementB = bfloat16_t;
using ElementC = float;
using LayoutTagC = layout::RowMajor;

// standard block: L1 <128,256,256>, L0C 128KB constrains one fp32 tile to 128x256.
using L1TileShape = Shape<Int<128>, Int<256>, Int<256>>;
using L0TileShape = Shape<Int<128>, Int<256>, Int<64>>;

template <class LayoutTagA, class LayoutTagB, int SwizzleDir>
struct Assembly {
    static constexpr bool enableUnitFlag = true;
    using DispatchPolicy = Gemm::MmadPingpong<ArchTag, enableUnitFlag>;

    using TileCopy = Gemm::Tile::PackedTileCopyTla<ArchTag, ElementA, LayoutTagA, ElementB, LayoutTagB, ElementC,
                                                   LayoutTagC>;
    using BlockMmad = Gemm::Block::BlockMmadTla<DispatchPolicy, L1TileShape, L0TileShape, ElementA, ElementB, ElementC,
                                                void, TileCopy>;
    using BlockScheduler = Gemm::Block::GemmIdentityBlockSwizzle<3, SwizzleDir>;

    using LayoutC = typename BlockMmad::LayoutC;

    using BlockEpilogue = Epilogue::Block::BlockEpilogueDecompositionGm<ArchTag, Int<128>, Int<256>, ElementC, LayoutC,
                                                                        LayoutTagA, LayoutTagB>;

    using Kernel = Gemm::Kernel::MatmulDecompositionGm<BlockMmad, BlockEpilogue, BlockScheduler, LayoutTagA,
                                                       LayoutTagB>;
};

#define MATMUL_EMU_SPLIT_WEIGHT_DISPATCH(layoutTagA, layoutTagB, swizzleDir)                                  \
    do {                                                                                                      \
        using Assembly_ = Assembly<layoutTagA, layoutTagB, swizzleDir>;                                       \
        using Kernel_ = typename Assembly_::Kernel;                                                           \
        auto layoutA = tla::MakeLayout<bfloat16_t, layoutTagA>(m, k);                                         \
        auto layoutB = tla::MakeLayout<bfloat16_t, layoutTagB>(k, n);                                         \
        auto layoutC = tla::MakeLayout<float, Catlass::layout::RowMajor>(m, n);                               \
        Catlass::GemmCoord problemShape(m, n, k);                                                             \
        typename Assembly_::BlockEpilogue::Params epilogueParams{y, layoutC, scale};                          \
        typename Kernel_::Params params{problemShape, x, layoutA, wHigh, wLow, layoutB, ws0, epilogueParams}; \
        Kernel_ kernel;                                                                                       \
        kernel(params);                                                                                       \
    } while (0)

template <int8_t API_LEVEL, int8_t A_TRANS, int8_t B_TRANS, int8_t SWIZZLE_DIR>
__global__ __aicore__ void matmul_emu_split_weight(GM_ADDR x, GM_ADDR wHigh, GM_ADDR wLow, GM_ADDR y, GM_ADDR workspace,
                                                   GM_ADDR tilingGm)
{
    REGISTER_TILING_DEFAULT(MatmulEmuSplitWeightTilingData);
    GET_TILING_DATA_WITH_STRUCT(MatmulEmuSplitWeightTilingData, tilingData, tilingGm);

    constexpr bool aTran = (A_TRANS == MATMUL_EMU_SPLIT_WEIGHT_TRANS);
    constexpr bool bTran = (B_TRANS == MATMUL_EMU_SPLIT_WEIGHT_TRANS);
    constexpr int swizzleDir = SWIZZLE_DIR;

    uint32_t m = tilingData.m;
    uint32_t n = tilingData.n;
    uint32_t k = tilingData.k;
    float scale = tilingData.scale;
    // GM staging workspace: [Y0 | Y1], each M*N fp32 (typed by MatmulDecompositionGm::Params).
    GM_ADDR ws0 = workspace;

    using layoutTagA = std::conditional_t<aTran, Catlass::layout::ColumnMajor, Catlass::layout::RowMajor>;
    using layoutTagB = std::conditional_t<bTran, Catlass::layout::ColumnMajor, Catlass::layout::RowMajor>;
    MATMUL_EMU_SPLIT_WEIGHT_DISPATCH(layoutTagA, layoutTagB, swizzleDir);
}

#endif

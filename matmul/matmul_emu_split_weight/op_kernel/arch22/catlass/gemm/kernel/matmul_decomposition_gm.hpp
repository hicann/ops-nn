/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// ============================================================================
// MatmulDecompositionGm — Atlas A2 fused Decomposition Matmul kernel.
//
//   Single BlockScheduler loop, per output tile:
//     [AIC] two blockMmad calls (X@W_high -> GM Y0, X@W_low -> GM Y1, L0C->GM),
//           explicit FIX_M event guard between the two calls (L0C[0] WAR).
//     [AIV] blockEpilogue split into LoadY0 / LoadY1 / FuseAndStore phases:
//           GM->UB read Y0/Y1, Y = Y0 + scale*Y1, UB->GM writeback.
//
//   CV overlap: A2/A3 has no L0C->UB path, so every mmad result is staged through GM.
//   The AIC publishes two separate flags — flagY0Ready (right after mmad0 + FIX_M guard)
//   and flagY1Ready (after mmad1). The AIV waits flagY0Ready and immediately issues the
//   Y0 GM->UB copy (MTE2), which overlaps the AIC's second mmad (Y1); it then waits
//   flagY1Ready before loading Y1. Single-direction handshake, same as
//   BasicMatmulTlaVisitor (AIC Set PIPE_FIX -> AIV Wait PIPE_MTE2).
//   Workspace = 2 * M * N * sizeof(float) (Y0 | Y1 GM staging).
// ============================================================================

#ifndef CATLASS_GEMM_KERNEL_MATMUL_DECOMPOSITION_GM_HPP
#define CATLASS_GEMM_KERNEL_MATMUL_DECOMPOSITION_GM_HPP

#include "../../matmul_emu_catlass.hpp"
#include "../../arch/matmul_emu_arch.hpp"
#include "../block/matmul_emu_block_swizzle.hpp"
#include "../tile/matmul_emu_tile_copy.hpp"
#include "../../matmul_emu_gemm_coord.hpp"
#include "../../layout/matmul_emu_layout.hpp"
#include "../../../tla/matmul_emu_layout.hpp"

namespace Catlass::Gemm::Kernel {

namespace detail {

template <class T, class = void>
struct HasSetSwizzleParams : std::false_type {};

template <class T>
struct HasSetSwizzleParams<T, std::void_t<decltype(std::declval<T>().SetSwizzleParams(0u, 0u))>> : std::true_type {};

} // namespace detail

template <class BlockMmad_, class BlockEpilogue_, class BlockScheduler_, class LayoutTagA_ = void,
          class LayoutTagB_ = void>
class MatmulDecompositionGm {
public:
    using BlockMmad = BlockMmad_;
    using ArchTag = typename BlockMmad::ArchTag;
    using L1TileShape = typename BlockMmad::L1TileShape;
    using ElementA = typename BlockMmad::ElementA;
    using LayoutA = std::conditional_t<std::is_void_v<LayoutTagA_>, typename BlockMmad::LayoutA,
                                       Catlass::detail::TagToLayout_t<ElementA, LayoutTagA_>>;
    using ElementB = typename BlockMmad::ElementB;
    using LayoutB = std::conditional_t<std::is_void_v<LayoutTagB_>, typename BlockMmad::LayoutB,
                                       Catlass::detail::TagToLayout_t<ElementB, LayoutTagB_>>;
    using ElementC = typename BlockMmad::ElementC;
    using LayoutC = typename BlockMmad::LayoutC;

    using BlockEpilogue = BlockEpilogue_;
    using EpilogueParams = typename BlockEpilogue::Params;
    using BlockScheduler = BlockScheduler_;

    static_assert(std::is_same_v<ArchTag, Catlass::Arch::AtlasA2>, "MatmulDecompositionGm only supports AtlasA2");
    static_assert(BlockEpilogue::USE_GM_WORKSPACE == true,
                  "BlockEpilogue must use GM workspace semantics (USE_GM_WORKSPACE == true)");

    static constexpr uint32_t L1_TILE_M = tla::get<0>(L1TileShape{});
    static constexpr uint32_t L1_TILE_N = tla::get<1>(L1TileShape{});
    static constexpr uint32_t L1_TILE_K = tla::get<2>(L1TileShape{});

    struct Params {
        Catlass::GemmCoord problemShape;
        GM_ADDR ptrX;
        LayoutA layoutA;
        GM_ADDR ptrW_high;
        GM_ADDR ptrW_low;
        LayoutB layoutB;
        GM_ADDR ptrWorkspace;
        EpilogueParams epilogueParams;

        CATLASS_HOST_DEVICE
        Params() {}

        CATLASS_HOST_DEVICE
        Params(Catlass::GemmCoord const& problemShape_, GM_ADDR ptrX_, LayoutA const& layoutA_, GM_ADDR ptrW_high_,
               GM_ADDR ptrW_low_, LayoutB const& layoutB_, GM_ADDR ptrWorkspace_, EpilogueParams const& epilogueParams_)
            : problemShape(problemShape_),
              ptrX(ptrX_),
              layoutA(layoutA_),
              ptrW_high(ptrW_high_),
              ptrW_low(ptrW_low_),
              layoutB(layoutB_),
              ptrWorkspace(ptrWorkspace_),
              epilogueParams(epilogueParams_)
        {}
    };

    static size_t GetWorkspaceSize(Params const& params)
    {
        return 2ul * params.problemShape.m() * params.problemShape.n() * sizeof(float);
    }

    CATLASS_DEVICE
    MatmulDecompositionGm() {}

    CATLASS_DEVICE
    void operator()(Params const& params)
    {
        BlockScheduler scheduler(params.problemShape, Catlass::MakeCoord(L1_TILE_M, L1_TILE_N));
        if constexpr (detail::HasSetSwizzleParams<BlockScheduler>::value) {
            uint32_t swizzleDir = (params.problemShape.m() > params.problemShape.n()) ? 0u : 1u;
            scheduler.SetSwizzleParams(3u, swizzleDir);
        }
        BlockMmad blockMmad(resource);
        BlockEpilogue blockEpilogue(resource, params.epilogueParams);
        RunCoreLoops(params, scheduler, blockMmad, blockEpilogue);
        AscendC::PipeBarrier<PIPE_ALL>();
    }

private:
    template <class Scheduler, class Mmad, class Epilogue>
    CATLASS_DEVICE void RunCoreLoops(Params const& params, Scheduler& scheduler, Mmad& blockMmad,
                                     Epilogue& blockEpilogue)
    {
        uint32_t coreLoops = scheduler.GetCoreLoops();

        AscendC::GlobalTensor<ElementA> gmX;
        gmX.SetGlobalBuffer((__gm__ ElementA*)params.ptrX);
        AscendC::GlobalTensor<ElementB> gmWh;
        gmWh.SetGlobalBuffer((__gm__ ElementB*)params.ptrW_high);
        AscendC::GlobalTensor<ElementB> gmWl;
        gmWl.SetGlobalBuffer((__gm__ ElementB*)params.ptrW_low);

        auto tensorX = tla::MakeTensor(gmX, params.layoutA, Catlass::Arch::PositionGM{});
        auto tensorWh = tla::MakeTensor(gmWh, params.layoutB, Catlass::Arch::PositionGM{});
        auto tensorWl = tla::MakeTensor(gmWl, params.layoutB, Catlass::Arch::PositionGM{});

        uint32_t m = params.problemShape.m();
        uint32_t n = params.problemShape.n();
        uint32_t mnElems = m * n;

        AscendC::GlobalTensor<ElementC> gmWs0;
        gmWs0.SetGlobalBuffer((__gm__ ElementC*)params.ptrWorkspace);
        AscendC::GlobalTensor<ElementC> gmWs1;
        gmWs1.SetGlobalBuffer((__gm__ ElementC*)params.ptrWorkspace + mnElems);

        auto layoutWs = tla::MakeLayout<ElementC, layout::RowMajor>(m, n);
        auto tensorWs0 = tla::MakeTensor(gmWs0, layoutWs, Catlass::Arch::PositionGM{});
        auto tensorWs1 = tla::MakeTensor(gmWs1, layoutWs, Catlass::Arch::PositionGM{});

        uint32_t aicoreIndex = AscendC::GetBlockIdx();
        if ASCEND_IS_AIV {
            aicoreIndex /= AscendC::GetSubBlockNum();
        }
        uint32_t loopStep = AscendC::GetBlockNum();
        Catlass::GemmCoord blockShape(L1_TILE_M, L1_TILE_N, L1_TILE_K);

        for (uint32_t loopIdx = aicoreIndex; loopIdx < coreLoops; loopIdx += loopStep) {
            ProcessOneBlock(scheduler, blockMmad, blockEpilogue, tensorX, tensorWh, tensorWl, tensorWs0, tensorWs1,
                            gmWs0, gmWs1, layoutWs, blockShape, loopIdx);
        }
    }

    template <class Scheduler, class Mmad, class Epilogue, class TensorX, class TensorW, class TensorWs, class GmWs,
              class LayoutWs>
    CATLASS_DEVICE void ProcessOneBlock(Scheduler& scheduler, Mmad& blockMmad, Epilogue& blockEpilogue,
                                        TensorX& tensorX, TensorW& tensorWh, TensorW& tensorWl, TensorWs& tensorWs0,
                                        TensorWs& tensorWs1, GmWs& gmWs0, GmWs& gmWs1, LayoutWs const& layoutWs,
                                        Catlass::GemmCoord const& blockShape, uint32_t loopIdx)
    {
        Catlass::GemmCoord blockCoord = scheduler.GetBlockCoord(loopIdx);
        Catlass::GemmCoord actualBlockShape = scheduler.GetActualBlockShape(blockCoord);

        auto tileOffset = tla::MakeCoord(blockCoord.m() * L1_TILE_M, blockCoord.n() * L1_TILE_N);
        auto tileShape = tla::MakeShape(actualBlockShape.m(), actualBlockShape.n());
        auto layoutTile = tla::GetTileLayout(layoutWs, tileShape, tla::MakeCoord(0, 0));

        if ASCEND_IS_AIC {
            RunAicBlock(blockMmad, tensorX, tensorWh, tensorWl, tensorWs0, tensorWs1, blockCoord, actualBlockShape,
                        tileOffset, tileShape);
        } else if ASCEND_IS_AIV {
            RunAivBlock(blockEpilogue, gmWs0, gmWs1, layoutWs, layoutTile, blockShape, blockCoord, actualBlockShape,
                        tileOffset);
        }
    }

    template <class Mmad, class TensorX, class TensorW, class TensorWs, class TileOffset, class TileShape>
    CATLASS_DEVICE void RunAicBlock(Mmad& blockMmad, TensorX& tensorX, TensorW& tensorWh, TensorW& tensorWl,
                                    TensorWs& tensorWs0, TensorWs& tensorWs1, Catlass::GemmCoord const& blockCoord,
                                    Catlass::GemmCoord const& actualBlockShape, TileOffset const& tileOffset,
                                    TileShape const& tileShape)
    {
        auto tensorBlockA = GetTile(tensorX, tla::MakeCoord(blockCoord.m() * L1_TILE_M, blockCoord.k() * L1_TILE_K),
                                    tla::MakeShape(actualBlockShape.m(), actualBlockShape.k()));
        auto tensorBlockBh = GetTile(tensorWh, tla::MakeCoord(blockCoord.k() * L1_TILE_K, blockCoord.n() * L1_TILE_N),
                                     tla::MakeShape(actualBlockShape.k(), actualBlockShape.n()));
        auto tensorBlockBl = GetTile(tensorWl, tla::MakeCoord(blockCoord.k() * L1_TILE_K, blockCoord.n() * L1_TILE_N),
                                     tla::MakeShape(actualBlockShape.k(), actualBlockShape.n()));
        auto tensorBlockY0 = GetTile(tensorWs0, tileOffset, tileShape);
        auto tensorBlockY1 = GetTile(tensorWs1, tileOffset, tileShape);

        blockMmad(tensorBlockA, tensorBlockBh, tensorBlockY0, actualBlockShape);

        AscendC::SetFlag<AscendC::HardEvent::FIX_M>(EVENT_L0C_REUSE);
        AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(EVENT_L0C_REUSE);
        Catlass::Arch::CrossCoreSetFlagWithReverse<0x2, PIPE_FIX>(flagY0Ready);

        blockMmad(tensorBlockA, tensorBlockBl, tensorBlockY1, actualBlockShape);
        Catlass::Arch::CrossCoreSetFlagWithReverse<0x2, PIPE_FIX>(flagY1Ready);
    }

    template <class Epilogue, class GmWs, class LayoutWs, class LayoutTile, class TileOffset>
    CATLASS_DEVICE void RunAivBlock(Epilogue& blockEpilogue, GmWs& gmWs0, GmWs& gmWs1, LayoutWs const& layoutWs,
                                    LayoutTile const& layoutTile, Catlass::GemmCoord const& blockShape,
                                    Catlass::GemmCoord const& blockCoord, Catlass::GemmCoord const& actualBlockShape,
                                    TileOffset const& tileOffset)
    {
        auto gmBlockY0 = gmWs0[layoutWs(tileOffset)];
        auto gmBlockY1 = gmWs1[layoutWs(tileOffset)];

        Catlass::Arch::CrossCoreWaitFlagWithReverse<0x2, PIPE_MTE2>(flagY0Ready);
        blockEpilogue.LoadY0(actualBlockShape, gmBlockY0, layoutTile);
        Catlass::Arch::CrossCoreWaitFlagWithReverse<0x2, PIPE_MTE2>(flagY1Ready);
        blockEpilogue.LoadY1(actualBlockShape, gmBlockY1, layoutTile);
        blockEpilogue.FuseAndStore(blockShape, blockCoord, actualBlockShape, layoutTile);
    }

    // Two independent single-direction handshakes: Y0 staged -> AIV load Y0 (overlaps
    // mmad1), Y1 staged -> AIV load Y1. Reverse flags drain the AIVs every 15 tiles.
    static constexpr Catlass::Arch::FlagID FLAG_AIC_Y0_READY = 0;
    static constexpr Catlass::Arch::FlagID RV_FLAG_AIC_Y0_READY = 1;
    static constexpr Catlass::Arch::FlagID FLAG_AIC_Y1_READY = 2;
    static constexpr Catlass::Arch::FlagID RV_FLAG_AIC_Y1_READY = 3;
    Catlass::Arch::CrossCoreFlagWithReverse<> flagY0Ready{FLAG_AIC_Y0_READY, RV_FLAG_AIC_Y0_READY};
    Catlass::Arch::CrossCoreFlagWithReverse<> flagY1Ready{FLAG_AIC_Y1_READY, RV_FLAG_AIC_Y1_READY};

    static constexpr int32_t EVENT_L0C_REUSE = 0;

    Catlass::Arch::Resource<ArchTag> resource;
};

} // namespace Catlass::Gemm::Kernel

#endif // CATLASS_GEMM_KERNEL_MATMUL_DECOMPOSITION_GM_HPP

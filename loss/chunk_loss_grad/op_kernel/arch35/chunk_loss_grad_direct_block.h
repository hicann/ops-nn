/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or
 * modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 *
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS
 * SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT
 * NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of
 * the software repository for the full text of the License.
 */

#pragma once
#include "blaze/gemm/block/block_mmad_matmul_basic.h"

namespace ChunkLossGradDirect {
// Source-backed custom Block: Basic's Copy/Mmad sequence, with explicit L1
// ownership and a split-M UB epilogue. No vendored header is modified.
template <class InputType>
class DirectBlock {
    using LA = asc::te::nd_ext_layout_ptn;
    using DN = asc::te::dn_ext_layout_ptn;
    using NZ = asc::te::nz_layout_ptn;
    using ZN = asc::te::zn_layout_ptn;

public:
    __aicore__ inline DirectBlock()
    {
        if ASCEND_IS_AIC {
            AscendC::SetMMLayoutTransform(true);
            ReleaseL0C(0);
            ReleaseL0C(1);
        }
    }
    __aicore__ inline ~DirectBlock()
    {
        if ASCEND_IS_AIC {
            WaitL0C(0);
            WaitL0C(1);
            AscendC::PipeBarrier<PIPE_ALL>();
            AscendC::SetMMLayoutTransform(false);
        }
    }

    template <bool l1Input, bool ubOutput, class A, class B, class C>
    __aicore__ inline void Run(A& matrixA, B& matrixB, C& matrixC, uint32_t rowCount, uint32_t columnCount,
                               uint32_t reductionSize)
    {
        const uint32_t accumulatorSlot = tiles & 1;
        WaitL0C(accumulatorSlot);
        auto l0cLayout = asc::te::frame_layout_format<NZ, asc::te::_16>{}(rowCount, columnCount);
        auto l0cTensor = asc::te::make_tensor(
            asc::te::make_mem_ptr<asc::te::location::l0c, float>(accumulatorSlot * 128 * 1024), l0cLayout);
        auto copyGmToL1 = asc::te::make_copy(asc::te::copy_gm_to_l1{});
        for (uint32_t reductionOffset = 0; reductionOffset < reductionSize; reductionOffset += 256) {
            const uint32_t reductionTileSize = reductionSize - reductionOffset < 256 ? reductionSize - reductionOffset :
                                                                                       256;
            const uint32_t stage = l1Sequence & 1;
            const uint32_t base = stage * 128 * 1024;
            auto l1aLayout = asc::te::frame_layout_format<NZ, asc::te::layout_trait_default<InputType>>{}(
                rowCount, reductionTileSize);
            auto l1aTensor = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::l1, InputType>(base),
                                                  l1aLayout);
            using BL = AscendC::Std::conditional_t<l1Input, NZ, ZN>;
            auto l1bLayout = asc::te::frame_layout_format<BL, asc::te::layout_trait_default<InputType>>{}(
                reductionTileSize, columnCount);
            // M1's A is already resident in the dedicated G slots. Its B
            // can use the entire128KiB staging slot (K256 x N256). M0
            // divides the same slot into64KiB A and64KiB B regions.
            constexpr uint32_t bOffset = l1Input ? 0 : 64 * 1024;
            auto l1bTensor = asc::te::make_tensor(
                asc::te::make_mem_ptr<asc::te::location::l1, InputType>(base + bOffset), l1bLayout);
            Blaze::Gemm::LockMte2(stage);
            if constexpr (!l1Input) {
                auto source = matrixA.slice(asc::te::make_coord(0, int64_t(reductionOffset)),
                                            asc::te::make_shape(int64_t(rowCount), int64_t(reductionTileSize)));
                asc::te::copy(copyGmToL1, l1aTensor, source);
            }
            auto bs = matrixB.slice(asc::te::make_coord(int64_t(reductionOffset), 0),
                                    asc::te::make_shape(int64_t(reductionTileSize), int64_t(columnCount)));
            asc::te::copy(copyGmToL1, l1bTensor, bs);
            Blaze::Gemm::UnLockMte2(stage);
            // N256/K64 keeps each double-buffered L0B tile within32KiB.
            constexpr uint32_t baseK = l1Input ? 64 : 128;
            for (uint32_t innerReductionOffset = 0; innerReductionOffset < reductionTileSize;
                 innerReductionOffset += baseK) {
                const uint32_t reductionCount = reductionTileSize - innerReductionOffset < baseK ?
                                                    reductionTileSize - innerReductionOffset :
                                                    baseK;
                const uint32_t slot = l0Sequence & 1;
                auto l0aLayout = asc::te::make_frame_layout<NZ, asc::te::layout_trait_default<InputType>>(
                    rowCount, reductionCount);
                auto l0bLayout = asc::te::make_frame_layout<ZN, asc::te::layout_trait_default<InputType>>(
                    reductionCount, columnCount);
                auto l0aTensor = asc::te::make_tensor(
                    asc::te::make_mem_ptr<asc::te::location::l0a, InputType>(slot * 32 * 1024), l0aLayout);
                auto l0bTensor = asc::te::make_tensor(
                    asc::te::make_mem_ptr<asc::te::location::l0b, InputType>(slot * 32 * 1024), l0bLayout);
                Blaze::Gemm::LockMte1(stage);
                Blaze::Gemm::LockMte1(slot + 4);
                auto copyL1ToL0A = asc::te::make_copy(asc::te::copy_l1_to_l0a{});
                if constexpr (l1Input) {
                    auto aSlice = matrixA.slice(asc::te::make_coord(0, int64_t(reductionOffset + innerReductionOffset)),
                                                asc::te::make_shape(int64_t(rowCount), int64_t(reductionCount)));
                    asc::te::copy(copyL1ToL0A, l0aTensor, aSlice);
                } else {
                    auto aSlice = l1aTensor.slice(asc::te::make_coord(0, int64_t(innerReductionOffset)),
                                                  asc::te::make_shape(int64_t(rowCount), int64_t(reductionCount)));
                    asc::te::copy(copyL1ToL0A, l0aTensor, aSlice);
                }
                auto bSlice = l1bTensor.slice(asc::te::make_coord(int64_t(innerReductionOffset), 0),
                                              asc::te::make_shape(int64_t(reductionCount), int64_t(columnCount)));
                auto copyL1ToL0B = asc::te::make_copy(asc::te::copy_l1_to_l0b{});
                asc::te::copy(copyL1ToL0B, l0bTensor, bSlice);
                Blaze::Gemm::UnLockMte1(stage);
                Blaze::Gemm::UnLockMte1(slot + 4);
                Blaze::Gemm::LockM(slot + 4);
                auto flag = reductionOffset + innerReductionOffset + reductionCount == reductionSize ?
                                asc::te::unit_flag_mode::enable_update :
                                asc::te::unit_flag_mode::enable_keep;
                asc::te::mmad_params params{uint16_t(rowCount), uint16_t(columnCount), uint16_t(reductionCount), flag,
                                            reductionOffset == 0 && innerReductionOffset == 0};
                constexpr auto op = asc::te::make_mmad(asc::te::mmad_operation{}, asc::te::mmad_trait_default{});
                asc::te::mmad(op.with(params), l0cTensor, l0aTensor, l0bTensor);
                Blaze::Gemm::UnLockM(slot + 4);
                ++l0Sequence;
            }
            ++l1Sequence;
        }
        if constexpr (ubOutput) {
            auto copyParams = asc::te::make_copy(asc::te::copy_l0c_to_ub{}, Blaze::Gemm::Tile::CopyL0C2UBTraitSplitM{});
            asc::te::l0c_to_ub_params params{asc::te::unit_flag_mode::enable_update};
            asc::te::copy(copyParams.with(params), matrixC, l0cTensor);
        } else {
            auto copyParams = asc::te::make_copy(asc::te::copy_l0c_to_gm{});
            asc::te::l0c_to_gm_params params{asc::te::unit_flag_mode::enable_update};
            asc::te::copy(copyParams.with(params), matrixC, l0cTensor);
        }
        ReleaseL0C(accumulatorSlot);
        ++tiles;
    }
    // Reuse each A (weight) K256 slice across up to four N128 fragments.
    // Adapted from this Block's streaming copy/MMAD sequence and Blaze's
    // A-L1-full-load ownership model. Full W does not fit beside resident G,
    // so only the current K slice stays resident across the N loop.
    // L1: A0/B0/A1/B1, each 64 KiB; L0A/B: the existing two32KiB slots.
    // M0 has four64KiB accumulators, disjoint until the ordered epilogue.
    // The epilogue publishes the original UB tiles in original N order.
    template <bool keepA0, class A, class B, class Epilogue>
    __aicore__ inline void RunLogitsReuse(A& matrixA, B& matrixB, uint32_t rowCount, uint32_t columnCount,
                                          uint32_t reductionSize, Epilogue& epilogue)
    {
        // Suspend the old two128KiB C-slot protocol before changing layouts.
        WaitL0C(0);
        WaitL0C(1);
        // IDs0/1: A K-slice slots. IDs2/3: independently rotated B slots.
        // MTE1 releases only after its last read of each corresponding slot.
        for (uint32_t aSlot = 0; aSlot < 2; ++aSlot) {
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(aSlot);
        }
        for (uint32_t bEvent = 2; bEvent < 4; ++bEvent) {
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(bEvent);
        }
        auto copyGmToL1 = asc::te::make_copy(asc::te::copy_gm_to_l1{});
        for (uint32_t reductionOffset = 0; reductionOffset < reductionSize; reductionOffset += 256) {
            const uint32_t reductionTileSize = reductionSize - reductionOffset < 256 ? reductionSize - reductionOffset :
                                                                                       256;
            const uint32_t aSlot = (reductionOffset / 256) & 1;
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(aSlot);
            auto l1aLayout = asc::te::frame_layout_format<NZ, asc::te::layout_trait_default<InputType>>{}(
                rowCount, reductionTileSize);
            auto l1aTensor = asc::te::make_tensor(
                asc::te::make_mem_ptr<asc::te::location::l1, InputType>(aSlot * 128 * 1024), l1aLayout);
            auto as = matrixA.slice(asc::te::make_coord(0, int64_t(reductionOffset)),
                                    asc::te::make_shape(int64_t(rowCount), int64_t(reductionTileSize)));
            asc::te::copy(copyGmToL1, l1aTensor, as);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(aSlot);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(aSlot);
            for (uint32_t tokenOffset = 0; tokenOffset < columnCount; tokenOffset += 128) {
                const uint32_t tokenCount = columnCount - tokenOffset < 128 ? columnCount - tokenOffset : 128;
                const uint32_t bSlot = l1Sequence & 1;
                const uint32_t bEvent = 2 + bSlot;
                AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(bEvent);
                auto l1bLayout = asc::te::frame_layout_format<ZN, asc::te::layout_trait_default<InputType>>{}(
                    reductionTileSize, tokenCount);
                auto l1bTensor = asc::te::make_tensor(
                    asc::te::make_mem_ptr<asc::te::location::l1, InputType>(bSlot * 128 * 1024 + 64 * 1024), l1bLayout);
                auto bs = matrixB.slice(asc::te::make_coord(int64_t(reductionOffset), int64_t(tokenOffset)),
                                        asc::te::make_shape(int64_t(reductionTileSize), int64_t(tokenCount)));
                asc::te::copy(copyGmToL1, l1bTensor, bs);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(bEvent);
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(bEvent);
                auto cl = asc::te::frame_layout_format<NZ, asc::te::_16>{}(rowCount, tokenCount);
                auto l0cTensor = asc::te::make_tensor(
                    asc::te::make_mem_ptr<asc::te::location::l0c, float>((tokenOffset / 128) * 64 * 1024), cl);
                for (uint32_t innerReductionOffset = 0; innerReductionOffset < reductionTileSize;
                     innerReductionOffset += 128) {
                    const uint32_t reductionCount = reductionTileSize - innerReductionOffset < 128 ?
                                                        reductionTileSize - innerReductionOffset :
                                                        128;
                    const uint32_t slot = l0Sequence & 1;
                    auto l0aLayout = asc::te::make_frame_layout<NZ, asc::te::layout_trait_default<InputType>>(
                        rowCount, reductionCount);
                    auto l0bLayout = asc::te::make_frame_layout<ZN, asc::te::layout_trait_default<InputType>>(
                        reductionCount, tokenCount);
                    auto l0aTensor = asc::te::make_tensor(
                        asc::te::make_mem_ptr<asc::te::location::l0a, InputType>(slot * 32 * 1024), l0aLayout);
                    auto l0bTensor = asc::te::make_tensor(
                        asc::te::make_mem_ptr<asc::te::location::l0b, InputType>(slot * 32 * 1024), l0bLayout);
                    Blaze::Gemm::LockMte1(slot + 4);
                    auto aSlice = l1aTensor.slice(asc::te::make_coord(0, int64_t(innerReductionOffset)),
                                                  asc::te::make_shape(int64_t(rowCount), int64_t(reductionCount)));
                    auto bSlice = l1bTensor.slice(asc::te::make_coord(int64_t(innerReductionOffset), 0),
                                                  asc::te::make_shape(int64_t(reductionCount), int64_t(tokenCount)));
                    auto copyL1ToL0A = asc::te::make_copy(asc::te::copy_l1_to_l0a{});
                    auto copyL1ToL0B = asc::te::make_copy(asc::te::copy_l1_to_l0b{});
                    // A's two K128 halves stay in their original L0A slots
                    // across N fragments. An odd single-half K tail rotates
                    // l0Sequence per N, so that case retains the original copy.
                    if constexpr (keepA0) {
                        if (tokenOffset == 0 || reductionTileSize <= 128) {
                            asc::te::copy(copyL1ToL0A, l0aTensor, aSlice);
                        }
                    } else {
                        asc::te::copy(copyL1ToL0A, l0aTensor, aSlice);
                    }
                    asc::te::copy(copyL1ToL0B, l0bTensor, bSlice);
                    Blaze::Gemm::UnLockMte1(slot + 4);
                    Blaze::Gemm::LockM(slot + 4);
                    const bool lastK = reductionOffset + innerReductionOffset + reductionCount == reductionSize;
                    auto flag = lastK ? asc::te::unit_flag_mode::enable_update : asc::te::unit_flag_mode::enable_keep;
                    asc::te::mmad_params params{uint16_t(rowCount), uint16_t(tokenCount), uint16_t(reductionCount),
                                                flag, reductionOffset == 0 && innerReductionOffset == 0};
                    constexpr auto op = asc::te::make_mmad(asc::te::mmad_operation{}, asc::te::mmad_trait_default{});
                    asc::te::mmad(op.with(params), l0cTensor, l0aTensor, l0bTensor);
                    Blaze::Gemm::UnLockM(slot + 4);
                    ++l0Sequence;
                }
                // Both K128 reads have been issued on MTE1 before releasing B.
                AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(bEvent);
                ++l1Sequence;
                // Queue each matching Fixpipe after its final update, in N order.
                // No C slot is reused until the group-wide FIX drain below.
                if (reductionOffset + reductionTileSize == reductionSize) {
                    epilogue(l0cTensor, tokenCount);
                }
            }
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(aSlot);
        }
        for (uint32_t aSlot = 0; aSlot < 2; ++aSlot) {
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(aSlot);
        }
        for (uint32_t bEvent = 2; bEvent < 4; ++bEvent) {
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(bEvent);
        }
        AscendC::PipeBarrier<PIPE_ALL>();
        // Resume the unchanged M1 two128KiB C-slot protocol.
        ReleaseL0C(0);
        ReleaseL0C(1);
    }

    __aicore__ inline void Drain()
    {
        for (uint32_t i = 0; i < 2; ++i) {
            WaitL0C(i);
        }
        AscendC::PipeBarrier<PIPE_ALL>();
        for (uint32_t i = 0; i < 2; ++i) {
            ReleaseL0C(i);
        }
    }

private:
    __aicore__ inline void WaitL0C(uint32_t slot) { AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(slot); }
    __aicore__ inline void ReleaseL0C(uint32_t slot) { AscendC::SetFlag<AscendC::HardEvent::FIX_M>(slot); }
    uint32_t tiles = 0, l1Sequence = 0, l0Sequence = 0;
};
} // namespace ChunkLossGradDirect

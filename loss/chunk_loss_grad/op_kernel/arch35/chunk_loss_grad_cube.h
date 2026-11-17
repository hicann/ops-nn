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

namespace ChunkLossGradOpt {
// Two GEMMs share the same physical L1/L0 arena. They execute on one AIC,
// never concurrently. Basic BlockMmad uses local mutex IDs 0..5, no cross-core
// flag IDs. Its own ND/DN copy handles logical tails independently of strides.
template <class InputType, bool transposeB>
class Cube {
    using LA = asc::te::nd_ext_layout_ptn;
    using LB = AscendC::Std::conditional_t<transposeB, asc::te::dn_ext_layout_ptn, LA>;
    using Policy = Blaze::Gemm::MatmulMultiBlockBasic<>;
    using Block = Blaze::Gemm::Block::BlockMmad<Policy, InputType, LA, InputType, LB, float, LA, float, LA>;

public:
    __aicore__ inline void Compute(GM_ADDR matrixA, GM_ADDR matrixB, GM_ADDR matrixC, uint32_t rows, uint32_t columns,
                                   uint32_t reductionSize, uint32_t aRowStride, uint32_t baseM)
    {
        typename Block::Params params;
        // baseM is a legal hardware tile (16/32/64/128), independent of
        // logical V tails. For larger tiles N256/K64 keeps double-buffered
        // L0B within64KiB and halves the number of output/Fixpipe blocks.
        // M128/N256/K256 with two L1 stages uses384KiB. Each M128/N256
        // FP32 L0C result uses one complete128KiB half.
        const uint32_t baseN = baseM > 16 ? 256 : 128;
        const uint32_t baseK = baseM > 16 ? 64 : 128;
        params.mL1 = baseM;
        params.nL1 = baseN;
        params.kL1 = 256;
        params.mL0 = baseM;
        params.nL0 = baseN;
        params.kL0 = baseK;
        const uint32_t cSlots = baseM > 16 ? 2 : 1;
        params.l1Stages = 2;
        params.l0cStages = cSlots;
        params.biasGmAddr = nullptr;
        block.Init(params);
        auto layoutA = Blaze::Gemm::MakeNDExtLayout<InputType>(rows, reductionSize, aRowStride);
        auto layoutB = asc::te::frame_layout_format<LB, asc::te::layout_trait_default<InputType>>{}(reductionSize,
                                                                                                    columns);
        auto layoutC = Blaze::Gemm::MakeNDExtLayout<float>(rows, columns, columns);
        auto layoutBias = Blaze::Gemm::MakeNDExtLayout<float>(1, columns, columns);
        auto gmTensorA = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::gm>((__gm__ InputType*)matrixA),
                                              layoutA);
        auto gmTensorB = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::gm>((__gm__ InputType*)matrixB),
                                              layoutB);
        auto gmTensorC = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::gm>((__gm__ float*)matrixC),
                                              layoutC);
        // Bias is disabled; a valid unused pointer avoids null pointer arithmetic.
        auto bias = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::gm>((__gm__ float*)matrixC),
                                         layoutBias);
        uint32_t tiles = 0;
        for (uint32_t rowOffset = 0; rowOffset < rows; rowOffset += baseM) {
            const uint32_t cm = rows - rowOffset < baseM ? rows - rowOffset : baseM;
            for (uint32_t columnOffset = 0; columnOffset < columns; columnOffset += baseN) {
                const uint32_t cn = columns - columnOffset < baseN ? columns - columnOffset : baseN;
                auto matrixSliceA = gmTensorA.slice(asc::te::make_coord(int64_t(rowOffset), int64_t(0)),
                                                    asc::te::make_shape(int64_t(cm), int64_t(reductionSize)));
                auto matrixSliceB = gmTensorB.slice(asc::te::make_coord(int64_t(0), int64_t(columnOffset)),
                                                    asc::te::make_shape(int64_t(reductionSize), int64_t(cn)));
                auto matrixSliceC = gmTensorC.slice(asc::te::make_coord(int64_t(rowOffset), int64_t(columnOffset)),
                                                    asc::te::make_shape(int64_t(cm), int64_t(cn)));
                auto biasSlice = bias.slice(asc::te::make_coord(int64_t(0), int64_t(columnOffset)),
                                            asc::te::make_shape(int64_t(1), int64_t(cn)));
                typename Block::TupleShape shape{cm, cn, reductionSize, 1};
                const uint32_t cSlot = tiles % cSlots;
                if (tiles >= cSlots) {
                    AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(cSlot);
                }
                block(matrixSliceA, matrixSliceB, biasSlice, matrixSliceC, shape);
                // Block selects 0/128KiB L0C halves in the same order. Wait
                // only when this half is reused, allowing the next tile's
                // Cube work to overlap the previous tile's Fixpipe.
                AscendC::SetFlag<AscendC::HardEvent::FIX_M>(cSlot);
                ++tiles;
            }
        }
        for (uint32_t slot = 0; slot < cSlots && slot < tiles; ++slot) {
            AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(slot);
        }
        // Complete all arena users before the other GEMM changes its layouts.
        AscendC::PipeBarrier<PIPE_ALL>();
    }

private:
    Block block;
};
} // namespace ChunkLossGradOpt

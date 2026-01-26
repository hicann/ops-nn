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
 * \file block_mmad_mx_weight_from_ub.h
 * \brief
 */

#ifndef QUANT_BATCH_MATMUL_V4_CMCT_BLOCK_BLOCK_MMAD_MX_WEIGHT_FROM_UB_H
#define QUANT_BATCH_MATMUL_V4_CMCT_BLOCK_BLOCK_MMAD_MX_WEIGHT_FROM_UB_H

#include "cmct/block/block_mmad.h"
#include "../../quant_batch_matmul_v4_tiling_data.h"

namespace QuantBatchMatmulV4 {
namespace Block {
using Cmct::Gemm::Block::BlockMmad;
template <
    class DispatchPolicy, class L1TileShape_, class L0TileShape_, class ATypeTuple_, class BType_, class CType_,
    class BiasType_, class TileCopy_, class TileMmad_>
class BlockMmadNN
    : public BlockMmad<
          DispatchPolicy, L1TileShape_, L0TileShape_, ATypeTuple_, BType_, CType_, BiasType_, TileCopy_, TileMmad_> {
public:
    using MmadCmct = BlockMmad<
        DispatchPolicy, L1TileShape_, L0TileShape_, ATypeTuple_, BType_, CType_, BiasType_, TileCopy_, TileMmad_>;
    using Arguments = typename MmadCmct::Arguments;
    using Params = typename MmadCmct::Params;
    __aicore__ inline BlockMmadNN() = delete;
    __aicore__ inline BlockMmadNN(const Params& params) : MmadCmct(params)
    {}
    template <typename ProblemShape>
    __aicore__ inline static Params ToUnderlyingArguments(
        ProblemShape const& problemShape, Arguments const& args,
        qbmmv4_tiling::QuantBatchMatmulV4TilingDataParams const* tiling)
    {
        auto stepKa = static_cast<uint32_t>(tiling->matmulTiling.stepKa);
        auto stepKb = static_cast<uint32_t>(tiling->matmulTiling.stepKa);
        auto baseK = static_cast<uint32_t>(tiling->matmulTiling.baseK);
        auto baseM = static_cast<uint32_t>(tiling->matmulTiling.baseM);
        auto baseN = static_cast<uint32_t>(tiling->matmulTiling.baseN);
        return {
            .ptrA = args.ptrA,
            .ptrC = args.ptrC,
            .ptrAScale = args.ptrAScale,
            .ptrBScale = args.ptrBScale,
            .layoutA = args.layoutA,
            .layoutC = args.layoutC,
            .layoutScale = args.layoutScale,
            .tileShapeL1 = AscendC::MakeShape(baseM, baseN, stepKa * baseK, stepKb * baseK),
            .tileShapeL0 = AscendC::MakeShape(baseM, baseN, baseK),
            .scaleFactor = tiling->matmulTiling.mxTypePara & 0xff, // 0xff：to obtain the lower 8 bits
            .aL1BufNum = tiling->AL1Pingpong,
            .isBias = bool(tiling->matmulTiling.isBias)};
    }
};
} // namespace Block
} // namespace QuantBatchMatmulV4
#endif
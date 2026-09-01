/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef MATMUL_EMU_CATLASS_GEMM_BLOCK_BLOCK_SWIZZLE_HPP
#define MATMUL_EMU_CATLASS_GEMM_BLOCK_BLOCK_SWIZZLE_HPP

#include "../../matmul_emu_catlass.hpp"
#include "../../matmul_emu_gemm_coord.hpp"

namespace Catlass::Gemm::Block {

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Block swizzling function for Gemms
template <uint32_t SwizzleOffset = 1, uint32_t SwizzleDirection = 0>
struct GemmIdentityBlockSwizzle {
    /// Data members

    GemmCoord problemShape;
    MatrixCoord tileMN;
    MatrixCoord loopsMN;

    /// Methods

    CATLASS_DEVICE
    GemmIdentityBlockSwizzle() {}

    CATLASS_DEVICE
    GemmIdentityBlockSwizzle(GemmCoord const& problemShape_, MatrixCoord const& tileMN_)
        : problemShape(problemShape_), tileMN(tileMN_)
    {
        loopsMN = CeilDiv(MatrixCoord(problemShape.GetCoordMN()), tileMN);
    }

    CATLASS_DEVICE
    GemmIdentityBlockSwizzle(GemmCoord const& problemShape_, MatrixCoord const& tileMN_, MatrixCoord const& loopsMN_)
        : problemShape(problemShape_), tileMN(tileMN_), loopsMN(loopsMN_)
    {}

    CATLASS_DEVICE
    void Update(GemmCoord const& problemShape_, MatrixCoord const& tileMN_)
    {
        problemShape = problemShape_;
        tileMN = tileMN_;

        loopsMN = CeilDiv(MatrixCoord(problemShape.GetCoordMN()), tileMN);
    }

    CATLASS_DEVICE
    void Update(GemmCoord const& problemShape_, MatrixCoord const& tileMN_, MatrixCoord const& loopsMN_)
    {
        problemShape = problemShape_;
        tileMN = tileMN_;
        loopsMN = loopsMN_;
    }

    CATLASS_DEVICE
    uint32_t GetCoreLoops() const { return loopsMN.row() * loopsMN.column(); }

    CATLASS_DEVICE
    uint32_t GetBatchIdx(uint32_t taskIdx) { return taskIdx / (GetCoreLoops()); }

    CATLASS_DEVICE
    GemmCoord GetBlockCoord(uint32_t taskIdx)
    {
        uint32_t innerIdx = taskIdx % GetCoreLoops();
        if constexpr (SwizzleDirection == 0) { // Zn
            uint32_t tileBlockLoop = CeilDiv(loopsMN.row(), SwizzleOffset);
            uint32_t tileBlockIdx = innerIdx / (SwizzleOffset * loopsMN.column());
            uint32_t inTileBlockIdx = innerIdx % (SwizzleOffset * loopsMN.column());

            uint32_t nRow = SwizzleOffset;
            if (tileBlockIdx == tileBlockLoop - 1) {
                nRow = loopsMN.row() - SwizzleOffset * tileBlockIdx;
            }
            uint32_t mIdx = tileBlockIdx * SwizzleOffset + inTileBlockIdx % nRow;
            uint32_t nIdx = inTileBlockIdx / nRow;
            if (tileBlockIdx % 2 == 1) {
                nIdx = loopsMN.column() - nIdx - 1;
            }
            return GemmCoord{mIdx, nIdx, 0};
        } else if constexpr (SwizzleDirection == 1) { // Nz
            uint32_t tileBlockLoop = CeilDiv(loopsMN.column(), SwizzleOffset);
            uint32_t tileBlockIdx = innerIdx / (SwizzleOffset * loopsMN.row());
            uint32_t inTileBlockIdx = innerIdx % (SwizzleOffset * loopsMN.row());

            uint32_t nCol = SwizzleOffset;
            if (tileBlockIdx == tileBlockLoop - 1) {
                nCol = loopsMN.column() - SwizzleOffset * tileBlockIdx;
            }
            uint32_t mIdx = inTileBlockIdx / nCol;
            uint32_t nIdx = tileBlockIdx * SwizzleOffset + inTileBlockIdx % nCol;
            if (tileBlockIdx % 2 == 1) {
                mIdx = loopsMN.row() - mIdx - 1;
            }
            return GemmCoord{mIdx, nIdx, 0};
        }
    }

    CATLASS_DEVICE
    GemmCoord GetActualBlockShape(GemmCoord blockCoord)
    {
        uint32_t mActual = (blockCoord.m() == (loopsMN.row() - 1)) ?
                               (problemShape.m() - blockCoord.m() * tileMN.row()) :
                               tileMN.row();
        uint32_t nActual = (blockCoord.n() == (loopsMN.column() - 1)) ?
                               (problemShape.n() - blockCoord.n() * tileMN.column()) :
                               tileMN.column();
        uint32_t kActual = problemShape.k();
        return GemmCoord{mActual, nActual, kActual};
    }
};

} // namespace Catlass::Gemm::Block

#endif // MATMUL_EMU_CATLASS_GEMM_BLOCK_BLOCK_SWIZZLE_HPP

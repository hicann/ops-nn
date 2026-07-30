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
 * \file quant_batch_matmul_v4_weight_quant_mx_swat_tiling_solver.h
 * \brief
 */

#pragma once

#include <cstdint>
#include <string>

#include "graph/types.h"
#include "../../../op_kernel/arch35/quant_batch_matmul_v4_tiling_data_apt.h"

namespace optiling {
struct WeightQuantMxSwatPlatformParam {
    uint64_t aicNum = 0UL;
    uint64_t ubSize = 0UL;
    uint64_t l1Size = 0UL;
    uint64_t l0aSize = 0UL;
    uint64_t l0bSize = 0UL;
    uint64_t l0cSize = 0UL;
};

struct WeightQuantMxSwatShapeParam {
    uint64_t m = 0UL;
    uint64_t n = 0UL;
    uint64_t k = 0UL;
};

class WeightQuantMxSwatTilingSolver {
public:
    explicit WeightQuantMxSwatTilingSolver(uint64_t targetL1BufferNum, bool allowConservativeFallback = false);

    bool Solve(const WeightQuantMxSwatPlatformParam& platform, const WeightQuantMxSwatShapeParam& shape,
               uint64_t groupSize, bool hasBias, bool hasX1Scale, bool hasX2Scale, bool weightNz, ge::DataType yDtype,
               qbmmv4_tiling::QuantBatchMatmulV4WeightQuantMxSwatTilingData& tilingData, std::string& reason);

private:
    struct RunInfo {
        uint64_t baseM = 0UL;
        uint64_t baseN = 0UL;
        uint64_t baseK = 0UL;
        uint64_t tileShapeKL1 = 0UL;
        uint64_t tileShapeScaleKL1 = 0UL;
        uint64_t nBubSize = 0UL;
        uint64_t kBubSize = 0UL;
        uint64_t mBlockCnt = 0UL;
        uint64_t nBlockCnt = 0UL;
        uint64_t totalBlockCnt = 0UL;
        uint64_t tailBlockCnt = 0UL;
        uint64_t mTailSize = 0UL;
        uint64_t nTailSize = 0UL;
        uint64_t mTailTile = 1UL;
        uint64_t nTailTile = 1UL;
        uint64_t mBaseTailSplitCnt = 1UL;
        uint64_t nBaseTailSplitCnt = 1UL;
        uint64_t mTailMain = 0UL;
        uint64_t nTailMain = 0UL;
    };

    bool Init(const WeightQuantMxSwatPlatformParam& platform, const WeightQuantMxSwatShapeParam& shape,
              std::string& reason);
    bool CalcBasicBlock(std::string& reason);
    bool CalcConservativeTiling(std::string& reason);
    void AdjustBasicBlock();
    void OptimizeEdgeBasicBlock();
    void CalcTailBasicBlock();
    bool CalcPathSpecificL1(std::string& reason);
    uint64_t CalcMaxScaleFactor(uint64_t stepK) const;
    bool ValidateTilingResult() const;
    bool IsL0Feasible(uint64_t baseM, uint64_t baseN, uint64_t baseK) const;
    bool IsL1Feasible(uint64_t tileShapeKL1, uint64_t tileShapeScaleKL1) const;
    bool IsBubTilingValid(uint64_t nBubSize, uint64_t kBubSize) const;
    void BuildTilingData(uint64_t groupSize, bool hasBias, bool hasX1Scale, bool hasX2Scale, bool weightNz,
                         ge::DataType yDtype,
                         qbmmv4_tiling::QuantBatchMatmulV4WeightQuantMxSwatTilingData& tilingData) const;
    bool FindBubSize(uint64_t nBl1Size, uint64_t kBl1Size, uint64_t& nBubSize, uint64_t& kBubSize) const;
    uint64_t FindKOnlyBubSize(uint64_t nBubSize, uint64_t kBl1Size) const;
    uint64_t GetBubSize(uint64_t bufferNum, uint64_t nDimSize, uint64_t kDimSize) const;
    uint64_t CalUsedCoreNum(uint64_t mTile, uint64_t nTile) const;
    uint64_t CalcValidSplitCount(uint64_t singleCoreM, uint64_t singleCoreN, uint64_t mTile, uint64_t nTile) const;
    void GetLogicalTileCoord(uint64_t tileIdx, uint64_t& mTileIdx, uint64_t& nTileIdx) const;
    uint64_t GetSingleCoreM(uint64_t mTileIdx) const;
    uint64_t GetSingleCoreN(uint64_t nTileIdx) const;
    static uint64_t GetBiasDataSize(ge::DataType yDtype);

    WeightQuantMxSwatShapeParam shape_;
    WeightQuantMxSwatPlatformParam platform_;
    RunInfo runInfo_;
    uint64_t targetL1BufferNum_ = 2UL;
    bool allowConservativeFallback_ = false;
    bool hasBias_ = false;
    bool weightNz_ = false;
    uint64_t biasDataSize_ = 0UL;
};
} // namespace optiling

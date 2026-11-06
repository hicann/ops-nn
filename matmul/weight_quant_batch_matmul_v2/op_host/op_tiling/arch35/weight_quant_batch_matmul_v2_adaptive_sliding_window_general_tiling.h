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
 * \file weight_quant_batch_matmul_v2_adaptive_sliding_window_tiling.h
 * \brief
 */

#pragma once

#include "../weight_quant_batch_matmul_v2_tiling.h"
#include "matmul/weight_quant_batch_matmul_v2/op_kernel/arch35/weight_quant_batch_matmul_v2_arch35_tiling_data.h"

namespace optiling {
namespace weight_quant_batch_matmul_v2 {

struct AdaptiveSlidingWindowGerneral {
    uint64_t baseM = 0; // 主窗口基本块大小
    uint64_t baseN = 0; // 主窗口基本块大小
    uint64_t baseK = 0;
    uint64_t mBlockCnt = 0;       // m方向基本块数量
    uint64_t nBlockCnt = 0;       // n方向基本块数量
    uint64_t totalBlockCnt = 0;   // 基本块总数
    uint64_t mTail = 0;           // m方向尾块的有效行数
    uint64_t nTail = 0;           // n方向尾块的有效列数
    uint64_t singleWinM = 0;      // 主窗口的m边长
    uint64_t singleWinN = 0;      // 主窗口的n边长
    uint64_t totalWinCnt = 0;     // 窗口总数，及核执行最大轮数
    uint64_t mainRow = 0;         // 主窗口行数（以窗口为单位，一个窗口为一行）
    uint64_t tailWinBlockCnt = 0; // 尾窗口包含的基本快数量
    uint64_t mTailTile = 1;       // 尾部窗口基本块m方向重切粒度
    uint64_t nTailTile = 1;       // 尾部窗口基本块n方向重切粒度
    uint64_t mainTileNum = 1;
    uint64_t lastRoundBaseN = 16;
    uint64_t lastRoundBaseNTail = 0;
    bool evadeDuplicateBlock = false; // 是否跳过尾核的重复计算
    bool useTailWinLogic = true;      // 是否使用尾窗口处理逻辑
    bool enablePreAlloc = true;       // MTE2是否开启PreAllocation
};

struct BasicTilingGerneral {
    uint32_t usedCoreNum = 1;
    uint32_t tilingMode = 0;
    uint32_t singleCoreM = 1;
    uint32_t singleCoreN = 1;
    uint32_t singleCoreK = 1;
    uint32_t baseM = 1;
    uint32_t baseN = 1;
    uint32_t baseK = 1;
    uint32_t stepKa = 1;
    uint32_t stepKb = 1;
    uint32_t depthA1 = 1;
    uint32_t depthB1 = 1;
    uint32_t stepM = 1;
    uint32_t stepN = 1;
    uint32_t iterateOrder = 0;
    uint32_t dbL0c = 1;
    uint32_t isMclash = 0;
    uint32_t isNclash = 0;
    uint32_t mTileCntl2 = 0;
    uint32_t nTileCntl2 = 0;
    uint32_t mTileBlock = 0;
    uint32_t nTileBlock = 0;
    uint32_t calOrder = 0;
    uint32_t ubCalcN = 0;
    uint32_t ubCalcM = 0;
    uint32_t scaleFactorA = 1;
    uint32_t scaleFactorB = 1;
};

struct BasicBlockParamGerneral {
    uint64_t mSize = 0;
    uint64_t nSize = 0;
    uint64_t kSize = 0;
    uint64_t singleM = 0;
    uint64_t singleN = 0;
    uint64_t mDim = 0;
    uint64_t nDim = 0;
    uint64_t baseM = 0;
    uint64_t baseN = 0;
    uint64_t baseK = 0;
    uint64_t aDtypeBits = 0;
    uint64_t bDtypeBits = 0;
    uint64_t biasDtypeBits = 0;
    double mte2Cost = 0;

    bool operator<(const BasicBlockParamGerneral& other) const
    {
        if (mte2Cost != other.mte2Cost) {
            return mte2Cost < other.mte2Cost;
        }

        bool thisMDivisible = (mSize % baseM == 0);
        bool otherMDivisible = (other.mSize % other.baseM == 0);
        if (thisMDivisible != otherMDivisible) {
            // M能整除的排在前面
            return thisMDivisible;
        }

        bool thisNDivisible = (nSize % baseN == 0);
        bool otherNDivisible = (other.nSize % other.baseN == 0);
        if (thisNDivisible != otherNDivisible) {
            // N能整除的排在前面
            return thisNDivisible;
        }

        if (baseN != other.baseN) {
            return baseN > other.baseN;
        }
        return baseM > other.baseM;
    }
};

class WeightQuantBatchMatmulV2TilingASWGeneral : public WeightQuantBatchMatmulV2Tiling {
public:
    explicit WeightQuantBatchMatmulV2TilingASWGeneral(gert::TilingContext* context)
        : WeightQuantBatchMatmulV2Tiling(context)
    {
        if (context->GetCompileInfo() == nullptr) {
            InitCompileInfo();
        }
    }
    ~WeightQuantBatchMatmulV2TilingASWGeneral() override = default;

protected:
    ge::graphStatus DoOpTiling() override;
    ge::graphStatus DoLibApiTiling() override;
    uint64_t GetTilingKey() const override;
    ge::graphStatus GetWorkspaceSize() override;
    ge::graphStatus PostTiling() override;

    std::unique_ptr<wqbmmv2_tiling::WeightQuantBatchMatmulV2ASWTilingData> tilingData_;
    size_t tilingDataSize_ = sizeof(wqbmmv2_tiling::WeightQuantBatchMatmulV2ASWTilingData);

    ge::graphStatus InstantiateTilingData();
    bool AnalyseSlidingWinInfo();
    bool CalcBasicBlock();
    void CalculateBaseN();
    constexpr bool Is4BitInput(ge::DataType dtype) const;
    uint64_t GetShapeWithDataType(uint64_t shapeSize, ge::DataType dtype) const;
    uint64_t GetSizeWithDataType(uint64_t shapeSize, ge::DataType dtype) const;
    void CalcTailBasicBlock();
    bool IsInValidWeighNzTailSplit(uint64_t splitCnt, bool isPreSplit) const;
    uint32_t CalUsedCoreNum();
    uint32_t CalUsedCoreNum(uint32_t mTile, uint32_t nTile);
    void CalL1Tiling();
    void CalStepKs();
    void SetTilingData();

private:
    // 滑窗相关信息
    AdaptiveSlidingWindowGerneral adaptiveWin_;
    BasicTilingGerneral basicTiling_;
    bool ClosePreAllocCond() const;
};

} // namespace weight_quant_batch_matmul_v2
} // namespace optiling

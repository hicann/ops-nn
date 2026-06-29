/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file adaptive_sliding_window_tiling.h
 * \brief Adaptive sliding-window tiling base definitions.
 */
#pragma once
#include <vector>

#include "../quant_batch_matmul_v3_tiling_base.h"
#include "quant_batch_matmul_v3_tiling_util.h"
#include "matmul/quant_batch_matmul_v3/op_kernel/arch35/quant_batch_matmul_v3_tiling_data.h"

namespace optiling {

void ResetAdaptiveSlidingWindowPlatformInfoCache();

struct AdaptiveSlidingWindow {
    uint64_t baseM = 0;             // Base M block size for the main window.
    uint64_t baseN = 0;             // Base N block size for the main window.
    uint64_t baseK = 0;             // Base K block size.
    uint64_t mBlockCnt = 0;         // Number of base blocks along M.
    uint64_t nBlockCnt = 0;         // Number of base blocks along N.
    uint64_t totalBlockCnt = 0;     // Total number of base blocks.
    uint64_t mTail = 0;             // Effective M size of the tail block.
    uint64_t nTail = 0;             // Effective N size of the tail block.
    uint64_t singleWinM = 0;        // M span covered by one window.
    uint64_t singleWinN = 0;        // N span covered by one window.
    uint64_t totalWinCnt = 0;       // Total number of windows, also the max execution rounds.
    uint64_t mainRow = 0;           // Number of rows in the main window layout.
    uint64_t tailWinBlockCnt = 0;   // Number of base blocks contained in the tail window.
    uint64_t mTailTile = 1;         // Extra split factor along M for the tail window.
    uint64_t nTailTile = 1;         // Extra split factor along N for the tail window.
    uint64_t mBaseTailSplitCnt = 1; // Number of merged base blocks for M-tail balancing.
    uint64_t nBaseTailSplitCnt = 1; // Number of merged base blocks for N-tail balancing.
    uint64_t mTailMain = 0;         // Rebalanced M size for the tail window.
    uint64_t nTailMain = 0;         // Rebalanced N size for the tail window.
    bool useTailWinLogic = true;    // Whether tail-window specific logic is enabled.
};

class AdaptiveSlidingWindowTiling : public QuantBatchMatmulV3TilingBase {
public:
    explicit AdaptiveSlidingWindowTiling(gert::TilingContext* context);
    AdaptiveSlidingWindowTiling(gert::TilingContext* context, DequantBmm::QuantBatchMatmulV3TilingDataParams* out);
    ~AdaptiveSlidingWindowTiling() override = default;

    // 1. Query platform information such as core count and memory sizes.
    ge::graphStatus GetPlatformInfo() override;
    // 2. Parse input shapes, dtypes and attributes.
    ge::graphStatus GetShapeAttrsInfo() override;
    // 3. Compute operator tiling data.
    ge::graphStatus DoOpTiling() override;
    // 4. Compute libapi tiling data for direct consumers such as MC2.
    ge::graphStatus DoLibApiTiling() override;
    // 5. Compute the tiling key.
    uint64_t GetTilingKey() const override;
    // 6. Compute workspace size.
    ge::graphStatus GetWorkspaceSize() override;
    // 7. Persist tiling data to the runtime context.
    ge::graphStatus PostTiling() override;

protected:
    void Reset();
    ge::graphStatus CheckContext() override;
    ge::graphStatus CalcUbTiling() override;
    virtual uint64_t GetBatchCoreCnt() const;
    virtual const void* GetTilingData() const;
    bool CheckDtype() const override;
    bool SetPlatformInfoForTiling() override;
    bool CheckShape(
        const std::vector<gert::Shape*>& mandtoryShape, const gert::StorageShape* biasShape,
        const gert::StorageShape* pertokenShape, const std::vector<int64_t>& dimValueOfMKN) const override;
    bool AnalyzeInputs() override;
    void CalcBlockWindowInfo();
    virtual bool CalL1Tiling() = 0;
    bool CheckBiasAndScale(uint64_t baseN, uint64_t dbL0c) const;
    bool AnalyseSlidingWinInfo();
    void SetBf16Compat();
    virtual void SetTilingData();
    uint32_t CalUsedCoreNum();
    virtual bool CalcBasicBlock();
    virtual void CalcTailBasicBlock();
    virtual void CalcTailBasicBlockAfullLoad();
    virtual void CalcTailBasicBlockBfullLoad();
    virtual void CalcTailBasicBlock4MmadS8S4();
    uint64_t GetTailBasicBlockSplitMax(bool isMSplit, uint64_t tileMax, uint64_t splitSize) const;
    bool CanIncreaseTailSplit(bool isPreSplitM, bool isPreSplit, uint64_t preSplit, uint64_t secSplit,
                              uint64_t splitMax);
    uint64_t GetTailSplitState(bool isPreSplitM, bool isPreSplit, uint64_t split, uint64_t splitSize) const;
    void CalcTailBasicBlockSplit(bool isPreSplitM, uint64_t preSplitMax, uint64_t secSplitMax,
                                 uint64_t preSplitSize, uint64_t secSplitSize);
    virtual void AnalyseFullLoadInfo() = 0;
    virtual void CalcTailRoundBasicBlockSplit();
    uint32_t CalUsedCoreNum(uint32_t mTile, uint32_t nTile);
    bool IsMxKOdd() const;
    bool IsMxBackwardTrans() const;
    uint64_t GetBiasMode() const;
    virtual uint64_t GetKernelType() const;
    virtual uint64_t GetApiLevel(NpuArch npuArch) const;

    bool IsInValidWeighNzTailSplit(uint64_t splitCnt, bool isPreSplit) const;

    void LoadBalanceDataReset();
    bool OptimizeEdgeBasicBlock();
    void CalculateCurrentPerf(uint64_t mergeLen, uint64_t nTail, uint64_t& newTailMain, uint64_t& curPerf);
    void GetOuterMAxisTailCnt(uint64_t& baseTailSplitCnt, uint64_t& tailMain);
    void GetOuterNAxisTailCnt(uint64_t& baseTailSplitCnt, uint64_t& tailMain);
    virtual bool CheckCoreNum() const;

    DequantBmm::QuantBatchMatmulV3TilingDataParams tilingDataSelf_;
    DequantBmm::QuantBatchMatmulV3TilingDataParams& tilingData_;
    AdaptiveSlidingWindow adaptiveWin_;
    BasicRunInfoTiling basicTiling_;
    bool isAFullLoad_ = false;
    bool isBFullLoad_ = false;
    bool isABFullLoad_ = false;
    bool isBf16Mix_ = false;
};
} // namespace optiling

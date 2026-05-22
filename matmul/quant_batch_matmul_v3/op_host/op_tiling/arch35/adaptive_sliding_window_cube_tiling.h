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
 * \file adaptive_sliding_window_cube_tiling.h
 * \brief
 */
#pragma once

#include "adaptive_sliding_window_tiling.h"

namespace optiling {

class AdaptiveSlidingWindowCubeTiling : public AdaptiveSlidingWindowTiling {
public:
    explicit AdaptiveSlidingWindowCubeTiling(gert::TilingContext* context);
    AdaptiveSlidingWindowCubeTiling(gert::TilingContext* context, DequantBmm::QuantBatchMatmulV3TilingDataParams* out);
    ~AdaptiveSlidingWindowCubeTiling() override = default;

protected:
    bool IsCapable() override;
    bool CalcBasicBlock() override;
    bool CalL1Tiling() override;
    virtual bool IsCalL1TilingDepth4MmadS8S4() const;
    void AnalyseFullLoadInfo() override;
    void CalcTailRoundBasicBlockSplit() override;
    virtual void CalL1TilingDepth4MmadS8S4(uint64_t leftL1Size);
    virtual void UpdateAFullLoadStatus();
    virtual void UpdateBFullLoadStatus();
    virtual void UpdateABFullLoadStatus();

    uint64_t singleCoreASizeWithFullLoad_ = 0;
    uint64_t singleCoreBSizeWithFullLoad_ = 0;

private:
    bool CalL1Tiling3510();
    bool CalL1TilingMmadS8S4();
    void CalcTailBasicBlockBfullLoad();
    void CalcTailBasicBlock4MmadS8S4();
    bool CheckL1Size(uint64_t leftL1Size, uint64_t tempStepKa, uint64_t tempStepKb) const;
    void AdjustStepK(uint64_t leftL1Size, uint64_t& tempStepKa, uint64_t& tempStepKb, bool isStepKa) const;
    void CarryDataSizePass(uint64_t leftL1Size, uint64_t maxStepK);
    void BalanceStepKPass(uint64_t leftL1Size);
    void PostCacheLinePass(uint64_t leftL1Size, uint64_t maxStepK);
    void L1FullLoadCacheLinePass(uint64_t& tempStepKa, uint64_t& tempStepKb, uint64_t aCacheLine, uint64_t bCacheLine);
    void NONL1FullLoadCacheLinePass(
        uint64_t& tempStepKa, uint64_t& tempStepKb, uint64_t aCacheLine, uint64_t bCacheLine);
};

} // namespace optiling

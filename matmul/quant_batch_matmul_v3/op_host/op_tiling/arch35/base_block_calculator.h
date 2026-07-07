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
 * \file base_block_calculator.h
 * \brief
 */
#pragma once

#include <cstdint>

#include "../quant_batch_matmul_v3_tiling_base.h"

namespace optiling {

struct BaseBlockRes {
    uint64_t baseM = 0UL;
    uint64_t baseN = 0UL;
    uint64_t baseK = 0UL;
    uint64_t mCnt = 1UL;
    uint64_t nCnt = 1UL;
    uint64_t preSplitKBlockCnt = 1UL;
    uint64_t streamKCnt = 1UL;
    uint64_t singleCoreK = 0UL;
    bool useTailWinLogic = true;
};

enum class BaseBlockMode { DEFAULT = 0, PERBLOCK, MMAD_S8S4, STREAMK };

class BaseBlockCalculator {
public:
    BaseBlockCalculator(const QuantBatchMatmulInfo& inputParams, const QuantBatchMatmulV3CompileInfo& compileInfo,
                        uint64_t batchCoreCnt = 1UL);
    virtual ~BaseBlockCalculator() = default;

    bool Compute(BaseBlockMode mode);
    const BaseBlockRes& GetOutput() const;

private:
    bool ValidateInput() const;
    bool ValidateBaseBlock() const;
    void ComputeBaseBlockDefault();
    void ComputeBaseBlockPerblock();
    void ComputeBaseBlockMmadS8S4();
    bool ComputeBaseBlockStreamK();
    bool InitStreamKBaseBlock();
    bool UpdateSmallMnStreamKBase();
    void UpdateTailStreamKBase();
    bool FinalizeStreamKBaseK();
    uint64_t GetBaseMAlignSize() const;
    uint64_t GetBaseNAlignSize() const;
    uint64_t GetBaseKAlignSize() const;
    bool OptimizeBaseBlockForCoreUtilization(BaseBlockMode mode);
    void OptimizeBaseBlockForLoadBalance();
    void SearchLoadBalanceBaseBlock(uint64_t roundLimit, uint64_t originLastRoundUsedCore,
                                    double originMemoryComputeScore, uint64_t& bestBaseM, uint64_t& bestBaseN) const;
    bool ShouldSkipLoadBalanceCandidate(uint64_t curBaseM, uint64_t curBaseN, uint64_t originLastRoundUsedCore,
                                        double originMemoryComputeScore) const;
    void TryApplyLoadBalanceBase(uint64_t bestBaseM, uint64_t bestBaseN);
    bool AdjustBaseBlockDefault();
    void TrySwapBaseMNForMxFalseTrue(uint64_t& baseM, uint64_t& baseN) const;
    bool AdjustBaseBlockPerblock();
    bool AdjustBaseBlockPertile(uint64_t coreNumMN);
    bool AdjustBaseBlockMmadS8S4(uint64_t oriBlock);
    uint64_t GetSingleCoreMaxRound(uint64_t baseM, uint64_t baseN) const;
    double GetBalanceRate(uint64_t baseM, uint64_t baseN) const;
    uint64_t GetLastRoundBlockCnt(uint64_t baseM, uint64_t baseN) const;
    bool CalculateOptimalSplit(uint64_t& baseM, uint64_t& baseN, uint64_t baseMAlignNum, uint64_t baseNAlignNum,
                               uint64_t baseKAlignNum) const;
    bool IsMxBackwardTrans() const;

    const QuantBatchMatmulInfo& inputParams_;
    const QuantBatchMatmulV3CompileInfo& compileInfo_;
    uint64_t batchCoreCnt_ = 1UL;
    BaseBlockRes baseBlockRes_;
};

} // namespace optiling
